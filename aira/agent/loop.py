"""
Agent execution loop — the core brain of Aira.
Ties together state machine, tools, planner, memory, and security.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from aira.agent.state import TaskState
from aira.agent.state_machine import StateMachine
from aira.agent.completion import is_task_complete
from aira.tools.registry import ToolRegistry, ToolResult
from aira.security.outbound_guard import sanitise_for_llm, SecurityError
from aira.security.cost_controller import CostController
from aira.memory.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class AgentLoop:
    """Main agent execution loop with deterministic state transitions."""

    def __init__(
        self,
        state_machine_factory: Callable[[str], StateMachine],
        tool_registry: ToolRegistry,
        planner: Any,        # ExternalPlanner or LocalQwen
        local_llm: Any,      # LocalQwen
        memory: MemoryStore,
        cost_controller: CostController,
        max_iterations: int = 15,
        max_output_chars: int = 20000,
        log_dir: Path = Path("./aira/logs"),
    ):
        self.sm_factory = state_machine_factory
        self.tools = tool_registry
        self.planner = planner
        self.local_llm = local_llm
        self.memory = memory
        self.cost_ctrl = cost_controller
        self.max_iterations = max_iterations
        self.max_output_chars = max_output_chars
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks for user interaction (set by Telegram adapter)
        self.on_update: Optional[Callable] = None   # async fn to push status updates
        self.on_confirm: Optional[Callable] = None  # async fn to ask Y/N

    async def run(self, task_description: str, task_id: str | None = None) -> dict:
        """
        Execute a task through the full state machine lifecycle.

        Args:
            task_description: What the user wants done.
            task_id: Optional existing task ID to resume.

        Returns:
            dict with final status, outputs, and cost summary.
        """
        task_id = task_id or str(uuid.uuid4())[:8]
        sm = self.sm_factory(task_id)
        log_file = self.log_dir / f"task_{task_id}_log.jsonl"
        iteration = 0
        last_tool_result: Optional[ToolResult] = None
        plan: dict = {}
        step_index = 0
        validation_errors: list[str] = []
        failed_plan_keys: set[str] = set()  # Track plans that already failed
        replan_count = 0  # Prevent infinite replans

        # Save task to memory
        await self.memory.save_task(task_id, {
            "description": task_description,
            "status": "INIT",
        })

        try:
            while iteration < self.max_iterations:
                iteration += 1
                iter_start = time.time()
                log_entry: dict = {
                    "iteration": iteration,
                    "task_id": task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # ── PLANNING ──
                if sm.current_state == TaskState.INIT or (
                    sm.current_state == TaskState.PLANNING
                    and not plan.get("steps")
                ):
                    if sm.current_state == TaskState.INIT:
                        sm.transition(TaskState.PLANNING, "start task")

                    # Get context from memory
                    summaries = await self.memory.get_recent_summaries(limit=3)
                    context = "\n".join(summaries) if summaries else ""

                    # Ask planner for a plan
                    try:
                        plan = await self._get_plan(
                            context, task_description,
                            skip_keywords=failed_plan_keys,
                        )
                        # Generate a key for this plan to detect repeated failures
                        plan_key = json.dumps(
                            plan.get("steps", []), sort_keys=True, default=str
                        )[:200]
                        step_index = 0
                        log_entry["plan_steps"] = len(plan.get("steps", []))
                    except Exception as e:
                        sm.transition(TaskState.FAILED, f"planning failed: {e}")
                        log_entry["error"] = str(e)
                        self._write_log(log_file, log_entry)
                        break

                # ── VALIDATING ──
                if sm.current_state == TaskState.PLANNING:
                    sm.transition(TaskState.VALIDATING, "plan ready")
                    validation_errors = self._validate_plan(plan)

                    if validation_errors:
                        log_entry["validation_errors"] = validation_errors
                        sm.transition(TaskState.PLANNING, "plan invalid — replan")
                        plan = {}  # Clear plan to force replan
                        self._write_log(log_file, log_entry)
                        continue
                    else:
                        sm.transition(TaskState.EXECUTING, "plan valid")

                # ── EXECUTING ──
                if sm.current_state == TaskState.EXECUTING:
                    steps = plan.get("steps", [])
                    if step_index >= len(steps):
                        # All steps done
                        sm.transition(TaskState.FILTERING, "all steps executed")
                    else:
                        step = steps[step_index]
                        tool_name = step.get("tool", "")
                        params = step.get("params", {})

                        log_entry["tool"] = tool_name
                        log_entry["params"] = params

                        exec_start = time.time()
                        last_tool_result = await self._execute_tool(tool_name, params)
                        exec_duration = int((time.time() - exec_start) * 1000)

                        log_entry["success"] = last_tool_result.success
                        log_entry["duration_ms"] = exec_duration
                        log_entry["output_size"] = len(last_tool_result.output)

                        # Save step to memory
                        await self.memory.save_step(task_id, {
                            "iteration": iteration,
                            "tool_name": tool_name,
                            "parameters": params,
                            "output": last_tool_result.output[:500],
                            "success": last_tool_result.success,
                            "duration_ms": exec_duration,
                        })

                        step_index += 1

                        if not last_tool_result.success:
                            log_entry["error"] = last_tool_result.error
                            # Track this plan as failed so we don't repeat it
                            plan_key = json.dumps(
                                plan.get("steps", []), sort_keys=True, default=str
                            )[:200]
                            failed_plan_keys.add(plan_key)
                            sm.transition(TaskState.FAILED, f"tool failed: {last_tool_result.error}")
                            self._write_log(log_file, log_entry)
                            break

                        sm.transition(TaskState.FILTERING, "tool done")

                # ── FILTERING ──
                if sm.current_state == TaskState.FILTERING:
                    try:
                        if last_tool_result:
                            sanitise_for_llm(
                                last_tool_result.output,
                                self.max_output_chars,
                            )
                        sm.transition(TaskState.UPDATING, "output safe")
                    except SecurityError as e:
                        sm.transition(TaskState.FAILED, f"security violation: {e}")
                        log_entry["security_error"] = str(e)
                        self._write_log(log_file, log_entry)
                        break

                # ── UPDATING ──
                if sm.current_state == TaskState.UPDATING:
                    planner_done = plan.get("done", False)
                    all_steps_done = step_index >= len(plan.get("steps", []))
                    tool_ok = last_tool_result.success if last_tool_result else False

                    if all_steps_done and is_task_complete(
                        planner_done=planner_done,
                        last_tool_success=tool_ok,
                        validation_errors=[],
                        output_valid=True,
                    ):
                        sm.transition(TaskState.CONFIRMING_COMPLETION, "all goals met")
                    else:
                        # More work needed or planner says not done
                        replan_count += 1
                        if replan_count > 3:
                            sm.transition(TaskState.FAILED, "max replans exceeded")
                            self._write_log(log_file, log_entry)
                            break
                        sm.transition(TaskState.PLANNING, "needs replan")
                        plan = {}  # Clear for next planning cycle

                # ── CONFIRMING COMPLETION ──
                if sm.current_state == TaskState.CONFIRMING_COMPLETION:
                    confirmed = await self._ask_confirmation()
                    if confirmed:
                        sm.transition(TaskState.COMPLETED, "user confirms")
                    else:
                        sm.transition(TaskState.PLANNING, "user rejects — replan")
                        plan = {}
                    self._write_log(log_file, log_entry)

                    if sm.is_terminal():
                        break

                # Log iteration
                log_entry["state"] = sm.current_state.value
                try:
                    log_entry["cost_summary"] = self.cost_ctrl.get_summary()
                except Exception:
                     log_entry["cost_summary"] = {} # Handle mock objects gracefully
                log_entry["iter_duration_ms"] = int((time.time() - iter_start) * 1000)
                self._write_log(log_file, log_entry)

                if sm.is_terminal():
                    break

            # Max iterations exceeded
            if not sm.is_terminal():
                try:
                    sm.transition(TaskState.FAILED, "max iterations exceeded")
                except Exception:
                    pass  # Already in terminal state

        except Exception as e:
            logger.error(f"Agent loop error: {e}", exc_info=True)
            try:
                sm.transition(TaskState.FAILED, f"unexpected error: {e}")
            except Exception:
                pass

        # Final status
        final = {
            "task_id": task_id,
            "final_state": sm.current_state.value,
            "iterations": iteration,
            "cost": self.cost_ctrl.get_summary() if hasattr(self.cost_ctrl, "get_summary") else {},
        }

        await self.memory.update_task_status(
            task_id, sm.current_state.value,
            summary=f"Completed in {iteration} iterations"
        )

        return final

    async def _get_plan(
        self,
        context: str,
        task_description: str,
        skip_keywords: set[str] | None = None,
    ) -> dict:
        """
        Generate a plan using:
        1. Keyword heuristic (fast, deterministic)
        2. Local LLM (flexible)
        3. External Planner (smart, costly)
        """
        # 1. Keyword Planner
        # We try this first because it's instant and reliable for simple tasks.
        keyword_plan = self._build_keyword_plan(task_description, skip_keywords)
        if keyword_plan:
            logger.info("Using keyword-based plan")
            return keyword_plan

        # 2. Local LLM Planner
        # Used for complex tasks that don't match simple keywords.
        system_prompt = (
             "You are a task planner. Given a context and task, produce a JSON plan.\n"
             "Respond ONLY with valid JSON:\n"
             '{"steps": [{"tool": "tool_name", "params": {...}}], "done": false}\n'
             "Available tools:\n"
             "- file_read(path)\n"
             "- file_write(path, content)\n"
             "- python_run(script_path)\n"
             "- system_read(action, path?, query?)\n"
             "- system_command(command)  <-- use for ANY shell/system command\n"
        )
        try:
            prompt = (
                f"Create a JSON plan for this task:\n{task_description}\n\n"
                f"Context:\n{context}\n\n"
                'Respond ONLY with valid JSON, no markdown, no explanation:\n'
                '{"steps": [{"tool": "tool_name", "params": {"key": "value"}, "description": "what this does"}], "confidence": 0.8, "done": true}\n\n'
                "Available tools (use EXACT names):\n"
                "- system_read: params {action, path, query}. Actions: read_file, list_dir, system_info, processes, disk_usage, gpu_info, network_info, installed_apps, search_files, env_var\n"
                "- system_command: params {command}. Runs PowerShell (requires user confirmation)\n"
                "- file_read: params {path}. Read file in workspace\n"
                "- file_write: params {path, content}. Write file in workspace\n"
                "- python_run: params {code}. Run Python code\n"
            )
            raw = await self.local_llm.generate(prompt)
            plan = self._extract_json(raw)
            if plan and plan.get("steps"):
                plan = self._normalize_plan(plan)
                # Validate — only use if all tools are valid
                errors = self._validate_plan(plan)
                if not errors:
                    logger.info(f"Local LLM plan valid with {len(plan['steps'])} steps")
                    return plan
                else:
                    logger.warning(f"LLM plan invalid: {errors}")
        except Exception as e:
            logger.warning(f"Local plan failed: {e}")

        # 3. External planner only if API key configured
        if hasattr(self.planner, "plan") and getattr(self.planner, "api_key", ""):
            try:
                return await self.planner.plan(context, description)
            except Exception as e:
                logger.warning(f"External planner failed: {e}")

        # 4. Ultimate fallback — system_info (better than empty/failed)
        return {"steps": [{"tool": "system_read", "params": {"action": "system_info"}, "description": "Get general system info"}], "confidence": 0.3, "done": True}

    def _normalize_plan(self, plan: dict) -> dict:
        """Fix common LLM mistakes in tool names and params."""
        TOOL_ALIASES = {
            "read_system": "system_read",
            "system": "system_read",
            "shell": "system_command",
            "shell_exec": "system_command",
            "shell_run": "system_command",
            "powershell": "system_command",
            "run_command": "system_command",
            "exec": "system_command",
            "read": "file_read",
            "write": "file_write",
            "python": "python_run",
            "run_python": "python_run",
        }

        for step in plan.get("steps", []):
            tool = step.get("tool", "")
            if tool in TOOL_ALIASES:
                step["tool"] = TOOL_ALIASES[tool]
            # If tool was aliased to system_read but has no action, infer it
            if step["tool"] == "system_read" and "action" not in step.get("params", {}):
                # The old tool name might BE the action
                if tool in ("system_info", "gpu_info", "disk_usage", "processes", "network_info"):
                    step.setdefault("params", {})["action"] = tool
        return plan

    def _extract_json(self, raw: str) -> dict:
        """Try hard to extract JSON from LLM output (handles markdown fences, etc)."""
        import re
        # Direct parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'(\{.*\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
        return {}

    def _build_keyword_plan(self, task: str, skip_keywords: set[str] | None = None) -> Optional[dict]:
        """
        Simple heuristic planner for common tasks.
        Returns a plan dict or None if no keywords match.
        """
        t = task.lower()
        
        # Helper to check if a plan was already attempted and failed
        def _make_plan(steps: list[dict], confidence: float = 0.9) -> Optional[dict]:
            plan = {"steps": steps, "confidence": confidence, "done": True}
            if skip_keywords:
                plan_key = json.dumps(steps, sort_keys=True, default=str)[:200]
                if plan_key in skip_keywords:
                    logger.info(f"Skipping failed keyword plan: {plan_key}")
                    return None
            return plan

        # System info queries
        if any(w in t for w in ["battery", "power", "charging"]):
            return _make_plan([{"tool": "system_read", "params": {"action": "system_info"}, "description": "Get system info including battery"}])

        if any(w in t for w in ["gpu", "vram", "graphics", "nvidia"]):
            return _make_plan([{"tool": "system_read", "params": {"action": "gpu_info"}, "description": "Get GPU information"}])

        if any(w in t for w in ["process", "running", "task manager", "what's running"]):
            query = ""
            # Extract app name if mentioned
            for word in ["chrome", "python", "node", "code", "discord", "spotify", "steam", "ollama"]:
                if word in t:
                    query = word
                    break
            return _make_plan([{"tool": "system_read", "params": {"action": "processes", "query": query}, "description": "List running processes"}])

        if any(w in t for w in ["disk", "storage", "space", "drive"]):
            return _make_plan([{"tool": "system_read", "params": {"action": "disk_usage"}, "description": "Check disk usage"}])

        if any(w in t for w in ["network", "ip", "wifi", "internet", "connection"]):
            return _make_plan([{"tool": "system_read", "params": {"action": "network_info"}, "description": "Get network info"}])
        
        if any(w in t for w in ["app", "installed", "program", "software"]):
             return _make_plan([{"tool": "system_read", "params": {"action": "installed_apps"}, "description": "List installed apps"}])

        # File listing (e.g. "what files in X")
        if any(w in t for w in ["list", "show files", "dir", "ls"]):
             path = self._extract_path(task)
             return _make_plan([{"tool": "system_read", "params": {"action": "list_dir", "path": str(path)}, "description": f"List files in {path}"}])

        return None

        if any(w in t for w in ["open ", "launch ", "start ", "close ", "kill ", "shutdown", "restart"]):
            # Build a shell command from the request
            if "open" in t or "launch" in t or "start" in t:
                app = task.split("open")[-1].split("launch")[-1].split("start")[-1].strip()
                cmd = f"Start-Process '{app}'"
            elif "close" in t or "kill" in t:
                app = task.split("close")[-1].split("kill")[-1].strip()
                cmd = f"Stop-Process -Name '{app}' -Force"
            else:
                cmd = task
            return _make_plan([{"tool": "system_command", "params": {"command": cmd}, "description": f"Shell: {cmd}"}])

        # Generic system query — try system_info as default
        if any(w in t for w in ["what", "how", "check", "tell me", "show"]):
            return _make_plan([{"tool": "system_read", "params": {"action": "system_info"}, "description": "Get system information"}])

        return None

    @staticmethod
    def _extract_path(text: str) -> str:
        """Try to extract a file/directory path from text."""
        import re
        import subprocess
        # Windows paths
        match = re.search(r'([A-Za-z]:\\[^\s,\'"]+)', text)
        if match:
            return match.group(1)
        # Use Windows API for common folders (handles OneDrive)
        folder_map = {
            "desktop": "Desktop",
            "documents": "MyDocuments",
            "downloads": None,
            "pictures": "MyPictures",
            "videos": "MyVideos",
            "music": "MyMusic",
        }
        for name, enum_name in folder_map.items():
            if name in text.lower():
                if enum_name:
                    try:
                        result = subprocess.run(
                            ["powershell", "-Command",
                             f"[Environment]::GetFolderPath('{enum_name}')"],
                            capture_output=True, text=True, timeout=3
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            return result.stdout.strip()
                    except Exception:
                        pass
                elif name == "downloads":
                    return str(Path.home() / "Downloads")
                return str(Path.home() / name.capitalize())
        return ""

    def _validate_plan(self, plan: dict) -> list[str]:
        """Validate plan structure. Returns list of errors."""
        errors = []
        if not isinstance(plan, dict):
            errors.append("Plan is not a dict")
            return errors
        if "steps" not in plan:
            errors.append("Plan missing 'steps' key")
        elif not isinstance(plan["steps"], list):
            errors.append("'steps' is not a list")
        else:
            for i, step in enumerate(plan["steps"]):
                if "tool" not in step:
                    errors.append(f"Step {i} missing 'tool' key")
                elif not self.tools.has(step["tool"]):
                    errors.append(f"Step {i} unknown tool: {step['tool']}")
        return errors

    async def _execute_tool(self, tool_name: str, params: dict) -> ToolResult:
        """Execute a tool by name with given parameters."""
        try:
            tool = self.tools.get(tool_name)
            import asyncio
            result = tool.run(**params)
            # Handle async tools (like shell_tool)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, ToolResult):
                return ToolResult(success=False, output="", error="Tool did not return ToolResult")
            return result
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    async def _ask_confirmation(self) -> bool:
        """Ask user for confirmation. Default True if no callback set."""
        if self.on_confirm:
            return await self.on_confirm()
        return True  # Auto-confirm in tests

    def _write_log(self, log_file: Path, entry: dict) -> None:
        """Append a log entry as JSON line."""
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
