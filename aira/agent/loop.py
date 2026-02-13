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
                        plan = await self._get_plan(context, task_description)
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

    async def _get_plan(self, context: str, description: str) -> dict:
        """Get a plan using the local or external planner."""
        # Try local model first (simple JSON plan)
        try:
            prompt = (
                f"Create a JSON plan for this task:\n{description}\n\n"
                f"Context:\n{context}\n\n"
                'Respond ONLY with JSON: {{"steps": [{{"tool": "name", "params": {{}}, "description": "..."}}], "confidence": 0.0-1.0, "done": false}}\n'
                "Available tools: file_read, file_write, python_run"
            )
            raw = await self.local_llm.generate(prompt)
            plan = json.loads(raw)
            return plan
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Local plan failed, trying external: {e}")

        # Fallback to external planner
        if hasattr(self.planner, "plan"):
            return await self.planner.plan(context, description)

        # Last resort — empty plan
        return {"steps": [], "confidence": 0.0, "done": False}

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
            result = tool.run(**params) if not callable(getattr(tool, 'run', None)) else tool.run(**params)
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
