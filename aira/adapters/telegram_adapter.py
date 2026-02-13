"""
Telegram adapter — connects Aira to Telegram via python-telegram-bot.
Routes messages to persona (chat) or agent loop (tasks).
Supports inline keyboard confirmation for shell commands.
"""

import asyncio
import logging
import uuid
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from aira.agent.loop import AgentLoop
from aira.persona.persona_manager import PersonaManager
from aira.llm.local_qwen import LocalQwen
from aira.memory.memory_store import MemoryStore
from aira.config.settings import Settings

logger = logging.getLogger(__name__)


class TelegramAdapter:
    """Telegram bot adapter for Aira."""

    def __init__(
        self,
        settings: Settings,
        agent_loop: AgentLoop,
        persona: PersonaManager,
        local_llm: LocalQwen,
        memory: MemoryStore,
        system_command_tool=None,
    ):
        self.settings = settings
        self.agent_loop = agent_loop
        self.persona = persona
        self.local_llm = local_llm
        self.memory = memory
        self.sys_cmd_tool = system_command_tool
        self._current_task_id: Optional[str] = None
        # Pending confirmation future for typed commands
        self._pending_confirm_future: Optional[asyncio.Future] = None

    async def _check_auth(self, update: Update) -> bool:
        """Check if user is authorized."""
        user_id = update.effective_user.id
        if not self.settings.ALLOWED_USER_IDS:
            return True
        if user_id not in self.settings.ALLOWED_USER_IDS:
            logger.warning(f"Unauthorized access attempt from user ID: {user_id}")
            if update.message:
                await update.message.reply_text("⛔ Unauthorized access.")
            return False
        return True

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not await self._check_auth(update): return

        await update.message.reply_text(
            "Hey~ I'm Aira 💫\n\n"
            "I'm your local AI companion. I can:\n"
            "💬 Chat with you naturally\n"
            "📂 Read ANY file on your system\n"
            "🖥️ Get system info (CPU, RAM, GPU, processes)\n"
            "⚡ Run shell commands (with your permission!)\n"
            "📝 Read & write files in my workspace\n\n"
            "Just talk to me or tell me what to do!\n\n"
            "Commands:\n"
            "/status — Check current task & costs\n"
            "/sysinfo — Quick system overview\n"
            "/gpu — GPU status\n"
            "/cancel — Cancel current task\n"
            "/help — Show this message"
        )

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update): return
        await self._start(update, context)

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update): return

        summary = self.agent_loop.cost_ctrl.get_summary()
        total_spent = summary.get('total_spent', 0.0)
        max_budget = summary.get('max_budget', 0.0)
        call_count = summary.get('call_count', 0)
        warning = summary.get('warning', False)

        status = (
            f"📊 Status\n"
            f"Current task: {self._current_task_id or 'None'}\n"
            f"Cost: ${total_spent:.4f} / ${max_budget:.2f}\n"
            f"Calls: {call_count}\n"
            f"{'⚠️ Budget warning!' if warning else '✅ Budget OK'}"
        )
        await update.message.reply_text(status)

    async def _sysinfo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Quick system info command."""
        if not await self._check_auth(update): return
        from aira.tools.system_read import SystemReadTool
        tool = SystemReadTool()
        result = tool.run(action="system_info")
        await update.message.reply_text(result.output if result.success else f"Error: {result.error}")

    async def _gpu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Quick GPU info command."""
        if not await self._check_auth(update): return
        from aira.tools.system_read import SystemReadTool
        tool = SystemReadTool()
        result = tool.run(action="gpu_info")
        await update.message.reply_text(result.output if result.success else f"Error: {result.error}")

    async def _cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_auth(update): return
        if self._current_task_id:
            self._current_task_id = None
            await update.message.reply_text("Task cancelled. What's next? 💫")
        else:
            await update.message.reply_text("No active task to cancel~")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle all text messages — route to chat or task."""
        if not await self._check_auth(update): return

        user_msg = update.message.text
        if not user_msg:
            return

        logger.info(f"Received from {update.effective_user.id}: {user_msg[:100]}")

        # 0. Check for pending typed confirmation
        if self._pending_confirm_future and not self._pending_confirm_future.done():
            self._pending_confirm_future.set_result(user_msg)
            return

        # First: try direct system query (bypasses agent loop entirely)
        handled = await self._try_system_query(update, user_msg)
        if handled:
            return

        # Detect intent via LLM / keywords
        intent = await self.persona.detect_intent(user_msg, self.local_llm)
        logger.info(f"Intent detected: {intent}")

        if intent == "chat":
            await self._handle_chat(update, user_msg)
        elif intent == "task":
            await self._handle_task(update, user_msg)
        elif intent == "cancel":
            await self._cancel(update, context)
        elif intent == "status":
            await self._status(update, context)
        else:
            await self._handle_chat(update, user_msg)

    async def _try_system_query(self, update: Update, msg: str) -> bool:
        """
        Try to handle system queries directly without the agent loop.
        Returns True if handled, False if the message should go to normal routing.
        """
        from aira.tools.system_read import SystemReadTool
        from pathlib import Path

        lower = msg.lower()
        tool = SystemReadTool()
        result = None

        # Battery
        if any(w in lower for w in ["battery", "power level", "charge", "charging"]):
            result = tool.run(action="battery_info")

        # GPU
        elif any(w in lower for w in ["gpu", "vram", "graphics card", "nvidia", "rtx"]):
            result = tool.run(action="gpu_info")

        # System info
        elif any(w in lower for w in ["system info", "sysinfo", "cpu", "ram ", "my pc", "computer info", "pc info"]):
            result = tool.run(action="system_info")

        # Processes
        elif any(w in lower for w in ["process", "running", "task manager", "what's running"]):
            query = ""
            for app in ["chrome", "python", "node", "code", "discord", "spotify", "steam", "ollama", "firefox", "edge"]:
                if app in lower:
                    query = app
                    break
            result = tool.run(action="processes", query=query)

        # Disk usage
        elif any(w in lower for w in ["disk", "storage", "drive", "space left", "free space"]):
            result = tool.run(action="disk_usage")

        # Network
        elif any(w in lower for w in ["network", "ip address", "wifi", "internet connection"]):
            result = tool.run(action="network_info")

        # Installed apps
        elif any(w in lower for w in ["installed apps", "installed programs", "what apps", "what software"]):
            query = ""
            result = tool.run(action="installed_apps", query=query)

        # List directory
        elif any(w in lower for w in ["what files", "list files", "show files", "what's on my desktop",
                                       "what's in my", "list my", "show my desktop", "files on"]):
            path = self._detect_path(lower)
            result = tool.run(action="list_dir", path=path)

        # Read a file
        elif any(w in lower for w in ["read file", "show file", "contents of", "what's in the file"]):
            path = self._detect_path(lower)
            if path:
                result = tool.run(action="read_file", path=path)

        if result is not None:
            if result.success:
                # Wrap response with Aira's personality
                try:
                    persona_response = await self.persona.wrap_response(
                        result.output, self.local_llm
                    )
                    await update.message.reply_text(persona_response)
                except Exception:
                    await update.message.reply_text(result.output)
            else:
                await update.message.reply_text(f"Couldn't get that info: {result.error}")

            try:
                await self.local_llm.unload()
            except Exception:
                pass
            return True

        return False

    @staticmethod
    def _detect_path(text: str) -> str:
        """Extract a file/directory path from text."""
        import re
        import subprocess
        from pathlib import Path

        # Explicit Windows path
        match = re.search(r'([A-Za-z]:\\[^\s,\'"]+)', text)
        if match:
            return match.group(1)

        # Use Windows API for known folders (handles OneDrive redirection)
        folder_map = {
            "desktop": "Desktop",
            "documents": "MyDocuments",
            "downloads": None,  # No .NET enum for Downloads
            "pictures": "MyPictures",
            "videos": "MyVideos",
            "music": "MyMusic",
        }
        for name, enum_name in folder_map.items():
            if name in text:
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
                    # Downloads is always under user profile
                    return str(Path.home() / "Downloads")
                return str(Path.home() / name.capitalize())
        return str(Path.home())

    async def _handle_chat(self, update: Update, user_msg: str) -> None:
        """Handle casual chat via persona + local LLM."""
        try:
            response = await self.persona.get_chat_response(user_msg, self.local_llm)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            await update.message.reply_text("Hmm, something went wrong. Try again? 💫")
        finally:
            try:
                await self.local_llm.unload()
            except Exception:
                pass

    async def _handle_task(self, update: Update, user_msg: str) -> None:
        """Handle task execution via agent loop."""
        await update.message.reply_text("On it~ Let me work on that for you 💪")

        task_id = str(uuid.uuid4())[:8]
        self._current_task_id = task_id

        # Set up shell confirmation callback (Typed Confirmation)
        if self.sys_cmd_tool:
            async def sys_confirm(command: str, risk_level, affected_paths: list[str], working_dir: str) -> dict:
                """Ask user to confirm a system command by re-typing it."""
                
                # Format warning message
                risk_icon = "🔴" if risk_level.value in ("HIGH", "CRITICAL") else "⚠️"
                instruction = f"Type `{command}` to confirm."
                if risk_level.value in ("HIGH", "CRITICAL"):
                    instruction = "Type `CONFIRM` (all caps) to execute."
                
                msg = (
                    f"{risk_icon} **CONFIRMATION REQUIRED** {risk_icon}\n\n"
                    f"**Command:** `{command}`\n"
                    f"**Risk:** {risk_level.value}\n"
                    f"**Location:** `{working_dir}`\n"
                )
                if affected_paths:
                    msg += f"**Affected:** {', '.join(affected_paths)}\n"
                
                msg += f"\n{instruction}\nOr type anything else to cancel."

                await update.message.reply_text(msg, parse_mode="Markdown")

                # Create future and wait for user reply
                future = asyncio.get_event_loop().create_future()
                self._pending_confirm_future = future
                
                try:
                    # Wait 60s for user to type
                    user_text = await asyncio.wait_for(future, timeout=60.0)
                    
                    # Validate
                    approved = False
                    if risk_level.value in ("HIGH", "CRITICAL"):
                        approved = (user_text.strip() == "CONFIRM")
                    else:
                        approved = (user_text.strip() == command.strip())
                    
                    if approved:
                        await update.message.reply_text("✅ Confirmed. Executing in 5 seconds...")
                    else:
                        await update.message.reply_text("❌ Cancelled/Mismatch.")

                    return {"approved": approved, "text": user_text}

                except asyncio.TimeoutError:
                    self._pending_confirm_future = None
                    await update.message.reply_text("⏰ Timed out.")
                    return {"approved": False, "text": "TIMEOUT"}
                except Exception as e:
                    self._pending_confirm_future = None
                    return {"approved": False, "text": str(e)}
                finally:
                    self._pending_confirm_future = None

            self.sys_cmd_tool.confirm_callback = sys_confirm

        # Set up task confirmation callback
        async def task_confirm():
            return True  # Auto-confirm task completion

        self.agent_loop.on_confirm = task_confirm

        try:
            result = await self.agent_loop.run(user_msg, task_id)

            state = result["final_state"]
            iterations = result["iterations"]
            cost = result.get("cost", {})
            total_spent = cost.get('total_spent', 0.0)

            if state == "COMPLETED":
                persona_response = await self.persona.wrap_response(
                    f"Task completed in {iterations} iterations. Cost: ${total_spent:.4f}",
                    self.local_llm,
                )
                await update.message.reply_text(persona_response)
            else:
                await update.message.reply_text(
                    f"Task finished with state: {state}\n"
                    f"Iterations: {iterations}\n"
                    f"Cost: ${total_spent:.4f}"
                )

        except Exception as e:
            logger.error(f"Task error: {e}", exc_info=True)
            await update.message.reply_text(f"Task failed: {str(e)[:200]}")
        finally:
            self._current_task_id = None
            try:
                await self.local_llm.unload()
            except Exception:
                pass

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard button presses."""
        query = update.callback_query
        await query.answer()
        await query.edit_message_text("Buttons are deprecated for command confirmation. Please use text.")

    def run(self) -> None:
        """Start the Telegram bot (blocking)."""
        token = self.settings.TELEGRAM_BOT_TOKEN
        if not token or token == "your-token":
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is not set. Add it to your .env file."
            )

        app = ApplicationBuilder().token(token).build()

        # Register handlers
        app.add_handler(CommandHandler("start", self._start))
        app.add_handler(CommandHandler("help", self._help))
        app.add_handler(CommandHandler("status", self._status))
        app.add_handler(CommandHandler("sysinfo", self._sysinfo))
        app.add_handler(CommandHandler("gpu", self._gpu))
        app.add_handler(CommandHandler("cancel", self._cancel))
        app.add_handler(CallbackQueryHandler(self._handle_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        logger.info("Aira is online via Telegram~ 💫")
        print("Aira is online via Telegram~ 💫")

        app.run_polling(drop_pending_updates=True)
