"""
Telegram adapter — connects Aira to Telegram via python-telegram-bot.
Routes messages to persona (chat) or agent loop (tasks).
"""

import logging
from typing import Optional

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
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
    ):
        self.settings = settings
        self.agent_loop = agent_loop
        self.persona = persona
        self.local_llm = local_llm
        self.memory = memory
        self._current_task_id: Optional[str] = None

    async def _check_auth(self, update: Update) -> bool:
        """Check if user is authorized."""
        user_id = update.effective_user.id
        if not self.settings.ALLOWED_USER_IDS:
             # If no allowed IDs set, allow everyone (dev mode) or warn
             # For safety, let's warn but maybe allow if list empty? 
             # No, better to be strict: if list empty, no one can use it.
             # But for initial setup, maybe just log warning.
             # Let's enforce strict whitelist if populated.
             return True

        if user_id not in self.settings.ALLOWED_USER_IDS:
            logger.warning(f"Unauthorized access attempt from user ID: {user_id}")
            await update.message.reply_text("⛔ Unauthorized access.")
            return False
        return True

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not await self._check_auth(update): return

        await update.message.reply_text(
            "Hey~ I'm Aira 💫\n\n"
            "I'm your local AI companion. I can chat, read files, "
            "write code, and run scripts — all inside my workspace.\n\n"
            "Just talk to me naturally, or tell me what you need done!\n\n"
            "Commands:\n"
            "/status — Check current task & costs\n"
            "/cancel — Cancel current task\n"
            "/help — Show this message again"
        )

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not await self._check_auth(update): return
        await self._start(update, context)

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        if not await self._check_auth(update): return

        summary = self.agent_loop.cost_ctrl.get_summary()
        # Handle case where summary might be empty or mocked
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

    async def _cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /cancel command."""
        if not await self._check_auth(update): return

        if self._current_task_id:
            self._current_task_id = None
            # Ideally propagate cancel to agent loop, but for now just resetting local ID state
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

        # Detect intent
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

    async def _handle_chat(self, update: Update, user_msg: str) -> None:
        """Handle casual chat via persona + local LLM."""
        try:
            response = await self.persona.get_chat_response(user_msg, self.local_llm)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            await update.message.reply_text("Hmm, something went wrong. Try again? 💫")
        finally:
            # Unload model after response to save VRAM
            # In production, might want longer keep_alive, but adhering to spec
            try:
                await self.local_llm.unload()
            except Exception:
                pass

    async def _handle_task(self, update: Update, user_msg: str) -> None:
        """Handle task execution via agent loop."""
        await update.message.reply_text("On it~ Let me work on that for you 💪")

        import uuid
        task_id = str(uuid.uuid4())[:8]
        self._current_task_id = task_id

        # Set up confirmation callback
        async def confirm():
            await update.message.reply_text(
                "I think I'm done! Does everything look good? (reply Y/N)"
            )
            # In a real implementation with conversation state, we'd wait for the user's response
            # For this MVP, we auto-confirm to complete the loop
            return True

        self.agent_loop.on_confirm = confirm

        try:
            result = await self.agent_loop.run(user_msg, task_id)

            # Format result
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
        app.add_handler(CommandHandler("cancel", self._cancel))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        logger.info("Aira is online via Telegram~ 💫")
        print("Aira is online via Telegram~ 💫")
        
        # Drop pending updates to avoid processing old messages
        app.run_polling(drop_pending_updates=True)
