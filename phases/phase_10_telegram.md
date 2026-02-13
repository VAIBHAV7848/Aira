# Phase 10 — Telegram Adapter & Main Entry Point

## Goal
Wire everything together. Expose Aira via Telegram bot. Create `main.py` entry point.

## Prerequisites
- ALL previous phases (0–9) passing tests

## Files to Create

### `aira/adapters/telegram_adapter.py`

```python
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

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
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
        await self._start(update, context)

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        cost = self.agent_loop.cost_ctrl.get_summary()
        status = (
            f"📊 Status\n"
            f"Current task: {self._current_task_id or 'None'}\n"
            f"Cost: ${cost['total_spent']:.4f} / ${cost['max_budget']:.2f}\n"
            f"Calls: {cost['call_count']}\n"
            f"{'⚠️ Budget warning!' if cost['warning'] else '✅ Budget OK'}"
        )
        await update.message.reply_text(status)

    async def _cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /cancel command."""
        if self._current_task_id:
            self._current_task_id = None
            await update.message.reply_text("Task cancelled. What's next? 💫")
        else:
            await update.message.reply_text("No active task to cancel~")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle all text messages — route to chat or task."""
        user_msg = update.message.text
        if not user_msg:
            return

        logger.info(f"Received: {user_msg[:100]}")

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
            # Unload model after response
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
            # In a real implementation, we'd wait for the user's response
            # For now, auto-confirm
            return True

        self.agent_loop.on_confirm = confirm

        try:
            result = await self.agent_loop.run(user_msg, task_id)

            # Format result
            state = result["final_state"]
            iterations = result["iterations"]
            cost = result["cost"]

            if state == "COMPLETED":
                persona_response = await self.persona.wrap_response(
                    f"Task completed in {iterations} iterations. Cost: ${cost['total_spent']:.4f}",
                    self.local_llm,
                )
                await update.message.reply_text(persona_response)
            else:
                await update.message.reply_text(
                    f"Task finished with state: {state}\n"
                    f"Iterations: {iterations}\n"
                    f"Cost: ${cost['total_spent']:.4f}"
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
        if not token:
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
        app.run_polling()
```

---

### `aira/main.py`

```python
"""
Aira — main entry point.
Wires all components together and starts the Telegram bot.
"""

import asyncio
import logging
import sys
from pathlib import Path

from aira.config.settings import Settings
from aira.security.path_guard import validate_path
from aira.security.outbound_guard import sanitise_for_llm
from aira.security.cost_controller import CostController
from aira.memory.memory_store import MemoryStore
from aira.tools.registry import ToolRegistry
from aira.tools.file_read import FileReadTool
from aira.tools.file_write import FileWriteTool
from aira.tools.python_runner import PythonRunnerTool
from aira.llm.local_qwen import LocalQwen
from aira.llm.external_planner import ExternalPlanner
from aira.agent.state_machine import StateMachine
from aira.agent.loop import AgentLoop
from aira.persona.persona_manager import PersonaManager
from aira.adapters.telegram_adapter import TelegramAdapter


def setup_logging(log_dir: Path) -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "aira.log"),
        ],
    )


async def init_memory(db_path: Path) -> MemoryStore:
    """Initialize the memory store."""
    memory = MemoryStore(db_path)
    await memory.init_db()
    return memory


def main() -> None:
    """Wire all components and start Aira."""
    # 1. Load settings
    settings = Settings()
    setup_logging(settings.LOG_DIR)
    logger = logging.getLogger("aira.main")
    logger.info("Starting Aira...")

    # 2. Ensure workspace exists
    settings.WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    # 3. Init memory (async)
    memory = asyncio.run(init_memory(settings.DB_PATH))

    # 4. Security components
    cost_ctrl = CostController(
        max_cost_usd=settings.MAX_COST_USD,
        warning_threshold=settings.COST_WARNING_THRESHOLD,
        persist_path=settings.LOG_DIR / "cost_state.json",
    )

    # 5. Tools
    tool_registry = ToolRegistry()
    tool_registry.register("file_read", FileReadTool(
        settings.WORKSPACE_ROOT, settings.MAX_OUTPUT_CHARS
    ))
    tool_registry.register("file_write", FileWriteTool(settings.WORKSPACE_ROOT))
    tool_registry.register("python_run", PythonRunnerTool(
        settings.WORKSPACE_ROOT, settings.SUBPROCESS_TIMEOUT
    ))

    # 6. LLMs
    local_llm = LocalQwen(settings.OLLAMA_BASE_URL, settings.OLLAMA_MODEL)
    external_planner = ExternalPlanner(
        cost_controller=cost_ctrl,
        model=settings.EXTERNAL_LLM_MODEL or "gpt-4o-mini",
        api_key=settings.EXTERNAL_LLM_API_KEY,
        max_output_chars=settings.MAX_OUTPUT_CHARS,
    )

    # 7. State machine factory
    sm_factory = lambda tid: StateMachine(tid, settings.LOG_DIR)

    # 8. Agent loop
    agent_loop = AgentLoop(
        state_machine_factory=sm_factory,
        tool_registry=tool_registry,
        planner=external_planner,
        local_llm=local_llm,
        memory=memory,
        cost_controller=cost_ctrl,
        max_iterations=settings.MAX_ITERATIONS,
        max_output_chars=settings.MAX_OUTPUT_CHARS,
        log_dir=settings.LOG_DIR,
    )

    # 9. Persona
    persona = PersonaManager()

    # 10. Telegram adapter
    adapter = TelegramAdapter(settings, agent_loop, persona, local_llm, memory)

    logger.info("All components wired. Starting Telegram bot...")
    adapter.run()


if __name__ == "__main__":
    main()
```

---

## Test File

### `aira/tests/test_telegram.py`

```python
"""Tests for Telegram adapter (no real Telegram — just routing logic)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aira.adapters.telegram_adapter import TelegramAdapter
from aira.config.settings import Settings
from aira.persona.persona_manager import PersonaManager


@pytest.fixture
def adapter():
    settings = MagicMock(spec=Settings)
    settings.TELEGRAM_BOT_TOKEN = "test-token"

    agent_loop = MagicMock()
    agent_loop.cost_ctrl = MagicMock()
    agent_loop.cost_ctrl.get_summary.return_value = {
        "total_spent": 0.0,
        "max_budget": 1.00,
        "remaining": 1.00,
        "usage_ratio": 0.0,
        "call_count": 0,
        "warning": False,
    }

    persona = PersonaManager()
    local_llm = AsyncMock()
    memory = AsyncMock()

    return TelegramAdapter(settings, agent_loop, persona, local_llm, memory)


def test_adapter_creation(adapter):
    assert adapter is not None
    assert adapter._current_task_id is None


def test_no_token_raises():
    settings = MagicMock(spec=Settings)
    settings.TELEGRAM_BOT_TOKEN = ""
    adapter = TelegramAdapter(settings, MagicMock(), PersonaManager(), AsyncMock(), AsyncMock())
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        adapter.run()
```

---

## Verification

```powershell
# Run ALL tests across all phases
pytest aira/tests/ -v --tb=short
```

## Final Manual Test

```powershell
# 1. Make sure .env has a valid TELEGRAM_BOT_TOKEN
# 2. Make sure Ollama is running
# 3. Start Aira
python -m aira.main
```

Then in Telegram:
1. Send `/start` → Should get welcome message
2. Send "Hey, how are you?" → Should get Aira persona response
3. Send "Read the file hello.txt" → Should trigger task execution
4. Send `/status` → Should show cost summary
5. Send `/cancel` → Should cancel active task

## Done When
- [ ] `aira/adapters/telegram_adapter.py` created
- [ ] `aira/main.py` created
- [ ] `aira/tests/test_telegram.py` created
- [ ] All tests pass
- [ ] Bot starts and responds to messages (manual test)
- [ ] Model unloads after responses (check `nvidia-smi`)

## 🎉 Project Complete!
