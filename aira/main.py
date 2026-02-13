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
            logging.FileHandler(log_dir / "aira.log", encoding="utf-8"),
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
