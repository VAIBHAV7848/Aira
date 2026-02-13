# Aira — Master Build Guide

> **Follow these phases in order. Do NOT skip ahead. Every phase has its own instruction file with complete, copy-paste code and tests.**

---

## Quick Start

```powershell
cd d:\Aira
```

## Build Order

| Phase | File | What You Build | Est. Time |
|---|---|---|---|
| **0** | [phase_00_project_skeleton.md](file:///d:/Aira/phases/phase_00_project_skeleton.md) | Folders, venv, `.gitignore`, `requirements.txt` | 30 min |
| **1** | [phase_01_config.md](file:///d:/Aira/phases/phase_01_config.md) | `settings.py` — loads `.env` into typed dataclass | 20 min |
| **2** | [phase_02_security.md](file:///d:/Aira/phases/phase_02_security.md) | `path_guard`, `injection_guard`, `outbound_guard`, `cost_controller` | 1.5 hrs |
| **3** | [phase_03_state_machine.md](file:///d:/Aira/phases/phase_03_state_machine.md) | Deterministic FSM with persist + transition whitelist | 1 hr |
| **4** | [phase_04_tools.md](file:///d:/Aira/phases/phase_04_tools.md) | `file_read`, `file_write`, `python_runner`, `registry` | 1.5 hrs |
| **5** | [phase_05_memory.md](file:///d:/Aira/phases/phase_05_memory.md) | SQLite memory store with token-capped summaries | 1 hr |
| **6** | [phase_06_local_llm.md](file:///d:/Aira/phases/phase_06_local_llm.md) | Lazy-loading Ollama/Qwen wrapper | 1 hr |
| **7** | [phase_07_external_planner.md](file:///d:/Aira/phases/phase_07_external_planner.md) | Cost-controlled external LLM planner | 1 hr |
| **8** | [phase_08_agent_loop.md](file:///d:/Aira/phases/phase_08_agent_loop.md) | Core agent loop tying everything together | 2 hrs |
| **9** | [phase_09_persona.md](file:///d:/Aira/phases/phase_09_persona.md) | Aira personality wrapper (zero execution authority) | 1 hr |
| **10** | [phase_10_telegram.md](file:///d:/Aira/phases/phase_10_telegram.md) | Telegram adapter + `main.py` entry point | 1.5 hrs |

**Total estimated: ~12 hours**

---

## Rules for Each Phase

1. **Open the phase file** — it has the EXACT code to copy
2. **Create the files** listed in that phase
3. **Run the tests** for that phase
4. **ALL tests must pass** before moving to the next phase
5. **If a test fails** — fix it before continuing

## Run ALL Tests

```powershell
cd d:\Aira
.\.venv\Scripts\Activate.ps1
pytest aira/tests/ -v --tb=short
```

## Start Aira

```powershell
# After ALL phases done:
python -m aira.main
```

---

## Key Documents

| Document | Purpose |
|---|---|
| [implementation_plan.md](file:///d:/Aira/implementation_plan.md) | Full PRD: tech stack, security architecture, acceptance criteria |
| [BUILD_GUIDE.md](file:///d:/Aira/BUILD_GUIDE.md) | This file — master index of all phases |
| `phases/` folder | Individual phase instructions with complete code |
