# Phase 0 — Project Skeleton

## Goal
Create the bare folder structure, virtual environment, and `.gitignore`. NO code yet.

## Prerequisites
- Python 3.11+ installed
- Ollama installed

## Steps

### Step 1: Create all directories
Run from `d:\Aira`:

```powershell
mkdir aira
mkdir aira\agent
mkdir aira\persona
mkdir aira\llm
mkdir aira\adapters
mkdir aira\tools
mkdir aira\security
mkdir aira\memory
mkdir aira\heartbeat
mkdir aira\config
mkdir aira\workspace
mkdir aira\logs
mkdir aira\tests
```

### Step 2: Create all `__init__.py` files
Create an **empty** `__init__.py` in every package directory:

```powershell
New-Item -ItemType File -Path aira\__init__.py
New-Item -ItemType File -Path aira\agent\__init__.py
New-Item -ItemType File -Path aira\persona\__init__.py
New-Item -ItemType File -Path aira\llm\__init__.py
New-Item -ItemType File -Path aira\adapters\__init__.py
New-Item -ItemType File -Path aira\tools\__init__.py
New-Item -ItemType File -Path aira\security\__init__.py
New-Item -ItemType File -Path aira\memory\__init__.py
New-Item -ItemType File -Path aira\heartbeat\__init__.py
New-Item -ItemType File -Path aira\config\__init__.py
New-Item -ItemType File -Path aira\tests\__init__.py
```

### Step 3: Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 4: Create `.gitignore`
Create file `d:\Aira\.gitignore` with this content:

```
.venv/
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
aira.db
aira/logs/*.json
aira/workspace/*
!aira/workspace/.gitkeep
.mypy_cache/
.pytest_cache/
.ruff_cache/
```

### Step 5: Create `.env.example`
Create file `d:\Aira\.env.example` with this content:

```
TELEGRAM_BOT_TOKEN=your-telegram-bot-token-here
WORKSPACE_ROOT=./aira/workspace
MAX_COST_USD=1.00
COST_WARNING_THRESHOLD=0.80
MAX_ITERATIONS=15
SUBPROCESS_TIMEOUT=30
MAX_OUTPUT_CHARS=20000
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
EXTERNAL_LLM_MODEL=
EXTERNAL_LLM_API_KEY=
LOG_DIR=./aira/logs
DB_PATH=./aira.db
```

### Step 6: Create `requirements.txt`
Create file `d:\Aira\requirements.txt`:

```
python-telegram-bot==21.*
aiosqlite==0.20.*
litellm==1.*
tiktoken==0.7.*
python-dotenv==1.*
click==8.*
httpx==0.27.*
pytest==8.*
pytest-asyncio==0.23.*
pytest-cov==5.*
ruff==0.4.*
mypy==1.10.*
```

### Step 7: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 8: Create workspace gitkeep

```powershell
New-Item -ItemType File -Path aira\workspace\.gitkeep
```

## Verification
Run this to confirm the structure:

```powershell
Get-ChildItem -Recurse -Name -Include "*.py","*.txt","*.example","*.gitignore","*.gitkeep" | Sort-Object
```

Expected output should show all `__init__.py` files, `requirements.txt`, `.env.example`, `.gitignore`, and `.gitkeep`.

## Done When
- [ ] All directories exist
- [ ] All `__init__.py` files exist
- [ ] `.venv` created and activated
- [ ] `pip install -r requirements.txt` succeeds
- [ ] `.gitignore` created
- [ ] `.env.example` created

## Next Phase
→ `phase_01_config.md`
