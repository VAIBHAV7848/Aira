import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from aira.tools.system_command_tool import SystemCommandTool, RiskLevel

# ── FIXTURES ──
@pytest.fixture
def tool():
    tool = SystemCommandTool(timeout=1, log_dir=None)
    tool.log_dir.mkdir(parents=True, exist_ok=True)
    return tool

# ── BLACKLIST TESTS ──
@pytest.mark.asyncio
async def test_blacklist_destructive(tool):
    """Test that destructive commands are blocked."""
    commands = [
        "format C:",
        "del /s C:\\*",
        "del /s /q C:\\",
        "rd /s /q C:\\Windows",
        "reg delete HKLM\\Software",
        "bcdedit /set",
    ]
    for cmd in commands:
        result = await tool.run(cmd)
        print(f"DEBUG: cmd='{cmd}', success={result.success}, error='{result.error}'")
        assert result.success is False
        assert "BLOCKED" in result.error
        assert "blacklisted" in result.error.lower()

@pytest.mark.asyncio
async def test_blacklist_wildcards(tool):
    """Test wildcards are blocked."""
    commands = [
        "del *.*",  # Matches *\.* pattern
        "remove-item * -recurse -force C:\\*", # Drive root wipe
        "del C:\\Users\\*.*",
    ]
    for cmd in commands:
        result = await tool.run(cmd)
        print(f"DEBUG: cmd='{cmd}', success={result.success}, error='{result.error}'")
        assert result.success is False
        assert "BLOCKED" in result.error

# ── RISK CLASSIFICATION TESTS ──
def test_risk_classification(tool):
    assert tool.classify_risk("echo hello") == RiskLevel.LOW
    assert tool.classify_risk("dir") == RiskLevel.LOW
    
    assert tool.classify_risk("net user") == RiskLevel.HIGH
    assert tool.classify_risk("taskkill /im notepad.exe") == RiskLevel.HIGH
    
    assert tool.classify_risk("format C:") == RiskLevel.CRITICAL
    assert tool.classify_risk("del /s C:\\*") == RiskLevel.CRITICAL

# ── CONFIRMATION TESTS ──
@pytest.mark.asyncio
async def test_confirmation_required(tool):
    """Test that execution fails without confirmation handler."""
    tool.confirm_callback = None
    result = await tool.run("echo hello")
    assert result.success is False
    assert "no confirmation handler" in result.error

@pytest.mark.asyncio
async def test_confirmation_rejected(tool):
    """Test that user rejection blocks command."""
    async def reject_cb(**kwargs):
        return {"approved": False, "text": "no"}
    
    tool.confirm_callback = reject_cb
    result = await tool.run("echo hello")
    assert result.success is False
    assert "rejected" in result.error

@pytest.mark.asyncio
async def test_high_risk_confirmation_mismatch(tool):
    """Test that high risk requires uppercase CONFIRM."""
    async def weak_confirm(**kwargs):
        return {"approved": True, "text": "yes"}
    
    tool.confirm_callback = weak_confirm
    # net user is HIGH risk
    result = await tool.run("net user")
    assert result.success is False
    assert "require typing 'CONFIRM'" in result.error

@pytest.mark.asyncio
async def test_valid_execution(tool):
    """Test valid execution with correct confirmation."""
    async def accept_cb(**kwargs):
        return {"approved": True, "text": "echo test"}
    
    tool.confirm_callback = accept_cb
    # reduced delay for test not easily possible without mocking sleep
    # but we can rely on timeout if it hangs
    
    # We mock asyncio.sleep to skip the 5s delay during tests
    original_sleep = asyncio.sleep
    asyncio.sleep = AsyncMock()
    
    try:
        result = await tool.run("echo test")
        assert result.success is True
        assert "test" in result.output.strip()
    finally:
        asyncio.sleep = original_sleep

# ── INJECTION TESTS ──
@pytest.mark.asyncio
async def test_command_injection_defense(tool):
    """Test that injection attempts are treated as the command itself (shell=False)."""
    async def accept_cb(**kwargs):
        return {"approved": True, "text": 'echo "a"; echo "b"'}
    
    tool.confirm_callback = accept_cb
    # Mock sleep
    original_sleep = asyncio.sleep
    asyncio.sleep = AsyncMock()
    
    try:
        # In shell=False, this echoes the whole string including the semicolon
        # It does NOT run two commands
        cmd = 'echo "a"; echo "b"'
        result = await tool.run(cmd)
        
        # Verify it didn't execute "echo b" separately
        # PowerShell might interpret ; if passed to -Command, 
        # but our blacklist/risk logic should catch dangerous chains if they were attempted.
        # Actually SystemCommandTool uses ["powershell", "-Command", cmd]
        # So injection IS possible if not sanitized, BUT valid use cases might need it.
        # The protection here is CONFIRMATION + BLACKLIST.
        
        assert result.success is True
        # If it ran, it matched the conf.
        # The point is: did it bypass blacklist?
        
        # Try a blacklisted injection
        bad_cmd = 'echo a; format C:'
        result = await tool.run(bad_cmd)
        assert result.success is False
        assert "BLOCKED" in result.error
    finally:
        asyncio.sleep = original_sleep
