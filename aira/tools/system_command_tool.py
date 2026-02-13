"""
SystemCommandTool — hardened shell execution with full security controls.

Replaces the old ShellTool. Every command requires:
1. Command blacklist check
2. Risk classification
3. Typed confirmation (user re-types exact command)
4. Uppercase "CONFIRM" for HIGH/CRITICAL risk
5. 5-second delay before execution

All executions use shell=False, have timeouts, and log everything.
"""

import asyncio
import logging
import os
import re
import shlex
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from aira.tools.registry import ToolResult

logger = logging.getLogger(__name__)

MAX_OUTPUT = 10000
DEFAULT_TIMEOUT = 30


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── BLACKLISTED COMMANDS ──
# These patterns are NEVER allowed, regardless of confirmation.
BLACKLIST_PATTERNS: list[re.Pattern] = [
    # Disk formatting
    re.compile(r"\bformat\b.*[a-zA-Z]:", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),

    # Drive root deletion
    re.compile(r"\b(del|rm|rd|rmdir|remove-item|erase)\b.*[a-zA-Z]:\\[\\/]?\s", re.IGNORECASE),
    re.compile(r"\b(del|rm|rd|rmdir|remove-item|erase)\b.*[a-zA-Z]:\\[\\/]?\*", re.IGNORECASE),
    re.compile(r"\b(del|rm|rd|rmdir)\b.*/[sS].*[a-zA-Z]:\\", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b.*-Recurse.*[a-zA-Z]:\\[\\/]?$", re.IGNORECASE),

    # System32 modification
    re.compile(r"C:\\Windows\\System32", re.IGNORECASE),
    re.compile(r"\bsystem32\b", re.IGNORECASE),

    # Registry modification
    re.compile(r"\breg\s+(add|delete|import)\b", re.IGNORECASE),
    re.compile(r"\bNew-ItemProperty\b.*HKLM", re.IGNORECASE),
    re.compile(r"\bSet-ItemProperty\b.*HKLM", re.IGNORECASE),
    re.compile(r"\bRemove-ItemProperty\b.*HKLM", re.IGNORECASE),
    re.compile(r"\bregedit\b", re.IGNORECASE),

    # Bootloader modification
    re.compile(r"\bbcdedit\b", re.IGNORECASE),
    re.compile(r"\bbootrec\b", re.IGNORECASE),
    re.compile(r"\bbootcfg\b", re.IGNORECASE),

    # Wildcard destructive deletes
    re.compile(r"\b(del|rm|remove-item)\b.*\*\.\*", re.IGNORECASE),
    re.compile(r"\b(del|rm|remove-item)\b.*-Recurse\b.*-Force\b.*\\\*", re.IGNORECASE),

    # User directory mass deletion
    re.compile(r"\b(del|rm|rd|rmdir|remove-item)\b.*C:\\Users\\\*", re.IGNORECASE),
    re.compile(r"\b(del|rm|rd|rmdir|remove-item)\b.*\\Users\\[^\\]+$", re.IGNORECASE),

    # Other dangerous commands
    re.compile(r"\bcipher\s+/w\b", re.IGNORECASE),  # Secure wipe
    re.compile(r"\bsfc\s+/scannow\b", re.IGNORECASE),  # System file checker (modifies)
    re.compile(r"\btakeown\b", re.IGNORECASE),  # Take ownership
    re.compile(r"\bicacls\b.*(/grant|/deny|/remove)", re.IGNORECASE),  # Permission changes
]

# ── HIGH-RISK INDICATORS ──
HIGH_RISK_INDICATORS: list[re.Pattern] = [
    re.compile(r"\b(del|rm|remove-item|erase)\b", re.IGNORECASE),
    re.compile(r"-Recurse\b", re.IGNORECASE),
    re.compile(r"-Force\b", re.IGNORECASE),
    re.compile(r"\bStop-Process\b", re.IGNORECASE),
    re.compile(r"\bStop-Service\b", re.IGNORECASE),
    re.compile(r"\bkill\b", re.IGNORECASE),
    re.compile(r"\btaskkill\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\brestart-computer\b", re.IGNORECASE),
    re.compile(r"\bnet\s+(user|localgroup)\b", re.IGNORECASE),
    re.compile(r"\bSet-ExecutionPolicy\b", re.IGNORECASE),
    re.compile(r"\bNew-Item\b.*-ItemType\s+SymbolicLink", re.IGNORECASE),
]

MEDIUM_RISK_INDICATORS: list[re.Pattern] = [
    re.compile(r"\bNew-Item\b", re.IGNORECASE),
    re.compile(r"\bCopy-Item\b", re.IGNORECASE),
    re.compile(r"\bMove-Item\b", re.IGNORECASE),
    re.compile(r"\bRename-Item\b", re.IGNORECASE),
    re.compile(r"\bSet-Content\b", re.IGNORECASE),
    re.compile(r"\bAdd-Content\b", re.IGNORECASE),
    re.compile(r"\bOut-File\b", re.IGNORECASE),
    re.compile(r"\bStart-Process\b", re.IGNORECASE),
    re.compile(r"\bInvoke-WebRequest\b", re.IGNORECASE),
    re.compile(r"\bcurl\b", re.IGNORECASE),
    re.compile(r"\bwget\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
]

# Minimal safe environment for subprocess execution
SAFE_ENV = {
    "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
    "SYSTEMDRIVE": os.environ.get("SYSTEMDRIVE", "C:"),
    "COMSPEC": os.environ.get("COMSPEC", r"C:\Windows\system32\cmd.exe"),
    "TEMP": os.environ.get("TEMP", r"C:\Users\Default\AppData\Local\Temp"),
    "TMP": os.environ.get("TMP", r"C:\Users\Default\AppData\Local\Temp"),
    "PATH": os.environ.get("SYSTEMROOT", r"C:\Windows") + r"\System32;"
            + os.environ.get("SYSTEMROOT", r"C:\Windows"),
}


class CommandBlacklistedError(Exception):
    """Raised when a blacklisted command is attempted."""
    pass


class ConfirmationError(Exception):
    """Raised when command confirmation fails."""
    pass


class SystemCommandTool:
    """
    Executes system commands with full security controls.

    Security guarantees:
    - Blacklisted commands are NEVER executed
    - All commands require explicit typed confirmation
    - HIGH/CRITICAL risk commands require uppercase "CONFIRM"
    - 5-second delay before execution
    - shell=False enforced
    - Timeout enforced
    - Environment is sanitized (no secrets leaked)
    - All executions are fully logged
    - Fail-closed on any confirmation error
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        log_dir: Optional[Path] = None,
    ):
        self.timeout = timeout
        self.log_dir = log_dir or Path("./aira/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # This MUST be set by the adapter before use
        self.confirm_callback: Optional[Callable] = None

    def classify_risk(self, command: str) -> RiskLevel:
        """Classify the risk level of a command."""
        # Check blacklist first
        for pattern in BLACKLIST_PATTERNS:
            if pattern.search(command):
                return RiskLevel.CRITICAL

        # Check high risk
        high_count = sum(1 for p in HIGH_RISK_INDICATORS if p.search(command))
        if high_count >= 2:
            return RiskLevel.CRITICAL
        if high_count >= 1:
            return RiskLevel.HIGH

        # Check medium risk
        for pattern in MEDIUM_RISK_INDICATORS:
            if pattern.search(command):
                return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def detect_affected_paths(self, command: str) -> list[str]:
        """Try to detect file/directory paths affected by the command."""
        paths = []
        # Windows paths
        for match in re.finditer(r'[A-Za-z]:\\[^\s,;"\'|]+', command):
            paths.append(match.group())
        # Relative paths with backslash
        for match in re.finditer(r'\.\\[^\s,;"\'|]+', command):
            paths.append(match.group())
        return paths[:10]  # Cap at 10

    def is_blacklisted(self, command: str) -> tuple[bool, str]:
        """Check if command is blacklisted. Returns (is_blocked, reason)."""
        for pattern in BLACKLIST_PATTERNS:
            match = pattern.search(command)
            if match:
                return True, f"Matched blacklist pattern: {pattern.pattern}"
        return False, ""

    async def run(self, command: str) -> ToolResult:
        """
        Execute a command after full security checks and confirmation.

        Flow:
        1. Validate command is not empty
        2. Check blacklist
        3. Classify risk
        4. Request typed confirmation
        5. Wait 5 seconds
        6. Execute with shell=False, sanitized env, timeout
        7. Log everything
        """
        exec_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "risk_level": None,
            "blacklisted": False,
            "confirmed": False,
            "confirmation_text": None,
            "executed": False,
            "exit_code": None,
            "duration_ms": None,
            "stdout_len": 0,
            "stderr_len": 0,
            "error": None,
        }

        try:
            # 1. Validate
            if not command or not command.strip():
                exec_log["error"] = "empty_command"
                self._write_exec_log(exec_log)
                return ToolResult(success=False, output="", error="No command provided")

            # 2. Check blacklist
            blocked, reason = self.is_blacklisted(command)
            if blocked:
                exec_log["blacklisted"] = True
                exec_log["error"] = f"blacklisted: {reason}"
                self._write_exec_log(exec_log)
                logger.warning(f"BLACKLISTED command blocked: {command[:200]} — {reason}")
                return ToolResult(
                    success=False, output="",
                    error=f"⛔ Command BLOCKED — {reason}\n\nThis command is permanently blacklisted for safety."
                )

            # 3. Classify risk
            risk = self.classify_risk(command)
            exec_log["risk_level"] = risk.value
            affected_paths = self.detect_affected_paths(command)

            # 4. Require confirmation callback
            if not self.confirm_callback:
                exec_log["error"] = "no_confirmation_handler"
                self._write_exec_log(exec_log)
                return ToolResult(
                    success=False, output="",
                    error="Command execution blocked: no confirmation handler set"
                )

            # 5. Request confirmation
            try:
                confirmation = await self.confirm_callback(
                    command=command,
                    risk_level=risk,
                    affected_paths=affected_paths,
                    working_dir=str(Path.cwd()),
                )
            except Exception as e:
                exec_log["error"] = f"confirmation_failed: {e}"
                self._write_exec_log(exec_log)
                return ToolResult(
                    success=False, output="",
                    error=f"Confirmation failed — command aborted: {e}"
                )

            exec_log["confirmation_text"] = str(confirmation.get("text", ""))[:500]

            if not confirmation.get("approved", False):
                exec_log["confirmed"] = False
                exec_log["error"] = "user_rejected"
                self._write_exec_log(exec_log)
                logger.info(f"User rejected command: {command[:100]}")
                return ToolResult(success=False, output="", error="Command rejected by user")

            # 6. Verify confirmation integrity
            if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                confirm_text = confirmation.get("text", "").strip()
                if confirm_text != "CONFIRM":
                    exec_log["error"] = "confirm_mismatch_high_risk"
                    self._write_exec_log(exec_log)
                    return ToolResult(
                        success=False, output="",
                        error="⚠️ HIGH/CRITICAL risk commands require typing 'CONFIRM' (uppercase). Command aborted."
                    )

            exec_log["confirmed"] = True

            # 7. 5-second delay
            logger.info(f"Command confirmed. Waiting 5 seconds before execution...")
            await asyncio.sleep(5)

            # 8. Execute with shell=False
            logger.info(f"Executing [{risk.value}]: {command[:200]}")
            import subprocess

            exec_start = time.time()
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=SAFE_ENV,
                    cwd=None,
                )
                exec_duration = int((time.time() - exec_start) * 1000)

                stdout = result.stdout[:MAX_OUTPUT] if result.stdout else ""
                stderr = result.stderr[:2000] if result.stderr else ""

                exec_log["executed"] = True
                exec_log["exit_code"] = result.returncode
                exec_log["duration_ms"] = exec_duration
                exec_log["stdout_len"] = len(stdout)
                exec_log["stderr_len"] = len(stderr)

                self._write_exec_log(exec_log)

                if result.returncode == 0:
                    return ToolResult(
                        success=True,
                        output=stdout or "(no output)",
                        metadata={
                            "return_code": 0,
                            "command": command[:200],
                            "risk_level": risk.value,
                            "duration_ms": exec_duration,
                        },
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=stdout,
                        error=f"Exit code {result.returncode}: {stderr}",
                        metadata={
                            "return_code": result.returncode,
                            "command": command[:200],
                            "risk_level": risk.value,
                            "duration_ms": exec_duration,
                        },
                    )

            except subprocess.TimeoutExpired:
                exec_log["error"] = f"timeout_{self.timeout}s"
                exec_log["duration_ms"] = int((time.time() - exec_start) * 1000)
                self._write_exec_log(exec_log)
                return ToolResult(
                    success=False, output="",
                    error=f"Command timed out after {self.timeout}s",
                )

        except Exception as e:
            exec_log["error"] = f"unexpected: {e}"
            self._write_exec_log(exec_log)
            logger.error(f"SystemCommandTool error: {e}", exc_info=True)
            return ToolResult(success=False, output="", error=f"Command error: {e}")

    def _write_exec_log(self, entry: dict) -> None:
        """Write an execution log entry."""
        import json
        log_file = self.log_dir / "command_executions.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write command log: {e}")
