"""
System read tool — read-only access to system information.
All user-provided inputs are sanitized before interpolation into commands.
"""

import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from aira.tools.registry import ToolResult

logger = logging.getLogger(__name__)

# Env vars that should be masked in output
SENSITIVE_VAR_NAMES = {
    "TELEGRAM_BOT_TOKEN", "EXTERNAL_LLM_API_KEY", "OPENAI_API_KEY",
    "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
    "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID",
    "GITHUB_TOKEN", "GH_TOKEN", "AZURE_KEY",
}


def _sanitize_query(query: str, max_len: int = 100) -> str:
    """Sanitize user query for safe interpolation into PowerShell commands.
    
    Strips all characters except alphanumeric, spaces, dots, hyphens, and underscores.
    Prevents command injection via $, `, |, ;, {, }, quotes, etc.
    """
    if not query:
        return ""
    # Whitelist: alphanumeric, spaces, dots, hyphens, underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s._\-]', '', query)
    return sanitized[:max_len].strip()


def _safe_read_env() -> dict:
    """Minimal environment for read-only subprocess calls. No secrets."""
    return {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "SYSTEMDRIVE": os.environ.get("SYSTEMDRIVE", "C:"),
        "COMSPEC": os.environ.get("COMSPEC", r"C:\Windows\system32\cmd.exe"),
        "TEMP": os.environ.get("TEMP", r"C:\Users\Default\AppData\Local\Temp"),
        "TMP": os.environ.get("TMP", r"C:\Users\Default\AppData\Local\Temp"),
        "PATH": os.environ.get("SYSTEMROOT", r"C:\Windows") + r"\System32;"
                + os.environ.get("SYSTEMROOT", r"C:\Windows"),
    }



class SystemReadTool:
    """Reads any file on the system and retrieves system information."""

    def __init__(self, max_chars: int = 20000):
        self.max_chars = max_chars

    def run(self, action: str, path: str = "", query: str = "") -> ToolResult:
        """
        Execute a read-only system action.

        Args:
            action: One of: read_file, list_dir, system_info, processes,
                    disk_usage, env_var, search_files, gpu_info
            path: File/directory path (for read_file, list_dir, search_files)
            query: Search query or env var name
        """
        try:
            if action == "read_file":
                return self._read_file(path)
            elif action == "list_dir":
                return self._list_dir(path)
            elif action == "system_info":
                return self._system_info()
            elif action == "processes":
                return self._processes(query)
            elif action == "disk_usage":
                return self._disk_usage()
            elif action == "env_var":
                return self._env_var(query)
            elif action == "search_files":
                return self._search_files(path, query)
            elif action == "gpu_info":
                return self._gpu_info()
            elif action == "network_info":
                return self._network_info()
            elif action == "installed_apps":
                return self._installed_apps(query)
            elif action in ("battery_info", "battery", "battery_percentage"):
                return self._battery_info()
            else:
                return ToolResult(
                    success=False, output="",
                    error=f"Unknown action: {action}. Available: read_file, list_dir, "
                          f"system_info, processes, disk_usage, env_var, search_files, "
                          f"gpu_info, network_info, installed_apps, battery_info"
                )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _read_file(self, path: str) -> ToolResult:
        """Read any file on the system."""
        if not path:
            return ToolResult(success=False, output="", error="No path provided")

        resolved = Path(path).resolve()
        if not resolved.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        if not resolved.is_file():
            return ToolResult(success=False, output="", error=f"Not a file: {path}")

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            truncated = len(content) > self.max_chars
            if truncated:
                content = content[:self.max_chars] + "\n\n... [TRUNCATED]"
            return ToolResult(
                success=True, output=content,
                metadata={"path": str(resolved), "size": resolved.stat().st_size, "truncated": truncated}
            )
        except Exception as e:
            # Try binary read for non-text files
            try:
                size = resolved.stat().st_size
                return ToolResult(
                    success=True,
                    output=f"[Binary file: {resolved.name}, size: {size:,} bytes]",
                    metadata={"path": str(resolved), "size": size, "binary": True}
                )
            except Exception:
                return ToolResult(success=False, output="", error=f"Cannot read: {e}")

    def _list_dir(self, path: str) -> ToolResult:
        """List directory contents."""
        if not path:
            path = "."
        resolved = Path(path).resolve()
        if not resolved.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {path}")
        if not resolved.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {path}")

        entries = []
        try:
            for entry in sorted(resolved.iterdir()):
                try:
                    if entry.is_dir():
                        child_count = sum(1 for _ in entry.iterdir()) if entry.is_dir() else 0
                        entries.append(f"📁 {entry.name}/ ({child_count} items)")
                    else:
                        size = entry.stat().st_size
                        entries.append(f"📄 {entry.name} ({self._fmt_size(size)})")
                except PermissionError:
                    entries.append(f"🔒 {entry.name} [access denied]")
                except Exception:
                    entries.append(f"❓ {entry.name}")

                if len(entries) >= 100:
                    entries.append(f"... and more (showing first 100)")
                    break
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Access denied: {path}")

        output = f"📂 {resolved}\n\n" + "\n".join(entries)
        return ToolResult(success=True, output=output, metadata={"path": str(resolved), "count": len(entries)})

    def _system_info(self) -> ToolResult:
        """Get comprehensive system information."""
        import shutil

        info_lines = [
            f"🖥️  System Information",
            f"",
            f"OS: {platform.system()} {platform.release()} ({platform.version()})",
            f"Machine: {platform.machine()}",
            f"Processor: {platform.processor()}",
            f"Python: {sys.version}",
            f"Username: {os.getlogin()}",
            f"Home: {Path.home()}",
            f"CWD: {Path.cwd()}",
        ]

        # CPU info
        try:
            cpu_count = os.cpu_count()
            info_lines.append(f"CPU Cores: {cpu_count}")
        except Exception:
            pass

        # Memory (via PowerShell)
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                total_mem = int(result.stdout.strip())
                info_lines.append(f"RAM: {self._fmt_size(total_mem)}")
        except Exception:
            pass

        # Disk
        try:
            total, used, free = shutil.disk_usage("/")
            info_lines.append(f"Disk (C:): {self._fmt_size(used)} used / {self._fmt_size(total)} total ({self._fmt_size(free)} free)")
        except Exception:
            pass

        # Battery
        try:
            bat = subprocess.run(
                ["powershell", "-Command",
                 "(Get-WmiObject Win32_Battery | Select-Object EstimatedChargeRemaining, BatteryStatus) | ForEach-Object { $_.EstimatedChargeRemaining.ToString() + '|' + $_.BatteryStatus.ToString() }"],
                capture_output=True, text=True, timeout=5
            )
            if bat.returncode == 0 and bat.stdout.strip():
                parts = bat.stdout.strip().split('|')
                pct = parts[0] if parts else '?'
                status_code = int(parts[1]) if len(parts) > 1 else 0
                status_map = {1: 'Discharging', 2: 'Plugged in', 3: 'Fully charged', 4: 'Low', 5: 'Critical'}
                status = status_map.get(status_code, 'Unknown')
                info_lines.append(f"🔋 Battery: {pct}% ({status})")
            else:
                info_lines.append("🔋 Battery: Not available (desktop PC?)")
        except Exception:
            info_lines.append("🔋 Battery: Could not read")

        return ToolResult(success=True, output="\n".join(info_lines))

    def _processes(self, query: str = "") -> ToolResult:
        """List running processes, optionally filtered."""
        try:
            safe_query = _sanitize_query(query)
            cmd = 'Get-Process | Sort-Object -Property WorkingSet64 -Descending | Select-Object -First 30 Name, Id, @{N="RAM(MB)";E={[math]::Round($_.WorkingSet64/1MB,1)}}, CPU | Format-Table -AutoSize'
            if safe_query:
                cmd = f'Get-Process | Where-Object {{$_.Name -like "*{safe_query}*"}} | Select-Object Name, Id, @{{N="RAM(MB)";E={{[math]::Round($_.WorkingSet64/1MB,1)}}}}, CPU | Format-Table -AutoSize'

            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True, text=True, timeout=10,
                env=_safe_read_env(),
            )
            output = result.stdout.strip() if result.stdout else "No processes found"
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _disk_usage(self) -> ToolResult:
        """Get disk usage for all drives."""
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{N='Used(GB)';E={[math]::Round($_.Used/1GB,1)}}, @{N='Free(GB)';E={[math]::Round($_.Free/1GB,1)}}, @{N='Total(GB)';E={[math]::Round(($_.Used+$_.Free)/1GB,1)}} | Format-Table -AutoSize"],
                capture_output=True, text=True, timeout=10
            )
            return ToolResult(success=True, output=result.stdout.strip())
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _env_var(self, name: str) -> ToolResult:
        """Get an environment variable value. Sensitive values are masked."""
        def _mask(key: str, val: str) -> str:
            upper = key.upper()
            for s in SENSITIVE_VAR_NAMES:
                if s in upper:
                    return f"{key}=***MASKED***"
            return f"{key}={val[:100]}"

        if not name:
            env_list = [_mask(k, v) for k, v in sorted(os.environ.items())]
            return ToolResult(success=True, output="\n".join(env_list[:50]))
        value = os.environ.get(name)
        if value is None:
            return ToolResult(success=False, output="", error=f"Env var not found: {name}")
        return ToolResult(success=True, output=_mask(name, value))

    def _search_files(self, path: str, query: str) -> ToolResult:
        """Search for files by name pattern."""
        if not query:
            return ToolResult(success=False, output="", error="No search query provided")
        safe_query = _sanitize_query(query, max_len=50)
        if not safe_query:
            return ToolResult(success=False, output="", error="Query contains only special characters")
        search_dir = Path(path).resolve() if path else Path.home()

        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f'Get-ChildItem -Path "{search_dir}" -Filter "*{safe_query}*" -Recurse -ErrorAction SilentlyContinue -Depth 3 | Select-Object -First 20 FullName, Length, LastWriteTime | Format-Table -AutoSize'],
                capture_output=True, text=True, timeout=15,
                env=_safe_read_env(),
            )
            output = result.stdout.strip() if result.stdout else "No files found"
            return ToolResult(success=True, output=output)
        except subprocess.TimeoutExpired:
            return ToolResult(success=True, output="Search timed out — try a narrower path")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _gpu_info(self) -> ToolResult:
        """Get GPU information via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 6:
                    output = (
                        f"🎮 GPU: {parts[0]}\n"
                        f"VRAM: {parts[2]}MB used / {parts[1]}MB total ({parts[3]}MB free)\n"
                        f"Temperature: {parts[4]}°C\n"
                        f"Utilization: {parts[5]}%"
                    )
                    return ToolResult(success=True, output=output)
            return ToolResult(success=True, output=result.stdout.strip())
        except FileNotFoundError:
            return ToolResult(success=False, output="", error="nvidia-smi not found (no NVIDIA GPU?)")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _network_info(self) -> ToolResult:
        """Get network adapter information."""
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -ne '127.0.0.1'} | Select-Object InterfaceAlias, IPAddress | Format-Table -AutoSize"],
                capture_output=True, text=True, timeout=5
            )
            return ToolResult(success=True, output=result.stdout.strip())
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _installed_apps(self, query: str = "") -> ToolResult:
        """List installed applications."""
        try:
            safe_query = _sanitize_query(query)
            filter_clause = ""
            if safe_query:
                filter_clause = f' | Where-Object {{$_.DisplayName -like "*{safe_query}*"}}'
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f'Get-ItemProperty HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*{filter_clause} | Select-Object -First 30 DisplayName, DisplayVersion, InstallDate | Sort-Object DisplayName | Format-Table -AutoSize'],
                capture_output=True, text=True, timeout=10,
                env=_safe_read_env(),
            )
            return ToolResult(success=True, output=result.stdout.strip() or "No apps found")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _battery_info(self) -> ToolResult:
        """Get battery percentage and status."""
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "(Get-WmiObject Win32_Battery | Select-Object EstimatedChargeRemaining, BatteryStatus) | ForEach-Object { $_.EstimatedChargeRemaining.ToString() + '|' + $_.BatteryStatus.ToString() }"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split('|')
                pct = parts[0] if parts else '?'
                status_code = int(parts[1]) if len(parts) > 1 else 0
                status_map = {1: 'Discharging', 2: 'Plugged in', 3: 'Fully charged', 4: 'Low', 5: 'Critical'}
                status = status_map.get(status_code, 'Unknown')
                return ToolResult(success=True, output=f"🔋 Battery: {pct}% ({status})")
            else:
                return ToolResult(success=True, output="🔋 Battery: Not available (desktop PC?)")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Battery check failed: {e}")

    @staticmethod
    def _fmt_size(size_bytes: int) -> str:
        """Format bytes to human readable."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
