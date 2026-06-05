#!/usr/bin/env python
"""
manage.py — Control script to start, stop, restart, or toggle background services (Watcher & Web UI/MCP server).
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent
PYTHONW_PATH = REPO_ROOT / ".venv" / "Scripts" / "pythonw.exe"
PYTHON_PATH = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
WATCHER_PATH = REPO_ROOT / "watcher.py"

# Self-re-execution check: automatically switch to virtual environment python if run globally
if sys.executable != str(PYTHON_PATH) and PYTHON_PATH.exists():
    os.execv(str(PYTHON_PATH), [str(PYTHON_PATH)] + sys.argv)

def check_startup_registration():
    """Ensure that the filewatcher is registered in Windows Startup folder."""
    appdata = os.getenv("APPDATA")
    if not appdata:
        return False
    
    startup_dir = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    bat_path = startup_dir / "start_repo_knowledge_watcher.bat"
    
    if not bat_path.exists():
        print("[INFO] Background watcher is not registered in Windows startup registry. Registering...")
        try:
            # Run the register_startup.py script
            subprocess.run([str(PYTHON_PATH), "register_startup.py"], check=True)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to run register_startup.py: {e}")
            return False
    return True

def find_process_by_cmdline(substring):
    """Find the PID of a running python process matching a specific command line substring."""
    try:
        # Use PowerShell to query process command lines, which works reliably on all Windows 10/11 systems
        cmd = f'powershell -Command "Get-CimInstance Win32_Process -Filter \\"Name=\'python.exe\' or Name=\'pythonw.exe\'\\" | Where-Object {{ $_.CommandLine -like \'*{substring}*\' }} | Select-Object -ExpandProperty ProcessId"'
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if res.returncode == 0:
            pids = res.stdout.strip().split()
            for pid in pids:
                if pid.isdigit():
                    return int(pid)
    except Exception:
        pass
    return None

def kill_process(pid):
    """Force kill a process by PID."""
    try:
        subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
        return True
    except Exception:
        return False

def get_status():
    """Get the running status of the watcher and server."""
    watcher_pid = find_process_by_cmdline("watcher.py")
    server_pid = find_process_by_cmdline("repo_knowledge.web_ui.server")
    return watcher_pid, server_pid

def start_services():
    """Start watcher and server in the background."""
    print("=== Starting REPO_KNOWLEDGE Services ===")
    
    # 1. Check startup registration
    check_startup_registration()
    
    watcher_pid, server_pid = get_status()
    
    # 2. Start Filewatcher
    if watcher_pid:
        print(f"[OK] Filewatcher is already running (PID: {watcher_pid})")
    else:
        print("[STARTING] Launching background Filewatcher...")
        if not PYTHONW_PATH.exists():
            print(f"[ERROR] Python virtual environment executable not found at: {PYTHONW_PATH}")
            return
        try:
            # Spawn process silently with no terminal window
            subprocess.Popen(
                [str(PYTHONW_PATH), str(WATCHER_PATH)],
                close_fds=True,
                creationflags=0x08000000 # CREATE_NO_WINDOW
            )
            time.sleep(0.5)
            new_pid = find_process_by_cmdline("watcher.py")
            if new_pid:
                print(f"[OK] Filewatcher started in background (PID: {new_pid})")
            else:
                print("[ERROR] Filewatcher failed to start.")
        except Exception as e:
            print(f"[ERROR] Failed to start Filewatcher: {e}")

    # 3. Start Web UI / SSE MCP Server
    if server_pid:
        print(f"[OK] Web UI / MCP Server is already running (PID: {server_pid})")
    else:
        print("[STARTING] Launching background Web UI & MCP Server...")
        try:
            subprocess.Popen(
                [str(PYTHONW_PATH), "-m", "repo_knowledge.web_ui.server"],
                close_fds=True,
                creationflags=0x08000000 # CREATE_NO_WINDOW
            )
            time.sleep(0.5)
            new_pid = find_process_by_cmdline("repo_knowledge.web_ui.server")
            if new_pid:
                print(f"[OK] Web UI / MCP Server started in background (PID: {new_pid})")
            else:
                print("[ERROR] Web UI / MCP Server failed to start.")
        except Exception as e:
            print(f"[ERROR] Failed to start Web UI / MCP Server: {e}")

def stop_services():
    """Stop running watcher and server."""
    print("=== Stopping REPO_KNOWLEDGE Services ===")
    watcher_pid, server_pid = get_status()
    
    if watcher_pid:
        print(f"[STOPPING] Terminating Filewatcher (PID: {watcher_pid})...")
        if kill_process(watcher_pid):
            print("[OK] Filewatcher stopped.")
        else:
            print("[ERROR] Failed to stop Filewatcher.")
    else:
        print("[INFO] Filewatcher is not running.")
        
    if server_pid:
        print(f"[STOPPING] Terminating Web UI & MCP Server (PID: {server_pid})...")
        if kill_process(server_pid):
            print("[OK] Web UI & MCP Server stopped.")
        else:
            print("[ERROR] Failed to stop Web UI & MCP Server.")
    else:
        print("[INFO] Web UI & MCP Server is not running.")

def print_status_report():
    """Display running status of both services."""
    watcher_pid, server_pid = get_status()
    print("=== REPO_KNOWLEDGE Service Status ===")
    print(f"Filewatcher:      {'RUNNING (PID: ' + str(watcher_pid) + ')' if watcher_pid else 'STOPPED'}")
    print(f"Web UI / MCP:     {'RUNNING (PID: ' + str(server_pid) + ')' if server_pid else 'STOPPED'}")

def main():
    if len(sys.argv) < 2:
        # Toggle mode
        watcher_pid, server_pid = get_status()
        if watcher_pid or server_pid:
            # Stop if running
            stop_services()
        else:
            # Start if stopped
            start_services()
        return

    cmd = sys.argv[1].lower()
    if cmd == "start":
        start_services()
    elif cmd == "stop":
        stop_services()
    elif cmd == "restart":
        stop_services()
        time.sleep(0.5)
        start_services()
    elif cmd in ("status", "info"):
        print_status_report()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python manage.py [start|stop|restart|status]")
        print("If run without arguments, it toggles the services (stops if running, starts if stopped).")

if __name__ == "__main__":
    main()
