"""
register_startup.py — Configures the OS filewatcher to run silently on Windows login.
"""

import os
from pathlib import Path
import click

def main():
    click.secho("=== REPO_KNOWLEDGE Windows Startup Configurator ===", fg="cyan", bold=True)

    appdata = os.getenv("APPDATA")
    if not appdata:
        click.secho("Error: APPDATA environment variable not found. Are you on Windows?", fg="red")
        return

    startup_dir = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    if not startup_dir.exists():
        click.secho(f"Error: Windows Startup folder not found at: {startup_dir}", fg="red")
        return

    repo_root = Path(__file__).resolve().parent
    watcher_path = repo_root / "watcher.py"
    pythonw_path = repo_root / ".venv" / "Scripts" / "pythonw.exe"

    if not watcher_path.exists():
        click.secho(f"Error: watcher.py not found at: {watcher_path}", fg="red")
        return

    if not pythonw_path.exists():
        click.secho(f"Error: pythonw.exe not found under .venv at: {pythonw_path}", fg="red")
        return

    bat_path = startup_dir / "start_repo_knowledge_watcher.bat"
    
    bat_content = f"""@echo off
rem Start REPO_KNOWLEDGE file watcher silently in the background on login
start /B "" "{pythonw_path}" "{watcher_path}"
"""

    try:
        bat_path.write_text(bat_content, encoding="utf-8")
        click.secho(f"\n[OK] Successfully registered silent background filewatcher at startup!", fg="green")
        click.echo(f"Startup Script Location: {bat_path}")
        click.echo(f"It will execute silently (no console window) using pythonw.exe on your next Windows login.")
    except Exception as e:
        click.secho(f"Error writing startup script: {e}", fg="red")

if __name__ == "__main__":
    main()
