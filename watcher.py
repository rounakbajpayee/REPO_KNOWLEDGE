"""
watcher.py — OS-level file watcher for REPO_KNOWLEDGE.

Watches PROJECTS_ROOT recursively for file modifications. When supported files
are created, modified, or deleted, triggers a debounced incremental reindex
for the corresponding project after 5 seconds of idle time.
"""

import os
import sys
import threading
import time
from pathlib import Path

import click
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Resolve REPO_KNOWLEDGE/src path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from repo_knowledge.config import IGNORE_DIRS, PROJECTS_ROOT, SUPPORTED_EXTENSIONS
from repo_knowledge.knowledge import KnowledgeService


class ProjectChangeHandler(FileSystemEventHandler):
    def __init__(self, service: KnowledgeService, debounce_seconds: float = 5.0):
        self.service = service
        self.debounce_seconds = debounce_seconds
        self.pending_projects: set[str] = set()
        self.timer: threading.Timer | None = None
        self.lock = threading.Lock()

    def on_any_event(self, event):
        if event.is_directory:
            return

        # We care about modified, created, deleted, moved
        if event.event_type not in {"modified", "created", "deleted", "moved"}:
            return

        src_path = Path(event.src_path).resolve()
        self._process_path(src_path)

        if event.event_type == "moved":
            dest_path = Path(event.dest_path).resolve()
            self._process_path(dest_path)

    def _process_path(self, path: Path):
        # 1. Filter by supported extensions
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        # 2. Filter by ignored directories (e.g. .git, .venv, node_modules)
        for part in path.parts:
            if part in IGNORE_DIRS or part.endswith(".egg-info") or part.startswith("."):
                return

        # 3. Determine the project name (the direct child folder of PROJECTS_ROOT)
        root_path = Path(PROJECTS_ROOT).resolve()
        root_str = str(root_path).lower().rstrip(os.sep)
        path_str = str(path).lower()

        if not path_str.startswith(root_str):
            return  # Path is not under PROJECTS_ROOT

        rel_str = path_str[len(root_str) :].lstrip(os.sep)
        parts = rel_str.split(os.sep)
        if not parts or not parts[0]:
            return

        project_name_lower = parts[0]

        # Find the correct casing of the project folder
        project_name = None
        try:
            for entry in root_path.iterdir():
                if entry.is_dir() and entry.name.lower() == project_name_lower:
                    project_name = entry.name
                    break
        except OSError:
            return

        if not project_name:
            return

        project_dir = root_path / project_name

        # Verify it is a valid git repository (contains .git folder)
        if not (project_dir / ".git").exists():
            return

        with self.lock:
            self.pending_projects.add(project_name)
            self._reset_timer_unlocked()

    def _reset_timer_unlocked(self):
        if self.timer is not None:
            self.timer.cancel()
        self.timer = threading.Timer(self.debounce_seconds, self._trigger_indexing)
        self.timer.start()

    def _trigger_indexing(self):
        with self.lock:
            projects_to_run = list(self.pending_projects)
            self.pending_projects.clear()
            self.timer = None

        for name in projects_to_run:
            click.secho(f"\n[WATCHER] Detected edits in '{name}'. Reindexing...", fg="cyan")
            t0 = time.monotonic()
            try:
                result = self.service.reindex_project(name, force=False)
                elapsed = round(time.monotonic() - t0, 1)
                if "error" in result:
                    click.secho(f"  [WATCHER ERROR] {result['error']} ({elapsed}s)", fg="red")
                else:
                    click.secho(f"  [WATCHER OK] {result['message']} ({elapsed}s)", fg="green")
            except Exception as e:
                click.secho(f"  [WATCHER CRITICAL ERROR] {e}", fg="red")


def main():
    click.secho("=== REPO_KNOWLEDGE OS-Event Filewatcher ===", fg="cyan", bold=True)
    click.secho(f"Monitoring: {PROJECTS_ROOT}", fg="white")
    click.secho("Supported extensions: " + ", ".join(SUPPORTED_EXTENSIONS), fg="white")
    click.secho("Waiting for file system events... Press Ctrl+C to stop.\n", fg="white")

    service = KnowledgeService()
    handler = ProjectChangeHandler(service)
    observer = Observer()
    observer.schedule(handler, path=PROJECTS_ROOT, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.secho("\nWatcher stopping...", fg="yellow")
        observer.stop()
    observer.join()
    click.secho("Watcher stopped.", fg="green")


if __name__ == "__main__":
    main()
