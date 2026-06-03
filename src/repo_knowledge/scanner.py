"""
scanner.py — Discovers Git repositories under PROJECTS_ROOT.

A directory is treated as a project if it contains a .git folder at its root.
Returns project name (directory name) and absolute path.

Heuristic stack detection checks for known manifest files — used by
get_project_context to give agents an instant orientation without file reads.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from repo_knowledge.config import IGNORE_DIRS, PROJECTS_ROOT


@dataclass
class Project:
    name: str
    path: Path
    stack: list[str] = field(default_factory=list)


# Maps manifest filename → human-readable stack label
_STACK_MARKERS: dict[str, str] = {
    "pyproject.toml": "Python",
    "requirements.txt": "Python",
    "setup.py": "Python",
    "package.json": "Node.js",
    "tsconfig.json": "TypeScript",
    "go.mod": "Go",
    "Cargo.toml": "Rust",
    "pom.xml": "Java/Maven",
    "build.gradle": "Java/Gradle",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "docker-compose.yml": "Docker",
    "docker-compose.yaml": "Docker",
    "Dockerfile": "Docker",
}


def detect_stack(project_path: Path) -> list[str]:
    """Return detected stack labels for a project directory."""
    found: list[str] = []
    seen: set[str] = set()
    for filename, label in _STACK_MARKERS.items():
        if label not in seen and (project_path / filename).exists():
            found.append(label)
            seen.add(label)
    return found


def scan_projects(root: str = PROJECTS_ROOT) -> list[Project]:
    """
    Walk root one level deep and return all Git repositories found.
    Does not recurse into nested repositories.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Projects root not found: {root_path}")

    projects: list[Project] = []

    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in IGNORE_DIRS or entry.name.startswith("."):
            continue
        if (entry / ".git").exists():
            projects.append(
                Project(
                    name=entry.name,
                    path=entry,
                    stack=detect_stack(entry),
                )
            )

    return projects


def get_project(name: str, root: str = PROJECTS_ROOT) -> Project | None:
    """Return a single project by name, or None if not found."""
    for project in scan_projects(root):
        if project.name == name:
            return project
    return None
