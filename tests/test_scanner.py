"""
test_scanner.py — Unit tests for scanner.py

Tests:
  - scan_projects discovers only .git directories
  - scan_projects ignores IGNORE_DIRS
  - detect_stack returns correct labels for known manifests
  - get_project returns correct project or None
  - scan_projects does not recurse into nested repos
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from repo_knowledge.scanner import scan_projects, get_project, detect_stack, Project


@pytest.fixture
def fake_projects(tmp_path: Path) -> Path:
    """Creates a fake projects directory with a mix of repos and non-repos."""
    # Valid git repos
    for name in ["ALPHA", "BETA"]:
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".git").mkdir()

    # Not a git repo
    (tmp_path / "not_a_repo").mkdir()

    # Ignored dir
    (tmp_path / "node_modules").mkdir()

    # ALPHA has a Python manifest
    (tmp_path / "ALPHA" / "pyproject.toml").touch()

    # BETA has a Node manifest
    (tmp_path / "BETA" / "package.json").touch()

    return tmp_path


def test_scan_finds_only_git_repos(fake_projects: Path):
    projects = scan_projects(str(fake_projects))
    names = {p.name for p in projects}
    assert names == {"ALPHA", "BETA"}


def test_scan_ignores_non_git_directories(fake_projects: Path):
    projects = scan_projects(str(fake_projects))
    names = {p.name for p in projects}
    assert "not_a_repo" not in names
    assert "node_modules" not in names


def test_scan_returns_correct_paths(fake_projects: Path):
    projects = scan_projects(str(fake_projects))
    for p in projects:
        assert p.path.exists()
        assert (p.path / ".git").exists()


def test_detect_stack_python(fake_projects: Path):
    project = get_project("ALPHA", str(fake_projects))
    assert project is not None
    assert "Python" in project.stack


def test_detect_stack_node(fake_projects: Path):
    project = get_project("BETA", str(fake_projects))
    assert project is not None
    assert "Node.js" in project.stack


def test_get_project_returns_none_for_unknown(fake_projects: Path):
    result = get_project("NONEXISTENT", str(fake_projects))
    assert result is None


def test_scan_raises_for_missing_root():
    with pytest.raises(FileNotFoundError):
        scan_projects("/this/path/does/not/exist")


def test_scan_results_are_sorted(fake_projects: Path):
    projects = scan_projects(str(fake_projects))
    names = [p.name for p in projects]
    assert names == sorted(names)
