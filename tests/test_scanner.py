import pytest
from pathlib import Path
from repo_knowledge.scanner import scan_projects, get_project, detect_stack, Project


@pytest.fixture
def fake_projects(tmp_path: Path) -> Path:
    for name in ["ALPHA", "BETA"]:
        repo = tmp_path / name
        repo.mkdir()
        (repo / ".git").mkdir()
    (tmp_path / "not_a_repo").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "ALPHA" / "pyproject.toml").touch()
    (tmp_path / "BETA" / "package.json").touch()
    return tmp_path


def test_scan_finds_only_git_repos(fake_projects):
    projects = scan_projects(str(fake_projects))
    names = {p.name for p in projects}
    assert names == {"ALPHA", "BETA"}


def test_scan_ignores_non_git_directories(fake_projects):
    projects = scan_projects(str(fake_projects))
    names = {p.name for p in projects}
    assert "not_a_repo" not in names
    assert "node_modules" not in names


def test_scan_returns_correct_paths(fake_projects):
    projects = scan_projects(str(fake_projects))
    for p in projects:
        assert p.path.exists()
        assert (p.path / ".git").exists()


def test_detect_stack_python(fake_projects):
    project = get_project("ALPHA", str(fake_projects))
    assert project is not None
    assert "Python" in project.stack


def test_detect_stack_node(fake_projects):
    project = get_project("BETA", str(fake_projects))
    assert project is not None
    assert "Node.js" in project.stack


def test_get_project_returns_none_for_unknown(fake_projects):
    result = get_project("NONEXISTENT", str(fake_projects))
    assert result is None


def test_scan_raises_for_missing_root():
    with pytest.raises(FileNotFoundError):
        scan_projects("/this/path/does/not/exist")


def test_scan_results_are_sorted(fake_projects):
    projects = scan_projects(str(fake_projects))
    names = [p.name for p in projects]
    assert names == sorted(names)


def test_get_project_fast_path(fake_projects):
    """get_project must return a Project without full scan when root/<name>/.git exists."""
    project = get_project("ALPHA", str(fake_projects))
    assert project is not None
    assert project.name == "ALPHA"
    assert (project.path / ".git").exists()


def test_get_project_fast_path_returns_none_for_non_git(fake_projects):
    """get_project fast path must return None when candidate dir has no .git."""
    result = get_project("not_a_repo", str(fake_projects))
    assert result is None
