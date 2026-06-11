from pathlib import Path

import pytest

from repo_knowledge.scanner import get_project, scan_projects


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


def test_list_project_files_fallback(tmp_path):
    """If not a git repo (or git fails), list_project_files should walk using fallback."""
    from repo_knowledge.scanner import list_project_files

    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    (project_dir / "src").mkdir()
    f1 = project_dir / "src" / "main.py"
    f1.touch()
    f2 = project_dir / "README.md"
    f2.touch()

    # Ignored files should not be listed
    f3 = project_dir / "node_modules" / "lib.js"
    f3.parent.mkdir()
    f3.touch()
    f4 = project_dir / ".git" / "config"
    f4.parent.mkdir()
    f4.touch()

    files = list_project_files(project_dir)
    paths = {f.name for f in files}
    assert "main.py" in paths
    assert "README.md" in paths
    assert "lib.js" not in paths
    assert "config" not in paths


def test_list_project_files_git(tmp_path):
    """If it is a git repo, list_project_files should use git ls-files."""
    import subprocess

    from repo_knowledge.scanner import list_project_files

    project_dir = tmp_path / "git_project"
    project_dir.mkdir()

    # Initialize actual git repo
    subprocess.run(["git", "init"], cwd=project_dir, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=project_dir, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=project_dir,
        capture_output=True,
        check=True,
    )

    f1 = project_dir / "main.py"
    f1.write_text("print(1)")
    f2 = project_dir / "ignored.tmp"
    f2.write_text("temp")

    # Add main.py to git, but keep ignored.tmp untracked
    subprocess.run(["git", "add", "main.py"], cwd=project_dir, capture_output=True, check=True)

    # Create a gitignore to ignore *.tmp files
    (project_dir / ".gitignore").write_text("*.tmp\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=project_dir, capture_output=True, check=True)

    files = list_project_files(project_dir)
    paths = {f.name for f in files}

    # main.py is tracked, .gitignore is tracked
    assert "main.py" in paths
    assert ".gitignore" in paths
    # ignored.tmp matches gitignore, so it must NOT be listed
    assert "ignored.tmp" not in paths
