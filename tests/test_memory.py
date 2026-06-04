"""
tests/test_memory.py — Unit tests for log_decision and get_decision_history
in KnowledgeService.
"""

import threading
import time
from pathlib import Path
import pytest
from repo_knowledge.knowledge import KnowledgeService


@pytest.fixture
def temp_vault_service(tmp_path: Path) -> KnowledgeService:
    """Returns a KnowledgeService instance pointed to a temporary projects root."""
    # We create a dummy PROJECTS_ROOT inside tmp_path
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    return KnowledgeService(projects_root=projects_root)


def test_log_decision_creates_file_with_frontmatter(temp_vault_service: KnowledgeService) -> None:
    """Calling log_decision for a new topic creates the markdown file and writes frontmatter."""
    svc = temp_vault_service
    res = svc.log_decision(
        topic="test_topic",
        name="initial_setup",
        description="This is a test description.",
        rationale="This is the rationale.",
    )

    assert "error" not in res
    assert res["topic"] == "test_topic"

    vault_file = Path(svc._projects_root) / "knowledge_vault" / "test_topic.md"
    assert vault_file.exists()

    content = vault_file.read_text(encoding="utf-8")
    assert content.startswith("---")
    assert "topic: test_topic" in content
    assert "entries_count: 1" in content
    assert "# Decision Log: Test Topic" in content
    assert "## [" in content
    assert "initial_setup" in content
    assert "This is a test description." in content


def test_log_decision_appends_and_updates_frontmatter(temp_vault_service: KnowledgeService) -> None:
    """Calling log_decision repeatedly appends entries and increments entry count."""
    svc = temp_vault_service

    # Write first
    svc.log_decision(
        topic="database_choice",
        name="use_sqlite",
        description="Using SQLite for local cache.",
        rationale="Simple and self-contained.",
    )

    # Write second
    res2 = svc.log_decision(
        topic="database_choice",
        name="migrate_to_postgres",
        description="Migrate to Postgres.",
        rationale="Need concurrency support.",
    )

    assert "error" not in res2

    vault_file = Path(svc._projects_root) / "knowledge_vault" / "database_choice.md"
    content = vault_file.read_text(encoding="utf-8")

    assert "entries_count: 2" in content
    assert "use_sqlite" in content
    assert "migrate_to_postgres" in content


def test_get_decision_history_defaults_limit(temp_vault_service: KnowledgeService) -> None:
    """get_decision_history defaults to last 3 entries unless full_history is true."""
    svc = temp_vault_service

    # Write 5 entries
    for i in range(1, 6):
        svc.log_decision(
            topic="deployment",
            name=f"deploy_step_{i}",
            description=f"Description {i}",
            rationale=f"Rationale {i}",
        )

    # Default history query (limit=3)
    res = svc.get_decision_history(topic="deployment")
    assert "error" not in res
    assert res["total_entries"] == 5
    assert res["shown_entries"] == 3
    assert "deploy_step_1" not in res["history"]
    assert "deploy_step_3" in res["history"]
    assert "deploy_step_5" in res["history"]
    assert "truncated" in res["history"].lower()

    # Full history query
    res_full = svc.get_decision_history(topic="deployment", full_history=True)
    assert res_full["shown_entries"] == 5
    assert "deploy_step_1" in res_full["history"]
    assert "deploy_step_5" in res_full["history"]


def test_log_decision_slugifies_topic(temp_vault_service: KnowledgeService) -> None:
    """Topic names are sanitized/slugified to avoid directory traversal and enforce lowercase."""
    svc = temp_vault_service

    res = svc.log_decision(
        topic="Test/../Topic!#Name",
        name="entry_one",
        description="desc",
        rationale="rat",
    )

    # Should slugify to 'test____topic__name'
    expected_slug = "test____topic__name"
    assert res["topic"] == expected_slug

    vault_file = Path(svc._projects_root) / "knowledge_vault" / f"{expected_slug}.md"
    assert vault_file.exists()


def test_log_decision_concurrency_thread_safety(temp_vault_service: KnowledgeService) -> None:
    """Multiple concurrent threads appending to the same topic do not corrupt files."""
    svc = temp_vault_service
    topic = "concurrent_topic"

    def worker(num: int):
        svc.log_decision(
            topic=topic,
            name=f"entry_{num}",
            description=f"description_{num}",
            rationale=f"rationale_{num}",
        )

    threads = []
    # Spin up 10 threads writing to the same file
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify count is 10 and no content is corrupt
    res = svc.get_decision_history(topic=topic, full_history=True)
    assert res["total_entries"] == 10
    for i in range(10):
        assert f"entry_{i}" in res["history"]
