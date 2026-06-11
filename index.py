"""
index.py — CLI entry point for indexing repositories.

Usage:
  # Index all projects under PROJECTS_ROOT
  python index.py

  # Index a single project by name
  python index.py --project LENS

  # Override projects root
  python index.py --root /path/to/projects
"""

import time

import click

from repo_knowledge.knowledge import KnowledgeService
from repo_knowledge.scanner import scan_projects
from repo_knowledge.tracer import trace


@click.command()
@click.option("--project", default=None, help="Index a single project by name.")
@click.option("--root", default=None, help="Override PROJECTS_ROOT from .env")
def main(project: str | None, root: str | None) -> None:
    kwargs = {}
    if root:
        kwargs["projects_root"] = root

    svc = KnowledgeService(**kwargs)

    if project:
        projects_to_index = [project]
    else:
        discovered = scan_projects(svc._projects_root)
        projects_to_index = [p.name for p in discovered]
        click.echo(
            f"Discovered {len(projects_to_index)} project(s): {', '.join(projects_to_index)}"
        )

    t_total = time.monotonic()
    for name in projects_to_index:
        click.echo(f"\n-> Indexing {name}...")
        t0 = time.monotonic()
        result = svc.reindex_project(name)
        elapsed = round(time.monotonic() - t0, 1)
        if "error" in result:
            click.secho(f"  [ERROR] {result['error']} ({elapsed}s)", fg="red")
            trace(
                "cli_index_error",
                subsystem="cli",
                severity="ERROR",
                project=name,
                error=result["error"],
                duration_s=elapsed,
            )
        else:
            click.secho(
                f"  [OK] {result['message']} ({elapsed}s)",
                fg="green",
            )
            trace(
                "cli_index_success",
                subsystem="cli",
                project=name,
                chunks=result["chunks_indexed"],
                duration_s=elapsed,
            )

    total_elapsed = round(time.monotonic() - t_total, 1)
    click.echo(f"\nDone in {total_elapsed}s.")
    trace(
        "cli_index_all_complete",
        subsystem="cli",
        projects=len(projects_to_index),
        duration_s=total_elapsed,
    )


if __name__ == "__main__":
    main()
