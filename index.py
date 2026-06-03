"""
index.py — CLI entry point for indexing repositories.

Usage:
  # Index all projects under PROJECTS_ROOT
  python index.py

  # Index a single project by name
  python index.py --project LENS

  # Index all, explicit root override
  python index.py --root /path/to/projects
"""

import click

from repo_knowledge.knowledge import KnowledgeService
from repo_knowledge.scanner import scan_projects


@click.command()
@click.option("--project", default=None, help="Index a single project by name. Default: all projects.")
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
        click.echo(f"Discovered {len(projects_to_index)} project(s): {', '.join(projects_to_index)}")

    for name in projects_to_index:
        click.echo(f"\n→ Indexing {name}...")
        result = svc.reindex_project(name)
        if "error" in result:
            click.secho(f"  ✗ {result['error']}", fg="red")
        else:
            click.secho(f"  ✓ {result['message']}", fg="green")

    click.echo("\nDone.")


if __name__ == "__main__":
    main()
