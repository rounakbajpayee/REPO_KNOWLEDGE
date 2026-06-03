"""
mcp_server.py — MCP server exposing repository knowledge to coding agents.

Transport: SSE (Server-Sent Events)
  - Compatible with Claude, Codex, OpenCode, and any HTTP client
  - Future REST/gRPC adapters should call KnowledgeService directly,
    not this module

Tools:
  list_projects         → all projects with stack + index status
  get_project_context   → full cold-start orientation for one project
  search_codebase       → semantic search, optional project filter
  get_file              → raw file contents by project + path
  reindex_project       → delete + rechunk + re-embed + store

All tools return structured dicts. Errors are returned as {"error": "..."} —
agents should check for this key rather than catching exceptions.

Startup checks:
  Qdrant and Ollama reachability are verified on server start.
  Server starts regardless — tools return clean error messages if
  a backend is unreachable at call time.
"""

import json
import logging

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from repo_knowledge.config import OLLAMA_URL, QDRANT_URL
from repo_knowledge.embedder import OllamaEmbedder
from repo_knowledge.knowledge import KnowledgeService
from repo_knowledge.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Server init ───────────────────────────────────────────────────────────────

server = Server("repo-knowledge")
_svc: KnowledgeService | None = None


def _get_service() -> KnowledgeService:
    global _svc
    if _svc is None:
        _svc = KnowledgeService()
    return _svc


# ── Tool definitions ──────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_projects",
            description=(
                "List all Git repositories found in the projects root. "
                "Returns project name, detected tech stack, and whether it has been indexed. "
                "Call this first to orient yourself before working on any project."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="get_project_context",
            description=(
                "Get full orientation context for a single project in one call. "
                "Returns README excerpt, directory tree, file count, and stack. "
                "Call this before starting or continuing work on a project. "
                "Avoids the need to read multiple files manually."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Exact project name as returned by list_projects",
                    }
                },
                "required": ["project"],
            },
        ),
        types.Tool(
            name="search_codebase",
            description=(
                "Semantic search over indexed code chunks. "
                "Returns the most relevant functions, classes, or sections matching the query. "
                "Each result includes file path, symbol name, line range, and content. "
                "Use this instead of reading entire files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what you are looking for",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: restrict search to a single project",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_file",
            description=(
                "Read the raw contents of a specific file. "
                "Requires both project name and file path relative to the project root. "
                "Use search_codebase first to find the relevant file path."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Exact project name",
                    },
                    "path": {
                        "type": "string",
                        "description": "File path relative to project root, e.g. src/auth/service.py",
                    },
                },
                "required": ["project", "path"],
            },
        ),
        types.Tool(
            name="reindex_project",
            description=(
                "Delete all existing vectors for a project and reindex it from scratch. "
                "Run this after making significant code changes to keep search results current. "
                "Returns chunk count on success."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Exact project name to reindex",
                    }
                },
                "required": ["project"],
            },
        ),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    svc = _get_service()

    try:
        if name == "list_projects":
            result = svc.list_projects()

        elif name == "get_project_context":
            project = arguments.get("project", "")
            if not project:
                result = {"error": "project is required"}
            else:
                result = svc.get_project_context(project)

        elif name == "search_codebase":
            query = arguments.get("query", "")
            if not query:
                result = {"error": "query is required"}
            else:
                result = svc.search(
                    query=query,
                    project=arguments.get("project"),
                    top_k=int(arguments.get("top_k", 5)),
                )

        elif name == "get_file":
            project = arguments.get("project", "")
            path = arguments.get("path", "")
            if not project or not path:
                result = {"error": "Both project and path are required"}
            else:
                result = svc.get_file(project, path)

        elif name == "reindex_project":
            project = arguments.get("project", "")
            if not project:
                result = {"error": "project is required"}
            else:
                result = svc.reindex_project(project)

        else:
            result = {"error": f"Unknown tool: {name}"}

    except RuntimeError as e:
        # Embedder or store connectivity failure — return clean error to agent
        result = {"error": str(e)}
    except Exception as e:
        log.exception("Unexpected error in tool %s", name)
        result = {"error": f"Internal error: {e}"}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("Starting REPO_KNOWLEDGE MCP server")
    log.info("Qdrant: %s", QDRANT_URL)
    log.info("Ollama: %s", OLLAMA_URL)

    # Startup health checks — warn but don't block
    embedder = OllamaEmbedder()
    store = Store()

    if not store.health_check():
        log.warning("Qdrant unreachable at %s — search tools will fail until resolved", QDRANT_URL)
    else:
        log.info("Qdrant OK")

    if not embedder.health_check():
        log.warning("Ollama unreachable at %s — search and reindex will fail until resolved", OLLAMA_URL)
    else:
        log.info("Ollama OK")

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
