# Issue: Remote MCP Access via SSE Transport

## Context
Currently, the MCP server in `src/repo_knowledge/mcp_server.py` is configured to run locally via standard input/output (`stdio`) transport. We want to allow remote clients to connect to this server (running on a Mac) without using SSH tunnels. To achieve this, we need to expose the MCP server over HTTP using Server-Sent Events (SSE).

## Architectural Constraints
- The project already runs a FastAPI backend for the web dashboard in `src/repo_knowledge/web_ui/server.py`.
- We should integrate the MCP SSE transport into this FastAPI application.
- `mcp_server.py` currently creates an `mcp.server.Server` instance and defines several tools (`@server.call_tool()`). You should reuse this server instance and its tool definitions so we don't duplicate logic.

## Implementation Details for Jules
1. Import `SseServerTransport` from `mcp.server.sse` in `web_ui/server.py` (or create a new `web_ui/mcp_sse.py` adapter module).
2. You will need to import the initialized `server` object from `repo_knowledge.mcp_server`. Note: `mcp_server.py` has `server = Server("repo-knowledge")`.
3. Create an SSE transport instance: `sse = SseServerTransport("/messages/")`.
4. Add an SSE GET route to `app` (FastAPI) that handles the connection lifecycle. Example logic:
   ```python
   @app.get("/sse")
   async def handle_sse(request: Request):
       async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
           # Run the imported MCP server
           await mcp_server.server.run(streams[0], streams[1], mcp_server.server.create_initialization_options())
   ```
5. Mount the POST route for messages: `app.mount("/messages/", sse.handle_post_message)`.
6. **Thread Safety**: `mcp_server.py`'s `call_tool` uses `asyncio.wait_for` and `loop.run_in_executor` to prevent blocking the async loop. Ensure that executing tools via the new FastAPI SSE transport doesn't introduce thread blocking issues.

## Acceptance Criteria
- An MCP client can connect to `http://<ip>:8000/sse` successfully.
- Tools execute correctly and do not block the FastAPI event loop.
- No duplicate implementation of the existing tools in `mcp_server.py`.
