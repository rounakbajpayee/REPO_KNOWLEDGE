from fastapi import FastAPI, Request
from mcp.server.sse import SseServerTransport
from repo_knowledge.mcp_server import server as mcp_server

app = FastAPI(title="Repo Knowledge MCP via SSE")
sse = SseServerTransport("/messages/")

@app.get("/sse")
async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())

app.mount("/messages/", sse.handle_post_message)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
