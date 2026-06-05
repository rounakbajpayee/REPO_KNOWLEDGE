from repo_knowledge.knowledge import KnowledgeService

svc = KnowledgeService()

print("=== list_projects ===")
import json
print(json.dumps(svc.list_projects(), indent=2))

print("\n=== get_project_context: LENS ===")
ctx = svc.get_project_context("LENS")
print(json.dumps({k: v for k, v in ctx.items() if k != "directory_tree"}, indent=2))
print("Tree:", [line.encode('ascii', errors='replace').decode('ascii') for line in ctx.get("directory_tree", [])[:5]])

print("\n=== search: authentication ===")
results = svc.search("authentication", top_k=3)
for r in results:
    print(f"  [{r['score']}] {r['project']}/{r['path']} — {r.get('symbol', '')}")
