import time
import logging
from repo_knowledge.knowledge import KnowledgeService
from repo_knowledge.config import QDRANT_URL, OLLAMA_URL

logging.basicConfig(level=logging.INFO)

print(f"QDRANT_URL: {QDRANT_URL}")
print(f"OLLAMA_URL: {OLLAMA_URL}")

print("Initializing...")
t0 = time.time()
svc = KnowledgeService()
print(f"Init took {time.time() - t0:.2f}s")

print("Searching...")
t1 = time.time()
try:
    res = svc.search("knowledge service embedding vector store", project="REPO_KNOWLEDGE", top_k=3)
    print(f"Search took {time.time() - t1:.2f}s")
    print(f"Found {len(res)} results.")
    for r in res:
        print(f" - {r.get('path')} (Score: {r.get('score')})")
except Exception as e:
    print(f"Search failed: {e}")
