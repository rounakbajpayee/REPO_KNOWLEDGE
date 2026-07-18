import sys
import os
import time
import logging

sys.path.insert(0, os.path.abspath('.'))
from src.repo_knowledge.knowledge import KnowledgeService

def test():
    svc = KnowledgeService()
    t0 = time.time()
    print("Starting get_project_context for session-memory...")
    try:
        res = svc.get_project_context("session-memory")
        print(f"Finished in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test()
