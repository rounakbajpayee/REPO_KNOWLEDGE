import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.repo_knowledge.knowledge import KnowledgeService
from repo_knowledge.scanner import scan_projects

def inspect():
    svc = KnowledgeService()
    scanned = {p.name: p for p in scan_projects(svc._projects_root)}
    print("Scanned:", list(scanned.keys()))
    
    indexed = set(svc._store.list_projects())
    print("Indexed:", indexed)
    
    for name in scanned:
        print(f"Project {name}: indexed={name in indexed}")

if __name__ == "__main__":
    inspect()
