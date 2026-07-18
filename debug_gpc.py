import sys
import os
import time
from pathlib import Path
sys.path.insert(0, os.path.abspath('.'))

from src.repo_knowledge.knowledge import KnowledgeService
from repo_knowledge.scanner import get_project

def debug_gpc():
    svc = KnowledgeService()
    project_name = "session-memory"
    
    t0 = time.time()
    project = get_project(project_name, svc._projects_root)
    print(f"get_project: {time.time()-t0:.4f}s")
    
    t0 = time.time()
    from repo_knowledge.knowledge import _read_readme, _build_tree
    readme_excerpt = _read_readme(project.path)
    print(f"_read_readme: {time.time()-t0:.4f}s")
    
    t0 = time.time()
    tree = _build_tree(project.path, max_depth=2)
    print(f"_build_tree: {time.time()-t0:.4f}s")
    
    t0 = time.time()
    from repo_knowledge.scanner import list_project_files
    file_count = len(list_project_files(project.path))
    print(f"list_project_files: {time.time()-t0:.4f}s")
    
    t0 = time.time()
    indexed = project_name in set(svc._store.list_projects())
    print(f"list_projects: {time.time()-t0:.4f}s")
    
    t0 = time.time()
    from repo_knowledge.tracer import trace
    trace(
        "get_project_context",
        project=project_name,
        file_count=file_count,
        indexed=indexed,
        subsystem="knowledge",
    )
    print(f"trace: {time.time()-t0:.4f}s")

if __name__ == "__main__":
    debug_gpc()
