import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.repo_knowledge.knowledge import KnowledgeService

def reindex_all():
    service = KnowledgeService()
    projects_resp = service.list_projects()
    
    # Extract project names
    if "projects" in projects_resp:
        projects = [p["name"] for p in projects_resp["projects"]]
    else:
        # If it returns a list directly
        projects = [p["name"] for p in projects_resp] if isinstance(projects_resp, list) else []

    print(f"Found {len(projects)} projects.")
    for proj in projects:
        print(f"Reindexing {proj}...")
        try:
            res = service.reindex_project(proj)
            print(res)
        except Exception as e:
            print(f"Failed {proj}: {e}")

if __name__ == "__main__":
    reindex_all()
