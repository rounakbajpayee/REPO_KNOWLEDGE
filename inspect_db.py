import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.repo_knowledge.postgres_store import PostgresStore
from src.repo_knowledge.store import Store

def inspect():
    pg = PostgresStore()
    print("Project names in DB:", pg.get_project_names())
    
    store = Store()
    print("Store projects:", store.list_projects())

if __name__ == "__main__":
    inspect()
