import re

with open('src/repo_knowledge/postgres_store.py', 'r') as f:
    content = f.read()

start = content.find('def _ensure_tables')
end = content.find('def health_check', start)
print(content[start:end])
