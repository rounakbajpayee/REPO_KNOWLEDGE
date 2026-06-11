import re

with open('src/repo_knowledge/postgres_store.py', 'r') as f:
    content = f.read()

start = content.find('def _ensure_tables')
end = content.find('def health_check', start)
section = content[start:end]

# print table creations
tables = re.findall(r'CREATE TABLE IF NOT EXISTS\s+(\w+)', section)
print("Tables:", tables)
