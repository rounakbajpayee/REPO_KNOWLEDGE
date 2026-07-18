import subprocess

out = subprocess.check_output('wmic process get processid,commandline', shell=True).decode('utf-8', errors='ignore')
for line in out.splitlines():
    if 'repo_knowledge.mcp_server' in line or ('session_memory' in line and 'mcp_server.py' in line):
        parts = line.strip().split()
        if not parts: continue
        pid = parts[-1]
        if pid.isdigit():
            print(f"Killing PID {pid}")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True)
