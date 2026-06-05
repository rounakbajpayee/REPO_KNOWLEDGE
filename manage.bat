@echo off
rem Run manage.py using the virtual environment's Python interpreter
"%~dp0.venv\Scripts\python.exe" "%~dp0manage.py" %*
