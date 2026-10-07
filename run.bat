@echo off
rem Start BroClips in this window, with its log (for finding problems). Close this window to stop it.
rem Normal use: the BroClips shortcut on your Desktop.
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PY="
if exist ".venv\python-path.txt" set /p PY=<".venv\python-path.txt"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=python"
echo Starting BroClips with %PY% ...
"%PY%" -m broclips %*
echo.
echo BroClips has stopped.
pause
