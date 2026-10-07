@echo off
rem BroClips installer: double-click this file once. It finds Python, checks ffmpeg, installs the packages,
rem downloads the fonts and makes the BroClips shortcut on your Desktop. Options: see scripts\install.ps1
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
echo.
pause
