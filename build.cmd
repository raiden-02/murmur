@echo off
rem Builds Murmur from any terminal and optionally runs one of its programs:
rem   build [x64-release^|x64-debug] [-Target name[,name...]] [-Test] [-Fresh] [-Run program]
rem `build --help` explains each option and lists the programs. tools\build.ps1 does the work.
if /i "%~1"=="--help" goto help
if /i "%~1"=="-h" goto help
if "%~1"=="/?" goto help
powershell -NoProfile -File "%~dp0tools\build.ps1" %*
exit /b %errorlevel%

:help
powershell -NoProfile -File "%~dp0tools\build.ps1" -Help
exit /b %errorlevel%
