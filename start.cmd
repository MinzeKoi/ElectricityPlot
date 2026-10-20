@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

set "APP_PYTHON="
if exist ".venv-m4\Scripts\python.exe" set "APP_PYTHON=%CD%\.venv-m4\Scripts\python.exe"
if not defined APP_PYTHON if exist ".venv-m2\Scripts\python.exe" set "APP_PYTHON=%CD%\.venv-m2\Scripts\python.exe"
if not defined APP_PYTHON if exist ".venv\Scripts\python.exe" set "APP_PYTHON=%CD%\.venv\Scripts\python.exe"
if not defined APP_PYTHON goto missing_environment

if not exist "tmp" mkdir "tmp"
if not exist "tmp\" goto failed
set "TEMP=%CD%\tmp"
set "TMP=%TEMP%"
echo Python: "%APP_PYTHON%"
"%APP_PYTHON%" -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'; import numpy, torch, streamlit, pandas, altair; print('Environment check passed.')"
if errorlevel 1 goto failed
if /i "%~1"=="--check" goto success

echo Open the local URL printed by Streamlit below in your browser.
echo Keep this window open. Press Ctrl+C to stop.
"%APP_PYTHON%" -m streamlit run scripts/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false %*
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" goto failed

:success
popd
exit /b 0

:missing_environment
echo [ERROR] No project Python environment found.
echo Create one with: py -3.12 -m venv .venv-m4
echo Then run: .venv-m4\Scripts\python.exe -m pip install -r requirements-m4-cpu.lock
goto failed

:failed
echo [ERROR] Startup failed. See the messages above.
if /i not "%~1"=="--check" pause
popd
exit /b 1
