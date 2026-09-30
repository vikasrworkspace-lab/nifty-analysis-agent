@echo off
REM Daily Fyers login: run this once before 09:00 IST on each trading day.
REM It mints an access token, pushes it to Secret Manager as a new
REM FYERS_ACCESS_TOKEN version, and verifies it. Nothing else is needed --
REM the Cloud Run jobs bind FYERS_ACCESS_TOKEN:latest.
REM
REM Doubling this file is the whole daily routine. See AGENTS.md "Fyers auth".

pushd "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo ERROR: could not find %PY%
    echo Run it from the project root, or recreate the venv.
    goto :end
)

if not exist ".env" (
    echo ERROR: .env is missing.
    echo It must define FYERS_APP_ID, FYERS_SECRET_KEY and FYERS_REDIRECT_URI.
    goto :end
)

where gcloud >nul 2>&1
if errorlevel 1 (
    echo ERROR: gcloud was not found on PATH.
    echo Install the Google Cloud SDK, then run: gcloud auth login
    goto :end
)

"%PY%" scripts\fyers_login.py --push-secret
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo RESULT: OK. You can close this window.
) else (
    echo RESULT: FAILED ^(exit %RC%^). Do not trade off this data today.
)

:end
echo.
pause
popd
exit /b %RC%
