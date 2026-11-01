@echo off
REM Automated 5-minute Intraday Loop: runs once, updates every 5 minutes until 15:31 IST.
pushd "%~dp0"

set "PY=.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo ERROR: could not find %PY%
    echo Run it from the project root, or recreate the venv.
    pause
    exit /b 1
)

title Nifty Analyst - Live 5m Intraday Loop
echo =========================================================================
echo  NIFTY ANALYST - LIVE 5-MINUTE INTRADAY AUTOMATION LOOP
echo =========================================================================
echo  This loop runs continuously during market hours.
echo  Every 5 minutes, it will:
echo   1. Fetch live Fyers 5m bars, Options Chain, and Futures positioning
echo   2. Re-calculate Institutional Defense Walls and Shift Signals
echo   3. Automatically update your dashboard at https://market-analysis-7dc40.web.app
echo   4. Auto-terminate cleanly after market close at 15:31 IST
echo =========================================================================
echo.
"%PY%" scripts\auto_intraday_loop.py
set "RC=%ERRORLEVEL%"

echo.
echo Loop finished with exit code %RC%.
pause
popd
exit /b %RC%
