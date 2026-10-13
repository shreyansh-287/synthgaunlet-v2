@echo off
echo ===================================================
echo   Starting SynthGauntlet Local Server
echo   URL: http://127.0.0.1:8080
echo ===================================================
call .\.venv\Scripts\activate.bat
python -m uvicorn main:app --host 127.0.0.1 --port 8080
pause
