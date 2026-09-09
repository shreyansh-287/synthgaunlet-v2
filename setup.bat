@echo off
echo ===================================================
echo   SynthGauntlet - Automated Setup Script
echo ===================================================
echo.

echo 1. Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo 2. Creating Virtual Environment (.venv)...
python -m venv .venv

echo 3. Activating Virtual Environment...
call .\.venv\Scripts\activate.bat

echo 4. Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist .env (
    echo 5. Creating .env file from .env.example...
    copy .env.example .env
    echo [NOTE] Please edit .env and add your GEMINI_API_KEY.
) else (
    echo 5. .env file already exists.
)

echo.
echo ===================================================
echo   SETUP COMPLETE!
echo   To start the app, run: python -m uvicorn main:app --port 8000
echo ===================================================
echo.
pause
