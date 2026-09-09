#!/usr/bin/env bash
echo "==================================================="
echo "  SynthGauntlet - Automated Setup Script (Mac/Linux)"
echo "==================================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed!"
    echo "Please install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

echo "1. Creating Virtual Environment (.venv)..."
python3 -m venv .venv

echo "2. Activating Virtual Environment..."
source .venv/bin/activate

echo "3. Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
    echo "4. Creating .env file from .env.example..."
    cp .env.example .env
    echo "[NOTE] Please edit .env and add your GEMINI_API_KEY."
else
    echo "4. .env file already exists."
fi

echo ""
echo "==================================================="
echo "  SETUP COMPLETE!"
echo "  To start the app, run: python3 -m uvicorn main:app --port 8000"
echo "==================================================="
