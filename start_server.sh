#!/bin/bash
echo "================================================"
echo "   F5-TTS Voice Cloning API Server"
echo "================================================"
echo

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated."
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
fi

echo
echo "Starting F5-TTS API Server on port 8001..."
echo "API Docs: http://localhost:8001/docs"
echo

python main.py --host 0.0.0.0 --port 8001
