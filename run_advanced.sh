#!/bin/bash

echo "=========================================="
echo "Advanced Agentic AI Chatbot with RAG"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

# Check if Ollama is running
echo "Checking Ollama status..."
if ! pgrep -f "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5
else
    echo "Ollama is running"
fi

# Check if Qwen3 model is available
echo "Checking Qwen3 model..."
if ollama list | grep -q "qwen3:8b"; then
    echo "Qwen3 8B model is available"
else
    echo "Pulling Qwen3 8B model (this may take a while)..."
    ollama pull qwen3:8b
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the advanced chatbot
echo "Starting Advanced Agentic AI Chatbot..."
echo "Features: RAG + Memory + Tool Registration + Python Analysis"
echo "=========================================="
streamlit run main_advanced.py

echo "Advanced chatbot stopped. Goodbye!"