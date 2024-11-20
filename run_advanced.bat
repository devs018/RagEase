@echo off
echo ==========================================
echo Advanced Agentic AI Chatbot with RAG
echo ==========================================

echo Checking if Ollama is running...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Ollama is running
) else (
    echo Starting Ollama server...
    start ollama serve
    timeout /t 5
)

echo Checking Qwen3 model...
ollama list | findstr "qwen3:8b" >nul
if %errorlevel% equ 0 (
    echo Qwen3 8B model is available
) else (
    echo Pulling Qwen3 8B model...
    ollama pull qwen3:8b
)

echo Starting Advanced Agentic AI Chatbot...
echo Features: RAG + Memory + Tool Registration + Python Analysis
echo ==========================================
streamlit run main_advanced.py

pause