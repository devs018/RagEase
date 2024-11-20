import os
from typing import Dict, Any

class Config:
    # Ollama Configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen3:4b"
    
    # Database Configuration
    DATABASE_PATH = "database/chatbot.db"
    
    # RAG Configuration
    RAG_FOLDER = "rag"
    VECTOR_DB_PATH = "database/vector_db"
    
    # Streamlit Configuration
    PAGE_TITLE = "Advanced Agentic AI Chatbot"
    PAGE_ICON = "robot"
    LAYOUT = "wide"
    
    # Agent Configuration
    AGENT_MODELS = {
        "database": "qwen3:8b",
        "plotting": "qwen3:8b", 
        "general": "qwen3:8b",
        "rag": "qwen3:8b"
    }
    
    # Advanced Agent Settings
    ENABLE_MEMORY = True
    ENABLE_PYTHON_EXECUTION = True
    ENABLE_VISUALIZATION = True
    MAX_MEMORY_ITEMS = 10
    
    # Conversation Settings
    MAX_CONVERSATION_HISTORY = 20
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    
    # Tool Registration Settings
    TOOLS_CONFIG = {
        "execute_sql_query": {
            "enabled": True,
            "safe_mode": True,
            "timeout": 30
        },
        "generate_python_analysis": {
            "enabled": True,
            "safe_imports": ["pandas", "numpy", "matplotlib", "seaborn", "plotly"],
            "timeout": 60
        },
        "create_visualization": {
            "enabled": True,
            "supported_types": ["bar", "line", "scatter", "histogram", "pie", "heatmap"],
            "max_data_points": 10000
        }
    }
    
    @classmethod
    def get_db_config(cls) -> Dict[str, Any]:
        return {
            "database_path": cls.DATABASE_PATH,
            "check_same_thread": False
        }
    
    @classmethod
    def get_ollama_config(cls) -> Dict[str, Any]:
        return {
            "base_url": cls.OLLAMA_BASE_URL,
            "model": cls.OLLAMA_MODEL,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096  # Increased for complex responses
        }
    
    @classmethod
    def get_tools_config(cls) -> Dict[str, Any]:
        return cls.TOOLS_CONFIG