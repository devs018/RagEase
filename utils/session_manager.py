import streamlit as st
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self):
        self.session_timeout = 3600  # 1 hour
        
    @staticmethod
    def initialize_session():
        """Initialize session state variables"""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = {}
        
        if "current_conversation" not in st.session_state:
            st.session_state.current_conversation = "default"
        
        if "last_activity" not in st.session_state:
            st.session_state.last_activity = time.time()
    
    @staticmethod
    def add_message(role: str, content: str, metadata: Dict = None):
        """Add a message to the current conversation"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        st.session_state.messages.append(message)
        st.session_state.last_activity = time.time()
    
    @staticmethod
    def get_messages() -> List[Dict]:
        """Get all messages from current conversation"""
        return st.session_state.messages
    
    @staticmethod
    def clear_current_conversation():
        """Clear the current conversation"""
        st.session_state.messages = []
    
    @staticmethod
    def start_new_conversation():
        """Start a new conversation"""
        # Save current conversation if it has messages
        if st.session_state.messages:
            SessionManager.save_conversation()
        
        # Clear current conversation
        SessionManager.clear_current_conversation()
        st.session_state.current_conversation = "default"
    
    @staticmethod
    def save_conversation(name: str = None):
        """Save current conversation to history"""
        if not st.session_state.messages:
            return False
        
        conversation_name = name or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        conversation_id = str(uuid.uuid4())
        
        st.session_state.conversation_history[conversation_id] = {
            "name": conversation_name,
            "messages": st.session_state.messages.copy(),
            "created_at": datetime.now().isoformat(),
            "message_count": len(st.session_state.messages)
        }
        return True
    
    @staticmethod
    def load_conversation(conversation_id: str):
        """Load a conversation from history"""
        if conversation_id in st.session_state.conversation_history:
            conversation = st.session_state.conversation_history[conversation_id]
            st.session_state.messages = conversation["messages"].copy()
            st.session_state.current_conversation = conversation_id
            return True
        return False
    
    @staticmethod
    def delete_conversation(conversation_id: str):
        """Delete a conversation from history"""
        if conversation_id in st.session_state.conversation_history:
            del st.session_state.conversation_history[conversation_id]
            return True
        return False
    
    @staticmethod
    def get_conversation_history() -> Dict:
        """Get all conversation history"""
        return st.session_state.conversation_history