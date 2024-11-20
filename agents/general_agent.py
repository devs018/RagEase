from typing import Dict, List, Any
from utils.ollama_client import OllamaClient

class GeneralAgent:
    def __init__(self, model: str = "qwen3:8b"):
        self.ollama_client = OllamaClient(model=model)
        
        self.system_prompt = """You are a helpful AI assistant with broad knowledge. You can:
1. Answer general questions on various topics
2. Provide explanations and educational content
3. Help with problem-solving and analysis
4. Engage in meaningful conversations

Be helpful, accurate, and conversational in your responses."""

    def handle_general_query(self, user_question: str, conversation_history: List[Dict] = None) -> str:
        """Handle general knowledge questions and conversations"""
        
        messages = [self.ollama_client.create_system_message(self.system_prompt)]
        
        # Add conversation history if available
        if conversation_history:
            # Add last few messages for context (limit to avoid token overflow)
            recent_history = conversation_history[-6:]  # Last 6 messages
            for msg in recent_history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add current user question
        messages.append(self.ollama_client.create_user_message(user_question))
        
        response = self.ollama_client.chat(messages)
        return response

    def classify_query_intent(self, user_question: str) -> Dict[str, Any]:
        """Classify what type of query this is"""
        
        # Simple keyword-based classification
        question_lower = user_question.lower()
        
        database_keywords = [
            'show', 'employee', 'employees', 'sales', 'salary', 'department', 
            'table', 'count', 'average', 'sum', 'data', 'list', 'total',
            'highest', 'lowest', 'recent', 'latest', 'top', 'bottom'
        ]
        plotting_keywords = ['plot', 'chart', 'graph', 'visualize', 'show me', 'histogram', 'bar chart']
        
        if any(keyword in question_lower for keyword in database_keywords):
            return {
                "primary_category": "database",
                "confidence": "medium",
                "explanation": "Contains database-related keywords",
                "requires_data": True
            }
        elif any(keyword in question_lower for keyword in plotting_keywords):
            return {
                "primary_category": "plotting", 
                "confidence": "medium",
                "explanation": "Contains plotting-related keywords",
                "requires_data": True
            }
        else:
            return {
                "primary_category": "general",
                "confidence": "medium", 
                "explanation": "General knowledge question",
                "requires_data": False
            }

    def provide_help(self) -> str:
        """Provide help information about the chatbot capabilities"""
        help_text = """
**Welcome to your Agentic AI Chatbot!**

I'm your AI assistant powered by Qwen3 8B. Here's what I can help you with:

**Database Queries**
- Ask questions about your data
- Generate and execute SQL queries
- Get insights and analysis from your database
- Examples: 
  - "Show me all employees"
  - "What's the average salary by department?"
  - "List recent sales"
  - "Show me top selling products"

**General Questions**
- Ask me anything about general topics
- Get explanations, help with problems
- Have conversations about various subjects
- Examples:
  - "Explain machine learning"
  - "Help me with Python programming"
  - "What is data science?"

**Tips:**
- Be specific in your questions for better results
- You can ask follow-up questions to continue conversations
- Use "show tables" to see available database tables
- Try asking about employees, sales, salaries, or departments

**Available Data:**
Your database contains sample data about:
- Employees (names, departments, salaries, hire dates)
- Sales (products, amounts, dates, regions)

What would you like to know?
        """
        return help_text.strip()