import json
import re
from typing import Dict, List, Any, Tuple, Optional
from utils.ollama_client import OllamaClient
from database.db_connection import DatabaseConnection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import streamlit as st

class AdvancedDatabaseAgent:
    def __init__(self, model: str = "qwen3:4b"):
        self.ollama_client = OllamaClient(model=model)
        self.db_connection = DatabaseConnection()
        
        # Load RAG context for database schemas
        self.rag_context = self.load_database_schemas()
        
        # System prompt with tools registration
        self.system_prompt = f"""You are an advanced database AI agent with the following capabilities:

AVAILABLE TOOLS:
1. execute_sql_query(query: str) - Execute SQL SELECT queries safely
2. generate_python_analysis(code: str) - Execute Python code for data analysis
3. create_visualization(chart_type: str, data_columns: list, title: str) - Create charts
4. get_memory_data() - Retrieve cached data from previous queries
5. store_memory_data(data: dict, description: str) - Cache query results

DATABASE SCHEMA CONTEXT:
{self.rag_context}

INSTRUCTIONS:
- Use the database schema to generate accurate SQL queries
- Always validate table and column names against the schema
- Store query results in memory for future reference
- When users ask follow-up questions, check memory first
- Generate Python code for complex analysis when needed
- Create visualizations when requested
- Provide descriptive insights based on data

RESPONSE FORMAT:
Always respond with JSON containing:
{{
    "reasoning": "Your thought process",
    "actions": [
        {{
            "tool": "tool_name",
            "parameters": {{"param": "value"}},
            "purpose": "why using this tool"
        }}
    ],
    "response": "Natural language response to user",
    "continue_conversation": true/false
}}

Be intelligent about when to use cached data vs fetching new data.
"""

    def load_database_schemas(self) -> str:
        """Load database schema information from RAG files and database"""
        context = "DATABASE SCHEMA INFORMATION:\n\n"
        
        # Get actual database schema
        try:
            tables = self.db_connection.get_table_names()
            for table_name in tables:
                schema = self.db_connection.get_table_schema(table_name)
                sample_data = self.db_connection.execute_query(f"SELECT * FROM {table_name} LIMIT 3")
                
                context += f"TABLE: {table_name}\n"
                context += "COLUMNS:\n"
                for col in schema:
                    context += f"  - {col['name']} ({col['type']})\n"
                
                if sample_data:
                    context += "SAMPLE DATA:\n"
                    for row in sample_data:
                        context += f"  {dict(row)}\n"
                context += "\n"
        except Exception as e:
            context += f"Error loading schema: {e}\n"
        
        # Load RAG documentation if available
        try:
            import os
            rag_dir = "rag"
            if os.path.exists(rag_dir):
                for filename in os.listdir(rag_dir):
                    if filename.endswith('.txt'):
                        with open(os.path.join(rag_dir, filename), 'r', encoding='utf-8') as f:
                            content = f.read()
                            context += f"DOCUMENTATION FROM {filename}:\n{content}\n\n"
        except:
            pass
        
        return context

    def get_memory_data(self) -> Dict[str, Any]:
        """Retrieve cached data from session state"""
        if 'agent_memory' not in st.session_state:
            st.session_state.agent_memory = {}
        return st.session_state.agent_memory

    def store_memory_data(self, data: Any, description: str, query: str = "") -> None:
        """Store data in session memory"""
        if 'agent_memory' not in st.session_state:
            st.session_state.agent_memory = {}
        
        memory_key = f"query_{len(st.session_state.agent_memory)}"
        st.session_state.agent_memory[memory_key] = {
            "data": data,
            "description": description,
            "query": query,
            "timestamp": pd.Timestamp.now().isoformat()
        }

    def execute_sql_query(self, query: str) -> Tuple[bool, Any, str]:
        """Execute SQL query and return results"""
        try:
            success, results, message = self.db_connection.execute_safe_query(query)
            if success and results:
                # Store in memory
                self.store_memory_data(results, f"SQL Query Results: {query[:50]}...", query)
            return success, results, message
        except Exception as e:
            return False, None, str(e)

    def generate_python_analysis(self, code: str, data: Any = None) -> Tuple[bool, Any, str]:
        """Execute Python code for data analysis"""
        try:
            # Prepare execution environment
            memory_data = self.get_memory_data()
            latest_data = None
            
            # Get the most recent data if no specific data provided
            if data is None and memory_data:
                latest_key = max(memory_data.keys())
                latest_data = memory_data[latest_key]["data"]
            else:
                latest_data = data
            
            # Create safe execution environment
            exec_globals = {
                'pd': pd,
                'plt': plt,
                'sns': sns,
                'px': px,
                'go': go,
                'data': latest_data,
                'df': pd.DataFrame(latest_data) if latest_data else None,
                'memory_data': memory_data,
                'np': __import__('numpy'),
                'result': None
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Return the result
            result = exec_globals.get('result', 'Code executed successfully')
            return True, result, "Analysis completed"
            
        except Exception as e:
            return False, None, f"Error executing Python code: {str(e)}"

    def create_visualization(self, chart_type: str, data_columns: List[str], title: str, data: Any = None) -> Tuple[bool, Any, str]:
        """Create visualization based on requirements"""
        try:
            # Get data from memory if not provided
            if data is None:
                memory_data = self.get_memory_data()
                if memory_data:
                    latest_key = max(memory_data.keys())
                    data = memory_data[latest_key]["data"]
            
            if not data:
                return False, None, "No data available for visualization"
            
            df = pd.DataFrame(data)
            
            # Create visualization based on type
            if chart_type.lower() == 'bar':
                if len(data_columns) >= 2:
                    fig = px.bar(df, x=data_columns[0], y=data_columns[1], title=title)
                else:
                    fig = px.bar(df, x=data_columns[0], title=title)
            elif chart_type.lower() == 'line':
                fig = px.line(df, x=data_columns[0], y=data_columns[1], title=title)
            elif chart_type.lower() == 'scatter':
                fig = px.scatter(df, x=data_columns[0], y=data_columns[1], title=title)
            elif chart_type.lower() == 'histogram':
                fig = px.histogram(df, x=data_columns[0], title=title)
            elif chart_type.lower() == 'pie':
                fig = px.pie(df, names=data_columns[0], values=data_columns[1], title=title)
            else:
                # Default to bar chart
                fig = px.bar(df, x=data_columns[0], y=data_columns[1] if len(data_columns) > 1 else data_columns[0], title=title)
            
            return True, fig, "Visualization created successfully"
            
        except Exception as e:
            return False, None, f"Error creating visualization: {str(e)}"

    def process_user_query(self, user_query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process user query using advanced agent capabilities"""
        
        # Prepare context with memory
        memory_data = self.get_memory_data()
        memory_context = ""
        if memory_data:
            memory_context = "CACHED DATA AVAILABLE:\n"
            for key, item in memory_data.items():
                memory_context += f"- {item['description']} (Query: {item['query'][:50]}...)\n"
        
        # Create conversation context
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-4:]  # Last 4 messages
            for msg in recent_history:
                conversation_context += f"{msg['role']}: {msg['content']}\n"
        
        # Prepare the prompt
        user_prompt = f"""
MEMORY CONTEXT:
{memory_context}

CONVERSATION HISTORY:
{conversation_context}

USER QUERY: {user_query}

Analyze the user's query and determine the best approach:
1. If it's a new data request, generate appropriate SQL query
2. If it's a follow-up question about existing data, use cached data
3. If it's asking for analysis or visualization, use appropriate tools
4. If it requires complex analysis, generate Python code

Provide your response in the specified JSON format with appropriate tool usage.
"""

        # Get response from LLM
        messages = [
            self.ollama_client.create_system_message(self.system_prompt),
            self.ollama_client.create_user_message(user_prompt)
        ]
        
        llm_response = self.ollama_client.chat(messages)
        
        # Parse LLM response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                parsed_response = json.loads(json_match.group())
            else:
                # Fallback response
                parsed_response = {
                    "reasoning": "Could not parse LLM response",
                    "actions": [],
                    "response": llm_response,
                    "continue_conversation": True
                }
        except:
            parsed_response = {
                "reasoning": "JSON parsing failed",
                "actions": [],
                "response": llm_response,
                "continue_conversation": True
            }
        
        # Execute the actions
        execution_results = []
        for action in parsed_response.get("actions", []):
            result = self.execute_action(action)
            execution_results.append(result)
        
        # Update response with execution results
        final_response = self.compile_final_response(parsed_response, execution_results)
        
        return final_response

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action from the LLM"""
        tool = action.get("tool", "")
        parameters = action.get("parameters", {})
        
        if tool == "execute_sql_query":
            query = parameters.get("query", "")
            success, results, message = self.execute_sql_query(query)
            return {
                "tool": tool,
                "success": success,
                "results": results,
                "message": message
            }
        
        elif tool == "generate_python_analysis":
            code = parameters.get("code", "")
            success, results, message = self.generate_python_analysis(code)
            return {
                "tool": tool,
                "success": success,
                "results": results,
                "message": message
            }
        
        elif tool == "create_visualization":
            chart_type = parameters.get("chart_type", "bar")
            data_columns = parameters.get("data_columns", [])
            title = parameters.get("title", "Chart")
            success, results, message = self.create_visualization(chart_type, data_columns, title)
            return {
                "tool": tool,
                "success": success,
                "results": results,
                "message": message
            }
        
        elif tool == "get_memory_data":
            memory_data = self.get_memory_data()
            return {
                "tool": tool,
                "success": True,
                "results": memory_data,
                "message": "Memory data retrieved"
            }
        
        elif tool == "store_memory_data":
            data = parameters.get("data", {})
            description = parameters.get("description", "")
            self.store_memory_data(data, description)
            return {
                "tool": tool,
                "success": True,
                "results": None,
                "message": "Data stored in memory"
            }
        
        else:
            return {
                "tool": tool,
                "success": False,
                "results": None,
                "message": f"Unknown tool: {tool}"
            }

    def compile_final_response(self, parsed_response: Dict[str, Any], execution_results: List[Dict]) -> Dict[str, Any]:
        """Compile final response with execution results"""
        
        # Gather all successful results
        sql_results = []
        visualizations = []
        analysis_results = []
        
        for result in execution_results:
            if result["success"]:
                if result["tool"] == "execute_sql_query":
                    sql_results.append(result["results"])
                elif result["tool"] == "create_visualization":
                    visualizations.append(result["results"])
                elif result["tool"] == "generate_python_analysis":
                    analysis_results.append(result["results"])
        
        # Enhance the response with execution results
        enhanced_response = parsed_response.copy()
        enhanced_response["execution_results"] = execution_results
        enhanced_response["sql_data"] = sql_results
        enhanced_response["visualizations"] = visualizations
        enhanced_response["analysis_results"] = analysis_results
        
        # Add success indicators
        enhanced_response["has_data"] = len(sql_results) > 0
        enhanced_response["has_visualizations"] = len(visualizations) > 0
        enhanced_response["has_analysis"] = len(analysis_results) > 0
        
        return enhanced_response

    def handle_database_query(self, user_question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Main entry point for handling database queries"""
        try:
            result = self.process_user_query(user_question, conversation_history)
            
            return {
                "success": True,
                "message": result.get("response", "Query processed"),
                "data": result.get("sql_data", []),
                "visualizations": result.get("visualizations", []),
                "analysis_results": result.get("analysis_results", []),
                "reasoning": result.get("reasoning", ""),
                "continue_conversation": result.get("continue_conversation", True),
                "agent_type": "advanced_database"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing query: {str(e)}",
                "data": None,
                "agent_type": "advanced_database"
            }

    def clear_memory(self):
        """Clear cached data from memory"""
        if 'agent_memory' in st.session_state:
            st.session_state.agent_memory = {}

    def get_memory_summary(self) -> str:
        """Get a summary of what's stored in memory"""
        memory_data = self.get_memory_data()
        if not memory_data:
            return "No data cached in memory"
        
        summary = f"Cached data ({len(memory_data)} items):\n"
        for key, item in memory_data.items():
            summary += f"- {item['description']}\n"
        
        return summary
