import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
from datetime import datetime
from config import Config
from utils.session_manager import SessionManager
from utils.ollama_client import OllamaClient
from agents.general_agent import GeneralAgent
# from agents.advanced_database_agent import AdvancedDatabaseAgent
from agents.smart_rag_database_agent import SmartRAGDatabaseAgent
from database.db_connection import DatabaseConnection
# from database.sample_data import create_sample_database
from database.enhanced_sample_data import create_enhanced_database

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class AdvancedAgenticChatbot:
    def __init__(self):
        # Initialize session
        SessionManager.initialize_session()
        
        # Initialize agents
        self.general_agent = GeneralAgent()
        # self.advanced_db_agent = AdvancedDatabaseAgent()
        self.smart_rag_agent = SmartRAGDatabaseAgent()
        self.db_connection = DatabaseConnection()
        
        # Check if sample database exists, create if not
        if not os.path.exists("database/chatbot.db"):
            # create_sample_database()
            create_enhanced_database()

    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=":robot:",
            layout=Config.LAYOUT,
            initial_sidebar_state="expanded"
        )
        
        st.title(":robot: Advanced Agentic AI Chatbot")
        st.markdown("**Powered by Qwen3 8B + RAG + Advanced Agents | Single LLM Call with Tools**")

    def render_sidebar(self):
        """Render enhanced sidebar with memory management and dynamic table view"""
        with st.sidebar:
            st.header(":speech_balloon: Conversations")

            # New conversation button
            if st.button(":heavy_plus_sign: New Conversation", width='stretch'):
                SessionManager.start_new_conversation()
                self.smart_rag_agent.clear_memory()
                st.rerun()

            # Conversation history
            conversations = SessionManager.get_conversation_history()

            if conversations:
                st.subheader("Previous Conversations")
                for conv_id, conv_data in conversations.items():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        if st.button(
                            f":memo: {conv_data['name'][:20]}...",
                            key=f"load_{conv_id}",
                            help=f"Messages: {conv_data['message_count']}"
                        ):
                            SessionManager.load_conversation(conv_id)
                            st.rerun()

                    with col2:
                        if st.button(":wastebasket:", key=f"del_{conv_id}", help="Delete"):
                            SessionManager.delete_conversation(conv_id)
                            st.rerun()

            st.divider()

            # Agent status and memory
            st.subheader(":robot: System Status")

            # Check Ollama connection
            ollama_client = OllamaClient()
            if ollama_client.is_model_available():
                st.success(":white_check_mark: Ollama Connected")
                st.info(f"Model: {Config.OLLAMA_MODEL}")
            else:
                st.error(":x: Ollama Not Available")
                st.warning("Please ensure Ollama is running and the model is pulled")
                with st.expander("Setup Instructions"):
                    st.code("ollama serve")
                    st.code("ollama pull qwen3:4b")

            # Enhanced Database status with dynamic table view
            try:
                tables = self.db_connection.get_table_names()
                st.success(f":white_check_mark: Database ({len(tables)} tables)")

                with st.expander("📊 Database Tables & Schema"):
                    for table in tables:
                        # Get table schema and row count
                        try:
                            schema = self.db_connection.get_table_schema(table)
                            row_count = self.db_connection.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                            count = row_count[0]['count'] if row_count else 0

                            st.write(f"**{table.upper()}** ({count} records)")

                            # Show column details in a compact format
                            col_details = []
                            for col in schema:
                                col_info = f"{col['name']} ({col['type']})"
                                if col.get('pk'):
                                    col_info += " [PK]"
                                col_details.append(col_info)

                            # Show columns in a more compact way
                            if len(col_details) <= 4:
                                st.caption(", ".join(col_details))
                            else:
                                # Show first 3 columns and "..." for more
                                st.caption(", ".join(col_details[:3]) + f", ... (+{len(col_details) - 3} more)")

                            st.write("---")

                        except Exception as e:
                            st.write(f"**{table.upper()}** (Error loading details)")
                            st.caption(f"Error: {str(e)}")

                # RAG System Status
                st.subheader(":brain: RAG System")
                try:
                    # Check if RAG folders exist
                    table_info_exists = os.path.exists("rag/table_info")
                    join_info_exists = os.path.exists("rag/table_joins")

                    if table_info_exists and join_info_exists:
                        # Count files
                        table_files = len([f for f in os.listdir("rag/table_info") if f.endswith('.txt')])
                        join_files = len([f for f in os.listdir("rag/table_joins") if f.endswith('.txt')])

                        st.success(f":white_check_mark: RAG Loaded")
                        st.info(f"Tables: {table_files} docs, Joins: {join_files} docs")

                        # Show which tables have documentation
                        with st.expander("📚 RAG Documentation Status"):
                            st.write("**Table Documentation:**")
                            for table in tables:
                                doc_file = f"{table}_table.txt"
                                doc_exists = os.path.exists(f"rag/table_info/{doc_file}")
                                status_icon = ":white_check_mark:" if doc_exists else ":x:"
                                st.write(f"{status_icon} {table}")

                        if st.button(":arrows_counterclockwise: Refresh RAG Data"):
                            with st.spinner("Refreshing RAG data..."):
                                success = self.smart_rag_agent.refresh_rag_data()
                                if success:
                                    st.success("RAG data refreshed!")
                                else:
                                    st.error("Failed to refresh RAG data")
                                st.rerun()
                    else:
                        st.warning(":exclamation: RAG folders not found")
                        st.info("Create rag/table_info and rag/table_joins folders")
                except Exception as e:
                    st.error(f":x: RAG Error: {str(e)}")

            except Exception as e:
                st.error(":x: Database Error")
                st.caption(str(e))

            # Memory status with enhanced details
            memory_summary = self.smart_rag_agent.get_memory_summary()
            if "No data cached" not in memory_summary:
                st.subheader(":brain: Memory Status")

                # Get detailed memory info
                memory_data = self.smart_rag_agent.get_memory_data()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Cached Items", len(memory_data))
                with col2:
                    total_records = sum(item.get('record_count', 0) for item in memory_data.values())
                    st.metric("Total Records", total_records)

                with st.expander("🧠 Cached Data Details"):
                    for key, item in memory_data.items():
                        st.write(f"**Query {key.split('_')[1]}:**")
                        st.caption(f"{item['description']}")
                        if item.get('tables_used'):
                            st.caption(f"Tables: {', '.join(item['tables_used'])}")
                        st.caption(f"Records: {item.get('record_count', 0)} | {item.get('timestamp', '')[:19]}")
                        st.write("---")

                if st.button(":wastebasket: Clear Memory"):
                    self.smart_rag_agent.clear_memory()
                    st.success("Memory cleared!")
                    st.rerun()
            else:
                st.info(":brain: No data in memory")

            st.divider()

            # Settings with enhanced database controls
            st.subheader(":gear: Settings")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(":arrows_counterclockwise: Recreate Database",
                             help="Recreate sample database with enhanced schema"):
                    with st.spinner("Creating enhanced database..."):
                        from database.enhanced_sample_data import create_enhanced_database
                        create_enhanced_database()
                        st.success("Enhanced database created!")
                        time.sleep(1)
                        st.rerun()

            with col2:
                if st.button(":chart_with_upwards_trend: Quick Stats", help="Show database statistics"):
                    with st.spinner("Calculating stats..."):
                        try:
                            stats = {}
                            for table in tables:
                                count = self.db_connection.execute_query(f"SELECT COUNT(*) as count FROM {table}")
                                stats[table] = count[0]['count'] if count else 0

                            st.write("**Database Statistics:**")
                            for table, count in stats.items():
                                st.write(f"• {table}: {count:,} records")
                        except Exception as e:
                            st.error(f"Error calculating stats: {e}")

            # Advanced features info with updated capabilities
            with st.expander(":sparkles: Advanced Features"):
                st.markdown("""
                **Smart RAG System:**
                - :brain: **Embedding-based Table Selection**: Uses vector similarity to find top 3 relevant tables
                - :link: **Intelligent Join Detection**: Automatically finds relevant table relationships
                - :floppy_disk: **Smart Memory Management**: Tracks tables used in cached data
                - :mag: **Context-Aware Queries**: Uses table documentation for accurate SQL generation

                **Enhanced Database:**
                - :building_with_columns: **8 Interconnected Tables**: Comprehensive business scenario
                - :chart_with_upwards_trend: **500+ Records**: Realistic data for analysis
                - :arrows_counterclockwise: **Foreign Key Relationships**: Proper data modeling

                **Tool Registration:**
                - :gear: **Single LLM Call**: Efficient processing with multiple tools
                - :snake: **Python Code Generation**: Statistical analysis and data processing
                - :bar_chart: **Smart Visualizations**: Context-appropriate charts
                """)

            # Help section with updated examples
            with st.expander(":question: Help & Examples"):
                st.markdown("""
                **Smart Database Queries:**
                - "Show employee information with departments" 
                - "What are the top selling products by revenue?"
                - "Analyze customer purchase patterns by type"
                - "Show supplier performance with ratings"

                **Advanced Analysis:**
                - "Calculate correlation between employee age and salary"
                - "Analyze seasonal sales trends by product category"
                - "Compare department performance metrics"
                - "Show customer lifetime value analysis"

                **Smart Follow-ups:**
                After any query, ask:
                - "Show me more details about this data"
                - "Create visualizations from this information"
                - "What insights can you find in this data?"
                - "How do these numbers compare to industry standards?"

                **Memory Usage:**
                - Data is automatically cached after queries
                - Follow-up questions intelligently use cached data
                - Ask "what data do we have in memory?" to see cached information
                """)

    def classify_and_route_query(self, user_input: str) -> dict:
        """Enhanced query classification and routing"""
        
        # Get conversation history for context
        conversation_history = SessionManager.get_messages()
        
        # Simple classification
        input_lower = user_input.lower()
        
        data_keywords = [
            'show', 'employee', 'employees', 'sales', 'salary', 'department', 
            'table', 'count', 'average', 'sum', 'data', 'list', 'total',
            'highest', 'lowest', 'recent', 'latest', 'top', 'bottom',
            'plot', 'chart', 'graph', 'visualize', 'histogram', 'bar',
            'analyze', 'analysis', 'correlation', 'summary', 'report',
            'trend', 'pattern', 'statistics'
        ]
        
        # Check if it's a data-related query or general query
        if any(keyword in input_lower for keyword in data_keywords):
            # Use advanced database agent
            result = self.smart_rag_agent.handle_database_query(user_input, conversation_history)
            return result
        else:
            # Use general agent for non-data queries
            if user_input.lower() == "help":
                response = self.general_agent.provide_help()
            else:
                response = self.general_agent.handle_general_query(user_input, conversation_history)
            
            return {
                "success": True,
                "message": response,
                "agent_type": "general"
            }

    def render_message(self, message: dict):
        """Enhanced message rendering with advanced features"""
        
        role = message["role"]
        content = message["content"]
        metadata = message.get("metadata", {})
        # st.write("INSIDE render_message")
        # st.write(message)
        
        with st.chat_message(role):
            st.write(content)
            
            # Show additional information based on agent type
            agent_type = metadata.get("agent_type")
            
            if agent_type == "smart_rag_database":
                # Show reasoning if available
                if metadata.get("reasoning"):
                    with st.expander(":brain: AI Reasoning"):
                        st.write(metadata["reasoning"])
                
                # Show SQL queries
                if metadata.get("sql_data"):
                    with st.expander(":memo: Query Details"):
                        if metadata.get("query"):
                            st.code(metadata["query"], language="sql")
                        
                        # Show data
                        for i, data in enumerate(metadata["sql_data"]):
                            if data:
                                df = pd.DataFrame(data)
                                st.write(f"**Results {i+1}:**")
                                st.dataframe(df, width='stretch')
                
                # Show visualizations
                if metadata.get("visualizations"):
                    st.write("**Generated Visualizations:**")
                    for i, fig in enumerate(metadata["visualizations"]):
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                
                # Show analysis results
                if metadata.get("analysis_results"):
                    with st.expander(":chart_with_upwards_trend: Analysis Results"):
                        for result in metadata["analysis_results"]:
                            st.write(result)

    def run(self):
        """Enhanced main application loop"""
        self.setup_page()
        self.render_sidebar()
        
        # Display chat messages
        messages = SessionManager.get_messages()
        
        # Show enhanced welcome message
        if not messages:
            with st.chat_message("assistant"):
                welcome_msg = """
:wave: **Welcome to your Advanced Agentic AI Chatbot!**

I'm powered by **Qwen3 8B** with advanced capabilities:

:gear: **Single LLM Call Architecture** - Efficient tool registration and execution
:brain: **RAG-Enhanced Database Queries** - Uses schema documentation for accurate SQL
:floppy_disk: **Smart Memory Management** - Caches data for follow-up questions  
:bar_chart: **Automatic Visualizations** - Creates charts based on your requests
:snake: **Python Analysis** - Generates and executes analysis code
:speech_balloon: **Conversation Continuity** - Maintains context across interactions

**Try these advanced queries:**
- "Show me employee data and create a salary analysis"
- "What's the correlation between age and salary?" 
- "Create a comprehensive sales dashboard"
- "Analyze department performance with visualizations"

**Smart Follow-ups:**
After I fetch data, ask follow-up questions like:
- "Show me more details about this"
- "Create a chart from this data"  
- "What insights can you find?"

What would you like to explore?
                """
                st.markdown(welcome_msg)
        
        # Display conversation history
        for message in messages:
            self.render_message(message)
        
        # Enhanced chat input
        if prompt := st.chat_input("Ask me anything! I can fetch data, analyze it, create visualizations, and answer follow-ups intelligently."):
            # Add user message
            SessionManager.add_message("user", prompt)
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Process the query with enhanced agent
            with st.chat_message("assistant"):
                with st.spinner("Processing with advanced AI agent..."):
                    result = self.classify_and_route_query(prompt)
                
                # Display response
                st.write(result["message"])
                
                # Prepare enhanced metadata
                metadata = {
                    "agent_type": result.get("agent_type", "general"),
                    "timestamp": datetime.now().isoformat(),
                    "reasoning": result.get("reasoning", ""),
                    "continue_conversation": result.get("continue_conversation", True)
                }
                
                # Add specific metadata for advanced database agent
                if result.get("agent_type") == "smart_rag_database":
                    metadata.update({
                        "sql_data": result.get("data", []),
                        "visualizations": result.get("visualizations", []),
                        "analysis_results": result.get("analysis_results", []),
                        "query": result.get("query", "")
                    })
                
                # Add assistant message with enhanced metadata
                SessionManager.add_message("assistant", result["message"], metadata)
                
                # Show advanced features in real-time
                if result.get("agent_type") == "smart_rag_database":
                    # Show reasoning
                    if result.get("reasoning"):
                        with st.expander(":brain: AI Reasoning"):
                            st.write(result["reasoning"])
                    
                    # Show data results
                    if result.get("data"):
                        with st.expander(":memo: Data Results"):
                            for i, data in enumerate(result["data"]):
                                if data:
                                    df = pd.DataFrame(data)
                                    st.write(f"**Dataset {i+1}:**")
                                    st.dataframe(df, width='stretch')
                    
                    # Show visualizations
                    if result.get("visualizations"):
                        st.write("**Generated Visualizations:**")
                        for fig in result["visualizations"]:
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Show analysis results
                    if result.get("analysis_results"):
                        with st.expander(":chart_with_upwards_trend: Analysis Results"):
                            for analysis in result["analysis_results"]:
                                st.write(analysis)
                    
                    # Show memory status update
                    if result.get("data"):
                        st.info(":brain: Data cached in memory for follow-up questions")


if __name__ == "__main__":
    app = AdvancedAgenticChatbot()
    app.run()
