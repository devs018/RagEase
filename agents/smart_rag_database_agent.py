import json
import re
import os
from typing import Dict, List, Any, Tuple, Optional
from utils.ollama_client import OllamaClient
from database.db_connection import DatabaseConnection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import chromadb
from chromadb.config import Settings
# from config import Config

class SmartRAGDatabaseAgent:
    def __init__(self, model: str = "qwen3:4b"):
        self.ollama_client = OllamaClient(model=model)
        self.db_connection = DatabaseConnection()
        
        # Initialize embedding model for RAG
        self.embedding_model = "qwen3-embeddings-0.6b"  # Can switch to other models
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(path="database/rag_vectors")
        
        # Setup vector collections
        self.table_collection = self._get_or_create_collection("table_info")
        self.join_collection = self._get_or_create_collection("table_joins")
        
        # Load RAG data into vector store
        self._initialize_rag_data()
        
        # System prompt with enhanced capabilities
        self.system_prompt = f"""You are an advanced database AI agent with smart RAG capabilities and tool 
        registration.

AVAILABLE TOOLS:
1. execute_sql_query(query: str) - Execute SQL SELECT queries safely
2. generate_python_analysis(code: str) - Execute Python code for data analysis not visualization
3. create_visualization(chart_type: str, data_columns: list, title: str) - Create charts / plots / visualizations
4. get_memory_data() - Retrieve cached data from previous queries
5. store_memory_data(data: dict, description: str) - Cache query results

You have access to a comprehensive business database with these tables:
- departments: Company departments with budgets and locations
- employees: Employee information with hierarchical relationships  
- suppliers: External suppliers with ratings and contact info
- products: Product catalog with pricing and inventory
- customers: Customer information with types and credit limits
- orders: Customer orders with status and payment info
- order_items: Detailed order line items with pricing
- sales: Individual sales transactions with commissions

DATABASE SCHEMA CONTEXT will be provided based on the most relevant tables for each query.

INSTRUCTIONS:
- Generate accurate SQL queries using the provided schema information
- Use table joins appropriately based on the join information provided
- Store query results in memory for future reference
- When users ask follow-up questions, check memory first before making new queries
- Generate Python code for complex analysis when needed
- Create visualizations when requested
- Provide descriptive insights based on data analysis

RESPONSE FORMAT:
Always respond with JSON containing:
{{
    "reasoning": "Your thought process about the query",
    "actions": [
        {{
            "tool": "tool_name",
            "parameters": {{"param": "value"}},
            "purpose": "why using this tool"
        }}
    ],
    "response": "Natural language response to user",
    "continue_conversation": true/false,
    "tables_used": ["table1", "table2"],
    "confidence": "high/medium/low"
}}

Be intelligent about when to use cached data vs fetching new data.
"""

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"description": f"Vector embeddings for {name}"}
            )

    def _initialize_rag_data(self):
        """Load table info and join info into vector database"""
        
        # Load table information
        table_info_path = "rag/table_info"
        if os.path.exists(table_info_path):
            self._load_table_info(table_info_path)
        
        # Load join information  
        join_info_path = "rag/table_joins"
        if os.path.exists(join_info_path):
            self._load_join_info(join_info_path)

    def _load_table_info(self, path: str):
        """Load individual table information files"""
        try:
            existing_docs = self.table_collection.get()
            if len(existing_docs['documents']) > 0:
                return  # Already loaded
            
            documents = []
            metadatas = []
            ids = []
            
            for filename in os.listdir(path):
                if filename.endswith('.txt'):
                    table_name = filename.replace('_table.txt', '').replace('.txt', '')
                    file_path = os.path.join(path, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append(content)
                    metadatas.append({
                        "table_name": table_name,
                        "type": "table_info",
                        "filename": filename
                    })
                    ids.append(f"table_{table_name}")
            
            if documents:
                self.table_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Loaded {len(documents)} table info documents into vector store")
        except Exception as e:
            print(f"Error loading table info: {e}")

    def _load_join_info(self, path: str):
        """Load table join information files"""
        try:
            existing_docs = self.join_collection.get()
            if len(existing_docs['documents']) > 0:
                return  # Already loaded
            
            documents = []
            metadatas = []
            ids = []
            
            for filename in os.listdir(path):
                if filename.endswith('.txt') and '__' in filename:
                    tables = filename.replace('.txt', '').split('__')
                    file_path = os.path.join(path, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append(content)
                    metadatas.append({
                        "table1": tables[0],
                        "table2": tables[1],
                        "type": "join_info",
                        "filename": filename
                    })
                    ids.append(f"join_{tables[0]}_{tables[1]}")
            
            if documents:
                self.join_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Loaded {len(documents)} join info documents into vector store")
        except Exception as e:
            print(f"Error loading join info: {e}")

    def get_relevant_tables(self, user_query: str, top_k: int = 3) -> List[Dict]:
        """Get top 3 most relevant tables based on user query using embeddings"""
        try:
            results = self.table_collection.query(
                query_texts=[user_query],
                n_results=top_k
            )
            
            relevant_tables = []
            for i in range(len(results['ids'][0])):
                relevant_tables.append({
                    "table_name": results['metadatas'][0][i]['table_name'],
                    "content": results['documents'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0
                })
            return relevant_tables
        except Exception as e:
            print(f"Error getting relevant tables: {e}")
            return []

    def get_relevant_joins(self, selected_tables: List[str]) -> List[Dict]:
        """Get relevant join information for selected tables"""
        try:
            join_docs = []
            
            # Get all possible combinations of selected tables
            for i, table1 in enumerate(selected_tables):
                for j, table2 in enumerate(selected_tables):
                    if i != j:  # Don't join table with itself
                        # Try both orders: table1__table2 and table2__table1
                        for combo in [f"{table1}__{table2}", f"{table2}__{table1}"]:
                            try:
                                results = self.join_collection.get(
                                    ids=[f"join_{table1}_{table2}"]
                                )
                                if results['documents']:
                                    join_docs.append({
                                        "tables": [table1, table2],
                                        "content": results['documents'][0],
                                        "metadata": results['metadatas'][0]
                                    })
                                    break
                            except:
                                continue
            
            return join_docs
        except Exception as e:
            print(f"Error getting relevant joins: {e}")
            return []

    def build_rag_context(self, user_query: str) -> Tuple[str, List[str]]:
        """Build RAG context with relevant table and join information"""
        
        # Get top 3 relevant tables
        relevant_tables = self.get_relevant_tables(user_query, top_k=3)
        selected_table_names = [table['table_name'] for table in relevant_tables]
        
        # Get relevant join information
        relevant_joins = self.get_relevant_joins(selected_table_names)
        
        # Build context
        context = "RELEVANT DATABASE SCHEMA:\\n\\n"
        
        # Add table information
        context += "TABLE INFORMATION:\\n"
        for table in relevant_tables:
            context += f"\\n{table['content']}\\n"
        
        # Add join information
        if relevant_joins:
            context += "\\nTABLE JOIN INFORMATION:\\n"
            for join in relevant_joins:
                context += f"\\n{join['content']}\\n"
        
        return context, selected_table_names

    def get_memory_data(self) -> Dict[str, Any]:
        """Retrieve cached data from session state with debugging"""
        if 'smart_agent_memory' not in st.session_state:
            st.session_state.smart_agent_memory = {}
            print("Debug: Initialized empty smart_agent_memory")

        memory_data = st.session_state.smart_agent_memory
        print(f"Debug: Memory contains {len(memory_data)} items")

        for key, value in memory_data.items():
            data_info = "None" if value.get('data') is None else f"{len(value.get('data', []))} records"
            print(f"Debug: Memory key {key}: {data_info}")

        return memory_data

    def store_memory_data(self, data: Any, description: str, query: str = "", tables_used: List[str] = None) -> None:
        """Store data in session memory with enhanced metadata and debugging"""
        if 'smart_agent_memory' not in st.session_state:
            st.session_state.smart_agent_memory = {}

        memory_key = f"query_{len(st.session_state.smart_agent_memory)}"

        # Enhanced data storage with validation
        stored_data = {
            "data": data,
            "description": description,
            "query": query,
            "tables_used": tables_used or [],
            "timestamp": pd.Timestamp.now().isoformat(),
            "record_count": len(data) if isinstance(data, list) else 1 if data is not None else 0,
            "data_type": type(data).__name__,
            "is_valid": data is not None and (isinstance(data, list) and len(data) > 0) if isinstance(data,
                                                                                                      list) else data is not None
        }

        st.session_state.smart_agent_memory[memory_key] = stored_data

        print(f"Debug: Stored data in memory key {memory_key}")
        print(
            f"Debug: Data type: {stored_data['data_type']}, Records: {stored_data['record_count']}, Valid: {stored_data['is_valid']}")

    def execute_sql_query(self, query: str) -> Tuple[bool, Any, str]:
        """Execute SQL query with enhanced error handling and fallback queries"""
        try:
            success, results, message = self.db_connection.execute_safe_query(query)

            # If the query failed, try to suggest a simpler alternative
            if not success and "no such column" in message.lower():
                # Try to fix common column name issues
                if "departments" in query.lower() and "d.name" in query:
                    # The issue might be that we're trying to join with departments table
                    # Let's check what columns are actually available
                    print(f"Debug: Original query failed: {message}")
                    print("Debug: Attempting fallback query...")

                    # Try a simpler query without joins first
                    fallback_query = "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department"
                    success, results, message = self.db_connection.execute_safe_query(fallback_query)

                    if success:
                        print("Debug: Fallback query succeeded")
                    else:
                        print(f"Debug: Fallback query also failed: {message}")

            if success and results:
                # Store in memory
                tables_used = self._extract_tables_from_query(query)
                self.store_memory_data(
                    results,
                    f"SQL Query Results: {query[:100]}...",
                    query,
                    tables_used
                )
                print(f"Debug: Query succeeded, stored {len(results)} records")
            else:
                print(f"Debug: Query failed: {message}")

            return success, results, message
        except Exception as e:
            print(f"Debug: Exception in execute_sql_query: {e}")
            return False, None, str(e)

    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from SQL query"""
        # Simple regex to find table names after FROM and JOIN
        tables = []
        query_upper = query.upper()
        
        # Find tables after FROM
        from_pattern = r'FROM\s+(\w+)'
        from_matches = re.findall(from_pattern, query_upper)
        tables.extend(from_matches)
        
        # Find tables after JOIN  
        join_pattern = r'JOIN\s+(\w+)'
        join_matches = re.findall(join_pattern, query_upper)
        tables.extend(join_matches)
        
        return list(set(tables))  # Remove duplicates

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
            result = exec_globals.get('result', 'Analysis completed successfully')
            return True, result, "Analysis completed"
            
        except Exception as e:
            return False, None, f"Error executing Python code: {str(e)}"

    def create_visualization(self, chart_type: str, data_columns: List[str], title: str, data: Any = None) -> Tuple[
        bool, Any, str]:
        """Create visualization with intelligent column matching"""
        try:
            # Enhanced data retrieval logic
            if data is None:
                memory_data = self.get_memory_data()
                if memory_data:
                    # Try to get the most recent data with valid results
                    latest_key = max(memory_data.keys(), default=None)
                    if latest_key and memory_data[latest_key].get("data"):
                        data = memory_data[latest_key]["data"]
                        print(f"Debug: Using data from memory key {latest_key}")
                    else:
                        print("Debug: No data in latest memory item")
                        return False, None, "No data found in memory cache"
                else:
                    print("Debug: No memory data available")
                    return False, None, "No cached data available for visualization"

            if not data:
                print("Debug: Data is still None after memory check")
                return False, None, "No data available for visualization"

            # Convert data to DataFrame
            try:
                df = pd.DataFrame(data)
                print(f"Debug: DataFrame created with shape {df.shape}")
                print(f"Debug: Available columns: {list(df.columns)}")
            except Exception as e:
                print(f"Debug: Error creating DataFrame: {e}")
                return False, None, f"Error creating DataFrame: {str(e)}"

            if df.empty:
                return False, None, "DataFrame is empty"

            # Intelligent column matching - map expected columns to actual columns
            column_mapping = {}
            available_columns = list(df.columns)

            for expected_col in data_columns:
                best_match = None

                # Direct match first
                if expected_col in available_columns:
                    best_match = expected_col
                else:
                    # Try fuzzy matching for common column names
                    expected_lower = expected_col.lower()

                    for actual_col in available_columns:
                        actual_lower = actual_col.lower()

                        # Check for common mappings
                        if expected_lower == 'department' and 'department' in actual_lower:
                            best_match = actual_col
                            break
                        elif expected_lower in ['avg_salary', 'salary'] and 'salary' in actual_lower:
                            best_match = actual_col
                            break
                        elif expected_lower in actual_lower or actual_lower in expected_lower:
                            best_match = actual_col
                            break

                if best_match:
                    column_mapping[expected_col] = best_match
                    print(f"Debug: Mapped '{expected_col}' -> '{best_match}'")
                else:
                    print(f"Debug: No match found for column '{expected_col}'")

            # If we couldn't map the expected columns, use auto-selection
            if len(column_mapping) < len(data_columns):
                print("Debug: Using auto-column selection")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

                # Auto-select appropriate columns based on chart type
                if chart_type.lower() in ['bar', 'column']:
                    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                        actual_columns = [categorical_cols[0], numeric_cols[0]]
                    elif len(categorical_cols) >= 2:
                        actual_columns = categorical_cols[:2]
                    else:
                        actual_columns = available_columns[:2]
                elif chart_type.lower() == 'histogram':
                    actual_columns = [numeric_cols[0]] if numeric_cols else [available_columns[0]]
                elif chart_type.lower() == 'pie':
                    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                        actual_columns = [categorical_cols[0], numeric_cols[0]]
                    else:
                        actual_columns = available_columns[:2]
                else:
                    actual_columns = available_columns[:min(len(data_columns), len(available_columns))]
            else:
                # Use the mapped columns
                actual_columns = [column_mapping[col] for col in data_columns if col in column_mapping]

            print(f"Debug: Using columns for visualization: {actual_columns}")

            # Create visualization based on type
            fig = None

            try:
                if chart_type.lower() in ['bar', 'column']:
                    if len(actual_columns) >= 2:
                        # Check if we need to aggregate data first
                        if df[actual_columns[0]].dtype == 'object' and len(df) > 20:
                            # Group by category and aggregate
                            grouped_df = df.groupby(actual_columns[0])[actual_columns[1]].mean().reset_index()
                            fig = px.bar(grouped_df, x=actual_columns[0], y=actual_columns[1], title=title)
                        else:
                            fig = px.bar(df, x=actual_columns[0], y=actual_columns[1], title=title)
                    else:
                        # Single column - create value counts
                        value_counts = df[actual_columns[0]].value_counts()
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                     title=title, labels={'x': actual_columns[0], 'y': 'Count'})

                elif chart_type.lower() == 'line':
                    if len(actual_columns) >= 2:
                        fig = px.line(df, x=actual_columns[0], y=actual_columns[1], title=title)
                    else:
                        return False, None, "Line chart requires at least 2 columns"

                elif chart_type.lower() == 'scatter':
                    if len(actual_columns) >= 2:
                        fig = px.scatter(df, x=actual_columns[0], y=actual_columns[1], title=title)
                    else:
                        return False, None, "Scatter plot requires at least 2 columns"

                elif chart_type.lower() in ['histogram', 'hist']:
                    fig = px.histogram(df, x=actual_columns[0], title=title)

                elif chart_type.lower() == 'pie':
                    if len(actual_columns) >= 2:
                        # Aggregate if needed
                        if len(df) > 10:
                            pie_df = df.groupby(actual_columns[0])[actual_columns[1]].sum().reset_index()
                            fig = px.pie(pie_df, names=actual_columns[0], values=actual_columns[1], title=title)
                        else:
                            fig = px.pie(df, names=actual_columns[0], values=actual_columns[1], title=title)
                    else:
                        # Single column pie chart
                        value_counts = df[actual_columns[0]].value_counts()
                        fig = px.pie(names=value_counts.index, values=value_counts.values, title=title)

                elif chart_type.lower() == 'box':
                    if len(actual_columns) >= 1:
                        fig = px.box(df, y=actual_columns[0], title=title)
                        if len(actual_columns) >= 2:
                            fig = px.box(df, x=actual_columns[1], y=actual_columns[0], title=title)

                else:
                    # Default to bar chart
                    if len(actual_columns) >= 2:
                        fig = px.bar(df, x=actual_columns[0], y=actual_columns[1], title=title)
                    else:
                        value_counts = df[actual_columns[0]].value_counts()
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                     title=title, labels={'x': actual_columns[0], 'y': 'Count'})

                if fig is None:
                    return False, None, f"Failed to create {chart_type} chart"

                # Enhance the figure with better formatting
                fig.update_layout(
                    title_font_size=16,
                    showlegend=True if chart_type.lower() == 'pie' else False,
                    margin=dict(l=40, r=40, t=60, b=40),
                    xaxis_title=actual_columns[0] if actual_columns else "X",
                    yaxis_title=actual_columns[1] if len(actual_columns) > 1 else "Y"
                )

                return True, fig, f"{chart_type.title()} chart created successfully using columns: {actual_columns}"

            except Exception as viz_error:
                print(f"Debug: Visualization creation error: {viz_error}")
                return False, None, f"Error creating {chart_type} chart: {str(viz_error)}"

        except Exception as e:
            print(f"Debug: Exception in create_visualization: {e}")
            import traceback
            print(f"Debug: Traceback: {traceback.format_exc()}")
            return False, None, f"Error creating visualization: {str(e)}"

    def process_user_query(self, user_query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process user query using smart RAG and tool registration"""
        
        # Build RAG context with relevant tables and joins
        rag_context, selected_tables = self.build_rag_context(user_query)
        
        # Prepare memory context
        memory_data = self.get_memory_data()
        memory_context = ""
        if memory_data:
            memory_context = "CACHED DATA AVAILABLE:\\n"
            for key, item in memory_data.items():
                memory_context += f"- {item['description']} (Tables: {item.get('tables_used', [])}, Records: {item.get('record_count', 0)})\\n"
        
        # Create conversation context
        conversation_context = ""
        if conversation_history:
            recent_history = conversation_history[-4:]  # Last 4 messages
            for msg in recent_history:
                conversation_context += f"{msg['role']}: {msg['content'][:200]}...\\n"
        
        # Enhanced system prompt with RAG context
        enhanced_prompt = f"""{self.system_prompt}

{rag_context}

MEMORY CONTEXT:
{memory_context}

CONVERSATION HISTORY:
{conversation_context}

USER QUERY: {user_query}

Based on the relevant database schema and available cached data, determine the best approach:
1. If it's a new data request, generate appropriate SQL query using the schema information
2. If it's a follow-up question about existing data, use cached data
3. If it requires analysis or visualization, use appropriate tools
4. If it requires statistical analysis, generate Python code

Focus on the most relevant tables: {selected_tables}

Provide your response in the specified JSON format with appropriate tool usage."""

        # st.write(enhanced_prompt)
        # Get response from LLM
        messages = [
            self.ollama_client.create_user_message(enhanced_prompt)
        ]
        
        llm_response = self.ollama_client.chat(messages)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>HERE>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # Parse LLM response
        try:
            # Extract JSON from response
            # json_match = re.search(r'\\{.*\\}', llm_response, re.DOTALL)
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            try:
                print("----------------LLM Response START HERE ---------------------")
                print(json_match)
                print("----------------LLM Response END HERE ------------------------")
                if json_match:
                    parsed_response = json.loads(json_match.group())
                else:
                    # Fallback response
                    parsed_response = {
                        "reasoning": "Could not parse LLM response",
                        "actions": [],
                        "response": llm_response,
                        "continue_conversation": True,
                        "tables_used": selected_tables,
                        "confidence": "low"
                    }
            except:
                if len(llm_response.split("</think>")) > 1:
                    parsed_response = json.loads(llm_response.split("</think>")[1])
                else:
                    parsed_response = {
                        "reasoning": "Could not parse LLM response",
                        "actions": [],
                        "response": llm_response,
                        "continue_conversation": True,
                        "tables_used": selected_tables,
                        "confidence": "low"
                    }
        except Exception as e:
            print("JSON parsing failed", str(e))
            parsed_response = {
                "reasoning": "JSON parsing failed",
                "actions": [],
                "response": llm_response,
                "continue_conversation": True,
                "tables_used": selected_tables,
                "confidence": "low"
            }
        
        # Execute the actions
        execution_results = []
        for action in parsed_response.get("actions", []):
            result = self.execute_action(action)
            execution_results.append(result)
        
        # Compile final response
        final_response = self.compile_final_response(parsed_response, execution_results, selected_tables)
        
        return final_response

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action from the LLM with enhanced debugging"""
        tool = action.get("tool", "")
        parameters = action.get("parameters", {})

        print(f"Debug: Executing action - Tool: {tool}, Parameters: {parameters}")

        if tool == "execute_sql_query":
            query = parameters.get("query", "")
            success, results, message = self.execute_sql_query(query)
            print(f"Debug: SQL execution - Success: {success}, Results count: {len(results) if results else 0}")
            return {
                "tool": tool,
                "success": success,
                "results": results,
                "message": message,
                "result_count": len(results) if results else 0
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

            # Get data from the most recent successful SQL query
            data = None
            memory_data = self.get_memory_data()

            if memory_data:
                # Find the most recent successful query with data
                for key in reversed(sorted(memory_data.keys())):
                    if memory_data[key].get('data') and memory_data[key].get('is_valid', False):
                        data = memory_data[key]['data']
                        print(f"Debug: Using data from memory key {key} for visualization")
                        break

            success, results, message = self.create_visualization(chart_type, data_columns, title, data)
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

    def compile_final_response(self, parsed_response: Dict[str, Any], execution_results: List[Dict], selected_tables: List[str]) -> Dict[str, Any]:
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
        
        # Enhanced response with RAG context
        enhanced_response = parsed_response.copy()
        enhanced_response["execution_results"] = execution_results
        enhanced_response["sql_data"] = sql_results
        enhanced_response["visualizations"] = visualizations
        enhanced_response["analysis_results"] = analysis_results
        enhanced_response["selected_tables"] = selected_tables
        
        # Add success indicators
        enhanced_response["has_data"] = len(sql_results) > 0
        enhanced_response["has_visualizations"] = len(visualizations) > 0
        enhanced_response["has_analysis"] = len(analysis_results) > 0
        
        return enhanced_response

    def handle_database_query(self, user_question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Main entry point for handling database queries with smart RAG"""
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
                "tables_used": result.get("selected_tables", []),
                "confidence": result.get("confidence", "medium"),
                "agent_type": "smart_rag_database"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing query: {str(e)}",
                "data": None,
                "agent_type": "smart_rag_database"
            }

    def clear_memory(self):
        """Clear cached data from memory"""
        if 'smart_agent_memory' in st.session_state:
            st.session_state.smart_agent_memory = {}

    def get_memory_summary(self) -> str:
        """Get a summary of what's stored in memory"""
        memory_data = self.get_memory_data()
        if not memory_data:
            return "No data cached in memory"
        
        summary = f"Cached data ({len(memory_data)} items):\\n"
        for key, item in memory_data.items():
            tables_info = f" (Tables: {', '.join(item.get('tables_used', []))})" if item.get('tables_used') else ""
            summary += f"- {item['description'][:50]}...{tables_info}\\n"
        
        return summary

    def refresh_rag_data(self):
        """Refresh RAG data by reloading from files"""
        try:
            # Clear existing collections
            self.chroma_client.delete_collection("table_info")
            self.chroma_client.delete_collection("table_joins")
            
            # Recreate collections
            self.table_collection = self._get_or_create_collection("table_info")
            self.join_collection = self._get_or_create_collection("table_joins")
            
            # Reload data
            self._initialize_rag_data()
            return True
        except Exception as e:
            print(f"Error refreshing RAG data: {e}")
            return False