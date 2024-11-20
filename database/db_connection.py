import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import os
from contextlib import contextmanager

class DatabaseConnection:
    def __init__(self, db_path: str = "database/chatbot.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database directory if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a SELECT query and return results as list of dictionaries"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results
    
    def execute_non_query(self, query: str, params: tuple = None) -> bool:
        """Execute INSERT, UPDATE, DELETE queries"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return True
        except Exception as e:
            print(f"Error executing query: {e}")
            return False
    
    def get_table_names(self) -> List[str]:
        """Get all table names in the database"""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        results = self.execute_query(query)
        return [row['name'] for row in results]
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Get schema information for a table"""
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate if a query is safe to execute"""
        # Basic validation - prevent dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        query_upper = query.upper().strip()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains dangerous keyword: {keyword}"
        
        if not query_upper.startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        return True, "Query is valid"
    
    def execute_safe_query(self, query: str) -> Tuple[bool, Any, str]:
        """Execute a query with validation"""
        is_valid, message = self.validate_query(query)
        if not is_valid:
            return False, None, message
        
        try:
            results = self.execute_query(query)
            return True, results, "Query executed successfully"
        except Exception as e:
            return False, None, f"Error executing query: {str(e)}"