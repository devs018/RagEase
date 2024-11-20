import re
import json
from typing import Dict, List, Any, Tuple
from utils.ollama_client import OllamaClient
from database.db_connection import DatabaseConnection

class DatabaseAgent:
    def __init__(self, model: str = "qwen3:8b"):
        self.ollama_client = OllamaClient(model=model)
        self.db_connection = DatabaseConnection()

    def generate_simple_sql(self, user_input: str) -> str:
        """Generate simple SQL queries based on common patterns"""
        input_lower = user_input.lower()
        
        # Pattern matching for common queries
        if "show" in input_lower and ("employee" in input_lower or "employees" in input_lower):
            return "SELECT * FROM employees LIMIT 10"
        elif "all employees" in input_lower:
            return "SELECT * FROM employees"
        elif "average salary" in input_lower and "department" in input_lower:
            return "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department"
        elif "salary" in input_lower and "department" in input_lower:
            return "SELECT department, name, salary FROM employees ORDER BY department, salary DESC"
        elif "sales" in input_lower and "region" in input_lower:
            return "SELECT region, SUM(amount) as total_sales, COUNT(*) as sales_count FROM sales GROUP BY region"
        elif "show tables" in input_lower or "list tables" in input_lower:
            return "SELECT name FROM sqlite_master WHERE type='table'"
        elif "count employees" in input_lower or "how many employees" in input_lower:
            return "SELECT COUNT(*) as total_employees FROM employees"
        elif "top" in input_lower and ("sales" in input_lower or "products" in input_lower):
            return "SELECT product, SUM(amount) as total_sales FROM sales GROUP BY product ORDER BY total_sales DESC LIMIT 5"
        elif "recent sales" in input_lower or "latest sales" in input_lower:
            return "SELECT s.*, e.name as employee_name FROM sales s JOIN employees e ON s.employee_id = e.id ORDER BY s.sale_date DESC LIMIT 10"
        elif "highest salary" in input_lower or "maximum salary" in input_lower:
            return "SELECT name, salary, department FROM employees ORDER BY salary DESC LIMIT 5"
        elif "departments" in input_lower and ("list" in input_lower or "show" in input_lower):
            return "SELECT DISTINCT department FROM employees"
        elif "total sales" in input_lower:
            return "SELECT SUM(amount) as total_sales FROM sales"
        elif "employees by department" in input_lower:
            return "SELECT department, COUNT(*) as employee_count FROM employees GROUP BY department"
        
        return None

    def analyze_results(self, results: List[Dict], original_question: str) -> str:
        """Analyze query results and provide insights"""
        if not results:
            return "No data found for your query."
        
        result_count = len(results)
        
        if result_count == 1 and len(results[0]) == 1:
            # Single value result
            value = list(results[0].values())[0]
            key = list(results[0].keys())[0]
            return f"The {key} is: {value}"
        
        # Multiple results - provide summary
        analysis = f"Found {result_count} records. "
        
        # Add some context based on the data
        first_row = results[0]
        columns = list(first_row.keys())
        
        if 'avg_salary' in [col.lower() for col in columns]:
            analysis += "Here's the salary breakdown by department."
        elif 'total_sales' in [col.lower() for col in columns]:
            analysis += "Here's the sales performance data."
        elif 'salary' in [col.lower() for col in columns] and 'name' in [col.lower() for col in columns]:
            analysis += "Here are the employee salary details."
        elif 'employee_count' in [col.lower() for col in columns]:
            analysis += "Here's the employee distribution."
        else:
            analysis += f"The data includes: {', '.join(columns)}."
        
        return analysis

    def handle_database_query(self, user_question: str) -> Dict[str, Any]:
        """Main method to handle database-related queries"""
        
        # Try simple pattern matching first
        query = self.generate_simple_sql(user_question)
        
        if not query:
            return {
                "success": False,
                "message": "I couldn't understand your database query. Try asking about employees, sales, salaries, or departments.",
                "data": None,
                "query": ""
            }
        
        # Execute query
        success, results, message = self.db_connection.execute_safe_query(query)
        
        if not success:
            return {
                "success": False,
                "message": f"Query failed: {message}",
                "data": None,
                "query": query
            }
        
        # Analyze results
        analysis = self.analyze_results(results, user_question)
        
        return {
            "success": True,
            "message": analysis,
            "data": results,
            "query": query,
            "explanation": "Generated from pattern matching"
        }