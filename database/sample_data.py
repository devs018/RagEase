import sqlite3
import random
from datetime import datetime, timedelta
import os

def create_sample_database(db_path: str = "database/chatbot.db"):
    """Create sample database with multiple tables for demonstration"""

    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create employees table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        salary REAL,
        hire_date DATE,
        age INTEGER
    )
    """)

    # Create sales table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        product TEXT,
        amount REAL,
        sale_date DATE,
        region TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees (id)
    )
    """)

    # Insert sample employees data
    employees_data = [
        ('John Smith', 'Engineering', 75000, '2022-01-15', 28),
        ('Sarah Johnson', 'Marketing', 65000, '2021-06-20', 32),
        ('Mike Brown', 'Sales', 55000, '2023-02-10', 26),
        ('Emily Davis', 'Engineering', 80000, '2020-08-05', 35),
        ('David Wilson', 'HR', 60000, '2022-11-30', 29)
    ]

    cursor.executemany("""
    INSERT OR REPLACE INTO employees (name, department, salary, hire_date, age) 
    VALUES (?, ?, ?, ?, ?)
    """, employees_data)

    # Insert sample sales data
    sales_data = []
    products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones']
    regions = ['North', 'South', 'East', 'West', 'Central']

    for i in range(20):  # 20 sales records
        employee_id = random.randint(1, 5)
        product = random.choice(products)
        amount = round(random.uniform(100, 2000), 2)
        sale_date = (datetime.now() - timedelta(days=random.randint(1, 365))).date()
        region = random.choice(regions)

        sales_data.append((employee_id, product, amount, sale_date, region))

    cursor.executemany("""
    INSERT OR REPLACE INTO sales (employee_id, product, amount, sale_date, region) 
    VALUES (?, ?, ?, ?, ?)
    """, sales_data)

    conn.commit()
    conn.close()

    print(f"Sample database created at {db_path}")

if __name__ == "__main__":
    create_sample_database()