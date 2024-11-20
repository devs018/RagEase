import sqlite3
import random
from datetime import datetime, timedelta
import os

def create_enhanced_database(db_path: str = "database/chatbot.db"):
    """Create enhanced database with 8 interconnected tables for comprehensive business scenario"""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables to recreate with new schema
    tables_to_drop = ['employees', 'sales', 'departments', 'products', 'customers', 'orders', 'order_items', 'suppliers']
    for table in tables_to_drop:
        cursor.execute(f'DROP TABLE IF EXISTS {table}')
    
    # 1. DEPARTMENTS table
    cursor.execute('''
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        manager_id INTEGER,
        budget REAL,
        location TEXT,
        created_date DATE,
        FOREIGN KEY (manager_id) REFERENCES employees (id)
    )
    ''')
    
    # 2. EMPLOYEES table
    cursor.execute('''
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        department_id INTEGER,
        position TEXT,
        salary REAL,
        hire_date DATE,
        age INTEGER,
        manager_id INTEGER,
        phone TEXT,
        address TEXT,
        FOREIGN KEY (department_id) REFERENCES departments (id),
        FOREIGN KEY (manager_id) REFERENCES employees (id)
    )
    ''')
    
    # 3. SUPPLIERS table
    cursor.execute('''
    CREATE TABLE suppliers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        contact_person TEXT,
        email TEXT,
        phone TEXT,
        address TEXT,
        city TEXT,
        country TEXT,
        rating REAL,
        contract_start_date DATE
    )
    ''')
    
    # 4. PRODUCTS table
    cursor.execute('''
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        supplier_id INTEGER,
        cost_price REAL,
        selling_price REAL,
        stock_quantity INTEGER,
        min_stock_level INTEGER,
        description TEXT,
        created_date DATE,
        FOREIGN KEY (supplier_id) REFERENCES suppliers (id)
    )
    ''')
    
    # 5. CUSTOMERS table
    cursor.execute('''
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        phone TEXT,
        address TEXT,
        city TEXT,
        country TEXT,
        customer_type TEXT,
        credit_limit REAL,
        registration_date DATE,
        last_purchase_date DATE
    )
    ''')
    
    # 6. ORDERS table
    cursor.execute('''
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        employee_id INTEGER,
        order_date DATE,
        ship_date DATE,
        total_amount REAL,
        status TEXT,
        payment_method TEXT,
        shipping_address TEXT,
        discount_percent REAL,
        FOREIGN KEY (customer_id) REFERENCES customers (id),
        FOREIGN KEY (employee_id) REFERENCES employees (id)
    )
    ''')
    
    # 7. ORDER_ITEMS table
    cursor.execute('''
    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        unit_price REAL,
        discount REAL,
        total_price REAL,
        FOREIGN KEY (order_id) REFERENCES orders (id),
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')
    
    # 8. SALES table (for individual sales tracking)
    cursor.execute('''
    CREATE TABLE sales (
        id INTEGER PRIMARY KEY,
        employee_id INTEGER,
        customer_id INTEGER,
        product_id INTEGER,
        sale_date DATE,
        quantity INTEGER,
        unit_price REAL,
        total_amount REAL,
        commission REAL,
        region TEXT,
        sale_type TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees (id),
        FOREIGN KEY (customer_id) REFERENCES customers (id),
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    ''')
    
    # Insert sample data
    
    # 1. Insert Departments
    departments_data = [
        ('Engineering', None, 500000, 'San Francisco', '2020-01-01'),
        ('Marketing', None, 300000, 'New York', '2020-01-01'),
        ('Sales', None, 400000, 'Chicago', '2020-01-01'),
        ('HR', None, 200000, 'Austin', '2020-01-01'),
        ('Finance', None, 350000, 'Boston', '2020-01-01'),
        ('Operations', None, 450000, 'Seattle', '2020-01-01'),
        ('Customer Service', None, 250000, 'Denver', '2020-01-01')
    ]
    
    cursor.executemany('''
    INSERT INTO departments (name, manager_id, budget, location, created_date) 
    VALUES (?, ?, ?, ?, ?)
    ''', departments_data)
    
    # 2. Insert Employees
    employees_data = [
        ('John Smith', 'john.smith@company.com', 1, 'Senior Developer', 85000, '2022-01-15', 32, None, '+1-555-0101', '123 Tech St, San Francisco'),
        ('Sarah Johnson', 'sarah.johnson@company.com', 2, 'Marketing Manager', 78000, '2021-06-20', 29, None, '+1-555-0102', '456 Market Ave, New York'),
        ('Mike Brown', 'mike.brown@company.com', 3, 'Sales Representative', 65000, '2023-02-10', 27, None, '+1-555-0103', '789 Sales Blvd, Chicago'),
        ('Emily Davis', 'emily.davis@company.com', 1, 'Lead Engineer', 95000, '2020-08-05', 35, 1, '+1-555-0104', '321 Code Lane, San Francisco'),
        ('David Wilson', 'david.wilson@company.com', 4, 'HR Specialist', 62000, '2022-11-30', 31, None, '+1-555-0105', '654 People St, Austin'),
        ('Lisa Anderson', 'lisa.anderson@company.com', 3, 'Sales Manager', 82000, '2021-09-12', 33, None, '+1-555-0106', '987 Revenue Rd, Chicago'),
        ('James Taylor', 'james.taylor@company.com', 2, 'Marketing Analyst', 58000, '2023-01-08', 26, 2, '+1-555-0107', '147 Brand Ave, New York'),
        ('Jennifer White', 'jennifer.white@company.com', 5, 'Financial Analyst', 71000, '2019-12-01', 34, None, '+1-555-0108', '258 Money St, Boston'),
        ('Robert Martinez', 'robert.martinez@company.com', 3, 'Sales Associate', 55000, '2022-07-18', 28, 6, '+1-555-0109', '369 Deal Ave, Chicago'),
        ('Maria Garcia', 'maria.garcia@company.com', 6, 'Operations Manager', 88000, '2021-04-25', 36, None, '+1-555-0110', '741 Process Blvd, Seattle'),
        ('Tom Anderson', 'tom.anderson@company.com', 7, 'Customer Service Rep', 45000, '2023-03-15', 24, None, '+1-555-0111', '852 Support St, Denver'),
        ('Anna Kim', 'anna.kim@company.com', 1, 'Software Developer', 76000, '2022-05-20', 30, 4, '+1-555-0112', '963 Code Ave, San Francisco')
    ]
    
    cursor.executemany('''
    INSERT INTO employees (name, email, department_id, position, salary, hire_date, age, manager_id, phone, address) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', employees_data)
    
    # Update department managers
    cursor.execute('UPDATE departments SET manager_id = 4 WHERE name = "Engineering"')
    cursor.execute('UPDATE departments SET manager_id = 2 WHERE name = "Marketing"')
    cursor.execute('UPDATE departments SET manager_id = 6 WHERE name = "Sales"')
    cursor.execute('UPDATE departments SET manager_id = 5 WHERE name = "HR"')
    cursor.execute('UPDATE departments SET manager_id = 8 WHERE name = "Finance"')
    cursor.execute('UPDATE departments SET manager_id = 10 WHERE name = "Operations"')
    
    # 3. Insert Suppliers
    suppliers_data = [
        ('TechCorp Industries', 'Alice Cooper', 'alice@techcorp.com', '+1-800-TECH-01', '100 Innovation Dr', 'Silicon Valley', 'USA', 4.5, '2019-01-01'),
        ('Global Electronics', 'Bob Smith', 'bob@globalelec.com', '+1-800-ELEC-02', '200 Circuit Ave', 'Austin', 'USA', 4.2, '2019-03-15'),
        ('Digital Solutions Ltd', 'Carol Jones', 'carol@digitalsol.com', '+44-20-DIGIT-1', '300 Tech Park', 'London', 'UK', 4.7, '2020-06-01'),
        ('Hardware Plus', 'Dave Brown', 'dave@hardwareplus.com', '+1-877-HARD-01', '400 Component St', 'Dallas', 'USA', 3.9, '2020-09-10'),
        ('Gadget World', 'Eva Martinez', 'eva@gadgetworld.com', '+49-30-GADGET', '500 Device Blvd', 'Berlin', 'Germany', 4.3, '2021-02-28')
    ]
    
    cursor.executemany('''
    INSERT INTO suppliers (name, contact_person, email, phone, address, city, country, rating, contract_start_date) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', suppliers_data)
    
    # 4. Insert Products
    products_data = [
        ('MacBook Pro 16"', 'Laptops', 1, 2200, 2800, 25, 5, 'High-performance laptop for professionals', '2023-01-15'),
        ('Dell XPS 13', 'Laptops', 2, 1100, 1400, 30, 8, 'Ultrabook for business users', '2023-01-20'),
        ('iPad Pro 12.9"', 'Tablets', 1, 900, 1200, 40, 10, 'Professional tablet with M2 chip', '2023-02-01'),
        ('Samsung Galaxy Tab S8', 'Tablets', 3, 650, 850, 35, 8, 'Android tablet for productivity', '2023-02-15'),
        ('iPhone 14 Pro', 'Smartphones', 1, 850, 1100, 50, 12, 'Latest iPhone with Pro camera', '2023-03-01'),
        ('Samsung Galaxy S23', 'Smartphones', 3, 720, 950, 45, 10, 'Android flagship smartphone', '2023-03-10'),
        ('AirPods Pro 2', 'Audio', 1, 180, 250, 100, 20, 'Noise-cancelling wireless earbuds', '2023-03-15'),
        ('Sony WH-1000XM5', 'Audio', 4, 320, 400, 60, 15, 'Premium noise-cancelling headphones', '2023-04-01'),
        ('Magic Mouse', 'Accessories', 1, 65, 95, 80, 25, 'Wireless mouse for Mac', '2023-04-10'),
        ('Logitech MX Master 3', 'Accessories', 5, 85, 120, 70, 20, 'Professional wireless mouse', '2023-04-15'),
        ('Dell UltraSharp 27"', 'Monitors', 2, 450, 650, 20, 5, '4K professional monitor', '2023-05-01'),
        ('LG UltraWide 34"', 'Monitors', 3, 550, 750, 15, 3, 'Curved ultrawide monitor', '2023-05-10')
    ]
    
    cursor.executemany('''
    INSERT INTO products (name, category, supplier_id, cost_price, selling_price, stock_quantity, min_stock_level, description, created_date) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', products_data)
    
    # 5. Insert Customers
    customers_data = [
        ('Acme Corporation', 'procurement@acme.com', '+1-555-ACME-01', '1000 Business Blvd', 'New York', 'USA', 'Enterprise', 50000, '2022-01-15', '2024-08-20'),
        ('StartupTech Inc', 'admin@startuptech.com', '+1-555-START-1', '2000 Innovation St', 'San Francisco', 'USA', 'SMB', 15000, '2022-03-20', '2024-09-01'),
        ('Global Consulting', 'orders@globalcons.com', '+1-555-GLOBAL-1', '3000 Consulting Ave', 'Chicago', 'USA', 'Enterprise', 75000, '2021-11-10', '2024-08-15'),
        ('Digital Agency Pro', 'tech@digitalagency.com', '+1-555-DIGITAL', '4000 Creative Blvd', 'Austin', 'USA', 'SMB', 25000, '2023-02-14', '2024-09-05'),
        ('Education Solutions', 'procurement@edusol.com', '+1-555-EDU-SOL', '5000 Learning Lane', 'Boston', 'USA', 'Non-Profit', 30000, '2022-09-30', '2024-07-22'),
        ('Healthcare Systems', 'it@healthsys.com', '+1-555-HEALTH-1', '6000 Medical Center Dr', 'Seattle', 'USA', 'Enterprise', 100000, '2021-06-15', '2024-08-30'),
        ('Retail Chain Plus', 'tech@retailchain.com', '+1-555-RETAIL-1', '7000 Commerce St', 'Denver', 'USA', 'Enterprise', 60000, '2022-12-01', '2024-09-10'),
        ('Freelancer John', 'john.freelancer@email.com', '+1-555-FREE-01', '8000 Home Office Ave', 'Portland', 'USA', 'Individual', 5000, '2023-04-18', '2024-08-25')
    ]
    
    cursor.executemany('''
    INSERT INTO customers (name, email, phone, address, city, country, customer_type, credit_limit, registration_date, last_purchase_date) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', customers_data)
    
    # 6. Insert Orders
    orders_data = []
    for i in range(50):  # Generate 50 orders
        customer_id = random.randint(1, 8)
        employee_id = random.choice([3, 6, 9])  # Sales employees
        order_date = datetime.now() - timedelta(days=random.randint(1, 365))
        ship_date = order_date + timedelta(days=random.randint(1, 7))
        total_amount = round(random.uniform(500, 15000), 2)
        status = random.choice(['Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled'])
        payment_method = random.choice(['Credit Card', 'Bank Transfer', 'PayPal', 'Check'])
        discount_percent = random.choice([0, 5, 10, 15, 20])
        
        orders_data.append((customer_id, employee_id, order_date.date(), ship_date.date(), 
                          total_amount, status, payment_method, f"Shipping Address {i+1}", discount_percent))
    
    cursor.executemany('''
    INSERT INTO orders (customer_id, employee_id, order_date, ship_date, total_amount, status, payment_method, shipping_address, discount_percent) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', orders_data)
    
    # 7. Insert Order Items
    order_items_data = []
    for order_id in range(1, 51):  # For each order
        num_items = random.randint(1, 5)
        for _ in range(num_items):
            product_id = random.randint(1, 12)
            quantity = random.randint(1, 10)
            # Get product price
            cursor.execute('SELECT selling_price FROM products WHERE id = ?', (product_id,))
            unit_price = cursor.fetchone()[0]
            discount = round(random.uniform(0, unit_price * 0.1), 2)
            total_price = round((unit_price - discount) * quantity, 2)
            
            order_items_data.append((order_id, product_id, quantity, unit_price, discount, total_price))
    
    cursor.executemany('''
    INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount, total_price) 
    VALUES (?, ?, ?, ?, ?, ?)
    ''', order_items_data)
    
    # 8. Insert Sales data
    sales_data = []
    for i in range(100):  # Generate 100 sales records
        employee_id = random.choice([3, 6, 9])  # Sales employees
        customer_id = random.randint(1, 8)
        product_id = random.randint(1, 12)
        sale_date = datetime.now() - timedelta(days=random.randint(1, 365))
        quantity = random.randint(1, 5)
        
        # Get product price
        cursor.execute('SELECT selling_price FROM products WHERE id = ?', (product_id,))
        unit_price = cursor.fetchone()[0]
        total_amount = round(unit_price * quantity, 2)
        commission = round(total_amount * 0.05, 2)  # 5% commission
        region = random.choice(['North', 'South', 'East', 'West', 'Central'])
        sale_type = random.choice(['Direct', 'Online', 'Partner', 'Referral'])
        
        sales_data.append((employee_id, customer_id, product_id, sale_date.date(), 
                          quantity, unit_price, total_amount, commission, region, sale_type))
    
    cursor.executemany('''
    INSERT INTO sales (employee_id, customer_id, product_id, sale_date, quantity, unit_price, total_amount, commission, region, sale_type) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', sales_data)
    
    conn.commit()
    conn.close()
    
    print(f"Enhanced database created at {db_path}")
    print("Tables created:")
    print("1. departments (7 departments)")
    print("2. employees (12 employees)")  
    print("3. suppliers (5 suppliers)")
    print("4. products (12 products)")
    print("5. customers (8 customers)")
    print("6. orders (50 orders)")
    print("7. order_items (multiple items per order)")
    print("8. sales (100 sales records)")
    print("\\nAll tables are interconnected with proper foreign key relationships!")

if __name__ == "__main__":
    create_enhanced_database()