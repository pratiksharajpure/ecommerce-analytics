"""
Shared data utilities for all dashboard pages
Column mappings, data loaders, and common functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st

# ===========================
# COLUMN MAPPINGS
# ===========================

COLUMN_MAPPINGS = {
    'customers': {
        'id': ['customer_id', 'id', 'cust_id', 'CustomerId'],
        'name': ['name', 'customer_name', 'full_name', 'CustomerName'],
        'email': ['email', 'email_address', 'EmailAddress'],
        'phone': ['phone', 'phone_number', 'PhoneNumber', 'contact'],
        'address': ['address', 'street_address', 'Address'],
        'city': ['city', 'City'],
        'country': ['country', 'Country', 'location', 'region'],
        'created': ['created_at', 'created_date', 'registration_date', 'date_joined'],
        'lifetime_value': ['lifetime_value', 'ltv', 'total_spent', 'clv']
    },
    'orders': {
        'id': ['order_id', 'id', 'OrderId'],
        'customer_id': ['customer_id', 'cust_id', 'CustomerId', 'user_id'],
        'date': ['order_date', 'created_at', 'OrderDate', 'date'],
        'amount': ['total_amount', 'amount', 'order_total', 'TotalAmount'],
        'status': ['status', 'order_status', 'Status'],
        'payment_method': ['payment_method', 'payment_type', 'PaymentMethod']
    },
    'products': {
        'id': ['product_id', 'id', 'ProductId'],
        'name': ['name', 'product_name', 'title', 'ProductName'],
        'price': ['price', 'unit_price', 'selling_price', 'Price'],
        'cost': ['cost', 'unit_cost', 'Cost'],
        'stock': ['stock', 'stock_quantity', 'quantity', 'qty', 'Quantity'],
        'category': ['category', 'product_category', 'type', 'Category'],
        'description': ['description', 'product_description', 'Description']
    },
    'inventory': {
        'id': ['inventory_id', 'id'],
        'product_id': ['product_id', 'prod_id', 'ProductId'],
        'quantity': ['quantity', 'stock', 'qty', 'Quantity'],
        'warehouse': ['warehouse', 'location', 'Warehouse'],
        'reorder_point': ['reorder_point', 'reorder_level', 'ReorderPoint']
    },
    'vendors': {
        'id': ['vendor_id', 'id', 'supplier_id', 'VendorId'],
        'name': ['name', 'vendor_name', 'supplier_name', 'VendorName'],
        'country': ['country', 'Country', 'location'],
        'rating': ['rating', 'vendor_rating', 'Rating'],
        'active': ['active', 'is_active', 'status']
    },
    'reviews': {
        'id': ['review_id', 'id', 'ReviewId'],
        'customer_id': ['customer_id', 'cust_id', 'user_id'],
        'product_id': ['product_id', 'prod_id', 'ProductId'],
        'rating': ['rating', 'score', 'stars', 'Rating'],
        'date': ['review_date', 'created_at', 'date', 'ReviewDate']
    },
    'returns': {
        'id': ['return_id', 'id', 'ReturnId'],
        'order_id': ['order_id', 'OrderId'],
        'date': ['return_date', 'created_at', 'date', 'ReturnDate'],
        'reason': ['reason', 'return_reason', 'Reason'],
        'status': ['status', 'return_status', 'Status'],
        'amount': ['refund_amount', 'amount', 'Amount']
    },
    'campaigns': {
        'id': ['campaign_id', 'id', 'CampaignId'],
        'name': ['name', 'campaign_name', 'CampaignName'],
        'type': ['type', 'campaign_type', 'Type'],
        'budget': ['budget', 'Budget'],
        'spent': ['spent', 'amount_spent', 'Spent'],
        'conversions': ['conversions', 'Conversions']
    },
    'transactions': {
        'id': ['transaction_id', 'id', 'TransactionId'],
        'order_id': ['order_id', 'OrderId'],
        'date': ['transaction_date', 'created_at', 'date', 'TransactionDate'],
        'amount': ['amount', 'transaction_amount', 'Amount'],
        'status': ['status', 'transaction_status', 'Status'],
        'payment_method': ['payment_method', 'PaymentMethod']
    }
}

# ===========================
# HELPER FUNCTIONS
# ===========================

def get_column(df, table_name, field_name):
    """
    Smart column finder - returns the actual column name from a dataframe
    
    Args:
        df: DataFrame to search
        table_name: Table name (e.g., 'orders', 'products')
        field_name: Field type (e.g., 'date', 'amount')
    
    Returns:
        str: Actual column name in the dataframe, or None if not found
    """
    if df is None or df.empty:
        return None
    
    if table_name not in COLUMN_MAPPINGS or field_name not in COLUMN_MAPPINGS[table_name]:
        return None
    
    possible_names = COLUMN_MAPPINGS[table_name][field_name]
    
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    
    return None

def get_columns(df, table_name, field_names):
    """
    Get multiple columns at once
    
    Returns:
        dict: {field_name: actual_column_name}
    """
    result = {}
    for field in field_names:
        col = get_column(df, table_name, field)
        if col:
            result[field] = col
    return result

def standardize_dataframe(df, table_name):
    """
    Rename columns to standard names
    
    Args:
        df: DataFrame to standardize
        table_name: Table name for mapping lookup
    
    Returns:
        DataFrame with standardized column names
    """
    if df is None or df.empty or table_name not in COLUMN_MAPPINGS:
        return df
    
    df = df.copy()
    rename_map = {}
    
    for standard_name, possible_names in COLUMN_MAPPINGS[table_name].items():
        for col in df.columns:
            if col in possible_names:
                rename_map[col] = standard_name
                break
    
    return df.rename(columns=rename_map)

# ===========================
# DATA VALIDATION
# ===========================

def validate_email(email):
    """Check if email is valid"""
    if pd.isna(email):
        return False
    email_str = str(email)
    return '@' in email_str and '.' in email_str.split('@')[-1]

def validate_phone(phone):
    """Check if phone number looks valid"""
    if pd.isna(phone):
        return False
    phone_str = str(phone).replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
    return len(phone_str) >= 10 and phone_str.replace('+', '').isdigit()

def detect_issues(df, table_name):
    """
    Detect data quality issues in a dataframe
    
    Returns:
        DataFrame with issues column added
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    df['issues'] = ''
    df['severity'] = 'low'
    
    if table_name == 'customers':
        email_col = get_column(df, 'customers', 'email')
        phone_col = get_column(df, 'customers', 'phone')
        address_col = get_column(df, 'customers', 'address')
        
        for idx, row in df.iterrows():
            issues = []
            severity = 'low'
            
            if email_col:
                if not validate_email(row[email_col]):
                    issues.append('Invalid Email')
                    severity = 'high'
            
            if phone_col:
                if pd.isna(row[phone_col]):
                    issues.append('Missing Phone')
                    if severity == 'low':
                        severity = 'medium'
                elif not validate_phone(row[phone_col]):
                    issues.append('Invalid Phone')
                    severity = 'medium'
            
            if address_col and pd.isna(row[address_col]):
                issues.append('Missing Address')
                if severity == 'low':
                    severity = 'medium'
            
            if issues:
                df.at[idx, 'issues'] = ', '.join(issues)
                df.at[idx, 'severity'] = severity
    
    return df

# ===========================
# DATA LOADERS
# ===========================

@st.cache_data(ttl=300)
def load_data_smart():
    """
    Smart data loader - tries multiple sources automatically
    """
    data = {}
    source = "Sample Data (Generated)"
    sql_errors = []
    
    # TRY 1: Load from SQL Database
    try:
        from utils.database import safe_table_query, table_exists, execute_sql_file, test_connection
        
        if test_connection():
            tables = ['customers', 'products', 'orders', 'inventory', 'vendors', 
                     'campaigns', 'reviews', 'returns', 'transactions']
            
            for table in tables:
                try:
                    df = safe_table_query(table, limit=50000)
                    if df is not None and not df.empty:
                        data[table] = df
                    elif df is None:
                        sql_errors.append(f"{table}: Table doesn't exist")
                except Exception as e:
                    sql_errors.append(f"{table}: {str(e)[:80]}")
                    continue
            
            # Load SQL reports
            sql_reports = {
                'dashboard_metrics': 'sql/reporting/dashboard_metrics.sql',
                'executive_summary': 'sql/reporting/executive_summary.sql',
                'table_health': 'sql/reporting/table_health_scores.sql',
            }
            
            for key, sql_file in sql_reports.items():
                if Path(sql_file).exists():
                    try:
                        df = execute_sql_file(sql_file)
                        if df is not None and not df.empty:
                            data[key] = df
                    except Exception as e:
                        sql_errors.append(f"{key}: {str(e)[:80]}")
            
            if len(data) > 0:
                source = f"MySQL Database ({len(data)} tables)"
                return data, source, sql_errors
    except ImportError:
        pass
    except Exception as e:
        sql_errors.append(f"MySQL: {str(e)[:80]}")
    
    # TRY 2: Load from CSV Files
    csv_files = {
        'customers': 'sample_data/core_data/customers.csv',
        'products': 'sample_data/core_data/products.csv',
        'orders': 'sample_data/core_data/orders.csv',
        'inventory': 'sample_data/core_data/inventory.csv',
        'vendors': 'sample_data/core_data/vendors.csv',
        'campaigns': 'sample_data/marketing_data/campaigns.csv',
        'reviews': 'sample_data/operational_data/reviews.csv',
        'returns': 'sample_data/operational_data/returns.csv',
        'transactions': 'sample_data/financial_data/transactions.csv',
    }
    
    csv_loaded = 0
    for table, csv_path in csv_files.items():
        if Path(csv_path).exists():
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    data[table] = df
                    csv_loaded += 1
            except:
                continue
    
    if len(data) > 0:
        source = f"CSV Files ({csv_loaded} files)"
        return data, source, []
    
    # TRY 3: Generate Sample Data
    data = generate_sample_data()
    return data, source, []

@st.cache_data(ttl=300)
def generate_sample_data():
    """Generate comprehensive sample data"""
    np.random.seed(42)
    data = {}
    
    # Customers
    num_customers = 10000
    data['customers'] = pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'name': [f'{np.random.choice(["John", "Jane", "Bob", "Alice", "Charlie"])} {np.random.choice(["Smith", "Johnson", "Brown", "Wilson", "Davis"])}' for _ in range(num_customers)],
        'email': [f'customer{i}@example.com' if i % 100 != 0 else None for i in range(num_customers)],
        'phone': [f'+1-555-{np.random.randint(1000, 9999)}' if i % 50 != 0 else None for i in range(num_customers)],
        'address': [f'{np.random.randint(1, 999)} Main St' if i % 30 != 0 else None for i in range(num_customers)],
        'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany'], num_customers),
        'created_at': pd.date_range(start='2023-01-01', periods=num_customers, freq='1H'),
        'lifetime_value': np.random.uniform(100, 5000, num_customers)
    })
    
    # Products
    num_products = 500
    data['products'] = pd.DataFrame({
        'product_id': range(1, num_products + 1),
        'name': [f'Product {i}' for i in range(1, num_products + 1)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], num_products),
        'price': np.random.uniform(10, 500, num_products),
        'cost': np.random.uniform(5, 250, num_products),
        'stock': np.random.randint(0, 200, num_products),
    })
    
    # Orders
    num_orders = 25000
    data['orders'] = pd.DataFrame({
        'order_id': range(1, num_orders + 1),
        'customer_id': np.random.randint(1, num_customers + 1, num_orders),
        'order_date': pd.date_range(end=datetime.now(), periods=num_orders, freq='15min'),
        'total_amount': np.random.uniform(20, 800, num_orders),
        'status': np.random.choice(['completed', 'pending', 'cancelled', 'shipped'], num_orders, p=[0.7, 0.15, 0.05, 0.1]),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'], num_orders)
    })
    
    # Inventory
    data['inventory'] = pd.DataFrame({
        'inventory_id': range(1, num_products + 1),
        'product_id': range(1, num_products + 1),
        'warehouse': np.random.choice(['Warehouse A', 'Warehouse B', 'Warehouse C'], num_products),
        'quantity': np.random.randint(0, 150, num_products),
        'reorder_point': np.random.randint(10, 30, num_products)
    })
    
    # Vendors
    num_vendors = 50
    data['vendors'] = pd.DataFrame({
        'vendor_id': range(1, num_vendors + 1),
        'name': [f'Vendor {i}' for i in range(1, num_vendors + 1)],
        'country': np.random.choice(['China', 'USA', 'Germany', 'Japan', 'India'], num_vendors),
        'rating': np.random.uniform(3.5, 5.0, num_vendors),
        'active': np.random.choice([True, False], num_vendors, p=[0.9, 0.1])
    })
    
    # Reviews
    num_reviews = 5000
    data['reviews'] = pd.DataFrame({
        'review_id': range(1, num_reviews + 1),
        'product_id': np.random.randint(1, num_products + 1, num_reviews),
        'customer_id': np.random.randint(1, num_customers + 1, num_reviews),
        'rating': np.random.randint(1, 6, num_reviews),
        'review_date': pd.date_range(end=datetime.now(), periods=num_reviews, freq='10min')
    })
    
    # Returns
    num_returns = 500
    data['returns'] = pd.DataFrame({
        'return_id': range(1, num_returns + 1),
        'order_id': np.random.randint(1, num_orders + 1, num_returns),
        'return_date': pd.date_range(end=datetime.now(), periods=num_returns, freq='2H'),
        'reason': np.random.choice(['Defective', 'Wrong Item', 'Not as Described', 'Changed Mind'], num_returns),
        'refund_amount': np.random.uniform(10, 300, num_returns),
        'status': np.random.choice(['Pending', 'Approved', 'Rejected'], num_returns)
    })
    
    # Campaigns
    num_campaigns = 30
    data['campaigns'] = pd.DataFrame({
        'campaign_id': range(1, num_campaigns + 1),
        'name': [f'Campaign {i}' for i in range(1, num_campaigns + 1)],
        'type': np.random.choice(['Email', 'Social', 'PPC', 'Display'], num_campaigns),
        'budget': np.random.uniform(1000, 20000, num_campaigns),
        'spent': np.random.uniform(500, 15000, num_campaigns),
        'conversions': np.random.randint(50, 1000, num_campaigns)
    })
    
    # Transactions
    num_transactions = int(num_orders * 1.05)
    data['transactions'] = pd.DataFrame({
        'transaction_id': range(1, num_transactions + 1),
        'order_id': np.random.randint(1, num_orders + 1, num_transactions),
        'transaction_date': pd.date_range(end=datetime.now(), periods=num_transactions, freq='10min'),
        'amount': np.random.uniform(10, 800, num_transactions),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal'], num_transactions),
        'status': np.random.choice(['success', 'failed', 'pending'], num_transactions, p=[0.85, 0.08, 0.07])
    })
    
    return data

# ===========================
# METRIC CALCULATORS
# ===========================

def calculate_dashboard_metrics(data, date_range_days=90):
    """Calculate key metrics from loaded data"""
    metrics = {}
    
    # Check if we have SQL dashboard_metrics
    if 'dashboard_metrics' in data and not data['dashboard_metrics'].empty:
        sql_df = data['dashboard_metrics']
        metrics['revenue'] = sql_df.get('total_revenue', [0])[0] if 'total_revenue' in sql_df.columns else 0
        metrics['orders'] = sql_df.get('total_orders', [0])[0] if 'total_orders' in sql_df.columns else 0
        if metrics['revenue'] > 0:
            return metrics
    
    # Calculate from orders data
    if 'orders' in data and not data['orders'].empty:
        orders = data['orders'].copy()
        date_col = get_column(orders, 'orders', 'date')
        amount_col = get_column(orders, 'orders', 'amount')
        
        if date_col and amount_col:
            orders[date_col] = pd.to_datetime(orders[date_col])
            end_date = datetime.now()
            start_date = end_date - timedelta(days=date_range_days)
            filtered = orders[orders[date_col] >= start_date]
            
            prev_start = start_date - timedelta(days=date_range_days)
            prev_filtered = orders[(orders[date_col] >= prev_start) & (orders[date_col] < start_date)]
            
            metrics['revenue'] = filtered[amount_col].sum()
            prev_revenue = prev_filtered[amount_col].sum()
            metrics['revenue_delta'] = ((metrics['revenue'] - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
            
            metrics['orders'] = len(filtered)
            prev_orders = len(prev_filtered)
            metrics['orders_delta'] = ((metrics['orders'] - prev_orders) / prev_orders * 100) if prev_orders > 0 else 0
            
            metrics['aov'] = filtered[amount_col].mean()
            metrics['conversion_rate'] = np.random.uniform(2.5, 4.5)
    
    # Customers
    if 'customers' in data:
        metrics['total_customers'] = len(data['customers'])
        metrics['new_customers'] = len(data['customers']) // 10
    
    # Products
    if 'products' in data:
        metrics['total_products'] = len(data['products'])
        stock_col = get_column(data['products'], 'products', 'stock')
        if stock_col:
            metrics['low_stock_items'] = (data['products'][stock_col] < 10).sum()
    
    # Returns
    if 'returns' in data:
        metrics['return_rate'] = (len(data['returns']) / max(metrics.get('orders', 1), 1)) * 100
    
    # Reviews
    if 'reviews' in data and not data['reviews'].empty:
        rating_col = get_column(data['reviews'], 'reviews', 'rating')
        if rating_col:
            metrics['avg_rating'] = data['reviews'][rating_col].mean()
    
    # Defaults
    defaults = {
        'revenue': 0, 'revenue_delta': 0, 'orders': 0, 'orders_delta': 0,
        'aov': 0, 'conversion_rate': 3.2, 'total_customers': 0, 'new_customers': 0,
        'total_products': 0, 'low_stock_items': 0, 'return_rate': 0, 'avg_rating': 4.2
    }
    for key, value in defaults.items():
        if key not in metrics:
            metrics[key] = value
    
    return metrics
