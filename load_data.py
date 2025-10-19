"""
load_data.py - Direct CSV loader (bypasses foreign keys completely)
Run this to load all CSV data into your MySQL database
"""

import pymysql
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Database config
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'ecommerce_analytics'),
    'port': int(os.getenv('DB_PORT', 3307)),
    'charset': 'utf8mb4'
}

# CSV files to load (in order)
CSV_FILES = [
    ('vendors', 'sample_data/core_data/vendors.csv'),
    ('customers', 'sample_data/core_data/customers.csv'),
    ('products', 'sample_data/core_data/products.csv'),
    ('inventory', 'sample_data/core_data/inventory.csv'),
    ('orders', 'sample_data/core_data/orders.csv'),
    ('campaigns', 'sample_data/marketing_data/campaigns.csv'),
    ('reviews', 'sample_data/operational_data/reviews.csv'),
    ('returns', 'sample_data/operational_data/returns.csv'),
]

def load_csv_to_db(table_name, csv_path):
    """Load CSV into database table"""
    try:
        if not Path(csv_path).exists():
            print(f"❌ {table_name}: CSV not found at {csv_path}")
            return False
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"📊 {table_name}: Read {len(df)} rows from CSV")
        
        # Connect
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Disable foreign keys
        cursor.execute("SET FOREIGN_KEY_CHECKS=0")
        
        # Get table columns
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        table_cols = [row[0] for row in cursor.fetchall()]
        print(f"   Table has {len(table_cols)} columns: {', '.join(table_cols)}")
        
        # Map CSV columns to table columns
        df_mapped = pd.DataFrame()
        for col in table_cols:
            if col in df.columns:
                df_mapped[col] = df[col]
        
        if df_mapped.empty:
            print(f"❌ {table_name}: No matching columns!")
            cursor.close()
            conn.close()
            return False
        
        # Build insert query
        cols = ', '.join([f"`{col}`" for col in df_mapped.columns])
        placeholders = ', '.join(['%s'] * len(df_mapped.columns))
        insert_query = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"
        
        # Convert to tuples
        data = [tuple(row) for row in df_mapped.values]
        
        # Insert all rows
        print(f"   Inserting {len(data)} rows...")
        cursor.executemany(insert_query, data)
        
        # Re-enable foreign keys
        cursor.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()
        
        cursor.close()
        conn.close()
        
        print(f"✅ {table_name}: Successfully loaded {len(data)} rows\n")
        return True
        
    except Exception as e:
        print(f"❌ {table_name}: {str(e)}\n")
        return False

def verify_data():
    """Verify data was loaded"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        
        print("\n" + "="*50)
        print("DATABASE VERIFICATION")
        print("="*50)
        
        total_rows = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            total_rows += count
            status = "✅" if count > 0 else "⚠️"
            print(f"{status} {table}: {count:,} rows")
        
        print("="*50)
        print(f"Total: {total_rows:,} rows across {len(tables)} tables")
        print("="*50)
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    print("="*50)
    print("CSV DATA LOADER")
    print("="*50 + "\n")
    
    success_count = 0
    for table_name, csv_path in CSV_FILES:
        if load_csv_to_db(table_name, csv_path):
            success_count += 1
    
    print(f"\nLoaded {success_count}/{len(CSV_FILES)} tables")
    
    # Verify
    verify_data()
