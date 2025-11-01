#!/usr/bin/env python3
"""
Fix and load reviews and returns data by filtering invalid foreign keys
"""

import pymysql
import pandas as pd
from pathlib import Path

def load_with_fk_validation():
    """Load reviews and returns, filtering out invalid foreign keys"""
    
    config = {
        'host': 'localhost',
        'port': 3307,
        'user': 'root',
        'password': '',
        'database': 'ecommerce_analytics',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }
    
    print("="*60)
    print("FIX REVIEWS AND RETURNS DATA")
    print("="*60)
    
    try:
        conn = pymysql.connect(**config)
        cursor = conn.cursor()
        
        # Get valid foreign key values
        print("\n[1/4] Getting valid foreign key values...")
        
        cursor.execute("SELECT product_id FROM products")
        valid_products = {row['product_id'] for row in cursor.fetchall()}
        print(f"  Valid products: {len(valid_products)}")
        
        cursor.execute("SELECT customer_id FROM customers")
        valid_customers = {row['customer_id'] for row in cursor.fetchall()}
        print(f"  Valid customers: {len(valid_customers)}")
        
        cursor.execute("SELECT order_id FROM orders")
        valid_orders = {row['order_id'] for row in cursor.fetchall()}
        print(f"  Valid orders: {len(valid_orders)}")
        
        cursor.execute("SELECT order_item_id FROM order_items")
        valid_order_items = {row['order_item_id'] for row in cursor.fetchall()}
        print(f"  Valid order_items: {len(valid_order_items)}")
        
        # Load and filter REVIEWS
        print("\n[2/4] Loading reviews...")
        reviews_file = Path('sample_data/operational_data/reviews.csv')
        
        if reviews_file.exists():
            df_reviews = pd.read_csv(reviews_file)
            df_reviews.columns = df_reviews.columns.str.strip()
            
            original_count = len(df_reviews)
            print(f"  Original reviews: {original_count}")
            
            # Filter valid foreign keys
            if 'product_id' in df_reviews.columns:
                df_reviews = df_reviews[df_reviews['product_id'].isin(valid_products)]
            
            if 'customer_id' in df_reviews.columns:
                df_reviews = df_reviews[
                    df_reviews['customer_id'].isin(valid_customers) | 
                    df_reviews['customer_id'].isna()
                ]
            
            filtered_count = len(df_reviews)
            print(f"  After filtering: {filtered_count} ({original_count - filtered_count} removed)")
            
            # Get table columns
            cursor.execute("DESCRIBE reviews")
            table_cols = {row['Field'] for row in cursor.fetchall()}
            
            # Match columns
            matching_cols = [c for c in df_reviews.columns if c in table_cols]
            df_clean = df_reviews[matching_cols].where(pd.notnull(df_reviews), None)
            
            if len(df_clean) > 0:
                # Insert
                cursor.execute("TRUNCATE TABLE reviews")
                
                cols = ', '.join(matching_cols)
                placeholders = ', '.join(['%s'] * len(matching_cols))
                sql = f"INSERT INTO reviews ({cols}) VALUES ({placeholders})"
                
                data = [tuple(row) for row in df_clean.values]
                cursor.executemany(sql, data)
                conn.commit()
                
                print(f"  [OK] Loaded {len(data)} reviews")
            else:
                print(f"  [WARN] No valid reviews to load")
        else:
            print(f"  [SKIP] reviews.csv not found")
        
        # Load and filter RETURNS
        print("\n[3/4] Loading returns...")
        returns_file = Path('sample_data/operational_data/returns.csv')
        
        if returns_file.exists():
            df_returns = pd.read_csv(returns_file)
            df_returns.columns = df_returns.columns.str.strip()
            
            original_count = len(df_returns)
            print(f"  Original returns: {original_count}")
            
            # Filter valid foreign keys
            if 'order_id' in df_returns.columns:
                df_returns = df_returns[df_returns['order_id'].isin(valid_orders)]
            
            if 'order_item_id' in df_returns.columns:
                df_returns = df_returns[
                    df_returns['order_item_id'].isin(valid_order_items) | 
                    df_returns['order_item_id'].isna()
                ]
            
            filtered_count = len(df_returns)
            print(f"  After filtering: {filtered_count} ({original_count - filtered_count} removed)")
            
            # Get table columns
            cursor.execute("DESCRIBE returns")
            table_cols = {row['Field'] for row in cursor.fetchall()}
            
            # Match columns
            matching_cols = [c for c in df_returns.columns if c in table_cols]
            df_clean = df_returns[matching_cols].where(pd.notnull(df_returns), None)
            
            if len(df_clean) > 0:
                # Insert
                cursor.execute("TRUNCATE TABLE returns")
                
                cols = ', '.join(matching_cols)
                placeholders = ', '.join(['%s'] * len(matching_cols))
                sql = f"INSERT INTO returns ({cols}) VALUES ({placeholders})"
                
                data = [tuple(row) for row in df_clean.values]
                cursor.executemany(sql, data)
                conn.commit()
                
                print(f"  [OK] Loaded {len(data)} returns")
            else:
                print(f"  [WARN] No valid returns to load")
        else:
            print(f"  [SKIP] returns.csv not found")
        
        # Final verification
        print("\n[4/4] Final verification...")
        print("\n" + "="*60)
        print("COMPLETE DATABASE STATUS")
        print("="*60)
        
        # Get all table counts
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'ecommerce_analytics' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        tables = [row['table_name'] for row in cursor.fetchall()]
        
        total_rows = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            count = cursor.fetchone()['cnt']
            total_rows += count
            if count > 0:
                print(f"  {table}: {count} rows")
        
        print("\n" + "="*60)
        print(f"TOTAL: {len(tables)} tables, {total_rows:,} rows")
        print("="*60)
        
        cursor.close()
        conn.close()
        
        print("\n[SUCCESS] Database is ready!")
        print("\nYou can now run your Streamlit app:")
        print("  streamlit run Home.py")
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    load_with_fk_validation()
