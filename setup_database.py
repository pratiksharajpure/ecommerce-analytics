"""
setup_database.py - Fixed with direct pymysql connection
Handles foreign key constraints properly
"""

import streamlit as st
import pymysql
import sqlalchemy as sa
import pandas as pd
from pathlib import Path
import time

st.set_page_config(
    page_title="Database Setup",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Database Setup Wizard")
st.markdown("Run your SQL files to set up the e-commerce analytics database")

def get_db_config():
    """Get database config from secrets"""
    try:
        return st.secrets["database"]
    except:
        return None

def create_pymysql_connection(use_database=True):
    """Create direct pymysql connection"""
    try:
        db = get_db_config()
        if not db:
            st.error("❌ Database config not found in secrets")
            return None
        
        config = {
            'host': db['host'],
            'user': db['username'],
            'password': db['password'],
            'port': db['port'],
            'charset': 'utf8mb4',
            'autocommit': True
        }
        
        if use_database:
            config['database'] = db['database']
        
        connection = pymysql.connect(**config)
        return connection
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        return None

def create_sqlalchemy_connection(use_database=True):
    """Create SQLAlchemy connection"""
    try:
        db = get_db_config()
        if use_database:
            connection_string = f"mysql+pymysql://{db['username']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8mb4"
        else:
            connection_string = f"mysql+pymysql://{db['username']}:{db['password']}@{db['host']}:{db['port']}?charset=utf8mb4"
        
        engine = sa.create_engine(connection_string, pool_pre_ping=True)
        return engine
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        return None

def execute_sql_file(file_path, use_database=True):
    """Execute a SQL file"""
    try:
        if not Path(file_path).exists():
            st.error(f"❌ File not found: {file_path}")
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        engine = create_sqlalchemy_connection(use_database=use_database)
        if not engine:
            return False
        
        statements = [s.strip() for s in sql_content.split(';') if s.strip()]
        
        with engine.connect() as conn:
            for statement in statements:
                if statement:
                    try:
                        conn.execute(sa.text(statement))
                    except Exception as e:
                        st.warning(f"⚠️ {str(e)[:100]}")
            conn.commit()
        
        return True
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False

def get_table_columns(table_name):
    """Get actual column names from MySQL table"""
    try:
        conn = create_pymysql_connection(use_database=True)
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return columns
    except Exception as e:
        st.warning(f"⚠️ Could not get columns for {table_name}: {e}")
        return []

def load_csv_to_table(csv_path, table_name):
    """Load CSV data into MySQL table with foreign key bypass"""
    try:
        if not Path(csv_path).exists():
            st.warning(f"⚠️ CSV not found: {csv_path}")
            return False
        
        df = pd.read_csv(csv_path)
        
        conn = create_pymysql_connection(use_database=True)
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Get actual table columns
        table_columns = get_table_columns(table_name)
        if not table_columns:
            st.warning(f"⚠️ Could not determine columns for {table_name}")
            cursor.close()
            conn.close()
            return False
        
        # Map CSV columns to table columns
        df_mapped = pd.DataFrame()
        for col in table_columns:
            if col in df.columns:
                df_mapped[col] = df[col]
        
        if df_mapped.empty:
            st.warning(f"⚠️ No matching columns found for {table_name}")
            cursor.close()
            conn.close()
            return False
        
        st.info(f"📊 Loading {len(df_mapped)} rows into {table_name}")
        
        try:
            # Disable foreign key checks
            cursor.execute("SET FOREIGN_KEY_CHECKS=0")
            
            # Build and execute insert statements
            cols = ', '.join([f"`{col}`" for col in df_mapped.columns])
            placeholders = ', '.join(['%s'] * len(df_mapped.columns))
            insert_query = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"
            
            # Convert dataframe to list of tuples
            data = [tuple(row) for row in df_mapped.values]
            
            # Execute batch insert
            cursor.executemany(insert_query, data)
            
            # Re-enable foreign key checks
            cursor.execute("SET FOREIGN_KEY_CHECKS=1")
            conn.commit()
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            cursor.execute("SET FOREIGN_KEY_CHECKS=1")
            conn.rollback()
            cursor.close()
            conn.close()
            error_msg = str(e)
            if "foreign key" in error_msg.lower():
                st.error(f"❌ {table_name}: Foreign key violation - {error_msg[:150]}")
            else:
                st.error(f"❌ {table_name}: {error_msg[:200]}")
            return False
            
    except Exception as e:
        st.error(f"❌ Error loading {csv_path}: {str(e)[:200]}")
        return False

st.markdown("---")

# Step 1: Test Connection
st.header("Step 1: Test Connection")
col1, col2 = st.columns([3, 1])
with col1:
    st.info("Test if MySQL connection is working")
with col2:
    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Testing..."):
            try:
                engine = create_sqlalchemy_connection(use_database=False)
                if engine:
                    with engine.connect() as conn:
                        result = conn.execute(sa.text("SELECT VERSION()"))
                        version = result.fetchone()[0]
                    st.success(f"✅ Connected! MySQL Version: {version}")
                else:
                    st.error("❌ Connection failed")
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.markdown("---")

# Step 2: Create Database
st.header("Step 2: Create Database")
col1, col2 = st.columns([3, 1])
with col1:
    st.info("Run: `sql/setup/create_database.sql`")
with col2:
    if st.button("🗄️ Create DB", use_container_width=True):
        with st.spinner("Creating database..."):
            success = execute_sql_file('sql/setup/create_database.sql', use_database=False)
            if success:
                st.success("✅ Database created!")
            else:
                st.error("❌ Failed to create database")

st.markdown("---")

# Step 3: Create Tables
st.header("Step 3: Create Tables")
col1, col2 = st.columns([3, 1])
with col1:
    st.info("Run: `sql/setup/create_tables.sql`")
with col2:
    if st.button("📋 Create Tables", use_container_width=True):
        with st.spinner("Creating tables..."):
            success = execute_sql_file('sql/setup/create_tables.sql', use_database=True)
            if success:
                st.success("✅ Tables created!")
            else:
                st.error("❌ Failed to create tables")

st.markdown("---")

# Step 4: Create Views
st.header("Step 4: Create Views")
col1, col2 = st.columns([3, 1])
with col1:
    st.info("Run: `sql/setup/create_views.sql`")
with col2:
    if st.button("👁️ Create Views", use_container_width=True):
        with st.spinner("Creating views..."):
            success = execute_sql_file('sql/setup/create_views.sql', use_database=True)
            if success:
                st.success("✅ Views created!")
            else:
                st.error("❌ Failed to create views")

st.markdown("---")

# Step 5: Load Sample Data
st.header("Step 5: Load Sample Data (CSV → MySQL)")

csv_load_order = [
    ('vendors', 'sample_data/core_data/vendors.csv'),
    ('customers', 'sample_data/core_data/customers.csv'),
    ('products', 'sample_data/core_data/products.csv'),
    ('inventory', 'sample_data/core_data/inventory.csv'),
    ('orders', 'sample_data/core_data/orders.csv'),
    ('campaigns', 'sample_data/marketing_data/campaigns.csv'),
    ('reviews', 'sample_data/operational_data/reviews.csv'),
    ('returns', 'sample_data/operational_data/returns.csv'),
    ('transactions', 'sample_data/financial_data/transactions.csv')
]

col1, col2 = st.columns([3, 1])
with col1:
    st.info(f"Load {len(csv_load_order)} CSV files (foreign keys bypassed during load)")
with col2:
    if st.button("📥 Load Data", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        total = len(csv_load_order)
        
        for idx, (table_name, csv_path) in enumerate(csv_load_order):
            status_text.text(f"Loading {table_name}... ({idx+1}/{total})")
            
            if load_csv_to_table(csv_path, table_name):
                st.success(f"✅ Loaded {table_name}")
                success_count += 1
            else:
                st.warning(f"⚠️ Skipped {table_name}")
            
            progress_bar.progress((idx + 1) / total)
            time.sleep(0.3)
        
        status_text.empty()
        progress_bar.empty()
        
        if success_count == total:
            st.success(f"🎉 All {total} tables loaded successfully!")
        elif success_count > 0:
            st.warning(f"⚠️ Loaded {success_count}/{total} tables")
        else:
            st.error("❌ No tables loaded")

st.markdown("---")

# Step 6: Verify Setup
st.header("Step 6: Verify Setup")
col1, col2 = st.columns([3, 1])
with col1:
    st.info("Check if all tables exist and have data")
with col2:
    if st.button("✅ Verify", use_container_width=True):
        with st.spinner("Verifying..."):
            try:
                engine = create_sqlalchemy_connection(use_database=True)
                
                with engine.connect() as conn:
                    tables_result = conn.execute(sa.text("SHOW TABLES"))
                    tables = [row[0] for row in tables_result]
                
                if tables:
                    st.success(f"✅ Found {len(tables)} tables")
                    
                    table_info = []
                    for table in tables:
                        with engine.connect() as conn:
                            count_result = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}"))
                            count = count_result.fetchone()[0]
                        table_info.append({'Table': table, 'Row Count': count})
                    
                    df = pd.DataFrame(table_info)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    total_rows = df['Row Count'].sum()
                    st.metric("Total Records", f"{total_rows:,}")
                else:
                    st.warning("⚠️ No tables found")
            except Exception as e:
                st.error(f"❌ Verification failed: {e}")

st.markdown("---")
st.caption("After setup is complete, you can run `Home.py` to start the dashboard!")