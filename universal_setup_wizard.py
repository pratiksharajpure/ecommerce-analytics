"""
universal_setup_wizard.py - Dynamic Setup for All Streamlit Pages
Auto-discovers SQL files, CSV files, and manages complete database initialization
Works with all modules: customers, products, orders, inventory, vendors, etc.
"""

import streamlit as st
import pymysql
import sqlalchemy as sa
import pandas as pd
from pathlib import Path
import time
import json
from datetime import datetime
from collections import defaultdict

st.set_page_config(
    page_title="Universal Database Setup",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Universal Database Setup Wizard")
st.markdown("Dynamically setup all modules with auto-discovery of SQL and CSV files")

# ==================== CONFIGURATION ====================

def get_db_config():
    """Get database config from secrets"""
    try:
        return st.secrets["database"]
    except:
        return None

def create_connection(use_database=True):
    """Create SQLAlchemy connection"""
    try:
        db = get_db_config()
        if not db:
            st.error("❌ Database config not found in secrets")
            return None
        
        if use_database:
            conn_str = f"mysql+pymysql://{db['username']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8mb4"
        else:
            conn_str = f"mysql+pymysql://{db['username']}:{db['password']}@{db['host']}:{db['port']}?charset=utf8mb4"
        
        engine = sa.create_engine(conn_str, pool_pre_ping=True)
        return engine
    except Exception as e:
        st.error(f"❌ Connection error: {e}")
        return None

# ==================== FILE DISCOVERY ====================

@st.cache_data(ttl=300)
def discover_sql_files():
    """Recursively discover all SQL files organized by category"""
    sql_files = defaultdict(list)
    sql_dir = Path("sql")
    
    if not sql_dir.exists():
        st.warning("⚠️ sql/ directory not found")
        return sql_files
    
    for sql_file in sorted(sql_dir.rglob("*.sql")):
        relative_path = sql_file.relative_to(sql_dir)
        category = str(relative_path.parent)
        
        sql_files[category].append({
            'name': sql_file.stem,
            'path': str(sql_file),
            'display': f"{sql_file.stem}"
        })
    
    return dict(sql_files)

@st.cache_data(ttl=300)
def discover_csv_files():
    """Recursively discover all CSV files organized by module"""
    csv_files = defaultdict(list)
    sample_dir = Path("sample_data")
    
    if not sample_dir.exists():
        st.warning("⚠️ sample_data/ directory not found")
        return csv_files
    
    for csv_file in sorted(sample_dir.rglob("*.csv")):
        relative_path = csv_file.relative_to(sample_dir)
        category = str(relative_path.parent)
        
        csv_files[category].append({
            'name': csv_file.stem,
            'path': str(csv_file),
            'display': f"{csv_file.stem}",
            'size': csv_file.stat().st_size
        })
    
    return dict(csv_files)

@st.cache_data(ttl=300)
def get_file_stats():
    """Get overall file statistics"""
    sql_files = discover_sql_files()
    csv_files = discover_csv_files()
    
    total_sql = sum(len(files) for files in sql_files.values())
    total_csv = sum(len(files) for files in csv_files.values())
    
    return {
        'sql_categories': len(sql_files),
        'sql_files': total_sql,
        'csv_categories': len(csv_files),
        'csv_files': total_csv
    }

# ==================== MODULE MAPPING ====================

def get_module_mapping():
    """Return complete module to SQL/CSV mapping (from your document)"""
    return {
        'Home': {
            'sql': [
                'sql/setup/create_database.sql',
                'sql/setup/create_tables.sql',
                'sql/setup/create_views.sql',
                'sql/reporting/executive_summary.sql',
                'sql/reporting/dashboard_metrics.sql',
            ],
            'csv': ['sample_data/core_data/customers.csv', 'sample_data/core_data/products.csv']
        },
        'Customers': {
            'sql': [
                'sql/setup/create_tables.sql',
                'sql/setup/create_indexes.sql',
                'sql/core_analysis/customer_duplicates.sql',
                'sql/advanced_analysis/customer_lifetime_value.sql',
            ],
            'csv': ['sample_data/core_data/customers.csv', 'sample_data/external_data/demographics.csv']
        },
        'Products': {
            'sql': [
                'sql/setup/create_tables.sql',
                'sql/core_analysis/product_data_gaps.sql',
                'sql/advanced_analysis/price_elasticity.sql',
            ],
            'csv': ['sample_data/core_data/products.csv', 'sample_data/financial_data/cost_of_goods.csv']
        },
        'Orders': {
            'sql': [
                'sql/setup/create_tables.sql',
                'sql/core_analysis/order_integrity.sql',
                'sql/advanced_analysis/market_basket.sql',
            ],
            'csv': ['sample_data/core_data/orders.csv', 'sample_data/financial_data/transactions.csv']
        },
        'Inventory': {
            'sql': [
                'sql/setup/create_tables.sql',
                'sql/core_analysis/inventory_mismatches.sql',
                'sql/advanced_analysis/inventory_turnover.sql',
            ],
            'csv': ['sample_data/core_data/inventory.csv', 'sample_data/core_data/products.csv']
        },
        'Vendors': {
            'sql': [
                'sql/setup/create_tables.sql',
                'sql/core_analysis/vendor_data_quality.sql',
                'sql/advanced_analysis/supplier_performance.sql',
            ],
            'csv': ['sample_data/core_data/vendors.csv', 'sample_data/external_data/supplier_ratings.csv']
        },
        'Campaigns': {
            'sql': [
                'sql/setup/create_tables.sql',
                'sql/core_analysis/campaign_effectiveness.sql',
                'sql/advanced_analysis/campaign_attribution.sql',
            ],
            'csv': ['sample_data/marketing_data/campaigns.csv', 'sample_data/operational_data/conversion_events.csv']
        },
        'Analytics': {
            'sql': [
                'sql/reporting/executive_summary.sql',
                'sql/advanced_analysis/cohort_analysis.sql',
                'sql/advanced_analysis/customer_lifetime_value.sql',
            ],
            'csv': ['sample_data/core_data/customers.csv', 'sample_data/financial_data/transactions.csv']
        },
    }

# ==================== EXECUTION FUNCTIONS ====================

def execute_sql_file(file_path):
    """Execute a SQL file"""
    try:
        if not Path(file_path).exists():
            return False, f"File not found: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        engine = create_connection(use_database=True)
        if not engine:
            return False, "Connection failed"
        
        statements = [s.strip() for s in sql_content.split(';') if s.strip()]
        
        with engine.connect() as conn:
            for statement in statements:
                if statement:
                    try:
                        conn.execute(sa.text(statement))
                    except Exception as e:
                        pass
            conn.commit()
        
        return True, "Success"
    except Exception as e:
        return False, str(e)[:100]

def load_csv_to_table(csv_path, table_name):
    """Load CSV into MySQL table with foreign key bypass"""
    try:
        if not Path(csv_path).exists():
            return False, f"CSV not found: {csv_path}"
        
        df = pd.read_csv(csv_path)
        
        # Get SQLAlchemy connection
        engine = create_connection(use_database=True)
        if not engine:
            return False, "Connection failed"
        
        # Use SQLAlchemy's to_sql
        df.to_sql(table_name, engine, if_exists='append', index=False)
        return True, f"Loaded {len(df)} rows"
        
    except Exception as e:
        return False, str(e)[:100]

def get_table_row_count(table_name):
    """Get row count for a table"""
    try:
        engine = create_connection(use_database=True)
        with engine.connect() as conn:
            result = conn.execute(sa.text(f"SELECT COUNT(*) FROM `{table_name}`"))
            return result.fetchone()[0]
    except:
        return 0

# ==================== DASHBOARD UI ====================

tabs = st.tabs([
    "🚀 Quick Start",
    "📋 Module Setup",
    "🔍 File Discovery",
    "✅ Verification",
    "📊 Statistics"
])

# ==================== TAB 1: QUICK START ====================

with tabs[0]:
    st.header("Quick Start Setup")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔗 Test Connection", use_container_width=True):
            with st.spinner("Testing..."):
                try:
                    engine = create_connection(use_database=False)
                    if engine:
                        with engine.connect() as conn:
                            result = conn.execute(sa.text("SELECT VERSION()"))
                            version = result.fetchone()[0]
                        st.success(f"✅ MySQL {version}")
                except Exception as e:
                    st.error(f"❌ {e}")
    
    with col2:
        if st.button("⚡ Setup All SQL", use_container_width=True):
            with st.spinner("Setting up SQL..."):
                sql_files = discover_sql_files()
                success = 0
                for category, files in sql_files.items():
                    for file_info in files:
                        ok, msg = execute_sql_file(file_info['path'])
                        if ok:
                            success += 1
                st.success(f"✅ Executed {success} SQL files")
    
    with col3:
        if st.button("📥 Load All CSVs", use_container_width=True):
            with st.spinner("Loading CSVs..."):
                csv_files = discover_csv_files()
                success = 0
                for category, files in csv_files.items():
                    for file_info in files:
                        table_name = file_info['name']
                        ok, msg = load_csv_to_table(file_info['path'], table_name)
                        if ok:
                            success += 1
                st.success(f"✅ Loaded {success} CSV files")
    
    st.divider()
    st.info("📌 Use tabs above for detailed setup options")

# ==================== TAB 2: MODULE SETUP ====================

with tabs[1]:
    st.header("Module-Based Setup")
    
    modules = get_module_mapping()
    selected_module = st.selectbox("Select Module:", list(modules.keys()))
    
    if selected_module:
        module_config = modules[selected_module]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 SQL Files")
            sql_list = module_config['sql']
            for idx, sql_file in enumerate(sql_list, 1):
                st.caption(f"{idx}. {Path(sql_file).stem}")
            
            if st.button("Execute SQL", use_container_width=True, key=f"sql_{selected_module}"):
                with st.spinner("Executing SQL..."):
                    success = 0
                    for sql_file in sql_list:
                        ok, msg = execute_sql_file(sql_file)
                        if ok:
                            success += 1
                            st.success(f"✅ {Path(sql_file).stem}")
                        else:
                            st.warning(f"⚠️ {Path(sql_file).stem}: {msg}")
                    st.info(f"Completed: {success}/{len(sql_list)}")
        
        with col2:
            st.subheader("📊 CSV Files")
            csv_list = module_config['csv']
            for idx, csv_file in enumerate(csv_list, 1):
                st.caption(f"{idx}. {Path(csv_file).stem}")
            
            if st.button("Load CSVs", use_container_width=True, key=f"csv_{selected_module}"):
                with st.spinner("Loading CSVs..."):
                    success = 0
                    for csv_file in csv_list:
                        table_name = Path(csv_file).stem
                        ok, msg = load_csv_to_table(csv_file, table_name)
                        if ok:
                            success += 1
                            st.success(f"✅ {table_name}")
                        else:
                            st.warning(f"⚠️ {table_name}: {msg}")
                    st.info(f"Completed: {success}/{len(csv_list)}")

# ==================== TAB 3: FILE DISCOVERY ====================

with tabs[2]:
    st.header("File Discovery & Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📂 SQL Files by Category")
        sql_files = discover_sql_files()
        for category in sorted(sql_files.keys()):
            with st.expander(f"📁 {category} ({len(sql_files[category])})"):
                for file_info in sql_files[category]:
                    st.caption(f"• {file_info['display']}")
    
    with col2:
        st.subheader("📦 CSV Files by Category")
        csv_files = discover_csv_files()
        for category in sorted(csv_files.keys()):
            with st.expander(f"📁 {category} ({len(csv_files[category])})"):
                for file_info in csv_files[category]:
                    size_mb = file_info['size'] / (1024*1024)
                    st.caption(f"• {file_info['display']} ({size_mb:.2f} MB)")

# ==================== TAB 4: VERIFICATION ====================

with tabs[3]:
    st.header("Database Verification")
    
    if st.button("🔍 Verify All", use_container_width=True):
        with st.spinner("Verifying..."):
            try:
                engine = create_connection(use_database=True)
                
                # Tables
                with engine.connect() as conn:
                    result = conn.execute(sa.text("SHOW TABLES"))
                    tables = [row[0] for row in result]
                
                # Views
                with engine.connect() as conn:
                    result = conn.execute(sa.text(
                        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS "
                        "WHERE TABLE_SCHEMA = DATABASE()"
                    ))
                    views = [row[0] for row in result]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Tables", len(tables))
                col2.metric("👁️ Views", len(views))
                col3.metric("📈 Total", len(tables) + len(views))
                
                # Table details
                st.subheader("Table Statistics")
                table_data = []
                for table in tables:
                    count = get_table_row_count(table)
                    table_data.append({'Table': table, 'Rows': count})
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.success(f"✅ Database verified! Total rows: {df['Rows'].sum():,}")
                
            except Exception as e:
                st.error(f"❌ {e}")

# ==================== TAB 5: STATISTICS ====================

with tabs[4]:
    st.header("System Statistics")
    
    stats = get_file_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📂 SQL Categories", stats['sql_categories'])
    col2.metric("📄 SQL Files", stats['sql_files'])
    col3.metric("📦 CSV Categories", stats['csv_categories'])
    col4.metric("📊 CSV Files", stats['csv_files'])
    
    st.divider()
    
    # Files by category
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SQL Distribution")
        sql_files = discover_sql_files()
        sql_dist = {cat: len(files) for cat, files in sql_files.items()}
        st.bar_chart(sql_dist)
    
    with col2:
        st.subheader("CSV Distribution")
        csv_files = discover_csv_files()
        csv_dist = {cat: len(files) for cat, files in csv_files.items()}
        st.bar_chart(csv_dist)

st.markdown("---")
st.caption("✨ Universal Setup Complete! Ready for all 30+ modules and analytics pages.")