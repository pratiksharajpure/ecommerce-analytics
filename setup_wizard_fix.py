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
import traceback

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
        if "database" in st.secrets:
            return st.secrets["database"]
        else:
            st.error("❌ 'database' key not found in secrets.toml")
            st.info("Please add database configuration to .streamlit/secrets.toml")
            return None
    except Exception as e:
        st.error(f"❌ Error reading secrets: {e}")
        return None

def create_connection(use_database=True):
    """Create SQLAlchemy connection"""
    try:
        db = get_db_config()
        if not db:
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

def discover_sql_files():
    """Recursively discover ALL SQL files organized by category"""
    sql_files = defaultdict(list)
    sql_dir = Path("sql")
    
    if not sql_dir.exists():
        st.warning(f"⚠️ sql/ directory not found at {sql_dir.absolute()}")
        return sql_files
    
    found_files = list(sql_dir.rglob("*.sql"))
    
    for sql_file in sorted(found_files):
        relative_path = sql_file.relative_to(sql_dir)
        category = str(relative_path.parent) if str(relative_path.parent) != '.' else 'root'
        
        sql_files[category].append({
            'name': sql_file.stem,
            'path': str(sql_file),
            'display': f"{sql_file.stem}",
            'full_path': sql_file
        })
    
    return dict(sql_files)

def discover_csv_files():
    """Recursively discover ALL CSV files organized by module"""
    csv_files = defaultdict(list)
    sample_dir = Path("sample_data")
    
    if not sample_dir.exists():
        st.warning(f"⚠️ sample_data/ directory not found at {sample_dir.absolute()}")
        return csv_files
    
    found_files = list(sample_dir.rglob("*.csv"))
    
    for csv_file in sorted(found_files):
        relative_path = csv_file.relative_to(sample_dir)
        category = str(relative_path.parent) if str(relative_path.parent) != '.' else 'root'
        
        csv_files[category].append({
            'name': csv_file.stem,
            'path': str(csv_file),
            'display': f"{csv_file.stem}",
            'size': csv_file.stat().st_size,
            'full_path': csv_file
        })
    
    return dict(csv_files)

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
        'csv_files': total_csv,
        'total_files': total_sql + total_csv
    }

# ==================== SQL EXECUTION ORDER ====================

def get_sql_execution_order():
    """Define the order in which SQL files should be executed"""
    return [
        'setup',              # Database, tables, indexes first
        'core_analysis',      # Then analysis queries
        'advanced_analysis',  # Advanced analytics
        'reporting',          # Reporting views
        'maintenance',        # Maintenance procedures
        'automation'          # Finally automation
    ]

def get_setup_file_order():
    """Specific order for setup files"""
    return [
        'create_database',
        'create_tables',
        'create_indexes',
        'create_views',
        'create_functions',
        'create_procedures',
        'create_triggers',
        'insert_lookup_data',
        'insert_sample_customers',
        'insert_sample_products',
        'insert_sample_orders',
        'insert_sample_inventory',
        'insert_sample_vendors',
        'insert_sample_campaigns',
        'insert_sample_reviews',
        'seed_data_with_issues'
    ]

# ==================== EXECUTION FUNCTIONS ====================

def execute_sql_file(file_path):
    """Execute a SQL file"""
    try:
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"
        
        with open(path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        if not sql_content.strip():
            return False, "File is empty"
        
        engine = create_connection(use_database=True)
        if not engine:
            return False, "Connection failed"
        
        statements = [s.strip() for s in sql_content.split(';') if s.strip()]
        
        with engine.connect() as conn:
            errors = []
            for i, statement in enumerate(statements):
                if statement:
                    try:
                        conn.execute(sa.text(statement))
                    except Exception as e:
                        errors.append(f"Statement {i+1}: {str(e)[:50]}")
            conn.commit()
        
        if errors:
            return False, "; ".join(errors[:2])
        
        return True, f"Executed {len(statements)} statements"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"

def load_csv_to_table(csv_path, table_name):
    """Load CSV into MySQL table"""
    try:
        path = Path(csv_path)
        if not path.exists():
            return False, f"CSV not found: {csv_path}"
        
        df = pd.read_csv(path)
        
        if df.empty:
            return False, "CSV is empty"
        
        engine = create_connection(use_database=True)
        if not engine:
            return False, "Connection failed"
        
        df.to_sql(table_name, engine, if_exists='append', index=False)
        return True, f"Loaded {len(df)} rows"
        
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"

def get_table_row_count(table_name):
    """Get row count for a table"""
    try:
        engine = create_connection(use_database=True)
        if not engine:
            return 0
        with engine.connect() as conn:
            result = conn.execute(sa.text(f"SELECT COUNT(*) FROM `{table_name}`"))
            return result.fetchone()[0]
    except:
        return 0

def execute_sql_in_order():
    """Execute SQL files in proper dependency order"""
    sql_files = discover_sql_files()
    execution_order = get_sql_execution_order()
    setup_order = get_setup_file_order()
    
    results = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }
    
    # Execute in category order
    for category in execution_order:
        if category not in sql_files:
            continue
        
        files = sql_files[category]
        
        # Special ordering for setup files
        if category == 'setup':
            ordered_files = []
            for order_name in setup_order:
                for file_info in files:
                    if file_info['name'] == order_name:
                        ordered_files.append(file_info)
                        break
            # Add any remaining setup files not in order list
            for file_info in files:
                if file_info not in ordered_files:
                    ordered_files.append(file_info)
            files = ordered_files
        
        # Execute files in category
        for file_info in files:
            results['total'] += 1
            ok, msg = execute_sql_file(file_info['path'])
            
            if ok:
                results['success'] += 1
                status = '✅'
            else:
                results['failed'] += 1
                status = '❌'
            
            results['details'].append({
                'category': category,
                'file': file_info['name'],
                'status': status,
                'message': msg
            })
    
    return results

# ==================== DASHBOARD UI ====================

tabs = st.tabs([
    "🚀 Quick Start",
    "📋 Category Setup",
    "🔍 File Browser",
    "✅ Verification",
    "📊 Statistics"
])

# ==================== TAB 1: QUICK START ====================

with tabs[0]:
    st.header("Quick Start Setup")
    
    # File count overview
    stats = get_file_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 SQL Files", stats['sql_files'])
    col2.metric("📊 CSV Files", stats['csv_files'])
    col3.metric("🗂️ Categories", stats['sql_categories'] + stats['csv_categories'])
    col4.metric("📦 Total Files", stats['total_files'])
    
    st.divider()
    
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
                    else:
                        st.error("❌ Failed to create connection")
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")
    
    with col2:
        if st.button("⚡ Setup All SQL (Ordered)", use_container_width=True, type="primary"):
            with st.spinner("Executing SQL in proper order..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = execute_sql_in_order()
                
                # Show results
                st.success(f"✅ Completed: {results['success']}/{results['total']} files")
                if results['failed'] > 0:
                    st.warning(f"⚠️ Failed: {results['failed']} files")
                
                # Show detailed results
                with st.expander("📋 Detailed Execution Log"):
                    for detail in results['details']:
                        st.text(f"{detail['status']} [{detail['category']}] {detail['file']}: {detail['message']}")
    
    with col3:
        if st.button("📥 Load All CSVs", use_container_width=True, type="primary"):
            with st.spinner("Loading all CSV files..."):
                csv_files = discover_csv_files()
                if not csv_files:
                    st.warning("⚠️ No CSV files found!")
                else:
                    success = 0
                    failed = 0
                    total_files = sum(len(files) for files in csv_files.values())
                    current = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for category, files in csv_files.items():
                        for file_info in files:
                            current += 1
                            table_name = file_info['name']
                            status_text.text(f"Loading {table_name}... ({current}/{total_files})")
                            progress_bar.progress(current / total_files)
                            
                            ok, msg = load_csv_to_table(file_info['path'], table_name)
                            if ok:
                                success += 1
                            else:
                                failed += 1
                    
                    st.success(f"✅ Loaded: {success}/{total_files} CSV files")
                    if failed > 0:
                        st.warning(f"⚠️ Failed: {failed} files")
    
    st.divider()
    
    # Show current directory
    st.info(f"📂 Working Directory: {Path.cwd()}")
    
    # Quick diagnostics
    with st.expander("🔍 Quick Diagnostics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SQL Files")
            sql_dir = Path("sql")
            if sql_dir.exists():
                st.success(f"✅ Directory exists")
                sql_count = len(list(sql_dir.rglob("*.sql")))
                st.info(f"Found {sql_count} SQL files")
            else:
                st.error(f"❌ Directory not found")
        
        with col2:
            st.subheader("CSV Files")
            csv_dir = Path("sample_data")
            if csv_dir.exists():
                st.success(f"✅ Directory exists")
                csv_count = len(list(csv_dir.rglob("*.csv")))
                st.info(f"Found {csv_count} CSV files")
            else:
                st.error(f"❌ Directory not found")

# ==================== TAB 2: CATEGORY SETUP ====================

with tabs[1]:
    st.header("Category-Based Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 SQL Categories")
        sql_files = discover_sql_files()
        
        if not sql_files:
            st.warning("⚠️ No SQL files found")
        else:
            for category in sorted(sql_files.keys()):
                with st.expander(f"📁 {category} ({len(sql_files[category])} files)"):
                    st.caption(f"Files in this category:")
                    for file_info in sql_files[category]:
                        st.text(f"  • {file_info['name']}")
                    
                    if st.button(f"Execute {category}", key=f"exec_{category}"):
                        with st.spinner(f"Executing {category} files..."):
                            success = 0
                            failed = 0
                            for file_info in sql_files[category]:
                                ok, msg = execute_sql_file(file_info['path'])
                                if ok:
                                    success += 1
                                    st.success(f"✅ {file_info['name']}")
                                else:
                                    failed += 1
                                    st.error(f"❌ {file_info['name']}: {msg}")
                            st.info(f"Complete: {success} success, {failed} failed")
    
    with col2:
        st.subheader("📊 CSV Categories")
        csv_files = discover_csv_files()
        
        if not csv_files:
            st.warning("⚠️ No CSV files found")
        else:
            for category in sorted(csv_files.keys()):
                with st.expander(f"📁 {category} ({len(csv_files[category])} files)"):
                    st.caption(f"Files in this category:")
                    for file_info in csv_files[category]:
                        size_mb = file_info['size'] / (1024*1024)
                        st.text(f"  • {file_info['name']} ({size_mb:.2f} MB)")
                    
                    if st.button(f"Load {category}", key=f"load_{category}"):
                        with st.spinner(f"Loading {category} files..."):
                            success = 0
                            failed = 0
                            for file_info in csv_files[category]:
                                table_name = file_info['name']
                                ok, msg = load_csv_to_table(file_info['path'], table_name)
                                if ok:
                                    success += 1
                                    st.success(f"✅ {table_name}: {msg}")
                                else:
                                    failed += 1
                                    st.error(f"❌ {table_name}: {msg}")
                            st.info(f"Complete: {success} success, {failed} failed")

# ==================== TAB 3: FILE BROWSER ====================

with tabs[2]:
    st.header("Complete File Browser")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📂 SQL Files (All 95 files)")
        sql_files = discover_sql_files()
        
        for category in sorted(sql_files.keys()):
            with st.expander(f"📁 {category} ({len(sql_files[category])} files)"):
                for file_info in sql_files[category]:
                    exists = file_info['full_path'].exists()
                    status = "✅" if exists else "❌"
                    st.caption(f"{status} {file_info['name']}")
    
    with col2:
        st.subheader("📦 CSV Files (All 84 files)")
        csv_files = discover_csv_files()
        
        for category in sorted(csv_files.keys()):
            with st.expander(f"📁 {category} ({len(csv_files[category])} files)"):
                for file_info in csv_files[category]:
                    exists = file_info['full_path'].exists()
                    size_mb = file_info['size'] / (1024*1024)
                    status = "✅" if exists else "❌"
                    st.caption(f"{status} {file_info['name']} ({size_mb:.2f} MB)")

# ==================== TAB 4: VERIFICATION ====================

with tabs[3]:
    st.header("Database Verification")
    
    if st.button("🔍 Verify Database", use_container_width=True, type="primary"):
        with st.spinner("Verifying database..."):
            try:
                engine = create_connection(use_database=True)
                
                if not engine:
                    st.error("❌ Could not connect to database")
                else:
                    # Get tables
                    with engine.connect() as conn:
                        result = conn.execute(sa.text("SHOW TABLES"))
                        tables = [row[0] for row in result]
                    
                    # Get views
                    with engine.connect() as conn:
                        result = conn.execute(sa.text(
                            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.VIEWS "
                            "WHERE TABLE_SCHEMA = DATABASE()"
                        ))
                        views = [row[0] for row in result]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📊 Tables", len(tables))
                    col2.metric("👁️ Views", len(views))
                    col3.metric("📈 Total Objects", len(tables) + len(views))
                    
                    # Table details with row counts
                    if tables:
                        st.subheader("📋 Table Statistics")
                        table_data = []
                        progress_bar = st.progress(0)
                        
                        for i, table in enumerate(tables):
                            progress_bar.progress((i + 1) / len(tables))
                            count = get_table_row_count(table)
                            table_data.append({'Table': table, 'Rows': count})
                        
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        total_rows = df['Rows'].sum()
                        st.success(f"✅ Database verified! Total rows: {total_rows:,}")
                    else:
                        st.warning("⚠️ No tables found in database")
                
            except Exception as e:
                st.error(f"❌ Verification Error: {e}")
                st.error(traceback.format_exc())

# ==================== TAB 5: STATISTICS ====================

with tabs[4]:
    st.header("Complete System Statistics")
    
    stats = get_file_stats()
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📂 SQL Categories", stats['sql_categories'])
    col2.metric("📄 SQL Files", stats['sql_files'])
    col3.metric("📦 CSV Categories", stats['csv_categories'])
    col4.metric("📊 CSV Files", stats['csv_files'])
    
    st.divider()
    
    # Expected vs Found
    st.subheader("📋 File Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Expected SQL Files", "95")
        st.metric("Found SQL Files", stats['sql_files'])
        if stats['sql_files'] == 95:
            st.success("✅ All SQL files present!")
        else:
            st.warning(f"⚠️ Missing {95 - stats['sql_files']} SQL files")
    
    with col2:
        st.metric("Expected CSV Files", "84")
        st.metric("Found CSV Files", stats['csv_files'])
        if stats['csv_files'] == 84:
            st.success("✅ All CSV files present!")
        else:
            st.warning(f"⚠️ Missing {84 - stats['csv_files']} CSV files")
    
    st.divider()
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SQL File Distribution")
        sql_files = discover_sql_files()
        if sql_files:
            sql_dist = {cat: len(files) for cat, files in sql_files.items()}
            st.bar_chart(sql_dist)
            
            # Detailed breakdown
            with st.expander("📊 Detailed SQL Breakdown"):
                for cat, count in sorted(sql_dist.items()):
                    st.text(f"{cat}: {count} files")
        else:
            st.info("No SQL files to display")
    
    with col2:
        st.subheader("CSV File Distribution")
        csv_files = discover_csv_files()
        if csv_files:
            csv_dist = {cat: len(files) for cat, files in csv_files.items()}
            st.bar_chart(csv_dist)
            
            # Detailed breakdown
            with st.expander("📊 Detailed CSV Breakdown"):
                for cat, count in sorted(csv_dist.items()):
                    st.text(f"{cat}: {count} files")
        else:
            st.info("No CSV files to display")

st.markdown("---")
st.caption(f"✨ Universal Setup Wizard | Managing {stats.get('total_files', 179)} files across SQL and CSV")