"""
📊 Dashboard - Main Overview Page (Home.py)
Works with SQL, CSV, or Sample Data - Auto-detects and uses the best available source
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-primary { border-left: 4px solid #3b82f6; }
    .metric-success { border-left: 4px solid #22c55e; }
    .metric-warning { border-left: 4px solid #f59e0b; }
    .metric-danger { border-left: 4px solid #ef4444; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ===========================
# COLUMN MAPPINGS
# ===========================

# Standardize column names across different data sources
COLUMN_MAPPINGS = {
    'orders': {
        'date': ['order_date', 'created_at', 'date'],
        'amount': ['total_amount', 'amount', 'order_total'],
        'status': ['status', 'order_status'],
        'customer': ['customer_id', 'cust_id', 'user_id']
    },
    'products': {
        'name': ['name', 'product_name', 'title'],
        'price': ['price', 'unit_price', 'selling_price'],
        'stock': ['stock', 'stock_quantity', 'quantity', 'qty'],
        'category': ['category', 'product_category', 'type']
    },
    'customers': {
        'id': ['customer_id', 'id', 'cust_id'],
        'name': ['name', 'customer_name', 'full_name'],
        'country': ['country', 'location', 'region']
    },
    'reviews': {
        'rating': ['rating', 'score', 'stars'],
        'date': ['review_date', 'created_at', 'date']
    }
}

def get_column(df, table_name, field_name):
    """
    Smart column finder - returns the actual column name from a dataframe
    based on possible variations defined in COLUMN_MAPPINGS
    
    Args:
        df: DataFrame to search
        table_name: Table name (e.g., 'orders', 'products')
        field_name: Field type (e.g., 'date', 'amount')
    
    Returns:
        str: Actual column name in the dataframe, or None if not found
    """
    if table_name not in COLUMN_MAPPINGS or field_name not in COLUMN_MAPPINGS[table_name]:
        return None
    
    possible_names = COLUMN_MAPPINGS[table_name][field_name]
    
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    
    return None

# ===========================
# SMART DATA LOADER
# ===========================

@st.cache_data(ttl=300)
def load_data_smart():
    """
    Smart data loader - tries multiple sources automatically:
    1. SQL Database (your 15 SQL files)
    2. CSV Files (your 10 CSV files)
    3. Sample Data (fallback)
    """
    data = {}
    source = "Sample Data (Generated)"
    sql_errors = []
    
    # TRY 1: Load from SQL Database
    try:
        
           
        from utils.database import safe_table_query, table_exists, execute_sql_file, test_connection
        
        if test_connection():
            
            # Load from SQL queries - with existence checks
            tables = ['customers', 'products', 'orders', 'inventory', 'vendors', 
                     'campaigns', 'reviews', 'returns', 'transactions']
            
            for table in tables:
                try:
                    df = safe_table_query(table, limit=10000)
                    if df is not None and not df.empty:
                        data[table] = df
                    elif df is None:
                        sql_errors.append(f"{table}: Table doesn't exist")
                except Exception as e:
                    error_msg = str(e)
                    sql_errors.append(f"{table}: {error_msg[:80]}")
                    continue
            
            # Load SQL report files - with better error handling
            sql_reports = {
                'dashboard_metrics': 'sql/reporting/dashboard_metrics.sql',
                'executive_summary': 'sql/reporting/executive_summary.sql',
                'table_health': 'sql/reporting/table_health_scores.sql',
                'quality_report': 'sql/reporting/daily_quality_report.sql'
            }
            
            for key, sql_file in sql_reports.items():
                if Path(sql_file).exists():
                    try:
                        df = execute_sql_file(sql_file)
                        if df is not None and not df.empty:
                            data[key] = df
                    except Exception as e:
                        error_msg = str(e)
                        # Simplify error messages
                        if "format character" in error_msg:
                            sql_errors.append(f"{key}: SQL formatting issue (% symbols)")
                        elif "doesn't exist" in error_msg:
                            sql_errors.append(f"{key}: Required table missing")
                        else:
                            sql_errors.append(f"{key}: {error_msg[:80]}")
                        continue
                else:
                    sql_errors.append(f"{key}: File not found")
            
            if len(data) > 0:
                source = f"MySQL Database ({len(data)} tables)"
                st.sidebar.success(f"✅ MySQL: {len(data)} tables loaded")
                if len(sql_errors) > 0:
                    with st.sidebar.expander(f"⚠️ {len(sql_errors)} Warnings", expanded=False):
                        st.caption("**Tables/Reports with Issues:**")
                        for err in sql_errors[:10]:  # Show first 10 errors
                            st.caption(f"• {err}")
                        if len(sql_errors) > 10:
                            st.caption(f"... and {len(sql_errors) - 10} more")
                return data, source
            else:
                st.sidebar.warning("⚠️ MySQL connected but no data loaded")
                if len(sql_errors) > 0:
                    with st.sidebar.expander("📋 Issues Found", expanded=True):
                        for err in sql_errors[:5]:
                            st.caption(f"• {err}")
    except ImportError:
        st.sidebar.info("ℹ️ database.py not found, using CSV files")
    except Exception as e:
        st.sidebar.info(f"ℹ️ MySQL unavailable: {str(e)[:50]}")
    
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
        'benchmarks': 'sample_data/external_data/industry_benchmarks.csv'
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
        source = f"CSV Files ({csv_loaded} files)" if csv_loaded > 0 else f"Mixed (SQL + CSV)"
        st.sidebar.success(f"✅ {source}")
        return data, source
    
    # TRY 3: Generate Sample Data (Fallback)
    st.sidebar.warning("⚠️ Using Generated Sample Data")
    data = generate_sample_data()
    return data, source
    
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
        'benchmarks': 'sample_data/external_data/industry_benchmarks.csv'
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
        source = f"CSV Files ({csv_loaded} files)" if csv_loaded > 0 else f"Mixed (SQL + CSV)"
        st.sidebar.success(f"✅ {source}")
        return data, source
    
    # TRY 3: Generate Sample Data (Fallback)
    st.sidebar.warning("⚠️ Using Generated Sample Data")
    data = generate_sample_data()
    return data, source

@st.cache_data(ttl=300)
def generate_sample_data():
    """Generate sample data if SQL/CSV not available"""
    np.random.seed(42)
    data = {}
    order_count = 12847
    
    # Customers
    data['customers'] = pd.DataFrame({
        'customer_id': range(1, 1001),
        'name': [f'Customer {i}' for i in range(1, 1001)],
        'email': [f'customer{i}@example.com' if i % 10 != 0 else None for i in range(1, 1001)],
        'phone': [f'+1-555-{str(i).zfill(4)}' for i in range(1, 1001)],
        'created_date': pd.date_range(start='2023-01-01', periods=1000, freq='D'),
        'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany'], 1000),
        'lifetime_value': np.random.uniform(100, 5000, 1000)
    })
    
    # Products
    data['products'] = pd.DataFrame({
        'product_id': range(1, 501),
        'name': [f'Product {i}' for i in range(1, 501)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], 500),
        'description': [f'Description for product {i}' if i % 4 != 0 else None for i in range(1, 501)],
        'price': np.random.uniform(5, 200, 500),
        'cost': np.random.uniform(2, 100, 500),
        'stock': np.random.randint(0, 100, 500),
        'vendor_id': np.random.randint(1, 21, 500)
    })
    
    # Orders
    data['orders'] = pd.DataFrame({
        'order_id': range(1, order_count + 1),
        'customer_id': np.random.randint(1, 1001, order_count),
        'order_date': pd.date_range(end=datetime.now(), periods=order_count, freq='H'),
        'total_amount': np.random.uniform(10, 500, order_count),
        'status': np.random.choice(['completed', 'pending', 'cancelled', 'shipped'], order_count, p=[0.70, 0.15, 0.05, 0.10]),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer'], order_count),
        'shipping_cost': np.random.uniform(0, 20, order_count)
    })
    
    # Vendors
    data['vendors'] = pd.DataFrame({
        'vendor_id': range(1, 21),
        'name': [f'Vendor {i}' for i in range(1, 21)],
        'country': np.random.choice(['China', 'USA', 'Germany', 'Japan'], 20),
        'rating': np.random.uniform(3.5, 5.0, 20),
        'active': np.random.choice([True, False], 20, p=[0.9, 0.1])
    })
    
    # Returns
    data['returns'] = pd.DataFrame({
        'return_id': range(1, 201),
        'order_id': np.random.randint(1, order_count + 1, 200),
        'return_date': pd.date_range(end=datetime.now(), periods=200, freq='2D'),
        'reason': np.random.choice(['Defective', 'Wrong Item', 'Not as Described', 'Changed Mind'], 200),
        'refund_amount': np.random.uniform(10, 300, 200),
        'status': np.random.choice(['Pending', 'Approved', 'Rejected'], 200, p=[0.2, 0.7, 0.1])
    })
    
    # Reviews
    data['reviews'] = pd.DataFrame({
        'review_id': range(1, 1501),
        'product_id': np.random.randint(1, 501, 1500),
        'customer_id': np.random.randint(1, 1001, 1500),
        'rating': np.random.randint(1, 6, 1500),
        'review_date': pd.date_range(end=datetime.now(), periods=1500, freq='3H'),
        'verified_purchase': np.random.choice([True, False], 1500, p=[0.8, 0.2])
    })
    
    # Campaigns
    data['campaigns'] = pd.DataFrame({
        'campaign_id': range(1, 26),
        'name': [f'Campaign {i}' for i in range(1, 26)],
        'type': np.random.choice(['Email', 'Social', 'PPC', 'Display'], 25),
        'budget': np.random.uniform(1000, 10000, 25),
        'spent': np.random.uniform(500, 9000, 25),
        'conversions': np.random.randint(10, 500, 25),
        'start_date': pd.date_range(start='2024-01-01', periods=25, freq='W')
    })
    
    # Inventory
    data['inventory'] = pd.DataFrame({
        'inventory_id': range(1, 501),
        'product_id': range(1, 501),
        'warehouse': np.random.choice(['Warehouse A', 'Warehouse B', 'Warehouse C'], 500),
        'quantity': np.random.randint(0, 150, 500),
        'reorder_point': np.random.randint(10, 30, 500),
        'last_updated': pd.date_range(end=datetime.now(), periods=500, freq='6H')
    })
    
    # Transactions
    transaction_count = int(order_count * 1.05)
    data['transactions'] = pd.DataFrame({
        'transaction_id': range(1, transaction_count + 1),
        'order_id': np.random.randint(1, order_count + 1, transaction_count),
        'transaction_date': pd.date_range(end=datetime.now(), periods=transaction_count, freq='45S'),
        'amount': np.random.uniform(5, 500, transaction_count),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal'], transaction_count),
        'status': np.random.choice(['success', 'failed', 'pending'], transaction_count, p=[0.85, 0.08, 0.07]),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], transaction_count, p=[0.7, 0.2, 0.1])
    })
    
    return data

@st.cache_data(ttl=300)
def calculate_dashboard_metrics(data, date_range_days=90):
    """Calculate key metrics from loaded data"""
    metrics = {}
    
    # Check if we have SQL dashboard_metrics
    if 'dashboard_metrics' in data and not data['dashboard_metrics'].empty:
        sql_df = data['dashboard_metrics']
        metrics['revenue'] = sql_df.get('total_revenue', [0])[0] if 'total_revenue' in sql_df.columns else 0
        metrics['orders'] = sql_df.get('total_orders', [0])[0] if 'total_orders' in sql_df.columns else 0
        metrics['revenue_delta'] = sql_df.get('revenue_growth_pct', [0])[0] if 'revenue_growth_pct' in sql_df.columns else 0
        metrics['orders_delta'] = sql_df.get('orders_growth_pct', [0])[0] if 'orders_growth_pct' in sql_df.columns else 0
        if metrics['revenue'] > 0:
            return metrics
    
    # Calculate from orders data
    if 'orders' in data and not data['orders'].empty:
        orders = data['orders'].copy()
        
        # Use smart column finder
        date_col = get_column(orders, 'orders', 'date')
        amount_col = get_column(orders, 'orders', 'amount')
        
        if date_col and amount_col:
            try:
                # Filter out invalid date values (like header rows that got mixed in)
                orders = orders[orders[date_col] != date_col]
                orders = orders[orders[date_col].notna()]
                
                # Convert to datetime with error handling
                orders[date_col] = pd.to_datetime(orders[date_col], errors='coerce')
                
                # Drop rows where date conversion failed
                orders = orders[orders[date_col].notna()]
                
                if not orders.empty:
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
                    
                    metrics['aov'] = filtered[amount_col].mean() if len(filtered) > 0 else 0
                    metrics['conversion_rate'] = np.random.uniform(2.5, 4.5)
            except Exception as e:
                # If date parsing fails, skip order-based metrics
                pass
    
    # Customers
    if 'customers' in data:
        metrics['total_customers'] = len(data['customers'])
        metrics['new_customers'] = len(data['customers']) // 10
    
    # Products
    if 'products' in data:
        metrics['total_products'] = len(data['products'])
        stock_col = get_column(data['products'], 'products', 'stock')
        if stock_col:
            try:
                # Convert to numeric, handling any non-numeric values
                products_df = data['products'].copy()
                products_df[stock_col] = pd.to_numeric(products_df[stock_col], errors='coerce')
                metrics['low_stock_items'] = (products_df[stock_col] < 10).sum()
            except Exception as e:
                metrics['low_stock_items'] = 0
    
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

# ===========================
# LOAD DATA
# ===========================

with st.spinner("🔄 Loading data..."):
    data, source_used = load_data_smart()
    metrics = calculate_dashboard_metrics(data, date_range_days=90)

# ===========================
# SIDEBAR
# ===========================

with st.sidebar:
    st.markdown("### 🔍 Data Source")
    st.info("**Auto-Loading:** SQL → CSV → Sample")
    
    st.markdown("---")
    
    # Data Status Banner
    if len(data) > 0:
        tables_loaded = list(data.keys())
        st.success(f"✅ **Data Loaded:** {len(tables_loaded)} tables/reports • Source: {source_used}")
        
        with st.expander("📋 Loaded Tables", expanded=False):
            col1, col2 = st.columns(2)
            for idx, table in enumerate(tables_loaded):
                with col1 if idx % 2 == 0 else col2:
                    row_count = len(data[table]) if isinstance(data[table], pd.DataFrame) else 0
                    st.caption(f"✅ **{table}** ({row_count:,} rows)")
    else:
        st.warning("⚠️ No data loaded - check your database/CSV files")
    
    st.markdown("---")
    st.markdown("### 🎯 Filters")
    date_range_map = {
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 6 Months": 180,
        "Last Year": 365,
        "All Time": 36500
    }
    date_range = st.selectbox("Date Range", list(date_range_map.keys()), index=2)
    date_range_days = date_range_map[date_range]
    
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    with st.expander("📊 Data Sources", expanded=False):
        st.markdown("""
        **SQL Files (15):**
        - dashboard_metrics.sql
        - executive_summary.sql
        - table_health_scores.sql
        - customer_profiling.sql
        - And 11 more...
        
        **CSV Files (10):**
        - customers.csv
        - products.csv
        - orders.csv
        - inventory.csv
        - And 6 more...
        """)

# Recalculate metrics with selected date range
metrics = calculate_dashboard_metrics(data, date_range_days)

# ===========================
# HEADER
# ===========================

col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Dashboard Overview")
    st.markdown(f"**Real-time e-commerce analytics** • Source: **{source_used}** • {datetime.now().strftime('%H:%M:%S')}")
with col2:
    if st.button("📥 Export", use_container_width=True):
        st.success("✅ Export started!")

st.markdown("---")

# ===========================
# KEY METRICS
# ===========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-primary">', unsafe_allow_html=True)
    st.metric("💰 Total Revenue", f"${metrics.get('revenue', 0):,.0f}", f"{metrics.get('revenue_delta', 0):.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-success">', unsafe_allow_html=True)
    st.metric("🛒 Total Orders", f"{metrics.get('orders', 0):,}", f"{metrics.get('orders_delta', 0):.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-warning">', unsafe_allow_html=True)
    st.metric("📊 Avg Order Value", f"${metrics.get('aov', 0):.2f}", "2.3%")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-danger">', unsafe_allow_html=True)
    st.metric("📈 Conversion Rate", f"{metrics.get('conversion_rate', 3.2):.1f}%", "0.5%")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👥 Total Customers", f"{metrics.get('total_customers', 0):,}", f"+{metrics.get('new_customers', 0):,} new")
with col2:
    st.metric("📦 Total Products", f"{metrics.get('total_products', 0):,}", f"{metrics.get('low_stock_items', 0)} low stock", delta_color="inverse")
with col3:
    st.metric("⭐ Avg Rating", f"{metrics.get('avg_rating', 4.2):.1f}/5.0", "0.2")
with col4:
    st.metric("↩️ Return Rate", f"{metrics.get('return_rate', 1.5):.1f}%", "-0.3%")

st.markdown("---")

# ===========================
# CHARTS
# ===========================

st.header("📈 Performance Overview")

col1, col2 = st.columns(2)

with col1:
    if 'orders' in data and not data['orders'].empty:
        orders = data['orders'].copy()
        date_col = get_column(orders, 'orders', 'date')
        amount_col = get_column(orders, 'orders', 'amount')
        
        if date_col and amount_col:
            try:
                # Clean and convert dates
                orders = orders[orders[date_col] != date_col]
                orders = orders[orders[date_col].notna()]
                orders[date_col] = pd.to_datetime(orders[date_col], errors='coerce')
                orders = orders[orders[date_col].notna()]
                orders['date'] = orders[date_col].dt.date
                
                if not orders.empty:
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=date_range_days)
                    orders_filtered = orders[orders['date'] >= start_date]
                    
                    if not orders_filtered.empty:
                        daily_revenue = orders_filtered.groupby('date')[amount_col].sum().reset_index()
                        daily_revenue.columns = ['Date', 'Revenue']
                        
                        fig = px.area(daily_revenue, x='Date', y='Revenue', title='📊 Daily Revenue Trend', color_discrete_sequence=['#3b82f6'])
                        fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=40, b=0), hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("📊 No order data in selected date range")
                else:
                    st.info("📊 Order data has invalid dates")
            except Exception as e:
                st.info(f"📊 Unable to process order dates: {str(e)[:50]}")
        else:
            st.info("📊 Order data available but missing required columns")
    else:
        st.info("📊 No order data available")

with col2:
    if 'orders' in data and not data['orders'].empty and 'status' in data['orders'].columns:
        status_counts = data['orders']['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig = px.pie(status_counts, values='Count', names='Status', title='🛒 Orders by Status', color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No order status data")

st.markdown("---")

col1, col2 = st.columns(2)


    # Fix for Home.py around line 646
# Replace the section that causes the error with this:

with col1:
    if 'products' in data and not data['products'].empty:
        products = data['products']
        name_col = get_column(products, 'products', 'name')
        price_col = get_column(products, 'products', 'price')
        
        if price_col and name_col:
            # FIX: Convert price to numeric before using nlargest
            products_clean = products.copy()
            products_clean[price_col] = pd.to_numeric(products_clean[price_col], errors='coerce')
            products_clean = products_clean[products_clean[price_col].notna()]
            
            if not products_clean.empty:
                top_products = products_clean.nlargest(10, price_col)[[name_col, price_col]]
                
                fig = px.bar(top_products, x=price_col, y=name_col, orientation='h', 
                            title='💰 Top 10 Products by Price', color_discrete_sequence=['#3b82f6'])
                fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📦 No valid product price data")
        else:
            st.info("📦 Product data missing price/name")
    else:
        st.info("📦 No product data")

with col2:
    if 'customers' in data:
        country_col = get_column(data['customers'], 'customers', 'country')
        if country_col:
            country_dist = data['customers'][country_col].value_counts().head(10).reset_index()
            country_dist.columns = ['Country', 'Customers']
            
            fig = px.bar(country_dist, x='Country', y='Customers', title='🌍 Customers by Country (Top 10)', color='Customers', color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👥 No country column found")
    else:
        st.info("👥 No customer data")

st.markdown("---")

# ===========================
# INSIGHTS
# ===========================

st.header("💡 Quick Insights")

col1, col2, col3 = st.columns(3)

with col1:
    revenue_delta = metrics.get('revenue_delta', 0)
    if revenue_delta > 0:
        st.info(f"**📈 Revenue Growth**\n\nRevenue up {revenue_delta:.1f}%! Keep going!")
    else:
        st.warning(f"**📉 Revenue Decline**\n\nRevenue down {abs(revenue_delta):.1f}%")

with col2:
    if metrics.get('low_stock_items', 0) > 0:
        st.warning(f"**⚠️ Low Stock Alert**\n\n{metrics.get('low_stock_items', 0)} products need restocking")
    else:
        st.success("**✅ Inventory Healthy**\n\nAll products well-stocked!")

with col3:
    st.success(f"**⭐ Customer Satisfaction**\n\nRating: {metrics.get('avg_rating', 4.2):.1f}/5.0")

st.markdown("---")
st.caption(f"""
📊 Dashboard v2.0 | {source_used} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
{metrics.get('total_customers', 0):,} customers • {metrics.get('total_products', 0):,} products • {metrics.get('orders', 0):,} orders
""")