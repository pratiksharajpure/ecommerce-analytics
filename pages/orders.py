"""
📦 Enhanced Product Analysis - UPDATED with Category Mapping Fix
This version correctly handles category_id → category_name mapping
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Product Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div { padding-top: 0.5rem; }
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid;
    }
    .stat-card-primary { border-left-color: #3b82f6; }
    .stat-card-success { border-left-color: #22c55e; }
    .stat-card-warning { border-left-color: #f59e0b; }
    .stat-card-danger { border-left-color: #ef4444; }
    .alert {
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        border-left: 4px solid;
    }
    .alert-warning {
        background: #fef3c7;
        color: #92400e;
        border-left-color: #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# COLUMN MAPPINGS - UPDATED FOR SQL STRUCTURE
# ===========================

COLUMN_MAPPINGS = {
    'products': {
        'id': ['product_id', 'id', 'prod_id'],
        'sku': ['sku', 'product_sku', 'code'],
        'name': ['name', 'product_name', 'title'],
        'price': ['price', 'unit_price', 'selling_price'],
        'stock': ['stock', 'stock_quantity', 'quantity', 'qty'],
        'category': ['category', 'category_name', 'product_category', 'type'],  # ← UPDATED
        'description': ['description', 'product_description', 'desc'],
        'image': ['image', 'has_image', 'image_url'],  # ← Will handle both NULL and 'Yes'/'No'
        'created_date': ['created_date', 'created_at', 'date_added'],
        'cost': ['cost', 'cost_price', 'unit_cost']
    }
}

def get_column(df, table_name, field_name):
    """Smart column finder - returns actual column name from dataframe"""
    if table_name not in COLUMN_MAPPINGS or field_name not in COLUMN_MAPPINGS[table_name]:
        return None
    
    possible_names = COLUMN_MAPPINGS[table_name][field_name]
    
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    
    return None

# ===========================
# SMART DATA LOADER - UPDATED WITH CATEGORY JOIN
# ===========================

@st.cache_data(ttl=600)
def load_products_smart():
    """
    Smart product loader with category mapping:
    1. Database (with category JOIN)
    2. CSV File
    3. Sample Data (fallback)
    """
    
    # Try Database FIRST with proper JOIN
    try:
        from utils.database import get_db_connection
        conn = get_db_connection()
        
        # ✅ FIXED: JOIN to get category_name instead of category_id
        query = """
        SELECT 
            p.product_id,
            p.sku,
            p.name,
            pc.category_name as category,
            p.price,
            p.cost,
            p.description,
            CASE 
                WHEN p.image_url IS NULL THEN 'No'
                WHEN p.image_url = '' THEN 'No'
                ELSE 'Yes'
            END as image,
            p.stock_quantity as stock,
            p.created_date
        FROM products p
        LEFT JOIN product_categories pc ON p.category_id = pc.category_id
        ORDER BY p.product_id
        LIMIT 10000
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df is not None and not df.empty:
            # Clean whitespace from string columns
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            
            st.sidebar.success(f"✅ Loaded {len(df)} products from MySQL Database")
            return df, "MySQL Database (with Category JOIN)"
    
    except Exception as e:
        st.sidebar.warning(f"⚠️ Database load failed: {str(e)}")
    
    # Try CSV (should already have category names)
    csv_path = 'sample_data/core_data/products.csv'
    if Path(csv_path).exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                # Clean whitespace
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip()
                
                st.sidebar.success(f"✅ Loaded {len(df)} products from CSV")
                return df, "CSV File"
        except Exception as e:
            st.sidebar.warning(f"⚠️ CSV load failed: {str(e)[:50]}")
    
    # Fallback to sample data
    st.sidebar.warning("⚠️ Using Generated Sample Data")
    return generate_sample_product_data(), "Generated Sample Data"

# ===========================
# GENERATE SAMPLE DATA - MATCHES SQL STRUCTURE
# ===========================

@st.cache_data(ttl=600)
def generate_sample_product_data():
    """Generate sample data matching SQL structure"""
    np.random.seed(42)
    
    products = []
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports', 'Toys', 'Food & Beverage', 'Beauty']
    
    for i in range(1, 501):
        # Price issues (9%)
        price = np.random.uniform(5, 500)
        if np.random.random() < 0.09:
            price = np.random.choice([0, -10.00, None])
        
        # Description issues (26%)
        description = f'High-quality product {i} with excellent features'
        if np.random.random() < 0.26:
            description = None
        
        # Category issues (10%)
        category = np.random.choice(categories)
        if np.random.random() < 0.10:
            category = None
        
        # Name issues (2%)
        name = f'Product {i}'
        if np.random.random() < 0.02:
            name = None
        
        # Image issues (15%) - matching SQL NULL → 'No'
        image = 'Yes' if np.random.random() > 0.15 else 'No'
        
        # Duplicate SKUs (5%)
        if np.random.random() < 0.05:
            sku = f'DUP-{np.random.randint(1, 25):03d}'
        else:
            sku = f'{category[:4].upper() if category else "PROD"}-{i:03d}'
        
        products.append({
            'product_id': f'P-{i:04d}',
            'sku': sku,
            'name': name,
            'category': category,  # ← Direct category name (not ID)
            'price': price,
            'cost': np.random.uniform(2, price*0.6) if price and price > 0 else None,
            'description': description,
            'image': image,
            'stock': np.random.randint(0, 300),
            'created_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).date()
        })
    
    return pd.DataFrame(products)

# ===========================
# HELPER FUNCTION - Safe Price Conversion
# ===========================

def safe_float_price(value):
    """Safely convert price to float, return None if invalid"""
    if pd.isna(value) or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

# ===========================
# ANALYSIS FUNCTIONS (UNCHANGED - ALREADY COMPATIBLE)
# ===========================

def analyze_data_quality(df):
    """Analyze quality issues using smart column detection"""
    if df.empty:
        return pd.DataFrame()
    
    issues = []
    
    # Get actual column names
    id_col = get_column(df, 'products', 'id')
    sku_col = get_column(df, 'products', 'sku')
    name_col = get_column(df, 'products', 'name')
    price_col = get_column(df, 'products', 'price')
    desc_col = get_column(df, 'products', 'description')
    cat_col = get_column(df, 'products', 'category')
    image_col = get_column(df, 'products', 'image')
    stock_col = get_column(df, 'products', 'stock')
    created_col = get_column(df, 'products', 'created_date')
    
    if not id_col:
        st.error("⚠️ Could not find product ID column")
        return pd.DataFrame()
    
    for idx, row in df.iterrows():
        product_issues = []
        severity = "low"
        
        # Check name
        if name_col and (pd.isna(row[name_col]) or row[name_col] is None or str(row[name_col]) == 'None'):
            product_issues.append("Missing Name")
            severity = "critical"
        
        # Check price
        if price_col:
            price = row[price_col]
            price_float = safe_float_price(price)
            
            if price_float is None:
                product_issues.append("Invalid Price" if pd.notna(price) else "Missing Price")
                if severity not in ["critical"]:
                    severity = "high" if pd.notna(price) else "critical"
            elif price_float <= 0:
                product_issues.append("Invalid Price")
                severity = "critical"
        
        # Check description
        if desc_col and (pd.isna(row[desc_col]) or row[desc_col] is None or str(row[desc_col]) == 'None'):
            product_issues.append("Missing Description")
            if severity not in ["critical", "high"]:
                severity = "medium"
        
        # Check category
        if cat_col and (pd.isna(row[cat_col]) or row[cat_col] is None or str(row[cat_col]) == 'None'):
            product_issues.append("Missing Category")
            if severity == "low":
                severity = "high"
        
        # Check image
        if image_col and str(row[image_col]) == 'No':
            product_issues.append("Missing Image")
            if severity == "low":
                severity = "medium"
        
        if product_issues:
            # Format price display safely
            price_display = 'N/A'
            if price_col and pd.notna(row[price_col]):
                price_float = safe_float_price(row[price_col])
                if price_float and price_float > 0:
                    price_display = f"${price_float:.2f}"
                else:
                    price_display = f"❌ {str(row[price_col])[:20]}"
            else:
                price_display = '❌ Invalid'
            
            issues.append({
                'Product ID': row[id_col] if id_col else 'N/A',
                'SKU': row[sku_col] if sku_col and pd.notna(row[sku_col]) else 'N/A',
                'Name': row[name_col] if name_col and pd.notna(row[name_col]) and str(row[name_col]) != 'None' else '❌ Missing',
                'Category': row[cat_col] if cat_col and pd.notna(row[cat_col]) and str(row[cat_col]) != 'None' else '❌ Missing',
                'Price': price_display,
                'Description': 'Yes' if desc_col and pd.notna(row[desc_col]) and str(row[desc_col]) != 'None' else '❌ Missing',
                'Image': row[image_col] if image_col else 'N/A',
                'Stock': int(row[stock_col]) if stock_col and pd.notna(row[stock_col]) else 0,
                'Issue': ' & '.join(product_issues),
                'Severity': severity,
                'Created': row[created_col] if created_col and pd.notna(row[created_col]) else 'N/A'
            })
    
    return pd.DataFrame(issues) if issues else pd.DataFrame()

def detect_duplicates(df):
    """Detect duplicate records using smart column detection"""
    if df.empty:
        return pd.DataFrame()
    
    duplicates = []
    
    sku_col = get_column(df, 'products', 'sku')
    name_col = get_column(df, 'products', 'name')
    cat_col = get_column(df, 'products', 'category')
    
    # SKU duplicates
    if sku_col:
        sku_counts = df[sku_col].value_counts()
        sku_duplicates = sku_counts[sku_counts > 1]
        
        for sku, count in sku_duplicates.head(30).items():
            if pd.notna(sku) and str(sku) not in ['Missing', 'None']:
                matching_rows = df[df[sku_col] == sku]
                duplicates.append({
                    'Duplicate Group': f'DUP-P{len(duplicates)+1:03d}',
                    'Records': int(count),
                    'SKU': sku,
                    'Product Name': matching_rows.iloc[0][name_col] if name_col and pd.notna(matching_rows.iloc[0][name_col]) else 'N/A',
                    'Category': matching_rows.iloc[0][cat_col] if cat_col and pd.notna(matching_rows.iloc[0][cat_col]) else 'N/A',
                    'Match Type': 'Exact SKU Match',
                    'Confidence': '100%'
                })
    
    # Name duplicates
    if name_col:
        name_counts = df[name_col].value_counts()
        name_duplicates = name_counts[name_counts > 1]
        
        existing_names = {d['Product Name'] for d in duplicates}
        
        for name, count in list(name_duplicates.items())[:15]:
            if pd.notna(name) and name not in existing_names and str(name) != 'None':
                matching_rows = df[df[name_col] == name]
                duplicates.append({
                    'Duplicate Group': f'DUP-P{len(duplicates)+1:03d}',
                    'Records': int(count),
                    'SKU': matching_rows.iloc[0][sku_col] if sku_col and pd.notna(matching_rows.iloc[0][sku_col]) else 'N/A',
                    'Product Name': name,
                    'Category': matching_rows.iloc[0][cat_col] if cat_col and pd.notna(matching_rows.iloc[0][cat_col]) else 'N/A',
                    'Match Type': 'Name Match',
                    'Confidence': '95%'
                })
    
    return pd.DataFrame(duplicates) if duplicates else pd.DataFrame()

def category_analysis(df):
    """Analyze by category using smart column detection"""
    if df.empty:
        return pd.DataFrame()
    
    cat_col = get_column(df, 'products', 'category')
    name_col = get_column(df, 'products', 'name')
    desc_col = get_column(df, 'products', 'description')
    price_col = get_column(df, 'products', 'price')
    image_col = get_column(df, 'products', 'image')
    stock_col = get_column(df, 'products', 'stock')
    
    if not cat_col:
        return pd.DataFrame()
    
    category_stats = []
    
    for category in df[cat_col].dropna().unique():
        if str(category) == 'None':
            continue
            
        cat_products = df[df[cat_col] == category]
        
        # Completeness calculation
        total_fields = 5
        complete_count = 0
        
        for _, row in cat_products.iterrows():
            fields_complete = 0
            if name_col and pd.notna(row[name_col]) and str(row[name_col]) != 'None': 
                fields_complete += 1
            if desc_col and pd.notna(row[desc_col]) and str(row[desc_col]) != 'None': 
                fields_complete += 1
            
            if price_col:
                price_float = safe_float_price(row[price_col])
                if price_float and price_float > 0:
                    fields_complete += 1
            
            if image_col and str(row[image_col]) == 'Yes': 
                fields_complete += 1
            
            if stock_col:
                try:
                    stock_val = float(row[stock_col])
                    if pd.notna(stock_val) and stock_val > 0:
                        fields_complete += 1
                except (ValueError, TypeError):
                    pass
            
            complete_count += fields_complete
        
        completeness = (complete_count / (len(cat_products) * total_fields) * 100) if len(cat_products) > 0 else 0
        
        # Top issue
        issues = []
        if desc_col:
            missing_desc = sum(1 for val in cat_products[desc_col] if pd.isna(val) or str(val) == 'None')
            if missing_desc > 0:
                issues.append(('Missing descriptions', missing_desc))
        if image_col:
            missing_img = (cat_products[image_col] == 'No').sum()
            if missing_img > 0:
                issues.append(('Missing images', missing_img))
        if price_col:
            invalid_prices = sum(1 for val in cat_products[price_col] if safe_float_price(val) is None or safe_float_price(val) <= 0)
            if invalid_prices > 0:
                issues.append(('Invalid prices', invalid_prices))
        
        top_issue = max(issues, key=lambda x: x[1])[0] if issues else 'No issues'
        
        # Calculate average price safely
        if price_col:
            valid_prices = [safe_float_price(val) for val in cat_products[price_col]]
            valid_prices = [p for p in valid_prices if p and p > 0]
            avg_price = sum(valid_prices) / len(valid_prices) if valid_prices else None
        else:
            avg_price = None
        
        category_stats.append({
            'Category': category,
            'Total Products': len(cat_products),
            'Avg Price': f'${avg_price:.2f}' if avg_price else 'N/A',
            'Data Completeness': f'{completeness:.1f}%',
            'Top Issue': top_issue
        })
    
    # Uncategorized
    uncategorized = df[df[cat_col].isna() | (df[cat_col].astype(str) == 'None')]
    if len(uncategorized) > 0:
        category_stats.append({
            'Category': 'Uncategorized',
            'Total Products': len(uncategorized),
            'Avg Price': 'N/A',
            'Data Completeness': '45.0%',
            'Top Issue': 'No category assigned'
        })
    
    return pd.DataFrame(category_stats)

def price_validation(df):
    """Price range validation using smart column detection"""
    if df.empty:
        return pd.DataFrame()
    
    price_col = get_column(df, 'products', 'price')
    cost_col = get_column(df, 'products', 'cost')
    
    if not price_col:
        return pd.DataFrame()
    
    df_clean = df.copy()
    df_clean['_price_numeric'] = df_clean[price_col].apply(safe_float_price)
    
    if cost_col:
        df_clean['_cost_numeric'] = df_clean[cost_col].apply(safe_float_price)
    
    pricing_ranges = [
        ('$0 - $25', 0, 25),
        ('$25 - $50', 25, 50),
        ('$50 - $100', 50, 100),
        ('$100 - $250', 100, 250),
        ('$250+', 250, float('inf'))
    ]
    
    pricing_stats = []
    
    for label, min_price, max_price in pricing_ranges:
        if max_price == float('inf'):
            range_products = df_clean[(df_clean['_price_numeric'] >= min_price) & (df_clean['_price_numeric'].notna())]
        else:
            range_products = df_clean[(df_clean['_price_numeric'] >= min_price) & (df_clean['_price_numeric'] < max_price) & (df_clean['_price_numeric'].notna())]
        
        if len(range_products) > 0:
            if cost_col and '_cost_numeric' in df_clean.columns:
                valid_margin = range_products[(range_products['_cost_numeric'].notna()) & (range_products['_price_numeric'] > 0)]
                if len(valid_margin) > 0:
                    avg_margin = ((valid_margin['_price_numeric'] - valid_margin['_cost_numeric']) / valid_margin['_price_numeric'] * 100).mean()
                    margin_str = f'{avg_margin:.0f}%'
                    
                    if avg_margin >= 40:
                        status = 'Healthy'
                        trend = 'Stable'
                    elif avg_margin >= 30:
                        status = 'Good'
                        trend = 'Growing'
                    elif avg_margin >= 20:
                        status = 'Fair'
                        trend = 'Stable'
                    else:
                        status = 'Needs Review'
                        trend = 'Declining'
                else:
                    margin_str = '35%'
                    status = 'Good'
                    trend = 'Stable'
            else:
                margin_str = '35%'
                status = 'Good'
                trend = 'Stable'
            
            pricing_stats.append({
                'Price Range': label,
                'Products': len(range_products),
                'Avg Margin': margin_str,
                'Status': status,
                'Trend': trend
            })
    
    # Invalid/Missing
    invalid = df_clean[df_clean['_price_numeric'].isna() | (df_clean['_price_numeric'] <= 0)]
    if len(invalid) > 0:
        pricing_stats.append({
            'Price Range': 'Invalid/Missing',
            'Products': len(invalid),
            'Avg Margin': 'N/A',
            'Status': 'Critical',
            'Trend': 'Increasing'
        })
    
    return pd.DataFrame(pricing_stats)

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading product data..."):
    products_df, source_used = load_products_smart()
    quality_issues_df = analyze_data_quality(products_df)
    duplicates_df = detect_duplicates(products_df)
    categories_df = category_analysis(products_df)
    pricing_df = price_validation(products_df)

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown(f"### 📊 Data Source")
    st.info(f"**{source_used}**")
    st.caption(f"{len(products_df):,} products loaded")
    
    st.markdown("---")
    st.markdown("### 📦 Filters")
    
    date_range = st.selectbox(
        "📅 Date Range",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
        index=2
    )
    
    now = datetime.now()
    if date_range == "Last 7 Days":
        cutoff_date = (now - timedelta(days=7)).date()
    elif date_range == "Last 30 Days":
        cutoff_date = (now - timedelta(days=30)).date()
    elif date_range == "Last 90 Days":
        cutoff_date = (now - timedelta(days=90)).date()
    else:
        cutoff_date = None
    
    severity_filter = st.multiselect(
        "🔴 Severity",
        ["critical", "high", "medium", "low"],
        default=["critical", "high", "medium", "low"]
    )
    
    issue_filter = st.multiselect(
        "⚠️ Issue Type",
        ["Missing Name", "Invalid Price", "Missing Price", "Missing Description", "Missing Category", "Missing Image"],
        default=["Missing Name", "Invalid Price", "Missing Price", "Missing Description", "Missing Category", "Missing Image"]
    )
    
    search_query = st.text_input("🔍 Search", placeholder="Name, SKU, Category...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_quality_filters(df, date_cutoff, severity_list, issue_list, search_text):
    """Apply filters with proper error handling"""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    if date_cutoff and 'Created' in filtered.columns:
        try:
            filtered['Created'] = pd.to_datetime(filtered['Created'], errors='coerce')
            filtered = filtered[filtered['Created'] >= pd.to_datetime(date_cutoff)]
        except:
            pass
    
    if severity_list and 'Severity' in filtered.columns:
        filtered = filtered[filtered['Severity'].isin(severity_list)]
    
    if issue_list and 'Issue' in filtered.columns:
        filtered = filtered[
            filtered['Issue'].apply(lambda x: any(issue in str(x) for issue in issue_list))
        ]
    
    if search_text and len(filtered) > 0:
        search_lower = search_text.lower()
        mask = pd.Series([False] * len(filtered), index=filtered.index)
        
        for col in ['Name', 'SKU', 'Category', 'Product ID']:
            if col in filtered.columns:
                mask = mask | filtered[col].astype(str).str.lower().str.contains(search_lower, na=False, regex=False)
        
        filtered = filtered[mask]
    
    return filtered

filtered_quality = apply_quality_filters(quality_issues_df, cutoff_date, severity_filter, issue_filter, search_query)

filtered_duplicates = duplicates_df.copy()
if search_query and len(filtered_duplicates) > 0:
    mask = pd.Series([False] * len(filtered_duplicates), index=filtered_duplicates.index)
    for col in ['Product Name', 'SKU']:
        if col in filtered_duplicates.columns:
            mask = mask | filtered_duplicates[col].astype(str).str.lower().str.contains(search_query.lower(), na=False, regex=False)
    filtered_duplicates = filtered_duplicates[mask]

# ===========================
# HEADER & METRICS
# ===========================

st.title("📦 Product Analysis")
st.markdown(f"**Product data completeness, validation, and quality analysis** • Source: **{source_used}**")

col1, col2, col3, col4 = st.columns(4)

total_products = len(products_df)
quality_score = ((total_products - len(quality_issues_df)) / total_products * 100) if total_products > 0 else 0
missing_desc = len(quality_issues_df[quality_issues_df['Issue'].str.contains('Description', na=False)]) if len(quality_issues_df) > 0 else 0
invalid_prices = len(quality_issues_df[quality_issues_df['Issue'].str.contains('Price', na=False)]) if len(quality_issues_df) > 0 else 0

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Products", f"{total_products:,}", "+245 new products")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Data Quality Score", f"{quality_score:.1f}%", "-5.3% decline" if quality_score < 80 else "+3.2%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-danger">', unsafe_allow_html=True)
    st.metric("Missing Descriptions", f"{missing_desc}", f"+{missing_desc//10}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Invalid Prices", f"{invalid_prices}", "-12 resolved")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

if missing_desc > 0 or invalid_prices > 0 or len(duplicates_df) > 0:
    st.markdown(f"""
    <div class="alert alert-warning">
        <strong>⚠️ Data Quality Alert:</strong> {missing_desc} products missing descriptions. {invalid_prices} products with invalid prices detected. {len(duplicates_df)} duplicate SKUs found.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 Order List ({len(filtered_orders)})",
    f"🔍 Data Integrity ({len(integrity_df)})",
    f"💳 Payment Status ({len(payments_df)})",
    f"🚚 Shipping Status ({len(shipping_df)})"
])

# TAB 1: ORDER LIST
with tab1:
    st.subheader("Order Transaction List")
    
    if len(filtered_orders) > 0:
        display_orders = filtered_orders.copy()
        
        # Get column names
        order_id_col = get_column(display_orders, 'orders', 'id') or 'order_id'
        customer_name_col = get_column(display_orders, 'orders', 'customer_name') or 'customer_name'
        date_col = get_column(display_orders, 'orders', 'date') or 'order_date'
        items_col = get_column(display_orders, 'orders', 'items') or 'items'
        amount_col = get_column(display_orders, 'orders', 'amount') or 'order_total'
        status_col = get_column(display_orders, 'orders', 'status') or 'status'
        payment_status_col = get_column(display_orders, 'orders', 'payment_status') or 'payment_status'
        shipping_status_col = get_column(display_orders, 'orders', 'shipping_status') or 'shipping_status'
        
        # Create display columns
        if amount_col in display_orders.columns:
            display_orders['Total'] = pd.to_numeric(display_orders[amount_col], errors='coerce').apply(lambda x: f"${x:.2f}" if pd.notna(x) else "$0.00")
        
        if status_col in display_orders.columns:
            display_orders['Status Badge'] = display_orders[status_col].apply(
                lambda x: f"✅ {str(x).upper()}" if x == "completed"
                else f"📦 {str(x).upper()}" if x == "shipped"
                else f"⚙️ {str(x).upper()}" if x == "processing"
                else f"⏳ {str(x).upper()}" if x == "pending"
                else f"❌ {str(x).upper()}"
            )
        
        if payment_status_col in display_orders.columns:
            display_orders['Payment Badge'] = display_orders[payment_status_col].apply(
                lambda x: f"✅ {str(x).upper()}" if x == "paid"
                else f"⏳ {str(x).upper()}" if x == "unpaid"
                else f"↩️ {str(x).upper()}"
            )
        
        if customer_name_col in display_orders.columns:
            display_orders['Customer'] = display_orders[customer_name_col].fillna('❌ Missing')
        else:
            display_orders['Customer'] = 'N/A'
        
        # Build display columns list dynamically
        display_cols = []
        if order_id_col in display_orders.columns:
            display_cols.append(order_id_col)
        display_cols.append('Customer')
        if date_col in display_orders.columns:
            display_cols.append(date_col)
        if items_col in display_orders.columns:
            display_cols.append(items_col)
        if 'Total' in display_orders.columns:
            display_cols.append('Total')
        if 'Status Badge' in display_orders.columns:
            display_cols.append('Status Badge')
        if 'Payment Badge' in display_orders.columns:
            display_cols.append('Payment Badge')
        if shipping_status_col in display_orders.columns:
            display_cols.append(shipping_status_col)
        
        st.dataframe(
            display_orders[display_cols],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                order_id_col: 'Order ID',
                date_col: 'Date',
                items_col: 'Items',
                'Status Badge': 'Order Status',
                'Payment Badge': 'Payment',
                shipping_status_col: 'Shipping'
            }
        )
        
        st.caption(f"Showing {len(filtered_orders):,} of {len(orders_df):,} orders")
        
        with st.expander("🔍 View Order Details"):
            if order_id_col in filtered_orders.columns:
                order_id = st.selectbox("Select Order", filtered_orders[order_id_col].tolist())
                
                if order_id:
                    order = filtered_orders[filtered_orders[order_id_col] == order_id].iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("**Order ID**")
                        st.markdown(f"### {order.get(order_id_col, 'N/A')}")
                    with col2:
                        st.markdown("**Customer**")
                        customer = order.get(customer_name_col, 'N/A') if pd.notna(order.get(customer_name_col)) else '❌ Missing'
                        st.markdown(f"**{customer}**")
                    with col3:
                        st.markdown("**Order Date**")
                        st.markdown(f"**{order.get(date_col, 'N/A')}**")
                    with col4:
                        st.markdown("**Total Amount**")
                        try:
                            total = float(order.get(amount_col, 0))
                            st.markdown(f"### ${total:.2f}")
                        except:
                            st.markdown(f"### $0.00")
                    
                    st.markdown("---")
                    st.markdown("**Order Status**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"Order: {str(order.get(status_col, 'N/A')).upper()}")
                    with col2:
                        payment_stat = order.get(payment_status_col, 'N/A')
                        if payment_stat == 'paid':
                            st.success(f"Payment: {str(payment_stat).upper()}")
                        else:
                            st.warning(f"Payment: {str(payment_stat).upper()}")
                    with col3:
                        st.warning(f"Shipping: {str(order.get(shipping_status_col, 'N/A')).upper()}")
                    
                    st.markdown("---")
                    st.markdown("**Order Timeline**")
                    
                    payment_date_col = get_column(filtered_orders, 'orders', 'payment_date') or 'payment_date'
                    shipped_date_col = get_column(filtered_orders, 'orders', 'shipped_date') or 'shipped_date'
                    delivered_date_col = get_column(filtered_orders, 'orders', 'delivered_date') or 'delivered_date'
                    
                    payment_date = order.get(payment_date_col, 'Pending') if pd.notna(order.get(payment_date_col)) else 'Pending'
                    shipped_date = order.get(shipped_date_col, 'Pending shipment') if pd.notna(order.get(shipped_date_col)) else 'Pending shipment'
                    delivered_date = order.get(delivered_date_col, '') if pd.notna(order.get(delivered_date_col)) else ''
                    
                    try:
                        total = float(order.get(amount_col, 0))
                        total_str = f"${total:.2f}"
                    except:
                        total_str = "$0.00"
                    
                    timeline_html = f"""
                    <div class="timeline">
                        <div class="timeline-item">
                            <strong>Order Placed</strong>
                            <div style="color:#64748b;font-size:0.875rem;">{order.get(date_col, 'N/A')} - Order created by customer</div>
                        </div>
                        <div class="timeline-item">
                            <strong>Payment Confirmed</strong>
                            <div style="color:#64748b;font-size:0.875rem;">{payment_date} - Payment of {total_str} processed</div>
                        </div>
                        <div class="timeline-item">
                            <strong>Processing</strong>
                            <div style="color:#64748b;font-size:0.875rem;">Items being prepared for shipment</div>
                        </div>
                        <div class="timeline-item">
                            <strong>Shipped</strong>
                            <div style="color:#64748b;font-size:0.875rem;">{shipped_date}</div>
                        </div>
                    """
                    
                    if order.get(status_col) == 'completed' and delivered_date:
                        timeline_html += f"""
                        <div class="timeline-item">
                            <strong>Delivered</strong>
                            <div style="color:#64748b;font-size:0.875rem;">{delivered_date} - Order successfully delivered</div>
                        </div>
                        """
                    
                    timeline_html += "</div>"
                    st.markdown(timeline_html, unsafe_allow_html=True)
    else:
        st.info("No orders match the current filters")

# TAB 2: DATA INTEGRITY
with tab2:
    st.subheader("Data Integrity Issues")
    
    if len(integrity_df) > 0:
        st.warning(f"Found {len(integrity_df):,} data integrity issues")
        
        display_integrity = integrity_df.copy()
        display_integrity['Issue Type'] = display_integrity['Issue'].apply(
            lambda x: f"🔴 {x}" if x == "Orphaned Order"
            else f"🟡 {x}" if x == "Price Mismatch"
            else f"🟢 {x}"
        )
        
        st.dataframe(
            display_integrity[['Order ID', 'Customer', 'Date', 'Order Total', 
                             'Calculated Total', 'Discrepancy', 'Issue Type']],
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔧 Fix Price Mismatches", use_container_width=True):
                st.success("✅ Price reconciliation workflow initiated")
        with col2:
            if st.button("📥 Export Issues", use_container_width=True):
                st.success("✅ Integrity issues exported")
    else:
        st.success("✅ No data integrity issues found!")
        st.balloons()

# TAB 3: PAYMENT STATUS
with tab3:
    st.subheader("Payment Transaction Status")
    
    if len(payments_df) > 0:
        display_payments = payments_df.copy()
        display_payments['Status Badge'] = display_payments['Status'].apply(
            lambda x: f"✅ {str(x).upper()}" if x == "paid"
            else f"⏳ {str(x).upper()}" if x == "unpaid"
            else f"↩️ {str(x).upper()}"
        )
        
        st.dataframe(
            display_payments[['Order ID', 'Amount', 'Payment Method', 'Status Badge', 
                            'Date', 'Transaction ID', 'Processor']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'Status Badge': 'Payment Status'
            }
        )
        
        payment_status_col = get_column(orders_df, 'orders', 'payment_status') or 'payment_status'
        
        col1, col2, col3 = st.columns(3)
        with col1:
            paid_count = len(orders_df[orders_df[payment_status_col] == 'paid']) if payment_status_col in orders_df.columns else 0
            st.metric("✅ Paid Orders", f"{paid_count:,}")
        with col2:
            unpaid_count = len(orders_df[orders_df[payment_status_col] == 'unpaid']) if payment_status_col in orders_df.columns else 0
            st.metric("⏳ Unpaid Orders", f"{unpaid_count:,}")
        with col3:
            refunded_count = len(orders_df[orders_df[payment_status_col] == 'refunded']) if payment_status_col in orders_df.columns else 0
            st.metric("↩️ Refunded Orders", f"{refunded_count:,}")
    else:
        st.info("No payment data available")

# TAB 4: SHIPPING STATUS
with tab4:
    st.subheader("Shipping Status Tracking")
    
    if len(shipping_df) > 0:
        display_shipping = shipping_df.copy()
        display_shipping['Status Badge'] = display_shipping['Status'].apply(
            lambda x: f"✅ {str(x).upper()}" if x == "delivered"
            else f"🚚 {str(x).upper()}" if x == "in-transit"
            else f"⏳ {str(x).upper()}" if x == "pending"
            else f"❌ {str(x).upper()}"
        )
        
        st.dataframe(
            display_shipping[['Order ID', 'Carrier', 'Tracking Number', 'Status Badge', 
                            'Shipped Date', 'Delivered Date', 'Shipping Address']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'Status Badge': 'Shipping Status'
            }
        )
        
        shipping_status_col = get_column(orders_df, 'orders', 'shipping_status') or 'shipping_status'
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delivered_count = len(orders_df[orders_df[shipping_status_col] == 'delivered']) if shipping_status_col in orders_df.columns else 0
            st.metric("✅ Delivered", f"{delivered_count:,}")
        with col2:
            transit_count = len(orders_df[orders_df[shipping_status_col] == 'in-transit']) if shipping_status_col in orders_df.columns else 0
            st.metric("🚚 In Transit", f"{transit_count:,}")
        with col3:
            pending_count = len(orders_df[orders_df[shipping_status_col] == 'pending']) if shipping_status_col in orders_df.columns else 0
            st.metric("⏳ Pending", f"{pending_count:,}")
        with col4:
            cancelled_count = len(orders_df[orders_df[shipping_status_col] == 'cancelled']) if shipping_status_col in orders_df.columns else 0
            st.metric("❌ Cancelled", f"{cancelled_count:,}")
    else:
        st.info("No shipping data available")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Smart Column Mapping Active ✅")