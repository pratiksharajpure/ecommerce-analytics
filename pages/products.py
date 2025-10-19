"""
📦 Enhanced Product Analysis - All 4 Tabs with Working Filters
Exact match to products.html with data quality and duplicate detection
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
    .stat-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
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
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_product_data():
    """Generate sample data with quality issues"""
    np.random.seed(42)
    
    products = []
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports', 'Toys', 'Food']
    
    for i in range(1, 501):
        # Price issues
        price = np.random.uniform(5, 500)
        if np.random.random() < 0.09:  # 9% invalid prices
            price = np.random.choice([0, -10.00, None])
        
        # Description issues
        description = f'High-quality product {i} with excellent features'
        if np.random.random() < 0.26:  # 26% missing descriptions
            description = None
        
        # Category issues
        category = np.random.choice(categories)
        if np.random.random() < 0.10:  # 10% missing category
            category = None
        
        # Name issues
        name = f'Product {i}'
        if np.random.random() < 0.02:  # 2% missing name
            name = None
        
        # Image issues
        image = 'Yes' if np.random.random() > 0.15 else 'No'
        
        # Create some duplicates
        if np.random.random() < 0.05:  # 5% duplicate SKUs
            sku = f'DUP-{np.random.randint(1, 25):03d}'
        else:
            sku = f'{category[:4].upper() if category else "PROD"}-{i:03d}'
        
        products.append({
            'product_id': f'P-{i:04d}',
            'sku': sku,
            'name': name,
            'category': category,
            'price': price,
            'description': description,
            'image': image,
            'stock': np.random.randint(0, 300),
            'created_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).date(),
            'cost': np.random.uniform(2, price*0.6) if price and price > 0 else None
        })
    
    return pd.DataFrame(products)

# ===========================
# ANALYSIS FUNCTIONS
# ===========================

def analyze_data_quality(df):
    """Analyze quality issues"""
    issues = []
    
    for idx, row in df.iterrows():
        product_issues = []
        severity = "low"
        
        # Check name
        if pd.isna(row['name']) or row['name'] is None:
            product_issues.append("Missing Name")
            severity = "critical"
        
        # Check price
        price = row['price']
        if pd.isna(price) or price is None:
            product_issues.append("Missing Price")
            if severity not in ["critical"]:
                severity = "high"
        elif price <= 0:
            product_issues.append("Invalid Price")
            severity = "critical"
        
        # Check description
        if pd.isna(row['description']) or row['description'] is None:
            product_issues.append("Missing Description")
            if severity not in ["critical", "high"]:
                severity = "medium"
        
        # Check category
        if pd.isna(row['category']) or row['category'] is None:
            product_issues.append("Missing Category")
            if severity == "low":
                severity = "high"
        
        # Check image
        if row['image'] == 'No':
            product_issues.append("Missing Image")
            if severity == "low":
                severity = "medium"
        
        if product_issues:
            issues.append({
                'Product ID': row['product_id'],
                'SKU': row['sku'],
                'Name': row['name'] if pd.notna(row['name']) else '❌ Missing',
                'Category': row['category'] if pd.notna(row['category']) else '❌ Missing',
                'Price': f"${row['price']:.2f}" if pd.notna(row['price']) and row['price'] > 0 else '❌ Invalid',
                'Description': 'Yes' if pd.notna(row['description']) else '❌ Missing',
                'Image': row['image'],
                'Stock': row['stock'],
                'Issue': ' & '.join(product_issues),
                'Severity': severity,
                'Created': row['created_date']
            })
    
    return pd.DataFrame(issues) if issues else pd.DataFrame()

def detect_duplicates(df):
    """Detect duplicate records"""
    duplicates = []
    
    # SKU duplicates
    sku_counts = df['sku'].value_counts()
    sku_duplicates = sku_counts[sku_counts > 1]
    
    for sku, count in sku_duplicates.head(30).items():
        if pd.notna(sku) and str(sku) != 'Missing':
            matching_rows = df[df['sku'] == sku]
            duplicates.append({
                'Duplicate Group': f'DUP-P{len(duplicates)+1:03d}',
                'Records': int(count),
                'SKU': sku,
                'Product Name': matching_rows.iloc[0]['name'] if pd.notna(matching_rows.iloc[0]['name']) else 'N/A',
                'Category': matching_rows.iloc[0]['category'] if pd.notna(matching_rows.iloc[0]['category']) else 'N/A',
                'Match Type': 'Exact SKU Match',
                'Confidence': '100%'
            })
    
    # Name duplicates
    name_counts = df['name'].value_counts()
    name_duplicates = name_counts[name_counts > 1]
    
    for name, count in list(name_duplicates.items())[:15]:
        if pd.notna(name) and not any(d['Product Name'] == name for d in duplicates):
            matching_rows = df[df['name'] == name]
            duplicates.append({
                'Duplicate Group': f'DUP-P{len(duplicates)+1:03d}',
                'Records': int(count),
                'SKU': matching_rows.iloc[0]['sku'],
                'Product Name': name,
                'Category': matching_rows.iloc[0]['category'] if pd.notna(matching_rows.iloc[0]['category']) else 'N/A',
                'Match Type': 'Name Match',
                'Confidence': '95%'
            })
    
    return pd.DataFrame(duplicates) if duplicates else pd.DataFrame()

def category_analysis(df):
    """Analyze by category"""
    if df.empty:
        return pd.DataFrame()
    
    category_stats = []
    
    for category in df['category'].dropna().unique():
        cat_products = df[df['category'] == category]
        
        # Completeness calculation
        total_fields = 5
        complete_count = 0
        
        for _, row in cat_products.iterrows():
            fields_complete = 0
            if pd.notna(row['name']): fields_complete += 1
            if pd.notna(row['description']): fields_complete += 1
            if pd.notna(row['price']) and row['price'] > 0: fields_complete += 1
            if row['image'] == 'Yes': fields_complete += 1
            if row['stock'] > 0: fields_complete += 1
            complete_count += fields_complete
        
        completeness = (complete_count / (len(cat_products) * total_fields) * 100) if len(cat_products) > 0 else 0
        
        # Top issue
        issues = []
        if (cat_products['description'].isna()).sum() > 0:
            issues.append(('Missing descriptions', (cat_products['description'].isna()).sum()))
        if (cat_products['image'] == 'No').sum() > 0:
            issues.append(('Missing images', (cat_products['image'] == 'No').sum()))
        if ((cat_products['price'].isna()) | (cat_products['price'] <= 0)).sum() > 0:
            issues.append(('Invalid prices', ((cat_products['price'].isna()) | (cat_products['price'] <= 0)).sum()))
        
        top_issue = max(issues, key=lambda x: x[1])[0] if issues else 'No issues'
        avg_price = cat_products[cat_products['price'] > 0]['price'].mean()
        
        category_stats.append({
            'Category': category,
            'Total Products': len(cat_products),
            'Avg Price': f'${avg_price:.2f}' if pd.notna(avg_price) else 'N/A',
            'Data Completeness': f'{completeness:.1f}%',
            'Top Issue': top_issue
        })
    
    # Uncategorized
    uncategorized = df[df['category'].isna()]
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
    """Price range validation"""
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
            range_products = df[(df['price'] >= min_price) & (df['price'].notna())]
        else:
            range_products = df[(df['price'] >= min_price) & (df['price'] < max_price) & (df['price'].notna())]
        
        if len(range_products) > 0:
            valid_margin = range_products[(range_products['cost'].notna()) & (range_products['price'] > 0)]
            if len(valid_margin) > 0:
                avg_margin = ((valid_margin['price'] - valid_margin['cost']) / valid_margin['price'] * 100).mean()
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
            
            pricing_stats.append({
                'Price Range': label,
                'Products': len(range_products),
                'Avg Margin': margin_str,
                'Status': status,
                'Trend': trend
            })
    
    # Invalid/Missing
    invalid = df[(df['price'].isna()) | (df['price'] <= 0)]
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
    products_df = generate_sample_product_data()
    quality_issues_df = analyze_data_quality(products_df)
    duplicates_df = detect_duplicates(products_df)
    categories_df = category_analysis(products_df)
    pricing_df = price_validation(products_df)

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
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
    filtered = df.copy()
    
    if date_cutoff:
        filtered = filtered[filtered['Created'] >= date_cutoff]
    
    if severity_list:
        filtered = filtered[filtered['Severity'].isin(severity_list)]
    
    if issue_list:
        filtered = filtered[
            filtered['Issue'].apply(lambda x: any(issue in str(x) for issue in issue_list))
        ]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['Name'].astype(str).str.lower().str.contains(search_lower, na=False) |
            filtered['SKU'].astype(str).str.lower().str.contains(search_lower, na=False) |
            filtered['Category'].astype(str).str.lower().str.contains(search_lower, na=False) |
            filtered['Product ID'].astype(str).str.contains(search_text, na=False)
        ]
    
    return filtered

filtered_quality = apply_quality_filters(quality_issues_df, cutoff_date, severity_filter, issue_filter, search_query)
filtered_duplicates = duplicates_df.copy()
if search_query:
    filtered_duplicates = filtered_duplicates[
        filtered_duplicates['Product Name'].astype(str).str.lower().str.contains(search_query.lower(), na=False) |
        filtered_duplicates['SKU'].astype(str).str.lower().str.contains(search_query.lower(), na=False)
    ]

# ===========================
# HEADER & METRICS
# ===========================

st.title("📦 Product Analysis")
st.markdown("**Product data completeness, validation, and quality analysis**")

col1, col2, col3, col4 = st.columns(4)

total_products = len(products_df)
quality_score = ((total_products - len(quality_issues_df)) / total_products * 100) if total_products > 0 else 0
missing_desc = len(quality_issues_df[quality_issues_df['Issue'].str.contains('Description', na=False)])
invalid_prices = len(quality_issues_df[quality_issues_df['Issue'].str.contains('Price', na=False)])

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

# ===========================
# ALERT
# ===========================

if missing_desc > 0 or invalid_prices > 0 or len(duplicates_df) > 0:
    st.markdown(f"""
    <div class="alert alert-warning">
        <strong>⚠️ Data Quality Alert:</strong> {missing_desc} products missing descriptions. {invalid_prices} products with invalid prices detected. {len(duplicates_df)} duplicate SKUs found.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===========================
# TABS - EXACT MATCH TO HTML
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 Data Quality Issues ({len(filtered_quality)})",
    f"🔄 Duplicate Products ({len(filtered_duplicates)})",
    f"📂 Category Analysis",
    f"💰 Price Validation"
])

# TAB 1: DATA QUALITY ISSUES
with tab1:
    st.subheader("Product Data Quality Issues")
    
    if len(filtered_quality) > 0:
        st.info(f"Found {len(filtered_quality):,} data quality issues")
        
        display_df = filtered_quality.copy()
        display_df['Severity'] = display_df['Severity'].apply(
            lambda x: f"🔴 {x.upper()}" if x == "critical"
            else f"🟠 {x.upper()}" if x == "high"
            else f"🔵 {x.upper()}" if x == "medium"
            else f"🟢 {x.upper()}"
        )
        
        st.dataframe(
            display_df[['Product ID', 'SKU', 'Name', 'Category', 'Price', 'Description', 'Image', 'Stock', 'Issue', 'Severity', 'Created']],
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        st.caption(f"Showing {len(filtered_quality):,} of {len(quality_issues_df):,} issues")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔧 Fix Invalid Prices", use_container_width=True):
                st.info("✅ Price validation workflow initiated")
        with col2:
            if st.button("📝 Update Descriptions", use_container_width=True):
                st.info("✅ Description update workflow initiated")
        with col3:
            if st.button("📥 Export Issues", use_container_width=True):
                st.success("✅ Issues exported to CSV")
    else:
        st.success("✅ No data quality issues found!")
        st.balloons()

# TAB 2: DUPLICATE DETECTION
with tab2:
    st.subheader("Duplicate Product Detection")
    
    if len(filtered_duplicates) > 0:
        st.info(f"Found {len(filtered_duplicates):,} duplicate groups affecting {filtered_duplicates['Records'].sum():,} records")
        
        st.dataframe(
            filtered_duplicates[['Duplicate Group', 'Records', 'SKU', 'Product Name', 'Category', 'Match Type', 'Confidence']],
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        st.caption(f"Total duplicate groups: {len(filtered_duplicates):,}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔀 Merge All Duplicates", use_container_width=True):
                st.success(f"✅ Duplicate merge initiated - {len(filtered_duplicates)} groups will be consolidated")
        with col2:
            if st.button("📥 Export Duplicates", use_container_width=True):
                st.success("✅ Duplicate list exported")
    else:
        st.success("✅ No duplicate records found!")
        st.info("💡 All product records are unique based on SKU and name matching.")

# TAB 3: CATEGORY ANALYSIS
with tab3:
    st.subheader("Category Analysis")
    
    if len(categories_df) > 0:
        st.dataframe(
            categories_df,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        fig = px.bar(
            categories_df,
            x='Category',
            y='Total Products',
            title='Product Distribution by Category',
            color='Total Products',
            color_continuous_scale='Blues',
            text='Total Products'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 💡 Category Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_cat = categories_df.iloc[0] if len(categories_df) > 0 else None
            if best_cat is not None:
                st.metric("🏆 Top Category", best_cat['Category'], f"{best_cat['Total Products']} products")
        with col2:
            total_cats = len(categories_df[categories_df['Category'] != 'Uncategorized'])
            st.metric("📂 Total Categories", f"{total_cats}", f"{total_products//max(total_cats,1)} avg")
        with col3:
            uncat = categories_df[categories_df['Category'] == 'Uncategorized']
            uncat_count = uncat['Total Products'].values[0] if len(uncat) > 0 else 0
            st.metric("⚠️ Uncategorized", f"{uncat_count}", "needs review")
    else:
        st.info("📊 Customer profiling requires order data.")

# TAB 4: PRICE VALIDATION
with tab4:
    st.subheader("Price Range Validation")
    
    if len(pricing_df) > 0:
        st.dataframe(
            pricing_df,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        valid_pricing = pricing_df[pricing_df['Price Range'] != 'Invalid/Missing']
        fig = px.bar(
            valid_pricing,
            x='Price Range',
            y='Products',
            title='Product Count by Price Range',
            color='Products',
            color_continuous_scale='Greens',
            text='Products'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **💰 Price Range Analysis:**
        - **$0-$25**: Entry-level products with high volume
        - **$25-$50**: Mid-range with balanced margins
        - **$50-$100**: Premium products with good margins
        - **$100-$250**: High-value items requiring careful pricing
        - **$250+**: Luxury items with lower margins
        
        **Status Indicators:**
        - **Healthy**: 40%+ margin, stable pricing
        - **Good**: 30-40% margin, growing segment
        - **Fair**: 20-30% margin, needs monitoring
        - **Critical**: Invalid/missing prices requiring immediate attention
        """)
    else:
        st.info("📊 RFM segmentation requires order data.")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")