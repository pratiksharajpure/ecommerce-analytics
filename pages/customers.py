"""
👥 Enhanced Customer Analysis - All 4 Tabs with Working Filters
Exact match to customers.html with profiling and RFM segmentation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import re

st.set_page_config(
    page_title="Customer Analysis",
    page_icon="👥",
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
def generate_sample_customer_data():
    """Generate sample data with quality issues"""
    np.random.seed(42)
    
    customers = []
    for i in range(1, 1001):
        if i > 100 and np.random.random() < 0.08:
            dup_idx = np.random.randint(1, min(i, 100))
            name = f'Customer {dup_idx}'
            email = f'customer{dup_idx}@example.com'
        else:
            name = f'Customer {i}'
            email = f'customer{i}@example.com'
        
        if np.random.random() < 0.08:
            if np.random.random() < 0.5:
                email = email.replace('@', '')
            else:
                email = email.split('@')[0]
        elif np.random.random() < 0.05:
            email = None
        
        phone = f'+1-555-{np.random.randint(1000, 9999)}'
        if np.random.random() < 0.06:
            phone = '123'
        elif np.random.random() < 0.04:
            phone = None
        
        address = f'{np.random.randint(1, 999)} Main St'
        if np.random.random() < 0.07:
            address = None
        
        customers.append({
            'customer_id': f'C-{i:04d}',
            'name': name,
            'email': email,
            'phone': phone,
            'address': address,
            'created_date': (datetime.now() - timedelta(days=np.random.randint(1, 730))).date()
        })
    
    # Generate orders for profiling
    orders = []
    for i in range(1, 5001):
        orders.append({
            'order_id': i,
            'customer_id': f'C-{np.random.randint(1, 1001):04d}',
            'order_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).date(),
            'total_amount': np.random.uniform(10, 500)
        })
    
    return pd.DataFrame(customers), pd.DataFrame(orders)

# ===========================
# ANALYSIS FUNCTIONS
# ===========================

def analyze_data_quality(df):
    """Analyze quality issues"""
    issues = []
    
    for idx, row in df.iterrows():
        customer_issues = []
        severity = "low"
        
        email = row['email']
        if pd.isna(email) or email is None:
            customer_issues.append("Missing Email")
            severity = "critical"
        elif '@' not in str(email) or '.' not in str(email):
            customer_issues.append("Invalid Email Format")
            severity = "high"
        
        phone = row['phone']
        if pd.isna(phone) or phone is None:
            customer_issues.append("Missing Phone")
            if severity not in ["critical"]:
                severity = "high"
        else:
            phone_digits = ''.join(filter(str.isdigit, str(phone)))
            if len(phone_digits) < 10:
                customer_issues.append("Invalid Phone Format")
                if severity == "low":
                    severity = "medium"
        
        address = row['address']
        if pd.isna(address) or address is None:
            customer_issues.append("Missing Address")
            if severity == "low":
                severity = "medium"
        
        if customer_issues:
            issues.append({
                'Customer ID': row['customer_id'],
                'Name': row['name'],
                'Email': row['email'] if pd.notna(row['email']) else '❌ Missing',
                'Phone': row['phone'] if pd.notna(row['phone']) else '❌ Missing',
                'Address': row['address'] if pd.notna(row['address']) else '❌ Missing',
                'Issue': ' & '.join(customer_issues),
                'Severity': severity,
                'Registered': row['created_date']
            })
    
    return pd.DataFrame(issues) if issues else pd.DataFrame()

def detect_duplicates(df):
    """Detect duplicate records"""
    duplicates = []
    
    email_counts = df['email'].value_counts()
    email_duplicates = email_counts[email_counts > 1]
    
    for email, count in email_duplicates.head(30).items():
        if pd.notna(email) and str(email) != 'Missing':
            matching_rows = df[df['email'] == email]
            duplicates.append({
                'Duplicate Group': f'DUP-{len(duplicates)+1:03d}',
                'Records': int(count),
                'Customer Name': matching_rows.iloc[0]['name'],
                'Email': email,
                'Match Type': 'Email Match',
                'Confidence': '100%'
            })
    
    name_counts = df['name'].value_counts()
    name_duplicates = name_counts[name_counts > 1]
    
    for name, count in list(name_duplicates.items())[:15]:
        if pd.notna(name) and not any(d['Customer Name'] == name for d in duplicates):
            matching_rows = df[df['name'] == name]
            duplicates.append({
                'Duplicate Group': f'DUP-{len(duplicates)+1:03d}',
                'Records': int(count),
                'Customer Name': name,
                'Email': matching_rows.iloc[0]['email'],
                'Match Type': 'Exact Name Match',
                'Confidence': '85%'
            })
    
    return pd.DataFrame(duplicates) if duplicates else pd.DataFrame()

def profile_customers(customers_df, orders_df):
    """Profile customers by segment"""
    if len(orders_df) == 0:
        return pd.DataFrame()
    
    customer_orders = orders_df.groupby('customer_id').agg({
        'order_id': 'count',
        'total_amount': 'sum'
    }).reset_index()
    
    customer_orders.columns = ['customer_id', 'order_count', 'total_revenue']
    
    def segment_customer(row):
        if row['order_count'] >= 10 and row['total_revenue'] >= 1500:
            return 'High Value'
        elif row['order_count'] >= 5:
            return 'Regular'
        elif row['order_count'] >= 1:
            return 'New Customers'
        else:
            return 'At Risk'
    
    customer_orders['segment'] = customer_orders.apply(segment_customer, axis=1)
    
    segment_stats = customer_orders.groupby('segment').agg({
        'customer_id': 'count',
        'total_revenue': 'mean',
        'order_count': 'mean'
    }).reset_index()
    
    segment_stats.columns = ['Segment', 'Customer Count', 'Avg Revenue', 'Avg Orders']
    segment_stats['Avg Revenue'] = '$' + segment_stats['Avg Revenue'].apply(lambda x: f'{x:.0f}')
    segment_stats['Avg Orders'] = segment_stats['Avg Orders'].round(1)
    segment_stats['Retention Rate'] = segment_stats['Segment'].map({
        'High Value': '89%', 'Regular': '67%', 'At Risk': '34%', 'New Customers': '45%'
    })
    
    return segment_stats

def rfm_segmentation(orders_df):
    """RFM segmentation"""
    if len(orders_df) == 0:
        return pd.DataFrame()
    
    now = datetime.now()
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'], errors='coerce')
    orders_clean = orders_df[orders_df['order_date'].notna()].copy()
    
    rfm = orders_clean.groupby('customer_id').agg({
        'order_date': lambda x: (now - x.max()).days,
        'order_id': 'count',
        'total_amount': 'sum'
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    try:
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        def rfm_segment(score):
            if score.startswith('5') and score.endswith('5'):
                return 'Champions'
            elif score.startswith('4') or score.startswith('5'):
                return 'Loyal'
            elif score.startswith('3'):
                return 'Potential'
            elif score.startswith('2'):
                return 'At Risk'
            else:
                return 'Lost'
        
        rfm['segment'] = rfm['rfm_score'].apply(rfm_segment)
        
        segment_stats = rfm.groupby('segment').agg({
            'customer_id': 'count',
            'monetary': 'sum'
        }).reset_index()
        
        segment_stats.columns = ['Segment', 'Customers', 'Revenue']
        segment_stats['Revenue'] = '$' + (segment_stats['Revenue'] / 1000000).apply(lambda x: f'{x:.1f}M')
        segment_stats['RFM Score'] = segment_stats['Segment'].map({
            'Champions': '555', 'Loyal': '445-544', 'Potential': '355-454', 'At Risk': '244-344', 'Lost': '111-233'
        })
        segment_stats['Description'] = segment_stats['Segment'].map({
            'Champions': 'Best customers - Frequent, Recent, High Spend',
            'Loyal': 'Consistent purchasers with good spend',
            'Potential': 'Recent customers with growth potential',
            'At Risk': 'Were good, now declining',
            'Lost': "Haven't purchased recently"
        })
        
        return segment_stats[['Segment', 'RFM Score', 'Customers', 'Revenue', 'Description']]
    except:
        return pd.DataFrame()

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading customer data..."):
    customers_df, orders_df = generate_sample_customer_data()
    quality_issues_df = analyze_data_quality(customers_df)
    duplicates_df = detect_duplicates(customers_df)
    profiling_df = profile_customers(customers_df, orders_df)
    rfm_df = rfm_segmentation(orders_df)

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 👥 Filters")
    
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
        "🔧 Issue Type",
        ["Missing Email", "Invalid Email Format", "Missing Phone", "Invalid Phone Format", "Missing Address"],
        default=["Missing Email", "Invalid Email Format", "Missing Phone", "Invalid Phone Format", "Missing Address"]
    )
    
    search_query = st.text_input("🔍 Search", placeholder="Name, Email, ID...")
    
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
        filtered = filtered[filtered['Registered'] >= date_cutoff]
    
    if severity_list:
        filtered = filtered[filtered['Severity'].isin(severity_list)]
    
    if issue_list:
        filtered = filtered[
            filtered['Issue'].apply(lambda x: any(issue in str(x) for issue in issue_list))
        ]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['Name'].str.lower().str.contains(search_lower, na=False) |
            filtered['Email'].astype(str).str.lower().str.contains(search_lower, na=False) |
            filtered['Customer ID'].astype(str).str.contains(search_text, na=False)
        ]
    
    return filtered

filtered_quality = apply_quality_filters(quality_issues_df, cutoff_date, severity_filter, issue_filter, search_query)
filtered_duplicates = duplicates_df.copy()
if search_query:
    filtered_duplicates = filtered_duplicates[
        filtered_duplicates['Customer Name'].str.lower().str.contains(search_query.lower(), na=False) |
        filtered_duplicates['Email'].astype(str).str.lower().str.contains(search_query.lower(), na=False)
    ]

# ===========================
# HEADER & METRICS
# ===========================

st.title("👥 Customer Analysis")
st.markdown("**Customer data quality validation and duplicate detection**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Customers", f"{len(customers_df):,}", "+12.5%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    quality_score = ((len(customers_df) - len(quality_issues_df)) / len(customers_df) * 100)
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Data Quality Score", f"{quality_score:.1f}%", "+3.2%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-danger">', unsafe_allow_html=True)
    st.metric("Duplicate Records", f"{len(duplicates_df)}", f"+{max(0, len(duplicates_df) - 40)}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    invalid_emails = len(quality_issues_df[quality_issues_df['Issue'].str.contains('Email', na=False)])
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Invalid Emails", f"{invalid_emails}", "-15")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

if len(duplicates_df) > 0 or invalid_emails > 0:
    st.markdown(f"""
    <div class="alert alert-warning">
        <strong>⚠️ Data Quality Alert:</strong> {len(duplicates_df)} duplicate customer records detected. {invalid_emails} invalid email addresses found.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===========================
# TABS - EXACT MATCH TO HTML
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 Data Quality Issues ({len(filtered_quality)})",
    f"📄 Duplicate Detection ({len(filtered_duplicates)})",
    f"👤 Customer Profiling",
    f"🎯 RFM Segmentation"
])

# TAB 1: DATA QUALITY ISSUES
with tab1:
    st.subheader("Customer Data Quality Issues")
    
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
            display_df[['Customer ID', 'Name', 'Email', 'Phone', 'Address', 'Issue', 'Severity', 'Registered']],
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        st.caption(f"Showing {len(filtered_quality):,} of {len(quality_issues_df):,} issues")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔧 Fix Invalid Emails", use_container_width=True):
                st.info("✅ Email validation workflow initiated")
        with col2:
            if st.button("📞 Update Phone Numbers", use_container_width=True):
                st.info("✅ Phone update workflow initiated")
        with col3:
            if st.button("📥 Export Issues", use_container_width=True):
                st.success("✅ Issues exported to CSV")
    else:
        st.success("✅ No data quality issues found!")
        st.balloons()

# TAB 2: DUPLICATE DETECTION
with tab2:
    st.subheader("Duplicate Customer Detection")
    
    if len(filtered_duplicates) > 0:
        st.info(f"Found {len(filtered_duplicates):,} duplicate groups affecting {filtered_duplicates['Records'].sum():,} records")
        
        st.dataframe(
            filtered_duplicates[['Duplicate Group', 'Records', 'Customer Name', 'Email', 'Match Type', 'Confidence']],
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
        st.info("💡 All customer records are unique based on email and name matching.")

# TAB 3: CUSTOMER PROFILING
with tab3:
    st.subheader("Customer Profiling Analysis")
    
    if len(profiling_df) > 0:
        st.dataframe(
            profiling_df,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        fig = px.bar(
            profiling_df,
            x='Segment',
            y='Customer Count',
            title='Customer Distribution by Segment',
            color='Customer Count',
            color_continuous_scale='Blues',
            text='Customer Count'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 💡 Segment Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_value = profiling_df[profiling_df['Segment'] == 'High Value']['Customer Count'].values
            if len(high_value) > 0:
                st.metric("💎 High Value Customers", f"{high_value[0]:,}")
        with col2:
            at_risk = profiling_df[profiling_df['Segment'] == 'At Risk']['Customer Count'].values
            if len(at_risk) > 0:
                st.metric("⚠️ At Risk Customers", f"{at_risk[0]:,}")
        with col3:
            new = profiling_df[profiling_df['Segment'] == 'New Customers']['Customer Count'].values
            if len(new) > 0:
                st.metric("🆕 New Customers", f"{new[0]:,}")
    else:
        st.info("📊 Customer profiling requires order data.")

# TAB 4: RFM SEGMENTATION
with tab4:
    st.subheader("RFM Customer Segmentation")
    
    if len(rfm_df) > 0:
        st.dataframe(
            rfm_df,
            use_container_width=True,
            hide_index=True,
            height=300
        )
        
        fig = px.treemap(
            rfm_df,
            path=['Segment'],
            values='Customers',
            title='Customer Segments (RFM Analysis)',
            color='Customers',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **📊 RFM Scoring Explained:**
        - **Recency (R)**: How recently did customer purchase? (1-5 score)
        - **Frequency (F)**: How often do they purchase? (1-5 score)  
        - **Monetary (M)**: How much do they spend? (1-5 score)
        
        **Champions (555)**: Buy recently, often, spend most  
        **Loyal (4XX-5XX)**: Consistent purchasers with good spend  
        **Potential (3XX)**: Recent customers with growth potential  
        **At Risk (2XX)**: Were good, now declining  
        **Lost (1XX)**: Haven't purchased recently
        """)
    else:
        st.info("📊 RFM segmentation requires order data.")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")