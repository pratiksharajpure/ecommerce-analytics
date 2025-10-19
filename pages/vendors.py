"""
🏭 Enhanced Supplier Data Quality - All 4 Tabs with Working Filters
Exact match to vendors.html with vendor management, ratings, contracts, and performance tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Supplier Data Quality",
    page_icon="🏭",
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
    .alert-info {
        background: #dbeafe;
        color: #1e40af;
        border-left-color: #3b82f6;
    }
    .performance-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_vendor_data():
    """Generate sample vendor data with quality metrics"""
    np.random.seed(42)
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Food & Beverage', 'Sports', 'Beauty', 'Books', 'Toys']
    countries = ['USA', 'China', 'Germany', 'Japan', 'Italy', 'France', 'UK', 'South Korea']
    statuses = ['active', 'inactive', 'pending']
    
    # Vendor List
    vendors = []
    for i in range(1, 235):
        category = np.random.choice(categories)
        status = np.random.choice(statuses, p=[0.85, 0.10, 0.05])
        
        # Rating distribution
        if np.random.random() < 0.20:
            rating = np.random.uniform(4.5, 5.0)
        elif np.random.random() < 0.60:
            rating = np.random.uniform(3.5, 4.5)
        else:
            rating = np.random.uniform(2.0, 3.5)
        
        rating = round(rating, 1)
        base_performance = (rating / 5.0) * 100
        on_time_delivery = int(max(75, min(99, base_performance + np.random.uniform(-5, 5))))
        quality_score = int(max(80, min(99, base_performance + np.random.uniform(-3, 7))))
        total_orders = np.random.randint(100, 2500)
        response_time = round(np.random.uniform(1.5, 6.0), 1)
        
        contact = None if np.random.random() < 0.034 else f'contact{i}@vendor{i}.com'
        phone = None if np.random.random() < 0.02 else f'+1-555-{np.random.randint(1000, 9999)}'
        
        vendors.append({
            'id': f'VEN-{1000+i}',
            'name': f'Vendor {i} - {category[:4]}',
            'category': category,
            'status': status,
            'rating': rating,
            'total_orders': total_orders,
            'on_time_delivery': on_time_delivery,
            'quality_score': quality_score,
            'response_time': response_time,
            'contact': contact,
            'phone': phone,
            'country': np.random.choice(countries)
        })
    
    # Vendor Ratings
    ratings = []
    top_vendors = sorted(vendors, key=lambda x: x['rating'], reverse=True)[:50]
    
    for vendor in top_vendors:
        reviews = int(vendor['total_orders'] * np.random.uniform(0.05, 0.15))
        quality = vendor['quality_score']
        delivery = vendor['on_time_delivery']
        communication = int(max(75, min(99, (quality + delivery) / 2 + np.random.randint(-5, 5))))
        pricing = int(max(70, min(99, (quality + delivery) / 2 + np.random.randint(-8, 3))))
        
        last_review_days = np.random.randint(1, 30)
        last_review = (datetime.now() - timedelta(days=last_review_days)).strftime('%Y-%m-%d')
        
        if vendor['rating'] >= 4.5:
            trend = np.random.choice(['up', 'stable'], p=[0.3, 0.7])
        elif vendor['rating'] >= 3.5:
            trend = np.random.choice(['up', 'stable', 'down'], p=[0.2, 0.6, 0.2])
        else:
            trend = np.random.choice(['stable', 'down'], p=[0.3, 0.7])
        
        ratings.append({
            'vendor': vendor['name'],
            'rating': vendor['rating'],
            'reviews': reviews,
            'quality': quality,
            'delivery': delivery,
            'communication': communication,
            'pricing': pricing,
            'last_review': last_review,
            'trend': trend
        })
    
    # Contracts
    contracts = []
    contract_vendors = [v for v in vendors if v['status'] == 'active' and v['rating'] >= 3.5][:80]
    
    for i, vendor in enumerate(contract_vendors):
        start_date = datetime.now() - timedelta(days=int(np.random.randint(30, 730)))
        duration = int(np.random.choice([365, 730]))
        end_date = start_date + timedelta(days=duration)
        
        days_until_expiry = (end_date - datetime.now()).days
        if days_until_expiry < 0:
            status = 'expired'
        elif days_until_expiry < 30:
            status = 'pending'
        else:
            status = 'active'
        
        contract_value = np.random.randint(500000, 5000000)
        payment_terms = np.random.choice(['Net 15', 'Net 30', 'Net 45', 'Net 60'])
        minimum_order = int(contract_value * np.random.uniform(0.01, 0.05))
        delivery_terms = np.random.choice(['FOB', 'CIF', 'EXW', 'DDP'])
        exclusivity = np.random.choice(['Yes', 'No'], p=[0.15, 0.85])
        auto_renew = np.random.choice(['Yes', 'No'], p=[0.60, 0.40])
        
        contracts.append({
            'vendor': vendor['name'],
            'contract_id': f'CNT-{2024}-{i+1:03d}',
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'status': status,
            'value': contract_value,
            'payment_terms': payment_terms,
            'minimum_order': minimum_order,
            'delivery_terms': delivery_terms,
            'exclusivity': exclusivity,
            'auto_renew': auto_renew
        })
    
    # Performance Metrics
    performance = []
    top_performance_vendors = sorted(vendors, key=lambda x: x['rating'], reverse=True)[:30]
    
    for vendor in top_performance_vendors:
        delivery_rate = vendor['on_time_delivery']
        quality_rate = vendor['quality_score']
        base_perf = (delivery_rate + quality_rate) / 2
        
        return_rate = round(max(0.5, (100 - base_perf) / 10 + np.random.uniform(-1, 1)), 1)
        defect_rate = round(max(0.3, (100 - base_perf) / 12 + np.random.uniform(-0.5, 0.5)), 1)
        lead_time = int(max(5, min(30, 25 - (base_perf - 80) / 3 + np.random.randint(-3, 3))))
        fill_rate = int(max(80, min(99, base_perf + np.random.randint(-3, 3))))
        cost_variance = int(max(1, min(15, (100 - base_perf) / 8 + np.random.randint(-2, 2))))
        innovation_score = int(max(60, min(98, base_perf + np.random.randint(-10, 5))))
        
        performance.append({
            'vendor': vendor['name'],
            'delivery_rate': delivery_rate,
            'quality_rate': quality_rate,
            'return_rate': return_rate,
            'defect_rate': defect_rate,
            'lead_time': lead_time,
            'fill_rate': fill_rate,
            'cost_variance': cost_variance,
            'innovation_score': innovation_score
        })
    
    return (pd.DataFrame(vendors), pd.DataFrame(ratings), 
            pd.DataFrame(contracts), pd.DataFrame(performance))

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading vendor data..."):
    vendors_df, ratings_df, contracts_df, performance_df = generate_sample_vendor_data()

# Calculate metrics
total_vendors = len(vendors_df)
active_vendors = len(vendors_df[vendors_df['status'] == 'active'])
invalid_contacts = len(vendors_df[vendors_df['contact'].isna()])
missing_fields = invalid_contacts + len(vendors_df[vendors_df['phone'].isna()])
data_quality_score = ((total_vendors - missing_fields) / total_vendors * 100)
expiring_contracts = len(contracts_df[contracts_df['status'] == 'pending'])
avg_rating = vendors_df['rating'].mean()

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 🏭 Filters")
    
    status_filter = st.multiselect(
        "📊 Status",
        ["active", "inactive", "pending"],
        default=["active", "inactive", "pending"]
    )
    
    rating_filter = st.selectbox(
        "⭐ Rating",
        ["All Ratings", "5 Stars", "4+ Stars", "3+ Stars", "Below 3 Stars"],
        index=0
    )
    
    category_filter = st.multiselect(
        "📦 Category",
        vendors_df['category'].unique().tolist(),
        default=vendors_df['category'].unique().tolist()
    )
    
    country_filter = st.multiselect(
        "🌍 Country",
        vendors_df['country'].unique().tolist(),
        default=vendors_df['country'].unique().tolist()
    )
    
    search_query = st.text_input("🔍 Search", placeholder="Name, ID...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_vendor_filters(df, status_list, rating_value, category_list, country_list, search_text):
    filtered = df.copy()
    
    if status_list:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    if rating_value == "5 Stars":
        filtered = filtered[filtered['rating'] >= 4.9]
    elif rating_value == "4+ Stars":
        filtered = filtered[filtered['rating'] >= 4.0]
    elif rating_value == "3+ Stars":
        filtered = filtered[filtered['rating'] >= 3.0]
    elif rating_value == "Below 3 Stars":
        filtered = filtered[filtered['rating'] < 3.0]
    
    if category_list:
        filtered = filtered[filtered['category'].isin(category_list)]
    
    if country_list:
        filtered = filtered[filtered['country'].isin(country_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['name'].str.lower().str.contains(search_lower, na=False) |
            filtered['id'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_vendors = apply_vendor_filters(vendors_df, status_filter, rating_filter, 
                                        category_filter, country_filter, search_query)
filtered_ratings = ratings_df[ratings_df['vendor'].isin(filtered_vendors['name'])]
filtered_contracts = contracts_df[contracts_df['vendor'].isin(filtered_vendors['name'])]
filtered_performance = performance_df[performance_df['vendor'].isin(filtered_vendors['name'])]

# ===========================
# HEADER & METRICS
# ===========================

st.title("🏭 Supplier Data Quality")
st.markdown("**Vendor management, performance tracking, and contract monitoring**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Vendors", f"{total_vendors}", "+12 new vendors")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Data Quality Score", f"{data_quality_score:.1f}%", "+2.1% improvement")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Invalid Contacts", f"{invalid_contacts}", "-3 resolved")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Missing Data Fields", f"{missing_fields}", "+2 new issues")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown(f"""
<div class="alert alert-info">
    <strong>📊 Vendor Status:</strong> {active_vendors} active vendors. {expiring_contracts} contracts expiring within 30 days. Overall performance score: {avg_rating:.1f}/5.0
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vendor Performance Trend")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    trend_data = pd.DataFrame({
        'Month': months,
        'On-Time Delivery': [92, 93, 94, 95, 96, 96],
        'Quality Score': [94, 95, 95, 96, 97, 97]
    })
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['On-Time Delivery'],
                              mode='lines+markers', name='On-Time Delivery %',
                              line=dict(color='#22c55e', width=3)))
    fig1.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Quality Score'],
                              mode='lines+markers', name='Quality Score %',
                              line=dict(color='#3b82f6', width=3)))
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                       yaxis_title='Percentage (%)',
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Rating Distribution")
    
    rating_dist = pd.DataFrame({
        'Rating': ['5 Stars', '4-4.9 Stars', '3-3.9 Stars', '2-2.9 Stars', '1-1.9 Stars'],
        'Count': [45, 128, 48, 10, 3]
    })
    
    fig2 = px.bar(rating_dist, x='Rating', y='Count',
                  color='Count',
                  color_continuous_scale=['#ef4444', '#f59e0b', '#3b82f6', '#22c55e', '#22c55e'])
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                       yaxis_title='Number of Vendors', showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 Vendor List ({len(filtered_vendors)})",
    f"⭐ Vendor Ratings ({len(filtered_ratings)})",
    f"📄 Contract Details ({len(filtered_contracts)})",
    f"📊 Performance Metrics ({len(filtered_performance)})"
])

# TAB 1: VENDOR LIST
with tab1:
    st.subheader("Vendor Master List")
    
    if len(filtered_vendors) > 0:
        display_vendors = filtered_vendors.copy()
        display_vendors['Status Badge'] = display_vendors['status'].apply(
            lambda x: f"🟢 {x.upper()}" if x == "active" else f"⚪ {x.upper()}" if x == "inactive" else f"🟡 {x.upper()}"
        )
        display_vendors['Rating Display'] = display_vendors['rating'].apply(lambda x: f"⭐ {x:.1f}")
        display_vendors['Contact Display'] = display_vendors['contact'].apply(lambda x: x if pd.notna(x) else '❌ Missing')
        
        st.dataframe(
            display_vendors[['id', 'name', 'category', 'Status Badge', 'Rating Display',
                           'total_orders', 'on_time_delivery', 'quality_score', 
                           'response_time', 'Contact Display', 'country']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'id': 'Vendor ID', 'name': 'Vendor Name', 'category': 'Category',
                'Status Badge': 'Status', 'Rating Display': 'Rating',
                'total_orders': 'Total Orders', 'on_time_delivery': 'On-Time %',
                'quality_score': 'Quality Score', 'response_time': 'Response Time (h)',
                'Contact Display': 'Contact', 'country': 'Country'
            }
        )
        
        st.caption(f"Showing {len(filtered_vendors):,} of {len(vendors_df):,} vendors")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            active_filtered = len(filtered_vendors[filtered_vendors['status'] == 'active'])
            st.metric("🟢 Active Vendors", f"{active_filtered:,}")
        with col2:
            avg_rating_filtered = filtered_vendors['rating'].mean()
            st.metric("⭐ Avg Rating", f"{avg_rating_filtered:.2f}/5.0")
        with col3:
            avg_quality = filtered_vendors['quality_score'].mean()
            st.metric("📊 Avg Quality Score", f"{avg_quality:.1f}%")
        with col4:
            avg_delivery = filtered_vendors['on_time_delivery'].mean()
            st.metric("🚚 Avg On-Time Delivery", f"{avg_delivery:.1f}%")
        
        if st.button("📧 Contact All Filtered Vendors", use_container_width=True):
            st.success(f"✅ Email notification sent to {len(filtered_vendors)} vendors")
    else:
        st.info("No vendors match the current filters")

# TAB 2: VENDOR RATINGS
with tab2:
    st.subheader("Vendor Rating Breakdown")
    
    if len(filtered_ratings) > 0:
        display_ratings = filtered_ratings.copy()
        display_ratings['Rating Display'] = display_ratings['rating'].apply(lambda x: f"⭐ {x:.1f}")
        display_ratings['Trend Display'] = display_ratings['trend'].apply(
            lambda x: f"📈 {x.upper()}" if x == "up" else f"📉 {x.upper()}" if x == "down" else f"➡️ {x.upper()}"
        )
        
        st.dataframe(
            display_ratings[['vendor', 'Rating Display', 'reviews', 'quality', 'delivery',
                           'communication', 'pricing', 'last_review', 'Trend Display']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'vendor': 'Vendor', 'Rating Display': 'Overall Rating', 'reviews': 'Reviews',
                'quality': 'Quality %', 'delivery': 'Delivery %', 'communication': 'Communication %',
                'pricing': 'Pricing %', 'last_review': 'Last Review', 'Trend Display': 'Trend'
            }
        )
        
        st.caption(f"Showing {len(filtered_ratings):,} vendor ratings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            excellent = len(filtered_ratings[filtered_ratings['rating'] >= 4.5])
            st.metric("🌟 Excellent (4.5+)", f"{excellent}")
        with col2:
            good = len(filtered_ratings[(filtered_ratings['rating'] >= 3.5) & (filtered_ratings['rating'] < 4.5)])
            st.metric("⭐ Good (3.5-4.5)", f"{good}")
        with col3:
            needs_improvement = len(filtered_ratings[filtered_ratings['rating'] < 3.5])
            st.metric("⚠️ Needs Improvement", f"{needs_improvement}")
        
        st.markdown("---")
        st.markdown("#### 🏆 Top 5 Performers")
        top_5 = filtered_ratings.nlargest(5, 'rating')[['vendor', 'rating', 'quality', 'delivery', 'reviews']]
        
        for idx, row in top_5.iterrows():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.markdown(f"**{row['vendor']}**")
            with col2:
                st.markdown(f"⭐ **{row['rating']:.1f}**")
            with col3:
                st.markdown(f"Quality: **{row['quality']}%**")
            with col4:
                st.markdown(f"Reviews: **{row['reviews']}**")
    else:
        st.info("No rating data available for filtered vendors")

# TAB 3: CONTRACT DETAILS
with tab3:
    st.subheader("Contract Management")
    
    if len(filtered_contracts) > 0:
        display_contracts = filtered_contracts.copy()
        display_contracts['Status Badge'] = display_contracts['status'].apply(
            lambda x: f"🟢 {x.upper()}" if x == "active" else f"🟡 {x.upper()}" if x == "pending" else f"🔴 {x.upper()}"
        )
        display_contracts['Value Display'] = display_contracts['value'].apply(lambda x: f"${x:,.0f}")
        display_contracts['Min Order Display'] = display_contracts['minimum_order'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(
            display_contracts[['vendor', 'contract_id', 'start_date', 'end_date',
                             'Status Badge', 'Value Display', 'payment_terms',
                             'Min Order Display', 'delivery_terms', 'exclusivity', 'auto_renew']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'vendor': 'Vendor', 'contract_id': 'Contract ID', 'start_date': 'Start Date',
                'end_date': 'End Date', 'Status Badge': 'Status', 'Value Display': 'Contract Value',
                'payment_terms': 'Payment Terms', 'Min Order Display': 'Min Order',
                'delivery_terms': 'Delivery Terms', 'exclusivity': 'Exclusivity', 'auto_renew': 'Auto Renew'
            }
        )
        
        st.caption(f"Showing {len(filtered_contracts):,} contracts")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            active_contracts = len(filtered_contracts[filtered_contracts['status'] == 'active'])
            st.metric("🟢 Active Contracts", f"{active_contracts}")
        with col2:
            pending_contracts = len(filtered_contracts[filtered_contracts['status'] == 'pending'])
            st.metric("🟡 Expiring Soon", f"{pending_contracts}")
        with col3:
            total_value = filtered_contracts['value'].sum()
            st.metric("💰 Total Value", f"${total_value/1000000:.1f}M")
        with col4:
            auto_renew_count = len(filtered_contracts[filtered_contracts['auto_renew'] == 'Yes'])
            st.metric("🔄 Auto-Renew", f"{auto_renew_count}")
        
        if pending_contracts > 0:
            st.markdown("---")
            st.warning(f"⚠️ **{pending_contracts} contracts expiring within 30 days - Action Required!**")
            expiring = filtered_contracts[filtered_contracts['status'] == 'pending'][['vendor', 'contract_id', 'end_date', 'value']].copy()
            expiring['value'] = expiring['value'].apply(lambda x: f"${x:,.0f}")
            expiring.columns = ['Vendor', 'Contract ID', 'End Date', 'Contract Value']
            st.dataframe(expiring, use_container_width=True, hide_index=True)
    else:
        st.info("No contract data available for filtered vendors")

# TAB 4: PERFORMANCE METRICS
with tab4:
    st.subheader("Vendor Performance Metrics")
    
    if len(filtered_performance) > 0:
        for idx, perf in filtered_performance.iterrows():
            overall_score = int((perf['delivery_rate'] + perf['quality_rate'] + perf['fill_rate'] + 
                               (100 - perf['return_rate']) + (100 - perf['defect_rate'])) / 5)
            
            if overall_score >= 95:
                score_color = '#22c55e'
                badge_class = 'Excellent'
                badge_bg = '#d1fae5'
                badge_color = '#065f46'
            elif overall_score >= 85:
                score_color = '#3b82f6'
                badge_class = 'Good'
                badge_bg = '#dbeafe'
                badge_color = '#1e40af'
            elif overall_score >= 75:
                score_color = '#f59e0b'
                badge_class = 'Fair'
                badge_bg = '#fef3c7'
                badge_color = '#92400e'
            else:
                score_color = '#ef4444'
                badge_class = 'Poor'
                badge_bg = '#fee2e2'
                badge_color = '#991b1b'
            
            st.markdown(f"""
            <div class="performance-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h4 style="font-size: 1.125rem; margin: 0;">{perf['vendor']}</h4>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.5rem; font-weight: 800; color: {score_color};">{overall_score}%</span>
                        <span style="padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; 
                                   background: {badge_bg}; color: {badge_color};">{badge_class}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delivery_color = '#22c55e' if perf['delivery_rate'] >= 95 else '#f59e0b' if perf['delivery_rate'] >= 85 else '#ef4444'
                st.markdown(f"**On-Time Delivery:** <span style='color: {delivery_color}; font-weight: bold;'>{perf['delivery_rate']}%</span>", unsafe_allow_html=True)
                quality_color = '#22c55e' if perf['quality_rate'] >= 95 else '#f59e0b' if perf['quality_rate'] >= 85 else '#ef4444'
                st.markdown(f"**Quality Rate:** <span style='color: {quality_color}; font-weight: bold;'>{perf['quality_rate']}%</span>", unsafe_allow_html=True)
            
            with col2:
                return_color = '#22c55e' if perf['return_rate'] <= 2 else '#f59e0b' if perf['return_rate'] <= 4 else '#ef4444'
                st.markdown(f"**Return Rate:** <span style='color: {return_color}; font-weight: bold;'>{perf['return_rate']}%</span>", unsafe_allow_html=True)
                defect_color = '#22c55e' if perf['defect_rate'] <= 1 else '#f59e0b' if perf['defect_rate'] <= 3 else '#ef4444'
                st.markdown(f"**Defect Rate:** <span style='color: {defect_color}; font-weight: bold;'>{perf['defect_rate']}%</span>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"**Lead Time:** **{perf['lead_time']}** days")
                fill_color = '#22c55e' if perf['fill_rate'] >= 95 else '#f59e0b' if perf['fill_rate'] >= 85 else '#ef4444'
                st.markdown(f"**Fill Rate:** <span style='color: {fill_color}; font-weight: bold;'>{perf['fill_rate']}%</span>", unsafe_allow_html=True)
            
            with col4:
                cost_color = '#22c55e' if perf['cost_variance'] <= 3 else '#f59e0b' if perf['cost_variance'] <= 6 else '#ef4444'
                st.markdown(f"**Cost Variance:** <span style='color: {cost_color}; font-weight: bold;'>{perf['cost_variance']}%</span>", unsafe_allow_html=True)
                innovation_color = '#22c55e' if perf['innovation_score'] >= 90 else '#f59e0b' if perf['innovation_score'] >= 80 else '#ef4444'
                st.markdown(f"**Innovation Score:** <span style='color: {innovation_color}; font-weight: bold;'>{perf['innovation_score']}%</span>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_delivery = filtered_performance['delivery_rate'].mean()
            st.metric("📦 Avg Delivery Rate", f"{avg_delivery:.1f}%")
        with col2:
            avg_quality = filtered_performance['quality_rate'].mean()
            st.metric("✅ Avg Quality Rate", f"{avg_quality:.1f}%")
        with col3:
            avg_return = filtered_performance['return_rate'].mean()
            st.metric("↩️ Avg Return Rate", f"{avg_return:.1f}%")
        with col4:
            avg_lead_time = filtered_performance['lead_time'].mean()
            st.metric("⏱️ Avg Lead Time", f"{avg_lead_time:.0f} days")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 10 by Delivery Rate")
            top_delivery = filtered_performance.nlargest(10, 'delivery_rate')[['vendor', 'delivery_rate']]
            fig_delivery = px.bar(top_delivery, x='delivery_rate', y='vendor', 
                                 orientation='h', color='delivery_rate',
                                 color_continuous_scale='Greens', text='delivery_rate')
            fig_delivery.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_delivery.update_layout(height=400, showlegend=False, 
                                      xaxis_title='Delivery Rate (%)', yaxis_title='')
            st.plotly_chart(fig_delivery, use_container_width=True)
        
        with col2:
            st.markdown("#### Top 10 by Quality Rate")
            top_quality = filtered_performance.nlargest(10, 'quality_rate')[['vendor', 'quality_rate']]
            fig_quality = px.bar(top_quality, x='quality_rate', y='vendor',
                                orientation='h', color='quality_rate',
                                color_continuous_scale='Blues', text='quality_rate')
            fig_quality.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_quality.update_layout(height=400, showlegend=False,
                                     xaxis_title='Quality Rate (%)', yaxis_title='')
            st.plotly_chart(fig_quality, use_container_width=True)
    else:
        st.info("No performance data available for filtered vendors")

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Vendor data refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'vendors': vendors_df.to_dict('records'),
            'ratings': ratings_df.to_dict('records'),
            'contracts': contracts_df.to_dict('records'),
            'performance': performance_df.to_dict('records'),
            'summary': {
                'total_vendors': total_vendors,
                'active_vendors': active_vendors,
                'avg_rating': f"{avg_rating:.2f}",
                'data_quality_score': f"{data_quality_score:.1f}%",
                'expiring_contracts': expiring_contracts
            }
        }
        st.success("✅ Vendor data exported successfully")
        st.json(export_data)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================
# ADDITIONAL INSIGHTS
# ===========================

with st.expander("📊 Vendor Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top Performers")
        
        top_vendors = vendors_df.nlargest(5, 'rating')[['name', 'rating', 'quality_score']]
        st.markdown("**Highest Rated Vendors:**")
        for idx, vendor in top_vendors.iterrows():
            st.markdown(f"- **{vendor['name']}**: ⭐ {vendor['rating']:.1f} (Quality: {vendor['quality_score']}%)")
        
        st.markdown("#### 💡 Best Practices")
        st.markdown("""
        - Maintain **95%+ quality scores** for excellent vendor relationships
        - Target **on-time delivery rate of 96%+** for optimal operations
        - Keep **response time under 3 hours** for critical issues
        - Review contracts **90 days before expiry** for renegotiation
        """)
    
    with col2:
        st.markdown("#### ⚠️ Areas of Concern")
        
        low_performers = vendors_df[vendors_df['rating'] < 3.5].nsmallest(5, 'rating')[['name', 'rating', 'quality_score']]
        if len(low_performers) > 0:
            st.markdown("**Vendors Requiring Attention:**")
            for idx, vendor in low_performers.iterrows():
                st.markdown(f"- **{vendor['name']}**: ⭐ {vendor['rating']:.1f} (Quality: {vendor['quality_score']}%)")
        else:
            st.success("✅ All vendors meeting minimum performance standards!")
        
        st.markdown("#### 🎯 Recommended Actions")
        st.markdown(f"""
        1. **Immediate**: Address {invalid_contacts} invalid vendor contacts
        2. **Short-term**: Renew {expiring_contracts} expiring contracts
        3. **Medium-term**: Performance review for vendors below 3.5 rating
        4. **Long-term**: Develop vendor diversification strategy
        """)
    
    st.markdown("---")
    st.markdown("### 📦 Category Analysis")
    
    category_stats = vendors_df.groupby('category').agg({
        'id': 'count',
        'rating': 'mean',
        'quality_score': 'mean',
        'on_time_delivery': 'mean'
    }).reset_index()
    category_stats.columns = ['Category', 'Vendor Count', 'Avg Rating', 'Avg Quality', 'Avg Delivery']
    category_stats = category_stats.sort_values('Avg Rating', ascending=False)
    
    st.dataframe(
        category_stats.style.format({
            'Avg Rating': '{:.2f}',
            'Avg Quality': '{:.1f}%',
            'Avg Delivery': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

# ===========================
# DIAGNOSTIC INFORMATION
# ===========================

with st.expander("🔧 System Diagnostics"):
    st.markdown("### Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(vendors_df):,}")
        st.metric("Data Completeness", f"{data_quality_score:.1f}%")
    
    with col2:
        st.metric("Missing Contacts", f"{invalid_contacts}")
        st.metric("Missing Phone Numbers", f"{len(vendors_df[vendors_df['phone'].isna()])}")
    
    with col3:
        st.metric("Active Contracts", f"{len(contracts_df[contracts_df['status'] == 'active'])}")
        st.metric("Performance Records", f"{len(performance_df)}")
    
    st.markdown("---")
    st.markdown("### Filter Status")
    st.info(f"""
    **Active Filters:**
    - Status: {', '.join(status_filter) if status_filter else 'None'}
    - Rating: {rating_filter}
    - Categories: {', '.join(category_filter[:3]) if category_filter else 'None'}{'...' if len(category_filter) > 3 else ''}
    - Countries: {', '.join(country_filter[:3]) if country_filter else 'None'}{'...' if len(country_filter) > 3 else ''}
    - Search Query: {'`' + search_query + '`' if search_query else 'None'}
    
    **Results:** {len(filtered_vendors):,} vendors shown out of {len(vendors_df):,} total
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Performance Benchmarks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Delivery Performance:**")
        excellent_delivery = len(vendors_df[vendors_df['on_time_delivery'] >= 95])
        good_delivery = len(vendors_df[(vendors_df['on_time_delivery'] >= 85) & (vendors_df['on_time_delivery'] < 95)])
        poor_delivery = len(vendors_df[vendors_df['on_time_delivery'] < 85])
        
        st.markdown(f"- ✅ Excellent (≥95%): **{excellent_delivery}** vendors")
        st.markdown(f"- ⚠️ Good (85-95%): **{good_delivery}** vendors")
        st.markdown(f"- ❌ Needs Improvement (<85%): **{poor_delivery}** vendors")
    
    with col2:
        st.markdown("**Quality Performance:**")
        excellent_quality = len(vendors_df[vendors_df['quality_score'] >= 95])
        good_quality = len(vendors_df[(vendors_df['quality_score'] >= 85) & (vendors_df['quality_score'] < 95)])
        poor_quality = len(vendors_df[vendors_df['quality_score'] < 85])
        
        st.markdown(f"- ✅ Excellent (≥95%): **{excellent_quality}** vendors")
        st.markdown(f"- ⚠️ Good (85-95%): **{good_quality}** vendors")
        st.markdown(f"- ❌ Needs Improvement (<85%): **{poor_quality}** vendors")

# ===========================
# CONTRACT ALERTS
# ===========================

if expiring_contracts > 0:
    with st.expander("⚠️ Contract Expiration Alerts", expanded=True):
        st.warning(f"**{expiring_contracts} contracts require attention!**")
        
        expiring_list = contracts_df[contracts_df['status'] == 'pending'].copy()
        expiring_list['Days Until Expiry'] = expiring_list['end_date'].apply(
            lambda x: (datetime.strptime(x, '%Y-%m-%d') - datetime.now()).days
        )
        expiring_list = expiring_list.sort_values('Days Until Expiry')
        
        for idx, contract in expiring_list.iterrows():
            days_left = contract['Days Until Expiry']
            urgency = "🔴 URGENT" if days_left < 7 else "🟡 SOON"
            
            st.markdown(f"""
            **{urgency}: {contract['vendor']}**
            - Contract ID: {contract['contract_id']}
            - Expires: {contract['end_date']} ({days_left} days)
            - Value: ${contract['value']:,.0f}
            - Auto-Renew: {contract['auto_renew']}
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📧 Contact Vendor", key=f"contact_{contract['contract_id']}"):
                    st.success(f"✅ Email sent to {contract['vendor']}")
            with col2:
                if st.button(f"📝 Renew Contract", key=f"renew_{contract['contract_id']}"):
                    st.success(f"✅ Renewal process initiated for {contract['contract_id']}")
            with col3:
                if st.button(f"📄 View Details", key=f"view_{contract['contract_id']}"):
                    st.info(f"Opening contract details for {contract['contract_id']}")
            
            st.markdown("---")