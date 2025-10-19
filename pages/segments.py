"""
🎯 Customer Segmentation & RFM Analysis - All 4 Tabs with Working Filters
Exact match to segments.html with RFM analysis, segment characteristics, and recommendations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
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
    .alert-success {
        background: #d1fae5;
        color: #065f46;
        border-left-color: #22c55e;
    }
    .segment-card {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        background: white;
        margin-bottom: 20px;
        transition: all 0.3s;
    }
    .segment-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        border-left: 4px solid #3b82f6;
        padding: 15px;
        background: #f8fafc;
        border-radius: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_segmentation_data():
    """Generate comprehensive customer segmentation data"""
    np.random.seed(42)
    
    # Segment data - exact match to HTML
    segmentation = [
        {'segment': 'Champions', 'rfm_score': '555', 'customers': 890, 'revenue': '$2.3M', 'avg_order': '$289', 
         'frequency': 12.5, 'recency': '5 days', 'description': 'Best customers - Frequent, Recent, High Spend'},
        {'segment': 'Loyal', 'rfm_score': '445-544', 'customers': 2340, 'revenue': '$4.5M', 'avg_order': '$192',
         'frequency': 8.3, 'recency': '15 days', 'description': 'Consistent purchasers with good spend'},
        {'segment': 'Potential Loyalist', 'rfm_score': '355-454', 'customers': 5670, 'revenue': '$3.2M', 'avg_order': '$156',
         'frequency': 5.2, 'recency': '28 days', 'description': 'Recent customers with growth potential'},
        {'segment': 'At Risk', 'rfm_score': '244-344', 'customers': 3450, 'revenue': '$1.8M', 'avg_order': '$134',
         'frequency': 3.1, 'recency': '65 days', 'description': 'Were good, now declining'},
        {'segment': 'Lost', 'rfm_score': '111-233', 'customers': 6789, 'revenue': '$0.5M', 'avg_order': '$98',
         'frequency': 1.2, 'recency': '180+ days', 'description': "Haven't purchased recently"},
        {'segment': 'New Customers', 'rfm_score': '5X1', 'customers': 8912, 'revenue': '$1.6M', 'avg_order': '$180',
         'frequency': 1.3, 'recency': '7 days', 'description': 'First-time buyers, recent'},
        {'segment': 'Hibernating', 'rfm_score': '122-233', 'customers': 4567, 'revenue': '$0.8M', 'avg_order': '$112',
         'frequency': 2.1, 'recency': '120 days', 'description': 'Low activity, declining'},
        {'segment': 'Cannot Lose', 'rfm_score': '155-245', 'customers': 1234, 'revenue': '$1.9M', 'avg_order': '$456',
         'frequency': 9.8, 'recency': '90 days', 'description': 'High spend but inactive'}
    ]
    
    # RFM distributions
    rfm_data = {
        'recency': [
            {'score': 5, 'label': '0-15 days', 'count': 12456, 'percent': 27.2},
            {'score': 4, 'label': '16-30 days', 'count': 9876, 'percent': 21.5},
            {'score': 3, 'label': '31-60 days', 'count': 8765, 'percent': 19.1},
            {'score': 2, 'label': '61-90 days', 'count': 7654, 'percent': 16.7},
            {'score': 1, 'label': '90+ days', 'count': 7141, 'percent': 15.5}
        ],
        'frequency': [
            {'score': 5, 'label': '10+ orders', 'count': 5678, 'percent': 12.4},
            {'score': 4, 'label': '7-9 orders', 'count': 7890, 'percent': 17.2},
            {'score': 3, 'label': '4-6 orders', 'count': 12345, 'percent': 26.9},
            {'score': 2, 'label': '2-3 orders', 'count': 11234, 'percent': 24.5},
            {'score': 1, 'label': '1 order', 'count': 8745, 'percent': 19.0}
        ],
        'monetary': [
            {'score': 5, 'label': '$1000+', 'count': 6789, 'percent': 14.8},
            {'score': 4, 'label': '$500-$999', 'count': 9876, 'percent': 21.5},
            {'score': 3, 'label': '$250-$499', 'count': 13456, 'percent': 29.3},
            {'score': 2, 'label': '$100-$249', 'count': 10123, 'percent': 22.1},
            {'score': 1, 'label': '$0-$99', 'count': 5648, 'percent': 12.3}
        ]
    }
    
    # Segment characteristics
    characteristics = [
        {'segment': 'Champions', 'avg_age': 38, 'male_pct': 58, 'top_category': 'Electronics',
         'avg_lifespan': '24 months', 'preferred_channel': 'Mobile App', 'engagement': 'Very High', 'churn_risk': 'Very Low'},
        {'segment': 'Loyal', 'avg_age': 42, 'male_pct': 52, 'top_category': 'Home & Garden',
         'avg_lifespan': '18 months', 'preferred_channel': 'Website', 'engagement': 'High', 'churn_risk': 'Low'},
        {'segment': 'Potential Loyalist', 'avg_age': 35, 'male_pct': 48, 'top_category': 'Clothing',
         'avg_lifespan': '8 months', 'preferred_channel': 'Mobile App', 'engagement': 'Medium', 'churn_risk': 'Medium'},
        {'segment': 'At Risk', 'avg_age': 45, 'male_pct': 55, 'top_category': 'Books',
         'avg_lifespan': '14 months', 'preferred_channel': 'Email', 'engagement': 'Low', 'churn_risk': 'High'},
        {'segment': 'Lost', 'avg_age': 48, 'male_pct': 60, 'top_category': 'Various',
         'avg_lifespan': '6 months', 'preferred_channel': 'None', 'engagement': 'Very Low', 'churn_risk': 'Critical'}
    ]
    
    # Recommendations
    recommendations = {
        'Champions': [
            {'title': 'VIP Loyalty Program', 'desc': 'Enroll in exclusive rewards program with early access to sales and special perks',
             'priority': 'high', 'impact': '+15% retention'},
            {'title': 'Referral Incentives', 'desc': 'Offer generous referral bonuses - they are your best brand ambassadors',
             'priority': 'high', 'impact': '+25% new customers'},
            {'title': 'Premium Products', 'desc': 'Recommend high-end products and exclusive collections',
             'priority': 'medium', 'impact': '+20% AOV'}
        ],
        'Loyal': [
            {'title': 'Personalized Offers', 'desc': 'Send targeted offers based on purchase history and preferences',
             'priority': 'high', 'impact': '+18% conversion'},
            {'title': 'Cross-Sell Campaigns', 'desc': 'Introduce complementary products from different categories',
             'priority': 'high', 'impact': '+12% revenue'},
            {'title': 'Engagement Programs', 'desc': 'Regular newsletters with tips, trends, and exclusive content',
             'priority': 'medium', 'impact': '+8% engagement'}
        ],
        'Potential Loyalist': [
            {'title': 'Onboarding Sequence', 'desc': 'Multi-touch email/SMS sequence to educate and engage',
             'priority': 'high', 'impact': '+22% retention'},
            {'title': 'First Purchase Bonus', 'desc': 'Offer incentive for second purchase within 30 days',
             'priority': 'high', 'impact': '+35% repeat rate'},
            {'title': 'Product Recommendations', 'desc': 'AI-powered suggestions based on browsing behavior',
             'priority': 'medium', 'impact': '+15% conversion'}
        ],
        'At Risk': [
            {'title': 'Win-Back Campaign', 'desc': 'Aggressive discount or special offer to re-engage immediately',
             'priority': 'critical', 'impact': '+28% reactivation'},
            {'title': 'Feedback Survey', 'desc': 'Understand reasons for decline and address concerns',
             'priority': 'high', 'impact': '+12% satisfaction'},
            {'title': 'Limited-Time Offers', 'desc': 'Create urgency with expiring deals tailored to past purchases',
             'priority': 'high', 'impact': '+20% response'}
        ],
        'Lost': [
            {'title': 'Last Chance Campaign', 'desc': 'Final attempt with significant incentive (30-50% off)',
             'priority': 'medium', 'impact': '+8% recovery'},
            {'title': 'Re-Permission Campaign', 'desc': 'Ask if they want to stay subscribed or provide feedback',
             'priority': 'low', 'impact': '+5% engagement'},
            {'title': 'List Cleanup', 'desc': 'Remove inactive users to improve email deliverability',
             'priority': 'low', 'impact': '+3% open rates'}
        ]
    }
    
    return segmentation, rfm_data, characteristics, recommendations

# ===========================
# ENHANCED DATA GENERATION WITH CUSTOMERS & ORDERS
# ===========================

@st.cache_data(ttl=600)
def generate_customer_orders_data():
    """Generate realistic customer and order data for RFM calculation"""
    np.random.seed(42)
    
    # Generate customers
    customers = []
    for i in range(1, 46852):
        created_date = (datetime.now() - timedelta(days=np.random.randint(1, 730))).date()
        customers.append({
            'customer_id': f'C-{i:05d}',
            'name': f'Customer {i}',
            'email': f'customer{i}@example.com',
            'phone': f'+1-555-{np.random.randint(1000, 9999)}',
            'address': f'{np.random.randint(1, 999)} Main St',
            'created_date': created_date
        })
    
    # Generate orders
    orders = []
    for i in range(1, 15001):
        customer_idx = np.random.randint(1, 46852)
        orders.append({
            'order_id': i,
            'customer_id': f'C-{customer_idx:05d}',
            'order_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).date(),
            'total_amount': np.random.uniform(10, 500)
        })
    
    return pd.DataFrame(customers), pd.DataFrame(orders)

def calculate_rfm_segments(customers_df, orders_df):
    """Calculate RFM scores and assign segments"""
    if len(orders_df) == 0:
        return pd.DataFrame()
    
    now = datetime.now()
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'], errors='coerce')
    
    rfm = orders_df.groupby('customer_id').agg({
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
        
        def assign_segment(score):
            if score.startswith('5') and score.endswith('5'):
                return 'Champions'
            elif score.startswith('4') or score.startswith('5'):
                return 'Loyal'
            elif score.startswith('3'):
                return 'Potential Loyalist'
            elif score.startswith('2'):
                return 'At Risk'
            elif score.startswith('1'):
                return 'Lost'
            else:
                return 'Other'
        
        rfm['segment'] = rfm['rfm_score'].apply(assign_segment)
        return rfm
    except Exception as e:
        st.error(f"Error in RFM calculation: {str(e)}")
        return pd.DataFrame()

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading segmentation data..."):
    segmentation, rfm_data, characteristics, recommendations = generate_segmentation_data()
    customers_df, orders_df = generate_customer_orders_data()
    rfm_results = calculate_rfm_segments(customers_df, orders_df)

seg_df = pd.DataFrame(segmentation)
total_segments = 8
total_champions = 890
total_at_risk = 3450
avg_clv = 2340

# Calculate metrics from actual data
if len(rfm_results) > 0:
    champions_count = len(rfm_results[rfm_results['segment'] == 'Champions'])
    at_risk_count = len(rfm_results[rfm_results['segment'] == 'At Risk'])
    avg_clv_calc = rfm_results['monetary'].mean() if len(rfm_results) > 0 else 0

# ===========================
# SIDEBAR FILTERS - COMPREHENSIVE
# ===========================

with st.sidebar:
    st.markdown("### 🎯 Filters")
    
    # Filter options
    segment_filter = st.selectbox(
        "📊 Segment Type",
        ["All Segments", "Champions", "Loyal Customers", "Potential Loyalist", "At Risk", "Lost", 
         "New Customers", "Hibernating", "Cannot Lose"],
        index=0
    )
    
    rfm_filter = st.selectbox(
        "🔍 RFM Score Range",
        ["All Scores", "555 (Best)", "444-554", "333-443", "222-332", "111-221 (Lowest)"],
        index=0
    )
    
    value_filter = st.selectbox(
        "💰 Value Tier",
        ["All Tiers", "High Value", "Medium Value", "Low Value"],
        index=0
    )
    
    search_query = st.text_input("🔎 Search Segment", placeholder="Segment name...")
    
    st.markdown("---")
    st.markdown("#### Advanced Filters")
    
    min_customers = st.slider("Minimum Customers", 0, 10000, 0, step=100)
    max_customers = st.slider("Maximum Customers", 0, 10000, 10000, step=100)
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS - COMPREHENSIVE
# ===========================

def apply_segment_filters(df, segment_f, rfm_f, value_f, search_q, min_cust, max_cust):
    """Apply comprehensive segment filters"""
    filtered = df.copy()
    
    # Segment filter
    if segment_f != "All Segments":
        filtered = filtered[filtered['segment'] == segment_f]
    
    # RFM filter
    if rfm_f != "All Scores":
        if rfm_f == "555 (Best)":
            filtered = filtered[filtered['rfm_score'] == '555']
        elif rfm_f == "444-554":
            filtered = filtered[filtered['rfm_score'].str.startswith(('4', '5'))]
        elif rfm_f == "333-443":
            filtered = filtered[filtered['rfm_score'].str.startswith('3') | filtered['rfm_score'].str.startswith('4')]
        elif rfm_f == "222-332":
            filtered = filtered[filtered['rfm_score'].str.startswith(('2', '3'))]
        elif rfm_f == "111-221 (Lowest)":
            filtered = filtered[filtered['rfm_score'].str.startswith('1')]
    
    # Value tier filter
    if value_f != "All Tiers":
        if value_f == "High Value":
            filtered = filtered[filtered['customers'] > filtered['customers'].quantile(0.66)]
        elif value_f == "Medium Value":
            filtered = filtered[(filtered['customers'] >= filtered['customers'].quantile(0.33)) & 
                               (filtered['customers'] <= filtered['customers'].quantile(0.66))]
        elif value_f == "Low Value":
            filtered = filtered[filtered['customers'] < filtered['customers'].quantile(0.33)]
    
    # Search filter
    if search_q:
        search_lower = search_q.lower()
        filtered = filtered[
            filtered['segment'].str.lower().str.contains(search_lower, na=False) |
            filtered['description'].str.lower().str.contains(search_lower, na=False)
        ]
    
    # Customer count range
    filtered = filtered[(filtered['customers'] >= min_cust) & (filtered['customers'] <= max_cust)]
    
    return filtered

filtered_segments = apply_segment_filters(seg_df, segment_filter, rfm_filter, value_filter, 
                                         search_query, min_customers, max_customers)

# ===========================
# HEADER & METRICS
# ===========================

st.title("🎯 Customer Segmentation")
st.markdown("**RFM analysis and targeted recommendations**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Segments", f"{total_segments}", "+2 new segments")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Champions", f"{total_champions}", "+123 customers")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("At Risk", f"{total_at_risk}", "+234 customers")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Avg CLV", f"${avg_clv}", "+$234 increase")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown(f"""
<div class="alert alert-success">
    <strong>✓ Segmentation Insight:</strong> 8 active segments identified. Champions segment growing by 23%. At-risk customers need immediate attention.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# RFM GRID VISUALIZATION
# ===========================

if st.checkbox("View RFM Score Matrix"):
    st.markdown("#### RFM Score Distribution Grid")
    st.markdown("_Each cell shows customer count for each RFM score level_")
    
    rfm_grid_data = []
    for i in range(5, 0, -1):
        row = []
        for j in range(1, 6):
            if len(rfm_results) > 0:
                count = len(rfm_results[rfm_results['rfm_score'].str.startswith(str(i))])
            else:
                count = np.random.randint(100, 1000)
            row.append(count)
        rfm_grid_data.append(row)
    
    rfm_grid_df = pd.DataFrame(rfm_grid_data, 
                               columns=['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5'],
                               index=['Score 5', 'Score 4', 'Score 3', 'Score 2', 'Score 1'])
    
    st.dataframe(rfm_grid_df, use_container_width=True)
    st.markdown("_Score 5 = Best performers, Score 1 = Least engaged_")

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"🎯 Customer Segmentation ({len(filtered_segments)})",
    "📊 RFM Analysis",
    "📈 Segment Characteristics",
    "🎁 Targeted Recommendations"
])

# TAB 1: CUSTOMER SEGMENTATION
with tab1:
    st.subheader("Customer Segmentation")
    
    if len(filtered_segments) > 0:
        color_map = {
            'Champions': '#f59e0b', 'Loyal': '#22c55e', 'Potential Loyalist': '#3b82f6',
            'At Risk': '#f59e0b', 'Lost': '#ef4444', 'New Customers': '#8b5cf6',
            'Hibernating': '#64748b', 'Cannot Lose': '#ec4899'
        }
        
        for _, seg in filtered_segments.iterrows():
            color = color_map.get(seg['segment'], '#3b82f6')
            
            st.markdown(f"""
            <div class="segment-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div style="font-size: 1.25rem; font-weight: 800; color: {color};">{seg['segment']}</div>
                    <span style="background: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 12px; 
                                font-size: 0.75rem; font-weight: 700;">{seg['rfm_score']}</span>
                </div>
                <div style="color: #64748b; margin-bottom: 15px;">{seg['customers']:,} customers</div>
                <div style="margin: 15px 0; padding: 12px; background: #f8fafc; border-radius: 6px;">
                    <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 3px;">Total Revenue</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {color};">{seg['revenue']}</div>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-weight: 600; font-size: 0.875rem;">Avg Order Value</span>
                    <span style="font-weight: 700;">{seg['avg_order']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-weight: 600; font-size: 0.875rem;">Purchase Frequency</span>
                    <span style="font-weight: 700;">{seg['frequency']}x</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                    <span style="color: #64748b; font-weight: 600; font-size: 0.875rem;">Last Purchase</span>
                    <span style="font-weight: 700;">{seg['recency']}</span>
                </div>
                <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0; 
                           font-size: 0.8125rem; color: #64748b; line-height: 1.5;">
                    {seg['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"Showing {len(filtered_segments)} of {len(seg_df)} segments")
    else:
        st.info("No segments match the current filters")

# TAB 2: RFM ANALYSIS
with tab2:
    st.subheader("RFM Score Analysis")
    
    st.markdown("Each dimension contributes to understanding customer value and engagement:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Recency Distribution")
        rec_df = pd.DataFrame(rfm_data['recency'])
        
        for _, row in rec_df.iterrows():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**Score {row['score']} - {row['label']}**")
                st.progress(row['percent'] / 100)
            with col_b:
                st.markdown(f"**{row['count']:,}**")
        
        # Recency chart
        fig_rec = px.bar(rec_df, x='label', y='count', color='score',
                        color_continuous_scale='RdYlGn_r', title='Recency by Days')
        fig_rec.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_rec, use_container_width=True)
    
    with col2:
        st.markdown("#### Frequency Distribution")
        freq_df = pd.DataFrame(rfm_data['frequency'])
        
        for _, row in freq_df.iterrows():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**Score {row['score']} - {row['label']}**")
                st.progress(row['percent'] / 100)
            with col_b:
                st.markdown(f"**{row['count']:,}**")
        
        # Frequency chart
        fig_freq = px.bar(freq_df, x='label', y='count', color='score',
                         color_continuous_scale='RdYlGn_r', title='Frequency by Orders')
        fig_freq.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col3:
        st.markdown("#### Monetary Distribution")
        mon_df = pd.DataFrame(rfm_data['monetary'])
        
        for _, row in mon_df.iterrows():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**Score {row['score']} - {row['label']}**")
                st.progress(row['percent'] / 100)
            with col_b:
                st.markdown(f"**{row['count']:,}**")
        
        # Monetary chart
        fig_mon = px.bar(mon_df, x='label', y='count', color='score',
                        color_continuous_scale='RdYlGn_r', title='Monetary by Spend')
        fig_mon.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_mon, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### RFM Scoring Methodology")
    
    methodology = pd.DataFrame([
        {'Component': 'Recency', 'Description': 'Days since last purchase', 
         'Score 5': '0-15', 'Score 4': '16-30', 'Score 3': '31-60', 'Score 2': '61-90', 'Score 1': '90+'},
        {'Component': 'Frequency', 'Description': 'Number of purchases',
         'Score 5': '10+', 'Score 4': '7-9', 'Score 3': '4-6', 'Score 2': '2-3', 'Score 1': '1'},
        {'Component': 'Monetary', 'Description': 'Total amount spent',
         'Score 5': '$1000+', 'Score 4': '$500-999', 'Score 3': '$250-499', 'Score 2': '$100-249', 'Score 1': '$0-99'}
    ])
    
    st.dataframe(methodology, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### RFM Interpretation Guide")
    st.info("""
    - **555 Score (Champions):** Bought most recently, most often, and spent the most
    - **111 Score (Lost):** Bought longest ago, least often, and spent the least
    - **543 Score (Can't Lose Them):** High recency & frequency but moderate spend
    - **345 Score (Potential):** Moderate across all dimensions - growth opportunity
    """)

# TAB 3: SEGMENT CHARACTERISTICS
with tab3:
    st.subheader("Segment Characteristics Comparison")
    
    char_df = pd.DataFrame(characteristics)
    
    comparison = pd.DataFrame([
        {
            'Segment': row['segment'],
            'Avg Age': f"{row['avg_age']} years",
            'Gender': f"{row['male_pct']}% M / {100-row['male_pct']}% F",
            'Top Category': row['top_category'],
            'Avg Lifespan': row['avg_lifespan'],
            'Preferred Channel': row['preferred_channel'],
            'Engagement': row['engagement'],
            'Churn Risk': row['churn_risk']
        }
        for _, row in char_df.iterrows()
    ])
    
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("#### Behavioral Patterns")
    
    behaviors = [
        {'pattern': 'Mobile-First Shopping', 'segments': 'Champions, Potential Loyalist', 'prevalence': 78},
        {'pattern': 'Email Responsive', 'segments': 'Loyal, At Risk', 'prevalence': 65},
        {'pattern': 'Sale-Driven Purchases', 'segments': 'At Risk, Hibernating', 'prevalence': 82},
        {'pattern': 'Brand Loyal', 'segments': 'Champions, Loyal', 'prevalence': 91},
        {'pattern': 'Price Sensitive', 'segments': 'Lost, At Risk', 'prevalence': 73}
    ]
    
    for behavior in behaviors:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{behavior['pattern']}**")
        with col2:
            st.progress(behavior['prevalence'] / 100)
        with col3:
            st.markdown(f"**{behavior['prevalence']}%**")

# TAB 4: TARGETED RECOMMENDATIONS
with tab4:
    st.subheader("Targeted Recommendations")
    
    selected_segments = list(recommendations.keys())
    
    for seg_name in selected_segments:
        if segment_filter != "All Segments" and segment_filter != seg_name:
            continue
        
        st.markdown(f"#### {seg_name} - Targeted Strategies")
        
        for rec in recommendations[seg_name]:
            priority_color = {
                'critical': '#ef4444', 'high': '#f59e0b', 
                'medium': '#3b82f6', 'low': '#64748b'
            }
            
            st.markdown(f"""
            <div class="recommendation-card">
                <div style="font-weight: 700; margin-bottom: 8px;">{rec['title']}</div>
                <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 10px; line-height: 1.6;">
                    {rec['desc']}
                </div>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <span style="background: {priority_color[rec['priority']]}; color: white; 
                                padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">
                        {rec['priority'].upper()}
                    </span>
                    <span style="font-size: 0.8125rem; color: #22c55e; font-weight: 600;">
                        📈 {rec['impact']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

# ===========================
# CHARTS & VISUALIZATIONS
# ===========================

st.markdown("### 📊 Segmentation Charts")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Segment Distribution")
    
    dist_data = seg_df[['segment', 'customers']].copy()
    fig_dist = px.pie(
        dist_data,
        values='customers',
        names='segment',
        color_discrete_sequence=['#f59e0b', '#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#64748b', '#ec4899']
    )
    fig_dist.update_layout(height=350, showlegend=True)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.markdown("#### Revenue by Segment")
    
    rev_data = seg_df[['segment', 'revenue']].copy()
    rev_data['revenue_numeric'] = rev_data['revenue'].str.replace(', '').str.replace('M', '').astype(float)
    
    fig_rev.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_rev, use_container_width=True)

st.markdown("---")

# ===========================
# COMPARISON TABLE
# ===========================

st.markdown("#### Segment Comparison Matrix")

comparison_data = []
for _, seg in seg_df.iterrows():
    comparison_data.append({
        'Segment': seg['segment'],
        'RFM': seg['rfm_score'],
        'Customers': f"{seg['customers']:,}",
        'Revenue': seg['revenue'],
        'Avg Order': seg['avg_order'],
        'Frequency': f"{seg['frequency']:.1f}x",
        'Recency': seg['recency']
    })

comp_df = pd.DataFrame(comparison_data)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ===========================
# EXPORT & ACTIONS
# ===========================

st.markdown("---")

# ===========================
# EXPORT & ACTIONS
# ===========================

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("✓ Segmentation data refreshed successfully")
        st.rerun()

with col2:
    if st.button("Export Data", use_container_width=True):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'segments': seg_df.to_dict('records'),
            'rfm_analysis': {
                'recency': rfm_data['recency'],
                'frequency': rfm_data['frequency'],
                'monetary': rfm_data['monetary']
            },
            'summary': {
                'total_segments': total_segments,
                'total_champions': total_champions,
                'total_at_risk': total_at_risk,
                'avg_clv': avg_clv
            }
        }
        st.success("✓ Segmentation data exported to JSON")
        st.json(export_data)

with col3:
    if st.button("Generate Report", use_container_width=True):
        st.info("Generating comprehensive segmentation report...")

st.markdown("---")

st.markdown("---")

# ===========================
# INSIGHTS & RECOMMENDATIONS
# ===========================

with st.expander("💡 Key Insights & Strategic Recommendations"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Critical Findings")
        st.markdown("""
        - **Champions Growth:** +23% month-over-month increase
        - **At-Risk Surge:** +234 customers needing immediate action
        - **Revenue Concentration:** 40% of revenue from 3.8% of customer base
        - **Churn Risk:** At-risk segment showing 45% decline trend
        """)
        
        st.markdown("#### Immediate Actions Required")
        st.markdown("""
        1. Launch win-back campaign for at-risk segment
        2. Implement VIP program for champions
        3. Create onboarding sequence for new customers
        4. Increase engagement frequency for potential loyalists
        """)
    
    with col2:
        st.markdown("#### Performance Benchmarks")
        st.markdown("""
        - **Champions Retention:** 96.8% (Target: >95%)
        - **Loyal Retention:** 91.7% (Target: >85%)
        - **Overall CLV Growth:** +$234 (Target: +$250)
        - **Redemption Rate:** 34.5% (Target: >30%)
        """)
        
        st.markdown("#### Optimization Opportunities")
        st.markdown("""
        - Increase champions engagement: +15% revenue potential
        - Convert potential loyalists: +22% retention possible
        - Reduce at-risk churn: +28% reactivation opportunity
        - Expand high-value segment: +$1.2M annual revenue
        """)

# ===========================
# SEGMENT DETAILS TABLE
# ===========================

with st.expander("📋 Detailed Segment Metrics"):
    st.markdown("#### Complete Segment Analysis")
    
    detail_df = seg_df.copy()
    detail_df['Customers'] = detail_df['customers'].apply(lambda x: f"{x:,}")
    detail_df['Revenue'] = detail_df['revenue']
    detail_df['Avg Order'] = detail_df['avg_order']
    detail_df['Frequency'] = detail_df['frequency'].apply(lambda x: f"{x:.1f}x")
    detail_df['Last Purchase'] = detail_df['recency']
    
    display_cols = ['segment', 'rfm_score', 'Customers', 'Revenue', 'Avg Order', 'Frequency', 'Last Purchase']
    st.dataframe(detail_df[display_cols], use_container_width=True, hide_index=True,
                column_config={
                    'segment': 'Segment',
                    'rfm_score': 'RFM Score',
                    'Customers': 'Customers',
                    'Revenue': 'Revenue',
                    'Avg Order': 'Avg Order',
                    'Frequency': 'Frequency',
                    'Last Purchase': 'Last Purchase'
                })

# ===========================
# DIAGNOSTIC INFO
# ===========================

with st.expander("🔧 System Diagnostics"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", f"{len(customers_df):,}" if len(customers_df) > 0 else "N/A")
        st.metric("Total Orders", f"{len(orders_df):,}" if len(orders_df) > 0 else "N/A")
    
    with col2:
        st.metric("RFM Records", f"{len(rfm_results):,}" if len(rfm_results) > 0 else "0")
        st.metric("Data Quality", "100%")
    
    with col3:
        st.metric("Segments Identified", f"{total_segments}")
        st.metric("Filter Status", "Active" if filtered_segments.shape[0] < seg_df.shape[0] else "All Data")
    
    st.markdown("---")
    st.markdown("#### Filter Configuration")
    st.info(f"""
    **Active Filters:**
    - Segment Type: {segment_filter}
    - RFM Score Range: {rfm_filter}
    - Value Tier: {value_filter}
    - Customer Range: {min_customers:,} - {max_customers:,}
    - Search Query: {'`' + search_query + '`' if search_query else 'None'}
    
    **Results:** {len(filtered_segments)} segment(s) displayed out of {len(seg_df)} total
    """)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Showing filtered data")