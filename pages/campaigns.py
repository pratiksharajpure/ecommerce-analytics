"""
📢 Enhanced Marketing Campaign Analysis - All 4 Tabs with Working Filters
Exact match to campaigns.html with ROI analysis, performance tracking, and A/B testing
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Marketing Campaign Analysis",
    page_icon="📢",
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
    .alert-info {
        background: #dbeafe;
        color: #1e40af;
        border-left-color: #3b82f6;
    }
    .performance-card {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_campaign_data():
    """Generate sample campaign data"""
    np.random.seed(42)
    
    campaigns = []
    channels = ['Email', 'Social Media', 'Display Ads', 'Search Ads']
    statuses = ['active', 'paused', 'completed', 'draft']
    
    campaign_names = [
        'Q4 Holiday Sale 2024', 'Black Friday Special', 'Summer Collection Launch',
        'Back to School Promo', 'New Customer Acquisition', 'Retargeting Campaign',
        'Spring Sale 2025', 'Valentine\'s Day Campaign', 'Easter Promotion',
        'Mother\'s Day Special', 'Father\'s Day Campaign', 'Cyber Monday Deal',
        'Flash Sale Event', 'Seasonal Clearance', 'Product Launch Campaign',
        'Customer Loyalty Drive', 'Referral Program Push', 'Win-back Campaign',
        'Bundle Deal Promotion', 'Anniversary Sale', 'Premium Tier Launch',
        'Mobile App Download', 'Newsletter Signup Drive', 'VIP Exclusive Offer'
    ]
    
    for i, name in enumerate(campaign_names, 1):
        status = np.random.choice(statuses, p=[0.4, 0.2, 0.2, 0.2])
        channel = np.random.choice(channels)
        budget = np.random.uniform(10000, 50000)
        
        if status == 'draft':
            spent = 0
            impressions = 0
            clicks = 0
            conversions = 0
            revenue = 0
        else:
            spent = budget * np.random.uniform(0.6, 1.0)
            impressions = np.random.randint(100000, 1000000)
            clicks = int(impressions * np.random.uniform(0.03, 0.08))
            conversions = int(clicks * np.random.uniform(0.02, 0.15))
            revenue = conversions * np.random.uniform(50, 300)
        
        roi = ((revenue - spent) / spent * 100) if spent > 0 else 0
        ctr = (clicks / impressions * 100) if impressions > 0 else 0
        cvr = (conversions / clicks * 100) if clicks > 0 else 0
        
        campaigns.append({
            'id': f'CMP-{1000+i}',
            'name': name,
            'channel': channel,
            'status': status,
            'budget': budget,
            'spent': spent,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'revenue': revenue,
            'roi': roi,
            'ctr': ctr,
            'cvr': cvr,
            'start_date': (datetime.now() - timedelta(days=np.random.randint(1, 180))).date(),
            'end_date': (datetime.now() - timedelta(days=np.random.randint(0, 60))).date() if status == 'completed' else None
        })
    
    # A/B Testing data
    ab_tests = []
    test_names = ['Email Subject Line A/B', 'Landing Page Layout', 'CTA Button Color']
    
    ab_tests.append({
        'test': 'Email Subject Line A/B',
        'variant': 'A: "Save 30% Today"',
        'impressions': 50000,
        'clicks': 2450,
        'conversions': 145,
        'ctr': 4.9,
        'cvr': 5.9,
        'winner': True
    })
    ab_tests.append({
        'test': 'Email Subject Line A/B',
        'variant': 'B: "Limited Time Offer"',
        'impressions': 50000,
        'clicks': 2150,
        'conversions': 118,
        'ctr': 4.3,
        'cvr': 5.5,
        'winner': False
    })
    ab_tests.append({
        'test': 'Landing Page Layout',
        'variant': 'A: Single Column',
        'impressions': 75000,
        'clicks': 3600,
        'conversions': 234,
        'ctr': 4.8,
        'cvr': 6.5,
        'winner': False
    })
    ab_tests.append({
        'test': 'Landing Page Layout',
        'variant': 'B: Two Column',
        'impressions': 75000,
        'clicks': 3750,
        'conversions': 289,
        'ctr': 5.0,
        'cvr': 7.7,
        'winner': True
    })
    ab_tests.append({
        'test': 'CTA Button Color',
        'variant': 'A: Blue Button',
        'impressions': 60000,
        'clicks': 2880,
        'conversions': 178,
        'ctr': 4.8,
        'cvr': 6.2,
        'winner': False
    })
    ab_tests.append({
        'test': 'CTA Button Color',
        'variant': 'B: Green Button',
        'impressions': 60000,
        'clicks': 3120,
        'conversions': 203,
        'ctr': 5.2,
        'cvr': 6.5,
        'winner': True
    })
    
    return pd.DataFrame(campaigns), pd.DataFrame(ab_tests)

# ===========================
# ANALYSIS FUNCTIONS
# ===========================

def calculate_performance_metrics(df):
    """Calculate overall performance metrics"""
    active_campaigns = len(df[df['status'] == 'active'])
    total_spend = df['spent'].sum()
    avg_roi = df[df['roi'] > 0]['roi'].mean() if len(df[df['roi'] > 0]) > 0 else 0
    total_conversions = df['conversions'].sum()
    
    return {
        'active_campaigns': active_campaigns,
        'total_spend': total_spend,
        'avg_roi': avg_roi,
        'total_conversions': total_conversions,
        'avg_ctr': df[df['ctr'] > 0]['ctr'].mean() if len(df[df['ctr'] > 0]) > 0 else 0,
        'avg_cvr': df[df['cvr'] > 0]['cvr'].mean() if len(df[df['cvr'] > 0]) > 0 else 0,
    }

def get_roi_analysis(df):
    """Prepare ROI analysis data"""
    roi_data = []
    
    for _, row in df[df['status'].isin(['active', 'completed'])].iterrows():
        profit = row['revenue'] - row['spent']
        roas = row['revenue'] / row['spent'] if row['spent'] > 0 else 0
        margin = (profit / row['revenue'] * 100) if row['revenue'] > 0 else 0
        
        roi_data.append({
            'Campaign': row['name'],
            'Spend': f"${row['spent']:,.0f}",
            'Revenue': f"${row['revenue']:,.0f}",
            'Profit': f"${profit:,.0f}",
            'ROI': f"{row['roi']:.0f}%",
            'ROAS': f"{roas:.2f}x",
            'Margin': f"{margin:.0f}%"
        })
    
    return pd.DataFrame(roi_data)

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading campaign data..."):
    campaigns_df, ab_tests_df = generate_sample_campaign_data()
    perf_metrics = calculate_performance_metrics(campaigns_df)

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 📢 Filters")
    
    date_range = st.selectbox(
        "📅 Date Range",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "This Year"],
        index=1
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
    
    status_filter = st.multiselect(
        "📋 Campaign Status",
        ["active", "paused", "completed", "draft"],
        default=["active", "paused", "completed", "draft"]
    )
    
    channel_filter = st.multiselect(
        "📡 Channel",
        ["Email", "Social Media", "Display Ads", "Search Ads"],
        default=["Email", "Social Media", "Display Ads", "Search Ads"]
    )
    
    search_query = st.text_input("🔍 Search Campaign", placeholder="Name, ID...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_campaign_filters(df, date_cutoff, status_list, channel_list, search_text):
    filtered = df.copy()
    
    if date_cutoff:
        filtered = filtered[filtered['start_date'] >= date_cutoff]
    
    if status_list:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    if channel_list:
        filtered = filtered[filtered['channel'].isin(channel_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['name'].str.lower().str.contains(search_lower, na=False) |
            filtered['id'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_campaigns = apply_campaign_filters(campaigns_df, cutoff_date, status_filter, channel_filter, search_query)

# ===========================
# HEADER & METRICS
# ===========================

st.title("📢 Marketing Campaign Analysis")
st.markdown("**Campaign performance tracking, ROI analysis, and conversion metrics**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Active Campaigns", f"{perf_metrics['active_campaigns']}", "+5 new campaigns")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Total Spend", f"${perf_metrics['total_spend']:,.0f}", "+$23,500 invested")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Average ROI", f"{perf_metrics['avg_roi']:.0f}%", "+45% improvement")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Total Conversions", f"{perf_metrics['total_conversions']:,.0f}", "+678 new conversions")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

st.markdown(f"""
<div class="alert alert-info">
    <strong>Campaign Update:</strong> {perf_metrics['active_campaigns']} active campaigns running. Q4 holiday campaign showing 340% ROI with 3,456 conversions.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# PERFORMANCE CARDS
# ===========================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="performance-card" style="background: linear-gradient(135deg, #3b82f6, #2563eb);">', unsafe_allow_html=True)
    st.markdown("#### Click-Through Rate")
    st.markdown(f"### {perf_metrics['avg_ctr']:.1f}%")
    st.markdown("*Above industry average*")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="performance-card" style="background: linear-gradient(135deg, #22c55e, #16a34a);">', unsafe_allow_html=True)
    st.markdown("#### Conversion Rate")
    st.markdown(f"### {perf_metrics['avg_cvr']:.1f}%")
    st.markdown("*+0.5% vs last month*")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="performance-card" style="background: linear-gradient(135deg, #8b5cf6, #7c3aed);">', unsafe_allow_html=True)
    st.markdown("#### Cost Per Acquisition")
    avg_cpa = (perf_metrics['total_spend'] / perf_metrics['total_conversions']) if perf_metrics['total_conversions'] > 0 else 0
    st.markdown(f"### ${avg_cpa:.2f}")
    st.markdown("*-$4.50 reduction*")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Campaign Performance Trend")
    
    weekly_data = []
    for i in range(4):
        week_num = i + 1
        weekly_data.append({
            'Week': f'Week {week_num}',
            'Conversions': [756, 892, 1045, 1234][i],
            'Revenue': [67, 78, 91, 105][i]
        })
    
    weekly_df = pd.DataFrame(weekly_data)
    fig1 = px.line(weekly_df, x='Week', y=['Conversions', 'Revenue'], markers=True)
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), hovermode='x unified')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Channel Distribution")
    
    channel_data = campaigns_df.groupby('channel')['conversions'].sum().reset_index()
    fig2 = px.pie(channel_data, values='conversions', names='channel',
                  color_discrete_sequence=['#3b82f6', '#8b5cf6', '#f59e0b', '#22c55e'])
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 Campaign List ({len(filtered_campaigns)})",
    f"📊 Performance Dashboard",
    f"💰 ROI Analysis",
    f"🧪 A/B Testing"
])

# TAB 1: CAMPAIGN LIST
with tab1:
    st.subheader("Marketing Campaign List")
    
    if len(filtered_campaigns) > 0:
        display_df = filtered_campaigns.copy()
        
        display_df['Status'] = display_df['status'].apply(
            lambda x: f"✅ {x.upper()}" if x == "active"
            else f"⏸ {x.upper()}" if x == "paused"
            else f"✔️ {x.upper()}" if x == "completed"
            else f"📝 {x.upper()}"
        )
        display_df['Budget'] = display_df['budget'].apply(lambda x: f"${x:,.0f}")
        display_df['Spent'] = display_df['spent'].apply(lambda x: f"${x:,.0f}")
        display_df['Impressions'] = display_df['impressions'].apply(lambda x: f"{x:,}")
        display_df['Clicks'] = display_df['clicks'].apply(lambda x: f"{x:,}")
        display_df['Conversions'] = display_df['conversions'].apply(lambda x: f"{x:,}")
        display_df['Revenue'] = display_df['revenue'].apply(lambda x: f"${x:,.0f}")
        display_df['ROI'] = display_df['roi'].apply(lambda x: f"{x:.0f}%")
        display_df['CTR'] = display_df['ctr'].apply(lambda x: f"{x:.1f}%")
        display_df['CVR'] = display_df['cvr'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_df[['id', 'name', 'channel', 'Status', 'Budget', 'Spent', 
                       'Impressions', 'Clicks', 'Conversions', 'Revenue', 'ROI', 'CTR', 'CVR']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'id': 'Campaign ID',
                'name': 'Campaign Name',
                'channel': 'Channel'
            }
        )
        
        st.caption(f"Showing {len(filtered_campaigns):,} of {len(campaigns_df):,} campaigns")
    else:
        st.info("No campaigns match the current filters")

# TAB 2: PERFORMANCE DASHBOARD
with tab2:
    st.subheader("Campaign Performance Dashboard")
    
    perf_data = []
    perf_data.append({
        'Metric': 'Total Impressions',
        'Value': f"{campaigns_df['impressions'].sum():,}",
        'Change': '+18%',
        'Target': '2.2M',
        'Progress': 114
    })
    perf_data.append({
        'Metric': 'Total Clicks',
        'Value': f"{campaigns_df['clicks'].sum():,}",
        'Change': '+22%',
        'Target': '100K',
        'Progress': 119
    })
    perf_data.append({
        'Metric': 'Total Conversions',
        'Value': f"{campaigns_df['conversions'].sum():,}",
        'Change': '+15%',
        'Target': '3,800',
        'Progress': 109
    })
    perf_data.append({
        'Metric': 'Total Revenue',
        'Value': f"${campaigns_df['revenue'].sum():,.0f}",
        'Change': '+28%',
        'Target': '$320K',
        'Progress': 121
    })
    perf_data.append({
        'Metric': 'Avg CTR',
        'Value': f"{perf_metrics['avg_ctr']:.1f}%",
        'Change': '+0.3%',
        'Target': '4.5%',
        'Progress': 107
    })
    perf_data.append({
        'Metric': 'Avg CVR',
        'Value': f"{perf_metrics['avg_cvr']:.1f}%",
        'Change': '+0.5%',
        'Target': '3.0%',
        'Progress': 117
    })
    
    perf_df = pd.DataFrame(perf_data)
    
    for _, row in perf_df.iterrows():
        col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
        
        with col1:
            st.markdown(f"**{row['Metric']}**")
        with col2:
            st.markdown(f"### {row['Value']}")
        with col3:
            st.markdown(f"<span style='color:green'>{row['Change']}</span>", unsafe_allow_html=True)
        with col4:
            progress_color = '#22c55e' if row['Progress'] >= 110 else '#3b82f6' if row['Progress'] >= 90 else '#f59e0b'
            st.markdown(f"""
                <div style="background:#e2e8f0;border-radius:4px;overflow:hidden;height:8px;margin-top:8px;">
                    <div style="width:{min(row['Progress'], 100)}%;height:100%;background:{progress_color};"></div>
                </div>
                <div style="text-align:right;font-weight:bold;color:{progress_color};font-size:0.875rem;margin-top:2px;">{row['Progress']}%</div>
            """, unsafe_allow_html=True)
        st.divider()

# TAB 3: ROI ANALYSIS
with tab3:
    st.subheader("ROI & ROAS Analysis")
    
    roi_df = get_roi_analysis(campaigns_df)
    
    if len(roi_df) > 0:
        st.dataframe(
            roi_df,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # ROI distribution chart
        roi_numeric = campaigns_df[campaigns_df['status'].isin(['active', 'completed'])]['roi'].values
        
        fig = px.histogram(
            pd.DataFrame({'ROI': roi_numeric}),
            x='ROI',
            nbins=20,
            title='ROI Distribution Across Campaigns',
            color_discrete_sequence=['#3b82f6']
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ROI data available for active or completed campaigns")

# TAB 4: A/B TESTING
with tab4:
    st.subheader("A/B Testing Results")
    
    if len(ab_tests_df) > 0:
        display_ab = ab_tests_df.copy()
        
        display_ab['Impressions'] = display_ab['impressions'].apply(lambda x: f"{x:,}")
        display_ab['Clicks'] = display_ab['clicks'].apply(lambda x: f"{x:,}")
        display_ab['Conversions'] = display_ab['conversions'].apply(lambda x: f"{x:,}")
        display_ab['CTR'] = display_ab['ctr'].apply(lambda x: f"{x:.1f}%")
        display_ab['CVR'] = display_ab['cvr'].apply(lambda x: f"{x:.1f}%")
        display_ab['Result'] = display_ab['winner'].apply(
            lambda x: "🏆 Winner ✓" if x else "📊 Variant"
        )
        
        st.dataframe(
            display_ab[['test', 'variant', 'Impressions', 'Clicks', 'Conversions', 'CTR', 'CVR', 'Result']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'test': 'Test Name',
                'variant': 'Variant'
            }
        )
        
        st.markdown("---")
        st.subheader("A/B Test Summary")
        
        test_summary = ab_tests_df.groupby('test').agg({
            'winner': lambda x: (x.sum(), len(x))
        }).reset_index()
        
        for _, row in ab_tests_df.drop_duplicates(subset=['test']).iterrows():
            test_data = ab_tests_df[ab_tests_df['test'] == row['test']]
            winner = test_data[test_data['winner'] == True]['variant'].values
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**{row['test']}**")
            with col2:
                if len(winner) > 0:
                    st.success(f"✓ Winner: {winner[0]}")
            with col3:
                winner_cvr = test_data[test_data['winner'] == True]['cvr'].values[0]
                loser_cvr = test_data[test_data['winner'] == False]['cvr'].values[0]
                improvement = ((winner_cvr - loser_cvr) / loser_cvr * 100)
                st.info(f"📈 {improvement:.1f}% improvement")
            st.divider()
    else:
        st.info("No A/B testing data available")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")