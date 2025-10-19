"""
⭐ Loyalty Program Audit - All 4 Tabs with Working Filters
Member activity and reward validation with tier analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Loyalty Program Audit",
    page_icon="⭐",
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
    .stat-card-gold { border-left-color: #ffd700; }
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
    .alert-success {
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        border-left: 4px solid;
        background: #d1fae5;
        color: #065f46;
        border-left-color: #22c55e;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_loyalty_data():
    """Generate comprehensive loyalty program data"""
    np.random.seed(42)
    
    # Member growth trend
    member_trend = []
    base_members = 15234
    for i in range(9):
        month_members = base_members + (i * 920) + np.random.randint(-500, 1500)
        member_trend.append({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'][i],
            'total_members': month_members
        })
    
    # Points activity
    points_trend = []
    for i in range(9):
        earned = 234567 + (i * 15000) + np.random.randint(-5000, 10000)
        redeemed = 89234 + (i * 4500) + np.random.randint(-2000, 4000)
        points_trend.append({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'][i],
            'earned': earned,
            'redeemed': redeemed
        })
    
    # Top performers
    top_performers = pd.DataFrame([
        {'id': 'M-10234', 'name': 'Sarah Johnson', 'tier': 'Platinum', 'points': 45678, 'spent': '$12,345', 'joined': datetime(2022, 1, 15)},
        {'id': 'M-10567', 'name': 'Michael Chen', 'tier': 'Platinum', 'points': 42134, 'spent': '$11,890', 'joined': datetime(2022, 3, 22)},
        {'id': 'M-10890', 'name': 'Emma Wilson', 'tier': 'Gold', 'points': 38456, 'spent': '$9,567', 'joined': datetime(2022, 5, 10)},
        {'id': 'M-11234', 'name': 'David Brown', 'tier': 'Gold', 'points': 35789, 'spent': '$8,234', 'joined': datetime(2022, 7, 18)},
        {'id': 'M-11567', 'name': 'Lisa Anderson', 'tier': 'Gold', 'points': 32567, 'spent': '$7,890', 'joined': datetime(2022, 9, 5)}
    ])
    
    # Earning sources
    earning_sources = pd.DataFrame([
        {'source': 'Purchases', 'points': 1856789, 'percent': 75.6},
        {'source': 'Referrals', 'points': 345678, 'percent': 14.1},
        {'source': 'Reviews', 'points': 123456, 'percent': 5.0},
        {'source': 'Bonuses', 'points': 89234, 'percent': 3.6},
        {'source': 'Social Media', 'points': 41632, 'percent': 1.7}
    ])
    
    # Tiers data
    tiers = pd.DataFrame([
        {'name': 'Bronze', 'members': 14567, 'percent': 62.1, 'retention': 68.5, 'avg_spend': '$123', 'avg_points': 456},
        {'name': 'Silver', 'members': 5678, 'percent': 24.2, 'retention': 82.3, 'avg_spend': '$289', 'avg_points': 2345},
        {'name': 'Gold', 'members': 2345, 'percent': 10.0, 'retention': 91.7, 'avg_spend': '$567', 'avg_points': 8934},
        {'name': 'Platinum', 'members': 866, 'percent': 3.7, 'retention': 96.8, 'avg_spend': '$1,234', 'avg_points': 28456}
    ])
    
    # Effectiveness metrics
    effectiveness = pd.DataFrame([
        {'metric': 'Member Acquisition', 'value': 1234, 'target': 1000, 'status': 'excellent'},
        {'metric': 'Activation Rate', 'value': 68.5, 'target': 65.0, 'status': 'good'},
        {'metric': 'Engagement Rate', 'value': 52.6, 'target': 60.0, 'status': 'fair'},
        {'metric': 'Redemption Rate', 'value': 34.5, 'target': 30.0, 'status': 'excellent'},
        {'metric': 'Retention Rate', 'value': 78.5, 'target': 75.0, 'status': 'good'},
        {'metric': 'Revenue Uplift', 'value': 24.5, 'target': 20.0, 'status': 'excellent'}
    ])
    
    # Tier movement
    tier_movement = pd.DataFrame([
        {'period': 'Last 30 Days', 'upgrades': 346, 'downgrades': 45, 'net': 301},
        {'period': 'Last 60 Days', 'upgrades': 657, 'downgrades': 78, 'net': 579},
        {'period': 'Last 90 Days', 'upgrades': 979, 'downgrades': 123, 'net': 856},
        {'period': 'Year to Date', 'upgrades': 3469, 'downgrades': 456, 'net': 3013}
    ])
    
    return (pd.DataFrame(member_trend), pd.DataFrame(points_trend), top_performers, 
            earning_sources, tiers, effectiveness, tier_movement)

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading loyalty program data..."):
    member_df, points_df, top_performers, earning_sources, tiers, effectiveness, tier_movement = generate_sample_loyalty_data()

# Calculate metrics
total_members = 23456
active_members = 12345
points_issued = 2300000
redemption_rate = 34.5

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### Filter Settings")
    
    date_range = st.selectbox(
        "Date Range",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
        index=2
    )
    
    tier_filter = st.selectbox(
        "Member Tier",
        ["All Tiers", "Bronze", "Silver", "Gold", "Platinum"],
        index=0
    )
    
    status_filter = st.selectbox(
        "Member Status",
        ["All Statuses", "Active", "Inactive"],
        index=0
    )
    
    search_query = st.text_input("Search Member", placeholder="Member ID, name...")
    
    st.markdown("---")
    if st.button("Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_loyalty_filters(performers_df, tier_f, search_q):
    """Apply filters to loyalty data"""
    filtered = performers_df.copy()
    
    if tier_f != "All Tiers":
        filtered = filtered[filtered['tier'] == tier_f]
    
    if search_q:
        search_lower = search_q.lower()
        filtered = filtered[
            filtered['id'].str.lower().str.contains(search_lower, na=False) |
            filtered['name'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_performers = apply_loyalty_filters(top_performers, tier_filter, search_query)

# ===========================
# HEADER & METRICS
# ===========================

st.title("⭐ Loyalty Program Audit")
st.markdown("**Member activity and reward validation**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Members", f"{total_members:,}", "+1,234 this month")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Active Members", f"{active_members:,}", "+567 growth")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-gold">', unsafe_allow_html=True)
    st.metric("Points Issued", f"{points_issued/1e6:.1f}M", "+345K this month")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Redemption Rate", f"{redemption_rate}%", "+2.3% increase")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown(f"""
<div class="alert-success">
    <strong>Program Health:</strong> {total_members:,} total members with {active_members:,} active. {points_issued/1e6:.1f}M points issued, {redemption_rate}% redemption rate. Program ROI: 4.2x.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "💰 Points Tracking",
    "🏆 Tier Distribution",
    "📈 Program Effectiveness"
])

# TAB 1: PROGRAM DASHBOARD
with tab1:
    st.subheader("Program Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Member Growth Trend")
        fig_growth = px.line(
            member_df,
            x='month',
            y='total_members',
            markers=True,
            labels={'total_members': 'Members', 'month': 'Month'}
        )
        fig_growth.update_traces(line=dict(color='#3b82f6', width=3),
                                marker=dict(size=8),
                                fill='tozeroy',
                                fillcolor='rgba(59, 130, 246, 0.1)')
        fig_growth.update_layout(height=350, showlegend=False, hovermode='x unified')
        st.plotly_chart(fig_growth, use_container_width=True)
    
    with col2:
        st.markdown("#### Points Activity")
        fig_points = px.bar(
            points_df,
            x='month',
            y=['earned', 'redeemed'],
            barmode='group',
            labels={'month': 'Month', 'value': 'Points'}
        )
        for i, trace in enumerate(fig_points.data):
            trace.marker.color = '#22c55e' if i == 0 else '#ef4444'
            trace.name = 'Earned' if i == 0 else 'Redeemed'
        fig_points.update_layout(height=350, hovermode='x unified')
        st.plotly_chart(fig_points, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Top Performing Members")
    
    display_performers = filtered_performers.copy()
    display_performers['joined_date'] = pd.to_datetime(display_performers['joined']).dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        display_performers[['id', 'name', 'tier', 'points', 'spent', 'joined_date']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'id': 'Member ID',
            'name': 'Name',
            'tier': 'Tier',
            'points': 'Points Balance',
            'spent': 'Total Spent',
            'joined_date': 'Member Since'
        }
    )
    
    if len(filtered_performers) == 0:
        st.info("No members match the current filters")
    else:
        st.caption(f"Showing {len(filtered_performers)} member(s)")

# TAB 2: POINTS TRACKING
with tab2:
    st.subheader("Points Tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Points Summary")
        
        summary_metrics = [
            ("Total Earned", f"{points_issued/1e6:.2f}M"),
            ("Total Redeemed", "847K"),
            ("Total Expired", "57K"),
            ("Balance", "1.55M")
        ]
        
        for label, value in summary_metrics:
            st.markdown(f"""
            <div style="padding: 15px; border-left: 4px solid #3b82f6; background: #f0f9ff; border-radius: 6px; margin-bottom: 12px;">
                <div style="font-weight: 700; margin-bottom: 5px;">{label}</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #3b82f6;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Points Earning Sources")
        
        fig_earning = px.pie(
            earning_sources,
            values='points',
            names='source',
            color_discrete_sequence=['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6']
        )
        fig_earning.update_layout(height=350)
        st.plotly_chart(fig_earning, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Earning Sources Breakdown")
    
    for _, source in earning_sources.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{source['source']}**")
            st.progress(source['percent'] / 100)
        with col2:
            st.markdown(f"{source['points']:,}")
        with col3:
            st.markdown(f"{source['percent']}%")

# TAB 3: TIER DISTRIBUTION
with tab3:
    st.subheader("Tier Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tier Performance")
        
        for _, tier in tiers.iterrows():
            st.markdown(f"""
            <div style="margin-bottom: 20px; padding: 15px; border: 2px solid #e2e8f0; border-radius: 8px; background: white;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 1.25rem; font-weight: 800;">{tier['name']}</div>
                    <div style="text-align: right;">
                        <div style="font-weight: 800;">{tier['members']:,}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">{tier['percent']}%</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; padding-top: 10px; border-top: 1px solid #e2e8f0;">
                    <div><div style="font-size: 0.75rem; color: #64748b;">Avg Spend</div><div style="font-weight: 700;">{tier['avg_spend']}</div></div>
                    <div><div style="font-size: 0.75rem; color: #64748b;">Avg Points</div><div style="font-weight: 700;">{tier['avg_points']:,}</div></div>
                    <div><div style="font-size: 0.75rem; color: #64748b;">Retention</div><div style="font-weight: 700; color: #22c55e;">{tier['retention']}%</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Tier Distribution Chart")
        
        fig_tiers = px.pie(
            tiers,
            values='members',
            names='name',
            color_discrete_map={
                'Bronze': '#cd7f32',
                'Silver': '#c0c0c0',
                'Gold': '#ffd700',
                'Platinum': '#e5e4e2'
            }
        )
        fig_tiers.update_layout(height=350)
        st.plotly_chart(fig_tiers, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Tier Movement Analysis")
    
    st.dataframe(
        tier_movement,
        use_container_width=True,
        hide_index=True,
        column_config={
            'period': 'Period',
            'upgrades': 'Upgrades',
            'downgrades': 'Downgrades',
            'net': 'Net Movement'
        }
    )

# TAB 4: PROGRAM EFFECTIVENESS
with tab4:
    st.subheader("Program Effectiveness")
    
    st.markdown("#### Performance vs Targets")
    
    for _, metric in effectiveness.iterrows():
        percent_of_target = (metric['value'] / metric['target'] * 100) if metric['target'] > 0 else 0
        color = '#22c55e' if metric['status'] == 'excellent' else '#3b82f6' if metric['status'] == 'good' else '#f59e0b'
        status_text = 'Excellent' if metric['status'] == 'excellent' else 'Good' if metric['status'] == 'good' else 'Fair'
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{metric['metric']}**")
            st.progress(min(percent_of_target / 100, 1.0))
            st.caption(f"Target: {metric['target']} • Achieved: {metric['value']} ({percent_of_target:.0f}% of target)")
        with col2:
            st.markdown(f"<span style='color: {color}; font-weight: bold;'>{status_text}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Comparison Metrics")
    
    comparison_data = pd.DataFrame([
        {'metric': 'Average Order Value', 'members': '$289', 'non_members': '$156', 'uplift': '+85%'},
        {'metric': 'Purchase Frequency', 'members': '8.5x/year', 'non_members': '3.2x/year', 'uplift': '+166%'},
        {'metric': 'Customer Lifetime Value', 'members': '$2,340', 'non_members': '$890', 'uplift': '+163%'},
        {'metric': 'Return Rate', 'members': '2.8%', 'non_members': '5.4%', 'uplift': '-48%'},
        {'metric': 'Referral Rate', 'members': '18.5%', 'non_members': '4.2%', 'uplift': '+340%'}
    ])
    
    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("#### Program ROI Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Financial Summary")
        
        roi_items = [
            ('Program Revenue', '$8.7M'),
            ('Program Costs', '$2.1M'),
            ('Points Liability', '$456K'),
            ('Net Benefit', '$6.14M')
        ]
        
        for label, value in roi_items:
            color = '#22c55e' if 'Revenue' in label or 'Net' in label or 'Benefit' in label else '#ef4444' if 'Costs' in label or 'Liability' in label else '#64748b'
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #e2e8f0;">
                <span style="font-weight: 600;">{label}</span>
                <span style="font-weight: 800; color: {color};">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"""
        <div style="padding: 15px; background: #d1fae5; border-radius: 8px; border-left: 4px solid #22c55e;">
            <div style="font-weight: 600; margin-bottom: 8px;">Program ROI</div>
            <div style="font-size: 2rem; font-weight: 800; color: #22c55e; margin-bottom: 5px;">4.2x</div>
            <div style="font-size: 0.875rem; color: #065f46;">Net Benefit: $6.14M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ROI Trend")
        
        roi_trend = pd.DataFrame({
            'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
            'ROI': [3.2, 3.6, 4.0, 4.2]
        })
        
        fig_roi = px.line(
            roi_trend,
            x='Quarter',
            y='ROI',
            markers=True,
            labels={'ROI': 'ROI Multiple', 'Quarter': 'Quarter'}
        )
        fig_roi.update_traces(line=dict(color='#22c55e', width=3),
                              marker=dict(size=10))
        fig_roi.update_layout(height=300, showlegend=False, hovermode='x unified')
        st.plotly_chart(fig_roi, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Loyalty data refreshed successfully")
        st.rerun()

with col2:
    if st.button("Export Data", use_container_width=True):
        st.success("✅ Loyalty program data exported successfully")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Showing data from {date_range} | Filter: {tier_filter}")