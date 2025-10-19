"""
📊 Customer Cohort Analysis - Cohort behavior tracking and retention analysis
Retention curves, cohort comparison, and lifecycle value metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Customer Cohort Analysis",
    page_icon="📊",
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
def generate_sample_cohort_data():
    """Generate comprehensive cohort data for last 24 months"""
    np.random.seed(42)
    
    # Last 12 months data
    cohorts_12m = pd.DataFrame([
        {'month': 'Jan 2024', 'year': 2024, 'size': 1567, 'm0': 100, 'm1': 78, 'm2': 72, 'm3': 68, 'm4': 65, 'm5': 62, 'clv': 2450, 'churn': 4.8, 'revenue': 3840000},
        {'month': 'Feb 2024', 'year': 2024, 'size': 1345, 'm0': 100, 'm1': 76, 'm2': 70, 'm3': 66, 'm4': 63, 'm5': 60, 'clv': 2380, 'churn': 5.2, 'revenue': 3201100},
        {'month': 'Mar 2024', 'year': 2024, 'size': 1456, 'm0': 100, 'm1': 79, 'm2': 73, 'm3': 69, 'm4': 66, 'm5': 63, 'clv': 2520, 'churn': 4.5, 'revenue': 3669120},
        {'month': 'Apr 2024', 'year': 2024, 'size': 1234, 'm0': 100, 'm1': 74, 'm2': 68, 'm3': 64, 'm4': 61, 'm5': 58, 'clv': 2290, 'churn': 5.6, 'revenue': 2825860},
        {'month': 'May 2024', 'year': 2024, 'size': 1398, 'm0': 100, 'm1': 80, 'm2': 74, 'm3': 70, 'm4': 67, 'm5': 64, 'clv': 2610, 'churn': 4.2, 'revenue': 3648780},
        {'month': 'Jun 2024', 'year': 2024, 'size': 1523, 'm0': 100, 'm1': 77, 'm2': 71, 'm3': 67, 'm4': 64, 'm5': 61, 'clv': 2470, 'churn': 4.9, 'revenue': 3761810},
        {'month': 'Jul 2024', 'year': 2024, 'size': 1612, 'm0': 100, 'm1': 81, 'm2': 75, 'm3': 71, 'm4': 68, 'm5': 65, 'clv': 2670, 'churn': 4.1, 'revenue': 4304040},
        {'month': 'Aug 2024', 'year': 2024, 'size': 1478, 'm0': 100, 'm1': 79, 'm2': 73, 'm3': 69, 'm4': 66, 'm5': 63, 'clv': 2540, 'churn': 4.6, 'revenue': 3754120},
        {'month': 'Sep 2024', 'year': 2024, 'size': 1589, 'm0': 100, 'm1': 82, 'm2': 76, 'm3': 72, 'm4': 69, 'm5': 66, 'clv': 2720, 'churn': 3.9, 'revenue': 4322080},
        {'month': 'Oct 2024', 'year': 2024, 'size': 1734, 'm0': 100, 'm1': 84, 'm2': 78, 'm3': 74, 'm4': 71, 'm5': 68, 'clv': 2890, 'churn': 3.5, 'revenue': 5011260},
        {'month': 'Nov 2024', 'year': 2024, 'size': 1823, 'm0': 100, 'm1': 85, 'm2': 79, 'm3': 75, 'm4': 72, 'm5': 69, 'clv': 2950, 'churn': 3.3, 'revenue': 5377850},
        {'month': 'Dec 2024', 'year': 2024, 'size': 1956, 'm0': 100, 'm1': 86, 'm2': 80, 'm3': 76, 'm4': 73, 'm5': 70, 'clv': 3010, 'churn': 3.1, 'revenue': 5885560}
    ])
    
    # Previous 12 months data (for 24 month view)
    cohorts_24m = pd.DataFrame([
        {'month': 'Jan 2023', 'year': 2023, 'size': 1234, 'm0': 100, 'm1': 70, 'm2': 64, 'm3': 60, 'm4': 57, 'm5': 54, 'clv': 2150, 'churn': 6.2, 'revenue': 2653100},
        {'month': 'Feb 2023', 'year': 2023, 'size': 1189, 'm0': 100, 'm1': 68, 'm2': 62, 'm3': 58, 'm4': 55, 'm5': 52, 'clv': 2090, 'churn': 6.5, 'revenue': 2485010},
        {'month': 'Mar 2023', 'year': 2023, 'size': 1298, 'm0': 100, 'm1': 72, 'm2': 66, 'm3': 62, 'm4': 59, 'm5': 56, 'clv': 2230, 'churn': 5.9, 'revenue': 2894540},
        {'month': 'Apr 2023', 'year': 2023, 'size': 1156, 'm0': 100, 'm1': 69, 'm2': 63, 'm3': 59, 'm4': 56, 'm5': 53, 'clv': 2120, 'churn': 6.4, 'revenue': 2450720},
        {'month': 'May 2023', 'year': 2023, 'size': 1267, 'm0': 100, 'm1': 73, 'm2': 67, 'm3': 63, 'm4': 60, 'm5': 57, 'clv': 2270, 'churn': 5.7, 'revenue': 2876090},
        {'month': 'Jun 2023', 'year': 2023, 'size': 1345, 'm0': 100, 'm1': 71, 'm2': 65, 'm3': 61, 'm4': 58, 'm5': 55, 'clv': 2190, 'churn': 6.1, 'revenue': 2945550},
        {'month': 'Jul 2023', 'year': 2023, 'size': 1423, 'm0': 100, 'm1': 74, 'm2': 68, 'm3': 64, 'm4': 61, 'm5': 58, 'clv': 2310, 'churn': 5.5, 'revenue': 3287130},
        {'month': 'Aug 2023', 'year': 2023, 'size': 1378, 'm0': 100, 'm1': 72, 'm2': 66, 'm3': 62, 'm4': 59, 'm5': 56, 'clv': 2250, 'churn': 5.8, 'revenue': 3100500},
        {'month': 'Sep 2023', 'year': 2023, 'size': 1456, 'm0': 100, 'm1': 75, 'm2': 69, 'm3': 65, 'm4': 62, 'm5': 59, 'clv': 2360, 'churn': 5.3, 'revenue': 3436160},
        {'month': 'Oct 2023', 'year': 2023, 'size': 1512, 'm0': 100, 'm1': 76, 'm2': 70, 'm3': 66, 'm4': 63, 'm5': 60, 'clv': 2410, 'churn': 5.1, 'revenue': 3643920},
        {'month': 'Nov 2023', 'year': 2023, 'size': 1589, 'm0': 100, 'm1': 77, 'm2': 71, 'm3': 67, 'm4': 64, 'm5': 61, 'clv': 2430, 'churn': 4.9, 'revenue': 3861270},
        {'month': 'Dec 2023', 'year': 2023, 'size': 1634, 'm0': 100, 'm1': 78, 'm2': 72, 'm3': 68, 'm4': 65, 'm5': 62, 'clv': 2460, 'churn': 4.7, 'revenue': 4019640}
    ])
    
    # Combine all data
    all_cohorts = pd.concat([cohorts_24m, cohorts_12m], ignore_index=True)
    
    return cohorts_12m, cohorts_24m, all_cohorts

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading cohort data..."):
    cohorts_12m, cohorts_24m, all_cohorts = generate_sample_cohort_data()

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 📊 Filter Settings")
    
    time_period = st.selectbox(
        "⏰ Time Period",
        ["Last 12 Months", "Last 24 Months"],
        index=0
    )
    
    cohort_type = st.selectbox(
        "👥 Cohort Type",
        ["Acquisition Month", "First Purchase"],
        index=0
    )
    
    metric_view = st.selectbox(
        "📈 Metric",
        ["Retention Rate", "Revenue", "CLV"],
        index=0
    )
    
    # Cohort selection filter
    if time_period == "Last 12 Months":
        available_cohorts = cohorts_12m['month'].tolist()
    else:
        available_cohorts = all_cohorts['month'].tolist()
    
    selected_cohorts = st.multiselect(
        "🎯 Select Specific Cohorts",
        options=available_cohorts,
        default=available_cohorts[:6] if len(available_cohorts) > 6 else available_cohorts
    )
    
    st.markdown("---")
    
    st.markdown("### 🔍 Quick Filters")
    show_top_performers = st.checkbox("Show Top Performers Only", value=False)
    show_improving = st.checkbox("Show Improving Cohorts", value=False)
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

# Select data based on time period
if time_period == "Last 12 Months":
    filtered_cohorts = cohorts_12m.copy()
else:
    filtered_cohorts = all_cohorts.copy()

# Filter by selected cohorts
if selected_cohorts:
    filtered_cohorts = filtered_cohorts[filtered_cohorts['month'].isin(selected_cohorts)]

# Apply quick filters
if show_top_performers:
    # Top 50% by CLV
    median_clv = filtered_cohorts['clv'].median()
    filtered_cohorts = filtered_cohorts[filtered_cohorts['clv'] >= median_clv]

if show_improving:
    # Cohorts with churn rate below average
    avg_churn = filtered_cohorts['churn'].mean()
    filtered_cohorts = filtered_cohorts[filtered_cohorts['churn'] < avg_churn]

# Calculate metrics from filtered data
total_cohorts = len(filtered_cohorts)
avg_retention = filtered_cohorts['m5'].mean() if len(filtered_cohorts) > 0 else 0
avg_cohort_size = filtered_cohorts['size'].mean() if len(filtered_cohorts) > 0 else 0
avg_churn_rate = filtered_cohorts['churn'].mean() if len(filtered_cohorts) > 0 else 0
total_customers = filtered_cohorts['size'].sum() if len(filtered_cohorts) > 0 else 0
total_revenue = filtered_cohorts['revenue'].sum() if len(filtered_cohorts) > 0 else 0
avg_clv = filtered_cohorts['clv'].mean() if len(filtered_cohorts) > 0 else 0

# ===========================
# HEADER & METRICS
# ===========================

st.title("📊 Customer Cohort Analysis")
st.markdown("**Cohort behavior tracking, retention analysis, and lifecycle value metrics**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Active Cohorts", f"{total_cohorts}", f"{time_period}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Avg Retention (M5)", f"{avg_retention:.1f}%", "+4.2% improvement")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Total Customers", f"{total_customers:,}", f"Across {total_cohorts} cohorts")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Avg Churn Rate", f"{avg_churn_rate:.1f}%", "-1.2% improvement")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Dynamic alert based on filters
filter_info = []
if time_period:
    filter_info.append(f"Period: {time_period}")
if metric_view:
    filter_info.append(f"Metric: {metric_view}")
if selected_cohorts and len(selected_cohorts) < len(available_cohorts):
    filter_info.append(f"{len(selected_cohorts)} cohorts selected")
if show_top_performers:
    filter_info.append("Top performers only")
if show_improving:
    filter_info.append("Improving cohorts only")

filter_text = " | ".join(filter_info) if filter_info else "All cohorts"

st.markdown(f"""
<div class="alert-success">
    <strong>🔍 Active Filters:</strong> {filter_text} | Showing {total_cohorts} cohort(s) with {total_customers:,} total customers | Average M5 retention: {avg_retention:.1f}% | Total revenue: ${total_revenue:,.0f}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Show warning if no data after filtering
if len(filtered_cohorts) == 0:
    st.warning("⚠️ No cohorts match your current filter criteria. Please adjust your filters.")
    st.stop()

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📈 {metric_view} Trend Over Time")
    
    # Determine which metric to show
    if metric_view == "Retention Rate":
        y_col = 'm1'
        y_label = 'Month 1 Retention %'
    elif metric_view == "Revenue":
        y_col = 'revenue'
        y_label = 'Revenue ($)'
    else:  # CLV
        y_col = 'clv'
        y_label = 'Customer Lifetime Value ($)'
    
    fig_trend = px.line(
        filtered_cohorts,
        x='month',
        y=y_col,
        markers=True,
        labels={y_col: y_label, 'month': 'Cohort'}
    )
    fig_trend.update_traces(line=dict(color='#22c55e', width=3), marker=dict(size=8))
    fig_trend.update_layout(height=350, showlegend=False, hovermode='x unified')
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.subheader("👥 Cohort Size Distribution")
    
    fig_size = px.bar(
        filtered_cohorts,
        x='month',
        y='size',
        color='size',
        color_continuous_scale='Blues',
        labels={'size': 'Cohort Size', 'month': 'Cohort'}
    )
    fig_size.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_size, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 CLV by Cohort")
    
    fig_clv = px.line(
        filtered_cohorts,
        x='month',
        y='clv',
        markers=True,
        labels={'clv': 'Customer Lifetime Value ($)', 'month': 'Cohort'}
    )
    fig_clv.update_traces(line=dict(color='#8b5cf6', width=3), marker=dict(size=8))
    fig_clv.update_layout(height=350, showlegend=False, hovermode='x unified')
    st.plotly_chart(fig_clv, use_container_width=True)

with col2:
    st.subheader("📉 Churn Rate by Cohort")
    
    fig_churn = px.bar(
        filtered_cohorts,
        x='month',
        y='churn',
        color='churn',
        color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444'],
        labels={'churn': 'Churn Rate %', 'month': 'Cohort'}
    )
    fig_churn.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_churn, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📊 Cohort Analysis ({total_cohorts})",
    "📈 Retention Curves",
    "🔍 Cohort Comparison",
    "💰 Lifecycle Value"
])

# TAB 1: COHORT ANALYSIS
with tab1:
    st.subheader("Cohort Retention Matrix")
    
    display_df = filtered_cohorts.copy()
    
    # Create retention matrix display
    retention_cols = ['month', 'size', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'clv', 'churn']
    display_cols = display_df[retention_cols].copy()
    display_cols.columns = ['Cohort', 'Size', 'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'CLV', 'Churn %']
    display_cols['CLV'] = display_cols['CLV'].apply(lambda x: f"${x:,}")
    
    # Format retention columns
    for col in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
        display_cols[col] = display_cols[col].apply(lambda x: f"{x}%")
    
    display_cols['Churn %'] = display_cols['Churn %'].apply(lambda x: f"{x}%")
    
    st.dataframe(
        display_cols,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🟢 **Green Zone:** >75% retention")
    with col2:
        st.info("🟡 **Yellow Zone:** 60-75% retention")
    with col3:
        st.info("🔴 **Red Zone:** <60% retention")

# TAB 2: RETENTION CURVES
with tab2:
    st.subheader("Retention Curves by Cohort")
    
    # Create retention curve visualization
    months = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']
    
    fig_curves = go.Figure()
    
    for _, cohort in filtered_cohorts.iterrows():
        retention_values = [cohort['m0'], cohort['m1'], cohort['m2'], cohort['m3'], cohort['m4'], cohort['m5']]
        fig_curves.add_trace(go.Scatter(
            x=months,
            y=retention_values,
            mode='lines+markers',
            name=cohort['month'],
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig_curves.update_layout(
        height=400,
        xaxis_title='Months Since Acquisition',
        yaxis_title='Retention Rate (%)',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    st.plotly_chart(fig_curves, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📊 Retention Statistics")
    
    cols = st.columns(6)
    for i, month in enumerate(['m0', 'm1', 'm2', 'm3', 'm4', 'm5']):
        with cols[i]:
            avg_val = filtered_cohorts[month].mean()
            st.metric(f"Avg M{i}", f"{avg_val:.1f}%")
    
    st.markdown("---")
    st.markdown("#### 🎯 Cohort Performance")
    
    # Best and worst performers
    best_cohort = filtered_cohorts.loc[filtered_cohorts['m5'].idxmax()]
    worst_cohort = filtered_cohorts.loc[filtered_cohorts['m5'].idxmin()]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Best Performer:** {best_cohort['month']} - {best_cohort['m5']:.1f}% M5 retention")
    with col2:
        st.warning(f"⚠️ **Needs Attention:** {worst_cohort['month']} - {worst_cohort['m5']:.1f}% M5 retention")

# TAB 3: COHORT COMPARISON
with tab3:
    st.subheader("Cohort Comparison")
    
    # Create comparison cards
    cols_per_row = 2
    cohort_list = filtered_cohorts.to_dict('records')
    
    for idx in range(0, len(cohort_list), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx, cohort in enumerate(cohort_list[idx:idx+cols_per_row]):
            avg_retention_cohort = (cohort['m1'] + cohort['m2'] + cohort['m3'] + cohort['m4'] + cohort['m5']) / 5
            
            with cols[col_idx]:
                # Determine performance color
                if avg_retention_cohort >= 70:
                    border_color = "#22c55e"
                elif avg_retention_cohort >= 60:
                    border_color = "#f59e0b"
                else:
                    border_color = "#ef4444"
                
                st.markdown(f"""
                <div style="padding: 20px; border: 2px solid {border_color}; border-radius: 10px; margin-bottom: 15px;">
                    <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 15px;">{cohort['month']}</div>
                    <div style="display: grid; gap: 10px;">
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0;">
                            <span style="color: #64748b; font-weight: 600;">Size</span>
                            <span style="font-size: 1.125rem; font-weight: 700;">{cohort['size']:,}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0;">
                            <span style="color: #64748b; font-weight: 600;">Avg Retention</span>
                            <span style="font-size: 1.125rem; font-weight: 700; color: {border_color};">{avg_retention_cohort:.1f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0;">
                            <span style="color: #64748b; font-weight: 600;">CLV</span>
                            <span style="font-size: 1.125rem; font-weight: 700;">${cohort['clv']:,}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0;">
                            <span style="color: #64748b; font-weight: 600;">Revenue</span>
                            <span style="font-size: 1.125rem; font-weight: 700;">${cohort['revenue']:,}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 8px 0;">
                            <span style="color: #64748b; font-weight: 600;">Churn Rate</span>
                            <span style="font-size: 1.125rem; font-weight: 700; color: #ef4444;">{cohort['churn']}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# TAB 4: LIFECYCLE VALUE
with tab4:
    st.subheader("Customer Lifecycle Value Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #8b5cf6, #6366f1); color: white; padding: 30px; border-radius: 10px; text-align: center;">
            <div style="font-size: 1.125rem; opacity: 0.9; margin-bottom: 15px;">Average Customer Lifetime Value</div>
            <div style="font-size: 3.5rem; font-weight: 900; margin: 15px 0;">${avg_clv:,.0f}</div>
            <div style="opacity: 0.8;">Across {total_cohorts} cohorts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 CLV Statistics")
        
        max_clv = filtered_cohorts['clv'].max()
        min_clv = filtered_cohorts['clv'].min()
        total_clv = (filtered_cohorts['clv'] * filtered_cohorts['size']).sum()
        
        st.metric("Average CLV", f"${avg_clv:,.0f}")
        st.metric("Highest CLV", f"${max_clv:,}", delta="Top performer")
        st.metric("Lowest CLV", f"${min_clv:,}", delta="Needs optimization")
        st.metric("Total CLV", f"${total_clv:,.0f}", delta="All cohorts combined")
    
    st.markdown("---")
    st.markdown("#### 💰 CLV by Cohort")
    
    clv_display = filtered_cohorts.copy()
    clv_display['Total Value'] = clv_display['size'] * clv_display['clv']
    clv_display['CLV'] = clv_display['clv'].apply(lambda x: f"${x:,}")
    clv_display['Total Value'] = clv_display['Total Value'].apply(lambda x: f"${x:,.0f}")
    clv_display['Revenue'] = clv_display['revenue'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(
        clv_display[['month', 'size', 'CLV', 'Revenue', 'Total Value']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'month': 'Cohort',
            'size': 'Size',
            'CLV': st.column_config.TextColumn('CLV per Customer'),
            'Revenue': 'Total Revenue',
            'Total Value': 'Total Cohort Value'
        }
    )
    
    st.markdown("---")
    st.markdown("#### 📈 CLV Trend Analysis")
    
    fig_clv_trend = go.Figure()
    
    fig_clv_trend.add_trace(go.Scatter(
        x=filtered_cohorts['month'],
        y=filtered_cohorts['clv'],
        mode='lines+markers',
        name='CLV',
        line=dict(color='#8b5cf6', width=3),
        marker=dict(size=10),
        yaxis='y'
    ))
    
    fig_clv_trend.add_trace(go.Scatter(
        x=filtered_cohorts['month'],
        y=filtered_cohorts['churn'],
        mode='lines+markers',
        name='Churn Rate',
        line=dict(color='#ef4444', width=2, dash='dash'),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig_clv_trend.update_layout(
        height=400,
        xaxis=dict(title='Cohort'),
        yaxis=dict(title='CLV ($)', side='left'),
        yaxis2=dict(title='Churn Rate (%)', side='right', overlaying='y'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_clv_trend, use_container_width=True)

st.markdown("---")

# ===========================
# INSIGHTS & RECOMMENDATIONS
# ===========================

with st.expander("💡 Cohort Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Performance Summary")
        st.markdown(f"""
        - **Total Cohorts Analyzed:** {total_cohorts}
        - **Total Customers:** {total_customers:,}
        - **Average CLV:** ${avg_clv:,.0f}
        - **Average M5 Retention:** {avg_retention:.1f}%
        - **Average Churn Rate:** {avg_churn_rate:.1f}%
        - **Total Revenue:** ${total_revenue:,.0f}
        """)
        
        # Calculate trends
        if len(filtered_cohorts) >= 2:
            first_cohort_clv = filtered_cohorts.iloc[0]['clv']
            last_cohort_clv = filtered_cohorts.iloc[-1]['clv']
            clv_trend = ((last_cohort_clv - first_cohort_clv) / first_cohort_clv * 100)
            
            st.markdown(f"""
            - **CLV Trend:** {clv_trend:+.1f}% from first to last cohort
            """)
    
    with col2:
        st.markdown("#### 🎯 Recommendations")
        
        # Generate dynamic recommendations based on data
        recommendations = []
        
        if avg_churn_rate > 5.0:
            recommendations.append("🔴 **High Churn Alert:** Churn rate is above 5%. Implement retention strategies.")
        
        if avg_retention < 65:
            recommendations.append("🟡 **Low Retention:** M5 retention below 65%. Focus on onboarding improvements.")
        
        best_cohort = filtered_cohorts.loc[filtered_cohorts['clv'].idxmax()]
        recommendations.append(f"🟢 **Best Practice:** Study {best_cohort['month']} cohort (CLV: ${best_cohort['clv']:,}) for success factors.")
        
        if len(filtered_cohorts) >= 3:
            recent_trend = filtered_cohorts.tail(3)['m1'].mean()
            older_trend = filtered_cohorts.head(3)['m1'].mean() if len(filtered_cohorts) >= 6 else filtered_cohorts['m1'].mean()
            
            if recent_trend > older_trend:
                recommendations.append("🟢 **Positive Trend:** Recent cohorts showing better M1 retention. Continue current strategies.")
            else:
                recommendations.append("🟡 **Declining Trend:** Recent cohorts have lower M1 retention. Review acquisition quality.")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        st.markdown("""
        
        **Action Items:**
        1. Analyze top-performing cohorts for success patterns
        2. Implement targeted re-engagement for at-risk cohorts
        3. Optimize onboarding for new cohorts
        4. Monitor churn triggers in month 1-3
        """)

# ===========================
# FILTER SUMMARY
# ===========================

with st.expander("🔍 Active Filter Summary"):
    st.markdown("### Current Filter Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Applied Filters")
        st.markdown(f"""
        - **Time Period:** {time_period}
        - **Cohort Type:** {cohort_type}
        - **Metric View:** {metric_view}
        - **Selected Cohorts:** {len(selected_cohorts)} of {len(available_cohorts)}
        - **Top Performers Only:** {'✅ Yes' if show_top_performers else '❌ No'}
        - **Improving Cohorts:** {'✅ Yes' if show_improving else '❌ No'}
        """)
    
    with col2:
        st.markdown("#### Results")
        st.markdown(f"""
        - **Cohorts Displayed:** {total_cohorts}
        - **Total Customers:** {total_customers:,}
        - **Data Points:** {total_cohorts * 6} (retention rates)
        - **Date Range:** {filtered_cohorts['month'].iloc[0]} to {filtered_cohorts['month'].iloc[-1]}
        """)
    
    st.markdown("---")
    st.markdown("#### 🎯 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 View All Cohorts", use_container_width=True, key="view_all_cohorts"):
            st.info("💡 Tip: Deselect all filters and select all cohorts in the sidebar")
    
    with col2:
        if st.button("🏆 Top 5 Cohorts", use_container_width=True, key="top_5"):
            st.info("💡 Tip: Enable 'Show Top Performers Only' filter")
    
    with col3:
        if st.button("📈 Recent Cohorts", use_container_width=True, key="recent"):
            st.info("💡 Tip: Select only the last 6 months in cohort selector")
    
    with col4:
        if st.button("💰 High CLV", use_container_width=True, key="high_clv"):
            st.info("💡 Tip: Sort by CLV in the Lifecycle Value tab")

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_cohorts"):
        st.cache_data.clear()
        st.success("✅ Cohort data refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True, key="export_cohorts"):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'filters': {
                'time_period': time_period,
                'cohort_type': cohort_type,
                'metric_view': metric_view,
                'selected_cohorts': selected_cohorts,
                'show_top_performers': show_top_performers,
                'show_improving': show_improving
            },
            'summary': {
                'total_cohorts': total_cohorts,
                'total_customers': total_customers,
                'avg_retention': f"{avg_retention:.1f}%",
                'avg_clv': f"${avg_clv:,.0f}",
                'avg_churn_rate': f"{avg_churn_rate:.1f}%",
                'total_revenue': f"${total_revenue:,.0f}"
            },
            'cohort_data': filtered_cohorts.to_dict('records')
        }
        st.success("✅ Cohort data exported successfully")
        st.json(export_data)

with col3:
    st.caption(f"📊 Displaying {total_cohorts} cohort(s) | {total_customers:,} customers | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")

# ===========================
# FOOTER
# ===========================

st.markdown("""
<div style="text-align:center;color:#64748b;font-size:0.875rem;padding:20px;border-top:1px solid #e2e8f0;">
    <strong>Customer Cohort Analysis System</strong> | 
    Real-time cohort tracking & retention analytics | 
    <a href="#" style="color:#3b82f6;text-decoration:none;">Documentation</a> | 
    <a href="#" style="color:#3b82f6;text-decoration:none;">Support</a>
</div>
""", unsafe_allow_html=True)