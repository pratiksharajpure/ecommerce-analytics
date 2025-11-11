"""
📅 Enhanced Seasonal Trend Analysis - All 4 Tabs with Working Filters
Exact match to seasonality.html with YoY comparison and holiday performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Seasonal Trend Analysis",
    page_icon="📅",
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
    .alert-success {
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
def generate_sample_seasonality_data():
    """Generate sample seasonality data with multiple years"""
    np.random.seed(42)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 2024 Data
    months_2024 = []
    base_revenue_2024 = [245678, 267890, 289456, 312345, 334567, 356789, 398765, 378901, 412345, 489123, 678901, 789456]
    base_orders_2024 = [1890, 2045, 2234, 2456, 2678, 2890, 3234, 3056, 3345, 3876, 5234, 6123]
    
    for i, month in enumerate(month_names):
        months_2024.append({
            'month': month,
            'year': 2024,
            'quarter': f'Q{(i//3)+1}',
            'season': ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Holiday', 'Holiday'][i],
            'revenue': base_revenue_2024[i],
            'orders': base_orders_2024[i],
            'customers': int(base_orders_2024[i] * 0.65),
            'aov': round(base_revenue_2024[i] / base_orders_2024[i], 2)
        })
    
    # 2023 Data
    months_2023 = []
    base_revenue_2023 = [198765, 212345, 234567, 256789, 278901, 298765, 334567, 312345, 356789, 412345, 567890, 678901]
    base_orders_2023 = [1567, 1678, 1876, 2056, 2245, 2456, 2756, 2567, 2934, 3345, 4456, 5234]
    
    for i, month in enumerate(month_names):
        months_2023.append({
            'month': month,
            'year': 2023,
            'quarter': f'Q{(i//3)+1}',
            'season': ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Holiday', 'Holiday'][i],
            'revenue': base_revenue_2023[i],
            'orders': base_orders_2023[i],
            'customers': int(base_orders_2023[i] * 0.65),
            'aov': round(base_revenue_2023[i] / base_orders_2023[i], 2)
        })
    
    # 2022 Data
    months_2022 = []
    base_revenue_2022 = [178765, 192345, 214567, 236789, 248901, 268765, 304567, 282345, 326789, 382345, 497890, 598901]
    base_orders_2022 = [1467, 1578, 1776, 1956, 2145, 2356, 2656, 2467, 2834, 3245, 4156, 4934]
    
    for i, month in enumerate(month_names):
        months_2022.append({
            'month': month,
            'year': 2022,
            'quarter': f'Q{(i//3)+1}',
            'season': ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Holiday', 'Holiday'][i],
            'revenue': base_revenue_2022[i],
            'orders': base_orders_2022[i],
            'customers': int(base_orders_2022[i] * 0.65),
            'aov': round(base_revenue_2022[i] / base_orders_2022[i], 2)
        })
    
    # Quarterly Data
    quarters = [
        {'quarter': 'Q1 2024', 'year': 2024, 'revenue': 802924, 'growth': 14.2, 'orders': 6169, 'customers': 4035},
        {'quarter': 'Q2 2024', 'year': 2024, 'revenue': 1003701, 'growth': 15.8, 'orders': 8024, 'customers': 5034},
        {'quarter': 'Q3 2024', 'year': 2024, 'revenue': 1190011, 'growth': 18.5, 'orders': 9635, 'customers': 5908},
        {'quarter': 'Q4 2024', 'year': 2024, 'revenue': 1957480, 'growth': 21.3, 'orders': 15233, 'customers': 9368}
    ]
    
    # Holiday Performance
    holidays = [
        {'name': 'Black Friday', 'date': 'Nov 29, 2024', 'revenue': 456789, 'orders': 3456, 'aov': 132, 'growth': 45.3, 'conversion': 8.9, 'traffic': 234},
        {'name': 'Cyber Monday', 'date': 'Dec 2, 2024', 'revenue': 512345, 'orders': 3890, 'aov': 132, 'growth': 52.8, 'conversion': 9.2, 'traffic': 267},
        {'name': 'Christmas', 'date': 'Dec 25, 2024', 'revenue': 389456, 'orders': 2987, 'aov': 130, 'growth': 38.7, 'conversion': 7.8, 'traffic': 198},
        {'name': "New Year", 'date': 'Jan 1, 2024', 'revenue': 234567, 'orders': 1876, 'aov': 125, 'growth': 28.4, 'conversion': 6.9, 'traffic': 156},
        {'name': "Valentine's Day", 'date': 'Feb 14, 2024', 'revenue': 198765, 'orders': 1567, 'aov': 127, 'growth': 32.1, 'conversion': 7.2, 'traffic': 145},
        {'name': "Mother's Day", 'date': 'May 12, 2024', 'revenue': 267890, 'orders': 2123, 'aov': 126, 'growth': 35.6, 'conversion': 7.5, 'traffic': 178}
    ]
    
    # Forecast Data
    forecast = {
        'next_quarter': {
            'revenue': 2134567,
            'confidence': 92,
            'range_low': 1967234,
            'range_high': 2301890,
            'orders': 16789,
            'customers': 10234
        },
        'next_year': {
            'revenue': 7234567,
            'confidence': 85,
            'range_low': 6789234,
            'range_high': 7679890,
            'orders': 58934,
            'customers': 36789
        }
    }
    
    all_months = months_2024 + months_2023 + months_2022
    
    return (pd.DataFrame(months_2024), pd.DataFrame(months_2023), pd.DataFrame(months_2022),
            pd.DataFrame(all_months), pd.DataFrame(quarters), pd.DataFrame(holidays), forecast)

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading seasonality data..."):
    months_2024_df, months_2023_df, months_2022_df, all_months_df, quarters_df, holidays_df, forecast_data = generate_sample_seasonality_data()

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 📅 Filters")
    
    analysis_period = st.selectbox(
        "📊 Analysis Period",
        ["Last Year", "Last 2 Years", "Last 3 Years"],
        index=1
    )
    
    metric_type = st.selectbox(
        "💰 Metric Type",
        ["Revenue", "Orders", "Customers", "Average Order Value"]
    )
    
    season_filter = st.selectbox(
        "🔄 Season",
        ["All Seasons", "Q1 (Winter)", "Q2 (Spring)", "Q3 (Summer)", "Q4 (Fall/Holiday)"]
    )
    
    comparison_year = st.selectbox(
        "📈 Comparison",
        ["2024 vs 2023", "2023 vs 2022"]
    )
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True, key="reset_filters_season"):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_season_filters(df, period, season):
    filtered = df.copy()
    
    if period == "Last Year":
        filtered = filtered[filtered['year'] == 2024]
    elif period == "Last 2 Years":
        filtered = filtered[filtered['year'].isin([2024, 2023])]
    elif period == "Last 3 Years":
        filtered = filtered[filtered['year'].isin([2024, 2023, 2022])]
    
    if season != "All Seasons":
        season_map = {
            "Q1 (Winter)": "Q1",
            "Q2 (Spring)": "Q2",
            "Q3 (Summer)": "Q3",
            "Q4 (Fall/Holiday)": "Q4"
        }
        if season in season_map:
            filtered = filtered[filtered['quarter'] == season_map[season]]
    
    return filtered

filtered_months = apply_season_filters(all_months_df, analysis_period, season_filter)

if comparison_year == "2024 vs 2023":
    current_year_df = months_2024_df
    previous_year_df = months_2023_df
    current_year = 2024
    previous_year = 2023
else:
    current_year_df = months_2023_df
    previous_year_df = months_2022_df
    current_year = 2023
    previous_year = 2022

if season_filter != "All Seasons":
    season_map = {
        "Q1 (Winter)": "Q1",
        "Q2 (Spring)": "Q2",
        "Q3 (Summer)": "Q3",
        "Q4 (Fall/Holiday)": "Q4"
    }
    if season_filter in season_map:
        quarter = season_map[season_filter]
        current_year_df = current_year_df[current_year_df['quarter'] == quarter]
        previous_year_df = previous_year_df[previous_year_df['quarter'] == quarter]

# ===========================
# CALCULATE METRICS
# ===========================

metric_column_map = {
    "Revenue": "revenue",
    "Orders": "orders",
    "Customers": "customers",
    "Average Order Value": "aov"
}
selected_metric = metric_column_map[metric_type]

total_current = current_year_df[selected_metric].sum() if selected_metric != 'aov' else current_year_df[selected_metric].mean()
total_previous = previous_year_df[selected_metric].sum() if selected_metric != 'aov' else previous_year_df[selected_metric].mean()
yoy_growth = ((total_current - total_previous) / total_previous * 100) if total_previous > 0 else 0

peak_value = filtered_months[selected_metric].max()
off_season_value = filtered_months[selected_metric].min()
seasonal_variability = (filtered_months[selected_metric].std() / filtered_months[selected_metric].mean() * 100) if filtered_months[selected_metric].mean() > 0 else 0

# ===========================
# HEADER & METRICS
# ===========================

st.title("📅 Seasonal Trend Analysis")
st.markdown("**Seasonal patterns, year-over-year comparison, forecasting, and holiday performance**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Seasonal Patterns", "12", "+2 new patterns identified")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    if selected_metric == 'revenue':
        st.metric("Peak Season Revenue", f"${peak_value:,.0f}", "+$345K vs last year")
    else:
        st.metric(f"Peak {metric_type}", f"{peak_value:,.0f}", "+12% vs last year")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    if selected_metric == 'revenue':
        st.metric("Off-Season Revenue", f"${off_season_value:,.0f}", "+$123K improvement")
    else:
        st.metric(f"Off-Season {metric_type}", f"{off_season_value:,.0f}", "+8% improvement")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Seasonal Variability", f"{seasonal_variability:.1f}%", "-3.2% more stable")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

st.markdown(f"""
<div class="alert alert-success">
    <strong>Seasonal Insights:</strong> {season_filter} selected. Analyzing {metric_type} trends for {analysis_period}. YoY Growth: {yoy_growth:+.1f}%. Viewing {comparison_year} comparison.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Monthly {metric_type} Trend ({comparison_year})")
    
    fig1 = go.Figure()
    
    y_current = current_year_df[selected_metric]
    y_previous = previous_year_df[selected_metric]
    
    fig1.add_trace(go.Scatter(
        x=current_year_df['month'], 
        y=y_current,
        mode='lines+markers', 
        name=str(current_year),
        line=dict(color='#3b82f6', width=3),
        fill='tozeroy', 
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    fig1.add_trace(go.Scatter(
        x=previous_year_df['month'], 
        y=y_previous,
        mode='lines+markers', 
        name=str(previous_year),
        line=dict(color='#94a3b8', width=2, dash='dash'),
        fill='tozeroy', 
        fillcolor='rgba(148, 163, 184, 0.1)'
    ))
    fig1.update_layout(
        height=400, 
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title=metric_type
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Quarterly Performance")
    
    display_quarters = quarters_df.copy()
    if season_filter != "All Seasons":
        quarter_num = season_filter[1]
        display_quarters = display_quarters[display_quarters['quarter'].str.contains(f'Q{quarter_num}')]
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig2 = px.bar(
            display_quarters, 
            x='quarter', 
            y='revenue',
            title='Quarterly Revenue 2024',
            color='quarter',
            color_discrete_sequence=['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6'],
            text='revenue'
        )
        fig2.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig2.update_layout(height=300, showlegend=False, xaxis_title='', yaxis_title='Revenue')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_b:
        fig3 = px.pie(
            display_quarters, 
            values='revenue', 
            names='quarter',
            title='Revenue Distribution',
            color_discrete_sequence=['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6']
        )
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📊 Seasonal Trends ({len(filtered_months)} months)",
    f"📈 Year-over-Year ({comparison_year})",
    "🔮 Forecasting",
    f"🎉 Holiday Performance ({len(holidays_df)})"
])

# TAB 1: SEASONAL TRENDS
with tab1:
    st.subheader(f"Seasonal Trends Analysis - {metric_type}")
    
    if season_filter == "All Seasons":
        seasons = [
            {'name': 'Q1 - Winter', 'icon': '❄️', 'quarter': 'Q1'},
            {'name': 'Q2 - Spring', 'icon': '🌸', 'quarter': 'Q2'},
            {'name': 'Q3 - Summer', 'icon': '☀️', 'quarter': 'Q3'},
            {'name': 'Q4 - Holiday', 'icon': '🎄', 'quarter': 'Q4'}
        ]
        
        cols = st.columns(4)
        for col, season in zip(cols, seasons):
            with col:
                quarter_data = quarters_df[quarters_df['quarter'].str.contains(season['quarter'])]
                if len(quarter_data) > 0:
                    row = quarter_data.iloc[0]
                    
                    st.markdown(f"""
                    <div style='background:#f8fafc;padding:20px;border-radius:10px;border-left:4px solid #3b82f6;'>
                        <div style='font-size:1.5rem;margin-bottom:10px;'>{season['icon']} {season['name']}</div>
                        <div style='font-size:0.875rem;color:#64748b;margin-bottom:10px;'>${row['revenue']/1000:.0f}K revenue</div>
                        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                            <div style='text-align:center;padding:8px;background:white;border-radius:6px;'>
                                <div style='font-weight:700;'>{row['orders']:,}</div>
                                <div style='font-size:0.75rem;color:#64748b;'>Orders</div>
                            </div>
                            <div style='text-align:center;padding:8px;background:white;border-radius:6px;'>
                                <div style='font-weight:700;color:#22c55e;'>{row['growth']:.1f}%</div>
                                <div style='font-size:0.75rem;color:#64748b;'>Growth</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"#### Monthly {metric_type} Patterns")
    
    display_filtered = filtered_months.copy()
    display_filtered['Month-Year'] = display_filtered['month'] + ' ' + display_filtered['year'].astype(str)
    display_filtered['Value Display'] = display_filtered[selected_metric].apply(
        lambda x: f"${x:,.0f}" if selected_metric in ['revenue', 'aov'] else f"{x:,.0f}"
    )
    
    st.dataframe(
        display_filtered[['Month-Year', 'quarter', 'season', 'Value Display', 'orders']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Month-Year': 'Period',
            'quarter': 'Quarter',
            'season': 'Season',
            'Value Display': metric_type,
            'orders': 'Orders'
        }
    )
    
    st.caption(f"Showing {len(filtered_months)} months of data based on filters")

# TAB 2: YEAR-OVER-YEAR
with tab2:
    st.subheader(f"Year-over-Year Comparison: {comparison_year}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if selected_metric in ['revenue', 'aov']:
            st.metric(f"{current_year} Total", f"${total_current:,.0f}")
        else:
            st.metric(f"{current_year} Total", f"{total_current:,.0f}")
    with col2:
        if selected_metric in ['revenue', 'aov']:
            st.metric(f"{previous_year} Total", f"${total_previous:,.0f}")
        else:
            st.metric(f"{previous_year} Total", f"{total_previous:,.0f}")
    with col3:
        st.metric("YoY Growth", f"{yoy_growth:+.1f}%")
    with col4:
        difference = total_current - total_previous
        if selected_metric in ['revenue', 'aov']:
            st.metric("Difference", f"${difference:,.0f}")
        else:
            st.metric("Difference", f"{difference:,.0f}")
    
    st.markdown("---")
    
    comparison_data = pd.DataFrame({
        'Month': current_year_df['month'],
        str(current_year): current_year_df[selected_metric],
        str(previous_year): previous_year_df[selected_metric]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comparison_data['Month'], 
        y=comparison_data[str(current_year)],
        name=str(current_year), 
        marker=dict(color='#3b82f6')
    ))
    fig.add_trace(go.Bar(
        x=comparison_data['Month'], 
        y=comparison_data[str(previous_year)],
        name=str(previous_year), 
        marker=dict(color='#94a3b8')
    ))
    fig.update_layout(
        title=f'Month-by-Month {metric_type} Comparison', 
        barmode='group', 
        height=400, 
        hovermode='x unified',
        yaxis_title=metric_type
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Detailed Comparison Table")
    
    # Calculate growth percentages properly
    growth_percentages = ((current_year_df[selected_metric].values - previous_year_df[selected_metric].values) / previous_year_df[selected_metric].values * 100)
    growth_percentages = np.round(growth_percentages, 1)
    growth_strings = [f"{val:+.1f}%" for val in growth_percentages]
    
    comparison_detail = pd.DataFrame({
        'Month': current_year_df['month'].values,
        f'{current_year} {metric_type}': current_year_df[selected_metric].apply(
            lambda x: f"${x:,.0f}" if selected_metric in ['revenue', 'aov'] else f"{x:,.0f}"
        ).values,
        f'{previous_year} {metric_type}': previous_year_df[selected_metric].apply(
            lambda x: f"${x:,.0f}" if selected_metric in ['revenue', 'aov'] else f"{x:,.0f}"
        ).values,
        'Growth %': growth_strings
    })
    
    st.dataframe(comparison_detail, use_container_width=True, hide_index=True)

# TAB 3: FORECASTING
with tab3:
    st.subheader("Seasonal Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Next Quarter Forecast")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);color:white;padding:25px;border-radius:10px;'>
            <div style='font-size:1.125rem;margin-bottom:10px;'>Projected Revenue</div>
            <div style='font-size:3rem;font-weight:900;'>${forecast_data['next_quarter']['revenue']/1000000:.2f}M</div>
            <div style='font-size:1rem;color:rgba(255,255,255,0.8);margin-top:10px;'>Confidence: {forecast_data['next_quarter']['confidence']}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Expected Orders", f"{forecast_data['next_quarter']['orders']:,}")
        st.metric("Expected Customers", f"{forecast_data['next_quarter']['customers']:,}")
    
    with col2:
        st.markdown("#### Annual Forecast (Next 12 Months)")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);color:white;padding:25px;border-radius:10px;'>
            <div style='font-size:1.125rem;margin-bottom:10px;'>Projected Annual Revenue</div>
            <div style='font-size:3rem;font-weight:900;'>${forecast_data['next_year']['revenue']/1000000:.2f}M</div>
            <div style='font-size:1rem;color:rgba(255,255,255,0.8);margin-top:10px;'>Confidence: {forecast_data['next_year']['confidence']}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Expected Orders", f"{forecast_data['next_year']['orders']:,}")
        st.metric("Expected Customers", f"{forecast_data['next_year']['customers']:,}")
    
    st.markdown("---")
    st.markdown("#### Forecast Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Historical Analysis**")
        st.progress(1.0)
        st.caption("3-year historical data analysis")
        
        st.markdown("**Seasonal Patterns**")
        st.progress(1.0)
        st.caption("12 identified seasonal patterns")
    
    with col2:
        st.markdown("**Market Trends**")
        st.progress(0.85)
        st.caption("Industry growth factors included")
        
        st.markdown("**External Factors**")
        st.progress(0.75)
        st.caption("Economic indicators considered")
    
    st.markdown("---")
    st.markdown("#### Next Quarter Monthly Breakdown")
    
    forecast_months = pd.DataFrame({
        'Month': ['Jan 2025', 'Feb 2025', 'Mar 2025'],
        'Forecasted Revenue': ['$678,901', '$712,345', '$743,321'],
        'Confidence': ['92%', '93%', '94%'],
        'Expected Orders': ['5,234', '5,534', '5,834'],
        'Growth vs Last Year': ['+18.5%', '+19.0%', '+19.5%']
    })
    
    st.dataframe(forecast_months, use_container_width=True, hide_index=True)

# TAB 4: HOLIDAY PERFORMANCE
with tab4:
    st.subheader("Holiday Performance Analysis")
    
    if len(holidays_df) > 0:
        for idx, holiday in holidays_df.iterrows():
            st.markdown(f"""
            <div style='background:#f8fafc;padding:20px;border-radius:8px;border-left:4px solid #3b82f6;margin-bottom:15px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                    <div style='font-weight:700;font-size:1.125rem;'>{holiday['name']}</div>
                    <div style='color:#64748b;font-size:0.875rem;'>{holiday['date']}</div>
                </div>
                <div style='margin-bottom:10px;'>
                    <span style='font-size:1.5rem;font-weight:800;color:#3b82f6;'>${holiday['revenue']:,}</span>
                    <span style='font-weight:700;color:#22c55e;margin-left:15px;'>+{holiday['growth']:.1f}% YoY</span>
                </div>
                <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px;text-align:center;'>
                    <div style='padding:10px;background:white;border-radius:6px;'>
                        <div style='font-weight:700;'>{holiday['orders']:,}</div>
                        <div style='font-size:0.75rem;color:#64748b;'>Orders</div>
                    </div>
                    <div style='padding:10px;background:white;border-radius:6px;'>
                        <div style='font-weight:700;'>${holiday['aov']}</div>
                        <div style='font-size:0.75rem;color:#64748b;'>AOV</div>
                    </div>
                    <div style='padding:10px;background:white;border-radius:6px;'>
                        <div style='font-weight:700;'>{holiday['conversion']:.1f}%</div>
                        <div style='font-size:0.75rem;color:#64748b;'>Conversion</div>
                    </div>
                </div>
                <div style='margin-top:10px;padding-top:10px;border-top:1px solid #e2e8f0;color:#64748b;font-size:0.875rem;'>
                    Traffic Increase: +{holiday['traffic']}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Top Performing Holidays")
            top_holidays = holidays_df.nlargest(5, 'revenue')
            for idx, row in top_holidays.iterrows():
                st.metric(f"{row['name']}", f"${row['revenue']/1000:.0f}K")
        
        with col2:
            st.markdown("#### Holiday Season Impact")
            total_holiday_revenue = holidays_df['revenue'].sum()
            st.metric("Total Holiday Revenue", f"${total_holiday_revenue/1000:.0f}K")
            st.metric("% of Annual Revenue", "22.8%")
            st.metric("Avg Growth Rate", "+38.5%")
        
        with col3:
            st.markdown("#### Preparation Status")
            st.info("""
            ✅ Inventory Stocked - 150% of normal levels
            ✅ Staff Scheduled - Additional support hired
            ⚠️ Marketing Campaigns - Launch 2 weeks before
            ⭕ Server Capacity - Scale up for traffic surge
            """)

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_btn_season"):
        st.cache_data.clear()
        st.success("✅ Seasonality data refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True, key="export_btn_season"):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'filtered_months': filtered_months.to_dict('records'),
            'current_year_data': current_year_df.to_dict('records'),
            'previous_year_data': previous_year_df.to_dict('records'),
            'quarters': quarters_df.to_dict('records'),
            'holidays': holidays_df.to_dict('records'),
            'forecast': forecast_data,
            'summary': {
                'metric_type': metric_type,
                'current_year_total': f"{total_current:,.2f}",
                'previous_year_total': f"{total_previous:,.2f}",
                'yoy_growth': f"{yoy_growth:.1f}%",
                'peak_value': f"{peak_value:,.2f}",
                'seasonal_variability': f"{seasonal_variability:.1f}%"
            }
        }
        st.success("✅ Seasonality data exported successfully")
        st.json(export_data)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================
# ADDITIONAL INSIGHTS
# ===========================

with st.expander("📊 Seasonal Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 📈 Performance Highlights ({metric_type})")
        if selected_metric in ['revenue', 'aov']:
            st.markdown(f"""
            - **${total_current:,.0f}** total {metric_type.lower()} in {current_year}
            - **${total_previous:,.0f}** total {metric_type.lower()} in {previous_year}
            - **{yoy_growth:+.1f}%** year-over-year growth
            - **${peak_value:,.0f}** peak season {metric_type.lower()}
            - **{seasonal_variability:.1f}%** seasonal variability
            """)
        else:
            st.markdown(f"""
            - **{total_current:,.0f}** total {metric_type.lower()} in {current_year}
            - **{total_previous:,.0f}** total {metric_type.lower()} in {previous_year}
            - **{yoy_growth:+.1f}%** year-over-year growth
            - **{peak_value:,.0f}** peak season {metric_type.lower()}
            - **{seasonal_variability:.1f}%** seasonal variability
            """)
        
        st.markdown("#### 💡 Recommended Actions")
        st.markdown("""
        1. **Immediate**: Prepare for Q4 peak season
        2. **Short-term**: Implement holiday campaigns
        3. **Medium-term**: Optimize inventory levels
        4. **Long-term**: Develop off-season strategies
        """)
    
    with col2:
        st.markdown("#### 🎯 Seasonal Patterns")
        
        for idx, quarter in quarters_df.iterrows():
            st.markdown(f"- **{quarter['quarter']}**: ${quarter['revenue']/1000000:.2f}M ({quarter['growth']:.1f}% growth)")
        
        st.markdown("#### 🏆 Holiday Success Metrics")
        st.markdown("""
        - Peak holiday performance: +52.8% (Cyber Monday)
        - Average holiday growth: +38.5%
        - Holiday season impact: 22.8% of annual revenue
        - Best conversion: 9.2% (Cyber Monday)
        """)

# ===========================
# DIAGNOSTIC INFORMATION
# ===========================

with st.expander("🔧 System Diagnostics"):
    st.markdown("### Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(all_months_df):,}")
        st.metric("Data Completeness", "100%")
    
    with col2:
        st.metric("Years Analyzed", str(len(all_months_df['year'].unique())))
        st.metric("Quarters Tracked", "4")
    
    with col3:
        st.metric("Holiday Events", len(holidays_df))
        st.metric("Forecast Confidence", f"{forecast_data['next_quarter']['confidence']}%")
    
    st.markdown("---")
    st.markdown("### Filter Status")
    st.info(f"""
    **Active Filters:**
    - Analysis Period: {analysis_period}
    - Metric Type: {metric_type}
    - Season Filter: {season_filter}
    - Comparison: {comparison_year}
    
    **Results:** {len(filtered_months)} months shown, comparing {current_year} vs {previous_year}
    """)
    
    st.markdown("---")
    st.markdown("### Trend Analysis Summary")
    
    growth_values = ((current_year_df[selected_metric].values - previous_year_df[selected_metric].values) / previous_year_df[selected_metric].values * 100)
    
    trend_metrics = pd.DataFrame({
        'Metric': [
            'Highest Growth Month', 
            'Lowest Growth Month', 
            f'Average Monthly {metric_type}', 
            f'{metric_type} Trend', 
            'Order Trend'
        ],
        f'{current_year}': [
            f"+{growth_values.max():.1f}%",
            f"+{growth_values.min():.1f}%",
            f"${current_year_df[selected_metric].mean():,.0f}" if selected_metric in ['revenue', 'aov'] else f"{current_year_df[selected_metric].mean():,.0f}",
            "📈 Upward" if yoy_growth > 0 else "📉 Downward",
            "📈 Upward"
        ]
    })
    
    st.dataframe(trend_metrics, use_container_width=True, hide_index=True)

# ===========================
# FILTER IMPACT SUMMARY
# ===========================

with st.expander("🎯 Filter Impact Analysis"):
    st.markdown("### How Filters Are Affecting Your View")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Data Scope")
        st.markdown(f"""
        **Period Filter:** `{analysis_period}`
        - Showing data from: {filtered_months['year'].min()} to {filtered_months['year'].max()}
        - Total months in view: {len(filtered_months)}
        
        **Season Filter:** `{season_filter}`
        - Current view: {season_filter}
        - Records matching filter: {len(filtered_months)}
        """)
    
    with col2:
        st.markdown("#### Metric Analysis")
        st.markdown(f"""
        **Metric Type:** `{metric_type}`
        - Current year total: {total_current:,.0f}
        - Previous year total: {total_previous:,.0f}
        - Growth rate: {yoy_growth:+.1f}%
        
        **Comparison:** `{comparison_year}`
        - Comparing: {current_year} vs {previous_year}
        - Months compared: {len(current_year_df)}
        """)
    
    st.markdown("---")
    st.markdown("#### Quick Filter Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 View All Data", use_container_width=True, key="view_all"):
            st.info("Set filters to: Last 3 Years, All Seasons")
    
    with col2:
        if st.button("🎄 Holiday Focus", use_container_width=True, key="holiday_focus"):
            st.info("Set season to: Q4 (Fall/Holiday)")
    
    with col3:
        if st.button("💰 Revenue Only", use_container_width=True, key="revenue_only"):
            st.info("Set metric to: Revenue")
    
    with col4:
        if st.button("📅 Current Year", use_container_width=True, key="current_year"):
            st.info("Set period to: Last Year")

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;font-size:0.875rem;padding:20px;">
    <strong>Seasonal Trend Analysis System</strong> | Powered by Advanced Analytics | 
    <a href="#" style="color:#3b82f6;text-decoration:none;">Documentation</a> | 
    <a href="#" style="color:#3b82f6;text-decoration:none;">Support</a>
</div>
""", unsafe_allow_html=True)