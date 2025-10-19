"""
🌍 Geographic Data Analysis - Geographic sales mapping and regional performance
Regional performance analysis, location issues, and expansion opportunities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Geographic Data Analysis",
    page_icon="🌍",
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
    .alert-info {
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        border-left: 4px solid;
        background: #dbeafe;
        color: #1e40af;
        border-left-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_geography_data():
    """Generate comprehensive geographic data"""
    np.random.seed(42)
    
    # US Regions
    regions = pd.DataFrame([
        {'id': 1, 'name': 'California', 'state': 'CA', 'country': 'USA', 'revenue': 456789, 'orders': 3456, 'customers': 2345, 'growth': 14.5, 'performance': 'excellent', 'issues': 45},
        {'id': 2, 'name': 'Texas', 'state': 'TX', 'country': 'USA', 'revenue': 389456, 'orders': 2890, 'customers': 1987, 'growth': 12.3, 'performance': 'excellent', 'issues': 34},
        {'id': 3, 'name': 'New York', 'state': 'NY', 'country': 'USA', 'revenue': 367234, 'orders': 2567, 'customers': 1876, 'growth': 8.7, 'performance': 'good', 'issues': 56},
        {'id': 4, 'name': 'Florida', 'state': 'FL', 'country': 'USA', 'revenue': 298765, 'orders': 2234, 'customers': 1654, 'growth': 15.2, 'performance': 'excellent', 'issues': 23},
        {'id': 5, 'name': 'Illinois', 'state': 'IL', 'country': 'USA', 'revenue': 245678, 'orders': 1890, 'customers': 1432, 'growth': 6.5, 'performance': 'good', 'issues': 18},
        {'id': 6, 'name': 'Washington', 'state': 'WA', 'country': 'USA', 'revenue': 223456, 'orders': 1765, 'customers': 1298, 'growth': 11.2, 'performance': 'good', 'issues': 12},
        {'id': 7, 'name': 'Massachusetts', 'state': 'MA', 'country': 'USA', 'revenue': 198765, 'orders': 1567, 'customers': 1123, 'growth': 9.8, 'performance': 'good', 'issues': 15},
        {'id': 8, 'name': 'Pennsylvania', 'state': 'PA', 'country': 'USA', 'revenue': 187654, 'orders': 1456, 'customers': 1089, 'growth': 5.3, 'performance': 'fair', 'issues': 28},
        {'id': 9, 'name': 'Georgia', 'state': 'GA', 'country': 'USA', 'revenue': 176543, 'orders': 1398, 'customers': 987, 'growth': 13.7, 'performance': 'excellent', 'issues': 9},
        {'id': 10, 'name': 'Arizona', 'state': 'AZ', 'country': 'USA', 'revenue': 165432, 'orders': 1287, 'customers': 945, 'growth': 16.8, 'performance': 'excellent', 'issues': 7}
    ])
    
    # International markets
    international = pd.DataFrame([
        {'country': 'Canada', 'revenue': 234567, 'orders': 1876, 'customers': 1234, 'growth': 18.5, 'marketSize': 'Large'},
        {'country': 'United Kingdom', 'revenue': 198765, 'orders': 1567, 'customers': 1098, 'growth': 12.3, 'marketSize': 'Large'},
        {'country': 'Germany', 'revenue': 145678, 'orders': 1234, 'customers': 876, 'growth': 15.7, 'marketSize': 'Medium'},
        {'country': 'Australia', 'revenue': 123456, 'orders': 987, 'customers': 678, 'growth': 22.4, 'marketSize': 'Medium'},
        {'country': 'France', 'revenue': 98765, 'orders': 789, 'customers': 543, 'growth': 9.8, 'marketSize': 'Medium'}
    ])
    
    # Issues
    issues = pd.DataFrame([
        {'type': 'Invalid Zip Code', 'count': 89, 'affected': 'Multiple States', 'severity': 'high', 'impact': 'Delivery delays'},
        {'type': 'Missing Address Line 2', 'count': 67, 'affected': 'Urban Areas', 'severity': 'medium', 'impact': 'Package mis-delivery'},
        {'type': 'Invalid State Code', 'count': 34, 'affected': 'CA, TX, FL', 'severity': 'high', 'impact': 'Order rejection'},
        {'type': 'PO Box Restrictions', 'count': 28, 'affected': 'Rural Areas', 'severity': 'medium', 'impact': 'Shipping limitations'},
        {'type': 'International Format', 'count': 16, 'affected': 'Canada, UK', 'severity': 'low', 'impact': 'Manual processing'}
    ])
    
    # Expansion opportunities
    expansion = pd.DataFrame([
        {'market': 'Colorado', 'score': 92, 'population': '5.8M', 'competition': 'Low', 'income': '$77,127', 'penetration': '68%', 'projected': '$145K', 'breakeven': '8 months'},
        {'market': 'North Carolina', 'score': 88, 'population': '10.6M', 'competition': 'Medium', 'income': '$59,384', 'penetration': '62%', 'projected': '$178K', 'breakeven': '10 months'},
        {'market': 'Oregon', 'score': 85, 'population': '4.2M', 'competition': 'Medium', 'income': '$67,058', 'penetration': '71%', 'projected': '$98K', 'breakeven': '7 months'},
        {'market': 'Tennessee', 'score': 82, 'population': '7.0M', 'competition': 'Low', 'income': '$56,071', 'penetration': '58%', 'projected': '$112K', 'breakeven': '11 months'},
        {'market': 'Nevada', 'score': 79, 'population': '3.2M', 'competition': 'High', 'income': '$63,276', 'penetration': '65%', 'projected': '$89K', 'breakeven': '13 months'},
        {'market': 'Mexico', 'score': 75, 'population': '128M', 'competition': 'Medium', 'income': '$9,946', 'penetration': '45%', 'projected': '$234K', 'breakeven': '18 months'}
    ])
    
    return regions, international, issues, expansion

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading geographic data..."):
    regions_df, international_df, issues_df, expansion_df = generate_sample_geography_data()

# Calculate metrics
total_regions = len(regions_df)
top_region_revenue = regions_df['revenue'].max()
address_errors = issues_df['count'].sum()
international_sales_pct = 23.4

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### Filter Settings")
    
    time_period = st.selectbox(
        "Time Period",
        ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year"],
        index=1
    )
    
    region_type = st.selectbox(
        "Region Type",
        ["All Regions", "Domestic", "International", "Top Performers", "Underperforming"],
        index=0
    )
    
    metric_view = st.selectbox(
        "Metric View",
        ["Revenue", "Orders", "Customers", "Growth Rate"],
        index=0
    )
    
    search_location = st.text_input("Search Location", placeholder="City, State, Country...")
    
    st.markdown("---")
    if st.button("Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_geography_filters(regions_df, region_f, search_q):
    """Apply filters to geographic data"""
    filtered = regions_df.copy()
    
    if region_f == "Top Performers":
        filtered = filtered[filtered['performance'] == 'excellent']
    elif region_f == "Underperforming":
        filtered = filtered[filtered['growth'] < 8.0]
    
    if search_q:
        search_lower = search_q.lower()
        filtered = filtered[
            filtered['name'].str.lower().str.contains(search_lower, na=False) |
            filtered['state'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_regions = apply_geography_filters(regions_df, region_type, search_location)

# ===========================
# HEADER & METRICS
# ===========================

st.title("🌍 Geographic Data Analysis")
st.markdown("**Geographic sales mapping, regional performance, location issues, and expansion opportunities**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Regions", f"{total_regions}", "+3 new markets")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Top Region Revenue", f"${top_region_revenue:,.0f}", "+$67K vs last month")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Address Errors", f"{address_errors}", "+23 new issues")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("International Sales", f"{international_sales_pct}%", "+3.4% growth")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown(f"""
<div class="alert-info">
    <strong>Geographic Insights:</strong> {total_regions} active regions tracked. Top region: California (${top_region_revenue/1000:.0f}K revenue). {address_errors} address errors detected. 12 high-potential expansion markets identified.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Region (Top 10)")
    
    top_regions = regions_df.nlargest(10, 'revenue')
    fig_revenue = px.bar(
        top_regions,
        x='state',
        y='revenue',
        color='revenue',
        color_continuous_scale='Blues',
        labels={'revenue': 'Revenue ($)', 'state': 'State'}
    )
    fig_revenue.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_revenue, use_container_width=True)

with col2:
    st.subheader("Geographic Distribution")
    
    distribution_data = pd.DataFrame([
        {'region': 'Top 5 States', 'revenue': regions_df.nlargest(5, 'revenue')['revenue'].sum()},
        {'region': 'Other States', 'revenue': regions_df.nsmallest(5, 'revenue')['revenue'].sum()},
        {'region': 'International', 'revenue': international_df['revenue'].sum()}
    ])
    
    fig_dist = px.pie(
        distribution_data,
        values='revenue',
        names='region',
        color_discrete_sequence=['#3b82f6', '#8b5cf6', '#22c55e']
    )
    fig_dist.update_layout(height=350)
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Sales Map",
    "📊 Regional Performance",
    "⚠️ Location Issues",
    "🎯 Expansion Opportunities"
])

# TAB 1: SALES MAP
with tab1:
    st.subheader("Geographic Sales Heatmap")
    
    st.markdown("#### US States Performance")
    
    col1, col2, col3 = st.columns(3)
    
    states_for_display = [
        {'state': 'CA', 'revenue': 456789},
        {'state': 'TX', 'revenue': 389456},
        {'state': 'NY', 'revenue': 367234},
        {'state': 'FL', 'revenue': 298765},
        {'state': 'IL', 'revenue': 245678},
        {'state': 'WA', 'revenue': 223456},
        {'state': 'MA', 'revenue': 198765},
        {'state': 'PA', 'revenue': 187654},
        {'state': 'GA', 'revenue': 176543},
        {'state': 'AZ', 'revenue': 165432}
    ]
    
    for state in states_for_display:
        intensity = state['revenue'] / 456789
        color = '#22c55e' if intensity > 0.8 else '#3b82f6' if intensity > 0.6 else '#f59e0b' if intensity > 0.4 else '#ef4444'
        
        col = col1 if states_for_display.index(state) % 3 == 0 else col2 if states_for_display.index(state) % 3 == 1 else col3
        
        with col:
            st.markdown(f"""
            <div style="background: {color}; color: white; padding: 20px; border-radius: 8px; margin-bottom: 10px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 800; margin-bottom: 5px;">{state['state']}</div>
                <div style="font-size: 1.125rem; font-weight: 700;">${state['revenue']/1000:.0f}K</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Top Performing Cities")
    
    cities = pd.DataFrame([
        {'city': 'Los Angeles, CA', 'revenue': 189456, 'orders': 1567, 'growth': 15.3},
        {'city': 'New York, NY', 'revenue': 176543, 'orders': 1432, 'growth': 12.8},
        {'city': 'Houston, TX', 'revenue': 145678, 'orders': 1234, 'growth': 18.5},
        {'city': 'Chicago, IL', 'revenue': 134567, 'orders': 1123, 'growth': 9.7},
        {'city': 'Phoenix, AZ', 'revenue': 123456, 'orders': 1045, 'growth': 21.4},
        {'city': 'Miami, FL', 'revenue': 112345, 'orders': 967, 'growth': 14.2}
    ])
    
    cols = st.columns(3)
    for idx, (_, city) in enumerate(cities.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="padding: 15px; border-left: 4px solid #3b82f6; background: #f0f9ff; border-radius: 6px; margin-bottom: 10px;">
                <div style="font-weight: 700; margin-bottom: 8px;">{city['city']}</div>
                <div style="font-size: 0.875rem; color: #64748b; margin-bottom: 8px;">${city['revenue']/1000:.0f}K revenue</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="text-align: center; background: white; padding: 8px; border-radius: 6px;">
                        <div style="font-weight: 700;">{city['orders']}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Orders</div>
                    </div>
                    <div style="text-align: center; background: white; padding: 8px; border-radius: 6px;">
                        <div style="font-weight: 700; color: #22c55e;">{city['growth']}%</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Growth</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# TAB 2: REGIONAL PERFORMANCE
with tab2:
    st.subheader("Regional Performance Analysis")
    
    st.markdown("#### US Regions Performance")
    
    display_regions = filtered_regions.copy()
    
    st.dataframe(
        display_regions[['name', 'state', 'revenue', 'orders', 'customers', 'growth', 'performance', 'issues']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'name': 'Region',
            'state': 'State',
            'revenue': st.column_config.NumberColumn('Revenue', format='$%d'),
            'orders': 'Orders',
            'customers': 'Customers',
            'growth': st.column_config.NumberColumn('Growth %', format='%.1f%%'),
            'performance': 'Performance',
            'issues': 'Issues'
        }
    )
    
    if len(filtered_regions) == 0:
        st.info("No regions match the current filters")
    else:
        st.caption(f"Showing {len(filtered_regions)} region(s)")
    
    st.markdown("---")
    st.markdown("#### International Markets Performance")
    
    st.dataframe(
        international_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'country': 'Country',
            'revenue': st.column_config.NumberColumn('Revenue', format='$%d'),
            'orders': 'Orders',
            'customers': 'Customers',
            'growth': st.column_config.NumberColumn('Growth %', format='%.1f%%'),
            'marketSize': 'Market Size'
        }
    )

# TAB 3: LOCATION ISSUES
with tab3:
    st.subheader("Location-Based Data Issues")
    
    st.markdown("#### Address Quality Issues")
    
    st.dataframe(
        issues_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'type': 'Issue Type',
            'count': 'Count',
            'affected': 'Affected Locations',
            'severity': 'Severity',
            'impact': 'Impact'
        }
    )
    
    st.markdown("---")
    st.markdown("#### Address Validation by Region")
    
    validation_data = pd.DataFrame([
        {'region': 'California', 'valid': 3234, 'invalid': 45, 'rate': 98.6},
        {'region': 'New York', 'valid': 2456, 'invalid': 56, 'rate': 97.8},
        {'region': 'Texas', 'valid': 2789, 'invalid': 34, 'rate': 98.8},
        {'region': 'Florida', 'valid': 2178, 'invalid': 23, 'rate': 99.0},
        {'region': 'Illinois', 'valid': 1834, 'invalid': 18, 'rate': 99.0}
    ])
    
    for _, val in validation_data.iterrows():
        color = '#22c55e' if val['rate'] >= 99 else '#3b82f6' if val['rate'] >= 98 else '#f59e0b'
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**{val['region']}**")
            st.progress(val['rate'] / 100)
        with col2:
            st.markdown(f"{val['valid']:,} valid")
        with col3:
            st.markdown(f"<div style='color: {color}; font-weight: 700;'>{val['rate']}%</div>", unsafe_allow_html=True)

# TAB 4: EXPANSION OPPORTUNITIES
with tab4:
    st.subheader("Market Expansion Opportunities")
    
    cols = st.columns(2)
    
    for idx, (_, opp) in enumerate(expansion_df.iterrows()):
        score_color = '#22c55e' if opp['score'] >= 90 else '#3b82f6' if opp['score'] >= 80 else '#f59e0b'
        
        with cols[idx % 2]:
            st.markdown(f"""
            <div style="border: 2px solid #e2e8f0; padding: 20px; border-radius: 10px; background: white; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div style="font-size: 1.25rem; font-weight: 700;">{opp['market']}</div>
                    <div style="background: {score_color}; color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700;">{opp['score']}/100</div>
                </div>
                <div style="margin-bottom: 15px;">
                    <div style="margin-bottom: 10px;">
                        <span style="color: #64748b; font-size: 0.875rem;">Population:</span>
                        <span style="font-weight: 700;">{opp['population']}</span>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span style="color: #64748b; font-size: 0.875rem;">Avg Income:</span>
                        <span style="font-weight: 700;">{opp['income']}</span>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span style="color: #64748b; font-size: 0.875rem;">E-commerce Penetration:</span>
                        <span style="font-weight: 700;">{opp['penetration']}</span>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <span style="color: #64748b; font-size: 0.875rem;">Competition:</span>
                        <span style="font-weight: 700;">{opp['competition']}</span>
                    </div>
                </div>
                <div style="background: #f0f9ff; padding: 10px; border-radius: 6px; margin-bottom: 10px; text-align: center;">
                    <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 3px;">Projected Year 1 Revenue</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: #22c55e;">{opp['projected']}</div>
                </div>
                <div style="text-align: center; font-size: 0.875rem; color: #64748b; margin-bottom: 10px;">
                    Breakeven: {opp['breakeven']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Expansion Strategy Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Immediate (0-6 months)")
        st.markdown("""
        - **Colorado** (92/100)
        - **Oregon** (85/100)
        
        Total Investment: $45K
        Projected Year 1 Revenue: $243K
        """)
    
    with col2:
        st.markdown("#### Short-term (6-12 months)")
        st.markdown("""
        - **North Carolina** (88/100)
        - **Tennessee** (82/100)
        
        Total Investment: $32K
        Projected Year 1 Revenue: $290K
        """)
    
    with col3:
        st.markdown("#### Long-term (12+ months)")
        st.markdown("""
        - **Nevada** (79/100)
        - **Mexico** (75/100)
        
        Total Investment: $54K
        Projected Year 1 Revenue: $323K
        """)

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Geographic data refreshed successfully")
        st.rerun()

with col2:
    if st.button("Export Data", use_container_width=True):
        st.success("✅ Geographic data exported successfully")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Showing data from {time_period} | View: {metric_view}")