"""
🚚 Enhanced Shipping Data Audit - All 4 Tabs with Working Filters
Exact match to shipping.html with delivery tracking, address validation, and carrier performance
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Shipping Data Audit",
    page_icon="🚚",
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
def generate_sample_shipping_data():
    """Generate sample shipping data with quality issues"""
    np.random.seed(42)
    
    # Shipping Records
    records = []
    carriers = ['UPS', 'FedEx', 'USPS', 'DHL']
    statuses = ['delivered', 'in-transit', 'pending', 'delayed', 'exception']
    destinations = [
        'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ',
        'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA', 'Dallas, TX', 'San Jose, CA'
    ]
    
    for i in range(1, 201):
        carrier = np.random.choice(carriers)
        status = np.random.choice(statuses, p=[0.45, 0.2, 0.15, 0.1, 0.1])
        
        shipped_date = (datetime.now() - timedelta(days=np.random.randint(0, 30))).date()
        
        if status == 'delivered':
            delivered_date = shipped_date + timedelta(days=np.random.randint(1, 8))
        elif status in ['in-transit', 'delayed']:
            delivered_date = None
        else:
            delivered_date = None
        
        # Address issues (5% will have problems)
        has_address_issue = np.random.random() < 0.05
        
        records.append({
            'shipment_id': f'SHP-{1000+i}',
            'order_id': f'ORD-{5000+i}',
            'carrier': carrier,
            'tracking_number': f'{carrier}{np.random.randint(100000000, 999999999)}',
            'status': status,
            'shipped_date': shipped_date,
            'delivered_date': delivered_date,
            'destination': np.random.choice(destinations),
            'cost': round(np.random.uniform(8, 30), 2),
            'address_issue': has_address_issue,
            'weight': round(np.random.uniform(0.5, 50), 2),
            'insurance': 'yes' if np.random.random() > 0.8 else 'no'
        })
    
    # Delivery Times Analysis
    delivery_times = []
    delivery_data = {
        'UPS': {'avg': 3.2, 'min': 2, 'max': 5, 'ontime': 96.5, 'shipments': 4234},
        'FedEx': {'avg': 3.5, 'min': 2, 'max': 6, 'ontime': 94.8, 'shipments': 3456},
        'USPS': {'avg': 4.1, 'min': 3, 'max': 7, 'ontime': 91.2, 'shipments': 2567},
        'DHL': {'avg': 3.8, 'min': 2, 'max': 8, 'ontime': 92.5, 'shipments': 977}
    }
    
    for carrier, data in delivery_data.items():
        delivery_times.append({
            'carrier': carrier,
            'avg_delivery_days': data['avg'],
            'min_days': data['min'],
            'max_days': data['max'],
            'ontime_rate': data['ontime'],
            'total_shipments': data['shipments']
        })
    
    # Shipping Costs Analysis
    costs = []
    cost_data = [
        {'zone': 'Zone 1 (Local)', 'avg_cost': 8.50, 'shipments': 2345, 'total': 19932, 'per_mile': 0.45},
        {'zone': 'Zone 2 (Regional)', 'avg_cost': 12.75, 'shipments': 3456, 'total': 44064, 'per_mile': 0.38},
        {'zone': 'Zone 3 (National)', 'avg_cost': 18.25, 'shipments': 4567, 'total': 83348, 'per_mile': 0.32},
        {'zone': 'Zone 4 (Far)', 'avg_cost': 25.50, 'shipments': 866, 'total': 22083, 'per_mile': 0.28}
    ]
    
    for item in cost_data:
        costs.append({
            'zone': item['zone'],
            'avg_cost': item['avg_cost'],
            'shipments': item['shipments'],
            'total_cost': item['total'],
            'cost_per_mile': item['per_mile']
        })
    
    # Carrier Performance
    carriers_perf = []
    carrier_data = {
        'UPS': {'rating': 4.6, 'shipments': 4234, 'ontime': 96.5, 'avg_delivery': 3.2, 'cost': 54234, 'issues': 23},
        'FedEx': {'rating': 4.4, 'shipments': 3456, 'ontime': 94.8, 'avg_delivery': 3.5, 'cost': 48567, 'issues': 34},
        'USPS': {'rating': 4.1, 'shipments': 2567, 'ontime': 91.2, 'avg_delivery': 4.1, 'cost': 21823, 'issues': 56},
        'DHL': {'rating': 4.3, 'shipments': 977, 'ontime': 92.5, 'avg_delivery': 3.8, 'cost': 17812, 'issues': 19}
    }
    
    for carrier, data in carrier_data.items():
        carriers_perf.append({
            'carrier': carrier,
            'rating': data['rating'],
            'total_shipments': data['shipments'],
            'ontime_rate': data['ontime'],
            'avg_delivery_days': data['avg_delivery'],
            'total_cost': data['cost'],
            'issues': data['issues'],
            'performance': 'excellent' if data['ontime'] >= 94 else 'good'
        })
    
    return (pd.DataFrame(records), pd.DataFrame(delivery_times), 
            pd.DataFrame(costs), pd.DataFrame(carriers_perf))

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading shipping data..."):
    records_df, delivery_df, costs_df, carriers_df = generate_sample_shipping_data()

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 🚚 Filters")
    
    date_range = st.selectbox(
        "📅 Date Range",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
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
    
    carrier_filter = st.multiselect(
        "🚛 Carrier",
        records_df['carrier'].unique().tolist(),
        default=records_df['carrier'].unique().tolist()
    )
    
    status_filter = st.multiselect(
        "📦 Status",
        ["delivered", "in-transit", "pending", "delayed", "exception"],
        default=["delivered", "in-transit", "pending", "delayed", "exception"]
    )
    
    search_query = st.text_input("🔍 Search", placeholder="Tracking, Order ID...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_shipping_filters(df, date_cutoff, carrier_list, status_list, search_text):
    filtered = df.copy()
    
    if date_cutoff:
        filtered = filtered[filtered['shipped_date'] >= date_cutoff]
    
    if carrier_list:
        filtered = filtered[filtered['carrier'].isin(carrier_list)]
    
    if status_list:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['shipment_id'].str.lower().str.contains(search_lower, na=False) |
            filtered['order_id'].str.lower().str.contains(search_lower, na=False) |
            filtered['tracking_number'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_records = apply_shipping_filters(records_df, cutoff_date, carrier_filter, status_filter, search_query)
filtered_carriers = carriers_df[carriers_df['carrier'].isin(carrier_filter)]

# ===========================
# CALCULATE METRICS
# ===========================

total_shipments = len(records_df)
on_time_delivery = (len(records_df[records_df['status'] == 'delivered']) / total_shipments * 100) if total_shipments > 0 else 0
address_issues = len(records_df[records_df['address_issue'] == True])
delayed_shipments = len(records_df[records_df['status'] == 'delayed'])

# ===========================
# HEADER & METRICS
# ===========================

st.title("🚚 Shipping Data Audit")
st.markdown("**Delivery tracking, address validation, shipping costs, and carrier performance**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Shipments", f"{total_shipments:,}", "+567 this month")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("On-Time Delivery", f"{on_time_delivery:.1f}%", "+2.1% improvement")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Address Issues", f"{address_issues}", "+12 new issues")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Delayed Shipments", f"{delayed_shipments}", "-45 vs last month")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

st.markdown(f"""
<div class="alert alert-warning">
    <strong>🚚 Shipping Alert:</strong> {total_shipments:,} total shipments processed. {address_issues} address issues detected. {delayed_shipments} delayed shipments ({delayed_shipments/total_shipments*100:.1f}%). {on_time_delivery:.1f}% on-time delivery rate.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Shipment Volume Trend")
    
    trend_data = pd.DataFrame({
        'Month': ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
        'Total Shipments': [9234, 9876, 10456, 10789, 11023, 11234],
        'Delayed': [298, 276, 289, 312, 267, 234]
    })
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Total Shipments'],
                              mode='lines+markers', name='Total Shipments',
                              line=dict(color='#3b82f6', width=3),
                              fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig1.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Delayed'],
                              mode='lines+markers', name='Delayed',
                              line=dict(color='#ef4444', width=2),
                              fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'))
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Delivery Status Distribution")
    
    status_counts = records_df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    color_map = {
        'delivered': '#22c55e',
        'in-transit': '#3b82f6',
        'pending': '#f59e0b',
        'delayed': '#ef4444',
        'exception': '#991b1b'
    }
    
    fig2 = px.pie(status_counts, values='Count', names='Status',
                  color='Status', color_discrete_map=color_map)
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 Shipping Records ({len(filtered_records)})",
    f"📊 Delivery Times",
    f"💰 Shipping Costs",
    f"🚛 Carrier Performance ({len(filtered_carriers)})"
])

# TAB 1: SHIPPING RECORDS
with tab1:
    st.subheader("Shipping Records")
    
    if len(filtered_records) > 0:
        display_records = filtered_records.copy()
        
        display_records['Cost Display'] = display_records['cost'].apply(lambda x: f"${x:.2f}")
        display_records['Status Badge'] = display_records['status'].apply(
            lambda x: f"✅ {x.upper()}" if x == "delivered"
            else f"🚚 {x.upper()}" if x == "in-transit"
            else f"⏳ {x.upper()}" if x == "pending"
            else f"⚠️ {x.upper()}" if x == "delayed"
            else f"❌ {x.upper()}"
        )
        display_records['Address Issue'] = display_records['address_issue'].apply(
            lambda x: "🔴 Issue" if x else "✅ Valid"
        )
        
        st.dataframe(
            display_records[['shipment_id', 'order_id', 'carrier', 'tracking_number',
                           'Status Badge', 'shipped_date', 'delivered_date', 'destination',
                           'Cost Display', 'Address Issue']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'shipment_id': 'Shipment ID',
                'order_id': 'Order ID',
                'carrier': 'Carrier',
                'tracking_number': 'Tracking Number',
                'Status Badge': 'Status',
                'shipped_date': 'Shipped Date',
                'delivered_date': 'Delivered Date',
                'destination': 'Destination',
                'Cost Display': 'Cost',
                'Address Issue': 'Address'
            }
        )
        
        st.caption(f"Showing {len(filtered_records):,} of {len(records_df):,} shipments")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delivered = len(filtered_records[filtered_records['status'] == 'delivered'])
            st.metric("✅ Delivered", f"{delivered:,}")
        with col2:
            in_transit = len(filtered_records[filtered_records['status'] == 'in-transit'])
            st.metric("🚚 In Transit", f"{in_transit:,}")
        with col3:
            pending = len(filtered_records[filtered_records['status'] == 'pending'])
            st.metric("⏳ Pending", f"{pending:,}")
        with col4:
            issues = len(filtered_records[filtered_records['address_issue'] == True])
            st.metric("⚠️ Address Issues", f"{issues}")
        
        if st.button("📍 Track Selected Shipment", use_container_width=True):
            st.info("Select a shipment to view detailed tracking information")
    else:
        st.info("No shipments match the current filters")

# TAB 2: DELIVERY TIMES
with tab2:
    st.subheader("Delivery Time Analysis")
    
    if len(delivery_df) > 0:
        display_delivery = delivery_df.copy()
        
        display_delivery['Avg Delivery'] = display_delivery['avg_delivery_days'].apply(lambda x: f"{x:.1f} days")
        display_delivery['On-Time Rate'] = display_delivery['ontime_rate'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_delivery[['carrier', 'Avg Delivery', 'min_days', 'max_days',
                            'On-Time Rate', 'total_shipments']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'carrier': 'Carrier',
                'Avg Delivery': 'Avg Delivery Time',
                'min_days': 'Min Days',
                'max_days': 'Max Days',
                'On-Time Rate': 'On-Time Rate',
                'total_shipments': 'Total Shipments'
            }
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Delivery Time Distribution")
            
            time_ranges = pd.DataFrame({
                'Range': ['1-2 days', '3-4 days', '5-6 days', '7+ days'],
                'Shipments': [3456, 5234, 1890, 654],
                'Percentage': [30.8, 46.6, 16.8, 5.8]
            })
            
            fig = px.bar(time_ranges, x='Range', y='Percentage',
                        title='Distribution of Delivery Times',
                        color='Percentage',
                        color_continuous_scale='Blues',
                        text='Percentage')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Delivery Time by Region")
            
            regions = pd.DataFrame({
                'Region': ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West Coast'],
                'Avg Time': [3.1, 3.8, 3.5, 4.2, 3.9],
                'On-Time %': [96.2, 93.5, 94.8, 91.3, 92.7]
            })
            
            st.dataframe(regions, use_container_width=True, hide_index=True)
    else:
        st.info("No delivery data available")

# TAB 3: SHIPPING COSTS
with tab3:
    st.subheader("Shipping Cost Analysis")
    
    if len(costs_df) > 0:
        display_costs = costs_df.copy()
        
        display_costs['Avg Cost'] = display_costs['avg_cost'].apply(lambda x: f"${x:.2f}")
        display_costs['Total Cost'] = display_costs['total_cost'].apply(lambda x: f"${x:,.0f}")
        display_costs['Cost/Mile'] = display_costs['cost_per_mile'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(
            display_costs[['zone', 'Avg Cost', 'shipments', 'Total Cost', 'Cost/Mile']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'zone': 'Shipping Zone',
                'Avg Cost': 'Avg Cost',
                'shipments': 'Shipments',
                'Total Cost': 'Total Cost',
                'Cost/Mile': 'Cost per Mile'
            }
        )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Cost Breakdown")
            st.metric("Base Shipping", "$98,234")
            st.metric("Fuel Surcharges", "$12,456")
            st.metric("Insurance", "$5,678")
            st.metric("Special Handling", "$3,245")
            st.markdown("---")
            st.metric("**Total Cost**", "$119,613")
        
        with col2:
            st.markdown("#### Cost by Carrier")
            
            carrier_costs = pd.DataFrame({
                'Carrier': ['UPS', 'FedEx', 'USPS', 'DHL'],
                'Total Cost': [54234, 48567, 21823, 17812]
            })
            
            fig = px.bar(carrier_costs, x='Carrier', y='Total Cost',
                        title='Total Shipping Cost by Carrier',
                        color='Total Cost',
                        color_continuous_scale='Oranges',
                        text='Total Cost')
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("#### Cost Optimization Opportunities")
            
            opportunities = pd.DataFrame({
                'Opportunity': ['Volume Discount', 'Zone Optimization', 'Packaging Efficiency', 'Carrier Mix'],
                'Potential Savings': ['$4,500', '$3,200', '$2,100', '$1,800']
            })
            
            st.dataframe(opportunities, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            total_savings = 4500 + 3200 + 2100 + 1800
            st.markdown(f"<div style='background:#dbeafe;padding:15px;border-radius:8px;text-align:center;'><div style='color:#1e40af;font-size:0.875rem;margin-bottom:5px;'>Total Potential Savings</div><div style='font-size:1.5rem;font-weight:800;color:#3b82f6;'>${total_savings:,}/month</div></div>", unsafe_allow_html=True)
    else:
        st.info("No cost data available")

# TAB 4: CARRIER PERFORMANCE
with tab4:
    st.subheader("Carrier Performance Comparison")
    
    if len(filtered_carriers) > 0:
        display_carriers = filtered_carriers.copy()
        
        display_carriers['Rating'] = display_carriers['rating'].apply(lambda x: f"⭐ {x:.1f}")
        display_carriers['On-Time %'] = display_carriers['ontime_rate'].apply(lambda x: f"{x:.1f}%")
        display_carriers['Avg Delivery'] = display_carriers['avg_delivery_days'].apply(lambda x: f"{x:.1f} days")
        display_carriers['Total Cost'] = display_carriers['total_cost'].apply(lambda x: f"${x:,}")
        display_carriers['Performance'] = display_carriers['performance'].apply(
            lambda x: f"🟢 {x.upper()}" if x == "excellent" else f"🟡 {x.upper()}"
        )
        
        st.dataframe(
            display_carriers[['carrier', 'Rating', 'total_shipments', 'On-Time %',
                            'Avg Delivery', 'Total Cost', 'issues', 'Performance']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'carrier': 'Carrier',
                'Rating': 'Rating',
                'total_shipments': 'Shipments',
                'On-Time %': 'On-Time Rate',
                'Avg Delivery': 'Avg Delivery',
                'Total Cost': 'Total Cost',
                'issues': 'Issues',
                'Performance': 'Performance'
            }
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance Metrics")
            
            fig = px.bar(display_carriers, x='carrier', y='ontime_rate',
                        title='On-Time Delivery Rate by Carrier',
                        color='ontime_rate',
                        color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
                        text='On-Time %')
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(height=300, showlegend=False, xaxis_title='Carrier', yaxis_title='On-Time %')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Issues by Carrier")
            
            fig = px.bar(display_carriers, x='carrier', y='issues',
                        title='Reported Issues by Carrier',
                        color='issues',
                        color_continuous_scale='Reds',
                        text='issues')
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(height=300, showlegend=False, xaxis_title='Carrier', yaxis_title='Number of Issues')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            top_carrier = display_carriers.loc[display_carriers['rating'].idxmax()]
            st.metric("🏆 Top Rated", top_carrier['carrier'], f"{top_carrier['rating']:.1f}⭐")
        with col2:
            best_ontime = display_carriers.loc[display_carriers['ontime_rate'].idxmax()]
            st.metric("✅ Best On-Time", best_ontime['carrier'], f"{best_ontime['ontime_rate']:.1f}%")
        with col3:
            lowest_issues = display_carriers.loc[display_carriers['issues'].idxmin()]
            st.metric("🟢 Fewest Issues", lowest_issues['carrier'], f"{lowest_issues['issues']} issues")
        with col4:
            lowest_cost = display_carriers.loc[display_carriers['total_cost'].idxmin()]
            st.metric("💰 Lowest Cost", lowest_cost['carrier'], lowest_cost['Total Cost'])
    else:
        st.info("No carriers match the current filters")

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_btn_1"):
        st.cache_data.clear()
        st.success("✅ Shipping data refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True, key="export_btn_1"):
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'shipments': records_df.to_dict('records'),
            'delivery_times': delivery_df.to_dict('records'),
            'shipping_costs': costs_df.to_dict('records'),
            'carrier_performance': carriers_df.to_dict('records'),
            'summary': {
                'total_shipments': total_shipments,
                'ontime_delivery_rate': f"{on_time_delivery:.1f}%",
                'address_issues': address_issues,
                'delayed_shipments': delayed_shipments
            }
        }
        st.success("✅ Shipping data exported successfully")
        st.json(export_data)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================
# ADDITIONAL INSIGHTS
# ===========================

with st.expander("📊 Shipping Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚚 Performance Highlights")
        st.markdown(f"""
        - **{on_time_delivery:.1f}%** on-time delivery rate
        - **{total_shipments:,}** total shipments processed
        - **{address_issues}** address validation issues detected
        - **{delayed_shipments}** delayed shipments requiring attention
        - **$119,613** total shipping costs across all carriers
        """)
        
        st.markdown("#### 💡 Recommended Actions")
        st.markdown("""
        1. **Immediate**: Address validation for flagged shipments
        2. **Short-term**: Investigate delayed shipments root causes
        3. **Medium-term**: Negotiate volume discounts with top carriers
        4. **Long-term**: Implement automated address verification system
        """)
    
    with col2:
        st.markdown("#### 📈 Performance Trends")
        
        carrier_performance = carriers_df.sort_values('ontime_rate', ascending=False)
        st.markdown("**Top Performing Carriers (On-Time Rate):**")
        for idx, row in carrier_performance.head(3).iterrows():
            st.markdown(f"- **{row['carrier']}**: {row['ontime_rate']:.1f}% ({row['total_shipments']:,} shipments)")
        
        st.markdown("#### 🎯 Optimization Opportunities")
        st.markdown("""
        - Reduce shipping costs by **$11,600/month** through optimization
        - Improve address accuracy from current to **99%+**
        - Target **95%+** on-time delivery across all carriers
        - Implement predictive analytics for delivery windows
        """)

# ===========================
# DIAGNOSTIC INFORMATION
# ===========================

with st.expander("🔧 System Diagnostics"):
    st.markdown("### Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Shipments", f"{len(records_df):,}")
        st.metric("Data Completeness", "100%")
    
    with col2:
        st.metric("Address Issues", f"{address_issues}")
        st.metric("Issue Rate", f"{address_issues/total_shipments*100:.2f}%")
    
    with col3:
        st.metric("Active Carriers", len(carriers_df))
        st.metric("Avg Delivery Time", f"{delivery_df['avg_delivery_days'].mean():.1f} days")
    
    st.markdown("---")
    st.markdown("### Filter Status")
    st.info(f"""
    **Active Filters:**
    - Date Range: {date_range}
    - Carriers: {', '.join(carrier_filter) if carrier_filter else 'None'}
    - Status: {', '.join(status_filter) if status_filter else 'None'}
    - Search Query: {'`' + search_query + '`' if search_query else 'None'}
    
    **Results:** {len(filtered_records):,} shipments shown out of {len(records_df):,} total
    """)