"""
↩️ Enhanced Returns & Refunds Audit - All 4 Tabs with Working Filters
Exact match to returns.html with return reason analysis and refund tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Returns & Refunds Audit",
    page_icon="↩️",
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
def generate_sample_return_data():
    """Generate sample return and refund data"""
    np.random.seed(42)
    
    # Return requests
    requests = []
    return_reasons = ['Defective Product', 'Wrong Item Received', 'Size/Fit Issue', 
                     'Quality Issues', 'Changed Mind', 'Not as Described']
    statuses = ['pending', 'approved', 'processing', 'completed', 'rejected']
    priorities = ['high', 'medium', 'low']
    products = ['Wireless Headphones', 'Running Shoes', 'Coffee Maker', 'Laptop Stand',
               'Smart Watch', 'Yoga Mat', 'Kitchen Blender', 'Desk Lamp', 'USB Cable',
               'Monitor Stand', 'Keyboard', 'Mouse Pad', 'Phone Case', 'Screen Protector']
    
    for i in range(1, 101):
        requests.append({
            'id': f'RET-{1000+i}',
            'order_id': f'ORD-{5000+i}',
            'customer': f'Customer {i}',
            'product': np.random.choice(products),
            'amount': round(np.random.uniform(30, 300), 2),
            'reason': np.random.choice(return_reasons),
            'request_date': (datetime.now() - timedelta(days=np.random.randint(0, 90))).date(),
            'status': np.random.choice(statuses, p=[0.15, 0.2, 0.1, 0.45, 0.1]),
            'priority': np.random.choice(priorities, p=[0.3, 0.5, 0.2])
        })
    
    # Refund processing
    refunds = []
    refund_methods = ['Original Payment', 'Store Credit', 'Bank Transfer']
    
    for i in range(1, 76):
        initiated_date = (datetime.now() - timedelta(days=np.random.randint(0, 90))).date()
        processing_days = np.random.randint(0, 30)
        completed_date = initiated_date + timedelta(days=processing_days) if np.random.random() > 0.2 else None
        
        refunds.append({
            'id': f'REF-{2000+i}',
            'return_id': f'RET-{1000+i}',
            'amount': round(np.random.uniform(30, 300), 2),
            'method': np.random.choice(refund_methods),
            'initiated_date': initiated_date,
            'completed_date': completed_date,
            'duration_days': processing_days,
            'status': 'completed' if completed_date else 'processing'
        })
    
    # Return reasons summary
    reasons_summary = []
    
    reason_data = {
        'Defective Product': {'count': 156, 'amount': '$12,456', 'trend': '+12%'},
        'Wrong Item Received': {'count': 134, 'amount': '$10,234', 'trend': '+8%'},
        'Size/Fit Issue': {'count': 98, 'amount': '$6,789', 'trend': '-5%'},
        'Quality Issues': {'count': 76, 'amount': '$4,567', 'trend': '+3%'},
        'Changed Mind': {'count': 45, 'amount': '$2,345', 'trend': '-2%'},
        'Not as Described': {'count': 34, 'amount': '$1,176', 'trend': '+15%'}
    }
    
    for reason, data in reason_data.items():
        total = sum(d['count'] for d in reason_data.values())
        percentage = (data['count'] / total) * 100
        reasons_summary.append({
            'reason': reason,
            'count': data['count'],
            'percentage': percentage,
            'amount': data['amount'],
            'trend': data['trend']
        })
    
    # Trends data
    trends = []
    months = ['Apr 2024', 'May 2024', 'Jun 2024', 'Jul 2024', 'Aug 2024', 'Sep 2024']
    for i, month in enumerate(months):
        returns_count = 412 + (i * 31)
        rate = 3.2 + (i * 0.2)
        refunds_amount = 28456 + (i * 1500)
        
        trends.append({
            'month': month,
            'returns': returns_count,
            'rate': rate,
            'refunds': refunds_amount
        })
    
    return (pd.DataFrame(requests), pd.DataFrame(refunds), 
            pd.DataFrame(reasons_summary), pd.DataFrame(trends))

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading return data..."):
    requests_df, refunds_df, reasons_df, trends_df = generate_sample_return_data()

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### ↩️ Filters")
    
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
    
    status_filter = st.multiselect(
        "📋 Status",
        ["pending", "approved", "processing", "completed", "rejected"],
        default=["pending", "approved", "processing", "completed", "rejected"]
    )
    
    reason_filter = st.multiselect(
        "📌 Return Reason",
        requests_df['reason'].unique().tolist(),
        default=requests_df['reason'].unique().tolist()
    )
    
    search_query = st.text_input("🔍 Search Return", placeholder="Return ID, Order ID...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_return_filters(df, date_cutoff, status_list, reason_list, search_text):
    filtered = df.copy()
    
    if date_cutoff:
        filtered = filtered[filtered['request_date'] >= date_cutoff]
    
    if status_list:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    if reason_list:
        filtered = filtered[filtered['reason'].isin(reason_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['id'].str.lower().str.contains(search_lower, na=False) |
            filtered['order_id'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_requests = apply_return_filters(requests_df, cutoff_date, status_filter, reason_filter, search_query)

# ===========================
# CALCULATE METRICS
# ===========================

total_returns = len(requests_df)
return_rate = (total_returns / 12899 * 100) if total_returns > 0 else 0  # Assuming 12,899 orders
total_refunded = requests_df['amount'].sum()
avg_processing_time = refunds_df[refunds_df['status'] == 'completed']['duration_days'].mean() if len(refunds_df[refunds_df['status'] == 'completed']) > 0 else 0

# ===========================
# HEADER & METRICS
# ===========================

st.title("↩️ Returns & Refunds Audit")
st.markdown("**Return reason analysis and refund tracking**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Total Returns", f"{total_returns}", "+45 this month")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Return Rate", f"{return_rate:.1f}%", "+0.3% vs last month")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-danger">', unsafe_allow_html=True)
    st.metric("Refund Amount", f"${total_refunded:,.0f}", "+$4,200 vs last month")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Avg Processing Time", f"{avg_processing_time:.0f} days", "-2 days improvement")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

st.markdown(f"""
<div class="alert alert-warning">
    <strong>Return Alert:</strong> {total_returns} total returns ({return_rate:.1f}% return rate). ${total_refunded:,.0f} in refunds processed. Average processing time: {avg_processing_time:.0f} days.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Return Trend Over Time")
    
    fig1 = px.line(trends_df, x='month', y=['returns', 'rate'], 
                   markers=True, labels={'value': 'Count', 'variable': 'Metric'})
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), hovermode='x unified')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Return Status Distribution")
    
    status_counts = requests_df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig2 = px.pie(status_counts, values='Count', names='Status',
                  color_discrete_sequence=['#fef3c7', '#dbeafe', '#e0e7ff', '#d1fae5', '#fee2e2'])
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 Return Requests ({len(filtered_requests)})",
    f"📌 Return Reasons",
    f"💳 Refund Processing",
    f"📈 Return Rate Trends"
])

# TAB 1: RETURN REQUESTS
with tab1:
    st.subheader("Return Requests")
    
    if len(filtered_requests) > 0:
        display_df = filtered_requests.copy()
        
        display_df['Amount'] = display_df['amount'].apply(lambda x: f"${x:.2f}")
        display_df['Status'] = display_df['status'].apply(
            lambda x: f"⏳ {x.upper()}" if x == "pending"
            else f"✅ {x.upper()}" if x == "approved"
            else f"⚙️ {x.upper()}" if x == "processing"
            else f"✔️ {x.upper()}" if x == "completed"
            else f"❌ {x.upper()}"
        )
        display_df['Priority'] = display_df['priority'].apply(
            lambda x: f"🔴 {x.upper()}" if x == "high"
            else f"🟡 {x.upper()}" if x == "medium"
            else f"🟢 {x.upper()}"
        )
        
        st.dataframe(
            display_df[['id', 'order_id', 'customer', 'product', 'Amount', 'reason', 
                       'request_date', 'Priority', 'Status']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'id': 'Return ID',
                'order_id': 'Order ID',
                'customer': 'Customer',
                'product': 'Product',
                'reason': 'Reason',
                'request_date': 'Request Date'
            }
        )
        
        st.caption(f"Showing {len(filtered_requests):,} of {len(requests_df):,} return requests")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ New Return Request", use_container_width=True):
                st.success("✅ Return request form opened")
        with col2:
            if st.button("📊 View Details", use_container_width=True):
                st.info("ℹ️ Select a return to view full details")
        with col3:
            if st.button("📥 Export Requests", use_container_width=True):
                st.success("✅ Return requests exported to CSV")
    else:
        st.info("No return requests match the current filters")

# TAB 2: RETURN REASONS
with tab2:
    st.subheader("Return Reason Analysis")
    
    # Reason bars
    st.markdown("#### Return Reasons Breakdown")
    
    for _, row in reasons_df.iterrows():
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.markdown(f"**{row['reason']}**")
        with col2:
            st.markdown(f"""
                <div style="background:#e2e8f0;border-radius:4px;overflow:hidden;height:30px;margin-top:8px;display:flex;align-items:center;padding:0 10px;color:white;font-weight:600;font-size:0.8125rem;background:linear-gradient(90deg,#ef4444,#f59e0b);">
                    {int(row['count'])} returns
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div style='font-weight:700;font-size:1rem;'>{row['percentage']:.1f}%</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Detailed Reason Breakdown")
        
        display_reasons = reasons_df.copy()
        display_reasons['Amount'] = display_reasons['amount']
        display_reasons['Trend'] = display_reasons['trend']
        
        st.dataframe(
            display_reasons[['reason', 'count', 'Amount', 'Trend']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'reason': 'Return Reason',
                'count': 'Count'
            }
        )
    
    with col2:
        st.markdown("#### Top Products Returned")
        
        top_products = pd.DataFrame({
            'Product': ['Wireless Headphones', 'Running Shoes', 'Smart Watch', 'Coffee Maker', 'Laptop Stand'],
            'Returns': [45, 38, 32, 28, 24],
            'Rate': ['12.3%', '8.9%', '7.2%', '6.8%', '5.4%'],
            'Refunded': ['$3,456', '$4,234', '$8,765', '$2,123', '$1,098']
        })
        
        st.dataframe(
            top_products,
            use_container_width=True,
            hide_index=True
        )

# TAB 3: REFUND PROCESSING
with tab3:
    st.subheader("Refund Processing")
    
    if len(refunds_df) > 0:
        display_refunds = refunds_df.copy()
        
        display_refunds['Amount'] = display_refunds['amount'].apply(lambda x: f"${x:.2f}")
        display_refunds['Status'] = display_refunds['status'].apply(
            lambda x: f"✔️ {x.upper()}" if x == "completed" else f"⚙️ {x.upper()}"
        )
        
        st.dataframe(
            display_refunds[['id', 'return_id', 'Amount', 'method', 'initiated_date', 
                           'completed_date', 'duration_days', 'Status']],
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                'id': 'Refund ID',
                'return_id': 'Return ID',
                'method': 'Method',
                'initiated_date': 'Initiated',
                'completed_date': 'Completed',
                'duration_days': 'Duration (Days)',
                'Status': 'Status'
            }
        )
    
    st.divider()
    
    # Refund metrics
    col1, col2, col3, col4 = st.columns(4)
    
    completed_refunds = refunds_df[refunds_df['status'] == 'completed']
    
    with col1:
        st.metric("Avg Processing Time", f"{avg_processing_time:.0f} days")
    with col2:
        st.metric("Fastest Refund", "0 days")
    with col3:
        slowest = refunds_df['duration_days'].max()
        st.metric("Slowest Refund", f"{slowest} days")
    with col4:
        on_time_rate = (len(completed_refunds[completed_refunds['duration_days'] <= 14]) / len(completed_refunds) * 100) if len(completed_refunds) > 0 else 0
        st.metric("On-Time Rate", f"{on_time_rate:.1f}%")
    
    st.divider()
    
    # Timeline example
    st.markdown("#### Refund Processing Timeline Example")
    
    timeline_events = [
        ("Oct 1, 2024 - 10:30 AM", "Return Requested", "Customer initiated return request"),
        ("Oct 2, 2024 - 2:15 PM", "Return Approved", "Return request approved by support team"),
        ("Oct 5, 2024 - 9:45 AM", "Item Received", "Returned item received at warehouse"),
        ("Oct 6, 2024 - 11:20 AM", "Quality Check", "Item inspected and approved for refund"),
        ("Oct 7, 2024 - 3:00 PM", "Refund Initiated", "Refund processed to original payment method"),
        ("Oct 9, 2024 - 8:30 AM", "Refund Completed", "Amount credited to customer account")
    ]
    
    for date, title, description in timeline_events:
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown("●")
        with col2:
            st.markdown(f"**{title}**")
            st.caption(f"{date} — {description}")

# TAB 4: RETURN RATE TRENDS
with tab4:
    st.subheader("Return Rate Trends (6-Month View)")
    
    display_trends = trends_df.copy()
    display_trends['Refunds'] = display_trends['refunds'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(
        display_trends[['month', 'returns', 'rate', 'Refunds']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'month': 'Month',
            'returns': 'Total Returns',
            'rate': 'Return Rate (%)'
        }
    )
    
    st.divider()
    
    # Trends metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Seasonal Patterns")
        
        seasonal_data = {
            'Q1 (Jan-Mar)': ('3.1%', 31),
            'Q2 (Apr-Jun)': ('3.5%', 35),
            'Q3 (Jul-Sep)': ('4.1%', 41)
        }
        
        for quarter, (rate, width) in seasonal_data.items():
            st.markdown(f"**{quarter}**")
            st.caption(f"{rate} avg return rate")
            st.progress(width / 100)
    
    with col2:
        st.markdown("#### Return Cost Impact")
        
        cost_breakdown = {
            'Refunds Issued': ('$34,567', '#ef4444'),
            'Restocking Costs': ('$5,234', '#f59e0b'),
            'Shipping Costs': ('$3,456', '#f59e0b'),
            'Total Impact': ('$43,257', '#ef4444')
        }
        
        for label, (amount, color) in cost_breakdown.items():
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.markdown(f"**{label}**")
            with col_right:
                st.markdown(f"<div style='color:{color};font-weight:700;'>{amount}</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### Benchmarks")
        
        benchmarks = [
            ("Industry Average", "5.5%", True),
            ("Your Current Rate", "4.2%", False),
            ("Target Goal", "3.0%", True)
        ]
        
        for label, value, is_positive in benchmarks:
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.markdown(f"**{label}**")
            with col_right:
                color = "#22c55e" if is_positive else "#f59e0b"
                st.markdown(f"<div style='color:{color};font-weight:700;'>{value}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")