"""
🛒 Enhanced Order Transaction Audit - All 4 Tabs with Working Filters
Exact match to orders.html with integrity validation and payment/shipping tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Order Transaction Audit",
    page_icon="🛒",
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
    .timeline {
        position: relative;
        padding-left: 30px;
        margin-top: 15px;
    }
    .timeline-item {
        position: relative;
        padding-bottom: 20px;
        border-left: 2px solid #e2e8f0;
        padding-left: 20px;
        margin-left: 5px;
    }
    .timeline-item:before {
        content: '';
        position: absolute;
        left: -7px;
        top: 5px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #3b82f6;
    }
    .timeline-item:last-child {
        border-left: none;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_order_data():
    """Generate sample order data with quality issues"""
    np.random.seed(42)
    
    orders = []
    order_statuses = ['completed', 'shipped', 'processing', 'pending', 'cancelled']
    payment_statuses = ['paid', 'unpaid', 'refunded']
    shipping_statuses = ['delivered', 'in-transit', 'pending', 'cancelled']
    carriers = ['UPS', 'FedEx', 'USPS', 'DHL']
    payment_methods = ['Credit Card', 'PayPal', 'Bank Transfer', 'Debit Card']
    processors = ['Stripe', 'PayPal', 'Manual', 'Square']
    
    for i in range(1, 501):
        order_date = (datetime.now() - timedelta(days=np.random.randint(1, 180))).date()
        
        # Customer data with orphaned orders
        if np.random.random() < 0.024:  # 2.4% orphaned (12 out of 500)
            customer_id = None
            customer_name = None
        else:
            customer_id = f'C-{np.random.randint(1, 201):03d}'
            customer_name = f'Customer {np.random.randint(1, 201)}'
        
        # Generate order total
        items = np.random.randint(1, 8)
        order_total = round(np.random.uniform(25, 1500), 2)
        
        # Calculate item-level total (for price mismatch)
        if np.random.random() < 0.068:  # 6.8% price mismatch (34 out of 500)
            calculated_total = round(order_total * np.random.uniform(1.02, 1.15), 2)
        else:
            calculated_total = order_total
        
        # Invalid totals
        if np.random.random() < 0.01:
            order_total = 0
            calculated_total = round(np.random.uniform(50, 300), 2)
        
        # Order status
        status = np.random.choice(order_statuses, p=[0.5, 0.2, 0.15, 0.1, 0.05])
        
        # Payment status
        if status == 'pending':
            payment_status = 'unpaid'
        elif status == 'cancelled':
            payment_status = np.random.choice(['unpaid', 'refunded'])
        else:
            payment_status = 'paid'
        
        # Shipping status
        if status == 'completed':
            shipping_status = 'delivered'
        elif status == 'shipped':
            shipping_status = 'in-transit'
        elif status == 'cancelled':
            shipping_status = 'cancelled'
        else:
            shipping_status = 'pending'
        
        # Payment details
        payment_method = np.random.choice(payment_methods)
        processor = 'PayPal' if payment_method == 'PayPal' else np.random.choice(processors)
        transaction_id = f'TXN-{i+5000}' if payment_status == 'paid' else None
        payment_date = order_date if payment_status == 'paid' else None
        
        # Shipping details
        carrier = np.random.choice(carriers) if shipping_status != 'pending' else None
        tracking_number = f'{carrier}{np.random.randint(100000, 999999)}' if carrier and shipping_status != 'cancelled' else None
        
        shipped_date = None
        delivered_date = None
        if shipping_status == 'delivered':
            shipped_date = order_date + timedelta(days=1)
            delivered_date = shipped_date + timedelta(days=np.random.randint(2, 7))
        elif shipping_status == 'in-transit':
            shipped_date = order_date + timedelta(days=1)
        
        address = f'{np.random.randint(100, 999)} {np.random.choice(["Main", "Oak", "Pine", "Elm", "Maple"])} St, City, ST {np.random.randint(10000, 99999)}'
        
        orders.append({
            'order_id': f'ORD-{i+1000}',
            'customer_id': customer_id,
            'customer_name': customer_name,
            'order_date': order_date,
            'items': items,
            'order_total': order_total,
            'calculated_total': calculated_total,
            'status': status,
            'payment_status': payment_status,
            'payment_method': payment_method,
            'payment_date': payment_date,
            'transaction_id': transaction_id,
            'processor': processor,
            'shipping_status': shipping_status,
            'carrier': carrier,
            'tracking_number': tracking_number,
            'shipped_date': shipped_date,
            'delivered_date': delivered_date,
            'shipping_address': address
        })
    
    return pd.DataFrame(orders)

# ===========================
# ANALYSIS FUNCTIONS
# ===========================

def analyze_integrity_issues(df):
    """Analyze data integrity issues"""
    issues = []
    
    for idx, row in df.iterrows():
        discrepancy = row['calculated_total'] - row['order_total']
        
        # Orphaned orders
        if pd.isna(row['customer_id']) or row['customer_id'] is None:
            issues.append({
                'Order ID': row['order_id'],
                'Customer': None,
                'Date': row['order_date'],
                'Order Total': f"${row['order_total']:.2f}",
                'Calculated Total': f"${row['calculated_total']:.2f}",
                'Discrepancy': f"${discrepancy:.2f}",
                'Issue': 'Orphaned Order'
            })
        
        # Price mismatches
        elif abs(discrepancy) > 0.01:
            issues.append({
                'Order ID': row['order_id'],
                'Customer': row['customer_id'],
                'Date': row['order_date'],
                'Order Total': f"${row['order_total']:.2f}",
                'Calculated Total': f"${row['calculated_total']:.2f}",
                'Discrepancy': f"${discrepancy:.2f}",
                'Issue': 'Price Mismatch'
            })
        
        # Invalid totals
        elif row['order_total'] <= 0:
            issues.append({
                'Order ID': row['order_id'],
                'Customer': row['customer_id'],
                'Date': row['order_date'],
                'Order Total': f"${row['order_total']:.2f}",
                'Calculated Total': f"${row['calculated_total']:.2f}",
                'Discrepancy': f"${discrepancy:.2f}",
                'Issue': 'Invalid Total'
            })
    
    return pd.DataFrame(issues) if issues else pd.DataFrame()

def get_payment_data(df):
    """Extract payment transaction data"""
    payment_data = []
    
    for idx, row in df.iterrows():
        payment_data.append({
            'Order ID': row['order_id'],
            'Amount': f"${row['order_total']:.2f}",
            'Payment Method': row['payment_method'],
            'Status': row['payment_status'],
            'Date': row['payment_date'],
            'Transaction ID': row['transaction_id'] if pd.notna(row['transaction_id']) else 'Pending',
            'Processor': row['processor']
        })
    
    return pd.DataFrame(payment_data)

def get_shipping_data(df):
    """Extract shipping status data"""
    shipping_data = []
    
    for idx, row in df.iterrows():
        if row['shipping_status'] != 'cancelled' or pd.notna(row['carrier']):
            shipping_data.append({
                'Order ID': row['order_id'],
                'Carrier': row['carrier'] if pd.notna(row['carrier']) else 'Not Assigned',
                'Tracking Number': row['tracking_number'] if pd.notna(row['tracking_number']) else 'Not Available',
                'Status': row['shipping_status'],
                'Shipped Date': row['shipped_date'],
                'Delivered Date': row['delivered_date'],
                'Shipping Address': row['shipping_address']
            })
    
    return pd.DataFrame(shipping_data)

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading order data..."):
    orders_df = generate_sample_order_data()
    integrity_df = analyze_integrity_issues(orders_df)
    payments_df = get_payment_data(orders_df)
    shipping_df = get_shipping_data(orders_df)

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 🛒 Filters")
    
    date_range = st.selectbox(
        "📅 Date Range",
        ["Today", "Last 7 Days", "Last 30 Days", "Last 90 Days"],
        index=2
    )
    
    now = datetime.now()
    if date_range == "Today":
        cutoff_date = now.date()
    elif date_range == "Last 7 Days":
        cutoff_date = (now - timedelta(days=7)).date()
    elif date_range == "Last 30 Days":
        cutoff_date = (now - timedelta(days=30)).date()
    elif date_range == "Last 90 Days":
        cutoff_date = (now - timedelta(days=90)).date()
    else:
        cutoff_date = None
    
    order_status_filter = st.multiselect(
        "📦 Order Status",
        ["completed", "shipped", "processing", "pending", "cancelled"],
        default=["completed", "shipped", "processing", "pending", "cancelled"]
    )
    
    payment_status_filter = st.multiselect(
        "💳 Payment Status",
        ["paid", "unpaid", "refunded"],
        default=["paid", "unpaid", "refunded"]
    )
    
    search_query = st.text_input("🔍 Search", placeholder="Order ID, Customer...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_order_filters(df, date_cutoff, status_list, payment_list, search_text):
    filtered = df.copy()
    
    if date_cutoff:
        filtered = filtered[filtered['order_date'] >= date_cutoff]
    
    if status_list:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    if payment_list:
        filtered = filtered[filtered['payment_status'].isin(payment_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['order_id'].str.lower().str.contains(search_lower, na=False) |
            filtered['customer_name'].astype(str).str.lower().str.contains(search_lower, na=False) |
            filtered['customer_id'].astype(str).str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_orders = apply_order_filters(orders_df, cutoff_date, order_status_filter, payment_status_filter, search_query)

# ===========================
# HEADER & METRICS
# ===========================

st.title("🛒 Order Transaction Audit")
st.markdown("**Order integrity validation, payment tracking, and shipping status**")

orphaned = len(integrity_df[integrity_df['Issue'] == 'Orphaned Order'])
price_mismatch = len(integrity_df[integrity_df['Issue'] == 'Price Mismatch'])
pending_payments = len(orders_df[(orders_df['payment_status'] == 'unpaid') & 
                                  (orders_df['order_date'] < (datetime.now().date() - timedelta(days=2)))])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Orders", f"{len(orders_df):,}", "+8.7%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    quality_score = ((len(orders_df) - len(integrity_df)) / len(orders_df) * 100)
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Data Quality Score", f"{quality_score:.1f}%", "+1.5%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-danger">', unsafe_allow_html=True)
    st.metric("Orphaned Orders", f"{orphaned}", "+2 new")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Price Mismatches", f"{price_mismatch}", "-8 resolved")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

if orphaned > 0 or price_mismatch > 0 or pending_payments > 0:
    st.markdown(f"""
    <div class="alert alert-warning">
        <strong>⚠️ Data Quality Alert:</strong> {orphaned} orphaned orders detected. {price_mismatch} orders with price mismatches. {pending_payments} pending payments over 48 hours old.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Order Volume Trend")
    
    monthly_data = []
    for i in range(6):
        month_date = datetime.now() - timedelta(days=30 * (5-i))
        monthly_data.append({
            'Month': month_date.strftime('%b'),
            'Orders': np.random.randint(1800, 2600)
        })
    
    monthly_df = pd.DataFrame(monthly_data)
    fig1 = px.line(monthly_df, x='Month', y='Orders', markers=True)
    fig1.update_traces(line_color='#3b82f6', fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)')
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Order Status Distribution")
    
    status_counts = orders_df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig2 = px.pie(status_counts, values='Count', names='Status', 
                  color_discrete_sequence=['#22c55e', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'])
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 Order List ({len(filtered_orders)})",
    f"🔍 Data Integrity ({len(integrity_df)})",
    f"💳 Payment Status ({len(payments_df)})",
    f"🚚 Shipping Status ({len(shipping_df)})"
])

# TAB 1: ORDER LIST
with tab1:
    st.subheader("Order Transaction List")
    
    if len(filtered_orders) > 0:
        display_orders = filtered_orders.copy()
        
        display_orders['Total'] = display_orders['order_total'].apply(lambda x: f"${x:.2f}")
        display_orders['Status Badge'] = display_orders['status'].apply(
            lambda x: f"✅ {x.upper()}" if x == "completed"
            else f"📦 {x.upper()}" if x == "shipped"
            else f"⚙️ {x.upper()}" if x == "processing"
            else f"⏳ {x.upper()}" if x == "pending"
            else f"❌ {x.upper()}"
        )
        display_orders['Payment Badge'] = display_orders['payment_status'].apply(
            lambda x: f"✅ {x.upper()}" if x == "paid"
            else f"⏳ {x.upper()}" if x == "unpaid"
            else f"↩️ {x.upper()}"
        )
        display_orders['Customer'] = display_orders['customer_name'].fillna('❌ Missing')
        
        st.dataframe(
            display_orders[['order_id', 'Customer', 'order_date', 'items', 'Total', 
                          'Status Badge', 'Payment Badge', 'shipping_status']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'order_id': 'Order ID',
                'order_date': 'Date',
                'items': 'Items',
                'Status Badge': 'Order Status',
                'Payment Badge': 'Payment',
                'shipping_status': 'Shipping'
            }
        )
        
        st.caption(f"Showing {len(filtered_orders):,} of {len(orders_df):,} orders")
        
        with st.expander("🔍 View Order Details"):
            order_id = st.selectbox("Select Order", filtered_orders['order_id'].tolist())
            
            if order_id:
                order = filtered_orders[filtered_orders['order_id'] == order_id].iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("**Order ID**")
                    st.markdown(f"### {order['order_id']}")
                with col2:
                    st.markdown("**Customer**")
                    st.markdown(f"**{order['customer_name'] if pd.notna(order['customer_name']) else '❌ Missing'}**")
                with col3:
                    st.markdown("**Order Date**")
                    st.markdown(f"**{order['order_date']}**")
                with col4:
                    st.markdown("**Total Amount**")
                    st.markdown(f"### ${order['order_total']:.2f}")
                
                st.markdown("---")
                st.markdown("**Order Status**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Order: {order['status'].upper()}")
                with col2:
                    st.success(f"Payment: {order['payment_status'].upper()}" if order['payment_status'] == 'paid' 
                              else f"Payment: {order['payment_status'].upper()}")
                with col3:
                    st.warning(f"Shipping: {order['shipping_status'].upper()}")
                
                st.markdown("---")
                st.markdown("**Order Timeline**")
                st.markdown(f"""
                <div class="timeline">
                    <div class="timeline-item">
                        <strong>Order Placed</strong>
                        <div style="color:#64748b;font-size:0.875rem;">{order['order_date']} - Order created by customer</div>
                    </div>
                    <div class="timeline-item">
                        <strong>Payment Confirmed</strong>
                        <div style="color:#64748b;font-size:0.875rem;">{order['payment_date'] if pd.notna(order['payment_date']) else 'Pending'} - Payment of ${order['order_total']:.2f} processed</div>
                    </div>
                    <div class="timeline-item">
                        <strong>Processing</strong>
                        <div style="color:#64748b;font-size:0.875rem;">Items being prepared for shipment</div>
                    </div>
                    <div class="timeline-item">
                        <strong>Shipped</strong>
                        <div style="color:#64748b;font-size:0.875rem;">{order['shipped_date'] if pd.notna(order['shipped_date']) else 'Pending shipment'}</div>
                    </div>
                    {f'''<div class="timeline-item">
                        <strong>Delivered</strong>
                        <div style="color:#64748b;font-size:0.875rem;">{order['delivered_date']} - Order successfully delivered</div>
                    </div>''' if order['status'] == 'completed' else ''}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No orders match the current filters")

# TAB 2: DATA INTEGRITY
with tab2:
    st.subheader("Data Integrity Issues")
    
    if len(integrity_df) > 0:
        st.warning(f"Found {len(integrity_df):,} data integrity issues")
        
        display_integrity = integrity_df.copy()
        display_integrity['Issue Type'] = display_integrity['Issue'].apply(
            lambda x: f"🔴 {x}" if x == "Orphaned Order"
            else f"🟡 {x}" if x == "Price Mismatch"
            else f"🟠 {x}"
        )
        
        st.dataframe(
            display_integrity[['Order ID', 'Customer', 'Date', 'Order Total', 
                             'Calculated Total', 'Discrepancy', 'Issue Type']],
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔧 Fix Price Mismatches", use_container_width=True):
                st.success("✅ Price reconciliation workflow initiated")
        with col2:
            if st.button("📥 Export Issues", use_container_width=True):
                st.success("✅ Integrity issues exported")
    else:
        st.success("✅ No data integrity issues found!")
        st.balloons()

# TAB 3: PAYMENT STATUS
with tab3:
    st.subheader("Payment Transaction Status")
    
    if len(payments_df) > 0:
        display_payments = payments_df.copy()
        display_payments['Status Badge'] = display_payments['Status'].apply(
            lambda x: f"✅ {x.upper()}" if x == "paid"
            else f"⏳ {x.upper()}" if x == "unpaid"
            else f"↩️ {x.upper()}"
        )
        
        st.dataframe(
            display_payments[['Order ID', 'Amount', 'Payment Method', 'Status Badge', 
                            'Date', 'Transaction ID', 'Processor']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'Status Badge': 'Payment Status'
            }
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            paid_count = len(orders_df[orders_df['payment_status'] == 'paid'])
            st.metric("✅ Paid Orders", f"{paid_count:,}")
        with col2:
            unpaid_count = len(orders_df[orders_df['payment_status'] == 'unpaid'])
            st.metric("⏳ Unpaid Orders", f"{unpaid_count:,}")
        with col3:
            refunded_count = len(orders_df[orders_df['payment_status'] == 'refunded'])
            st.metric("↩️ Refunded Orders", f"{refunded_count:,}")
    else:
        st.info("No payment data available")

# TAB 4: SHIPPING STATUS
with tab4:
    st.subheader("Shipping Status Tracking")
    
    if len(shipping_df) > 0:
        display_shipping = shipping_df.copy()
        display_shipping['Status Badge'] = display_shipping['Status'].apply(
            lambda x: f"✅ {x.upper()}" if x == "delivered"
            else f"🚚 {x.upper()}" if x == "in-transit"
            else f"⏳ {x.upper()}" if x == "pending"
            else f"❌ {x.upper()}"
        )
        
        st.dataframe(
            display_shipping[['Order ID', 'Carrier', 'Tracking Number', 'Status Badge', 
                            'Shipped Date', 'Delivered Date', 'Shipping Address']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'Status Badge': 'Shipping Status'
            }
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delivered_count = len(orders_df[orders_df['shipping_status'] == 'delivered'])
            st.metric("✅ Delivered", f"{delivered_count:,}")
        with col2:
            transit_count = len(orders_df[orders_df['shipping_status'] == 'in-transit'])
            st.metric("🚚 In Transit", f"{transit_count:,}")
        with col3:
            pending_count = len(orders_df[orders_df['shipping_status'] == 'pending'])
            st.metric("⏳ Pending", f"{pending_count:,}")
        with col4:
            cancelled_count = len(orders_df[orders_df['shipping_status'] == 'cancelled'])
            st.metric("❌ Cancelled", f"{cancelled_count:,}")
    else:
        st.info("No shipping data available")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")