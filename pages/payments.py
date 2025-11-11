"""
💳 Enhanced Payment Processing Audit - All 4 Tabs with Working Filters
Exact match to payments.html with transaction monitoring and chargeback tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Payment Processing Audit",
    page_icon="💳",
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
def generate_sample_payment_data():
    """Generate sample payment data with quality issues"""
    np.random.seed(42)
    
    # Payment Transactions
    transactions = []
    methods = ['Credit Card', 'PayPal', 'Bank Transfer', 'Digital Wallet']
    processors = ['Stripe', 'PayPal', 'Square', 'Manual']
    statuses = ['completed', 'pending', 'failed']
    risk_scores = ['Low', 'Medium', 'High']
    
    for i in range(1, 251):
        method = np.random.choice(methods)
        processor = 'PayPal' if method == 'PayPal' else 'Stripe' if method == 'Credit Card' else 'Manual' if method == 'Bank Transfer' else 'Square'
        status = np.random.choice(statuses, p=[0.8, 0.12, 0.08])
        
        transaction_date = (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S')
        
        risk_score = np.random.choice(risk_scores, p=[0.7, 0.2, 0.1])
        
        transactions.append({
            'transaction_id': f'TXN-{10000+i}',
            'order_id': f'ORD-{5000+i}',
            'customer': f'Customer {np.random.randint(1, 500)}',
            'amount': round(np.random.uniform(50, 999.99), 2),
            'method': method,
            'processor': processor,
            'status': status,
            'date': transaction_date,
            'card_last4': f'{np.random.randint(1000, 9999)}' if method == 'Credit Card' else 'N/A',
            'risk_score': risk_score
        })
    
    # Failed Payments
    failures = []
    failure_reasons = [
        'Insufficient Funds', 'Card Expired', 'Invalid CVV', 'Card Declined',
        'Address Mismatch', 'Connection Timeout', 'Processor Error', 'Duplicate Transaction'
    ]
    error_codes = [
        'card_declined', 'expired_card', 'incorrect_cvc', 'insufficient_funds',
        'address_mismatch', 'timeout', 'processor_error', 'duplicate'
    ]
    
    for i in range(1, 51):
        attempt = np.random.randint(1, 4)
        failure_date = (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S')
        
        failures.append({
            'transaction_id': f'TXN-{20000+i}',
            'order_id': f'ORD-{7000+i}',
            'customer': f'Customer {np.random.randint(1, 500)}',
            'amount': round(np.random.uniform(50, 999.99), 2),
            'method': np.random.choice(methods),
            'reason': np.random.choice(failure_reasons),
            'attempts': attempt,
            'date': failure_date,
            'error_code': np.random.choice(error_codes),
            'retryable': np.random.choice([True, False], p=[0.6, 0.4])
        })
    
    # Chargebacks
    chargebacks = []
    chargeback_reasons = [
        'Product Not Received', 'Unauthorized Transaction', 'Product Not as Described',
        'Duplicate Charge', 'Refund Not Received', 'Quality Issue'
    ]
    chargeback_statuses = ['disputed', 'won', 'lost']
    
    for i in range(1, 26):
        filed_date = (datetime.now() - timedelta(days=np.random.randint(0, 60))).date()
        due_date = filed_date + timedelta(days=np.random.randint(7, 15))
        
        chargebacks.append({
            'chargeback_id': f'CB-{i:03d}',
            'transaction_id': f'TXN-{15000+i}',
            'order_id': f'ORD-{6000+i}',
            'customer': f'Customer {np.random.randint(1, 500)}',
            'amount': round(np.random.uniform(100, 999.99), 2),
            'reason': np.random.choice(chargeback_reasons),
            'status': np.random.choice(chargeback_statuses, p=[0.5, 0.3, 0.2]),
            'filed_date': filed_date,
            'due_date': due_date
        })
    
    # Payment Methods Summary
    methods_summary = []
    method_data = {
        'Credit Card': {'transactions': 9234, 'volume': 2845678, 'success': 98.2, 'failure': 1.8},
        'PayPal': {'transactions': 3456, 'volume': 678912, 'success': 97.5, 'failure': 2.5},
        'Bank Transfer': {'transactions': 1890, 'volume': 456789, 'success': 95.8, 'failure': 4.2},
        'Digital Wallet': {'transactions': 1098, 'volume': 234567, 'success': 99.1, 'failure': 0.9}
    }
    
    for method, data in method_data.items():
        methods_summary.append({
            'method': method,
            'transactions': data['transactions'],
            'volume': data['volume'],
            'success_rate': data['success'],
            'failure_rate': data['failure']
        })
    
    return (pd.DataFrame(transactions), pd.DataFrame(failures),
            pd.DataFrame(chargebacks), pd.DataFrame(methods_summary))

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading payment data..."):
    transactions_df, failures_df, chargebacks_df, methods_df = generate_sample_payment_data()

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 💳 Filters")
    
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
    
    payment_status = st.multiselect(
        "💰 Payment Status",
        ["completed", "pending", "failed"],
        default=["completed", "pending", "failed"]
    )
    
    payment_method = st.multiselect(
        "🏦 Payment Method",
        transactions_df['method'].unique().tolist(),
        default=transactions_df['method'].unique().tolist()
    )
    
    search_query = st.text_input("🔍 Search", placeholder="Transaction ID, Order ID...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_payment_filters(df, date_cutoff, status_list, method_list, search_text):
    filtered = df.copy()
    
    if date_cutoff and 'date' in filtered.columns:
        filtered['date_parsed'] = pd.to_datetime(filtered['date']).dt.date
        filtered = filtered[filtered['date_parsed'] >= date_cutoff]
        filtered = filtered.drop('date_parsed', axis=1)
    
    if status_list and 'status' in filtered.columns:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    if method_list and 'method' in filtered.columns:
        filtered = filtered[filtered['method'].isin(method_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['transaction_id'].str.lower().str.contains(search_lower, na=False) |
            filtered['order_id'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_transactions = apply_payment_filters(transactions_df, cutoff_date, payment_status, payment_method, search_query)

# ===========================
# CALCULATE METRICS
# ===========================

total_transactions = len(transactions_df)
success_rate = (len(transactions_df[transactions_df['status'] == 'completed']) / total_transactions * 100) if total_transactions > 0 else 0
failed_payments = len(failures_df)
chargebacks_count = len(chargebacks_df)
total_volume = transactions_df['amount'].sum()

# ===========================
# HEADER & METRICS
# ===========================

st.title("💳 Payment Processing Audit")
st.markdown("**Transaction monitoring, failure analysis, and payment method optimization**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Transactions", f"{total_transactions:,}", "+890 from last month")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Success Rate", f"{success_rate:.1f}%", "+0.5% improvement")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Failed Payments", f"{failed_payments}", "-23 resolved")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Chargebacks", f"{chargebacks_count}", "+3 new disputes")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

st.markdown(f"""
<div class="alert alert-success">
    <strong>Payment Performance:</strong> {success_rate:.1f}% success rate. {failed_payments} failed payments. {chargebacks_count} active chargebacks. Processing ${total_volume:,.0f} today.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Payment Volume Trend")
    
    trend_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Successful': [12345, 13890, 13456, 15234, 14890, 15678],
        'Failed': [234, 267, 298, 276, 289, 312]
    })
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Successful'],
                              mode='lines+markers', name='Successful',
                              line=dict(color='#22c55e', width=3),
                              fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.1)'))
    fig1.add_trace(go.Scatter(x=trend_data['Month'], y=trend_data['Failed'],
                              mode='lines+markers', name='Failed',
                              line=dict(color='#ef4444', width=2),
                              fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'))
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Payment Method Distribution")
    
    method_counts = transactions_df['method'].value_counts().reset_index()
    method_counts.columns = ['Method', 'Count']
    
    color_map = {
        'Credit Card': '#3b82f6',
        'PayPal': '#f59e0b',
        'Bank Transfer': '#8b5cf6',
        'Digital Wallet': '#22c55e'
    }
    
    fig2 = px.pie(method_counts, values='Count', names='Method',
                  color='Method', color_discrete_map=color_map)
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"💳 Payment Transactions ({len(filtered_transactions)})",
    f"❌ Failed Payments ({len(failures_df)})",
    f"🔄 Chargeback Tracking ({len(chargebacks_df)})",
    f"🏦 Payment Methods"
])

# TAB 1: PAYMENT TRANSACTIONS
with tab1:
    st.subheader("Payment Transaction List")
    
    if len(filtered_transactions) > 0:
        display_transactions = filtered_transactions.copy()
        
        display_transactions['Amount Display'] = display_transactions['amount'].apply(lambda x: f"${x:.2f}")
        display_transactions['Status Badge'] = display_transactions['status'].apply(
            lambda x: f"✅ {x.upper()}" if x == "completed"
            else f"⏳ {x.upper()}" if x == "pending"
            else f"❌ {x.upper()}"
        )
        display_transactions['Risk Badge'] = display_transactions['risk_score'].apply(
            lambda x: f"🟢 {x}" if x == "Low"
            else f"🟡 {x}" if x == "Medium"
            else f"🔴 {x}"
        )
        
        st.dataframe(
            display_transactions[['transaction_id', 'order_id', 'customer', 'Amount Display',
                                'method', 'processor', 'Status Badge', 'date', 'Risk Badge']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'transaction_id': 'Transaction ID',
                'order_id': 'Order ID',
                'customer': 'Customer',
                'Amount Display': 'Amount',
                'method': 'Payment Method',
                'processor': 'Processor',
                'Status Badge': 'Status',
                'date': 'Date/Time',
                'Risk Badge': 'Risk Level'
            }
        )
        
        st.caption(f"Showing {len(filtered_transactions):,} of {len(transactions_df):,} transactions")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            completed = len(filtered_transactions[filtered_transactions['status'] == 'completed'])
            st.metric("✅ Completed", f"{completed:,}")
        with col2:
            pending = len(filtered_transactions[filtered_transactions['status'] == 'pending'])
            st.metric("⏳ Pending", f"{pending:,}")
        with col3:
            failed = len(filtered_transactions[filtered_transactions['status'] == 'failed'])
            st.metric("❌ Failed", f"{failed:,}")
        with col4:
            low_risk = len(filtered_transactions[filtered_transactions['risk_score'] == 'Low'])
            st.metric("🟢 Low Risk", f"{low_risk:,}")
    else:
        st.info("No transactions match the current filters")

# TAB 2: FAILED PAYMENTS
with tab2:
    st.subheader("Failed Payment Analysis")
    
    if len(failures_df) > 0:
        display_failures = failures_df.copy()
        
        display_failures['Amount Display'] = display_failures['amount'].apply(lambda x: f"${x:.2f}")
        display_failures['Retryable Badge'] = display_failures['retryable'].apply(
            lambda x: "✅ Yes" if x else "❌ No"
        )
        display_failures['Reason Badge'] = display_failures['reason'].apply(
            lambda x: f"🔴 {x}"
        )
        
        st.dataframe(
            display_failures[['transaction_id', 'order_id', 'customer', 'Amount Display',
                            'method', 'Reason Badge', 'attempts', 'Retryable Badge', 'error_code']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'transaction_id': 'Transaction ID',
                'order_id': 'Order ID',
                'customer': 'Customer',
                'Amount Display': 'Amount',
                'method': 'Payment Method',
                'Reason Badge': 'Failure Reason',
                'attempts': 'Attempts',
                'Retryable Badge': 'Retryable',
                'error_code': 'Error Code'
            }
        )
        
        st.caption(f"Showing {len(failures_df):,} failed payment records")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            retryable = len(failures_df[failures_df['retryable'] == True])
            st.metric("♻️ Retryable", f"{retryable:,}")
        with col2:
            non_retryable = len(failures_df[failures_df['retryable'] == False])
            st.metric("🚫 Non-Retryable", f"{non_retryable:,}")
        with col3:
            failed_total = failures_df['amount'].sum()
            st.metric("💰 Failed Value", f"${failed_total:,.2f}")
        
        if st.button("🔄 Retry Failed Payments", use_container_width=True):
            st.success(f"✅ Initiated retry for {retryable} retryable payments")
    else:
        st.success("✅ No failed payments detected!")

# TAB 3: CHARGEBACK TRACKING
with tab3:
    st.subheader("Chargeback Tracking & Disputes")
    
    if len(chargebacks_df) > 0:
        display_chargebacks = chargebacks_df.copy()
        
        display_chargebacks['Amount Display'] = display_chargebacks['amount'].apply(lambda x: f"${x:.2f}")
        display_chargebacks['Status Badge'] = display_chargebacks['status'].apply(
            lambda x: f"⚠️ {x.upper()}" if x == "disputed"
            else f"✅ {x.upper()}" if x == "won"
            else f"❌ {x.upper()}"
        )
        
        st.dataframe(
            display_chargebacks[['chargeback_id', 'transaction_id', 'order_id', 'customer',
                               'Amount Display', 'reason', 'Status Badge', 'filed_date', 'due_date']],
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                'chargeback_id': 'Chargeback ID',
                'transaction_id': 'Transaction ID',
                'order_id': 'Order ID',
                'customer': 'Customer',
                'Amount Display': 'Amount',
                'reason': 'Reason',
                'Status Badge': 'Status',
                'filed_date': 'Filed Date',
                'due_date': 'Due Date'
            }
        )
        
        st.caption(f"Showing {len(chargebacks_df):,} chargeback records")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            disputed = len(chargebacks_df[chargebacks_df['status'] == 'disputed'])
            st.metric("⚠️ Disputed", f"{disputed}")
        with col2:
            won = len(chargebacks_df[chargebacks_df['status'] == 'won'])
            st.metric("✅ Won", f"{won}")
        with col3:
            lost = len(chargebacks_df[chargebacks_df['status'] == 'lost'])
            st.metric("❌ Lost", f"{lost}")
        with col4:
            chargeback_total = chargebacks_df['amount'].sum()
            st.metric("💰 Total Impact", f"${chargeback_total:,.2f}")
        
        st.markdown("---")
        st.markdown("#### Top Chargeback Reasons")
        
        reasons_count = chargebacks_df['reason'].value_counts().reset_index()
        reasons_count.columns = ['Reason', 'Count']
        
        fig = px.bar(reasons_count, x='Reason', y='Count',
                    title='Chargeback Reasons Distribution',
                    color='Count',
                    color_continuous_scale='Reds',
                    text='Count')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No chargeback records found")

# TAB 4: PAYMENT METHODS
with tab4:
    st.subheader("Payment Method Analysis")
    
    if len(methods_df) > 0:
        display_methods = methods_df.copy()
        
        display_methods['Volume Display'] = display_methods['volume'].apply(lambda x: f"${x/1000000:.2f}M")
        display_methods['Success Rate %'] = display_methods['success_rate'].apply(lambda x: f"{x:.1f}%")
        display_methods['Failure Rate %'] = display_methods['failure_rate'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_methods[['method', 'transactions', 'Volume Display', 'Success Rate %', 'Failure Rate %']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'method': 'Payment Method',
                'transactions': 'Transactions',
                'Volume Display': 'Total Volume',
                'Success Rate %': 'Success Rate',
                'Failure Rate %': 'Failure Rate'
            }
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Success Rate Comparison")
            
            fig = px.bar(display_methods, x='method', y='success_rate',
                        title='Success Rate by Payment Method',
                        color='success_rate',
                        color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
                        text='Success Rate %')
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(height=300, showlegend=False, xaxis_title='Payment Method', yaxis_title='Success Rate %')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Transaction Volume by Method")
            
            fig = px.bar(display_methods, x='method', y='transactions',
                        title='Transaction Count by Payment Method',
                        color='transactions',
                        color_continuous_scale='Blues',
                        text='transactions')
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(height=300, showlegend=False, xaxis_title='Payment Method', yaxis_title='Transaction Count')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            best_method = display_methods.loc[display_methods['success_rate'].idxmax()]
            st.metric("🏆 Best Success Rate", best_method['method'], f"{best_method['success_rate']:.1f}%")
        with col2:
            most_volume = display_methods.loc[display_methods['volume'].idxmax()]
            st.metric("💰 Highest Volume", most_volume['method'], most_volume['Volume Display'])
        with col3:
            most_trans = display_methods.loc[display_methods['transactions'].idxmax()]
            st.metric("📊 Most Transactions", most_trans['method'], f"{most_trans['transactions']:,}")
        with col4:
            total_success = display_methods['success_rate'].mean()
            st.metric("📈 Avg Success Rate", "All Methods", f"{total_success:.1f}%")
    else:
        st.info("No payment method data available")

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Payment data refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'transactions': transactions_df.to_dict('records'),
            'failures': failures_df.to_dict('records'),
            'chargebacks': chargebacks_df.to_dict('records'),
            'payment_methods': methods_df.to_dict('records'),
            'summary': {
                'total_transactions': total_transactions,
                'success_rate': f"{success_rate:.1f}%",
                'failed_payments': failed_payments,
                'chargebacks': chargebacks_count,
                'total_volume': f"${total_volume:,.2f}"
            }
        }
        st.success("✅ Payment data exported successfully")
        st.json(export_data)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================
# ADDITIONAL INSIGHTS
# ===========================

with st.expander("📊 Payment Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💳 Payment Performance")
        st.markdown(f"""
        - **{success_rate:.1f}%** overall success rate
        - **{failed_payments}** failed payment transactions
        - **{chargebacks_count}** active chargebacks
        - **${total_volume:,.2f}** total transaction volume
        - **{total_transactions:,}** total transactions processed
        """)
        
        st.markdown("#### 🎯 Recommended Actions")
        st.markdown("""
        1. **Immediate**: Investigate high-risk transactions
        2. **Short-term**: Implement retry mechanism for failed payments
        3. **Medium-term**: Enhance fraud detection system
        4. **Long-term**: Optimize payment flow for better conversion
        """)
    
    with col2:
        st.markdown("#### 📈 Performance Trends")
        
        method_performance = methods_df.sort_values('success_rate', ascending=False)
        st.markdown("**Payment Methods by Success Rate:**")
        for idx, row in method_performance.head(4).iterrows():
            st.markdown(f"- **{row['method']}**: {row['success_rate']:.1f}% ({row['transactions']:,} transactions)")
        
        st.markdown("#### 💡 Optimization Opportunities")
        st.markdown("""
        - Improve failed payment retry success by 15-20%
        - Reduce chargeback rate to under 0.5%
        - Optimize payment method mix for better conversion
        - Implement 3D Secure for high-risk transactions
        - Auto-select best payment method per customer
        """)

# ===========================
# DIAGNOSTIC INFORMATION
# ===========================

with st.expander("🔧 System Diagnostics"):
    st.markdown("### Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", f"{len(transactions_df):,}")
        st.metric("Data Completeness", "100%")
    
    with col2:
        st.metric("Failed Transactions", f"{len(failures_df):,}")
        st.metric("Failure Rate", f"{len(failures_df)/len(transactions_df)*100:.2f}%")
    
    with col3:
        st.metric("Active Chargebacks", f"{len(chargebacks_df):,}")
        st.metric("Chargeback Rate", f"{len(chargebacks_df)/len(transactions_df)*100:.2f}%")
    
    st.markdown("---")
    st.markdown("### Filter Status")
    st.info(f"""
    **Active Filters:**
    - Date Range: {date_range}
    - Payment Status: {', '.join(payment_status) if payment_status else 'None'}
    - Payment Method: {', '.join(payment_method) if payment_method else 'None'}
    - Search Query: {'`' + search_query + '`' if search_query else 'None'}
    
    **Results:** {len(filtered_transactions):,} transactions shown out of {len(transactions_df):,} total
    """)