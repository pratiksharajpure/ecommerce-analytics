"""
🔒 Enhanced Fraud Detection Analysis - All 4 Tabs with Working Filters
Exact match to fraud.html with risk scoring and pattern detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Fraud Detection Analysis",
    page_icon="🔒",
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
    .alert-danger {
        background: #fee2e2;
        color: #991b1b;
        border-left-color: #ef4444;
    }
    .fraud-card {
        border-left: 4px solid;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 12px;
        background: white;
        border: 1px solid #e2e8f0;
    }
    .fraud-card-critical {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    .fraud-card-high {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    .fraud-card-medium {
        border-left-color: #3b82f6;
        background: #eff6ff;
    }
    .pattern-card {
        padding: 15px;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 15px;
        cursor: pointer;
        transition: all 0.3s;
        background: white;
    }
    .pattern-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_fraud_data():
    """Generate sample fraud detection data"""
    np.random.seed(42)
    
    # Suspicious Transactions
    suspicious = []
    statuses = ['blocked', 'flagged', 'reviewing', 'cleared']
    card_types = ['Visa', 'Mastercard', 'Amex', 'Discover']
    countries = ['Nigeria', 'Romania', 'China', 'Russia', 'India', 'Brazil', 'Ukraine', 'USA', 'UK', 'Unknown']
    
    for i in range(1, 101):
        # Risk score determines status
        risk_score = np.random.randint(50, 100)
        
        if risk_score >= 90:
            status = np.random.choice(['blocked', 'reviewing'], p=[0.8, 0.2])
            risk_level = 'critical'
        elif risk_score >= 70:
            status = np.random.choice(['flagged', 'reviewing', 'blocked'], p=[0.5, 0.3, 0.2])
            risk_level = 'high'
        else:
            status = np.random.choice(['flagged', 'reviewing', 'cleared'], p=[0.4, 0.3, 0.3])
            risk_level = 'medium'
        
        amount = round(np.random.uniform(100, 5000), 2)
        
        # Generate indicators based on risk level
        all_indicators = [
            'Stolen Card', 'High Value', 'New Customer', 'Shipping Rush',
            'Velocity Check', 'Multiple Orders', 'Different Cards',
            'Suspicious IP', 'Proxy Detected', 'Mismatched Billing',
            'High Risk Country', 'First Purchase', 'Card Testing',
            'Multiple Attempts', 'Bot Detected', 'Invalid Email'
        ]
        
        num_indicators = 4 if risk_level == 'critical' else 3 if risk_level == 'high' else 2
        indicators = list(np.random.choice(all_indicators, size=num_indicators, replace=False))
        
        date_offset = int(np.random.randint(0, 30))
        order_date = (datetime.now() - timedelta(days=date_offset, 
                                                 hours=int(np.random.randint(0, 24)),
                                                 minutes=int(np.random.randint(0, 60))))
        
        ip_address = f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
        
        suspicious.append({
            'order_id': f'ORD-{12000+i}',
            'customer': f'Customer {np.random.randint(1, 500)}',
            'amount': amount,
            'date': order_date,
            'risk_score': risk_score,
            'status': status,
            'indicators': indicators,
            'ip_address': ip_address,
            'location': np.random.choice(countries, p=[0.15, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.10, 0.10, 0.05]),
            'card_type': f"{np.random.choice(card_types)} ending {np.random.randint(1000, 9999)}",
            'risk_level': risk_level
        })
    
    # Fraud Patterns
    patterns = [
        {
            'name': 'Card Testing Pattern',
            'description': 'Multiple small transactions followed by large purchase attempt',
            'occurrences': 23,
            'blocked': 21,
            'severity': 'critical',
            'indicators': ['Multiple small charges', 'Rapid succession', 'Different cards', 'Same IP'],
            'recent_activity': '2 hours ago'
        },
        {
            'name': 'Velocity Abuse',
            'description': 'Unusual number of orders in short time period',
            'occurrences': 18,
            'blocked': 16,
            'severity': 'high',
            'indicators': ['Multiple orders', 'Same customer', 'Different shipping', 'Time clustering'],
            'recent_activity': '4 hours ago'
        },
        {
            'name': 'Account Takeover',
            'description': 'Legitimate account used by unauthorized party',
            'occurrences': 12,
            'blocked': 11,
            'severity': 'critical',
            'indicators': ['Login from new location', 'Password change', 'Shipping address change', 'High value order'],
            'recent_activity': '6 hours ago'
        },
        {
            'name': 'Triangulation Fraud',
            'description': 'Stolen card used to purchase from legitimate retailer',
            'occurrences': 9,
            'blocked': 8,
            'severity': 'high',
            'indicators': ['Gift cards purchased', 'Digital goods', 'Expedited shipping', 'Mismatched details'],
            'recent_activity': '12 hours ago'
        },
        {
            'name': 'Friendly Fraud',
            'description': 'Legitimate purchase followed by chargeback claim',
            'occurrences': 15,
            'blocked': 7,
            'severity': 'medium',
            'indicators': ['Previous chargebacks', 'High refund rate', 'Item not received claims'],
            'recent_activity': '1 day ago'
        }
    ]
    
    # Risk Scoring Factors
    scoring_factors = [
        {'factor': 'Payment Method', 'weight': 25, 'description': 'Card type, BIN validation, CVV match', 'impact': 'High'},
        {'factor': 'Customer History', 'weight': 20, 'description': 'Order history, account age, previous flags', 'impact': 'High'},
        {'factor': 'Transaction Velocity', 'weight': 18, 'description': 'Number of orders in timeframe', 'impact': 'High'},
        {'factor': 'Geolocation', 'weight': 15, 'description': 'IP location vs billing/shipping mismatch', 'impact': 'Medium'},
        {'factor': 'Order Value', 'weight': 12, 'description': 'Unusually high or low order amounts', 'impact': 'Medium'},
        {'factor': 'Device Fingerprint', 'weight': 10, 'description': 'Device consistency, browser data', 'impact': 'Low'}
    ]
    
    # Risk Thresholds
    thresholds = [
        {'range': '0-49', 'level': 'Low Risk', 'action': 'Approve automatically', 'color': '#22c55e'},
        {'range': '50-69', 'level': 'Medium Risk', 'action': 'Manual review recommended', 'color': '#3b82f6'},
        {'range': '70-89', 'level': 'High Risk', 'action': 'Additional verification required', 'color': '#f59e0b'},
        {'range': '90-100', 'level': 'Critical Risk', 'action': 'Block transaction immediately', 'color': '#ef4444'}
    ]
    
    # Scoring Examples
    examples = [
        {'scenario': 'New customer, first order, normal value', 'score': 35, 'level': 'low'},
        {'scenario': 'Returning customer, unusual location', 'score': 58, 'level': 'medium'},
        {'scenario': 'Multiple failed card attempts', 'score': 82, 'level': 'high'},
        {'scenario': 'Stolen card detected, high value', 'score': 98, 'level': 'critical'}
    ]
    
    return (pd.DataFrame(suspicious), patterns, scoring_factors, thresholds, examples)

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading fraud detection data..."):
    suspicious_df, patterns_data, scoring_factors, thresholds, examples = generate_sample_fraud_data()

# Calculate metrics
flagged_orders = len(suspicious_df[suspicious_df['status'].isin(['flagged', 'blocked', 'reviewing'])])
blocked_transactions = len(suspicious_df[suspicious_df['status'] == 'blocked'])
amount_saved = suspicious_df[suspicious_df['status'] == 'blocked']['amount'].sum()
false_positives = 12

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 🔒 Filters")
    
    date_range = st.selectbox(
        "📅 Date Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
        index=2
    )
    
    now = datetime.now()
    if date_range == "Last 24 Hours":
        cutoff_date = now - timedelta(days=1)
    elif date_range == "Last 7 Days":
        cutoff_date = now - timedelta(days=7)
    elif date_range == "Last 30 Days":
        cutoff_date = now - timedelta(days=30)
    else:
        cutoff_date = None
    
    risk_filter = st.selectbox(
        "⚠️ Risk Level",
        ["All Risk Levels", "Critical (90-100)", "High (70-89)", "Medium (50-69)", "Low (0-49)"],
        index=0
    )
    
    status_filter = st.multiselect(
        "📊 Status",
        ["flagged", "blocked", "reviewing", "cleared"],
        default=["flagged", "blocked", "reviewing", "cleared"]
    )
    
    search_query = st.text_input("🔍 Search", placeholder="Order ID, customer...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_fraud_filters(df, date_cutoff, risk_value, status_list, search_text):
    filtered = df.copy()
    
    if date_cutoff:
        filtered = filtered[filtered['date'] >= date_cutoff]
    
    # Apply risk filter
    if risk_value == "Critical (90-100)":
        filtered = filtered[filtered['risk_score'] >= 90]
    elif risk_value == "High (70-89)":
        filtered = filtered[(filtered['risk_score'] >= 70) & (filtered['risk_score'] < 90)]
    elif risk_value == "Medium (50-69)":
        filtered = filtered[(filtered['risk_score'] >= 50) & (filtered['risk_score'] < 70)]
    elif risk_value == "Low (0-49)":
        filtered = filtered[filtered['risk_score'] < 50]
    
    if status_list:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['order_id'].str.lower().str.contains(search_lower, na=False) |
            filtered['customer'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_suspicious = apply_fraud_filters(suspicious_df, cutoff_date, risk_filter, 
                                         status_filter, search_query)

# ===========================
# HEADER & METRICS
# ===========================

st.title("🔒 Fraud Detection Analysis")
st.markdown("**Suspicious activity and pattern detection**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-danger">', unsafe_allow_html=True)
    st.metric("Flagged Orders", f"{flagged_orders}", "+8 new flags")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Blocked Transactions", f"{blocked_transactions}", "+5 today")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Amount Saved", f"${amount_saved:,.0f}", "+$5,600 prevented")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("False Positives", f"{false_positives}", "-3 improvement")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown(f"""
<div class="alert alert-danger">
    <strong>🚨 Security Alert:</strong> {flagged_orders} flagged orders detected. {blocked_transactions} blocked transactions. ${amount_saved:,.0f} in potential fraud prevented. {false_positives} false positives identified.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fraud Detection Trend")
    
    trend_data = pd.DataFrame({
        'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        'Fraud Attempts': [156, 178, 189, 148],
        'Blocked': [145, 167, 178, 144]
    })
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=trend_data['Week'], y=trend_data['Fraud Attempts'],
                              mode='lines+markers', name='Fraud Attempts',
                              line=dict(color='#ef4444', width=3),
                              fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'))
    fig1.add_trace(go.Scatter(x=trend_data['Week'], y=trend_data['Blocked'],
                              mode='lines+markers', name='Blocked',
                              line=dict(color='#22c55e', width=3),
                              fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.1)'))
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Risk Level Distribution")
    
    risk_dist = pd.DataFrame({
        'Risk Level': ['Critical (90-100)', 'High (70-89)', 'Medium (50-69)', 'Low (0-49)'],
        'Count': [89, 145, 198, 239]
    })
    
    fig2 = px.pie(risk_dist, values='Count', names='Risk Level',
                  color_discrete_sequence=['#ef4444', '#f59e0b', '#3b82f6', '#22c55e'])
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Fraud Dashboard",
    f"⚠️ Suspicious Transactions ({len(filtered_suspicious)})",
    "🔍 Fraud Patterns",
    "📈 Risk Scoring"
])

# TAB 1: FRAUD DASHBOARD
with tab1:
    st.subheader("Fraud Detection Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; background: white;">
            <div style="font-size: 0.875rem; color: #64748b; font-weight: 600; margin-bottom: 10px;">Total Fraud Attempts</div>
            <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">671</div>
            <div style="font-size: 0.875rem; color: #64748b;">Last 30 days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; background: white;">
            <div style="font-size: 0.875rem; color: #64748b; font-weight: 600; margin-bottom: 10px;">Successfully Blocked</div>
            <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">634</div>
            <div style="font-size: 0.875rem; color: #64748b;">94.5% success rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px; background: white;">
            <div style="font-size: 0.875rem; color: #64748b; font-weight: 600; margin-bottom: 10px;">Revenue Protected</div>
            <div style="font-size: 2rem; font-weight: 800; margin-bottom: 8px;">$23,450</div>
            <div style="font-size: 0.875rem; color: #64748b;">Potential losses prevented</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚫 Recent Blocked Transactions")
    
    recent_blocks = filtered_suspicious[filtered_suspicious['status'] == 'blocked'].nlargest(5, 'date')
    
    if len(recent_blocks) > 0:
        for idx, block in recent_blocks.iterrows():
            time_diff = datetime.now() - block['date']
            if time_diff.total_seconds() < 3600:
                time_ago = f"{int(time_diff.total_seconds() / 60)} min ago"
            elif time_diff.total_seconds() < 86400:
                time_ago = f"{int(time_diff.total_seconds() / 3600)} hours ago"
            else:
                time_ago = f"{time_diff.days} days ago"
            
            risk_color = '#ef4444' if block['risk_score'] >= 90 else '#f59e0b' if block['risk_score'] >= 70 else '#3b82f6'
            
            col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 2])
            with col1:
                st.markdown(f"**{time_ago}**")
            with col2:
                st.markdown(f"**{block['order_id']}**")
            with col3:
                st.markdown(f"${block['amount']:,.2f}")
            with col4:
                st.markdown(f"<span style='color: {risk_color}; font-weight: bold;'>{block['risk_score']}</span>", unsafe_allow_html=True)
            with col5:
                st.markdown(f"{block['indicators'][0] if len(block['indicators']) > 0 else 'Fraud detected'}")
    else:
        st.info("No recent blocked transactions")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Top Fraud Indicators")
        indicators_data = {
            'Stolen Card Data': 89,
            'Velocity Violations': 67,
            'Suspicious IP/Location': 54,
            'Failed Verification': 45,
            'Device Fingerprint Mismatch': 25
        }
        
        for name, count in indicators_data.items():
            percent = int((count / 280) * 100)
            st.markdown(f"**{name}**: {count}")
            st.progress(percent / 100)
            st.caption(f"{percent}% of total flags")
    
    with col2:
        st.markdown("### 🎯 Detection Algorithm Performance")
        st.markdown("**Machine Learning Model**: <span style='color: #22c55e; font-weight: bold;'>96.5%</span>", unsafe_allow_html=True)
        st.markdown("**Rule-Based Engine**: <span style='color: #22c55e; font-weight: bold;'>92.3%</span>", unsafe_allow_html=True)
        st.markdown("**Behavioral Analysis**: <span style='color: #22c55e; font-weight: bold;'>88.7%</span>", unsafe_allow_html=True)
        st.markdown("**Consortium Data**: <span style='color: #3b82f6; font-weight: bold;'>85.4%</span>", unsafe_allow_html=True)

# TAB 2: SUSPICIOUS TRANSACTIONS
with tab2:
    st.subheader("Suspicious Transactions")
    
    if len(filtered_suspicious) > 0:
        for idx, trans in filtered_suspicious.iterrows():
            risk_class = 'critical' if trans['risk_level'] == 'critical' else 'high' if trans['risk_level'] == 'high' else 'medium'
            risk_color = '#ef4444' if trans['risk_level'] == 'critical' else '#f59e0b' if trans['risk_level'] == 'high' else '#3b82f6'
            
            st.markdown(f"""
            <div class="fraud-card fraud-card-{risk_class}">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                    <div>
                        <div style="font-weight: 700; font-size: 1rem;">{trans['order_id']}</div>
                        <div style="font-size: 0.875rem; color: #64748b; margin-top: 3px;">{trans['customer']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.25rem; font-weight: 800; color: {risk_color};">{trans['risk_score']}</div>
                        <div style="font-size: 0.75rem; color: #64748b;">Risk Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.markdown(f"**Amount:** ${trans['amount']:,.2f}")
            with col2:
                st.markdown(f"**Date:** {trans['date'].strftime('%Y-%m-%d %H:%M')}")
            with col3:
                status_color = '#ef4444' if trans['status'] == 'blocked' else '#f59e0b' if trans['status'] == 'flagged' else '#22c55e'
                st.markdown(f"**Status:** <span style='background: {status_color}20; color: {status_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;'>{trans['status'].upper()}</span>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"**IP:** {trans['ip_address']}")
            with col5:
                st.markdown(f"**Location:** {trans['location']}")
            with col6:
                st.markdown(f"**Card:** {trans['card_type']}")
            
            st.markdown(f"**Fraud Indicators:** {', '.join(trans['indicators'])}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ Approve", key=f"approve_{trans['order_id']}"):
                    st.success(f"Transaction {trans['order_id']} approved")
            with col2:
                if st.button(f"🚫 Block", key=f"block_{trans['order_id']}"):
                    st.error(f"Transaction {trans['order_id']} blocked")
            with col3:
                if st.button(f"🔍 Investigate", key=f"investigate_{trans['order_id']}"):
                    st.info(f"Opening investigation for {trans['order_id']}")
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.caption(f"Showing {len(filtered_suspicious):,} suspicious transactions")
    else:
        st.info("No suspicious transactions match the current filters")

# TAB 3: FRAUD PATTERNS
with tab3:
    st.subheader("Fraud Pattern Detection")
    
    for pattern in patterns_data:
        success_rate = (pattern['blocked'] / pattern['occurrences'] * 100)
        severity_color = '#ef4444' if pattern['severity'] == 'critical' else '#f59e0b' if pattern['severity'] == 'high' else '#3b82f6'
        
        st.markdown(f"""
        <div class="pattern-card">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                <div style="flex: 1;">
                    <div style="font-weight: 700; font-size: 1rem; margin-bottom: 5px;">{pattern['name']}</div>
                    <span style="background: {severity_color}20; color: {severity_color}; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">{pattern['severity'].upper()}</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.5rem; font-weight: 800; color: #ef4444;">{pattern['occurrences']}</div>
                    <div style="font-size: 0.75rem; color: #64748b;">occurrences</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Description:** {pattern['description']}")
        st.markdown(f"**Common Indicators:** {', '.join(pattern['indicators'])}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Blocked:** {pattern['blocked']}/{pattern['occurrences']}")
        with col2:
            st.markdown(f"**Success Rate:** {success_rate:.1f}%")
        with col3:
            st.markdown(f"**Last Seen:** {pattern['recent_activity']}")
        
        if st.button(f"View Pattern Details", key=f"pattern_{pattern['name']}"):
            st.info(f"Viewing details for {pattern['name']}")
        
        st.markdown("---")
    
    st.markdown("### 📊 Pattern Detection Timeline")
    pattern_names = [p['name'] for p in patterns_data]
    pattern_counts = [p['occurrences'] for p in patterns_data]
    
    fig_pattern = px.bar(x=pattern_counts, y=pattern_names, orientation='h',
                         color=pattern_counts,
                         color_continuous_scale=['#3b82f6', '#f59e0b', '#ef4444'],
                         labels={'x': 'Occurrences', 'y': 'Pattern Type'})
    fig_pattern.update_layout(height=350, showlegend=False,
                             xaxis_title='Occurrences', yaxis_title='')
    st.plotly_chart(fig_pattern, use_container_width=True)

# TAB 4: RISK SCORING
with tab4:
    st.subheader("Risk Scoring System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Risk Scoring Factors")
        
        st.dataframe(
            pd.DataFrame(scoring_factors),
            use_container_width=True,
            hide_index=True,
            column_config={
                'factor': 'Factor',
                'weight': st.column_config.NumberColumn('Weight (%)', format='%d%%'),
                'description': 'Description',
                'impact': 'Impact Level'
            }
        )
    
    with col2:
        st.markdown("### 🎯 Risk Score Example")
        
        # Create risk meter visualization
        risk_score = 85
        risk_level_text = 'High Risk'
        risk_color = '#f59e0b'
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': risk_level_text},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 49], 'color': '#d1fae5'},
                    {'range': [50, 69], 'color': '#dbeafe'},
                    {'range': [70, 89], 'color': '#fef3c7'},
                    {'range': [90, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.caption("Real-time calculation based on transaction data")
    
    st.markdown("---")
    st.markdown("### 📏 Risk Level Thresholds")
    
    for threshold in thresholds:
        st.markdown(f"""
        <div style="margin-bottom: 20px; padding: 15px; border-left: 4px solid {threshold['color']}; 
             background: #f8fafc; border-radius: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>
                    <div style="font-weight: 700; font-size: 1rem;">{threshold['level']}</div>
                    <div style="font-size: 0.875rem; color: #64748b;">Score Range: {threshold['range']}</div>
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {threshold['color']};">
                    {threshold['range'].split('-')[0]}
                </div>
            </div>
            <div style="font-size: 0.875rem; color: #64748b;">
                <strong>Action:</strong> {threshold['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Example Scenarios")
    
    examples_df = pd.DataFrame(examples)
    
    for idx, ex in examples_df.iterrows():
        level_color = '#ef4444' if ex['level'] == 'critical' else '#f59e0b' if ex['level'] == 'high' else '#3b82f6' if ex['level'] == 'medium' else '#22c55e'
        
        col1, col2, col3 = st.columns([3, 1, 2])
        with col1:
            st.markdown(f"**{ex['scenario']}**")
        with col2:
            st.markdown(f"<span style='color: {level_color}; font-weight: bold; font-size: 1.25rem;'>{ex['score']}</span>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<span style='background: {level_color}20; color: {level_color}; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;'>{ex['level'].upper()}</span>", unsafe_allow_html=True)
        
        st.markdown("---")

st.markdown("---")

# ===========================
# DETECTION PERFORMANCE
# ===========================

st.markdown("### 📈 Detection Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="padding: 15px; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div style="font-weight: 600; color: #64748b; margin-bottom: 10px;">Detection Rate</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #22c55e; margin-bottom: 5px;">94.5%</div>
        <div style="font-size: 0.8125rem; color: #64748b;">634 of 671 fraud attempts caught</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(0.945)

with col2:
    st.markdown("""
    <div style="padding: 15px; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div style="font-weight: 600; color: #64748b; margin-bottom: 10px;">False Positive Rate</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #22c55e; margin-bottom: 5px;">2.1%</div>
        <div style="font-size: 0.8125rem; color: #64748b;">12 of 568 flagged incorrectly</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(0.021)

with col3:
    st.markdown("""
    <div style="padding: 15px; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div style="font-weight: 600; color: #64748b; margin-bottom: 10px;">Avg Response Time</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #3b82f6; margin-bottom: 5px;">1.8s</div>
        <div style="font-size: 0.8125rem; color: #64748b;">Real-time detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(0.85)

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Fraud data refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'module': 'fraud_detection',
            'suspicious_transactions': suspicious_df.to_dict('records'),
            'patterns': patterns_data,
            'summary': {
                'flagged_orders': flagged_orders,
                'blocked_transactions': blocked_transactions,
                'amount_saved': f"${amount_saved:,.2f}",
                'false_positives': false_positives,
                'detection_rate': '94.5%',
                'false_positive_rate': '2.1%'
            }
        }
        st.success("✅ Fraud data exported successfully")
        st.json(export_data)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================
# ADDITIONAL INSIGHTS
# ===========================

with st.expander("🔍 Fraud Detection Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 High Risk Transactions")
        
        high_risk = suspicious_df[suspicious_df['risk_score'] >= 90].nlargest(5, 'risk_score')
        if len(high_risk) > 0:
            st.markdown("**Top 5 Critical Risk Transactions:**")
            for idx, trans in high_risk.iterrows():
                st.markdown(f"- **{trans['order_id']}**: Risk Score {trans['risk_score']} - {trans['indicators'][0]}")
        else:
            st.success("✅ No critical risk transactions currently!")
        
        st.markdown("#### 💡 Best Practices")
        st.markdown("""
        - **Real-time monitoring** of all transactions
        - **Multi-layered detection** using ML + rules + behavioral analysis
        - **Manual review** for medium-high risk (50-89 score)
        - **Automatic blocking** for critical risk (90+ score)
        - **Regular model updates** to catch new fraud patterns
        """)
    
    with col2:
        st.markdown("#### 📊 Detection Statistics")
        
        st.markdown(f"""
        **Last 30 Days Performance:**
        - Total Fraud Attempts: **671**
        - Successfully Blocked: **634** (94.5%)
        - Missed Fraud: **37** (5.5%)
        - False Positives: **{false_positives}** (2.1%)
        - Revenue Protected: **${amount_saved:,.0f}**
        - Chargebacks Prevented: **89**
        """)
        
        st.markdown("#### 🎯 Recommended Actions")
        st.markdown(f"""
        1. **Immediate**: Review {len(filtered_suspicious[filtered_suspicious['status'] == 'reviewing'])} transactions in review queue
        2. **Short-term**: Investigate {len(filtered_suspicious[filtered_suspicious['status'] == 'flagged'])} flagged transactions
        3. **Medium-term**: Analyze false positives to improve model accuracy
        4. **Long-term**: Implement advanced device fingerprinting
        """)
    
    st.markdown("---")
    st.markdown("### 🌍 Geographic Risk Analysis")
    
    location_stats = suspicious_df.groupby('location').agg({
        'order_id': 'count',
        'risk_score': 'mean',
        'amount': 'sum'
    }).reset_index()
    location_stats.columns = ['Location', 'Transaction Count', 'Avg Risk Score', 'Total Amount']
    location_stats = location_stats.sort_values('Transaction Count', ascending=False).head(10)
    
    st.dataframe(
        location_stats.style.format({
            'Avg Risk Score': '{:.1f}',
            'Total Amount': '${:,.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("### 🔐 Security Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Immediate Actions:**
        - ✅ Enable 3D Secure authentication for high-risk regions
        - ✅ Implement velocity checks (max 3 orders/hour per customer)
        - ✅ Require CVV verification for all transactions
        - ✅ Block transactions from known proxy/VPN services
        """)
    
    with col2:
        st.markdown("""
        **Long-term Improvements:**
        - 🔹 Deploy machine learning model v2.0 (expected +3% accuracy)
        - 🔹 Integrate consortium fraud data feeds
        - 🔹 Implement behavioral biometrics
        - 🔹 Add email/phone verification for new accounts
        """)

# ===========================
# DIAGNOSTIC INFORMATION
# ===========================

with st.expander("🔧 System Diagnostics"):
    st.markdown("### Detection System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions Analyzed", f"{len(suspicious_df):,}")
        st.metric("Detection Rules Active", "127")
    
    with col2:
        st.metric("ML Model Accuracy", "96.5%")
        st.metric("Average Processing Time", "1.8s")
    
    with col3:
        st.metric("System Uptime", "99.9%")
        st.metric("Last Model Update", "2 days ago")
    
    st.markdown("---")
    st.markdown("### Filter Status")
    st.info(f"""
    **Active Filters:**
    - Date Range: {date_range}
    - Risk Level: {risk_filter}
    - Status: {', '.join(status_filter) if status_filter else 'None'}
    - Search Query: {'`' + search_query + '`' if search_query else 'None'}
    
    **Results:** {len(filtered_suspicious):,} transactions shown out of {len(suspicious_df):,} total
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Detection Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**By Risk Level:**")
        critical = len(suspicious_df[suspicious_df['risk_score'] >= 90])
        high = len(suspicious_df[(suspicious_df['risk_score'] >= 70) & (suspicious_df['risk_score'] < 90)])
        medium = len(suspicious_df[(suspicious_df['risk_score'] >= 50) & (suspicious_df['risk_score'] < 70)])
        low = len(suspicious_df[suspicious_df['risk_score'] < 50])
        
        st.markdown(f"- 🔴 Critical (90-100): **{critical}** transactions")
        st.markdown(f"- 🟡 High (70-89): **{high}** transactions")
        st.markdown(f"- 🔵 Medium (50-69): **{medium}** transactions")
        st.markdown(f"- 🟢 Low (0-49): **{low}** transactions")
    
    with col2:
        st.markdown("**By Status:**")
        blocked = len(suspicious_df[suspicious_df['status'] == 'blocked'])
        flagged = len(suspicious_df[suspicious_df['status'] == 'flagged'])
        reviewing = len(suspicious_df[suspicious_df['status'] == 'reviewing'])
        cleared = len(suspicious_df[suspicious_df['status'] == 'cleared'])
        
        st.markdown(f"- 🚫 Blocked: **{blocked}** transactions")
        st.markdown(f"- ⚠️ Flagged: **{flagged}** transactions")
        st.markdown(f"- 🔍 Reviewing: **{reviewing}** transactions")
        st.markdown(f"- ✅ Cleared: **{cleared}** transactions")

# ===========================
# FRAUD PATTERN INSIGHTS
# ===========================

with st.expander("🔍 Fraud Pattern Analysis"):
    st.markdown("### Pattern Detection Summary")
    
    patterns_summary = pd.DataFrame(patterns_data)
    patterns_summary['Success Rate'] = (patterns_summary['blocked'] / patterns_summary['occurrences'] * 100).round(1)
    
    fig_patterns = px.bar(patterns_summary, x='name', y='occurrences',
                         color='Success Rate',
                         color_continuous_scale='RdYlGn',
                         labels={'name': 'Pattern Type', 'occurrences': 'Occurrences'},
                         title='Fraud Patterns by Occurrence and Success Rate')
    fig_patterns.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_patterns, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Pattern Details")
    
    for pattern in patterns_data:
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(pattern['name'], f"{pattern['occurrences']}", "occurrences")
            with col2:
                success_rate = (pattern['blocked'] / pattern['occurrences'] * 100)
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col3:
                st.metric("Blocked", f"{pattern['blocked']}/{pattern['occurrences']}")
            with col4:
                severity_emoji = "🔴" if pattern['severity'] == 'critical' else "🟡" if pattern['severity'] == 'high' else "🔵"
                st.metric("Severity", f"{severity_emoji} {pattern['severity'].upper()}")
            
            st.markdown(f"**Last Activity:** {pattern['recent_activity']}")
            st.markdown("---")

# ===========================
# ALERTS & NOTIFICATIONS
# ===========================

if blocked_transactions > 20 or flagged_orders > 40:
    with st.expander("⚠️ Critical Alerts", expanded=True):
        st.warning("**High Volume of Suspicious Activity Detected!**")
        
        st.markdown(f"""
        - 🚨 **{blocked_transactions} blocked transactions** in the last 30 days
        - ⚠️ **{flagged_orders} flagged orders** requiring review
        - 💰 **${amount_saved:,.0f}** in potential losses prevented
        
        **Recommended Actions:**
        1. Review all blocked transactions for false positives
        2. Investigate flagged orders within 24 hours
        3. Update fraud detection rules based on new patterns
        4. Contact payment processor for additional verification
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📧 Notify Security Team", use_container_width=True):
                st.success("✅ Security team has been notified")
        with col2:
            if st.button("📊 Generate Detailed Report", use_container_width=True):
                st.success("✅ Detailed fraud report generated")
                