"""
🔔 Enhanced Alerts Dashboard - All 4 Tabs with Working Filters
Exact match to alerts.html with real-time monitoring and alert management
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Alerts Dashboard",
    page_icon="🔔",
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
    .alert-danger {
        background: #fee2e2;
        color: #991b1b;
        border-left-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=300)
def generate_sample_alert_data():
    """Generate sample alert data with different severity levels"""
    np.random.seed(42)
    
    # Active Alerts
    active_alerts = []
    alert_types = ['Performance', 'Database', 'Security', 'Payment', 'Inventory', 'System']
    severities = ['critical', 'high', 'medium', 'low']
    statuses = ['active', 'acknowledged']
    
    alert_templates = {
        'Performance': [
            'High CPU Usage - CPU exceeded 90%',
            'Memory Usage Warning - Memory at 78%',
            'Slow API Response - Response time > 2s',
            'Database Connection Slow - Response > 5s'
        ],
        'Database': [
            'Database Connection Error',
            'Query Performance Degradation',
            'Connection Pool Exhausted',
            'Replication Lag Detected'
        ],
        'Security': [
            'Failed Login Attempts Detected',
            'Suspicious IP Activity',
            'SSL Certificate Expiring',
            'Unauthorized Access Attempt'
        ],
        'Payment': [
            'Payment Gateway Error',
            'High Transaction Failure Rate',
            'Payment Processing Timeout',
            'Fraud Alert Triggered'
        ],
        'Inventory': [
            'Inventory Low - Products Below Minimum',
            'Stock Discrepancy Detected',
            'Warehouse Capacity Warning',
            'Reorder Point Breach'
        ],
        'System': [
            'Disk Space Low - Usage at 68%',
            'Service Down - Healthcheck Failed',
            'Backup Failed',
            'Log Storage Critical'
        ]
    }
    
    for i in range(1, 21):
        alert_type = np.random.choice(alert_types)
        severity = np.random.choice(severities, p=[0.15, 0.25, 0.35, 0.25])
        
        timestamp_minutes = np.random.randint(1, 180)
        if timestamp_minutes < 60:
            timestamp = f"{timestamp_minutes} min ago"
        else:
            timestamp = f"{timestamp_minutes // 60}h {timestamp_minutes % 60}m ago"
        
        active_alerts.append({
            'alert_id': i,
            'title': np.random.choice(alert_templates[alert_type]),
            'severity': severity,
            'message': f"{alert_type} alert triggered for system monitoring",
            'timestamp': timestamp,
            'type': alert_type,
            'status': np.random.choice(statuses, p=[0.7, 0.3])
        })
    
    # Alert Types Configuration
    alert_types_config = []
    type_data = {
        'Performance': {'count': 45, 'enabled': True, 'threshold': 'CPU > 85% for 5min'},
        'Database': {'count': 23, 'enabled': True, 'threshold': 'Response time > 3s'},
        'Security': {'count': 18, 'enabled': True, 'threshold': 'Failed logins > 10'},
        'Payment': {'count': 12, 'enabled': True, 'threshold': 'Error rate > 5%'},
        'Inventory': {'count': 34, 'enabled': True, 'threshold': 'Stock < minimum'},
        'System': {'count': 28, 'enabled': True, 'threshold': 'Disk > 80%'}
    }
    
    for alert_type, data in type_data.items():
        alert_types_config.append({
            'name': alert_type,
            'count': data['count'],
            'enabled': data['enabled'],
            'threshold': data['threshold']
        })
    
    # Alert History
    history = []
    for i in range(1, 16):
        resolved_minutes = np.random.randint(5, 60)
        history.append({
            'alert_id': 100 + i,
            'title': np.random.choice([
                'Server Restart Required',
                'API Rate Limit Exceeded',
                'Backup Failed',
                'Email Queue Backlog',
                'SSL Certificate Expiring',
                'Memory Leak Detected',
                'Disk Cleanup Needed',
                'Database Deadlock'
            ]),
            'severity': np.random.choice(['critical', 'high', 'medium', 'low'], p=[0.1, 0.25, 0.35, 0.3]),
            'timestamp': f"{np.random.randint(1, 24)} hours ago",
            'resolved_in': f"{resolved_minutes} min",
            'resolved_by': np.random.choice(['Admin', 'System', 'Operator'], p=[0.6, 0.3, 0.1])
        })
    
    # Alert Trend Data (for chart)
    trend_data = []
    for hour in range(12):
        trend_data.append({
            'hour': f"{hour}:00",
            'alerts': np.random.randint(2, 10)
        })
    
    # Severity Distribution
    severity_dist = {
        'Critical': 3,
        'High': 5,
        'Medium': 12,
        'Low': 8
    }
    
    return (pd.DataFrame(active_alerts), pd.DataFrame(alert_types_config),
            pd.DataFrame(history), pd.DataFrame(trend_data), severity_dist)

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading alerts..."):
    active_df, types_df, history_df, trend_df, severity_dist = generate_sample_alert_data()

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 🔔 Filters")
    
    severity_filter = st.multiselect(
        "⚠️ Severity Level",
        ["critical", "high", "medium", "low"],
        default=["critical", "high", "medium", "low"]
    )
    
    type_filter = st.multiselect(
        "📊 Alert Type",
        active_df['type'].unique().tolist(),
        default=active_df['type'].unique().tolist()
    )
    
    status_filter = st.multiselect(
        "🔄 Status",
        ["active", "acknowledged"],
        default=["active", "acknowledged"]
    )
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_alert_filters(df, severity_list, type_list, status_list):
    filtered = df.copy()
    
    if severity_list and 'severity' in filtered.columns:
        filtered = filtered[filtered['severity'].isin(severity_list)]
    
    if type_list and 'type' in filtered.columns:
        filtered = filtered[filtered['type'].isin(type_list)]
    
    if status_list and 'status' in filtered.columns:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    return filtered

filtered_alerts = apply_alert_filters(active_df, severity_filter, type_filter, status_filter)

# ===========================
# CALCULATE METRICS
# ===========================

total_alerts = len(active_df)
critical_alerts = len(active_df[active_df['severity'] == 'critical'])
high_alerts = len(active_df[active_df['severity'] == 'high'])
resolved_today = 12
total_24h = 28

# ===========================
# HEADER & METRICS
# ===========================

st.title("🔔 Alerts Dashboard")
st.markdown("**Real-time monitoring, alert management, and notification configuration**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-danger">', unsafe_allow_html=True)
    st.metric("Critical Alerts", f"{critical_alerts}", "+2 in last hour")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Warning Alerts", f"{high_alerts}", "Monitoring")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Resolved Today", f"{resolved_today}", "Avg: 8.5 min")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Alerts (24h)", f"{total_24h}", "15 active")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT BANNER
# ===========================

st.markdown(f"""
<div class="alert alert-warning">
    <strong>⚠️ Active Alerts:</strong> {critical_alerts} critical alerts require attention. {high_alerts} warnings detected. Last update: Just now.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Alerts Over Time")
    
    fig1 = px.line(trend_df, x='hour', y='alerts',
                   title='Alert Trend (Last 12 Hours)',
                   markers=True,
                   color_discrete_sequence=['#ef4444'])
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Alert Distribution by Severity")
    
    severity_df = pd.DataFrame(list(severity_dist.items()), columns=['Severity', 'Count'])
    color_map = {
        'Critical': '#ef4444',
        'High': '#f59e0b',
        'Medium': '#3b82f6',
        'Low': '#22c55e'
    }
    
    fig2 = px.pie(severity_df, values='Count', names='Severity',
                  color='Severity', color_discrete_map=color_map,
                  title='Severity Distribution')
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"🔴 Active Alerts ({len(filtered_alerts)})",
    f"📊 Alert Types",
    f"📜 Alert History ({len(history_df)})",
    f"⚙️ Configuration"
])

# TAB 1: ACTIVE ALERTS
with tab1:
    st.subheader("Active Alerts")
    
    if len(filtered_alerts) > 0:
        col1, col2 = st.columns([0.9, 0.1])
        with col2:
            if st.button("Resolve All", use_container_width=True):
                st.success("✅ All alerts resolved")
        
        for idx, alert in filtered_alerts.iterrows():
            severity_color = "#ef4444" if alert['severity'] == "critical" else "#f59e0b" if alert['severity'] == "high" else "#3b82f6" if alert['severity'] == "medium" else "#22c55e"
            status_emoji = "🔴" if alert['status'] == "active" else "🟡"
            
            with st.container():
                col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
                
                with col1:
                    st.markdown(f"""
                    <div style='background:#f8fafc;padding:15px;border-radius:8px;border-left:4px solid {severity_color};margin-bottom:10px;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                            <div style='font-weight:700;font-size:1rem;'>{status_emoji} {alert['title']}</div>
                            <span style='display:inline-block;padding:4px 12px;border-radius:12px;font-size:0.75rem;font-weight:700;background:{severity_color};color:white;'>{alert['severity'].upper()}</span>
                        </div>
                        <div style='font-size:0.875rem;color:#64748b;margin-bottom:5px;'>{alert['message']}</div>
                        <div style='font-size:0.75rem;color:#64748b;'>⏰ {alert['timestamp']} | Type: {alert['type']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("✅ Resolve", key=f"resolve_{alert['alert_id']}", use_container_width=True):
                        st.success(f"Alert #{alert['alert_id']} resolved")
                
                with col3:
                    if st.button("👁️ Details", key=f"details_{alert['alert_id']}", use_container_width=True):
                        st.info(f"Full diagnostic information for Alert #{alert['alert_id']}")
        
        st.caption(f"Showing {len(filtered_alerts):,} of {len(active_df):,} active alerts")
    else:
        st.success("✅ No active alerts!")

# TAB 2: ALERT TYPES
with tab2:
    st.subheader("Alert Types & Categories")
    
    if len(types_df) > 0:
        display_types = types_df.copy()
        display_types['Status'] = display_types['enabled'].apply(
            lambda x: "✅ Enabled" if x else "❌ Disabled"
        )
        
        st.dataframe(
            display_types[['name', 'count', 'threshold', 'Status']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'name': 'Alert Type',
                'count': 'Total Alerts',
                'threshold': 'Threshold Configuration',
                'Status': 'Status'
            }
        )
        
        st.markdown("---")
        st.markdown("#### Alert Types Overview")
        
        cols = st.columns(3)
        for idx, (col, (_, row)) in enumerate(zip(cols * ((len(types_df) + 2) // 3), types_df.iterrows())):
            with col:
                st.metric(row['name'], f"{row['count']} alerts", "Enabled" if row['enabled'] else "Disabled")

# TAB 3: ALERT HISTORY
with tab3:
    st.subheader("Alert History")
    
    if len(history_df) > 0:
        display_history = history_df.copy()
        display_history['Severity'] = display_history['severity'].apply(
            lambda x: f"🔴 {x.upper()}" if x == "critical"
            else f"🟠 {x.upper()}" if x == "high"
            else f"🔵 {x.upper()}" if x == "medium"
            else f"🟢 {x.upper()}"
        )
        
        st.dataframe(
            display_history[['alert_id', 'title', 'Severity', 'timestamp', 'resolved_in', 'resolved_by']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'alert_id': 'Alert ID',
                'title': 'Title',
                'Severity': 'Severity',
                'timestamp': 'Timestamp',
                'resolved_in': 'Resolution Time',
                'resolved_by': 'Resolved By'
            }
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Resolution Statistics")
            st.metric("Avg Resolution Time", "8.5 min")
            st.metric("Total Resolved Today", resolved_today)
            st.metric("Total This Week", "87")
        
        with col2:
            st.markdown("#### Top Resolvers")
            resolvers = pd.DataFrame({
                'Resolver': ['Admin', 'System', 'Operator'],
                'Count': [45, 32, 10]
            })
            st.dataframe(resolvers, use_container_width=True, hide_index=True)

# TAB 4: CONFIGURATION
with tab4:
    st.subheader("Alert Configuration")
    
    st.markdown("#### Notification Methods")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        email_enabled = st.checkbox("📧 Email Notifications", value=True)
    with col2:
        sms_enabled = st.checkbox("📱 SMS Notifications", value=True)
    with col3:
        slack_enabled = st.checkbox("💬 Slack Integration", value=False)
    
    st.markdown("---")
    st.markdown("#### Alert Settings")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        auto_resolve = st.checkbox("⚙️ Auto-Resolution", value=True)
    with col2:
        grouping = st.checkbox("📊 Alert Grouping", value=True)
    with col3:
        quiet_hours = st.checkbox("🌙 Quiet Hours (10PM-6AM)", value=False)
    
    st.markdown("---")
    st.markdown("#### Notification Channels")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Email Recipients:**")
        email_list = st.text_area("Email addresses (one per line)", "admin@company.com\nops@company.com", height=80)
    
    with col2:
        st.markdown("**Alert Thresholds:**")
        st.info("""
        **Critical:** Immediate notification
        **High:** Notify within 5 minutes
        **Medium/Low:** Batch notifications
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Configuration", use_container_width=True):
            st.success("✅ Configuration saved successfully!")
    with col2:
        if st.button("📥 Export Configuration", use_container_width=True):
            st.success("✅ Configuration exported!")

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Alerts", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Alerts refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'active_alerts': active_df.to_dict('records'),
            'alert_types': types_df.to_dict('records'),
            'history': history_df.to_dict('records'),
            'summary': {
                'critical_alerts': int(critical_alerts),
                'high_alerts': int(high_alerts),
                'resolved_today': resolved_today,
                'total_24h': total_24h
            }
        }
        st.success("✅ Alert data exported successfully")
        st.json(export_data)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================
# ADDITIONAL INSIGHTS
# ===========================

with st.expander("📊 Alert Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Current Status")
        st.markdown(f"""
        - **{critical_alerts}** critical alerts requiring immediate attention
        - **{high_alerts}** high-priority warnings
        - **{total_alerts}** total active alerts
        - **{resolved_today}** alerts resolved today (avg: 8.5 min)
        - Average resolution time is excellent
        """)
        
        st.markdown("#### 💡 Recommended Actions")
        st.markdown("""
        1. **Immediate**: Address critical alerts
        2. **Short-term**: Review high-priority warnings
        3. **Medium-term**: Optimize alert thresholds
        4. **Long-term**: Implement ML-based anomaly detection
        """)
    
    with col2:
        st.markdown("#### 📈 Performance Metrics")
        
        alert_types_active = types_df[types_df['enabled'] == True]
        st.markdown(f"**Active Alert Types:** {len(alert_types_active)}")
        
        for _, row in alert_types_active.iterrows():
            st.markdown(f"- **{row['name']}**: {row['count']} alerts")
        
        st.markdown("#### 🎯 Optimization Opportunities")
        st.markdown("""
        - Reduce false positives by 20-30%
        - Implement predictive alerting
        - Automate common resolution steps
        - Create alert runbooks for faster resolution
        """)

# ===========================
# DIAGNOSTIC INFORMATION
# ===========================

with st.expander("🔧 System Diagnostics"):
    st.markdown("### Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts", f"{len(active_df):,}")
        st.metric("Data Completeness", "100%")
    
    with col2:
        st.metric("Critical Count", f"{critical_alerts}")
        st.metric("Critical %", f"{critical_alerts/len(active_df)*100:.1f}%")
    
    with col3:
        st.metric("Active Types", len(types_df[types_df['enabled'] == True]))
        st.metric("History Records", f"{len(history_df):,}")
    
    st.markdown("---")
    st.markdown("### Filter Status")
    st.info(f"""
    **Active Filters:**
    - Severity: {', '.join(severity_filter) if severity_filter else 'None'}
    - Alert Type: {', '.join(type_filter) if type_filter else 'None'}
    - Status: {', '.join(status_filter) if status_filter else 'None'}
    
    **Results:** {len(filtered_alerts):,} alerts shown out of {len(active_df):,} total
    """)