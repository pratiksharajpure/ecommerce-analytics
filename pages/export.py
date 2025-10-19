"""
📥 Data Export Center - Export data in multiple formats with scheduling
Complete export functionality with history and scheduled exports
"""

import streamlit as st
import pandas as pd
import io
import json
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Data Export Center",
    page_icon="📥",
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
    .alert {
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        border-left: 4px solid;
        background: #dbeafe;
        color: #0f172a;
        border-left-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_export_data():
    """Generate sample export history and schedule data"""
    
    export_history = pd.DataFrame([
        {'id': 1, 'name': 'Customer_Report_2025', 'format': 'CSV', 'size': '2.3 MB', 'date': datetime.now() - timedelta(hours=2), 'status': 'completed'},
        {'id': 2, 'name': 'Sales_Analysis_Q3', 'format': 'Excel', 'size': '5.8 MB', 'date': datetime.now() - timedelta(hours=4), 'status': 'completed'},
        {'id': 3, 'name': 'Product_Inventory', 'format': 'CSV', 'size': '1.2 MB', 'date': datetime.now() - timedelta(hours=6), 'status': 'completed'},
        {'id': 4, 'name': 'Monthly_Dashboard', 'format': 'PDF', 'size': '12.4 MB', 'date': datetime.now() - timedelta(days=1, hours=6), 'status': 'completed'},
        {'id': 5, 'name': 'Order_Details_Full', 'format': 'Excel', 'size': '8.9 MB', 'date': datetime.now() - timedelta(days=1, hours=9), 'status': 'completed'},
        {'id': 6, 'name': 'Analytics_Summary', 'format': 'PDF', 'size': '3.1 MB', 'date': datetime.now() - timedelta(days=1, hours=13), 'status': 'failed'},
    ])
    
    schedules = pd.DataFrame([
        {'id': 1, 'name': 'Daily Sales Report', 'format': 'CSV', 'frequency': 'Daily', 'time': '09:00 AM', 'enabled': True, 'lastRun': datetime.now().date()},
        {'id': 2, 'name': 'Weekly Inventory', 'format': 'Excel', 'frequency': 'Weekly', 'time': 'Monday 06:00 AM', 'enabled': True, 'lastRun': datetime.now().date() - timedelta(days=4)},
        {'id': 3, 'name': 'Monthly Analytics', 'format': 'PDF', 'frequency': 'Monthly', 'time': '1st at 08:00 AM', 'enabled': True, 'lastRun': datetime.now().date() - timedelta(days=10)},
        {'id': 4, 'name': 'Customer Backup', 'format': 'CSV', 'frequency': 'Daily', 'time': '11:00 PM', 'enabled': False, 'lastRun': datetime.now().date() - timedelta(days=2)},
    ])
    
    return export_history, schedules

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading export data..."):
    export_history, schedules = generate_sample_export_data()

# Calculate metrics
total_exports = len(export_history)
active_schedules = len(schedules[schedules['enabled'] == True])
# Extract numeric values from size column and sum them
total_size = sum([float(''.join(c for c in size if c.isdigit() or c == '.')) for size in export_history['size']])
success_rate = (len(export_history[export_history['status'] == 'completed']) / len(export_history) * 100) if len(export_history) > 0 else 0

# ===========================
# STATE MANAGEMENT
# ===========================

if 'selected_format' not in st.session_state:
    st.session_state.selected_format = 'csv'
if 'selected_datasets' not in st.session_state:
    st.session_state.selected_datasets = []

# ===========================
# HEADER & METRICS
# ===========================

st.title("📥 Data Export Center")
st.markdown("**Export your data in various formats, manage export history, and schedule automated exports**")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.success("✅ History cleared")
with col2:
    if st.button("+ New Export", use_container_width=True):
        st.info("ℹ️ Configure your export settings below")

st.markdown("---")

st.markdown(f"""
<div class="alert">
    <strong>📊 Export Stats:</strong> {total_exports} total exports. Last export: 2 hours ago. {active_schedules} scheduled exports active.
</div>
""", unsafe_allow_html=True)

# Top Stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total Exports", f"{total_exports}", "This month")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Active Schedules", f"{active_schedules}", "Running")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Total Size", f"{total_size:.1f} MB", "All exports")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Success Rate", f"{success_rate:.1f}%", "Last 30 days")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3 = st.tabs([
    "📄 New Export",
    "📋 Export History",
    "⏱️ Scheduled Exports"
])

# ===========================
# TAB 1: NEW EXPORT
# ===========================

with tab1:
    st.subheader("Create New Export")
    
    st.markdown("#### Select Export Format")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 CSV", use_container_width=True, key="format_csv"):
            st.session_state.selected_format = 'csv'
        if st.session_state.selected_format == 'csv':
            st.success("✅ CSV Selected")
        st.caption("Comma-separated values • Best for spreadsheets")
    
    with col2:
        if st.button("📊 Excel", use_container_width=True, key="format_excel"):
            st.session_state.selected_format = 'excel'
        if st.session_state.selected_format == 'excel':
            st.success("✅ Excel Selected")
        st.caption("Microsoft Excel format • Multiple sheets supported")
    
    with col3:
        if st.button("📑 PDF", use_container_width=True, key="format_pdf"):
            st.session_state.selected_format = 'pdf'
        if st.session_state.selected_format == 'pdf':
            st.success("✅ PDF Selected")
        st.caption("Portable document format • Perfect for reports")
    
    st.markdown("---")
    st.markdown("#### Select Data to Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Datasets:**")
        
        datasets = ['Customers', 'Products', 'Orders', 'Inventory', 'Analytics', 'Reports']
        
        for dataset in datasets:
            if st.checkbox(dataset, value=dataset in st.session_state.selected_datasets, key=f"dataset_{dataset}"):
                if dataset not in st.session_state.selected_datasets:
                    st.session_state.selected_datasets.append(dataset)
            else:
                if dataset in st.session_state.selected_datasets:
                    st.session_state.selected_datasets.remove(dataset)
        
        st.caption(f"Selected: {len(st.session_state.selected_datasets)} dataset(s)")
    
    with col2:
        st.markdown("**Export Options:**")
        
        file_name = st.text_input("File Name", value=f"export_{datetime.now().strftime('%Y-%m-%d')}")
        
        date_range = st.selectbox(
            "Date Range",
            ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "This Year", "Custom Range"]
        )
        
        if date_range == "Custom Range":
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            with col_end:
                end_date = st.date_input("End Date", value=datetime.now())
        
        include_metadata = st.selectbox(
            "Include Metadata",
            ["Yes", "No"]
        )
        
        compression = st.selectbox(
            "Compression",
            ["None", "ZIP", "GZIP"]
        )
    
    st.markdown("---")
    
    if st.button("🚀 Start Export", use_container_width=True, key="start_export"):
        if len(st.session_state.selected_datasets) == 0:
            st.error("❌ Please select at least one dataset to export")
        else:
            with st.spinner(f"Exporting {len(st.session_state.selected_datasets)} dataset(s)..."):
                # Simulate export
                import time
                time.sleep(2)
                
                st.success(f"✅ Export completed successfully!")
                st.markdown(f"""
                **Export Details:**
                - **File Name:** {file_name}.{st.session_state.selected_format}
                - **Format:** {st.session_state.selected_format.upper()}
                - **Datasets:** {', '.join(st.session_state.selected_datasets)}
                - **Date Range:** {date_range}
                - **Size:** ~{np.random.uniform(1, 15):.1f} MB
                - **Compression:** {compression}
                """)
                
                # Create sample data
                if st.session_state.selected_format == 'csv':
                    data = pd.DataFrame({
                        'ID': range(100),
                        'Name': [f'Item_{i}' for i in range(100)],
                        'Value': np.random.randint(100, 10000, 100),
                        'Date': [datetime.now() - timedelta(days=np.random.randint(0, 90)) for _ in range(100)]
                    })
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{file_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                elif st.session_state.selected_format == 'excel':
                    data = pd.DataFrame({
                        'ID': range(100),
                        'Name': [f'Item_{i}' for i in range(100)],
                        'Value': np.random.randint(100, 10000, 100),
                        'Date': [datetime.now() - timedelta(days=np.random.randint(0, 90)) for _ in range(100)]
                    })
                    excel_buffer = io.BytesIO()
                    data.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"{file_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

# ===========================
# TAB 2: EXPORT HISTORY
# ===========================

with tab2:
    st.subheader("Export History")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Filter by Status:**")
        status_filter = st.selectbox("Status", ["All", "Completed", "Failed"])
    with col2:
        st.markdown("**Filter by Format:**")
        format_filter = st.selectbox("Format", ["All", "CSV", "Excel", "PDF"])
    
    # Apply filters
    filtered_history = export_history.copy()
    if status_filter != "All":
        filtered_history = filtered_history[filtered_history['status'] == status_filter.lower()]
    if format_filter != "All":
        filtered_history = filtered_history[filtered_history['format'] == format_filter]
    
    # Display table
    display_df = filtered_history.copy()
    display_df['date_str'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['Status'] = display_df['status'].apply(
        lambda x: "✅ Completed" if x == "completed" else "❌ Failed"
    )
    
    st.dataframe(
        display_df[['name', 'format', 'size', 'date_str', 'Status']],
        use_container_width=True,
        hide_index=True,
        column_config={
            'name': 'File Name',
            'format': 'Format',
            'size': 'Size',
            'date_str': 'Date',
            'Status': 'Status'
        }
    )
    
    st.markdown("---")
    st.markdown("#### Export Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**By Format**")
        format_counts = filtered_history['format'].value_counts()
        for fmt, count in format_counts.items():
            st.markdown(f"- {fmt}: **{count}**")
    
    with col2:
        st.markdown("**By Status**")
        status_counts = filtered_history['status'].value_counts()
        for status, count in status_counts.items():
            st.markdown(f"- {status.title()}: **{count}**")
    
    with col3:
        st.markdown("**Storage**")
        filtered_total_size = sum([float(''.join(c for c in size if c.isdigit() or c == '.')) for size in filtered_history['size']])
        avg_size = filtered_total_size / len(filtered_history) if len(filtered_history) > 0 else 0
        st.markdown(f"- Total Size: **{filtered_total_size:.1f} MB**")
        st.markdown(f"- Average Size: **{avg_size:.1f} MB**")

# ===========================
# TAB 3: SCHEDULED EXPORTS
# ===========================

with tab3:
    st.subheader("Scheduled Exports")
    
    st.markdown("#### Active Schedules")
    
    for idx, row in schedules.iterrows():
        with st.expander(f"📅 {row['name']}", expanded=row['enabled']):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Format**")
                st.markdown(f"``{row['format']}``")
            
            with col2:
                st.markdown("**Frequency**")
                st.markdown(f"``{row['frequency']}``")
            
            with col3:
                st.markdown("**Time**")
                st.markdown(f"``{row['time']}``")
            
            with col4:
                st.markdown("**Last Run**")
                st.markdown(f"``{row['lastRun']}``")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                enabled_new = st.checkbox(
                    "Enabled",
                    value=row['enabled'],
                    key=f"schedule_enabled_{row['id']}"
                )
                if enabled_new != row['enabled']:
                    st.success(f"✅ Schedule {'' if enabled_new else 'dis'}abled")
            
            with col2:
                if st.button("✏️ Edit", use_container_width=True, key=f"edit_{row['id']}"):
                    st.info(f"Edit dialog for {row['name']}")
            
            with col3:
                if st.button("▶️ Run Now", use_container_width=True, key=f"run_{row['id']}"):
                    st.success(f"✅ {row['name']} running...")
            
            with col4:
                if st.button("🗑️ Delete", use_container_width=True, key=f"delete_{row['id']}"):
                    st.error(f"Schedule deleted")
    
    st.markdown("---")
    st.markdown("#### Create New Schedule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        schedule_name = st.text_input("Schedule Name", value="My Export Schedule")
        schedule_frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
        schedule_time = st.time_input("Time", value=datetime.strptime("09:00", "%H:%M").time())
    
    with col2:
        schedule_format = st.selectbox("Format", ["CSV", "Excel", "PDF"])
        schedule_datasets = st.multiselect(
            "Datasets",
            ["Customers", "Products", "Orders", "Inventory", "Analytics", "Reports"],
            default=["Customers"]
        )
        schedule_email = st.text_input("Email Recipients (comma separated)")
    
    if st.button("✅ Create Schedule", use_container_width=True):
        if schedule_name and schedule_datasets:
            st.success(f"✅ Schedule '{schedule_name}' created successfully!")
            st.markdown(f"""
            **Schedule Details:**
            - **Name:** {schedule_name}
            - **Frequency:** {schedule_frequency}
            - **Time:** {schedule_time}
            - **Format:** {schedule_format}
            - **Datasets:** {', '.join(schedule_datasets)}
            - **Email:** {schedule_email if schedule_email else 'No email recipients'}
            """)
        else:
            st.error("Please fill in all required fields")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
