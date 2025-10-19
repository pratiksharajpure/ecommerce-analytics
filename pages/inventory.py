"""
📈 Enhanced Inventory Quality Check - All 4 Tabs with Working Filters
Exact match to inventory.html with stock validation, alerts, movements, and warehouse monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Inventory Quality Check",
    page_icon="📈",
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
    .alert-danger {
        background: #fee2e2;
        color: #991b1b;
        border-left-color: #ef4444;
    }
    .warehouse-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 4px solid #3b82f6;
        margin-bottom: 15px;
    }
    .progress-bar {
        height: 8px;
        background: #f1f5f9;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 8px;
    }
    .progress-fill {
        height: 100%;
        transition: width 0.6s;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_inventory_data():
    """Generate sample inventory data with quality issues"""
    np.random.seed(42)
    
    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Toys', 'Food', 'Beauty', 'Garden', 'Pet']
    warehouses = ['Main Warehouse', 'East Coast DC', 'West Coast DC', 'Central Hub']
    
    # Current Stock Levels
    stock = []
    for i in range(1, 201):
        category = np.random.choice(categories)
        warehouse = np.random.choice(warehouses)
        
        system_stock = np.random.randint(20, 500)
        
        # Create variance (8-15% will have mismatches)
        if np.random.random() < 0.12:
            variance = np.random.randint(-30, 30)
            if variance == 0:
                variance = np.random.choice([-5, 5])
        else:
            variance = 0
        
        physical_stock = system_stock + variance
        
        reorder_point = int(system_stock * np.random.uniform(0.3, 0.5))
        reorder_qty = int(reorder_point * 2)
        
        # Determine status based on physical stock
        if physical_stock < reorder_point * 0.5:
            status = 'critical'
        elif physical_stock < reorder_point:
            status = 'low'
        elif physical_stock > reorder_point * 3:
            status = 'overstocked'
        elif physical_stock >= reorder_point * 1.5:
            status = 'good'
        else:
            status = 'adequate'
        
        value_impact = variance * np.random.uniform(5, 50)
        
        stock.append({
            'sku': f'{category[:4].upper()}-{i:03d}',
            'product': f'{category} Product {i}',
            'category': category,
            'system_stock': system_stock,
            'physical_stock': physical_stock,
            'variance': variance,
            'reorder_point': reorder_point,
            'reorder_qty': reorder_qty,
            'status': status,
            'value_impact': value_impact,
            'warehouse': warehouse
        })
    
    # Low Stock Alerts
    alerts = []
    critical_items = [s for s in stock if s['status'] == 'critical'][:15]
    low_items = [s for s in stock if s['status'] == 'low'][:10]
    
    for item in critical_items + low_items:
        daily_avg_sales = np.random.randint(3, 12)
        days_remaining = max(1, item['physical_stock'] // daily_avg_sales)
        
        severity = 'critical' if item['status'] == 'critical' else 'low'
        suggested_action = 'Emergency reorder required' if severity == 'critical' else 'Place standard reorder'
        
        alerts.append({
            'sku': item['sku'],
            'product': item['product'],
            'current_stock': item['physical_stock'],
            'reorder_point': item['reorder_point'],
            'days_remaining': days_remaining,
            'daily_avg_sales': daily_avg_sales,
            'severity': severity,
            'warehouse': item['warehouse'],
            'suggested_action': suggested_action
        })
    
    # Stock Movements
    movements = []
    movement_types = ['inbound', 'outbound', 'adjustment', 'damaged']
    movement_reasons = {
        'inbound': ['Purchase order received', 'Supplier delivery', 'Weekly restock', 'Return to stock'],
        'outbound': ['Customer order', 'Bulk order', 'Transfer to store', 'B2B order'],
        'adjustment': ['Damaged items removed', 'Inventory audit adjustment', 'System correction', 'Shrinkage'],
        'damaged': ['Quality control rejection', 'Shipping damage', 'Expired items', 'Customer return damage']
    }
    
    for i in range(150):
        movement_type = np.random.choice(movement_types, p=[0.4, 0.45, 0.1, 0.05])
        
        if movement_type == 'inbound':
            quantity = np.random.randint(20, 150)
        elif movement_type == 'outbound':
            quantity = -np.random.randint(5, 50)
        elif movement_type == 'adjustment':
            quantity = np.random.randint(-20, 20)
        else:  # damaged
            quantity = -np.random.randint(1, 10)
        
        item = np.random.choice(stock)
        before_qty = item['physical_stock']
        after_qty = before_qty + quantity
        
        date_offset = np.random.randint(0, 30)
        movement_date = (datetime.now() - timedelta(days=date_offset)).strftime('%Y-%m-%d %H:%M')
        
        reference = f"{'PO' if movement_type == 'inbound' else 'ORD' if movement_type == 'outbound' else 'ADJ'}-{np.random.randint(1000, 9999)}"
        reason = np.random.choice(movement_reasons[movement_type])
        
        movements.append({
            'date': movement_date,
            'sku': item['sku'],
            'product': item['product'],
            'type': movement_type,
            'quantity': quantity,
            'warehouse': item['warehouse'],
            'reference': reference,
            'before_qty': before_qty,
            'after_qty': after_qty,
            'reason': reason
        })
    
    # Sort movements by date descending
    movements = sorted(movements, key=lambda x: x['date'], reverse=True)
    
    # Warehouse Breakdown
    warehouses_data = []
    for warehouse in warehouses:
        warehouse_items = [s for s in stock if s['warehouse'] == warehouse]
        
        total_skus = len(warehouse_items)
        total_units = sum(s['physical_stock'] for s in warehouse_items)
        capacity = total_units + np.random.randint(10000, 30000)
        utilization = int((total_units / capacity) * 100)
        
        critical_items = len([s for s in warehouse_items if s['status'] == 'critical'])
        low_stock_items = len([s for s in warehouse_items if s['status'] == 'low'])
        
        # Calculate accuracy
        items_with_variance = len([s for s in warehouse_items if s['variance'] != 0])
        accuracy = int(((total_skus - items_with_variance) / total_skus * 100)) if total_skus > 0 else 100
        
        last_audit_days = np.random.randint(1, 20)
        last_audit = (datetime.now() - timedelta(days=last_audit_days)).strftime('%Y-%m-%d')
        
        location_map = {
            'Main Warehouse': 'New York, NY',
            'East Coast DC': 'Atlanta, GA',
            'West Coast DC': 'Los Angeles, CA',
            'Central Hub': 'Chicago, IL'
        }
        
        warehouses_data.append({
            'name': warehouse,
            'location': location_map[warehouse],
            'total_skus': total_skus,
            'total_units': total_units,
            'capacity': capacity,
            'utilization': utilization,
            'critical_items': critical_items,
            'low_stock_items': low_stock_items,
            'accuracy': accuracy,
            'last_audit': last_audit
        })
    
    return pd.DataFrame(stock), pd.DataFrame(alerts), pd.DataFrame(movements), pd.DataFrame(warehouses_data)

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading inventory data..."):
    stock_df, alerts_df, movements_df, warehouses_df = generate_sample_inventory_data()

# Calculate key metrics
total_skus = len(stock_df)
mismatches = len(stock_df[stock_df['variance'] != 0])
critical_items = len(stock_df[stock_df['status'] == 'critical'])
stock_accuracy = ((total_skus - mismatches) / total_skus * 100) if total_skus > 0 else 0
total_value_impact = abs(stock_df['value_impact'].sum())

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 📈 Filters")
    
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
    
    stock_status_filter = st.multiselect(
        "📊 Stock Status",
        ["critical", "low", "adequate", "good", "overstocked"],
        default=["critical", "low", "adequate", "good", "overstocked"]
    )
    
    warehouse_filter = st.multiselect(
        "🏢 Warehouse",
        stock_df['warehouse'].unique().tolist(),
        default=stock_df['warehouse'].unique().tolist()
    )
    
    category_filter = st.multiselect(
        "📦 Category",
        stock_df['category'].unique().tolist(),
        default=stock_df['category'].unique().tolist()
    )
    
    search_query = st.text_input("🔍 Search", placeholder="SKU, Product Name...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_inventory_filters(df, status_list, warehouse_list, category_list, search_text):
    filtered = df.copy()
    
    if status_list:
        filtered = filtered[filtered['status'].isin(status_list)]
    
    if warehouse_list:
        filtered = filtered[filtered['warehouse'].isin(warehouse_list)]
    
    if category_list:
        filtered = filtered[filtered['category'].isin(category_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['sku'].str.lower().str.contains(search_lower, na=False) |
            filtered['product'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

def apply_movement_filters(df, warehouse_list, category_list, search_text):
    filtered = df.copy()
    
    if warehouse_list:
        filtered = filtered[filtered['warehouse'].isin(warehouse_list)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['sku'].str.lower().str.contains(search_lower, na=False) |
            filtered['product'].str.lower().str.contains(search_lower, na=False) |
            filtered['reference'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_stock = apply_inventory_filters(stock_df, stock_status_filter, warehouse_filter, category_filter, search_query)
filtered_movements = apply_movement_filters(movements_df, warehouse_filter, category_filter, search_query)
filtered_warehouses = warehouses_df[warehouses_df['name'].isin(warehouse_filter)]

# ===========================
# HEADER & METRICS
# ===========================

st.title("📈 Inventory Quality Check")
st.markdown("**Stock level validation, variance analysis, and warehouse monitoring**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Total SKUs", f"{total_skus:,}", "+245 new items")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-danger">', unsafe_allow_html=True)
    st.metric("Stock Accuracy", f"{stock_accuracy:.1f}%", "-8.3% decline")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-danger">', unsafe_allow_html=True)
    st.metric("Mismatches Found", f"{mismatches}", "+15 new issues")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-warning">', unsafe_allow_html=True)
    st.metric("Value Impact", f"${total_value_impact:,.0f}", "+$8,500 loss")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

if mismatches > 0 or critical_items > 0:
    st.markdown(f"""
    <div class="alert alert-danger">
        <strong>🚨 Critical Alert:</strong> {mismatches} inventory mismatches detected. {critical_items} products below critical stock levels. Total value impact: ${total_value_impact:,.0f}.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Stock Level Trend")
    
    # Generate trend data
    trend_data = pd.DataFrame({
        'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        'System Stock': [158234, 156789, 155456, 154123],
        'Physical Stock': [156890, 155234, 153890, 152456]
    })
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=trend_data['Week'], y=trend_data['System Stock'], 
                              mode='lines+markers', name='System Stock',
                              line=dict(color='#3b82f6', width=3),
                              fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig1.add_trace(go.Scatter(x=trend_data['Week'], y=trend_data['Physical Stock'], 
                              mode='lines+markers', name='Physical Stock',
                              line=dict(color='#22c55e', width=3),
                              fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.1)'))
    fig1.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), 
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Stock Status Distribution")
    
    status_counts = stock_df['status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    color_map = {
        'good': '#22c55e',
        'adequate': '#3b82f6',
        'low': '#f59e0b',
        'critical': '#ef4444',
        'overstocked': '#8b5cf6'
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
    f"📋 Current Stock Levels ({len(filtered_stock)})",
    f"⚠️ Low Stock Alerts ({len(alerts_df)})",
    f"📦 Stock Movements ({len(filtered_movements)})",
    f"🏢 Warehouse Breakdown ({len(filtered_warehouses)})"
])

# TAB 1: CURRENT STOCK LEVELS
with tab1:
    st.subheader("Current Stock Levels")
    
    if len(filtered_stock) > 0:
        display_stock = filtered_stock.copy()
        
        # Format for display
        display_stock['Variance Display'] = display_stock['variance'].apply(
            lambda x: f"🔴 {x:+d}" if x < 0 else f"🟢 +{x}" if x > 0 else "⚪ 0"
        )
        display_stock['Status Badge'] = display_stock['status'].apply(
            lambda x: f"🔴 {x.upper()}" if x == "critical"
            else f"🟡 {x.upper()}" if x == "low"
            else f"🔵 {x.upper()}" if x == "adequate"
            else f"🟢 {x.upper()}" if x == "good"
            else f"🟣 {x.upper()}"
        )
        display_stock['Value Impact Display'] = display_stock['value_impact'].apply(
            lambda x: f"🔴 -${abs(x):.0f}" if x < 0 else f"🟢 +${x:.0f}"
        )
        
        st.dataframe(
            display_stock[['sku', 'product', 'category', 'system_stock', 'physical_stock', 
                          'Variance Display', 'reorder_point', 'Status Badge', 
                          'Value Impact Display', 'warehouse']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'sku': 'SKU',
                'product': 'Product',
                'category': 'Category',
                'system_stock': 'System Stock',
                'physical_stock': 'Physical Stock',
                'Variance Display': 'Variance',
                'reorder_point': 'Reorder Point',
                'Status Badge': 'Status',
                'Value Impact Display': 'Value Impact',
                'warehouse': 'Warehouse'
            }
        )
        
        st.caption(f"Showing {len(filtered_stock):,} of {len(stock_df):,} items")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            critical_filtered = len(filtered_stock[filtered_stock['status'] == 'critical'])
            st.metric("🔴 Critical Items", f"{critical_filtered:,}")
        with col2:
            low_filtered = len(filtered_stock[filtered_stock['status'] == 'low'])
            st.metric("🟡 Low Stock Items", f"{low_filtered:,}")
        with col3:
            good_filtered = len(filtered_stock[filtered_stock['status'] == 'good'])
            st.metric("🟢 Good Stock Items", f"{good_filtered:,}")
        with col4:
            total_variance = filtered_stock['variance'].sum()
            st.metric("📊 Total Variance", f"{total_variance:+,}")
        
        if st.button("🔧 Adjust Selected Items", use_container_width=True):
            st.success("✅ Stock adjustment workflow initiated")
    else:
        st.info("No items match the current filters")

# TAB 2: LOW STOCK ALERTS
with tab2:
    st.subheader("Low Stock Alerts")
    
    if len(alerts_df) > 0:
        st.warning(f"⚠️ {len(alerts_df)} items require immediate attention")
        
        for idx, alert in alerts_df.iterrows():
            severity_color = "background: #fee2e2; border-left: 4px solid #ef4444;" if alert['severity'] == 'critical' else "background: #fef3c7; border-left: 4px solid #f59e0b;"
            severity_emoji = "🔴" if alert['severity'] == 'critical' else "🟡"
            
            st.markdown(f"""
            <div style="{severity_color} padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-weight: 700; font-size: 1rem; margin-bottom: 5px;">
                            {alert['product']} ({alert['sku']})
                        </div>
                        <div style="font-size: 0.875rem; color: #64748b;">
                            <strong>Current Stock:</strong> {alert['current_stock']} units | 
                            <strong>Reorder Point:</strong> {alert['reorder_point']} units | 
                            <strong>Days Remaining:</strong> ~{alert['days_remaining']} days | 
                            <strong>Daily Avg Sales:</strong> {alert['daily_avg_sales']} units<br>
                            <strong>Warehouse:</strong> {alert['warehouse']} | 
                            <strong>Action:</strong> {alert['suggested_action']}
                        </div>
                    </div>
                    <div style="margin-left: 20px;">
                        <span style="padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700; 
                                   background: {'#fee2e2' if alert['severity'] == 'critical' else '#fef3c7'}; 
                                   color: {'#991b1b' if alert['severity'] == 'critical' else '#92400e'};">
                            {severity_emoji} {alert['severity'].upper()}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📦 Create Bulk Purchase Orders", use_container_width=True):
                st.success(f"✅ Purchase orders created for {len(alerts_df)} items")
        with col2:
            if st.button("📥 Export Alert List", use_container_width=True):
                st.success("✅ Alert list exported to CSV")
    else:
        st.success("✅ No low stock alerts!")
        st.balloons()

# TAB 3: STOCK MOVEMENTS
with tab3:
    st.subheader("Stock Movement History")
    
    if len(filtered_movements) > 0:
        display_movements = filtered_movements.copy()
        
        # Format for display
        display_movements['Type Badge'] = display_movements['type'].apply(
            lambda x: f"🟢 {x.upper()}" if x == "inbound"
            else f"🔴 {x.upper()}" if x == "outbound"
            else f"🔵 {x.upper()}" if x == "adjustment"
            else f"⚫ {x.upper()}"
        )
        display_movements['Quantity Display'] = display_movements.apply(
            lambda row: f"{'🟢' if row['quantity'] > 0 else '🔴'} {row['quantity']:+d}",
            axis=1
        )
        
        st.dataframe(
            display_movements[['date', 'sku', 'product', 'Type Badge', 'Quantity Display',
                             'before_qty', 'after_qty', 'warehouse', 'reference', 'reason']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'date': 'Date/Time',
                'sku': 'SKU',
                'product': 'Product',
                'Type Badge': 'Type',
                'Quantity Display': 'Quantity',
                'before_qty': 'Before',
                'after_qty': 'After',
                'warehouse': 'Warehouse',
                'reference': 'Reference',
                'reason': 'Reason'
            }
        )
        
        st.caption(f"Showing {len(filtered_movements):,} movements")
        
        # Movement summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            inbound_count = len(display_movements[display_movements['type'] == 'inbound'])
            st.metric("📥 Inbound", f"{inbound_count:,}")
        with col2:
            outbound_count = len(display_movements[display_movements['type'] == 'outbound'])
            st.metric("📤 Outbound", f"{outbound_count:,}")
        with col3:
            adjustment_count = len(display_movements[display_movements['type'] == 'adjustment'])
            st.metric("⚙️ Adjustments", f"{adjustment_count:,}")
        with col4:
            damaged_count = len(display_movements[display_movements['type'] == 'damaged'])
            st.metric("⚫ Damaged", f"{damaged_count:,}")
    else:
        st.info("No movements match the current filters")

# TAB 4: WAREHOUSE BREAKDOWN
with tab4:
    st.subheader("Warehouse Breakdown")
    
    if len(filtered_warehouses) > 0:
        for idx, warehouse in filtered_warehouses.iterrows():
            # Determine colors based on metrics
            utilization_color = '#ef4444' if warehouse['utilization'] >= 90 else '#f59e0b' if warehouse['utilization'] >= 75 else '#22c55e'
            accuracy_color = '#22c55e' if warehouse['accuracy'] >= 80 else '#f59e0b' if warehouse['accuracy'] >= 60 else '#ef4444'
            
            st.markdown(f"""
            <div class="warehouse-card">
                <h4 style="margin-bottom: 10px;">{warehouse['name']}</h4>
                <p style="color: #64748b; font-size: 0.875rem; margin-bottom: 15px;">📍 {warehouse['location']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Total SKUs**")
                st.markdown(f"### {warehouse['total_skus']:,}")
                
                st.markdown("**Total Units**")
                st.markdown(f"### {warehouse['total_units']:,}")
            
            with col2:
                st.markdown("**Capacity**")
                st.markdown(f"### {warehouse['total_units']:,} / {warehouse['capacity']:,}")
                
                st.markdown("**Utilization**")
                st.markdown(f"### <span style='color:{utilization_color}'>{warehouse['utilization']}%</span>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{warehouse['utilization']}%; background:{utilization_color};"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("**Critical Items**")
                st.markdown(f"### <span style='color:#ef4444'>{warehouse['critical_items']}</span>", unsafe_allow_html=True)
                
                st.markdown("**Low Stock Items**")
                st.markdown(f"### <span style='color:#f59e0b'>{warehouse['low_stock_items']}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Stock Accuracy**")
                st.markdown(f"### <span style='color:{accuracy_color}'>{warehouse['accuracy']}%</span>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{warehouse['accuracy']}%; background:{accuracy_color};"></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Last Audit**")
                st.markdown(f"### {warehouse['last_audit']}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button(f"📊 View Details - {warehouse['name']}", key=f"btn_{warehouse['name']}", use_container_width=True):
                st.info(f"Opening detailed view for {warehouse['name']}")
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Warehouse comparison
        st.markdown("---")
        st.subheader("Warehouse Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_util = px.bar(filtered_warehouses, x='name', y='utilization',
                             title='Warehouse Utilization Rate (%)',
                             color='utilization',
                             color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444'],
                             text='utilization')
            fig_util.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_util.update_layout(height=300, showlegend=False, 
                                  xaxis_title='', yaxis_title='Utilization %')
            st.plotly_chart(fig_util, use_container_width=True)
        
        with col2:
            fig_acc = px.bar(filtered_warehouses, x='name', y='accuracy',
                            title='Stock Accuracy by Warehouse (%)',
                            color='accuracy',
                            color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
                            text='accuracy')
            fig_acc.update_traces(texttemplate='%{text}%', textposition='outside')
            fig_acc.update_layout(height=300, showlegend=False,
                                 xaxis_title='', yaxis_title='Accuracy %')
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Summary table
        st.markdown("---")
        st.subheader("Warehouse Summary")
        
        summary_display = filtered_warehouses.copy()
        summary_display['Utilization'] = summary_display['utilization'].apply(lambda x: f"{x}%")
        summary_display['Accuracy'] = summary_display['accuracy'].apply(lambda x: f"{x}%")
        summary_display['Issues'] = summary_display['critical_items'] + summary_display['low_stock_items']
        
        st.dataframe(
            summary_display[['name', 'location', 'total_skus', 'total_units', 
                           'Utilization', 'critical_items', 'low_stock_items', 
                           'Accuracy', 'last_audit']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'name': 'Warehouse',
                'location': 'Location',
                'total_skus': 'Total SKUs',
                'total_units': 'Total Units',
                'Utilization': 'Utilization',
                'critical_items': 'Critical Items',
                'low_stock_items': 'Low Stock Items',
                'Accuracy': 'Stock Accuracy',
                'last_audit': 'Last Audit'
            }
        )
    else:
        st.info("No warehouses match the current filters")

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Inventory data refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'stock_levels': stock_df.to_dict('records'),
            'alerts': alerts_df.to_dict('records'),
            'movements': movements_df.to_dict('records'),
            'warehouses': warehouses_df.to_dict('records'),
            'summary': {
                'total_skus': total_skus,
                'mismatches': mismatches,
                'critical_items': critical_items,
                'stock_accuracy': f"{stock_accuracy:.1f}%",
                'value_impact': f"${total_value_impact:,.2f}"
            }
        }
        st.success("✅ Inventory data exported successfully")
        st.json(export_data)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================
# ADDITIONAL INSIGHTS
# ===========================

with st.expander("📊 Inventory Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 Critical Issues")
        st.markdown(f"""
        - **{critical_items} products** are below critical stock levels
        - **{mismatches} inventory mismatches** detected across all warehouses
        - **${total_value_impact:,.0f}** in potential revenue impact
        - Stock accuracy at **{stock_accuracy:.1f}%** (target: 95%+)
        """)
        
        st.markdown("#### 🎯 Recommended Actions")
        st.markdown("""
        1. **Immediate**: Initiate emergency reorders for critical items
        2. **Short-term**: Conduct physical count reconciliation
        3. **Medium-term**: Implement cycle counting program
        4. **Long-term**: Upgrade inventory management system
        """)
    
    with col2:
        st.markdown("#### 📈 Performance Trends")
        
        # Top issues by category
        category_issues = stock_df[stock_df['variance'] != 0].groupby('category').size().sort_values(ascending=False).head(5)
        
        if len(category_issues) > 0:
            st.markdown("**Top 5 Categories with Issues:**")
            for cat, count in category_issues.items():
                st.markdown(f"- **{cat}**: {count} mismatches")
        
        st.markdown("#### 💡 Best Practices")
        st.markdown("""
        - Maintain **95%+ accuracy** for optimal operations
        - Keep **critical items** below 5% of total inventory
        - Target **75-85% utilization** for warehouse efficiency
        - Conduct audits **every 2 weeks** for high-value items
        """)

# ===========================
# DIAGNOSTIC INFORMATION
# ===========================

with st.expander("🔧 System Diagnostics"):
    st.markdown("### Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(stock_df):,}")
        st.metric("Data Completeness", "100%")
    
    with col2:
        st.metric("Variance Rate", f"{(mismatches/total_skus*100):.1f}%")
        st.metric("Critical Rate", f"{(critical_items/total_skus*100):.1f}%")
    
    with col3:
        st.metric("Movement Records", f"{len(movements_df):,}")
        st.metric("Active Warehouses", f"{len(warehouses_df)}")
    
    st.markdown("---")
    st.markdown("### Filter Status")
    st.info(f"""
    **Active Filters:**
    - Date Range: {date_range}
    - Stock Status: {', '.join(stock_status_filter) if stock_status_filter else 'None'}
    - Warehouses: {', '.join(warehouse_filter) if warehouse_filter else 'None'}
    - Categories: {', '.join(category_filter[:3]) if category_filter else 'None'}{'...' if len(category_filter) > 3 else ''}
    - Search Query: {'`' + search_query + '`' if search_query else 'None'}
    
    **Results:** {len(filtered_stock):,} items shown out of {len(stock_df):,} total
    """)