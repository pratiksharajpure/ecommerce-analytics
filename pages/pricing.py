"""
💰 Enhanced Price Optimization Audit - All 4 Tabs with Working Filters
Dynamic pricing analysis, elasticity tracking, margin optimization, and competitive pricing
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Price Optimization Audit",
    page_icon="💰",
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
    .comparison-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px;
        background: #f8fafc;
        border-radius: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# GENERATE SAMPLE DATA
# ===========================

@st.cache_data(ttl=600)
def generate_sample_pricing_data():
    """Generate sample pricing data"""
    np.random.seed(42)
    
    # Price Optimization Data
    optimization = []
    products = [
        ('ELEC-001', 'Wireless Headphones', 79.99, 89.99, 45.00, 'Electronics'),
        ('CLOTH-045', 'Cotton T-Shirt', 29.99, 24.99, 12.00, 'Clothing'),
        ('HOME-089', 'Kitchen Blender', 149.99, 149.99, 75.00, 'Home & Garden'),
        ('SPORT-567', 'Yoga Mat Premium', 59.99, 69.99, 28.00, 'Sports'),
        ('BOOK-234', 'Programming Guide', 49.99, 44.99, 18.00, 'Books'),
        ('TOY-123', 'Educational Toy Set', 39.99, 39.99, 18.00, 'Toys'),
        ('FOOD-789', 'Organic Snack Pack', 12.99, 14.99, 6.50, 'Food'),
        ('ELEC-002', 'Smart Watch', 199.99, 219.99, 120.00, 'Electronics'),
        ('CLOTH-046', 'Denim Jeans', 69.99, 64.99, 30.00, 'Clothing'),
        ('HOME-090', 'Vacuum Cleaner', 299.99, 289.99, 180.00, 'Home & Garden'),
        ('SPORT-568', 'Running Shoes', 89.99, 99.99, 45.00, 'Sports'),
        ('BOOK-235', 'Data Science Book', 39.99, 34.99, 15.00, 'Books'),
        ('ELEC-003', 'Bluetooth Speaker', 49.99, 54.99, 25.00, 'Electronics'),
        ('CLOTH-047', 'Winter Jacket', 149.99, 139.99, 70.00, 'Clothing'),
        ('HOME-091', 'Coffee Maker', 79.99, 84.99, 40.00, 'Home & Garden'),
    ]
    
    for sku, product, current, optimal, cost, category in products:
        change = ((optimal - current) / current) * 100
        revenue_impact = np.random.randint(5000, 20000)
        demand_impact = -int(change * 0.8)
        
        if abs(change) < 1:
            status = 'optimal'
        elif change > 0:
            status = 'underpriced'
        else:
            status = 'overpriced'
        
        current_margin = ((current - cost) / current) * 100
        optimal_margin = ((optimal - cost) / optimal) * 100
        
        optimization.append({
            'sku': sku,
            'product': product,
            'category': category,
            'current_price': current,
            'optimal_price': optimal,
            'cost': cost,
            'suggested_change': change,
            'revenue_impact': revenue_impact,
            'demand_impact': demand_impact,
            'status': status,
            'current_margin': current_margin,
            'optimal_margin': optimal_margin
        })
    
    # Price Elasticity Data
    elasticity = []
    for item in optimization:
        elast_coef = round(np.random.uniform(-2.5, -0.5), 2)
        elast_type = 'elastic' if abs(elast_coef) > 1.0 else 'inelastic'
        price_change = -10
        demand_change = int(abs(elast_coef) * 10)
        revenue_max = np.random.randint(80000, 500000)
        
        elasticity.append({
            'product': item['product'],
            'category': item['category'],
            'elasticity': elast_coef,
            'type': elast_type,
            'price_change_10': price_change,
            'demand_change_10': demand_change,
            'optimal_price_point': item['optimal_price'],
            'revenue_max': revenue_max
        })
    
    # Competitive Pricing Data
    competitive = []
    for item in optimization:
        our_price = item['current_price']
        comp1 = our_price * np.random.uniform(0.95, 1.15)
        comp2 = our_price * np.random.uniform(0.90, 1.10)
        comp3 = our_price * np.random.uniform(0.92, 1.12)
        avg_comp = (comp1 + comp2 + comp3) / 3
        price_diff = ((our_price - avg_comp) / avg_comp) * 100
        
        if abs(price_diff) < 5:
            position = 'competitive'
        elif price_diff > 5:
            position = 'overpriced'
        else:
            position = 'underpriced'
        
        market_share = np.random.randint(10, 30)
        
        competitive.append({
            'product': item['product'],
            'category': item['category'],
            'our_price': our_price,
            'comp1': comp1,
            'comp2': comp2,
            'comp3': comp3,
            'avg_comp_price': avg_comp,
            'price_diff': price_diff,
            'position': position,
            'market_share': market_share
        })
    
    # Margin Analysis Data
    margins = []
    categories = {
        'Electronics': (1234, 245.67, 147.40, 40.0, 42),
        'Clothing': (2345, 45.23, 18.09, 60.0, 58),
        'Home & Garden': (1567, 67.89, 40.73, 40.0, 42),
        'Sports': (890, 89.99, 54.00, 40.0, 45),
        'Books': (678, 23.45, 9.38, 60.0, 60),
        'Food': (456, 15.67, 7.83, 50.0, 52)
    }
    
    for cat, (products, avg_price, avg_cost, avg_margin, target) in categories.items():
        revenue = products * avg_price
        profit = revenue * (avg_margin / 100)
        
        margins.append({
            'category': cat,
            'products': products,
            'avg_price': avg_price,
            'avg_cost': avg_cost,
            'avg_margin': avg_margin,
            'revenue': revenue,
            'profit': profit,
            'target_margin': target
        })
    
    return (pd.DataFrame(optimization), pd.DataFrame(elasticity), 
            pd.DataFrame(competitive), pd.DataFrame(margins))

# ===========================
# LOAD DATA
# ===========================

with st.spinner("Loading pricing data..."):
    optimization_df, elasticity_df, competitive_df, margins_df = generate_sample_pricing_data()

# ===========================
# SIDEBAR FILTERS
# ===========================

with st.sidebar:
    st.markdown("### 💰 Filters")
    
    category_filter = st.multiselect(
        "📦 Category",
        optimization_df['category'].unique().tolist(),
        default=optimization_df['category'].unique().tolist()
    )
    
    price_range = st.selectbox(
        "💵 Price Range",
        ["All Prices", "$0 - $25", "$25 - $50", "$50 - $100", "$100+"],
        index=0
    )
    
    status_filter = st.multiselect(
        "📊 Status",
        ["optimal", "overpriced", "underpriced"],
        default=["optimal", "overpriced", "underpriced"]
    )
    
    search_query = st.text_input("🔍 Search", placeholder="Name, SKU...")
    
    st.markdown("---")
    if st.button("🔄 Reset Filters", use_container_width=True, key="reset_filters"):
        st.cache_data.clear()
        st.rerun()

# ===========================
# APPLY FILTERS
# ===========================

def apply_pricing_filters(df, categories, price_rng, statuses, search_text):
    filtered = df.copy()
    
    if categories:
        filtered = filtered[filtered['category'].isin(categories)]
    
    if price_rng == "$0 - $25":
        filtered = filtered[filtered['current_price'] <= 25]
    elif price_rng == "$25 - $50":
        filtered = filtered[(filtered['current_price'] > 25) & (filtered['current_price'] <= 50)]
    elif price_rng == "$50 - $100":
        filtered = filtered[(filtered['current_price'] > 50) & (filtered['current_price'] <= 100)]
    elif price_rng == "$100+":
        filtered = filtered[filtered['current_price'] > 100]
    
    if statuses and 'status' in filtered.columns:
        filtered = filtered[filtered['status'].isin(statuses)]
    
    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered['product'].str.lower().str.contains(search_lower, na=False) |
            filtered['sku'].str.lower().str.contains(search_lower, na=False)
        ]
    
    return filtered

filtered_optimization = apply_pricing_filters(optimization_df, category_filter, price_range, status_filter, search_query)
filtered_elasticity = elasticity_df[elasticity_df['category'].isin(category_filter)]
filtered_competitive = competitive_df[competitive_df['category'].isin(category_filter)]

# ===========================
# CALCULATE METRICS
# ===========================

total_products = len(optimization_df)
avg_elasticity = elasticity_df['elasticity'].mean()
optimal_prices = len(optimization_df[optimization_df['status'] == 'optimal'])
revenue_impact = optimization_df['revenue_impact'].sum()

# ===========================
# HEADER & METRICS
# ===========================

st.title("💰 Price Optimization Audit")
st.markdown("**Dynamic pricing analysis, elasticity tracking, and margin optimization**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="stat-card stat-card-primary">', unsafe_allow_html=True)
    st.metric("Products Analyzed", f"{total_products:,}", "+245 new products")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Avg Price Elasticity", f"{avg_elasticity:.1f}", "-0.2 improved")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Optimal Prices Set", f"{optimal_prices:,}", "+456 optimized")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="stat-card stat-card-success">', unsafe_allow_html=True)
    st.metric("Revenue Impact", f"+${revenue_impact/1000:.0f}K", "+$45K potential")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===========================
# ALERT
# ===========================

optimal_count = len(optimization_df[optimization_df['status'] == 'optimal'])
st.markdown(f"""
<div class="alert alert-success">
    <strong>💰 Optimization Opportunity:</strong> {total_products:,} products analyzed. {optimal_count} products at optimal pricing. Potential revenue increase of ${revenue_impact/1000:.0f}K identified through optimal pricing adjustments.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===========================
# CHARTS
# ===========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price vs Demand Curve")
    
    demand_data = pd.DataFrame({
        'Price': ['$60', '$70', '$80', '$90', '$100', '$110', '$120'],
        'Demand': [2800, 2450, 2100, 1750, 1400, 1050, 700],
        'Revenue': [168, 171.5, 168, 157.5, 140, 115.5, 84]
    })
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=demand_data['Price'], y=demand_data['Demand'],
                              mode='lines+markers', name='Demand (Units)',
                              line=dict(color='#3b82f6', width=3),
                              fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig1.add_trace(go.Scatter(x=demand_data['Price'], y=demand_data['Revenue'],
                              mode='lines+markers', name='Revenue ($K)',
                              line=dict(color='#22c55e', width=3),
                              fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.1)',
                              yaxis='y2'))
    fig1.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title='Demand (Units)'),
        yaxis2=dict(title='Revenue ($K)', overlaying='y', side='right')
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Margin Distribution")
    
    margin_chart_data = margins_df[['category', 'avg_margin', 'target_margin']].copy()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=margin_chart_data['category'], y=margin_chart_data['avg_margin'],
                          name='Current Margin %', marker_color='#3b82f6'))
    fig2.add_trace(go.Bar(x=margin_chart_data['category'], y=margin_chart_data['target_margin'],
                          name='Target Margin %', marker_color='#22c55e'))
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                       yaxis=dict(title='Margin %'))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4 = st.tabs([
    f"💵 Price Optimization ({len(filtered_optimization)})",
    f"📊 Price Elasticity",
    f"🎯 Competitive Pricing",
    f"📈 Margin Analysis"
])

# TAB 1: PRICE OPTIMIZATION
with tab1:
    st.subheader("Price Optimization Recommendations")
    
    if len(filtered_optimization) > 0:
        display_opt = filtered_optimization.copy()
        
        display_opt['Current Price'] = display_opt['current_price'].apply(lambda x: f"${x:.2f}")
        display_opt['Optimal Price'] = display_opt['optimal_price'].apply(lambda x: f"${x:.2f}")
        display_opt['Cost'] = display_opt['cost'].apply(lambda x: f"${x:.2f}")
        display_opt['Change %'] = display_opt['suggested_change'].apply(lambda x: f"{x:+.1f}%")
        display_opt['Revenue Impact'] = display_opt['revenue_impact'].apply(lambda x: f"+${x:,}")
        display_opt['Demand Impact'] = display_opt['demand_impact'].apply(lambda x: f"{x:+d}%")
        display_opt['Status Badge'] = display_opt['status'].apply(
            lambda x: f"🟢 {x.upper()}" if x == "optimal"
            else f"🔴 {x.upper()}" if x == "overpriced"
            else f"🟡 {x.upper()}"
        )
        display_opt['Current Margin %'] = display_opt['current_margin'].apply(lambda x: f"{x:.1f}%")
        display_opt['Optimal Margin %'] = display_opt['optimal_margin'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_opt[['sku', 'product', 'category', 'Current Price', 'Optimal Price', 
                        'Cost', 'Change %', 'Revenue Impact', 'Demand Impact', 
                        'Status Badge', 'Current Margin %', 'Optimal Margin %']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'sku': 'SKU',
                'product': 'Product',
                'category': 'Category',
                'Current Price': 'Current Price',
                'Optimal Price': 'Optimal Price',
                'Cost': 'Cost',
                'Change %': 'Suggested Change',
                'Revenue Impact': 'Revenue Impact',
                'Demand Impact': 'Demand Impact',
                'Status Badge': 'Status',
                'Current Margin %': 'Current Margin',
                'Optimal Margin %': 'Optimal Margin'
            }
        )
        
        st.caption(f"Showing {len(filtered_optimization):,} of {len(optimization_df):,} products")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            optimal = len(filtered_optimization[filtered_optimization['status'] == 'optimal'])
            st.metric("🟢 Optimal", f"{optimal:,}")
        with col2:
            overpriced = len(filtered_optimization[filtered_optimization['status'] == 'overpriced'])
            st.metric("🔴 Overpriced", f"{overpriced:,}")
        with col3:
            underpriced = len(filtered_optimization[filtered_optimization['status'] == 'underpriced'])
            st.metric("🟡 Underpriced", f"{underpriced:,}")
        with col4:
            total_impact = filtered_optimization['revenue_impact'].sum()
            st.metric("💰 Total Impact", f"${total_impact/1000:.0f}K")
    else:
        st.info("No products match the current filters")

# TAB 2: PRICE ELASTICITY
with tab2:
    st.subheader("Price Elasticity Analysis")
    
    if len(filtered_elasticity) > 0:
        display_elast = filtered_elasticity.copy()
        
        display_elast['Elasticity'] = display_elast['elasticity'].apply(lambda x: f"{x:.2f}")
        display_elast['Type Badge'] = display_elast['type'].apply(
            lambda x: f"🔴 {x.upper()}" if x == "elastic" else f"🔵 {x.upper()}"
        )
        display_elast['Price Change'] = display_elast['price_change_10'].apply(lambda x: f"{x}%")
        display_elast['Demand Change'] = display_elast['demand_change_10'].apply(lambda x: f"+{x}%")
        display_elast['Optimal Price'] = display_elast['optimal_price_point'].apply(lambda x: f"${x:.2f}")
        display_elast['Revenue Max'] = display_elast['revenue_max'].apply(lambda x: f"${x:,}")
        
        st.dataframe(
            display_elast[['product', 'category', 'Elasticity', 'Type Badge', 
                          'Price Change', 'Demand Change', 'Optimal Price', 'Revenue Max']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'product': 'Product',
                'category': 'Category',
                'Elasticity': 'Elasticity Coefficient',
                'Type Badge': 'Type',
                'Price Change': 'Price Change (10%)',
                'Demand Change': 'Demand Change',
                'Optimal Price': 'Optimal Price Point',
                'Revenue Max': 'Revenue Maximization'
            }
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Elasticity Distribution")
            
            elast_dist = filtered_elasticity['type'].value_counts().reset_index()
            elast_dist.columns = ['Type', 'Count']
            
            fig = px.pie(elast_dist, values='Count', names='Type',
                        color='Type',
                        color_discrete_map={'elastic': '#ef4444', 'inelastic': '#3b82f6'})
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Elasticity by Category")
            
            cat_elast = filtered_elasticity.groupby('category')['elasticity'].mean().reset_index()
            cat_elast['elasticity'] = cat_elast['elasticity'].abs()
            
            fig = px.bar(cat_elast, x='category', y='elasticity',
                        color='elasticity',
                        color_continuous_scale='Reds',
                        labels={'elasticity': 'Avg Elasticity', 'category': 'Category'})
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No elasticity data available for selected filters")

# TAB 3: COMPETITIVE PRICING
with tab3:
    st.subheader("Competitive Price Analysis")
    
    if len(filtered_competitive) > 0:
        display_comp = filtered_competitive.copy()
        
        display_comp['Our Price'] = display_comp['our_price'].apply(lambda x: f"${x:.2f}")
        display_comp['Comp 1'] = display_comp['comp1'].apply(lambda x: f"${x:.2f}")
        display_comp['Comp 2'] = display_comp['comp2'].apply(lambda x: f"${x:.2f}")
        display_comp['Comp 3'] = display_comp['comp3'].apply(lambda x: f"${x:.2f}")
        display_comp['Avg Comp'] = display_comp['avg_comp_price'].apply(lambda x: f"${x:.2f}")
        display_comp['Price Diff'] = display_comp['price_diff'].apply(lambda x: f"{x:+.2f}%")
        display_comp['Position Badge'] = display_comp['position'].apply(
            lambda x: f"🟢 {x.upper()}" if x == "competitive"
            else f"🔴 {x.upper()}" if x == "overpriced"
            else f"🟡 {x.upper()}"
        )
        display_comp['Market Share'] = display_comp['market_share'].apply(lambda x: f"{x}%")
        
        st.dataframe(
            display_comp[['product', 'category', 'Our Price', 'Comp 1', 'Comp 2', 'Comp 3',
                         'Avg Comp', 'Price Diff', 'Position Badge', 'Market Share']],
            use_container_width=True,
            hide_index=True,
            height=500,
            column_config={
                'product': 'Product',
                'category': 'Category',
                'Our Price': 'Our Price',
                'Comp 1': 'Competitor 1',
                'Comp 2': 'Competitor 2',
                'Comp 3': 'Competitor 3',
                'Avg Comp': 'Avg Comp Price',
                'Price Diff': 'Price Difference',
                'Position Badge': 'Position',
                'Market Share': 'Market Share'
            }
        )
        
        st.markdown("---")
        
        competitive_count = len(filtered_competitive[filtered_competitive['position'] == 'competitive'])
        overpriced_count = len(filtered_competitive[filtered_competitive['position'] == 'overpriced'])
        underpriced_count = len(filtered_competitive[filtered_competitive['position'] == 'underpriced'])
        avg_market_share = filtered_competitive['market_share'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="comparison-item">', unsafe_allow_html=True)
            st.markdown(f'<span class="comparison-label">🎯 Competitive Products</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="comparison-value" style="color:#22c55e;">{competitive_count} of {len(filtered_competitive)}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="comparison-item">', unsafe_allow_html=True)
            st.markdown(f'<span class="comparison-label">⚠️ Overpriced Products</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="comparison-value" style="color:#ef4444;">{overpriced_count} of {len(filtered_competitive)}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="comparison-item">', unsafe_allow_html=True)
            st.markdown(f'<span class="comparison-label">💰 Underpriced Products</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="comparison-value" style="color:#f59e0b;">{underpriced_count} of {len(filtered_competitive)}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="comparison-item">', unsafe_allow_html=True)
            st.markdown(f'<span class="comparison-label">📊 Avg Market Share</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="comparison-value" style="color:#3b82f6;">{avg_market_share:.1f}%</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Competitive Position Distribution")
            
            position_dist = filtered_competitive['position'].value_counts().reset_index()
            position_dist.columns = ['Position', 'Count']
            
            color_map = {
                'competitive': '#22c55e',
                'overpriced': '#ef4444',
                'underpriced': '#f59e0b'
            }
            
            fig = px.bar(position_dist, x='Position', y='Count',
                        color='Position',
                        color_discrete_map=color_map)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Market Share by Category")
            
            cat_share = filtered_competitive.groupby('category')['market_share'].mean().reset_index()
            
            fig = px.bar(cat_share, x='category', y='market_share',
                        color='market_share',
                        color_continuous_scale='Blues',
                        labels={'market_share': 'Avg Market Share %', 'category': 'Category'})
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No competitive data available for selected filters")

# TAB 4: MARGIN ANALYSIS
with tab4:
    st.subheader("Margin Analysis by Category")
    
    if len(margins_df) > 0:
        # Display margin cards in a grid
        cols = st.columns(3)
        
        for idx, row in margins_df.iterrows():
            col_idx = idx % 3
            
            with cols[col_idx]:
                margin_color = "🟢" if row['avg_margin'] >= row['target_margin'] else "🟡"
                margin_status = "On Target" if row['avg_margin'] >= row['target_margin'] else "Below Target"
                margin_progress = min((row['avg_margin'] / row['target_margin']) * 100, 100)
                
                # Create a container for each category
                with st.container():
                    st.markdown(f"### {margin_color} {row['category']}")
                    
                    # Progress bar for margin performance
                    st.progress(margin_progress / 100, text=f"{margin_status}")
                    
                    # Metrics in columns
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Products", f"{row['products']:,}")
                        st.metric("Avg Price", f"${row['avg_price']:.2f}")
                        st.metric("Current Margin", f"{row['avg_margin']:.1f}%")
                        st.metric("Revenue", f"${row['revenue']/1000000:.2f}M")
                    
                    with m2:
                        st.metric("Avg Cost", f"${row['avg_cost']:.2f}")
                        st.metric("Target Margin", f"{row['target_margin']}%")
                        st.metric("Profit", f"${row['profit']/1000000:.2f}M")
                    
                    st.markdown("---")
        
        st.markdown("---")
        
        st.subheader("Margin Summary Table")
        
        display_margins = margins_df.copy()
        display_margins['Avg Price'] = display_margins['avg_price'].apply(lambda x: f"${x:.2f}")
        display_margins['Avg Cost'] = display_margins['avg_cost'].apply(lambda x: f"${x:.2f}")
        display_margins['Current Margin'] = display_margins['avg_margin'].apply(lambda x: f"{x:.1f}%")
        display_margins['Target Margin'] = display_margins['target_margin'].apply(lambda x: f"{x}%")
        display_margins['Revenue'] = display_margins['revenue'].apply(lambda x: f"${x/1000000:.2f}M")
        display_margins['Profit'] = display_margins['profit'].apply(lambda x: f"${x/1000000:.2f}M")
        
        st.dataframe(
            display_margins[['category', 'products', 'Avg Price', 'Avg Cost', 
                           'Current Margin', 'Target Margin', 'Revenue', 'Profit']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'category': 'Category',
                'products': 'Products',
                'Avg Price': 'Avg Price',
                'Avg Cost': 'Avg Cost',
                'Current Margin': 'Current Margin',
                'Target Margin': 'Target Margin',
                'Revenue': 'Revenue',
                'Profit': 'Profit'
            }
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products_margin = margins_df['products'].sum()
            st.metric("📦 Total Products", f"{total_products_margin:,}")
        
        with col2:
            avg_margin = margins_df['avg_margin'].mean()
            st.metric("📊 Avg Margin", f"{avg_margin:.1f}%")
        
        with col3:
            total_revenue = margins_df['revenue'].sum()
            st.metric("💰 Total Revenue", f"${total_revenue/1000000:.1f}M")
        
        with col4:
            total_profit = margins_df['profit'].sum()
            st.metric("✅ Total Profit", f"${total_profit/1000000:.1f}M")
    else:
        st.info("No margin data available")

st.markdown("---")

# ===========================
# EXPORT & REFRESH
# ===========================

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True, key="refresh_btn_pricing"):
        st.cache_data.clear()
        st.success("✅ Pricing data refreshed successfully")
        st.rerun()

with col2:
    if st.button("📥 Export Data", use_container_width=True, key="export_btn_pricing"):
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'optimization': optimization_df.to_dict('records'),
            'elasticity': elasticity_df.to_dict('records'),
            'competitive': competitive_df.to_dict('records'),
            'margins': margins_df.to_dict('records'),
            'summary': {
                'total_products': total_products,
                'avg_elasticity': f"{avg_elasticity:.2f}",
                'optimal_prices': optimal_prices,
                'revenue_impact': f"${revenue_impact/1000:.0f}K"
            }
        }
        st.success("✅ Pricing data exported successfully")
        st.json(export_data)

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===========================
# ADDITIONAL INSIGHTS
# ===========================

with st.expander("💡 Pricing Insights & Recommendations"):
    st.markdown("### Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Pricing Highlights")
        st.markdown(f"""
        - **{total_products:,}** products analyzed for pricing optimization
        - **{avg_elasticity:.1f}** average price elasticity coefficient
        - **{optimal_prices:,}** products currently at optimal pricing
        - **${revenue_impact/1000:.0f}K** potential revenue increase identified
        - **{len(optimization_df[optimization_df['status'] == 'underpriced'])}** underpriced products detected
        """)
        
        st.markdown("#### 💡 Recommended Actions")
        st.markdown("""
        1. **Immediate**: Adjust underpriced products to optimal levels
        2. **Short-term**: Review overpriced products and competitive positioning
        3. **Medium-term**: Implement dynamic pricing based on elasticity
        4. **Long-term**: Develop AI-powered pricing optimization engine
        """)
    
    with col2:
        st.markdown("#### 📈 Performance Trends")
        
        top_categories = margins_df.nlargest(3, 'avg_margin')
        st.markdown("**Top Performing Categories (Margin):**")
        for idx, row in top_categories.iterrows():
            st.markdown(f"- **{row['category']}**: {row['avg_margin']:.1f}% margin (${row['revenue']/1000000:.2f}M revenue)")
        
        st.markdown("#### 🎯 Optimization Opportunities")
        st.markdown(f"""
        - Increase revenue by **${revenue_impact/1000:.0f}K** through optimal pricing
        - Improve profit margins by **2-5%** in underperforming categories
        - Capture **15-20%** more market share with competitive pricing
        - Implement elasticity-based dynamic pricing for high-volume products
        """)

# ===========================
# DIAGNOSTIC INFORMATION
# ===========================

with st.expander("🔧 System Diagnostics"):
    st.markdown("### Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Products", f"{len(optimization_df):,}")
        st.metric("Data Completeness", "100%")
    
    with col2:
        st.metric("Categories", len(optimization_df['category'].unique()))
        st.metric("Avg Elasticity", f"{avg_elasticity:.2f}")
    
    with col3:
        st.metric("Optimal Prices", f"{optimal_prices:,}")
        st.metric("Revenue Opportunity", f"${revenue_impact/1000:.0f}K")
    
    st.markdown("---")
    st.markdown("### Filter Status")
    st.info(f"""
    **Active Filters:**
    - Categories: {', '.join(category_filter) if category_filter else 'None'}
    - Price Range: {price_range}
    - Status: {', '.join(status_filter) if status_filter else 'None'}
    - Search Query: {'`' + search_query + '`' if search_query else 'None'}
    
    **Results:** {len(filtered_optimization):,} products shown out of {len(optimization_df):,} total
    """)

# ===========================
# PRICING STRATEGIES
# ===========================

with st.expander("📊 Pricing Strategies & Best Practices"):
    st.markdown("### Dynamic Pricing Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Elasticity-Based Pricing")
        st.markdown("""
        **Elastic Products (|E| > 1.0):**
        - Small price decreases can significantly increase demand
        - Focus on volume over margin
        - Use promotional pricing strategically
        - Monitor competitor prices closely
        
        **Inelastic Products (|E| < 1.0):**
        - Price increases have minimal impact on demand
        - Optimize for maximum margin
        - Premium positioning opportunities
        - Less sensitive to competitive pricing
        """)
    
    with col2:
        st.markdown("#### 🏆 Competitive Positioning")
        st.markdown("""
        **Competitive Strategy:**
        - Price within 5% of market average
        - Differentiate on value, not just price
        - Monitor top 3 competitors regularly
        - Adjust pricing dynamically based on inventory
        
        **Market Leadership:**
        - Premium pricing for unique products
        - Bundle pricing for increased value
        - Loyalty discounts for retention
        - Dynamic pricing during peak demand
        """)
    
    st.markdown("---")
    
    st.markdown("### Margin Optimization Tactics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Cost Reduction:**")
        st.markdown("""
        - Negotiate supplier discounts
        - Optimize shipping costs
        - Reduce packaging expenses
        - Bulk purchasing strategies
        """)
    
    with col2:
        st.markdown("**Value Addition:**")
        st.markdown("""
        - Product bundling
        - Add premium options
        - Extended warranties
        - Subscription models
        """)
    
    with col3:
        st.markdown("**Price Optimization:**")
        st.markdown("""
        - A/B price testing
        - Seasonal adjustments
        - Geographic pricing
        - Time-based pricing
        """)

# ===========================
# FOOTER
# ===========================

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;font-size:0.875rem;padding:20px;">
    <strong>Price Optimization Audit System</strong> | Powered by Advanced Analytics | 
    <a href="#" style="color:#3b82f6;text-decoration:none;">Documentation</a> | 
    <a href="#" style="color:#3b82f6;text-decoration:none;">Support</a>
</div>
""", unsafe_allow_html=True)