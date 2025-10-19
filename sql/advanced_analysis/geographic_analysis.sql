-- ========================================
-- GEOGRAPHIC ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Sales by Region, Regional Trends, Market Penetration
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. SALES PERFORMANCE BY REGION
-- Revenue, orders, and customers by state/city
-- ========================================

WITH regional_sales AS (
    SELECT 
        c.state,
        c.city,
        c.country,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        SUM(o.total_amount - o.shipping_cost - o.tax_amount) AS net_revenue,
        -- Recent performance
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            THEN o.order_id 
        END) AS orders_last_30d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            THEN o.total_amount ELSE 0 
        END) AS revenue_last_30d,
        -- Customer behavior
        COUNT(DISTINCT CASE 
            WHEN (SELECT COUNT(*) FROM orders o2 
                  WHERE o2.customer_id = c.customer_id 
                    AND o2.status IN ('delivered', 'shipped', 'processing')) > 1 
            THEN c.customer_id 
        END) AS repeat_customers
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE c.status = 'active'
    GROUP BY c.state, c.city, c.country
),
state_rankings AS (
    SELECT 
        state,
        SUM(total_revenue) AS state_revenue,
        SUM(total_customers) AS state_customers,
        SUM(total_orders) AS state_orders
    FROM regional_sales
    GROUP BY state
)
SELECT 
    rs.state,
    rs.city,
    rs.country,
    rs.total_customers,
    rs.total_orders,
    ROUND(rs.total_revenue, 2) AS total_revenue,
    ROUND(rs.net_revenue, 2) AS net_revenue,
    ROUND(rs.avg_order_value, 2) AS avg_order_value,
    rs.orders_last_30d,
    ROUND(rs.revenue_last_30d, 2) AS revenue_last_30d,
    -- Customer metrics
    rs.repeat_customers,
    ROUND(rs.repeat_customers * 100.0 / NULLIF(rs.total_customers, 0), 2) AS repeat_customer_pct,
    ROUND(rs.total_orders * 1.0 / NULLIF(rs.total_customers, 0), 2) AS orders_per_customer,
    ROUND(rs.total_revenue / NULLIF(rs.total_customers, 0), 2) AS revenue_per_customer,
    -- State context
    sr.state_revenue,
    ROUND(rs.total_revenue * 100.0 / NULLIF(sr.state_revenue, 0), 2) AS pct_of_state_revenue,
    -- National context
    ROUND(rs.total_revenue * 100.0 / (SELECT SUM(total_revenue) FROM regional_sales), 2) AS pct_of_national_revenue,
    -- Rankings
    RANK() OVER (ORDER BY rs.total_revenue DESC) AS national_revenue_rank,
    RANK() OVER (PARTITION BY rs.state ORDER BY rs.total_revenue DESC) AS state_revenue_rank,
    -- Market classification
    CASE 
        WHEN rs.total_revenue >= (SELECT AVG(total_revenue) FROM regional_sales) * 2 THEN 'Major Market'
        WHEN rs.total_revenue >= (SELECT AVG(total_revenue) FROM regional_sales) THEN 'Strong Market'
        WHEN rs.total_revenue >= (SELECT AVG(total_revenue) FROM regional_sales) * 0.5 THEN 'Developing Market'
        ELSE 'Emerging Market'
    END AS market_tier,
    -- Growth potential
    CASE 
        WHEN rs.orders_last_30d * 12 > rs.total_orders * 1.2 THEN 'High Growth'
        WHEN rs.orders_last_30d * 12 > rs.total_orders * 0.9 THEN 'Stable Growth'
        WHEN rs.orders_last_30d * 12 > rs.total_orders * 0.7 THEN 'Slowing'
        ELSE 'Declining'
    END AS growth_trend
FROM regional_sales rs
JOIN state_rankings sr ON rs.state = sr.state
WHERE rs.total_revenue > 0
ORDER BY rs.total_revenue DESC;

-- ========================================
-- 2. STATE-LEVEL MARKET PENETRATION
-- Market share and penetration analysis
-- ========================================

WITH state_metrics AS (
    SELECT 
        c.state,
        c.country,
        COUNT(DISTINCT c.customer_id) AS active_customers,
        COUNT(DISTINCT CASE 
            WHEN c.created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN c.customer_id 
        END) AS new_customers_90d,
        SUM(o.total_amount) AS total_revenue,
        COUNT(DISTINCT o.order_id) AS total_orders,
        AVG(o.total_amount) AS avg_order_value,
        -- Customer lifetime value
        AVG((SELECT SUM(o2.total_amount) 
             FROM orders o2 
             WHERE o2.customer_id = c.customer_id 
               AND o2.status IN ('delivered', 'shipped', 'processing'))) AS avg_customer_ltv,
        -- Product diversity
        COUNT(DISTINCT oi.product_id) AS unique_products_sold,
        COUNT(DISTINCT p.category_id) AS categories_purchased
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    WHERE c.status = 'active'
    GROUP BY c.state, c.country
),
-- US Census population data (simplified - would typically come from external source)
state_population AS (
    SELECT 'CA' AS state, 39538223 AS population
    UNION ALL SELECT 'TX', 29145505
    UNION ALL SELECT 'FL', 21538187
    UNION ALL SELECT 'NY', 20201249
    UNION ALL SELECT 'PA', 13002700
    -- Add more states as needed
),
penetration_metrics AS (
    SELECT 
        sm.state,
        sm.country,
        sm.active_customers,
        sm.new_customers_90d,
        ROUND(sm.total_revenue, 2) AS total_revenue,
        sm.total_orders,
        ROUND(sm.avg_order_value, 2) AS avg_order_value,
        ROUND(sm.avg_customer_ltv, 2) AS avg_customer_ltv,
        sm.unique_products_sold,
        sm.categories_purchased,
        COALESCE(sp.population, 1000000) AS estimated_population,
        -- Penetration calculations
        ROUND(sm.active_customers * 100.0 / NULLIF(COALESCE(sp.population, 1000000), 0), 4) AS market_penetration_pct,
        ROUND(sm.total_revenue / NULLIF(COALESCE(sp.population, 1000000), 0), 2) AS revenue_per_capita
    FROM state_metrics sm
    LEFT JOIN state_population sp ON sm.state = sp.state
)
SELECT 
    state,
    country,
    active_customers,
    new_customers_90d,
    total_revenue,
    total_orders,
    avg_order_value,
    avg_customer_ltv,
    unique_products_sold,
    categories_purchased,
    estimated_population,
    market_penetration_pct,
    revenue_per_capita,
    -- Market potential
    ROUND((estimated_population * 0.10) - active_customers, 0) AS addressable_market_customers,
    ROUND(((estimated_population * 0.10) - active_customers) * avg_customer_ltv, 2) AS market_opportunity_value,
    -- Growth metrics
    ROUND(new_customers_90d * 4, 0) AS projected_annual_new_customers,
    ROUND((new_customers_90d * 4) * avg_customer_ltv, 2) AS projected_annual_new_revenue,
    -- Market maturity
    CASE 
        WHEN market_penetration_pct >= 1.0 THEN 'Mature Market'
        WHEN market_penetration_pct >= 0.5 THEN 'Growing Market'
        WHEN market_penetration_pct >= 0.1 THEN 'Developing Market'
        ELSE 'Untapped Market'
    END AS market_maturity,
    -- Strategic priority
    CASE 
        WHEN market_penetration_pct < 0.1 AND estimated_population > 5000000 
            THEN 'High Priority - Large untapped market'
        WHEN market_penetration_pct < 0.5 AND total_revenue > 50000 
            THEN 'Medium Priority - Growth opportunity'
        WHEN market_penetration_pct >= 1.0 
            THEN 'Low Priority - Mature market'
        ELSE 'Monitor'
    END AS expansion_priority
FROM penetration_metrics
ORDER BY total_revenue DESC;

-- ========================================
-- 3. REGIONAL GROWTH TRENDS
-- Month-over-month and year-over-year trends
-- ========================================

WITH monthly_regional_sales AS (
    SELECT 
        c.state,
        c.city,
        DATE_FORMAT(o.order_date, '%Y-%m') AS sales_month,
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS customers,
        SUM(o.total_amount) AS revenue,
        AVG(o.total_amount) AS avg_order_value
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY c.state, c.city, DATE_FORMAT(o.order_date, '%Y-%m')
),
trend_analysis AS (
    SELECT 
        state,
        city,
        sales_month,
        orders,
        customers,
        revenue,
        avg_order_value,
        -- Month-over-month comparison
        LAG(revenue, 1) OVER (PARTITION BY state, city ORDER BY sales_month) AS prev_month_revenue,
        LAG(orders, 1) OVER (PARTITION BY state, city ORDER BY sales_month) AS prev_month_orders,
        -- Year-over-year comparison
        LAG(revenue, 12) OVER (PARTITION BY state, city ORDER BY sales_month) AS yoy_revenue,
        LAG(orders, 12) OVER (PARTITION BY state, city ORDER BY sales_month) AS yoy_orders,
        -- Moving averages
        AVG(revenue) OVER (
            PARTITION BY state, city 
            ORDER BY sales_month 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS revenue_3mo_avg,
        AVG(revenue) OVER (
            PARTITION BY state, city 
            ORDER BY sales_month 
            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
        ) AS revenue_6mo_avg
    FROM monthly_regional_sales
)
SELECT 
    state,
    city,
    sales_month,
    orders,
    customers,
    ROUND(revenue, 2) AS monthly_revenue,
    ROUND(avg_order_value, 2) AS avg_order_value,
    -- Trend metrics
    ROUND(revenue_3mo_avg, 2) AS revenue_3mo_moving_avg,
    ROUND(revenue_6mo_avg, 2) AS revenue_6mo_moving_avg,
    -- Month-over-month growth
    ROUND((revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0) * 100, 2) AS mom_revenue_growth_pct,
    ROUND((orders - prev_month_orders) / NULLIF(prev_month_orders, 0) * 100, 2) AS mom_order_growth_pct,
    -- Year-over-year growth
    ROUND((revenue - yoy_revenue) / NULLIF(yoy_revenue, 0) * 100, 2) AS yoy_revenue_growth_pct,
    ROUND((orders - yoy_orders) / NULLIF(yoy_orders, 0) * 100, 2) AS yoy_order_growth_pct,
    -- Trend classification
    CASE 
        WHEN (revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0) > 0.20 THEN 'Surging'
        WHEN (revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0) > 0.10 THEN 'Strong Growth'
        WHEN (revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0) > 0 THEN 'Growing'
        WHEN (revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0) > -0.10 THEN 'Flat'
        WHEN (revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0) > -0.20 THEN 'Declining'
        ELSE 'Contracting'
    END AS trend_status,
    -- Volatility indicator
    CASE 
        WHEN ABS(revenue - revenue_3mo_avg) / NULLIF(revenue_3mo_avg, 0) > 0.30 THEN 'High Volatility'
        WHEN ABS(revenue - revenue_3mo_avg) / NULLIF(revenue_3mo_avg, 0) > 0.15 THEN 'Moderate Volatility'
        ELSE 'Stable'
    END AS volatility_level
FROM trend_analysis
WHERE sales_month >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 12 MONTH), '%Y-%m')
ORDER BY state, city, sales_month DESC;

-- ========================================
-- 4. REGIONAL PRODUCT PREFERENCES
-- What products sell best in each region
-- ========================================

WITH regional_product_sales AS (
    SELECT 
        c.state,
        pc.category_name,
        p.product_name,
        COUNT(DISTINCT o.order_id) AS orders,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS revenue,
        AVG(oi.unit_price) AS avg_selling_price,
        COUNT(DISTINCT o.customer_id) AS unique_customers
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY c.state, pc.category_name, p.product_name
),
state_totals AS (
    SELECT 
        state,
        SUM(revenue) AS total_state_revenue
    FROM regional_product_sales
    GROUP BY state
),
product_rankings AS (
    SELECT 
        rps.state,
        rps.category_name,
        rps.product_name,
        rps.orders,
        rps.units_sold,
        ROUND(rps.revenue, 2) AS revenue,
        ROUND(rps.avg_selling_price, 2) AS avg_price,
        rps.unique_customers,
        ROUND(rps.revenue * 100.0 / NULLIF(st.total_state_revenue, 0), 2) AS pct_of_state_revenue,
        ROW_NUMBER() OVER (PARTITION BY rps.state ORDER BY rps.revenue DESC) AS state_product_rank,
        ROW_NUMBER() OVER (PARTITION BY rps.state, rps.category_name ORDER BY rps.revenue DESC) AS category_rank
    FROM regional_product_sales rps
    JOIN state_totals st ON rps.state = st.state
)
SELECT 
    state,
    category_name,
    product_name,
    state_product_rank,
    category_rank,
    orders,
    units_sold,
    revenue,
    avg_price,
    unique_customers,
    pct_of_state_revenue,
    -- Penetration
    ROUND(unique_customers * 100.0 / (SELECT COUNT(DISTINCT customer_id) 
                                       FROM customers 
                                       WHERE state = pr.state), 2) AS customer_penetration_pct,
    -- Classification
    CASE 
        WHEN state_product_rank <= 5 THEN 'Top Product'
        WHEN state_product_rank <= 20 THEN 'Popular Product'
        ELSE 'Standard Product'
    END AS product_status
FROM product_rankings pr
WHERE state_product_rank <= 10  -- Top 10 products per state
ORDER BY state, state_product_rank;

-- ========================================
-- 5. REGIONAL CUSTOMER SEGMENTATION
-- Customer behavior patterns by region
-- ========================================

WITH regional_customer_metrics AS (
    SELECT 
        c.state,
        c.customer_id,
        COUNT(DISTINCT o.order_id) AS lifetime_orders,
        SUM(o.total_amount) AS lifetime_value,
        AVG(o.total_amount) AS avg_order_value,
        MIN(o.order_date) AS first_order_date,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) AS customer_lifespan_days,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order,
        -- Product diversity
        COUNT(DISTINCT oi.product_id) AS unique_products,
        COUNT(DISTINCT p.category_id) AS unique_categories
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    WHERE c.status = 'active'
    GROUP BY c.state, c.customer_id
),
customer_segments AS (
    SELECT 
        state,
        customer_id,
        lifetime_orders,
        lifetime_value,
        avg_order_value,
        days_since_last_order,
        customer_lifespan_days,
        unique_products,
        unique_categories,
        -- RFM-style segmentation
        CASE 
            WHEN days_since_last_order <= 30 THEN 5
            WHEN days_since_last_order <= 60 THEN 4
            WHEN days_since_last_order <= 90 THEN 3
            WHEN days_since_last_order <= 180 THEN 2
            ELSE 1
        END AS recency_score,
        CASE 
            WHEN lifetime_orders >= 10 THEN 5
            WHEN lifetime_orders >= 5 THEN 4
            WHEN lifetime_orders >= 3 THEN 3
            WHEN lifetime_orders >= 2 THEN 2
            ELSE 1
        END AS frequency_score,
        CASE 
            WHEN lifetime_value >= 2000 THEN 5
            WHEN lifetime_value >= 1000 THEN 4
            WHEN lifetime_value >= 500 THEN 3
            WHEN lifetime_value >= 250 THEN 2
            ELSE 1
        END AS monetary_score
    FROM regional_customer_metrics
    WHERE lifetime_orders > 0
),
regional_segmentation AS (
    SELECT 
        state,
        COUNT(DISTINCT customer_id) AS total_customers,
        -- Customer value tiers
        SUM(CASE WHEN monetary_score = 5 THEN 1 ELSE 0 END) AS vip_customers,
        SUM(CASE WHEN monetary_score >= 4 THEN 1 ELSE 0 END) AS high_value_customers,
        SUM(CASE WHEN monetary_score = 3 THEN 1 ELSE 0 END) AS medium_value_customers,
        SUM(CASE WHEN monetary_score <= 2 THEN 1 ELSE 0 END) AS low_value_customers,
        -- Engagement levels
        SUM(CASE WHEN recency_score >= 4 THEN 1 ELSE 0 END) AS active_customers,
        SUM(CASE WHEN recency_score = 3 THEN 1 ELSE 0 END) AS at_risk_customers,
        SUM(CASE WHEN recency_score <= 2 THEN 1 ELSE 0 END) AS inactive_customers,
        -- Purchase behavior
        SUM(CASE WHEN frequency_score >= 4 THEN 1 ELSE 0 END) AS loyal_customers,
        SUM(CASE WHEN frequency_score <= 2 THEN 1 ELSE 0 END) AS occasional_buyers,
        -- Averages
        AVG(lifetime_value) AS avg_customer_ltv,
        AVG(lifetime_orders) AS avg_orders_per_customer,
        AVG(avg_order_value) AS avg_order_value
    FROM customer_segments
    GROUP BY state
)
SELECT 
    state,
    total_customers,
    vip_customers,
    high_value_customers,
    medium_value_customers,
    low_value_customers,
    active_customers,
    at_risk_customers,
    inactive_customers,
    loyal_customers,
    occasional_buyers,
    -- Percentages
    ROUND(vip_customers * 100.0 / NULLIF(total_customers, 0), 2) AS vip_pct,
    ROUND(high_value_customers * 100.0 / NULLIF(total_customers, 0), 2) AS high_value_pct,
    ROUND(active_customers * 100.0 / NULLIF(total_customers, 0), 2) AS active_pct,
    ROUND(at_risk_customers * 100.0 / NULLIF(total_customers, 0), 2) AS at_risk_pct,
    ROUND(loyal_customers * 100.0 / NULLIF(total_customers, 0), 2) AS loyal_pct,
    -- Financial metrics
    ROUND(avg_customer_ltv, 2) AS avg_customer_ltv,
    ROUND(avg_orders_per_customer, 2) AS avg_orders_per_customer,
    ROUND(avg_order_value, 2) AS avg_order_value,
    -- Health score
    ROUND(
        (active_customers * 100.0 / NULLIF(total_customers, 0)) * 0.35 +
        (high_value_customers * 100.0 / NULLIF(total_customers, 0)) * 0.30 +
        (loyal_customers * 100.0 / NULLIF(total_customers, 0)) * 0.20 +
        (vip_customers * 100.0 / NULLIF(total_customers, 0)) * 0.15,
        0
    ) AS regional_health_score,
    -- Strategic focus
    CASE 
        WHEN at_risk_customers * 100.0 / NULLIF(total_customers, 0) > 30 
            THEN 'Retention Focus'
        WHEN vip_customers * 100.0 / NULLIF(total_customers, 0) > 15 
            THEN 'VIP Nurturing'
        WHEN loyal_customers * 100.0 / NULLIF(total_customers, 0) < 20 
            THEN 'Loyalty Building'
        ELSE 'Balanced Growth'
    END AS strategic_focus
FROM regional_segmentation
ORDER BY total_customers DESC;

-- ========================================
-- 6. SHIPPING PERFORMANCE BY REGION
-- Delivery times and costs by geography
-- ========================================

WITH regional_shipping AS (
    SELECT 
        c.state,
        c.city,
        o.order_id,
        o.order_date,
        o.status,
        o.shipping_cost,
        o.total_amount,
        -- Calculate delivery time (simplified using status update)
        CASE 
            WHEN o.status = 'delivered' THEN DATEDIFF(o.updated_at, o.order_date)
            ELSE NULL
        END AS delivery_days
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND o.payment_status = 'paid'
),
shipping_metrics AS (
    SELECT 
        state,
        city,
        COUNT(DISTINCT order_id) AS total_orders,
        AVG(shipping_cost) AS avg_shipping_cost,
        AVG(delivery_days) AS avg_delivery_days,
        MIN(delivery_days) AS fastest_delivery,
        MAX(delivery_days) AS slowest_delivery,
        STDDEV(delivery_days) AS delivery_consistency,
        SUM(shipping_cost) AS total_shipping_revenue,
        AVG(shipping_cost / NULLIF(total_amount, 0) * 100) AS shipping_as_pct_of_order
    FROM regional_shipping
    GROUP BY state, city
)
SELECT 
    state,
    city,
    total_orders,
    ROUND(avg_shipping_cost, 2) AS avg_shipping_cost,
    ROUND(avg_delivery_days, 1) AS avg_delivery_days,
    fastest_delivery AS best_delivery_days,
    slowest_delivery AS worst_delivery_days,
    ROUND(delivery_consistency, 1) AS delivery_variability,
    ROUND(total_shipping_revenue, 2) AS total_shipping_revenue,
    ROUND(shipping_as_pct_of_order, 2) AS shipping_pct_of_order_value,
    -- Performance classification
    CASE 
        WHEN avg_delivery_days <= 3 THEN 'Excellent'
        WHEN avg_delivery_days <= 5 THEN 'Good'
        WHEN avg_delivery_days <= 7 THEN 'Acceptable'
        ELSE 'Needs Improvement'
    END AS delivery_performance,
    -- Cost efficiency
    CASE 
        WHEN avg_shipping_cost / NULLIF(total_shipping_revenue / total_orders, 0) < 0.08 THEN 'Cost Efficient'
        WHEN avg_shipping_cost / NULLIF(total_shipping_revenue / total_orders, 0) < 0.12 THEN 'Average Cost'
        ELSE 'High Cost'
    END AS shipping_cost_tier,
    -- Recommendations
    CASE 
        WHEN avg_delivery_days > 7 THEN 'Optimize logistics partner'
        WHEN delivery_consistency > 3 THEN 'Improve delivery consistency'
        WHEN avg_shipping_cost > 15 THEN 'Negotiate better shipping rates'
        ELSE 'Maintain current operations'
    END AS shipping_recommendation
FROM shipping_metrics
WHERE total_orders >= 10  -- Minimum volume for meaningful metrics
ORDER BY total_orders DESC;

-- ========================================
-- 7. REGIONAL COMPETITIVE ANALYSIS
-- Market share and competitive positioning
-- ========================================

WITH regional_market_data AS (
    SELECT 
        c.state,
        COUNT(DISTINCT c.customer_id) AS our_customers,
        SUM(o.total_amount) AS our_revenue,
        COUNT(DISTINCT o.order_id) AS our_orders,
        AVG(o.total_amount) AS our_avg_order_value,
        -- Estimated market size (simplified - would use external data)
        COUNT(DISTINCT c.customer_id) * 10 AS estimated_total_market_customers,
        SUM(o.total_amount) * 5 AS estimated_total_market_revenue
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE c.status = 'active'
    GROUP BY c.state
)
SELECT 
    state,
    our_customers,
    ROUND(our_revenue, 2) AS our_revenue,
    our_orders,
    ROUND(our_avg_order_value, 2) AS our_avg_order_value,
    estimated_total_market_customers,
    ROUND(estimated_total_market_revenue, 2) AS estimated_market_revenue,
    -- Market share
    ROUND(our_customers * 100.0 / NULLIF(estimated_total_market_customers, 0), 2) AS estimated_customer_share_pct,
    ROUND(our_revenue * 100.0 / NULLIF(estimated_total_market_revenue, 0), 2) AS estimated_revenue_share_pct,
    -- Opportunity sizing
    estimated_total_market_customers - our_customers AS addressable_customers,
    ROUND(estimated_total_market_revenue - our_revenue, 2) AS addressable_revenue,
    -- Competitive position
    CASE 
        WHEN our_customers * 100.0 / NULLIF(estimated_total_market_customers, 0) >= 20 THEN 'Market Leader'
        WHEN our_customers * 100.0 / NULLIF(estimated_total_market_customers, 0) >= 10 THEN 'Strong Player'
        WHEN our_customers * 100.0 / NULLIF(estimated_total_market_customers, 0) >= 5 THEN 'Challenger'
        ELSE 'Niche Player'
    END AS competitive_position,
    -- Growth strategy
    CASE 
        WHEN our_customers * 100.0 / NULLIF(estimated_total_market_customers, 0) < 5 
             AND estimated_total_market_revenue > 500000
            THEN 'Aggressive Expansion - Large untapped market'
        WHEN our_customers * 100.0 / NULLIF(estimated_total_market_customers, 0) < 10
            THEN 'Growth Focus - Significant opportunity'
        WHEN our_customers * 100.0 / NULLIF(estimated_total_market_customers, 0) < 20
            THEN 'Market Share Defense - Consolidate position'
        ELSE 'Market Leadership - Maintain dominance'
    END AS recommended_strategy
FROM regional_market_data
ORDER BY our_revenue DESC;

-- ========================================
-- 8. REGIONAL PROFITABILITY ANALYSIS
-- Profit margins and operational efficiency by region
-- ========================================

WITH regional_costs AS (
    SELECT 
        c.state,
        c.city,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS gross_revenue,
        SUM(o.shipping_cost) AS total_shipping_costs,
        SUM(o.tax_amount) AS total_taxes,
        SUM(oi.quantity * p.cost) AS cogs,
        SUM(o.total_amount - o.shipping_cost - o.tax_amount) AS net_revenue,
        COUNT(DISTINCT c.customer_id) AS total_customers
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY c.state, c.city
),
profitability_calc AS (
    SELECT 
        state,
        city,
        total_orders,
        total_customers,
        ROUND(gross_revenue, 2) AS gross_revenue,
        ROUND(net_revenue, 2) AS net_revenue,
        ROUND(total_shipping_costs, 2) AS shipping_costs,
        ROUND(total_taxes, 2) AS taxes_collected,
        ROUND(cogs, 2) AS cogs,
        -- Gross profit calculation
        ROUND(gross_revenue - cogs - total_shipping_costs, 2) AS gross_profit,
        ROUND((gross_revenue - cogs - total_shipping_costs) / NULLIF(gross_revenue, 0) * 100, 2) AS gross_margin_pct,
        -- Per-order metrics
        ROUND(gross_revenue / NULLIF(total_orders, 0), 2) AS revenue_per_order,
        ROUND((gross_revenue - cogs - total_shipping_costs) / NULLIF(total_orders, 0), 2) AS profit_per_order,
        -- Per-customer metrics
        ROUND(gross_revenue / NULLIF(total_customers, 0), 2) AS revenue_per_customer,
        ROUND((gross_revenue - cogs - total_shipping_costs) / NULLIF(total_customers, 0), 2) AS profit_per_customer
    FROM regional_costs
)
SELECT 
    state,
    city,
    total_orders,
    total_customers,
    gross_revenue,
    net_revenue,
    cogs,
    shipping_costs,
    gross_profit,
    gross_margin_pct,
    revenue_per_order,
    profit_per_order,
    revenue_per_customer,
    profit_per_customer,
    -- Profitability tier
    CASE 
        WHEN gross_margin_pct >= 50 THEN 'Highly Profitable'
        WHEN gross_margin_pct >= 40 THEN 'Very Profitable'
        WHEN gross_margin_pct >= 30 THEN 'Profitable'
        WHEN gross_margin_pct >= 20 THEN 'Marginally Profitable'
        ELSE 'Low Margin'
    END AS profitability_tier,
    -- Efficiency score (0-100)
    ROUND(
        LEAST(100,
            (gross_margin_pct * 1.5) * 0.50 +  -- Margin (50%)
            (profit_per_customer / 10) * 0.30 +  -- Customer profitability (30%)
            (profit_per_order / 5) * 0.20  -- Order efficiency (20%)
        ),
        0
    ) AS regional_efficiency_score,
    -- Strategic recommendation
    CASE 
        WHEN gross_margin_pct < 25 AND shipping_costs / NULLIF(gross_revenue, 0) > 0.15
            THEN 'Optimize shipping costs'
        WHEN gross_margin_pct < 30 AND cogs / NULLIF(gross_revenue, 0) > 0.60
            THEN 'Review product mix - high COGS'
        WHEN gross_margin_pct >= 40 AND total_orders < 50
            THEN 'Scale operations - high margin opportunity'
        WHEN profit_per_customer < 50
            THEN 'Increase customer lifetime value'
        ELSE 'Maintain current operations'
    END AS profitability_action
FROM profitability_calc
WHERE total_orders >= 5
ORDER BY gross_profit DESC;

-- ========================================
-- 9. GEOGRAPHIC EXPANSION OPPORTUNITY SCORING
-- Data-driven expansion recommendations
-- ========================================

WITH expansion_metrics AS (
    SELECT 
        c.state,
        c.city,
        COUNT(DISTINCT c.customer_id) AS current_customers,
        SUM(o.total_amount) AS current_revenue,
        AVG(o.total_amount) AS avg_order_value,
        COUNT(DISTINCT o.order_id) AS total_orders,
        -- Growth indicators
        COUNT(DISTINCT CASE 
            WHEN c.created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN c.customer_id 
        END) AS new_customers_90d,
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            THEN o.order_id 
        END) AS orders_last_30d,
        -- Customer quality
        AVG((SELECT COUNT(*) FROM orders o2 
             WHERE o2.customer_id = c.customer_id 
               AND o2.status IN ('delivered', 'shipped', 'processing'))) AS avg_orders_per_customer,
        AVG((SELECT SUM(total_amount) FROM orders o3 
             WHERE o3.customer_id = c.customer_id 
               AND o3.status IN ('delivered', 'shipped', 'processing'))) AS avg_customer_ltv
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE c.status = 'active'
    GROUP BY c.state, c.city
),
opportunity_scoring AS (
    SELECT 
        state,
        city,
        current_customers,
        ROUND(current_revenue, 2) AS current_revenue,
        ROUND(avg_order_value, 2) AS avg_order_value,
        total_orders,
        new_customers_90d,
        orders_last_30d,
        ROUND(avg_orders_per_customer, 2) AS avg_orders_per_customer,
        ROUND(avg_customer_ltv, 2) AS avg_customer_ltv,
        -- Scoring components (0-25 points each)
        -- 1. Growth momentum (25 points)
        LEAST(25, (new_customers_90d * 4) / NULLIF(current_customers, 0) * 100 * 0.25) AS growth_score,
        -- 2. Market size potential (25 points)
        LEAST(25, current_revenue / 1000 * 0.25) AS market_size_score,
        -- 3. Customer value (25 points)
        LEAST(25, avg_customer_ltv / 40 * 0.25) AS customer_value_score,
        -- 4. Activity level (25 points)
        LEAST(25, (orders_last_30d * 12) / NULLIF(total_orders, 0) * 100 * 0.25) AS activity_score
    FROM expansion_metrics
)
SELECT 
    state,
    city,
    current_customers,
    current_revenue,
    avg_order_value,
    avg_customer_ltv,
    new_customers_90d,
    -- Opportunity score
    ROUND(growth_score + market_size_score + customer_value_score + activity_score, 0) AS expansion_opportunity_score,
    ROUND(growth_score, 1) AS growth_component,
    ROUND(market_size_score, 1) AS size_component,
    ROUND(customer_value_score, 1) AS value_component,
    ROUND(activity_score, 1) AS activity_component,
    -- Classification
    CASE 
        WHEN (growth_score + market_size_score + customer_value_score + activity_score) >= 80
            THEN 'Tier 1 - High Priority'
        WHEN (growth_score + market_size_score + customer_value_score + activity_score) >= 60
            THEN 'Tier 2 - Strong Opportunity'
        WHEN (growth_score + market_size_score + customer_value_score + activity_score) >= 40
            THEN 'Tier 3 - Moderate Opportunity'
        WHEN (growth_score + market_size_score + customer_value_score + activity_score) >= 20
            THEN 'Tier 4 - Watch List'
        ELSE 'Tier 5 - Low Priority'
    END AS opportunity_tier,
    -- Investment recommendation
    CASE 
        WHEN (growth_score + market_size_score + customer_value_score + activity_score) >= 80
            THEN 'Major investment: Local warehouse, dedicated marketing'
        WHEN (growth_score + market_size_score + customer_value_score + activity_score) >= 60
            THEN 'Significant investment: Targeted campaigns, partnerships'
        WHEN (growth_score + market_size_score + customer_value_score + activity_score) >= 40
            THEN 'Moderate investment: Digital marketing, test promotions'
        WHEN (growth_score + market_size_score + customer_value_score + activity_score) >= 20
            THEN 'Light investment: Monitor and optimize existing efforts'
        ELSE 'Minimal investment: Maintain baseline presence'
    END AS investment_recommendation
FROM opportunity_scoring
ORDER BY expansion_opportunity_score DESC, current_revenue DESC;

-- ========================================
-- 10. REGIONAL PERFORMANCE DASHBOARD
-- Executive summary by region
-- ========================================

WITH regional_summary AS (
    SELECT 
        c.state,
        c.country,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        SUM(o.total_amount - o.shipping_cost - o.tax_amount) AS net_revenue,
        AVG(o.total_amount) AS avg_order_value,
        -- Time-based metrics
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            THEN o.total_amount ELSE 0 
        END) AS revenue_last_30d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            AND o.order_date < DATE_SUB(CURDATE(), INTERVAL 60 DAY)
            THEN o.total_amount ELSE 0 
        END) AS revenue_prev_30d,
        -- Customer metrics
        COUNT(DISTINCT CASE 
            WHEN (SELECT COUNT(*) FROM orders o2 
                  WHERE o2.customer_id = c.customer_id 
                    AND o2.status IN ('delivered', 'shipped', 'processing')) > 1 
            THEN c.customer_id 
        END) AS repeat_customers,
        -- Product metrics
        COUNT(DISTINCT oi.product_id) AS unique_products_sold,
        COUNT(DISTINCT p.category_id) AS categories_sold
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    WHERE c.status = 'active'
    GROUP BY c.state, c.country
)
SELECT 
    state,
    country,
    total_customers,
    total_orders,
    ROUND(total_revenue, 2) AS total_revenue_12m,
    ROUND(net_revenue, 2) AS net_revenue_12m,
    ROUND(avg_order_value, 2) AS avg_order_value,
    ROUND(revenue_last_30d, 2) AS revenue_last_30d,
    -- Growth metrics
    ROUND((revenue_last_30d - revenue_prev_30d) / NULLIF(revenue_prev_30d, 0) * 100, 2) AS mom_revenue_growth_pct,
    ROUND(revenue_last_30d * 12, 2) AS annualized_revenue_projection,
    -- Customer metrics
    repeat_customers,
    ROUND(repeat_customers * 100.0 / NULLIF(total_customers, 0), 2) AS repeat_customer_rate_pct,
    ROUND(total_orders * 1.0 / NULLIF(total_customers, 0), 2) AS orders_per_customer,
    ROUND(total_revenue / NULLIF(total_customers, 0), 2) AS revenue_per_customer,
    -- Product diversity
    unique_products_sold,
    categories_sold,
    -- Market share (relative to total)
    ROUND(total_revenue * 100.0 / (SELECT SUM(total_revenue) FROM regional_summary), 2) AS pct_of_total_revenue,
    -- Performance ranking
    RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
    RANK() OVER (ORDER BY total_customers DESC) AS customer_rank,
    -- Overall health score (0-100)
    ROUND(
        LEAST(100,
            -- Revenue contribution (30%)
            (total_revenue / 10000) * 0.30 +
            -- Customer engagement (25%)
            (repeat_customers * 100.0 / NULLIF(total_customers, 0)) * 0.25 +
            -- Growth momentum (25%)
            GREATEST(0, ((revenue_last_30d - revenue_prev_30d) / NULLIF(revenue_prev_30d, 0) * 100)) * 0.25 +
            -- Market penetration (20%)
            (orders_per_customer * 10) * 0.20
        ),
        0
    ) AS regional_health_score,
    -- Status indicator
    CASE 
        WHEN (revenue_last_30d - revenue_prev_30d) / NULLIF(revenue_prev_30d, 0) > 0.20 
            THEN '🟢 High Growth'
        WHEN (revenue_last_30d - revenue_prev_30d) / NULLIF(revenue_prev_30d, 0) > 0.05 
            THEN '🟢 Growing'
        WHEN (revenue_last_30d - revenue_prev_30d) / NULLIF(revenue_prev_30d, 0) > -0.05 
            THEN '🟡 Stable'
        WHEN (revenue_last_30d - revenue_prev_30d) / NULLIF(revenue_prev_30d, 0) > -0.20 
            THEN '🟡 Declining'
        ELSE '🔴 Contracting'
    END AS growth_status,
    -- Strategic priority
    CASE 
        WHEN total_revenue >= 100000 AND (revenue_last_30d - revenue_prev_30d) / NULLIF(revenue_prev_30d, 0) > 0.10
            THEN 'Strategic Market - Scale aggressively'
        WHEN total_revenue >= 50000
            THEN 'Key Market - Invest in growth'
        WHEN total_revenue >= 10000 AND (revenue_last_30d - revenue_prev_30d) / NULLIF(revenue_prev_30d, 0) > 0.15
            THEN 'Emerging Market - Accelerate development'
        WHEN total_revenue < 10000 AND total_customers < 50
            THEN 'New Market - Test and learn'
        ELSE 'Established Market - Optimize operations'
    END AS strategic_classification
FROM regional_summary
ORDER BY total_revenue DESC;

-- ========================================
-- End of Geographic Analysis
-- ========================================