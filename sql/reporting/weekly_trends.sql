-- ========================================
-- WEEKLY TRENDS ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Week-over-Week Analysis, Performance, Trend Indicators
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- REPORT HEADER
-- ========================================

SELECT 
    '═══════════════════════════════════════════════════════════' AS separator,
    'WEEKLY TRENDS ANALYSIS REPORT' AS title,
    CONCAT('Week of ', DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY), '%B %d, %Y')) AS current_week,
    DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s') AS generated_at,
    '═══════════════════════════════════════════════════════════' AS separator2;

-- ========================================
-- 1. WEEKLY REVENUE TRENDS (Last 12 Weeks)
-- Comprehensive week-over-week revenue analysis
-- ========================================

WITH weekly_metrics AS (
    SELECT 
        YEARWEEK(order_date, 1) AS year_week,
        WEEK(order_date, 1) AS week_num,
        YEAR(order_date) AS year,
        DATE_SUB(order_date, INTERVAL WEEKDAY(order_date) DAY) AS week_start_date,
        DATE_FORMAT(DATE_SUB(order_date, INTERVAL WEEKDAY(order_date) DAY), '%b %d') AS week_label,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        SUM(CASE WHEN payment_status = 'paid' THEN total_amount ELSE 0 END) AS revenue,
        AVG(CASE WHEN payment_status = 'paid' THEN total_amount END) AS avg_order_value,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered_orders,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK)
    GROUP BY 
        YEARWEEK(order_date, 1),
        WEEK(order_date, 1),
        YEAR(order_date),
        DATE_SUB(order_date, INTERVAL WEEKDAY(order_date) DAY),
        week_label
)
SELECT 
    year_week,
    CONCAT('Week ', week_num, ', ', year) AS week_period,
    week_label,
    CONCAT('$', FORMAT(revenue, 2)) AS revenue,
    FORMAT(orders, 0) AS orders,
    FORMAT(customers, 0) AS customers,
    CONCAT('$', FORMAT(avg_order_value, 2)) AS avg_order_value,
    -- Week-over-week changes
    CONCAT(
        CASE 
            WHEN LAG(revenue) OVER (ORDER BY year_week) IS NULL THEN '0'
            ELSE FORMAT(ROUND((revenue - LAG(revenue) OVER (ORDER BY year_week)) / 
                 NULLIF(LAG(revenue) OVER (ORDER BY year_week), 0) * 100, 2), 2)
        END, '%'
    ) AS revenue_wow_change,
    CONCAT(
        CASE 
            WHEN LAG(orders) OVER (ORDER BY year_week) IS NULL THEN '0'
            ELSE FORMAT(ROUND((orders - LAG(orders) OVER (ORDER BY year_week)) / 
                 NULLIF(LAG(orders) OVER (ORDER BY year_week), 0) * 100, 2), 2)
        END, '%'
    ) AS orders_wow_change,
    -- 4-week moving average
    CONCAT('$', FORMAT(
        AVG(revenue) OVER (ORDER BY year_week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 2
    )) AS revenue_4wk_avg,
    -- Trend indicators
    CASE 
        WHEN revenue > LAG(revenue, 1) OVER (ORDER BY year_week) 
            AND LAG(revenue, 1) OVER (ORDER BY year_week) > LAG(revenue, 2) OVER (ORDER BY year_week)
            THEN '🚀 Accelerating'
        WHEN revenue < LAG(revenue, 1) OVER (ORDER BY year_week) 
            AND LAG(revenue, 1) OVER (ORDER BY year_week) < LAG(revenue, 2) OVER (ORDER BY year_week)
            THEN ''TRENDING_DOWN' Declining'
        WHEN revenue > LAG(revenue) OVER (ORDER BY year_week)
            THEN ''TRENDING_UP' Growing'
        WHEN revenue < LAG(revenue) OVER (ORDER BY year_week)
            THEN '↘️ Slowing'
        ELSE '➡️ Stable'
    END AS trend,
    ROUND(delivered_orders * 100.0 / NULLIF(orders, 0), 2) AS fulfillment_rate,
    ROUND(cancelled_orders * 100.0 / NULLIF(orders, 0), 2) AS cancellation_rate
FROM weekly_metrics
ORDER BY year_week DESC;

-- ========================================
-- 2. CUSTOMER ACQUISITION TRENDS
-- Weekly new customer growth analysis
-- ========================================

WITH weekly_customers AS (
    SELECT 
        YEARWEEK(created_at, 1) AS year_week,
        DATE_SUB(created_at, INTERVAL WEEKDAY(created_at) DAY) AS week_start_date,
        DATE_FORMAT(DATE_SUB(created_at, INTERVAL WEEKDAY(created_at) DAY), '%b %d, %Y') AS week_label,
        COUNT(DISTINCT customer_id) AS new_customers,
        COUNT(DISTINCT CASE WHEN status = 'active' THEN customer_id END) AS active_customers
    FROM customers
    WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK)
    GROUP BY 
        YEARWEEK(created_at, 1),
        week_start_date,
        week_label
),
weekly_orders AS (
    SELECT 
        YEARWEEK(o.order_date, 1) AS year_week,
        COUNT(DISTINCT CASE 
            WHEN (SELECT COUNT(*) FROM orders o2 
                  WHERE o2.customer_id = o.customer_id 
                  AND o2.order_date < o.order_date) = 0 
            THEN o.customer_id 
        END) AS first_time_buyers,
        COUNT(DISTINCT CASE 
            WHEN (SELECT COUNT(*) FROM orders o2 
                  WHERE o2.customer_id = o.customer_id 
                  AND o2.order_date < o.order_date) > 0 
            THEN o.customer_id 
        END) AS repeat_buyers,
        SUM(CASE 
            WHEN (SELECT COUNT(*) FROM orders o2 
                  WHERE o2.customer_id = o.customer_id 
                  AND o2.order_date < o.order_date) = 0 
            AND payment_status = 'paid'
            THEN total_amount 
            ELSE 0 
        END) AS first_time_revenue,
        SUM(CASE 
            WHEN (SELECT COUNT(*) FROM orders o2 
                  WHERE o2.customer_id = o.customer_id 
                  AND o2.order_date < o.order_date) > 0 
            AND payment_status = 'paid'
            THEN total_amount 
            ELSE 0 
        END) AS repeat_revenue
    FROM orders o
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK)
    GROUP BY YEARWEEK(o.order_date, 1)
)
SELECT 
    wc.week_label,
    FORMAT(wc.new_customers, 0) AS new_customers,
    FORMAT(wo.first_time_buyers, 0) AS first_time_buyers,
    FORMAT(wo.repeat_buyers, 0) AS repeat_buyers,
    CONCAT('$', FORMAT(wo.first_time_revenue, 2)) AS first_time_revenue,
    CONCAT('$', FORMAT(wo.repeat_revenue, 2)) AS repeat_revenue,
    CONCAT('$', FORMAT(
        wo.first_time_revenue / NULLIF(wo.first_time_buyers, 0), 2
    )) AS avg_first_purchase,
    CONCAT('$', FORMAT(
        wo.repeat_revenue / NULLIF(wo.repeat_buyers, 0), 2
    )) AS avg_repeat_purchase,
    -- Week-over-week growth
    CONCAT(
        FORMAT(ROUND((wc.new_customers - LAG(wc.new_customers) OVER (ORDER BY wc.year_week)) / 
               NULLIF(LAG(wc.new_customers) OVER (ORDER BY wc.year_week), 0) * 100, 2), 2), '%'
    ) AS new_customers_wow,
    -- Repeat buyer rate
    ROUND(wo.repeat_buyers * 100.0 / NULLIF(wo.first_time_buyers + wo.repeat_buyers, 0), 2) AS repeat_rate_pct,
    CASE 
        WHEN wc.new_customers > LAG(wc.new_customers) OVER (ORDER BY wc.year_week) THEN ''TRENDING_UP' Growing'
        WHEN wc.new_customers < LAG(wc.new_customers) OVER (ORDER BY wc.year_week) THEN ''TRENDING_DOWN' Declining'
        ELSE '➡️ Stable'
    END AS trend
FROM weekly_customers wc
LEFT JOIN weekly_orders wo ON wc.year_week = wo.year_week
ORDER BY wc.year_week DESC;

-- ========================================
-- 3. PRODUCT CATEGORY TRENDS
-- Weekly performance by category
-- ========================================

WITH weekly_category_sales AS (
    SELECT 
        YEARWEEK(o.order_date, 1) AS year_week,
        DATE_FORMAT(DATE_SUB(o.order_date, INTERVAL WEEKDAY(o.order_date) DAY), '%b %d') AS week_label,
        pc.category_name,
        COUNT(DISTINCT o.order_id) AS orders,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS revenue,
        AVG(oi.unit_price) AS avg_price
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
        AND o.payment_status = 'paid'
    GROUP BY 
        YEARWEEK(o.order_date, 1),
        week_label,
        pc.category_id,
        pc.category_name
)
SELECT 
    week_label,
    category_name,
    CONCAT('$', FORMAT(revenue, 2)) AS revenue,
    FORMAT(units_sold, 0) AS units_sold,
    FORMAT(orders, 0) AS orders,
    CONCAT('$', FORMAT(avg_price, 2)) AS avg_price,
    -- Calculate share of week's total
    ROUND(revenue * 100.0 / SUM(revenue) OVER (PARTITION BY year_week), 2) AS revenue_share_pct,
    -- Week-over-week change
    CONCAT(
        FORMAT(ROUND((revenue - LAG(revenue) OVER (PARTITION BY category_name ORDER BY year_week)) / 
               NULLIF(LAG(revenue) OVER (PARTITION BY category_name ORDER BY year_week), 0) * 100, 2), 2), '%'
    ) AS wow_change,
    -- Trend indicator
    CASE 
        WHEN revenue > LAG(revenue) OVER (PARTITION BY category_name ORDER BY year_week) THEN ''TRENDING_UP''
        WHEN revenue < LAG(revenue) OVER (PARTITION BY category_name ORDER BY year_week) THEN ''TRENDING_DOWN''
        ELSE '➡️'
    END AS trend
FROM weekly_category_sales
ORDER BY year_week DESC, revenue DESC;

-- ========================================
-- 4. TOP PRODUCTS TRENDING UP
-- Products gaining momentum week-over-week
-- ========================================

WITH weekly_product_sales AS (
    SELECT 
        YEARWEEK(o.order_date, 1) AS year_week,
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 4 WEEK)
        AND o.payment_status = 'paid'
    GROUP BY 
        YEARWEEK(o.order_date, 1),
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name
),
current_vs_previous AS (
    SELECT 
        product_id,
        product_name,
        sku,
        category_name,
        SUM(CASE WHEN year_week = YEARWEEK(CURDATE(), 1) THEN revenue ELSE 0 END) AS current_week_revenue,
        SUM(CASE WHEN year_week = YEARWEEK(DATE_SUB(CURDATE(), INTERVAL 1 WEEK), 1) THEN revenue ELSE 0 END) AS previous_week_revenue,
        SUM(CASE WHEN year_week = YEARWEEK(CURDATE(), 1) THEN units_sold ELSE 0 END) AS current_week_units,
        SUM(CASE WHEN year_week = YEARWEEK(DATE_SUB(CURDATE(), INTERVAL 1 WEEK), 1) THEN units_sold ELSE 0 END) AS previous_week_units
    FROM weekly_product_sales
    GROUP BY product_id, product_name, sku, category_name
    HAVING current_week_revenue > 0 AND previous_week_revenue > 0
)
SELECT 
    product_name,
    sku,
    category_name,
    CONCAT('$', FORMAT(current_week_revenue, 2)) AS this_week_revenue,
    CONCAT('$', FORMAT(previous_week_revenue, 2)) AS last_week_revenue,
    FORMAT(current_week_units, 0) AS this_week_units,
    FORMAT(previous_week_units, 0) AS last_week_units,
    CONCAT(
        FORMAT(ROUND((current_week_revenue - previous_week_revenue) / 
               NULLIF(previous_week_revenue, 0) * 100, 2), 2), '%'
    ) AS revenue_growth,
    CONCAT(
        FORMAT(ROUND((current_week_units - previous_week_units) / 
               NULLIF(previous_week_units, 0) * 100, 2), 2), '%'
    ) AS unit_growth,
    '🔥' AS status
FROM current_vs_previous
WHERE (current_week_revenue - previous_week_revenue) / NULLIF(previous_week_revenue, 0) >= 0.20
ORDER BY (current_week_revenue - previous_week_revenue) / NULLIF(previous_week_revenue, 0) DESC
LIMIT 15;

-- ========================================
-- 5. DECLINING PRODUCTS ALERT
-- Products losing momentum
-- ========================================

WITH weekly_product_sales AS (
    SELECT 
        YEARWEEK(o.order_date, 1) AS year_week,
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 4 WEEK)
        AND o.payment_status = 'paid'
    GROUP BY 
        YEARWEEK(o.order_date, 1),
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name
),
current_vs_previous AS (
    SELECT 
        product_id,
        product_name,
        sku,
        category_name,
        SUM(CASE WHEN year_week = YEARWEEK(CURDATE(), 1) THEN revenue ELSE 0 END) AS current_week_revenue,
        SUM(CASE WHEN year_week = YEARWEEK(DATE_SUB(CURDATE(), INTERVAL 1 WEEK), 1) THEN revenue ELSE 0 END) AS previous_week_revenue,
        SUM(CASE WHEN year_week = YEARWEEK(CURDATE(), 1) THEN units_sold ELSE 0 END) AS current_week_units,
        SUM(CASE WHEN year_week = YEARWEEK(DATE_SUB(CURDATE(), INTERVAL 1 WEEK), 1) THEN units_sold ELSE 0 END) AS previous_week_units
    FROM weekly_product_sales
    GROUP BY product_id, product_name, sku, category_name
    HAVING previous_week_revenue > 100
)
SELECT 
    product_name,
    sku,
    category_name,
    CONCAT('$', FORMAT(current_week_revenue, 2)) AS this_week_revenue,
    CONCAT('$', FORMAT(previous_week_revenue, 2)) AS last_week_revenue,
    FORMAT(current_week_units, 0) AS this_week_units,
    FORMAT(previous_week_units, 0) AS last_week_units,
    CONCAT(
        FORMAT(ROUND((current_week_revenue - previous_week_revenue) / 
               NULLIF(previous_week_revenue, 0) * 100, 2), 2), '%'
    ) AS revenue_change,
    ''WARNING'' AS alert,
    CASE 
        WHEN (current_week_revenue - previous_week_revenue) / NULLIF(previous_week_revenue, 0) <= -0.50 THEN 'Critical - Review immediately'
        WHEN (current_week_revenue - previous_week_revenue) / NULLIF(previous_week_revenue, 0) <= -0.30 THEN 'High - Investigate decline'
        ELSE 'Medium - Monitor trend'
    END AS recommended_action
FROM current_vs_previous
WHERE (current_week_revenue - previous_week_revenue) / NULLIF(previous_week_revenue, 0) <= -0.20
ORDER BY (current_week_revenue - previous_week_revenue) / NULLIF(previous_week_revenue, 0) ASC
LIMIT 15;

-- ========================================
-- 6. WEEKLY ORDER METRICS BREAKDOWN
-- Detailed order statistics by week
-- ========================================

WITH weekly_orders AS (
    SELECT 
        YEARWEEK(order_date, 1) AS year_week,
        DATE_FORMAT(DATE_SUB(order_date, INTERVAL WEEKDAY(order_date) DAY), '%b %d') AS week_label,
        COUNT(*) AS total_orders,
        SUM(CASE WHEN payment_status = 'paid' THEN 1 ELSE 0 END) AS paid_orders,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) AS failed_payments,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled,
        SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing,
        AVG(total_amount) AS avg_order_value,
        AVG(CASE WHEN status = 'delivered' 
            THEN TIMESTAMPDIFF(DAY, order_date, updated_at) END) AS avg_delivery_days
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
    GROUP BY 
        YEARWEEK(order_date, 1),
        week_label
)
SELECT 
    week_label,
    FORMAT(total_orders, 0) AS total_orders,
    FORMAT(paid_orders, 0) AS paid_orders,
    FORMAT(failed_payments, 0) AS failed_payments,
    ROUND(paid_orders * 100.0 / total_orders, 2) AS payment_success_rate,
    FORMAT(delivered, 0) AS delivered,
    FORMAT(cancelled, 0) AS cancelled,
    ROUND(delivered * 100.0 / NULLIF(total_orders, 0), 2) AS fulfillment_rate,
    ROUND(cancelled * 100.0 / NULLIF(total_orders, 0), 2) AS cancellation_rate,
    CONCAT('$', FORMAT(avg_order_value, 2)) AS avg_order_value,
    ROUND(avg_delivery_days, 1) AS avg_delivery_days,
    -- Week-over-week comparison
    CONCAT(
        FORMAT(ROUND((total_orders - LAG(total_orders) OVER (ORDER BY year_week)) / 
               NULLIF(LAG(total_orders) OVER (ORDER BY year_week), 0) * 100, 2), 2), '%'
    ) AS orders_wow,
    CASE 
        WHEN delivered * 100.0 / NULLIF(total_orders, 0) >= 90 THEN ''GREEN' Excellent'
        WHEN delivered * 100.0 / NULLIF(total_orders, 0) >= 80 THEN ''YELLOW' Good'
        ELSE ''RED' Needs Improvement'
    END AS performance_grade
FROM weekly_orders
ORDER BY year_week DESC;

-- ========================================
-- 7. GEOGRAPHIC TRENDS
-- Weekly performance by state/region
-- ========================================

WITH weekly_geographic AS (
    SELECT 
        YEARWEEK(o.order_date, 1) AS year_week,
        DATE_FORMAT(DATE_SUB(o.order_date, INTERVAL WEEKDAY(o.order_date) DAY), '%b %d') AS week_label,
        c.state,
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS customers,
        SUM(CASE WHEN o.payment_status = 'paid' THEN o.total_amount ELSE 0 END) AS revenue
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
        AND c.state IS NOT NULL
    GROUP BY 
        YEARWEEK(o.order_date, 1),
        week_label,
        c.state
),
ranked_states AS (
    SELECT 
        year_week,
        week_label,
        state,
        revenue,
        orders,
        customers,
        ROW_NUMBER() OVER (PARTITION BY year_week ORDER BY revenue DESC) AS revenue_rank
    FROM weekly_geographic
)
SELECT 
    week_label,
    state,
    CONCAT('$', FORMAT(revenue, 2)) AS revenue,
    FORMAT(orders, 0) AS orders,
    FORMAT(customers, 0) AS customers,
    revenue_rank,
    CONCAT(
        FORMAT(ROUND((revenue - LAG(revenue) OVER (PARTITION BY state ORDER BY year_week)) / 
               NULLIF(LAG(revenue) OVER (PARTITION BY state ORDER BY year_week), 0) * 100, 2), 2), '%'
    ) AS wow_change,
    CASE 
        WHEN revenue > LAG(revenue) OVER (PARTITION BY state ORDER BY year_week) THEN ''TRENDING_UP''
        WHEN revenue < LAG(revenue) OVER (PARTITION BY state ORDER BY year_week) THEN ''TRENDING_DOWN''
        ELSE '➡️'
    END AS trend
FROM ranked_states
WHERE revenue_rank <= 5
ORDER BY year_week DESC, revenue_rank;

-- ========================================
-- 8. MARKETING CAMPAIGN TRENDS
-- Weekly campaign performance
-- ========================================

WITH weekly_campaigns AS (
    SELECT 
        YEARWEEK(cp.report_date, 1) AS year_week,
        DATE_FORMAT(DATE_SUB(cp.report_date, INTERVAL WEEKDAY(cp.report_date) DAY), '%b %d') AS week_label,
        c.campaign_type,
        COUNT(DISTINCT c.campaign_id) AS active_campaigns,
        SUM(cp.impressions) AS total_impressions,
        SUM(cp.clicks) AS total_clicks,
        SUM(cp.conversions) AS total_conversions,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue
    FROM campaign_performance cp
    JOIN campaigns c ON cp.campaign_id = c.campaign_id
    WHERE cp.report_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
    GROUP BY 
        YEARWEEK(cp.report_date, 1),
        week_label,
        c.campaign_type
)
SELECT 
    week_label,
    campaign_type,
    FORMAT(active_campaigns, 0) AS active_campaigns,
    FORMAT(total_impressions, 0) AS impressions,
    FORMAT(total_clicks, 0) AS clicks,
    FORMAT(total_conversions, 0) AS conversions,
    ROUND(total_clicks * 100.0 / NULLIF(total_impressions, 0), 3) AS ctr_pct,
    ROUND(total_conversions * 100.0 / NULLIF(total_clicks, 0), 2) AS conversion_rate,
    CONCAT('$', FORMAT(total_spend, 2)) AS spend,
    CONCAT('$', FORMAT(total_revenue, 2)) AS revenue,
    ROUND((total_revenue - total_spend) / NULLIF(total_spend, 0) * 100, 2) AS roi_pct,
    CASE 
        WHEN (total_revenue - total_spend) / NULLIF(total_spend, 0) >= 2 THEN ''GREEN' Excellent'
        WHEN (total_revenue - total_spend) / NULLIF(total_spend, 0) >= 1 THEN ''YELLOW' Good'
        WHEN (total_revenue - total_spend) / NULLIF(total_spend, 0) >= 0 THEN '🟠 Break-even'
        ELSE ''RED' Loss'
    END AS performance
FROM weekly_campaigns
ORDER BY year_week DESC, total_revenue DESC;

-- ========================================
-- 9. INVENTORY TRENDS
-- Weekly stock level changes
-- ========================================

WITH weekly_inventory_snapshot AS (
    SELECT 
        YEARWEEK(CURDATE(), 1) AS year_week,
        'Current Week' AS week_label,
        COUNT(DISTINCT p.product_id) AS total_products,
        SUM(p.stock_quantity) AS total_units,
        SUM(CASE WHEN p.stock_quantity = 0 THEN 1 ELSE 0 END) AS out_of_stock_count,
        SUM(CASE WHEN p.stock_quantity <= 10 THEN 1 ELSE 0 END) AS low_stock_count,
        SUM(p.stock_quantity * p.cost) AS inventory_value
    FROM products p
    WHERE p.status = 'active'
),
product_velocity AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.stock_quantity AS current_stock,
        COALESCE(SUM(oi.quantity), 0) AS units_sold_this_week,
        COALESCE(SUM(oi.quantity) / 7.0, 0) AS daily_velocity,
        p.stock_quantity / NULLIF(COALESCE(SUM(oi.quantity) / 7.0, 0), 0) AS days_of_stock
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id 
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        AND o.payment_status = 'paid'
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, p.stock_quantity
    HAVING units_sold_this_week > 0
)
SELECT 
    'INVENTORY SUMMARY' AS section,
    FORMAT(total_products, 0) AS total_products,
    FORMAT(total_units, 0) AS total_units_on_hand,
    FORMAT(out_of_stock_count, 0) AS out_of_stock,
    FORMAT(low_stock_count, 0) AS low_stock,
    CONCAT('$', FORMAT(inventory_value, 2)) AS inventory_value,
    ROUND(out_of_stock_count * 100.0 / total_products, 2) AS stockout_rate_pct,
    '' AS spacer
FROM weekly_inventory_snapshot

UNION ALL

SELECT 
    'CRITICAL STOCK ALERTS',
    product_name,
    FORMAT(current_stock, 0),
    FORMAT(units_sold_this_week, 0),
    FORMAT(daily_velocity, 1),
    FORMAT(days_of_stock, 1),
    CASE 
        WHEN days_of_stock <= 3 THEN ''RED' Critical'
        WHEN days_of_stock <= 7 THEN '🟠 Warning'
        ELSE ''GREEN' Adequate'
    END,
    ''
FROM product_velocity
WHERE days_of_stock <= 7
ORDER BY days_of_stock
LIMIT 10;

-- ========================================
-- 10. CUSTOMER RETENTION TRENDS
-- Weekly retention and churn analysis
-- ========================================

WITH weekly_cohorts AS (
    SELECT 
        YEARWEEK(first_order, 1) AS cohort_week,
        DATE_FORMAT(first_order, '%b %d, %Y') AS cohort_label,
        customer_id,
        first_order
    FROM (
        SELECT 
            customer_id,
            MIN(order_date) AS first_order
        FROM orders
        WHERE payment_status = 'paid'
        GROUP BY customer_id
    ) AS first_orders
    WHERE first_order >= DATE_SUB(CURDATE(), INTERVAL 12 WEEK)
),
retention_data AS (
    SELECT 
        wc.cohort_week,
        wc.cohort_label,
        COUNT(DISTINCT wc.customer_id) AS cohort_size,
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_ADD(wc.first_order, INTERVAL 1 WEEK)
            AND o.order_date < DATE_ADD(wc.first_order, INTERVAL 2 WEEK)
            THEN wc.customer_id 
        END) AS week_1_retained,
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_ADD(wc.first_order, INTERVAL 4 WEEK)
            AND o.order_date < DATE_ADD(wc.first_order, INTERVAL 5 WEEK)
            THEN wc.customer_id 
        END) AS week_4_retained,
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_ADD(wc.first_order, INTERVAL 8 WEEK)
            AND o.order_date < DATE_ADD(wc.first_order, INTERVAL 9 WEEK)
            THEN wc.customer_id 
        END) AS week_8_retained
    FROM weekly_cohorts wc
    LEFT JOIN orders o ON wc.customer_id = o.customer_id 
        AND o.payment_status = 'paid'
    GROUP BY wc.cohort_week, wc.cohort_label
)
SELECT 
    cohort_label AS acquisition_week,
    FORMAT(cohort_size, 0) AS cohort_size,
    FORMAT(week_1_retained, 0) AS retained_week_1,
    ROUND(week_1_retained * 100.0 / cohort_size, 2) AS retention_week_1_pct,
    FORMAT(week_4_retained, 0) AS retained_week_4,
    ROUND(week_4_retained * 100.0 / cohort_size, 2) AS retention_week_4_pct,
    FORMAT(week_8_retained, 0) AS retained_week_8,
    ROUND(week_8_retained * 100.0 / cohort_size, 2) AS retention_week_8_pct,
    CASE 
        WHEN week_1_retained * 100.0 / cohort_size >= 30 THEN ''GREEN' Strong'
        WHEN week_1_retained * 100.0 / cohort_size >= 20 THEN ''YELLOW' Moderate'
        WHEN week_1_retained * 100.0 / cohort_size >= 10 THEN '🟠 Weak'
        ELSE ''RED' Poor'
    END AS retention_health
FROM retention_data
WHERE cohort_size >= 10
ORDER BY cohort_week DESC;

-- ========================================
-- 11. RETURN & REFUND TRENDS
-- Weekly returns analysis
-- ========================================

WITH weekly_returns AS (
    SELECT 
        YEARWEEK(r.created_at, 1) AS year_week,
        DATE_FORMAT(DATE_SUB(r.created_at, INTERVAL WEEKDAY(r.created_at) DAY), '%b %d') AS week_label,
        COUNT(DISTINCT r.return_id) AS return_count,
        COUNT(DISTINCT CASE WHEN r.status = 'approved' THEN r.return_id END) AS approved_returns,
        SUM(CASE WHEN r.status = 'refunded' THEN r.refund_amount ELSE 0 END) AS refunded_amount,
        r.reason,
        COUNT(*) AS reason_count
    FROM returns r
    WHERE r.created_at >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
    GROUP BY 
        YEARWEEK(r.created_at, 1),
        week_label,
        r.reason
),
weekly_orders_for_return_rate AS (
    SELECT 
        YEARWEEK(order_date, 1) AS year_week,
        COUNT(DISTINCT order_id) AS total_orders
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
        AND payment_status = 'paid'
    GROUP BY YEARWEEK(order_date, 1)
)
SELECT 
    wr.week_label,
    FORMAT(wr.return_count, 0) AS total_returns,
    FORMAT(wr.approved_returns, 0) AS approved_returns,
    CONCAT(', FORMAT(wr.refunded_amount, 2)) AS refunded_amount,
    ROUND(wr.return_count * 100.0 / NULLIF(wo.total_orders, 0), 2) AS return_rate_pct,
    wr.reason AS top_reason,
    FORMAT(wr.reason_count, 0) AS reason_count,
    CONCAT(
        FORMAT(ROUND((wr.return_count - LAG(wr.return_count) OVER (ORDER BY wr.year_week)) / 
               NULLIF(LAG(wr.return_count) OVER (ORDER BY wr.year_week), 0) * 100, 2), 2), '%'
    ) AS wow_change,
    CASE 
        WHEN wr.return_count * 100.0 / NULLIF(wo.total_orders, 0) <= 5 THEN ''GREEN' Excellent'
        WHEN wr.return_count * 100.0 / NULLIF(wo.total_orders, 0) <= 10 THEN ''YELLOW' Acceptable'
        WHEN wr.return_count * 100.0 / NULLIF(wo.total_orders, 0) <= 15 THEN '🟠 High'
        ELSE ''RED' Critical'
    END AS status
FROM (
    SELECT 
        year_week,
        week_label,
        SUM(return_count) AS return_count,
        SUM(approved_returns) AS approved_returns,
        SUM(refunded_amount) AS refunded_amount,
        reason,
        reason_count,
        ROW_NUMBER() OVER (PARTITION BY year_week ORDER BY reason_count DESC) AS reason_rank
    FROM weekly_returns
    GROUP BY year_week, week_label, reason, reason_count
) wr
JOIN weekly_orders_for_return_rate wo ON wr.year_week = wo.year_week
WHERE wr.reason_rank = 1
ORDER BY wr.year_week DESC;

-- ========================================
-- 12. REVIEW & RATING TRENDS
-- Weekly customer feedback analysis
-- ========================================

WITH weekly_reviews AS (
    SELECT 
        YEARWEEK(created_at, 1) AS year_week,
        DATE_FORMAT(DATE_SUB(created_at, INTERVAL WEEKDAY(created_at) DAY), '%b %d') AS week_label,
        COUNT(*) AS total_reviews,
        COUNT(CASE WHEN status = 'approved' THEN 1 END) AS approved_reviews,
        AVG(CASE WHEN status = 'approved' THEN rating END) AS avg_rating,
        COUNT(CASE WHEN rating = 5 THEN 1 END) AS five_star,
        COUNT(CASE WHEN rating = 4 THEN 1 END) AS four_star,
        COUNT(CASE WHEN rating = 3 THEN 1 END) AS three_star,
        COUNT(CASE WHEN rating <= 2 THEN 1 END) AS negative_reviews,
        COUNT(CASE WHEN is_verified_purchase = TRUE THEN 1 END) AS verified_purchases
    FROM reviews
    WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
    GROUP BY 
        YEARWEEK(created_at, 1),
        week_label
)
SELECT 
    week_label,
    FORMAT(total_reviews, 0) AS total_reviews,
    FORMAT(approved_reviews, 0) AS approved_reviews,
    ROUND(avg_rating, 2) AS avg_rating,
    FORMAT(five_star, 0) AS five_star_count,
    FORMAT(negative_reviews, 0) AS negative_reviews,
    ROUND((five_star + four_star) * 100.0 / NULLIF(total_reviews, 0), 2) AS positive_review_pct,
    ROUND(verified_purchases * 100.0 / NULLIF(total_reviews, 0), 2) AS verified_pct,
    CONCAT(
        FORMAT(ROUND((avg_rating - LAG(avg_rating) OVER (ORDER BY year_week)), 2), 2)
    ) AS rating_change,
    CASE 
        WHEN avg_rating >= 4.5 THEN ''GREEN' Excellent'
        WHEN avg_rating >= 4.0 THEN ''YELLOW' Good'
        WHEN avg_rating >= 3.5 THEN '🟠 Fair'
        ELSE ''RED' Poor'
    END AS satisfaction_level,
    CASE 
        WHEN avg_rating > LAG(avg_rating) OVER (ORDER BY year_week) THEN ''TRENDING_UP' Improving'
        WHEN avg_rating < LAG(avg_rating) OVER (ORDER BY year_week) THEN ''TRENDING_DOWN' Declining'
        ELSE '➡️ Stable'
    END AS trend
FROM weekly_reviews
ORDER BY year_week DESC;

-- ========================================
-- 13. PAYMENT METHOD TRENDS
-- Weekly payment preferences
-- ========================================

WITH weekly_payments AS (
    SELECT 
        YEARWEEK(o.order_date, 1) AS year_week,
        DATE_FORMAT(DATE_SUB(o.order_date, INTERVAL WEEKDAY(o.order_date) DAY), '%b %d') AS week_label,
        pm.payment_type,
        COUNT(DISTINCT o.order_id) AS transactions,
        SUM(CASE WHEN o.payment_status = 'paid' THEN o.total_amount ELSE 0 END) AS successful_amount,
        SUM(CASE WHEN o.payment_status = 'failed' THEN 1 ELSE 0 END) AS failed_transactions,
        COUNT(DISTINCT o.customer_id) AS unique_customers
    FROM orders o
    JOIN payment_methods pm ON o.customer_id = pm.customer_id AND pm.is_default = TRUE
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
    GROUP BY 
        YEARWEEK(o.order_date, 1),
        week_label,
        pm.payment_type
)
SELECT 
    week_label,
    payment_type,
    FORMAT(transactions, 0) AS transactions,
    CONCAT(', FORMAT(successful_amount, 2)) AS successful_amount,
    FORMAT(failed_transactions, 0) AS failed_count,
    ROUND((transactions - failed_transactions) * 100.0 / transactions, 2) AS success_rate_pct,
    ROUND(successful_amount * 100.0 / SUM(successful_amount) OVER (PARTITION BY year_week), 2) AS revenue_share_pct,
    CASE payment_type
        WHEN 'credit_card' THEN '💳'
        WHEN 'debit_card' THEN '💳'
        WHEN 'paypal' THEN '🅿️'
        WHEN 'bank_transfer' THEN '🏦'
        ELSE ''MONEY''
    END AS icon
FROM weekly_payments
ORDER BY year_week DESC, successful_amount DESC;

-- ========================================
-- 14. SHIPPING PERFORMANCE TRENDS
-- Weekly delivery metrics
-- ========================================

WITH weekly_shipping AS (
    SELECT 
        YEARWEEK(order_date, 1) AS year_week,
        DATE_FORMAT(DATE_SUB(order_date, INTERVAL WEEKDAY(order_date) DAY), '%b %d') AS week_label,
        COUNT(DISTINCT order_id) AS total_orders,
        SUM(CASE WHEN status = 'shipped' THEN 1 ELSE 0 END) AS shipped,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered,
        AVG(CASE WHEN status IN ('shipped', 'delivered') 
            THEN TIMESTAMPDIFF(DAY, order_date, updated_at) END) AS avg_ship_time,
        AVG(CASE WHEN status = 'delivered' 
            THEN TIMESTAMPDIFF(DAY, order_date, updated_at) END) AS avg_delivery_time,
        SUM(shipping_cost) AS total_shipping_revenue
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
    GROUP BY 
        YEARWEEK(order_date, 1),
        week_label
)
SELECT 
    week_label,
    FORMAT(total_orders, 0) AS total_orders,
    FORMAT(shipped, 0) AS shipped,
    FORMAT(delivered, 0) AS delivered,
    ROUND(shipped * 100.0 / total_orders, 2) AS ship_rate_pct,
    ROUND(delivered * 100.0 / total_orders, 2) AS delivery_rate_pct,
    ROUND(avg_ship_time, 1) AS avg_ship_days,
    ROUND(avg_delivery_time, 1) AS avg_delivery_days,
    CONCAT(', FORMAT(total_shipping_revenue, 2)) AS shipping_revenue,
    CASE 
        WHEN avg_delivery_time <= 3 THEN ''GREEN' Excellent'
        WHEN avg_delivery_time <= 5 THEN ''YELLOW' Good'
        WHEN avg_delivery_time <= 7 THEN '🟠 Acceptable'
        ELSE ''RED' Slow'
    END AS delivery_performance,
    CONCAT(
        FORMAT(ROUND((avg_delivery_time - LAG(avg_delivery_time) OVER (ORDER BY year_week)), 1), 1), ' days'
    ) AS delivery_time_change
FROM weekly_shipping
ORDER BY year_week DESC;

-- ========================================
-- 15. WEEKLY TREND SUMMARY SCORECARD
-- Comprehensive week-over-week summary
-- ========================================

WITH current_week_metrics AS (
    SELECT 
        COUNT(DISTINCT CASE WHEN payment_status = 'paid' THEN order_id END) AS orders,
        SUM(CASE WHEN payment_status = 'paid' THEN total_amount ELSE 0 END) AS revenue,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(CASE WHEN payment_status = 'paid' THEN total_amount END) AS aov
    FROM orders
    WHERE YEARWEEK(order_date, 1) = YEARWEEK(CURDATE(), 1)
),
previous_week_metrics AS (
    SELECT 
        COUNT(DISTINCT CASE WHEN payment_status = 'paid' THEN order_id END) AS orders,
        SUM(CASE WHEN payment_status = 'paid' THEN total_amount ELSE 0 END) AS revenue,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(CASE WHEN payment_status = 'paid' THEN total_amount END) AS aov
    FROM orders
    WHERE YEARWEEK(order_date, 1) = YEARWEEK(DATE_SUB(CURDATE(), INTERVAL 1 WEEK), 1)
)
SELECT 
    ''CHART' WEEK-OVER-WEEK SUMMARY' AS metric_category,
    '' AS current_week,
    '' AS previous_week,
    '' AS change,
    '' AS trend,
    '' AS status

UNION ALL

SELECT 
    'Total Revenue',
    CONCAT(', FORMAT(cw.revenue, 2)),
    CONCAT(', FORMAT(pw.revenue, 2)),
    CONCAT(FORMAT(ROUND((cw.revenue - pw.revenue) / NULLIF(pw.revenue, 0) * 100, 2), 2), '%'),
    CASE 
        WHEN cw.revenue > pw.revenue THEN ''TRENDING_UP''
        WHEN cw.revenue < pw.revenue THEN ''TRENDING_DOWN''
        ELSE '➡️'
    END,
    CASE 
        WHEN (cw.revenue - pw.revenue) / NULLIF(pw.revenue, 0) >= 0.10 THEN ''GREEN' Strong Growth'
        WHEN (cw.revenue - pw.revenue) / NULLIF(pw.revenue, 0) >= 0 THEN ''YELLOW' Growing'
        WHEN (cw.revenue - pw.revenue) / NULLIF(pw.revenue, 0) >= -0.05 THEN '🟠 Slight Decline'
        ELSE ''RED' Significant Decline'
    END
FROM current_week_metrics cw, previous_week_metrics pw

UNION ALL

SELECT 
    'Total Orders',
    FORMAT(cw.orders, 0),
    FORMAT(pw.orders, 0),
    CONCAT(FORMAT(ROUND((cw.orders - pw.orders) / NULLIF(pw.orders, 0) * 100, 2), 2), '%'),
    CASE 
        WHEN cw.orders > pw.orders THEN ''TRENDING_UP''
        WHEN cw.orders < pw.orders THEN ''TRENDING_DOWN''
        ELSE '➡️'
    END,
    CASE 
        WHEN (cw.orders - pw.orders) / NULLIF(pw.orders, 0) >= 0.10 THEN ''GREEN' Strong Growth'
        WHEN (cw.orders - pw.orders) / NULLIF(pw.orders, 0) >= 0 THEN ''YELLOW' Growing'
        WHEN (cw.orders - pw.orders) / NULLIF(pw.orders, 0) >= -0.05 THEN '🟠 Slight Decline'
        ELSE ''RED' Significant Decline'
    END
FROM current_week_metrics cw, previous_week_metrics pw

UNION ALL

SELECT 
    'Active Customers',
    FORMAT(cw.customers, 0),
    FORMAT(pw.customers, 0),
    CONCAT(FORMAT(ROUND((cw.customers - pw.customers) / NULLIF(pw.customers, 0) * 100, 2), 2), '%'),
    CASE 
        WHEN cw.customers > pw.customers THEN ''TRENDING_UP''
        WHEN cw.customers < pw.customers THEN ''TRENDING_DOWN''
        ELSE '➡️'
    END,
    CASE 
        WHEN (cw.customers - pw.customers) / NULLIF(pw.customers, 0) >= 0.10 THEN ''GREEN' Strong Growth'
        WHEN (cw.customers - pw.customers) / NULLIF(pw.customers, 0) >= 0 THEN ''YELLOW' Growing'
        WHEN (cw.customers - pw.customers) / NULLIF(pw.customers, 0) >= -0.05 THEN '🟠 Slight Decline'
        ELSE ''RED' Significant Decline'
    END
FROM current_week_metrics cw, previous_week_metrics pw

UNION ALL

SELECT 
    'Average Order Value',
    CONCAT(', FORMAT(cw.aov, 2)),
    CONCAT(', FORMAT(pw.aov, 2)),
    CONCAT(FORMAT(ROUND((cw.aov - pw.aov) / NULLIF(pw.aov, 0) * 100, 2), 2), '%'),
    CASE 
        WHEN cw.aov > pw.aov THEN ''TRENDING_UP''
        WHEN cw.aov < pw.aov THEN ''TRENDING_DOWN''
        ELSE '➡️'
    END,
    CASE 
        WHEN (cw.aov - pw.aov) / NULLIF(pw.aov, 0) >= 0.05 THEN ''GREEN' Improving'
        WHEN (cw.aov - pw.aov) / NULLIF(pw.aov, 0) >= 0 THEN ''YELLOW' Stable'
        WHEN (cw.aov - pw.aov) / NULLIF(pw.aov, 0) >= -0.05 THEN '🟠 Slight Decline'
        ELSE ''RED' Declining'
    END
FROM current_week_metrics cw, previous_week_metrics pw;

-- ========================================
-- 16. KEY INSIGHTS & RECOMMENDATIONS
-- Automated insights based on trends
-- ========================================

SELECT 
    '💡 KEY INSIGHTS & RECOMMENDATIONS' AS section,
    '' AS insight,
    '' AS data_point,
    '' AS recommendation;

WITH trend_analysis AS (
    SELECT 
        (SELECT SUM(CASE WHEN payment_status = 'paid' THEN total_amount ELSE 0 END)
         FROM orders 
         WHERE YEARWEEK(order_date, 1) = YEARWEEK(CURDATE(), 1)) AS current_week_revenue,
        (SELECT SUM(CASE WHEN payment_status = 'paid' THEN total_amount ELSE 0 END)
         FROM orders 
         WHERE YEARWEEK(order_date, 1) = YEARWEEK(DATE_SUB(CURDATE(), INTERVAL 1 WEEK), 1)) AS last_week_revenue,
        (SELECT AVG(rating) FROM reviews 
         WHERE YEARWEEK(created_at, 1) = YEARWEEK(CURDATE(), 1) 
         AND status = 'approved') AS current_avg_rating,
        (SELECT COUNT(*) FROM products 
         WHERE status = 'active' AND stock_quantity = 0) AS out_of_stock_count
)
SELECT 
    '1',
    'Revenue Trend',
    CONCAT('Week-over-week change: ', 
           FORMAT(ROUND((current_week_revenue - last_week_revenue) / 
                  NULLIF(last_week_revenue, 0) * 100, 2), 2), '%'),
    CASE 
        WHEN (current_week_revenue - last_week_revenue) / NULLIF(last_week_revenue, 0) > 0.20 
        THEN 'Exceptional growth - analyze success factors and scale winning strategies'
        WHEN (current_week_revenue - last_week_revenue) / NULLIF(last_week_revenue, 0) < -0.15 
        THEN 'Concerning decline - investigate marketing effectiveness and product availability'
        ELSE 'Performance within normal range - maintain current strategies'
    END
FROM trend_analysis

UNION ALL

SELECT 
    '2',
    'Customer Satisfaction',
    CONCAT('Current average rating: ', FORMAT(current_avg_rating, 2), '/5'),
    CASE 
        WHEN current_avg_rating < 4.0 
        THEN 'Below target - review negative feedback and implement quality improvements'
        WHEN current_avg_rating >= 4.5 
        THEN 'Excellent satisfaction - leverage positive reviews in marketing'
        ELSE 'Satisfactory - continue monitoring and incremental improvements'
    END
FROM trend_analysis

UNION ALL

SELECT 
    '3',
    'Inventory Management',
    CONCAT(out_of_stock_count, ' products currently out of stock'),
    CASE 
        WHEN out_of_stock_count > 20 
        THEN 'High stockout rate - expedite restocking and improve demand forecasting'
        WHEN out_of_stock_count > 10 
        THEN 'Moderate stockouts - prioritize fast-moving items for reorder'
        ELSE 'Inventory levels adequate - maintain current replenishment schedule'
    END
FROM trend_analysis;

-- ========================================
-- REPORT FOOTER
-- ========================================

SELECT 
    '═══════════════════════════════════════════════════════════' AS separator,
    CONCAT('Analysis Period: Last 12 Weeks (', 
           DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 12 WEEK), '%b %d'), ' - ',
           DATE_FORMAT(CURDATE(), '%b %d, %Y'), ')') AS period,
    CONCAT('Report Generated: ', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')) AS timestamp,
    ''SUCCESS' Weekly Trends Analysis Complete' AS status;