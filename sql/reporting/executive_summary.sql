-- ========================================
-- EXECUTIVE SUMMARY DASHBOARD
-- E-commerce Revenue Analytics Engine
-- High-Level KPIs, Executive Metrics, Key Trends
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. EXECUTIVE KPI OVERVIEW
-- Top-level business performance metrics
-- ========================================

WITH current_period AS (
    SELECT 
        SUM(o.total_amount) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS customers,
        AVG(o.total_amount) AS avg_order_value,
        SUM(oi.quantity) AS units_sold,
        SUM(o.total_amount) / COUNT(DISTINCT o.customer_id) AS revenue_per_customer,
        COUNT(DISTINCT o.order_id) / COUNT(DISTINCT o.customer_id) AS orders_per_customer
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
        AND o.order_date < CURDATE()
),
previous_period AS (
    SELECT 
        SUM(o.total_amount) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS customers,
        AVG(o.total_amount) AS avg_order_value,
        SUM(oi.quantity) AS units_sold,
        SUM(o.total_amount) / COUNT(DISTINCT o.customer_id) AS revenue_per_customer,
        COUNT(DISTINCT o.order_id) / COUNT(DISTINCT o.customer_id) AS orders_per_customer
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 2 MONTH)
        AND o.order_date < DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
),
ytd_metrics AS (
    SELECT 
        SUM(o.total_amount) AS ytd_revenue,
        COUNT(DISTINCT o.order_id) AS ytd_orders,
        COUNT(DISTINCT o.customer_id) AS ytd_customers
    FROM orders o
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND YEAR(o.order_date) = YEAR(CURDATE())
)
SELECT 
    'Total Revenue' AS kpi_name,
    CONCAT('$', FORMAT(cp.revenue, 2)) AS current_value,
    CONCAT('$', FORMAT(pp.revenue, 2)) AS previous_value,
    CONCAT('$', FORMAT(ym.ytd_revenue, 2)) AS ytd_value,
    ROUND((cp.revenue - pp.revenue) / pp.revenue * 100, 2) AS mom_change_pct,
    CASE 
        WHEN (cp.revenue - pp.revenue) / pp.revenue * 100 >= 10 THEN '🚀 Strong Growth'
        WHEN (cp.revenue - pp.revenue) / pp.revenue * 100 >= 5 THEN '📈 Growing'
        WHEN (cp.revenue - pp.revenue) / pp.revenue * 100 >= 0 THEN '➡️ Stable'
        WHEN (cp.revenue - pp.revenue) / pp.revenue * 100 >= -5 THEN '⚠️ Declining'
        ELSE '🔴 Alert'
    END AS status
FROM current_period cp, previous_period pp, ytd_metrics ym

UNION ALL

SELECT 
    'Total Orders',
    FORMAT(cp.orders, 0),
    FORMAT(pp.orders, 0),
    FORMAT(ym.ytd_orders, 0),
    ROUND((cp.orders - pp.orders) / pp.orders * 100, 2),
    CASE 
        WHEN (cp.orders - pp.orders) / pp.orders * 100 >= 10 THEN '🚀 Strong Growth'
        WHEN (cp.orders - pp.orders) / pp.orders * 100 >= 5 THEN '📈 Growing'
        WHEN (cp.orders - pp.orders) / pp.orders * 100 >= 0 THEN '➡️ Stable'
        WHEN (cp.orders - pp.orders) / pp.orders * 100 >= -5 THEN '⚠️ Declining'
        ELSE '🔴 Alert'
    END
FROM current_period cp, previous_period pp, ytd_metrics ym

UNION ALL

SELECT 
    'Active Customers',
    FORMAT(cp.customers, 0),
    FORMAT(pp.customers, 0),
    FORMAT(ym.ytd_customers, 0),
    ROUND((cp.customers - pp.customers) / pp.customers * 100, 2),
    CASE 
        WHEN (cp.customers - pp.customers) / pp.customers * 100 >= 10 THEN '🚀 Strong Growth'
        WHEN (cp.customers - pp.customers) / pp.customers * 100 >= 5 THEN '📈 Growing'
        WHEN (cp.customers - pp.customers) / pp.customers * 100 >= 0 THEN '➡️ Stable'
        WHEN (cp.customers - pp.customers) / pp.customers * 100 >= -5 THEN '⚠️ Declining'
        ELSE '🔴 Alert'
    END
FROM current_period cp, previous_period pp, ytd_metrics ym

UNION ALL

SELECT 
    'Avg Order Value',
    CONCAT('$', FORMAT(cp.avg_order_value, 2)),
    CONCAT('$', FORMAT(pp.avg_order_value, 2)),
    '-',
    ROUND((cp.avg_order_value - pp.avg_order_value) / pp.avg_order_value * 100, 2),
    CASE 
        WHEN (cp.avg_order_value - pp.avg_order_value) / pp.avg_order_value * 100 >= 5 THEN '🚀 Strong Growth'
        WHEN (cp.avg_order_value - pp.avg_order_value) / pp.avg_order_value * 100 >= 2 THEN '📈 Growing'
        WHEN (cp.avg_order_value - pp.avg_order_value) / pp.avg_order_value * 100 >= -2 THEN '➡️ Stable'
        WHEN (cp.avg_order_value - pp.avg_order_value) / pp.avg_order_value * 100 >= -5 THEN '⚠️ Declining'
        ELSE '🔴 Alert'
    END
FROM current_period cp, previous_period pp, ytd_metrics ym

UNION ALL

SELECT 
    'Units Sold',
    FORMAT(cp.units_sold, 0),
    FORMAT(pp.units_sold, 0),
    '-',
    ROUND((cp.units_sold - pp.units_sold) / pp.units_sold * 100, 2),
    CASE 
        WHEN (cp.units_sold - pp.units_sold) / pp.units_sold * 100 >= 10 THEN '🚀 Strong Growth'
        WHEN (cp.units_sold - pp.units_sold) / pp.units_sold * 100 >= 5 THEN '📈 Growing'
        WHEN (cp.units_sold - pp.units_sold) / pp.units_sold * 100 >= 0 THEN '➡️ Stable'
        WHEN (cp.units_sold - pp.units_sold) / pp.units_sold * 100 >= -5 THEN '⚠️ Declining'
        ELSE '🔴 Alert'
    END
FROM current_period cp, previous_period pp, ytd_metrics ym

UNION ALL

SELECT 
    'Revenue per Customer',
    CONCAT('$', FORMAT(cp.revenue_per_customer, 2)),
    CONCAT('$', FORMAT(pp.revenue_per_customer, 2)),
    '-',
    ROUND((cp.revenue_per_customer - pp.revenue_per_customer) / pp.revenue_per_customer * 100, 2),
    CASE 
        WHEN (cp.revenue_per_customer - pp.revenue_per_customer) / pp.revenue_per_customer * 100 >= 5 THEN '🚀 Strong Growth'
        WHEN (cp.revenue_per_customer - pp.revenue_per_customer) / pp.revenue_per_customer * 100 >= 2 THEN '📈 Growing'
        WHEN (cp.revenue_per_customer - pp.revenue_per_customer) / pp.revenue_per_customer * 100 >= -2 THEN '➡️ Stable'
        WHEN (cp.revenue_per_customer - pp.revenue_per_customer) / pp.revenue_per_customer * 100 >= -5 THEN '⚠️ Declining'
        ELSE '🔴 Alert'
    END
FROM current_period cp, previous_period pp, ytd_metrics ym;

-- ========================================
-- 2. REVENUE TRENDS - MONTHLY VIEW
-- 12-month revenue performance with trends
-- ========================================

WITH monthly_revenue AS (
    SELECT 
        DATE_FORMAT(o.order_date, '%Y-%m') AS month,
        DATE_FORMAT(o.order_date, '%b %Y') AS month_name,
        SUM(o.total_amount) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS customers,
        AVG(o.total_amount) AS avg_order_value,
        SUM(o.total_amount) / COUNT(DISTINCT o.customer_id) AS revenue_per_customer
    FROM orders o
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY DATE_FORMAT(o.order_date, '%Y-%m'), DATE_FORMAT(o.order_date, '%b %Y')
)
SELECT 
    month,
    month_name,
    ROUND(revenue, 2) AS revenue,
    orders,
    customers,
    ROUND(avg_order_value, 2) AS avg_order_value,
    ROUND(revenue_per_customer, 2) AS revenue_per_customer,
    ROUND((revenue - LAG(revenue) OVER (ORDER BY month)) / 
        NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100, 2) AS mom_revenue_growth_pct,
    ROUND((orders - LAG(orders) OVER (ORDER BY month)) / 
        NULLIF(LAG(orders) OVER (ORDER BY month), 0) * 100, 2) AS mom_order_growth_pct,
    ROUND(AVG(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS revenue_3mo_avg,
    CASE 
        WHEN revenue > LAG(revenue, 1) OVER (ORDER BY month) 
            AND LAG(revenue, 1) OVER (ORDER BY month) > LAG(revenue, 2) OVER (ORDER BY month)
            THEN '📈 Upward Trend'
        WHEN revenue < LAG(revenue, 1) OVER (ORDER BY month) 
            AND LAG(revenue, 1) OVER (ORDER BY month) < LAG(revenue, 2) OVER (ORDER BY month)
            THEN '📉 Downward Trend'
        WHEN revenue > LAG(revenue) OVER (ORDER BY month)
            THEN '↗️ Growing'
        WHEN revenue < LAG(revenue) OVER (ORDER BY month)
            THEN '↘️ Declining'
        ELSE '➡️ Stable'
    END AS trend
FROM monthly_revenue
ORDER BY month DESC;

-- ========================================
-- 3. CATEGORY PERFORMANCE SUMMARY
-- Top performing categories by revenue
-- ========================================

WITH category_metrics AS (
    SELECT 
        pc.category_name,
        SUM(o.total_amount) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS customers,
        AVG(o.total_amount) AS avg_order_value,
        SUM(oi.quantity) AS units_sold,
        COUNT(DISTINCT p.product_id) AS product_count,
        AVG(rev.rating) AS avg_rating
    FROM product_categories pc
    JOIN products p ON pc.category_id = p.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    LEFT JOIN reviews rev ON p.product_id = rev.product_id 
        AND rev.status = 'approved'
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
    GROUP BY pc.category_name
),
total_revenue AS (
    SELECT SUM(revenue) AS total FROM category_metrics
)
SELECT 
    cm.category_name,
    ROUND(cm.revenue, 2) AS revenue,
    ROUND(cm.revenue * 100.0 / tr.total, 2) AS revenue_share_pct,
    cm.orders,
    cm.customers,
    ROUND(cm.avg_order_value, 2) AS avg_order_value,
    cm.units_sold,
    cm.product_count,
    ROUND(cm.avg_rating, 2) AS avg_rating,
    RANK() OVER (ORDER BY cm.revenue DESC) AS revenue_rank,
    CASE 
        WHEN cm.revenue * 100.0 / tr.total >= 20 THEN '⭐ Top Performer'
        WHEN cm.revenue * 100.0 / tr.total >= 10 THEN '🔥 Strong'
        WHEN cm.revenue * 100.0 / tr.total >= 5 THEN '✓ Good'
        ELSE '○ Moderate'
    END AS performance_tier
FROM category_metrics cm
CROSS JOIN total_revenue tr
ORDER BY cm.revenue DESC;

-- ========================================
-- 4. CUSTOMER SEGMENTATION OVERVIEW
-- High-value, regular, and at-risk customers
-- ========================================

WITH customer_segments AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order,
        MIN(o.order_date) AS first_order_date,
        DATEDIFF(CURDATE(), MIN(o.order_date)) AS customer_lifetime_days
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email
),
segment_classification AS (
    SELECT 
        *,
        CASE 
            WHEN total_revenue >= 1000 AND days_since_last_order <= 30 THEN 'VIP Active'
            WHEN total_revenue >= 1000 AND days_since_last_order <= 90 THEN 'VIP At Risk'
            WHEN total_revenue >= 1000 THEN 'VIP Churned'
            WHEN total_orders >= 3 AND days_since_last_order <= 30 THEN 'Loyal Active'
            WHEN total_orders >= 3 AND days_since_last_order <= 90 THEN 'Loyal At Risk'
            WHEN total_orders >= 3 THEN 'Loyal Churned'
            WHEN total_orders = 1 AND days_since_last_order <= 30 THEN 'New'
            WHEN days_since_last_order <= 90 THEN 'Regular'
            ELSE 'At Risk'
        END AS segment
    FROM customer_segments
)
SELECT 
    segment,
    COUNT(DISTINCT customer_id) AS customer_count,
    ROUND(SUM(total_revenue), 2) AS total_revenue,
    ROUND(AVG(total_revenue), 2) AS avg_customer_value,
    ROUND(AVG(total_orders), 1) AS avg_orders,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(AVG(days_since_last_order), 0) AS avg_days_since_last_order,
    ROUND(COUNT(DISTINCT customer_id) * 100.0 / 
        (SELECT COUNT(DISTINCT customer_id) FROM segment_classification), 2) AS customer_share_pct,
    ROUND(SUM(total_revenue) * 100.0 / 
        (SELECT SUM(total_revenue) FROM segment_classification), 2) AS revenue_share_pct,
    CASE 
        WHEN segment LIKE '%VIP%' THEN '🔴 High Priority'
        WHEN segment LIKE '%At Risk%' THEN '🟡 Medium Priority'
        WHEN segment = 'New' THEN '🟢 Nurture'
        ELSE '🔵 Monitor'
    END AS action_priority
FROM segment_classification
GROUP BY segment
ORDER BY total_revenue DESC;

-- ========================================
-- 5. PRODUCT PERFORMANCE HIGHLIGHTS
-- Top and bottom performing products
-- ========================================

WITH product_performance AS (
    SELECT 
        p.product_id,
        p.product_name,
        pc.category_name,
        p.price,
        p.cost,
        (p.price - p.cost) / p.price * 100 AS margin_pct,
        COUNT(DISTINCT o.order_id) AS orders,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS revenue,
        AVG(rev.rating) AS avg_rating,
        COUNT(DISTINCT rev.review_id) AS review_count,
        p.stock_quantity
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
    LEFT JOIN reviews rev ON p.product_id = rev.product_id 
        AND rev.status = 'approved'
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, pc.category_name, p.price, p.cost, p.stock_quantity
)
(
    SELECT 
        'TOP PERFORMER' AS performance_type,
        product_name,
        category_name,
        ROUND(revenue, 2) AS revenue,
        units_sold,
        orders,
        ROUND(price, 2) AS price,
        ROUND(margin_pct, 2) AS margin_pct,
        ROUND(avg_rating, 2) AS avg_rating,
        stock_quantity,
        '🌟 Keep investing' AS recommendation
    FROM product_performance
    WHERE revenue IS NOT NULL
    ORDER BY revenue DESC
    LIMIT 10
)
UNION ALL
(
    SELECT 
        'NEEDS ATTENTION',
        product_name,
        category_name,
        ROUND(COALESCE(revenue, 0), 2),
        COALESCE(units_sold, 0),
        COALESCE(orders, 0),
        ROUND(price, 2),
        ROUND(margin_pct, 2),
        ROUND(avg_rating, 2),
        stock_quantity,
        CASE 
            WHEN revenue IS NULL OR revenue = 0 THEN '🔴 No sales - Review pricing/visibility'
            WHEN units_sold < 5 THEN '⚠️ Low sales - Needs promotion'
            WHEN avg_rating < 3.5 THEN '⚠️ Poor rating - Quality review'
            ELSE '○ Monitor performance'
        END
    FROM product_performance
    ORDER BY COALESCE(revenue, 0) ASC
    LIMIT 10
);

-- ========================================
-- 6. OPERATIONAL METRICS DASHBOARD
-- Fulfillment, returns, and customer satisfaction
-- ========================================

WITH operational_metrics AS (
    SELECT 
        COUNT(DISTINCT CASE WHEN o.status = 'delivered' THEN o.order_id END) AS delivered_orders,
        COUNT(DISTINCT CASE WHEN o.status = 'shipped' THEN o.order_id END) AS shipped_orders,
        COUNT(DISTINCT CASE WHEN o.status = 'processing' THEN o.order_id END) AS processing_orders,
        COUNT(DISTINCT CASE WHEN o.status = 'cancelled' THEN o.order_id END) AS cancelled_orders,
        COUNT(DISTINCT o.order_id) AS total_orders,
        AVG(CASE WHEN o.status = 'delivered' 
            THEN DATEDIFF(o.updated_at, o.order_date) END) AS avg_delivery_days,
        COUNT(DISTINCT r.return_id) AS total_returns,
        COUNT(DISTINCT CASE WHEN r.status = 'approved' THEN r.return_id END) AS approved_returns,
        SUM(CASE WHEN r.status = 'approved' THEN r.refund_amount ELSE 0 END) AS total_refunds,
        AVG(rev.rating) AS avg_rating,
        COUNT(DISTINCT rev.review_id) AS total_reviews,
        COUNT(DISTINCT CASE WHEN rev.rating >= 4 THEN rev.review_id END) AS positive_reviews,
        COUNT(DISTINCT CASE WHEN rev.rating <= 2 THEN rev.review_id END) AS negative_reviews
    FROM orders o
    LEFT JOIN returns r ON o.order_id = r.order_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN reviews rev ON oi.product_id = rev.product_id 
        AND rev.status = 'approved'
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
)
SELECT 
    'Order Fulfillment Rate' AS metric_name,
    CONCAT(ROUND(delivered_orders * 100.0 / total_orders, 2), '%') AS value,
    CASE 
        WHEN delivered_orders * 100.0 / total_orders >= 95 THEN '🟢 Excellent'
        WHEN delivered_orders * 100.0 / total_orders >= 90 THEN '🟡 Good'
        ELSE '🔴 Needs Improvement'
    END AS status,
    'Target: 95%+' AS benchmark
FROM operational_metrics

UNION ALL

SELECT 
    'Orders In Processing',
    FORMAT(processing_orders, 0),
    CASE 
        WHEN processing_orders * 100.0 / total_orders <= 10 THEN '🟢 Normal'
        WHEN processing_orders * 100.0 / total_orders <= 20 THEN '🟡 Monitor'
        ELSE '🔴 Backlog'
    END,
    'Target: <10%'
FROM operational_metrics

UNION ALL

SELECT 
    'Avg Delivery Time',
    CONCAT(ROUND(avg_delivery_days, 1), ' days'),
    CASE 
        WHEN avg_delivery_days <= 3 THEN '🟢 Excellent'
        WHEN avg_delivery_days <= 5 THEN '🟡 Good'
        WHEN avg_delivery_days <= 7 THEN '🟡 Average'
        ELSE '🔴 Slow'
    END,
    'Target: <5 days'
FROM operational_metrics

UNION ALL

SELECT 
    'Return Rate',
    CONCAT(ROUND(total_returns * 100.0 / total_orders, 2), '%'),
    CASE 
        WHEN total_returns * 100.0 / total_orders <= 5 THEN '🟢 Excellent'
        WHEN total_returns * 100.0 / total_orders <= 10 THEN '🟡 Acceptable'
        WHEN total_returns * 100.0 / total_orders <= 15 THEN '🟡 High'
        ELSE '🔴 Critical'
    END,
    'Target: <10%'
FROM operational_metrics

UNION ALL

SELECT 
    'Cancellation Rate',
    CONCAT(ROUND(cancelled_orders * 100.0 / total_orders, 2), '%'),
    CASE 
        WHEN cancelled_orders * 100.0 / total_orders <= 2 THEN '🟢 Excellent'
        WHEN cancelled_orders * 100.0 / total_orders <= 5 THEN '🟡 Acceptable'
        ELSE '🔴 High'
    END,
    'Target: <2%'
FROM operational_metrics

UNION ALL

SELECT 
    'Customer Satisfaction',
    CONCAT(ROUND(avg_rating, 2), '/5'),
    CASE 
        WHEN avg_rating >= 4.5 THEN '🟢 Excellent'
        WHEN avg_rating >= 4.0 THEN '🟢 Good'
        WHEN avg_rating >= 3.5 THEN '🟡 Average'
        ELSE '🔴 Poor'
    END,
    'Target: 4.0+'
FROM operational_metrics

UNION ALL

SELECT 
    'Positive Review Rate',
    CONCAT(ROUND(positive_reviews * 100.0 / NULLIF(total_reviews, 0), 2), '%'),
    CASE 
        WHEN positive_reviews * 100.0 / NULLIF(total_reviews, 0) >= 80 THEN '🟢 Excellent'
        WHEN positive_reviews * 100.0 / NULLIF(total_reviews, 0) >= 70 THEN '🟡 Good'
        ELSE '🔴 Needs Improvement'
    END,
    'Target: 80%+'
FROM operational_metrics;

-- ========================================
-- 7. FINANCIAL SUMMARY
-- Revenue, costs, and profitability metrics
-- ========================================

WITH financial_data AS (
    SELECT 
        SUM(o.total_amount) AS total_revenue,
        SUM(oi.quantity * p.cost) AS total_cogs,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(CASE WHEN r.status = 'approved' THEN r.refund_amount ELSE 0 END) AS total_refunds,
        AVG(o.total_amount) AS avg_order_value
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
)
SELECT 
    'Total Revenue' AS metric,
    CONCAT('$', FORMAT(total_revenue, 2)) AS value,
    '100%' AS percentage,
    '📊 Revenue Stream' AS category
FROM financial_data

UNION ALL

SELECT 
    'Cost of Goods Sold',
    CONCAT('$', FORMAT(total_cogs, 2)),
    CONCAT(ROUND(total_cogs * 100.0 / total_revenue, 2), '%'),
    '💰 Costs'
FROM financial_data

UNION ALL

SELECT 
    'Gross Profit',
    CONCAT('$', FORMAT(total_revenue - total_cogs, 2)),
    CONCAT(ROUND((total_revenue - total_cogs) * 100.0 / total_revenue, 2), '%'),
    '💚 Profitability'
FROM financial_data

UNION ALL

SELECT 
    'Refunds Issued',
    CONCAT('$', FORMAT(total_refunds, 2)),
    CONCAT(ROUND(total_refunds * 100.0 / total_revenue, 2), '%'),
    '🔄 Returns'
FROM financial_data

UNION ALL

SELECT 
    'Net Revenue (After Refunds)',
    CONCAT('$', FORMAT(total_revenue - total_refunds, 2)),
    CONCAT(ROUND((total_revenue - total_refunds) * 100.0 / total_revenue, 2), '%'),
    '📈 Net Performance'
FROM financial_data

UNION ALL

SELECT 
    'Avg Order Value',
    CONCAT('$', FORMAT(avg_order_value, 2)),
    '-',
    '📊 Order Metrics'
FROM financial_data;

-- ========================================
-- 8. KEY ALERTS AND RECOMMENDATIONS
-- Automated alerts for executive attention
-- ========================================

WITH alert_metrics AS (
    SELECT 
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
         AND status IN ('delivered', 'shipped', 'processing')
         AND payment_status = 'paid') AS revenue_7d,
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 14 DAY) 
         AND order_date < DATE_SUB(CURDATE(), INTERVAL 7 DAY)
         AND status IN ('delivered', 'shipped', 'processing')
         AND payment_status = 'paid') AS revenue_prev_7d,
        (SELECT COUNT(*) FROM products 
         WHERE stock_quantity <= 10 AND status = 'active') AS low_stock_products,
        (SELECT COUNT(*) FROM products 
         WHERE stock_quantity = 0 AND status = 'active') AS out_of_stock_products,
        (SELECT COUNT(DISTINCT customer_id) FROM orders 
         WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS active_customers_7d,
        (SELECT COUNT(*) FROM orders 
         WHERE status = 'processing' 
         AND DATEDIFF(CURDATE(), order_date) > 3) AS delayed_orders,
        (SELECT COUNT(*) FROM returns 
         WHERE status = 'pending' 
         AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS pending_returns_7d,
        (SELECT COUNT(*) FROM reviews 
         WHERE status = 'pending') AS pending_reviews,
        (SELECT AVG(rating) FROM reviews 
         WHERE status = 'approved' 
         AND created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) AS avg_rating_30d
)
SELECT 
    CASE 
        WHEN (revenue_7d - revenue_prev_7d) / NULLIF(revenue_prev_7d, 0) < -0.15 
        THEN '🔴 CRITICAL'
        WHEN out_of_stock_products > 20 
        THEN '🔴 CRITICAL'
        WHEN low_stock_products > 50 
        THEN '🟡 WARNING'
        WHEN delayed_orders > 10 
        THEN '🟡 WARNING'
        WHEN pending_returns_7d > 20 
        THEN '🟡 WARNING'
        WHEN avg_rating_30d < 3.5 
        THEN '🔴 CRITICAL'
        ELSE '🟢 NORMAL'
    END AS alert_level,
    'Revenue Trend' AS alert_category,
    CONCAT('Week-over-week revenue change: ',
           ROUND((revenue_7d - revenue_prev_7d) / NULLIF(revenue_prev_7d, 0) * 100, 2), '%') AS alert_message,
    CASE 
        WHEN (revenue_7d - revenue_prev_7d) / NULLIF(revenue_prev_7d, 0) < -0.15 
        THEN 'Review marketing campaigns and pricing strategy'
        WHEN (revenue_7d - revenue_prev_7d) / NULLIF(revenue_prev_7d, 0) > 0.20 
        THEN 'Maintain momentum - analyze success drivers'
        ELSE 'Continue monitoring'
    END AS recommended_action
FROM alert_metrics

UNION ALL

SELECT 
    CASE 
        WHEN out_of_stock_products > 20 THEN '🔴 CRITICAL'
        WHEN out_of_stock_products > 10 THEN '🟡 WARNING'
        ELSE '🟢 NORMAL'
    END,
    'Inventory - Out of Stock',
    CONCAT(out_of_stock_products, ' active products are out of stock'),
    CASE 
        WHEN out_of_stock_products > 20 THEN 'URGENT: Expedite restocking for high-demand items'
        WHEN out_of_stock_products > 10 THEN 'Prioritize reordering for out-of-stock products'
        ELSE 'Maintain current inventory practices'
    END
FROM alert_metrics

UNION ALL

SELECT 
    CASE 
        WHEN low_stock_products > 50 THEN '🟡 WARNING'
        WHEN low_stock_products > 100 THEN '🔴 CRITICAL'
        ELSE '🟢 NORMAL'
    END,
    'Inventory - Low Stock',
    CONCAT(low_stock_products, ' products below reorder level'),
    CASE 
        WHEN low_stock_products > 50 THEN 'Review and process reorder requests'
        ELSE 'Monitor inventory levels'
    END
FROM alert_metrics

UNION ALL

SELECT 
    CASE 
        WHEN delayed_orders > 10 THEN '🟡 WARNING'
        WHEN delayed_orders > 25 THEN '🔴 CRITICAL'
        ELSE '🟢 NORMAL'
    END,
    'Operations - Delayed Orders',
    CONCAT(delayed_orders, ' orders in processing for 3+ days'),
    CASE 
        WHEN delayed_orders > 25 THEN 'URGENT: Review fulfillment bottlenecks'
        WHEN delayed_orders > 10 THEN 'Investigate processing delays'
        ELSE 'Maintain current operations'
    END
FROM alert_metrics

UNION ALL

SELECT 
    CASE 
        WHEN pending_returns_7d > 20 THEN '🟡 WARNING'
        WHEN pending_returns_7d > 50 THEN '🔴 CRITICAL'
        ELSE '🟢 NORMAL'
    END,
    'Customer Service - Returns',
    CONCAT(pending_returns_7d, ' pending return requests this week'),
    CASE 
        WHEN pending_returns_7d > 50 THEN 'URGENT: Allocate additional CS resources'
        WHEN pending_returns_7d > 20 THEN 'Expedite return processing'
        ELSE 'Continue normal processing'
    END
FROM alert_metrics

UNION ALL

SELECT 
    CASE 
        WHEN avg_rating_30d < 3.5 THEN '🔴 CRITICAL'
        WHEN avg_rating_30d < 4.0 THEN '🟡 WARNING'
        ELSE '🟢 NORMAL'
    END,
    'Quality - Customer Satisfaction',
    CONCAT('30-day average rating: ', ROUND(avg_rating_30d, 2), '/5.0'),
    CASE 
        WHEN avg_rating_30d < 3.5 THEN 'CRITICAL: Review product quality and customer feedback'
        WHEN avg_rating_30d < 4.0 THEN 'Analyze negative reviews and implement improvements'
        ELSE 'Maintain quality standards'
    END
FROM alert_metrics

UNION ALL

SELECT 
    CASE 
        WHEN pending_reviews > 100 THEN '🟡 WARNING'
        ELSE '🟢 NORMAL'
    END,
    'Content Moderation',
    CONCAT(pending_reviews, ' reviews awaiting moderation'),
    CASE 
        WHEN pending_reviews > 100 THEN 'Increase review moderation capacity'
        ELSE 'Maintain current moderation pace'
    END
FROM alert_metrics;

-- ========================================
-- 9. GROWTH OPPORTUNITIES ANALYSIS
-- Identify potential areas for business growth
-- ========================================

WITH growth_analysis AS (
    SELECT 
        'Cross-Sell Potential' AS opportunity,
        COUNT(DISTINCT o.customer_id) AS affected_customers,
        CONCAT(', FORMAT(AVG(o.total_amount) * 1.25, 2)) AS potential_value,
        'Customers buying from single category' AS description,
        '📈 High' AS priority
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
        AND o.payment_status = 'paid'
    GROUP BY o.customer_id
    HAVING COUNT(DISTINCT p.category_id) = 1
    
    UNION ALL
    
    SELECT 
        'Cart Abandonment Recovery',
        COUNT(DISTINCT customer_id),
        CONCAT(', FORMAT(SUM(total_amount), 2)),
        'Pending orders older than 24 hours',
        '🔴 Critical'
    FROM orders
    WHERE status = 'pending'
        AND created_at < DATE_SUB(NOW(), INTERVAL 1 DAY)
        AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    
    UNION ALL
    
    SELECT 
        'Loyalty Program Expansion',
        (SELECT COUNT(*) FROM customers 
         WHERE status = 'active' 
         AND customer_id NOT IN (SELECT customer_id FROM loyalty_program)),
        CONCAT(', FORMAT(AVG(total_spent) * 1.3, 2)),
        'Active customers not in loyalty program',
        '📈 High'
    FROM (
        SELECT customer_id, SUM(total_amount) AS total_spent
        FROM orders
        WHERE payment_status = 'paid'
        GROUP BY customer_id
        HAVING SUM(total_amount) > 200
    ) AS high_value_customers
    
    UNION ALL
    
    SELECT 
        'Win-Back Campaign',
        COUNT(DISTINCT customer_id),
        CONCAT(', FORMAT(AVG(historical_value), 2)),
        'Customers inactive for 90+ days',
        '🟡 Medium'
    FROM (
        SELECT 
            o.customer_id,
            MAX(o.order_date) AS last_order,
            SUM(o.total_amount) AS historical_value
        FROM orders o
        WHERE o.payment_status = 'paid'
        GROUP BY o.customer_id
        HAVING MAX(o.order_date) < DATE_SUB(CURDATE(), INTERVAL 90 DAY)
            AND SUM(o.total_amount) > 300
    ) AS churned_customers
)
SELECT * FROM growth_analysis
ORDER BY 
    CASE priority
        WHEN '🔴 Critical' THEN 1
        WHEN '📈 High' THEN 2
        WHEN '🟡 Medium' THEN 3
        ELSE 4
    END;

-- ========================================
-- 10. WEEKLY EXECUTIVE SCORECARD
-- Comprehensive weekly performance summary
-- ========================================

WITH weekly_metrics AS (
    SELECT 
        YEARWEEK(order_date, 1) AS year_week,
        CONCAT('Week ', WEEK(order_date, 1), ' ', YEAR(order_date)) AS week_label,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS avg_order_value
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 8 WEEK)
        AND status IN ('delivered', 'shipped', 'processing')
        AND payment_status = 'paid'
    GROUP BY YEARWEEK(order_date, 1), week_label
),
weekly_comparison AS (
    SELECT 
        week_label,
        revenue,
        orders,
        customers,
        avg_order_value,
        LAG(revenue) OVER (ORDER BY year_week) AS prev_revenue,
        LAG(orders) OVER (ORDER BY year_week) AS prev_orders,
        LAG(customers) OVER (ORDER BY year_week) AS prev_customers
    FROM weekly_metrics
)
SELECT 
    week_label,
    CONCAT(', FORMAT(revenue, 2)) AS revenue,
    CONCAT(FORMAT(orders, 0), ' orders') AS orders,
    CONCAT(FORMAT(customers, 0), ' customers') AS customers,
    CONCAT(', FORMAT(avg_order_value, 2)) AS avg_order_value,
    CONCAT(ROUND((revenue - prev_revenue) / NULLIF(prev_revenue, 0) * 100, 1), '%') AS revenue_wow,
    CONCAT(ROUND((orders - prev_orders) / NULLIF(prev_orders, 0) * 100, 1), '%') AS orders_wow,
    CASE 
        WHEN (revenue - prev_revenue) / NULLIF(prev_revenue, 0) >= 0.10 THEN '🚀 Accelerating'
        WHEN (revenue - prev_revenue) / NULLIF(prev_revenue, 0) >= 0.05 THEN '📈 Growing'
        WHEN (revenue - prev_revenue) / NULLIF(prev_revenue, 0) >= 0 THEN '➡️ Stable'
        WHEN (revenue - prev_revenue) / NULLIF(prev_revenue, 0) >= -0.05 THEN '⚠️ Slowing'
        ELSE '🔴 Declining'
    END AS trend
FROM weekly_comparison
ORDER BY year_week DESC
LIMIT 8;

-- ========================================
-- 11. MARKET SHARE BY CATEGORY
-- Category contribution to total business
-- ========================================

WITH category_revenue AS (
    SELECT 
        pc.category_name,
        SUM(o.total_amount) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS unique_customers,
        SUM(oi.quantity) AS units_sold
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
    GROUP BY pc.category_name
),
total_metrics AS (
    SELECT 
        SUM(revenue) AS total_revenue,
        SUM(orders) AS total_orders,
        SUM(unique_customers) AS total_customers
    FROM category_revenue
)
SELECT 
    cr.category_name,
    CONCAT(', FORMAT(cr.revenue, 2)) AS revenue,
    ROUND(cr.revenue * 100.0 / tm.total_revenue, 2) AS revenue_share_pct,
    FORMAT(cr.orders, 0) AS orders,
    ROUND(cr.orders * 100.0 / tm.total_orders, 2) AS order_share_pct,
    FORMAT(cr.unique_customers, 0) AS customers,
    FORMAT(cr.units_sold, 0) AS units_sold,
    CONCAT(', FORMAT(cr.revenue / cr.orders, 2)) AS avg_order_value,
    RANK() OVER (ORDER BY cr.revenue DESC) AS revenue_rank,
    CASE 
        WHEN cr.revenue * 100.0 / tm.total_revenue >= 25 THEN '👑 Dominant'
        WHEN cr.revenue * 100.0 / tm.total_revenue >= 15 THEN '⭐ Leading'
        WHEN cr.revenue * 100.0 / tm.total_revenue >= 10 THEN '🔥 Strong'
        WHEN cr.revenue * 100.0 / tm.total_revenue >= 5 THEN '✓ Growing'
        ELSE '○ Emerging'
    END AS market_position
FROM category_revenue cr
CROSS JOIN total_metrics tm
ORDER BY cr.revenue DESC;

-- ========================================
-- 12. CUSTOMER ACQUISITION COST (CAC) ANALYSIS
-- Marketing efficiency metrics
-- ========================================

WITH marketing_spend AS (
    SELECT 
        SUM(cp.spend) AS total_spend,
        SUM(cp.conversions) AS total_conversions
    FROM campaign_performance cp
    WHERE cp.report_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
),
new_customers AS (
    SELECT 
        COUNT(DISTINCT customer_id) AS new_customer_count,
        SUM(total_amount) AS new_customer_revenue
    FROM orders
    WHERE customer_id IN (
        SELECT customer_id 
        FROM customers 
        WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
    )
    AND payment_status = 'paid'
    AND order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
)
SELECT 
    'Customer Acquisition Cost (CAC)' AS metric,
    CONCAT(', FORMAT(ms.total_spend / NULLIF(nc.new_customer_count, 0), 2)) AS value,
    CASE 
        WHEN ms.total_spend / NULLIF(nc.new_customer_count, 0) < 50 THEN '🟢 Excellent'
        WHEN ms.total_spend / NULLIF(nc.new_customer_count, 0) < 100 THEN '🟡 Good'
        WHEN ms.total_spend / NULLIF(nc.new_customer_count, 0) < 150 THEN '🟡 Acceptable'
        ELSE '🔴 High'
    END AS assessment,
    'Target: <$50' AS benchmark
FROM marketing_spend ms
CROSS JOIN new_customers nc

UNION ALL

SELECT 
    'New Customers Acquired',
    FORMAT(nc.new_customer_count, 0),
    CASE 
        WHEN nc.new_customer_count >= 100 THEN '🟢 Strong'
        WHEN nc.new_customer_count >= 50 THEN '🟡 Moderate'
        ELSE '🔴 Low'
    END,
    'Target: 100+/month'
FROM new_customers nc

UNION ALL

SELECT 
    'CAC Payback Period (months)',
    FORMAT(
        (ms.total_spend / NULLIF(nc.new_customer_count, 0)) / 
        NULLIF((nc.new_customer_revenue / NULLIF(nc.new_customer_count, 0)), 0), 1
    ),
    CASE 
        WHEN (ms.total_spend / NULLIF(nc.new_customer_count, 0)) / 
             NULLIF((nc.new_customer_revenue / NULLIF(nc.new_customer_count, 0)), 0) < 3 THEN '🟢 Fast'
        WHEN (ms.total_spend / NULLIF(nc.new_customer_count, 0)) / 
             NULLIF((nc.new_customer_revenue / NULLIF(nc.new_customer_count, 0)), 0) < 6 THEN '🟡 Moderate'
        ELSE '🔴 Slow'
    END,
    'Target: <3 months'
FROM marketing_spend ms
CROSS JOIN new_customers nc

UNION ALL

SELECT 
    'Marketing ROI',
    CONCAT(FORMAT(
        (nc.new_customer_revenue - ms.total_spend) / NULLIF(ms.total_spend, 0) * 100, 2
    ), '%'),
    CASE 
        WHEN (nc.new_customer_revenue - ms.total_spend) / NULLIF(ms.total_spend, 0) >= 2 THEN '🟢 Excellent'
        WHEN (nc.new_customer_revenue - ms.total_spend) / NULLIF(ms.total_spend, 0) >= 1 THEN '🟡 Good'
        WHEN (nc.new_customer_revenue - ms.total_spend) / NULLIF(ms.total_spend, 0) >= 0 THEN '🟡 Break-even'
        ELSE '🔴 Negative'
    END,
    'Target: 200%+'
FROM marketing_spend ms
CROSS JOIN new_customers nc;

-- ========================================
-- 13. EXECUTIVE SUMMARY - SINGLE VIEW
-- Complete business snapshot
-- ========================================

SELECT 
    '═══════════════════════════════════════' AS separator,
    'EXECUTIVE DASHBOARD SUMMARY' AS title,
    DATE_FORMAT(CURDATE(), '%W, %M %d, %Y') AS report_date,
    '═══════════════════════════════════════' AS separator2

UNION ALL

SELECT 
    '📊 FINANCIAL PERFORMANCE', '', '', ''
FROM (
    SELECT 
        CONCAT('Revenue (MTD): , FORMAT(SUM(total_amount), 2)) AS metric1,
        CONCAT('Orders: ', FORMAT(COUNT(*), 0)) AS metric2,
        CONCAT('AOV: , FORMAT(AVG(total_amount), 2)) AS metric3,
        '' AS metric4
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
        AND payment_status = 'paid'
        AND status IN ('delivered', 'shipped', 'processing')
) AS financial

UNION ALL

SELECT 
    '👥 CUSTOMER METRICS', '', '', ''
FROM (
    SELECT 
        CONCAT('Active Customers: ', FORMAT(COUNT(*), 0)) AS metric1,
        '' AS metric2, '' AS metric3, '' AS metric4
    FROM customers
    WHERE status = 'active'
) AS customers

UNION ALL

SELECT 
    '📦 OPERATIONAL STATUS', '', '', ''
FROM (
    SELECT 
        CONCAT('Products: ', FORMAT(COUNT(*), 0)) AS metric1,
        CONCAT('Low Stock: ', FORMAT(SUM(CASE WHEN stock_quantity <= 10 THEN 1 ELSE 0 END), 0)) AS metric2,
        '' AS metric3, '' AS metric4
    FROM products
    WHERE status = 'active'
) AS operations

UNION ALL

SELECT 
    '⭐ QUALITY METRICS', '', '', ''
FROM (
    SELECT 
        CONCAT('Avg Rating: ', FORMAT(AVG(rating), 2), '/5') AS metric1,
        CONCAT('Return Rate: ', FORMAT(
            COUNT(DISTINCT r.return_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0), 2
        ), '%') AS metric2,
        '' AS metric3, '' AS metric4
    FROM orders o
    LEFT JOIN returns r ON o.order_id = r.order_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN reviews rev ON oi.product_id = rev.product_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
) AS quality;

-- ========================================
-- COMPLETION MESSAGE
-- ========================================

SELECT 
    '✅ Executive Summary Generated Successfully' AS status,
    CONCAT('Report Date: ', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')) AS timestamp,
    'All metrics calculated for management review' AS message;