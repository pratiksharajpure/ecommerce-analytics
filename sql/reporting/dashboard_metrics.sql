-- ========================================
-- DASHBOARD METRICS SQL
-- Real-Time Dashboard Data & Snapshot Metrics
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. REAL-TIME KPI CARDS
-- Core metrics for dashboard header
-- ========================================

-- Today's Revenue vs Yesterday
SELECT 
    'TODAY_REVENUE' AS metric_id,
    'Today\'s Revenue' AS metric_name,
    CONCAT('$', FORMAT(
        COALESCE(SUM(CASE 
            WHEN DATE(order_date) = CURDATE() 
            AND payment_status = 'paid'
            THEN total_amount 
        END), 0), 2
    )) AS current_value,
    CONCAT('$', FORMAT(
        COALESCE(SUM(CASE 
            WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
            AND payment_status = 'paid'
            THEN total_amount 
        END), 0), 2
    )) AS previous_value,
    ROUND(
        (COALESCE(SUM(CASE WHEN DATE(order_date) = CURDATE() AND payment_status = 'paid' THEN total_amount END), 0) -
         COALESCE(SUM(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) AND payment_status = 'paid' THEN total_amount END), 0)) /
        NULLIF(COALESCE(SUM(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) AND payment_status = 'paid' THEN total_amount END), 0), 0) * 100, 2
    ) AS change_pct,
    CASE 
        WHEN COALESCE(SUM(CASE WHEN DATE(order_date) = CURDATE() AND payment_status = 'paid' THEN total_amount END), 0) > 
             COALESCE(SUM(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) AND payment_status = 'paid' THEN total_amount END), 0)
        THEN 'up'
        ELSE 'down'
    END AS trend_direction,
    'primary' AS card_color,
    ''MONEY'' AS icon
FROM orders
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 DAY)

UNION ALL

-- Today's Orders
SELECT 
    'TODAY_ORDERS',
    'Today\'s Orders',
    FORMAT(COUNT(CASE WHEN DATE(order_date) = CURDATE() THEN 1 END), 0),
    FORMAT(COUNT(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) THEN 1 END), 0),
    ROUND(
        (COUNT(CASE WHEN DATE(order_date) = CURDATE() THEN 1 END) -
         COUNT(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) THEN 1 END)) /
        NULLIF(COUNT(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) THEN 1 END), 0) * 100, 2
    ),
    CASE 
        WHEN COUNT(CASE WHEN DATE(order_date) = CURDATE() THEN 1 END) > 
             COUNT(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) THEN 1 END)
        THEN 'up'
        ELSE 'down'
    END,
    'success',
    '📦'
FROM orders

UNION ALL

-- Active Customers (Last 24 Hours)
SELECT 
    'ACTIVE_CUSTOMERS_24H',
    'Active Customers (24h)',
    FORMAT(COUNT(DISTINCT CASE 
        WHEN order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR) 
        THEN customer_id 
    END), 0),
    FORMAT(COUNT(DISTINCT CASE 
        WHEN order_date >= DATE_SUB(NOW(), INTERVAL 48 HOUR)
        AND order_date < DATE_SUB(NOW(), INTERVAL 24 HOUR)
        THEN customer_id 
    END), 0),
    ROUND(
        (COUNT(DISTINCT CASE WHEN order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR) THEN customer_id END) -
         COUNT(DISTINCT CASE WHEN order_date >= DATE_SUB(NOW(), INTERVAL 48 HOUR) AND order_date < DATE_SUB(NOW(), INTERVAL 24 HOUR) THEN customer_id END)) /
        NULLIF(COUNT(DISTINCT CASE WHEN order_date >= DATE_SUB(NOW(), INTERVAL 48 HOUR) AND order_date < DATE_SUB(NOW(), INTERVAL 24 HOUR) THEN customer_id END), 0) * 100, 2
    ),
    CASE 
        WHEN COUNT(DISTINCT CASE WHEN order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR) THEN customer_id END) > 
             COUNT(DISTINCT CASE WHEN order_date >= DATE_SUB(NOW(), INTERVAL 48 HOUR) AND order_date < DATE_SUB(NOW(), INTERVAL 24 HOUR) THEN customer_id END)
        THEN 'up'
        ELSE 'down'
    END,
    'info',
    '👥'
FROM orders

UNION ALL

-- Average Order Value (Today)
SELECT 
    'TODAY_AOV',
    'Avg Order Value (Today)',
    CONCAT('$', FORMAT(AVG(CASE 
        WHEN DATE(order_date) = CURDATE() 
        AND payment_status = 'paid'
        THEN total_amount 
    END), 2)),
    CONCAT('$', FORMAT(AVG(CASE 
        WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
        AND payment_status = 'paid'
        THEN total_amount 
    END), 2)),
    ROUND(
        (AVG(CASE WHEN DATE(order_date) = CURDATE() AND payment_status = 'paid' THEN total_amount END) -
         AVG(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) AND payment_status = 'paid' THEN total_amount END)) /
        NULLIF(AVG(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) AND payment_status = 'paid' THEN total_amount END), 0) * 100, 2
    ),
    CASE 
        WHEN AVG(CASE WHEN DATE(order_date) = CURDATE() AND payment_status = 'paid' THEN total_amount END) > 
             AVG(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY) AND payment_status = 'paid' THEN total_amount END)
        THEN 'up'
        ELSE 'down'
    END,
    'warning',
    '💳'
FROM orders;

-- ========================================
-- 2. HOURLY PERFORMANCE (Last 24 Hours)
-- Real-time hourly breakdown
-- ========================================

SELECT 
    DATE_FORMAT(order_date, '%Y-%m-%d %H:00:00') AS hour_timestamp,
    DATE_FORMAT(order_date, '%h %p') AS hour_label,
    COUNT(DISTINCT order_id) AS orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(CASE WHEN payment_status = 'paid' THEN total_amount ELSE 0 END) AS revenue,
    AVG(CASE WHEN payment_status = 'paid' THEN total_amount END) AS avg_order_value,
    SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered_orders,
    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing_orders,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    -- Moving average
    AVG(COUNT(DISTINCT order_id)) OVER (
        ORDER BY DATE_FORMAT(order_date, '%Y-%m-%d %H:00:00')
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS orders_3hr_avg
FROM orders
WHERE order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY DATE_FORMAT(order_date, '%Y-%m-%d %H:00:00'), DATE_FORMAT(order_date, '%h %p')
ORDER BY hour_timestamp DESC;

-- ========================================
-- 3. REAL-TIME ORDER STATUS BREAKDOWN
-- Current order pipeline status
-- ========================================

SELECT 
    status AS order_status,
    COUNT(*) AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS total_value,
    COUNT(DISTINCT customer_id) AS unique_customers,
    CASE status
        WHEN 'pending' THEN '⏳'
        WHEN 'processing' THEN '🔄'
        WHEN 'shipped' THEN '🚚'
        WHEN 'delivered' THEN ''SUCCESS''
        WHEN 'cancelled' THEN ''ERROR''
        ELSE '❓'
    END AS status_icon,
    CASE status
        WHEN 'pending' THEN 'warning'
        WHEN 'processing' THEN 'info'
        WHEN 'shipped' THEN 'primary'
        WHEN 'delivered' THEN 'success'
        WHEN 'cancelled' THEN 'danger'
        ELSE 'secondary'
    END AS badge_color,
    -- Age of orders in this status
    AVG(TIMESTAMPDIFF(HOUR, created_at, NOW())) AS avg_age_hours
FROM orders
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
GROUP BY status
ORDER BY 
    FIELD(status, 'pending', 'processing', 'shipped', 'delivered', 'cancelled');

-- ========================================
-- 4. TOP PRODUCTS - REAL-TIME
-- Best sellers in last 24 hours
-- ========================================

SELECT 
    p.product_id,
    p.product_name,
    p.sku,
    pc.category_name,
    COUNT(DISTINCT oi.order_id) AS orders_24h,
    SUM(oi.quantity) AS units_sold_24h,
    CONCAT('$', FORMAT(SUM(oi.subtotal), 2)) AS revenue_24h,
    CONCAT('$', FORMAT(p.price, 2)) AS current_price,
    p.stock_quantity,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    -- Velocity (units per hour)
    ROUND(SUM(oi.quantity) / 24.0, 2) AS units_per_hour,
    CASE 
        WHEN p.stock_quantity = 0 THEN ''RED' Out of Stock'
        WHEN p.stock_quantity <= 10 THEN ''YELLOW' Low Stock'
        WHEN SUM(oi.quantity) / 24.0 > 5 THEN '🔥 Hot Item'
        ELSE ''GREEN' Available'
    END AS stock_status
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN products p ON oi.product_id = p.product_id
JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN reviews r ON p.product_id = r.product_id AND r.status = 'approved'
WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    AND o.payment_status = 'paid'
GROUP BY p.product_id, p.product_name, p.sku, pc.category_name, p.price, p.stock_quantity
ORDER BY revenue_24h DESC
LIMIT 10;

-- ========================================
-- 5. REVENUE SNAPSHOT - MULTIPLE TIMEFRAMES
-- Quick revenue comparison across periods
-- ========================================

WITH revenue_metrics AS (
    SELECT 
        SUM(CASE WHEN order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR) 
            AND payment_status = 'paid' THEN total_amount ELSE 0 END) AS last_hour,
        SUM(CASE WHEN DATE(order_date) = CURDATE() 
            AND payment_status = 'paid' THEN total_amount ELSE 0 END) AS today,
        SUM(CASE WHEN DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
            AND payment_status = 'paid' THEN total_amount ELSE 0 END) AS yesterday,
        SUM(CASE WHEN order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
            AND payment_status = 'paid' THEN total_amount ELSE 0 END) AS last_7_days,
        SUM(CASE WHEN order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
            AND payment_status = 'paid' THEN total_amount ELSE 0 END) AS last_30_days,
        SUM(CASE WHEN YEAR(order_date) = YEAR(CURDATE())
            AND payment_status = 'paid' THEN total_amount ELSE 0 END) AS ytd
    FROM orders
)
SELECT 
    'Last Hour' AS time_period,
    CONCAT('$', FORMAT(last_hour, 2)) AS revenue,
    CONCAT('$', FORMAT(last_hour / 1, 2)) AS avg_per_period,
    1 AS period_hours
FROM revenue_metrics

UNION ALL

SELECT 
    'Today',
    CONCAT('$', FORMAT(today, 2)),
    CONCAT('$', FORMAT(today / NULLIF(HOUR(NOW()), 0), 2)),
    HOUR(NOW())
FROM revenue_metrics

UNION ALL

SELECT 
    'Yesterday',
    CONCAT('$', FORMAT(yesterday, 2)),
    CONCAT('$', FORMAT(yesterday / 24, 2)),
    24
FROM revenue_metrics

UNION ALL

SELECT 
    'Last 7 Days',
    CONCAT('$', FORMAT(last_7_days, 2)),
    CONCAT('$', FORMAT(last_7_days / 7, 2)),
    168
FROM revenue_metrics

UNION ALL

SELECT 
    'Last 30 Days',
    CONCAT('$', FORMAT(last_30_days, 2)),
    CONCAT('$', FORMAT(last_30_days / 30, 2)),
    720
FROM revenue_metrics

UNION ALL

SELECT 
    'Year to Date',
    CONCAT('$', FORMAT(ytd, 2)),
    CONCAT('$', FORMAT(ytd / DAYOFYEAR(CURDATE()), 2)),
    DAYOFYEAR(CURDATE()) * 24
FROM revenue_metrics;

-- ========================================
-- 6. CUSTOMER ACTIVITY - REAL-TIME
-- Live customer engagement metrics
-- ========================================

SELECT 
    'New Customers Today' AS metric,
    COUNT(*) AS count,
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(o.total_amount) 
         FROM orders o 
         WHERE o.customer_id = c.customer_id 
         AND o.payment_status = 'paid'), 0)
    / COUNT(*), 2)) AS avg_value_per_customer,
    '👤' AS icon,
    'success' AS color
FROM customers c
WHERE DATE(c.created_at) = CURDATE()

UNION ALL

SELECT 
    'Repeat Purchasers (24h)',
    COUNT(DISTINCT customer_id),
    CONCAT('$', FORMAT(AVG(total_amount), 2)),
    '🔄',
    'primary'
FROM orders
WHERE order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    AND customer_id IN (
        SELECT customer_id 
        FROM orders 
        GROUP BY customer_id 
        HAVING COUNT(*) > 1
    )
    AND payment_status = 'paid'

UNION ALL

SELECT 
    'First-Time Buyers (24h)',
    COUNT(DISTINCT o.customer_id),
    CONCAT('$', FORMAT(AVG(o.total_amount), 2)),
    '🎉',
    'info'
FROM orders o
WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    AND o.customer_id IN (
        SELECT customer_id 
        FROM orders 
        GROUP BY customer_id 
        HAVING COUNT(*) = 1
    )
    AND o.payment_status = 'paid'

UNION ALL

SELECT 
    'VIP Customers Active Today',
    COUNT(DISTINCT o.customer_id),
    CONCAT('$', FORMAT(AVG(o.total_amount), 2)),
    '👑',
    'warning'
FROM orders o
WHERE DATE(o.order_date) = CURDATE()
    AND o.customer_id IN (
        SELECT customer_id 
        FROM orders 
        WHERE payment_status = 'paid'
        GROUP BY customer_id 
        HAVING SUM(total_amount) >= 1000
    );

-- ========================================
-- 7. CONVERSION FUNNEL - REAL-TIME
-- Order lifecycle metrics
-- ========================================

WITH funnel_data AS (
    SELECT 
        COUNT(*) AS total_orders,
        SUM(CASE WHEN payment_status = 'paid' THEN 1 ELSE 0 END) AS paid_orders,
        SUM(CASE WHEN status IN ('processing', 'shipped', 'delivered') THEN 1 ELSE 0 END) AS fulfilled_orders,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered_orders,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
)
SELECT 
    'Orders Created' AS funnel_stage,
    total_orders AS count,
    100.00 AS percentage,
    total_orders AS remaining,
    0 AS drop_off
FROM funnel_data

UNION ALL

SELECT 
    'Payment Completed',
    paid_orders,
    ROUND(paid_orders * 100.0 / total_orders, 2),
    paid_orders,
    total_orders - paid_orders
FROM funnel_data

UNION ALL

SELECT 
    'Order Fulfillment',
    fulfilled_orders,
    ROUND(fulfilled_orders * 100.0 / total_orders, 2),
    fulfilled_orders,
    paid_orders - fulfilled_orders
FROM funnel_data

UNION ALL

SELECT 
    'Successfully Delivered',
    delivered_orders,
    ROUND(delivered_orders * 100.0 / total_orders, 2),
    delivered_orders,
    fulfilled_orders - delivered_orders
FROM funnel_data

UNION ALL

SELECT 
    'Cancelled/Failed',
    cancelled_orders,
    ROUND(cancelled_orders * 100.0 / total_orders, 2),
    0,
    cancelled_orders
FROM funnel_data;

-- ========================================
-- 8. INVENTORY ALERTS - REAL-TIME
-- Critical stock level notifications
-- ========================================

SELECT 
    p.product_id,
    p.product_name,
    p.sku,
    pc.category_name,
    p.stock_quantity,
    i.reorder_level,
    i.quantity_reserved,
    i.quantity_available,
    COALESCE(SUM(oi.quantity), 0) AS sold_last_7_days,
    ROUND(COALESCE(SUM(oi.quantity), 0) / 7.0, 2) AS daily_velocity,
    ROUND(p.stock_quantity / NULLIF(COALESCE(SUM(oi.quantity), 0) / 7.0, 0), 1) AS days_until_stockout,
    CASE 
        WHEN p.stock_quantity = 0 THEN ''RED' OUT OF STOCK'
        WHEN p.stock_quantity <= i.reorder_level THEN ''YELLOW' REORDER NOW'
        WHEN p.stock_quantity / NULLIF(COALESCE(SUM(oi.quantity), 0) / 7.0, 0) <= 7 THEN '🟠 LOW STOCK'
        ELSE ''GREEN' ADEQUATE'
    END AS alert_status,
    CASE 
        WHEN p.stock_quantity = 0 THEN 1
        WHEN p.stock_quantity <= i.reorder_level THEN 2
        WHEN p.stock_quantity / NULLIF(COALESCE(SUM(oi.quantity), 0) / 7.0, 0) <= 7 THEN 3
        ELSE 4
    END AS priority_order
FROM products p
JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN inventory i ON p.product_id = i.product_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id 
    AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    AND o.payment_status = 'paid'
WHERE p.status = 'active'
GROUP BY p.product_id, p.product_name, p.sku, pc.category_name, 
         p.stock_quantity, i.reorder_level, i.quantity_reserved, i.quantity_available
HAVING alert_status != ''GREEN' ADEQUATE'
ORDER BY priority_order, days_until_stockout;

-- ========================================
-- 9. PAYMENT STATUS MONITORING
-- Real-time payment health
-- ========================================

SELECT 
    payment_status,
    COUNT(*) AS transaction_count,
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS total_value,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    AVG(TIMESTAMPDIFF(MINUTE, created_at, updated_at)) AS avg_processing_minutes,
    CASE payment_status
        WHEN 'paid' THEN ''SUCCESS' Successful'
        WHEN 'pending' THEN '⏳ Processing'
        WHEN 'failed' THEN ''ERROR' Failed'
        WHEN 'refunded' THEN '🔄 Refunded'
        ELSE '❓ Unknown'
    END AS status_label,
    CASE payment_status
        WHEN 'paid' THEN 'success'
        WHEN 'pending' THEN 'warning'
        WHEN 'failed' THEN 'danger'
        WHEN 'refunded' THEN 'info'
        ELSE 'secondary'
    END AS badge_color
FROM orders
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 DAY)
GROUP BY payment_status
ORDER BY 
    FIELD(payment_status, 'paid', 'pending', 'failed', 'refunded');

-- ========================================
-- 10. GEOGRAPHIC PERFORMANCE - REAL-TIME
-- Top performing regions/states
-- ========================================

SELECT 
    COALESCE(c.state, 'Unknown') AS state,
    COUNT(DISTINCT o.order_id) AS orders,
    COUNT(DISTINCT o.customer_id) AS unique_customers,
    CONCAT('$', FORMAT(SUM(o.total_amount), 2)) AS revenue,
    CONCAT('$', FORMAT(AVG(o.total_amount), 2)) AS avg_order_value,
    ROUND(SUM(o.total_amount) * 100.0 / 
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) 
         AND payment_status = 'paid'), 2) AS revenue_share_pct,
    RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS revenue_rank
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    AND o.payment_status = 'paid'
    AND c.state IS NOT NULL
GROUP BY c.state
ORDER BY revenue DESC
LIMIT 10;

-- ========================================
-- 11. CATEGORY PERFORMANCE SNAPSHOT
-- Quick category health check
-- ========================================

SELECT 
    pc.category_name,
    COUNT(DISTINCT oi.order_id) AS orders_today,
    SUM(oi.quantity) AS units_sold_today,
    CONCAT('$', FORMAT(SUM(oi.subtotal), 2)) AS revenue_today,
    CONCAT('$', FORMAT(AVG(oi.unit_price), 2)) AS avg_selling_price,
    COUNT(DISTINCT p.product_id) AS active_products,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    -- Compare to yesterday
    ROUND(
        (SUM(oi.subtotal) - 
         COALESCE((SELECT SUM(oi2.subtotal) 
                   FROM order_items oi2 
                   JOIN orders o2 ON oi2.order_id = o2.order_id
                   JOIN products p2 ON oi2.product_id = p2.product_id
                   WHERE p2.category_id = pc.category_id
                   AND DATE(o2.order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
                   AND o2.payment_status = 'paid'), 0)) /
        NULLIF(COALESCE((SELECT SUM(oi2.subtotal) 
                         FROM order_items oi2 
                         JOIN orders o2 ON oi2.order_id = o2.order_id
                         JOIN products p2 ON oi2.product_id = p2.product_id
                         WHERE p2.category_id = pc.category_id
                         AND DATE(o2.order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
                         AND o2.payment_status = 'paid'), 0), 0) * 100, 2
    ) AS change_vs_yesterday_pct,
    CASE 
        WHEN SUM(oi.subtotal) > 
             COALESCE((SELECT SUM(oi2.subtotal) 
                       FROM order_items oi2 
                       JOIN orders o2 ON oi2.order_id = o2.order_id
                       JOIN products p2 ON oi2.product_id = p2.product_id
                       WHERE p2.category_id = pc.category_id
                       AND DATE(o2.order_date) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
                       AND o2.payment_status = 'paid'), 0)
        THEN ''TRENDING_UP' Up'
        ELSE ''TRENDING_DOWN' Down'
    END AS trend
FROM product_categories pc
JOIN products p ON pc.category_id = p.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id 
    AND DATE(o.order_date) = CURDATE()
    AND o.payment_status = 'paid'
LEFT JOIN reviews r ON p.product_id = r.product_id 
    AND r.status = 'approved'
WHERE p.status = 'active'
GROUP BY pc.category_id, pc.category_name
ORDER BY revenue_today DESC;

-- ========================================
-- 12. RETURNS & REFUNDS - REAL-TIME
-- Current returns status
-- ========================================

SELECT 
    r.status AS return_status,
    COUNT(*) AS return_count,
    CONCAT('$', FORMAT(SUM(r.refund_amount), 2)) AS refund_value,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    AVG(TIMESTAMPDIFF(HOUR, r.created_at, NOW())) AS avg_age_hours,
    CASE r.status
        WHEN 'requested' THEN '📝 Requested'
        WHEN 'approved' THEN ''SUCCESS' Approved'
        WHEN 'rejected' THEN ''ERROR' Rejected'
        WHEN 'received' THEN '📦 Received'
        WHEN 'refunded' THEN ''MONEY' Refunded'
        ELSE '❓ Unknown'
    END AS status_label,
    CASE r.status
        WHEN 'requested' THEN 'warning'
        WHEN 'approved' THEN 'info'
        WHEN 'rejected' THEN 'danger'
        WHEN 'received' THEN 'primary'
        WHEN 'refunded' THEN 'success'
        ELSE 'secondary'
    END AS badge_color
FROM returns r
WHERE r.created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
GROUP BY r.status
ORDER BY 
    FIELD(r.status, 'requested', 'approved', 'received', 'refunded', 'rejected');

-- ========================================
-- 13. CAMPAIGN PERFORMANCE - LIVE
-- Active marketing campaigns snapshot
-- ========================================

SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.status AS campaign_status,
    DATEDIFF(c.end_date, CURDATE()) AS days_remaining,
    CONCAT('$', FORMAT(cp.spend, 2)) AS spend_today,
    CONCAT('$', FORMAT(cp.revenue, 2)) AS revenue_today,
    cp.impressions AS impressions_today,
    cp.clicks AS clicks_today,
    cp.conversions AS conversions_today,
    ROUND(cp.clicks * 100.0 / NULLIF(cp.impressions, 0), 2) AS ctr_pct,
    ROUND(cp.conversions * 100.0 / NULLIF(cp.clicks, 0), 2) AS conversion_rate_pct,
    ROUND((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100, 2) AS roi_pct,
    CASE 
        WHEN (cp.revenue - cp.spend) / NULLIF(cp.spend, 0) >= 2 THEN ''GREEN' Excellent'
        WHEN (cp.revenue - cp.spend) / NULLIF(cp.spend, 0) >= 1 THEN ''YELLOW' Good'
        WHEN (cp.revenue - cp.spend) / NULLIF(cp.spend, 0) >= 0 THEN '🟠 Break-even'
        ELSE ''RED' Loss'
    END AS performance_status
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    AND cp.report_date = CURDATE()
WHERE c.status = 'active'
ORDER BY roi_pct DESC;

-- ========================================
-- 14. REVIEWS & RATINGS - REAL-TIME
-- Latest customer feedback snapshot
-- ========================================

SELECT 
    'Reviews Submitted Today' AS metric,
    COUNT(*) AS count,
    ROUND(AVG(rating), 2) AS avg_rating,
    '⭐' AS icon
FROM reviews
WHERE DATE(created_at) = CURDATE()

UNION ALL

SELECT 
    'Pending Moderation',
    COUNT(*),
    ROUND(AVG(rating), 2),
    '⏳'
FROM reviews
WHERE status = 'pending'

UNION ALL

SELECT 
    'Approved Today',
    COUNT(*),
    ROUND(AVG(rating), 2),
    ''SUCCESS''
FROM reviews
WHERE status = 'approved'
    AND DATE(created_at) = CURDATE()

UNION ALL

SELECT 
    '5-Star Reviews (7d)',
    COUNT(*),
    5.00,
    '🌟'
FROM reviews
WHERE rating = 5
    AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    AND status = 'approved'

UNION ALL

SELECT 
    'Negative Reviews (7d)',
    COUNT(*),
    ROUND(AVG(rating), 2),
    ''WARNING''
FROM reviews
WHERE rating <= 2
    AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    AND status = 'approved';

-- ========================================
-- 15. SYSTEM HEALTH INDICATORS
-- Database and operational metrics
-- ========================================

SELECT 
    'Total Customers' AS metric_name,
    FORMAT(COUNT(*), 0) AS current_value,
    FORMAT(COUNT(CASE WHEN status = 'active' THEN 1 END), 0) AS active_count,
    ROUND(COUNT(CASE WHEN status = 'active' THEN 1 END) * 100.0 / COUNT(*), 2) AS active_pct,
    '👥' AS icon,
    'info' AS color
FROM customers

UNION ALL

SELECT 
    'Total Products',
    FORMAT(COUNT(*), 0),
    FORMAT(COUNT(CASE WHEN status = 'active' THEN 1 END), 0),
    ROUND(COUNT(CASE WHEN status = 'active' THEN 1 END) * 100.0 / COUNT(*), 2),
    '📦',
    'primary'
FROM products

UNION ALL

SELECT 
    'Total Orders',
    FORMAT(COUNT(*), 0),
    FORMAT(COUNT(CASE WHEN payment_status = 'paid' THEN 1 END), 0),
    ROUND(COUNT(CASE WHEN payment_status = 'paid' THEN 1 END) * 100.0 / COUNT(*), 2),
    '🛒',
    'success'
FROM orders

UNION ALL

SELECT 
    'Active Campaigns',
    FORMAT(COUNT(*), 0),
    FORMAT(COUNT(CASE WHEN status = 'active' THEN 1 END), 0),
    ROUND(COUNT(CASE WHEN status = 'active' THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 2),
    '📢',
    'warning'
FROM campaigns

UNION ALL

SELECT 
    'Loyalty Members',
    FORMAT(COUNT(*), 0),
    FORMAT(COUNT(CASE WHEN tier IN ('gold', 'platinum') THEN 1 END), 0),
    ROUND(COUNT(CASE WHEN tier IN ('gold', 'platinum') THEN 1 END) * 100.0 / COUNT(*), 2),
    '👑',
    'warning'
FROM loyalty_program;

-- ========================================
-- 16. LIVE ORDER FEED
-- Most recent orders (Last 20)
-- ========================================

SELECT 
    o.order_id,
    o.order_date,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email AS customer_email,
    CONCAT(', FORMAT(o.total_amount, 2)) AS order_value,
    o.status AS order_status,
    o.payment_status,
    COUNT(oi.order_item_id) AS item_count,
    TIMESTAMPDIFF(MINUTE, o.created_at, NOW()) AS minutes_ago,
    CASE 
        WHEN o.status = 'delivered' THEN ''SUCCESS''
        WHEN o.status = 'shipped' THEN '🚚'
        WHEN o.status = 'processing' THEN '🔄'
        WHEN o.status = 'pending' THEN '⏳'
        WHEN o.status = 'cancelled' THEN ''ERROR''
        ELSE '❓'
    END AS status_icon,
    CASE o.status
        WHEN 'delivered' THEN 'success'
        WHEN 'shipped' THEN 'primary'
        WHEN 'processing' THEN 'info'
        WHEN 'pending' THEN 'warning'
        WHEN 'cancelled' THEN 'danger'
        ELSE 'secondary'
    END AS badge_color
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.order_date, c.first_name, c.last_name, 
         c.email, o.total_amount, o.status, o.payment_status, o.created_at
ORDER BY o.order_date DESC
LIMIT 20;

-- ========================================
-- 17. QUICK STATS - DASHBOARD SUMMARY
-- Single row summary for quick view
-- ========================================

SELECT 
    -- Today metrics
    (SELECT COUNT(*) FROM orders WHERE DATE(order_date) = CURDATE()) AS orders_today,
    (SELECT CONCAT(', FORMAT(SUM(total_amount), 2)) 
     FROM orders 
     WHERE DATE(order_date) = CURDATE() 
     AND payment_status = 'paid') AS revenue_today,
    
    -- Customer metrics
    (SELECT COUNT(*) FROM customers WHERE status = 'active') AS active_customers,
    (SELECT COUNT(*) FROM customers WHERE DATE(created_at) = CURDATE()) AS new_customers_today,
    
    -- Product metrics
    (SELECT COUNT(*) FROM products WHERE status = 'active') AS active_products,
    (SELECT COUNT(*) FROM products WHERE stock_quantity = 0 AND status = 'active') AS out_of_stock,
    (SELECT COUNT(*) FROM products WHERE stock_quantity <= 10 AND status = 'active') AS low_stock,
    
    -- Order pipeline
    (SELECT COUNT(*) FROM orders WHERE status = 'pending') AS pending_orders,
    (SELECT COUNT(*) FROM orders WHERE status = 'processing') AS processing_orders,
    (SELECT COUNT(*) FROM orders WHERE status = 'shipped') AS shipped_orders,
    
    -- Returns
    (SELECT COUNT(*) FROM returns WHERE status = 'requested') AS pending_returns,
    
    -- Reviews
    (SELECT COUNT(*) FROM reviews WHERE status = 'pending') AS pending_reviews,
    (SELECT ROUND(AVG(rating), 2) 
     FROM reviews 
     WHERE status = 'approved' 
     AND created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS avg_rating_7d,
    
    -- Campaign
    (SELECT COUNT(*) FROM campaigns WHERE status = 'active') AS active_campaigns,
    
    -- Timestamp
    NOW() AS snapshot_time;

-- ========================================
-- 18. PERFORMANCE INDICATORS - 24H
-- Key performance metrics for last 24 hours
-- ========================================

WITH metrics_24h AS (
    SELECT 
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS customers,
        SUM(o.total_amount) AS revenue,
        AVG(o.total_amount) AS avg_order_value,
        SUM(oi.quantity) AS units_sold,
        COUNT(DISTINCT CASE WHEN o.status = 'delivered' THEN o.order_id END) AS delivered,
        COUNT(DISTINCT CASE WHEN o.status = 'cancelled' THEN o.order_id END) AS cancelled,
        AVG(CASE WHEN o.status = 'delivered' 
            THEN TIMESTAMPDIFF(HOUR, o.order_date, o.updated_at) END) AS avg_delivery_hours
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
),
metrics_prev_24h AS (
    SELECT 
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT o.customer_id) AS customers,
        SUM(o.total_amount) AS revenue,
        AVG(o.total_amount) AS avg_order_value
    FROM orders o
    WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 48 HOUR)
        AND o.order_date < DATE_SUB(NOW(), INTERVAL 24 HOUR)
)
SELECT 
    'Orders (24h)' AS metric,
    FORMAT(m24.orders, 0) AS current_value,
    FORMAT(mp24.orders, 0) AS previous_value,
    ROUND((m24.orders - mp24.orders) * 100.0 / NULLIF(mp24.orders, 0), 2) AS change_pct,
    CASE WHEN m24.orders > mp24.orders THEN ''TRENDING_UP'' ELSE ''TRENDING_DOWN'' END AS trend
FROM metrics_24h m24, metrics_prev_24h mp24

UNION ALL

SELECT 
    'Revenue (24h)',
    CONCAT(', FORMAT(m24.revenue, 2)),
    CONCAT(', FORMAT(mp24.revenue, 2)),
    ROUND((m24.revenue - mp24.revenue) * 100.0 / NULLIF(mp24.revenue, 0), 2),
    CASE WHEN m24.revenue > mp24.revenue THEN ''TRENDING_UP'' ELSE ''TRENDING_DOWN'' END
FROM metrics_24h m24, metrics_prev_24h mp24

UNION ALL

SELECT 
    'Avg Order Value (24h)',
    CONCAT(', FORMAT(m24.avg_order_value, 2)),
    CONCAT(', FORMAT(mp24.avg_order_value, 2)),
    ROUND((m24.avg_order_value - mp24.avg_order_value) * 100.0 / NULLIF(mp24.avg_order_value, 0), 2),
    CASE WHEN m24.avg_order_value > mp24.avg_order_value THEN ''TRENDING_UP'' ELSE ''TRENDING_DOWN'' END
FROM metrics_24h m24, metrics_prev_24h mp24

UNION ALL

SELECT 
    'Customers (24h)',
    FORMAT(m24.customers, 0),
    FORMAT(mp24.customers, 0),
    ROUND((m24.customers - mp24.customers) * 100.0 / NULLIF(mp24.customers, 0), 2),
    CASE WHEN m24.customers > mp24.customers THEN ''TRENDING_UP'' ELSE ''TRENDING_DOWN'' END
FROM metrics_24h m24, metrics_prev_24h mp24

UNION ALL

SELECT 
    'Units Sold (24h)',
    FORMAT(m24.units_sold, 0),
    '-',
    NULL,
    '📦'
FROM metrics_24h m24

UNION ALL

SELECT 
    'Delivery Rate (24h)',
    CONCAT(ROUND(m24.delivered * 100.0 / NULLIF(m24.orders, 0), 2), '%'),
    '-',
    NULL,
    '🚚'
FROM metrics_24h m24

UNION ALL

SELECT 
    'Cancellation Rate (24h)',
    CONCAT(ROUND(m24.cancelled * 100.0 / NULLIF(m24.orders, 0), 2), '%'),
    '-',
    NULL,
    ''ERROR''
FROM metrics_24h m24

UNION ALL

SELECT 
    'Avg Delivery Time',
    CONCAT(ROUND(m24.avg_delivery_hours, 1), ' hours'),
    '-',
    NULL,
    '⏱️'
FROM metrics_24h m24;

-- ========================================
-- 19. TOP CUSTOMERS - LIVE
-- Highest value customers today
-- ========================================

SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.state,
    COUNT(DISTINCT o.order_id) AS orders_today,
    CONCAT(', FORMAT(SUM(o.total_amount), 2)) AS spent_today,
    CONCAT(', FORMAT(AVG(o.total_amount), 2)) AS avg_order_value,
    -- Lifetime metrics
    (SELECT COUNT(*) FROM orders WHERE customer_id = c.customer_id AND payment_status = 'paid') AS lifetime_orders,
    (SELECT CONCAT(', FORMAT(SUM(total_amount), 2)) 
     FROM orders 
     WHERE customer_id = c.customer_id AND payment_status = 'paid') AS lifetime_value,
    -- Loyalty tier
    COALESCE(lp.tier, 'none') AS loyalty_tier,
    COALESCE(lp.points_balance, 0) AS points_balance,
    CASE 
        WHEN (SELECT SUM(total_amount) FROM orders WHERE customer_id = c.customer_id AND payment_status = 'paid') >= 5000 THEN '👑 VIP'
        WHEN (SELECT SUM(total_amount) FROM orders WHERE customer_id = c.customer_id AND payment_status = 'paid') >= 1000 THEN '⭐ Gold'
        WHEN (SELECT COUNT(*) FROM orders WHERE customer_id = c.customer_id AND payment_status = 'paid') >= 5 THEN '🔄 Loyal'
        ELSE '👤 Regular'
    END AS customer_segment
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
WHERE DATE(o.order_date) = CURDATE()
    AND o.payment_status = 'paid'
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.state, lp.tier, lp.points_balance
ORDER BY SUM(o.total_amount) DESC
LIMIT 15;

-- ========================================
-- 20. TRAFFIC & CONVERSION METRICS
-- Order creation and conversion tracking
-- ========================================

WITH hourly_traffic AS (
    SELECT 
        DATE_FORMAT(created_at, '%H:00') AS hour,
        COUNT(*) AS orders_created,
        COUNT(CASE WHEN payment_status = 'paid' THEN 1 END) AS orders_paid,
        COUNT(CASE WHEN status = 'cancelled' THEN 1 END) AS orders_cancelled
    FROM orders
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    GROUP BY DATE_FORMAT(created_at, '%H:00')
)
SELECT 
    hour,
    orders_created,
    orders_paid,
    orders_cancelled,
    ROUND(orders_paid * 100.0 / NULLIF(orders_created, 0), 2) AS conversion_rate_pct,
    ROUND(orders_cancelled * 100.0 / NULLIF(orders_created, 0), 2) AS abandonment_rate_pct,
    CASE 
        WHEN orders_paid * 100.0 / NULLIF(orders_created, 0) >= 80 THEN ''GREEN' Excellent'
        WHEN orders_paid * 100.0 / NULLIF(orders_created, 0) >= 60 THEN ''YELLOW' Good'
        WHEN orders_paid * 100.0 / NULLIF(orders_created, 0) >= 40 THEN '🟠 Fair'
        ELSE ''RED' Poor'
    END AS conversion_health
FROM hourly_traffic
ORDER BY hour DESC;

-- ========================================
-- 21. REVENUE BY PAYMENT METHOD
-- Payment method breakdown
-- ========================================

SELECT 
    pm.payment_type,
    COUNT(DISTINCT o.order_id) AS transactions,
    CONCAT(', FORMAT(SUM(o.total_amount), 2)) AS total_value,
    ROUND(SUM(o.total_amount) * 100.0 / 
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) 
         AND payment_status = 'paid'), 2) AS revenue_share_pct,
    CONCAT(', FORMAT(AVG(o.total_amount), 2)) AS avg_transaction_value,
    COUNT(DISTINCT o.customer_id) AS unique_customers,
    CASE pm.payment_type
        WHEN 'credit_card' THEN '💳'
        WHEN 'debit_card' THEN '💳'
        WHEN 'paypal' THEN '🅿️'
        WHEN 'bank_transfer' THEN '🏦'
        ELSE ''MONEY''
    END AS payment_icon
FROM orders o
JOIN payment_methods pm ON o.customer_id = pm.customer_id AND pm.is_default = TRUE
WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    AND o.payment_status = 'paid'
GROUP BY pm.payment_type
ORDER BY total_value DESC;

-- ========================================
-- 22. VENDOR PERFORMANCE SNAPSHOT
-- Top performing vendors
-- ========================================

SELECT 
    v.vendor_id,
    v.vendor_name,
    v.rating AS vendor_rating,
    v.status AS vendor_status,
    COUNT(DISTINCT p.product_id) AS products_supplied,
    COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
    COALESCE(SUM(oi.quantity), 0) AS units_sold_7d,
    CONCAT(', FORMAT(COALESCE(SUM(oi.subtotal), 0), 2)) AS revenue_7d,
    CONCAT(', FORMAT(AVG(p.price - p.cost), 2)) AS avg_margin_per_unit,
    COUNT(DISTINCT vc.contract_id) AS active_contracts,
    CASE 
        WHEN v.rating >= 4.5 THEN '⭐ Excellent'
        WHEN v.rating >= 4.0 THEN ''GREEN' Good'
        WHEN v.rating >= 3.5 THEN ''YELLOW' Fair'
        ELSE ''RED' Poor'
    END AS performance_rating
FROM vendors v
LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id AND vc.status = 'active'
LEFT JOIN products p ON vc.product_id = p.product_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id 
    AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    AND o.payment_status = 'paid'
WHERE v.status = 'active'
GROUP BY v.vendor_id, v.vendor_name, v.rating, v.status
ORDER BY revenue_7d DESC
LIMIT 10;

-- ========================================
-- 23. DAILY GOAL TRACKER
-- Track progress toward daily targets
-- ========================================

WITH daily_targets AS (
    SELECT 
        10000 AS revenue_target,
        50 AS orders_target,
        30 AS new_customers_target,
        100 AS avg_order_value_target
),
daily_actual AS (
    SELECT 
        COALESCE(SUM(CASE WHEN payment_status = 'paid' THEN total_amount END), 0) AS revenue_actual,
        COUNT(*) AS orders_actual,
        (SELECT COUNT(*) FROM customers WHERE DATE(created_at) = CURDATE()) AS new_customers_actual,
        AVG(CASE WHEN payment_status = 'paid' THEN total_amount END) AS avg_order_value_actual
    FROM orders
    WHERE DATE(order_date) = CURDATE()
)
SELECT 
    'Revenue' AS goal_metric,
    CONCAT(', FORMAT(dt.revenue_target, 2)) AS target,
    CONCAT(', FORMAT(da.revenue_actual, 2)) AS actual,
    ROUND(da.revenue_actual * 100.0 / dt.revenue_target, 2) AS progress_pct,
    CONCAT(', FORMAT(dt.revenue_target - da.revenue_actual, 2)) AS remaining,
    CASE 
        WHEN da.revenue_actual >= dt.revenue_target THEN ''SUCCESS' Target Met'
        WHEN da.revenue_actual >= dt.revenue_target * 0.8 THEN ''YELLOW' On Track'
        WHEN da.revenue_actual >= dt.revenue_target * 0.5 THEN '🟠 Behind'
        ELSE ''RED' Critical'
    END AS status
FROM daily_targets dt, daily_actual da

UNION ALL

SELECT 
    'Orders',
    FORMAT(dt.orders_target, 0),
    FORMAT(da.orders_actual, 0),
    ROUND(da.orders_actual * 100.0 / dt.orders_target, 2),
    FORMAT(dt.orders_target - da.orders_actual, 0),
    CASE 
        WHEN da.orders_actual >= dt.orders_target THEN ''SUCCESS' Target Met'
        WHEN da.orders_actual >= dt.orders_target * 0.8 THEN ''YELLOW' On Track'
        WHEN da.orders_actual >= dt.orders_target * 0.5 THEN '🟠 Behind'
        ELSE ''RED' Critical'
    END
FROM daily_targets dt, daily_actual da

UNION ALL

SELECT 
    'New Customers',
    FORMAT(dt.new_customers_target, 0),
    FORMAT(da.new_customers_actual, 0),
    ROUND(da.new_customers_actual * 100.0 / dt.new_customers_target, 2),
    FORMAT(dt.new_customers_target - da.new_customers_actual, 0),
    CASE 
        WHEN da.new_customers_actual >= dt.new_customers_target THEN ''SUCCESS' Target Met'
        WHEN da.new_customers_actual >= dt.new_customers_target * 0.8 THEN ''YELLOW' On Track'
        WHEN da.new_customers_actual >= dt.new_customers_target * 0.5 THEN '🟠 Behind'
        ELSE ''RED' Critical'
    END
FROM daily_targets dt, daily_actual da

UNION ALL

SELECT 
    'Avg Order Value',
    CONCAT(', FORMAT(dt.avg_order_value_target, 2)),
    CONCAT(', FORMAT(da.avg_order_value_actual, 2)),
    ROUND(da.avg_order_value_actual * 100.0 / dt.avg_order_value_target, 2),
    CONCAT(', FORMAT(dt.avg_order_value_target - da.avg_order_value_actual, 2)),
    CASE 
        WHEN da.avg_order_value_actual >= dt.avg_order_value_target THEN ''SUCCESS' Target Met'
        WHEN da.avg_order_value_actual >= dt.avg_order_value_target * 0.9 THEN ''YELLOW' On Track'
        WHEN da.avg_order_value_actual >= dt.avg_order_value_target * 0.8 THEN '🟠 Behind'
        ELSE ''RED' Critical'
    END
FROM daily_targets dt, daily_actual da;

-- ========================================
-- COMPLETION MESSAGE
-- ========================================

SELECT 
    ''SUCCESS' Dashboard Metrics Generated Successfully' AS status,
    CONCAT('Snapshot Time: ', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')) AS timestamp,
    'Real-time data ready for display' AS message,
    CONCAT('Active Users: ', (SELECT COUNT(DISTINCT customer_id) 
                               FROM orders 
                               WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR))) AS active_now;