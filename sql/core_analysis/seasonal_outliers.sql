-- ========================================
-- SEASONAL OUTLIERS ANALYSIS
-- Day 6-7: Order & Transaction Queries
-- Unusual Seasonal Patterns & Time-based Anomalies
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. UNUSUAL DAILY ORDER VOLUME PATTERNS
-- ========================================

-- Detect days with abnormally high or low order volumes
WITH daily_stats AS (
    SELECT 
        DATE(order_date) AS order_day,
        COUNT(order_id) AS order_count,
        SUM(total_amount) AS daily_revenue,
        ROUND(AVG(total_amount), 2) AS avg_order_value
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY DATE(order_date)
),
overall_stats AS (
    SELECT 
        AVG(order_count) AS avg_daily_orders,
        STDDEV(order_count) AS stddev_daily_orders,
        AVG(daily_revenue) AS avg_daily_revenue,
        STDDEV(daily_revenue) AS stddev_daily_revenue
    FROM daily_stats
)
SELECT 
    ds.order_day,
    DAYNAME(ds.order_day) AS day_of_week,
    ds.order_count,
    ds.daily_revenue,
    ds.avg_order_value,
    ROUND(os.avg_daily_orders, 2) AS expected_orders,
    ROUND((ds.order_count - os.avg_daily_orders) / NULLIF(os.stddev_daily_orders, 0), 2) AS order_z_score,
    ROUND((ds.daily_revenue - os.avg_daily_revenue) / NULLIF(os.stddev_daily_revenue, 0), 2) AS revenue_z_score,
    CASE 
        WHEN ABS((ds.order_count - os.avg_daily_orders) / NULLIF(os.stddev_daily_orders, 0)) > 3 THEN 'EXTREME_OUTLIER'
        WHEN ABS((ds.order_count - os.avg_daily_orders) / NULLIF(os.stddev_daily_orders, 0)) > 2 THEN 'SIGNIFICANT_OUTLIER'
        ELSE 'NORMAL'
    END AS outlier_status
FROM daily_stats ds
CROSS JOIN overall_stats os
WHERE ABS((ds.order_count - os.avg_daily_orders) / NULLIF(os.stddev_daily_orders, 0)) > 2
ORDER BY ABS((ds.order_count - os.avg_daily_orders) / NULLIF(os.stddev_daily_orders, 0)) DESC;

-- ========================================
-- 2. WEEKEND VS WEEKDAY ANOMALIES
-- ========================================

WITH order_patterns AS (
    SELECT 
        DATE(order_date) AS order_day,
        CASE 
            WHEN DAYOFWEEK(order_date) IN (1, 7) THEN 'Weekend'
            ELSE 'Weekday'
        END AS day_type,
        COUNT(order_id) AS order_count,
        SUM(total_amount) AS daily_revenue,
        AVG(total_amount) AS avg_order_value
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY DATE(order_date), day_type
),
day_type_averages AS (
    SELECT 
        day_type,
        AVG(order_count) AS avg_orders,
        STDDEV(order_count) AS stddev_orders,
        AVG(daily_revenue) AS avg_revenue,
        STDDEV(daily_revenue) AS stddev_revenue
    FROM order_patterns
    GROUP BY day_type
)
SELECT 
    op.order_day,
    op.day_type,
    op.order_count,
    ROUND(dta.avg_orders, 2) AS expected_orders,
    op.daily_revenue,
    ROUND(dta.avg_revenue, 2) AS expected_revenue,
    ROUND((op.order_count - dta.avg_orders) / NULLIF(dta.stddev_orders, 0), 2) AS order_deviation,
    ROUND(((op.order_count - dta.avg_orders) / NULLIF(dta.avg_orders, 0)) * 100, 2) AS pct_difference
FROM order_patterns op
JOIN day_type_averages dta ON op.day_type = dta.day_type
WHERE ABS((op.order_count - dta.avg_orders) / NULLIF(dta.stddev_orders, 0)) > 2
ORDER BY ABS((op.order_count - dta.avg_orders) / NULLIF(dta.stddev_orders, 0)) DESC;

-- ========================================
-- 3. MONTHLY SEASONALITY ANALYSIS
-- ========================================

-- Compare monthly performance to detect unusual patterns
WITH monthly_metrics AS (
    SELECT 
        YEAR(order_date) AS order_year,
        MONTH(order_date) AS order_month,
        DATE_FORMAT(order_date, '%Y-%m') AS year_month,
        COUNT(order_id) AS order_count,
        SUM(total_amount) AS monthly_revenue,
        AVG(total_amount) AS avg_order_value,
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY YEAR(order_date), MONTH(order_date), DATE_FORMAT(order_date, '%Y-%m')
),
month_averages AS (
    SELECT 
        order_month,
        AVG(order_count) AS avg_orders,
        STDDEV(order_count) AS stddev_orders,
        AVG(monthly_revenue) AS avg_revenue,
        STDDEV(monthly_revenue) AS stddev_revenue
    FROM monthly_metrics
    GROUP BY order_month
)
SELECT 
    mm.year_month,
    mm.order_month,
    MONTHNAME(STR_TO_DATE(mm.order_month, '%m')) AS month_name,
    mm.order_count,
    ROUND(ma.avg_orders, 2) AS historical_avg_orders,
    mm.monthly_revenue,
    ROUND(ma.avg_revenue, 2) AS historical_avg_revenue,
    ROUND((mm.order_count - ma.avg_orders) / NULLIF(ma.stddev_orders, 0), 2) AS order_z_score,
    ROUND(((mm.order_count - ma.avg_orders) / NULLIF(ma.avg_orders, 0)) * 100, 2) AS pct_variance,
    mm.unique_customers,
    mm.avg_order_value
FROM monthly_metrics mm
JOIN month_averages ma ON mm.order_month = ma.order_month
ORDER BY mm.year_month DESC;

-- ========================================
-- 4. HOURLY TRANSACTION PATTERNS
-- ========================================

-- Identify unusual hourly patterns (potential bot activity or fraud)
WITH hourly_orders AS (
    SELECT 
        DATE(order_date) AS order_day,
        HOUR(order_date) AS order_hour,
        COUNT(order_id) AS order_count,
        SUM(total_amount) AS hourly_revenue,
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
    GROUP BY DATE(order_date), HOUR(order_date)
),
hour_averages AS (
    SELECT 
        order_hour,
        AVG(order_count) AS avg_orders,
        STDDEV(order_count) AS stddev_orders
    FROM hourly_orders
    GROUP BY order_hour
)
SELECT 
    ho.order_day,
    ho.order_hour,
    CASE 
        WHEN ho.order_hour BETWEEN 0 AND 5 THEN 'Late Night'
        WHEN ho.order_hour BETWEEN 6 AND 11 THEN 'Morning'
        WHEN ho.order_hour BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE 'Evening'
    END AS time_period,
    ho.order_count,
    ROUND(ha.avg_orders, 2) AS expected_orders,
    ho.unique_customers,
    ROUND(ho.order_count / NULLIF(ho.unique_customers, 0), 2) AS orders_per_customer,
    ROUND((ho.order_count - ha.avg_orders) / NULLIF(ha.stddev_orders, 0), 2) AS z_score
FROM hourly_orders ho
JOIN hour_averages ha ON ho.order_hour = ha.order_hour
WHERE ABS((ho.order_count - ha.avg_orders) / NULLIF(ha.stddev_orders, 0)) > 2
   OR (ho.order_count / NULLIF(ho.unique_customers, 0)) > 5  -- More than 5 orders per customer per hour
ORDER BY ho.order_day DESC, ho.order_hour;

-- ========================================
-- 5. PRODUCT CATEGORY SEASONAL PATTERNS
-- ========================================

-- Identify products with unusual seasonal demand
WITH category_monthly AS (
    SELECT 
        pc.category_name,
        DATE_FORMAT(o.order_date, '%Y-%m') AS year_month,
        MONTH(o.order_date) AS order_month,
        COUNT(oi.order_item_id) AS items_sold,
        SUM(oi.subtotal) AS category_revenue
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 18 MONTH)
    GROUP BY pc.category_name, DATE_FORMAT(o.order_date, '%Y-%m'), MONTH(o.order_date)
),
category_month_avg AS (
    SELECT 
        category_name,
        order_month,
        AVG(items_sold) AS avg_items,
        STDDEV(items_sold) AS stddev_items
    FROM category_monthly
    GROUP BY category_name, order_month
)
SELECT 
    cm.category_name,
    cm.year_month,
    MONTHNAME(STR_TO_DATE(cm.order_month, '%m')) AS month_name,
    cm.items_sold,
    ROUND(cma.avg_items, 2) AS expected_items,
    ROUND(((cm.items_sold - cma.avg_items) / NULLIF(cma.avg_items, 0)) * 100, 2) AS pct_variance,
    cm.category_revenue,
    CASE 
        WHEN cm.items_sold > cma.avg_items * 1.5 THEN 'UNUSUALLY_HIGH'
        WHEN cm.items_sold < cma.avg_items * 0.5 THEN 'UNUSUALLY_LOW'
        ELSE 'NORMAL'
    END AS seasonality_flag
FROM category_monthly cm
JOIN category_month_avg cma ON cm.category_name = cma.category_name 
    AND cm.order_month = cma.order_month
WHERE cm.items_sold > cma.avg_items * 1.5
   OR cm.items_sold < cma.avg_items * 0.5
ORDER BY cm.year_month DESC, ABS(pct_variance) DESC;

-- ========================================
-- 6. RAPID ORDER SUCCESSION DETECTION
-- ========================================

-- Detect customers placing multiple orders in short time spans
WITH order_intervals AS (
    SELECT 
        customer_id,
        order_id,
        order_date,
        LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_order_date,
        TIMESTAMPDIFF(MINUTE, 
            LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date),
            order_date
        ) AS minutes_since_last_order
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
)
SELECT 
    oi.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    oi.order_id,
    oi.order_date,
    oi.prev_order_date,
    oi.minutes_since_last_order,
    o.total_amount,
    o.status,
    CASE 
        WHEN oi.minutes_since_last_order < 5 THEN 'EXTREMELY_RAPID'
        WHEN oi.minutes_since_last_order < 15 THEN 'VERY_RAPID'
        WHEN oi.minutes_since_last_order < 60 THEN 'RAPID'
        ELSE 'NORMAL'
    END AS order_pattern
FROM order_intervals oi
JOIN orders o ON oi.order_id = o.order_id
JOIN customers c ON oi.customer_id = c.customer_id
WHERE oi.minutes_since_last_order IS NOT NULL
  AND oi.minutes_since_last_order < 60
ORDER BY oi.minutes_since_last_order ASC, oi.customer_id, oi.order_date;

-- ========================================
-- 7. OFF-PEAK HOUR HIGH-VALUE ORDERS
-- ========================================

-- Identify high-value orders placed during unusual hours
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    DATE(o.order_date) AS order_day,
    HOUR(o.order_date) AS order_hour,
    CASE 
        WHEN HOUR(o.order_date) BETWEEN 0 AND 5 THEN 'Late Night (12AM-5AM)'
        WHEN HOUR(o.order_date) BETWEEN 6 AND 11 THEN 'Morning (6AM-11AM)'
        WHEN HOUR(o.order_date) BETWEEN 12 AND 17 THEN 'Afternoon (12PM-5PM)'
        ELSE 'Evening (6PM-11PM)'
    END AS time_period,
    o.total_amount,
    o.status,
    o.payment_status,
    COUNT(oi.order_item_id) AS item_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE HOUR(o.order_date) BETWEEN 0 AND 5  -- Late night hours
  AND o.total_amount > 500  -- High value
  AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, o.order_date, 
         o.total_amount, o.status, o.payment_status
ORDER BY o.order_date DESC, o.total_amount DESC;

-- ========================================
-- 8. SUDDEN CUSTOMER BEHAVIOR CHANGES
-- ========================================

-- Customers with sudden spikes in order frequency or value
WITH customer_periods AS (
    SELECT 
        customer_id,
        CASE 
            WHEN order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN 'Recent'
            WHEN order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) THEN 'Previous'
            ELSE 'Historical'
        END AS period,
        COUNT(order_id) AS order_count,
        SUM(total_amount) AS total_spent,
        AVG(total_amount) AS avg_order_value
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
    GROUP BY customer_id, period
),
customer_comparison AS (
    SELECT 
        r.customer_id,
        r.order_count AS recent_orders,
        r.total_spent AS recent_spent,
        r.avg_order_value AS recent_avg,
        p.order_count AS previous_orders,
        p.total_spent AS previous_spent,
        p.avg_order_value AS previous_avg
    FROM customer_periods r
    LEFT JOIN customer_periods p ON r.customer_id = p.customer_id AND p.period = 'Previous'
    WHERE r.period = 'Recent'
)
SELECT 
    cc.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    cc.recent_orders,
    COALESCE(cc.previous_orders, 0) AS previous_orders,
    ROUND(cc.recent_spent, 2) AS recent_spent,
    ROUND(COALESCE(cc.previous_spent, 0), 2) AS previous_spent,
    ROUND(cc.recent_avg, 2) AS recent_avg_order,
    ROUND(COALESCE(cc.previous_avg, 0), 2) AS previous_avg_order,
    CASE 
        WHEN cc.previous_orders > 0 THEN 
            ROUND(((cc.recent_orders - cc.previous_orders) / cc.previous_orders) * 100, 2)
        ELSE NULL
    END AS order_frequency_change_pct,
    CASE 
        WHEN cc.previous_avg > 0 THEN 
            ROUND(((cc.recent_avg - cc.previous_avg) / cc.previous_avg) * 100, 2)
        ELSE NULL
    END AS avg_value_change_pct,
    c.status AS customer_status
FROM customer_comparison cc
JOIN customers c ON cc.customer_id = c.customer_id
WHERE (cc.previous_orders > 0 AND cc.recent_orders > cc.previous_orders * 2)
   OR (cc.previous_avg > 0 AND cc.recent_avg > cc.previous_avg * 1.5)
ORDER BY order_frequency_change_pct DESC NULLS LAST;

-- ========================================
-- 9. QUARTERLY TREND ANOMALIES
-- ========================================

WITH quarterly_metrics AS (
    SELECT 
        YEAR(order_date) AS order_year,
        QUARTER(order_date) AS order_quarter,
        CONCAT(YEAR(order_date), '-Q', QUARTER(order_date)) AS year_quarter,
        COUNT(order_id) AS order_count,
        SUM(total_amount) AS quarterly_revenue,
        AVG(total_amount) AS avg_order_value,
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY YEAR(order_date), QUARTER(order_date)
),
quarter_growth AS (
    SELECT 
        qm.*,
        LAG(qm.quarterly_revenue) OVER (ORDER BY qm.order_year, qm.order_quarter) AS prev_quarter_revenue,
        LAG(qm.order_count) OVER (ORDER BY qm.order_year, qm.order_quarter) AS prev_quarter_orders,
        ROUND(((qm.quarterly_revenue - LAG(qm.quarterly_revenue) OVER (ORDER BY qm.order_year, qm.order_quarter)) / 
               NULLIF(LAG(qm.quarterly_revenue) OVER (ORDER BY qm.order_year, qm.order_quarter), 0)) * 100, 2) AS revenue_growth_pct,
        ROUND(((qm.order_count - LAG(qm.order_count) OVER (ORDER BY qm.order_year, qm.order_quarter)) / 
               NULLIF(LAG(qm.order_count) OVER (ORDER BY qm.order_year, qm.order_quarter), 0)) * 100, 2) AS order_growth_pct
    FROM quarterly_metrics qm
)
SELECT 
    year_quarter,
    order_count,
    prev_quarter_orders,
    quarterly_revenue,
    prev_quarter_revenue,
    revenue_growth_pct,
    order_growth_pct,
    unique_customers,
    avg_order_value,
    CASE 
        WHEN revenue_growth_pct > 50 THEN 'EXCEPTIONAL_GROWTH'
        WHEN revenue_growth_pct < -30 THEN 'SIGNIFICANT_DECLINE'
        WHEN ABS(revenue_growth_pct) < 5 THEN 'STAGNANT'
        ELSE 'NORMAL'
    END AS trend_flag
FROM quarter_growth
WHERE prev_quarter_revenue IS NOT NULL
ORDER BY order_year DESC, order_quarter DESC;

-- ========================================
-- 10. HOLIDAY SEASON PERFORMANCE
-- ========================================

-- Compare holiday periods to non-holiday periods
WITH date_classification AS (
    SELECT 
        DATE(order_date) AS order_day,
        CASE 
            -- Black Friday to Cyber Monday (approximate - last Thu-Mon of November)
            WHEN MONTH(order_date) = 11 AND DAY(order_date) BETWEEN 23 AND 30 THEN 'Black Friday Week'
            -- Holiday season (Dec 1 - Dec 31)
            WHEN MONTH(order_date) = 12 THEN 'Holiday Season'
            -- Back to School (August)
            WHEN MONTH(order_date) = 8 THEN 'Back to School'
            -- Valentine's Day
            WHEN MONTH(order_date) = 2 AND DAY(order_date) BETWEEN 10 AND 14 THEN 'Valentine''s Day'
            -- Mother's Day (approximate - 2nd Sunday in May)
            WHEN MONTH(order_date) = 5 AND DAY(order_date) BETWEEN 8 AND 14 THEN 'Mother''s Day'
            ELSE 'Regular'
        END AS period_type,
        COUNT(order_id) AS order_count,
        SUM(total_amount) AS daily_revenue,
        AVG(total_amount) AS avg_order_value
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY DATE(order_date), period_type
)
SELECT 
    period_type,
    COUNT(DISTINCT order_day) AS number_of_days,
    SUM(order_count) AS total_orders,
    ROUND(AVG(order_count), 2) AS avg_daily_orders,
    SUM(daily_revenue) AS total_revenue,
    ROUND(AVG(daily_revenue), 2) AS avg_daily_revenue,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    MAX(order_count) AS peak_day_orders,
    MIN(order_count) AS lowest_day_orders
FROM date_classification
GROUP BY period_type
ORDER BY avg_daily_revenue DESC;

-- ========================================
-- 11. TIME-BASED ORDER CANCELLATION PATTERNS
-- ========================================

-- Analyze cancellation patterns by time period
SELECT 
    DATE_FORMAT(order_date, '%Y-%m') AS year_month,
    HOUR(order_date) AS order_hour,
    CASE 
        WHEN HOUR(order_date) BETWEEN 0 AND 5 THEN 'Late Night'
        WHEN HOUR(order_date) BETWEEN 6 AND 11 THEN 'Morning'
        WHEN HOUR(order_date) BETWEEN 12 AND 17 THEN 'Afternoon'
        ELSE 'Evening'
    END AS time_period,
    COUNT(order_id) AS total_orders,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    ROUND((SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(order_id)) * 100, 2) AS cancellation_rate,
    ROUND(AVG(total_amount), 2) AS avg_order_value,
    ROUND(AVG(CASE WHEN status = 'cancelled' THEN total_amount END), 2) AS avg_cancelled_value
FROM orders
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
GROUP BY DATE_FORMAT(order_date, '%Y-%m'), HOUR(order_date), time_period
HAVING cancelled_orders > 0
ORDER BY cancellation_rate DESC, year_month DESC;

-- ========================================
-- 12. PRODUCT LIFECYCLE ANOMALIES
-- ========================================

-- Identify products with unusual sales patterns over time
WITH product_monthly AS (
    SELECT 
        p.product_id,
        p.product_name,
        DATE_FORMAT(o.order_date, '%Y-%m') AS year_month,
        COUNT(oi.order_item_id) AS units_sold,
        SUM(oi.subtotal) AS monthly_revenue
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY p.product_id, p.product_name, DATE_FORMAT(o.order_date, '%Y-%m')
),
product_stats AS (
    SELECT 
        product_id,
        product_name,
        AVG(units_sold) AS avg_monthly_sales,
        STDDEV(units_sold) AS stddev_sales,
        MAX(units_sold) AS peak_sales,
        MIN(units_sold) AS lowest_sales
    FROM product_monthly
    GROUP BY product_id, product_name
)
SELECT 
    pm.product_id,
    pm.product_name,
    pm.year_month,
    pm.units_sold,
    ROUND(ps.avg_monthly_sales, 2) AS avg_monthly_sales,
    ROUND((pm.units_sold - ps.avg_monthly_sales) / NULLIF(ps.stddev_sales, 0), 2) AS z_score,
    ROUND(((pm.units_sold - ps.avg_monthly_sales) / NULLIF(ps.avg_monthly_sales, 0)) * 100, 2) AS pct_variance,
    pm.monthly_revenue,
    CASE 
        WHEN pm.units_sold > ps.avg_monthly_sales * 2 THEN 'SUDDEN_SPIKE'
        WHEN pm.units_sold < ps.avg_monthly_sales * 0.3 AND ps.avg_monthly_sales > 10 THEN 'SUDDEN_DROP'
        ELSE 'NORMAL'
    END AS anomaly_type
FROM product_monthly pm
JOIN product_stats ps ON pm.product_id = ps.product_id
WHERE (pm.units_sold > ps.avg_monthly_sales * 2 OR pm.units_sold < ps.avg_monthly_sales * 0.3)
  AND ps.avg_monthly_sales > 5  -- Filter out low-volume products
ORDER BY ABS(pct_variance) DESC, pm.year_month DESC;

-- ========================================
-- 13. RETURN SEASONALITY PATTERNS
-- ========================================

-- Analyze return patterns over time
SELECT 
    DATE_FORMAT(r.created_at, '%Y-%m') AS return_month,
    MONTHNAME(r.created_at) AS month_name,
    COUNT(r.return_id) AS total_returns,
    COUNT(DISTINCT r.order_id) AS affected_orders,
    SUM(r.refund_amount) AS total_refunded,
    ROUND(AVG(r.refund_amount), 2) AS avg_refund,
    r.reason,
    COUNT(*) AS return_count_by_reason,
    ROUND((COUNT(*) / (SELECT COUNT(*) FROM returns r2 
                       WHERE DATE_FORMAT(r2.created_at, '%Y-%m') = DATE_FORMAT(r.created_at, '%Y-%m'))) * 100, 2) AS pct_of_monthly_returns
FROM returns r
WHERE r.created_at >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY DATE_FORMAT(r.created_at, '%Y-%m'), MONTHNAME(r.created_at), r.reason
ORDER BY return_month DESC, return_count_by_reason DESC;

-- ========================================
-- 14. INVENTORY RESTOCK PATTERN ANALYSIS
-- ========================================

-- Identify unusual inventory fluctuation patterns
WITH inventory_changes AS (
    SELECT 
        i.product_id,
        p.product_name,
        DATE(i.last_updated) AS change_date,
        i.quantity_on_hand,
        LAG(i.quantity_on_hand) OVER (PARTITION BY i.product_id ORDER BY i.last_updated) AS prev_quantity,
        i.quantity_on_hand - LAG(i.quantity_on_hand) OVER (PARTITION BY i.product_id ORDER BY i.last_updated) AS quantity_change
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    WHERE i.last_updated >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
)
SELECT 
    product_id,
    product_name,
    change_date,
    prev_quantity,
    quantity_on_hand,
    quantity_change,
    CASE 
        WHEN quantity_change > 1000 THEN 'LARGE_RESTOCK'
        WHEN quantity_change < -500 THEN 'LARGE_DEPLETION'
        WHEN prev_quantity > 0 AND quantity_on_hand = 0 THEN 'STOCKOUT'
        WHEN prev_quantity = 0 AND quantity_on_hand > 0 THEN 'BACK_IN_STOCK'
        ELSE 'NORMAL'
    END AS change_type
FROM inventory_changes
WHERE quantity_change IS NOT NULL
  AND (ABS(quantity_change) > 500 OR (prev_quantity > 0 AND quantity_on_hand = 0))
ORDER BY ABS(quantity_change) DESC, change_date DESC;

-- ========================================
-- 15. CUSTOMER ACQUISITION SEASONALITY
-- ========================================

-- Analyze new customer acquisition patterns
WITH monthly_customers AS (
    SELECT 
        DATE_FORMAT(created_at, '%Y-%m') AS signup_month,
        MONTH(created_at) AS month_num,
        YEAR(created_at) AS year_num,
        COUNT(customer_id) AS new_customers,
        COUNT(CASE WHEN status = 'active' THEN 1 END) AS active_customers
    FROM customers
    WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 18 MONTH)
    GROUP BY DATE_FORMAT(created_at, '%Y-%m'), MONTH(created_at), YEAR(created_at)
),
month_averages AS (
    SELECT 
        month_num,
        AVG(new_customers) AS avg_new_customers,
        STDDEV(new_customers) AS stddev_new_customers
    FROM monthly_customers
    GROUP BY month_num
)
SELECT 
    mc.signup_month,
    MONTHNAME(STR_TO_DATE(mc.month_num, '%m')) AS month_name,
    mc.new_customers,
    mc.active_customers,
    ROUND(ma.avg_new_customers, 2) AS expected_new_customers,
    ROUND((mc.new_customers - ma.avg_new_customers) / NULLIF(ma.stddev_new_customers, 0), 2) AS z_score,
    ROUND(((mc.new_customers - ma.avg_new_customers) / NULLIF(ma.avg_new_customers, 0)) * 100, 2) AS pct_variance,
    CASE 
        WHEN mc.new_customers > ma.avg_new_customers * 1.5 THEN 'HIGH_ACQUISITION'
        WHEN mc.new_customers < ma.avg_new_customers * 0.5 THEN 'LOW_ACQUISITION'
        ELSE 'NORMAL'
    END AS acquisition_pattern
FROM monthly_customers mc
JOIN month_averages ma ON mc.month_num = ma.month_num
ORDER BY mc.signup_month DESC;

-- ========================================
-- END OF SEASONAL OUTLIERS ANALYSIS
-- ========================================