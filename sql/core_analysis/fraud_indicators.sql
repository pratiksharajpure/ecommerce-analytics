-- ========================================
-- FRAUD INDICATORS ANALYSIS
-- Day 6-7: Order & Transaction Queries
-- Fraud Pattern Detection & Risk Scoring
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. MULTIPLE ORDERS SAME IP/DEVICE DETECTION
-- ========================================

-- Detect customers with suspicious multiple orders in short timeframe
WITH rapid_orders AS (
    SELECT 
        customer_id,
        COUNT(order_id) AS order_count,
        MIN(order_date) AS first_order,
        MAX(order_date) AS last_order,
        TIMESTAMPDIFF(MINUTE, MIN(order_date), MAX(order_date)) AS timespan_minutes,
        SUM(total_amount) AS total_spent,
        AVG(total_amount) AS avg_order_value,
        COUNT(DISTINCT DATE(order_date)) AS unique_days
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    GROUP BY customer_id
    HAVING COUNT(order_id) >= 5
       AND TIMESTAMPDIFF(HOUR, MIN(order_date), MAX(order_date)) < 24
)
SELECT 
    ro.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.phone,
    ro.order_count,
    ro.first_order,
    ro.last_order,
    ro.timespan_minutes,
    ro.total_spent,
    ro.avg_order_value,
    ro.unique_days,
    c.created_at AS customer_since,
    DATEDIFF(ro.first_order, c.created_at) AS days_since_signup,
    CASE 
        WHEN ro.order_count >= 10 AND ro.timespan_minutes < 120 THEN 'CRITICAL'
        WHEN ro.order_count >= 7 AND ro.timespan_minutes < 360 THEN 'HIGH'
        WHEN ro.order_count >= 5 AND ro.timespan_minutes < 1440 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_level
FROM rapid_orders ro
JOIN customers c ON ro.customer_id = c.customer_id
ORDER BY ro.order_count DESC, ro.timespan_minutes ASC;

-- ========================================
-- 2. NEW ACCOUNT HIGH-VALUE TRANSACTIONS
-- ========================================

-- Detect new accounts making unusually large purchases
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.created_at AS account_created,
    o.order_id,
    o.order_date,
    TIMESTAMPDIFF(HOUR, c.created_at, o.order_date) AS hours_after_signup,
    o.total_amount,
    o.status,
    o.payment_status,
    COUNT(oi.order_item_id) AS item_count,
    CASE 
        WHEN o.total_amount > 2000 AND TIMESTAMPDIFF(HOUR, c.created_at, o.order_date) < 1 THEN 'CRITICAL'
        WHEN o.total_amount > 1000 AND TIMESTAMPDIFF(HOUR, c.created_at, o.order_date) < 24 THEN 'HIGH'
        WHEN o.total_amount > 500 AND TIMESTAMPDIFF(HOUR, c.created_at, o.order_date) < 72 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS fraud_risk
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE c.created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
  AND TIMESTAMPDIFF(HOUR, c.created_at, o.order_date) < 168  -- Within 7 days
  AND o.total_amount > 500
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.created_at,
         o.order_id, o.order_date, o.total_amount, o.status, o.payment_status
ORDER BY o.total_amount DESC, hours_after_signup ASC;

-- ========================================
-- 3. FAILED PAYMENT ATTEMPT PATTERNS
-- ========================================

-- Multiple failed payment attempts (card testing)
WITH payment_attempts AS (
    SELECT 
        customer_id,
        DATE(order_date) AS attempt_date,
        COUNT(CASE WHEN payment_status = 'failed' THEN 1 END) AS failed_count,
        COUNT(CASE WHEN payment_status = 'paid' THEN 1 END) AS success_count,
        COUNT(order_id) AS total_attempts,
        SUM(CASE WHEN payment_status = 'failed' THEN total_amount ELSE 0 END) AS failed_amount,
        SUM(CASE WHEN payment_status = 'paid' THEN total_amount ELSE 0 END) AS success_amount
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 14 DAY)
    GROUP BY customer_id, DATE(order_date)
    HAVING failed_count >= 3
)
SELECT 
    pa.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.phone,
    pa.attempt_date,
    pa.failed_count,
    pa.success_count,
    pa.total_attempts,
    ROUND(pa.failed_amount, 2) AS failed_amount,
    ROUND(pa.success_amount, 2) AS success_amount,
    ROUND((pa.failed_count / pa.total_attempts) * 100, 2) AS failure_rate,
    c.status AS customer_status,
    CASE 
        WHEN pa.failed_count >= 10 THEN 'CRITICAL_CARD_TESTING'
        WHEN pa.failed_count >= 5 THEN 'HIGH_RISK'
        ELSE 'MEDIUM_RISK'
    END AS risk_assessment
FROM payment_attempts pa
JOIN customers c ON pa.customer_id = c.customer_id
ORDER BY pa.failed_count DESC, pa.attempt_date DESC;

-- ========================================
-- 4. UNUSUAL SHIPPING ADDRESS PATTERNS
-- ========================================

-- Multiple customers sharing same shipping address
WITH shared_addresses AS (
    SELECT 
        CONCAT(UPPER(TRIM(address_line1)), '|', UPPER(TRIM(city)), '|', 
               UPPER(TRIM(state)), '|', UPPER(TRIM(zip_code))) AS address_key,
        address_line1,
        city,
        state,
        zip_code,
        COUNT(DISTINCT customer_id) AS customer_count,
        GROUP_CONCAT(DISTINCT customer_id ORDER BY customer_id) AS customer_ids,
        SUM(CASE WHEN is_default = TRUE THEN 1 ELSE 0 END) AS default_address_count
    FROM shipping_addresses
    WHERE address_line1 IS NOT NULL
    GROUP BY address_key, address_line1, city, state, zip_code
    HAVING COUNT(DISTINCT customer_id) >= 3
)
SELECT 
    sa.address_line1,
    sa.city,
    sa.state,
    sa.zip_code,
    sa.customer_count,
    sa.customer_ids,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_order_value,
    MAX(o.order_date) AS last_order_date,
    CASE 
        WHEN sa.customer_count >= 10 THEN 'CRITICAL_RESHIPPING'
        WHEN sa.customer_count >= 5 THEN 'HIGH_RISK'
        ELSE 'MEDIUM_RISK'
    END AS fraud_indicator
FROM shared_addresses sa
LEFT JOIN shipping_addresses sha ON sa.address_key = CONCAT(UPPER(TRIM(sha.address_line1)), '|', 
                                                              UPPER(TRIM(sha.city)), '|', 
                                                              UPPER(TRIM(sha.state)), '|', 
                                                              UPPER(TRIM(sha.zip_code)))
LEFT JOIN orders o ON sha.customer_id = o.customer_id
GROUP BY sa.address_line1, sa.city, sa.state, sa.zip_code, sa.customer_count, sa.customer_ids
ORDER BY sa.customer_count DESC, total_order_value DESC;

-- ========================================
-- 5. VELOCITY FRAUD DETECTION
-- ========================================

-- Customers exceeding normal transaction velocity
WITH customer_velocity AS (
    SELECT 
        customer_id,
        DATE(order_date) AS order_day,
        COUNT(order_id) AS daily_orders,
        SUM(total_amount) AS daily_spent,
        AVG(total_amount) AS avg_order_value,
        MIN(order_date) AS first_order_time,
        MAX(order_date) AS last_order_time,
        TIMESTAMPDIFF(HOUR, MIN(order_date), MAX(order_date)) AS order_timespan_hours
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
    GROUP BY customer_id, DATE(order_date)
),
customer_history AS (
    SELECT 
        customer_id,
        COUNT(DISTINCT order_day) AS active_days,
        SUM(daily_orders) AS total_orders,
        MAX(daily_orders) AS max_daily_orders,
        SUM(daily_spent) AS total_spent,
        MAX(daily_spent) AS max_daily_spent,
        AVG(daily_orders) AS avg_daily_orders,
        AVG(daily_spent) AS avg_daily_spent
    FROM customer_velocity
    GROUP BY customer_id
)
SELECT 
    ch.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    ch.total_orders,
    ch.active_days,
    ch.max_daily_orders,
    ROUND(ch.avg_daily_orders, 2) AS avg_daily_orders,
    ROUND(ch.max_daily_spent, 2) AS max_daily_spent,
    ROUND(ch.avg_daily_spent, 2) AS avg_daily_spent,
    ROUND(ch.total_spent, 2) AS total_spent_30days,
    c.created_at AS customer_since,
    DATEDIFF(CURDATE(), c.created_at) AS account_age_days,
    CASE 
        WHEN ch.max_daily_orders >= 10 OR ch.max_daily_spent > 5000 THEN 'CRITICAL'
        WHEN ch.max_daily_orders >= 5 OR ch.max_daily_spent > 2000 THEN 'HIGH'
        WHEN ch.max_daily_orders >= 3 OR ch.max_daily_spent > 1000 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS velocity_risk
FROM customer_history ch
JOIN customers c ON ch.customer_id = c.customer_id
WHERE ch.max_daily_orders >= 3 OR ch.max_daily_spent > 1000
ORDER BY ch.max_daily_orders DESC, ch.max_daily_spent DESC;

-- ========================================
-- 6. MISMATCHED ORDER INFORMATION
-- ========================================

-- Orders with suspicious mismatches (email/name/address patterns)
SELECT 
    o.order_id,
    o.customer_id,
    c.email,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.phone,
    c.city AS billing_city,
    c.state AS billing_state,
    sa.city AS shipping_city,
    sa.state AS shipping_state,
    o.total_amount,
    o.order_date,
    o.status,
    o.payment_status,
    CASE 
        WHEN c.email LIKE '%test%' OR c.email LIKE '%fake%' OR c.email LIKE '%temp%' THEN 'SUSPICIOUS_EMAIL'
        ELSE NULL
    END AS email_flag,
    CASE 
        WHEN c.first_name = c.last_name THEN 'SAME_FIRST_LAST_NAME'
        WHEN LENGTH(c.first_name) <= 2 OR LENGTH(c.last_name) <= 2 THEN 'SHORT_NAME'
        ELSE NULL
    END AS name_flag,
    CASE 
        WHEN c.state != sa.state THEN 'DIFFERENT_STATE'
        ELSE NULL
    END AS address_flag,
    CASE 
        WHEN c.phone IS NULL OR LENGTH(c.phone) < 10 THEN 'INVALID_PHONE'
        ELSE NULL
    END AS phone_flag
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id AND sa.is_default = TRUE
WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
  AND (
    c.email LIKE '%test%' OR c.email LIKE '%fake%' OR c.email LIKE '%temp%'
    OR c.first_name = c.last_name
    OR LENGTH(c.first_name) <= 2 OR LENGTH(c.last_name) <= 2
    OR c.state != sa.state
    OR c.phone IS NULL OR LENGTH(c.phone) < 10
  )
ORDER BY o.total_amount DESC, o.order_date DESC;

-- ========================================
-- 7. HIGH-RISK PRODUCT COMBINATIONS
-- ========================================

-- Orders with commonly fraudulent product combinations (high-value electronics, gift cards)
WITH order_categories AS (
    SELECT 
        o.order_id,
        o.customer_id,
        o.total_amount,
        o.order_date,
        GROUP_CONCAT(DISTINCT pc.category_name ORDER BY pc.category_name) AS categories,
        COUNT(DISTINCT oi.product_id) AS unique_products,
        SUM(oi.quantity) AS total_items,
        COUNT(DISTINCT pc.category_id) AS category_count
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
    GROUP BY o.order_id, o.customer_id, o.total_amount, o.order_date
)
SELECT 
    oc.order_id,
    oc.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    oc.order_date,
    oc.total_amount,
    oc.categories,
    oc.unique_products,
    oc.total_items,
    oc.category_count,
    o.status,
    o.payment_status,
    DATEDIFF(oc.order_date, c.created_at) AS days_since_signup,
    CASE 
        WHEN oc.total_amount > 2000 AND oc.category_count = 1 THEN 'HIGH_VALUE_SINGLE_CATEGORY'
        WHEN oc.unique_products >= 10 AND TIMESTAMPDIFF(HOUR, c.created_at, oc.order_date) < 24 THEN 'NEW_ACCOUNT_BULK_BUY'
        WHEN oc.total_items > 20 AND oc.unique_products <= 3 THEN 'BULK_SAME_ITEMS'
        ELSE 'REVIEW'
    END AS fraud_pattern
FROM order_categories oc
JOIN customers c ON oc.customer_id = c.customer_id
JOIN orders o ON oc.order_id = o.order_id
WHERE oc.total_amount > 1000
   OR (oc.unique_products >= 10 AND TIMESTAMPDIFF(HOUR, c.created_at, oc.order_date) < 24)
   OR (oc.total_items > 20 AND oc.unique_products <= 3)
ORDER BY oc.total_amount DESC, oc.order_date DESC;

-- ========================================
-- 8. REFUND ABUSE PATTERNS
-- ========================================

-- Customers with excessive return/refund patterns
WITH customer_returns AS (
    SELECT 
        c.customer_id,
        COUNT(DISTINCT r.return_id) AS total_returns,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(r.refund_amount) AS total_refunded,
        SUM(o.total_amount) AS total_spent,
        ROUND((COUNT(DISTINCT r.return_id) / NULLIF(COUNT(DISTINCT o.order_id), 0)) * 100, 2) AS return_rate,
        MAX(r.created_at) AS last_return_date,
        GROUP_CONCAT(DISTINCT r.reason ORDER BY r.created_at DESC SEPARATOR ', ') AS return_reasons
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY c.customer_id
    HAVING COUNT(DISTINCT r.return_id) >= 3
)
SELECT 
    cr.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.status,
    cr.total_returns,
    cr.total_orders,
    cr.return_rate,
    ROUND(cr.total_refunded, 2) AS total_refunded,
    ROUND(cr.total_spent, 2) AS total_spent,
    ROUND((cr.total_refunded / NULLIF(cr.total_spent, 0)) * 100, 2) AS refund_rate,
    cr.last_return_date,
    cr.return_reasons,
    c.created_at AS customer_since,
    CASE 
        WHEN cr.return_rate >= 50 AND cr.total_returns >= 5 THEN 'CRITICAL_ABUSE'
        WHEN cr.return_rate >= 30 AND cr.total_returns >= 3 THEN 'HIGH_RISK'
        WHEN cr.return_rate >= 20 THEN 'MEDIUM_RISK'
        ELSE 'MONITOR'
    END AS abuse_risk
FROM customer_returns cr
JOIN customers c ON cr.customer_id = c.customer_id
WHERE cr.return_rate >= 20
ORDER BY cr.return_rate DESC, cr.total_returns DESC;

-- ========================================
-- 9. DUPLICATE CUSTOMER DETECTION
-- ========================================

-- Identify potential duplicate accounts (same person, multiple accounts)
WITH customer_similarity AS (
    SELECT 
        c1.customer_id AS customer_id_1,
        c2.customer_id AS customer_id_2,
        CONCAT(c1.first_name, ' ', c1.last_name) AS name_1,
        CONCAT(c2.first_name, ' ', c2.last_name) AS name_2,
        c1.email AS email_1,
        c2.email AS email_2,
        c1.phone AS phone_1,
        c2.phone AS phone_2,
        c1.address_line1 AS address_1,
        c2.address_line1 AS address_2,
        CASE 
            WHEN c1.phone = c2.phone AND c1.phone IS NOT NULL THEN 'SAME_PHONE'
            ELSE NULL
        END AS phone_match,
        CASE 
            WHEN c1.address_line1 = c2.address_line1 AND c1.address_line1 IS NOT NULL THEN 'SAME_ADDRESS'
            ELSE NULL
        END AS address_match,
        CASE 
            WHEN SUBSTRING_INDEX(c1.email, '@', 1) = SUBSTRING_INDEX(c2.email, '@', 1) THEN 'SIMILAR_EMAIL'
            ELSE NULL
        END AS email_match,
        CASE 
            WHEN SOUNDEX(CONCAT(c1.first_name, c1.last_name)) = SOUNDEX(CONCAT(c2.first_name, c2.last_name)) THEN 'SIMILAR_NAME'
            ELSE NULL
        END AS name_match
    FROM customers c1
    JOIN customers c2 ON c1.customer_id < c2.customer_id
    WHERE (c1.phone = c2.phone AND c1.phone IS NOT NULL)
       OR (c1.address_line1 = c2.address_line1 AND c1.zip_code = c2.zip_code AND c1.address_line1 IS NOT NULL)
       OR (SUBSTRING_INDEX(c1.email, '@', 1) = SUBSTRING_INDEX(c2.email, '@', 1))
       OR (SOUNDEX(CONCAT(c1.first_name, c1.last_name)) = SOUNDEX(CONCAT(c2.first_name, c2.last_name))
           AND c1.zip_code = c2.zip_code)
)
SELECT 
    cs.*,
    o1.order_count AS orders_account_1,
    o2.order_count AS orders_account_2,
    CASE 
        WHEN phone_match IS NOT NULL AND address_match IS NOT NULL THEN 'VERY_HIGH'
        WHEN phone_match IS NOT NULL OR address_match IS NOT NULL THEN 'HIGH'
        WHEN email_match IS NOT NULL AND name_match IS NOT NULL THEN 'MEDIUM'
        ELSE 'LOW'
    END AS duplicate_probability
FROM customer_similarity cs
LEFT JOIN (
    SELECT customer_id, COUNT(order_id) AS order_count 
    FROM orders 
    GROUP BY customer_id
) o1 ON cs.customer_id_1 = o1.customer_id
LEFT JOIN (
    SELECT customer_id, COUNT(order_id) AS order_count 
    FROM orders 
    GROUP BY customer_id
) o2 ON cs.customer_id_2 = o2.customer_id
ORDER BY duplicate_probability DESC, cs.customer_id_1;

-- ========================================
-- 10. ACCOUNT TAKEOVER INDICATORS
-- ========================================

-- Sudden changes in customer behavior (potential account compromise)
WITH customer_behavior AS (
    SELECT 
        customer_id,
        DATE(order_date) AS order_day,
        COUNT(order_id) AS daily_orders,
        SUM(total_amount) AS daily_spent,
        AVG(total_amount) AS avg_order_value
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
    GROUP BY customer_id, DATE(order_date)
),
customer_baseline AS (
    SELECT 
        customer_id,
        AVG(daily_orders) AS baseline_daily_orders,
        STDDEV(daily_orders) AS stddev_orders,
        AVG(daily_spent) AS baseline_daily_spent,
        STDDEV(daily_spent) AS stddev_spent,
        MAX(order_day) AS last_order_day
    FROM customer_behavior
    WHERE order_day <= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    GROUP BY customer_id
),
recent_behavior AS (
    SELECT 
        customer_id,
        AVG(daily_orders) AS recent_daily_orders,
        AVG(daily_spent) AS recent_daily_spent
    FROM customer_behavior
    WHERE order_day > DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    GROUP BY customer_id
)
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.phone,
    ROUND(cb.baseline_daily_orders, 2) AS baseline_orders,
    ROUND(rb.recent_daily_orders, 2) AS recent_orders,
    ROUND(cb.baseline_daily_spent, 2) AS baseline_spent,
    ROUND(rb.recent_daily_spent, 2) AS recent_spent,
    ROUND(((rb.recent_daily_orders - cb.baseline_daily_orders) / NULLIF(cb.baseline_daily_orders, 0)) * 100, 2) AS order_change_pct,
    ROUND(((rb.recent_daily_spent - cb.baseline_daily_spent) / NULLIF(cb.baseline_daily_spent, 0)) * 100, 2) AS spend_change_pct,
    cb.last_order_day,
    c.updated_at AS account_last_updated,
    CASE 
        WHEN rb.recent_daily_orders > cb.baseline_daily_orders * 3 
         AND rb.recent_daily_spent > cb.baseline_daily_spent * 3 THEN 'CRITICAL_TAKEOVER'
        WHEN rb.recent_daily_orders > cb.baseline_daily_orders * 2 
          OR rb.recent_daily_spent > cb.baseline_daily_spent * 2 THEN 'HIGH_RISK'
        ELSE 'MONITOR'
    END AS takeover_risk
FROM customer_baseline cb
JOIN recent_behavior rb ON cb.customer_id = rb.customer_id
JOIN customers c ON cb.customer_id = c.customer_id
WHERE rb.recent_daily_orders > cb.baseline_daily_orders * 2
   OR rb.recent_daily_spent > cb.baseline_daily_spent * 2
ORDER BY order_change_pct DESC;

-- ========================================
-- 11. INTERNATIONAL ORDER FRAUD RISK
-- ========================================

-- High-risk international orders
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.country AS billing_country,
    sa.country AS shipping_country,
    c.city AS billing_city,
    sa.city AS shipping_city,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    DATEDIFF(o.order_date, c.created_at) AS days_since_signup,
    COUNT(oi.order_item_id) AS item_count,
    CASE 
        WHEN c.country != sa.country THEN 'INTERNATIONAL_SHIPPING'
        ELSE NULL
    END AS country_mismatch,
    CASE 
        WHEN c.country != 'USA' AND o.total_amount > 1000 THEN 'HIGH_VALUE_INTERNATIONAL'
        ELSE NULL
    END AS high_value_flag,
    CASE 
        WHEN DATEDIFF(o.order_date, c.created_at) < 7 AND o.total_amount > 500 THEN 'NEW_ACCOUNT_HIGH_VALUE'
        ELSE NULL
    END AS new_account_flag
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id AND sa.is_default = TRUE
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
  AND (
    (c.country != 'USA' AND o.total_amount > 500)
    OR (c.country != sa.country)
    OR (DATEDIFF(o.order_date, c.created_at) < 7 AND o.total_amount > 500)
  )
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, c.email, c.country,
         sa.country, c.city, sa.city, o.order_date, o.total_amount, o.status,
         o.payment_status, c.created_at
ORDER BY o.total_amount DESC, o.order_date DESC;

-- ========================================
-- 12. COMPREHENSIVE FRAUD RISK SCORE
-- ========================================

-- Multi-factor fraud risk scoring system
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        c.status,
        c.created_at,
        DATEDIFF(CURDATE(), c.created_at) AS account_age_days,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_spent,
        AVG(o.total_amount) AS avg_order_value,
        COUNT(DISTINCT DATE(o.order_date)) AS active_days,
        MAX(o.order_date) AS last_order_date,
        SUM(CASE WHEN o.payment_status = 'failed' THEN 1 ELSE 0 END) AS failed_payments,
        SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
        COUNT(DISTINCT sa.address_id) AS shipping_addresses,
        COUNT(DISTINCT r.return_id) AS total_returns
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
       OR o.order_id IS NULL
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.status, c.created_at
)
SELECT 
    cm.customer_id,
    cm.customer_name,
    cm.email,
    cm.account_age_days,
    cm.total_orders,
    ROUND(cm.total_spent, 2) AS total_spent,
    ROUND(cm.avg_order_value, 2) AS avg_order_value,
    cm.failed_payments,
    cm.cancelled_orders,
    cm.shipping_addresses,
    cm.total_returns,
    -- Risk score calculation (0-100)
    (
        CASE WHEN cm.account_age_days < 7 THEN 20 ELSE 0 END +  -- New account
        CASE WHEN cm.total_orders > 10 AND cm.account_age_days < 30 THEN 15 ELSE 0 END +  -- High velocity
        CASE WHEN cm.failed_payments >= 3 THEN 20 ELSE cm.failed_payments * 5 END +  -- Failed payments
        CASE WHEN cm.shipping_addresses >= 5 THEN 15 ELSE 0 END +  -- Multiple addresses
        CASE WHEN cm.avg_order_value > 1000 AND cm.total_orders < 5 THEN 15 ELSE 0 END +  -- High value, few orders
        CASE WHEN cm.total_returns > 0 AND (cm.total_returns / NULLIF(cm.total_orders, 0)) > 0.3 THEN 15 ELSE 0 END  -- High return rate
    ) AS fraud_risk_score,
    CASE 
        WHEN (
            CASE WHEN cm.account_age_days < 7 THEN 20 ELSE 0 END +
            CASE WHEN cm.total_orders > 10 AND cm.account_age_days < 30 THEN 15 ELSE 0 END +
            CASE WHEN cm.failed_payments >= 3 THEN 20 ELSE cm.failed_payments * 5 END +
            CASE WHEN cm.shipping_addresses >= 5 THEN 15 ELSE 0 END +
            CASE WHEN cm.avg_order_value > 1000 AND cm.total_orders < 5 THEN 15 ELSE 0 END +
            CASE WHEN cm.total_returns > 0 AND (cm.total_returns / NULLIF(cm.total_orders, 0)) > 0.3 THEN 15 ELSE 0 END
        ) >= 60 THEN 'CRITICAL'
        WHEN (
            CASE WHEN cm.account_age_days < 7 THEN 20 ELSE 0 END +
            CASE WHEN cm.total_orders > 10 AND cm.account_age_days < 30 THEN 15 ELSE 0 END +
            CASE WHEN cm.failed_payments >= 3 THEN 20 ELSE cm.failed_payments * 5 END +
            CASE WHEN cm.shipping_addresses >= 5 THEN 15 ELSE 0 END +
            CASE WHEN cm.avg_order_value > 1000 AND cm.total_orders < 5 THEN 15 ELSE 0 END +
            CASE WHEN cm.total_returns > 0 AND (cm.total_returns / NULLIF(cm.total_orders, 0)) > 0.3 THEN 15 ELSE 0 END
        ) >= 40 THEN 'HIGH'
        WHEN (
            CASE WHEN cm.account_age_days < 7 THEN 20 ELSE 0 END +
            CASE WHEN cm.total_orders > 10 AND cm.account_age_days < 30 THEN 15 ELSE 0 END +
            CASE WHEN cm.failed_payments >= 3 THEN 20 ELSE cm.failed_payments * 5 END +
            CASE WHEN cm.shipping_addresses >= 5 THEN 15 ELSE 0 END +
            CASE WHEN cm.avg_order_value > 1000 AND cm.total_orders < 5 THEN 15 ELSE 0 END +
            CASE WHEN cm.total_returns > 0 AND (cm.total_returns / NULLIF(cm.total_orders, 0)) > 0.3 THEN 15 ELSE 0 END
        ) >= 20 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_level,
    cm.last_order_date,
    cm.status AS customer_status
FROM customer_metrics cm
WHERE cm.total_orders > 0
ORDER BY fraud_risk_score DESC, cm.total_spent DESC
LIMIT 100;

-- ========================================
-- 13. SUSPICIOUS TIME PATTERN ORDERS
-- ========================================

-- Orders placed at unusual times with other risk factors
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    o.order_date,
    HOUR(o.order_date) AS order_hour,
    DAYNAME(o.order_date) AS day_of_week,
    o.total_amount,
    o.status,
    o.payment_status,
    COUNT(oi.order_item_id) AS item_count,
    DATEDIFF(o.order_date, c.created_at) AS account_age_at_order,
    CASE 
        WHEN HOUR(o.order_date) BETWEEN 2 AND 5 THEN 'LATE_NIGHT'
        ELSE NULL
    END AS time_flag,
    CASE 
        WHEN o.total_amount > 1500 THEN 'HIGH_VALUE'
        ELSE NULL
    END AS value_flag,
    CASE 
        WHEN DATEDIFF(o.order_date, c.created_at) < 1 THEN 'SAME_DAY_SIGNUP'
        ELSE NULL
    END AS account_flag
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
  AND (
    (HOUR(o.order_date) BETWEEN 2 AND 5 AND o.total_amount > 500)
    OR (HOUR(o.order_date) BETWEEN 2 AND 5 AND DATEDIFF(o.order_date, c.created_at) < 7)
  )
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, c.email, o.order_date,
         o.total_amount, o.status, o.payment_status, c.created_at
ORDER BY o.total_amount DESC, o.order_date DESC;

-- ========================================
-- END OF FRAUD INDICATORS ANALYSIS
-- ========================================