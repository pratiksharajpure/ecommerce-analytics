-- ========================================
-- PAYMENT_ANOMALIES.SQL
-- Data Quality Check: Payment Anomalies
-- Path: sql/core_analysis/payment_anomalies.sql
-- Identifies payment mismatches and failed transactions
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Paid Orders with Failed Payment Status
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    o.order_date,
    o.total_amount,
    o.status AS order_status,
    o.payment_status,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_old,
    'Delivered but payment failed' AS issue_type
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status IN ('delivered', 'shipped')
    AND o.payment_status IN ('failed', 'pending')
ORDER BY o.total_amount DESC;

-- ========================================
-- 2. Refunded Orders Still Marked as Paid
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    o.total_amount,
    o.payment_status,
    r.return_id,
    r.status AS return_status,
    r.refund_amount,
    r.refund_method,
    'Order refunded but still marked paid' AS issue_type
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN returns r ON o.order_id = r.order_id
WHERE r.status = 'refunded'
    AND o.payment_status = 'paid'
ORDER BY r.refund_amount DESC;

-- ========================================
-- 3. Multiple Payment Methods for Same Customer
-- ========================================
WITH customer_payment_count AS (
    SELECT 
        customer_id,
        COUNT(*) AS payment_method_count,
        COUNT(CASE WHEN is_default = TRUE THEN 1 END) AS default_count
    FROM payment_methods
    GROUP BY customer_id
)
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    cpc.payment_method_count,
    cpc.default_count,
    CASE 
        WHEN cpc.default_count = 0 THEN 'No default payment method'
        WHEN cpc.default_count > 1 THEN 'Multiple default payment methods'
    END AS issue_type
FROM customers c
INNER JOIN customer_payment_count cpc ON c.customer_id = cpc.customer_id
WHERE cpc.default_count != 1
ORDER BY cpc.payment_method_count DESC;

-- ========================================
-- 4. Expired Payment Methods Still Set as Default
-- ========================================
SELECT 
    pm.payment_method_id,
    pm.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    pm.payment_type,
    pm.card_brand,
    pm.card_last_four,
    CONCAT(pm.expiry_month, '/', pm.expiry_year) AS expiry_date,
    pm.is_default,
    COUNT(o.order_id) AS recent_orders,
    'Expired card set as default' AS issue_type
FROM payment_methods pm
INNER JOIN customers c ON pm.customer_id = c.customer_id
LEFT JOIN orders o ON c.customer_id = o.customer_id 
    AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
WHERE pm.is_default = TRUE
    AND pm.payment_type IN ('credit_card', 'debit_card')
    AND (
        (pm.expiry_year < YEAR(CURRENT_DATE)) OR
        (pm.expiry_year = YEAR(CURRENT_DATE) AND pm.expiry_month < MONTH(CURRENT_DATE))
    )
GROUP BY pm.payment_method_id, pm.customer_id, c.first_name, c.last_name, c.email,
         pm.payment_type, pm.card_brand, pm.card_last_four, pm.expiry_month, pm.expiry_year, pm.is_default
ORDER BY recent_orders DESC;

-- ========================================
-- 5. High-Value Orders with Pending Payment
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    o.order_date,
    o.total_amount,
    o.status AS order_status,
    o.payment_status,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_pending,
    'High-value order with pending payment' AS issue_type,
    'URGENT - Revenue at risk' AS priority
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.payment_status = 'pending'
    AND o.total_amount >= 500
    AND o.status NOT IN ('cancelled')
ORDER BY o.total_amount DESC, days_pending DESC;

-- ========================================
-- 6. Payment Status Changes Without Updates
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    o.order_date,
    o.created_at,
    o.updated_at,
    o.total_amount,
    o.status,
    o.payment_status,
    TIMESTAMPDIFF(DAY, o.created_at, o.updated_at) AS days_since_update,
    CASE 
        WHEN o.payment_status = 'pending' AND o.created_at = o.updated_at 
            AND DATEDIFF(CURRENT_DATE, DATE(o.created_at)) > 7 THEN 'Long-pending, never updated'
        WHEN o.payment_status = 'failed' AND DATEDIFF(CURRENT_DATE, DATE(o.updated_at)) > 30 THEN 'Failed payment, no follow-up'
    END AS issue_type
FROM orders o
WHERE (
    (o.payment_status = 'pending' AND o.created_at = o.updated_at 
     AND DATEDIFF(CURRENT_DATE, DATE(o.created_at)) > 7) OR
    (o.payment_status = 'failed' AND DATEDIFF(CURRENT_DATE, DATE(o.updated_at)) > 30)
)
ORDER BY days_since_update DESC;

-- ========================================
-- 7. Refund Amount Exceeds Order Total
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    o.total_amount AS order_total,
    r.refund_amount,
    ROUND(r.refund_amount - o.total_amount, 2) AS excess_refund,
    ROUND((r.refund_amount / o.total_amount) * 100, 2) AS refund_percentage,
    r.status AS return_status,
    'Refund exceeds order total' AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE r.refund_amount > o.total_amount
ORDER BY excess_refund DESC;

-- ========================================
-- 8. Orders with Suspicious Payment Patterns
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    o.order_date,
    o.total_amount,
    o.payment_status,
    COUNT(prev_orders.order_id) AS previous_failed_payments,
    SUM(CASE WHEN prev_orders.payment_status = 'failed' THEN prev_orders.total_amount ELSE 0 END) AS total_failed_amount,
    'Multiple failed payments from customer' AS issue_type
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN orders prev_orders ON c.customer_id = prev_orders.customer_id 
    AND prev_orders.payment_status = 'failed'
    AND prev_orders.order_date < o.order_date
    AND prev_orders.order_date >= DATE_SUB(o.order_date, INTERVAL 90 DAY)
WHERE o.payment_status IN ('pending', 'failed')
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, c.email,
         o.order_date, o.total_amount, o.payment_status
HAVING COUNT(prev_orders.order_id) >= 2
ORDER BY previous_failed_payments DESC, o.total_amount DESC;

-- ========================================
-- 9. Payment Method Mismatches
-- ========================================
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(CASE WHEN o.payment_status = 'paid' THEN 1 ELSE 0 END) AS paid_orders,
    SUM(CASE WHEN o.payment_status = 'failed' THEN 1 ELSE 0 END) AS failed_orders,
    COUNT(DISTINCT pm.payment_method_id) AS stored_payment_methods,
    'Active customer with no payment methods' AS issue_type
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN payment_methods pm ON c.customer_id = pm.customer_id
WHERE c.status = 'active'
    AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
GROUP BY c.customer_id, c.first_name, c.last_name, c.email
HAVING COUNT(DISTINCT pm.payment_method_id) = 0
ORDER BY total_orders DESC;

-- ========================================
-- 10. Payment Anomalies Summary Report
-- ========================================
SELECT 
    'Delivered Orders Not Paid' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(total_amount), 2) AS revenue_at_risk
FROM orders
WHERE status IN ('delivered', 'shipped')
    AND payment_status IN ('failed', 'pending')

UNION ALL

SELECT 
    'Refunded but Marked Paid' AS metric,
    COUNT(DISTINCT o.order_id) AS count,
    ROUND(SUM(r.refund_amount), 2) AS revenue_at_risk
FROM orders o
INNER JOIN returns r ON o.order_id = r.order_id
WHERE r.status = 'refunded' AND o.payment_status = 'paid'

UNION ALL

SELECT 
    'High-Value Pending Payments' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(total_amount), 2) AS revenue_at_risk
FROM orders
WHERE payment_status = 'pending'
    AND total_amount >= 500
    AND status NOT IN ('cancelled')

UNION ALL

SELECT 
    'Refunds Exceeding Order Total' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(r.refund_amount - o.total_amount), 2) AS revenue_at_risk
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
WHERE r.refund_amount > o.total_amount

UNION ALL

SELECT 
    'Expired Default Payment Methods' AS metric,
    COUNT(*) AS count,
    NULL AS revenue_at_risk
FROM payment_methods pm
WHERE pm.is_default = TRUE
    AND pm.payment_type IN ('credit_card', 'debit_card')
    AND (
        (pm.expiry_year < YEAR(CURRENT_DATE)) OR
        (pm.expiry_year = YEAR(CURRENT_DATE) AND pm.expiry_month < MONTH(CURRENT_DATE))
    )

UNION ALL

SELECT 
    'Customers with Multiple Defaults' AS metric,
    COUNT(*) AS count,
    NULL AS revenue_at_risk
FROM (
    SELECT customer_id
    FROM payment_methods
    WHERE is_default = TRUE
    GROUP BY customer_id
    HAVING COUNT(*) > 1
) AS multiple_defaults;

-- ========================================
-- 11. Failed Payment Recovery Opportunities
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.phone,
    o.order_date,
    o.total_amount,
    o.payment_status,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_since_order,
    COUNT(pm.payment_method_id) AS available_payment_methods,
    'Recovery opportunity' AS action_required
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN payment_methods pm ON c.customer_id = pm.customer_id
WHERE o.payment_status = 'failed'
    AND o.status NOT IN ('cancelled')
    AND DATEDIFF(CURRENT_DATE, DATE(o.order_date)) BETWEEN 1 AND 30
    AND c.status = 'active'
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, c.email, 
         c.phone, o.order_date, o.total_amount, o.payment_status
ORDER BY o.total_amount DESC;

-- ========================================
-- 12. Payment Velocity Anomalies
-- ========================================
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    COUNT(o.order_id) AS orders_today,
    SUM(o.total_amount) AS total_spent_today,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value,
    COUNT(CASE WHEN o.payment_status = 'paid' THEN 1 END) AS paid_orders,
    COUNT(CASE WHEN o.payment_status = 'failed' THEN 1 END) AS failed_orders,
    'Unusual payment velocity' AS issue_type
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE DATE(o.order_date) = CURRENT_DATE
GROUP BY c.customer_id, c.first_name, c.last_name, c.email
HAVING COUNT(o.order_id) >= 5
    OR SUM(o.total_amount) >= 5000
ORDER BY orders_today DESC, total_spent_today DESC;