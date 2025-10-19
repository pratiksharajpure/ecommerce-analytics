-- ========================================
-- RETURN_DATA_INTEGRITY.SQL
-- Data Quality Check: Return Record Validation
-- Path: sql/core_analysis/return_data_integrity.sql
-- Validates return records and refund integrity
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Returns Without Valid Orders
-- ========================================
SELECT 
    r.return_id,
    r.order_id AS invalid_order_id,
    r.reason,
    r.status,
    r.refund_amount,
    r.created_at,
    'Return references non-existent order' AS issue_type
FROM returns r
LEFT JOIN orders o ON r.order_id = o.order_id
WHERE o.order_id IS NULL
ORDER BY r.created_at DESC;

-- ========================================
-- 2. Returns Without Valid Order Items
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    r.order_item_id AS invalid_item_id,
    o.customer_id,
    r.reason,
    r.status,
    r.refund_amount,
    'Return references non-existent order item' AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
LEFT JOIN order_items oi ON r.order_item_id = oi.order_item_id
WHERE r.order_item_id IS NOT NULL
    AND oi.order_item_id IS NULL
ORDER BY r.created_at DESC;

-- ========================================
-- 3. Refund Amount Exceeds Order Item Value
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    r.order_item_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    oi.product_id,
    p.product_name,
    oi.subtotal AS item_value,
    r.refund_amount,
    ROUND(r.refund_amount - oi.subtotal, 2) AS excess_refund,
    r.status,
    'Refund exceeds item value' AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
INNER JOIN order_items oi ON r.order_item_id = oi.order_item_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE r.refund_amount > oi.subtotal
ORDER BY excess_refund DESC;

-- ========================================
-- 4. Returns Approved But Not Refunded
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    r.created_at AS return_date,
    r.updated_at AS last_update,
    DATEDIFF(CURRENT_DATE, r.updated_at) AS days_since_update,
    r.refund_amount,
    r.status,
    'Return approved but not refunded' AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE r.status IN ('approved', 'received')
    AND DATEDIFF(CURRENT_DATE, r.updated_at) > 7
ORDER BY days_since_update DESC;

-- ========================================
-- 5. Multiple Returns for Same Order Item
-- ========================================
SELECT 
    oi.order_item_id,
    oi.order_id,
    p.product_name,
    oi.quantity AS ordered_quantity,
    COUNT(r.return_id) AS return_count,
    SUM(r.refund_amount) AS total_refunded,
    oi.subtotal AS original_value,
    ROUND(SUM(r.refund_amount) - oi.subtotal, 2) AS over_refunded,
    'Multiple returns for same item' AS issue_type
FROM order_items oi
INNER JOIN returns r ON oi.order_item_id = r.order_item_id
LEFT JOIN products p ON oi.product_id = p.product_id
GROUP BY oi.order_item_id, oi.order_id, p.product_name, oi.quantity, oi.subtotal
HAVING COUNT(r.return_id) > 1
    OR SUM(r.refund_amount) > oi.subtotal
ORDER BY return_count DESC, over_refunded DESC;

-- ========================================
-- 6. Returns with NULL or Zero Refund Amount
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    r.order_item_id,
    oi.subtotal AS item_value,
    r.refund_amount,
    r.status,
    r.reason,
    CASE 
        WHEN r.refund_amount IS NULL THEN 'NULL refund amount'
        WHEN r.refund_amount = 0 THEN 'Zero refund amount'
        WHEN r.refund_amount < 0 THEN 'Negative refund amount'
    END AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON r.order_item_id = oi.order_item_id
WHERE r.status = 'refunded'
    AND (r.refund_amount IS NULL OR r.refund_amount <= 0)
ORDER BY r.created_at DESC;

-- ========================================
-- 7. Returns Requested for Cancelled Orders
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    o.total_amount,
    o.status AS order_status,
    o.payment_status,
    r.created_at AS return_date,
    r.status AS return_status,
    r.refund_amount,
    'Return requested for cancelled order' AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'cancelled'
ORDER BY r.created_at DESC;

-- ========================================
-- 8. Return Status vs Order Payment Status Conflicts
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    r.status AS return_status,
    o.payment_status,
    r.refund_amount,
    r.refund_method,
    CASE 
        WHEN r.status = 'refunded' AND o.payment_status != 'refunded' THEN 'Return refunded but order not marked refunded'
        WHEN r.status = 'refunded' AND o.payment_status = 'pending' THEN 'Refund issued for unpaid order'
        WHEN r.status = 'refunded' AND o.payment_status = 'failed' THEN 'Refund issued for failed payment'
    END AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE (r.status = 'refunded' AND o.payment_status NOT IN ('refunded', 'paid'))
    OR (r.status = 'refunded' AND o.payment_status IN ('pending', 'failed'))
ORDER BY r.refund_amount DESC;

-- ========================================
-- 9. High Return Rate Products
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    COUNT(DISTINCT r.return_id) AS total_returns,
    ROUND((COUNT(DISTINCT r.return_id) * 100.0) / COUNT(DISTINCT oi.order_id), 2) AS return_rate_pct,
    SUM(oi.subtotal) AS total_revenue,
    SUM(CASE WHEN r.status = 'refunded' THEN r.refund_amount ELSE 0 END) AS total_refunded,
    'High return rate - quality issue?' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
INNER JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name
HAVING COUNT(DISTINCT oi.order_id) >= 10
    AND (COUNT(DISTINCT r.return_id) * 100.0) / COUNT(DISTINCT oi.order_id) > 20
ORDER BY return_rate_pct DESC;

-- ========================================
-- 10. Returns with Missing Reason Details
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    r.reason,
    r.reason_details,
    r.status,
    r.refund_amount,
    r.created_at,
    CASE 
        WHEN r.reason_details IS NULL OR r.reason_details = '' THEN 'Missing reason details'
        WHEN r.reason = 'other' AND (r.reason_details IS NULL OR r.reason_details = '') THEN 'Reason "other" without explanation'
    END AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE (r.reason_details IS NULL OR r.reason_details = '')
    OR (r.reason = 'other' AND (r.reason_details IS NULL OR r.reason_details = ''))
ORDER BY r.refund_amount DESC;

-- ========================================
-- 11. Stuck Return Requests
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    r.created_at AS return_requested,
    r.updated_at AS last_updated,
    DATEDIFF(CURRENT_DATE, r.created_at) AS days_pending,
    r.status,
    r.refund_amount,
    CASE 
        WHEN r.status = 'requested' AND DATEDIFF(CURRENT_DATE, r.created_at) > 14 THEN 'Return request pending >14 days'
        WHEN r.status = 'approved' AND DATEDIFF(CURRENT_DATE, r.updated_at) > 7 THEN 'Approved but not received >7 days'
        WHEN r.status = 'received' AND DATEDIFF(CURRENT_DATE, r.updated_at) > 5 THEN 'Received but not refunded >5 days'
    END AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE (r.status = 'requested' AND DATEDIFF(CURRENT_DATE, r.created_at) > 14)
    OR (r.status = 'approved' AND DATEDIFF(CURRENT_DATE, r.updated_at) > 7)
    OR (r.status = 'received' AND DATEDIFF(CURRENT_DATE, r.updated_at) > 5)
ORDER BY days_pending DESC;

-- ========================================
-- 12. Return Data Integrity Summary
-- ========================================
SELECT 
    'Total Returns' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(refund_amount), 2) AS total_amount
FROM returns

UNION ALL

SELECT 
    'Returns Without Valid Orders' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(r.refund_amount), 2) AS total_amount
FROM returns r
LEFT JOIN orders o ON r.order_id = o.order_id
WHERE o.order_id IS NULL

UNION ALL

SELECT 
    'Refunds Exceeding Item Value' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(r.refund_amount - oi.subtotal), 2) AS total_amount
FROM returns r
INNER JOIN order_items oi ON r.order_item_id = oi.order_item_id
WHERE r.refund_amount > oi.subtotal

UNION ALL

SELECT 
    'Approved But Not Refunded (>7 days)' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(refund_amount), 2) AS total_amount
FROM returns
WHERE status IN ('approved', 'received')
    AND DATEDIFF(CURRENT_DATE, updated_at) > 7

UNION ALL

SELECT 
    'Returns for Cancelled Orders' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(r.refund_amount), 2) AS total_amount
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
WHERE o.status = 'cancelled'

UNION ALL

SELECT 
    'Refunded with Zero/NULL Amount' AS metric,
    COUNT(*) AS count,
    NULL AS total_amount
FROM returns
WHERE status = 'refunded'
    AND (refund_amount IS NULL OR refund_amount <= 0)

UNION ALL

SELECT 
    'Stuck Return Requests' AS metric,
    COUNT(*) AS count,
    ROUND(SUM(refund_amount), 2) AS total_amount
FROM returns
WHERE (status = 'requested' AND DATEDIFF(CURRENT_DATE, created_at) > 14)
    OR (status = 'approved' AND DATEDIFF(CURRENT_DATE, updated_at) > 7)
    OR (status = 'received' AND DATEDIFF(CURRENT_DATE, updated_at) > 5);

-- ========================================
-- 13. Return Fraud Indicators
-- ========================================
SELECT 
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    COUNT(DISTINCT r.return_id) AS total_returns,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND((COUNT(DISTINCT r.return_id) * 100.0) / COUNT(DISTINCT o.order_id), 2) AS return_rate_pct,
    SUM(CASE WHEN r.status = 'refunded' THEN r.refund_amount ELSE 0 END) AS total_refunded,
    SUM(o.total_amount) AS total_spent,
    'Suspicious return pattern' AS issue_type
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN returns r ON o.order_id = r.order_id
WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 180 DAY)
GROUP BY o.customer_id, c.first_name, c.last_name, c.email
HAVING COUNT(DISTINCT r.return_id) >= 5
    AND (COUNT(DISTINCT r.return_id) * 100.0) / COUNT(DISTINCT o.order_id) > 50
ORDER BY return_rate_pct DESC, total_returns DESC;