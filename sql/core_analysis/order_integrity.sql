-- ========================================
-- ORDER_INTEGRITY.SQL
-- Data Quality Check: Order Validation
-- Path: sql/core_analysis/order_integrity.sql
-- Validates order data integrity and consistency
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Orders with Mismatched Total Amounts
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    o.total_amount AS recorded_total,
    ROUND(SUM(oi.subtotal), 2) AS calculated_subtotal,
    o.shipping_cost,
    o.tax_amount,
    ROUND(SUM(oi.subtotal) + o.shipping_cost + o.tax_amount, 2) AS calculated_total,
    ROUND(o.total_amount - (SUM(oi.subtotal) + o.shipping_cost + o.tax_amount), 2) AS discrepancy,
    o.status,
    'Total amount mismatch' AS issue_type
FROM orders o
INNER JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, o.order_date,
         o.total_amount, o.shipping_cost, o.tax_amount, o.status
HAVING ABS(o.total_amount - (SUM(oi.subtotal) + o.shipping_cost + o.tax_amount)) > 0.01
ORDER BY ABS(discrepancy) DESC;

-- ========================================
-- 2. Orders Without Order Items
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    'Order has no items' AS issue_type,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_old
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE oi.order_item_id IS NULL
ORDER BY o.order_date DESC;

-- ========================================
-- 3. Orders with Zero or Negative Total Amount
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    o.total_amount,
    o.shipping_cost,
    o.tax_amount,
    COUNT(oi.order_item_id) AS item_count,
    SUM(oi.subtotal) AS items_subtotal,
    o.status,
    o.payment_status,
    CASE 
        WHEN o.total_amount = 0 THEN 'Zero total amount'
        WHEN o.total_amount < 0 THEN 'Negative total amount'
    END AS issue_type
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.total_amount <= 0
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, 
         o.order_date, o.total_amount, o.shipping_cost, o.tax_amount, o.status, o.payment_status
ORDER BY o.total_amount ASC;

-- ========================================
-- 4. Payment Status vs Order Status Conflicts
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    o.total_amount,
    o.status AS order_status,
    o.payment_status,
    CASE 
        WHEN o.status IN ('shipped', 'delivered') AND o.payment_status != 'paid' THEN 'Shipped without payment'
        WHEN o.status = 'cancelled' AND o.payment_status = 'paid' THEN 'Cancelled but paid'
        WHEN o.status = 'pending' AND o.payment_status = 'paid' THEN 'Paid but still pending'
        WHEN o.status = 'delivered' AND o.payment_status IN ('pending', 'failed') THEN 'Delivered without payment'
    END AS issue_type,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_since_order
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE (
    (o.status IN ('shipped', 'delivered') AND o.payment_status != 'paid') OR
    (o.status = 'cancelled' AND o.payment_status = 'paid') OR
    (o.status = 'pending' AND o.payment_status = 'paid' AND DATEDIFF(CURRENT_DATE, DATE(o.order_date)) > 7) OR
    (o.status = 'delivered' AND o.payment_status IN ('pending', 'failed'))
)
ORDER BY o.order_date DESC;

-- ========================================
-- 5. Orders with Invalid Shipping or Tax
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    o.order_date,
    o.total_amount,
    o.shipping_cost,
    o.tax_amount,
    SUM(oi.subtotal) AS items_subtotal,
    ROUND((o.tax_amount / NULLIF(SUM(oi.subtotal), 0)) * 100, 2) AS tax_rate_pct,
    o.status,
    CASE 
        WHEN o.shipping_cost < 0 THEN 'Negative shipping cost'
        WHEN o.tax_amount < 0 THEN 'Negative tax amount'
        WHEN o.shipping_cost > o.total_amount THEN 'Shipping exceeds total'
        WHEN o.tax_amount > o.total_amount THEN 'Tax exceeds total'
        WHEN (o.tax_amount / NULLIF(SUM(oi.subtotal), 0)) > 0.20 THEN 'Tax rate over 20%'
    END AS issue_type
FROM orders o
INNER JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.customer_id, o.order_date, o.total_amount, 
         o.shipping_cost, o.tax_amount, o.status
HAVING 
    o.shipping_cost < 0 OR
    o.tax_amount < 0 OR
    o.shipping_cost > o.total_amount OR
    o.tax_amount > o.total_amount OR
    (o.tax_amount / NULLIF(SUM(oi.subtotal), 0)) > 0.20
ORDER BY o.order_date DESC;

-- ========================================
-- 6. Duplicate Order Detection
-- ========================================
SELECT 
    o1.order_id AS order_id_1,
    o2.order_id AS order_id_2,
    o1.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o1.order_date AS order_date_1,
    o2.order_date AS order_date_2,
    o1.total_amount AS amount_1,
    o2.total_amount AS amount_2,
    TIMESTAMPDIFF(MINUTE, o1.order_date, o2.order_date) AS minutes_apart,
    'Potential duplicate order' AS issue_type
FROM orders o1
INNER JOIN orders o2 ON o1.customer_id = o2.customer_id
    AND o1.order_id < o2.order_id
    AND ABS(o1.total_amount - o2.total_amount) < 0.01
    AND TIMESTAMPDIFF(MINUTE, o1.order_date, o2.order_date) <= 60
LEFT JOIN customers c ON o1.customer_id = c.customer_id
WHERE o1.status != 'cancelled'
    AND o2.status != 'cancelled'
ORDER BY minutes_apart ASC;

-- ========================================
-- 7. Orders Stuck in Processing
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_in_status,
    CASE 
        WHEN o.status = 'pending' AND DATEDIFF(CURRENT_DATE, DATE(o.order_date)) > 7 THEN 'Pending too long'
        WHEN o.status = 'processing' AND DATEDIFF(CURRENT_DATE, DATE(o.order_date)) > 5 THEN 'Processing too long'
        WHEN o.status = 'shipped' AND DATEDIFF(CURRENT_DATE, DATE(o.order_date)) > 14 THEN 'Shipped but not delivered'
    END AS issue_type
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE (
    (o.status = 'pending' AND DATEDIFF(CURRENT_DATE, DATE(o.order_date)) > 7) OR
    (o.status = 'processing' AND DATEDIFF(CURRENT_DATE, DATE(o.order_date)) > 5) OR
    (o.status = 'shipped' AND DATEDIFF(CURRENT_DATE, DATE(o.order_date)) > 14)
)
ORDER BY days_in_status DESC;

-- ========================================
-- 8. Order Items with Price Mismatches
-- ========================================
SELECT 
    oi.order_item_id,
    oi.order_id,
    o.order_date,
    p.product_id,
    p.sku,
    p.product_name,
    p.price AS current_price,
    oi.unit_price AS order_unit_price,
    ROUND(p.price - oi.unit_price, 2) AS price_difference,
    ROUND(((p.price - oi.unit_price) / p.price) * 100, 2) AS price_diff_pct,
    oi.quantity,
    oi.subtotal,
    'Significant price variance' AS issue_type
FROM order_items oi
INNER JOIN orders o ON oi.order_id = o.order_id
INNER JOIN products p ON oi.product_id = p.product_id
WHERE ABS(p.price - oi.unit_price) > (p.price * 0.50)  -- More than 50% difference
    AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
ORDER BY ABS(price_difference) DESC;

-- ========================================
-- 9. Order Integrity Summary Report
-- ========================================
SELECT 
    'Total Orders' AS metric,
    COUNT(*) AS count,
    '100%' AS percentage
FROM orders

UNION ALL

SELECT 
    'Orders Without Items' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2), '%') AS percentage
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE oi.order_item_id IS NULL

UNION ALL

SELECT 
    'Total Amount Mismatches' AS metric,
    COUNT(DISTINCT o.order_id) AS count,
    CONCAT(ROUND(COUNT(DISTINCT o.order_id) * 100.0 / (SELECT COUNT(*) FROM orders), 2), '%') AS percentage
FROM orders o
INNER JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.total_amount, o.shipping_cost, o.tax_amount
HAVING ABS(o.total_amount - (SUM(oi.subtotal) + o.shipping_cost + o.tax_amount)) > 0.01

UNION ALL

SELECT 
    'Payment/Status Conflicts' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2), '%') AS percentage
FROM orders
WHERE (
    (status IN ('shipped', 'delivered') AND payment_status != 'paid') OR
    (status = 'cancelled' AND payment_status = 'paid') OR
    (status = 'delivered' AND payment_status IN ('pending', 'failed'))
)

UNION ALL

SELECT 
    'Invalid Shipping/Tax' AS metric,
    COUNT(DISTINCT o.order_id) AS count,
    CONCAT(ROUND(COUNT(DISTINCT o.order_id) * 100.0 / (SELECT COUNT(*) FROM orders), 2), '%') AS percentage
FROM orders o
WHERE o.shipping_cost < 0 OR o.tax_amount < 0

UNION ALL

SELECT 
    'Stuck in Processing' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2), '%') AS percentage
FROM orders
WHERE (
    (status = 'pending' AND DATEDIFF(CURRENT_DATE, DATE(order_date)) > 7) OR
    (status = 'processing' AND DATEDIFF(CURRENT_DATE, DATE(order_date)) > 5) OR
    (status = 'shipped' AND DATEDIFF(CURRENT_DATE, DATE(order_date)) > 14)
);

-- ========================================
-- 10. High-Value Orders with Issues
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    COUNT(oi.order_item_id) AS item_count,
    CASE 
        WHEN oi.order_item_id IS NULL THEN 'No items'
        WHEN o.payment_status != 'paid' AND o.status IN ('shipped', 'delivered') THEN 'Not paid'
        WHEN ABS(o.total_amount - (SUM(oi.subtotal) + o.shipping_cost + o.tax_amount)) > 0.01 THEN 'Amount mismatch'
    END AS critical_issue,
    'High-value order - needs attention' AS priority
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.total_amount >= 1000
    AND (
        oi.order_item_id IS NULL OR
        (o.payment_status != 'paid' AND o.status IN ('shipped', 'delivered')) OR
        o.status IN ('pending', 'processing')
    )
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, c.email,
         o.order_date, o.total_amount, o.status, o.payment_status, o.shipping_cost, o.tax_amount
ORDER BY o.total_amount DESC;