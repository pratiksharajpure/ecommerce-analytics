-- ========================================
-- EXCEPTION REPORT - E-COMMERCE ANALYTICS
-- Data Exceptions, Outliers & Anomaly Detection
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. EXECUTIVE EXCEPTION SUMMARY
-- ========================================
SELECT 
    'EXCEPTION SUMMARY DASHBOARD' AS report_section,
    '===========================' AS separator;

-- Count all critical exceptions
SELECT 
    'Critical Exceptions Overview' AS metric,
    (SELECT COUNT(*) FROM orders WHERE total_amount <= 0 OR total_amount IS NULL) AS invalid_order_amounts,
    (SELECT COUNT(*) FROM products WHERE price <= 0 OR price IS NULL) AS invalid_product_prices,
    (SELECT COUNT(*) FROM inventory WHERE quantity_available < 0) AS negative_inventory,
    (SELECT COUNT(*) FROM customers WHERE email IS NULL OR email = '' OR email NOT LIKE '%@%') AS invalid_emails,
    (SELECT COUNT(*) FROM orders WHERE order_date > NOW()) AS future_dated_orders,
    (SELECT COUNT(*) FROM returns WHERE refund_amount > (SELECT MAX(total_amount) FROM orders)) AS excessive_refunds,
    CASE 
        WHEN (
            (SELECT COUNT(*) FROM orders WHERE total_amount <= 0) +
            (SELECT COUNT(*) FROM products WHERE price <= 0) +
            (SELECT COUNT(*) FROM inventory WHERE quantity_available < 0)
        ) > 100 THEN 'CRITICAL - IMMEDIATE ATTENTION REQUIRED'
        WHEN (
            (SELECT COUNT(*) FROM orders WHERE total_amount <= 0) +
            (SELECT COUNT(*) FROM products WHERE price <= 0) +
            (SELECT COUNT(*) FROM inventory WHERE quantity_available < 0)
        ) > 50 THEN 'HIGH - ACTION NEEDED'
        WHEN (
            (SELECT COUNT(*) FROM orders WHERE total_amount <= 0) +
            (SELECT COUNT(*) FROM products WHERE price <= 0) +
            (SELECT COUNT(*) FROM inventory WHERE quantity_available < 0)
        ) > 10 THEN 'MEDIUM - MONITOR CLOSELY'
        ELSE 'LOW - WITHIN ACCEPTABLE RANGE'
    END AS exception_severity;

-- ========================================
-- 2. ORDER DATA EXCEPTIONS
-- ========================================
SELECT 
    '' AS blank_line,
    'ORDER DATA EXCEPTIONS' AS report_section,
    '=====================' AS separator;

-- Invalid order amounts
SELECT 
    'Invalid Order Amounts' AS exception_type,
    o.order_id,
    o.customer_id,
    c.email AS customer_email,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    CASE 
        WHEN o.total_amount IS NULL THEN 'NULL amount'
        WHEN o.total_amount < 0 THEN 'Negative amount'
        WHEN o.total_amount = 0 THEN 'Zero amount'
        ELSE 'Unknown issue'
    END AS exception_reason,
    'CRITICAL' AS severity
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.total_amount <= 0 OR o.total_amount IS NULL
ORDER BY o.order_date DESC
LIMIT 50;

-- Order-items mismatch (order total doesn't match sum of items)
SELECT 
    'Order Amount Mismatch' AS exception_type,
    o.order_id,
    o.total_amount AS recorded_total,
    COALESCE(SUM(oi.subtotal), 0) AS calculated_subtotal,
    o.shipping_cost,
    o.tax_amount,
    COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount AS calculated_total,
    ABS(o.total_amount - (COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount)) AS discrepancy,
    CASE 
        WHEN ABS(o.total_amount - (COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount)) > 100 
        THEN 'CRITICAL'
        WHEN ABS(o.total_amount - (COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount)) > 10 
        THEN 'HIGH'
        ELSE 'MEDIUM'
    END AS severity
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY o.order_id, o.total_amount, o.shipping_cost, o.tax_amount
HAVING ABS(o.total_amount - (COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount)) > 0.01
ORDER BY discrepancy DESC
LIMIT 50;

-- Orphaned order items (no parent order)
SELECT 
    'Orphaned Order Items' AS exception_type,
    oi.order_item_id,
    oi.order_id,
    oi.product_id,
    oi.quantity,
    oi.unit_price,
    oi.subtotal,
    'Order record missing' AS exception_reason,
    'HIGH' AS severity
FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL
LIMIT 50;

-- Future-dated orders
SELECT 
    'Future Dated Orders' AS exception_type,
    o.order_id,
    o.customer_id,
    c.email AS customer_email,
    o.order_date,
    DATEDIFF(o.order_date, NOW()) AS days_in_future,
    o.total_amount,
    o.status,
    'Invalid order date' AS exception_reason,
    'CRITICAL' AS severity
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date > NOW()
ORDER BY o.order_date DESC;

-- ========================================
-- 3. PRODUCT DATA EXCEPTIONS
-- ========================================
SELECT 
    '' AS blank_line,
    'PRODUCT DATA EXCEPTIONS' AS report_section,
    '=======================' AS separator;

-- Invalid pricing
SELECT 
    'Invalid Product Pricing' AS exception_type,
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    p.status,
    CASE 
        WHEN p.price IS NULL THEN 'NULL price'
        WHEN p.price <= 0 THEN 'Zero or negative price'
        WHEN p.cost > p.price THEN 'Cost exceeds price'
        WHEN p.cost IS NULL THEN 'NULL cost'
        ELSE 'Unknown issue'
    END AS exception_reason,
    CASE 
        WHEN p.price IS NULL OR p.price <= 0 THEN 'CRITICAL'
        WHEN p.cost > p.price THEN 'HIGH'
        ELSE 'MEDIUM'
    END AS severity
FROM products p
WHERE p.price IS NULL 
    OR p.price <= 0 
    OR p.cost IS NULL 
    OR p.cost <= 0
    OR p.cost > p.price
ORDER BY severity DESC, p.product_id
LIMIT 50;

-- Products with no category
SELECT 
    'Uncategorized Products' AS exception_type,
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.status,
    'Missing category assignment' AS exception_reason,
    'MEDIUM' AS severity
FROM products p
WHERE p.category_id IS NULL
    AND p.status = 'active'
LIMIT 50;

-- Duplicate SKUs
SELECT 
    'Duplicate SKUs' AS exception_type,
    p.sku,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(p.product_id ORDER BY p.product_id SEPARATOR ', ') AS affected_product_ids,
    GROUP_CONCAT(p.product_name ORDER BY p.product_id SEPARATOR ' | ') AS product_names,
    'SKU uniqueness violation' AS exception_reason,
    'HIGH' AS severity
FROM products p
WHERE p.sku IS NOT NULL AND p.sku != ''
GROUP BY p.sku
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;

-- ========================================
-- 4. INVENTORY EXCEPTIONS
-- ========================================
SELECT 
    '' AS blank_line,
    'INVENTORY EXCEPTIONS' AS report_section,
    '====================' AS separator;

-- Negative inventory
SELECT 
    'Negative Inventory' AS exception_type,
    i.inventory_id,
    i.product_id,
    p.sku,
    p.product_name,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_available,
    i.reorder_level,
    'Negative available quantity' AS exception_reason,
    'CRITICAL' AS severity
FROM inventory i
JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_available < 0
ORDER BY i.quantity_available ASC
LIMIT 50;

-- Reserved quantity exceeds on-hand
SELECT 
    'Invalid Reserved Quantity' AS exception_type,
    i.inventory_id,
    i.product_id,
    p.sku,
    p.product_name,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_reserved - i.quantity_on_hand AS excess_reserved,
    'Reserved exceeds on-hand' AS exception_reason,
    'HIGH' AS severity
FROM inventory i
JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_reserved > i.quantity_on_hand
ORDER BY excess_reserved DESC
LIMIT 50;

-- Products with no inventory record
SELECT 
    'Missing Inventory Records' AS exception_type,
    p.product_id,
    p.sku,
    p.product_name,
    p.status,
    'No inventory record exists' AS exception_reason,
    CASE 
        WHEN p.status = 'active' THEN 'CRITICAL'
        ELSE 'MEDIUM'
    END AS severity
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
WHERE i.inventory_id IS NULL
    AND p.status IN ('active', 'out_of_stock')
LIMIT 50;

-- ========================================
-- 5. CUSTOMER DATA EXCEPTIONS
-- ========================================
SELECT 
    '' AS blank_line,
    'CUSTOMER DATA EXCEPTIONS' AS report_section,
    '========================' AS separator;

-- Invalid email addresses
SELECT 
    'Invalid Email Addresses' AS exception_type,
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    c.status,
    c.created_at,
    CASE 
        WHEN c.email IS NULL OR c.email = '' THEN 'Missing email'
        WHEN c.email NOT LIKE '%@%' THEN 'Invalid format - no @'
        WHEN c.email NOT LIKE '%.%' THEN 'Invalid format - no domain'
        WHEN c.email LIKE '%@%@%' THEN 'Multiple @ symbols'
        ELSE 'Format issue'
    END AS exception_reason,
    'HIGH' AS severity
FROM customers c
WHERE c.status = 'active'
    AND (
        c.email IS NULL 
        OR c.email = '' 
        OR c.email NOT LIKE '%@%'
        OR c.email NOT LIKE '%.%'
        OR c.email LIKE '%@%@%'
    )
LIMIT 50;

-- Duplicate email addresses
SELECT 
    'Duplicate Customer Emails' AS exception_type,
    c.email,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(c.customer_id ORDER BY c.created_at SEPARATOR ', ') AS customer_ids,
    MIN(c.created_at) AS first_created,
    MAX(c.created_at) AS last_created,
    'Potential duplicate accounts' AS exception_reason,
    'MEDIUM' AS severity
FROM customers c
WHERE c.email IS NOT NULL 
    AND c.email != ''
    AND c.status = 'active'
GROUP BY c.email
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 50;

-- Incomplete customer addresses
SELECT 
    'Incomplete Customer Addresses' AS exception_type,
    c.customer_id,
    c.email,
    c.address_line1,
    c.city,
    c.state,
    c.zip_code,
    c.status,
    CONCAT_WS(', ',
        CASE WHEN c.address_line1 IS NULL OR c.address_line1 = '' THEN 'Missing address_line1' END,
        CASE WHEN c.city IS NULL OR c.city = '' THEN 'Missing city' END,
        CASE WHEN c.state IS NULL OR c.state = '' THEN 'Missing state' END,
        CASE WHEN c.zip_code IS NULL OR c.zip_code = '' THEN 'Missing zip_code' END
    ) AS missing_fields,
    'MEDIUM' AS severity
FROM customers c
WHERE c.status = 'active'
    AND (
        c.address_line1 IS NULL OR c.address_line1 = ''
        OR c.city IS NULL OR c.city = ''
        OR c.state IS NULL OR c.state = ''
        OR c.zip_code IS NULL OR c.zip_code = ''
    )
LIMIT 50;

-- ========================================
-- 6. PAYMENT & FINANCIAL EXCEPTIONS
-- ========================================
SELECT 
    '' AS blank_line,
    'PAYMENT & FINANCIAL EXCEPTIONS' AS report_section,
    '===============================' AS separator;

-- Paid orders with cancelled status
SELECT 
    'Status-Payment Mismatch' AS exception_type,
    o.order_id,
    o.customer_id,
    o.order_date,
    o.status AS order_status,
    o.payment_status,
    o.total_amount,
    'Paid order marked as cancelled' AS exception_reason,
    'CRITICAL' AS severity
FROM orders o
WHERE o.status = 'cancelled' 
    AND o.payment_status = 'paid'
    AND o.order_date >= DATE_SUB(NOW(), INTERVAL 180 DAY)
ORDER BY o.order_date DESC
LIMIT 50;

-- Orders without payment methods
SELECT 
    'Missing Payment Method' AS exception_type,
    o.order_id,
    o.customer_id,
    c.email,
    o.order_date,
    o.total_amount,
    o.payment_status,
    COUNT(pm.payment_method_id) AS payment_methods_on_file,
    'Customer has no payment method' AS exception_reason,
    CASE 
        WHEN o.payment_status = 'paid' THEN 'CRITICAL'
        ELSE 'HIGH'
    END AS severity
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN payment_methods pm ON c.customer_id = pm.customer_id
WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY o.order_id, o.customer_id, c.email, o.order_date, o.total_amount, o.payment_status
HAVING COUNT(pm.payment_method_id) = 0
ORDER BY o.order_date DESC
LIMIT 50;

-- Expired payment methods still marked as default
SELECT 
    'Expired Default Payment Methods' AS exception_type,
    pm.payment_method_id,
    pm.customer_id,
    c.email,
    pm.payment_type,
    pm.card_brand,
    pm.card_last_four,
    CONCAT(pm.expiry_month, '/', pm.expiry_year) AS expiry_date,
    'Expired card still set as default' AS exception_reason,
    'HIGH' AS severity
FROM payment_methods pm
JOIN customers c ON pm.customer_id = c.customer_id
WHERE pm.is_default = TRUE
    AND pm.payment_type IN ('credit_card', 'debit_card')
    AND (
        pm.expiry_year < YEAR(NOW())
        OR (pm.expiry_year = YEAR(NOW()) AND pm.expiry_month < MONTH(NOW()))
    )
LIMIT 50;

-- ========================================
-- 7. RETURN & REFUND EXCEPTIONS
-- ========================================
SELECT 
    '' AS blank_line,
    'RETURN & REFUND EXCEPTIONS' AS report_section,
    '==========================' AS separator;

-- Refunds exceeding original order amount
SELECT 
    'Excessive Refund Amounts' AS exception_type,
    r.return_id,
    r.order_id,
    o.total_amount AS original_order_amount,
    r.refund_amount,
    r.refund_amount - o.total_amount AS excess_amount,
    r.status AS return_status,
    r.reason,
    'Refund exceeds order total' AS exception_reason,
    'CRITICAL' AS severity
FROM returns r
JOIN orders o ON r.order_id = o.order_id
WHERE r.refund_amount > o.total_amount
ORDER BY excess_amount DESC
LIMIT 50;

-- Returns without refund amounts
SELECT 
    'Returns Missing Refund Amount' AS exception_type,
    r.return_id,
    r.order_id,
    o.total_amount AS order_amount,
    r.status,
    r.reason,
    DATEDIFF(NOW(), r.created_at) AS days_since_return,
    'Approved/Refunded return with no amount' AS exception_reason,
    CASE 
        WHEN r.status = 'refunded' THEN 'CRITICAL'
        WHEN r.status = 'approved' THEN 'HIGH'
        ELSE 'MEDIUM'
    END AS severity
FROM returns r
JOIN orders o ON r.order_id = o.order_id
WHERE r.status IN ('approved', 'refunded')
    AND (r.refund_amount IS NULL OR r.refund_amount = 0)
ORDER BY days_since_return DESC
LIMIT 50;

-- Returns for non-existent orders
SELECT 
    'Orphaned Returns' AS exception_type,
    r.return_id,
    r.order_id,
    r.status,
    r.refund_amount,
    r.created_at,
    'Return references non-existent order' AS exception_reason,
    'CRITICAL' AS severity
FROM returns r
LEFT JOIN orders o ON r.order_id = o.order_id
WHERE o.order_id IS NULL
LIMIT 50;

-- ========================================
-- 8. OUTLIER DETECTION - ORDER VALUES
-- ========================================
SELECT 
    '' AS blank_line,
    'STATISTICAL OUTLIERS - ORDER VALUES' AS report_section,
    '===================================' AS separator;

-- Orders with unusually high amounts (>3 standard deviations)
WITH order_stats AS (
    SELECT 
        AVG(total_amount) AS avg_amount,
        STDDEV(total_amount) AS stddev_amount
    FROM orders
    WHERE status != 'cancelled'
        AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
)
SELECT 
    'Unusually High Order Values' AS outlier_type,
    o.order_id,
    o.customer_id,
    c.email,
    o.order_date,
    o.total_amount,
    CONCAT('$', FORMAT(os.avg_amount, 2)) AS average_order_value,
    ROUND((o.total_amount - os.avg_amount) / os.stddev_amount, 2) AS std_deviations_from_mean,
    o.status,
    o.payment_status,
    CASE 
        WHEN (o.total_amount - os.avg_amount) / os.stddev_amount > 5 THEN 'EXTREME OUTLIER'
        WHEN (o.total_amount - os.avg_amount) / os.stddev_amount > 3 THEN 'SIGNIFICANT OUTLIER'
        ELSE 'MODERATE OUTLIER'
    END AS outlier_severity
FROM orders o
CROSS JOIN order_stats os
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status != 'cancelled'
    AND o.order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
    AND (o.total_amount - os.avg_amount) / os.stddev_amount > 3
ORDER BY std_deviations_from_mean DESC
LIMIT 50;

-- Orders with unusually low amounts
WITH order_stats AS (
    SELECT 
        AVG(total_amount) AS avg_amount,
        STDDEV(total_amount) AS stddev_amount
    FROM orders
    WHERE status != 'cancelled'
        AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
)
SELECT 
    'Unusually Low Order Values' AS outlier_type,
    o.order_id,
    o.customer_id,
    c.email,
    o.order_date,
    o.total_amount,
    CONCAT('$', FORMAT(os.avg_amount, 2)) AS average_order_value,
    ROUND((os.avg_amount - o.total_amount) / os.stddev_amount, 2) AS std_deviations_below_mean,
    o.status,
    o.payment_status,
    'Potential pricing error or test order' AS note
FROM orders o
CROSS JOIN order_stats os
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status != 'cancelled'
    AND o.order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
    AND (os.avg_amount - o.total_amount) / os.stddev_amount > 2
    AND o.total_amount > 0
ORDER BY std_deviations_below_mean DESC
LIMIT 50;

-- ========================================
-- 9. OUTLIER DETECTION - CUSTOMER BEHAVIOR
-- ========================================
SELECT 
    '' AS blank_line,
    'STATISTICAL OUTLIERS - CUSTOMER BEHAVIOR' AS report_section,
    '=========================================' AS separator;

-- Customers with abnormally high order frequency
WITH customer_order_stats AS (
    SELECT 
        customer_id,
        COUNT(*) AS order_count,
        DATEDIFF(NOW(), MIN(order_date)) AS customer_age_days,
        COUNT(*) / NULLIF(DATEDIFF(NOW(), MIN(order_date)), 0) AS orders_per_day
    FROM orders
    WHERE order_date >= DATE_SUB(NOW(), INTERVAL 180 DAY)
    GROUP BY customer_id
),
avg_stats AS (
    SELECT 
        AVG(orders_per_day) AS avg_orders_per_day,
        STDDEV(orders_per_day) AS stddev_orders_per_day
    FROM customer_order_stats
)
SELECT 
    'Abnormally High Order Frequency' AS outlier_type,
    cos.customer_id,
    c.email,
    c.status,
    cos.order_count,
    cos.customer_age_days,
    ROUND(cos.orders_per_day, 3) AS orders_per_day,
    ROUND(avgs.avg_orders_per_day, 3) AS avg_orders_per_day,
    ROUND((cos.orders_per_day - avgs.avg_orders_per_day) / NULLIF(avgs.stddev_orders_per_day, 0), 2) AS std_deviations,
    'Potential bot or fraud activity' AS warning
FROM customer_order_stats cos
CROSS JOIN avg_stats avgs
JOIN customers c ON cos.customer_id = c.customer_id
WHERE (cos.orders_per_day - avgs.avg_orders_per_day) / NULLIF(avgs.stddev_orders_per_day, 0) > 3
ORDER BY std_deviations DESC
LIMIT 50;

-- ========================================
-- 10. ANOMALY DETECTION - TIME-BASED PATTERNS
-- ========================================
SELECT 
    '' AS blank_line,
    'TIME-BASED ANOMALIES' AS report_section,
    '====================' AS separator;

-- Orders created outside business hours (potential fraud)
SELECT 
    'Off-Hours Order Activity' AS anomaly_type,
    o.order_id,
    o.customer_id,
    c.email,
    o.order_date,
    HOUR(o.order_date) AS order_hour,
    o.total_amount,
    o.status,
    CASE 
        WHEN HOUR(o.order_date) BETWEEN 2 AND 5 THEN 'Late night (2-5 AM)'
        ELSE 'Early morning (0-2 AM)'
    END AS time_window,
    'Unusual transaction time' AS note
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE HOUR(o.order_date) BETWEEN 0 AND 5
    AND o.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    AND o.total_amount > 500
ORDER BY o.order_date DESC
LIMIT 50;

-- Rapid successive orders from same customer
SELECT 
    'Rapid Successive Orders' AS anomaly_type,
    o1.customer_id,
    c.email,
    o1.order_id AS order_1,
    o1.order_date AS order_1_date,
    o2.order_id AS order_2,
    o2.order_date AS order_2_date,
    TIMESTAMPDIFF(MINUTE, o1.order_date, o2.order_date) AS minutes_between,
    o1.total_amount + o2.total_amount AS combined_amount,
    'Potential fraud or duplicate order' AS warning
FROM orders o1
JOIN orders o2 ON o1.customer_id = o2.customer_id 
    AND o2.order_date > o1.order_date
    AND TIMESTAMPDIFF(MINUTE, o1.order_date, o2.order_date) <= 15
JOIN customers c ON o1.customer_id = c.customer_id
WHERE o1.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
ORDER BY minutes_between ASC
LIMIT 50;

-- Display completion message
SELECT 
    '' AS blank_line,
    'EXCEPTION REPORT COMPLETE' AS status,
    'Review all critical exceptions immediately' AS recommendation,
    NOW() AS generated_at;