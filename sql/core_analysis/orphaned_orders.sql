-- ========================================
-- ORPHANED_ORDERS.SQL
-- Data Quality Check: Orphaned Orders
-- Path: sql/core_analysis/orphaned_orders.sql
-- Identifies orders without valid customer references
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Orders with NULL Customer ID
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    COUNT(oi.order_item_id) AS item_count,
    SUM(oi.subtotal) AS items_total,
    'NULL customer_id' AS issue_type,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_old
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.customer_id IS NULL
GROUP BY o.order_id, o.customer_id, o.order_date, o.total_amount, o.status, o.payment_status
ORDER BY o.total_amount DESC;

-- ========================================
-- 2. Orders with Deleted/Non-Existent Customers
-- ========================================
SELECT 
    o.order_id,
    o.customer_id AS invalid_customer_id,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    o.shipping_cost,
    o.tax_amount,
    COUNT(oi.order_item_id) AS item_count,
    'Customer record does not exist' AS issue_type,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_old
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.customer_id IS NOT NULL
    AND c.customer_id IS NULL
GROUP BY o.order_id, o.customer_id, o.order_date, o.total_amount, 
         o.status, o.payment_status, o.shipping_cost, o.tax_amount
ORDER BY o.total_amount DESC;

-- ========================================
-- 3. Orders from Suspended/Inactive Customers
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.status AS customer_status,
    o.order_date,
    o.total_amount,
    o.status AS order_status,
    o.payment_status,
    COUNT(oi.order_item_id) AS item_count,
    CASE 
        WHEN c.status = 'suspended' THEN 'Order from suspended customer'
        WHEN c.status = 'inactive' THEN 'Order from inactive customer'
    END AS issue_type
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE c.status IN ('suspended', 'inactive')
    AND o.status NOT IN ('cancelled')
    AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 180 DAY)
GROUP BY o.order_id, o.customer_id, c.first_name, c.last_name, c.email,
         c.status, o.order_date, o.total_amount, o.status, o.payment_status
ORDER BY o.order_date DESC;

-- ========================================
-- 4. Orders Without Shipping Address
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.address_line1 AS customer_default_address,
    o.order_date,
    o.total_amount,
    o.status,
    o.shipping_cost,
    'No shipping address available' AS issue_type
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
WHERE sa.address_id IS NULL
    AND (c.address_line1 IS NULL OR c.address_line1 = '')
    AND o.status IN ('pending', 'processing', 'shipped')
ORDER BY o.order_date DESC;

-- ========================================
-- 5. Orders Without Payment Method
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
    'No payment method on file' AS issue_type,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_since_order
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN payment_methods pm ON c.customer_id = pm.customer_id
WHERE pm.payment_method_id IS NULL
    AND o.payment_status IN ('pending', 'failed')
    AND o.status != 'cancelled'
ORDER BY o.total_amount DESC;

-- ========================================
-- 6. Orphaned Order Items (No Parent Order)
-- ========================================
SELECT 
    oi.order_item_id,
    oi.order_id AS invalid_order_id,
    oi.product_id,
    p.product_name,
    oi.quantity,
    oi.unit_price,
    oi.subtotal,
    'Order item without valid order' AS issue_type,
    oi.created_at
FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE o.order_id IS NULL
ORDER BY oi.created_at DESC;

-- ========================================
-- 7. Orders with Missing Customer Contact Info
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.phone,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    CASE 
        WHEN (c.email IS NULL OR c.email = '') AND (c.phone IS NULL OR c.phone = '') THEN 'No contact information'
        WHEN (c.email IS NULL OR c.email = '') THEN 'No email address'
        WHEN (c.phone IS NULL OR c.phone = '') THEN 'No phone number'
    END AS issue_type,
    'Cannot contact customer about order' AS impact
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE (
    (c.email IS NULL OR c.email = '') OR
    (c.phone IS NULL OR c.phone = '')
)
    AND o.status IN ('pending', 'processing', 'shipped')
ORDER BY o.order_date DESC;

-- ========================================
-- 8. Revenue Impact of Orphaned Orders
-- ========================================
SELECT 
    issue_category,
    COUNT(DISTINCT order_id) AS affected_orders,
    SUM(total_amount) AS total_revenue_at_risk,
    AVG(total_amount) AS avg_order_value,
    MIN(order_date) AS oldest_order,
    MAX(order_date) AS newest_order
FROM (
    -- Orders with NULL customer
    SELECT 
        'NULL Customer ID' AS issue_category,
        o.order_id,
        o.total_amount,
        o.order_date
    FROM orders o
    WHERE o.customer_id IS NULL
    
    UNION ALL
    
    -- Orders with deleted customers
    SELECT 
        'Deleted Customer' AS issue_category,
        o.order_id,
        o.total_amount,
        o.order_date
    FROM orders o
    LEFT JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.customer_id IS NOT NULL AND c.customer_id IS NULL
    
    UNION ALL
    
    -- Orders from suspended customers
    SELECT 
        'Suspended Customer' AS issue_category,
        o.order_id,
        o.total_amount,
        o.order_date
    FROM orders o
    INNER JOIN customers c ON o.customer_id = c.customer_id
    WHERE c.status = 'suspended' AND o.status NOT IN ('cancelled')
    
    UNION ALL
    
    -- Orders without shipping address
    SELECT 
        'No Shipping Address' AS issue_category,
        o.order_id,
        o.total_amount,
        o.order_date
    FROM orders o
    INNER JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
    WHERE sa.address_id IS NULL
        AND (c.address_line1 IS NULL OR c.address_line1 = '')
        AND o.status IN ('pending', 'processing', 'shipped')
) AS orphaned_orders
GROUP BY issue_category
ORDER BY total_revenue_at_risk DESC;

-- ========================================
-- 9. Orphaned Records Summary
-- ========================================
SELECT 
    'Orders with NULL Customer' AS metric,
    COUNT(*) AS count
FROM orders
WHERE customer_id IS NULL

UNION ALL

SELECT 
    'Orders with Deleted Customer' AS metric,
    COUNT(*) AS count
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.customer_id IS NOT NULL AND c.customer_id IS NULL

UNION ALL

SELECT 
    'Orders from Suspended Customers' AS metric,
    COUNT(*) AS count
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE c.status = 'suspended'

UNION ALL

SELECT 
    'Orders Without Shipping Address' AS metric,
    COUNT(*) AS count
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
WHERE sa.address_id IS NULL
    AND (c.address_line1 IS NULL OR c.address_line1 = '')
    AND o.status IN ('pending', 'processing', 'shipped')

UNION ALL

SELECT 
    'Order Items Without Orders' AS metric,
    COUNT(*) AS count
FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL

UNION ALL

SELECT 
    'Orders Without Contact Info' AS metric,
    COUNT(*) AS count
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE ((c.email IS NULL OR c.email = '') OR (c.phone IS NULL OR c.phone = ''))
    AND o.status IN ('pending', 'processing', 'shipped');

-- ========================================
-- 10. Critical Orphaned Orders (Requires Immediate Action)
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    COALESCE(CONCAT(c.first_name, ' ', c.last_name), 'UNKNOWN') AS customer_name,
    COALESCE(c.email, 'NO EMAIL') AS email,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    DATEDIFF(CURRENT_DATE, DATE(o.order_date)) AS days_old,
    CASE 
        WHEN o.customer_id IS NULL THEN 'NULL customer_id'
        WHEN c.customer_id IS NULL THEN 'Customer deleted'
        WHEN c.status = 'suspended' THEN 'Customer suspended'
        WHEN sa.address_id IS NULL AND (c.address_line1 IS NULL OR c.address_line1 = '') THEN 'No shipping address'
    END AS critical_issue,
    'URGENT - Cannot fulfill order' AS priority
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
WHERE o.status IN ('pending', 'processing', 'shipped')
    AND o.total_amount >= 100
    AND (
        o.customer_id IS NULL OR
        c.customer_id IS NULL OR
        c.status = 'suspended' OR
        (sa.address_id IS NULL AND (c.address_line1 IS NULL OR c.address_line1 = ''))
    )
ORDER BY o.total_amount DESC, o.order_date ASC
LIMIT 50;