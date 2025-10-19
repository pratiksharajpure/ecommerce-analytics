-- ========================================
-- INVALID_DATES.SQL
-- Data Quality Check: Invalid Date Values
-- Path: sql/core_analysis/invalid_dates.sql
-- Identifies future dates, invalid date ranges, and temporal anomalies
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Orders with Future Dates
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_date,
    DATEDIFF(o.order_date, CURRENT_TIMESTAMP) AS days_in_future,
    o.total_amount,
    o.status,
    o.payment_status,
    'Order date in future' AS issue_type
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date > CURRENT_TIMESTAMP
ORDER BY o.order_date DESC;

-- ========================================
-- 2. Orders Older Than Customer Creation
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.created_at AS customer_created,
    o.order_date,
    DATEDIFF(c.created_at, o.order_date) AS days_before_customer,
    o.total_amount,
    o.status,
    'Order predates customer account' AS issue_type
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date < c.created_at
ORDER BY days_before_customer DESC;

-- ========================================
-- 3. Order Items Created Before Order
-- ========================================
SELECT 
    oi.order_item_id,
    oi.order_id,
    o.order_date,
    oi.created_at AS item_created,
    TIMESTAMPDIFF(MINUTE, o.order_date, oi.created_at) AS minutes_before_order,
    p.product_name,
    oi.quantity,
    oi.subtotal,
    'Order item predates order' AS issue_type
FROM order_items oi
INNER JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE oi.created_at < o.order_date
ORDER BY minutes_before_order DESC;

-- ========================================
-- 4. Updated_at Before Created_at
-- ========================================
-- Products
SELECT 
    'Product' AS table_name,
    product_id AS record_id,
    sku,
    product_name AS record_name,
    created_at,
    updated_at,
    TIMESTAMPDIFF(MINUTE, updated_at, created_at) AS minutes_difference,
    'Updated before created' AS issue_type
FROM products
WHERE updated_at < created_at

UNION ALL

-- Customers
SELECT 
    'Customer' AS table_name,
    customer_id AS record_id,
    email AS sku,
    CONCAT(first_name, ' ', last_name) AS record_name,
    created_at,
    updated_at,
    TIMESTAMPDIFF(MINUTE, updated_at, created_at) AS minutes_difference,
    'Updated before created' AS issue_type
FROM customers
WHERE updated_at < created_at

UNION ALL

-- Orders
SELECT 
    'Order' AS table_name,
    order_id AS record_id,
    CAST(order_id AS CHAR) AS sku,
    CAST(total_amount AS CHAR) AS record_name,
    created_at,
    updated_at,
    TIMESTAMPDIFF(MINUTE, updated_at, created_at) AS minutes_difference,
    'Updated before created' AS issue_type
FROM orders
WHERE updated_at < created_at

ORDER BY minutes_difference DESC;

-- ========================================
-- 5. Vendor Contracts with Invalid Date Ranges
-- ========================================
SELECT 
    vc.contract_id,
    vc.vendor_id,
    v.vendor_name,
    vc.product_id,
    p.product_name,
    vc.start_date,
    vc.end_date,
    DATEDIFF(vc.end_date, vc.start_date) AS contract_duration_days,
    vc.status,
    CASE 
        WHEN vc.end_date < vc.start_date THEN 'End date before start date'
        WHEN vc.start_date > CURRENT_DATE AND vc.status = 'active' THEN 'Active contract not yet started'
        WHEN vc.end_date < CURRENT_DATE AND vc.status = 'active' THEN 'Contract expired but still active'
        WHEN vc.start_date > CURRENT_DATE + INTERVAL 1 YEAR THEN 'Start date too far in future'
    END AS issue_type
FROM vendor_contracts vc
INNER JOIN vendors v ON vc.vendor_id = v.vendor_id
LEFT JOIN products p ON vc.product_id = p.product_id
WHERE vc.end_date < vc.start_date
    OR (vc.start_date > CURRENT_DATE AND vc.status = 'active')
    OR (vc.end_date < CURRENT_DATE AND vc.status = 'active')
    OR vc.start_date > CURRENT_DATE + INTERVAL 1 YEAR
ORDER BY vc.start_date DESC;

-- ========================================
-- 6. Campaigns with Invalid Date Ranges
-- ========================================
SELECT 
    campaign_id,
    campaign_name,
    campaign_type,
    start_date,
    end_date,
    DATEDIFF(end_date, start_date) AS campaign_duration_days,
    budget,
    status,
    CASE 
        WHEN end_date < start_date THEN 'End date before start date'
        WHEN start_date > CURRENT_DATE AND status = 'active' THEN 'Active campaign not yet started'
        WHEN end_date < CURRENT_DATE AND status = 'active' THEN 'Campaign ended but still active'
        WHEN DATEDIFF(end_date, start_date) < 1 THEN 'Campaign duration less than 1 day'
        WHEN DATEDIFF(end_date, start_date) > 365 THEN 'Campaign duration over 1 year'
    END AS issue_type
FROM campaigns
WHERE end_date < start_date
    OR (start_date > CURRENT_DATE AND status = 'active')
    OR (end_date < CURRENT_DATE AND status = 'active')
    OR DATEDIFF(end_date, start_date) < 1
    OR DATEDIFF(end_date, start_date) > 365
ORDER BY start_date DESC;

-- ========================================
-- 7. Orders with Impossible Timestamps
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    o.order_date,
    YEAR(o.order_date) AS order_year,
    o.created_at,
    o.updated_at,
    o.total_amount,
    o.status,
    CASE 
        WHEN YEAR(o.order_date) < 2000 THEN 'Order date before year 2000'
        WHEN YEAR(o.order_date) > YEAR(CURRENT_DATE) + 1 THEN 'Order date more than 1 year in future'
        WHEN o.order_date < '2010-01-01' THEN 'Order date unrealistically old'
    END AS issue_type
FROM orders o
WHERE YEAR(o.order_date) < 2000
    OR YEAR(o.order_date) > YEAR(CURRENT_DATE) + 1
    OR o.order_date < '2010-01-01'
ORDER BY o.order_date ASC;

-- ========================================
-- 8. Returns Requested Before Order Date
-- ========================================
SELECT 
    r.return_id,
    r.order_id,
    o.order_date,
    r.created_at AS return_requested_date,
    DATEDIFF(o.order_date, r.created_at) AS days_before_order,
    r.reason,
    r.status AS return_status,
    o.status AS order_status,
    r.refund_amount,
    'Return requested before order placed' AS issue_type
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
WHERE r.created_at < o.order_date
ORDER BY days_before_order DESC;

-- ========================================
-- 9. Reviews Created Before Product Purchase
-- ========================================
SELECT 
    r.review_id,
    r.product_id,
    p.product_name,
    r.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    r.created_at AS review_date,
    MIN(o.order_date) AS first_purchase_date,
    DATEDIFF(MIN(o.order_date), r.created_at) AS days_before_purchase,
    r.rating,
    r.is_verified_purchase,
    'Review before purchase' AS issue_type
FROM reviews r
LEFT JOIN products p ON r.product_id = p.product_id
LEFT JOIN customers c ON r.customer_id = c.customer_id
LEFT JOIN order_items oi ON r.product_id = oi.product_id AND r.customer_id = oi.order_id
LEFT JOIN orders o ON oi.order_id = o.order_id AND o.customer_id = r.customer_id
GROUP BY r.review_id, r.product_id, p.product_name, r.customer_id, 
         c.first_name, c.last_name, r.created_at, r.rating, r.is_verified_purchase
HAVING MIN(o.order_date) IS NULL OR r.created_at < MIN(o.order_date)
ORDER BY days_before_purchase DESC;

-- ========================================
-- 10. Loyalty Program Anomalies
-- ========================================
SELECT 
    lp.loyalty_id,
    lp.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    lp.joined_date,
    c.created_at AS customer_created,
    lp.tier_start_date,
    DATEDIFF(c.created_at, lp.joined_date) AS days_before_customer,
    lp.tier,
    lp.points_balance,
    CASE 
        WHEN lp.joined_date < c.created_at THEN 'Joined loyalty before customer created'
        WHEN lp.tier_start_date < lp.joined_date THEN 'Tier started before joining'
        WHEN lp.joined_date > CURRENT_DATE THEN 'Joined date in future'
    END AS issue_type
FROM loyalty_program lp
INNER JOIN customers c ON lp.customer_id = c.customer_id
WHERE lp.joined_date < c.created_at
    OR lp.tier_start_date < lp.joined_date
    OR lp.joined_date > CURRENT_DATE
ORDER BY days_before_customer DESC;

-- ========================================
-- 11. Date Validation Summary Report
-- ========================================
SELECT 
    'Orders with Future Dates' AS metric,
    COUNT(*) AS count
FROM orders
WHERE order_date > CURRENT_TIMESTAMP

UNION ALL

SELECT 
    'Orders Before Customer Creation' AS metric,
    COUNT(*) AS count
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date < c.created_at

UNION ALL

SELECT 
    'Updated Before Created Records' AS metric,
    COUNT(*) AS count
FROM (
    SELECT product_id FROM products WHERE updated_at < created_at
    UNION ALL
    SELECT customer_id FROM customers WHERE updated_at < created_at
    UNION ALL
    SELECT order_id FROM orders WHERE updated_at < created_at
) AS all_records

UNION ALL

SELECT 
    'Invalid Contract Date Ranges' AS metric,
    COUNT(*) AS count
FROM vendor_contracts
WHERE end_date < start_date
    OR (end_date < CURRENT_DATE AND status = 'active')

UNION ALL

SELECT 
    'Invalid Campaign Date Ranges' AS metric,
    COUNT(*) AS count
FROM campaigns
WHERE end_date < start_date
    OR (end_date < CURRENT_DATE AND status = 'active')

UNION ALL

SELECT 
    'Returns Before Order Date' AS metric,
    COUNT(*) AS count
FROM returns r
INNER JOIN orders o ON r.order_id = o.order_id
WHERE r.created_at < o.order_date

UNION ALL

SELECT 
    'Reviews Before Purchase' AS metric,
    COUNT(DISTINCT r.review_id) AS count
FROM reviews r
LEFT JOIN order_items oi ON r.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id AND o.customer_id = r.customer_id
GROUP BY r.review_id, r.created_at
HAVING MIN(o.order_date) IS NULL OR r.created_at < MIN(o.order_date);

-- ========================================
-- 12. Temporal Sequence Violations
-- ========================================
SELECT 
    o.order_id,
    o.customer_id,
    o.order_date,
    o.status,
    o.created_at AS order_created,
    MIN(oi.created_at) AS first_item_added,
    MAX(oi.created_at) AS last_item_added,
    o.updated_at AS order_updated,
    TIMESTAMPDIFF(MINUTE, o.order_date, MIN(oi.created_at)) AS minutes_before_order,
    'Items added before order created' AS issue_type
FROM orders o
INNER JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.customer_id, o.order_date, o.status, o.created_at, o.updated_at
HAVING MIN(oi.created_at) < o.order_date
ORDER BY minutes_before_order DESC;