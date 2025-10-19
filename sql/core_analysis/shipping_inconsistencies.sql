-- ========================================
-- SHIPPING_INCONSISTENCIES.SQL
-- Data Quality Check: Shipping Address Issues
-- Path: sql/core_analysis/shipping_inconsistencies.sql
-- Identifies shipping address problems and inconsistencies
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Customers with Multiple Default Shipping Addresses
-- ========================================
SELECT 
    sa.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    COUNT(*) AS default_address_count,
    GROUP_CONCAT(sa.address_id ORDER BY sa.created_at) AS address_ids,
    'Multiple default shipping addresses' AS issue_type
FROM shipping_addresses sa
INNER JOIN customers c ON sa.customer_id = c.customer_id
WHERE sa.is_default = TRUE
GROUP BY sa.customer_id, c.first_name, c.last_name, c.email
HAVING COUNT(*) > 1
ORDER BY default_address_count DESC;

-- ========================================
-- 2. Customers Without Any Shipping Address
-- ========================================
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.address_line1 AS profile_address,
    c.city,
    c.state,
    c.status,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_revenue,
    MAX(o.order_date) AS last_order_date,
    CASE 
        WHEN c.address_line1 IS NULL OR c.address_line1 = '' THEN 'No shipping address at all'
        ELSE 'No shipping_addresses record'
    END AS issue_type
FROM customers c
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE sa.address_id IS NULL
    AND c.status = 'active'
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, 
         c.address_line1, c.city, c.state, c.status
ORDER BY total_orders DESC;

-- ========================================
-- 3. Incomplete Shipping Addresses
-- ========================================
SELECT 
    sa.address_id,
    sa.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    sa.address_label,
    sa.address_line1,
    sa.city,
    sa.state,
    sa.zip_code,
    sa.country,
    sa.is_default,
    CASE 
        WHEN (sa.address_line1 IS NULL OR sa.address_line1 = '') THEN 'Missing street address'
        WHEN (sa.city IS NULL OR sa.city = '') THEN 'Missing city'
        WHEN (sa.state IS NULL OR sa.state = '') THEN 'Missing state'
        WHEN (sa.zip_code IS NULL OR sa.zip_code = '') THEN 'Missing zip code'
    END AS issue_type
FROM shipping_addresses sa
INNER JOIN customers c ON sa.customer_id = c.customer_id
WHERE (sa.address_line1 IS NULL OR sa.address_line1 = '')
    OR (sa.city IS NULL OR sa.city = '')
    OR (sa.state IS NULL OR sa.state = '')
    OR (sa.zip_code IS NULL OR sa.zip_code = '')
ORDER BY sa.is_default DESC, sa.created_at DESC;

-- ========================================
-- 4. Invalid Zip Code Formats
-- ========================================
SELECT 
    sa.address_id,
    sa.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    sa.address_label,
    sa.zip_code,
    sa.city,
    sa.state,
    sa.is_default,
    'Invalid zip code format' AS issue_type
FROM shipping_addresses sa
INNER JOIN customers c ON sa.customer_id = c.customer_id
WHERE sa.zip_code IS NOT NULL
    AND sa.zip_code != ''
    AND sa.zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$'
ORDER BY sa.is_default DESC;

-- ========================================
-- 5. Mismatched Address Between Tables
-- ========================================
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.address_line1 AS customer_address,
    c.city AS customer_city,
    c.state AS customer_state,
    c.zip_code AS customer_zip,
    sa.address_line1 AS shipping_address,
    sa.city AS shipping_city,
    sa.state AS shipping_state,
    sa.zip_code AS shipping_zip,
    sa.is_default,
    'Address mismatch between tables' AS issue_type
FROM customers c
INNER JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
WHERE sa.is_default = TRUE
    AND (
        c.address_line1 != sa.address_line1 OR
        c.city != sa.city OR
        c.state != sa.state OR
        c.zip_code != sa.zip_code
    )
    AND c.address_line1 IS NOT NULL
    AND c.address_line1 != ''
ORDER BY c.customer_id;

-- ========================================
-- 6. Shipping Addresses with No Associated Orders
-- ========================================
SELECT 
    sa.address_id,
    sa.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    sa.address_label,
    sa.address_line1,
    sa.city,
    sa.state,
    sa.zip_code,
    sa.is_default,
    sa.created_at,
    DATEDIFF(CURRENT_DATE, sa.created_at) AS days_since_created,
    'Shipping address never used' AS issue_type
FROM shipping_addresses sa
INNER JOIN customers c ON sa.customer_id = c.customer_id
LEFT JOIN orders o ON sa.customer_id = o.customer_id
WHERE o.order_id IS NULL
    AND DATEDIFF(CURRENT_DATE, sa.created_at) > 90
ORDER BY days_since_created DESC;

-- ========================================
-- 7. Duplicate Shipping Addresses
-- ========================================
SELECT 
    sa1.address_id AS address_id_1,
    sa2.address_id AS address_id_2,
    sa1.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    sa1.address_line1,
    sa1.city,
    sa1.state,
    sa1.zip_code,
    sa1.is_default AS is_default_1,
    sa2.is_default AS is_default_2,
    'Duplicate shipping addresses' AS issue_type
FROM shipping_addresses sa1
INNER JOIN shipping_addresses sa2 ON sa1.customer_id = sa2.customer_id
    AND sa1.address_id < sa2.address_id
    AND sa1.address_line1 = sa2.address_line1
    AND sa1.city = sa2.city
    AND sa1.state = sa2.state
    AND sa1.zip_code = sa2.zip_code
INNER JOIN customers c ON sa1.customer_id = c.customer_id
ORDER BY sa1.customer_id;

-- ========================================
-- 8. Shipping Cost Anomalies
-- ========================================
WITH shipping_stats AS (
    SELECT 
        AVG(shipping_cost) AS avg_shipping,
        STDDEV(shipping_cost) AS stddev_shipping,
        MIN(shipping_cost) AS min_shipping,
        MAX(shipping_cost) AS max_shipping
    FROM orders
    WHERE shipping_cost > 0
        AND status NOT IN ('cancelled')
)
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.city,
    c.state,
    o.order_date,
    o.total_amount,
    o.shipping_cost,
    ROUND(ss.avg_shipping, 2) AS avg_shipping_cost,
    ROUND((o.shipping_cost - ss.avg_shipping) / ss.stddev_shipping, 2) AS std_deviations,
    o.status,
    CASE 
        WHEN o.shipping_cost = 0 AND o.total_amount < 100 THEN 'Free shipping on small order'
        WHEN o.shipping_cost > (ss.avg_shipping + 3 * ss.stddev_shipping) THEN 'Unusually high shipping'
        WHEN o.shipping_cost > o.total_amount THEN 'Shipping exceeds order value'
    END AS issue_type
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
CROSS JOIN shipping_stats ss
WHERE o.status NOT IN ('cancelled')
    AND (
        (o.shipping_cost = 0 AND o.total_amount < 100) OR
        o.shipping_cost > (ss.avg_shipping + 3 * ss.stddev_shipping) OR
        o.shipping_cost > o.total_amount
    )
ORDER BY std_deviations DESC;

-- ========================================
-- 9. International Shipping Issues
-- ========================================
SELECT 
    sa.address_id,
    sa.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    sa.address_line1,
    sa.city,
    sa.state,
    sa.country,
    COUNT(DISTINCT o.order_id) AS total_orders,
    CASE 
        WHEN sa.country IS NULL OR sa.country = '' THEN 'Missing country'
        WHEN sa.country != 'USA' AND (sa.state IS NOT NULL AND sa.state != '') THEN 'International address with US state'
        WHEN sa.country != 'USA' AND sa.zip_code REGEXP '^[0-9]{5}' THEN 'International address with US zip format'
    END AS issue_type
FROM shipping_addresses sa
INNER JOIN customers c ON sa.customer_id = c.customer_id
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE (sa.country IS NULL OR sa.country = '')
    OR (sa.country != 'USA' AND (sa.state IS NOT NULL AND sa.state != ''))
    OR (sa.country != 'USA' AND sa.zip_code REGEXP '^[0-9]{5}')
GROUP BY sa.address_id, sa.customer_id, c.first_name, c.last_name, c.email,
         sa.address_line1, sa.city, sa.state, sa.country, sa.zip_code
ORDER BY total_orders DESC;

-- ========================================
-- 10. Shipping Inconsistencies Summary
-- ========================================
SELECT 
    'Total Shipping Addresses' AS metric,
    COUNT(*) AS count
FROM shipping_addresses

UNION ALL

SELECT 
    'Incomplete Addresses' AS metric,
    COUNT(*) AS count
FROM shipping_addresses
WHERE (address_line1 IS NULL OR address_line1 = '')
    OR (city IS NULL OR city = '')
    OR (state IS NULL OR state = '')
    OR (zip_code IS NULL OR zip_code = '')

UNION ALL

SELECT 
    'Invalid Zip Codes' AS metric,
    COUNT(*) AS count
FROM shipping_addresses
WHERE zip_code IS NOT NULL
    AND zip_code != ''
    AND zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$'

UNION ALL

SELECT 
    'Customers with Multiple Defaults' AS metric,
    COUNT(DISTINCT customer_id) AS count
FROM (
    SELECT customer_id
    FROM shipping_addresses
    WHERE is_default = TRUE
    GROUP BY customer_id
    HAVING COUNT(*) > 1
) AS multi_defaults

UNION ALL

SELECT 
    'Active Customers Without Address' AS metric,
    COUNT(*) AS count
FROM customers c
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
WHERE sa.address_id IS NULL
    AND c.status = 'active'

UNION ALL

SELECT 
    'Duplicate Addresses' AS metric,
    COUNT(*) AS count
FROM (
    SELECT sa1.address_id
    FROM shipping_addresses sa1
    INNER JOIN shipping_addresses sa2 ON sa1.customer_id = sa2.customer_id
        AND sa1.address_id < sa2.address_id
        AND sa1.address_line1 = sa2.address_line1
        AND sa1.city = sa2.city
        AND sa1.zip_code = sa2.zip_code
) AS duplicates;