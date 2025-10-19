-- ========================================
-- MISSING_ADDRESSES.SQL
-- Data Quality Check: Incomplete Address Records
-- Identifies customers with missing or incomplete address information
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Customers with Completely Missing Addresses
-- ========================================
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    phone,
    status,
    created_at,
    'No address information' AS issue_type
FROM customers
WHERE (address_line1 IS NULL OR address_line1 = '')
    AND (city IS NULL OR city = '')
    AND (state IS NULL OR state = '')
    AND (zip_code IS NULL OR zip_code = '')
ORDER BY created_at DESC;

-- ========================================
-- 2. Customers with Partial Address Information
-- ========================================
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    address_line1,
    city,
    state,
    zip_code,
    country,
    CASE 
        WHEN (address_line1 IS NULL OR address_line1 = '') THEN 'Missing street address'
        WHEN (city IS NULL OR city = '') THEN 'Missing city'
        WHEN (state IS NULL OR state = '') THEN 'Missing state'
        WHEN (zip_code IS NULL OR zip_code = '') THEN 'Missing zip code'
    END AS missing_field,
    status,
    created_at
FROM customers
WHERE status = 'active'
    AND (
        (address_line1 IS NULL OR address_line1 = '') OR
        (city IS NULL OR city = '') OR
        (state IS NULL OR state = '') OR
        (zip_code IS NULL OR zip_code = '')
    )
ORDER BY created_at DESC;

-- ========================================
-- 3. Invalid Zip Code Patterns
-- ========================================
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    address_line1,
    city,
    state,
    zip_code,
    'Invalid zip code format' AS issue_type,
    created_at
FROM customers
WHERE zip_code IS NOT NULL
    AND zip_code != ''
    AND (
        LENGTH(zip_code) NOT IN (5, 10) OR  -- US zip codes are 5 or 10 chars (with hyphen)
        zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$'  -- Must match XXXXX or XXXXX-XXXX
    )
ORDER BY created_at DESC;

-- ========================================
-- 4. Missing Contact Information
-- ========================================
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    phone,
    CASE 
        WHEN (email IS NULL OR email = '') AND (phone IS NULL OR phone = '') THEN 'No contact info'
        WHEN (email IS NULL OR email = '') THEN 'Missing email'
        WHEN (phone IS NULL OR phone = '') THEN 'Missing phone'
    END AS issue_type,
    status,
    created_at
FROM customers
WHERE (email IS NULL OR email = '' OR phone IS NULL OR phone = '')
    AND status = 'active'
ORDER BY created_at DESC;

-- ========================================
-- 5. Customers with Orders but No Shipping Address
-- ========================================
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_revenue,
    MAX(o.order_date) AS last_order_date,
    'Has orders but no shipping address' AS issue_type
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
WHERE sa.address_id IS NULL
    AND o.status IN ('pending', 'processing', 'shipped', 'delivered')
GROUP BY c.customer_id, c.first_name, c.last_name, c.email
ORDER BY total_revenue DESC;

-- ========================================
-- 6. Summary Statistics - Address Data Quality
-- ========================================
SELECT 
    'Total Customers' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2), '%') AS percentage
FROM customers

UNION ALL

SELECT 
    'Missing Address Line 1' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2), '%') AS percentage
FROM customers
WHERE address_line1 IS NULL OR address_line1 = ''

UNION ALL

SELECT 
    'Missing City' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2), '%') AS percentage
FROM customers
WHERE city IS NULL OR city = ''

UNION ALL

SELECT 
    'Missing State' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2), '%') AS percentage
FROM customers
WHERE state IS NULL OR state = ''

UNION ALL

SELECT 
    'Missing Zip Code' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2), '%') AS percentage
FROM customers
WHERE zip_code IS NULL OR zip_code = ''

UNION ALL

SELECT 
    'Invalid Zip Code Format' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2), '%') AS percentage
FROM customers
WHERE zip_code IS NOT NULL 
    AND zip_code != ''
    AND zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$'

UNION ALL

SELECT 
    'Complete Address Information' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2), '%') AS percentage
FROM customers
WHERE address_line1 IS NOT NULL AND address_line1 != ''
    AND city IS NOT NULL AND city != ''
    AND state IS NOT NULL AND state != ''
    AND zip_code IS NOT NULL AND zip_code != '';

-- ========================================
-- 7. Active Customers Priority Fix List
-- ========================================
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.phone,
    c.address_line1,
    c.city,
    c.state,
    c.zip_code,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.total_amount), 0) AS lifetime_value,
    MAX(o.order_date) AS last_order_date,
    DATEDIFF(CURRENT_DATE, MAX(o.order_date)) AS days_since_last_order,
    'High priority - Active customer with incomplete address' AS priority
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.status = 'active'
    AND (
        c.address_line1 IS NULL OR c.address_line1 = '' OR
        c.city IS NULL OR c.city = '' OR
        c.state IS NULL OR c.state = '' OR
        c.zip_code IS NULL OR c.zip_code = ''
    )
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.phone, 
         c.address_line1, c.city, c.state, c.zip_code
HAVING order_count > 0
ORDER BY lifetime_value DESC, days_since_last_order ASC
LIMIT 100;