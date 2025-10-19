-- ========================================
-- GEOGRAPHIC ANOMALIES ANALYSIS
-- Day 6-7: Order & Transaction Queries
-- Invalid Locations, ZIP Codes & Address Validation
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. INVALID ZIP CODE FORMATS
-- ========================================

-- Check for invalid US ZIP code formats (should be 5 or 9 digits)
SELECT 
    'customers' AS source_table,
    customer_id AS record_id,
    CONCAT(first_name, ' ', last_name) AS name,
    city,
    state,
    zip_code,
    country,
    CASE 
        WHEN zip_code IS NULL THEN 'NULL'
        WHEN LENGTH(zip_code) = 0 THEN 'EMPTY'
        WHEN country = 'USA' AND zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$' THEN 'INVALID_FORMAT'
        WHEN country = 'USA' AND zip_code REGEXP '^[0-9]{5}$' AND CAST(zip_code AS UNSIGNED) > 99999 THEN 'OUT_OF_RANGE'
        ELSE 'OTHER_ISSUE'
    END AS issue_type
FROM customers
WHERE (country = 'USA' OR country IS NULL)
  AND (zip_code IS NULL 
       OR LENGTH(zip_code) = 0
       OR zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$')

UNION ALL

SELECT 
    'shipping_addresses' AS source_table,
    address_id AS record_id,
    customer_id AS name,
    city,
    state,
    zip_code,
    country,
    CASE 
        WHEN zip_code IS NULL THEN 'NULL'
        WHEN LENGTH(zip_code) = 0 THEN 'EMPTY'
        WHEN country = 'USA' AND zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$' THEN 'INVALID_FORMAT'
        WHEN country = 'USA' AND zip_code REGEXP '^[0-9]{5}$' AND CAST(zip_code AS UNSIGNED) > 99999 THEN 'OUT_OF_RANGE'
        ELSE 'OTHER_ISSUE'
    END AS issue_type
FROM shipping_addresses
WHERE (country = 'USA' OR country IS NULL)
  AND (zip_code IS NULL 
       OR LENGTH(zip_code) = 0
       OR zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$')

UNION ALL

SELECT 
    'vendors' AS source_table,
    vendor_id AS record_id,
    vendor_name AS name,
    city,
    state,
    zip_code,
    'USA' AS country,
    CASE 
        WHEN zip_code IS NULL THEN 'NULL'
        WHEN LENGTH(zip_code) = 0 THEN 'EMPTY'
        WHEN zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$' THEN 'INVALID_FORMAT'
        ELSE 'OTHER_ISSUE'
    END AS issue_type
FROM vendors
WHERE zip_code IS NULL 
   OR LENGTH(zip_code) = 0
   OR zip_code NOT REGEXP '^[0-9]{5}(-[0-9]{4})?$'
ORDER BY source_table, record_id;

-- ========================================
-- 2. MISSING OR INCOMPLETE ADDRESS DATA
-- ========================================

SELECT 
    'customers' AS source_table,
    customer_id AS record_id,
    CONCAT(first_name, ' ', last_name) AS name,
    email,
    CASE 
        WHEN address_line1 IS NULL OR LENGTH(TRIM(address_line1)) = 0 THEN 'MISSING_ADDRESS_LINE1'
        ELSE NULL
    END AS address_issue,
    CASE 
        WHEN city IS NULL OR LENGTH(TRIM(city)) = 0 THEN 'MISSING_CITY'
        ELSE NULL
    END AS city_issue,
    CASE 
        WHEN state IS NULL OR LENGTH(TRIM(state)) = 0 THEN 'MISSING_STATE'
        ELSE NULL
    END AS state_issue,
    CASE 
        WHEN zip_code IS NULL OR LENGTH(TRIM(zip_code)) = 0 THEN 'MISSING_ZIP'
        ELSE NULL
    END AS zip_issue,
    CASE 
        WHEN country IS NULL OR LENGTH(TRIM(country)) = 0 THEN 'MISSING_COUNTRY'
        ELSE NULL
    END AS country_issue
FROM customers
WHERE (address_line1 IS NULL OR LENGTH(TRIM(address_line1)) = 0)
   OR (city IS NULL OR LENGTH(TRIM(city)) = 0)
   OR (state IS NULL OR LENGTH(TRIM(state)) = 0)
   OR (zip_code IS NULL OR LENGTH(TRIM(zip_code)) = 0)

UNION ALL

SELECT 
    'shipping_addresses' AS source_table,
    address_id AS record_id,
    customer_id AS name,
    NULL AS email,
    CASE 
        WHEN address_line1 IS NULL OR LENGTH(TRIM(address_line1)) = 0 THEN 'MISSING_ADDRESS_LINE1'
        ELSE NULL
    END AS address_issue,
    CASE 
        WHEN city IS NULL OR LENGTH(TRIM(city)) = 0 THEN 'MISSING_CITY'
        ELSE NULL
    END AS city_issue,
    CASE 
        WHEN state IS NULL OR LENGTH(TRIM(state)) = 0 THEN 'MISSING_STATE'
        ELSE NULL
    END AS state_issue,
    CASE 
        WHEN zip_code IS NULL OR LENGTH(TRIM(zip_code)) = 0 THEN 'MISSING_ZIP'
        ELSE NULL
    END AS zip_issue,
    CASE 
        WHEN country IS NULL OR LENGTH(TRIM(country)) = 0 THEN 'MISSING_COUNTRY'
        ELSE NULL
    END AS country_issue
FROM shipping_addresses
WHERE (address_line1 IS NULL OR LENGTH(TRIM(address_line1)) = 0)
   OR (city IS NULL OR LENGTH(TRIM(city)) = 0)
   OR (state IS NULL OR LENGTH(TRIM(state)) = 0)
   OR (zip_code IS NULL OR LENGTH(TRIM(zip_code)) = 0)
ORDER BY source_table, record_id;

-- ========================================
-- 3. INVALID STATE CODES
-- ========================================

-- Check for invalid US state abbreviations
SELECT 
    'customers' AS source_table,
    customer_id AS record_id,
    CONCAT(first_name, ' ', last_name) AS name,
    city,
    state,
    zip_code,
    CASE 
        WHEN state NOT IN (
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
        ) THEN 'INVALID_STATE_CODE'
        WHEN LENGTH(state) != 2 THEN 'INVALID_LENGTH'
        ELSE 'OTHER'
    END AS issue_type
FROM customers
WHERE country = 'USA'
  AND (state IS NULL 
       OR LENGTH(state) != 2
       OR state NOT IN (
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
       ))

UNION ALL

SELECT 
    'shipping_addresses' AS source_table,
    address_id AS record_id,
    customer_id AS name,
    city,
    state,
    zip_code,
    CASE 
        WHEN state NOT IN (
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
        ) THEN 'INVALID_STATE_CODE'
        WHEN LENGTH(state) != 2 THEN 'INVALID_LENGTH'
        ELSE 'OTHER'
    END AS issue_type
FROM shipping_addresses
WHERE country = 'USA'
  AND (state IS NULL 
       OR LENGTH(state) != 2
       OR state NOT IN (
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
       ))
ORDER BY source_table, record_id;

-- ========================================
-- 4. STATE-ZIP CODE MISMATCH DETECTION
-- ========================================

-- Common state-ZIP code first digit mappings
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.city,
    c.state,
    c.zip_code,
    SUBSTRING(c.zip_code, 1, 1) AS zip_first_digit,
    CASE 
        -- Examples of expected ZIP ranges by state
        WHEN c.state IN ('CT', 'MA', 'ME', 'NH', 'NJ', 'RI', 'VT') AND SUBSTRING(c.zip_code, 1, 1) != '0' THEN 'MISMATCH'
        WHEN c.state IN ('DC', 'DE', 'MD', 'NY', 'PA') AND SUBSTRING(c.zip_code, 1, 1) NOT IN ('0', '1') THEN 'MISMATCH'
        WHEN c.state IN ('FL', 'GA', 'NC', 'SC', 'VA', 'WV') AND SUBSTRING(c.zip_code, 1, 1) NOT IN ('2', '3') THEN 'MISMATCH'
        WHEN c.state IN ('AL', 'KY', 'MS', 'TN') AND SUBSTRING(c.zip_code, 1, 1) NOT IN ('3', '4') THEN 'MISMATCH'
        WHEN c.state IN ('IN', 'MI', 'OH') AND SUBSTRING(c.zip_code, 1, 1) NOT IN ('4', '5') THEN 'MISMATCH'
        WHEN c.state IN ('IA', 'IL', 'MN', 'MO', 'WI') AND SUBSTRING(c.zip_code, 1, 1) NOT IN ('5', '6') THEN 'MISMATCH'
        WHEN c.state IN ('AR', 'KS', 'LA', 'NE', 'OK', 'SD', 'TX') AND SUBSTRING(c.zip_code, 1, 1) NOT IN ('6', '7') THEN 'MISMATCH'
        WHEN c.state IN ('AZ', 'CO', 'ID', 'MT', 'ND', 'NM', 'UT', 'WY') AND SUBSTRING(c.zip_code, 1, 1) NOT IN ('7', '8') THEN 'MISMATCH'
        WHEN c.state IN ('AK', 'CA', 'HI', 'NV', 'OR', 'WA') AND SUBSTRING(c.zip_code, 1, 1) != '9' THEN 'MISMATCH'
        ELSE 'OK'
    END AS validation_status
FROM customers c
WHERE c.country = 'USA'
  AND c.zip_code REGEXP '^[0-9]{5}'
HAVING validation_status = 'MISMATCH'
ORDER BY c.state, c.zip_code;

-- ========================================
-- 5. DUPLICATE ADDRESSES
-- ========================================

-- Find exact duplicate addresses across customers
SELECT 
    address_line1,
    address_line2,
    city,
    state,
    zip_code,
    country,
    COUNT(DISTINCT customer_id) AS customer_count,
    GROUP_CONCAT(DISTINCT customer_id ORDER BY customer_id) AS customer_ids,
    GROUP_CONCAT(DISTINCT CONCAT(first_name, ' ', last_name) SEPARATOR '; ') AS customer_names
FROM customers
WHERE address_line1 IS NOT NULL
GROUP BY address_line1, address_line2, city, state, zip_code, country
HAVING COUNT(DISTINCT customer_id) > 1
ORDER BY customer_count DESC, address_line1;

-- ========================================
-- 6. SUSPICIOUS CITY-STATE COMBINATIONS
-- ========================================

-- Cities that appear in multiple states (potential data entry errors)
WITH city_states AS (
    SELECT 
        city,
        state,
        COUNT(*) AS occurrence_count
    FROM (
        SELECT UPPER(TRIM(city)) AS city, state FROM customers WHERE country = 'USA'
        UNION ALL
        SELECT UPPER(TRIM(city)) AS city, state FROM shipping_addresses WHERE country = 'USA'
    ) AS all_addresses
    WHERE city IS NOT NULL AND state IS NOT NULL
    GROUP BY city, state
),
cities_multiple_states AS (
    SELECT 
        city,
        COUNT(DISTINCT state) AS state_count,
        GROUP_CONCAT(DISTINCT state ORDER BY state) AS states,
        SUM(occurrence_count) AS total_occurrences
    FROM city_states
    GROUP BY city
    HAVING COUNT(DISTINCT state) > 1
)
SELECT 
    cms.city,
    cms.state_count,
    cms.states,
    cms.total_occurrences,
    cs.state,
    cs.occurrence_count
FROM cities_multiple_states cms
JOIN city_states cs ON cms.city = cs.city
WHERE cms.total_occurrences >= 3  -- Focus on cities with significant occurrences
ORDER BY cms.total_occurrences DESC, cms.city, cs.occurrence_count DESC;

-- ========================================
-- 7. ORDERS WITH MISMATCHED GEOGRAPHY
-- ========================================

-- Orders where customer and shipping address have different locations
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.city AS customer_city,
    c.state AS customer_state,
    c.zip_code AS customer_zip,
    sa.city AS shipping_city,
    sa.state AS shipping_state,
    sa.zip_code AS shipping_zip,
    o.order_date,
    o.total_amount,
    o.status
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id AND sa.is_default = TRUE
WHERE (c.city != sa.city OR c.state != sa.state OR c.zip_code != sa.zip_code)
  AND sa.address_id IS NOT NULL
ORDER BY o.order_date DESC;

-- ========================================
-- 8. INTERNATIONAL ADDRESS VALIDATION
-- ========================================

-- Check for addresses with country != USA but using US-style formatting
SELECT 
    'customers' AS source_table,
    customer_id AS record_id,
    CONCAT(first_name, ' ', last_name) AS name,
    country,
    state,
    zip_code,
    'Non-USA country with US state code' AS issue
FROM customers
WHERE country != 'USA'
  AND country IS NOT NULL
  AND state IN (
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
  )

UNION ALL

SELECT 
    'shipping_addresses' AS source_table,
    address_id AS record_id,
    customer_id AS name,
    country,
    state,
    zip_code,
    'Non-USA country with US state code' AS issue
FROM shipping_addresses
WHERE country != 'USA'
  AND country IS NOT NULL
  AND state IN (
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
  )
ORDER BY source_table, record_id;

-- ========================================
-- 9. GEOGRAPHIC CONCENTRATION ANALYSIS
-- ========================================

-- Identify unusual geographic concentrations (potential data quality issues or fraud)
SELECT 
    c.state,
    c.city,
    c.zip_code,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    COUNT(DISTINCT o.order_id) AS order_count,
    SUM(o.total_amount) AS total_revenue,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value,
    COUNT(DISTINCT o.order_id) / COUNT(DISTINCT c.customer_id) AS orders_per_customer
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.country = 'USA'
GROUP BY c.state, c.city, c.zip_code
HAVING customer_count >= 5
ORDER BY customer_count DESC, order_count DESC
LIMIT 50;

-- ========================================
-- 10. PO BOX ADDRESS DETECTION
-- ========================================

-- Identify PO Box addresses (may require special shipping handling)
SELECT 
    'customers' AS source_table,
    customer_id AS record_id,
    CONCAT(first_name, ' ', last_name) AS name,
    address_line1,
    address_line2,
    city,
    state,
    zip_code
FROM customers
WHERE UPPER(address_line1) REGEXP 'P\\.?O\\.? BOX|POST OFFICE BOX|POBOX'
   OR UPPER(address_line2) REGEXP 'P\\.?O\\.? BOX|POST OFFICE BOX|POBOX'

UNION ALL

SELECT 
    'shipping_addresses' AS source_table,
    address_id AS record_id,
    customer_id AS name,
    address_line1,
    address_line2,
    city,
    state,
    zip_code
FROM shipping_addresses
WHERE UPPER(address_line1) REGEXP 'P\\.?O\\.? BOX|POST OFFICE BOX|POBOX'
   OR UPPER(address_line2) REGEXP 'P\\.?O\\.? BOX|POST OFFICE BOX|POBOX'
ORDER BY source_table, record_id;

-- ========================================
-- 11. CUSTOMERS WITH MULTIPLE SHIPPING ADDRESSES
-- ========================================

-- Analyze customers with unusually high number of shipping addresses
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.status,
    COUNT(sa.address_id) AS address_count,
    COUNT(DISTINCT sa.state) AS unique_states,
    COUNT(DISTINCT sa.city) AS unique_cities,
    GROUP_CONCAT(DISTINCT sa.state ORDER BY sa.state) AS states,
    c.created_at AS customer_since
FROM customers c
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.status, c.created_at
HAVING COUNT(sa.address_id) >= 5
ORDER BY address_count DESC, unique_states DESC;

-- ========================================
-- 12. ADDRESS QUALITY SCORE
-- ========================================

-- Comprehensive address quality scoring
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    address_line1,
    city,
    state,
    zip_code,
    country,
    CASE WHEN address_line1 IS NOT NULL AND LENGTH(TRIM(address_line1)) > 0 THEN 20 ELSE 0 END +
    CASE WHEN city IS NOT NULL AND LENGTH(TRIM(city)) > 0 THEN 20 ELSE 0 END +
    CASE WHEN state IS NOT NULL AND LENGTH(state) = 2 THEN 20 ELSE 0 END +
    CASE WHEN zip_code REGEXP '^[0-9]{5}(-[0-9]{4})? THEN 20 ELSE 0 END +
    CASE WHEN country IS NOT NULL AND LENGTH(TRIM(country)) > 0 THEN 10 ELSE 0 END +
    CASE WHEN phone IS NOT NULL AND LENGTH(TRIM(phone)) >= 10 THEN 10 ELSE 0 END AS quality_score,
    CASE 
        WHEN (
            CASE WHEN address_line1 IS NOT NULL AND LENGTH(TRIM(address_line1)) > 0 THEN 20 ELSE 0 END +
            CASE WHEN city IS NOT NULL AND LENGTH(TRIM(city)) > 0 THEN 20 ELSE 0 END +
            CASE WHEN state IS NOT NULL AND LENGTH(state) = 2 THEN 20 ELSE 0 END +
            CASE WHEN zip_code REGEXP '^[0-9]{5}(-[0-9]{4})? THEN 20 ELSE 0 END +
            CASE WHEN country IS NOT NULL AND LENGTH(TRIM(country)) > 0 THEN 10 ELSE 0 END +
            CASE WHEN phone IS NOT NULL AND LENGTH(TRIM(phone)) >= 10 THEN 10 ELSE 0 END
        ) >= 90 THEN 'EXCELLENT'
        WHEN (
            CASE WHEN address_line1 IS NOT NULL AND LENGTH(TRIM(address_line1)) > 0 THEN 20 ELSE 0 END +
            CASE WHEN city IS NOT NULL AND LENGTH(TRIM(city)) > 0 THEN 20 ELSE 0 END +
            CASE WHEN state IS NOT NULL AND LENGTH(state) = 2 THEN 20 ELSE 0 END +
            CASE WHEN zip_code REGEXP '^[0-9]{5}(-[0-9]{4})? THEN 20 ELSE 0 END +
            CASE WHEN country IS NOT NULL AND LENGTH(TRIM(country)) > 0 THEN 10 ELSE 0 END +
            CASE WHEN phone IS NOT NULL AND LENGTH(TRIM(phone)) >= 10 THEN 10 ELSE 0 END
        ) >= 70 THEN 'GOOD'
        WHEN (
            CASE WHEN address_line1 IS NOT NULL AND LENGTH(TRIM(address_line1)) > 0 THEN 20 ELSE 0 END +
            CASE WHEN city IS NOT NULL AND LENGTH(TRIM(city)) > 0 THEN 20 ELSE 0 END +
            CASE WHEN state IS NOT NULL AND LENGTH(state) = 2 THEN 20 ELSE 0 END +
            CASE WHEN zip_code REGEXP '^[0-9]{5}(-[0-9]{4})? THEN 20 ELSE 0 END +
            CASE WHEN country IS NOT NULL AND LENGTH(TRIM(country)) > 0 THEN 10 ELSE 0 END +
            CASE WHEN phone IS NOT NULL AND LENGTH(TRIM(phone)) >= 10 THEN 10 ELSE 0 END
        ) >= 50 THEN 'FAIR'
        ELSE 'POOR'
    END AS quality_grade
FROM customers
WHERE country = 'USA' OR country IS NULL
ORDER BY quality_score ASC, customer_id
LIMIT 100;

-- ========================================
-- 13. GEOGRAPHIC OUTLIER ORDERS
-- ========================================

-- Orders shipped to locations far from customer's registered address
SELECT 
    o.order_id,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.state AS customer_state,
    c.city AS customer_city,
    sa.state AS shipping_state,
    sa.city AS shipping_city,
    o.total_amount,
    o.order_date,
    o.status,
    CASE 
        WHEN c.state != sa.state THEN 'DIFFERENT_STATE'
        WHEN c.city != sa.city THEN 'DIFFERENT_CITY'
        ELSE 'SAME_LOCATION'
    END AS location_difference
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN shipping_addresses sa ON c.customer_id = sa.customer_id
WHERE sa.is_default = TRUE
  AND (c.state != sa.state OR c.city != sa.city)
  AND o.total_amount > 500  -- Focus on high-value orders
ORDER BY o.total_amount DESC, o.order_date DESC;

-- ========================================
-- END OF GEOGRAPHIC ANOMALIES ANALYSIS
-- ========================================