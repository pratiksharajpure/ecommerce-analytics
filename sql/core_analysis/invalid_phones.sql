-- ========================================
-- INVALID PHONE NUMBER VALIDATION
-- Week 2, Day 3: Phone format and validation
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. MISSING PHONE NUMBERS
-- ========================================
SELECT 
    'Missing Phone Numbers' AS validation_type,
    COUNT(*) AS affected_customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers
WHERE phone IS NULL OR phone = '' OR TRIM(phone) = '';

-- ========================================
-- 2. PHONE NUMBERS WITH LETTERS
-- ========================================
SELECT 
    'Contains Letters' AS validation_type,
    customer_id,
    phone,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    created_at
FROM customers
WHERE phone IS NOT NULL 
    AND phone REGEXP '[a-zA-Z]'
ORDER BY customer_id;

-- ========================================
-- 3. PHONE NUMBERS TOO SHORT OR TOO LONG
-- ========================================
SELECT 
    'Invalid Length' AS validation_type,
    customer_id,
    phone,
    LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) AS digit_count,
    CASE 
        WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) < 10 THEN 'Too Short (< 10 digits)'
        WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) > 15 THEN 'Too Long (> 15 digits)'
    END AS length_issue,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email
FROM customers
WHERE phone IS NOT NULL 
    AND phone != ''
    AND (
        LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) < 10 OR 
        LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) > 15
    )
ORDER BY digit_count, customer_id;

-- ========================================
-- 4. PHONE NUMBERS WITH INVALID CHARACTERS
-- ========================================
SELECT 
    'Invalid Characters' AS validation_type,
    customer_id,
    phone,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email
FROM customers
WHERE phone IS NOT NULL 
    AND phone != ''
    AND phone REGEXP '[^0-9\\(\\)\\-\\+\\. ]'
ORDER BY customer_id;

-- ========================================
-- 5. SUSPICIOUS PATTERNS - ALL SAME DIGITS
-- ========================================
SELECT 
    'All Same Digits' AS validation_type,
    customer_id,
    phone,
    REGEXP_REPLACE(phone, '[^0-9]', '') AS digits_only,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE phone IS NOT NULL 
    AND phone != ''
    AND (
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^0+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^1+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^2+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^3+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^4+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^5+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^6+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^7+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^8+$' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') REGEXP '^9+$'
    )
ORDER BY customer_id;

-- ========================================
-- 6. SEQUENTIAL NUMBERS (123456... or 987654...)
-- ========================================
SELECT 
    'Sequential Digits' AS validation_type,
    customer_id,
    phone,
    REGEXP_REPLACE(phone, '[^0-9]', '') AS digits_only,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE phone IS NOT NULL 
    AND phone != ''
    AND (
        REGEXP_REPLACE(phone, '[^0-9]', '') LIKE '%0123456%' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') LIKE '%1234567%' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') LIKE '%2345678%' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') LIKE '%3456789%' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') LIKE '%9876543%' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') LIKE '%8765432%' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') LIKE '%7654321%'
    )
ORDER BY customer_id;

-- ========================================
-- 7. TEST/FAKE PHONE NUMBERS
-- ========================================
SELECT 
    'Test/Fake Numbers' AS validation_type,
    customer_id,
    phone,
    REGEXP_REPLACE(phone, '[^0-9]', '') AS digits_only,
    CASE 
        WHEN phone LIKE '555-555-%' THEN 'Hollywood fake (555-555-xxxx)'
        WHEN phone LIKE '%5555555%' THEN 'Repeated 5s pattern'
        WHEN REGEXP_REPLACE(phone, '[^0-9]', '') = '0000000000' THEN 'All zeros'
        WHEN REGEXP_REPLACE(phone, '[^0-9]', '') = '1111111111' THEN 'All ones'
        WHEN REGEXP_REPLACE(phone, '[^0-9]', '') = '1234567890' THEN 'Sequential test number'
    END AS pattern_type,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE phone IS NOT NULL 
    AND (
        phone LIKE '555-555-%' OR
        phone LIKE '%5555555%' OR
        REGEXP_REPLACE(phone, '[^0-9]', '') IN ('0000000000', '1111111111', '1234567890')
    )
ORDER BY pattern_type, customer_id;

-- ========================================
-- 8. PHONE NUMBERS WITH UNUSUAL FORMATTING
-- ========================================
SELECT 
    'Unusual Formatting' AS validation_type,
    customer_id,
    phone,
    LENGTH(phone) AS total_length,
    LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) AS digit_count,
    CASE 
        WHEN phone LIKE '% %' AND phone LIKE '% % %' THEN 'Multiple spaces'
        WHEN phone LIKE '%  %' THEN 'Consecutive spaces'
        WHEN phone LIKE '%.%' THEN 'Contains dots'
        WHEN phone LIKE '%/%' THEN 'Contains slashes'
        WHEN phone NOT REGEXP '^[\\+]?[0-9\\(\\)\\-\\. ]+$' THEN 'Mixed unusual chars'
    END AS format_issue,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE phone IS NOT NULL 
    AND phone != ''
    AND (
        (phone LIKE '% %' AND phone LIKE '% % %') OR
        phone LIKE '%  %' OR
        phone LIKE '%.%' OR
        phone LIKE '%/%' OR
        phone NOT REGEXP '^[\\+]?[0-9\\(\\)\\-\\. ]+$'
    )
ORDER BY customer_id
LIMIT 50;

-- ========================================
-- 9. US PHONE NUMBERS WITH INVALID AREA CODES
-- ========================================
-- Area codes 000-199 and certain others are invalid
SELECT 
    'Invalid US Area Code' AS validation_type,
    customer_id,
    phone,
    SUBSTRING(REGEXP_REPLACE(phone, '[^0-9]', ''), 1, 3) AS area_code,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    state
FROM customers
WHERE phone IS NOT NULL 
    AND phone != ''
    AND LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10
    AND country = 'USA'
    AND (
        CAST(SUBSTRING(REGEXP_REPLACE(phone, '[^0-9]', ''), 1, 3) AS UNSIGNED) < 200 OR
        SUBSTRING(REGEXP_REPLACE(phone, '[^0-9]', ''), 1, 3) IN ('555', '911', '000', '111')
    )
ORDER BY area_code, customer_id;

-- ========================================
-- 10. PHONES STARTING WITH COUNTRY CODE +1 FOR NON-US
-- ========================================
SELECT 
    'Country Code Mismatch' AS validation_type,
    customer_id,
    phone,
    country,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE phone IS NOT NULL 
    AND phone LIKE '+1%'
    AND country != 'USA'
    AND country != 'Canada'
ORDER BY country, customer_id;

-- ========================================
-- 11. PHONE NUMBER FORMAT CONSISTENCY
-- ========================================
SELECT 
    'Format Patterns' AS analysis_type,
    CASE 
        WHEN phone REGEXP '^\\+[0-9]' THEN 'International (+X...)'
        WHEN phone REGEXP '^\\([0-9]{3}\\)' THEN 'US Format (XXX) XXX-XXXX'
        WHEN phone REGEXP '^[0-9]{3}-[0-9]{3}-[0-9]{4}$' THEN 'US Format XXX-XXX-XXXX'
        WHEN phone REGEXP '^[0-9]{10}$' THEN 'US Format No separators'
        WHEN phone REGEXP '^[0-9]{3}\\.[0-9]{3}\\.[0-9]{4}$' THEN 'Dotted format'
        WHEN phone REGEXP '^[0-9 ]+$' THEN 'Space-separated'
        ELSE 'Other/Mixed format'
    END AS format_type,
    COUNT(*) AS phone_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers WHERE phone IS NOT NULL AND phone != ''), 2) AS percentage
FROM customers
WHERE phone IS NOT NULL AND phone != ''
GROUP BY format_type
ORDER BY phone_count DESC;

-- ========================================
-- 12. PHONE NUMBERS BY DIGIT LENGTH
-- ========================================
SELECT 
    'Digit Length Distribution' AS analysis_type,
    LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) AS digit_count,
    COUNT(*) AS phone_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers WHERE phone IS NOT NULL AND phone != ''), 2) AS percentage,
    CASE 
        WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 10 THEN 'US/Canada Standard'
        WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) = 11 THEN 'International with country code'
        WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) < 10 THEN 'Too Short'
        WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) > 11 THEN 'Too Long or Different country'
    END AS classification
FROM customers
WHERE phone IS NOT NULL AND phone != ''
GROUP BY digit_count
ORDER BY phone_count DESC;

-- ========================================
-- 13. EMPTY OR WHITESPACE-ONLY PHONES
-- ========================================
SELECT 
    'Empty/Whitespace' AS validation_type,
    customer_id,
    CONCAT('''', phone, '''') AS phone_with_quotes,
    LENGTH(phone) AS total_length,
    LENGTH(TRIM(phone)) AS trimmed_length,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE phone IS NOT NULL 
    AND (
        phone = '' OR 
        TRIM(phone) = '' OR
        phone REGEXP '^[ \\t\\n\\r]+$'
    )
ORDER BY customer_id;

-- ========================================
-- 14. PHONES WITH EXTENSIONS OR ADDITIONAL TEXT
-- ========================================
SELECT 
    'Contains Extension/Text' AS validation_type,
    customer_id,
    phone,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE phone IS NOT NULL 
    AND (
        phone LIKE '%ext%' OR
        phone LIKE '%x%' OR
        phone LIKE '%extension%' OR
        phone LIKE '%#%'
    )
ORDER BY customer_id
LIMIT 50;

-- ========================================
-- 15. COMPREHENSIVE PHONE VALIDATION SUMMARY
-- ========================================
SELECT 
    'VALIDATION SUMMARY' AS report_type,
    COUNT(*) AS total_customers,
    COUNT(CASE WHEN phone IS NULL OR phone = '' OR TRIM(phone) = '' THEN 1 END) AS missing_phones,
    COUNT(CASE WHEN phone REGEXP '[a-zA-Z]' THEN 1 END) AS contains_letters,
    COUNT(CASE WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) < 10 THEN 1 END) AS too_short,
    COUNT(CASE WHEN LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) > 15 THEN 1 END) AS too_long,
    COUNT(CASE WHEN phone REGEXP '[^0-9\\(\\)\\-\\+\\. ]' THEN 1 END) AS invalid_characters,
    COUNT(CASE WHEN 
        phone IS NOT NULL AND phone != '' 
        AND phone NOT REGEXP '[a-zA-Z]'
        AND LENGTH(REGEXP_REPLACE(phone, '[^0-9]', '')) BETWEEN 10 AND 15
        AND phone REGEXP '^[\\+]?[0-9\\(\\)\\-\\. ]+$'
    THEN 1 END) AS potentially_valid
FROM customers;

-- ========================================
-- 16. CUSTOMERS WITH BOTH INVALID EMAIL AND PHONE
-- ========================================
SELECT 
    'Critical: Both Invalid' AS validation_type,
    c.customer_id,
    c.email,
    c.phone,
    c.status,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.created_at,
    COALESCE(o.order_count, 0) AS order_count
FROM customers c
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS order_count 
    FROM orders 
    GROUP BY customer_id
) o ON c.customer_id = o.customer_id
WHERE 
    (c.email IS NULL OR c.email = '' OR c.email NOT LIKE '%@%') 
    AND (c.phone IS NULL OR c.phone = '' OR LENGTH(REGEXP_REPLACE(c.phone, '[^0-9]', '')) < 10)
ORDER BY order_count DESC, c.customer_id
LIMIT 100;

-- Summary Report
SELECT '========== PHONE VALIDATION COMPLETE ==========' AS status;