-- ========================================
-- CUSTOMER DUPLICATE DETECTION
-- Week 2, Day 2: Identify duplicate customers
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. DUPLICATE EMAILS
-- ========================================
SELECT 
    'Duplicate Emails' AS issue_type,
    email,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS customer_ids,
    MIN(created_at) AS first_registration,
    MAX(created_at) AS last_registration
FROM customers
WHERE email IS NOT NULL AND email != ''
GROUP BY email
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, email;

-- ========================================
-- 2. DUPLICATE PHONE NUMBERS
-- ========================================
SELECT 
    'Duplicate Phones' AS issue_type,
    phone,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS customer_ids,
    GROUP_CONCAT(CONCAT(first_name, ' ', last_name) SEPARATOR ' | ') AS customer_names,
    MIN(created_at) AS first_registration,
    MAX(created_at) AS last_registration
FROM customers
WHERE phone IS NOT NULL AND phone != ''
GROUP BY phone
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, phone;

-- ========================================
-- 3. EXACT NAME DUPLICATES (First + Last Name)
-- ========================================
SELECT 
    'Duplicate Names' AS issue_type,
    first_name,
    last_name,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS customer_ids,
    GROUP_CONCAT(email SEPARATOR ' | ') AS emails,
    GROUP_CONCAT(phone SEPARATOR ' | ') AS phones
FROM customers
WHERE first_name IS NOT NULL AND last_name IS NOT NULL
GROUP BY first_name, last_name
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, last_name, first_name
LIMIT 50;

-- ========================================
-- 4. SAME ADDRESS DUPLICATES
-- ========================================
SELECT 
    'Duplicate Addresses' AS issue_type,
    address_line1,
    city,
    state,
    zip_code,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS customer_ids,
    GROUP_CONCAT(CONCAT(first_name, ' ', last_name) SEPARATOR ' | ') AS customer_names,
    GROUP_CONCAT(email SEPARATOR ' | ') AS emails
FROM customers
WHERE address_line1 IS NOT NULL AND address_line1 != ''
GROUP BY address_line1, city, state, zip_code
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 30;

-- ========================================
-- 5. POTENTIAL DUPLICATES: EMAIL + NAME MATCH
-- ========================================
SELECT 
    'Email + Name Match' AS match_type,
    c1.customer_id AS customer_id_1,
    c2.customer_id AS customer_id_2,
    c1.email,
    CONCAT(c1.first_name, ' ', c1.last_name) AS name_1,
    CONCAT(c2.first_name, ' ', c2.last_name) AS name_2,
    c1.created_at AS registration_1,
    c2.created_at AS registration_2
FROM customers c1
JOIN customers c2 ON c1.email = c2.email 
    AND c1.customer_id < c2.customer_id
    AND c1.first_name = c2.first_name 
    AND c1.last_name = c2.last_name
WHERE c1.email IS NOT NULL
ORDER BY c1.email, c1.customer_id;

-- ========================================
-- 6. POTENTIAL DUPLICATES: SIMILAR NAMES + SAME ADDRESS
-- ========================================
SELECT 
    'Similar Name + Address' AS match_type,
    c1.customer_id AS customer_id_1,
    c2.customer_id AS customer_id_2,
    CONCAT(c1.first_name, ' ', c1.last_name) AS name_1,
    CONCAT(c2.first_name, ' ', c2.last_name) AS name_2,
    c1.email AS email_1,
    c2.email AS email_2,
    c1.address_line1,
    c1.city,
    c1.state
FROM customers c1
JOIN customers c2 ON c1.customer_id < c2.customer_id
    AND c1.address_line1 = c2.address_line1
    AND c1.city = c2.city
    AND c1.state = c2.state
    AND c1.zip_code = c2.zip_code
    AND SOUNDEX(c1.last_name) = SOUNDEX(c2.last_name)
WHERE c1.address_line1 IS NOT NULL AND c1.address_line1 != ''
ORDER BY c1.address_line1, c1.customer_id
LIMIT 50;

-- ========================================
-- 7. PHONE NUMBER VARIATIONS (Same digits, different formats)
-- ========================================
-- Remove non-numeric characters and compare
SELECT 
    'Phone Format Variations' AS issue_type,
    phone,
    REGEXP_REPLACE(phone, '[^0-9]', '') AS digits_only,
    COUNT(*) AS customer_count,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS customer_ids,
    GROUP_CONCAT(CONCAT(first_name, ' ', last_name) SEPARATOR ' | ') AS customer_names
FROM customers
WHERE phone IS NOT NULL AND phone != ''
GROUP BY REGEXP_REPLACE(phone, '[^0-9]', '')
HAVING COUNT(*) > 1
ORDER BY customer_count DESC
LIMIT 30;

-- ========================================
-- 8. EMAIL DOMAIN TYPOS (Common misspellings)
-- ========================================
SELECT 
    'Potential Email Typos' AS issue_type,
    email,
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    SUBSTRING_INDEX(email, '@', -1) AS domain
FROM customers
WHERE email IS NOT NULL 
    AND (
        email LIKE '%@gmial.com' OR
        email LIKE '%@gmai.com' OR
        email LIKE '%@yahooo.com' OR
        email LIKE '%@yaho.com' OR
        email LIKE '%@hotmial.com' OR
        email LIKE '%@outlok.com' OR
        email LIKE '%@iclod.com' OR
        email LIKE '%@aol.co'
    )
ORDER BY email;

-- ========================================
-- 9. CASE-SENSITIVE EMAIL DUPLICATES
-- ========================================
SELECT 
    'Case-Sensitive Duplicates' AS issue_type,
    LOWER(email) AS email_normalized,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(email SEPARATOR ' | ') AS email_variations,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS customer_ids
FROM customers
WHERE email IS NOT NULL AND email != ''
GROUP BY LOWER(email)
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;

-- ========================================
-- 10. SUSPICIOUS PATTERN: Multiple accounts created same day
-- ========================================
SELECT 
    'Same Day Registrations' AS pattern,
    DATE(created_at) AS registration_date,
    COUNT(*) AS accounts_created,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS customer_ids,
    GROUP_CONCAT(email SEPARATOR ' | ') AS emails
FROM customers
GROUP BY DATE(created_at), 
         CASE 
             WHEN address_line1 IS NOT NULL THEN address_line1
             WHEN phone IS NOT NULL THEN phone
             ELSE 'unknown'
         END
HAVING COUNT(*) >= 3
ORDER BY registration_date DESC, accounts_created DESC
LIMIT 20;

-- ========================================
-- 11. WHITESPACE-TRIMMED EMAIL DUPLICATES
-- ========================================
SELECT 
    'Whitespace Issues' AS issue_type,
    TRIM(email) AS email_trimmed,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(CONCAT('''', email, '''') SEPARATOR ' | ') AS email_with_spaces,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS customer_ids
FROM customers
WHERE email IS NOT NULL AND email != ''
GROUP BY TRIM(email)
HAVING COUNT(*) > 1 OR MAX(LENGTH(email)) != MAX(LENGTH(TRIM(email)))
ORDER BY duplicate_count DESC;

-- ========================================
-- 12. COMPREHENSIVE DUPLICATE SUMMARY
-- ========================================
SELECT 
    'DUPLICATE SUMMARY' AS report_type,
    SUM(CASE WHEN dup_type = 'email' THEN 1 ELSE 0 END) AS duplicate_emails,
    SUM(CASE WHEN dup_type = 'phone' THEN 1 ELSE 0 END) AS duplicate_phones,
    SUM(CASE WHEN dup_type = 'name' THEN 1 ELSE 0 END) AS duplicate_names,
    SUM(CASE WHEN dup_type = 'address' THEN 1 ELSE 0 END) AS duplicate_addresses
FROM (
    SELECT 'email' AS dup_type, email AS value FROM customers 
    WHERE email IS NOT NULL GROUP BY email HAVING COUNT(*) > 1
    UNION ALL
    SELECT 'phone', phone FROM customers 
    WHERE phone IS NOT NULL GROUP BY phone HAVING COUNT(*) > 1
    UNION ALL
    SELECT 'name', CONCAT(first_name, ' ', last_name) FROM customers 
    WHERE first_name IS NOT NULL AND last_name IS NOT NULL 
    GROUP BY first_name, last_name HAVING COUNT(*) > 1
    UNION ALL
    SELECT 'address', address_line1 FROM customers 
    WHERE address_line1 IS NOT NULL 
    GROUP BY address_line1, city, state, zip_code HAVING COUNT(*) > 1
) AS all_duplicates;

-- ========================================
-- 13. CUSTOMERS WITH MULTIPLE DUPLICATE INDICATORS
-- ========================================
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    c.phone,
    c.status,
    c.created_at,
    CASE WHEN e.email IS NOT NULL THEN 1 ELSE 0 END AS has_duplicate_email,
    CASE WHEN p.phone IS NOT NULL THEN 1 ELSE 0 END AS has_duplicate_phone,
    CASE WHEN n.name_key IS NOT NULL THEN 1 ELSE 0 END AS has_duplicate_name,
    (CASE WHEN e.email IS NOT NULL THEN 1 ELSE 0 END +
     CASE WHEN p.phone IS NOT NULL THEN 1 ELSE 0 END +
     CASE WHEN n.name_key IS NOT NULL THEN 1 ELSE 0 END) AS total_duplicate_flags
FROM customers c
LEFT JOIN (
    SELECT email FROM customers WHERE email IS NOT NULL 
    GROUP BY email HAVING COUNT(*) > 1
) e ON c.email = e.email
LEFT JOIN (
    SELECT phone FROM customers WHERE phone IS NOT NULL 
    GROUP BY phone HAVING COUNT(*) > 1
) p ON c.phone = p.phone
LEFT JOIN (
    SELECT CONCAT(first_name, '|', last_name) AS name_key 
    FROM customers WHERE first_name IS NOT NULL AND last_name IS NOT NULL
    GROUP BY first_name, last_name HAVING COUNT(*) > 1
) n ON CONCAT(c.first_name, '|', c.last_name) = n.name_key
WHERE e.email IS NOT NULL OR p.phone IS NOT NULL OR n.name_key IS NOT NULL
ORDER BY total_duplicate_flags DESC, c.customer_id;

-- Summary Report
SELECT '========== DUPLICATE DETECTION COMPLETE ==========' AS status;