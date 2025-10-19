-- ========================================
-- INVALID EMAIL VALIDATION
-- Week 2, Day 2: Email format and domain validation
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. MISSING EMAILS
-- ========================================
SELECT 
    'Missing Emails' AS validation_type,
    COUNT(*) AS affected_customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers
WHERE email IS NULL OR email = '' OR TRIM(email) = '';

-- ========================================
-- 2. EMAILS WITHOUT @ SYMBOL
-- ========================================
SELECT 
    'Missing @ Symbol' AS validation_type,
    customer_id,
    email,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    created_at
FROM customers
WHERE email IS NOT NULL 
    AND email != ''
    AND email NOT LIKE '%@%'
ORDER BY customer_id;

-- ========================================
-- 3. EMAILS WITH MULTIPLE @ SYMBOLS
-- ========================================
SELECT 
    'Multiple @ Symbols' AS validation_type,
    customer_id,
    email,
    LENGTH(email) - LENGTH(REPLACE(email, '@', '')) AS at_symbol_count,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND LENGTH(email) - LENGTH(REPLACE(email, '@', '')) > 1
ORDER BY at_symbol_count DESC, customer_id;

-- ========================================
-- 4. EMAILS WITHOUT DOMAIN EXTENSION
-- ========================================
SELECT 
    'Missing Domain Extension' AS validation_type,
    customer_id,
    email,
    SUBSTRING_INDEX(email, '@', -1) AS domain,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND email LIKE '%@%'
    AND SUBSTRING_INDEX(email, '@', -1) NOT LIKE '%.%'
ORDER BY customer_id;

-- ========================================
-- 5. EMAILS WITH SPACES
-- ========================================
SELECT 
    'Contains Spaces' AS validation_type,
    customer_id,
    CONCAT('''', email, '''') AS email_with_quotes,
    LENGTH(email) AS total_length,
    LENGTH(TRIM(email)) AS trimmed_length,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND (email LIKE '% %' OR email != TRIM(email))
ORDER BY customer_id;

-- ========================================
-- 6. EMAILS STARTING/ENDING WITH SPECIAL CHARACTERS
-- ========================================
SELECT 
    'Invalid Start/End Characters' AS validation_type,
    customer_id,
    email,
    CASE 
        WHEN email REGEXP '^[^a-zA-Z0-9]' THEN 'Starts with special char'
        WHEN email REGEXP '@[^a-zA-Z0-9]' THEN 'Domain starts with special char'
        WHEN email REGEXP '[^a-zA-Z0-9]$' THEN 'Ends with special char'
        ELSE 'Other issue'
    END AS issue_detail,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND (
        email REGEXP '^[^a-zA-Z0-9]' OR 
        email REGEXP '@[^a-zA-Z0-9]' OR 
        email REGEXP '[^a-zA-Z0-9]$'
    )
ORDER BY customer_id;

-- ========================================
-- 7. EMAILS WITH CONSECUTIVE DOTS
-- ========================================
SELECT 
    'Consecutive Dots' AS validation_type,
    customer_id,
    email,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND email LIKE '%..%'
ORDER BY customer_id;

-- ========================================
-- 8. EMAILS TOO SHORT OR TOO LONG
-- ========================================
SELECT 
    'Invalid Length' AS validation_type,
    customer_id,
    email,
    LENGTH(email) AS email_length,
    CASE 
        WHEN LENGTH(email) < 5 THEN 'Too Short (< 5 chars)'
        WHEN LENGTH(email) > 100 THEN 'Too Long (> 100 chars)'
    END AS length_issue,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND (LENGTH(email) < 5 OR LENGTH(email) > 100)
ORDER BY email_length DESC, customer_id;

-- ========================================
-- 9. SUSPICIOUS EMAIL PATTERNS
-- ========================================
SELECT 
    'Suspicious Pattern' AS validation_type,
    customer_id,
    email,
    CASE 
        WHEN email LIKE 'test%' THEN 'Test account'
        WHEN email LIKE 'temp%' THEN 'Temporary email'
        WHEN email LIKE 'fake%' THEN 'Fake email'
        WHEN email LIKE '%@test.%' THEN 'Test domain'
        WHEN email LIKE '%@example.%' THEN 'Example domain'
        WHEN email LIKE 'noreply%' THEN 'No-reply email'
        WHEN email LIKE '%+%' THEN 'Plus addressing'
        WHEN email REGEXP '[0-9]{5,}' THEN 'Many consecutive numbers'
    END AS pattern_type,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND (
        email LIKE 'test%' OR
        email LIKE 'temp%' OR
        email LIKE 'fake%' OR
        email LIKE '%@test.%' OR
        email LIKE '%@example.%' OR
        email LIKE 'noreply%' OR
        email LIKE '%+%' OR
        email REGEXP '[0-9]{5,}'
    )
ORDER BY pattern_type, customer_id
LIMIT 50;

-- ========================================
-- 10. COMMON DOMAIN TYPOS
-- ========================================
SELECT 
    'Domain Typo' AS validation_type,
    customer_id,
    email,
    SUBSTRING_INDEX(email, '@', -1) AS current_domain,
    CASE 
        WHEN email LIKE '%@gmial.com' THEN 'gmail.com'
        WHEN email LIKE '%@gmai.com' THEN 'gmail.com'
        WHEN email LIKE '%@gmil.com' THEN 'gmail.com'
        WHEN email LIKE '%@yahooo.com' THEN 'yahoo.com'
        WHEN email LIKE '%@yaho.com' THEN 'yahoo.com'
        WHEN email LIKE '%@yahho.com' THEN 'yahoo.com'
        WHEN email LIKE '%@hotmial.com' THEN 'hotmail.com'
        WHEN email LIKE '%@hotmil.com' THEN 'hotmail.com'
        WHEN email LIKE '%@outlok.com' THEN 'outlook.com'
        WHEN email LIKE '%@outloo.com' THEN 'outlook.com'
        WHEN email LIKE '%@iclod.com' THEN 'icloud.com'
        WHEN email LIKE '%@iclould.com' THEN 'icloud.com'
    END AS suggested_domain,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email LIKE '%@gmial.com' OR email LIKE '%@gmai.com' OR email LIKE '%@gmil.com'
   OR email LIKE '%@yahooo.com' OR email LIKE '%@yaho.com' OR email LIKE '%@yahho.com'
   OR email LIKE '%@hotmial.com' OR email LIKE '%@hotmil.com'
   OR email LIKE '%@outlok.com' OR email LIKE '%@outloo.com'
   OR email LIKE '%@iclod.com' OR email LIKE '%@iclould.com'
ORDER BY suggested_domain, customer_id;

-- ========================================
-- 11. INVALID TOP-LEVEL DOMAINS (TLD)
-- ========================================
SELECT 
    'Invalid/Suspicious TLD' AS validation_type,
    customer_id,
    email,
    SUBSTRING_INDEX(email, '.', -1) AS tld,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND email LIKE '%@%.%'
    AND (
        LENGTH(SUBSTRING_INDEX(email, '.', -1)) < 2 OR
        LENGTH(SUBSTRING_INDEX(email, '.', -1)) > 6 OR
        SUBSTRING_INDEX(email, '.', -1) REGEXP '[^a-zA-Z]'
    )
ORDER BY tld, customer_id
LIMIT 50;

-- ========================================
-- 12. DISPOSABLE EMAIL DOMAINS
-- ========================================
SELECT 
    'Disposable Email' AS validation_type,
    customer_id,
    email,
    SUBSTRING_INDEX(email, '@', -1) AS domain,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    created_at
FROM customers
WHERE email IS NOT NULL 
    AND (
        email LIKE '%@mailinator.com' OR
        email LIKE '%@guerrillamail.%' OR
        email LIKE '%@10minutemail.%' OR
        email LIKE '%@temp-mail.%' OR
        email LIKE '%@throwaway.email' OR
        email LIKE '%@trashmail.%' OR
        email LIKE '%@fakeinbox.%' OR
        email LIKE '%@maildrop.cc' OR
        email LIKE '%@yopmail.%' OR
        email LIKE '%@tempmail.%'
    )
ORDER BY domain, customer_id;

-- ========================================
-- 13. EMAIL LOCAL PART VALIDATION (before @)
-- ========================================
SELECT 
    'Invalid Local Part' AS validation_type,
    customer_id,
    email,
    SUBSTRING_INDEX(email, '@', 1) AS local_part,
    CASE 
        WHEN SUBSTRING_INDEX(email, '@', 1) LIKE '.%' THEN 'Starts with dot'
        WHEN SUBSTRING_INDEX(email, '@', 1) LIKE '%.' THEN 'Ends with dot'
        WHEN SUBSTRING_INDEX(email, '@', 1) LIKE '' THEN 'Empty local part'
        WHEN LENGTH(SUBSTRING_INDEX(email, '@', 1)) > 64 THEN 'Local part too long'
    END AS issue_detail,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND email LIKE '%@%'
    AND (
        SUBSTRING_INDEX(email, '@', 1) LIKE '.%' OR
        SUBSTRING_INDEX(email, '@', 1) LIKE '%.' OR
        SUBSTRING_INDEX(email, '@', 1) = '' OR
        LENGTH(SUBSTRING_INDEX(email, '@', 1)) > 64
    )
ORDER BY customer_id;

-- ========================================
-- 14. DOMAIN PART VALIDATION (after @)
-- ========================================
SELECT 
    'Invalid Domain Part' AS validation_type,
    customer_id,
    email,
    SUBSTRING_INDEX(email, '@', -1) AS domain_part,
    CASE 
        WHEN SUBSTRING_INDEX(email, '@', -1) LIKE '%..' THEN 'Consecutive dots in domain'
        WHEN SUBSTRING_INDEX(email, '@', -1) LIKE '.%' THEN 'Domain starts with dot'
        WHEN SUBSTRING_INDEX(email, '@', -1) LIKE '%.' THEN 'Domain ends with dot'
        WHEN LENGTH(SUBSTRING_INDEX(email, '@', -1)) > 255 THEN 'Domain too long'
    END AS issue_detail,
    CONCAT(first_name, ' ', last_name) AS customer_name
FROM customers
WHERE email IS NOT NULL 
    AND email LIKE '%@%'
    AND (
        SUBSTRING_INDEX(email, '@', -1) LIKE '%..' OR
        SUBSTRING_INDEX(email, '@', -1) LIKE '.%' OR
        SUBSTRING_INDEX(email, '@', -1) LIKE '%.' OR
        LENGTH(SUBSTRING_INDEX(email, '@', -1)) > 255
    )
ORDER BY customer_id;

-- ========================================
-- 15. EMAIL DOMAIN DISTRIBUTION
-- ========================================
SELECT 
    'Domain Distribution' AS analysis_type,
    SUBSTRING_INDEX(email, '@', -1) AS domain,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers WHERE email LIKE '%@%'), 2) AS percentage
FROM customers
WHERE email IS NOT NULL AND email LIKE '%@%'
GROUP BY SUBSTRING_INDEX(email, '@', -1)
ORDER BY customer_count DESC
LIMIT 30;

-- ========================================
-- 16. COMPREHENSIVE EMAIL VALIDATION SUMMARY
-- ========================================
SELECT 
    'VALIDATION SUMMARY' AS report_type,
    COUNT(*) AS total_customers,
    COUNT(CASE WHEN email IS NULL OR email = '' THEN 1 END) AS missing_emails,
    COUNT(CASE WHEN email NOT LIKE '%@%' AND email IS NOT NULL AND email != '' THEN 1 END) AS no_at_symbol,
    COUNT(CASE WHEN email LIKE '%@%' AND SUBSTRING_INDEX(email, '@', -1) NOT LIKE '%.%' THEN 1 END) AS no_domain_extension,
    COUNT(CASE WHEN email LIKE '% %' THEN 1 END) AS contains_spaces,
    COUNT(CASE WHEN LENGTH(email) < 5 AND email IS NOT NULL THEN 1 END) AS too_short,
    COUNT(CASE WHEN LENGTH(email) > 100 THEN 1 END) AS too_long,
    COUNT(CASE WHEN email LIKE '%..%' THEN 1 END) AS consecutive_dots,
    COUNT(CASE WHEN 
        email IS NOT NULL AND email != '' AND email LIKE '%@%' 
        AND SUBSTRING_INDEX(email, '@', -1) LIKE '%.%'
        AND email NOT LIKE '% %' 
        AND LENGTH(email) >= 5 
        AND LENGTH(email) <= 100
        AND email NOT LIKE '%..%'
    THEN 1 END) AS potentially_valid
FROM customers;

-- ========================================
-- 17. EMAILS REQUIRING IMMEDIATE ATTENTION
-- ========================================
SELECT 
    'Priority Issues' AS validation_type,
    c.customer_id,
    c.email,
    c.status,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    o.order_count,
    CASE 
        WHEN c.email IS NULL OR c.email = '' THEN 'CRITICAL: Missing email'
        WHEN c.email NOT LIKE '%@%' THEN 'CRITICAL: No @ symbol'
        WHEN c.email LIKE '%@%' AND SUBSTRING_INDEX(c.email, '@', -1) NOT LIKE '%.%' THEN 'HIGH: No domain extension'
        WHEN c.email LIKE '% %' THEN 'HIGH: Contains spaces'
        WHEN LENGTH(c.email) < 5 THEN 'MEDIUM: Too short'
        ELSE 'LOW: Other validation issue'
    END AS priority_level
FROM customers c
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS order_count 
    FROM orders 
    GROUP BY customer_id
) o ON c.customer_id = o.customer_id
WHERE 
    c.email IS NULL OR c.email = '' OR
    c.email NOT LIKE '%@%' OR
    (c.email LIKE '%@%' AND SUBSTRING_INDEX(c.email, '@', -1) NOT LIKE '%.%') OR
    c.email LIKE '% %' OR
    LENGTH(c.email) < 5
ORDER BY 
    CASE 
        WHEN c.email IS NULL OR c.email = '' THEN 1
        WHEN c.email NOT LIKE '%@%' THEN 2
        WHEN c.email LIKE '%@%' AND SUBSTRING_INDEX(c.email, '@', -1) NOT LIKE '%.%' THEN 3
        WHEN c.email LIKE '% %' THEN 4
        ELSE 5
    END,
    COALESCE(o.order_count, 0) DESC,
    c.customer_id
LIMIT 100;

-- Summary Report
SELECT '========== EMAIL VALIDATION COMPLETE ==========' AS status;