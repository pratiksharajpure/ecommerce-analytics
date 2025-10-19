-- ========================================
-- CUSTOMER PROFILING & STATISTICS
-- Week 2, Day 1: Customer Analysis Queries
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. OVERALL CUSTOMER STATISTICS
-- ========================================
SELECT 
    'Customer Statistics' AS metric_category,
    COUNT(*) AS total_customers,
    COUNT(CASE WHEN status = 'active' THEN 1 END) AS active_customers,
    COUNT(CASE WHEN status = 'inactive' THEN 1 END) AS inactive_customers,
    COUNT(CASE WHEN status = 'suspended' THEN 1 END) AS suspended_customers,
    ROUND(COUNT(CASE WHEN status = 'active' THEN 1 END) * 100.0 / COUNT(*), 2) AS active_percentage,
    COUNT(DISTINCT email) AS unique_emails,
    COUNT(DISTINCT phone) AS unique_phones
FROM customers;

-- ========================================
-- 2. CUSTOMER REGISTRATION TRENDS
-- ========================================
SELECT 
    DATE_FORMAT(created_at, '%Y-%m') AS registration_month,
    COUNT(*) AS new_customers,
    SUM(COUNT(*)) OVER (ORDER BY DATE_FORMAT(created_at, '%Y-%m')) AS cumulative_customers
FROM customers
GROUP BY DATE_FORMAT(created_at, '%Y-%m')
ORDER BY registration_month DESC
LIMIT 12;

-- ========================================
-- 3. GEOGRAPHIC DISTRIBUTION
-- ========================================
SELECT 
    country,
    state,
    city,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers
WHERE city IS NOT NULL
GROUP BY country, state, city
ORDER BY customer_count DESC
LIMIT 20;

-- ========================================
-- 4. STATE-LEVEL DISTRIBUTION (USA)
-- ========================================
SELECT 
    state,
    COUNT(*) AS customer_count,
    COUNT(CASE WHEN status = 'active' THEN 1 END) AS active_customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers WHERE country = 'USA'), 2) AS percentage
FROM customers
WHERE country = 'USA' AND state IS NOT NULL
GROUP BY state
ORDER BY customer_count DESC;

-- ========================================
-- 5. CUSTOMER LIFECYCLE ANALYSIS
-- ========================================
SELECT 
    CASE 
        WHEN DATEDIFF(CURDATE(), created_at) <= 30 THEN '0-30 days'
        WHEN DATEDIFF(CURDATE(), created_at) <= 90 THEN '31-90 days'
        WHEN DATEDIFF(CURDATE(), created_at) <= 180 THEN '91-180 days'
        WHEN DATEDIFF(CURDATE(), created_at) <= 365 THEN '181-365 days'
        ELSE '365+ days'
    END AS customer_age,
    COUNT(*) AS customer_count,
    ROUND(AVG(DATEDIFF(CURDATE(), created_at)), 0) AS avg_days_since_registration
FROM customers
GROUP BY customer_age
ORDER BY MIN(DATEDIFF(CURDATE(), created_at));

-- ========================================
-- 6. CUSTOMER DATA COMPLETENESS PROFILE
-- ========================================
SELECT 
    'Data Completeness' AS metric,
    COUNT(*) AS total_customers,
    COUNT(CASE WHEN email IS NOT NULL AND email != '' THEN 1 END) AS has_email,
    COUNT(CASE WHEN phone IS NOT NULL AND phone != '' THEN 1 END) AS has_phone,
    COUNT(CASE WHEN address_line1 IS NOT NULL AND address_line1 != '' THEN 1 END) AS has_address,
    COUNT(CASE WHEN city IS NOT NULL AND city != '' THEN 1 END) AS has_city,
    COUNT(CASE WHEN state IS NOT NULL AND state != '' THEN 1 END) AS has_state,
    COUNT(CASE WHEN zip_code IS NOT NULL AND zip_code != '' THEN 1 END) AS has_zipcode,
    ROUND(COUNT(CASE WHEN 
        email IS NOT NULL AND 
        phone IS NOT NULL AND 
        address_line1 IS NOT NULL AND 
        city IS NOT NULL AND 
        state IS NOT NULL AND 
        zip_code IS NOT NULL 
    THEN 1 END) * 100.0 / COUNT(*), 2) AS complete_profile_percentage
FROM customers;

-- ========================================
-- 7. CUSTOMER SEGMENTATION BY DATA QUALITY
-- ========================================
SELECT 
    CASE 
        WHEN field_count >= 6 THEN 'Complete Profile'
        WHEN field_count >= 4 THEN 'Good Profile'
        WHEN field_count >= 2 THEN 'Partial Profile'
        ELSE 'Minimal Profile'
    END AS profile_quality,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM (
    SELECT 
        customer_id,
        (CASE WHEN email IS NOT NULL AND email != '' THEN 1 ELSE 0 END +
         CASE WHEN phone IS NOT NULL AND phone != '' THEN 1 ELSE 0 END +
         CASE WHEN address_line1 IS NOT NULL AND address_line1 != '' THEN 1 ELSE 0 END +
         CASE WHEN address_line2 IS NOT NULL AND address_line2 != '' THEN 1 ELSE 0 END +
         CASE WHEN city IS NOT NULL AND city != '' THEN 1 ELSE 0 END +
         CASE WHEN state IS NOT NULL AND state != '' THEN 1 ELSE 0 END +
         CASE WHEN zip_code IS NOT NULL AND zip_code != '' THEN 1 ELSE 0 END) AS field_count
    FROM customers
) AS profile_scores
GROUP BY profile_quality
ORDER BY MIN(field_count) DESC;

-- ========================================
-- 8. CUSTOMERS WITH ORDERS VS WITHOUT
-- ========================================
SELECT 
    CASE WHEN o.customer_id IS NOT NULL THEN 'Has Orders' ELSE 'No Orders' END AS customer_type,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    ROUND(COUNT(DISTINCT c.customer_id) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY customer_type;

-- ========================================
-- 9. CUSTOMER STATUS CHANGE ANALYSIS
-- ========================================
SELECT 
    status,
    COUNT(*) AS customer_count,
    ROUND(AVG(DATEDIFF(updated_at, created_at)), 0) AS avg_days_since_creation,
    MIN(updated_at) AS earliest_update,
    MAX(updated_at) AS latest_update
FROM customers
GROUP BY status
ORDER BY customer_count DESC;

-- ========================================
-- 10. TOP CITIES BY CUSTOMER CONCENTRATION
-- ========================================
SELECT 
    city,
    state,
    country,
    COUNT(*) AS customer_count,
    COUNT(CASE WHEN status = 'active' THEN 1 END) AS active_customers,
    ROUND(COUNT(CASE WHEN status = 'active' THEN 1 END) * 100.0 / COUNT(*), 2) AS active_percentage
FROM customers
WHERE city IS NOT NULL
GROUP BY city, state, country
HAVING customer_count >= 5
ORDER BY customer_count DESC
LIMIT 25;

-- ========================================
-- 11. CUSTOMER NAME ANALYSIS
-- ========================================
SELECT 
    'Name Analysis' AS metric,
    COUNT(*) AS total_customers,
    COUNT(CASE WHEN first_name IS NOT NULL AND first_name != '' THEN 1 END) AS has_first_name,
    COUNT(CASE WHEN last_name IS NOT NULL AND last_name != '' THEN 1 END) AS has_last_name,
    COUNT(CASE WHEN 
        first_name IS NOT NULL AND first_name != '' AND 
        last_name IS NOT NULL AND last_name != '' 
    THEN 1 END) AS has_full_name,
    COUNT(CASE WHEN first_name IS NULL OR first_name = '' THEN 1 END) AS missing_first_name,
    COUNT(CASE WHEN last_name IS NULL OR last_name = '' THEN 1 END) AS missing_last_name
FROM customers;

-- ========================================
-- 12. RECENT CUSTOMER ACTIVITY
-- ========================================
SELECT 
    CASE 
        WHEN DATEDIFF(CURDATE(), updated_at) <= 7 THEN 'Last 7 days'
        WHEN DATEDIFF(CURDATE(), updated_at) <= 30 THEN 'Last 30 days'
        WHEN DATEDIFF(CURDATE(), updated_at) <= 90 THEN 'Last 90 days'
        WHEN DATEDIFF(CURDATE(), updated_at) <= 180 THEN 'Last 180 days'
        ELSE 'Over 180 days ago'
    END AS last_activity,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage
FROM customers
GROUP BY last_activity
ORDER BY MIN(DATEDIFF(CURDATE(), updated_at));

-- Summary Report
SELECT '========== CUSTOMER PROFILING COMPLETE ==========' AS status;