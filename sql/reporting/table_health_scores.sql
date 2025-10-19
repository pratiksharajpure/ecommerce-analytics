-- ========================================
-- TABLE HEALTH SCORES - COMPLETE REPORT
-- Individual table quality assessment
-- Completeness & Accuracy Metrics
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CUSTOMERS TABLE HEALTH SCORE
-- ========================================
SELECT 
    'CUSTOMERS' AS table_name,
    COUNT(*) AS total_records,
    
    -- Completeness Metrics
    ROUND(100.0 * SUM(CASE WHEN first_name IS NOT NULL AND first_name != '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS first_name_completeness,
    ROUND(100.0 * SUM(CASE WHEN last_name IS NOT NULL AND last_name != '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS last_name_completeness,
    ROUND(100.0 * SUM(CASE WHEN email IS NOT NULL AND email != '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS email_completeness,
    ROUND(100.0 * SUM(CASE WHEN phone IS NOT NULL AND phone != '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS phone_completeness,
    ROUND(100.0 * SUM(CASE WHEN address_line1 IS NOT NULL AND address_line1 != '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS address_completeness,
    
    -- Accuracy Metrics
    ROUND(100.0 * SUM(CASE WHEN email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' THEN 1 ELSE 0 END) / COUNT(*), 2) AS email_accuracy,
    ROUND(100.0 * SUM(CASE WHEN phone REGEXP '^[0-9]{10,15}$' OR phone REGEXP '^\+?[0-9]{1,3}[-.\s]?[0-9]{10,15}$' THEN 1 ELSE 0 END) / COUNT(*), 2) AS phone_accuracy,
    
    -- Duplicate Detection
    COUNT(*) - COUNT(DISTINCT email) AS duplicate_emails,
    
    -- Overall Health Score
    ROUND((
        (100.0 * SUM(CASE WHEN first_name IS NOT NULL AND first_name != '' THEN 1 ELSE 0 END) / COUNT(*) * 0.10) +
        (100.0 * SUM(CASE WHEN last_name IS NOT NULL AND last_name != '' THEN 1 ELSE 0 END) / COUNT(*) * 0.10) +
        (100.0 * SUM(CASE WHEN email IS NOT NULL AND email != '' THEN 1 ELSE 0 END) / COUNT(*) * 0.20) +
        (100.0 * SUM(CASE WHEN email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
        (100.0 * SUM(CASE WHEN address_line1 IS NOT NULL AND address_line1 != '' THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) AS overall_health_score
FROM customers;

-- ========================================
-- 2. PRODUCTS TABLE HEALTH SCORE
-- ========================================
SELECT 
    'PRODUCTS' AS table_name,
    COUNT(*) AS total_records,
    
    -- Completeness Metrics
    ROUND(100.0 * SUM(CASE WHEN sku IS NOT NULL AND sku != '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS sku_completeness,
    ROUND(100.0 * SUM(CASE WHEN product_name IS NOT NULL AND product_name != '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS product_name_completeness,
    ROUND(100.0 * SUM(CASE WHEN description IS NOT NULL AND description != '' THEN 1 ELSE 0 END) / COUNT(*), 2) AS description_completeness,
    ROUND(100.0 * SUM(CASE WHEN price IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS price_completeness,
    
    -- Accuracy Metrics
    ROUND(100.0 * SUM(CASE WHEN price > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS price_accuracy,
    ROUND(100.0 * SUM(CASE WHEN cost >= 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS cost_accuracy,
    ROUND(100.0 * SUM(CASE WHEN price > cost THEN 1 ELSE 0 END) / COUNT(*), 2) AS margin_validity,
    
    -- Business Issues
    SUM(CASE WHEN price <= 0 OR price IS NULL THEN 1 ELSE 0 END) AS invalid_prices,
    SUM(CASE WHEN price <= cost THEN 1 ELSE 0 END) AS negative_margin_products,
    
    -- Overall Health Score
    ROUND((
        (100.0 * SUM(CASE WHEN sku IS NOT NULL AND sku != '' THEN 1 ELSE 0 END) / COUNT(*) * 0.20) +
        (100.0 * SUM(CASE WHEN product_name IS NOT NULL AND product_name != '' THEN 1 ELSE 0 END) / COUNT(*) * 0.15) +
        (100.0 * SUM(CASE WHEN price > 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.35) +
        (100.0 * SUM(CASE WHEN cost >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) AS overall_health_score
FROM products;

-- ========================================
-- 3. ORDERS TABLE HEALTH SCORE
-- ========================================
SELECT 
    'ORDERS' AS table_name,
    COUNT(*) AS total_records,
    
    -- Completeness Metrics
    ROUND(100.0 * SUM(CASE WHEN customer_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS customer_id_completeness,
    ROUND(100.0 * SUM(CASE WHEN total_amount IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS total_amount_completeness,
    
    -- Accuracy Metrics
    ROUND(100.0 * SUM(CASE WHEN total_amount >= 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS amount_accuracy,
    ROUND(100.0 * SUM(CASE WHEN order_date <= NOW() THEN 1 ELSE 0 END) / COUNT(*), 2) AS date_validity,
    
    -- Business Issues
    SUM(CASE WHEN total_amount < 0 THEN 1 ELSE 0 END) AS negative_amounts,
    SUM(CASE WHEN order_date > NOW() THEN 1 ELSE 0 END) AS future_dates,
    
    -- Overall Health Score
    ROUND((
        (100.0 * SUM(CASE WHEN customer_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
        (100.0 * SUM(CASE WHEN total_amount >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN order_date <= NOW() THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) AS overall_health_score
FROM orders;

-- ========================================
-- 4. CONSOLIDATED HEALTH SUMMARY
-- ========================================
SELECT 
    table_name,
    total_records,
    overall_health_score,
    CASE 
        WHEN overall_health_score >= 90 THEN 'Excellent'
        WHEN overall_health_score >= 75 THEN 'Good'
        WHEN overall_health_score >= 60 THEN 'Fair'
        ELSE 'Poor'
    END AS health_status
FROM (
    SELECT 'CUSTOMERS' AS table_name, COUNT(*) AS total_records,
        ROUND((
            (100.0 * SUM(CASE WHEN email IS NOT NULL AND email != '' THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
            (100.0 * SUM(CASE WHEN email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN address_line1 IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
        ), 2) AS overall_health_score
    FROM customers
    
    UNION ALL
    
    SELECT 'PRODUCTS', COUNT(*),
        ROUND((
            (100.0 * SUM(CASE WHEN sku IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.25) +
            (100.0 * SUM(CASE WHEN price > 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN cost >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.35)
        ), 2)
    FROM products
    
    UNION ALL
    
    SELECT 'ORDERS', COUNT(*),
        ROUND((
            (100.0 * SUM(CASE WHEN customer_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
            (100.0 * SUM(CASE WHEN total_amount >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN order_date <= NOW() THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
        ), 2)
    FROM orders
    
    UNION ALL
    
    SELECT 'INVENTORY', COUNT(*),
        ROUND((
            (100.0 * SUM(CASE WHEN product_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
            (100.0 * SUM(CASE WHEN quantity_on_hand >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN quantity_reserved <= quantity_on_hand THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
        ), 2)
    FROM inventory
    
    UNION ALL
    
    SELECT 'VENDORS', COUNT(*),
        ROUND((
            (100.0 * SUM(CASE WHEN vendor_name IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
            (100.0 * SUM(CASE WHEN email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN rating IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
        ), 2)
    FROM vendors
    
    UNION ALL
    
    SELECT 'CAMPAIGNS', COUNT(*),
        ROUND((
            (100.0 * SUM(CASE WHEN campaign_name IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
            (100.0 * SUM(CASE WHEN budget > 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN start_date <= end_date THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
        ), 2)
    FROM campaigns
    
    UNION ALL
    
    SELECT 'REVIEWS', COUNT(*),
        ROUND((
            (100.0 * SUM(CASE WHEN product_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
            (100.0 * SUM(CASE WHEN rating BETWEEN 1 AND 5 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN review_comment IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
        ), 2)
    FROM reviews
    
    UNION ALL
    
    SELECT 'RETURNS', COUNT(*),
        ROUND((
            (100.0 * SUM(CASE WHEN order_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN refund_amount >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
            (100.0 * SUM(CASE WHEN reason IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.20)
        ), 2)
    FROM returns
) AS health_summary
ORDER BY overall_health_score DESC;

-- ========================================
-- 5. CRITICAL ISSUES SUMMARY
-- ========================================
SELECT 
    'Critical Issues' AS report_section,
    issue_description,
    issue_count,
    severity,
    'Immediate action required' AS action_needed
FROM (
    SELECT 'Customers with invalid emails' AS issue_description,
        COUNT(*) AS issue_count, 'High' AS severity
    FROM customers
    WHERE email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' OR email IS NULL
    HAVING COUNT(*) > 0
    
    UNION ALL
    
    SELECT 'Products with invalid prices', COUNT(*), 'Critical'
    FROM products
    WHERE price <= 0 OR price IS NULL
    HAVING COUNT(*) > 0
    
    UNION ALL
    
    SELECT 'Products with negative margins', COUNT(*), 'High'
    FROM products
    WHERE price <= cost
    HAVING COUNT(*) > 0
    
    UNION ALL
    
    SELECT 'Orders with negative amounts', COUNT(*), 'Critical'
    FROM orders
    WHERE total_amount < 0
    HAVING COUNT(*) > 0
    
    UNION ALL
    
    SELECT 'Inventory with negative quantities', COUNT(*), 'Critical'
    FROM inventory
    WHERE quantity_on_hand < 0
    HAVING COUNT(*) > 0
    
    UNION ALL
    
    SELECT 'Inventory over-reserved', COUNT(*), 'High'
    FROM inventory
    WHERE quantity_reserved > quantity_on_hand
    HAVING COUNT(*) > 0
) AS issues
ORDER BY 
    CASE severity
        WHEN 'Critical' THEN 1
        WHEN 'High' THEN 2
        ELSE 3
    END,
    issue_count DESC;

-- ========================================
-- 6. EXECUTIVE SCORECARD
-- ========================================
SELECT 
    'OVERALL SYSTEM HEALTH' AS metric_category,
    ROUND(AVG(health_score), 2) AS average_health_score,
    ROUND(MIN(health_score), 2) AS lowest_table_score,
    ROUND(MAX(health_score), 2) AS highest_table_score,
    COUNT(*) AS tables_analyzed,
    SUM(CASE WHEN health_score >= 90 THEN 1 ELSE 0 END) AS excellent_tables,
    SUM(CASE WHEN health_score >= 75 AND health_score < 90 THEN 1 ELSE 0 END) AS good_tables,
    SUM(CASE WHEN health_score >= 60 AND health_score < 75 THEN 1 ELSE 0 END) AS fair_tables,
    SUM(CASE WHEN health_score < 60 THEN 1 ELSE 0 END) AS poor_tables
FROM (
    SELECT ROUND((
        (100.0 * SUM(CASE WHEN email IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
        (100.0 * SUM(CASE WHEN email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN address_line1 IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) AS health_score FROM customers
    UNION ALL
    SELECT ROUND((
        (100.0 * SUM(CASE WHEN sku IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.25) +
        (100.0 * SUM(CASE WHEN price > 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN cost >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.35)
    ), 2) FROM products
    UNION ALL
    SELECT ROUND((
        (100.0 * SUM(CASE WHEN customer_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
        (100.0 * SUM(CASE WHEN total_amount >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN order_date <= NOW() THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) FROM orders
    UNION ALL
    SELECT ROUND((
        (100.0 * SUM(CASE WHEN product_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
        (100.0 * SUM(CASE WHEN quantity_on_hand >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN quantity_reserved <= quantity_on_hand THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) FROM inventory
    UNION ALL
    SELECT ROUND((
        (100.0 * SUM(CASE WHEN vendor_name IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
        (100.0 * SUM(CASE WHEN email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN rating IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) FROM vendors
    UNION ALL
    SELECT ROUND((
        (100.0 * SUM(CASE WHEN campaign_name IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
        (100.0 * SUM(CASE WHEN budget > 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN start_date <= end_date THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) FROM campaigns
    UNION ALL
    SELECT ROUND((
        (100.0 * SUM(CASE WHEN product_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30) +
        (100.0 * SUM(CASE WHEN rating BETWEEN 1 AND 5 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN review_comment IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.30)
    ), 2) FROM reviews
    UNION ALL
    SELECT ROUND((
        (100.0 * SUM(CASE WHEN order_id IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN refund_amount >= 0 THEN 1 ELSE 0 END) / COUNT(*) * 0.40) +
        (100.0 * SUM(CASE WHEN reason IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) * 0.20)
    ), 2) FROM returns
) AS all_scores;

-- ========================================
-- REPORT COMPLETE
-- ========================================
SELECT 
    'Table Health Scores Analysis Complete' AS status,
    NOW() AS generated_at,
    '6 comprehensive sections executed' AS summary;