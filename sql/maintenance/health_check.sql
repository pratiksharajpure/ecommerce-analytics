-- ========================================
-- HEALTH CHECK & DIAGNOSTICS SCRIPT
-- E-commerce Revenue Analytics Engine
-- System Health, Integrity & Performance Checks
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- SECTION 1: DATABASE CONNECTION & STATUS
-- ========================================
SELECT '========== DATABASE CONNECTION STATUS ==========' AS '';

SELECT 
    'Database Connection' AS Check_Type,
    DATABASE() AS Current_Database,
    VERSION() AS MySQL_Version,
    NOW() AS Current_Timestamp,
    'PASS' AS Status;

-- ========================================
-- SECTION 2: TABLE EXISTENCE CHECK
-- ========================================
SELECT '========== TABLE EXISTENCE CHECK ==========' AS '';

SELECT 
    'Table Count' AS Check_Type,
    COUNT(*) AS Total_Tables,
    CASE 
        WHEN COUNT(*) >= 15 THEN 'PASS'
        ELSE 'FAIL'
    END AS Status
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'ecommerce_analytics';

-- Detailed table list
SELECT 
    TABLE_NAME AS Table_Name,
    TABLE_ROWS AS Estimated_Rows,
    ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) AS Size_MB,
    ENGINE AS Storage_Engine,
    TABLE_COLLATION AS Collation
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
ORDER BY TABLE_NAME;

-- ========================================
-- SECTION 3: DATA INTEGRITY CHECKS
-- ========================================
SELECT '========== DATA INTEGRITY CHECKS ==========' AS '';

-- Check for orphaned order items (orders that don't exist)
SELECT 
    'Orphaned Order Items' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;

-- Check for orders without items
SELECT 
    'Orders Without Items' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE oi.order_item_id IS NULL;

-- Check for products without categories
SELECT 
    'Products Without Categories' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'INFO'
    END AS Status
FROM products
WHERE category_id IS NULL;

-- Check for negative inventory
SELECT 
    'Negative Inventory' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS Status
FROM inventory
WHERE quantity_on_hand < 0 OR quantity_reserved < 0;

-- Check for invalid email formats in customers
SELECT 
    'Invalid Customer Emails' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM customers
WHERE email NOT LIKE '%_@__%.__%' OR email IS NULL;

-- Check for orders with mismatched totals
SELECT 
    'Order Total Mismatches' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM (
    SELECT 
        o.order_id,
        o.total_amount,
        COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount AS calculated_total
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY o.order_id, o.total_amount, o.shipping_cost, o.tax_amount
    HAVING ABS(o.total_amount - calculated_total) > 0.01
) AS mismatches;

-- ========================================
-- SECTION 4: REFERENTIAL INTEGRITY
-- ========================================
SELECT '========== REFERENTIAL INTEGRITY ==========' AS '';

-- Check foreign key constraints
SELECT 
    'Foreign Key Constraints' AS Check_Type,
    COUNT(*) AS Total_Constraints,
    'PASS' AS Status
FROM information_schema.TABLE_CONSTRAINTS
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
AND CONSTRAINT_TYPE = 'FOREIGN KEY';

-- List all foreign key relationships
SELECT 
    TABLE_NAME AS Child_Table,
    COLUMN_NAME AS Foreign_Key_Column,
    REFERENCED_TABLE_NAME AS Parent_Table,
    REFERENCED_COLUMN_NAME AS Referenced_Column
FROM information_schema.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
AND REFERENCED_TABLE_NAME IS NOT NULL
ORDER BY TABLE_NAME, COLUMN_NAME;

-- ========================================
-- SECTION 5: INDEX HEALTH CHECK
-- ========================================
SELECT '========== INDEX HEALTH CHECK ==========' AS '';

-- Count indexes per table
SELECT 
    TABLE_NAME AS Table_Name,
    COUNT(*) AS Index_Count,
    CASE 
        WHEN COUNT(*) > 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
GROUP BY TABLE_NAME
ORDER BY Index_Count DESC;

-- Check for missing primary keys
SELECT 
    'Tables Without Primary Keys' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS Status
FROM information_schema.TABLES t
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
AND NOT EXISTS (
    SELECT 1 
    FROM information_schema.TABLE_CONSTRAINTS tc
    WHERE tc.TABLE_SCHEMA = t.TABLE_SCHEMA
    AND tc.TABLE_NAME = t.TABLE_NAME
    AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
);

-- ========================================
-- SECTION 6: PERFORMANCE METRICS
-- ========================================
SELECT '========== PERFORMANCE METRICS ==========' AS '';

-- Table size and row count analysis
SELECT 
    TABLE_NAME AS Table_Name,
    TABLE_ROWS AS Row_Count,
    ROUND((DATA_LENGTH) / 1024 / 1024, 2) AS Data_Size_MB,
    ROUND((INDEX_LENGTH) / 1024 / 1024, 2) AS Index_Size_MB,
    ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) AS Total_Size_MB,
    CASE 
        WHEN TABLE_ROWS > 1000000 THEN 'LARGE'
        WHEN TABLE_ROWS > 100000 THEN 'MEDIUM'
        ELSE 'SMALL'
    END AS Table_Size_Category
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC;

-- Total database size
SELECT 
    'Total Database Size' AS Metric,
    ROUND(SUM(DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) AS Size_MB,
    ROUND(SUM(DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024 / 1024, 2) AS Size_GB
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'ecommerce_analytics';

-- ========================================
-- SECTION 7: DATA QUALITY CHECKS
-- ========================================
SELECT '========== DATA QUALITY CHECKS ==========' AS '';

-- Check for duplicate SKUs in products
SELECT 
    'Duplicate Product SKUs' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS Status
FROM (
    SELECT sku
    FROM products
    WHERE sku IS NOT NULL
    GROUP BY sku
    HAVING COUNT(*) > 1
) AS duplicates;

-- Check for duplicate customer emails
SELECT 
    'Duplicate Customer Emails' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM (
    SELECT email
    FROM customers
    WHERE email IS NOT NULL
    GROUP BY email
    HAVING COUNT(*) > 1
) AS duplicates;

-- Check for products with price = 0 or NULL
SELECT 
    'Products With Zero/NULL Price' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM products
WHERE price IS NULL OR price = 0;

-- Check for orders with invalid dates
SELECT 
    'Orders With Future Dates' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM orders
WHERE order_date > NOW();

-- Check for reviews with invalid ratings
SELECT 
    'Reviews With Invalid Ratings' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'FAIL'
    END AS Status
FROM reviews
WHERE rating NOT BETWEEN 1 AND 5;

-- ========================================
-- SECTION 8: BUSINESS LOGIC CHECKS
-- ========================================
SELECT '========== BUSINESS LOGIC CHECKS ==========' AS '';

-- Check for customers with no orders
SELECT 
    'Customers Without Orders' AS Check_Type,
    COUNT(*) AS Count,
    'INFO' AS Status
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;

-- Check for products never ordered
SELECT 
    'Products Never Ordered' AS Check_Type,
    COUNT(*) AS Count,
    'INFO' AS Status
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.order_item_id IS NULL;

-- Check for expired vendor contracts still marked as active
SELECT 
    'Expired Active Contracts' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) = 0 THEN 'PASS'
        ELSE 'WARN'
    END AS Status
FROM vendor_contracts
WHERE status = 'active' AND end_date < CURDATE();

-- Check inventory below reorder level
SELECT 
    'Products Below Reorder Level' AS Check_Type,
    COUNT(*) AS Count,
    'INFO' AS Status
FROM inventory
WHERE quantity_available < reorder_level;

-- ========================================
-- SECTION 9: RECENT ACTIVITY CHECK
-- ========================================
SELECT '========== RECENT ACTIVITY CHECK ==========' AS '';

-- Check for recent orders (last 24 hours)
SELECT 
    'Orders (Last 24h)' AS Activity_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) > 0 THEN 'ACTIVE'
        ELSE 'IDLE'
    END AS Status
FROM orders
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR);

-- Check for recent customers (last 7 days)
SELECT 
    'New Customers (Last 7 days)' AS Activity_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN COUNT(*) > 0 THEN 'ACTIVE'
        ELSE 'IDLE'
    END AS Status
FROM customers
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY);

-- Check for recent reviews (last 30 days)
SELECT 
    'New Reviews (Last 30 days)' AS Activity_Type,
    COUNT(*) AS Count,
    'INFO' AS Status
FROM reviews
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY);

-- ========================================
-- SECTION 10: SYSTEM RECOMMENDATIONS
-- ========================================
SELECT '========== SYSTEM RECOMMENDATIONS ==========' AS '';

-- Tables that might need optimization
SELECT 
    TABLE_NAME AS Table_Name,
    'Consider OPTIMIZE TABLE' AS Recommendation,
    CONCAT('Table has ', TABLE_ROWS, ' rows') AS Reason
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
AND TABLE_ROWS > 100000
ORDER BY TABLE_ROWS DESC
LIMIT 5;

-- ========================================
-- SECTION 11: OVERALL HEALTH SUMMARY
-- ========================================
SELECT '========== OVERALL HEALTH SUMMARY ==========' AS '';

SELECT 
    'Database Health Check' AS Summary,
    CASE 
        WHEN (SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = 'ecommerce_analytics') >= 15
        AND (SELECT COUNT(*) FROM order_items oi LEFT JOIN orders o ON oi.order_id = o.order_id WHERE o.order_id IS NULL) = 0
        AND (SELECT COUNT(*) FROM inventory WHERE quantity_on_hand < 0) = 0
        THEN 'HEALTHY'
        ELSE 'NEEDS ATTENTION'
    END AS Status,
    NOW() AS Check_Completed_At;

-- Display completion message
SELECT '========================================' AS '';
SELECT 'Health Check Completed Successfully' AS Result;
SELECT 'Review all sections above for detailed diagnostics' AS Note;
SELECT '========================================' AS '';