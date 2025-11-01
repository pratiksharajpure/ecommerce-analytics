-- ========================================
-- CLEANUP PROCEDURES
-- E-commerce Revenue Analytics Engine
-- Delete old records, Archive data, Clean temp tables
-- ========================================

USE ecommerce_analytics;

DELIMITER $$

-- ========================================
-- 1. ARCHIVE OLD ORDERS (Move to Archive Table)
-- ========================================
DROP PROCEDURE IF EXISTS sp_archive_old_orders$$
CREATE PROCEDURE sp_archive_old_orders(
    IN p_months_old INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_archived_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_months_old MONTH);
    
    -- Create archive table if not exists
    CREATE TABLE IF NOT EXISTS orders_archive LIKE orders;
    
    -- Insert old orders into archive
    INSERT INTO orders_archive
    SELECT * FROM orders
    WHERE order_date < v_cutoff_date
    AND status IN ('delivered', 'cancelled');
    
    SET v_archived_count = ROW_COUNT();
    
    -- Delete archived orders from main table
    DELETE FROM orders
    WHERE order_date < v_cutoff_date
    AND status IN ('delivered', 'cancelled');
    
    SELECT CONCAT('Archived ', v_archived_count, ' orders older than ', p_months_old, ' months') AS Result;
END$$

-- ========================================
-- 2. ARCHIVE OLD ORDER ITEMS
-- ========================================
DROP PROCEDURE IF EXISTS sp_archive_old_order_items$$
CREATE PROCEDURE sp_archive_old_order_items(
    IN p_months_old INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_archived_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_months_old MONTH);
    
    -- Create archive table if not exists
    CREATE TABLE IF NOT EXISTS order_items_archive LIKE order_items;
    
    -- Archive order items for archived orders
    INSERT INTO order_items_archive
    SELECT oi.* FROM order_items oi
    INNER JOIN orders_archive oa ON oi.order_id = oa.order_id
    WHERE oi.order_id NOT IN (SELECT order_item_id FROM order_items_archive);
    
    SET v_archived_count = ROW_COUNT();
    
    -- Delete archived order items
    DELETE oi FROM order_items oi
    INNER JOIN orders_archive oa ON oi.order_id = oa.order_id;
    
    SELECT CONCAT('Archived ', v_archived_count, ' order items') AS Result;
END$$

-- ========================================
-- 3. DELETE OLD CAMPAIGN PERFORMANCE DATA
-- ========================================
DROP PROCEDURE IF EXISTS sp_cleanup_old_campaign_data$$
CREATE PROCEDURE sp_cleanup_old_campaign_data(
    IN p_months_old INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_deleted_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_months_old MONTH);
    
    -- Create archive table if not exists
    CREATE TABLE IF NOT EXISTS campaign_performance_archive LIKE campaign_performance;
    
    -- Archive old performance data
    INSERT INTO campaign_performance_archive
    SELECT * FROM campaign_performance
    WHERE report_date < v_cutoff_date;
    
    SET v_deleted_count = ROW_COUNT();
    
    -- Delete old campaign performance data
    DELETE FROM campaign_performance
    WHERE report_date < v_cutoff_date;
    
    SELECT CONCAT('Archived and deleted ', v_deleted_count, ' campaign performance records older than ', p_months_old, ' months') AS Result;
END$$

-- ========================================
-- 4. DELETE ORPHANED RECORDS
-- ========================================
DROP PROCEDURE IF EXISTS sp_delete_orphaned_records$$
CREATE PROCEDURE sp_delete_orphaned_records()
BEGIN
DECLARE v_deleted_count INT DEFAULT 0;
    DECLARE v_total_deleted INT DEFAULT 0;
    
    -- Delete order items with no parent order
    DELETE FROM order_items
    WHERE order_id NOT IN (SELECT order_id FROM orders);
    

    












SET v_deleted_count = ROW_COUNT();
    SET v_total_deleted = v_total_deleted + v_deleted_count;
    
    -- Delete inventory records for deleted products
    DELETE FROM inventory
    WHERE product_id NOT IN (SELECT product_id FROM products);
    SET v_deleted_count = ROW_COUNT();
    SET v_total_deleted = v_total_deleted + v_deleted_count;
    
    -- Delete vendor contracts for deleted vendors or products
    DELETE FROM vendor_contracts
    WHERE vendor_id NOT IN (SELECT vendor_id FROM vendors)
    OR product_id NOT IN (SELECT product_id FROM products);
    SET v_deleted_count = ROW_COUNT();
    SET v_total_deleted = v_total_deleted + v_deleted_count;
    
    -- Delete shipping addresses for deleted customers
    DELETE FROM shipping_addresses
    WHERE customer_id NOT IN (SELECT customer_id FROM customers);
    SET v_deleted_count = ROW_COUNT();
    SET v_total_deleted = v_total_deleted + v_deleted_count;
    
    -- Delete payment methods for deleted customers
    DELETE FROM payment_methods
    WHERE customer_id NOT IN (SELECT customer_id FROM customers);
    SET v_deleted_count = ROW_COUNT();
    SET v_total_deleted = v_total_deleted + v_deleted_count;
    
    -- Delete reviews for deleted products
    DELETE FROM reviews
    WHERE product_id NOT IN (SELECT product_id FROM products);
    SET v_deleted_count = ROW_COUNT();
    SET v_total_deleted = v_total_deleted + v_deleted_count;
    
    SELECT CONCAT('Deleted ', v_total_deleted, ' orphaned records') AS Result;
END$$

-- ========================================
-- 5. PURGE REJECTED REVIEWS
-- ========================================
DROP PROCEDURE IF EXISTS sp_purge_rejected_reviews$$
CREATE PROCEDURE sp_purge_rejected_reviews(
    IN p_days_old INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_deleted_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_days_old DAY);
    
    DELETE FROM reviews
    WHERE status = 'rejected'
    AND created_at < v_cutoff_date;
    
    SET v_deleted_count = ROW_COUNT();
    
    SELECT CONCAT('Deleted ', v_deleted_count, ' rejected reviews older than ', p_days_old, ' days') AS Result;
END$$

-- ========================================
-- 6. CLEANUP INACTIVE CUSTOMERS
-- ========================================
DROP PROCEDURE IF EXISTS sp_cleanup_inactive_customers$$
CREATE PROCEDURE sp_cleanup_inactive_customers(
    IN p_years_inactive INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_updated_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_years_inactive YEAR);
    
    -- Mark customers as inactive if no orders in specified period
    UPDATE customers c
    SET c.status = 'inactive'
    WHERE c.customer_id NOT IN (
        SELECT DISTINCT customer_id 
        FROM orders 
        WHERE order_date >= v_cutoff_date
        AND customer_id IS NOT NULL
    )
    AND c.status = 'active';
    
    SET v_updated_count = ROW_COUNT();
    
    SELECT CONCAT('Marked ', v_updated_count, ' customers as inactive after ', p_years_inactive, ' years of inactivity') AS Result;
END$$

-- ========================================
-- 7. DELETE CANCELLED PENDING ORDERS
-- ========================================
DROP PROCEDURE IF EXISTS sp_delete_old_cancelled_orders$$
CREATE PROCEDURE sp_delete_old_cancelled_orders(
    IN p_days_old INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_deleted_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_days_old DAY);
    
    -- Delete cancelled orders older than specified days
    DELETE FROM orders
    WHERE status = 'cancelled'
    AND order_date < v_cutoff_date;
    
    SET v_deleted_count = ROW_COUNT();
    
    SELECT CONCAT('Deleted ', v_deleted_count, ' cancelled orders older than ', p_days_old, ' days') AS Result;
END$$

-- ========================================
-- 8. ARCHIVE COMPLETED RETURNS
-- ========================================
DROP PROCEDURE IF EXISTS sp_archive_completed_returns$$
CREATE PROCEDURE sp_archive_completed_returns(
    IN p_months_old INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_archived_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_months_old MONTH);
    
    -- Create archive table if not exists
    CREATE TABLE IF NOT EXISTS returns_archive LIKE returns;
    
    -- Archive completed returns
    INSERT INTO returns_archive
    SELECT * FROM returns
    WHERE updated_at < v_cutoff_date
    AND status IN ('refunded', 'rejected');
    
    SET v_archived_count = ROW_COUNT();
    
    -- Delete archived returns
    DELETE FROM returns
    WHERE updated_at < v_cutoff_date
    AND status IN ('refunded', 'rejected');
    
    SELECT CONCAT('Archived and deleted ', v_archived_count, ' completed returns older than ', p_months_old, ' months') AS Result;
END$$

-- ========================================
-- 9. CLEANUP EXPIRED VENDOR CONTRACTS
-- ========================================
DROP PROCEDURE IF EXISTS sp_cleanup_expired_contracts$$
CREATE PROCEDURE sp_cleanup_expired_contracts(
    IN p_years_old INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_deleted_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_years_old YEAR);
    
    -- Create archive table if not exists
    CREATE TABLE IF NOT EXISTS vendor_contracts_archive LIKE vendor_contracts;
    
    -- Archive expired contracts
    INSERT INTO vendor_contracts_archive
    SELECT * FROM vendor_contracts
    WHERE end_date < v_cutoff_date
    AND status = 'expired';
    
    SET v_deleted_count = ROW_COUNT();
    
    -- Delete archived contracts
    DELETE FROM vendor_contracts
    WHERE end_date < v_cutoff_date
    AND status = 'expired';
    
    SELECT CONCAT('Archived and deleted ', v_deleted_count, ' expired vendor contracts older than ', p_years_old, ' years') AS Result;
END$$

-- ========================================
-- 10. COMPREHENSIVE CLEANUP MASTER PROCEDURE
-- ========================================
DROP PROCEDURE IF EXISTS sp_master_cleanup$$
CREATE PROCEDURE sp_master_cleanup()
BEGIN
DECLARE v_start_time DATETIME;
    

    












SET v_start_time = NOW();
    
    SELECT 'Starting comprehensive cleanup process...' AS Status;
    
    -- Archive old orders (older than 24 months)
    CALL sp_archive_old_orders(24);
    
    -- Archive old order items
    CALL sp_archive_old_order_items(24);
    
    -- Cleanup campaign data (older than 12 months)
    CALL sp_cleanup_old_campaign_data(12);
    
    -- Delete orphaned records
    CALL sp_delete_orphaned_records();
    
    -- Purge rejected reviews (older than 90 days)
    CALL sp_purge_rejected_reviews(90);
    
    -- Cleanup inactive customers (inactive for 3 years)
    CALL sp_cleanup_inactive_customers(3);
    
    -- Delete old cancelled orders (older than 90 days)
    CALL sp_delete_old_cancelled_orders(90);
    
    -- Archive completed returns (older than 6 months)
    CALL sp_archive_completed_returns(6);
    
    -- Cleanup expired vendor contracts (older than 2 years)
    CALL sp_cleanup_expired_contracts(2);
    
    SELECT CONCAT('Cleanup completed in ', TIMESTAMPDIFF(SECOND, v_start_time, NOW()), ' seconds') AS Status;
END$$

-- ========================================
-- 11. OPTIMIZE ALL TABLES
-- ========================================
DROP PROCEDURE IF EXISTS sp_optimize_all_tables$$
CREATE PROCEDURE sp_optimize_all_tables()
BEGIN
    -- Optimize main tables
    OPTIMIZE TABLE customers;
    OPTIMIZE TABLE products;
    OPTIMIZE TABLE product_categories;
    OPTIMIZE TABLE orders;
    OPTIMIZE TABLE order_items;
    OPTIMIZE TABLE inventory;
    OPTIMIZE TABLE vendors;
    OPTIMIZE TABLE vendor_contracts;
    OPTIMIZE TABLE shipping_addresses;
    OPTIMIZE TABLE payment_methods;
    OPTIMIZE TABLE campaigns;
    OPTIMIZE TABLE campaign_performance;
    OPTIMIZE TABLE reviews;
    OPTIMIZE TABLE returns;
    OPTIMIZE TABLE loyalty_program;
    
    SELECT 'All tables optimized successfully' AS Status;
END$$

-- ========================================
-- 12. VACUUM STATISTICS UPDATE
-- ========================================
DROP PROCEDURE IF EXISTS sp_update_table_statistics$$
CREATE PROCEDURE sp_update_table_statistics()
BEGIN
    -- Analyze tables to update statistics
    ANALYZE TABLE customers;
    ANALYZE TABLE products;
    ANALYZE TABLE product_categories;
    ANALYZE TABLE orders;
    ANALYZE TABLE order_items;
    ANALYZE TABLE inventory;
    ANALYZE TABLE vendors;
    ANALYZE TABLE vendor_contracts;
    ANALYZE TABLE campaigns;
    ANALYZE TABLE campaign_performance;
    ANALYZE TABLE reviews;
    ANALYZE TABLE returns;
    
    SELECT 'Table statistics updated successfully' AS Status;
END$$

DELIMITER ;

-- ========================================
-- USAGE EXAMPLES
-- ========================================

-- Archive orders older than 24 months
-- CALL sp_archive_old_orders(24);

-- Delete orphaned records
-- CALL sp_delete_orphaned_records();

-- Run comprehensive cleanup
-- CALL sp_master_cleanup();

-- Optimize all tables
-- CALL sp_optimize_all_tables();

-- Update table statistics
-- CALL sp_update_table_statistics();

SELECT 'All cleanup procedures created successfully' AS Status;