-- ========================================
-- BACKUP SCRIPTS
-- E-commerce Revenue Analytics Engine
-- Full, Incremental, and Table-Level Backups
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- BACKUP TRACKING TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS backup_log (
    backup_id INT PRIMARY KEY AUTO_INCREMENT,
    backup_type ENUM('full', 'incremental', 'table', 'differential') NOT NULL,
    backup_name VARCHAR(200) NOT NULL,
    backup_path VARCHAR(500),
    tables_backed_up TEXT,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    duration_seconds INT,
    status ENUM('in_progress', 'completed', 'failed') DEFAULT 'in_progress',
    backup_size_mb DECIMAL(10,2),
    row_count INT,
    error_message TEXT,
    created_by VARCHAR(100),
    INDEX idx_backup_type (backup_type),
    INDEX idx_status (status),
    INDEX idx_start_time (start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- INCREMENTAL BACKUP TRACKING TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS backup_incremental_tracker (
    tracker_id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100) NOT NULL,
    last_backup_time DATETIME NOT NULL,
    last_backup_id INT,
    records_backed_up INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_table_name (table_name),
    FOREIGN KEY (last_backup_id) REFERENCES backup_log(backup_id) ON DELETE SET NULL,
    INDEX idx_last_backup_time (last_backup_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

DELIMITER $$

-- ========================================
-- 1. FULL DATABASE BACKUP PROCEDURE
-- ========================================
DROP PROCEDURE IF EXISTS sp_full_database_backup$$
CREATE PROCEDURE sp_full_database_backup(
    IN p_backup_path VARCHAR(500),
    IN p_created_by VARCHAR(100)
)
BEGIN
DECLARE v_backup_id INT;
    DECLARE v_start_time DATETIME;
    DECLARE v_backup_name VARCHAR(200);
    DECLARE v_total_rows INT DEFAULT 0;
    
    

    












SET v_start_time = NOW();
    SET v_backup_name = CONCAT('full_backup_', DATE_FORMAT(v_start_time, '%Y%m%d_%H%i%s'));
    
    -- Log backup start
    INSERT INTO backup_log (backup_type, backup_name, backup_path, start_time, status, created_by)
    VALUES ('full', v_backup_name, p_backup_path, v_start_time, 'in_progress', p_created_by);
    
    SET v_backup_id = LAST_INSERT_ID();
    
    -- Count total rows across all tables
    SELECT 
        (SELECT COUNT(*) FROM customers) +
        (SELECT COUNT(*) FROM products) +
        (SELECT COUNT(*) FROM product_categories) +
        (SELECT COUNT(*) FROM orders) +
        (SELECT COUNT(*) FROM order_items) +
        (SELECT COUNT(*) FROM inventory) +
        (SELECT COUNT(*) FROM vendors) +
        (SELECT COUNT(*) FROM vendor_contracts) +
        (SELECT COUNT(*) FROM shipping_addresses) +
        (SELECT COUNT(*) FROM payment_methods) +
        (SELECT COUNT(*) FROM campaigns) +
        (SELECT COUNT(*) FROM campaign_performance) +
        (SELECT COUNT(*) FROM reviews) +
        (SELECT COUNT(*) FROM returns) +
        (SELECT COUNT(*) FROM loyalty_program)
    INTO v_total_rows;
    
    -- Update backup log with completion
    UPDATE backup_log
    SET end_time = NOW(),
        duration_seconds = TIMESTAMPDIFF(SECOND, start_time, NOW()),
        status = 'completed',
        row_count = v_total_rows,
        tables_backed_up = 'ALL_TABLES'
    WHERE backup_id = v_backup_id;
    
    -- Update incremental tracker for all tables
    INSERT INTO backup_incremental_tracker (table_name, last_backup_time, last_backup_id)
    VALUES 
        ('customers', v_start_time, v_backup_id),
        ('products', v_start_time, v_backup_id),
        ('orders', v_start_time, v_backup_id),
        ('order_items', v_start_time, v_backup_id),
        ('inventory', v_start_time, v_backup_id),
        ('campaigns', v_start_time, v_backup_id),
        ('returns', v_start_time, v_backup_id)
    ON DUPLICATE KEY UPDATE 
        last_backup_time = v_start_time,
        last_backup_id = v_backup_id,
        records_backed_up = 0;
    
    SELECT 
        v_backup_id AS backup_id,
        v_backup_name AS backup_name,
        v_total_rows AS total_rows_backed_up,
        TIMESTAMPDIFF(SECOND, v_start_time, NOW()) AS duration_seconds,
        'Full backup completed successfully' AS status;
END$$

-- ========================================
-- 2. INCREMENTAL BACKUP - ORDERS
-- ========================================
DROP PROCEDURE IF EXISTS sp_incremental_backup_orders$$
CREATE PROCEDURE sp_incremental_backup_orders(
    IN p_backup_path VARCHAR(500),
    IN p_created_by VARCHAR(100)
)
BEGIN
DECLARE v_backup_id INT;
    DECLARE v_start_time DATETIME;
    DECLARE v_last_backup_time DATETIME;
    DECLARE v_backup_name VARCHAR(200);
    DECLARE v_row_count INT DEFAULT 0;
    
    

    












SET v_start_time = NOW();
    SET v_backup_name = CONCAT('incremental_orders_', DATE_FORMAT(v_start_time, '%Y%m%d_%H%i%s'));
    
    -- Get last backup time
    SELECT COALESCE(last_backup_time, '1900-01-01') INTO v_last_backup_time
    FROM backup_incremental_tracker
    WHERE table_name = 'orders';
    
    -- Log backup start
    INSERT INTO backup_log (backup_type, backup_name, backup_path, start_time, status, created_by, tables_backed_up)
    VALUES ('incremental', v_backup_name, p_backup_path, v_start_time, 'in_progress', p_created_by, 'orders');
    
    SET v_backup_id = LAST_INSERT_ID();
    
    -- Create temporary backup table
    DROP TEMPORARY TABLE IF EXISTS tmp_orders_backup;
    CREATE TEMPORARY TABLE tmp_orders_backup AS
    SELECT * FROM orders
    WHERE created_at > v_last_backup_time OR updated_at > v_last_backup_time;
    
    SELECT COUNT(*) INTO v_row_count FROM tmp_orders_backup;
    
    -- Update backup log
    UPDATE backup_log
    SET end_time = NOW(),
        duration_seconds = TIMESTAMPDIFF(SECOND, start_time, NOW()),
        status = 'completed',
        row_count = v_row_count
    WHERE backup_id = v_backup_id;
    
    -- Update tracker
    INSERT INTO backup_incremental_tracker (table_name, last_backup_time, last_backup_id, records_backed_up)
    VALUES ('orders', v_start_time, v_backup_id, v_row_count)
    ON DUPLICATE KEY UPDATE 
        last_backup_time = v_start_time,
        last_backup_id = v_backup_id,
        records_backed_up = v_row_count;
    
    SELECT 
        v_backup_id AS backup_id,
        v_backup_name AS backup_name,
        v_row_count AS rows_backed_up,
        v_last_backup_time AS since_last_backup,
        'Incremental backup completed' AS status;
    
    DROP TEMPORARY TABLE IF EXISTS tmp_orders_backup;
END$$

-- ========================================
-- 3. INCREMENTAL BACKUP - CUSTOMERS
-- ========================================
DROP PROCEDURE IF EXISTS sp_incremental_backup_customers$$
CREATE PROCEDURE sp_incremental_backup_customers(
    IN p_backup_path VARCHAR(500),
    IN p_created_by VARCHAR(100)
)
BEGIN
DECLARE v_backup_id INT;
    DECLARE v_start_time DATETIME;
    DECLARE v_last_backup_time DATETIME;
    DECLARE v_backup_name VARCHAR(200);
    DECLARE v_row_count INT DEFAULT 0;
    
    

    












SET v_start_time = NOW();
    SET v_backup_name = CONCAT('incremental_customers_', DATE_FORMAT(v_start_time, '%Y%m%d_%H%i%s'));
    
    SELECT COALESCE(last_backup_time, '1900-01-01') INTO v_last_backup_time
    FROM backup_incremental_tracker
    WHERE table_name = 'customers';
    
    INSERT INTO backup_log (backup_type, backup_name, backup_path, start_time, status, created_by, tables_backed_up)
    VALUES ('incremental', v_backup_name, p_backup_path, v_start_time, 'in_progress', p_created_by, 'customers');
    
    SET v_backup_id = LAST_INSERT_ID();
    
    DROP TEMPORARY TABLE IF EXISTS tmp_customers_backup;
    CREATE TEMPORARY TABLE tmp_customers_backup AS
    SELECT * FROM customers
    WHERE created_at > v_last_backup_time OR updated_at > v_last_backup_time;
    
    SELECT COUNT(*) INTO v_row_count FROM tmp_customers_backup;
    
    UPDATE backup_log
    SET end_time = NOW(),
        duration_seconds = TIMESTAMPDIFF(SECOND, start_time, NOW()),
        status = 'completed',
        row_count = v_row_count
    WHERE backup_id = v_backup_id;
    
    INSERT INTO backup_incremental_tracker (table_name, last_backup_time, last_backup_id, records_backed_up)
    VALUES ('customers', v_start_time, v_backup_id, v_row_count)
    ON DUPLICATE KEY UPDATE 
        last_backup_time = v_start_time,
        last_backup_id = v_backup_id,
        records_backed_up = v_row_count;
    
    SELECT 
        v_backup_id AS backup_id,
        v_backup_name AS backup_name,
        v_row_count AS rows_backed_up,
        'Incremental backup completed' AS status;
    
    DROP TEMPORARY TABLE IF EXISTS tmp_customers_backup;
END$$

-- ========================================
-- 4. TABLE-LEVEL BACKUP - SPECIFIC TABLE
-- ========================================
DROP PROCEDURE IF EXISTS sp_table_backup$$
CREATE PROCEDURE sp_table_backup(
    IN p_table_name VARCHAR(100),
    IN p_backup_path VARCHAR(500),
    IN p_created_by VARCHAR(100)
)
BEGIN
DECLARE v_backup_id INT;
    DECLARE v_start_time DATETIME;
    DECLARE v_backup_name VARCHAR(200);
    DECLARE v_row_count INT DEFAULT 0;
    DECLARE v_sql_query TEXT;
    
    

    












SET v_start_time = NOW();
    SET v_backup_name = CONCAT('table_', p_table_name, '_', DATE_FORMAT(v_start_time, '%Y%m%d_%H%i%s'));
    
    INSERT INTO backup_log (backup_type, backup_name, backup_path, start_time, status, created_by, tables_backed_up)
    VALUES ('table', v_backup_name, p_backup_path, v_start_time, 'in_progress', p_created_by, p_table_name);
    
    SET v_backup_id = LAST_INSERT_ID();
    
    -- Get row count
    SET @count_sql = CONCAT('SELECT COUNT(*) INTO @v_count FROM ', p_table_name);
    PREPARE stmt FROM @count_sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    SET v_row_count = @v_count;
    
    -- Create backup table
    SET @backup_sql = CONCAT('CREATE TABLE IF NOT EXISTS ', p_table_name, '_backup_', 
                             DATE_FORMAT(v_start_time, '%Y%m%d_%H%i%s'), 
                             ' AS SELECT * FROM ', p_table_name);
    PREPARE stmt FROM @backup_sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    UPDATE backup_log
    SET end_time = NOW(),
        duration_seconds = TIMESTAMPDIFF(SECOND, start_time, NOW()),
        status = 'completed',
        row_count = v_row_count
    WHERE backup_id = v_backup_id;
    
    SELECT 
        v_backup_id AS backup_id,
        v_backup_name AS backup_name,
        v_row_count AS rows_backed_up,
        CONCAT(p_table_name, '_backup_', DATE_FORMAT(v_start_time, '%Y%m%d_%H%i%s')) AS backup_table_name,
        'Table backup completed' AS status;
END$$

-- ========================================
-- 5. BACKUP CRITICAL TABLES
-- ========================================
DROP PROCEDURE IF EXISTS sp_backup_critical_tables$$
CREATE PROCEDURE sp_backup_critical_tables(
    IN p_backup_path VARCHAR(500),
    IN p_created_by VARCHAR(100)
)
BEGIN
DECLARE v_backup_id INT;
    DECLARE v_start_time DATETIME;
    DECLARE v_backup_name VARCHAR(200);
    DECLARE v_total_rows INT DEFAULT 0;
    
    

    












SET v_start_time = NOW();
    SET v_backup_name = CONCAT('critical_tables_', DATE_FORMAT(v_start_time, '%Y%m%d_%H%i%s'));
    
    INSERT INTO backup_log (backup_type, backup_name, backup_path, start_time, status, created_by)
    VALUES ('table', v_backup_name, p_backup_path, v_start_time, 'in_progress', p_created_by);
    
    SET v_backup_id = LAST_INSERT_ID();
    
    -- Backup customers
    DROP TABLE IF EXISTS customers_backup_latest;
    CREATE TABLE customers_backup_latest AS SELECT * FROM customers;
    
    -- Backup orders
    DROP TABLE IF EXISTS orders_backup_latest;
    CREATE TABLE orders_backup_latest AS SELECT * FROM orders;
    
    -- Backup products
    DROP TABLE IF EXISTS products_backup_latest;
    CREATE TABLE products_backup_latest AS SELECT * FROM products;
    
    -- Backup order_items
    DROP TABLE IF EXISTS order_items_backup_latest;
    CREATE TABLE order_items_backup_latest AS SELECT * FROM order_items;
    
    -- Count total rows
    SELECT 
        (SELECT COUNT(*) FROM customers_backup_latest) +
        (SELECT COUNT(*) FROM orders_backup_latest) +
        (SELECT COUNT(*) FROM products_backup_latest) +
        (SELECT COUNT(*) FROM order_items_backup_latest)
    INTO v_total_rows;
    
    UPDATE backup_log
    SET end_time = NOW(),
        duration_seconds = TIMESTAMPDIFF(SECOND, start_time, NOW()),
        status = 'completed',
        row_count = v_total_rows,
        tables_backed_up = 'customers,orders,products,order_items'
    WHERE backup_id = v_backup_id;
    
    SELECT 
        v_backup_id AS backup_id,
        v_backup_name AS backup_name,
        v_total_rows AS total_rows_backed_up,
        'Critical tables backed up successfully' AS status;
END$$

-- ========================================
-- 6. DIFFERENTIAL BACKUP (Since Last Full)
-- ========================================
DROP PROCEDURE IF EXISTS sp_differential_backup$$
CREATE PROCEDURE sp_differential_backup(
    IN p_backup_path VARCHAR(500),
    IN p_created_by VARCHAR(100)
)
BEGIN
DECLARE v_backup_id INT;
    DECLARE v_start_time DATETIME;
    DECLARE v_last_full_backup_time DATETIME;
    DECLARE v_backup_name VARCHAR(200);
    DECLARE v_row_count INT DEFAULT 0;
    
    

    












SET v_start_time = NOW();
    SET v_backup_name = CONCAT('differential_', DATE_FORMAT(v_start_time, '%Y%m%d_%H%i%s'));
    
    -- Get last full backup time
    SELECT COALESCE(MAX(start_time), '1900-01-01') INTO v_last_full_backup_time
    FROM backup_log
    WHERE backup_type = 'full' AND status = 'completed';
    
    INSERT INTO backup_log (backup_type, backup_name, backup_path, start_time, status, created_by)
    VALUES ('differential', v_backup_name, p_backup_path, v_start_time, 'in_progress', p_created_by);
    
    SET v_backup_id = LAST_INSERT_ID();
    
    -- Backup changed orders
    DROP TEMPORARY TABLE IF EXISTS tmp_orders_diff;
    CREATE TEMPORARY TABLE tmp_orders_diff AS
    SELECT * FROM orders
    WHERE created_at > v_last_full_backup_time OR updated_at > v_last_full_backup_time;
    
    -- Backup changed customers
    DROP TEMPORARY TABLE IF EXISTS tmp_customers_diff;
    CREATE TEMPORARY TABLE tmp_customers_diff AS
    SELECT * FROM customers
    WHERE created_at > v_last_full_backup_time OR updated_at > v_last_full_backup_time;
    
    -- Backup changed products
    DROP TEMPORARY TABLE IF EXISTS tmp_products_diff;
    CREATE TEMPORARY TABLE tmp_products_diff AS
    SELECT * FROM products
    WHERE created_at > v_last_full_backup_time OR updated_at > v_last_full_backup_time;
    
    SELECT 
        (SELECT COUNT(*) FROM tmp_orders_diff) +
        (SELECT COUNT(*) FROM tmp_customers_diff) +
        (SELECT COUNT(*) FROM tmp_products_diff)
    INTO v_row_count;
    
    UPDATE backup_log
    SET end_time = NOW(),
        duration_seconds = TIMESTAMPDIFF(SECOND, start_time, NOW()),
        status = 'completed',
        row_count = v_row_count,
        tables_backed_up = 'orders,customers,products (differential)'
    WHERE backup_id = v_backup_id;
    
    SELECT 
        v_backup_id AS backup_id,
        v_backup_name AS backup_name,
        v_row_count AS rows_backed_up,
        v_last_full_backup_time AS since_full_backup,
        'Differential backup completed' AS status;
    
    DROP TEMPORARY TABLE IF EXISTS tmp_orders_diff;
    DROP TEMPORARY TABLE IF EXISTS tmp_customers_diff;
    DROP TEMPORARY TABLE IF EXISTS tmp_products_diff;
END$$

-- ========================================
-- 7. BACKUP STATUS REPORT
-- ========================================
DROP PROCEDURE IF EXISTS sp_backup_status_report$$
CREATE PROCEDURE sp_backup_status_report(
    IN p_days_back INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_days_back DAY);
    
    SELECT 
        backup_id,
        backup_type,
        backup_name,
        tables_backed_up,
        start_time,
        end_time,
        duration_seconds,
        status,
        row_count,
        backup_size_mb,
        created_by
    FROM backup_log
    WHERE start_time >= v_cutoff_date
    ORDER BY start_time DESC;
END$$

-- ========================================
-- 8. CLEANUP OLD BACKUPS
-- ========================================
DROP PROCEDURE IF EXISTS sp_cleanup_old_backups$$
CREATE PROCEDURE sp_cleanup_old_backups(
    IN p_days_to_keep INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_deleted_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_days_to_keep DAY);
    
    -- Delete old backup logs
    DELETE FROM backup_log
    WHERE start_time < v_cutoff_date
    AND status = 'completed';
    
    SET v_deleted_count = ROW_COUNT();
    
    SELECT CONCAT('Deleted ', v_deleted_count, ' backup log entries older than ', p_days_to_keep, ' days') AS Result;
END$$

-- ========================================
-- 9. VERIFY BACKUP INTEGRITY
-- ========================================
DROP PROCEDURE IF EXISTS sp_verify_backup_integrity$$
CREATE PROCEDURE sp_verify_backup_integrity(
    IN p_backup_id INT
)
BEGIN
    DECLARE v_backup_type VARCHAR(20);
    DECLARE v_row_count INT;
    DECLARE v_tables_backed_up TEXT;
    
    SELECT backup_type, row_count, tables_backed_up
    INTO v_backup_type, v_row_count, v_tables_backed_up
    FROM backup_log
    WHERE backup_id = p_backup_id;
    
    SELECT 
        p_backup_id AS backup_id,
        v_backup_type AS backup_type,
        v_tables_backed_up AS tables_included,
        v_row_count AS total_rows,
        CASE 
            WHEN v_row_count > 0 THEN 'VALID'
            WHEN v_row_count = 0 THEN 'WARNING: No rows'
            ELSE 'ERROR: Unable to verify'
        END AS integrity_status
    FROM backup_log
    WHERE backup_id = p_backup_id;
END$$

-- ========================================
-- 10. AUTOMATED BACKUP SCHEDULER INFO
-- ========================================
DROP PROCEDURE IF EXISTS sp_backup_schedule_recommendation$$
CREATE PROCEDURE sp_backup_schedule_recommendation()
BEGIN
    SELECT 
        'Full Backup' AS backup_type,
        'Weekly (Sunday 2:00 AM)' AS recommended_schedule,
        'CALL sp_full_database_backup("/backups/full/", "scheduler")' AS command
    UNION ALL
    SELECT 
        'Incremental - Orders' AS backup_type,
        'Daily (Every 6 hours)' AS recommended_schedule,
        'CALL sp_incremental_backup_orders("/backups/incremental/", "scheduler")' AS command
    UNION ALL
    SELECT 
        'Incremental - Customers' AS backup_type,
        'Daily (Midnight)' AS recommended_schedule,
        'CALL sp_incremental_backup_customers("/backups/incremental/", "scheduler")' AS command
    UNION ALL
    SELECT 
        'Critical Tables' AS backup_type,
        'Daily (11:00 PM)' AS recommended_schedule,
        'CALL sp_backup_critical_tables("/backups/critical/", "scheduler")' AS command
    UNION ALL
    SELECT 
        'Differential Backup' AS backup_type,
        'Daily (3:00 AM)' AS recommended_schedule,
        'CALL sp_differential_backup("/backups/differential/", "scheduler")' AS command;
END$$

DELIMITER ;

-- ========================================
-- USAGE EXAMPLES
-- ========================================

-- Full database backup
-- CALL sp_full_database_backup('/backups/full/', 'admin');

-- Incremental backup - orders
-- CALL sp_incremental_backup_orders('/backups/incremental/', 'admin');

-- Incremental backup - customers
-- CALL sp_incremental_backup_customers('/backups/incremental/', 'admin');

-- Backup specific table
-- CALL sp_table_backup('products', '/backups/tables/', 'admin');

-- Backup critical tables
-- CALL sp_backup_critical_tables('/backups/critical/', 'admin');

-- Differential backup
-- CALL sp_differential_backup('/backups/differential/', 'admin');

-- View backup status
-- CALL sp_backup_status_report(30);

-- Verify backup integrity
-- CALL sp_verify_backup_integrity(1);

-- Get schedule recommendations
-- CALL sp_backup_schedule_recommendation();

-- Cleanup old backups
-- CALL sp_cleanup_old_backups(90);

SELECT 'All backup procedures created successfully' AS Status;