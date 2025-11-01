-- ========================================
-- RESTORE SCRIPTS
-- E-commerce Revenue Analytics Engine
-- Point-in-Time Recovery & Selective Restore
-- ========================================

-- ========================================
-- 1. FULL DATABASE RESTORE PROCEDURE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_restore_full_database(
    IN backup_file_path VARCHAR(500),
    IN target_database VARCHAR(100)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'ERROR: Full restore failed' AS Status;
    END;
    
    START TRANSACTION;
    
    -- Drop existing database if exists
    

    












SET @drop_db = CONCAT('DROP DATABASE IF EXISTS ', target_database);
    PREPARE stmt FROM @drop_db;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    -- Create new database
    SET @create_db = CONCAT('CREATE DATABASE ', target_database, ' CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci');
    PREPARE stmt FROM @create_db;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    -- Use the new database
    SET @use_db = CONCAT('USE ', target_database);
    PREPARE stmt FROM @use_db;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    -- Source the backup file
    SET @restore_cmd = CONCAT('SOURCE ', backup_file_path);
    PREPARE stmt FROM @restore_cmd;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    COMMIT;
    SELECT CONCAT('SUCCESS: Database ', target_database, ' restored successfully') AS Status;
END //

DELIMITER ;

-- ========================================
-- 2. POINT-IN-TIME RECOVERY PROCEDURE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_point_in_time_recovery(
    IN backup_file_path VARCHAR(500),
    IN binlog_file VARCHAR(200),
    IN stop_datetime DATETIME,
    IN target_database VARCHAR(100)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'ERROR: Point-in-time recovery failed' AS Status;
    END;
    
    START TRANSACTION;
    
    -- Step 1: Restore from full backup
    CALL sp_restore_full_database(backup_file_path, target_database);
    
    -- Step 2: Apply binary logs up to specified point in time
    -- Note: This requires mysqlbinlog utility to be run externally
    -- The following is a template for the command
    SELECT CONCAT(
        'Execute externally: ',
        'mysqlbinlog --stop-datetime="', stop_datetime, '" ',
        binlog_file, ' | mysql -u root -p ', target_database
    ) AS 'Next Step';
    
    COMMIT;
    SELECT CONCAT('Point-in-time recovery initiated for ', target_database) AS Status;
END //

DELIMITER ;

-- ========================================
-- 3. SELECTIVE TABLE RESTORE PROCEDURE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_restore_selective_tables(
    IN backup_database VARCHAR(100),
    IN target_database VARCHAR(100),
    IN table_list TEXT  -- Comma-separated list of tables
)
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE table_name VARCHAR(100);
    DECLARE table_cursor CURSOR FOR 
        SELECT TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(table_list, ',', n.n), ',', -1)) AS table_name
        FROM (
            SELECT 1 AS n UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5
            UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10
            UNION SELECT 11 UNION SELECT 12 UNION SELECT 13 UNION SELECT 14 UNION SELECT 15
        ) n
        WHERE n.n <= 1 + (LENGTH(table_list) - LENGTH(REPLACE(table_list, ',', '')));
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'ERROR: Selective restore failed' AS Status;
    END;
    
    START TRANSACTION;
    
    OPEN table_cursor;
    
    restore_loop: LOOP
        FETCH table_cursor INTO table_name;
        IF done THEN
            LEAVE restore_loop;
        END IF;
        
        -- Drop table if exists in target
        SET @drop_stmt = CONCAT('DROP TABLE IF EXISTS ', target_database, '.', table_name);
        PREPARE stmt FROM @drop_stmt;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
        
        -- Copy table structure and data
        SET @copy_stmt = CONCAT(
            'CREATE TABLE ', target_database, '.', table_name, 
            ' LIKE ', backup_database, '.', table_name
        );
        PREPARE stmt FROM @copy_stmt;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
        
        SET @insert_stmt = CONCAT(
            'INSERT INTO ', target_database, '.', table_name,
            ' SELECT * FROM ', backup_database, '.', table_name
        );
        PREPARE stmt FROM @insert_stmt;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
        
        SELECT CONCAT('Restored table: ', table_name) AS Progress;
    END LOOP;
    
    CLOSE table_cursor;
    COMMIT;
    
    SELECT 'SUCCESS: Selective tables restored' AS Status;
END //

DELIMITER ;

-- ========================================
-- 4. RESTORE WITH DATA VALIDATION
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_restore_with_validation(
    IN backup_file_path VARCHAR(500),
    IN target_database VARCHAR(100)
)
BEGIN
DECLARE row_count_customers INT;
    DECLARE row_count_orders INT;
    DECLARE row_count_products INT;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'ERROR: Restore with validation failed' AS Status;
    END;
    
    START TRANSACTION;
    
    -- Perform restore
    CALL sp_restore_full_database(backup_file_path, target_database);
    
    -- Validate critical tables
    

    












SET @count_stmt = CONCAT('SELECT COUNT(*) INTO @row_count_customers FROM ', target_database, '.customers');
    PREPARE stmt FROM @count_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    SET row_count_customers = @row_count_customers;
    
    SET @count_stmt = CONCAT('SELECT COUNT(*) INTO @row_count_orders FROM ', target_database, '.orders');
    PREPARE stmt FROM @count_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    SET row_count_orders = @row_count_orders;
    
    SET @count_stmt = CONCAT('SELECT COUNT(*) INTO @row_count_products FROM ', target_database, '.products');
    PREPARE stmt FROM @count_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    SET row_count_products = @row_count_products;
    
    -- Display validation results
    SELECT 
        'Data Validation Report' AS Report,
        row_count_customers AS 'Customers',
        row_count_orders AS 'Orders',
        row_count_products AS 'Products';
    
    COMMIT;
    SELECT 'SUCCESS: Restore completed with validation' AS Status;
END //

DELIMITER ;

-- ========================================
-- 5. INCREMENTAL RESTORE PROCEDURE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_restore_incremental(
    IN full_backup_path VARCHAR(500),
    IN incremental_backup_path VARCHAR(500),
    IN target_database VARCHAR(100)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'ERROR: Incremental restore failed' AS Status;
    END;
    
    START TRANSACTION;
    
    -- Step 1: Restore full backup
    CALL sp_restore_full_database(full_backup_path, target_database);
    
    -- Step 2: Apply incremental backup
    

    












SET @use_db = CONCAT('USE ', target_database);
    PREPARE stmt FROM @use_db;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    SET @restore_inc = CONCAT('SOURCE ', incremental_backup_path);
    PREPARE stmt FROM @restore_inc;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    COMMIT;
    SELECT CONCAT('SUCCESS: Incremental restore completed for ', target_database) AS Status;
END //

DELIMITER ;

-- ========================================
-- 6. RESTORE SPECIFIC DATE RANGE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_restore_date_range(
    IN backup_database VARCHAR(100),
    IN target_database VARCHAR(100),
    IN start_date DATE,
    IN end_date DATE
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'ERROR: Date range restore failed' AS Status;
    END;
    
    START TRANSACTION;
    
    -- Restore orders within date range
    

    












SET @drop_stmt = CONCAT('DROP TABLE IF EXISTS ', target_database, '.orders');
    PREPARE stmt FROM @drop_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    SET @create_stmt = CONCAT(
        'CREATE TABLE ', target_database, '.orders ',
        'LIKE ', backup_database, '.orders'
    );
    PREPARE stmt FROM @create_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    SET @insert_stmt = CONCAT(
        'INSERT INTO ', target_database, '.orders ',
        'SELECT * FROM ', backup_database, '.orders ',
        'WHERE DATE(order_date) BETWEEN "', start_date, '" AND "', end_date, '"'
    );
    PREPARE stmt FROM @insert_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    -- Restore related order_items
    SET @drop_stmt = CONCAT('DROP TABLE IF EXISTS ', target_database, '.order_items');
    PREPARE stmt FROM @drop_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    SET @create_stmt = CONCAT(
        'CREATE TABLE ', target_database, '.order_items ',
        'LIKE ', backup_database, '.order_items'
    );
    PREPARE stmt FROM @create_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    SET @insert_stmt = CONCAT(
        'INSERT INTO ', target_database, '.order_items ',
        'SELECT oi.* FROM ', backup_database, '.order_items oi ',
        'INNER JOIN ', target_database, '.orders o ON oi.order_id = o.order_id'
    );
    PREPARE stmt FROM @insert_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    COMMIT;
    SELECT CONCAT('SUCCESS: Date range restore completed (', start_date, ' to ', end_date, ')') AS Status;
END //

DELIMITER ;

-- ========================================
-- 7. EMERGENCY QUICK RESTORE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_emergency_restore(
    IN latest_backup_path VARCHAR(500)
)
BEGIN
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        SELECT 'CRITICAL ERROR: Emergency restore failed - Manual intervention required' AS Status;
    END;
    
    -- Quick restore without extensive validation
    CALL sp_restore_full_database(latest_backup_path, 'ecommerce_analytics');
    
    SELECT 'EMERGENCY RESTORE COMPLETED' AS Status,
           NOW() AS 'Restored At',
           'Run validation queries immediately' AS 'Action Required';
END //

DELIMITER ;

-- ========================================
-- USAGE EXAMPLES
-- ========================================

-- Example 1: Full Database Restore
-- CALL sp_restore_full_database('/backup/ecommerce_full_20250131.sql', 'ecommerce_analytics_restored');

-- Example 2: Point-in-Time Recovery
-- CALL sp_point_in_time_recovery('/backup/ecommerce_full.sql', '/var/log/mysql/binlog.000001', '2025-01-31 14:30:00', 'ecommerce_analytics_pitr');

-- Example 3: Selective Table Restore
-- CALL sp_restore_selective_tables('ecommerce_backup', 'ecommerce_analytics', 'customers,orders,order_items');

-- Example 4: Restore with Validation
-- CALL sp_restore_with_validation('/backup/ecommerce_full.sql', 'ecommerce_analytics');

-- Example 5: Incremental Restore
-- CALL sp_restore_incremental('/backup/full_backup.sql', '/backup/incremental_backup.sql', 'ecommerce_analytics');

-- Example 6: Date Range Restore
-- CALL sp_restore_date_range('ecommerce_backup', 'ecommerce_analytics_jan', '2025-01-01', '2025-01-31');

-- Example 7: Emergency Restore
-- CALL sp_emergency_restore('/backup/latest_backup.sql');

SELECT 'All restore procedures created successfully' AS Status;