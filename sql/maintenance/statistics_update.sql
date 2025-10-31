-- ========================================
-- STATISTICS UPDATE SCRIPTS
-- E-commerce Revenue Analytics Engine
-- Update Statistics, Analyze Query Plans, Refresh Metadata
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CREATE STATISTICS TRACKING TABLES
-- ========================================

-- Table to track statistics updates
CREATE TABLE IF NOT EXISTS statistics_update_log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100),
    rows_before BIGINT,
    rows_after BIGINT,
    avg_row_length_before INT,
    avg_row_length_after INT,
    data_length_mb_before DECIMAL(10,2),
    data_length_mb_after DECIMAL(10,2),
    update_duration_seconds INT,
    update_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_table_name (table_name),
    INDEX idx_timestamp (update_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table to store query execution plans
CREATE TABLE IF NOT EXISTS query_plan_analysis (
    plan_id INT PRIMARY KEY AUTO_INCREMENT,
    query_name VARCHAR(200),
    query_text TEXT,
    execution_plan JSON,
    estimated_rows BIGINT,
    actual_rows BIGINT,
    execution_time_ms DECIMAL(10,2),
    rows_examined BIGINT,
    using_index BOOLEAN,
    using_filesort BOOLEAN,
    using_temporary BOOLEAN,
    recommendations TEXT,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_query_name (query_name),
    INDEX idx_analyzed_at (analyzed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table to track metadata refresh
CREATE TABLE IF NOT EXISTS metadata_refresh_log (
    refresh_id INT PRIMARY KEY AUTO_INCREMENT,
    metadata_type VARCHAR(50),
    object_name VARCHAR(200),
    status ENUM('success', 'failed', 'warning') DEFAULT 'success',
    message TEXT,
    refresh_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metadata_type (metadata_type),
    INDEX idx_timestamp (refresh_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- 2. UPDATE TABLE STATISTICS
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_update_table_statistics(
    IN target_database VARCHAR(100),
    IN specific_table VARCHAR(100)
)
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE v_table_name VARCHAR(100);
    DECLARE v_rows_before BIGINT;
    DECLARE v_rows_after BIGINT;
    DECLARE v_avg_row_before INT;
    DECLARE v_avg_row_after INT;
    DECLARE v_data_length_before BIGINT;
    DECLARE v_data_length_after BIGINT;
    DECLARE v_start_time DATETIME;
    DECLARE v_duration INT;
    DECLARE v_update_count INT DEFAULT 0;
    
    DECLARE table_cursor CURSOR FOR
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = target_database
        AND table_type = 'BASE TABLE'
        AND (specific_table IS NULL OR table_name = specific_table);
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    SELECT CONCAT('Starting statistics update for database: ', target_database) AS Status;
    
    OPEN table_cursor;
    
    update_loop: LOOP
        FETCH table_cursor INTO v_table_name;
        IF done THEN
            LEAVE update_loop;
        END IF;
        
        -- Get statistics before update
        SELECT table_rows, avg_row_length, data_length
        INTO v_rows_before, v_avg_row_before, v_data_length_before
        FROM information_schema.tables
        WHERE table_schema = target_database
        AND table_name = v_table_name;
        
        SET v_start_time = NOW();
        
        -- Run ANALYZE TABLE
        SET @analyze_stmt = CONCAT('ANALYZE TABLE ', target_database, '.', v_table_name);
        PREPARE stmt FROM @analyze_stmt;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
        
        SET v_duration = TIMESTAMPDIFF(SECOND, v_start_time, NOW());
        
        -- Get statistics after update
        SELECT table_rows, avg_row_length, data_length
        INTO v_rows_after, v_avg_row_after, v_data_length_after
        FROM information_schema.tables
        WHERE table_schema = target_database
        AND table_name = v_table_name;
        
        -- Log the update
        INSERT INTO statistics_update_log (
            table_name, rows_before, rows_after,
            avg_row_length_before, avg_row_length_after,
            data_length_mb_before, data_length_mb_after,
            update_duration_seconds
        ) VALUES (
            v_table_name, v_rows_before, v_rows_after,
            v_avg_row_before, v_avg_row_after,
            ROUND(v_data_length_before / 1024 / 1024, 2),
            ROUND(v_data_length_after / 1024 / 1024, 2),
            v_duration
        );
        
        SET v_update_count = v_update_count + 1;
        SELECT CONCAT('Updated: ', v_table_name, ' (Rows: ', v_rows_after, ', Duration: ', v_duration, 's)') AS Progress;
    END LOOP;
    
    CLOSE table_cursor;
    
    SELECT CONCAT('SUCCESS: Updated statistics for ', v_update_count, ' tables') AS Status;
END //

DELIMITER ;

-- ========================================
-- 3. ANALYZE QUERY EXECUTION PLANS
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_analyze_query_plan(
    IN query_name VARCHAR(200),
    IN query_text TEXT
)
BEGIN
    DECLARE v_plan_json JSON;
    DECLARE v_execution_time DECIMAL(10,2);
    DECLARE v_start_time DATETIME;
    DECLARE v_using_index BOOLEAN DEFAULT FALSE;
    DECLARE v_using_filesort BOOLEAN DEFAULT FALSE;
    DECLARE v_using_temporary BOOLEAN DEFAULT FALSE;
    DECLARE v_recommendations TEXT DEFAULT '';
    
    SET v_start_time = NOW();
    
    -- Get execution plan
    SET @explain_stmt = CONCAT('EXPLAIN FORMAT=JSON ', query_text);
    PREPARE stmt FROM @explain_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    -- Note: Actual plan extraction would require more complex logic
    -- This is a simplified version
    
    SET v_execution_time = TIMESTAMPDIFF(MICROSECOND, v_start_time, NOW()) / 1000;
    
    -- Analyze for common issues
    IF query_text LIKE '%ORDER BY%' AND query_text NOT LIKE '%INDEX%' THEN
        SET v_using_filesort = TRUE;
        SET v_recommendations = CONCAT(v_recommendations, 'Consider adding index for ORDER BY clause. ');
    END IF;
    
    IF query_text LIKE '%GROUP BY%' AND query_text NOT LIKE '%INDEX%' THEN
        SET v_using_temporary = TRUE;
        SET v_recommendations = CONCAT(v_recommendations, 'Consider adding index for GROUP BY clause. ');
    END IF;
    
    IF query_text LIKE '%WHERE%' AND query_text NOT LIKE '%INDEX%' THEN
        SET v_recommendations = CONCAT(v_recommendations, 'Verify WHERE clause uses indexed columns. ');
    END IF;
    
    IF query_text LIKE '%SELECT *%' THEN
        SET v_recommendations = CONCAT(v_recommendations, 'Avoid SELECT *, specify columns explicitly. ');
    END IF;
    
    -- Store analysis
    INSERT INTO query_plan_analysis (
        query_name, query_text, execution_time_ms,
        using_filesort, using_temporary, recommendations
    ) VALUES (
        query_name, query_text, v_execution_time,
        v_using_filesort, v_using_temporary, v_recommendations
    );
    
    SELECT query_name AS 'Query Name',
           v_execution_time AS 'Execution Time (ms)',
           v_using_filesort AS 'Using Filesort',
           v_using_temporary AS 'Using Temporary',
           v_recommendations AS 'Recommendations';
END //

DELIMITER ;

-- ========================================
-- 4. BATCH ANALYZE CRITICAL QUERIES
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_batch_analyze_critical_queries(
    IN target_database VARCHAR(100)
)
BEGIN
    SELECT 'Analyzing critical e-commerce queries...' AS Status;
    
    -- Query 1: Customer order history
    CALL sp_analyze_query_plan(
        'customer_order_history',
        'SELECT c.customer_id, c.email, COUNT(o.order_id) AS order_count, SUM(o.total_amount) AS total_spent
         FROM customers c
         LEFT JOIN orders o ON c.customer_id = o.customer_id
         WHERE c.status = "active"
         GROUP BY c.customer_id, c.email
         ORDER BY total_spent DESC'
    );
    
    -- Query 2: Top selling products
    CALL sp_analyze_query_plan(
        'top_selling_products',
        'SELECT p.product_id, p.product_name, SUM(oi.quantity) AS total_sold, SUM(oi.subtotal) AS revenue
         FROM products p
         JOIN order_items oi ON p.product_id = oi.product_id
         JOIN orders o ON oi.order_id = o.order_id
         WHERE o.status = "delivered"
         GROUP BY p.product_id, p.product_name
         ORDER BY total_sold DESC
         LIMIT 20'
    );
    
    -- Query 3: Monthly revenue report
    CALL sp_analyze_query_plan(
        'monthly_revenue_report',
        'SELECT DATE_FORMAT(order_date, "%Y-%m") AS month, 
                COUNT(order_id) AS total_orders,
                SUM(total_amount) AS revenue,
                AVG(total_amount) AS avg_order_value
         FROM orders
         WHERE payment_status = "paid"
         GROUP BY DATE_FORMAT(order_date, "%Y-%m")
         ORDER BY month DESC'
    );
    
    -- Query 4: Inventory alerts
    CALL sp_analyze_query_plan(
        'inventory_alerts',
        'SELECT p.product_id, p.product_name, i.quantity_available, i.reorder_level
         FROM products p
         JOIN inventory i ON p.product_id = i.product_id
         WHERE i.quantity_available < i.reorder_level
         ORDER BY i.quantity_available ASC'
    );
    
    SELECT 'Batch analysis completed' AS Status;
END //

DELIMITER ;

-- ========================================
-- 5. REFRESH METADATA
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_refresh_metadata(
    IN target_database VARCHAR(100)
)
BEGIN
    DECLARE v_refresh_count INT DEFAULT 0;
    
    SELECT 'Starting metadata refresh...' AS Status;
    
    -- Refresh table statistics
    INSERT INTO metadata_refresh_log (metadata_type, object_name, status, message)
    VALUES ('table_statistics', target_database, 'success', 'Refreshing table statistics');
    
    CALL sp_update_table_statistics(target_database, NULL);
    SET v_refresh_count = v_refresh_count + 1;
    
    -- Refresh information_schema cache
    SELECT COUNT(*) INTO @dummy FROM information_schema.tables WHERE table_schema = target_database;
    INSERT INTO metadata_refresh_log (metadata_type, object_name, status, message)
    VALUES ('information_schema', target_database, 'success', 'Refreshed information_schema cache');
    SET v_refresh_count = v_refresh_count + 1;
    
    -- Flush table cache
    FLUSH TABLES;
    INSERT INTO metadata_refresh_log (metadata_type, object_name, status, message)
    VALUES ('table_cache', 'all', 'success', 'Flushed table cache');
    SET v_refresh_count = v_refresh_count + 1;
    
    -- Refresh InnoDB statistics
    SET GLOBAL innodb_stats_on_metadata = 1;
    INSERT INTO metadata_refresh_log (metadata_type, object_name, status, message)
    VALUES ('innodb_stats', target_database, 'success', 'Refreshed InnoDB statistics');
    SET v_refresh_count = v_refresh_count + 1;
    
    SELECT CONCAT('SUCCESS: Refreshed ', v_refresh_count, ' metadata components') AS Status;
END //

DELIMITER ;

-- ========================================
-- 6. GENERATE STATISTICS REPORT
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_generate_statistics_report(
    IN target_database VARCHAR(100),
    IN days_back INT
)
BEGIN
    DECLARE v_report_date DATE;
    SET v_report_date = DATE_SUB(CURDATE(), INTERVAL days_back DAY);
    
    SELECT '=== DATABASE STATISTICS REPORT ===' AS Report;
    
    -- 1. Table size summary
    SELECT 
        table_name,
        table_rows AS 'Estimated Rows',
        ROUND(data_length / 1024 / 1024, 2) AS 'Data Size (MB)',
        ROUND(index_length / 1024 / 1024, 2) AS 'Index Size (MB)',
        ROUND((data_length + index_length) / 1024 / 1024, 2) AS 'Total Size (MB)',
        ROUND(data_free / 1024 / 1024, 2) AS 'Free Space (MB)'
    FROM information_schema.tables
    WHERE table_schema = target_database
    AND table_type = 'BASE TABLE'
    ORDER BY (data_length + index_length) DESC;
    
    -- 2. Recent statistics updates
    SELECT 
        table_name,
        rows_after AS 'Current Rows',
        ROUND(data_length_mb_after, 2) AS 'Data Size (MB)',
        update_duration_seconds AS 'Update Duration (s)',
        update_timestamp AS 'Last Updated'
    FROM statistics_update_log
    WHERE DATE(update_timestamp) >= v_report_date
    ORDER BY update_timestamp DESC
    LIMIT 20;
    
    -- 3. Query performance summary
    SELECT 
        query_name,
        COUNT(*) AS 'Analysis Count',
        AVG(execution_time_ms) AS 'Avg Execution Time (ms)',
        MAX(execution_time_ms) AS 'Max Execution Time (ms)',
        SUM(using_filesort) AS 'Times Used Filesort',
        SUM(using_temporary) AS 'Times Used Temporary'
    FROM query_plan_analysis
    WHERE DATE(analyzed_at) >= v_report_date
    GROUP BY query_name
    ORDER BY AVG(execution_time_ms) DESC;
    
    -- 4. Top recommendations
    SELECT 
        query_name,
        recommendations,
        execution_time_ms AS 'Execution Time (ms)',
        analyzed_at AS 'Analyzed At'
    FROM query_plan_analysis
    WHERE recommendations IS NOT NULL 
    AND recommendations != ''
    AND DATE(analyzed_at) >= v_report_date
    ORDER BY execution_time_ms DESC
    LIMIT 10;
    
    -- 5. Metadata refresh history
    SELECT 
        metadata_type,
        COUNT(*) AS 'Refresh Count',
        MAX(refresh_timestamp) AS 'Last Refresh'
    FROM metadata_refresh_log
    WHERE DATE(refresh_timestamp) >= v_report_date
    GROUP BY metadata_type;
    
    SELECT '=== END OF REPORT ===' AS Report;
END //

DELIMITER ;

-- ========================================
-- 7. AUTO STATISTICS MAINTENANCE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_auto_statistics_maintenance(
    IN target_database VARCHAR(100)
)
BEGIN
    DECLARE v_start_time DATETIME;
    DECLARE v_end_time DATETIME;
    
    SET v_start_time = NOW();
    
    SELECT CONCAT('Starting automatic statistics maintenance at ', v_start_time) AS Status;
    
    -- Step 1: Update table statistics
    CALL sp_update_table_statistics(target_database, NULL);
    
    -- Step 2: Analyze critical queries
    CALL sp_batch_analyze_critical_queries(target_database);
    
    -- Step 3: Refresh metadata
    CALL sp_refresh_metadata(target_database);
    
    -- Step 4: Generate report
    CALL sp_generate_statistics_report(target_database, 7);
    
    SET v_end_time = NOW();
    
    SELECT CONCAT('Maintenance completed in ', TIMESTAMPDIFF(SECOND, v_start_time, v_end_time), ' seconds') AS Status;
END //

DELIMITER ;

-- ========================================
-- 8. STATISTICS COMPARISON
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_compare_statistics(
    IN target_database VARCHAR(100),
    IN table_name_filter VARCHAR(100)
)
BEGIN
    SELECT 
        sul1.table_name,
        sul1.rows_after AS 'Current Rows',
        sul2.rows_after AS 'Previous Rows',
        sul1.rows_after - sul2.rows_after AS 'Row Change',
        ROUND(((sul1.rows_after - sul2.rows_after) / sul2.rows_after) * 100, 2) AS 'Change %',
        sul1.data_length_mb_after AS 'Current Size (MB)',
        sul2.data_length_mb_after AS 'Previous Size (MB)',
        sul1.update_timestamp AS 'Current Update',
        sul2.update_timestamp AS 'Previous Update'
    FROM statistics_update_log sul1
    INNER JOIN statistics_update_log sul2 
        ON sul1.table_name = sul2.table_name
        AND sul1.log_id > sul2.log_id
    WHERE sul1.log_id IN (
        SELECT MAX(log_id)
        FROM statistics_update_log
        WHERE table_name LIKE CONCAT('%', IFNULL(table_name_filter, ''), '%')
        GROUP BY table_name
    )
    AND sul2.log_id IN (
        SELECT MAX(log_id)
        FROM statistics_update_log sul3
        WHERE sul3.table_name = sul1.table_name
        AND sul3.log_id < sul1.log_id
    )
    ORDER BY ABS(sul1.rows_after - sul2.rows_after) DESC;
END //

DELIMITER ;

-- ========================================
-- 9. IDENTIFY STALE STATISTICS
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_identify_stale_statistics(
    IN target_database VARCHAR(100),
    IN days_threshold INT
)
BEGIN
    DROP TEMPORARY TABLE IF EXISTS tmp_stale_stats;
    CREATE TEMPORARY TABLE tmp_stale_stats (
        table_name VARCHAR(100),
        current_rows BIGINT,
        last_analyzed DATE,
        days_since_analysis INT,
        recommendation VARCHAR(200)
    );
    
    INSERT INTO tmp_stale_stats
    SELECT 
        t.table_name,
        t.table_rows,
        MAX(DATE(sul.update_timestamp)) AS last_analyzed,
        DATEDIFF(CURDATE(), MAX(DATE(sul.update_timestamp))) AS days_since_analysis,
        CASE
            WHEN DATEDIFF(CURDATE(), MAX(DATE(sul.update_timestamp))) > days_threshold * 2 THEN 'CRITICAL - Update Immediately'
            WHEN DATEDIFF(CURDATE(), MAX(DATE(sul.update_timestamp))) > days_threshold THEN 'WARNING - Update Soon'
            ELSE 'OK'
        END AS recommendation
    FROM information_schema.tables t
    LEFT JOIN statistics_update_log sul ON t.table_name = sul.table_name
    WHERE t.table_schema = target_database
    AND t.table_type = 'BASE TABLE'
    GROUP BY t.table_name, t.table_rows;
    
    SELECT * FROM tmp_stale_stats
    WHERE days_since_analysis > days_threshold OR days_since_analysis IS NULL
    ORDER BY days_since_analysis DESC;
    
    DROP TEMPORARY TABLE IF EXISTS tmp_stale_stats;
END //

DELIMITER ;

-- ========================================
-- USAGE EXAMPLES
-- ========================================

-- Example 1: Update Statistics for All Tables
-- CALL sp_update_table_statistics('ecommerce_analytics', NULL);

-- Example 2: Update Statistics for Specific Table
-- CALL sp_update_table_statistics('ecommerce_analytics', 'orders');

-- Example 3: Analyze Single Query Plan
-- CALL sp_analyze_query_plan('test_query', 'SELECT * FROM orders WHERE status = "pending"');

-- Example 4: Batch Analyze Critical Queries
-- CALL sp_batch_analyze_critical_queries('ecommerce_analytics');

-- Example 5: Refresh All Metadata
-- CALL sp_refresh_metadata('ecommerce_analytics');

-- Example 6: Generate Statistics Report (Last 7 Days)
-- CALL sp_generate_statistics_report('ecommerce_analytics', 7);

-- Example 7: Run Auto Maintenance
-- CALL sp_auto_statistics_maintenance('ecommerce_analytics');

-- Example 8: Compare Statistics
-- CALL sp_compare_statistics('ecommerce_analytics', NULL);

-- Example 9: Identify Stale Statistics (>30 days)
-- CALL sp_identify_stale_statistics('ecommerce_analytics', 30);

-- ========================================
-- SCHEDULE AS CRON JOB
-- ========================================
-- Daily statistics update (2 AM):
-- 0 2 * * * mysql -u root -p -e "CALL ecommerce_analytics.sp_auto_statistics_maintenance('ecommerce_analytics');"

-- Weekly full report (Sunday 3 AM):
-- 0 3 * * 0 mysql -u root -p -e "CALL ecommerce_analytics.sp_generate_statistics_report('ecommerce_analytics', 30);"

SELECT 'All statistics update procedures created successfully' AS Status;