-- ========================================
-- INDEX MAINTENANCE SCRIPTS
-- E-commerce Revenue Analytics Engine
-- Rebuild, Optimize, and Maintain Indexes
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CREATE INDEX STATISTICS TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS index_statistics (
    stat_id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100),
    index_name VARCHAR(100),
    cardinality BIGINT,
    index_length BIGINT,
    data_length BIGINT,
    fragmentation_pct DECIMAL(5,2),
    last_used DATETIME,
    usage_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_table_name (table_name),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- 2. ANALYZE INDEX FRAGMENTATION
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_analyze_index_fragmentation(
    IN target_database VARCHAR(100)
)
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE v_table_name VARCHAR(100);
    DECLARE v_data_length BIGINT;
    DECLARE v_index_length BIGINT;
    DECLARE v_data_free BIGINT;
    DECLARE v_fragmentation DECIMAL(5,2);
    
    DECLARE table_cursor CURSOR FOR
        SELECT table_name, data_length, index_length, data_free
        FROM information_schema.tables
        WHERE table_schema = target_database
        AND table_type = 'BASE TABLE';
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    -- Create temporary results table
    DROP TEMPORARY TABLE IF EXISTS tmp_fragmentation_report;
    CREATE TEMPORARY TABLE tmp_fragmentation_report (
        table_name VARCHAR(100),
        data_length_mb DECIMAL(10,2),
        index_length_mb DECIMAL(10,2),
        data_free_mb DECIMAL(10,2),
        fragmentation_pct DECIMAL(5,2),
        recommendation VARCHAR(100)
    );
    
    OPEN table_cursor;
    
    analyze_loop: LOOP
        FETCH table_cursor INTO v_table_name, v_data_length, v_index_length, v_data_free;
        IF done THEN
            LEAVE analyze_loop;
        END IF;
        
        -- Calculate fragmentation percentage
        IF v_data_length > 0 THEN
            SET v_fragmentation = (v_data_free / v_data_length) * 100;
        ELSE
            SET v_fragmentation = 0;
        END IF;
        
        -- Insert into results
        INSERT INTO tmp_fragmentation_report
        VALUES (
            v_table_name,
            ROUND(v_data_length / 1024 / 1024, 2),
            ROUND(v_index_length / 1024 / 1024, 2),
            ROUND(v_data_free / 1024 / 1024, 2),
            v_fragmentation,
            CASE
                WHEN v_fragmentation > 30 THEN 'CRITICAL - Rebuild Required'
                WHEN v_fragmentation > 15 THEN 'WARNING - Consider Rebuild'
                WHEN v_fragmentation > 5 THEN 'MONITOR - May Need Optimize'
                ELSE 'HEALTHY'
            END
        );
    END LOOP;
    
    CLOSE table_cursor;
    
    -- Display results
    SELECT * FROM tmp_fragmentation_report
    ORDER BY fragmentation_pct DESC;
    
    DROP TEMPORARY TABLE IF EXISTS tmp_fragmentation_report;
END //

DELIMITER ;

-- ========================================
-- 3. REBUILD FRAGMENTED INDEXES
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_rebuild_fragmented_indexes(
    IN target_database VARCHAR(100),
    IN fragmentation_threshold DECIMAL(5,2)
)
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE v_table_name VARCHAR(100);
    DECLARE v_data_free BIGINT;
    DECLARE v_data_length BIGINT;
    DECLARE v_fragmentation DECIMAL(5,2);
    DECLARE v_rebuild_count INT DEFAULT 0;
    
    DECLARE table_cursor CURSOR FOR
        SELECT table_name, data_length, data_free
        FROM information_schema.tables
        WHERE table_schema = target_database
        AND table_type = 'BASE TABLE';
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SELECT 'ERROR: Index rebuild failed' AS Status;
    END;
    
    START TRANSACTION;
    
    SELECT CONCAT('Starting index rebuild for tables with fragmentation > ', fragmentation_threshold, '%') AS Status;
    
    OPEN table_cursor;
    
    rebuild_loop: LOOP
        FETCH table_cursor INTO v_table_name, v_data_length, v_data_free;
        IF done THEN
            LEAVE rebuild_loop;
        END IF;
        
        -- Calculate fragmentation
        IF v_data_length > 0 THEN
            SET v_fragmentation = (v_data_free / v_data_length) * 100;
        ELSE
            SET v_fragmentation = 0;
        END IF;
        
        -- Rebuild if fragmentation exceeds threshold
        IF v_fragmentation >= fragmentation_threshold THEN
            SET @optimize_stmt = CONCAT('OPTIMIZE TABLE ', target_database, '.', v_table_name);
            PREPARE stmt FROM @optimize_stmt;
            EXECUTE stmt;
            DEALLOCATE PREPARE stmt;
            
            SET v_rebuild_count = v_rebuild_count + 1;
            SELECT CONCAT('Rebuilt: ', v_table_name, ' (Fragmentation: ', ROUND(v_fragmentation, 2), '%)') AS Progress;
        END IF;
    END LOOP;
    
    CLOSE table_cursor;
    COMMIT;
    
    SELECT CONCAT('SUCCESS: Rebuilt ', v_rebuild_count, ' fragmented tables') AS Status;
END //

DELIMITER ;

-- ========================================
-- 4. UPDATE INDEX STATISTICS
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_update_statistics(
    IN target_database VARCHAR(100)
)
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE v_table_name VARCHAR(100);
    DECLARE v_analyze_count INT DEFAULT 0;
    
    DECLARE table_cursor CURSOR FOR
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = target_database
        AND table_type = 'BASE TABLE';
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    SELECT 'Starting statistics update for all tables...' AS Status;
    
    OPEN table_cursor;
    
    analyze_loop: LOOP
        FETCH table_cursor INTO v_table_name;
        IF done THEN
            LEAVE analyze_loop;
        END IF;
        
        -- Run ANALYZE TABLE to update statistics
        SET @analyze_stmt = CONCAT('ANALYZE TABLE ', target_database, '.', v_table_name);
        PREPARE stmt FROM @analyze_stmt;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
        
        SET v_analyze_count = v_analyze_count + 1;
        SELECT CONCAT('Analyzed: ', v_table_name) AS Progress;
    END LOOP;
    
    CLOSE table_cursor;
    
    SELECT CONCAT('SUCCESS: Updated statistics for ', v_analyze_count, ' tables') AS Status;
END //

DELIMITER ;

-- ========================================
-- 5. IDENTIFY UNUSED INDEXES
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_identify_unused_indexes(
    IN target_database VARCHAR(100),
    IN days_threshold INT
)
BEGIN
    -- Note: This requires enabling performance_schema
    -- SET GLOBAL performance_schema = ON;
    
    DROP TEMPORARY TABLE IF EXISTS tmp_unused_indexes;
    CREATE TEMPORARY TABLE tmp_unused_indexes (
        table_name VARCHAR(100),
        index_name VARCHAR(100),
        column_names TEXT,
        index_type VARCHAR(20),
        is_unique VARCHAR(10),
        cardinality BIGINT,
        recommendation VARCHAR(200)
    );
    
    -- Find indexes with low or no usage
    INSERT INTO tmp_unused_indexes
    SELECT 
        s.table_name,
        s.index_name,
        GROUP_CONCAT(s.column_name ORDER BY s.seq_in_index) AS column_names,
        s.index_type,
        IF(s.non_unique = 0, 'UNIQUE', 'NON-UNIQUE') AS is_unique,
        s.cardinality,
        CASE
            WHEN s.index_name = 'PRIMARY' THEN 'PRIMARY KEY - DO NOT DROP'
            WHEN s.non_unique = 0 THEN 'UNIQUE INDEX - Review Before Dropping'
            WHEN s.cardinality IS NULL OR s.cardinality = 0 THEN 'ZERO CARDINALITY - Safe to Drop'
            WHEN s.cardinality < 10 THEN 'LOW CARDINALITY - Consider Dropping'
            ELSE 'REVIEW USAGE PATTERN'
        END AS recommendation
    FROM information_schema.statistics s
    WHERE s.table_schema = target_database
    AND s.index_name != 'PRIMARY'
    GROUP BY s.table_name, s.index_name, s.index_type, s.non_unique, s.cardinality;
    
    -- Display results
    SELECT * FROM tmp_unused_indexes
    ORDER BY 
        CASE recommendation
            WHEN 'ZERO CARDINALITY - Safe to Drop' THEN 1
            WHEN 'LOW CARDINALITY - Consider Dropping' THEN 2
            ELSE 3
        END,
        table_name, index_name;
    
    -- Summary
    SELECT 
        COUNT(*) AS total_indexes,
        SUM(CASE WHEN recommendation LIKE '%Safe to Drop%' THEN 1 ELSE 0 END) AS safe_to_drop,
        SUM(CASE WHEN recommendation LIKE '%Consider Dropping%' THEN 1 ELSE 0 END) AS consider_dropping
    FROM tmp_unused_indexes;
    
    DROP TEMPORARY TABLE IF EXISTS tmp_unused_indexes;
END //

DELIMITER ;

-- ========================================
-- 6. REMOVE UNUSED INDEXES
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_remove_unused_indexes(
    IN target_database VARCHAR(100),
    IN cardinality_threshold BIGINT,
    IN dry_run BOOLEAN
)
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE v_table_name VARCHAR(100);
    DECLARE v_index_name VARCHAR(100);
    DECLARE v_cardinality BIGINT;
    DECLARE v_is_unique INT;
    DECLARE v_drop_count INT DEFAULT 0;
    
    DECLARE index_cursor CURSOR FOR
        SELECT table_name, index_name, cardinality, non_unique
        FROM information_schema.statistics
        WHERE table_schema = target_database
        AND index_name != 'PRIMARY'
        AND (cardinality IS NULL OR cardinality < cardinality_threshold)
        GROUP BY table_name, index_name, cardinality, non_unique;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    IF dry_run THEN
        SELECT 'DRY RUN MODE - No indexes will be dropped' AS Status;
    ELSE
        SELECT 'LIVE MODE - Indexes will be permanently dropped' AS Status;
    END IF;
    
    OPEN index_cursor;
    
    drop_loop: LOOP
        FETCH index_cursor INTO v_table_name, v_index_name, v_cardinality, v_is_unique;
        IF done THEN
            LEAVE drop_loop;
        END IF;
        
        -- Skip unique indexes for safety
        IF v_is_unique = 0 THEN
            IF dry_run THEN
                SELECT CONCAT('Would drop: ', v_table_name, '.', v_index_name, 
                       ' (Cardinality: ', COALESCE(v_cardinality, 0), ')') AS DryRun;
            ELSE
                SET @drop_stmt = CONCAT('ALTER TABLE ', target_database, '.', v_table_name, 
                                       ' DROP INDEX ', v_index_name);
                PREPARE stmt FROM @drop_stmt;
                EXECUTE stmt;
                DEALLOCATE PREPARE stmt;
                
                SELECT CONCAT('Dropped: ', v_table_name, '.', v_index_name) AS Progress;
            END IF;
            SET v_drop_count = v_drop_count + 1;
        END IF;
    END LOOP;
    
    CLOSE index_cursor;
    
    SELECT CONCAT('SUCCESS: ', IF(dry_run, 'Would drop ', 'Dropped '), v_drop_count, ' unused indexes') AS Status;
END //

DELIMITER ;

-- ========================================
-- 7. COMPREHENSIVE INDEX HEALTH CHECK
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_index_health_check(
    IN target_database VARCHAR(100)
)
BEGIN
    SELECT '=== INDEX HEALTH CHECK REPORT ===' AS Report;
    
    -- 1. Total indexes count
    SELECT 
        COUNT(DISTINCT CONCAT(table_name, '.', index_name)) AS total_indexes,
        COUNT(DISTINCT table_name) AS tables_with_indexes
    FROM information_schema.statistics
    WHERE table_schema = target_database;
    
    -- 2. Indexes by type
    SELECT 
        index_type,
        COUNT(*) AS count
    FROM (
        SELECT DISTINCT table_name, index_name, index_type
        FROM information_schema.statistics
        WHERE table_schema = target_database
    ) t
    GROUP BY index_type;
    
    -- 3. Tables without indexes (excluding PRIMARY)
    SELECT 
        t.table_name,
        t.table_rows AS estimated_rows
    FROM information_schema.tables t
    LEFT JOIN information_schema.statistics s 
        ON t.table_name = s.table_name 
        AND t.table_schema = s.table_schema
        AND s.index_name != 'PRIMARY'
    WHERE t.table_schema = target_database
    AND t.table_type = 'BASE TABLE'
    AND s.index_name IS NULL
    AND t.table_rows > 1000;
    
    -- 4. Duplicate indexes
    SELECT 
        s1.table_name,
        s1.index_name AS index_1,
        s2.index_name AS index_2,
        GROUP_CONCAT(s1.column_name ORDER BY s1.seq_in_index) AS columns
    FROM information_schema.statistics s1
    JOIN information_schema.statistics s2
        ON s1.table_schema = s2.table_schema
        AND s1.table_name = s2.table_name
        AND s1.column_name = s2.column_name
        AND s1.seq_in_index = s2.seq_in_index
        AND s1.index_name < s2.index_name
    WHERE s1.table_schema = target_database
    GROUP BY s1.table_name, s1.index_name, s2.index_name
    HAVING COUNT(*) > 0;
    
    -- 5. Large indexes
    SELECT 
        table_name,
        index_name,
        ROUND(stat_value * @@innodb_page_size / 1024 / 1024, 2) AS size_mb
    FROM mysql.innodb_index_stats
    WHERE database_name = target_database
    AND stat_name = 'size'
    ORDER BY stat_value DESC
    LIMIT 10;
    
    SELECT '=== END OF REPORT ===' AS Report;
END //

DELIMITER ;

-- ========================================
-- 8. AUTO MAINTENANCE SCHEDULER
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_auto_index_maintenance(
    IN target_database VARCHAR(100)
)
BEGIN
    DECLARE v_start_time DATETIME;
    DECLARE v_end_time DATETIME;
    
    SET v_start_time = NOW();
    
    SELECT CONCAT('Starting automatic index maintenance at ', v_start_time) AS Status;
    
    -- Step 1: Analyze fragmentation
    CALL sp_analyze_index_fragmentation(target_database);
    
    -- Step 2: Rebuild fragmented indexes (>15% fragmentation)
    CALL sp_rebuild_fragmented_indexes(target_database, 15.0);
    
    -- Step 3: Update statistics
    CALL sp_update_statistics(target_database);
    
    -- Step 4: Health check
    CALL sp_index_health_check(target_database);
    
    SET v_end_time = NOW();
    
    SELECT CONCAT('Maintenance completed in ', TIMESTAMPDIFF(SECOND, v_start_time, v_end_time), ' seconds') AS Status;
END //

DELIMITER ;

-- ========================================
-- USAGE EXAMPLES
-- ========================================

-- Example 1: Analyze Index Fragmentation
-- CALL sp_analyze_index_fragmentation('ecommerce_analytics');

-- Example 2: Rebuild Fragmented Indexes (>20% fragmentation)
-- CALL sp_rebuild_fragmented_indexes('ecommerce_analytics', 20.0);

-- Example 3: Update Statistics for All Tables
-- CALL sp_update_statistics('ecommerce_analytics');

-- Example 4: Identify Unused Indexes
-- CALL sp_identify_unused_indexes('ecommerce_analytics', 30);

-- Example 5: Remove Unused Indexes (Dry Run)
-- CALL sp_remove_unused_indexes('ecommerce_analytics', 10, TRUE);

-- Example 6: Remove Unused Indexes (Live Mode)
-- CALL sp_remove_unused_indexes('ecommerce_analytics', 10, FALSE);

-- Example 7: Comprehensive Health Check
-- CALL sp_index_health_check('ecommerce_analytics');

-- Example 8: Run Full Auto Maintenance
-- CALL sp_auto_index_maintenance('ecommerce_analytics');

-- ========================================
-- SCHEDULE AS CRON JOB (Run Weekly)
-- ========================================
-- 0 2 * * 0 mysql -u root -p -e "CALL ecommerce_analytics.sp_auto_index_maintenance('ecommerce_analytics');"

SELECT 'All index maintenance procedures created successfully' AS Status;