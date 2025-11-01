-- ========================================
-- OPTIMIZATION QUERIES
-- E-commerce Revenue Analytics Engine
-- Query Optimization, Index Suggestions, Performance Tuning
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- PERFORMANCE TRACKING TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS query_performance_log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    query_name VARCHAR(200) NOT NULL,
    query_text TEXT,
    execution_time_ms DECIMAL(10,2),
    rows_examined BIGINT,
    rows_returned INT,
    execution_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    database_name VARCHAR(100),
    user_name VARCHAR(100),
    INDEX idx_query_name (query_name),
    INDEX idx_execution_date (execution_date),
    INDEX idx_execution_time (execution_time_ms)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

DELIMITER $$

-- ========================================
-- 1. ANALYZE TABLE SIZES AND ROW COUNTS
-- ========================================
DROP PROCEDURE IF EXISTS sp_analyze_table_sizes$$
CREATE PROCEDURE sp_analyze_table_sizes()
BEGIN
DECLARE v_cutoff_date DATE;
    

    SELECT 
        TABLE_NAME,
        TABLE_ROWS AS estimated_rows,
        ROUND(DATA_LENGTH / 1024 / 1024, 2) AS data_size_mb,
        ROUND(INDEX_LENGTH / 1024 / 1024, 2) AS index_size_mb,
        ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) AS total_size_mb,
        ROUND(INDEX_LENGTH / DATA_LENGTH * 100, 2) AS index_ratio_percent,
        ENGINE,
        TABLE_COLLATION
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = 'ecommerce_analytics'
    AND TABLE_TYPE = 'BASE TABLE'
    ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC;
END$$

-- ========================================
-- 2. ANALYZE INDEX USAGE
-- ========================================
DROP PROCEDURE IF EXISTS sp_analyze_index_usage$$
CREATE PROCEDURE sp_analyze_index_usage()
BEGIN
    SELECT 
        TABLE_NAME,
        INDEX_NAME,
        SEQ_IN_INDEX,
        COLUMN_NAME,
        CARDINALITY,
        INDEX_TYPE,
        CASE 
            WHEN NON_UNIQUE = 0 THEN 'UNIQUE'
            ELSE 'NON-UNIQUE'
        END AS index_uniqueness,
        CASE 
            WHEN CARDINALITY IS NULL THEN 'Not analyzed'
            WHEN CARDINALITY < 100 THEN 'Low cardinality - Consider removal'
            WHEN CARDINALITY < 1000 THEN 'Medium cardinality'
            ELSE 'Good cardinality'
        END AS cardinality_assessment
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = 'ecommerce_analytics'
    ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX;
END$$

-- ========================================
-- 3. FIND MISSING INDEXES
-- ========================================
DROP PROCEDURE IF EXISTS sp_suggest_missing_indexes$$
CREATE PROCEDURE sp_suggest_missing_indexes()
BEGIN
    -- Suggest indexes based on foreign keys without indexes
    SELECT 
        'Add index on foreign key' AS suggestion_type,
        tc.TABLE_NAME,
        tc.COLUMN_NAME,
        CONCAT('CREATE INDEX idx_', tc.COLUMN_NAME, ' ON ', tc.TABLE_NAME, '(', tc.COLUMN_NAME, ');') AS suggested_index
    FROM information_schema.KEY_COLUMN_USAGE tc
    WHERE tc.TABLE_SCHEMA = 'ecommerce_analytics'
    AND tc.REFERENCED_TABLE_NAME IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.STATISTICS s
        WHERE s.TABLE_SCHEMA = tc.TABLE_SCHEMA
        AND s.TABLE_NAME = tc.TABLE_NAME
        AND s.COLUMN_NAME = tc.COLUMN_NAME
        AND s.SEQ_IN_INDEX = 1
    )
    
    UNION ALL
    
    -- Suggest composite indexes for common query patterns
    SELECT 
        'Composite index for date range queries' AS suggestion_type,
        'orders' AS TABLE_NAME,
        'order_date, status' AS COLUMN_NAME,
        'CREATE INDEX idx_order_date_status ON orders(order_date, status);' AS suggested_index
    
    UNION ALL
    
    SELECT 
        'Composite index for customer orders' AS suggestion_type,
        'orders' AS TABLE_NAME,
        'customer_id, order_date' AS COLUMN_NAME,
        'CREATE INDEX idx_customer_order_date ON orders(customer_id, order_date);' AS suggested_index
    
    UNION ALL
    
    SELECT 
        'Composite index for product sales analysis' AS suggestion_type,
        'order_items' AS TABLE_NAME,
        'product_id, order_id' AS COLUMN_NAME,
        'CREATE INDEX idx_product_order ON order_items(product_id, order_id);' AS suggested_index;
END$$

-- ========================================
-- 4. IDENTIFY DUPLICATE INDEXES
-- ========================================
DROP PROCEDURE IF EXISTS sp_find_duplicate_indexes$$
CREATE PROCEDURE sp_find_duplicate_indexes()
BEGIN
    SELECT 
        s1.TABLE_NAME,
        s1.INDEX_NAME AS index_1,
        s2.INDEX_NAME AS index_2,
        GROUP_CONCAT(s1.COLUMN_NAME ORDER BY s1.SEQ_IN_INDEX) AS columns,
        'Potential duplicate - consider removing one' AS recommendation
    FROM information_schema.STATISTICS s1
    JOIN information_schema.STATISTICS s2 
        ON s1.TABLE_SCHEMA = s2.TABLE_SCHEMA
        AND s1.TABLE_NAME = s2.TABLE_NAME
        AND s1.COLUMN_NAME = s2.COLUMN_NAME
        AND s1.INDEX_NAME < s2.INDEX_NAME
    WHERE s1.TABLE_SCHEMA = 'ecommerce_analytics'
    GROUP BY s1.TABLE_NAME, s1.INDEX_NAME, s2.INDEX_NAME
    HAVING COUNT(*) > 0;
END$$

-- ========================================
-- 5. ANALYZE SLOW QUERIES
-- ========================================
DROP PROCEDURE IF EXISTS sp_analyze_slow_queries$$
CREATE PROCEDURE sp_analyze_slow_queries(
    IN p_days_back INT
)
BEGIN
    












SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL p_days_back DAY);
    
    SELECT 
        query_name,
        COUNT(*) AS execution_count,
        AVG(execution_time_ms) AS avg_time_ms,
        MIN(execution_time_ms) AS min_time_ms,
        MAX(execution_time_ms) AS max_time_ms,
        AVG(rows_examined) AS avg_rows_examined,
        AVG(rows_returned) AS avg_rows_returned,
        CASE 
            WHEN AVG(execution_time_ms) > 5000 THEN 'CRITICAL - Needs immediate optimization'
            WHEN AVG(execution_time_ms) > 1000 THEN 'HIGH - Should be optimized'
            WHEN AVG(execution_time_ms) > 500 THEN 'MEDIUM - Monitor closely'
            ELSE 'LOW - Performance acceptable'
        END AS priority
    FROM query_performance_log
    WHERE execution_date >= v_cutoff_date
    GROUP BY query_name
    ORDER BY AVG(execution_time_ms) DESC;
END$$

-- ========================================
-- 6. TABLE FRAGMENTATION ANALYSIS
-- ========================================
DROP PROCEDURE IF EXISTS sp_analyze_fragmentation$$
CREATE PROCEDURE sp_analyze_fragmentation()
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_table_name VARCHAR(100);
    DECLARE v_fragmentation DECIMAL(10,2);
    
    DECLARE table_cursor CURSOR FOR
        SELECT TABLE_NAME, ROUND(DATA_FREE / DATA_LENGTH * 100, 2) AS frag_pct
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = 'ecommerce_analytics'
        AND DATA_LENGTH > 0
        AND (DATA_FREE / DATA_LENGTH * 100) > 10
        AND TABLE_TYPE = 'BASE TABLE';
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    SELECT 
        TABLE_NAME,
        ROUND(DATA_LENGTH / 1024 / 1024, 2) AS data_size_mb,
        ROUND(DATA_FREE / 1024 / 1024, 2) AS fragmented_mb,
        ROUND(DATA_FREE / DATA_LENGTH * 100, 2) AS fragmentation_percent,
        CASE 
            WHEN (DATA_FREE / DATA_LENGTH * 100) > 25 THEN CONCAT('OPTIMIZE TABLE ', TABLE_NAME, ';')
            WHEN (DATA_FREE / DATA_LENGTH * 100) > 10 THEN 'Consider optimization'
            ELSE 'Fragmentation acceptable'
        END AS recommendation
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = 'ecommerce_analytics'
    AND DATA_LENGTH > 0
    AND TABLE_TYPE = 'BASE TABLE'
    ORDER BY fragmentation_percent DESC;
END$$

-- ========================================
-- 7. OPTIMIZE FREQUENTLY ACCESSED TABLES
-- ========================================
DROP PROCEDURE IF EXISTS sp_optimize_hot_tables$$
CREATE PROCEDURE sp_optimize_hot_tables()
BEGIN
    












SET done = TRUE;
    
    OPEN table_cursor;
    
    read_loop: LOOP
        FETCH table_cursor INTO v_table_name, v_fragmentation;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        SET @optimize_sql = CONCAT('OPTIMIZE TABLE ', v_table_name);
        PREPARE stmt FROM @optimize_sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
        
        SELECT CONCAT('Optimized table: ', v_table_name, ' (fragmentation: ', v_fragmentation, '%)') AS status;
    END LOOP;
    
    CLOSE table_cursor;
    
    SELECT 'All fragmented tables optimized' AS final_status;
END$$

-- ========================================
-- 8. QUERY EXECUTION PLAN ANALYZER
-- ========================================
DROP PROCEDURE IF EXISTS sp_explain_query$$
CREATE PROCEDURE sp_explain_query(
    IN p_query TEXT
)
BEGIN
    -- This procedure helps analyze query execution plans
    SET @explain_query = CONCAT('EXPLAIN ', p_query);
    PREPARE stmt FROM @explain_query;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
END$$

-- ========================================
-- 9. RECOMMEND COMPOSITE INDEXES
-- ========================================
DROP PROCEDURE IF EXISTS sp_recommend_composite_indexes$$
CREATE PROCEDURE sp_recommend_composite_indexes()
BEGIN
    -- Recommendations based on common query patterns
    SELECT 
        'orders' AS table_name,
        'customer_id, order_date, status' AS columns,
        'CREATE INDEX idx_customer_date_status ON orders(customer_id, order_date, status);' AS create_statement,
        'Optimizes customer order history queries with date ranges and status filters' AS benefit
    
    UNION ALL
    
    SELECT 
        'order_items' AS table_name,
        'order_id, product_id, quantity' AS columns,
        'CREATE INDEX idx_order_product_qty ON order_items(order_id, product_id, quantity);' AS create_statement,
        'Optimizes order detail queries and product quantity aggregations' AS benefit
    
    UNION ALL
    
    SELECT 
        'products' AS table_name,
        'category_id, status, price' AS columns,
        'CREATE INDEX idx_category_status_price ON products(category_id, status, price);' AS create_statement,
        'Optimizes product catalog queries with category and price filtering' AS benefit
    
    UNION ALL
    
    SELECT 
        'reviews' AS table_name,
        'product_id, status, rating' AS columns,
        'CREATE INDEX idx_product_status_rating ON reviews(product_id, status, rating);' AS create_statement,
        'Optimizes product review queries and rating calculations' AS benefit
    
    UNION ALL
    
    SELECT 
        'campaign_performance' AS table_name,
        'campaign_id, report_date' AS columns,
        'CREATE INDEX idx_campaign_date ON campaign_performance(campaign_id, report_date);' AS create_statement,
        'Optimizes campaign reporting queries with date ranges' AS benefit
    
    UNION ALL
    
    SELECT 
        'inventory' AS table_name,
        'product_id, warehouse_id' AS columns,
        'CREATE INDEX idx_product_warehouse ON inventory(product_id, warehouse_id);' AS create_statement,
        'Optimizes inventory lookup by product and warehouse' AS benefit;
END$$

-- ========================================
-- 10. COVERING INDEX SUGGESTIONS
-- ========================================
DROP PROCEDURE IF EXISTS sp_suggest_covering_indexes$$
CREATE PROCEDURE sp_suggest_covering_indexes()
BEGIN
    -- Covering indexes include all columns needed for a query
    SELECT 
        'High-value customer query' AS query_type,
        'customers' AS table_name,
        'CREATE INDEX idx_customer_status_cover ON customers(status, customer_id, email, first_name, last_name);' AS create_statement,
        'Covers: SELECT customer_id, email, first_name, last_name FROM customers WHERE status = "active"' AS covered_query
    
    UNION ALL
    
    SELECT 
        'Product availability query' AS query_type,
        'products' AS table_name,
        'CREATE INDEX idx_product_status_cover ON products(status, product_id, product_name, price, stock_quantity);' AS create_statement,
        'Covers: SELECT product_id, product_name, price, stock_quantity FROM products WHERE status = "active"' AS covered_query
    
    UNION ALL
    
    SELECT 
        'Order summary query' AS query_type,
        'orders' AS table_name,
        'CREATE INDEX idx_order_customer_cover ON orders(customer_id, order_id, order_date, total_amount, status);' AS create_statement,
        'Covers: SELECT order_id, order_date, total_amount, status FROM orders WHERE customer_id = ?' AS covered_query;
END$$

-- ========================================
-- 11. PARTITION RECOMMENDATIONS
-- ========================================
DROP PROCEDURE IF EXISTS sp_partition_recommendations$$
CREATE PROCEDURE sp_partition_recommendations()
BEGIN
    SELECT 
        'orders' AS table_name,
        'Range partitioning by order_date' AS partition_type,
        'Partition by year or quarter for historical data' AS recommendation,
        'ALTER TABLE orders PARTITION BY RANGE (YEAR(order_date)) (...);' AS example_syntax,
        'Improves query performance for date-range queries and simplifies archiving' AS benefit
    
    UNION ALL
    
    SELECT 
        'campaign_performance' AS table_name,
        'Range partitioning by report_date' AS partition_type,
        'Partition by month for time-series data' AS recommendation,
        'ALTER TABLE campaign_performance PARTITION BY RANGE (TO_DAYS(report_date)) (...);' AS example_syntax,
        'Enables efficient data pruning for date-based analytics' AS benefit
    
    UNION ALL
    
    SELECT 
        'order_items' AS table_name,
        'Hash partitioning by order_id' AS partition_type,
        'Distribute data evenly across partitions' AS recommendation,
        'ALTER TABLE order_items PARTITION BY HASH(order_id) PARTITIONS 8;' AS example_syntax,
        'Balances load and improves parallel query execution' AS benefit;
END$$

-- ========================================
-- 12. QUERY CACHE ANALYSIS
-- ========================================
DROP PROCEDURE IF EXISTS sp_query_cache_stats$$
CREATE PROCEDURE sp_query_cache_stats()
BEGIN
    -- Note: Query cache deprecated in MySQL 8.0+
    SELECT 
        'Query Cache' AS metric,
        @@query_cache_size AS cache_size_bytes,
        CASE 
            WHEN @@query_cache_type = 0 THEN 'OFF'
            WHEN @@query_cache_type = 1 THEN 'ON'
            WHEN @@query_cache_type = 2 THEN 'DEMAND'
        END AS cache_status,
        'Note: Query cache deprecated in MySQL 8.0+' AS note;
END$$

-- ========================================
-- 13. BUFFER POOL USAGE ANALYSIS
-- ========================================
DROP PROCEDURE IF EXISTS sp_buffer_pool_analysis$$
CREATE PROCEDURE sp_buffer_pool_analysis()
BEGIN
    SELECT 
        'InnoDB Buffer Pool Size' AS metric,
        @@innodb_buffer_pool_size / 1024 / 1024 / 1024 AS size_gb,
        'Recommended: 70-80% of total RAM' AS recommendation;
    
    -- Show buffer pool stats if available
    SELECT 
        POOL_ID,
        POOL_SIZE AS pages,
        ROUND(POOL_SIZE * 16 / 1024, 2) AS size_mb,
        FREE_BUFFERS,
        DATABASE_PAGES,
        OLD_DATABASE_PAGES
    FROM information_schema.INNODB_BUFFER_POOL_STATS;
END$$

-- ========================================
-- 14. LONG-RUNNING QUERY DETECTOR
-- ========================================
DROP PROCEDURE IF EXISTS sp_find_long_running_queries$$
CREATE PROCEDURE sp_find_long_running_queries(
    IN p_min_seconds INT
)
BEGIN
    SELECT 
        ID,
        USER,
        HOST,
        DB,
        COMMAND,
        TIME AS seconds_running,
        STATE,
        LEFT(INFO, 100) AS query_preview,
        CASE 
            WHEN TIME > 300 THEN 'CRITICAL - Consider killing'
            WHEN TIME > 60 THEN 'HIGH - Monitor closely'
            ELSE 'MEDIUM - Track performance'
        END AS priority
    FROM information_schema.PROCESSLIST
    WHERE COMMAND != 'Sleep'
    AND TIME > p_min_seconds
    ORDER BY TIME DESC;
END$$

-- ========================================
-- 15. COMPREHENSIVE OPTIMIZATION REPORT
-- ========================================
DROP PROCEDURE IF EXISTS sp_optimization_report$$
CREATE PROCEDURE sp_optimization_report()
BEGIN
    -- Section 1: Table Sizes
    SELECT '========== TABLE SIZE ANALYSIS ==========' AS section;
    CALL sp_analyze_table_sizes();
    
    -- Section 2: Missing Indexes
    SELECT '========== MISSING INDEX SUGGESTIONS ==========' AS section;
    CALL sp_suggest_missing_indexes();
    
    -- Section 3: Composite Index Recommendations
    SELECT '========== COMPOSITE INDEX RECOMMENDATIONS ==========' AS section;
    CALL sp_recommend_composite_indexes();
    
    -- Section 4: Fragmentation
    SELECT '========== FRAGMENTATION ANALYSIS ==========' AS section;
    CALL sp_analyze_fragmentation();
    
    -- Section 5: Configuration
    SELECT '========== KEY CONFIGURATION SETTINGS ==========' AS section;
    SELECT 
        'innodb_buffer_pool_size' AS setting,
        @@innodb_buffer_pool_size / 1024 / 1024 / 1024 AS value_gb,
        'Recommended: 70-80% of RAM' AS recommendation
    UNION ALL
    SELECT 
        'max_connections' AS setting,
        @@max_connections AS value_gb,
        'Adjust based on concurrent user load' AS recommendation
    UNION ALL
    SELECT 
        'innodb_log_file_size' AS setting,
        @@innodb_log_file_size / 1024 / 1024 AS value_gb,
        'Larger values improve write performance' AS recommendation;
END$$

-- ========================================
-- 16. LOG QUERY PERFORMANCE
-- ========================================
DROP PROCEDURE IF EXISTS sp_log_query_performance$$
CREATE PROCEDURE sp_log_query_performance(
    IN p_query_name VARCHAR(200),
    IN p_query_text TEXT,
    IN p_execution_time_ms DECIMAL(10,2),
    IN p_rows_examined BIGINT,
    IN p_rows_returned INT
)
BEGIN
    INSERT INTO query_performance_log 
        (query_name, query_text, execution_time_ms, rows_examined, rows_returned, database_name, user_name)
    VALUES 
        (p_query_name, p_query_text, p_execution_time_ms, p_rows_examined, p_rows_returned, 
         DATABASE(), USER());
    
    SELECT 'Query performance logged successfully' AS status;
END$$

-- ========================================
-- 17. OPTIMAL JOIN ORDER SUGGESTIONS
-- ========================================
DROP PROCEDURE IF EXISTS sp_suggest_join_optimization$$
CREATE PROCEDURE sp_suggest_join_optimization()
BEGIN
    SELECT 
        'Order of JOINs matters' AS principle,
        '1. Start with the table that filters the most rows (smallest result set)' AS rule_1,
        '2. Use INNER JOIN instead of WHERE for join conditions' AS rule_2,
        '3. Ensure all JOIN columns are indexed' AS rule_3,
        '4. Use STRAIGHT_JOIN only when you know the optimal order' AS rule_4;
    
    -- Example optimizations
    SELECT 
        'Customer orders with items' AS query_type,
        'Good: FROM orders o INNER JOIN customers c ... INNER JOIN order_items oi ...' AS good_practice,
        'Bad: FROM customers c INNER JOIN orders o ... (if filtering by order_date)' AS bad_practice,
        'Start with orders if filtering by date; start with customers if filtering by customer attributes' AS guidance;
END$$

DELIMITER ;

-- ========================================
-- APPLY RECOMMENDED INDEXES
-- ========================================

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_customer_date_status ON orders(customer_id, order_date, status);
CREATE INDEX IF NOT EXISTS idx_order_product ON order_items(product_id, order_id);
CREATE INDEX IF NOT EXISTS idx_category_status_price ON products(category_id, status, price);
CREATE INDEX IF NOT EXISTS idx_product_status_rating ON reviews(product_id, status, rating);
CREATE INDEX IF NOT EXISTS idx_campaign_date ON campaign_performance(campaign_id, report_date);

-- Covering indexes for frequently accessed queries
CREATE INDEX IF NOT EXISTS idx_order_customer_summary ON orders(customer_id, order_id, order_date, total_amount, status);

-- ========================================
-- USAGE EXAMPLES
-- ========================================

-- Analyze table sizes and growth
-- CALL sp_analyze_table_sizes();

-- Check index usage
-- CALL sp_analyze_index_usage();

-- Get missing index suggestions
-- CALL sp_suggest_missing_indexes();

-- Find duplicate indexes
-- CALL sp_find_duplicate_indexes();

-- Analyze slow queries from last 7 days
-- CALL sp_analyze_slow_queries(7);

-- Check table fragmentation
-- CALL sp_analyze_fragmentation();

-- Optimize fragmented tables
-- CALL sp_optimize_hot_tables();

-- Get composite index recommendations
-- CALL sp_recommend_composite_indexes();

-- Get covering index suggestions
-- CALL sp_suggest_covering_indexes();

-- Get partition recommendations
-- CALL sp_partition_recommendations();

-- Find long-running queries (> 10 seconds)
-- CALL sp_find_long_running_queries(10);

-- Generate comprehensive optimization report
-- CALL sp_optimization_report();

-- Log a query's performance
-- CALL sp_log_query_performance('customer_orders_query', 'SELECT ...', 125.50, 1000, 50);

-- Get join optimization suggestions
-- CALL sp_suggest_join_optimization();

SELECT 'All optimization procedures and indexes created successfully' AS Status;