-- ========================================
-- DATA ARCHIVING SCRIPTS
-- E-commerce Revenue Analytics Engine
-- Archive Strategy, Move to Archive Tables, Retention Policies
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CREATE ARCHIVE TRACKING TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS archive_log (
    archive_id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100),
    archive_type ENUM('full', 'partial', 'date_range') NOT NULL,
    records_archived INT,
    archive_date_from DATE,
    archive_date_to DATE,
    archive_size_mb DECIMAL(10,2),
    status ENUM('in_progress', 'completed', 'failed') DEFAULT 'in_progress',
    error_message TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    INDEX idx_table_name (table_name),
    INDEX idx_status (status),
    INDEX idx_started_at (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- 2. CREATE DATA RETENTION POLICY TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS data_retention_policy (
    policy_id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100) UNIQUE,
    retention_days INT NOT NULL,
    archive_enabled BOOLEAN DEFAULT TRUE,
    delete_after_archive BOOLEAN DEFAULT FALSE,
    last_archive_date DATE,
    next_archive_date DATE,
    status ENUM('active', 'paused', 'disabled') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_table_name (table_name),
    INDEX idx_next_archive_date (next_archive_date),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- 3. CREATE ARCHIVE TABLES
-- ========================================

-- Archive table for orders
CREATE TABLE IF NOT EXISTS orders_archive (
    archive_record_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT,
    customer_id INT,
    order_date TIMESTAMP,
    total_amount DECIMAL(10,2),
    status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled'),
    payment_status ENUM('pending', 'paid', 'failed', 'refunded'),
    shipping_cost DECIMAL(10,2),
    tax_amount DECIMAL(10,2),
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_created_at TIMESTAMP,
    INDEX idx_order_id (order_id),
    INDEX idx_customer_id (customer_id),
    INDEX idx_order_date (order_date),
    INDEX idx_archived_at (archived_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Archive table for order items
CREATE TABLE IF NOT EXISTS order_items_archive (
    archive_record_id INT PRIMARY KEY AUTO_INCREMENT,
    order_item_id INT,
    order_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    discount DECIMAL(10,2),
    subtotal DECIMAL(10,2),
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_created_at TIMESTAMP,
    INDEX idx_order_item_id (order_item_id),
    INDEX idx_order_id (order_id),
    INDEX idx_archived_at (archived_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Archive table for campaign performance
CREATE TABLE IF NOT EXISTS campaign_performance_archive (
    archive_record_id INT PRIMARY KEY AUTO_INCREMENT,
    performance_id INT,
    campaign_id INT,
    report_date DATE,
    impressions INT,
    clicks INT,
    conversions INT,
    spend DECIMAL(10,2),
    revenue DECIMAL(10,2),
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_created_at TIMESTAMP,
    INDEX idx_campaign_id (campaign_id),
    INDEX idx_report_date (report_date),
    INDEX idx_archived_at (archived_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Archive table for reviews
CREATE TABLE IF NOT EXISTS reviews_archive (
    archive_record_id INT PRIMARY KEY AUTO_INCREMENT,
    review_id INT,
    product_id INT,
    customer_id INT,
    rating INT,
    review_title VARCHAR(200),
    review_comment TEXT,
    is_verified_purchase BOOLEAN,
    helpful_count INT,
    status ENUM('pending', 'approved', 'rejected'),
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_created_at TIMESTAMP,
    INDEX idx_review_id (review_id),
    INDEX idx_product_id (product_id),
    INDEX idx_archived_at (archived_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Archive table for returns
CREATE TABLE IF NOT EXISTS returns_archive (
    archive_record_id INT PRIMARY KEY AUTO_INCREMENT,
    return_id INT,
    order_id INT,
    order_item_id INT,
    reason ENUM('defective', 'wrong_item', 'not_as_described', 'changed_mind', 'other'),
    reason_details TEXT,
    status ENUM('requested', 'approved', 'rejected', 'received', 'refunded'),
    refund_amount DECIMAL(10,2),
    refund_method ENUM('original_payment', 'store_credit'),
    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    original_created_at TIMESTAMP,
    INDEX idx_return_id (return_id),
    INDEX idx_order_id (order_id),
    INDEX idx_archived_at (archived_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- 4. INITIALIZE RETENTION POLICIES
-- ========================================
INSERT INTO data_retention_policy (table_name, retention_days, archive_enabled, delete_after_archive) VALUES
('orders', 1825, TRUE, FALSE),              -- 5 years, archive but keep
('order_items', 1825, TRUE, FALSE),         -- 5 years, archive but keep
('campaign_performance', 730, TRUE, TRUE),  -- 2 years, archive and delete
('reviews', 1095, TRUE, FALSE),             -- 3 years, archive but keep
('returns', 1095, TRUE, FALSE),             -- 3 years, archive but keep
('customers', 2555, TRUE, FALSE),           -- 7 years (legal requirement)
('payments', 2555, TRUE, FALSE)             -- 7 years (legal requirement)
ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP;

-- ========================================
-- 5. ARCHIVE ORDERS BY DATE RANGE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_archive_orders_by_date(
    IN archive_from_date DATE,
    IN archive_to_date DATE,
    IN delete_after_archive BOOLEAN
)
BEGIN
    DECLARE v_records_archived INT DEFAULT 0;
    DECLARE v_archive_log_id INT;
    DECLARE v_start_time DATETIME;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        UPDATE archive_log 
        SET status = 'failed', 
            error_message = 'Archive failed due to SQL exception',
            completed_at = NOW()
        WHERE archive_id = v_archive_log_id;
        SELECT 'ERROR: Archive failed' AS Status;
    END;
    
    SET v_start_time = NOW();
    
    -- Create log entry
    INSERT INTO archive_log (table_name, archive_type, archive_date_from, archive_date_to, status)
    VALUES ('orders', 'date_range', archive_from_date, archive_to_date, 'in_progress');
    SET v_archive_log_id = LAST_INSERT_ID();
    
    START TRANSACTION;
    
    -- Archive order items first (maintain referential integrity)
    INSERT INTO order_items_archive (
        order_item_id, order_id, product_id, quantity, unit_price, 
        discount, subtotal, original_created_at
    )
    SELECT 
        oi.order_item_id, oi.order_id, oi.product_id, oi.quantity, oi.unit_price,
        oi.discount, oi.subtotal, oi.created_at
    FROM order_items oi
    INNER JOIN orders o ON oi.order_id = o.order_id
    WHERE DATE(o.order_date) BETWEEN archive_from_date AND archive_to_date
    AND o.status IN ('delivered', 'cancelled');
    
    -- Archive orders
    INSERT INTO orders_archive (
        order_id, customer_id, order_date, total_amount, status,
        payment_status, shipping_cost, tax_amount, original_created_at
    )
    SELECT 
        order_id, customer_id, order_date, total_amount, status,
        payment_status, shipping_cost, tax_amount, created_at
    FROM orders
    WHERE DATE(order_date) BETWEEN archive_from_date AND archive_to_date
    AND status IN ('delivered', 'cancelled');
    
    SET v_records_archived = ROW_COUNT();
    
    -- Delete if requested
    IF delete_after_archive THEN
        DELETE oi FROM order_items oi
        INNER JOIN orders o ON oi.order_id = o.order_id
        WHERE DATE(o.order_date) BETWEEN archive_from_date AND archive_to_date
        AND o.status IN ('delivered', 'cancelled');
        
        DELETE FROM orders
        WHERE DATE(order_date) BETWEEN archive_from_date AND archive_to_date
        AND status IN ('delivered', 'cancelled');
    END IF;
    
    -- Update log
    UPDATE archive_log 
    SET status = 'completed',
        records_archived = v_records_archived,
        completed_at = NOW()
    WHERE archive_id = v_archive_log_id;
    
    COMMIT;
    
    SELECT CONCAT('SUCCESS: Archived ', v_records_archived, ' orders from ', 
                  archive_from_date, ' to ', archive_to_date) AS Status;
END //

DELIMITER ;

-- ========================================
-- 6. ARCHIVE CAMPAIGN PERFORMANCE DATA
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_archive_campaign_performance(
    IN days_to_keep INT,
    IN delete_after_archive BOOLEAN
)
BEGIN
    DECLARE v_records_archived INT DEFAULT 0;
    DECLARE v_archive_log_id INT;
    DECLARE v_cutoff_date DATE;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        UPDATE archive_log 
        SET status = 'failed', 
            error_message = 'Archive failed',
            completed_at = NOW()
        WHERE archive_id = v_archive_log_id;
    END;
    
    SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL days_to_keep DAY);
    
    INSERT INTO archive_log (table_name, archive_type, archive_date_to, status)
    VALUES ('campaign_performance', 'partial', v_cutoff_date, 'in_progress');
    SET v_archive_log_id = LAST_INSERT_ID();
    
    START TRANSACTION;
    
    -- Archive old campaign performance data
    INSERT INTO campaign_performance_archive (
        performance_id, campaign_id, report_date, impressions, clicks,
        conversions, spend, revenue, original_created_at
    )
    SELECT 
        performance_id, campaign_id, report_date, impressions, clicks,
        conversions, spend, revenue, created_at
    FROM campaign_performance
    WHERE report_date < v_cutoff_date;
    
    SET v_records_archived = ROW_COUNT();
    
    -- Delete if requested
    IF delete_after_archive THEN
        DELETE FROM campaign_performance
        WHERE report_date < v_cutoff_date;
    END IF;
    
    UPDATE archive_log 
    SET status = 'completed',
        records_archived = v_records_archived,
        completed_at = NOW()
    WHERE archive_id = v_archive_log_id;
    
    COMMIT;
    
    SELECT CONCAT('SUCCESS: Archived ', v_records_archived, ' campaign performance records') AS Status;
END //

DELIMITER ;

-- ========================================
-- 7. ARCHIVE REVIEWS
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_archive_old_reviews(
    IN days_to_keep INT,
    IN delete_after_archive BOOLEAN
)
BEGIN
    DECLARE v_records_archived INT DEFAULT 0;
    DECLARE v_archive_log_id INT;
    DECLARE v_cutoff_date DATE;
    
    SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL days_to_keep DAY);
    
    INSERT INTO archive_log (table_name, archive_type, archive_date_to, status)
    VALUES ('reviews', 'partial', v_cutoff_date, 'in_progress');
    SET v_archive_log_id = LAST_INSERT_ID();
    
    START TRANSACTION;
    
    INSERT INTO reviews_archive (
        review_id, product_id, customer_id, rating, review_title,
        review_comment, is_verified_purchase, helpful_count, status, original_created_at
    )
    SELECT 
        review_id, product_id, customer_id, rating, review_title,
        review_comment, is_verified_purchase, helpful_count, status, created_at
    FROM reviews
    WHERE DATE(created_at) < v_cutoff_date
    AND status IN ('approved', 'rejected');
    
    SET v_records_archived = ROW_COUNT();
    
    IF delete_after_archive THEN
        DELETE FROM reviews
        WHERE DATE(created_at) < v_cutoff_date
        AND status IN ('approved', 'rejected');
    END IF;
    
    UPDATE archive_log 
    SET status = 'completed',
        records_archived = v_records_archived,
        completed_at = NOW()
    WHERE archive_id = v_archive_log_id;
    
    COMMIT;
    
    SELECT CONCAT('SUCCESS: Archived ', v_records_archived, ' reviews') AS Status;
END //

DELIMITER ;

-- ========================================
-- 8. ARCHIVE RETURNS
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_archive_old_returns(
    IN days_to_keep INT,
    IN delete_after_archive BOOLEAN
)
BEGIN
    DECLARE v_records_archived INT DEFAULT 0;
    DECLARE v_archive_log_id INT;
    DECLARE v_cutoff_date DATE;
    
    SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL days_to_keep DAY);
    
    INSERT INTO archive_log (table_name, archive_type, archive_date_to, status)
    VALUES ('returns', 'partial', v_cutoff_date, 'in_progress');
    SET v_archive_log_id = LAST_INSERT_ID();
    
    START TRANSACTION;
    
    INSERT INTO returns_archive (
        return_id, order_id, order_item_id, reason, reason_details,
        status, refund_amount, refund_method, original_created_at
    )
    SELECT 
        return_id, order_id, order_item_id, reason, reason_details,
        status, refund_amount, refund_method, created_at
    FROM returns
    WHERE DATE(created_at) < v_cutoff_date
    AND status IN ('refunded', 'rejected');
    
    SET v_records_archived = ROW_COUNT();
    
    IF delete_after_archive THEN
        DELETE FROM returns
        WHERE DATE(created_at) < v_cutoff_date
        AND status IN ('refunded', 'rejected');
    END IF;
    
    UPDATE archive_log 
    SET status = 'completed',
        records_archived = v_records_archived,
        completed_at = NOW()
    WHERE archive_id = v_archive_log_id;
    
    COMMIT;
    
    SELECT CONCAT('SUCCESS: Archived ', v_records_archived, ' returns') AS Status;
END //

DELIMITER ;

-- ========================================
-- 9. AUTO ARCHIVE BY RETENTION POLICY
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_auto_archive_by_policy()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE v_table_name VARCHAR(100);
    DECLARE v_retention_days INT;
    DECLARE v_delete_after BOOLEAN;
    DECLARE v_cutoff_date DATE;
    
    DECLARE policy_cursor CURSOR FOR
        SELECT table_name, retention_days, delete_after_archive
        FROM data_retention_policy
        WHERE status = 'active'
        AND archive_enabled = TRUE
        AND (next_archive_date IS NULL OR next_archive_date <= CURDATE());
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    SELECT 'Starting automatic archive based on retention policies...' AS Status;
    
    OPEN policy_cursor;
    
    archive_loop: LOOP
        FETCH policy_cursor INTO v_table_name, v_retention_days, v_delete_after;
        IF done THEN
            LEAVE archive_loop;
        END IF;
        
        SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL v_retention_days DAY);
        
        -- Archive based on table name
        CASE v_table_name
            WHEN 'orders' THEN
                CALL sp_archive_orders_by_date(
                    DATE_SUB(v_cutoff_date, INTERVAL 1 YEAR),
                    v_cutoff_date,
                    v_delete_after
                );
            WHEN 'campaign_performance' THEN
                CALL sp_archive_campaign_performance(v_retention_days, v_delete_after);
            WHEN 'reviews' THEN
                CALL sp_archive_old_reviews(v_retention_days, v_delete_after);
            WHEN 'returns' THEN
                CALL sp_archive_old_returns(v_retention_days, v_delete_after);
            ELSE
                SELECT CONCAT('No archive procedure defined for ', v_table_name) AS Warning;
        END CASE;
        
        -- Update policy next archive date
        UPDATE data_retention_policy
        SET last_archive_date = CURDATE(),
            next_archive_date = DATE_ADD(CURDATE(), INTERVAL 30 DAY)
        WHERE table_name = v_table_name;
        
        SELECT CONCAT('Archived: ', v_table_name) AS Progress;
    END LOOP;
    
    CLOSE policy_cursor;
    
    SELECT 'Automatic archive completed' AS Status;
END //

DELIMITER ;

-- ========================================
-- 10. GENERATE ARCHIVE REPORT
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_generate_archive_report(
    IN days_back INT
)
BEGIN
    DECLARE v_report_date DATE;
    SET v_report_date = DATE_SUB(CURDATE(), INTERVAL days_back DAY);
    
    SELECT '=== ARCHIVE REPORT ===' AS Report;
    
    -- 1. Archive activity summary
    SELECT 
        table_name,
        COUNT(*) AS 'Archive Operations',
        SUM(records_archived) AS 'Total Records Archived',
        ROUND(SUM(archive_size_mb), 2) AS 'Total Size (MB)',
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS 'Successful',
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS 'Failed'
    FROM archive_log
    WHERE DATE(started_at) >= v_report_date
    GROUP BY table_name;
    
    -- 2. Recent archive operations
    SELECT 
        archive_id,
        table_name,
        archive_type,
        records_archived,
        archive_date_from,
        archive_date_to,
        status,
        TIMESTAMPDIFF(SECOND, started_at, completed_at) AS 'Duration (seconds)',
        started_at
    FROM archive_log
    WHERE DATE(started_at) >= v_report_date
    ORDER BY started_at DESC
    LIMIT 20;
    
    -- 3. Current retention policies
    SELECT 
        table_name,
        retention_days,
        ROUND(retention_days / 365, 1) AS 'Retention Years',
        archive_enabled,
        delete_after_archive,
        last_archive_date,
        next_archive_date,
        status
    FROM data_retention_policy
    ORDER BY retention_days DESC;
    
    -- 4. Archive table sizes
    SELECT 
        table_name,
        table_rows AS 'Archived Records',
        ROUND(data_length / 1024 / 1024, 2) AS 'Data Size (MB)',
        ROUND(index_length / 1024 / 1024, 2) AS 'Index Size (MB)',
        ROUND((data_length + index_length) / 1024 / 1024, 2) AS 'Total Size (MB)'
    FROM information_schema.tables
    WHERE table_schema = 'ecommerce_analytics'
    AND table_name LIKE '%_archive'
    ORDER BY (data_length + index_length) DESC;
    
    -- 5. Failed archives
    SELECT 
        archive_id,
        table_name,
        error_message,
        started_at
    FROM archive_log
    WHERE status = 'failed'
    AND DATE(started_at) >= v_report_date;
    
    SELECT '=== END OF REPORT ===' AS Report;
END //

DELIMITER ;

-- ========================================
-- 11. RESTORE FROM ARCHIVE
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_restore_from_archive(
    IN source_table VARCHAR(100),
    IN restore_from_date DATE,
    IN restore_to_date DATE
)
BEGIN
    DECLARE v_records_restored INT DEFAULT 0;
    DECLARE v_archive_table VARCHAR(100);
    
    SET v_archive_table = CONCAT(source_table, '_archive');
    
    SELECT CONCAT('Restoring data from ', v_archive_table, ' to ', source_table) AS Status;
    
    START TRANSACTION;
    
    -- Restore based on table type
    CASE source_table
        WHEN 'orders' THEN
            INSERT IGNORE INTO orders (
                order_id, customer_id, order_date, total_amount, status,
                payment_status, shipping_cost, tax_amount, created_at
            )
            SELECT 
                order_id, customer_id, order_date, total_amount, status,
                payment_status, shipping_cost, tax_amount, original_created_at
            FROM orders_archive
            WHERE DATE(order_date) BETWEEN restore_from_date AND restore_to_date;
            
            SET v_records_restored = ROW_COUNT();
            
        WHEN 'reviews' THEN
            INSERT IGNORE INTO reviews (
                review_id, product_id, customer_id, rating, review_title,
                review_comment, is_verified_purchase, helpful_count, status, created_at
            )
            SELECT 
                review_id, product_id, customer_id, rating, review_title,
                review_comment, is_verified_purchase, helpful_count, status, original_created_at
            FROM reviews_archive
            WHERE DATE(original_created_at) BETWEEN restore_from_date AND restore_to_date;
            
            SET v_records_restored = ROW_COUNT();
    END CASE;
    
    COMMIT;
    
    SELECT CONCAT('SUCCESS: Restored ', v_records_restored, ' records') AS Status;
END //

DELIMITER ;

-- ========================================
-- 12. PURGE OLD ARCHIVE DATA
-- ========================================
DELIMITER //

CREATE PROCEDURE sp_purge_old_archives(
    IN archive_table VARCHAR(100),
    IN days_to_keep INT
)
BEGIN
    DECLARE v_records_deleted INT DEFAULT 0;
    DECLARE v_cutoff_date DATE;
    
    SET v_cutoff_date = DATE_SUB(CURDATE(), INTERVAL days_to_keep DAY);
    
    START TRANSACTION;
    
    SET @purge_stmt = CONCAT(
        'DELETE FROM ', archive_table,
        ' WHERE DATE(archived_at) < "', v_cutoff_date, '"'
    );
    PREPARE stmt FROM @purge_stmt;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    SET v_records_deleted = ROW_COUNT();
    
    COMMIT;
    
    SELECT CONCAT('SUCCESS: Purged ', v_records_deleted, ' old archive records from ', archive_table) AS Status;
END //

DELIMITER ;

-- ========================================
-- USAGE EXAMPLES
-- ========================================

-- Example 1: Archive Orders from Specific Date Range
-- CALL sp_archive_orders_by_date('2023-01-01', '2023-12-31', FALSE);

-- Example 2: Archive Campaign Performance (older than 2 years)
-- CALL sp_archive_campaign_performance(730, TRUE);

-- Example 3: Archive Old Reviews (older than 3 years)
-- CALL sp_archive_old_reviews(1095, FALSE);

-- Example 4: Archive Old Returns (older than 3 years)
-- CALL sp_archive_old_returns(1095, FALSE);

-- Example 5: Run Auto Archive Based on Policies
-- CALL sp_auto_archive_by_policy();

-- Example 6: Generate Archive Report (Last 30 Days)
-- CALL sp_generate_archive_report(30);

-- Example 7: Restore from Archive
-- CALL sp_restore_from_archive('orders', '2023-01-01', '2023-01-31');

-- Example 8: Purge Old Archive Data (older than 10 years)
-- CALL sp_purge_old_archives('orders_archive', 3650);

-- ========================================
-- SCHEDULE AS CRON JOBS
-- ========================================

-- Monthly archive (1st day of month, 1 AM):
-- 0 1 1 * * mysql -u root -p -e "CALL ecommerce_analytics.sp_auto_archive_by_policy();"

-- Quarterly archive report (1st day of quarter, 2 AM):
-- 0 2 1 */3 * mysql -u root -p -e "CALL ecommerce_analytics.sp_generate_archive_report(90);"

-- Annual purge of very old archives (January 1st, 3 AM):
-- 0 3 1 1 * mysql -u root -p -e "CALL ecommerce_analytics.sp_purge_old_archives('orders_archive', 3650);"

SELECT 'All archive procedures created successfully' AS Status;