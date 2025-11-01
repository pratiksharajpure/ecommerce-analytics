-- ========================================
-- BATCH PROCESSING & ETL SYSTEM
-- E-commerce Revenue Analytics Engine
-- Batch Jobs, ETL Processes, Data Transformations
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for procedure creation
DELIMITER //

-- ========================================
-- BATCH PROCESSING INFRASTRUCTURE
-- ========================================

-- Batch Job Definitions
CREATE TABLE IF NOT EXISTS batch_job_definitions (
    job_id INT PRIMARY KEY AUTO_INCREMENT,
    job_name VARCHAR(200) NOT NULL UNIQUE,
    job_description TEXT,
    job_type ENUM('etl', 'aggregation', 'cleanup', 'transformation', 'export', 'calculation') NOT NULL,
    job_category ENUM('daily', 'weekly', 'monthly', 'real_time', 'on_demand') DEFAULT 'daily',
    execution_order INT DEFAULT 100,
    schedule_expression VARCHAR(100),
    estimated_duration_minutes INT,
    is_active BOOLEAN DEFAULT TRUE,
    depends_on_jobs VARCHAR(500),
    max_retry_attempts INT DEFAULT 3,
    timeout_minutes INT DEFAULT 60,
    notification_email VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_job_type (job_type),
    INDEX idx_job_category (job_category),
    INDEX idx_is_active (is_active),
    INDEX idx_execution_order (execution_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Batch Job Execution Log
CREATE TABLE IF NOT EXISTS batch_job_execution_log (
    execution_id INT PRIMARY KEY AUTO_INCREMENT,
    job_id INT NOT NULL,
    execution_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_end TIMESTAMP NULL,
    execution_duration_seconds INT,
    status ENUM('running', 'completed', 'failed', 'timeout', 'cancelled') DEFAULT 'running',
    records_processed INT DEFAULT 0,
    records_inserted INT DEFAULT 0,
    records_updated INT DEFAULT 0,
    records_deleted INT DEFAULT 0,
    records_failed INT DEFAULT 0,
    execution_message TEXT,
    error_message TEXT,
    retry_attempt INT DEFAULT 0,
    executed_by VARCHAR(100),
    FOREIGN KEY (job_id) REFERENCES batch_job_definitions(job_id) ON DELETE CASCADE,
    INDEX idx_job_id (job_id),
    INDEX idx_execution_start (execution_start),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Batch Job Dependencies
CREATE TABLE IF NOT EXISTS batch_job_dependencies (
    dependency_id INT PRIMARY KEY AUTO_INCREMENT,
    job_id INT NOT NULL,
    depends_on_job_id INT NOT NULL,
    dependency_type ENUM('required', 'optional', 'conditional') DEFAULT 'required',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES batch_job_definitions(job_id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_job_id) REFERENCES batch_job_definitions(job_id) ON DELETE CASCADE,
    UNIQUE KEY unique_dependency (job_id, depends_on_job_id),
    INDEX idx_job_id (job_id),
    INDEX idx_depends_on_job_id (depends_on_job_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ETL Staging Tables
CREATE TABLE IF NOT EXISTS etl_staging_orders (
    staging_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT,
    customer_id INT,
    order_date TIMESTAMP,
    total_amount DECIMAL(10,2),
    status VARCHAR(50),
    payment_status VARCHAR(50),
    batch_id INT,
    load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    INDEX idx_batch_id (batch_id),
    INDEX idx_processed (processed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Data Quality Issues Log
CREATE TABLE IF NOT EXISTS data_quality_issues (
    issue_id INT PRIMARY KEY AUTO_INCREMENT,
    execution_id INT,
    issue_type ENUM('missing_value', 'invalid_format', 'duplicate', 'constraint_violation', 'business_rule') NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    table_name VARCHAR(100),
    record_id INT,
    column_name VARCHAR(100),
    issue_description TEXT,
    data_value VARCHAR(500),
    resolution_status ENUM('open', 'resolved', 'ignored') DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES batch_job_execution_log(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_issue_type (issue_type),
    INDEX idx_severity (severity),
    INDEX idx_resolution_status (resolution_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- ANALYTICAL TABLES (TARGET FOR ETL)
-- ========================================

-- Daily Sales Summary
CREATE TABLE IF NOT EXISTS fact_daily_sales (
    date_key DATE PRIMARY KEY,
    total_orders INT DEFAULT 0,
    total_revenue DECIMAL(15,2) DEFAULT 0,
    total_items_sold INT DEFAULT 0,
    unique_customers INT DEFAULT 0,
    avg_order_value DECIMAL(10,2) DEFAULT 0,
    total_shipping_cost DECIMAL(10,2) DEFAULT 0,
    total_tax DECIMAL(10,2) DEFAULT 0,
    total_discounts DECIMAL(10,2) DEFAULT 0,
    new_customers INT DEFAULT 0,
    returning_customers INT DEFAULT 0,
    orders_completed INT DEFAULT 0,
    orders_cancelled INT DEFAULT 0,
    gross_profit DECIMAL(15,2) DEFAULT 0,
    gross_margin_pct DECIMAL(5,2) DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_date_key (date_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Product Performance Summary
CREATE TABLE IF NOT EXISTS fact_product_performance (
    product_id INT,
    date_key DATE,
    units_sold INT DEFAULT 0,
    revenue DECIMAL(10,2) DEFAULT 0,
    cost DECIMAL(10,2) DEFAULT 0,
    gross_profit DECIMAL(10,2) DEFAULT 0,
    orders_count INT DEFAULT 0,
    returns_count INT DEFAULT 0,
    reviews_count INT DEFAULT 0,
    avg_rating DECIMAL(3,2) DEFAULT 0,
    inventory_turnover DECIMAL(10,2) DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (product_id, date_key),
    INDEX idx_date_key (date_key),
    INDEX idx_product_id (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Customer Behavior Summary
CREATE TABLE IF NOT EXISTS fact_customer_behavior (
    customer_id INT,
    date_key DATE,
    orders_count INT DEFAULT 0,
    total_spent DECIMAL(10,2) DEFAULT 0,
    items_purchased INT DEFAULT 0,
    avg_order_value DECIMAL(10,2) DEFAULT 0,
    days_since_last_order INT,
    total_lifetime_value DECIMAL(10,2) DEFAULT 0,
    total_lifetime_orders INT DEFAULT 0,
    loyalty_points_earned INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id, date_key),
    INDEX idx_date_key (date_key),
    INDEX idx_customer_id (customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Category Performance Summary
CREATE TABLE IF NOT EXISTS fact_category_performance (
    category_id INT,
    date_key DATE,
    products_count INT DEFAULT 0,
    units_sold INT DEFAULT 0,
    revenue DECIMAL(10,2) DEFAULT 0,
    cost DECIMAL(10,2) DEFAULT 0,
    gross_profit DECIMAL(10,2) DEFAULT 0,
    orders_count INT DEFAULT 0,
    unique_customers INT DEFAULT 0,
    avg_rating DECIMAL(3,2) DEFAULT 0,
    return_rate DECIMAL(5,2) DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (category_id, date_key),
    INDEX idx_date_key (date_key),
    INDEX idx_category_id (category_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- BATCH JOB MANAGEMENT PROCEDURES
-- ========================================

-- Start batch job execution
CREATE PROCEDURE sp_start_batch_job(
    IN p_job_id INT,
    OUT p_execution_id INT
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_records_processed INT DEFAULT 0;
    DECLARE v_records_inserted INT DEFAULT 0;
    DECLARE v_records_updated INT DEFAULT 0;
    DECLARE v_error_msg TEXT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        CALL sp_complete_batch_job(v_execution_id, 'failed', v_records_processed, 
                                   v_records_inserted, v_records_updated, 0, 0, v_error_msg);
        RESIGNAL;
    END;
    
    -- Start job execution
    CALL sp_start_batch_job(1, v_execution_id);
    
    -- Delete existing data for the date
    DELETE FROM fact_daily_sales WHERE date_key = p_process_date;
    
    -- Insert/Update daily sales summary
    INSERT INTO fact_daily_sales (
        date_key,
        total_orders,
        total_revenue,
        total_items_sold,
        unique_customers,
        avg_order_value,
        total_shipping_cost,
        total_tax,
        total_discounts,
        new_customers,
        returning_customers,
        orders_completed,
        orders_cancelled,
        gross_profit,
        gross_margin_pct
    )
    SELECT 
        DATE(o.order_date) AS date_key,
        COUNT(DISTINCT o.order_id) AS total_orders,
        COALESCE(SUM(o.total_amount), 0) AS total_revenue,
        COALESCE(SUM(oi.quantity), 0) AS total_items_sold,
        COUNT(DISTINCT o.customer_id) AS unique_customers,
        COALESCE(AVG(o.total_amount), 0) AS avg_order_value,
        COALESCE(SUM(o.shipping_cost), 0) AS total_shipping_cost,
        COALESCE(SUM(o.tax_amount), 0) AS total_tax,
        COALESCE(SUM(oi.discount), 0) AS total_discounts,
        -- New customers (first order on this date)
        COUNT(DISTINCT CASE 
            WHEN NOT EXISTS (
                SELECT 1 FROM orders o2 
                WHERE o2.customer_id = o.customer_id 
                AND DATE(o2.order_date) < DATE(o.order_date)
            ) THEN o.customer_id 
        END) AS new_customers,
        -- Returning customers
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o2 
                WHERE o2.customer_id = o.customer_id 
                AND DATE(o2.order_date) < DATE(o.order_date)
            ) THEN o.customer_id 
        END) AS returning_customers,
        SUM(CASE WHEN o.status = 'delivered' THEN 1 ELSE 0 END) AS orders_completed,
        SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) AS orders_cancelled,
        COALESCE(SUM(oi.quantity * oi.unit_price) - SUM(oi.quantity * p.cost), 0) AS gross_profit,
        CASE 
            WHEN SUM(oi.quantity * oi.unit_price) > 0 
            THEN ((SUM(oi.quantity * oi.unit_price) - SUM(oi.quantity * p.cost)) / 
                  SUM(oi.quantity * oi.unit_price)) * 100
            ELSE 0 
        END AS gross_margin_pct
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    WHERE DATE(o.order_date) = p_process_date
        AND o.payment_status = 'paid'
    GROUP BY DATE(o.order_date);
    
    

    INSERT INTO batch_job_execution_log (job_id, executed_by, retry_attempt)
    VALUES (p_job_id, USER(), 0);
    
    











SET p_execution_id = LAST_INSERT_ID();
END//

-- Complete batch job execution
CREATE PROCEDURE sp_complete_batch_job(
    IN p_execution_id INT,
    IN p_status VARCHAR(20),
    IN p_records_processed INT,
    IN p_records_inserted INT,
    IN p_records_updated INT,
    IN p_records_deleted INT,
    IN p_records_failed INT,
    IN p_message TEXT
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_records_processed INT DEFAULT 0;
    DECLARE v_records_inserted INT DEFAULT 0;
    DECLARE v_error_msg TEXT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        CALL sp_complete_batch_job(v_execution_id, 'failed', v_records_processed, 
                                   v_records_inserted, 0, 0, 0, v_error_msg);
        RESIGNAL;
    END;
    
    CALL sp_start_batch_job(2, v_execution_id);
    
    -- Delete existing data
    DELETE FROM fact_product_performance WHERE date_key = p_process_date;
    
    -- Insert product performance data
    INSERT INTO fact_product_performance (
        product_id,
        date_key,
        units_sold,
        revenue,
        cost,
        gross_profit,
        orders_count,
        returns_count,
        reviews_count,
        avg_rating
    )
    SELECT 
        p.product_id,
        p_process_date AS date_key,
        COALESCE(SUM(oi.quantity), 0) AS units_sold,
        COALESCE(SUM(oi.subtotal), 0) AS revenue,
        COALESCE(SUM(oi.quantity * p.cost), 0) AS cost,
        COALESCE(SUM(oi.subtotal) - SUM(oi.quantity * p.cost), 0) AS gross_profit,
        COUNT(DISTINCT o.order_id) AS orders_count,
        -- Returns count
        (SELECT COUNT(*) 
         FROM returns r 
         JOIN order_items oi2 ON r.order_item_id = oi2.order_item_id
         WHERE oi2.product_id = p.product_id 
         AND DATE(r.created_at) = p_process_date) AS returns_count,
        -- Reviews count
        (SELECT COUNT(*) 
         FROM reviews rev 
         WHERE rev.product_id = p.product_id 
         AND DATE(rev.created_at) = p_process_date) AS reviews_count,
        -- Average rating
        (SELECT AVG(rating) 
         FROM reviews rev 
         WHERE rev.product_id = p.product_id 
         AND rev.status = 'approved') AS avg_rating
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id 
        AND DATE(o.order_date) = p_process_date
        AND o.payment_status = 'paid'
    WHERE p.status = 'active'
    GROUP BY p.product_id
    HAVING units_sold > 0 OR returns_count > 0 OR reviews_count > 0;
    
    

    


    UPDATE batch_job_execution_log
    








SET 
        execution_end = NOW(),
        execution_duration_seconds = TIMESTAMPDIFF(SECOND, execution_start, NOW()),
        status = p_status,
        records_processed = p_records_processed,
        records_inserted = p_records_inserted,
        records_updated = p_records_updated,
        records_deleted = p_records_deleted,
        records_failed = p_records_failed,
        execution_message = p_message
    WHERE execution_id = p_execution_id;
END//

-- Log data quality issue
CREATE PROCEDURE sp_log_data_quality_issue(
    IN p_execution_id INT,
    IN p_issue_type VARCHAR(50),
    IN p_severity VARCHAR(20),
    IN p_table_name VARCHAR(100),
    IN p_record_id INT,
    IN p_column_name VARCHAR(100),
    IN p_description TEXT,
    IN p_data_value VARCHAR(500)
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_records_processed INT DEFAULT 0;
    DECLARE v_records_inserted INT DEFAULT 0;
    DECLARE v_error_msg TEXT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        CALL sp_complete_batch_job(v_execution_id, 'failed', v_records_processed, 
                                   v_records_inserted, 0, 0, 0, v_error_msg);
        RESIGNAL;
    END;
    
    CALL sp_start_batch_job(3, v_execution_id);
    
    -- Delete existing data
    DELETE FROM fact_customer_behavior WHERE date_key = p_process_date;
    
    -- Insert customer behavior data
    INSERT INTO fact_customer_behavior (
        customer_id,
        date_key,
        orders_count,
        total_spent,
        items_purchased,
        avg_order_value,
        days_since_last_order,
        total_lifetime_value,
        total_lifetime_orders,
        loyalty_points_earned
    )
    SELECT 
        c.customer_id,
        p_process_date AS date_key,
        -- Orders on this date
        COUNT(DISTINCT CASE WHEN DATE(o.order_date) = p_process_date THEN o.order_id END) AS orders_count,
        COALESCE(SUM(CASE WHEN DATE(o.order_date) = p_process_date THEN o.total_amount ELSE 0 END), 0) AS total_spent,
        COALESCE(SUM(CASE WHEN DATE(o.order_date) = p_process_date THEN oi.quantity ELSE 0 END), 0) AS items_purchased,
        AVG(CASE WHEN DATE(o.order_date) = p_process_date THEN o.total_amount END) AS avg_order_value,
        -- Days since last order before process date
        DATEDIFF(p_process_date, MAX(CASE WHEN DATE(o.order_date) < p_process_date THEN o.order_date END)) AS days_since_last_order,
        -- Lifetime metrics up to process date
        SUM(CASE WHEN DATE(o.order_date) <= p_process_date THEN o.total_amount ELSE 0 END) AS total_lifetime_value,
        COUNT(DISTINCT CASE WHEN DATE(o.order_date) <= p_process_date THEN o.order_id END) AS total_lifetime_orders,
        COALESCE(lp.points_earned_lifetime, 0) AS loyalty_points_earned
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
    WHERE c.status = 'active'
    GROUP BY c.customer_id, lp.points_earned_lifetime
    HAVING orders_count > 0;
    
    

    





    INSERT INTO data_quality_issues (
        execution_id,
        issue_type,
        severity,
        table_name,
        record_id,
        column_name,
        issue_description,
        data_value
    ) VALUES (
        p_execution_id,
        p_issue_type,
        p_severity,
        p_table_name,
        p_record_id,
        p_column_name,
        p_description,
        p_data_value
    );
END//

-- ========================================
-- ETL JOB 1: Daily Sales Aggregation
-- ========================================
CREATE PROCEDURE sp_etl_daily_sales_summary(
    IN p_process_date DATE
)
BEGIN
    






SET v_records_inserted = ROW_COUNT();
    SET v_records_processed = v_records_inserted;
    
    -- Complete job execution
    CALL sp_complete_batch_job(
        v_execution_id, 
        'completed', 
        v_records_processed,
        v_records_inserted,
        0,
        0,
        0,
        CONCAT('Daily sales summary completed for ', p_process_date)
    );
    
    SELECT 'Daily sales summary completed' AS Status, v_records_inserted AS Records_Inserted;
END//

-- ========================================
-- ETL JOB 2: Product Performance Aggregation
-- ========================================
CREATE PROCEDURE sp_etl_product_performance(
    IN p_process_date DATE
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_records_processed INT DEFAULT 0;
    DECLARE v_records_inserted INT DEFAULT 0;
    DECLARE v_error_msg TEXT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        CALL sp_complete_batch_job(v_execution_id, 'failed', v_records_processed, 
                                   v_records_inserted, 0, 0, 0, v_error_msg);
        RESIGNAL;
    END;
    
    CALL sp_start_batch_job(4, v_execution_id);
    
    -- Delete existing data
    DELETE FROM fact_category_performance WHERE date_key = p_process_date;
    
    -- Insert category performance data
    INSERT INTO fact_category_performance (
        category_id,
        date_key,
        products_count,
        units_sold,
        revenue,
        cost,
        gross_profit,
        orders_count,
        unique_customers,
        avg_rating,
        return_rate
    )
    SELECT 
        pc.category_id,
        p_process_date AS date_key,
        COUNT(DISTINCT p.product_id) AS products_count,
        COALESCE(SUM(oi.quantity), 0) AS units_sold,
        COALESCE(SUM(oi.subtotal), 0) AS revenue,
        COALESCE(SUM(oi.quantity * p.cost), 0) AS cost,
        COALESCE(SUM(oi.subtotal) - SUM(oi.quantity * p.cost), 0) AS gross_profit,
        COUNT(DISTINCT o.order_id) AS orders_count,
        COUNT(DISTINCT o.customer_id) AS unique_customers,
        AVG(rev.rating) AS avg_rating,
        -- Return rate
        CASE 
            WHEN COUNT(DISTINCT o.order_id) > 0 
            THEN (COUNT(DISTINCT r.return_id) / COUNT(DISTINCT o.order_id)) * 100
            ELSE 0 
        END AS return_rate
    FROM product_categories pc
    LEFT JOIN products p ON pc.category_id = p.category_id AND p.status = 'active'
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id 
        AND DATE(o.order_date) = p_process_date
        AND o.payment_status = 'paid'
    LEFT JOIN reviews rev ON p.product_id = rev.product_id AND rev.status = 'approved'
    LEFT JOIN returns r ON oi.order_item_id = r.order_item_id 
        AND DATE(r.created_at) = p_process_date
    GROUP BY pc.category_id
    HAVING units_sold > 0;
    
    

    












SET v_records_inserted = ROW_COUNT();
    SET v_records_processed = v_records_inserted;
    
    CALL sp_complete_batch_job(
        v_execution_id, 
        'completed', 
        v_records_processed,
        v_records_inserted,
        0,
        0,
        0,
        CONCAT('Product performance completed for ', p_process_date)
    );
    
    SELECT 'Product performance aggregation completed' AS Status, v_records_inserted AS Records_Inserted;
END//

-- ========================================
-- ETL JOB 3: Customer Behavior Aggregation
-- ========================================
CREATE PROCEDURE sp_etl_customer_behavior(
    IN p_process_date DATE
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_issues_found INT DEFAULT 0;
    DECLARE v_error_msg TEXT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        CALL sp_complete_batch_job(v_execution_id, 'failed', 0, 0, 0, 0, 0, v_error_msg);
        RESIGNAL;
    END;
    
    CALL sp_start_batch_job(5, v_execution_id);
    
    -- Check 1: Orders with missing customer
    INSERT INTO data_quality_issues (execution_id, issue_type, severity, table_name, record_id, issue_description)
    SELECT 
        v_execution_id,
        'missing_value',
        'high',
        'orders',
        order_id,
        'Order has null customer_id'
    FROM orders
    WHERE customer_id IS NULL
        AND order_date >= DATE_SUB(NOW(), INTERVAL 7 DAY);
    
    

    












SET v_records_inserted = ROW_COUNT();
    SET v_records_processed = v_records_inserted;
    
    CALL sp_complete_batch_job(
        v_execution_id, 
        'completed', 
        v_records_processed,
        v_records_inserted,
        0,
        0,
        0,
        CONCAT('Customer behavior completed for ', p_process_date)
    );
    
    SELECT 'Customer behavior aggregation completed' AS Status, v_records_inserted AS Records_Inserted;
END//

-- ========================================
-- ETL JOB 4: Category Performance Aggregation
-- ========================================
CREATE PROCEDURE sp_etl_category_performance(
    IN p_process_date DATE
)
BEGIN

SET v_records_inserted = ROW_COUNT();
    SET v_records_processed = v_records_inserted;
    
    CALL sp_complete_batch_job(
        v_execution_id, 
        'completed', 
        v_records_processed,
        v_records_inserted,
        0,
        0,
        0,
        CONCAT('Category performance completed for ', p_process_date)
    );
    
    SELECT 'Category performance aggregation completed' AS Status, v_records_inserted AS Records_Inserted;
END//

-- ========================================
-- ETL JOB 5: Data Quality Validation
-- ========================================
CREATE PROCEDURE sp_etl_data_quality_check()
BEGIN

SET v_issues_found = v_issues_found + ROW_COUNT();
    
    -- Check 2: Products with invalid prices
    INSERT INTO data_quality_issues (execution_id, issue_type, severity, table_name, record_id, column_name, issue_description, data_value)
    SELECT 
        v_execution_id,
        'invalid_format',
        'critical',
        'products',
        product_id,
        'price',
        'Product has negative or zero price',
        price
    FROM products
    WHERE price <= 0 AND status = 'active';
    
    SET v_issues_found = v_issues_found + ROW_COUNT();
    
    -- Check 3: Duplicate SKUs
    INSERT INTO data_quality_issues (execution_id, issue_type, severity, table_name, record_id, column_name, issue_description, data_value)
    SELECT 
        v_execution_id,
        'duplicate',
        'high',
        'products',
        MIN(product_id),
        'sku',
        CONCAT('Duplicate SKU found: ', COUNT(*), ' occurrences'),
        sku
    FROM products
    WHERE sku IS NOT NULL
    GROUP BY sku
    HAVING COUNT(*) > 1;
    
    SET v_issues_found = v_issues_found + ROW_COUNT();
    
    -- Check 4: Customers with invalid emails
    INSERT INTO data_quality_issues (execution_id, issue_type, severity, table_name, record_id, column_name, issue_description, data_value)
    SELECT 
        v_execution_id,
        'invalid_format',
        'medium',
        'customers',
        customer_id,
        'email',
        'Invalid email format',
        email
    FROM customers
    WHERE email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'
        AND email IS NOT NULL
        AND status = 'active';
    
    SET v_issues_found = v_issues_found + ROW_COUNT();
    
    -- Check 5: Inventory constraint violations
    INSERT INTO data_quality_issues (execution_id, issue_type, severity, table_name, record_id, issue_description, data_value)
    SELECT 
        v_execution_id,
        'constraint_violation',
        'critical',
        'inventory',
        inventory_id,
        CONCAT('Reserved quantity (', quantity_reserved, ') exceeds on-hand (', quantity_on_hand, ')'),
        quantity_reserved - quantity_on_hand
    FROM inventory
    WHERE quantity_reserved > quantity_on_hand;
    
    SET v_issues_found = v_issues_found + ROW_COUNT();
    
    CALL sp_complete_batch_job(
        v_execution_id, 
        'completed', 
        v_issues_found,
        v_issues_found,
        0,
        0,
        0,
        CONCAT('Data quality check completed. Issues found: ', v_issues_found)
    );
    
    SELECT 'Data quality check completed' AS Status, v_issues_found AS Issues_Found;
END//

-- ========================================
-- ETL JOB 6: Data Cleanup and Archive
-- ========================================
CREATE PROCEDURE sp_etl_cleanup_old_data(
    IN p_retention_days INT
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_records_deleted INT DEFAULT 0;
    DECLARE v_staging_deleted INT DEFAULT 0;
    DECLARE v_logs_deleted INT DEFAULT 0;
    DECLARE v_error_msg TEXT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        CALL sp_complete_batch_job(v_execution_id, 'failed', 0, 0, 0, v_records_deleted, 0, v_error_msg);
        RESIGNAL;
    END;
    
    CALL sp_start_batch_job(6, v_execution_id);
    
    -- Clean up staging tables
    DELETE FROM etl_staging_orders
    WHERE load_timestamp < DATE_SUB(NOW(), INTERVAL p_retention_days DAY)
        AND processed = TRUE;
    

    












SET v_staging_deleted = ROW_COUNT();
    
    -- Clean up old execution logs (keep only summary)
    DELETE FROM batch_job_execution_log
    WHERE execution_start < DATE_SUB(NOW(), INTERVAL p_retention_days DAY)
        AND status IN ('completed', 'failed');
    SET v_logs_deleted = ROW_COUNT();
    
    -- Clean up resolved data quality issues
    DELETE FROM data_quality_issues
    WHERE created_at < DATE_SUB(NOW(), INTERVAL p_retention_days DAY)
        AND resolution_status = 'resolved';
    
    SET v_records_deleted = v_staging_deleted + v_logs_deleted + ROW_COUNT();
    
    CALL sp_complete_batch_job(
        v_execution_id, 
        'completed', 
        v_records_deleted,
        0,
        0,
        v_records_deleted,
        0,
        CONCAT('Cleanup completed. Deleted: ', v_records_deleted, ' records')
    );
    
    SELECT 'Data cleanup completed' AS Status, v_records_deleted AS Records_Deleted;
END//

-- ========================================
-- ETL JOB 7: Calculate Derived Metrics
-- ========================================
CREATE PROCEDURE sp_etl_calculate_derived_metrics(
    IN p_process_date DATE
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_records_updated INT DEFAULT 0;
    DECLARE v_error_msg TEXT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        CALL sp_complete_batch_job(v_execution_id, 'failed', 0, 0, v_records_updated, 0, 0, v_error_msg);
        RESIGNAL;
    END;
    
    CALL sp_start_batch_job(7, v_execution_id);
    
    -- Update product inventory turnover
    UPDATE fact_product_performance fpp
    JOIN inventory i ON fpp.product_id = i.product_id
    SET fpp.inventory_turnover = CASE 
        WHEN i.quantity_on_hand > 0 
        THEN fpp.units_sold / i.quantity_on_hand
        ELSE 0 
    END
    WHERE fpp.date_key = p_process_date;
    
    

    












SET v_records_updated = ROW_COUNT();
    
    -- Update customer lifetime metrics
    UPDATE fact_customer_behavior fcb
    SET 
        total_lifetime_value = (
            SELECT SUM(total_amount)
            FROM orders o
            WHERE o.customer_id = fcb.customer_id
                AND o.payment_status = 'paid'
                AND DATE(o.order_date) <= fcb.date_key
        ),
        total_lifetime_orders = (
            SELECT COUNT(*)
            FROM orders o
            WHERE o.customer_id = fcb.customer_id
                AND o.payment_status = 'paid'
                AND DATE(o.order_date) <= fcb.date_key
        )
    WHERE fcb.date_key = p_process_date;
    
    SET v_records_updated = v_records_updated + ROW_COUNT();
    
    CALL sp_complete_batch_job(
        v_execution_id, 
        'completed', 
        v_records_updated,
        0,
        v_records_updated,
        0,
        0,
        CONCAT('Derived metrics calculated for ', p_process_date)
    );
    
    SELECT 'Derived metrics calculation completed' AS Status, v_records_updated AS Records_Updated;
END//

-- ========================================
-- MASTER ETL ORCHESTRATOR
-- ========================================
CREATE PROCEDURE sp_run_daily_etl_pipeline(
    IN p_process_date DATE
)
BEGIN
DECLARE v_start_time TIMESTAMP;
    DECLARE v_end_time TIMESTAMP;
    DECLARE v_total_duration INT;
    DECLARE v_error_msg TEXT;
    DECLARE v_jobs_completed INT DEFAULT 0;
    DECLARE v_jobs_failed INT DEFAULT 0;
    
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        SELECT CONCAT('ETL Pipeline failed: ', v_error_msg) AS Error;
        

    












SET v_jobs_failed = v_jobs_failed + 1;
    END;
    
    SET v_start_time = NOW();
    
    SELECT CONCAT('Starting ETL Pipeline for date: ', p_process_date) AS Status;
    
    -- Job 1: Daily Sales Summary
    CALL sp_etl_daily_sales_summary(p_process_date);
    SET v_jobs_completed = v_jobs_completed + 1;
    
    -- Job 2: Product Performance
    CALL sp_etl_product_performance(p_process_date);
    SET v_jobs_completed = v_jobs_completed + 1;
    
    -- Job 3: Customer Behavior
    CALL sp_etl_customer_behavior(p_process_date);
    SET v_jobs_completed = v_jobs_completed + 1;
    
    -- Job 4: Category Performance
    CALL sp_etl_category_performance(p_process_date);
    SET v_jobs_completed = v_jobs_completed + 1;
    
    -- Job 5: Calculate Derived Metrics
    CALL sp_etl_calculate_derived_metrics(p_process_date);
    SET v_jobs_completed = v_jobs_completed + 1;
    
    -- Job 6: Data Quality Check
    CALL sp_etl_data_quality_check();
    SET v_jobs_completed = v_jobs_completed + 1;
    
    SET v_end_time = NOW();
    SET v_total_duration = TIMESTAMPDIFF(SECOND, v_start_time, v_end_time);
    
    -- Pipeline summary
    SELECT 
        'ETL Pipeline Completed' AS Status,
        p_process_date AS Process_Date,
        v_jobs_completed AS Jobs_Completed,
        v_jobs_failed AS Jobs_Failed,
        v_total_duration AS Duration_Seconds,
        CONCAT(FLOOR(v_total_duration / 60), 'm ', MOD(v_total_duration, 60), 's') AS Duration_Formatted;
END//

-- ========================================
-- TRANSFORMATION PROCEDURES
-- ========================================

-- Transform: Normalize customer data
CREATE PROCEDURE sp_transform_normalize_customers()
BEGIN
DECLARE v_records_updated INT;
    
    -- Update order totals from order items
    UPDATE orders o
    

    -- Standardize phone numbers (remove non-numeric)
    UPDATE customers
    











SET phone = REGEXP_REPLACE(phone, '[^0-9]', '')
    WHERE phone IS NOT NULL;
    
    -- Standardize email (lowercase)
    UPDATE customers
    SET email = LOWER(TRIM(email))
    WHERE email IS NOT NULL;
    
    -- Standardize names (proper case)
    UPDATE customers
    SET 
        first_name = CONCAT(UPPER(SUBSTRING(first_name, 1, 1)), LOWER(SUBSTRING(first_name, 2))),
        last_name = CONCAT(UPPER(SUBSTRING(last_name, 1, 1)), LOWER(SUBSTRING(last_name, 2)))
    WHERE first_name IS NOT NULL OR last_name IS NOT NULL;
    
    SELECT 'Customer data normalization completed' AS Status;
END//

-- Transform: Enrich order data
CREATE PROCEDURE sp_transform_enrich_orders()
BEGIN
    
SET total_amount = (
        SELECT COALESCE(SUM(quantity * unit_price - discount), 0)
        FROM order_items oi
        WHERE oi.order_id = o.order_id
    )
    WHERE EXISTS (
        SELECT 1 FROM order_items WHERE order_id = o.order_id
    );
    
    SET v_records_updated = ROW_COUNT();
    
    SELECT 'Order enrichment completed' AS Status, v_records_updated AS Records_Updated;
END//

-- Transform: Calculate customer segments
CREATE PROCEDURE sp_transform_customer_segments()
BEGIN
    -- Create temporary table for segmentation
    CREATE TEMPORARY TABLE IF NOT EXISTS temp_customer_segments (
        customer_id INT PRIMARY KEY,
        segment VARCHAR(50),
        total_spent DECIMAL(10,2),
        order_count INT,
        recency_days INT
    );
    
    TRUNCATE TABLE temp_customer_segments;
    
    -- Calculate customer segments based on RFM (Recency, Frequency, Monetary)
    INSERT INTO temp_customer_segments (customer_id, segment, total_spent, order_count, recency_days)
    SELECT 
        c.customer_id,
        CASE 
            WHEN total_spent >= 10000 AND order_count >= 10 AND recency_days <= 30 THEN 'VIP'
            WHEN total_spent >= 5000 AND order_count >= 5 AND recency_days <= 60 THEN 'High Value'
            WHEN total_spent >= 1000 AND order_count >= 3 AND recency_days <= 90 THEN 'Regular'
            WHEN recency_days <= 90 THEN 'Active'
            WHEN recency_days <= 180 THEN 'At Risk'
            ELSE 'Dormant'
        END AS segment,
        COALESCE(SUM(o.total_amount), 0) AS total_spent,
        COUNT(o.order_id) AS order_count,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS recency_days
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    WHERE c.status = 'active'
    GROUP BY c.customer_id;
    
    SELECT 
        segment,
        COUNT(*) AS customer_count,
        AVG(total_spent) AS avg_spent,
        AVG(order_count) AS avg_orders,
        AVG(recency_days) AS avg_recency_days
    FROM temp_customer_segments
    GROUP BY segment
    ORDER BY AVG(total_spent) DESC;
END//

-- ========================================
-- BATCH JOB MONITORING VIEWS
-- ========================================

DELIMITER ;

-- Batch Job Execution Dashboard
CREATE OR REPLACE VIEW v_batch_job_dashboard AS
SELECT 
    bjd.job_id,
    bjd.job_name,
    bjd.job_type,
    bjd.job_category,
    bjd.is_active,
    bjd.execution_order,
    -- Last execution
    (SELECT execution_start 
     FROM batch_job_execution_log 
     WHERE job_id = bjd.job_id 
     ORDER BY execution_start DESC 
     LIMIT 1) AS last_execution_time,
    (SELECT status 
     FROM batch_job_execution_log 
     WHERE job_id = bjd.job_id 
     ORDER BY execution_start DESC 
     LIMIT 1) AS last_execution_status,
    (SELECT execution_duration_seconds 
     FROM batch_job_execution_log 
     WHERE job_id = bjd.job_id 
     ORDER BY execution_start DESC 
     LIMIT 1) AS last_duration_seconds,
    (SELECT records_processed 
     FROM batch_job_execution_log 
     WHERE job_id = bjd.job_id 
     ORDER BY execution_start DESC 
     LIMIT 1) AS last_records_processed,
    -- Statistics (last 30 days)
    (SELECT COUNT(*) 
     FROM batch_job_execution_log 
     WHERE job_id = bjd.job_id 
     AND execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)) AS executions_last_30_days,
    (SELECT COUNT(*) 
     FROM batch_job_execution_log 
     WHERE job_id = bjd.job_id 
     AND status = 'failed'
     AND execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)) AS failures_last_30_days,
    (SELECT AVG(execution_duration_seconds) 
     FROM batch_job_execution_log 
     WHERE job_id = bjd.job_id 
     AND status = 'completed'
     AND execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)) AS avg_duration_seconds,
    (SELECT SUM(records_processed) 
     FROM batch_job_execution_log 
     WHERE job_id = bjd.job_id 
     AND execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)) AS total_records_processed_30_days
FROM batch_job_definitions bjd
ORDER BY bjd.execution_order, bjd.job_id;

-- ETL Job Performance Trends
CREATE OR REPLACE VIEW v_etl_performance_trends AS
SELECT 
    DATE(bjel.execution_start) AS execution_date,
    bjd.job_name,
    bjd.job_type,
    COUNT(*) AS execution_count,
    SUM(CASE WHEN bjel.status = 'completed' THEN 1 ELSE 0 END) AS successful_runs,
    SUM(CASE WHEN bjel.status = 'failed' THEN 1 ELSE 0 END) AS failed_runs,
    AVG(bjel.execution_duration_seconds) AS avg_duration_seconds,
    SUM(bjel.records_processed) AS total_records_processed,
    SUM(bjel.records_inserted) AS total_records_inserted,
    SUM(bjel.records_updated) AS total_records_updated,
    SUM(bjel.records_failed) AS total_records_failed
FROM batch_job_execution_log bjel
JOIN batch_job_definitions bjd ON bjel.job_id = bjd.job_id
WHERE bjel.execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(bjel.execution_start), bjd.job_name, bjd.job_type
ORDER BY execution_date DESC, bjd.job_name;

-- Data Quality Issues Summary
CREATE OR REPLACE VIEW v_data_quality_summary AS
SELECT 
    dqi.issue_type,
    dqi.severity,
    dqi.table_name,
    COUNT(*) AS issue_count,
    COUNT(CASE WHEN dqi.resolution_status = 'open' THEN 1 END) AS open_issues,
    COUNT(CASE WHEN dqi.resolution_status = 'resolved' THEN 1 END) AS resolved_issues,
    MIN(dqi.created_at) AS oldest_issue,
    MAX(dqi.created_at) AS newest_issue
FROM data_quality_issues dqi
WHERE dqi.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY dqi.issue_type, dqi.severity, dqi.table_name
ORDER BY 
    FIELD(dqi.severity, 'critical', 'high', 'medium', 'low'),
    issue_count DESC;

-- Failed Jobs Report
CREATE OR REPLACE VIEW v_failed_batch_jobs AS
SELECT 
    bjd.job_name,
    bjd.job_type,
    bjel.execution_start,
    bjel.execution_duration_seconds,
    bjel.records_processed,
    bjel.records_failed,
    bjel.error_message,
    bjel.retry_attempt
FROM batch_job_execution_log bjel
JOIN batch_job_definitions bjd ON bjel.job_id = bjd.job_id
WHERE bjel.status = 'failed'
    AND bjel.execution_start >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY bjel.execution_start DESC;

-- ETL Pipeline Health
CREATE OR REPLACE VIEW v_etl_pipeline_health AS
SELECT 
    bjd.job_category,
    COUNT(DISTINCT bjd.job_id) AS total_jobs,
    SUM(CASE WHEN bjd.is_active = TRUE THEN 1 ELSE 0 END) AS active_jobs,
    -- Last 24 hours metrics
    SUM(CASE WHEN bjel.execution_start >= DATE_SUB(NOW(), INTERVAL 24 HOUR) THEN 1 ELSE 0 END) AS executions_24h,
    SUM(CASE WHEN bjel.execution_start >= DATE_SUB(NOW(), INTERVAL 24 HOUR) 
             AND bjel.status = 'completed' THEN 1 ELSE 0 END) AS successful_24h,
    SUM(CASE WHEN bjel.execution_start >= DATE_SUB(NOW(), INTERVAL 24 HOUR) 
             AND bjel.status = 'failed' THEN 1 ELSE 0 END) AS failed_24h,
    ROUND(AVG(CASE WHEN bjel.execution_start >= DATE_SUB(NOW(), INTERVAL 24 HOUR) 
                   THEN bjel.execution_duration_seconds END), 2) AS avg_duration_24h,
    -- Health indicator
    CASE 
        WHEN SUM(CASE WHEN bjel.execution_start >= DATE_SUB(NOW(), INTERVAL 24 HOUR) 
                      AND bjel.status = 'failed' THEN 1 ELSE 0 END) = 0 
             THEN 'Healthy'
        WHEN SUM(CASE WHEN bjel.execution_start >= DATE_SUB(NOW(), INTERVAL 24 HOUR) 
                      AND bjel.status = 'failed' THEN 1 ELSE 0 END) <= 2 
             THEN 'Warning'
        ELSE 'Critical'
    END AS health_status
FROM batch_job_definitions bjd
LEFT JOIN batch_job_execution_log bjel ON bjd.job_id = bjel.job_id
GROUP BY bjd.job_category
ORDER BY bjd.job_category;

-- ========================================
-- INSERT BATCH JOB DEFINITIONS
-- ========================================

INSERT INTO batch_job_definitions (job_name, job_description, job_type, job_category, execution_order, estimated_duration_minutes, is_active) VALUES
('Daily Sales Summary ETL', 'Aggregates daily sales metrics into fact table', 'etl', 'daily', 10, 15, TRUE),
('Product Performance ETL', 'Aggregates product performance metrics', 'etl', 'daily', 20, 20, TRUE),
('Customer Behavior ETL', 'Aggregates customer behavior and lifetime value', 'etl', 'daily', 30, 25, TRUE),
('Category Performance ETL', 'Aggregates category-level performance metrics', 'etl', 'daily', 40, 15, TRUE),
('Data Quality Validation', 'Validates data quality and logs issues', 'etl', 'daily', 50, 10, TRUE),
('Data Cleanup and Archive', 'Cleans up old staging and log data', 'cleanup', 'weekly', 60, 5, TRUE),
('Calculate Derived Metrics', 'Calculates additional derived metrics', 'calculation', 'daily', 70, 10, TRUE),
('Customer Segmentation', 'Segments customers based on RFM analysis', 'transformation', 'weekly', 80, 15, TRUE),
('Inventory Snapshot', 'Creates daily inventory snapshot', 'aggregation', 'daily', 90, 5, TRUE),
('Monthly Revenue Rollup', 'Rolls up daily sales into monthly summaries', 'aggregation', 'monthly', 100, 20, TRUE);

-- ========================================
-- QUICK START COMMANDS
-- ========================================

SELECT '============================================' AS '';
SELECT 'Batch Processing & ETL System Created Successfully' AS Status;
SELECT '============================================' AS '';
SELECT '' AS '';
SELECT 'QUICK START COMMANDS:' AS '';
SELECT '-------------------------------------------' AS '';
SELECT '1. Run complete ETL pipeline for today:' AS '';
SELECT '   CALL sp_run_daily_etl_pipeline(CURDATE());' AS '';
SELECT '' AS '';
SELECT '2. Run individual ETL jobs:' AS '';
SELECT '   CALL sp_etl_daily_sales_summary(CURDATE());' AS '';
SELECT '   CALL sp_etl_product_performance(CURDATE());' AS '';
SELECT '   CALL sp_etl_customer_behavior(CURDATE());' AS '';
SELECT '' AS '';
SELECT '3. Run data quality check:' AS '';
SELECT '   CALL sp_etl_data_quality_check();' AS '';
SELECT '' AS '';
SELECT '4. View batch job dashboard:' AS '';
SELECT '   SELECT * FROM v_batch_job_dashboard;' AS '';
SELECT '' AS '';
SELECT '5. View ETL pipeline health:' AS '';
SELECT '   SELECT * FROM v_etl_pipeline_health;' AS '';
SELECT '' AS '';
SELECT '6. View data quality issues:' AS '';
SELECT '   SELECT * FROM v_data_quality_summary;' AS '';
SELECT '' AS '';
SELECT '7. View failed jobs:' AS '';
SELECT '   SELECT * FROM v_failed_batch_jobs;' AS '';
SELECT '' AS '';
SELECT '8. Run data cleanup (90 day retention):' AS '';
SELECT '   CALL sp_etl_cleanup_old_data(90);' AS '';
SELECT '' AS '';
SELECT '9. Transform customer segments:' AS '';
SELECT '   CALL sp_transform_customer_segments();' AS '';
SELECT '' AS '';
SELECT '10. Normalize customer data:' AS '';
SELECT '    CALL sp_transform_normalize_customers();' AS '';
SELECT '' AS '';
SELECT '============================================' AS '';

-- Show configured batch jobs
SELECT 'Configured Batch Jobs:' AS '';
SELECT 
    job_name,
    job_type,
    job_category,
    execution_order,
    estimated_duration_minutes,
    is_active
FROM batch_job_definitions
ORDER BY execution_order;