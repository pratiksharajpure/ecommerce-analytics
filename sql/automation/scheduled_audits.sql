-- ========================================
-- SCHEDULED AUDITS & AUTOMATED JOBS
-- E-commerce Revenue Analytics Engine
-- Audit Jobs, Schedule Definitions, Frequency Control
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for procedure creation
DELIMITER //

-- ========================================
-- AUDIT INFRASTRUCTURE TABLES
-- ========================================

-- Audit Job Definitions
CREATE TABLE IF NOT EXISTS audit_job_definitions (
    job_id INT PRIMARY KEY AUTO_INCREMENT,
    job_name VARCHAR(200) NOT NULL UNIQUE,
    job_description TEXT,
    job_type ENUM('data_quality', 'financial', 'inventory', 'security', 'performance', 'compliance') NOT NULL,
    schedule_frequency ENUM('hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'on_demand') DEFAULT 'daily',
    schedule_time TIME,
    schedule_day_of_week ENUM('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'),
    schedule_day_of_month INT,
    is_active BOOLEAN DEFAULT TRUE,
    alert_on_failure BOOLEAN DEFAULT TRUE,
    alert_on_threshold_breach BOOLEAN DEFAULT TRUE,
    threshold_value DECIMAL(10,2),
    notification_email VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_job_type (job_type),
    INDEX idx_schedule_frequency (schedule_frequency),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Audit Execution Log
CREATE TABLE IF NOT EXISTS audit_execution_log (
    execution_id INT PRIMARY KEY AUTO_INCREMENT,
    job_id INT NOT NULL,
    execution_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_end TIMESTAMP NULL,
    execution_duration_seconds INT,
    status ENUM('running', 'completed', 'failed', 'warning') DEFAULT 'running',
    records_audited INT DEFAULT 0,
    issues_found INT DEFAULT 0,
    warnings_found INT DEFAULT 0,
    execution_result TEXT,
    error_message TEXT,
    executed_by VARCHAR(100),
    FOREIGN KEY (job_id) REFERENCES audit_job_definitions(job_id) ON DELETE CASCADE,
    INDEX idx_job_id (job_id),
    INDEX idx_execution_start (execution_start),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Audit Findings Table
CREATE TABLE IF NOT EXISTS audit_findings (
    finding_id INT PRIMARY KEY AUTO_INCREMENT,
    execution_id INT NOT NULL,
    finding_type ENUM('error', 'warning', 'information') DEFAULT 'warning',
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    finding_category VARCHAR(100),
    finding_description TEXT,
    affected_table VARCHAR(100),
    affected_record_id INT,
    current_value VARCHAR(500),
    expected_value VARCHAR(500),
    resolution_status ENUM('open', 'in_progress', 'resolved', 'false_positive') DEFAULT 'open',
    resolution_notes TEXT,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES audit_execution_log(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_finding_type (finding_type),
    INDEX idx_severity (severity),
    INDEX idx_resolution_status (resolution_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Audit Metrics History
CREATE TABLE IF NOT EXISTS audit_metrics_history (
    metric_id INT PRIMARY KEY AUTO_INCREMENT,
    execution_id INT NOT NULL,
    metric_name VARCHAR(200),
    metric_value DECIMAL(15,2),
    metric_date DATE,
    previous_value DECIMAL(15,2),
    variance_percentage DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES audit_execution_log(execution_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_metric_name (metric_name),
    INDEX idx_metric_date (metric_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- AUDIT STORED PROCEDURES
-- ========================================

-- Helper procedure to log audit execution
CREATE PROCEDURE sp_start_audit_execution(
    IN p_job_id INT,
    OUT p_execution_id INT
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_issues INT DEFAULT 0;
    DECLARE v_warnings INT DEFAULT 0;
    DECLARE v_records INT DEFAULT 0;
    DECLARE v_orphaned_items INT;
    DECLARE v_negative_prices INT;
    DECLARE v_invalid_emails INT;
    DECLARE v_result TEXT;
    
    -- Start audit execution
    CALL sp_start_audit_execution(1, v_execution_id);
    
    -- Check 1: Orphaned order items (no parent order)
    SELECT COUNT(*) INTO v_orphaned_items
    FROM order_items oi
    LEFT JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_id IS NULL;
    
    IF v_orphaned_items > 0 THEN
        

    INSERT INTO audit_execution_log (job_id, executed_by)
    VALUES (p_job_id, USER());
    
    











SET p_execution_id = LAST_INSERT_ID();
END//

-- Helper procedure to complete audit execution
CREATE PROCEDURE sp_complete_audit_execution(
    IN p_execution_id INT,
    IN p_status VARCHAR(20),
    IN p_records_audited INT,
    IN p_issues_found INT,
    IN p_warnings_found INT,
    IN p_result TEXT
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_issues INT DEFAULT 0;
    DECLARE v_warnings INT DEFAULT 0;
    DECLARE v_records INT DEFAULT 0;
    DECLARE v_order_total DECIMAL(15,2);
    DECLARE v_item_total DECIMAL(15,2);
    DECLARE v_variance DECIMAL(15,2);
    DECLARE v_result TEXT;
    
    CALL sp_start_audit_execution(2, v_execution_id);
    
    -- Calculate order totals vs item totals
    SELECT 
        SUM(o.total_amount),
        SUM(oi.subtotal)
    INTO v_order_total, v_item_total
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);
    
    

    


    UPDATE audit_execution_log
    








SET 
        execution_end = NOW(),
        execution_duration_seconds = TIMESTAMPDIFF(SECOND, execution_start, NOW()),
        status = p_status,
        records_audited = p_records_audited,
        issues_found = p_issues_found,
        warnings_found = p_warnings_found,
        execution_result = p_result
    WHERE execution_id = p_execution_id;
END//

-- Helper procedure to log audit finding
CREATE PROCEDURE sp_log_audit_finding(
    IN p_execution_id INT,
    IN p_finding_type VARCHAR(20),
    IN p_severity VARCHAR(20),
    IN p_category VARCHAR(100),
    IN p_description TEXT,
    IN p_table VARCHAR(100),
    IN p_record_id INT,
    IN p_current_value VARCHAR(500),
    IN p_expected_value VARCHAR(500)
)
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_issues INT DEFAULT 0;
    DECLARE v_warnings INT DEFAULT 0;
    DECLARE v_records INT DEFAULT 0;
    DECLARE v_result TEXT;
    
    CALL sp_start_audit_execution(3, v_execution_id);
    
    -- Check 1: Negative inventory
    SELECT COUNT(*) INTO v_records
    FROM inventory
    WHERE quantity_on_hand < 0 OR quantity_reserved < 0;
    
    IF v_records > 0 THEN
        

    





    INSERT INTO audit_findings (
        execution_id,
        finding_type,
        severity,
        finding_category,
        finding_description,
        affected_table,
        affected_record_id,
        current_value,
        expected_value
    ) VALUES (
        p_execution_id,
        p_finding_type,
        p_severity,
        p_category,
        p_description,
        p_table,
        p_record_id,
        p_current_value,
        p_expected_value
    );
END//

-- ========================================
-- AUDIT JOB 1: Data Integrity Audit
-- ========================================
CREATE PROCEDURE sp_audit_data_integrity()
BEGIN
    






SET v_issues = v_issues + v_orphaned_items;
        CALL sp_log_audit_finding(
            v_execution_id,
            'error',
            'high',
            'Referential Integrity',
            CONCAT('Found ', v_orphaned_items, ' orphaned order items without parent orders'),
            'order_items',
            NULL,
            v_orphaned_items,
            '0'
        );
    END IF;
    
    -- Check 2: Products with negative prices
    SELECT COUNT(*) INTO v_negative_prices
    FROM products
    WHERE price < 0 OR cost < 0;
    
    IF v_negative_prices > 0 THEN
        SET v_issues = v_issues + v_negative_prices;
        CALL sp_log_audit_finding(
            v_execution_id,
            'error',
            'critical',
            'Data Validity',
            CONCAT('Found ', v_negative_prices, ' products with negative prices or costs'),
            'products',
            NULL,
            v_negative_prices,
            '0'
        );
    END IF;
    
    -- Check 3: Invalid email formats
    SELECT COUNT(*) INTO v_invalid_emails
    FROM customers
    WHERE email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
        AND email IS NOT NULL;
    
    IF v_invalid_emails > 0 THEN
        SET v_warnings = v_warnings + v_invalid_emails;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'medium',
            'Data Quality',
            CONCAT('Found ', v_invalid_emails, ' customers with invalid email formats'),
            'customers',
            NULL,
            v_invalid_emails,
            '0'
        );
    END IF;
    
    -- Check 4: Orders with mismatched totals
    INSERT INTO audit_findings (execution_id, finding_type, severity, finding_category, finding_description, affected_table, affected_record_id, current_value, expected_value)
    SELECT 
        v_execution_id,
        'error',
        'high',
        'Financial Integrity',
        CONCAT('Order #', o.order_id, ' total mismatch: stored=', o.total_amount, ' calculated=', calculated_total),
        'orders',
        o.order_id,
        o.total_amount,
        calculated_total
    FROM orders o
    INNER JOIN (
        SELECT 
            order_id,
            SUM(quantity * unit_price - discount) as calculated_total
        FROM order_items
        GROUP BY order_id
    ) oi ON o.order_id = oi.order_id
    WHERE ABS(o.total_amount - oi.calculated_total) > 0.01;
    
    SET v_issues = v_issues + ROW_COUNT();
    
    -- Calculate total records audited
    SELECT 
        (SELECT COUNT(*) FROM orders) +
        (SELECT COUNT(*) FROM order_items) +
        (SELECT COUNT(*) FROM products) +
        (SELECT COUNT(*) FROM customers)
    INTO v_records;
    
    -- Complete audit
    SET v_result = CONCAT(
        'Data Integrity Audit Complete. ',
        'Records Audited: ', v_records, ', ',
        'Issues: ', v_issues, ', ',
        'Warnings: ', v_warnings
    );
    
    CALL sp_complete_audit_execution(
        v_execution_id,
        CASE WHEN v_issues > 0 THEN 'failed' ELSE 'completed' END,
        v_records,
        v_issues,
        v_warnings,
        v_result
    );
END//

-- ========================================
-- AUDIT JOB 2: Financial Reconciliation
-- ========================================
CREATE PROCEDURE sp_audit_financial_reconciliation()
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_issues INT DEFAULT 0;
    DECLARE v_warnings INT DEFAULT 0;
    DECLARE v_records INT DEFAULT 0;
    DECLARE v_result TEXT;
    
    CALL sp_start_audit_execution(4, v_execution_id);
    
    -- Check 1: Customers without proper contact info
    SELECT COUNT(*) INTO v_records
    FROM customers
    WHERE (email IS NULL OR email = '')
        AND (phone IS NULL OR phone = '')
        AND status = 'active';
    
    IF v_records > 0 THEN
        

    












SET v_variance = ABS(v_order_total - v_item_total);
    
    -- Log metric
    INSERT INTO audit_metrics_history (execution_id, metric_name, metric_value, metric_date)
    VALUES (v_execution_id, 'Monthly Revenue (Orders)', v_order_total, CURDATE()),
           (v_execution_id, 'Monthly Revenue (Items)', v_item_total, CURDATE()),
           (v_execution_id, 'Revenue Variance', v_variance, CURDATE());
    
    -- Check for significant variance (more than $100)
    IF v_variance > 100 THEN
        SET v_issues = v_issues + 1;
        CALL sp_log_audit_finding(
            v_execution_id,
            'error',
            'critical',
            'Revenue Reconciliation',
            CONCAT('Significant variance detected: $', v_variance, ' between order totals and item totals'),
            'orders',
            NULL,
            v_order_total,
            v_item_total
        );
    END IF;
    
    -- Check for orders without items
    SELECT COUNT(*) INTO v_records
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE oi.order_item_id IS NULL
        AND o.payment_status = 'paid';
    
    IF v_records > 0 THEN
        SET v_issues = v_issues + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'error',
            'high',
            'Order Completeness',
            CONCAT('Found ', v_records, ' paid orders without any items'),
            'orders',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    -- Check for refunds without return records
    SELECT COUNT(*) INTO v_records
    FROM orders o
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE o.payment_status = 'refunded'
        AND r.return_id IS NULL;
    
    IF v_records > 0 THEN
        SET v_warnings = v_warnings + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'medium',
            'Refund Documentation',
            CONCAT('Found ', v_records, ' refunded orders without return records'),
            'orders',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    SET v_result = CONCAT('Financial Reconciliation Complete. Issues: ', v_issues, ', Warnings: ', v_warnings);
    
    CALL sp_complete_audit_execution(
        v_execution_id,
        CASE WHEN v_issues > 0 THEN 'failed' ELSE 'completed' END,
        (SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)),
        v_issues,
        v_warnings,
        v_result
    );
END//

-- ========================================
-- AUDIT JOB 3: Inventory Accuracy Audit
-- ========================================
CREATE PROCEDURE sp_audit_inventory_accuracy()
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_issues INT DEFAULT 0;
    DECLARE v_warnings INT DEFAULT 0;
    DECLARE v_records INT DEFAULT 0;
    DECLARE v_result TEXT;
    DECLARE v_large_tables INT DEFAULT 0;
    DECLARE v_fragmented_tables INT DEFAULT 0;
    
    CALL sp_start_audit_execution(5, v_execution_id);
    
    -- Check 1: Large tables without recent optimization
    SELECT COUNT(*) INTO v_large_tables
    FROM information_schema.tables
    WHERE table_schema = 'ecommerce_analytics'
        AND table_rows > 100000
        AND (update_time IS NULL OR update_time < DATE_SUB(NOW(), INTERVAL 30 DAY));
    
    IF v_large_tables > 0 THEN
        

    












SET v_issues = v_issues + v_records;
        
        INSERT INTO audit_findings (execution_id, finding_type, severity, finding_category, finding_description, affected_table, affected_record_id)
        SELECT 
            v_execution_id,
            'error',
            'critical',
            'Inventory Integrity',
            CONCAT('Product ID ', product_id, ' has negative inventory: on_hand=', quantity_on_hand, ' reserved=', quantity_reserved),
            'inventory',
            inventory_id
        FROM inventory
        WHERE quantity_on_hand < 0 OR quantity_reserved < 0;
    END IF;
    
    -- Check 2: Reserved quantity exceeds on-hand
    SELECT COUNT(*) INTO v_records
    FROM inventory
    WHERE quantity_reserved > quantity_on_hand;
    
    IF v_records > 0 THEN
        SET v_issues = v_issues + v_records;
        
        INSERT INTO audit_findings (execution_id, finding_type, severity, finding_category, finding_description, affected_table, affected_record_id)
        SELECT 
            v_execution_id,
            'error',
            'high',
            'Inventory Logic',
            CONCAT('Product ID ', product_id, ' has reserved (', quantity_reserved, ') > on_hand (', quantity_on_hand, ')'),
            'inventory',
            inventory_id
        FROM inventory
        WHERE quantity_reserved > quantity_on_hand;
    END IF;
    
    -- Check 3: Products out of stock but marked as active
    SELECT COUNT(*) INTO v_records
    FROM products p
    LEFT JOIN inventory i ON p.product_id = i.product_id
    WHERE p.status = 'active'
        AND (i.quantity_available IS NULL OR i.quantity_available <= 0);
    
    IF v_records > 0 THEN
        SET v_warnings = v_warnings + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'medium',
            'Product Status',
            CONCAT('Found ', v_records, ' products marked as active but have no available inventory'),
            'products',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    -- Check 4: Inventory below reorder level
    SELECT COUNT(*) INTO v_records
    FROM inventory
    WHERE quantity_available <= reorder_level
        AND quantity_available > 0;
    
    IF v_records > 0 THEN
        SET v_warnings = v_warnings + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'low',
            'Reorder Alert',
            CONCAT(v_records, ' products need reordering (below reorder level)'),
            'inventory',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    SET v_result = CONCAT('Inventory Audit Complete. Issues: ', v_issues, ', Warnings: ', v_warnings);
    
    CALL sp_complete_audit_execution(
        v_execution_id,
        CASE WHEN v_issues > 0 THEN 'failed' ELSE 'completed' END,
        (SELECT COUNT(*) FROM inventory),
        v_issues,
        v_warnings,
        v_result
    );
END//

-- ========================================
-- AUDIT JOB 4: Security & Compliance Audit
-- ========================================
CREATE PROCEDURE sp_audit_security_compliance()
BEGIN

SET v_warnings = v_warnings + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'medium',
            'Customer Data Completeness',
            CONCAT('Found ', v_records, ' active customers without email or phone'),
            'customers',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    -- Check 2: Suspended customers with recent orders
    SELECT COUNT(*) INTO v_records
    FROM customers c
    INNER JOIN orders o ON c.customer_id = o.customer_id
    WHERE c.status = 'suspended'
        AND o.order_date >= DATE_SUB(NOW(), INTERVAL 7 DAY);
    
    IF v_records > 0 THEN
        SET v_issues = v_issues + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'error',
            'high',
            'Account Security',
            CONCAT('Found ', v_records, ' orders from suspended customers in last 7 days'),
            'orders',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    -- Check 3: Payment methods with expired cards
    SELECT COUNT(*) INTO v_records
    FROM payment_methods
    WHERE payment_type IN ('credit_card', 'debit_card')
        AND (
            expiry_year < YEAR(CURDATE())
            OR (expiry_year = YEAR(CURDATE()) AND expiry_month < MONTH(CURDATE()))
        );
    
    IF v_records > 0 THEN
        SET v_warnings = v_warnings + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'low',
            'Payment Method Hygiene',
            CONCAT('Found ', v_records, ' expired payment methods that should be archived'),
            'payment_methods',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    -- Check 4: Reviews without proper moderation
    SELECT COUNT(*) INTO v_records
    FROM reviews
    WHERE status = 'pending'
        AND created_at < DATE_SUB(NOW(), INTERVAL 7 DAY);
    
    IF v_records > 0 THEN
        SET v_warnings = v_warnings + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'medium',
            'Content Moderation',
            CONCAT('Found ', v_records, ' reviews pending for more than 7 days'),
            'reviews',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    SET v_result = CONCAT('Security & Compliance Audit Complete. Issues: ', v_issues, ', Warnings: ', v_warnings);
    
    CALL sp_complete_audit_execution(
        v_execution_id,
        CASE WHEN v_issues > 0 THEN 'failed' ELSE 'completed' END,
        (SELECT COUNT(*) FROM customers) + (SELECT COUNT(*) FROM payment_methods) + (SELECT COUNT(*) FROM reviews),
        v_issues,
        v_warnings,
        v_result
    );
END//

-- ========================================
-- AUDIT JOB 5: Performance Metrics Audit
-- ========================================
CREATE PROCEDURE sp_audit_performance_metrics()
BEGIN

SET v_warnings = v_warnings + v_large_tables;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'medium',
            'Database Maintenance',
            CONCAT('Found ', v_large_tables, ' large tables not optimized in last 30 days'),
            'information_schema.tables',
            NULL,
            v_large_tables,
            '0'
        );
    END IF;
    
    -- Check 2: Tables with high fragmentation
    SELECT COUNT(*) INTO v_fragmented_tables
    FROM information_schema.tables
    WHERE table_schema = 'ecommerce_analytics'
        AND data_free / NULLIF(data_length, 0) > 0.2;
    
    IF v_fragmented_tables > 0 THEN
        SET v_warnings = v_warnings + v_fragmented_tables;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'medium',
            'Table Fragmentation',
            CONCAT('Found ', v_fragmented_tables, ' tables with >20% fragmentation'),
            'information_schema.tables',
            NULL,
            v_fragmented_tables,
            '0'
        );
    END IF;
    
    -- Check 3: Slow query patterns (orders with many items)
    SELECT COUNT(*) INTO v_records
    FROM orders o
    WHERE (
        SELECT COUNT(*) 
        FROM order_items oi 
        WHERE oi.order_id = o.order_id
    ) > 50;
    
    IF v_records > 10 THEN
        SET v_warnings = v_warnings + 1;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'low',
            'Query Performance',
            CONCAT('Found ', v_records, ' orders with 50+ items (potential performance impact)'),
            'orders',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    SET v_result = CONCAT('Performance Metrics Audit Complete. Warnings: ', v_warnings);
    
    CALL sp_complete_audit_execution(
        v_execution_id,
        'completed',
        v_large_tables + v_fragmented_tables,
        v_issues,
        v_warnings,
        v_result
    );
END//

-- ========================================
-- AUDIT JOB 6: Campaign Performance Audit
-- ========================================
CREATE PROCEDURE sp_audit_campaign_performance()
BEGIN
DECLARE v_execution_id INT;
    DECLARE v_issues INT DEFAULT 0;
    DECLARE v_warnings INT DEFAULT 0;
    DECLARE v_records INT DEFAULT 0;
    DECLARE v_result TEXT;
    
    CALL sp_start_audit_execution(6, v_execution_id);
    
    -- Check 1: Active campaigns with no performance data
    SELECT COUNT(*) INTO v_records
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
        AND cp.report_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    WHERE c.status = 'active'
        AND c.start_date <= CURDATE()
        AND cp.performance_id IS NULL;
    
    IF v_records > 0 THEN
        

    












SET v_warnings = v_warnings + v_records;
        CALL sp_log_audit_finding(
            v_execution_id,
            'warning',
            'medium',
            'Campaign Tracking',
            CONCAT('Found ', v_records, ' active campaigns without recent performance data'),
            'campaigns',
            NULL,
            v_records,
            '0'
        );
    END IF;
    
    -- Check 2: Campaigns exceeding budget
    INSERT INTO audit_findings (execution_id, finding_type, severity, finding_category, finding_description, affected_table, affected_record_id, current_value, expected_value)
    SELECT 
        v_execution_id,
        'error',
        'high',
        'Budget Control',
        CONCAT('Campaign "', c.campaign_name, '" has exceeded budget: $', total_spend, ' of $', c.budget),
        'campaigns',
        c.campaign_id,
        total_spend,
        c.budget
    FROM campaigns c
    INNER JOIN (
        SELECT campaign_id, SUM(spend) as total_spend
        FROM campaign_performance
        GROUP BY campaign_id
    ) cp ON c.campaign_id = cp.campaign_id
    WHERE cp.total_spend > c.budget
        AND c.status IN ('active', 'completed');
    
    SET v_issues = v_issues + ROW_COUNT();
    
    -- Check 3: Campaigns with poor ROI
    INSERT INTO audit_findings (execution_id, finding_type, severity, finding_category, finding_description, affected_table, affected_record_id, current_value, expected_value)
    SELECT 
        v_execution_id,
        'warning',
        'medium',
        'Campaign Effectiveness',
        CONCAT('Campaign "', c.campaign_name, '" has negative ROI: revenue=$', total_revenue, ' spend=$', total_spend),
        'campaigns',
        c.campaign_id,
        total_revenue - total_spend,
        '0'
    FROM campaigns c
    INNER JOIN (
        SELECT 
            campaign_id, 
            SUM(spend) as total_spend,
            SUM(revenue) as total_revenue
        FROM campaign_performance
        GROUP BY campaign_id
    ) cp ON c.campaign_id = cp.campaign_id
    WHERE cp.total_revenue < cp.total_spend
        AND c.status IN ('active', 'completed');
    
    SET v_warnings = v_warnings + ROW_COUNT();
    
    SET v_result = CONCAT('Campaign Performance Audit Complete. Issues: ', v_issues, ', Warnings: ', v_warnings);
    
    CALL sp_complete_audit_execution(
        v_execution_id,
        CASE WHEN v_issues > 0 THEN 'warning' ELSE 'completed' END,
        (SELECT COUNT(*) FROM campaigns WHERE status IN ('active', 'completed')),
        v_issues,
        v_warnings,
        v_result
    );
END//

-- Reset delimiter
DELIMITER ;

-- ========================================
-- INSERT AUDIT JOB DEFINITIONS
-- ========================================
INSERT INTO audit_job_definitions (job_name, job_description, job_type, schedule_frequency, schedule_time, is_active, alert_on_failure, notification_email) VALUES
('Data Integrity Audit', 'Checks referential integrity, data validity, and consistency across tables', 'data_quality', 'daily', '02:00:00', TRUE, TRUE, 'data-quality@company.com'),
('Financial Reconciliation', 'Validates financial data, revenue calculations, and payment records', 'financial', 'daily', '03:00:00', TRUE, TRUE, 'finance@company.com'),
('Inventory Accuracy Audit', 'Monitors inventory levels, stock accuracy, and reorder alerts', 'inventory', 'daily', '01:00:00', TRUE, TRUE, 'inventory@company.com'),
('Security & Compliance Audit', 'Reviews security policies, data completeness, and compliance requirements', 'security', 'weekly', '04:00:00', TRUE, TRUE, 'security@company.com'),
('Performance Metrics Audit', 'Analyzes database performance, table sizes, and optimization opportunities', 'performance', 'weekly', '05:00:00', TRUE, FALSE, 'dba@company.com'),
('Campaign Performance Audit', 'Evaluates marketing campaign effectiveness and budget compliance', 'financial', 'daily', '06:00:00', TRUE, TRUE, 'marketing@company.com');

-- ========================================
-- VIEWS FOR AUDIT MONITORING
-- ========================================

-- Active Audit Jobs Dashboard
CREATE OR REPLACE VIEW v_audit_jobs_dashboard AS
SELECT 
    ajd.job_id,
    ajd.job_name,
    ajd.job_type,
    ajd.schedule_frequency,
    ajd.schedule_time,
    ajd.is_active,
    ajd.notification_email,
    -- Last execution info
    (SELECT execution_start 
     FROM audit_execution_log 
     WHERE job_id = ajd.job_id 
     ORDER BY execution_start DESC 
     LIMIT 1) AS last_execution,
    (SELECT status 
     FROM audit_execution_log 
     WHERE job_id = ajd.job_id 
     ORDER BY execution_start DESC 
     LIMIT 1) AS last_status,
    (SELECT issues_found 
     FROM audit_execution_log 
     WHERE job_id = ajd.job_id 
     ORDER BY execution_start DESC 
     LIMIT 1) AS last_issues_found,
    (SELECT warnings_found 
     FROM audit_execution_log 
     WHERE job_id = ajd.job_id 
     ORDER BY execution_start DESC 
     LIMIT 1) AS last_warnings_found,
    -- Execution statistics
    (SELECT COUNT(*) 
     FROM audit_execution_log 
     WHERE job_id = ajd.job_id 
     AND execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)) AS executions_last_30_days,
    (SELECT COUNT(*) 
     FROM audit_execution_log 
     WHERE job_id = ajd.job_id 
     AND status = 'failed'
     AND execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)) AS failures_last_30_days,
    (SELECT AVG(execution_duration_seconds) 
     FROM audit_execution_log 
     WHERE job_id = ajd.job_id 
     AND status = 'completed'
     AND execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)) AS avg_duration_seconds,
    -- Next scheduled run (simplified calculation)
    CASE 
        WHEN ajd.schedule_frequency = 'hourly' THEN DATE_ADD(NOW(), INTERVAL 1 HOUR)
        WHEN ajd.schedule_frequency = 'daily' THEN 
            DATE_ADD(CONCAT(CURDATE(), ' ', ajd.schedule_time), INTERVAL 
                CASE WHEN TIME(NOW()) >= ajd.schedule_time THEN 1 ELSE 0 END DAY)
        WHEN ajd.schedule_frequency = 'weekly' THEN DATE_ADD(NOW(), INTERVAL 7 DAY)
        WHEN ajd.schedule_frequency = 'monthly' THEN DATE_ADD(NOW(), INTERVAL 1 MONTH)
        ELSE NULL
    END AS next_scheduled_run
FROM audit_job_definitions ajd
WHERE ajd.is_active = TRUE
ORDER BY ajd.job_id;

-- Audit Findings Summary
CREATE OR REPLACE VIEW v_audit_findings_summary AS
SELECT 
    ajd.job_name,
    ajd.job_type,
    af.finding_category,
    af.severity,
    af.resolution_status,
    COUNT(*) AS finding_count,
    COUNT(CASE WHEN af.resolution_status = 'open' THEN 1 END) AS open_count,
    COUNT(CASE WHEN af.resolution_status = 'resolved' THEN 1 END) AS resolved_count,
    MIN(af.created_at) AS oldest_finding,
    MAX(af.created_at) AS newest_finding
FROM audit_findings af
JOIN audit_execution_log ael ON af.execution_id = ael.execution_id
JOIN audit_job_definitions ajd ON ael.job_id = ajd.job_id
WHERE af.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY ajd.job_name, ajd.job_type, af.finding_category, af.severity, af.resolution_status
ORDER BY 
    FIELD(af.severity, 'critical', 'high', 'medium', 'low'),
    finding_count DESC;

-- Critical Findings Requiring Attention
CREATE OR REPLACE VIEW v_critical_findings AS
SELECT 
    af.finding_id,
    ajd.job_name,
    af.severity,
    af.finding_category,
    af.finding_description,
    af.affected_table,
    af.affected_record_id,
    af.current_value,
    af.expected_value,
    af.resolution_status,
    af.created_at,
    DATEDIFF(NOW(), af.created_at) AS days_open
FROM audit_findings af
JOIN audit_execution_log ael ON af.execution_id = ael.execution_id
JOIN audit_job_definitions ajd ON ael.job_id = ajd.job_id
WHERE af.resolution_status IN ('open', 'in_progress')
    AND af.severity IN ('critical', 'high')
ORDER BY 
    FIELD(af.severity, 'critical', 'high'),
    af.created_at ASC;

-- Audit Execution Trends
CREATE OR REPLACE VIEW v_audit_execution_trends AS
SELECT 
    DATE(ael.execution_start) AS execution_date,
    ajd.job_name,
    ajd.job_type,
    COUNT(*) AS execution_count,
    SUM(CASE WHEN ael.status = 'completed' THEN 1 ELSE 0 END) AS successful_count,
    SUM(CASE WHEN ael.status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
    SUM(ael.issues_found) AS total_issues,
    SUM(ael.warnings_found) AS total_warnings,
    AVG(ael.execution_duration_seconds) AS avg_duration_seconds
FROM audit_execution_log ael
JOIN audit_job_definitions ajd ON ael.job_id = ajd.job_id
WHERE ael.execution_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(ael.execution_start), ajd.job_name, ajd.job_type
ORDER BY execution_date DESC, ajd.job_name;

-- Audit Metrics Trends
CREATE OR REPLACE VIEW v_audit_metrics_trends AS
SELECT 
    amh.metric_name,
    amh.metric_date,
    amh.metric_value,
    amh.previous_value,
    amh.variance_percentage,
    ajd.job_name,
    CASE 
        WHEN ABS(amh.variance_percentage) > 20 THEN 'High Variance'
        WHEN ABS(amh.variance_percentage) > 10 THEN 'Medium Variance'
        ELSE 'Normal'
    END AS variance_level
FROM audit_metrics_history amh
JOIN audit_execution_log ael ON amh.execution_id = ael.execution_id
JOIN audit_job_definitions ajd ON ael.job_id = ajd.job_id
WHERE amh.metric_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
ORDER BY amh.metric_date DESC, amh.metric_name;

-- ========================================
-- AUDIT MANAGEMENT PROCEDURES
-- ========================================

DELIMITER //

-- Procedure to run all active audits
CREATE PROCEDURE sp_run_all_audits()
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_job_id INT;
    DECLARE v_job_name VARCHAR(200);
    DECLARE v_error_occurred BOOLEAN DEFAULT FALSE;
    DECLARE v_error_message TEXT;
    
    DECLARE job_cursor CURSOR FOR 
        SELECT job_id, job_name 
        FROM audit_job_definitions 
        WHERE is_active = TRUE
        ORDER BY job_id;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
    BEGIN
        SET v_error_occurred = TRUE;
        GET DIAGNOSTICS CONDITION 1 v_error_message = MESSAGE_TEXT;
    END;
    
    OPEN job_cursor;
    
    audit_loop: LOOP
        FETCH job_cursor INTO v_job_id, v_job_name;
        
        IF done THEN
            LEAVE audit_loop;
        END IF;
        
        -- Execute specific audit based on job_id
        CASE v_job_id
            WHEN 1 THEN CALL sp_audit_data_integrity();
            WHEN 2 THEN CALL sp_audit_financial_reconciliation();
            WHEN 3 THEN CALL sp_audit_inventory_accuracy();
            WHEN 4 THEN CALL sp_audit_security_compliance();
            WHEN 5 THEN CALL sp_audit_performance_metrics();
            WHEN 6 THEN CALL sp_audit_campaign_performance();
        END CASE;
        
        IF v_error_occurred THEN
            SELECT CONCAT('Error running audit "', v_job_name, '": ', v_error_message) AS Error;
            SET v_error_occurred = FALSE;
        END IF;
        
    END LOOP;
    
    CLOSE job_cursor;
    
    SELECT 'All scheduled audits completed' AS Status;
END//

-- Procedure to resolve audit finding
CREATE PROCEDURE sp_resolve_audit_finding(
    IN p_finding_id INT,
    IN p_resolution_status VARCHAR(20),
    IN p_resolution_notes TEXT,
    IN p_resolved_by VARCHAR(100)
)
BEGIN
DECLARE v_deleted_executions INT;
    DECLARE v_deleted_findings INT;
    DECLARE v_deleted_metrics INT;
    
    -- Delete old audit findings
    DELETE FROM audit_findings
    WHERE created_at < DATE_SUB(NOW(), INTERVAL p_retention_days DAY);
    

    UPDATE audit_findings
    











SET 
        resolution_status = p_resolution_status,
        resolution_notes = p_resolution_notes,
        resolved_by = p_resolved_by,
        resolved_at = NOW()
    WHERE finding_id = p_finding_id;
    
    SELECT CONCAT('Finding #', p_finding_id, ' marked as ', p_resolution_status) AS Result;
END//

-- Procedure to get audit report
CREATE PROCEDURE sp_get_audit_report(
    IN p_days_back INT
)
BEGIN
    -- Summary statistics
    SELECT 
        'Audit Summary Report' AS Report_Type,
        COUNT(DISTINCT ael.execution_id) AS Total_Executions,
        SUM(ael.records_audited) AS Total_Records_Audited,
        SUM(ael.issues_found) AS Total_Issues_Found,
        SUM(ael.warnings_found) AS Total_Warnings_Found,
        SUM(CASE WHEN ael.status = 'failed' THEN 1 ELSE 0 END) AS Failed_Audits,
        ROUND(AVG(ael.execution_duration_seconds), 2) AS Avg_Duration_Seconds
    FROM audit_execution_log ael
    WHERE ael.execution_start >= DATE_SUB(NOW(), INTERVAL p_days_back DAY);
    
    -- Findings by severity
    SELECT 
        'Findings by Severity' AS Report_Section,
        af.severity,
        COUNT(*) AS finding_count,
        SUM(CASE WHEN af.resolution_status = 'open' THEN 1 ELSE 0 END) AS open_count,
        SUM(CASE WHEN af.resolution_status = 'resolved' THEN 1 ELSE 0 END) AS resolved_count
    FROM audit_findings af
    JOIN audit_execution_log ael ON af.execution_id = ael.execution_id
    WHERE ael.execution_start >= DATE_SUB(NOW(), INTERVAL p_days_back DAY)
    GROUP BY af.severity
    ORDER BY FIELD(af.severity, 'critical', 'high', 'medium', 'low');
    
    -- Top issues by category
    SELECT 
        'Top Issues by Category' AS Report_Section,
        af.finding_category,
        COUNT(*) AS issue_count,
        COUNT(DISTINCT af.affected_table) AS affected_tables
    FROM audit_findings af
    JOIN audit_execution_log ael ON af.execution_id = ael.execution_id
    WHERE ael.execution_start >= DATE_SUB(NOW(), INTERVAL p_days_back DAY)
        AND af.finding_type = 'error'
    GROUP BY af.finding_category
    ORDER BY issue_count DESC
    LIMIT 10;
END//

-- Procedure to clean up old audit data
CREATE PROCEDURE sp_cleanup_old_audit_data(
    IN p_retention_days INT
)
BEGIN
    
SET v_deleted_findings = ROW_COUNT();
    
    -- Delete old metrics
    DELETE FROM audit_metrics_history
    WHERE created_at < DATE_SUB(NOW(), INTERVAL p_retention_days DAY);
    SET v_deleted_metrics = ROW_COUNT();
    
    -- Delete old execution logs (this will cascade to findings via FK)
    DELETE FROM audit_execution_log
    WHERE execution_start < DATE_SUB(NOW(), INTERVAL p_retention_days DAY);
    SET v_deleted_executions = ROW_COUNT();
    
    SELECT 
        'Cleanup Complete' AS Status,
        v_deleted_executions AS Executions_Deleted,
        v_deleted_findings AS Findings_Deleted,
        v_deleted_metrics AS Metrics_Deleted,
        p_retention_days AS Retention_Days;
END//

-- Procedure to export audit findings
CREATE PROCEDURE sp_export_audit_findings(
    IN p_severity VARCHAR(20),
    IN p_status VARCHAR(20),
    IN p_days_back INT
)
BEGIN
    SELECT 
        af.finding_id,
        ajd.job_name,
        ael.execution_start,
        af.finding_type,
        af.severity,
        af.finding_category,
        af.finding_description,
        af.affected_table,
        af.affected_record_id,
        af.current_value,
        af.expected_value,
        af.resolution_status,
        af.resolution_notes,
        af.resolved_by,
        af.resolved_at,
        af.created_at
    FROM audit_findings af
    JOIN audit_execution_log ael ON af.execution_id = ael.execution_id
    JOIN audit_job_definitions ajd ON ael.job_id = ajd.job_id
    WHERE 
        (p_severity IS NULL OR af.severity = p_severity)
        AND (p_status IS NULL OR af.resolution_status = p_status)
        AND af.created_at >= DATE_SUB(NOW(), INTERVAL p_days_back DAY)
    ORDER BY 
        FIELD(af.severity, 'critical', 'high', 'medium', 'low'),
        af.created_at DESC;
END//

DELIMITER ;

-- ========================================
-- EXAMPLE MYSQL EVENT SCHEDULER SETUP
-- (Requires EVENT scheduler to be enabled)
-- ========================================

-- Enable event scheduler (if not already enabled)
-- SET GLOBAL event_scheduler = ON;

-- Schedule daily data integrity audit at 2 AM
CREATE EVENT IF NOT EXISTS evt_daily_data_integrity
ON SCHEDULE EVERY 1 DAY
STARTS CONCAT(CURDATE() + INTERVAL 1 DAY, ' 02:00:00')
DO CALL sp_audit_data_integrity();

-- Schedule daily financial reconciliation at 3 AM
CREATE EVENT IF NOT EXISTS evt_daily_financial_reconciliation
ON SCHEDULE EVERY 1 DAY
STARTS CONCAT(CURDATE() + INTERVAL 1 DAY, ' 03:00:00')
DO CALL sp_audit_financial_reconciliation();

-- Schedule daily inventory audit at 1 AM
CREATE EVENT IF NOT EXISTS evt_daily_inventory_audit
ON SCHEDULE EVERY 1 DAY
STARTS CONCAT(CURDATE() + INTERVAL 1 DAY, ' 01:00:00')
DO CALL sp_audit_inventory_accuracy();

-- Schedule weekly security audit (every Monday at 4 AM)
CREATE EVENT IF NOT EXISTS evt_weekly_security_audit
ON SCHEDULE EVERY 1 WEEK
STARTS (CURDATE() + INTERVAL (8 - DAYOFWEEK(CURDATE())) DAY + INTERVAL 4 HOUR)
DO CALL sp_audit_security_compliance();

-- Schedule weekly performance audit (every Monday at 5 AM)
CREATE EVENT IF NOT EXISTS evt_weekly_performance_audit
ON SCHEDULE EVERY 1 WEEK
STARTS (CURDATE() + INTERVAL (8 - DAYOFWEEK(CURDATE())) DAY + INTERVAL 5 HOUR)
DO CALL sp_audit_performance_metrics();

-- Schedule daily campaign audit at 6 AM
CREATE EVENT IF NOT EXISTS evt_daily_campaign_audit
ON SCHEDULE EVERY 1 DAY
STARTS CONCAT(CURDATE() + INTERVAL 1 DAY, ' 06:00:00')
DO CALL sp_audit_campaign_performance();

-- Schedule monthly cleanup (1st of each month at midnight)
CREATE EVENT IF NOT EXISTS evt_monthly_audit_cleanup
ON SCHEDULE EVERY 1 MONTH
STARTS CONCAT(LAST_DAY(CURDATE()) + INTERVAL 1 DAY, ' 00:00:00')
DO CALL sp_cleanup_old_audit_data(365);

-- ========================================
-- QUICK START COMMANDS
-- ========================================

-- Display completion message and next steps
SELECT '============================================' AS '';
SELECT 'Scheduled Audits System Created Successfully' AS Status;
SELECT '============================================' AS '';
SELECT '' AS '';
SELECT 'QUICK START COMMANDS:' AS '';
SELECT '-------------------------------------------' AS '';
SELECT '1. Run all audits manually:' AS '';
SELECT '   CALL sp_run_all_audits();' AS '';
SELECT '' AS '';
SELECT '2. Run specific audit:' AS '';
SELECT '   CALL sp_audit_data_integrity();' AS '';
SELECT '   CALL sp_audit_financial_reconciliation();' AS '';
SELECT '   CALL sp_audit_inventory_accuracy();' AS '';
SELECT '' AS '';
SELECT '3. View active alerts:' AS '';
SELECT '   SELECT * FROM v_critical_findings;' AS '';
SELECT '   SELECT * FROM v_audit_findings_summary;' AS '';
SELECT '' AS '';
SELECT '4. View audit dashboard:' AS '';
SELECT '   SELECT * FROM v_audit_jobs_dashboard;' AS '';
SELECT '' AS '';
SELECT '5. Get audit report (last 30 days):' AS '';
SELECT '   CALL sp_get_audit_report(30);' AS '';
SELECT '' AS '';
SELECT '6. Resolve a finding:' AS '';
SELECT '   CALL sp_resolve_audit_finding(1, "resolved", "Fixed the issue", "admin");' AS '';
SELECT '' AS '';
SELECT '7. Export findings:' AS '';
SELECT '   CALL sp_export_audit_findings("critical", "open", 30);' AS '';
SELECT '' AS '';
SELECT '8. Enable event scheduler (for automated runs):' AS '';
SELECT '   SET GLOBAL event_scheduler = ON;' AS '';
SELECT '' AS '';
SELECT '============================================' AS '';

-- Show initial audit job configuration
SELECT 'Configured Audit Jobs:' AS '';
SELECT 
    job_name,
    job_type,
    schedule_frequency,
    schedule_time,
    is_active
FROM audit_job_definitions
ORDER BY job_id;