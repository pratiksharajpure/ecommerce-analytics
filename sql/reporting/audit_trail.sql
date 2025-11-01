-- ========================================
-- AUDIT TRAIL & CHANGE TRACKING SYSTEM
-- Complete change tracking
-- User activity logs
-- Data modifications history
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- SECTION 1: AUDIT TRAIL TABLES
-- ========================================

-- 1.1 User Sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    username VARCHAR(100) NOT NULL,
    user_role ENUM('admin', 'analyst', 'developer', 'read_only', 'system') NOT NULL,
    login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    logout_time TIMESTAMP NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    session_status ENUM('active', 'expired', 'terminated') DEFAULT 'active',
    session_duration_minutes INT AS (TIMESTAMPDIFF(MINUTE, login_time, logout_time)) STORED,
    INDEX idx_user_id (user_id),
    INDEX idx_username (username),
    INDEX idx_login_time (login_time),
    INDEX idx_session_status (session_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.2 Data Modification Audit
CREATE TABLE IF NOT EXISTS audit_data_changes (
    audit_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100) NOT NULL,
    operation_type ENUM('INSERT', 'UPDATE', 'DELETE') NOT NULL,
    record_id VARCHAR(100),
    changed_by VARCHAR(100) NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    old_values JSON,
    new_values JSON,
    changed_columns JSON,
    ip_address VARCHAR(45),
    application VARCHAR(100),
    session_id INT,
    transaction_id VARCHAR(100),
    INDEX idx_table_name (table_name),
    INDEX idx_operation_type (operation_type),
    INDEX idx_changed_by (changed_by),
    INDEX idx_changed_at (changed_at),
    INDEX idx_record_id (record_id),
    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.3 Query Execution Log
CREATE TABLE IF NOT EXISTS audit_query_log (
    query_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id INT,
    username VARCHAR(100) NOT NULL,
    query_text TEXT NOT NULL,
    query_type ENUM('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DDL', 'OTHER') NOT NULL,
    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_duration_ms INT,
    rows_affected INT,
    query_status ENUM('success', 'failed', 'timeout') DEFAULT 'success',
    error_message TEXT,
    database_name VARCHAR(100),
    tables_accessed JSON,
    INDEX idx_session_id (session_id),
    INDEX idx_username (username),
    INDEX idx_execution_time (execution_time),
    INDEX idx_query_type (query_type),
    INDEX idx_query_status (query_status),
    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.4 Schema Changes Audit
CREATE TABLE IF NOT EXISTS audit_schema_changes (
    schema_change_id INT PRIMARY KEY AUTO_INCREMENT,
    change_type ENUM('CREATE', 'ALTER', 'DROP', 'RENAME', 'TRUNCATE') NOT NULL,
    object_type ENUM('TABLE', 'VIEW', 'INDEX', 'COLUMN', 'CONSTRAINT', 'PROCEDURE', 'TRIGGER') NOT NULL,
    object_name VARCHAR(200) NOT NULL,
    ddl_statement TEXT,
    changed_by VARCHAR(100) NOT NULL,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    change_reason TEXT,
    approval_required BOOLEAN DEFAULT TRUE,
    approved_by VARCHAR(100),
    approved_at TIMESTAMP NULL,
    rollback_script TEXT,
    INDEX idx_object_type (object_type),
    INDEX idx_changed_by (changed_by),
    INDEX idx_changed_at (changed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.5 Data Access Log
CREATE TABLE IF NOT EXISTS audit_data_access (
    access_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id INT,
    username VARCHAR(100) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    access_type ENUM('READ', 'EXPORT', 'PRINT', 'VIEW') NOT NULL,
    record_count INT,
    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filter_criteria TEXT,
    columns_accessed JSON,
    export_format VARCHAR(50),
    sensitive_data_flag BOOLEAN DEFAULT FALSE,
    INDEX idx_username (username),
    INDEX idx_table_name (table_name),
    INDEX idx_accessed_at (accessed_at),
    INDEX idx_sensitive_data (sensitive_data_flag),
    FOREIGN KEY (session_id) REFERENCES user_sessions(session_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.6 Security Events
CREATE TABLE IF NOT EXISTS audit_security_events (
    event_id INT PRIMARY KEY AUTO_INCREMENT,
    event_type ENUM('login_success', 'login_failure', 'logout', 'permission_denied', 'suspicious_activity', 'password_change', 'account_locked') NOT NULL,
    username VARCHAR(100) NOT NULL,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT,
    event_details JSON,
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'low',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP NULL,
    INDEX idx_event_type (event_type),
    INDEX idx_username (username),
    INDEX idx_event_timestamp (event_timestamp),
    INDEX idx_severity (severity),
    INDEX idx_resolved (resolved)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.7 Data Retention Audit
CREATE TABLE IF NOT EXISTS audit_data_retention (
    retention_id INT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100) NOT NULL,
    operation ENUM('archive', 'delete', 'purge') NOT NULL,
    records_affected INT,
    retention_policy VARCHAR(200),
    executed_by VARCHAR(100) NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    date_range_start DATE,
    date_range_end DATE,
    archive_location VARCHAR(500),
    verification_checksum VARCHAR(100),
    INDEX idx_table_name (table_name),
    INDEX idx_executed_at (executed_at),
    INDEX idx_operation (operation)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.8 Compliance Audit
CREATE TABLE IF NOT EXISTS audit_compliance (
    compliance_id INT PRIMARY KEY AUTO_INCREMENT,
    audit_type ENUM('GDPR', 'CCPA', 'SOX', 'HIPAA', 'PCI_DSS', 'INTERNAL') NOT NULL,
    audit_date DATE NOT NULL,
    auditor_name VARCHAR(100),
    table_name VARCHAR(100),
    compliance_rule VARCHAR(500),
    compliance_status ENUM('compliant', 'non_compliant', 'partially_compliant', 'pending_review') NOT NULL,
    findings TEXT,
    remediation_plan TEXT,
    remediation_deadline DATE,
    remediation_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audit_type (audit_type),
    INDEX idx_audit_date (audit_date),
    INDEX idx_compliance_status (compliance_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- SECTION 2: AUDIT TRIGGERS FOR KEY TABLES
-- ========================================

-- 2.1 Customers Table Audit Trigger
DELIMITER //

CREATE TRIGGER IF NOT EXISTS trg_customers_audit_insert
AFTER INSERT ON customers
FOR EACH ROW
BEGIN
DECLARE changed_cols JSON;
    
    

    INSERT INTO audit_data_changes (
        table_name, operation_type, record_id, changed_by, 
        new_values, changed_columns, ip_address
    ) VALUES (
        'customers', 
        'INSERT', 
        NEW.customer_id,
        COALESCE(USER(), 'system'),
        JSON_OBJECT(
            'customer_id', NEW.customer_id,
            'first_name', NEW.first_name,
            'last_name', NEW.last_name,
            'email', NEW.email,
            'phone', NEW.phone,
            'status', NEW.status
        ),
        JSON_ARRAY('customer_id', 'first_name', 'last_name', 'email', 'phone', 'status'),
        COALESCE(@client_ip, '0.0.0.0')
    );
END//

CREATE TRIGGER IF NOT EXISTS trg_customers_audit_update
AFTER UPDATE ON customers
FOR EACH ROW
BEGIN
    












SET changed_cols = JSON_ARRAY();
    
    IF OLD.first_name != NEW.first_name OR (OLD.first_name IS NULL AND NEW.first_name IS NOT NULL) OR (OLD.first_name IS NOT NULL AND NEW.first_name IS NULL) THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'first_name');
    END IF;
    IF OLD.last_name != NEW.last_name OR (OLD.last_name IS NULL AND NEW.last_name IS NOT NULL) OR (OLD.last_name IS NOT NULL AND NEW.last_name IS NULL) THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'last_name');
    END IF;
    IF OLD.email != NEW.email OR (OLD.email IS NULL AND NEW.email IS NOT NULL) OR (OLD.email IS NOT NULL AND NEW.email IS NULL) THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'email');
    END IF;
    IF OLD.phone != NEW.phone OR (OLD.phone IS NULL AND NEW.phone IS NOT NULL) OR (OLD.phone IS NOT NULL AND NEW.phone IS NULL) THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'phone');
    END IF;
    IF OLD.status != NEW.status THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'status');
    END IF;
    
    INSERT INTO audit_data_changes (
        table_name, operation_type, record_id, changed_by,
        old_values, new_values, changed_columns, ip_address
    ) VALUES (
        'customers',
        'UPDATE',
        NEW.customer_id,
        COALESCE(USER(), 'system'),
        JSON_OBJECT(
            'first_name', OLD.first_name,
            'last_name', OLD.last_name,
            'email', OLD.email,
            'phone', OLD.phone,
            'status', OLD.status
        ),
        JSON_OBJECT(
            'first_name', NEW.first_name,
            'last_name', NEW.last_name,
            'email', NEW.email,
            'phone', NEW.phone,
            'status', NEW.status
        ),
        changed_cols,
        COALESCE(@client_ip, '0.0.0.0')
    );
END//

CREATE TRIGGER IF NOT EXISTS trg_customers_audit_delete
AFTER DELETE ON customers
FOR EACH ROW
BEGIN
    INSERT INTO audit_data_changes (
        table_name, operation_type, record_id, changed_by,
        old_values, ip_address
    ) VALUES (
        'customers',
        'DELETE',
        OLD.customer_id,
        COALESCE(USER(), 'system'),
        JSON_OBJECT(
            'customer_id', OLD.customer_id,
            'first_name', OLD.first_name,
            'last_name', OLD.last_name,
            'email', OLD.email,
            'phone', OLD.phone,
            'status', OLD.status
        ),
        COALESCE(@client_ip, '0.0.0.0')
    );
END//

-- 2.2 Orders Table Audit Trigger
CREATE TRIGGER IF NOT EXISTS trg_orders_audit_insert
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    INSERT INTO audit_data_changes (
        table_name, operation_type, record_id, changed_by,
        new_values, changed_columns, ip_address
    ) VALUES (
        'orders',
        'INSERT',
        NEW.order_id,
        COALESCE(USER(), 'system'),
        JSON_OBJECT(
            'order_id', NEW.order_id,
            'customer_id', NEW.customer_id,
            'total_amount', NEW.total_amount,
            'status', NEW.status,
            'payment_status', NEW.payment_status
        ),
        JSON_ARRAY('order_id', 'customer_id', 'total_amount', 'status', 'payment_status'),
        COALESCE(@client_ip, '0.0.0.0')
    );
END//

CREATE TRIGGER IF NOT EXISTS trg_orders_audit_update
AFTER UPDATE ON orders
FOR EACH ROW
BEGIN
    DECLARE changed_cols JSON;
    SET changed_cols = JSON_ARRAY();
    
    IF OLD.total_amount != NEW.total_amount THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'total_amount');
    END IF;
    IF OLD.status != NEW.status THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'status');
    END IF;
    IF OLD.payment_status != NEW.payment_status THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'payment_status');
    END IF;
    
    INSERT INTO audit_data_changes (
        table_name, operation_type, record_id, changed_by,
        old_values, new_values, changed_columns, ip_address
    ) VALUES (
        'orders',
        'UPDATE',
        NEW.order_id,
        COALESCE(USER(), 'system'),
        JSON_OBJECT(
            'total_amount', OLD.total_amount,
            'status', OLD.status,
            'payment_status', OLD.payment_status
        ),
        JSON_OBJECT(
            'total_amount', NEW.total_amount,
            'status', NEW.status,
            'payment_status', NEW.payment_status
        ),
        changed_cols,
        COALESCE(@client_ip, '0.0.0.0')
    );
END//

-- 2.3 Products Table Audit Trigger
CREATE TRIGGER IF NOT EXISTS trg_products_audit_update
AFTER UPDATE ON products
FOR EACH ROW
BEGIN
    DECLARE changed_cols JSON;
    SET changed_cols = JSON_ARRAY();
    
    IF OLD.price != NEW.price OR (OLD.price IS NULL AND NEW.price IS NOT NULL) OR (OLD.price IS NOT NULL AND NEW.price IS NULL) THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'price');
    END IF;
    IF OLD.cost != NEW.cost OR (OLD.cost IS NULL AND NEW.cost IS NOT NULL) OR (OLD.cost IS NOT NULL AND NEW.cost IS NULL) THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'cost');
    END IF;
    IF OLD.stock_quantity != NEW.stock_quantity THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'stock_quantity');
    END IF;
    IF OLD.status != NEW.status THEN
        SET changed_cols = JSON_ARRAY_APPEND(changed_cols, '$', 'status');
    END IF;
    
    INSERT INTO audit_data_changes (
        table_name, operation_type, record_id, changed_by,
        old_values, new_values, changed_columns, ip_address
    ) VALUES (
        'products',
        'UPDATE',
        NEW.product_id,
        COALESCE(USER(), 'system'),
        JSON_OBJECT(
            'price', OLD.price,
            'cost', OLD.cost,
            'stock_quantity', OLD.stock_quantity,
            'status', OLD.status
        ),
        JSON_OBJECT(
            'price', NEW.price,
            'cost', NEW.cost,
            'stock_quantity', NEW.stock_quantity,
            'status', NEW.status
        ),
        changed_cols,
        COALESCE(@client_ip, '0.0.0.0')
    );
END//

DELIMITER ;

-- ========================================
-- SECTION 3: AUDIT QUERY PROCEDURES
-- ========================================

-- 3.1 Get Complete Audit Trail for Record
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS sp_get_record_audit_trail(
    IN p_table_name VARCHAR(100),
    IN p_record_id VARCHAR(100)
)
BEGIN
DECLARE v_record_state JSON;
    
    -- Get the latest state before the specified date
    SELECT new_values INTO v_record_state
    FROM audit_data_changes
    WHERE table_name = p_table_name
        AND record_id = p_record_id
        AND changed_at <= p_as_of_date
        AND operation_type != 'DELETE'
    ORDER BY changed_at DESC
    LIMIT 1;
    
    SELECT 
        p_table_name AS table_name,
        p_record_id AS record_id,
        p_as_of_date AS as_of_date,
        v_record_state AS record_state,
        CASE 
            WHEN v_record_state IS NULL THEN 'Record did not exist at this time'
            ELSE 'Record state reconstructed'
        END AS status;
END//

-- 6.2 Track Field-Level Changes Over Time
CREATE PROCEDURE IF NOT EXISTS sp_track_field_changes(
    IN p_table_name VARCHAR(100),
    IN p_record_id VARCHAR(100),
    IN p_field_name VARCHAR(100)
)
BEGIN
    SELECT 
        audit_id,
        changed_at,
        changed_by,
        JSON_UNQUOTE(JSON_EXTRACT(old_values, CONCAT('$.', p_field_name))) AS old_value,
        JSON_UNQUOTE(JSON_EXTRACT(new_values, CONCAT('$.', p_field_name))) AS new_value,
        ip_address
    FROM audit_data_changes
    WHERE table_name = p_table_name
        AND record_id = p_record_id
        AND JSON_CONTAINS(changed_columns, JSON_QUOTE(p_field_name))
    ORDER BY changed_at;
END//

-- 6.3 Identify Anomalous User Behavior
CREATE PROCEDURE IF NOT EXISTS sp_detect_anomalous_behavior(
    IN p_username VARCHAR(100)
)
BEGIN
    -- Calculate user's baseline behavior
    WITH user_baseline AS (
        SELECT 
            AVG(hourly_changes) AS avg_changes_per_hour,
            STDDEV(hourly_changes) AS stddev_changes
        FROM (
            SELECT 
                DATE(changed_at) AS change_date,
                HOUR(changed_at) AS hour_of_day,
                COUNT(*) AS hourly_changes
            FROM audit_data_changes
            WHERE changed_by = p_username
                AND changed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(changed_at), HOUR(changed_at)
        ) hourly_stats
    ),
    recent_activity AS (
        SELECT 
            DATE(changed_at) AS change_date,
            HOUR(changed_at) AS hour_of_day,
            COUNT(*) AS hourly_changes
        FROM audit_data_changes
        WHERE changed_by = p_username
            AND changed_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        GROUP BY DATE(changed_at), HOUR(changed_at)
    )
    SELECT 
        ra.change_date,
        ra.hour_of_day,
        ra.hourly_changes,
        ub.avg_changes_per_hour,
        ROUND((ra.hourly_changes - ub.avg_changes_per_hour) / NULLIF(ub.stddev_changes, 0), 2) AS z_score,
        CASE 
            WHEN (ra.hourly_changes - ub.avg_changes_per_hour) / NULLIF(ub.stddev_changes, 0) > 3 THEN 'CRITICAL ANOMALY'
            WHEN (ra.hourly_changes - ub.avg_changes_per_hour) / NULLIF(ub.stddev_changes, 0) > 2 THEN 'WARNING'
            ELSE 'NORMAL'
        END AS anomaly_status
    FROM recent_activity ra
    CROSS JOIN user_baseline ub
    ORDER BY z_score DESC;
END//

-- 6.4 Compare Before/After States
CREATE PROCEDURE IF NOT EXISTS sp_compare_record_states(
    IN p_audit_id BIGINT
)
BEGIN
    SELECT 
        audit_id,
        table_name,
        record_id,
        operation_type,
        changed_by,
        changed_at,
        JSON_KEYS(old_values) AS changed_fields,
        old_values AS before_state,
        new_values AS after_state,
        changed_columns
    FROM audit_data_changes
    WHERE audit_id = p_audit_id;
    
    -- Show field-by-field comparison
    SELECT 
        field_name,
        JSON_UNQUOTE(JSON_EXTRACT(old_values, CONCAT('$.', field_name))) AS old_value,
        JSON_UNQUOTE(JSON_EXTRACT(new_values, CONCAT('$.', field_name))) AS new_value
    FROM (
        SELECT JSON_KEYS(old_values) AS fields
        FROM audit_data_changes
        WHERE audit_id = p_audit_id
    ) AS field_list
    CROSS JOIN JSON_TABLE(
        field_list.fields,
        '$[*]' COLUMNS (field_name VARCHAR(100) PATH ')
    ) AS jt
    CROSS JOIN audit_data_changes adc
    WHERE adc.audit_id = p_audit_id;
END//

DELIMITER ;

-- ========================================
-- SECTION 7: AUTOMATED AUDIT MAINTENANCE
-- ========================================

-- 7.1 Archive Old Audit Records
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS sp_archive_old_audit_records(
    IN p_retention_days INT
)
BEGIN
    DECLARE v_cutoff_date DATETIME;
    DECLARE v_archived_count INT DEFAULT 0;
    
    

    SELECT 
        audit_id,
        operation_type,
        changed_by,
        changed_at,
        old_values,
        new_values,
        changed_columns,
        ip_address,
        application
    FROM audit_data_changes
    WHERE table_name = p_table_name
        AND record_id = p_record_id
    ORDER BY changed_at DESC;
END//

-- 3.2 Get User Activity Summary
CREATE PROCEDURE IF NOT EXISTS sp_get_user_activity_summary(
    IN p_username VARCHAR(100),
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    SELECT 
        'Data Changes' AS activity_type,
        COUNT(*) AS activity_count,
        MIN(changed_at) AS first_activity,
        MAX(changed_at) AS last_activity
    FROM audit_data_changes
    WHERE changed_by = p_username
        AND DATE(changed_at) BETWEEN p_start_date AND p_end_date
    
    UNION ALL
    
    SELECT 
        'Queries Executed',
        COUNT(*),
        MIN(execution_time),
        MAX(execution_time)
    FROM audit_query_log
    WHERE username = p_username
        AND DATE(execution_time) BETWEEN p_start_date AND p_end_date
    
    UNION ALL
    
    SELECT 
        'Data Access',
        COUNT(*),
        MIN(accessed_at),
        MAX(accessed_at)
    FROM audit_data_access
    WHERE username = p_username
        AND DATE(accessed_at) BETWEEN p_start_date AND p_end_date;
END//

-- 3.3 Get Suspicious Activity Report
CREATE PROCEDURE IF NOT EXISTS sp_get_suspicious_activity()
BEGIN
    -- Multiple failed logins
    SELECT 
        'Multiple Failed Logins' AS alert_type,
        username,
        COUNT(*) AS event_count,
        MAX(event_timestamp) AS last_event,
        GROUP_CONCAT(DISTINCT ip_address) AS ip_addresses
    FROM audit_security_events
    WHERE event_type = 'login_failure'
        AND event_timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
    GROUP BY username
    HAVING COUNT(*) >= 5
    
    UNION ALL
    
    -- After-hours data access
    SELECT 
        'After Hours Access',
        username,
        COUNT(*),
        MAX(accessed_at),
        NULL
    FROM audit_data_access
    WHERE (HOUR(accessed_at) < 6 OR HOUR(accessed_at) > 22)
        AND accessed_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    GROUP BY username
    HAVING COUNT(*) >= 10
    
    UNION ALL
    
    -- Bulk data exports
    SELECT 
        'Bulk Data Export',
        username,
        SUM(record_count),
        MAX(accessed_at),
        NULL
    FROM audit_data_access
    WHERE access_type = 'EXPORT'
        AND accessed_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    GROUP BY username
    HAVING SUM(record_count) >= 10000;
END//

-- 3.4 Get Data Change History by Date Range
CREATE PROCEDURE IF NOT EXISTS sp_get_changes_by_date_range(
    IN p_start_date DATETIME,
    IN p_end_date DATETIME
)
BEGIN
    SELECT 
        table_name,
        operation_type,
        COUNT(*) AS change_count,
        COUNT(DISTINCT changed_by) AS unique_users,
        MIN(changed_at) AS first_change,
        MAX(changed_at) AS last_change
    FROM audit_data_changes
    WHERE changed_at BETWEEN p_start_date AND p_end_date
    GROUP BY table_name, operation_type
    ORDER BY change_count DESC;
END//

DELIMITER ;

-- ========================================
-- SECTION 4: INSERT SAMPLE AUDIT DATA
-- ========================================

-- 4.1 Sample User Sessions
INSERT INTO user_sessions (user_id, username, user_role, login_time, ip_address, user_agent, session_status) VALUES
(1, 'john.doe', 'admin', DATE_SUB(NOW(), INTERVAL 2 HOUR), '192.168.1.100', 'Mozilla/5.0', 'active'),
(2, 'jane.smith', 'analyst', DATE_SUB(NOW(), INTERVAL 4 HOUR), '192.168.1.101', 'Mozilla/5.0', 'active'),
(3, 'bob.johnson', 'developer', DATE_SUB(NOW(), INTERVAL 6 HOUR), '192.168.1.102', 'Mozilla/5.0', 'expired'),
(4, 'alice.williams', 'read_only', DATE_SUB(NOW(), INTERVAL 1 HOUR), '192.168.1.103', 'Mozilla/5.0', 'active'),
(5, 'system', 'system', DATE_SUB(NOW(), INTERVAL 24 HOUR), '127.0.0.1', 'System Process', 'active');

-- 4.2 Sample Security Events
INSERT INTO audit_security_events (event_type, username, event_timestamp, ip_address, severity, event_details) VALUES
('login_success', 'john.doe', DATE_SUB(NOW(), INTERVAL 2 HOUR), '192.168.1.100', 'low', '{"browser": "Chrome", "os": "Windows"}'),
('login_success', 'jane.smith', DATE_SUB(NOW(), INTERVAL 4 HOUR), '192.168.1.101', 'low', '{"browser": "Firefox", "os": "Mac"}'),
('login_failure', 'unknown_user', DATE_SUB(NOW(), INTERVAL 1 HOUR), '203.0.113.42', 'medium', '{"reason": "Invalid credentials"}'),
('login_failure', 'unknown_user', DATE_SUB(NOW(), INTERVAL 55 MINUTE), '203.0.113.42', 'medium', '{"reason": "Invalid credentials"}'),
('permission_denied', 'alice.williams', DATE_SUB(NOW(), INTERVAL 30 MINUTE), '192.168.1.103', 'high', '{"attempted_action": "DELETE", "table": "customers"}');

-- 4.3 Sample Schema Changes
INSERT INTO audit_schema_changes (change_type, object_type, object_name, ddl_statement, changed_by, change_reason, approved_by, approved_at) VALUES
('CREATE', 'TABLE', 'audit_data_changes', 'CREATE TABLE audit_data_changes...', 'system', 'Initial audit system setup', 'admin', NOW()),
('ALTER', 'TABLE', 'customers', 'ALTER TABLE customers ADD COLUMN loyalty_points INT', 'john.doe', 'Add loyalty program support', 'john.doe', DATE_SUB(NOW(), INTERVAL 7 DAY)),
('CREATE', 'INDEX', 'idx_customer_email', 'CREATE INDEX idx_customer_email ON customers(email)', 'bob.johnson', 'Performance optimization', 'john.doe', DATE_SUB(NOW(), INTERVAL 14 DAY));

-- 4.4 Sample Data Retention Records
INSERT INTO audit_data_retention (table_name, operation, records_affected, retention_policy, executed_by, date_range_start, date_range_end) VALUES
('orders', 'archive', 15000, 'Archive orders older than 7 years', 'system', '2010-01-01', '2016-12-31'),
('audit_query_log', 'delete', 50000, 'Delete query logs older than 90 days', 'system', '2024-01-01', '2024-07-01'),
('audit_data_access', 'purge', 100000, 'Purge access logs older than 1 year', 'system', '2022-01-01', '2023-12-31');

-- ========================================
-- SECTION 5: AUDIT REPORTING QUERIES
-- ========================================

-- 5.1 Most Active Users Report
SELECT 
    changed_by AS username,
    COUNT(*) AS total_changes,
    SUM(CASE WHEN operation_type = 'INSERT' THEN 1 ELSE 0 END) AS inserts,
    SUM(CASE WHEN operation_type = 'UPDATE' THEN 1 ELSE 0 END) AS updates,
    SUM(CASE WHEN operation_type = 'DELETE' THEN 1 ELSE 0 END) AS deletes,
    COUNT(DISTINCT table_name) AS tables_modified,
    MIN(changed_at) AS first_change,
    MAX(changed_at) AS last_change
FROM audit_data_changes
WHERE changed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY changed_by
ORDER BY total_changes DESC
LIMIT 20;

-- 5.2 Most Modified Tables Report
SELECT 
    table_name,
    COUNT(*) AS total_changes,
    COUNT(DISTINCT changed_by) AS unique_modifiers,
    SUM(CASE WHEN operation_type = 'INSERT' THEN 1 ELSE 0 END) AS inserts,
    SUM(CASE WHEN operation_type = 'UPDATE' THEN 1 ELSE 0 END) AS updates,
    SUM(CASE WHEN operation_type = 'DELETE' THEN 1 ELSE 0 END) AS deletes,
    MIN(changed_at) AS first_change,
    MAX(changed_at) AS last_change,
    ROUND(COUNT(*) / DATEDIFF(MAX(changed_at), MIN(changed_at)), 2) AS avg_changes_per_day
FROM audit_data_changes
WHERE changed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY table_name
ORDER BY total_changes DESC;

-- 5.3 Recent Critical Changes Report
SELECT 
    adc.audit_id,
    adc.table_name,
    adc.operation_type,
    adc.record_id,
    adc.changed_by,
    adc.changed_at,
    adc.changed_columns,
    adc.ip_address,
    CASE 
        WHEN adc.operation_type = 'DELETE' THEN 'CRITICAL'
        WHEN adc.table_name IN ('customers', 'orders', 'products') AND adc.operation_type = 'UPDATE' THEN 'HIGH'
        ELSE 'MEDIUM'
    END AS change_severity
FROM audit_data_changes adc
WHERE adc.changed_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY adc.changed_at DESC
LIMIT 50;

-- 5.4 User Session Analytics
SELECT 
    us.username,
    us.user_role,
    COUNT(DISTINCT us.session_id) AS total_sessions,
    ROUND(AVG(us.session_duration_minutes), 2) AS avg_session_duration_min,
    MAX(us.session_duration_minutes) AS max_session_duration_min,
    COUNT(DISTINCT us.ip_address) AS unique_ip_addresses,
    MIN(us.login_time) AS first_login,
    MAX(us.login_time) AS last_login
FROM user_sessions us
WHERE us.login_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY us.username, us.user_role
ORDER BY total_sessions DESC;

-- 5.5 Data Access Patterns
SELECT 
    ada.username,
    ada.table_name,
    ada.access_type,
    COUNT(*) AS access_count,
    SUM(ada.record_count) AS total_records_accessed,
    MAX(ada.accessed_at) AS last_access,
    SUM(CASE WHEN ada.sensitive_data_flag = TRUE THEN 1 ELSE 0 END) AS sensitive_accesses,
    COUNT(DISTINCT DATE(ada.accessed_at)) AS days_active
FROM audit_data_access ada
WHERE ada.accessed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY ada.username, ada.table_name, ada.access_type
ORDER BY total_records_accessed DESC
LIMIT 50;

-- 5.6 Query Performance Analysis
SELECT 
    aql.query_type,
    COUNT(*) AS total_queries,
    ROUND(AVG(aql.execution_duration_ms), 2) AS avg_duration_ms,
    ROUND(MAX(aql.execution_duration_ms), 2) AS max_duration_ms,
    ROUND(MIN(aql.execution_duration_ms), 2) AS min_duration_ms,
    SUM(CASE WHEN aql.query_status = 'failed' THEN 1 ELSE 0 END) AS failed_queries,
    ROUND(100.0 * SUM(CASE WHEN aql.query_status = 'failed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_rate_pct
FROM audit_query_log aql
WHERE aql.execution_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY aql.query_type
ORDER BY total_queries DESC;

-- 5.7 Security Events Dashboard
SELECT 
    ase.event_type,
    ase.severity,
    COUNT(*) AS event_count,
    COUNT(DISTINCT ase.username) AS affected_users,
    COUNT(DISTINCT ase.ip_address) AS unique_ips,
    MAX(ase.event_timestamp) AS last_occurrence,
    SUM(CASE WHEN ase.resolved = FALSE THEN 1 ELSE 0 END) AS unresolved_count
FROM audit_security_events ase
WHERE ase.event_timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY ase.event_type, ase.severity
ORDER BY event_count DESC;

-- 5.8 Compliance Audit Status
SELECT 
    ac.audit_type,
    ac.compliance_status,
    COUNT(*) AS audit_count,
    SUM(CASE WHEN ac.remediation_completed = TRUE THEN 1 ELSE 0 END) AS completed_remediations,
    SUM(CASE WHEN ac.remediation_deadline < CURDATE() AND ac.remediation_completed = FALSE THEN 1 ELSE 0 END) AS overdue_remediations,
    COUNT(DISTINCT ac.table_name) AS tables_audited
FROM audit_compliance ac
WHERE ac.audit_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
GROUP BY ac.audit_type, ac.compliance_status
ORDER BY audit_count DESC;

-- 5.9 Change Frequency Heatmap
SELECT 
    DATE(changed_at) AS change_date,
    HOUR(changed_at) AS hour_of_day,
    COUNT(*) AS change_count,
    COUNT(DISTINCT changed_by) AS unique_users,
    GROUP_CONCAT(DISTINCT table_name ORDER BY table_name) AS affected_tables
FROM audit_data_changes
WHERE changed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY DATE(changed_at), HOUR(changed_at)
ORDER BY change_date DESC, hour_of_day;

-- 5.10 Audit Trail Completeness Check
SELECT 
    'audit_data_changes' AS audit_table,
    COUNT(*) AS record_count,
    MIN(changed_at) AS earliest_record,
    MAX(changed_at) AS latest_record,
    DATEDIFF(MAX(changed_at), MIN(changed_at)) AS days_of_coverage,
    ROUND(COUNT(*) / NULLIF(DATEDIFF(MAX(changed_at), MIN(changed_at)), 0), 2) AS avg_records_per_day
FROM audit_data_changes

UNION ALL

SELECT 
    'audit_query_log',
    COUNT(*),
    MIN(execution_time),
    MAX(execution_time),
    DATEDIFF(MAX(execution_time), MIN(execution_time)),
    ROUND(COUNT(*) / NULLIF(DATEDIFF(MAX(execution_time), MIN(execution_time)), 0), 2)
FROM audit_query_log

UNION ALL

SELECT 
    'audit_data_access',
    COUNT(*),
    MIN(accessed_at),
    MAX(accessed_at),
    DATEDIFF(MAX(accessed_at), MIN(accessed_at)),
    ROUND(COUNT(*) / NULLIF(DATEDIFF(MAX(accessed_at), MIN(accessed_at)), 0), 2)
FROM audit_data_access

UNION ALL

SELECT 
    'audit_security_events',
    COUNT(*),
    MIN(event_timestamp),
    MAX(event_timestamp),
    DATEDIFF(MAX(event_timestamp), MIN(event_timestamp)),
    ROUND(COUNT(*) / NULLIF(DATEDIFF(MAX(event_timestamp), MIN(event_timestamp)), 0), 2)
FROM audit_security_events;

-- ========================================
-- SECTION 6: ADVANCED AUDIT ANALYSIS
-- ========================================

-- 6.1 Reconstruct Record State at Point in Time
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS sp_reconstruct_record_state(
    IN p_table_name VARCHAR(100),
    IN p_record_id VARCHAR(100),
    IN p_as_of_date DATETIME
)
BEGIN
    












SET v_cutoff_date = DATE_SUB(NOW(), INTERVAL p_retention_days DAY);
    
    -- Archive audit_data_changes
    INSERT INTO audit_data_retention (
        table_name, operation, records_affected, retention_policy, 
        executed_by, date_range_start, date_range_end
    )
    SELECT 
        'audit_data_changes',
        'archive',
        COUNT(*),
        CONCAT('Archive records older than ', p_retention_days, ' days'),
        USER(),
        MIN(DATE(changed_at)),
        MAX(DATE(changed_at))
    FROM audit_data_changes
    WHERE changed_at < v_cutoff_date;
    
    -- Get count before deletion
    SELECT COUNT(*) INTO v_archived_count
    FROM audit_data_changes
    WHERE changed_at < v_cutoff_date;
    
    -- Delete archived records
    DELETE FROM audit_data_changes
    WHERE changed_at < v_cutoff_date;
    
    SELECT 
        'Audit Archive Complete' AS status,
        v_archived_count AS records_archived,
        v_cutoff_date AS cutoff_date;
END//

-- 7.2 Clean Up Old Sessions
CREATE PROCEDURE IF NOT EXISTS sp_cleanup_expired_sessions()
BEGIN
DECLARE v_cleaned_count INT;
    
    -- Update active sessions that have been inactive for > 24 hours
    UPDATE user_sessions
    

    












SET session_status = 'expired',
        logout_time = DATE_ADD(login_time, INTERVAL 24 HOUR)
    WHERE session_status = 'active'
        AND login_time < DATE_SUB(NOW(), INTERVAL 24 HOUR)
        AND logout_time IS NULL;
    
    SET v_cleaned_count = ROW_COUNT();
    
    SELECT 
        'Session Cleanup Complete' AS status,
        v_cleaned_count AS sessions_expired;
END//

-- 7.3 Generate Audit Summary Report
CREATE PROCEDURE IF NOT EXISTS sp_generate_audit_summary(
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    -- Overall Statistics
    SELECT 
        'AUDIT SUMMARY REPORT' AS report_title,
        p_start_date AS period_start,
        p_end_date AS period_end,
        DATEDIFF(p_end_date, p_start_date) + 1 AS days_covered;
    
    -- Data Changes Summary
    SELECT 
        'DATA CHANGES' AS metric_category,
        COUNT(*) AS total_changes,
        COUNT(DISTINCT table_name) AS tables_affected,
        COUNT(DISTINCT changed_by) AS unique_users,
        SUM(CASE WHEN operation_type = 'INSERT' THEN 1 ELSE 0 END) AS inserts,
        SUM(CASE WHEN operation_type = 'UPDATE' THEN 1 ELSE 0 END) AS updates,
        SUM(CASE WHEN operation_type = 'DELETE' THEN 1 ELSE 0 END) AS deletes
    FROM audit_data_changes
    WHERE DATE(changed_at) BETWEEN p_start_date AND p_end_date;
    
    -- User Activity
    SELECT 
        'USER ACTIVITY' AS metric_category,
        COUNT(DISTINCT session_id) AS total_sessions,
        COUNT(DISTINCT username) AS active_users,
        ROUND(AVG(session_duration_minutes), 2) AS avg_session_duration,
        SUM(session_duration_minutes) AS total_session_time
    FROM user_sessions
    WHERE DATE(login_time) BETWEEN p_start_date AND p_end_date;
    
    -- Security Events
    SELECT 
        'SECURITY EVENTS' AS metric_category,
        COUNT(*) AS total_events,
        SUM(CASE WHEN event_type = 'login_failure' THEN 1 ELSE 0 END) AS failed_logins,
        SUM(CASE WHEN event_type = 'permission_denied' THEN 1 ELSE 0 END) AS permission_denials,
        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical_events,
        SUM(CASE WHEN resolved = FALSE THEN 1 ELSE 0 END) AS unresolved_events
    FROM audit_security_events
    WHERE DATE(event_timestamp) BETWEEN p_start_date AND p_end_date;
    
    -- Top Active Users
    SELECT 
        'TOP 10 ACTIVE USERS' AS metric_category,
        changed_by AS username,
        COUNT(*) AS total_changes,
        COUNT(DISTINCT table_name) AS tables_modified
    FROM audit_data_changes
    WHERE DATE(changed_at) BETWEEN p_start_date AND p_end_date
    GROUP BY changed_by
    ORDER BY total_changes DESC
    LIMIT 10;
    
    -- Most Modified Tables
    SELECT 
        'TOP 10 MODIFIED TABLES' AS metric_category,
        table_name,
        COUNT(*) AS modification_count,
        COUNT(DISTINCT changed_by) AS unique_modifiers
    FROM audit_data_changes
    WHERE DATE(changed_at) BETWEEN p_start_date AND p_end_date
    GROUP BY table_name
    ORDER BY modification_count DESC
    LIMIT 10;
END//

DELIMITER ;

-- ========================================
-- SECTION 8: REAL-TIME AUDIT VIEWS
-- ========================================

-- 8.1 Current Active Sessions View
CREATE OR REPLACE VIEW vw_active_sessions AS
SELECT 
    us.session_id,
    us.username,
    us.user_role,
    us.login_time,
    TIMESTAMPDIFF(MINUTE, us.login_time, NOW()) AS session_age_minutes,
    us.ip_address,
    COUNT(DISTINCT adc.table_name) AS tables_modified,
    SUM(CASE WHEN adc.operation_type = 'UPDATE' THEN 1 ELSE 0 END) AS updates_made,
    MAX(adc.changed_at) AS last_activity
FROM user_sessions us
LEFT JOIN audit_data_changes adc ON us.session_id = adc.session_id
WHERE us.session_status = 'active'
GROUP BY us.session_id, us.username, us.user_role, us.login_time, us.ip_address;

-- 8.2 Recent Changes View
CREATE OR REPLACE VIEW vw_recent_changes AS
SELECT 
    adc.audit_id,
    adc.table_name,
    adc.operation_type,
    adc.record_id,
    adc.changed_by,
    adc.changed_at,
    TIMESTAMPDIFF(MINUTE, adc.changed_at, NOW()) AS minutes_ago,
    JSON_LENGTH(adc.changed_columns) AS fields_changed,
    adc.ip_address
FROM audit_data_changes adc
WHERE adc.changed_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
ORDER BY adc.changed_at DESC;

-- 8.3 Security Alerts View
CREATE OR REPLACE VIEW vw_security_alerts AS
SELECT 
    ase.event_id,
    ase.event_type,
    ase.username,
    ase.event_timestamp,
    TIMESTAMPDIFF(MINUTE, ase.event_timestamp, NOW()) AS minutes_ago,
    ase.severity,
    ase.ip_address,
    ase.resolved,
    CASE 
        WHEN ase.severity = 'critical' AND ase.resolved = FALSE THEN 'URGENT'
        WHEN ase.severity = 'high' AND ase.resolved = FALSE THEN 'HIGH PRIORITY'
        WHEN ase.resolved = FALSE THEN 'NEEDS ATTENTION'
        ELSE 'RESOLVED'
    END AS alert_status
FROM audit_security_events ase
WHERE ase.event_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
ORDER BY 
    CASE ase.severity
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        ELSE 4
    END,
    ase.event_timestamp DESC;

-- 8.4 Data Quality Issues View
CREATE OR REPLACE VIEW vw_audit_quality_issues AS
SELECT 
    'Missing User Attribution' AS issue_type,
    COUNT(*) AS issue_count,
    MIN(changed_at) AS first_occurrence,
    MAX(changed_at) AS last_occurrence
FROM audit_data_changes
WHERE changed_by IS NULL OR changed_by = ''
    AND changed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)

UNION ALL

SELECT 
    'Missing IP Address',
    COUNT(*),
    MIN(changed_at),
    MAX(changed_at)
FROM audit_data_changes
WHERE (ip_address IS NULL OR ip_address = '0.0.0.0')
    AND changed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)

UNION ALL

SELECT 
    'Unresolved Security Events',
    COUNT(*),
    MIN(event_timestamp),
    MAX(event_timestamp)
FROM audit_security_events
WHERE resolved = FALSE
    AND event_timestamp < DATE_SUB(NOW(), INTERVAL 24 HOUR);

-- ========================================
-- SECTION 9: COMPLIANCE REPORTING
-- ========================================

-- 9.1 GDPR Right to be Forgotten Audit
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS sp_gdpr_deletion_audit(
    IN p_customer_id INT
)
BEGIN
    -- Show all data related to the customer across tables
    SELECT 
        'customers' AS table_name,
        COUNT(*) AS record_count,
        'Personal data exists' AS status
    FROM customers
    WHERE customer_id = p_customer_id
    
    UNION ALL
    
    SELECT 
        'orders',
        COUNT(*),
        'Transaction history exists'
    FROM orders
    WHERE customer_id = p_customer_id
    
    UNION ALL
    
    SELECT 
        'audit_data_changes',
        COUNT(*),
        'Audit trail exists'
    FROM audit_data_changes
    WHERE table_name = 'customers' AND record_id = p_customer_id;
    
    -- Show recent access to customer data
    SELECT 
        'Recent Access to Customer Data' AS report_section,
        username,
        table_name,
        access_type,
        accessed_at,
        record_count
    FROM audit_data_access
    WHERE table_name = 'customers'
        AND accessed_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
    ORDER BY accessed_at DESC;
END//

-- 9.2 SOX Compliance Report
CREATE PROCEDURE IF NOT EXISTS sp_sox_compliance_report(
    IN p_report_month DATE
)
BEGIN
    -- Financial data changes
    SELECT 
        'FINANCIAL DATA CHANGES' AS section,
        table_name,
        operation_type,
        COUNT(*) AS change_count,
        COUNT(DISTINCT changed_by) AS users_involved,
        GROUP_CONCAT(DISTINCT changed_by) AS user_list
    FROM audit_data_changes
    WHERE table_name IN ('orders', 'order_items', 'returns', 'products')
        AND YEAR(changed_at) = YEAR(p_report_month)
        AND MONTH(changed_at) = MONTH(p_report_month)
    GROUP BY table_name, operation_type;
    
    -- Unauthorized access attempts
    SELECT 
        'UNAUTHORIZED ACCESS ATTEMPTS' AS section,
        username,
        event_type,
        COUNT(*) AS attempt_count,
        MAX(event_timestamp) AS last_attempt
    FROM audit_security_events
    WHERE event_type IN ('permission_denied', 'login_failure')
        AND YEAR(event_timestamp) = YEAR(p_report_month)
        AND MONTH(event_timestamp) = MONTH(p_report_month)
    GROUP BY username, event_type;
    
    -- Schema changes requiring approval
    SELECT 
        'SCHEMA CHANGES' AS section,
        change_type,
        object_type,
        object_name,
        changed_by,
        approved_by,
        changed_at
    FROM audit_schema_changes
    WHERE YEAR(changed_at) = YEAR(p_report_month)
        AND MONTH(changed_at) = MONTH(p_report_month)
    ORDER BY changed_at DESC;
END//

DELIMITER ;

-- ========================================
-- SECTION 10: AUDIT DASHBOARD SUMMARY
-- ========================================

SELECT '================================================' AS separator;
SELECT 'AUDIT TRAIL SYSTEM - DASHBOARD SUMMARY' AS title;
SELECT '================================================' AS separator;

SELECT 
    'Total Audit Records' AS metric,
    COUNT(*) AS value
FROM audit_data_changes

UNION ALL

SELECT 
    'Active Sessions',
    COUNT(*)
FROM user_sessions
WHERE session_status = 'active'

UNION ALL

SELECT 
    'Unresolved Security Events',
    COUNT(*)
FROM audit_security_events
WHERE resolved = FALSE

UNION ALL

SELECT 
    'Tables with Audit Triggers',
    COUNT(DISTINCT table_name)
FROM audit_data_changes

UNION ALL

SELECT 
    'Unique Users Tracked',
    COUNT(DISTINCT changed_by)
FROM audit_data_changes;

-- Display completion message
SELECT 
    'Audit Trail System Initialized Successfully' AS status,
    'All tables, triggers, procedures, and views created' AS message,
    NOW() AS timestamp;