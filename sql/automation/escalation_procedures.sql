-- ========================================
-- ESCALATION PROCEDURES
-- E-commerce Revenue Analytics Engine
-- Issue Escalation, Severity Levels & Notification Routing
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for procedure creation
DELIMITER //

-- ========================================
-- SUPPORT TABLE: Issues/Incidents
-- ========================================
CREATE TABLE IF NOT EXISTS incidents (
    incident_id INT PRIMARY KEY AUTO_INCREMENT,
    incident_type ENUM('order_failure', 'payment_issue', 'inventory_critical', 
                       'data_quality', 'system_error', 'fraud_alert', 'customer_complaint') NOT NULL,
    severity_level ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    status ENUM('open', 'assigned', 'in_progress', 'escalated', 'resolved', 'closed') DEFAULT 'open',
    description TEXT,
    affected_entity_type VARCHAR(50), -- 'order', 'customer', 'product', etc.
    affected_entity_id INT,
    assigned_to VARCHAR(100),
    escalated_to VARCHAR(100),
    escalation_level INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    escalated_at TIMESTAMP NULL,
    resolved_at TIMESTAMP NULL,
    INDEX idx_severity (severity_level),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_incident_type (incident_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- SUPPORT TABLE: Notification Log
-- ========================================
CREATE TABLE IF NOT EXISTS notification_log (
    notification_id INT PRIMARY KEY AUTO_INCREMENT,
    incident_id INT,
    recipient VARCHAR(100) NOT NULL,
    notification_type ENUM('email', 'sms', 'slack', 'pagerduty') NOT NULL,
    subject VARCHAR(255),
    message TEXT,
    status ENUM('pending', 'sent', 'failed') DEFAULT 'pending',
    sent_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id) ON DELETE CASCADE,
    INDEX idx_incident_id (incident_id),
    INDEX idx_status (status),
    INDEX idx_sent_at (sent_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- SUPPORT TABLE: Escalation Rules
-- ========================================
CREATE TABLE IF NOT EXISTS escalation_rules (
    rule_id INT PRIMARY KEY AUTO_INCREMENT,
    incident_type ENUM('order_failure', 'payment_issue', 'inventory_critical', 
                       'data_quality', 'system_error', 'fraud_alert', 'customer_complaint') NOT NULL,
    severity_level ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    time_threshold_minutes INT NOT NULL, -- Time before escalation
    escalation_level INT NOT NULL,
    escalate_to VARCHAR(100) NOT NULL, -- Role or email
    notification_type ENUM('email', 'sms', 'slack', 'pagerduty') NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_incident_type (incident_type),
    INDEX idx_severity (severity_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- PROCEDURE: Create Incident
-- ========================================
CREATE PROCEDURE sp_create_incident(
    IN p_incident_type VARCHAR(50),
    IN p_severity_level VARCHAR(20),
    IN p_description TEXT,
    IN p_entity_type VARCHAR(50),
    IN p_entity_id INT,
    OUT p_incident_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_assigned_to VARCHAR(100);
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error creating incident';
        SET p_incident_id = NULL;
    END;
    
    START TRANSACTION;
    
    -- Determine initial assignment based on severity
    SET v_assigned_to = CASE p_severity_level
        WHEN 'low' THEN 'tier1_support@company.com'
        WHEN 'medium' THEN 'tier2_support@company.com'
        WHEN 'high' THEN 'senior_support@company.com'
        WHEN 'critical' THEN 'oncall_manager@company.com'
        ELSE 'tier1_support@company.com'
    END;
    
    -- Create incident
    INSERT INTO incidents (
        incident_type, 
        severity_level, 
        description, 
        affected_entity_type,
        affected_entity_id,
        assigned_to,
        status
    ) VALUES (
        p_incident_type,
        p_severity_level,
        p_description,
        p_entity_type,
        p_entity_id,
        v_assigned_to,
        'open'
    );
    
    SET p_incident_id = LAST_INSERT_ID();
    
    -- Send initial notification
    CALL sp_send_notification(
        p_incident_id,
        v_assigned_to,
        'email',
        CONCAT('New ', p_severity_level, ' Incident: ', p_incident_type),
        p_description,
        @notify_status
    );
    
    COMMIT;
    SET p_status_message = CONCAT('Incident created with ID: ', p_incident_id);
END//

-- ========================================
-- PROCEDURE: Escalate Incident
-- ========================================
CREATE PROCEDURE sp_escalate_incident(
    IN p_incident_id INT,
    IN p_reason TEXT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_current_level INT;
    DECLARE v_severity VARCHAR(20);
    DECLARE v_incident_type VARCHAR(50);
    DECLARE v_escalate_to VARCHAR(100);
    DECLARE v_notification_type VARCHAR(20);
    DECLARE v_time_threshold INT;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error escalating incident';
    END;
    
    START TRANSACTION;
    
    -- Get current incident details
    SELECT escalation_level, severity_level, incident_type
    INTO v_current_level, v_severity, v_incident_type
    FROM incidents
    WHERE incident_id = p_incident_id;
    
    -- Get escalation rule
    SELECT escalate_to, notification_type
    INTO v_escalate_to, v_notification_type
    FROM escalation_rules
    WHERE incident_type = v_incident_type
    AND severity_level = v_severity
    AND escalation_level = v_current_level + 1
    AND is_active = TRUE
    LIMIT 1;
    
    IF v_escalate_to IS NOT NULL THEN
        -- Update incident
        UPDATE incidents
        SET escalation_level = escalation_level + 1,
            escalated_to = v_escalate_to,
            escalated_at = NOW(),
            status = 'escalated',
            description = CONCAT(description, '\n\nESCALATION: ', p_reason)
        WHERE incident_id = p_incident_id;
        
        -- Send escalation notification
        CALL sp_send_notification(
            p_incident_id,
            v_escalate_to,
            v_notification_type,
            CONCAT('ESCALATED: ', v_severity, ' - ', v_incident_type),
            CONCAT('Incident escalated. Reason: ', p_reason),
            @notify_status
        );
        
        SET p_status_message = CONCAT('Incident escalated to: ', v_escalate_to);
    ELSE
        SET p_status_message = 'No escalation rule found for next level';
    END IF;
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE: Auto-Escalate Based on Time
-- ========================================
CREATE PROCEDURE sp_auto_escalate_incidents()
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_incident_id INT;
    DECLARE v_severity VARCHAR(20);
    DECLARE v_incident_type VARCHAR(50);
    DECLARE v_created_at TIMESTAMP;
    DECLARE v_escalation_level INT;
    DECLARE v_time_threshold INT;
    
    DECLARE incident_cursor CURSOR FOR
        SELECT i.incident_id, i.severity_level, i.incident_type, 
               i.created_at, i.escalation_level
        FROM incidents i
        WHERE i.status IN ('open', 'assigned', 'in_progress')
        AND i.resolved_at IS NULL;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    OPEN incident_cursor;
    
    read_loop: LOOP
        FETCH incident_cursor INTO v_incident_id, v_severity, v_incident_type, 
                                   v_created_at, v_escalation_level;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        -- Get time threshold for escalation
        SELECT time_threshold_minutes
        INTO v_time_threshold
        FROM escalation_rules
        WHERE incident_type = v_incident_type
        AND severity_level = v_severity
        AND escalation_level = v_escalation_level + 1
        AND is_active = TRUE
        LIMIT 1;
        
        -- Check if threshold exceeded
        IF v_time_threshold IS NOT NULL AND 
           TIMESTAMPDIFF(MINUTE, v_created_at, NOW()) >= v_time_threshold THEN
            
            CALL sp_escalate_incident(
                v_incident_id,
                CONCAT('Auto-escalated after ', v_time_threshold, ' minutes'),
                @escalate_status
            );
        END IF;
        
    END LOOP;
    
    CLOSE incident_cursor;
END//

-- ========================================
-- PROCEDURE: Send Notification
-- ========================================
CREATE PROCEDURE sp_send_notification(
    IN p_incident_id INT,
    IN p_recipient VARCHAR(100),
    IN p_notification_type VARCHAR(20),
    IN p_subject VARCHAR(255),
    IN p_message TEXT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        

    












SET p_status_message = 'Error sending notification';
    END;
    
    -- Log notification (actual sending would be handled by external service)
    INSERT INTO notification_log (
        incident_id,
        recipient,
        notification_type,
        subject,
        message,
        status,
        sent_at
    ) VALUES (
        p_incident_id,
        p_recipient,
        p_notification_type,
        p_subject,
        p_message,
        'sent', -- In production, this would be 'pending' until confirmed
        NOW()
    );
    
    SET p_status_message = CONCAT('Notification logged for: ', p_recipient);
END//

-- ========================================
-- PROCEDURE: Resolve Incident
-- ========================================
CREATE PROCEDURE sp_resolve_incident(
    IN p_incident_id INT,
    IN p_resolution_notes TEXT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_assigned_to VARCHAR(100);
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error resolving incident';
    END;
    
    START TRANSACTION;
    
    -- Get assigned person for notification
    SELECT assigned_to INTO v_assigned_to
    FROM incidents
    WHERE incident_id = p_incident_id;
    
    -- Update incident status
    UPDATE incidents
    SET status = 'resolved',
        resolved_at = NOW(),
        description = CONCAT(description, '\n\nRESOLUTION: ', p_resolution_notes)
    WHERE incident_id = p_incident_id;
    
    -- Send resolution notification
    CALL sp_send_notification(
        p_incident_id,
        v_assigned_to,
        'email',
        'Incident Resolved',
        p_resolution_notes,
        @notify_status
    );
    
    COMMIT;
    SET p_status_message = 'Incident resolved successfully';
END//

-- ========================================
-- PROCEDURE: Check Critical Thresholds
-- ========================================
CREATE PROCEDURE sp_check_critical_thresholds()
BEGIN
    DECLARE v_low_inventory_count INT;
    DECLARE v_failed_payments_count INT;
    DECLARE v_pending_returns_count INT;
    
    -- Check for critical low inventory
    SELECT COUNT(*)
    INTO v_low_inventory_count
    FROM inventory
    WHERE quantity_available <= reorder_level
    AND quantity_available > 0;
    
    IF v_low_inventory_count > 10 THEN
        CALL sp_create_incident(
            'inventory_critical',
            'high',
            CONCAT(v_low_inventory_count, ' products below reorder level'),
            'inventory',
            NULL,
            @incident_id,
            @status
        );
    END IF;
    
    -- Check for failed payments in last hour
    SELECT COUNT(*)
    INTO v_failed_payments_count
    FROM orders
    WHERE payment_status = 'failed'
    AND created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR);
    
    IF v_failed_payments_count > 5 THEN
        CALL sp_create_incident(
            'payment_issue',
            'critical',
            CONCAT(v_failed_payments_count, ' failed payments in last hour'),
            'order',
            NULL,
            @incident_id,
            @status
        );
    END IF;
    
    -- Check for pending returns older than 7 days
    SELECT COUNT(*)
    INTO v_pending_returns_count
    FROM returns
    WHERE status = 'requested'
    AND created_at <= DATE_SUB(NOW(), INTERVAL 7 DAY);
    
    IF v_pending_returns_count > 0 THEN
        CALL sp_create_incident(
            'customer_complaint',
            'medium',
            CONCAT(v_pending_returns_count, ' returns pending over 7 days'),
            'returns',
            NULL,
            @incident_id,
            @status
        );
    END IF;
END//

-- ========================================
-- PROCEDURE: Generate Escalation Report
-- ========================================
CREATE PROCEDURE sp_generate_escalation_report(
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    SELECT 
        incident_type,
        severity_level,
        COUNT(*) AS total_incidents,
        SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) AS resolved_count,
        SUM(CASE WHEN escalation_level > 0 THEN 1 ELSE 0 END) AS escalated_count,
        AVG(TIMESTAMPDIFF(MINUTE, created_at, 
            COALESCE(resolved_at, NOW()))) AS avg_resolution_time_minutes,
        MAX(escalation_level) AS max_escalation_level
    FROM incidents
    WHERE DATE(created_at) BETWEEN p_start_date AND p_end_date
    GROUP BY incident_type, severity_level
    ORDER BY severity_level DESC, total_incidents DESC;
END//

-- Reset delimiter
DELIMITER ;

-- ========================================
-- SEED ESCALATION RULES
-- ========================================
INSERT INTO escalation_rules (incident_type, severity_level, time_threshold_minutes, escalation_level, escalate_to, notification_type) VALUES
-- Critical incidents
('payment_issue', 'critical', 15, 1, 'payment_team_lead@company.com', 'slack'),
('payment_issue', 'critical', 30, 2, 'cto@company.com', 'pagerduty'),
('fraud_alert', 'critical', 10, 1, 'fraud_team@company.com', 'pagerduty'),
('fraud_alert', 'critical', 20, 2, 'security_officer@company.com', 'pagerduty'),
('system_error', 'critical', 15, 1, 'devops_lead@company.com', 'slack'),
('system_error', 'critical', 30, 2, 'cto@company.com', 'pagerduty'),

-- High severity incidents
('order_failure', 'high', 30, 1, 'operations_manager@company.com', 'email'),
('order_failure', 'high', 120, 2, 'vp_operations@company.com', 'slack'),
('inventory_critical', 'high', 60, 1, 'inventory_manager@company.com', 'email'),
('inventory_critical', 'high', 240, 2, 'supply_chain_director@company.com', 'email'),

-- Medium severity incidents
('customer_complaint', 'medium', 120, 1, 'customer_success_lead@company.com', 'email'),
('customer_complaint', 'medium', 480, 2, 'customer_success_director@company.com', 'email'),
('data_quality', 'medium', 240, 1, 'data_team_lead@company.com', 'email'),

-- Low severity incidents
('data_quality', 'low', 1440, 1, 'data_analyst@company.com', 'email');

-- Display confirmation
SELECT 'Escalation procedures, tables, and rules created successfully' AS Status;

-- ========================================
-- USAGE EXAMPLES
-- ========================================
/*
-- Create a new incident
CALL sp_create_incident(
    'payment_issue',
    'critical',
    'Multiple payment gateway timeouts detected',
    'order',
    12345,
    @incident_id,
    @status
);
SELECT @incident_id, @status;

-- Manually escalate an incident
CALL sp_escalate_incident(
    1,
    'Customer is VIP - needs immediate attention',
    @status
);

-- Run auto-escalation check (should be scheduled)
CALL sp_auto_escalate_incidents();

-- Check for critical thresholds
CALL sp_check_critical_thresholds();

-- Resolve an incident
CALL sp_resolve_incident(
    1,
    'Payment gateway restored. All pending orders processed.',
    @status
);

-- Generate escalation report
CALL sp_generate_escalation_report('2025-10-01', '2025-10-31');
*/