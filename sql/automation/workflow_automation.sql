-- ========================================
-- WORKFLOW AUTOMATION SYSTEM
-- E-commerce Revenue Analytics Engine
-- Automated Workflows, Process Automation & Task Scheduling
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for procedure creation
DELIMITER //

-- ========================================
-- TABLE: Workflow Definitions
-- ========================================
CREATE TABLE IF NOT EXISTS workflow_definitions (
    workflow_id INT PRIMARY KEY AUTO_INCREMENT,
    workflow_name VARCHAR(100) UNIQUE NOT NULL,
    workflow_type ENUM('order_processing', 'inventory_management', 'customer_lifecycle',
                       'marketing_automation', 'data_quality', 'reporting', 'maintenance') NOT NULL,
    description TEXT,
    trigger_type ENUM('manual', 'scheduled', 'event_based', 'conditional') NOT NULL,
    trigger_config JSON, -- Schedule or event configuration
    is_active BOOLEAN DEFAULT TRUE,
    priority INT DEFAULT 5, -- 1-10, higher = more important
    timeout_minutes INT DEFAULT 60,
    max_retries INT DEFAULT 3,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    INDEX idx_workflow_type (workflow_type),
    INDEX idx_is_active (is_active),
    INDEX idx_trigger_type (trigger_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Workflow Steps
-- ========================================
CREATE TABLE IF NOT EXISTS workflow_steps (
    step_id INT PRIMARY KEY AUTO_INCREMENT,
    workflow_id INT NOT NULL,
    step_order INT NOT NULL,
    step_name VARCHAR(100) NOT NULL,
    step_type ENUM('procedure_call', 'function_call', 'sql_query', 'notification',
                   'condition', 'wait', 'approval', 'external_api') NOT NULL,
    action_config JSON NOT NULL, -- Procedure name, parameters, etc.
    retry_on_failure BOOLEAN DEFAULT TRUE,
    continue_on_error BOOLEAN DEFAULT FALSE,
    timeout_seconds INT DEFAULT 300,
    depends_on_step_id INT NULL, -- For conditional execution
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflow_definitions(workflow_id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_step_id) REFERENCES workflow_steps(step_id) ON DELETE SET NULL,
    INDEX idx_workflow_id (workflow_id),
    INDEX idx_step_order (step_order),
    UNIQUE KEY unique_workflow_step (workflow_id, step_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Workflow Executions
-- ========================================
CREATE TABLE IF NOT EXISTS workflow_executions (
    execution_id INT PRIMARY KEY AUTO_INCREMENT,
    workflow_id INT NOT NULL,
    status ENUM('pending', 'running', 'completed', 'failed', 'cancelled', 'paused') DEFAULT 'pending',
    trigger_source VARCHAR(100), -- What triggered this execution
    input_parameters JSON,
    output_data JSON,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workflow_id) REFERENCES workflow_definitions(workflow_id) ON DELETE CASCADE,
    INDEX idx_workflow_id (workflow_id),
    INDEX idx_status (status),
    INDEX idx_started_at (started_at),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Step Executions
-- ========================================
CREATE TABLE IF NOT EXISTS step_executions (
    step_execution_id INT PRIMARY KEY AUTO_INCREMENT,
    execution_id INT NOT NULL,
    step_id INT NOT NULL,
    status ENUM('pending', 'running', 'completed', 'failed', 'skipped') DEFAULT 'pending',
    input_data JSON,
    output_data JSON,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    duration_seconds INT,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES workflow_executions(execution_id) ON DELETE CASCADE,
    FOREIGN KEY (step_id) REFERENCES workflow_steps(step_id) ON DELETE CASCADE,
    INDEX idx_execution_id (execution_id),
    INDEX idx_step_id (step_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Scheduled Tasks
-- ========================================
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    task_id INT PRIMARY KEY AUTO_INCREMENT,
    task_name VARCHAR(100) UNIQUE NOT NULL,
    task_type ENUM('workflow', 'procedure', 'query', 'maintenance') NOT NULL,
    schedule_type ENUM('once', 'hourly', 'daily', 'weekly', 'monthly', 'cron') NOT NULL,
    schedule_config JSON NOT NULL, -- Cron expression or specific time
    target_action VARCHAR(255) NOT NULL, -- Workflow ID or procedure name
    parameters JSON,
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP NULL,
    next_run_at TIMESTAMP NULL,
    last_status ENUM('success', 'failed', 'skipped') NULL,
    run_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_task_name (task_name),
    INDEX idx_is_active (is_active),
    INDEX idx_next_run_at (next_run_at),
    INDEX idx_schedule_type (schedule_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Task Execution Log
-- ========================================
CREATE TABLE IF NOT EXISTS task_execution_log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    task_id INT NOT NULL,
    status ENUM('success', 'failed', 'timeout') NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP NULL,
    duration_seconds INT,
    output_message TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES scheduled_tasks(task_id) ON DELETE CASCADE,
    INDEX idx_task_id (task_id),
    INDEX idx_status (status),
    INDEX idx_started_at (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Automation Rules
-- ========================================
CREATE TABLE IF NOT EXISTS automation_rules (
    rule_id INT PRIMARY KEY AUTO_INCREMENT,
    rule_name VARCHAR(100) UNIQUE NOT NULL,
    rule_category ENUM('order', 'inventory', 'customer', 'marketing', 'quality') NOT NULL,
    condition_expression TEXT NOT NULL, -- SQL WHERE clause
    action_type ENUM('workflow', 'notification', 'update', 'procedure') NOT NULL,
    action_config JSON NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    priority INT DEFAULT 5,
    execution_limit INT DEFAULT NULL, -- Max executions per day
    executions_today INT DEFAULT 0,
    last_executed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_rule_category (rule_category),
    INDEX idx_is_active (is_active),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Process Queue
-- ========================================
CREATE TABLE IF NOT EXISTS process_queue (
    queue_id INT PRIMARY KEY AUTO_INCREMENT,
    process_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50), -- 'order', 'customer', etc.
    entity_id INT,
    priority INT DEFAULT 5,
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    parameters JSON,
    result_data JSON,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    scheduled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_process_type (process_type),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_scheduled_at (scheduled_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- PROCEDURE: Create Workflow
-- ========================================
CREATE PROCEDURE sp_create_workflow(
    IN p_workflow_name VARCHAR(100),
    IN p_workflow_type VARCHAR(50),
    IN p_trigger_type VARCHAR(50),
    IN p_trigger_config JSON,
    IN p_created_by VARCHAR(100),
    OUT p_workflow_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error creating workflow';
        SET p_workflow_id = NULL;
    END;
    
    START TRANSACTION;
    
    INSERT INTO workflow_definitions (
        workflow_name,
        workflow_type,
        trigger_type,
        trigger_config,
        created_by,
        is_active
    ) VALUES (
        p_workflow_name,
        p_workflow_type,
        p_trigger_type,
        p_trigger_config,
        p_created_by,
        TRUE
    );
    
    SET p_workflow_id = LAST_INSERT_ID();
    SET p_status_message = CONCAT('Workflow created with ID: ', p_workflow_id);
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE: Add Workflow Step
-- ========================================
CREATE PROCEDURE sp_add_workflow_step(
    IN p_workflow_id INT,
    IN p_step_order INT,
    IN p_step_name VARCHAR(100),
    IN p_step_type VARCHAR(50),
    IN p_action_config JSON,
    OUT p_step_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error adding workflow step';
        SET p_step_id = NULL;
    END;
    
    START TRANSACTION;
    
    INSERT INTO workflow_steps (
        workflow_id,
        step_order,
        step_name,
        step_type,
        action_config
    ) VALUES (
        p_workflow_id,
        p_step_order,
        p_step_name,
        p_step_type,
        p_action_config
    );
    
    SET p_step_id = LAST_INSERT_ID();
    SET p_status_message = CONCAT('Step added with ID: ', p_step_id);
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE: Execute Workflow
-- ========================================
CREATE PROCEDURE sp_execute_workflow(
    IN p_workflow_id INT,
    IN p_input_parameters JSON,
    IN p_trigger_source VARCHAR(100),
    OUT p_execution_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_is_active BOOLEAN;
    DECLARE v_workflow_name VARCHAR(100);
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error executing workflow';
        SET p_execution_id = NULL;
    END;
    
    START TRANSACTION;
    
    -- Check if workflow is active
    SELECT is_active, workflow_name
    INTO v_is_active, v_workflow_name
    FROM workflow_definitions
    WHERE workflow_id = p_workflow_id;
    
    IF v_is_active IS NULL THEN
        SET p_status_message = 'Workflow not found';
        ROLLBACK;
    ELSEIF v_is_active = FALSE THEN
        SET p_status_message = 'Workflow is not active';
        ROLLBACK;
    ELSE
        -- Create execution record
        INSERT INTO workflow_executions (
            workflow_id,
            status,
            trigger_source,
            input_parameters,
            started_at
        ) VALUES (
            p_workflow_id,
            'running',
            p_trigger_source,
            p_input_parameters,
            NOW()
        );
        
        SET p_execution_id = LAST_INSERT_ID();
        
        -- Execute workflow steps
        CALL sp_process_workflow_steps(p_execution_id, @step_status);
        
        SET p_status_message = CONCAT('Workflow "', v_workflow_name, '" execution started');
        
        COMMIT;
    END IF;
END//

-- ========================================
-- PROCEDURE: Process Workflow Steps
-- ========================================
CREATE PROCEDURE sp_process_workflow_steps(
    IN p_execution_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_step_id INT;
    DECLARE v_step_name VARCHAR(100);
    DECLARE v_step_type VARCHAR(50);
    DECLARE v_action_config JSON;
    DECLARE v_workflow_id INT;
    DECLARE v_step_status VARCHAR(20);
    
    DECLARE step_cursor CURSOR FOR
        SELECT ws.step_id, ws.step_name, ws.step_type, ws.action_config
        FROM workflow_steps ws
        JOIN workflow_executions we ON ws.workflow_id = we.workflow_id
        WHERE we.execution_id = p_execution_id
        ORDER BY ws.step_order;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    -- Get workflow_id
    SELECT workflow_id INTO v_workflow_id
    FROM workflow_executions
    WHERE execution_id = p_execution_id;
    
    OPEN step_cursor;
    
    step_loop: LOOP
        FETCH step_cursor INTO v_step_id, v_step_name, v_step_type, v_action_config;
        
        IF done THEN
            LEAVE step_loop;
        END IF;
        
        -- Create step execution record
        INSERT INTO step_executions (
            execution_id,
            step_id,
            status,
            started_at
        ) VALUES (
            p_execution_id,
            v_step_id,
            'running',
            NOW()
        );
        
        -- Execute step based on type
        SET v_step_status = 'completed'; -- Simplified; would actually execute
        
        -- Update step execution
        UPDATE step_executions
        SET status = v_step_status,
            completed_at = NOW(),
            duration_seconds = TIMESTAMPDIFF(SECOND, started_at, NOW())
        WHERE execution_id = p_execution_id
        AND step_id = v_step_id;
        
    END LOOP;
    
    CLOSE step_cursor;
    
    -- Update workflow execution status
    UPDATE workflow_executions
    SET status = 'completed',
        completed_at = NOW()
    WHERE execution_id = p_execution_id;
    
    SET p_status_message = 'Workflow steps processed successfully';
END//

-- ========================================
-- PROCEDURE: Schedule Task
-- ========================================
CREATE PROCEDURE sp_schedule_task(
    IN p_task_name VARCHAR(100),
    IN p_task_type VARCHAR(50),
    IN p_schedule_type VARCHAR(50),
    IN p_schedule_config JSON,
    IN p_target_action VARCHAR(255),
    IN p_parameters JSON,
    OUT p_task_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_next_run TIMESTAMP;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error scheduling task';
        SET p_task_id = NULL;
    END;
    
    START TRANSACTION;
    
    -- Calculate next run time
    SET v_next_run = CASE p_schedule_type
        WHEN 'hourly' THEN DATE_ADD(NOW(), INTERVAL 1 HOUR)
        WHEN 'daily' THEN DATE_ADD(NOW(), INTERVAL 1 DAY)
        WHEN 'weekly' THEN DATE_ADD(NOW(), INTERVAL 1 WEEK)
        WHEN 'monthly' THEN DATE_ADD(NOW(), INTERVAL 1 MONTH)
        ELSE NOW()
    END;
    
    INSERT INTO scheduled_tasks (
        task_name,
        task_type,
        schedule_type,
        schedule_config,
        target_action,
        parameters,
        next_run_at,
        is_active
    ) VALUES (
        p_task_name,
        p_task_type,
        p_schedule_type,
        p_schedule_config,
        p_target_action,
        p_parameters,
        v_next_run,
        TRUE
    );
    
    SET p_task_id = LAST_INSERT_ID();
    SET p_status_message = CONCAT('Task scheduled with ID: ', p_task_id);
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE: Process Scheduled Tasks
-- ========================================
CREATE PROCEDURE sp_process_scheduled_tasks()
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_task_id INT;
    DECLARE v_task_name VARCHAR(100);
    DECLARE v_task_type VARCHAR(50);
    DECLARE v_target_action VARCHAR(255);
    DECLARE v_parameters JSON;
    DECLARE v_schedule_type VARCHAR(50);
    DECLARE v_execution_status VARCHAR(20);
    DECLARE v_error_msg TEXT;
    DECLARE v_start_time TIMESTAMP;
    DECLARE v_next_run TIMESTAMP;
    
    DECLARE task_cursor CURSOR FOR
        SELECT task_id, task_name, task_type, target_action, parameters, schedule_type
        FROM scheduled_tasks
        WHERE is_active = TRUE
        AND next_run_at <= NOW()
        ORDER BY next_run_at
        LIMIT 50;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    OPEN task_cursor;
    
    task_loop: LOOP
        FETCH task_cursor INTO v_task_id, v_task_name, v_task_type, 
                               v_target_action, v_parameters, v_schedule_type;
        
        IF done THEN
            LEAVE task_loop;
        END IF;
        
        SET v_start_time = NOW();
        SET v_execution_status = 'success';
        SET v_error_msg = NULL;
        
        -- Execute based on task type
        BEGIN
            DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
            BEGIN
                SET v_execution_status = 'failed';
                SET v_error_msg = 'Task execution failed';
            END;
            
            IF v_task_type = 'workflow' THEN
                CALL sp_execute_workflow(
                    CAST(v_target_action AS UNSIGNED),
                    v_parameters,
                    'scheduled_task',
                    @exec_id,
                    @exec_msg
                );
            ELSEIF v_task_type = 'procedure' THEN
                -- Would execute stored procedure dynamically
                SET v_execution_status = 'success';
            END IF;
        END;
        
        -- Calculate next run time
        SET v_next_run = CASE v_schedule_type
            WHEN 'hourly' THEN DATE_ADD(NOW(), INTERVAL 1 HOUR)
            WHEN 'daily' THEN DATE_ADD(NOW(), INTERVAL 1 DAY)
            WHEN 'weekly' THEN DATE_ADD(NOW(), INTERVAL 1 WEEK)
            WHEN 'monthly' THEN DATE_ADD(NOW(), INTERVAL 1 MONTH)
            ELSE NULL
        END;
        
        -- Update task
        UPDATE scheduled_tasks
        SET last_run_at = v_start_time,
            next_run_at = v_next_run,
            last_status = v_execution_status,
            run_count = run_count + 1,
            failure_count = failure_count + CASE WHEN v_execution_status = 'failed' THEN 1 ELSE 0 END
        WHERE task_id = v_task_id;
        
        -- Log execution
        INSERT INTO task_execution_log (
            task_id,
            status,
            started_at,
            completed_at,
            duration_seconds,
            error_message
        ) VALUES (
            v_task_id,
            v_execution_status,
            v_start_time,
            NOW(),
            TIMESTAMPDIFF(SECOND, v_start_time, NOW()),
            v_error_msg
        );
        
    END LOOP;
    
    CLOSE task_cursor;
END//

-- ========================================
-- PROCEDURE: Add to Process Queue
-- ========================================
CREATE PROCEDURE sp_add_to_process_queue(
    IN p_process_type VARCHAR(50),
    IN p_entity_type VARCHAR(50),
    IN p_entity_id INT,
    IN p_parameters JSON,
    IN p_priority INT,
    OUT p_queue_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error adding to process queue';
        SET p_queue_id = NULL;
    END;
    
    START TRANSACTION;
    
    INSERT INTO process_queue (
        process_type,
        entity_type,
        entity_id,
        parameters,
        priority,
        status
    ) VALUES (
        p_process_type,
        p_entity_type,
        p_entity_id,
        p_parameters,
        IFNULL(p_priority, 5),
        'pending'
    );
    
    SET p_queue_id = LAST_INSERT_ID();
    SET p_status_message = 'Added to process queue';
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE: Process Queue Items
-- ========================================
CREATE PROCEDURE sp_process_queue_items(
    IN p_batch_size INT
)
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_queue_id INT;
    DECLARE v_process_type VARCHAR(50);
    DECLARE v_entity_type VARCHAR(50);
    DECLARE v_entity_id INT;
    DECLARE v_parameters JSON;
    
    DECLARE queue_cursor CURSOR FOR
        SELECT queue_id, process_type, entity_type, entity_id, parameters
        FROM process_queue
        WHERE status = 'pending'
        AND scheduled_at <= NOW()
        ORDER BY priority DESC, created_at ASC
        LIMIT p_batch_size;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    OPEN queue_cursor;
    
    queue_loop: LOOP
        FETCH queue_cursor INTO v_queue_id, v_process_type, v_entity_type, 
                                v_entity_id, v_parameters;
        
        IF done THEN
            LEAVE queue_loop;
        END IF;
        
        -- Update status to processing
        UPDATE process_queue
        SET status = 'processing',
            started_at = NOW()
        WHERE queue_id = v_queue_id;
        
        -- Process based on type (simplified)
        BEGIN
            DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
            BEGIN
                UPDATE process_queue
                SET status = 'failed',
                    completed_at = NOW(),
                    error_message = 'Processing failed',
                    retry_count = retry_count + 1
                WHERE queue_id = v_queue_id;
            END;
            
            -- Actual processing logic would go here
            
            UPDATE process_queue
            SET status = 'completed',
                completed_at = NOW()
            WHERE queue_id = v_queue_id;
        END;
        
    END LOOP;
    
    CLOSE queue_cursor;
END//

-- ========================================
-- PROCEDURE: Execute Automation Rules
-- ========================================
CREATE PROCEDURE sp_execute_automation_rules()
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_rule_id INT;
    DECLARE v_rule_name VARCHAR(100);
    DECLARE v_action_type VARCHAR(50);
    DECLARE v_action_config JSON;
    
    DECLARE rule_cursor CURSOR FOR
        SELECT rule_id, rule_name, action_type, action_config
        FROM automation_rules
        WHERE is_active = TRUE
        AND (execution_limit IS NULL OR executions_today < execution_limit)
        ORDER BY priority DESC;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    OPEN rule_cursor;
    
    rule_loop: LOOP
        FETCH rule_cursor INTO v_rule_id, v_rule_name, v_action_type, v_action_config;
        
        IF done THEN
            LEAVE rule_loop;
        END IF;
        
        -- Execute action based on type
        -- This would check conditions and execute appropriate actions
        
        -- Update execution count
        UPDATE automation_rules
        SET executions_today = executions_today + 1,
            last_executed_at = NOW()
        WHERE rule_id = v_rule_id;
        
    END LOOP;
    
    CLOSE rule_cursor;
END//

-- ========================================
-- PROCEDURE: Reset Daily Counters
-- ========================================
CREATE PROCEDURE sp_reset_daily_counters()
BEGIN
    -- Reset automation rule counters
    UPDATE automation_rules
    SET executions_today = 0
    WHERE executions_today > 0;
    
    -- Clean up old completed queue items
    DELETE FROM process_queue
    WHERE status = 'completed'
    AND completed_at < DATE_SUB(NOW(), INTERVAL 7 DAY);
    
    -- Clean up old workflow executions
    DELETE FROM workflow_executions
    WHERE status IN ('completed', 'failed')
    AND completed_at < DATE_SUB(NOW(), INTERVAL 30 DAY);
END//

-- ========================================
-- PROCEDURE: Generate Workflow Report
-- ========================================
CREATE PROCEDURE sp_generate_workflow_report(
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    SELECT 
        wd.workflow_name,
        wd.workflow_type,
        COUNT(we.execution_id) AS total_executions,
        SUM(CASE WHEN we.status = 'completed' THEN 1 ELSE 0 END) AS successful_executions,
        SUM(CASE WHEN we.status = 'failed' THEN 1 ELSE 0 END) AS failed_executions,
        AVG(TIMESTAMPDIFF(SECOND, we.started_at, we.completed_at)) AS avg_duration_seconds,
        MAX(TIMESTAMPDIFF(SECOND, we.started_at, we.completed_at)) AS max_duration_seconds
    FROM workflow_definitions wd
    LEFT JOIN workflow_executions we ON wd.workflow_id = we.workflow_id
        AND DATE(we.created_at) BETWEEN p_start_date AND p_end_date
    GROUP BY wd.workflow_id, wd.workflow_name, wd.workflow_type
    ORDER BY total_executions DESC;
END//

-- Reset delimiter
DELIMITER ;

-- ========================================
-- SEED WORKFLOW DEFINITIONS
-- ========================================
-- Create sample workflows
CALL sp_create_workflow(
    'Daily Order Processing',
    'order_processing',
    'scheduled',
    '{"schedule": "daily", "time": "06:00"}',
    'system',
    @wf1_id,
    @wf1_msg
);

CALL sp_create_workflow(
    'Low Inventory Alert',
    'inventory_management',
    'event_based',
    '{"trigger": "inventory_below_threshold"}',
    'system',
    @wf2_id,
    @wf2_msg
);

CALL sp_create_workflow(
    'Customer Onboarding',
    'customer_lifecycle',
    'event_based',
    '{"trigger": "customer_created"}',
    'system',
    @wf3_id,
    @wf3_msg
);

-- ========================================
-- SEED SCHEDULED TASKS
-- ========================================
CALL sp_schedule_task(
    'Hourly Health Check',
    'procedure',
    'hourly',
    '{"hour": "*"}',
    'sp_check_critical_thresholds',
    NULL,
    @task1_id,
    @task1_msg
);

CALL sp_schedule_task(
    'Daily Notification Cleanup',
    'procedure',
    'daily',
    '{"hour": "02:00"}',
    'sp_reset_daily_counters',
    NULL,
    @task2_id,
    @task2_msg
);

CALL sp_schedule_task(
    'Daily Abandoned Cart Emails',
    'procedure',
    'daily',
    '{"hour": "10:00"}',
    'sp_send_abandoned_cart_reminders',
    NULL,
    @task3_id,
    @task3_msg
);

-- ========================================
-- SEED AUTOMATION RULES
-- ========================================
INSERT INTO automation_rules (rule_name, rule_category, condition_expression, action_type, action_config, priority) VALUES
('High Value Order Alert', 'order', 
 'total_amount > 1000', 
 'notification', 
 '{"template": "high_value_order_alert", "recipients": ["sales@company.com"]}',
 8),

('Low Stock Auto-Reorder', 'inventory',
 'quantity_available <= reorder_level',
 'workflow',
 '{"workflow_id": 2}',
 7),

('Failed Payment Retry', 'order',
 'payment_status = "failed"',
 'procedure',
 '{"procedure": "sp_retry_payment"}',
 9),

('VIP Customer Welcome', 'customer',
 'status = "active" AND created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)',
 'workflow',
 '{"workflow_id": 3}',
 6);

-- Display confirmation
SELECT 'Workflow automation system created successfully' AS Status;

-- ========================================
-- USAGE EXAMPLES
-- ========================================
/*
-- Create a new workflow
CALL sp_create_workflow(
    'Order Fulfillment Process',
    'order_processing',
    'event_based',
    '{"trigger": "order_placed"}',
    'admin',
    @workflow_id,
    @status
);

-- Add steps to workflow
CALL sp_add_workflow_step(
    @workflow_id,
    1,
    'Validate Order',
    'procedure_call',
    '{"procedure": "sp_validate_order", "parameters": ["order_id"]}',
    @step1_id,
    @step1_msg
);

CALL sp_add_workflow_step(
    @workflow_id,
    2,
    'Check Inventory',
    'procedure_call',
    '{"procedure": "sp_check_inventory_availability", "parameters": ["order_id"]}',
    @step2_id,
    @step2_msg
);

CALL sp_add_workflow_step(
    @workflow_id,
    3,
    'Process Payment',
    'procedure_call',
    '{"procedure": "sp_process_payment", "parameters": ["order_id"]}',
    @step3_id,
    @step3_msg
);

CALL sp_add_workflow_step(
    @workflow_id,
    4,
    'Send Confirmation',
    'notification',
    '{"template": "order_confirmation", "recipient_field": "customer_email"}',
    @step4_id,
    @step4_msg
);

-- Execute a workflow
CALL sp_execute_workflow(
    1,
    '{"order_id": 12345}',
    'manual_trigger',
    @execution_id,
    @exec_status
);

-- Schedule a new task
CALL sp_schedule_task(
    'Weekly Sales Report',
    'procedure',
    'weekly',
    '{"day": "Monday", "hour": "09:00"}',
    'sp_generate_report',
    '{"report_type": "sales", "period": "weekly"}',
    @task_id,
    @task_status
);

-- Add item to process queue
CALL sp_add_to_process_queue(
    'order_fulfillment',
    'order',
    12345,
    '{"priority_shipping": true}',
    8,
    @queue_id,
    @queue_status
);

-- Process scheduled tasks (run via cron)
CALL sp_process_scheduled_tasks();

-- Process queue items (run continuously)
CALL sp_process_queue_items(50);

-- Execute automation rules (run every few minutes)
CALL sp_execute_automation_rules();

-- Generate workflow performance report
CALL sp_generate_workflow_report('2025-10-01', '2025-10-31');

-- Query workflow execution history
SELECT 
    wd.workflow_name,
    we.execution_id,
    we.status,
    we.started_at,
    we.completed_at,
    TIMESTAMPDIFF(SECOND, we.started_at, we.completed_at) AS duration_seconds
FROM workflow_executions we
JOIN workflow_definitions wd ON we.workflow_id = wd.workflow_id
WHERE DATE(we.created_at) = CURDATE()
ORDER BY we.started_at DESC;

-- View pending tasks
SELECT 
    task_name,
    schedule_type,
    next_run_at,
    last_status,
    run_count,
    failure_count
FROM scheduled_tasks
WHERE is_active = TRUE
ORDER BY next_run_at;

-- Check process queue status
SELECT 
    process_type,
    status,
    COUNT(*) AS count,
    AVG(TIMESTAMPDIFF(SECOND, scheduled_at, completed_at)) AS avg_processing_time
FROM process_queue
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY process_type, status;

-- Monitor automation rules
SELECT 
    rule_name,
    rule_category,
    executions_today,
    execution_limit,
    last_executed_at,
    is_active
FROM automation_rules
ORDER BY priority DESC;
*/