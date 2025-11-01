-- ========================================
-- INTEGRATION SCRIPTS
-- E-commerce Revenue Analytics Engine
-- Third-Party Integrations, Data Sync & API Logic
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for procedure creation
DELIMITER //

-- ========================================
-- TABLE: Integration Endpoints
-- ========================================
CREATE TABLE IF NOT EXISTS integration_endpoints (
    endpoint_id INT PRIMARY KEY AUTO_INCREMENT,
    endpoint_name VARCHAR(100) UNIQUE NOT NULL,
    provider VARCHAR(50) NOT NULL, -- 'stripe', 'shopify', 'mailchimp', etc.
    endpoint_type ENUM('payment', 'shipping', 'marketing', 'analytics', 
                       'crm', 'inventory', 'accounting', 'erp') NOT NULL,
    base_url VARCHAR(255) NOT NULL,
    api_version VARCHAR(20),
    auth_type ENUM('api_key', 'oauth2', 'basic', 'bearer', 'hmac') NOT NULL,
    auth_config JSON, -- Encrypted credentials
    rate_limit_per_minute INT DEFAULT 60,
    timeout_seconds INT DEFAULT 30,
    is_active BOOLEAN DEFAULT TRUE,
    is_sandbox BOOLEAN DEFAULT FALSE,
    last_sync_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_provider (provider),
    INDEX idx_endpoint_type (endpoint_type),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Integration Mappings
-- ========================================
CREATE TABLE IF NOT EXISTS integration_mappings (
    mapping_id INT PRIMARY KEY AUTO_INCREMENT,
    endpoint_id INT NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- 'customer', 'order', 'product'
    source_field VARCHAR(100) NOT NULL,
    target_field VARCHAR(100) NOT NULL,
    field_type ENUM('string', 'integer', 'decimal', 'date', 'boolean', 'json') NOT NULL,
    transformation_rule TEXT, -- SQL expression or function
    is_required BOOLEAN DEFAULT FALSE,
    default_value VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (endpoint_id) REFERENCES integration_endpoints(endpoint_id) ON DELETE CASCADE,
    INDEX idx_endpoint_id (endpoint_id),
    INDEX idx_entity_type (entity_type),
    UNIQUE KEY unique_mapping (endpoint_id, entity_type, source_field)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Sync Jobs
-- ========================================
CREATE TABLE IF NOT EXISTS sync_jobs (
    job_id INT PRIMARY KEY AUTO_INCREMENT,
    endpoint_id INT NOT NULL,
    job_name VARCHAR(100) NOT NULL,
    sync_type ENUM('full', 'incremental', 'real_time') NOT NULL,
    sync_direction ENUM('import', 'export', 'bidirectional') NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    schedule_config JSON, -- Cron expression or interval
    filter_criteria JSON, -- WHERE conditions
    batch_size INT DEFAULT 100,
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP NULL,
    last_run_status ENUM('success', 'failed', 'partial') NULL,
    next_run_at TIMESTAMP NULL,
    records_processed INT DEFAULT 0,
    records_failed INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (endpoint_id) REFERENCES integration_endpoints(endpoint_id) ON DELETE CASCADE,
    INDEX idx_endpoint_id (endpoint_id),
    INDEX idx_is_active (is_active),
    INDEX idx_next_run_at (next_run_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Sync Execution Log
-- ========================================
CREATE TABLE IF NOT EXISTS sync_execution_log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    job_id INT NOT NULL,
    execution_status ENUM('started', 'running', 'completed', 'failed') NOT NULL,
    records_synced INT DEFAULT 0,
    records_failed INT DEFAULT 0,
    records_skipped INT DEFAULT 0,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP NULL,
    duration_seconds INT,
    error_message TEXT,
    response_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES sync_jobs(job_id) ON DELETE CASCADE,
    INDEX idx_job_id (job_id),
    INDEX idx_started_at (started_at),
    INDEX idx_execution_status (execution_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: API Request Log
-- ========================================
CREATE TABLE IF NOT EXISTS api_request_log (
    request_id INT PRIMARY KEY AUTO_INCREMENT,
    endpoint_id INT NOT NULL,
    request_method ENUM('GET', 'POST', 'PUT', 'PATCH', 'DELETE') NOT NULL,
    request_url VARCHAR(500) NOT NULL,
    request_headers JSON,
    request_body JSON,
    response_status INT,
    response_headers JSON,
    response_body JSON,
    response_time_ms INT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (endpoint_id) REFERENCES integration_endpoints(endpoint_id) ON DELETE CASCADE,
    INDEX idx_endpoint_id (endpoint_id),
    INDEX idx_created_at (created_at),
    INDEX idx_response_status (response_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Webhook Events
-- ========================================
CREATE TABLE IF NOT EXISTS webhook_events (
    event_id INT PRIMARY KEY AUTO_INCREMENT,
    endpoint_id INT,
    event_type VARCHAR(100) NOT NULL,
    event_source VARCHAR(100) NOT NULL,
    payload JSON NOT NULL,
    signature VARCHAR(255), -- For verification
    status ENUM('received', 'processing', 'processed', 'failed', 'ignored') DEFAULT 'received',
    processed_at TIMESTAMP NULL,
    retry_count INT DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (endpoint_id) REFERENCES integration_endpoints(endpoint_id) ON DELETE SET NULL,
    INDEX idx_endpoint_id (endpoint_id),
    INDEX idx_event_type (event_type),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Data Transformation Cache
-- ========================================
CREATE TABLE IF NOT EXISTS data_transformation_cache (
    cache_id INT PRIMARY KEY AUTO_INCREMENT,
    endpoint_id INT NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id INT NOT NULL,
    external_id VARCHAR(255), -- ID from external system
    cached_data JSON,
    last_synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sync_hash VARCHAR(64), -- MD5 hash for change detection
    is_dirty BOOLEAN DEFAULT FALSE, -- Needs re-sync
    FOREIGN KEY (endpoint_id) REFERENCES integration_endpoints(endpoint_id) ON DELETE CASCADE,
    UNIQUE KEY unique_entity_mapping (endpoint_id, entity_type, entity_id),
    INDEX idx_endpoint_id (endpoint_id),
    INDEX idx_entity_type (entity_type),
    INDEX idx_external_id (external_id),
    INDEX idx_is_dirty (is_dirty)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Integration Errors
-- ========================================
CREATE TABLE IF NOT EXISTS integration_errors (
    error_id INT PRIMARY KEY AUTO_INCREMENT,
    endpoint_id INT,
    error_type ENUM('connection', 'authentication', 'rate_limit', 'validation', 
                    'mapping', 'data', 'timeout', 'unknown') NOT NULL,
    error_code VARCHAR(50),
    error_message TEXT NOT NULL,
    context_data JSON,
    resolution_status ENUM('unresolved', 'investigating', 'resolved', 'ignored') DEFAULT 'unresolved',
    resolved_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (endpoint_id) REFERENCES integration_endpoints(endpoint_id) ON DELETE SET NULL,
    INDEX idx_endpoint_id (endpoint_id),
    INDEX idx_error_type (error_type),
    INDEX idx_resolution_status (resolution_status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- PROCEDURE: Register Integration Endpoint
-- ========================================
CREATE PROCEDURE sp_register_integration(
    IN p_endpoint_name VARCHAR(100),
    IN p_provider VARCHAR(50),
    IN p_endpoint_type VARCHAR(50),
    IN p_base_url VARCHAR(255),
    IN p_auth_type VARCHAR(20),
    IN p_auth_config JSON,
    OUT p_endpoint_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error registering integration';
        SET p_endpoint_id = NULL;
    END;
    
    START TRANSACTION;
    
    INSERT INTO integration_endpoints (
        endpoint_name,
        provider,
        endpoint_type,
        base_url,
        auth_type,
        auth_config,
        is_active
    ) VALUES (
        p_endpoint_name,
        p_provider,
        p_endpoint_type,
        p_base_url,
        p_auth_type,
        p_auth_config,
        TRUE
    );
    
    SET p_endpoint_id = LAST_INSERT_ID();
    SET p_status_message = CONCAT('Integration registered with ID: ', p_endpoint_id);
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE: Create Field Mapping
-- ========================================
CREATE PROCEDURE sp_create_field_mapping(
    IN p_endpoint_id INT,
    IN p_entity_type VARCHAR(50),
    IN p_source_field VARCHAR(100),
    IN p_target_field VARCHAR(100),
    IN p_field_type VARCHAR(20),
    IN p_transformation_rule TEXT,
    OUT p_mapping_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error creating field mapping';
        SET p_mapping_id = NULL;
    END;
    
    START TRANSACTION;
    
    INSERT INTO integration_mappings (
        endpoint_id,
        entity_type,
        source_field,
        target_field,
        field_type,
        transformation_rule
    ) VALUES (
        p_endpoint_id,
        p_entity_type,
        p_source_field,
        p_target_field,
        p_field_type,
        p_transformation_rule
    );
    
    SET p_mapping_id = LAST_INSERT_ID();
    SET p_status_message = 'Field mapping created successfully';
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE: Create Sync Job
-- ========================================
CREATE PROCEDURE sp_create_sync_job(
    IN p_endpoint_id INT,
    IN p_job_name VARCHAR(100),
    IN p_sync_type VARCHAR(20),
    IN p_sync_direction VARCHAR(20),
    IN p_entity_type VARCHAR(50),
    IN p_schedule_config JSON,
    OUT p_job_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_next_run TIMESTAMP;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error creating sync job';
        SET p_job_id = NULL;
    END;
    
    START TRANSACTION;
    
    -- Calculate next run time (simplified)
    SET v_next_run = DATE_ADD(NOW(), INTERVAL 1 HOUR);
    
    INSERT INTO sync_jobs (
        endpoint_id,
        job_name,
        sync_type,
        sync_direction,
        entity_type,
        schedule_config,
        next_run_at,
        is_active
    ) VALUES (
        p_endpoint_id,
        p_job_name,
        p_sync_type,
        p_sync_direction,
        p_entity_type,
        p_schedule_config,
        v_next_run,
        TRUE
    );
    
    SET p_job_id = LAST_INSERT_ID();
    SET p_status_message = CONCAT('Sync job created with ID: ', p_job_id);
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE: Execute Sync Job
-- ========================================
CREATE PROCEDURE sp_execute_sync_job(
    IN p_job_id INT,
    OUT p_log_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_endpoint_id INT;
    DECLARE v_entity_type VARCHAR(50);
    DECLARE v_sync_direction VARCHAR(20);
    DECLARE v_sync_type VARCHAR(20);
    DECLARE v_batch_size INT;
    DECLARE v_start_time TIMESTAMP;
    DECLARE v_records_synced INT DEFAULT 0;
    DECLARE v_records_failed INT DEFAULT 0;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        
        UPDATE sync_execution_log
        

    












SET execution_status = 'failed',
            completed_at = NOW(),
            duration_seconds = TIMESTAMPDIFF(SECOND, started_at, NOW()),
            error_message = 'Sync job execution failed'
        WHERE log_id = p_log_id;
        
        SET p_status_message = 'Sync job failed';
    END;
    
    START TRANSACTION;
    
    SET v_start_time = NOW();
    
    -- Get job details
    SELECT endpoint_id, entity_type, sync_direction, sync_type, batch_size
    INTO v_endpoint_id, v_entity_type, v_sync_direction, v_sync_type, v_batch_size
    FROM sync_jobs
    WHERE job_id = p_job_id;
    
    -- Create execution log entry
    INSERT INTO sync_execution_log (
        job_id,
        execution_status,
        started_at
    ) VALUES (
        p_job_id,
        'started',
        v_start_time
    );
    
    SET p_log_id = LAST_INSERT_ID();
    
    -- Update log to running
    UPDATE sync_execution_log
    SET execution_status = 'running'
    WHERE log_id = p_log_id;
    
    -- Execute sync based on direction and entity type
    IF v_sync_direction = 'export' THEN
        CALL sp_export_data(v_endpoint_id, v_entity_type, v_batch_size, v_records_synced, v_records_failed);
    ELSEIF v_sync_direction = 'import' THEN
        CALL sp_import_data(v_endpoint_id, v_entity_type, v_batch_size, v_records_synced, v_records_failed);
    END IF;
    
    -- Update execution log
    UPDATE sync_execution_log
    SET execution_status = 'completed',
        records_synced = v_records_synced,
        records_failed = v_records_failed,
        completed_at = NOW(),
        duration_seconds = TIMESTAMPDIFF(SECOND, started_at, NOW())
    WHERE log_id = p_log_id;
    
    -- Update sync job
    UPDATE sync_jobs
    SET last_run_at = v_start_time,
        last_run_status = CASE WHEN v_records_failed > 0 THEN 'partial' ELSE 'success' END,
        records_processed = records_processed + v_records_synced,
        records_failed = records_failed + v_records_failed,
        next_run_at = DATE_ADD(NOW(), INTERVAL 1 HOUR)
    WHERE job_id = p_job_id;
    
    COMMIT;
    
    SET p_status_message = CONCAT('Sync completed: ', v_records_synced, ' synced, ', v_records_failed, ' failed');
END//

-- ========================================
-- PROCEDURE: Export Data
-- ========================================
CREATE PROCEDURE sp_export_data(
    IN p_endpoint_id INT,
    IN p_entity_type VARCHAR(50),
    IN p_batch_size INT,
    OUT p_records_synced INT,
    OUT p_records_failed INT
)
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_entity_id INT;
    DECLARE v_sync_hash VARCHAR(64);
    DECLARE v_current_hash VARCHAR(64);
    
    -- Cursor for entities that need export
    DECLARE export_cursor CURSOR FOR
        SELECT entity_id, sync_hash
        FROM data_transformation_cache
        WHERE endpoint_id = p_endpoint_id
        AND entity_type = p_entity_type
        AND is_dirty = TRUE
        LIMIT p_batch_size;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    SET p_records_synced = 0;
    SET p_records_failed = 0;
    
    OPEN export_cursor;
    
    export_loop: LOOP
        FETCH export_cursor INTO v_entity_id, v_sync_hash;
        
        IF done THEN
            LEAVE export_loop;
        END IF;
        
        -- Simulate API call to export data
        -- In production, this would call external API
        
        BEGIN
            DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
            BEGIN
                SET p_records_failed = p_records_failed + 1;
            END;
            
            -- Mark as synced
            UPDATE data_transformation_cache
            SET is_dirty = FALSE,
                last_synced_at = NOW()
            WHERE endpoint_id = p_endpoint_id
            AND entity_type = p_entity_type
            AND entity_id = v_entity_id;
            
            SET p_records_synced = p_records_synced + 1;
        END;
        
    END LOOP;
    
    CLOSE export_cursor;
END//

-- ========================================
-- PROCEDURE: Import Data
-- ========================================
CREATE PROCEDURE sp_import_data(
    IN p_endpoint_id INT,
    IN p_entity_type VARCHAR(50),
    IN p_batch_size INT,
    OUT p_records_synced INT,
    OUT p_records_failed INT
)
BEGIN
DECLARE v_endpoint_id INT;
    DECLARE v_event_type VARCHAR(100);
    DECLARE v_payload JSON;
    DECLARE v_entity_type VARCHAR(50);
    DECLARE v_entity_id INT;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        UPDATE webhook_events
        

    -- Simulate fetching data from external API
    -- In production, this would:
    -- 1. Call external API
    -- 2. Apply field mappings
    -- 3. Transform data
    -- 4. Insert/update local database
    -- 5. Update cache
    
    











SET p_records_synced = 0;
    SET p_records_failed = 0;
    
    -- Simplified simulation
    SET p_records_synced = p_batch_size;
END//

-- ========================================
-- PROCEDURE: Process Webhook Event
-- ========================================
CREATE PROCEDURE sp_process_webhook_event(
    IN p_event_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_sync_hash VARCHAR(64);
    
    -- Calculate hash of data
    

    INSERT INTO api_request_log (
        endpoint_id,
        request_method,
        request_url,
        request_body,
        response_status,
        response_body,
        response_time_ms
    ) VALUES (
        p_endpoint_id,
        p_method,
        p_url,
        p_request_body,
        p_response_status,
        p_response_body,
        p_response_time_ms
    );
    
    

    









SET status = 'failed',
            error_message = 'Error processing webhook event',
            retry_count = retry_count + 1
        WHERE event_id = p_event_id;
        
        SET p_status_message = 'Webhook processing failed';
    END;
    
    -- Get event details
    SELECT endpoint_id, event_type, payload
    INTO v_endpoint_id, v_event_type, v_payload
    FROM webhook_events
    WHERE event_id = p_event_id;
    
    -- Update status to processing
    UPDATE webhook_events
    SET status = 'processing'
    WHERE event_id = p_event_id;
    
    -- Process based on event type
    IF v_event_type = 'order.created' THEN
        -- Extract order data and create/update order
        SET v_entity_type = 'order';
        -- Would parse JSON and create order
    ELSEIF v_event_type = 'customer.updated' THEN
        SET v_entity_type = 'customer';
        -- Would parse JSON and update customer
    ELSEIF v_event_type = 'payment.succeeded' THEN
        -- Update payment status
        SET v_entity_type = 'payment';
    END IF;
    
    -- Mark as processed
    UPDATE webhook_events
    SET status = 'processed',
        processed_at = NOW()
    WHERE event_id = p_event_id;
    
    SET p_status_message = 'Webhook event processed successfully';
END//

-- ========================================
-- PROCEDURE: Log API Request
-- ========================================
CREATE PROCEDURE sp_log_api_request(
    IN p_endpoint_id INT,
    IN p_method VARCHAR(10),
    IN p_url VARCHAR(500),
    IN p_request_body JSON,
    IN p_response_status INT,
    IN p_response_body JSON,
    IN p_response_time_ms INT,
    OUT p_request_id INT
)
BEGIN
DECLARE v_customer_data JSON;
    DECLARE v_external_id VARCHAR(255);
    
    -- Get customer data
    SELECT JSON_OBJECT(
        'customer_id', customer_id,
        'first_name', first_name,
        'last_name', last_name,
        'email', email,
        'phone', phone,
        'status', status,
        'created_at', created_at
    ) INTO v_customer_data
    FROM customers
    WHERE customer_id = p_customer_id;
    
    -- Apply field mappings and transformations
    -- (In production, this would map fields using integration_mappings table)
    
    -- Simulate API call to CRM
    -- In production: Make actual API request
    

    -- Mark specific entities as dirty (need re-sync)
    UPDATE data_transformation_cache
    

    









SET p_request_id = LAST_INSERT_ID();
END//

-- ========================================
-- PROCEDURE: Update Integration Cache
-- ========================================
CREATE PROCEDURE sp_update_integration_cache(
    IN p_endpoint_id INT,
    IN p_entity_type VARCHAR(50),
    IN p_entity_id INT,
    IN p_external_id VARCHAR(255),
    IN p_cached_data JSON,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_order_data JSON;
    DECLARE v_items_data JSON;
    
    -- Get order with items
    SELECT JSON_OBJECT(
        'order_id', o.order_id,
        'customer_email', c.email,
        'order_date', o.order_date,
        'total_amount', o.total_amount,
        'status', o.status,
        'shipping_address', JSON_OBJECT(
            'address_line1', c.address_line1,
            'city', c.city,
            'state', c.state,
            'zip_code', c.zip_code
        )
    ) INTO v_order_data
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_id = p_order_id;
    
    -- Get order items
    SELECT JSON_ARRAYAGG(
        JSON_OBJECT(
            'product_id', oi.product_id,
            'sku', p.sku,
            'quantity', oi.quantity,
            'unit_price', oi.unit_price
        )
    ) INTO v_items_data
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    WHERE oi.order_id = p_order_id;
    
    -- Combine order and items
    

    


    









SET v_sync_hash = MD5(CAST(p_cached_data AS CHAR));
    
    INSERT INTO data_transformation_cache (
        endpoint_id,
        entity_type,
        entity_id,
        external_id,
        cached_data,
        sync_hash,
        last_synced_at,
        is_dirty
    ) VALUES (
        p_endpoint_id,
        p_entity_type,
        p_entity_id,
        p_external_id,
        p_cached_data,
        v_sync_hash,
        NOW(),
        FALSE
    ) ON DUPLICATE KEY UPDATE
        external_id = p_external_id,
        cached_data = p_cached_data,
        sync_hash = v_sync_hash,
        last_synced_at = NOW(),
        is_dirty = FALSE;
    
    SET p_status_message = 'Cache updated successfully';
END//

-- ========================================
-- PROCEDURE: Mark Entities for Sync
-- ========================================
CREATE PROCEDURE sp_mark_entities_for_sync(
    IN p_endpoint_id INT,
    IN p_entity_type VARCHAR(50),
    IN p_entity_ids JSON
)
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_job_id INT;
    
    DECLARE job_cursor CURSOR FOR
        SELECT job_id
        FROM sync_jobs
        WHERE is_active = TRUE
        AND next_run_at <= NOW()
        ORDER BY next_run_at
        LIMIT 10;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET is_dirty = TRUE
    WHERE endpoint_id = p_endpoint_id
    AND entity_type = p_entity_type
    AND JSON_CONTAINS(p_entity_ids, CAST(entity_id AS JSON));
END//

-- ========================================
-- PROCEDURE: Sync Customer to CRM
-- ========================================
CREATE PROCEDURE sp_sync_customer_to_crm(
    IN p_customer_id INT,
    IN p_crm_endpoint_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN

SET v_external_id = CONCAT('CRM_', p_customer_id);
    
    -- Update cache
    CALL sp_update_integration_cache(
        p_crm_endpoint_id,
        'customer',
        p_customer_id,
        v_external_id,
        v_customer_data,
        @cache_status
    );
    
    -- Log API request
    CALL sp_log_api_request(
        p_crm_endpoint_id,
        'POST',
        '/api/customers',
        v_customer_data,
        200,
        JSON_OBJECT('id', v_external_id, 'status', 'success'),
        250,
        @request_id
    );
    
    SET p_status_message = 'Customer synced to CRM successfully';
END//

-- ========================================
-- PROCEDURE: Sync Order to Fulfillment
-- ========================================
CREATE PROCEDURE sp_sync_order_to_fulfillment(
    IN p_order_id INT,
    IN p_fulfillment_endpoint_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_event_id INT;
    
    DECLARE webhook_cursor CURSOR FOR
        SELECT event_id
        FROM webhook_events
        WHERE status IN ('received', 'failed')
        AND (retry_count < 3 OR status = 'received')
        ORDER BY created_at
        LIMIT 50;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET v_order_data = JSON_SET(v_order_data, '$.items', v_items_data);
    
    -- Log API request (simulated)
    CALL sp_log_api_request(
        p_fulfillment_endpoint_id,
        'POST',
        '/api/orders',
        v_order_data,
        201,
        JSON_OBJECT('fulfillment_id', CONCAT('FUL_', p_order_id)),
        180,
        @request_id
    );
    
    SET p_status_message = 'Order synced to fulfillment system successfully';
END//

-- ========================================
-- PROCEDURE: Process Pending Sync Jobs
-- ========================================
CREATE PROCEDURE sp_process_pending_sync_jobs()
BEGIN

SET done = TRUE;
    
    OPEN job_cursor;
    
    job_loop: LOOP
        FETCH job_cursor INTO v_job_id;
        
        IF done THEN
            LEAVE job_loop;
        END IF;
        
        -- Execute sync job
        CALL sp_execute_sync_job(v_job_id, @log_id, @status);
        
    END LOOP;
    
    CLOSE job_cursor;
END//

-- ========================================
-- PROCEDURE: Process Pending Webhooks
-- ========================================
CREATE PROCEDURE sp_process_pending_webhooks()
BEGIN
DECLARE v_last_status VARCHAR(20);
    
    SELECT last_run_status INTO v_last_status
    FROM sync_jobs
    WHERE job_id = p_job_id;
    
    IF v_last_status = 'failed' THEN
        -- Execute the sync job again
        CALL sp_execute_sync_job(p_job_id, @log_id, @exec_status);
        

    SELECT 
        ie.endpoint_name,
        ie.provider,
        COUNT(DISTINCT sj.job_id) AS total_jobs,
        SUM(sel.records_synced) AS total_records_synced,
        SUM(sel.records_failed) AS total_records_failed,
        AVG(sel.duration_seconds) AS avg_sync_duration,
        COUNT(DISTINCT irl.request_id) AS total_api_calls,
        AVG(irl.response_time_ms) AS avg_response_time_ms,
        SUM(CASE WHEN irl.response_status >= 200 AND irl.response_status < 300 THEN 1 ELSE 0 END) AS successful_requests,
        SUM(CASE WHEN irl.response_status >= 400 THEN 1 ELSE 0 END) AS failed_requests
    FROM integration_endpoints ie
    LEFT JOIN sync_jobs sj ON ie.endpoint_id = sj.endpoint_id
    LEFT JOIN sync_execution_log sel ON sj.job_id = sel.job_id
        AND DATE(sel.started_at) BETWEEN p_start_date AND p_end_date
    LEFT JOIN api_request_log irl ON ie.endpoint_id = irl.endpoint_id
        AND DATE(irl.created_at) BETWEEN p_start_date AND p_end_date
    GROUP BY ie.endpoint_id, ie.endpoint_name, ie.provider
    ORDER BY total_records_synced DESC;
END//

-- ========================================
-- PROCEDURE: Handle Integration Error
-- ========================================
CREATE PROCEDURE sp_handle_integration_error(
    IN p_endpoint_id INT,
    IN p_error_type VARCHAR(50),
    IN p_error_code VARCHAR(50),
    IN p_error_message TEXT,
    IN p_context_data JSON,
    OUT p_error_id INT
)
BEGIN
    INSERT INTO integration_errors (
        endpoint_id,
        error_type,
        error_code,
        error_message,
        context_data,
        resolution_status
    ) VALUES (
        p_endpoint_id,
        p_error_type,
        p_error_code,
        p_error_message,
        p_context_data,
        'unresolved'
    );
    
    











SET done = TRUE;
    
    OPEN webhook_cursor;
    
    webhook_loop: LOOP
        FETCH webhook_cursor INTO v_event_id;
        
        IF done THEN
            LEAVE webhook_loop;
        END IF;
        
        -- Process webhook
        CALL sp_process_webhook_event(v_event_id, @status);
        
    END LOOP;
    
    CLOSE webhook_cursor;
END//

-- ========================================
-- PROCEDURE: Generate Integration Report
-- ========================================
CREATE PROCEDURE sp_generate_integration_report(
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN

SET p_error_id = LAST_INSERT_ID();
    
    -- Create incident for critical errors
    IF p_error_type IN ('authentication', 'connection') THEN
        CALL sp_create_incident(
            'system_error',
            'high',
            CONCAT('Integration error: ', p_error_message),
            'integration',
            p_endpoint_id,
            @incident_id,
            @status
        );
    END IF;
END//

-- ========================================
-- PROCEDURE: Retry Failed Sync
-- ========================================
CREATE PROCEDURE sp_retry_failed_sync(
    IN p_job_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_cutoff_date TIMESTAMP;
    
    

    


    









SET p_status_message = @exec_status;
    ELSE
        SET p_status_message = 'Job does not need retry';
    END IF;
END//

-- ========================================
-- PROCEDURE: Clean Up Old Integration Data
-- ========================================
CREATE PROCEDURE sp_cleanup_integration_data(
    IN p_days_to_keep INT
)
BEGIN

SET v_cutoff_date = DATE_SUB(NOW(), INTERVAL p_days_to_keep DAY);
    
    -- Clean up old API request logs
    DELETE FROM api_request_log
    WHERE created_at < v_cutoff_date;
    
    -- Clean up old sync execution logs
    DELETE FROM sync_execution_log
    WHERE created_at < v_cutoff_date
    AND execution_status = 'completed';
    
    -- Clean up processed webhook events
    DELETE FROM webhook_events
    WHERE created_at < v_cutoff_date
    AND status = 'processed';
    
    -- Clean up resolved errors
    DELETE FROM integration_errors
    WHERE created_at < v_cutoff_date
    AND resolution_status = 'resolved';
    
    SELECT CONCAT('Cleaned up integration data older than ', p_days_to_keep, ' days') AS Status;
END//

-- ========================================
-- FUNCTION: Get External ID
-- ========================================
CREATE FUNCTION fn_get_external_id(
    p_endpoint_id INT,
    p_entity_type VARCHAR(50),
    p_entity_id INT
)
RETURNS VARCHAR(255)
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_external_id VARCHAR(255);
    
    SELECT external_id INTO v_external_id
    FROM data_transformation_cache
    WHERE endpoint_id = p_endpoint_id
    AND entity_type = p_entity_type
    AND entity_id = p_entity_id;
    
    RETURN v_external_id;
END//

-- ========================================
-- FUNCTION: Check Rate Limit
-- ========================================
CREATE FUNCTION fn_check_rate_limit(
    p_endpoint_id INT
)
RETURNS BOOLEAN
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_rate_limit INT;
    DECLARE v_recent_requests INT;
    
    -- Get rate limit
    SELECT rate_limit_per_minute INTO v_rate_limit
    FROM integration_endpoints
    WHERE endpoint_id = p_endpoint_id;
    
    -- Count recent requests
    SELECT COUNT(*) INTO v_recent_requests
    FROM api_request_log
    WHERE endpoint_id = p_endpoint_id
    AND created_at >= DATE_SUB(NOW(), INTERVAL 1 MINUTE);
    
    RETURN (v_recent_requests < v_rate_limit);
END//

-- Reset delimiter
DELIMITER ;

-- ========================================
-- SEED INTEGRATION ENDPOINTS
-- ========================================
CALL sp_register_integration(
    'Stripe Payment Gateway',
    'stripe',
    'payment',
    'https://api.stripe.com/v1',
    'bearer',
    '{"api_key": "sk_test_xxxxx"}',
    @stripe_id,
    @stripe_msg
);

CALL sp_register_integration(
    'Shopify Storefront',
    'shopify',
    'inventory',
    'https://mystore.myshopify.com/admin/api/2024-01',
    'bearer',
    '{"access_token": "shpat_xxxxx"}',
    @shopify_id,
    @shopify_msg
);

CALL sp_register_integration(
    'Mailchimp Marketing',
    'mailchimp',
    'marketing',
    'https://us1.api.mailchimp.com/3.0',
    'api_key',
    '{"api_key": "xxxxx-us1"}',
    @mailchimp_id,
    @mailchimp_msg
);

CALL sp_register_integration(
    'ShipStation Fulfillment',
    'shipstation',
    'shipping',
    'https://ssapi.shipstation.com',
    'basic',
    '{"username": "xxxxx", "password": "xxxxx"}',
    @shipstation_id,
    @shipstation_msg
);

CALL sp_register_integration(
    'Salesforce CRM',
    'salesforce',
    'crm',
    'https://myinstance.salesforce.com/services/data/v58.0',
    'oauth2',
    '{"client_id": "xxxxx", "client_secret": "xxxxx"}',
    @salesforce_id,
    @salesforce_msg
);

-- ========================================
-- SEED FIELD MAPPINGS (Example: Customer to CRM)
-- ========================================
CALL sp_create_field_mapping(
    @salesforce_id,
    'customer',
    'first_name',
    'FirstName',
    'string',
    NULL,
    @map1_id,
    @map1_msg
);

CALL sp_create_field_mapping(
    @salesforce_id,
    'customer',
    'last_name',
    'LastName',
    'string',
    NULL,
    @map2_id,
    @map2_msg
);

CALL sp_create_field_mapping(
    @salesforce_id,
    'customer',
    'email',
    'Email',
    'string',
    NULL,
    @map3_id,
    @map3_msg
);

CALL sp_create_field_mapping(
    @salesforce_id,
    'customer',
    'phone',
    'Phone',
    'string',
    'REPLACE(phone, "-", "")', -- Transform: remove dashes
    @map4_id,
    @map4_msg
);

-- ========================================
-- SEED SYNC JOBS
-- ========================================
CALL sp_create_sync_job(
    @shopify_id,
    'Hourly Inventory Sync',
    'incremental',
    'bidirectional',
    'product',
    '{"interval": "hourly"}',
    @job1_id,
    @job1_msg
);

CALL sp_create_sync_job(
    @salesforce_id,
    'Daily Customer Sync',
    'incremental',
    'export',
    'customer',
    '{"interval": "daily", "time": "02:00"}',
    @job2_id,
    @job2_msg
);

CALL sp_create_sync_job(
    @mailchimp_id,
    'Weekly Marketing List Sync',
    'full',
    'export',
    'customer',
    '{"interval": "weekly", "day": "Monday"}',
    @job3_id,
    @job3_msg
);

CALL sp_create_sync_job(
    @shipstation_id,
    'Real-time Order Sync',
    'real_time',
    'export',
    'order',
    '{"trigger": "order_placed"}',
    @job4_id,
    @job4_msg
);

-- Display confirmation
SELECT 'Integration system created successfully' AS Status;

-- ========================================
-- USAGE EXAMPLES
-- ========================================
/*
-- Register a new integration endpoint
CALL sp_register_integration(
    'Google Analytics',
    'google',
    'analytics',
    'https://www.googleapis.com/analytics/v3',
    'oauth2',
    '{"client_id": "xxxxx", "client_secret": "xxxxx", "refresh_token": "xxxxx"}',
    @endpoint_id,
    @status
);

-- Create field mappings
CALL sp_create_field_mapping(
    1,
    'customer',
    'customer_id',
    'external_customer_id',
    'integer',
    NULL,
    @mapping_id,
    @status
);

-- Create a sync job
CALL sp_create_sync_job(
    1,
    'Nightly Customer Export',
    'incremental',
    'export',
    'customer',
    '{"schedule": "0 2 * * *"}',
    @job_id,
    @status
);

-- Execute a sync job manually
CALL sp_execute_sync_job(1, @log_id, @status);
SELECT @status;

-- Sync specific customer to CRM
CALL sp_sync_customer_to_crm(123, 5, @status);

-- Sync order to fulfillment system
CALL sp_sync_order_to_fulfillment(456, 4, @status);

-- Process pending sync jobs (run via scheduler)
CALL sp_process_pending_sync_jobs();

-- Process pending webhooks (run via scheduler)
CALL sp_process_pending_webhooks();

-- Process a specific webhook event
CALL sp_process_webhook_event(789, @status);

-- Update integration cache
CALL sp_update_integration_cache(
    1,
    'customer',
    123,
    'EXT_CUST_123',
    '{"name": "John Doe", "email": "john@example.com"}',
    @status
);

-- Mark entities for re-sync
CALL sp_mark_entities_for_sync(1, 'customer', '[123, 456, 789]');

-- Handle integration error
CALL sp_handle_integration_error(
    1,
    'rate_limit',
    'RATE_LIMIT_EXCEEDED',
    'API rate limit exceeded. Retry after 60 seconds.',
    '{"requests": 1000, "limit": 1000, "window": "1 hour"}',
    @error_id
);

-- Retry failed sync
CALL sp_retry_failed_sync(1, @status);

-- Generate integration report
CALL sp_generate_integration_report('2025-10-01', '2025-10-31');

-- Clean up old integration data (keep last 90 days)
CALL sp_cleanup_integration_data(90);

-- Get external ID for an entity
SELECT fn_get_external_id(1, 'customer', 123) AS external_id;

-- Check if rate limit allows more requests
SELECT fn_check_rate_limit(1) AS can_make_request;

-- View sync job status
SELECT 
    sj.job_name,
    ie.provider,
    sj.sync_type,
    sj.last_run_at,
    sj.last_run_status,
    sj.next_run_at,
    sj.records_processed,
    sj.records_failed
FROM sync_jobs sj
JOIN integration_endpoints ie ON sj.endpoint_id = ie.endpoint_id
WHERE sj.is_active = TRUE
ORDER BY sj.next_run_at;

-- View recent API requests
SELECT 
    ie.endpoint_name,
    arl.request_method,
    arl.request_url,
    arl.response_status,
    arl.response_time_ms,
    arl.created_at
FROM api_request_log arl
JOIN integration_endpoints ie ON arl.endpoint_id = ie.endpoint_id
WHERE arl.created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
ORDER BY arl.created_at DESC
LIMIT 50;

-- View pending webhook events
SELECT 
    event_id,
    event_type,
    event_source,
    status,
    retry_count,
    created_at
FROM webhook_events
WHERE status IN ('received', 'failed')
ORDER BY created_at;

-- View integration errors
SELECT 
    ie.endpoint_name,
    ier.error_type,
    ier.error_code,
    ier.error_message,
    ier.resolution_status,
    ier.created_at
FROM integration_errors ier
LEFT JOIN integration_endpoints ie ON ier.endpoint_id = ie.endpoint_id
WHERE ier.resolution_status = 'unresolved'
ORDER BY ier.created_at DESC;

-- View entities needing sync
SELECT 
    ie.endpoint_name,
    dtc.entity_type,
    dtc.entity_id,
    dtc.external_id,
    dtc.last_synced_at
FROM data_transformation_cache dtc
JOIN integration_endpoints ie ON dtc.endpoint_id = ie.endpoint_id
WHERE dtc.is_dirty = TRUE
ORDER BY dtc.last_synced_at;

-- Check sync performance metrics
SELECT 
    DATE(sel.started_at) AS sync_date,
    COUNT(DISTINCT sel.job_id) AS jobs_run,
    SUM(sel.records_synced) AS total_synced,
    SUM(sel.records_failed) AS total_failed,
    AVG(sel.duration_seconds) AS avg_duration,
    MAX(sel.duration_seconds) AS max_duration
FROM sync_execution_log sel
WHERE sel.started_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY DATE(sel.started_at)
ORDER BY sync_date DESC;
*/