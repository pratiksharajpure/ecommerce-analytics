-- ========================================
-- NOTIFICATION SYSTEM
-- E-commerce Revenue Analytics Engine
-- Notification Management, Email Triggers & Alert Distribution
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for procedure creation
DELIMITER //

-- ========================================
-- TABLE: Notification Templates
-- ========================================
CREATE TABLE IF NOT EXISTS notification_templates (
    template_id INT PRIMARY KEY AUTO_INCREMENT,
    template_name VARCHAR(100) UNIQUE NOT NULL,
    template_type ENUM('order', 'payment', 'shipping', 'marketing', 'system', 'alert') NOT NULL,
    channel ENUM('email', 'sms', 'push', 'slack', 'webhook') NOT NULL,
    subject_template VARCHAR(255),
    body_template TEXT NOT NULL,
    variables JSON, -- List of variables that can be used: ["customer_name", "order_id"]
    priority ENUM('low', 'medium', 'high', 'urgent') DEFAULT 'medium',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_template_name (template_name),
    INDEX idx_template_type (template_type),
    INDEX idx_channel (channel)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Notification Queue
-- ========================================
CREATE TABLE IF NOT EXISTS notification_queue (
    queue_id INT PRIMARY KEY AUTO_INCREMENT,
    recipient VARCHAR(255) NOT NULL,
    channel ENUM('email', 'sms', 'push', 'slack', 'webhook') NOT NULL,
    template_id INT,
    subject VARCHAR(255),
    message TEXT NOT NULL,
    variables JSON,
    priority ENUM('low', 'medium', 'high', 'urgent') DEFAULT 'medium',
    status ENUM('pending', 'processing', 'sent', 'failed', 'retry') DEFAULT 'pending',
    scheduled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sent_at TIMESTAMP NULL,
    failed_at TIMESTAMP NULL,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    error_message TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (template_id) REFERENCES notification_templates(template_id) ON DELETE SET NULL,
    INDEX idx_recipient (recipient),
    INDEX idx_status (status),
    INDEX idx_channel (channel),
    INDEX idx_scheduled_at (scheduled_at),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Notification Preferences
-- ========================================
CREATE TABLE IF NOT EXISTS notification_preferences (
    preference_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT NOT NULL,
    notification_type ENUM('order_updates', 'marketing', 'promotional', 'system_alerts', 
                           'review_requests', 'loyalty_updates', 'shipping_updates') NOT NULL,
    channel ENUM('email', 'sms', 'push') NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    frequency ENUM('immediate', 'daily_digest', 'weekly_digest') DEFAULT 'immediate',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    UNIQUE KEY unique_preference (customer_id, notification_type, channel),
    INDEX idx_customer_id (customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Notification Triggers
-- ========================================
CREATE TABLE IF NOT EXISTS notification_triggers (
    trigger_id INT PRIMARY KEY AUTO_INCREMENT,
    trigger_name VARCHAR(100) UNIQUE NOT NULL,
    trigger_event ENUM('order_placed', 'order_shipped', 'order_delivered', 'payment_failed',
                       'payment_success', 'return_requested', 'review_reminder', 
                       'abandoned_cart', 'low_stock', 'price_drop', 'back_in_stock',
                       'loyalty_milestone', 'account_created', 'password_reset') NOT NULL,
    template_id INT NOT NULL,
    delay_minutes INT DEFAULT 0, -- Delay before sending
    conditions JSON, -- Additional conditions: {"min_order_value": 100}
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (template_id) REFERENCES notification_templates(template_id) ON DELETE CASCADE,
    INDEX idx_trigger_event (trigger_event),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Alert Subscriptions
-- ========================================
CREATE TABLE IF NOT EXISTS alert_subscriptions (
    subscription_id INT PRIMARY KEY AUTO_INCREMENT,
    subscriber_email VARCHAR(255) NOT NULL,
    subscriber_role ENUM('admin', 'manager', 'analyst', 'developer', 'support') NOT NULL,
    alert_category ENUM('sales', 'inventory', 'fraud', 'performance', 'errors', 'all') NOT NULL,
    min_severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    channel ENUM('email', 'sms', 'slack', 'webhook') NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_subscriber_email (subscriber_email),
    INDEX idx_alert_category (alert_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- TABLE: Notification Analytics
-- ========================================
CREATE TABLE IF NOT EXISTS notification_analytics (
    analytics_id INT PRIMARY KEY AUTO_INCREMENT,
    template_id INT,
    channel ENUM('email', 'sms', 'push', 'slack', 'webhook'),
    sent_count INT DEFAULT 0,
    delivered_count INT DEFAULT 0,
    opened_count INT DEFAULT 0,
    clicked_count INT DEFAULT 0,
    failed_count INT DEFAULT 0,
    unsubscribed_count INT DEFAULT 0,
    report_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (template_id) REFERENCES notification_templates(template_id) ON DELETE CASCADE,
    UNIQUE KEY unique_daily_stats (template_id, channel, report_date),
    INDEX idx_report_date (report_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- PROCEDURE: Queue Notification
-- ========================================
CREATE PROCEDURE sp_queue_notification(
    IN p_recipient VARCHAR(255),
    IN p_channel VARCHAR(20),
    IN p_template_name VARCHAR(100),
    IN p_variables JSON,
    IN p_priority VARCHAR(20),
    IN p_scheduled_at TIMESTAMP,
    OUT p_queue_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
DECLARE v_template_id INT;
    DECLARE v_subject_template VARCHAR(255);
    DECLARE v_body_template TEXT;
    DECLARE v_final_subject VARCHAR(255);
    DECLARE v_final_body TEXT;
    
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        

    












SET p_status_message = 'Error queueing notification';
        SET p_queue_id = NULL;
    END;
    
    START TRANSACTION;
    
    -- Get template details
    SELECT template_id, subject_template, body_template
    INTO v_template_id, v_subject_template, v_body_template
    FROM notification_templates
    WHERE template_name = p_template_name
    AND channel = p_channel
    AND is_active = TRUE;
    
    IF v_template_id IS NULL THEN
        SET p_status_message = 'Template not found or inactive';
        ROLLBACK;
    ELSE
        -- In production, replace variables in templates here
        SET v_final_subject = v_subject_template;
        SET v_final_body = v_body_template;
        
        -- Insert into queue
        INSERT INTO notification_queue (
            recipient,
            channel,
            template_id,
            subject,
            message,
            variables,
            priority,
            scheduled_at,
            status
        ) VALUES (
            p_recipient,
            p_channel,
            v_template_id,
            v_final_subject,
            v_final_body,
            p_variables,
            IFNULL(p_priority, 'medium'),
            IFNULL(p_scheduled_at, NOW()),
            'pending'
        );
        
        SET p_queue_id = LAST_INSERT_ID();
        SET p_status_message = 'Notification queued successfully';
        
        COMMIT;
    END IF;
END//

-- ========================================
-- PROCEDURE: Process Notification Queue
-- ========================================
CREATE PROCEDURE sp_process_notification_queue(
    IN p_batch_size INT
)
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_queue_id INT;
    DECLARE v_recipient VARCHAR(255);
    DECLARE v_channel VARCHAR(20);
    DECLARE v_message TEXT;
    DECLARE v_retry_count INT;
    
    DECLARE queue_cursor CURSOR FOR
        SELECT queue_id, recipient, channel, message, retry_count
        FROM notification_queue
        WHERE status IN ('pending', 'retry')
        AND scheduled_at <= NOW()
        ORDER BY priority DESC, scheduled_at ASC
        LIMIT p_batch_size;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    OPEN queue_cursor;
    
    process_loop: LOOP
        FETCH queue_cursor INTO v_queue_id, v_recipient, v_channel, v_message, v_retry_count;
        
        IF done THEN
            LEAVE process_loop;
        END IF;
        
        -- Update status to processing
        UPDATE notification_queue
        SET status = 'processing'
        WHERE queue_id = v_queue_id;
        
        -- Simulate sending (in production, call external API)
        -- For this demo, we'll mark as sent
        UPDATE notification_queue
        SET status = 'sent',
            sent_at = NOW()
        WHERE queue_id = v_queue_id;
        
        -- Update analytics
        CALL sp_update_notification_analytics(v_queue_id, 'sent');
        
    END LOOP;
    
    CLOSE queue_cursor;
END//

-- ========================================
-- PROCEDURE: Send Order Notification
-- ========================================
CREATE PROCEDURE sp_send_order_notification(
    IN p_order_id INT,
    IN p_notification_type VARCHAR(50)
)
BEGIN
DECLARE v_customer_id INT;
    DECLARE v_customer_email VARCHAR(255);
    DECLARE v_customer_name VARCHAR(100);
    DECLARE v_order_total DECIMAL(10,2);
    DECLARE v_template_name VARCHAR(100);
    DECLARE v_variables JSON;
    
    -- Get order and customer details
    SELECT 
        o.customer_id,
        c.email,
        CONCAT(c.first_name, ' ', c.last_name),
        o.total_amount
    INTO v_customer_id, v_customer_email, v_customer_name, v_order_total
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_id = p_order_id;
    
    -- Determine template based on notification type
    

    












SET v_template_name = CASE p_notification_type
        WHEN 'order_placed' THEN 'order_confirmation'
        WHEN 'order_shipped' THEN 'shipping_notification'
        WHEN 'order_delivered' THEN 'delivery_confirmation'
        ELSE 'order_update'
    END;
    
    -- Build variables JSON
    SET v_variables = JSON_OBJECT(
        'customer_name', v_customer_name,
        'order_id', p_order_id,
        'order_total', v_order_total
    );
    
    -- Check customer preferences
    IF EXISTS (
        SELECT 1 FROM notification_preferences
        WHERE customer_id = v_customer_id
        AND notification_type = 'order_updates'
        AND channel = 'email'
        AND is_enabled = TRUE
    ) OR NOT EXISTS (
        SELECT 1 FROM notification_preferences
        WHERE customer_id = v_customer_id
    ) THEN
        -- Queue notification
        CALL sp_queue_notification(
            v_customer_email,
            'email',
            v_template_name,
            v_variables,
            'high',
            NOW(),
            @queue_id,
            @status
        );
    END IF;
END//

-- ========================================
-- PROCEDURE: Send Alert to Subscribers
-- ========================================
CREATE PROCEDURE sp_send_alert_to_subscribers(
    IN p_alert_category VARCHAR(50),
    IN p_severity VARCHAR(20),
    IN p_subject VARCHAR(255),
    IN p_message TEXT
)
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_subscriber_email VARCHAR(255);
    DECLARE v_channel VARCHAR(20);
    
    DECLARE subscriber_cursor CURSOR FOR
        SELECT subscriber_email, channel
        FROM alert_subscriptions
        WHERE (alert_category = p_alert_category OR alert_category = 'all')
        AND (
            (min_severity = 'low') OR
            (min_severity = 'medium' AND p_severity IN ('medium', 'high', 'critical')) OR
            (min_severity = 'high' AND p_severity IN ('high', 'critical')) OR
            (min_severity = 'critical' AND p_severity = 'critical')
        )
        AND is_active = TRUE;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    OPEN subscriber_cursor;
    
    subscriber_loop: LOOP
        FETCH subscriber_cursor INTO v_subscriber_email, v_channel;
        
        IF done THEN
            LEAVE subscriber_loop;
        END IF;
        
        -- Queue alert
        INSERT INTO notification_queue (
            recipient,
            channel,
            subject,
            message,
            priority,
            status
        ) VALUES (
            v_subscriber_email,
            v_channel,
            CONCAT('[', UPPER(p_severity), '] ', p_subject),
            p_message,
            CASE p_severity
                WHEN 'critical' THEN 'urgent'
                WHEN 'high' THEN 'high'
                ELSE 'medium'
            END,
            'pending'
        );
        
    END LOOP;
    
    CLOSE subscriber_cursor;
END//

-- ========================================
-- PROCEDURE: Send Abandoned Cart Reminder
-- ========================================
CREATE PROCEDURE sp_send_abandoned_cart_reminders()
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_customer_id INT;
    DECLARE v_customer_email VARCHAR(255);
    DECLARE v_customer_name VARCHAR(100);
    DECLARE v_cart_value DECIMAL(10,2);
    
    -- Find customers with items added over 24 hours ago but no order
    DECLARE cart_cursor CURSOR FOR
        SELECT DISTINCT
            c.customer_id,
            c.email,
            CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
            0.00 AS cart_value -- Simplified; would calculate from cart table
        FROM customers c
        WHERE c.customer_id NOT IN (
            SELECT customer_id 
            FROM orders 
            WHERE order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        )
        AND c.status = 'active'
        LIMIT 100; -- Process in batches
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    OPEN cart_cursor;
    
    cart_loop: LOOP
        FETCH cart_cursor INTO v_customer_id, v_customer_email, v_customer_name, v_cart_value;
        
        IF done THEN
            LEAVE cart_loop;
        END IF;
        
        -- Queue abandoned cart email
        CALL sp_queue_notification(
            v_customer_email,
            'email',
            'abandoned_cart_reminder',
            JSON_OBJECT(
                'customer_name', v_customer_name,
                'cart_value', v_cart_value
            ),
            'medium',
            NOW(),
            @queue_id,
            @status
        );
        
    END LOOP;
    
    CLOSE cart_cursor;
END//

-- ========================================
-- PROCEDURE: Update Notification Analytics
-- ========================================
CREATE PROCEDURE sp_update_notification_analytics(
    IN p_queue_id INT,
    IN p_event_type VARCHAR(50)
)
BEGIN
    DECLARE v_template_id INT;
    DECLARE v_channel VARCHAR(20);
    DECLARE v_report_date DATE;
    
    SELECT template_id, channel, CURDATE()
    INTO v_template_id, v_channel, v_report_date
    FROM notification_queue
    WHERE queue_id = p_queue_id;
    
    IF v_template_id IS NOT NULL THEN
        INSERT INTO notification_analytics (
            template_id, channel, report_date,
            sent_count, delivered_count, opened_count, 
            clicked_count, failed_count
        ) VALUES (
            v_template_id, v_channel, v_report_date,
            CASE WHEN p_event_type = 'sent' THEN 1 ELSE 0 END,
            CASE WHEN p_event_type = 'delivered' THEN 1 ELSE 0 END,
            CASE WHEN p_event_type = 'opened' THEN 1 ELSE 0 END,
            CASE WHEN p_event_type = 'clicked' THEN 1 ELSE 0 END,
            CASE WHEN p_event_type = 'failed' THEN 1 ELSE 0 END
        ) ON DUPLICATE KEY UPDATE
            sent_count = sent_count + CASE WHEN p_event_type = 'sent' THEN 1 ELSE 0 END,
            delivered_count = delivered_count + CASE WHEN p_event_type = 'delivered' THEN 1 ELSE 0 END,
            opened_count = opened_count + CASE WHEN p_event_type = 'opened' THEN 1 ELSE 0 END,
            clicked_count = clicked_count + CASE WHEN p_event_type = 'clicked' THEN 1 ELSE 0 END,
            failed_count = failed_count + CASE WHEN p_event_type = 'failed' THEN 1 ELSE 0 END;
    END IF;
END//

-- ========================================
-- PROCEDURE: Generate Notification Report
-- ========================================
CREATE PROCEDURE sp_generate_notification_report(
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    SELECT 
        nt.template_name,
        na.channel,
        SUM(na.sent_count) AS total_sent,
        SUM(na.delivered_count) AS total_delivered,
        SUM(na.opened_count) AS total_opened,
        SUM(na.clicked_count) AS total_clicked,
        SUM(na.failed_count) AS total_failed,
        ROUND(SUM(na.delivered_count) * 100.0 / NULLIF(SUM(na.sent_count), 0), 2) AS delivery_rate,
        ROUND(SUM(na.opened_count) * 100.0 / NULLIF(SUM(na.delivered_count), 0), 2) AS open_rate,
        ROUND(SUM(na.clicked_count) * 100.0 / NULLIF(SUM(na.opened_count), 0), 2) AS click_rate
    FROM notification_analytics na
    JOIN notification_templates nt ON na.template_id = nt.template_id
    WHERE na.report_date BETWEEN p_start_date AND p_end_date
    GROUP BY nt.template_name, na.channel
    ORDER BY total_sent DESC;
END//

-- ========================================
-- FUNCTION: Check Notification Preference
-- ========================================
CREATE FUNCTION fn_check_notification_preference(
    p_customer_id INT,
    p_notification_type VARCHAR(50),
    p_channel VARCHAR(20)
)
RETURNS BOOLEAN
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_is_enabled BOOLEAN;
    
    SELECT is_enabled INTO v_is_enabled
    FROM notification_preferences
    WHERE customer_id = p_customer_id
    AND notification_type = p_notification_type
    AND channel = p_channel;
    
    -- If no preference set, default to enabled
    RETURN IFNULL(v_is_enabled, TRUE);
END//

-- Reset delimiter
DELIMITER ;

-- ========================================
-- SEED NOTIFICATION TEMPLATES
-- ========================================
INSERT INTO notification_templates (template_name, template_type, channel, subject_template, body_template, variables, priority) VALUES
-- Order notifications
('order_confirmation', 'order', 'email', 
 'Order Confirmation - Order #{{order_id}}',
 'Hi {{customer_name}},\n\nThank you for your order! Your order #{{order_id}} totaling ${{order_total}} has been confirmed.\n\nWe''ll notify you when it ships.',
 '["customer_name", "order_id", "order_total"]', 'high'),

('shipping_notification', 'shipping', 'email',
 'Your Order Has Shipped - Order #{{order_id}}',
 'Hi {{customer_name}},\n\nGreat news! Your order #{{order_id}} has shipped.\n\nTracking: {{tracking_number}}',
 '["customer_name", "order_id", "tracking_number"]', 'high'),

('delivery_confirmation', 'shipping', 'email',
 'Your Order Has Been Delivered - Order #{{order_id}}',
 'Hi {{customer_name}},\n\nYour order #{{order_id}} has been delivered. We hope you love it!\n\nPlease consider leaving a review.',
 '["customer_name", "order_id"]', 'medium'),

-- Payment notifications
('payment_failed', 'payment', 'email',
 'Payment Failed - Action Required',
 'Hi {{customer_name}},\n\nWe couldn''t process your payment for order #{{order_id}}. Please update your payment method.',
 '["customer_name", "order_id"]', 'urgent'),

('payment_success', 'payment', 'email',
 'Payment Received - Order #{{order_id}}',
 'Hi {{customer_name}},\n\nWe''ve received your payment of ${{amount}} for order #{{order_id}}.',
 '["customer_name", "order_id", "amount"]', 'high'),

-- Marketing notifications
('abandoned_cart_reminder', 'marketing', 'email',
 'You Left Something Behind!',
 'Hi {{customer_name}},\n\nWe noticed you left items in your cart worth ${{cart_value}}. Complete your purchase now!',
 '["customer_name", "cart_value"]', 'medium'),

('price_drop_alert', 'marketing', 'email',
 'Price Drop Alert - {{product_name}}',
 'Hi {{customer_name}},\n\nGood news! {{product_name}} is now on sale. Original: ${{old_price}}, Now: ${{new_price}}',
 '["customer_name", "product_name", "old_price", "new_price"]', 'medium'),

-- System alerts
('low_stock_alert', 'alert', 'slack',
 'Low Stock Alert',
 'Product {{product_name}} (SKU: {{sku}}) has only {{quantity}} units remaining.',
 '["product_name", "sku", "quantity"]', 'high'),

('fraud_alert', 'alert', 'slack',
 'Fraud Detection Alert',
 'Suspicious activity detected on order #{{order_id}}. Customer: {{customer_email}}',
 '["order_id", "customer_email"]', 'urgent');

-- ========================================
-- SEED NOTIFICATION TRIGGERS
-- ========================================
INSERT INTO notification_triggers (trigger_name, trigger_event, template_id, delay_minutes, is_active) VALUES
('order_confirmation_trigger', 'order_placed', 1, 0, TRUE),
('shipping_notification_trigger', 'order_shipped', 2, 0, TRUE),
('delivery_confirmation_trigger', 'order_delivered', 3, 0, TRUE),
('payment_failed_trigger', 'payment_failed', 4, 0, TRUE),
('abandoned_cart_trigger', 'abandoned_cart', 6, 1440, TRUE), -- 24 hours
('low_stock_trigger', 'low_stock', 8, 0, TRUE);

-- ========================================
-- SEED ALERT SUBSCRIPTIONS
-- ========================================
INSERT INTO alert_subscriptions (subscriber_email, subscriber_role, alert_category, min_severity, channel) VALUES
('ops@company.com', 'manager', 'inventory', 'high', 'email'),
('fraud@company.com', 'analyst', 'fraud', 'medium', 'slack'),
('dev@company.com', 'developer', 'errors', 'high', 'slack'),
('cto@company.com', 'admin', 'all', 'critical', 'email'),
('sales@company.com', 'manager', 'sales', 'medium', 'email');

-- Display confirmation
SELECT 'Notification system created successfully' AS Status;

-- ========================================
-- USAGE EXAMPLES
-- ========================================
/*
-- Queue a notification
CALL sp_queue_notification(
    'customer@email.com',
    'email',
    'order_confirmation',
    '{"customer_name": "John Doe", "order_id": 12345, "order_total": 299.99}',
    'high',
    NOW(),
    @queue_id,
    @status
);

-- Process notification queue (run periodically)
CALL sp_process_notification_queue(50);

-- Send order notification
CALL sp_send_order_notification(12345, 'order_placed');

-- Send alert to subscribers
CALL sp_send_alert_to_subscribers(
    'inventory',
    'high',
    'Critical Low Stock',
    'Multiple products below reorder level'
);

-- Send abandoned cart reminders
CALL sp_send_abandoned_cart_reminders();

-- Generate notification report
CALL sp_generate_notification_report('2025-10-01', '2025-10-31');

-- Check customer preference
SELECT fn_check_notification_preference(123, 'order_updates', 'email');
*/