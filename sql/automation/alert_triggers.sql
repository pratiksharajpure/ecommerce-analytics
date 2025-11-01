-- ========================================
-- ALERT TRIGGERS & THRESHOLD MONITORING
-- E-commerce Revenue Analytics Engine
-- Automated Alerting and Escalation System
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for trigger creation
DELIMITER //

-- ========================================
-- ALERT INFRASTRUCTURE TABLES
-- ========================================

-- Alert Configuration Table
CREATE TABLE IF NOT EXISTS alert_config (
    config_id INT PRIMARY KEY AUTO_INCREMENT,
    alert_type VARCHAR(100) NOT NULL,
    alert_name VARCHAR(200) NOT NULL,
    threshold_value DECIMAL(10,2),
    comparison_operator ENUM('GT', 'LT', 'EQ', 'GTE', 'LTE') DEFAULT 'GT',
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    is_active BOOLEAN DEFAULT TRUE,
    notification_email VARCHAR(200),
    escalation_minutes INT DEFAULT 60,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_alert_type (alert_type),
    INDEX idx_severity (severity),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Alert Log Table
CREATE TABLE IF NOT EXISTS alert_log (
    alert_id INT PRIMARY KEY AUTO_INCREMENT,
    alert_type VARCHAR(100) NOT NULL,
    alert_name VARCHAR(200) NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    alert_message TEXT,
    metric_value DECIMAL(10,2),
    threshold_value DECIMAL(10,2),
    affected_entity_type VARCHAR(50),
    affected_entity_id INT,
    status ENUM('new', 'acknowledged', 'resolved', 'escalated') DEFAULT 'new',
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP NULL,
    resolved_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_alert_type (alert_type),
    INDEX idx_severity (severity),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Alert Escalation Log
CREATE TABLE IF NOT EXISTS alert_escalation (
    escalation_id INT PRIMARY KEY AUTO_INCREMENT,
    alert_id INT NOT NULL,
    escalation_level INT DEFAULT 1,
    escalated_to VARCHAR(200),
    escalation_reason TEXT,
    escalated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (alert_id) REFERENCES alert_log(alert_id) ON DELETE CASCADE,
    INDEX idx_alert_id (alert_id),
    INDEX idx_escalation_level (escalation_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- STORED PROCEDURES FOR ALERT MANAGEMENT
-- ========================================

-- Procedure to create an alert
CREATE PROCEDURE sp_create_alert(
    IN p_alert_type VARCHAR(100),
    IN p_alert_name VARCHAR(200),
    IN p_severity VARCHAR(20),
    IN p_message TEXT,
    IN p_metric_value DECIMAL(10,2),
    IN p_threshold_value DECIMAL(10,2),
    IN p_entity_type VARCHAR(50),
    IN p_entity_id INT
)
BEGIN
DECLARE v_product_name VARCHAR(200);
    DECLARE v_alert_message TEXT;
    
    -- Check if inventory falls below reorder level
    IF NEW.quantity_available <= NEW.reorder_level AND NEW.quantity_available > 0 THEN
        -- Get product name
        SELECT product_name INTO v_product_name
        FROM products
        WHERE product_id = NEW.product_id;
        
        

    INSERT INTO alert_log (
        alert_type,
        alert_name,
        severity,
        alert_message,
        metric_value,
        threshold_value,
        affected_entity_type,
        affected_entity_id,
        status
    ) VALUES (
        p_alert_type,
        p_alert_name,
        p_severity,
        p_message,
        p_metric_value,
        p_threshold_value,
        p_entity_type,
        p_entity_id,
        'new'
    );
END//

-- Procedure to escalate alerts
CREATE PROCEDURE sp_escalate_alert(
    IN p_alert_id INT,
    IN p_escalation_level INT,
    IN p_escalated_to VARCHAR(200),
    IN p_reason TEXT
)
BEGIN
    -- Update alert status
    UPDATE alert_log
    











SET status = 'escalated'
    WHERE alert_id = p_alert_id;
    
    -- Log escalation
    INSERT INTO alert_escalation (
        alert_id,
        escalation_level,
        escalated_to,
        escalation_reason
    ) VALUES (
        p_alert_id,
        p_escalation_level,
        p_escalated_to,
        p_reason
    );
END//

-- ========================================
-- TRIGGER 1: Low Inventory Alert
-- ========================================
CREATE TRIGGER trg_alert_low_inventory
AFTER UPDATE ON inventory
FOR EACH ROW
BEGIN
    
SET v_alert_message = CONCAT(
            'Low inventory alert: Product "', v_product_name, 
            '" (ID: ', NEW.product_id, ') has ', NEW.quantity_available,
            ' units available. Reorder level: ', NEW.reorder_level
        );
        
        CALL sp_create_alert(
            'INVENTORY',
            'Low Stock Alert',
            'medium',
            v_alert_message,
            NEW.quantity_available,
            NEW.reorder_level,
            'product',
            NEW.product_id
        );
    END IF;
    
    -- Critical alert for out of stock
    IF NEW.quantity_available <= 0 THEN
        SELECT product_name INTO v_product_name
        FROM products
        WHERE product_id = NEW.product_id;
        
        SET v_alert_message = CONCAT(
            'CRITICAL: Product "', v_product_name, 
            '" (ID: ', NEW.product_id, ') is OUT OF STOCK!'
        );
        
        CALL sp_create_alert(
            'INVENTORY',
            'Out of Stock Critical',
            'critical',
            v_alert_message,
            0,
            0,
            'product',
            NEW.product_id
        );
    END IF;
END//

-- ========================================
-- TRIGGER 2: High Return Rate Alert
-- ========================================
CREATE TRIGGER trg_alert_high_return_rate
AFTER INSERT ON returns
FOR EACH ROW
BEGIN
    DECLARE v_product_name VARCHAR(200);
    DECLARE v_return_count INT;
    DECLARE v_order_count INT;
    DECLARE v_return_rate DECIMAL(5,2);
    DECLARE v_alert_message TEXT;
    
    -- Get product from order item
    SELECT p.product_name, oi.product_id
    INTO v_product_name, @product_id
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    WHERE oi.order_item_id = NEW.order_item_id;
    
    -- Calculate return rate for this product (last 30 days)
    SELECT 
        COUNT(DISTINCT r.return_id),
        COUNT(DISTINCT oi.order_id)
    INTO v_return_count, v_order_count
    FROM order_items oi
    LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
        AND r.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    WHERE oi.product_id = @product_id
        AND oi.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY);
    
    SET v_return_rate = (v_return_count / NULLIF(v_order_count, 0)) * 100;
    
    -- Alert if return rate exceeds 15%
    IF v_return_rate > 15 AND v_order_count >= 10 THEN
        SET v_alert_message = CONCAT(
            'High return rate detected: Product "', v_product_name,
            '" has a ', ROUND(v_return_rate, 2), '% return rate (',
            v_return_count, ' returns out of ', v_order_count, ' orders in last 30 days). ',
            'Latest return reason: ', NEW.reason
        );
        
        CALL sp_create_alert(
            'QUALITY',
            'High Return Rate',
            'high',
            v_alert_message,
            v_return_rate,
            15.00,
            'product',
            @product_id
        );
    END IF;
END//

-- ========================================
-- TRIGGER 3: Failed Payment Alert
-- ========================================
CREATE TRIGGER trg_alert_failed_payment
AFTER UPDATE ON orders
FOR EACH ROW
BEGIN
    DECLARE v_customer_name VARCHAR(200);
    DECLARE v_alert_message TEXT;
    
    -- Alert when payment status changes to failed
    IF OLD.payment_status != 'failed' AND NEW.payment_status = 'failed' THEN
        SELECT CONCAT(first_name, ' ', last_name)
        INTO v_customer_name
        FROM customers
        WHERE customer_id = NEW.customer_id;
        
        SET v_alert_message = CONCAT(
            'Payment failed for Order #', NEW.order_id,
            ' - Customer: ', v_customer_name,
            ' - Amount: $', NEW.total_amount
        );
        
        CALL sp_create_alert(
            'PAYMENT',
            'Payment Failed',
            'high',
            v_alert_message,
            NEW.total_amount,
            0,
            'order',
            NEW.order_id
        );
    END IF;
END//

-- ========================================
-- TRIGGER 4: Large Order Alert
-- ========================================
CREATE TRIGGER trg_alert_large_order
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    DECLARE v_customer_name VARCHAR(200);
    DECLARE v_alert_message TEXT;
    DECLARE v_threshold DECIMAL(10,2) DEFAULT 5000.00;
    
    -- Alert for orders over threshold
    IF NEW.total_amount >= v_threshold THEN
        SELECT CONCAT(first_name, ' ', last_name)
        INTO v_customer_name
        FROM customers
        WHERE customer_id = NEW.customer_id;
        
        SET v_alert_message = CONCAT(
            'Large order received: Order #', NEW.order_id,
            ' - Customer: ', v_customer_name,
            ' - Amount: $', NEW.total_amount,
            ' - Requires verification'
        );
        
        CALL sp_create_alert(
            'REVENUE',
            'Large Order Alert',
            'medium',
            v_alert_message,
            NEW.total_amount,
            v_threshold,
            'order',
            NEW.order_id
        );
    END IF;
END//

-- ========================================
-- TRIGGER 5: Suspicious Activity Alert
-- ========================================
CREATE TRIGGER trg_alert_suspicious_activity
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    DECLARE v_recent_orders INT;
    DECLARE v_customer_name VARCHAR(200);
    DECLARE v_alert_message TEXT;
    
    -- Count orders from this customer in last hour
    SELECT COUNT(*)
    INTO v_recent_orders
    FROM orders
    WHERE customer_id = NEW.customer_id
        AND order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR);
    
    -- Alert if more than 5 orders in an hour
    IF v_recent_orders > 5 THEN
        SELECT CONCAT(first_name, ' ', last_name)
        INTO v_customer_name
        FROM customers
        WHERE customer_id = NEW.customer_id;
        
        SET v_alert_message = CONCAT(
            'SUSPICIOUS: Customer "', v_customer_name,
            '" (ID: ', NEW.customer_id, ') has placed ',
            v_recent_orders, ' orders in the last hour. ',
            'Latest order #', NEW.order_id, ' - Amount: $', NEW.total_amount
        );
        
        CALL sp_create_alert(
            'FRAUD',
            'Suspicious Order Pattern',
            'critical',
            v_alert_message,
            v_recent_orders,
            5,
            'customer',
            NEW.customer_id
        );
    END IF;
END//

-- ========================================
-- TRIGGER 6: Campaign Budget Alert
-- ========================================
CREATE TRIGGER trg_alert_campaign_budget
AFTER INSERT ON campaign_performance
FOR EACH ROW
BEGIN
    DECLARE v_campaign_name VARCHAR(200);
    DECLARE v_total_spend DECIMAL(10,2);
    DECLARE v_campaign_budget DECIMAL(10,2);
    DECLARE v_budget_utilization DECIMAL(5,2);
    DECLARE v_alert_message TEXT;
    
    -- Get campaign details and calculate spend
    SELECT 
        c.campaign_name,
        c.budget,
        SUM(cp.spend)
    INTO v_campaign_name, v_campaign_budget, v_total_spend
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE c.campaign_id = NEW.campaign_id
    GROUP BY c.campaign_name, c.budget;
    
    SET v_budget_utilization = (v_total_spend / NULLIF(v_campaign_budget, 0)) * 100;
    
    -- Alert at 80% budget utilization
    IF v_budget_utilization >= 80 AND v_budget_utilization < 100 THEN
        SET v_alert_message = CONCAT(
            'Campaign "', v_campaign_name, '" has used ',
            ROUND(v_budget_utilization, 2), '% of budget ($',
            v_total_spend, ' of $', v_campaign_budget, ')'
        );
        
        CALL sp_create_alert(
            'MARKETING',
            'Campaign Budget Warning',
            'medium',
            v_alert_message,
            v_budget_utilization,
            80.00,
            'campaign',
            NEW.campaign_id
        );
    END IF;
    
    -- Critical alert at 100% budget
    IF v_budget_utilization >= 100 THEN
        SET v_alert_message = CONCAT(
            'CRITICAL: Campaign "', v_campaign_name, 
            '" has EXCEEDED budget! Spent: $', v_total_spend,
            ' - Budget: $', v_campaign_budget
        );
        
        CALL sp_create_alert(
            'MARKETING',
            'Campaign Budget Exceeded',
            'critical',
            v_alert_message,
            v_budget_utilization,
            100.00,
            'campaign',
            NEW.campaign_id
        );
    END IF;
END//

-- ========================================
-- TRIGGER 7: Vendor Performance Alert
-- ========================================
CREATE TRIGGER trg_alert_vendor_rating
AFTER UPDATE ON vendors
FOR EACH ROW
BEGIN
    DECLARE v_alert_message TEXT;
    
    -- Alert when vendor rating drops below 3.0
    IF NEW.rating < 3.0 AND OLD.rating >= 3.0 THEN
        SET v_alert_message = CONCAT(
            'Vendor "', NEW.vendor_name, '" rating has dropped to ',
            NEW.rating, '. Review vendor contracts and performance.'
        );
        
        CALL sp_create_alert(
            'VENDOR',
            'Low Vendor Rating',
            'high',
            v_alert_message,
            NEW.rating,
            3.00,
            'vendor',
            NEW.vendor_id
        );
    END IF;
    
    -- Critical alert if rating drops below 2.0
    IF NEW.rating < 2.0 THEN
        SET v_alert_message = CONCAT(
            'CRITICAL: Vendor "', NEW.vendor_name, 
            '" rating is critically low at ', NEW.rating,
            '. Immediate action required!'
        );
        
        CALL sp_create_alert(
            'VENDOR',
            'Critical Vendor Rating',
            'critical',
            v_alert_message,
            NEW.rating,
            2.00,
            'vendor',
            NEW.vendor_id
        );
    END IF;
END//

-- ========================================
-- TRIGGER 8: Customer Churn Risk Alert
-- ========================================
CREATE TRIGGER trg_alert_customer_churn
AFTER UPDATE ON customers
FOR EACH ROW
BEGIN
    DECLARE v_days_since_last_order INT;
    DECLARE v_total_orders INT;
    DECLARE v_alert_message TEXT;
    
    -- Check if customer becomes inactive
    IF OLD.status = 'active' AND NEW.status = 'inactive' THEN
        -- Calculate days since last order
        SELECT 
            DATEDIFF(NOW(), MAX(order_date)),
            COUNT(*)
        INTO v_days_since_last_order, v_total_orders
        FROM orders
        WHERE customer_id = NEW.customer_id
            AND payment_status = 'paid';
        
        -- Only alert for previously active customers
        IF v_total_orders >= 3 THEN
            SET v_alert_message = CONCAT(
                'Customer churn risk: ', NEW.first_name, ' ', NEW.last_name,
                ' (', v_total_orders, ' previous orders) has been inactive for ',
                v_days_since_last_order, ' days. Consider re-engagement campaign.'
            );
            
            CALL sp_create_alert(
                'CUSTOMER',
                'Customer Churn Risk',
                'medium',
                v_alert_message,
                v_days_since_last_order,
                90,
                'customer',
                NEW.customer_id
            );
        END IF;
    END IF;
END//

-- ========================================
-- TRIGGER 9: Negative Review Alert
-- ========================================
CREATE TRIGGER trg_alert_negative_review
AFTER INSERT ON reviews
FOR EACH ROW
BEGIN
    DECLARE v_product_name VARCHAR(200);
    DECLARE v_customer_name VARCHAR(200);
    DECLARE v_alert_message TEXT;
    
    -- Alert for 1 or 2 star reviews
    IF NEW.rating <= 2 THEN
        SELECT product_name INTO v_product_name
        FROM products WHERE product_id = NEW.product_id;
        
        SELECT CONCAT(first_name, ' ', last_name) INTO v_customer_name
        FROM customers WHERE customer_id = NEW.customer_id;
        
        SET v_alert_message = CONCAT(
            'Negative review alert: Product "', v_product_name,
            '" received ', NEW.rating, '-star review from ', v_customer_name,
            '. Title: "', COALESCE(NEW.review_title, 'No title'), '"'
        );
        
        CALL sp_create_alert(
            'QUALITY',
            'Negative Product Review',
            CASE WHEN NEW.rating = 1 THEN 'high' ELSE 'medium' END,
            v_alert_message,
            NEW.rating,
            2,
            'review',
            NEW.review_id
        );
    END IF;
END//

-- ========================================
-- TRIGGER 10: Loyalty Tier Change Alert
-- ========================================
CREATE TRIGGER trg_alert_loyalty_tier_change
AFTER UPDATE ON loyalty_program
FOR EACH ROW
BEGIN
    DECLARE v_customer_name VARCHAR(200);
    DECLARE v_alert_message TEXT;
    
    -- Alert when customer reaches higher tier
    IF NEW.tier != OLD.tier THEN
        SELECT CONCAT(first_name, ' ', last_name)
        INTO v_customer_name
        FROM customers
        WHERE customer_id = NEW.customer_id;
        
        SET v_alert_message = CONCAT(
            'Loyalty tier change: Customer "', v_customer_name,
            '" moved from ', OLD.tier, ' to ', NEW.tier,
            ' tier. Points balance: ', NEW.points_balance
        );
        
        CALL sp_create_alert(
            'LOYALTY',
            'Customer Tier Upgrade',
            'low',
            v_alert_message,
            0,
            0,
            'customer',
            NEW.customer_id
        );
    END IF;
END//

-- Reset delimiter
DELIMITER ;

-- ========================================
-- INSERT DEFAULT ALERT CONFIGURATIONS
-- ========================================
INSERT INTO alert_config (alert_type, alert_name, threshold_value, comparison_operator, severity, notification_email) VALUES
('INVENTORY', 'Low Stock Warning', 10, 'LTE', 'medium', 'inventory@company.com'),
('INVENTORY', 'Out of Stock Critical', 0, 'EQ', 'critical', 'inventory@company.com'),
('QUALITY', 'High Return Rate', 15, 'GT', 'high', 'quality@company.com'),
('PAYMENT', 'Payment Failed', 0, 'GT', 'high', 'finance@company.com'),
('REVENUE', 'Large Order Alert', 5000, 'GTE', 'medium', 'sales@company.com'),
('FRAUD', 'Suspicious Order Pattern', 5, 'GT', 'critical', 'security@company.com'),
('MARKETING', 'Campaign Budget Warning', 80, 'GTE', 'medium', 'marketing@company.com'),
('MARKETING', 'Campaign Budget Exceeded', 100, 'GTE', 'critical', 'marketing@company.com'),
('VENDOR', 'Low Vendor Rating', 3.0, 'LT', 'high', 'procurement@company.com'),
('VENDOR', 'Critical Vendor Rating', 2.0, 'LT', 'critical', 'procurement@company.com'),
('CUSTOMER', 'Customer Churn Risk', 90, 'GT', 'medium', 'retention@company.com'),
('QUALITY', 'Negative Product Review', 2, 'LTE', 'medium', 'quality@company.com'),
('LOYALTY', 'Customer Tier Upgrade', 0, 'GT', 'low', 'loyalty@company.com');

-- ========================================
-- VIEWS FOR ALERT MONITORING
-- ========================================

-- Active Alerts Dashboard
CREATE OR REPLACE VIEW v_active_alerts AS
SELECT 
    al.alert_id,
    al.alert_type,
    al.alert_name,
    al.severity,
    al.alert_message,
    al.metric_value,
    al.threshold_value,
    al.affected_entity_type,
    al.affected_entity_id,
    al.status,
    al.created_at,
    TIMESTAMPDIFF(MINUTE, al.created_at, NOW()) AS minutes_since_creation,
    CASE 
        WHEN TIMESTAMPDIFF(MINUTE, al.created_at, NOW()) > 60 THEN 'NEEDS_ESCALATION'
        ELSE 'NORMAL'
    END AS escalation_status
FROM alert_log al
WHERE al.status IN ('new', 'acknowledged')
ORDER BY 
    FIELD(al.severity, 'critical', 'high', 'medium', 'low'),
    al.created_at DESC;

-- Alert Summary by Type
CREATE OR REPLACE VIEW v_alert_summary AS
SELECT 
    alert_type,
    COUNT(*) AS total_alerts,
    SUM(CASE WHEN status = 'new' THEN 1 ELSE 0 END) AS new_alerts,
    SUM(CASE WHEN status = 'acknowledged' THEN 1 ELSE 0 END) AS acknowledged_alerts,
    SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) AS resolved_alerts,
    SUM(CASE WHEN status = 'escalated' THEN 1 ELSE 0 END) AS escalated_alerts,
    SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical_count,
    SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) AS high_count,
    MAX(created_at) AS last_alert_time
FROM alert_log
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY alert_type
ORDER BY critical_count DESC, high_count DESC;

-- Display confirmation
SELECT 'Alert triggers and monitoring system created successfully' AS Status;
SELECT COUNT(*) AS 'Alert Configurations Created' FROM alert_config;
SELECT 'Run SELECT * FROM v_active_alerts to view current alerts' AS Next_Steps;