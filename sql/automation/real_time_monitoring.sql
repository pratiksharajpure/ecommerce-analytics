-- ========================================
-- REAL-TIME MONITORING SYSTEM
-- E-commerce Revenue Analytics Engine
-- Change Detection & Live Dashboards
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- SECTION 1: CREATE MONITORING INFRASTRUCTURE
-- ========================================
SELECT '========== CREATING REAL-TIME MONITORING INFRASTRUCTURE ==========' AS '';

-- Table to track real-time metrics snapshots
CREATE TABLE IF NOT EXISTS rt_metrics_snapshot (
    snapshot_id INT PRIMARY KEY AUTO_INCREMENT,
    snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_type ENUM('orders', 'revenue', 'inventory', 'customers', 'performance') NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,2),
    metric_count INT,
    metric_text VARCHAR(255),
    INDEX idx_snapshot_time (snapshot_time),
    INDEX idx_metric_type (metric_type),
    INDEX idx_metric_name (metric_name)
) ENGINE=InnoDB;

-- Table to track data changes/deltas
CREATE TABLE IF NOT EXISTS rt_change_log (
    change_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(100) NOT NULL,
    record_id VARCHAR(50),
    change_type ENUM('INSERT', 'UPDATE', 'DELETE') NOT NULL,
    column_name VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_table_name (table_name),
    INDEX idx_changed_at (changed_at),
    INDEX idx_change_type (change_type)
) ENGINE=InnoDB;

-- Table for alert definitions
CREATE TABLE IF NOT EXISTS rt_alert_rules (
    alert_id INT PRIMARY KEY AUTO_INCREMENT,
    alert_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    condition_type ENUM('threshold_above', 'threshold_below', 'percentage_change', 'sudden_spike', 'sudden_drop') NOT NULL,
    threshold_value DECIMAL(15,2),
    percentage_threshold DECIMAL(5,2),
    time_window_minutes INT DEFAULT 5,
    severity ENUM('critical', 'high', 'medium', 'low') DEFAULT 'medium',
    is_active BOOLEAN DEFAULT TRUE,
    alert_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_name (metric_name),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB;

-- Table to store triggered alerts
CREATE TABLE IF NOT EXISTS rt_alert_history (
    history_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    alert_id INT NOT NULL,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_value DECIMAL(15,2),
    threshold_value DECIMAL(15,2),
    alert_message TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP NULL,
    FOREIGN KEY (alert_id) REFERENCES rt_alert_rules(alert_id) ON DELETE CASCADE,
    INDEX idx_triggered_at (triggered_at),
    INDEX idx_acknowledged (acknowledged)
) ENGINE=InnoDB;

-- Table for real-time dashboard KPIs
CREATE TABLE IF NOT EXISTS rt_dashboard_kpis (
    kpi_id INT PRIMARY KEY AUTO_INCREMENT,
    kpi_name VARCHAR(100) NOT NULL UNIQUE,
    kpi_category ENUM('sales', 'operations', 'customer', 'inventory', 'marketing') NOT NULL,
    current_value DECIMAL(15,2),
    previous_value DECIMAL(15,2),
    change_amount DECIMAL(15,2),
    change_percentage DECIMAL(8,2),
    trend ENUM('up', 'down', 'stable') DEFAULT 'stable',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    update_frequency_seconds INT DEFAULT 60,
    INDEX idx_kpi_category (kpi_category),
    INDEX idx_last_updated (last_updated)
) ENGINE=InnoDB;

-- Table for activity stream (real-time events)
CREATE TABLE IF NOT EXISTS rt_activity_stream (
    activity_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    activity_type ENUM('order_placed', 'order_shipped', 'payment_received', 'customer_registered', 'product_low_stock', 'review_posted', 'return_requested') NOT NULL,
    activity_title VARCHAR(200),
    activity_details TEXT,
    related_table VARCHAR(100),
    related_id INT,
    severity ENUM('info', 'warning', 'critical') DEFAULT 'info',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_activity_type (activity_type),
    INDEX idx_created_at (created_at),
    INDEX idx_severity (severity)
) ENGINE=InnoDB;

SELECT 'Real-time monitoring infrastructure created' AS Status;

-- ========================================
-- SECTION 2: DEFINE ALERT RULES
-- ========================================
SELECT '========== DEFINING ALERT RULES ==========' AS '';

-- Clear existing alert rules
DELETE FROM rt_alert_rules;

-- Alert: High order volume spike
INSERT INTO rt_alert_rules (alert_name, metric_name, condition_type, percentage_threshold, time_window_minutes, severity, alert_message)
VALUES (
    'High Order Volume Spike',
    'orders_per_minute',
    'percentage_change',
    50.00,
    5,
    'high',
    'Order volume increased by more than 50% in the last 5 minutes'
);

-- Alert: Revenue threshold
INSERT INTO rt_alert_rules (alert_name, metric_name, condition_type, threshold_value, time_window_minutes, severity, alert_message)
VALUES (
    'Hourly Revenue Below Target',
    'revenue_per_hour',
    'threshold_below',
    1000.00,
    60,
    'medium',
    'Hourly revenue is below $1000 target'
);

-- Alert: Critical inventory level
INSERT INTO rt_alert_rules (alert_name, metric_name, condition_type, threshold_value, time_window_minutes, severity, alert_message)
VALUES (
    'Critical Inventory Level',
    'products_out_of_stock',
    'threshold_above',
    5.00,
    15,
    'critical',
    'More than 5 products are out of stock'
);

-- Alert: Payment failure rate
INSERT INTO rt_alert_rules (alert_name, metric_name, condition_type, threshold_value, time_window_minutes, severity, alert_message)
VALUES (
    'High Payment Failure Rate',
    'payment_failure_rate',
    'threshold_above',
    10.00,
    30,
    'high',
    'Payment failure rate exceeds 10%'
);

-- Alert: Sudden drop in traffic
INSERT INTO rt_alert_rules (alert_name, metric_name, condition_type, percentage_threshold, time_window_minutes, severity, alert_message)
VALUES (
    'Traffic Drop Alert',
    'new_customers_per_hour',
    'sudden_drop',
    30.00,
    60,
    'medium',
    'New customer registrations dropped by 30%'
);

SELECT 'Alert rules configured' AS Status;

-- ========================================
-- SECTION 3: INITIALIZE DASHBOARD KPIs
-- ========================================
SELECT '========== INITIALIZING DASHBOARD KPIs ==========' AS '';

-- Initialize or update KPIs
INSERT INTO rt_dashboard_kpis (kpi_name, kpi_category, current_value, previous_value, update_frequency_seconds)
VALUES 
    ('total_orders_today', 'sales', 0, 0, 60),
    ('total_revenue_today', 'sales', 0, 0, 60),
    ('average_order_value', 'sales', 0, 0, 300),
    ('orders_last_hour', 'operations', 0, 0, 60),
    ('pending_orders', 'operations', 0, 0, 120),
    ('active_customers_today', 'customer', 0, 0, 300),
    ('new_customers_today', 'customer', 0, 0, 300),
    ('products_low_stock', 'inventory', 0, 0, 300),
    ('products_out_of_stock', 'inventory', 0, 0, 300),
    ('campaign_conversions_today', 'marketing', 0, 0, 600)
ON DUPLICATE KEY UPDATE 
    kpi_name = VALUES(kpi_name);

SELECT 'Dashboard KPIs initialized' AS Status;

-- ========================================
-- SECTION 4: CAPTURE CURRENT METRICS SNAPSHOT
-- ========================================
SELECT '========== CAPTURING METRICS SNAPSHOT ==========' AS '';

-- Insert current snapshot of key metrics
INSERT INTO rt_metrics_snapshot (metric_type, metric_name, metric_value, metric_count)
VALUES
    -- Order metrics
    ('orders', 'total_orders_today', 
     (SELECT COUNT(*) FROM orders WHERE DATE(order_date) = CURDATE()), 
     (SELECT COUNT(*) FROM orders WHERE DATE(order_date) = CURDATE())),
    
    ('orders', 'orders_last_hour',
     (SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR)),
     (SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR))),
    
    ('orders', 'pending_orders',
     (SELECT COUNT(*) FROM orders WHERE status = 'pending'),
     (SELECT COUNT(*) FROM orders WHERE status = 'pending')),
    
    -- Revenue metrics
    ('revenue', 'total_revenue_today',
     (SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE DATE(order_date) = CURDATE()),
     NULL),
    
    ('revenue', 'revenue_last_hour',
     (SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR)),
     NULL),
    
    ('revenue', 'average_order_value',
     (SELECT COALESCE(AVG(total_amount), 0) FROM orders WHERE DATE(order_date) = CURDATE()),
     NULL),
    
    -- Customer metrics
    ('customers', 'new_customers_today',
     (SELECT COUNT(*) FROM customers WHERE DATE(created_at) = CURDATE()),
     (SELECT COUNT(*) FROM customers WHERE DATE(created_at) = CURDATE())),
    
    ('customers', 'active_customers_today',
     (SELECT COUNT(DISTINCT customer_id) FROM orders WHERE DATE(order_date) = CURDATE()),
     (SELECT COUNT(DISTINCT customer_id) FROM orders WHERE DATE(order_date) = CURDATE())),
    
    -- Inventory metrics
    ('inventory', 'products_out_of_stock',
     (SELECT COUNT(*) FROM products WHERE stock_quantity = 0 AND status = 'active'),
     (SELECT COUNT(*) FROM products WHERE stock_quantity = 0 AND status = 'active')),
    
    ('inventory', 'products_low_stock',
     (SELECT COUNT(*) FROM products WHERE stock_quantity > 0 AND stock_quantity < 10 AND status = 'active'),
     (SELECT COUNT(*) FROM products WHERE stock_quantity > 0 AND stock_quantity < 10 AND status = 'active'));

SELECT 'Metrics snapshot captured' AS Status, ROW_COUNT() AS Metrics_Captured;

-- ========================================
-- SECTION 5: UPDATE DASHBOARD KPIs
-- ========================================
SELECT '========== UPDATING DASHBOARD KPIs ==========' AS '';

-- Update KPIs with current values and calculate trends
UPDATE rt_dashboard_kpis SET previous_value = current_value;

-- Sales KPIs
UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COUNT(*) FROM orders WHERE DATE(order_date) = CURDATE())
WHERE kpi_name = 'total_orders_today';

UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE DATE(order_date) = CURDATE())
WHERE kpi_name = 'total_revenue_today';

UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COALESCE(AVG(total_amount), 0) FROM orders WHERE DATE(order_date) = CURDATE())
WHERE kpi_name = 'average_order_value';

-- Operations KPIs
UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR))
WHERE kpi_name = 'orders_last_hour';

UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COUNT(*) FROM orders WHERE status = 'pending')
WHERE kpi_name = 'pending_orders';

-- Customer KPIs
UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COUNT(DISTINCT customer_id) FROM orders WHERE DATE(order_date) = CURDATE())
WHERE kpi_name = 'active_customers_today';

UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COUNT(*) FROM customers WHERE DATE(created_at) = CURDATE())
WHERE kpi_name = 'new_customers_today';

-- Inventory KPIs
UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COUNT(*) FROM products WHERE stock_quantity > 0 AND stock_quantity < 10 AND status = 'active')
WHERE kpi_name = 'products_low_stock';

UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COUNT(*) FROM products WHERE stock_quantity = 0 AND status = 'active')
WHERE kpi_name = 'products_out_of_stock';

-- Marketing KPIs
UPDATE rt_dashboard_kpis 
SET current_value = (SELECT COALESCE(SUM(conversions), 0) FROM campaign_performance WHERE report_date = CURDATE())
WHERE kpi_name = 'campaign_conversions_today';

-- Calculate changes and trends
UPDATE rt_dashboard_kpis 
SET 
    change_amount = current_value - previous_value,
    change_percentage = CASE 
        WHEN previous_value = 0 THEN 0
        ELSE ((current_value - previous_value) * 100.0 / previous_value)
    END,
    trend = CASE 
        WHEN current_value > previous_value THEN 'up'
        WHEN current_value < previous_value THEN 'down'
        ELSE 'stable'
    END;

SELECT 'Dashboard KPIs updated' AS Status, ROW_COUNT() AS KPIs_Updated;

-- ========================================
-- SECTION 6: DETECT CHANGES IN LAST HOUR
-- ========================================
SELECT '========== DETECTING RECENT CHANGES ==========' AS '';

-- Detect new orders in last hour
INSERT INTO rt_activity_stream (activity_type, activity_title, activity_details, related_table, related_id, severity)
SELECT 
    'order_placed' AS activity_type,
    CONCAT('Order #', order_id, ' - $', ROUND(total_amount, 2)) AS activity_title,
    CONCAT('Customer: ', customer_id, ', Items: ', 
           (SELECT COUNT(*) FROM order_items WHERE order_id = o.order_id)) AS activity_details,
    'orders' AS related_table,
    order_id AS related_id,
    'info' AS severity
FROM orders o
WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
AND NOT EXISTS (
    SELECT 1 FROM rt_activity_stream 
    WHERE activity_type = 'order_placed' 
    AND related_id = o.order_id 
    AND related_table = 'orders'
)
LIMIT 20;

-- Detect low stock alerts
INSERT INTO rt_activity_stream (activity_type, activity_title, activity_details, related_table, related_id, severity)
SELECT 
    'product_low_stock' AS activity_type,
    CONCAT(product_name, ' - Low Stock') AS activity_title,
    CONCAT('Current stock: ', stock_quantity, ', SKU: ', sku) AS activity_details,
    'products' AS related_table,
    product_id AS related_id,
    'warning' AS severity
FROM products
WHERE stock_quantity > 0 AND stock_quantity < 10 AND status = 'active'
AND NOT EXISTS (
    SELECT 1 FROM rt_activity_stream 
    WHERE activity_type = 'product_low_stock' 
    AND related_id = products.product_id 
    AND related_table = 'products'
    AND created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
)
LIMIT 10;

-- Detect out of stock
INSERT INTO rt_activity_stream (activity_type, activity_title, activity_details, related_table, related_id, severity)
SELECT 
    'product_low_stock' AS activity_type,
    CONCAT(product_name, ' - OUT OF STOCK') AS activity_title,
    CONCAT('Stock depleted, SKU: ', sku) AS activity_details,
    'products' AS related_table,
    product_id AS related_id,
    'critical' AS severity
FROM products
WHERE stock_quantity = 0 AND status = 'active'
AND NOT EXISTS (
    SELECT 1 FROM rt_activity_stream 
    WHERE activity_type = 'product_low_stock' 
    AND related_id = products.product_id 
    AND related_table = 'products'
    AND created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
)
LIMIT 10;

SELECT 'Change detection completed' AS Status;

-- ========================================
-- SECTION 7: CHECK ALERT RULES
-- ========================================
SELECT '========== CHECKING ALERT RULES ==========' AS '';

-- Check for alert conditions and trigger if necessary
-- Alert 1: Check order volume spike
SET @current_orders = (SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 5 MINUTE));
SET @previous_orders = (SELECT COUNT(*) FROM orders WHERE order_date BETWEEN DATE_SUB(NOW(), INTERVAL 10 MINUTE) AND DATE_SUB(NOW(), INTERVAL 5 MINUTE));
SET @order_change_pct = CASE WHEN @previous_orders > 0 THEN ((@current_orders - @previous_orders) * 100.0 / @previous_orders) ELSE 0 END;

INSERT INTO rt_alert_history (alert_id, metric_value, threshold_value, alert_message)
SELECT 
    alert_id,
    @order_change_pct AS metric_value,
    percentage_threshold AS threshold_value,
    CONCAT(alert_message, ' (Current: ', ROUND(@order_change_pct, 2), '%)') AS alert_message
FROM rt_alert_rules
WHERE alert_name = 'High Order Volume Spike'
AND is_active = TRUE
AND @order_change_pct > percentage_threshold
AND NOT EXISTS (
    SELECT 1 FROM rt_alert_history ah
    WHERE ah.alert_id = rt_alert_rules.alert_id
    AND ah.triggered_at >= DATE_SUB(NOW(), INTERVAL time_window_minutes MINUTE)
);

-- Alert 2: Check inventory critical level
SET @out_of_stock_count = (SELECT COUNT(*) FROM products WHERE stock_quantity = 0 AND status = 'active');

INSERT INTO rt_alert_history (alert_id, metric_value, threshold_value, alert_message)
SELECT 
    alert_id,
    @out_of_stock_count AS metric_value,
    threshold_value,
    CONCAT(alert_message, ' (Current: ', @out_of_stock_count, ' products)') AS alert_message
FROM rt_alert_rules
WHERE alert_name = 'Critical Inventory Level'
AND is_active = TRUE
AND @out_of_stock_count > threshold_value
AND NOT EXISTS (
    SELECT 1 FROM rt_alert_history ah
    WHERE ah.alert_id = rt_alert_rules.alert_id
    AND ah.triggered_at >= DATE_SUB(NOW(), INTERVAL time_window_minutes MINUTE)
);

-- Alert 3: Check payment failure rate
SET @total_orders_hour = (SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR));
SET @failed_payments = (SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR) AND payment_status = 'failed');
SET @failure_rate = CASE WHEN @total_orders_hour > 0 THEN (@failed_payments * 100.0 / @total_orders_hour) ELSE 0 END;

INSERT INTO rt_alert_history (alert_id, metric_value, threshold_value, alert_message)
SELECT 
    alert_id,
    @failure_rate AS metric_value,
    threshold_value,
    CONCAT(alert_message, ' (Current: ', ROUND(@failure_rate, 2), '%)') AS alert_message
FROM rt_alert_rules
WHERE alert_name = 'High Payment Failure Rate'
AND is_active = TRUE
AND @failure_rate > threshold_value
AND NOT EXISTS (
    SELECT 1 FROM rt_alert_history ah
    WHERE ah.alert_id = rt_alert_rules.alert_id
    AND ah.triggered_at >= DATE_SUB(NOW(), INTERVAL 30 MINUTE)
);

SELECT 'Alert rules checked' AS Status;

-- ========================================
-- SECTION 8: LIVE DASHBOARD - CURRENT SNAPSHOT
-- ========================================
SELECT '========== LIVE DASHBOARD - REAL-TIME KPIs ==========' AS '';

SELECT 
    kpi_name AS KPI,
    kpi_category AS Category,
    ROUND(current_value, 2) AS Current_Value,
    ROUND(previous_value, 2) AS Previous_Value,
    ROUND(change_amount, 2) AS Change,
    CONCAT(ROUND(change_percentage, 2), '%') AS Change_Pct,
    trend AS Trend,
    CONCAT(update_frequency_seconds, 's') AS Update_Frequency,
    last_updated AS Last_Updated
FROM rt_dashboard_kpis
ORDER BY 
    FIELD(kpi_category, 'sales', 'operations', 'customer', 'inventory', 'marketing'),
    kpi_name;

-- ========================================
-- SECTION 9: ACTIVE ALERTS
-- ========================================
SELECT '========== ACTIVE ALERTS - REQUIRE ATTENTION ==========' AS '';

SELECT 
    ar.alert_name AS Alert_Name,
    ar.severity AS Severity,
    ah.triggered_at AS Triggered_At,
    TIMESTAMPDIFF(MINUTE, ah.triggered_at, NOW()) AS Minutes_Ago,
    ROUND(ah.metric_value, 2) AS Current_Value,
    ROUND(ah.threshold_value, 2) AS Threshold,
    ah.alert_message AS Message,
    ah.acknowledged AS Acknowledged
FROM rt_alert_history ah
JOIN rt_alert_rules ar ON ah.alert_id = ar.alert_id
WHERE ah.acknowledged = FALSE
AND ah.triggered_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
ORDER BY 
    FIELD(ar.severity, 'critical', 'high', 'medium', 'low'),
    ah.triggered_at DESC;

-- ========================================
-- SECTION 10: RECENT ACTIVITY STREAM
-- ========================================
SELECT '========== RECENT ACTIVITY STREAM (LAST HOUR) ==========' AS '';

SELECT 
    activity_type AS Activity_Type,
    activity_title AS Title,
    activity_details AS Details,
    severity AS Severity,
    TIMESTAMPDIFF(MINUTE, created_at, NOW()) AS Minutes_Ago,
    created_at AS Timestamp
FROM rt_activity_stream
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
ORDER BY created_at DESC
LIMIT 50;

-- ========================================
-- SECTION 11: CHANGE DETECTION SUMMARY
-- ========================================
SELECT '========== CHANGE DETECTION SUMMARY (LAST HOUR) ==========' AS '';

SELECT 
    table_name AS Table_Name,
    change_type AS Change_Type,
    COUNT(*) AS Change_Count,
    MIN(changed_at) AS First_Change,
    MAX(changed_at) AS Last_Change
FROM rt_change_log
WHERE changed_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
GROUP BY table_name, change_type
ORDER BY Change_Count DESC;

-- ========================================
-- SECTION 12: PERFORMANCE METRICS
-- ========================================
SELECT '========== SYSTEM PERFORMANCE METRICS ==========' AS '';

SELECT 
    'Orders Processing Rate' AS Metric,
    CONCAT(COUNT(*), ' orders/hour') AS Value,
    'Orders created in last hour' AS Description
FROM orders
WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 HOUR)

UNION ALL

SELECT 
    'Average Response Time' AS Metric,
    'N/A' AS Value,
    'Requires application-level monitoring' AS Description

UNION ALL

SELECT 
    'Database Size' AS Metric,
    CONCAT(ROUND(SUM(DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2), ' MB') AS Value,
    'Total database size' AS Description
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'ecommerce_analytics'

UNION ALL

SELECT 
    'Active Connections' AS Metric,
    CAST(COUNT(*) AS CHAR) AS Value,
    'Current database connections' AS Description
FROM information_schema.PROCESSLIST;

-- ========================================
-- SECTION 13: TOP PRODUCTS TRENDING NOW
-- ========================================
SELECT '========== TRENDING PRODUCTS (LAST 24 HOURS) ==========' AS '';

SELECT 
    p.product_name AS Product_Name,
    COUNT(DISTINCT oi.order_id) AS Orders_Count,
    SUM(oi.quantity) AS Units_Sold,
    ROUND(SUM(oi.subtotal), 2) AS Revenue,
    p.stock_quantity AS Current_Stock,
    CASE 
        WHEN p.stock_quantity = 0 THEN 'OUT OF STOCK'
        WHEN p.stock_quantity < 10 THEN 'LOW STOCK'
        ELSE 'IN STOCK'
    END AS Stock_Status
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY p.product_id, p.product_name, p.stock_quantity
ORDER BY Units_Sold DESC
LIMIT 10;

-- ========================================
-- SECTION 14: HOURLY TREND ANALYSIS
-- ========================================
SELECT '========== HOURLY TREND ANALYSIS (TODAY) ==========' AS '';

SELECT 
    HOUR(order_date) AS Hour_Of_Day,
    COUNT(*) AS Orders,
    ROUND(SUM(total_amount), 2) AS Revenue,
    ROUND(AVG(total_amount), 2) AS Avg_Order_Value,
    COUNT(DISTINCT customer_id) AS Unique_Customers
FROM orders
WHERE DATE(order_date) = CURDATE()
GROUP BY HOUR(order_date)
ORDER BY Hour_Of_Day;

-- ========================================
-- SECTION 15: MONITORING HEALTH CHECK
-- ========================================
SELECT '========== MONITORING SYSTEM HEALTH CHECK ==========' AS '';

SELECT 
    'Metrics Captured (Last Hour)' AS Check_Type,
    COUNT(*) AS Count,
    CASE WHEN COUNT(*) > 0 THEN 'ACTIVE' ELSE 'IDLE' END AS Status
FROM rt_metrics_snapshot
WHERE snapshot_time >= DATE_SUB(NOW(), INTERVAL 1 HOUR)

UNION ALL

SELECT 
    'Active Alerts' AS Check_Type,
    COUNT(*) AS Count,
    CASE WHEN COUNT(*) > 0 THEN 'ALERTS PENDING' ELSE 'NO ALERTS' END AS Status
FROM rt_alert_history
WHERE acknowledged = FALSE AND triggered_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)

UNION ALL

SELECT 
    'Activity Events (Last Hour)' AS Check_Type,
    COUNT(*) AS Count,
    CASE WHEN COUNT(*) > 0 THEN 'ACTIVE' ELSE 'IDLE' END AS Status
FROM rt_activity_stream
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)

UNION ALL

SELECT 
    'KPI Updates Status' AS Check_Type,
    COUNT(*) AS Count,
    CASE 
        WHEN MAX(last_updated) >= DATE_SUB(NOW(), INTERVAL 5 MINUTE) THEN 'CURRENT'
        ELSE 'STALE'
    END AS Status
FROM rt_dashboard_kpis;

-- ========================================
-- FINAL MESSAGE
-- ========================================
SELECT '========================================' AS '';
SELECT 'Real-Time Monitoring Snapshot Complete' AS Result;
SELECT NOW() AS Snapshot_Time;
SELECT 'Dashboard refreshes automatically based on KPI frequency' AS Note;
SELECT '========================================' AS '';