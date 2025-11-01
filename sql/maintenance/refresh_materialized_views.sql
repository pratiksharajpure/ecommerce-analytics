-- ========================================
-- REFRESH MATERIALIZED VIEWS SCRIPT
-- E-commerce Revenue Analytics Engine
-- Full & Incremental Refresh with Dependencies
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- SECTION 1: CREATE MATERIALIZED VIEWS INFRASTRUCTURE
-- ========================================
SELECT '========== SETTING UP MATERIALIZED VIEWS INFRASTRUCTURE ==========' AS '';

-- Create table to track materialized view refresh history
CREATE TABLE IF NOT EXISTS mv_refresh_log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    view_name VARCHAR(100) NOT NULL,
    refresh_type ENUM('full', 'incremental') NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    rows_affected INT,
    status ENUM('running', 'completed', 'failed') DEFAULT 'running',
    error_message TEXT,
    INDEX idx_view_name (view_name),
    INDEX idx_start_time (start_time),
    INDEX idx_status (status)
) ENGINE=InnoDB;

-- Create table to track last refresh timestamp for incremental updates
CREATE TABLE IF NOT EXISTS mv_last_refresh (
    view_name VARCHAR(100) PRIMARY KEY,
    last_refresh_time TIMESTAMP NOT NULL,
    next_refresh_due TIMESTAMP,
    refresh_frequency_minutes INT DEFAULT 60,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

SELECT 'Materialized views infrastructure created' AS Status;

-- ========================================
-- SECTION 2: CREATE MATERIALIZED VIEW TABLES
-- ========================================
SELECT '========== CREATING MATERIALIZED VIEW TABLES ==========' AS '';

-- MV 1: Daily Sales Summary (Materialized)
CREATE TABLE IF NOT EXISTS mv_daily_sales_summary (
    summary_date DATE PRIMARY KEY,
    total_orders INT DEFAULT 0,
    total_revenue DECIMAL(15,2) DEFAULT 0.00,
    total_items_sold INT DEFAULT 0,
    unique_customers INT DEFAULT 0,
    average_order_value DECIMAL(10,2) DEFAULT 0.00,
    total_shipping_cost DECIMAL(12,2) DEFAULT 0.00,
    total_tax DECIMAL(12,2) DEFAULT 0.00,
    cancelled_orders INT DEFAULT 0,
    refunded_orders INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_summary_date (summary_date),
    INDEX idx_total_revenue (total_revenue)
) ENGINE=InnoDB;

-- MV 2: Product Performance Summary (Materialized)
CREATE TABLE IF NOT EXISTS mv_product_performance (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(200),
    category_name VARCHAR(100),
    total_orders INT DEFAULT 0,
    total_quantity_sold INT DEFAULT 0,
    total_revenue DECIMAL(15,2) DEFAULT 0.00,
    total_cost DECIMAL(15,2) DEFAULT 0.00,
    total_profit DECIMAL(15,2) DEFAULT 0.00,
    profit_margin_pct DECIMAL(5,2) DEFAULT 0.00,
    average_rating DECIMAL(3,2),
    review_count INT DEFAULT 0,
    return_count INT DEFAULT 0,
    return_rate_pct DECIMAL(5,2) DEFAULT 0.00,
    last_order_date TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_product_name (product_name),
    INDEX idx_total_revenue (total_revenue),
    INDEX idx_profit_margin (profit_margin_pct)
) ENGINE=InnoDB;

-- MV 3: Customer Lifetime Value (Materialized)
CREATE TABLE IF NOT EXISTS mv_customer_ltv (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    email VARCHAR(100),
    total_orders INT DEFAULT 0,
    lifetime_value DECIMAL(15,2) DEFAULT 0.00,
    average_order_value DECIMAL(10,2) DEFAULT 0.00,
    first_order_date TIMESTAMP,
    last_order_date TIMESTAMP,
    days_since_last_order INT,
    customer_tier ENUM('bronze', 'silver', 'gold', 'platinum'),
    loyalty_points INT DEFAULT 0,
    total_reviews INT DEFAULT 0,
    average_review_rating DECIMAL(3,2),
    total_returns INT DEFAULT 0,
    churn_risk ENUM('low', 'medium', 'high'),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_lifetime_value (lifetime_value),
    INDEX idx_customer_tier (customer_tier),
    INDEX idx_churn_risk (churn_risk)
) ENGINE=InnoDB;

-- MV 4: Inventory Alert Summary (Materialized)
CREATE TABLE IF NOT EXISTS mv_inventory_alerts (
    product_id INT PRIMARY KEY,
    sku VARCHAR(50),
    product_name VARCHAR(200),
    current_stock INT DEFAULT 0,
    reserved_stock INT DEFAULT 0,
    available_stock INT DEFAULT 0,
    reorder_level INT DEFAULT 0,
    stock_status ENUM('critical', 'low', 'normal', 'overstocked'),
    days_until_stockout INT,
    recommended_order_quantity INT,
    vendor_count INT DEFAULT 0,
    best_vendor_id INT,
    best_cost_per_unit DECIMAL(10,2),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_stock_status (stock_status),
    INDEX idx_days_until_stockout (days_until_stockout)
) ENGINE=InnoDB;

-- MV 5: Campaign Performance Dashboard (Materialized)
CREATE TABLE IF NOT EXISTS mv_campaign_dashboard (
    campaign_id INT PRIMARY KEY,
    campaign_name VARCHAR(200),
    campaign_type VARCHAR(50),
    total_impressions BIGINT DEFAULT 0,
    total_clicks INT DEFAULT 0,
    total_conversions INT DEFAULT 0,
    total_spend DECIMAL(12,2) DEFAULT 0.00,
    total_revenue DECIMAL(15,2) DEFAULT 0.00,
    ctr_pct DECIMAL(5,2) DEFAULT 0.00,
    conversion_rate_pct DECIMAL(5,2) DEFAULT 0.00,
    cpa DECIMAL(10,2) DEFAULT 0.00,
    roas DECIMAL(10,2) DEFAULT 0.00,
    roi_pct DECIMAL(10,2) DEFAULT 0.00,
    campaign_status VARCHAR(20),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_campaign_type (campaign_type),
    INDEX idx_roi_pct (roi_pct)
) ENGINE=InnoDB;

SELECT 'All materialized view tables created' AS Status;

-- ========================================
-- SECTION 3: FULL REFRESH - DAILY SALES SUMMARY
-- ========================================
SELECT '========== REFRESHING: Daily Sales Summary (FULL) ==========' AS '';

-- Log refresh start
INSERT INTO mv_refresh_log (view_name, refresh_type, status)
VALUES ('mv_daily_sales_summary', 'full', 'running');
SET @log_id_daily_sales = LAST_INSERT_ID();

-- Truncate and rebuild
TRUNCATE TABLE mv_daily_sales_summary;

INSERT INTO mv_daily_sales_summary (
    summary_date, total_orders, total_revenue, total_items_sold, 
    unique_customers, average_order_value, total_shipping_cost, 
    total_tax, cancelled_orders, refunded_orders
)
SELECT 
    DATE(o.order_date) AS summary_date,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COALESCE(SUM(o.total_amount), 0) AS total_revenue,
    COALESCE(SUM(oi.quantity), 0) AS total_items_sold,
    COUNT(DISTINCT o.customer_id) AS unique_customers,
    COALESCE(AVG(o.total_amount), 0) AS average_order_value,
    COALESCE(SUM(o.shipping_cost), 0) AS total_shipping_cost,
    COALESCE(SUM(o.tax_amount), 0) AS total_tax,
    SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    SUM(CASE WHEN o.payment_status = 'refunded' THEN 1 ELSE 0 END) AS refunded_orders
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY DATE(o.order_date);

-- Update refresh log
UPDATE mv_refresh_log 
SET end_time = NOW(), 
    rows_affected = ROW_COUNT(), 
    status = 'completed'
WHERE log_id = @log_id_daily_sales;

-- Update last refresh tracker
INSERT INTO mv_last_refresh (view_name, last_refresh_time, next_refresh_due, refresh_frequency_minutes)
VALUES ('mv_daily_sales_summary', NOW(), DATE_ADD(NOW(), INTERVAL 60 MINUTE), 60)
ON DUPLICATE KEY UPDATE 
    last_refresh_time = NOW(),
    next_refresh_due = DATE_ADD(NOW(), INTERVAL refresh_frequency_minutes MINUTE);

SELECT 'Daily Sales Summary refreshed' AS Status, ROW_COUNT() AS Rows_Updated;

-- ========================================
-- SECTION 4: FULL REFRESH - PRODUCT PERFORMANCE
-- ========================================
SELECT '========== REFRESHING: Product Performance (FULL) ==========' AS '';

-- Log refresh start
INSERT INTO mv_refresh_log (view_name, refresh_type, status)
VALUES ('mv_product_performance', 'full', 'running');
SET @log_id_product = LAST_INSERT_ID();

-- Truncate and rebuild
TRUNCATE TABLE mv_product_performance;

INSERT INTO mv_product_performance (
    product_id, product_name, category_name, total_orders,
    total_quantity_sold, total_revenue, total_cost, total_profit,
    profit_margin_pct, average_rating, review_count, return_count,
    return_rate_pct, last_order_date
)
SELECT 
    p.product_id,
    p.product_name,
    pc.category_name,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    COALESCE(SUM(oi.quantity), 0) AS total_quantity_sold,
    COALESCE(SUM(oi.subtotal), 0) AS total_revenue,
    COALESCE(SUM(oi.quantity * p.cost), 0) AS total_cost,
    COALESCE(SUM(oi.subtotal) - SUM(oi.quantity * p.cost), 0) AS total_profit,
    CASE 
        WHEN SUM(oi.subtotal) > 0 
        THEN ((SUM(oi.subtotal) - SUM(oi.quantity * p.cost)) * 100.0 / SUM(oi.subtotal))
        ELSE 0 
    END AS profit_margin_pct,
    (SELECT AVG(rating) FROM reviews WHERE product_id = p.product_id AND status = 'approved') AS average_rating,
    (SELECT COUNT(*) FROM reviews WHERE product_id = p.product_id AND status = 'approved') AS review_count,
    (SELECT COUNT(*) FROM returns r JOIN order_items oi2 ON r.order_item_id = oi2.order_item_id WHERE oi2.product_id = p.product_id) AS return_count,
    CASE 
        WHEN COUNT(DISTINCT oi.order_id) > 0
        THEN ((SELECT COUNT(*) FROM returns r JOIN order_items oi2 ON r.order_item_id = oi2.order_item_id WHERE oi2.product_id = p.product_id) * 100.0 / COUNT(DISTINCT oi.order_id))
        ELSE 0
    END AS return_rate_pct,
    MAX(o.order_date) AS last_order_date
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id
GROUP BY p.product_id, p.product_name, pc.category_name, p.cost;

-- Update refresh log
UPDATE mv_refresh_log 
SET end_time = NOW(), 
    rows_affected = ROW_COUNT(), 
    status = 'completed'
WHERE log_id = @log_id_product;

-- Update last refresh tracker
INSERT INTO mv_last_refresh (view_name, last_refresh_time, next_refresh_due, refresh_frequency_minutes)
VALUES ('mv_product_performance', NOW(), DATE_ADD(NOW(), INTERVAL 120 MINUTE), 120)
ON DUPLICATE KEY UPDATE 
    last_refresh_time = NOW(),
    next_refresh_due = DATE_ADD(NOW(), INTERVAL refresh_frequency_minutes MINUTE);

SELECT 'Product Performance refreshed' AS Status, ROW_COUNT() AS Rows_Updated;

-- ========================================
-- SECTION 5: FULL REFRESH - CUSTOMER LTV
-- ========================================
SELECT '========== REFRESHING: Customer Lifetime Value (FULL) ==========' AS '';

-- Log refresh start
INSERT INTO mv_refresh_log (view_name, refresh_type, status)
VALUES ('mv_customer_ltv', 'full', 'running');
SET @log_id_ltv = LAST_INSERT_ID();

-- Truncate and rebuild
TRUNCATE TABLE mv_customer_ltv;

INSERT INTO mv_customer_ltv (
    customer_id, customer_name, email, total_orders,
    lifetime_value, average_order_value, first_order_date,
    last_order_date, days_since_last_order, customer_tier,
    loyalty_points, total_reviews, average_review_rating,
    total_returns, churn_risk
)
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COALESCE(SUM(o.total_amount), 0) AS lifetime_value,
    COALESCE(AVG(o.total_amount), 0) AS average_order_value,
    MIN(o.order_date) AS first_order_date,
    MAX(o.order_date) AS last_order_date,
    DATEDIFF(NOW(), MAX(o.order_date)) AS days_since_last_order,
    COALESCE(lp.tier, 'bronze') AS customer_tier,
    COALESCE(lp.points_balance, 0) AS loyalty_points,
    (SELECT COUNT(*) FROM reviews WHERE customer_id = c.customer_id) AS total_reviews,
    (SELECT AVG(rating) FROM reviews WHERE customer_id = c.customer_id) AS average_review_rating,
    (SELECT COUNT(*) FROM returns r WHERE r.order_id IN (SELECT order_id FROM orders WHERE customer_id = c.customer_id)) AS total_returns,
    CASE 
        WHEN DATEDIFF(NOW(), MAX(o.order_date)) > 180 THEN 'high'
        WHEN DATEDIFF(NOW(), MAX(o.order_date)) > 90 THEN 'medium'
        ELSE 'low'
    END AS churn_risk
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, lp.tier, lp.points_balance;

-- Update refresh log
UPDATE mv_refresh_log 
SET end_time = NOW(), 
    rows_affected = ROW_COUNT(), 
    status = 'completed'
WHERE log_id = @log_id_ltv;

-- Update last refresh tracker
INSERT INTO mv_last_refresh (view_name, last_refresh_time, next_refresh_due, refresh_frequency_minutes)
VALUES ('mv_customer_ltv', NOW(), DATE_ADD(NOW(), INTERVAL 240 MINUTE), 240)
ON DUPLICATE KEY UPDATE 
    last_refresh_time = NOW(),
    next_refresh_due = DATE_ADD(NOW(), INTERVAL refresh_frequency_minutes MINUTE);

SELECT 'Customer LTV refreshed' AS Status, ROW_COUNT() AS Rows_Updated;

-- ========================================
-- SECTION 6: FULL REFRESH - INVENTORY ALERTS
-- ========================================
SELECT '========== REFRESHING: Inventory Alerts (FULL) ==========' AS '';

-- Log refresh start
INSERT INTO mv_refresh_log (view_name, refresh_type, status)
VALUES ('mv_inventory_alerts', 'full', 'running');
SET @log_id_inventory = LAST_INSERT_ID();

-- Truncate and rebuild
TRUNCATE TABLE mv_inventory_alerts;

INSERT INTO mv_inventory_alerts (
    product_id, sku, product_name, current_stock,
    reserved_stock, available_stock, reorder_level,
    stock_status, days_until_stockout, recommended_order_quantity,
    vendor_count, best_vendor_id, best_cost_per_unit
)
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    COALESCE(SUM(i.quantity_on_hand), 0) AS current_stock,
    COALESCE(SUM(i.quantity_reserved), 0) AS reserved_stock,
    COALESCE(SUM(i.quantity_available), 0) AS available_stock,
    COALESCE(MAX(i.reorder_level), 10) AS reorder_level,
    CASE 
        WHEN COALESCE(SUM(i.quantity_available), 0) = 0 THEN 'critical'
        WHEN COALESCE(SUM(i.quantity_available), 0) < COALESCE(MAX(i.reorder_level), 10) THEN 'low'
        WHEN COALESCE(SUM(i.quantity_available), 0) > (COALESCE(MAX(i.reorder_level), 10) * 5) THEN 'overstocked'
        ELSE 'normal'
    END AS stock_status,
    CASE 
        WHEN (SELECT AVG(oi.quantity) FROM order_items oi WHERE oi.product_id = p.product_id AND oi.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)) > 0
        THEN FLOOR(COALESCE(SUM(i.quantity_available), 0) / (SELECT AVG(oi.quantity) FROM order_items oi WHERE oi.product_id = p.product_id AND oi.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)))
        ELSE 999
    END AS days_until_stockout,
    GREATEST(COALESCE(MAX(i.reorder_level), 10) * 3 - COALESCE(SUM(i.quantity_available), 0), 0) AS recommended_order_quantity,
    (SELECT COUNT(*) FROM vendor_contracts WHERE product_id = p.product_id AND status = 'active') AS vendor_count,
    (SELECT vendor_id FROM vendor_contracts WHERE product_id = p.product_id AND status = 'active' ORDER BY cost_per_unit LIMIT 1) AS best_vendor_id,
    (SELECT MIN(cost_per_unit) FROM vendor_contracts WHERE product_id = p.product_id AND status = 'active') AS best_cost_per_unit
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
GROUP BY p.product_id, p.sku, p.product_name;

-- Update refresh log
UPDATE mv_refresh_log 
SET end_time = NOW(), 
    rows_affected = ROW_COUNT(), 
    status = 'completed'
WHERE log_id = @log_id_inventory;

-- Update last refresh tracker
INSERT INTO mv_last_refresh (view_name, last_refresh_time, next_refresh_due, refresh_frequency_minutes)
VALUES ('mv_inventory_alerts', NOW(), DATE_ADD(NOW(), INTERVAL 30 MINUTE), 30)
ON DUPLICATE KEY UPDATE 
    last_refresh_time = NOW(),
    next_refresh_due = DATE_ADD(NOW(), INTERVAL refresh_frequency_minutes MINUTE);

SELECT 'Inventory Alerts refreshed' AS Status, ROW_COUNT() AS Rows_Updated;

-- ========================================
-- SECTION 7: FULL REFRESH - CAMPAIGN DASHBOARD
-- ========================================
SELECT '========== REFRESHING: Campaign Dashboard (FULL) ==========' AS '';

-- Log refresh start
INSERT INTO mv_refresh_log (view_name, refresh_type, status)
VALUES ('mv_campaign_dashboard', 'full', 'running');
SET @log_id_campaign = LAST_INSERT_ID();

-- Truncate and rebuild
TRUNCATE TABLE mv_campaign_dashboard;

INSERT INTO mv_campaign_dashboard (
    campaign_id, campaign_name, campaign_type, total_impressions,
    total_clicks, total_conversions, total_spend, total_revenue,
    ctr_pct, conversion_rate_pct, cpa, roas, roi_pct, campaign_status
)
SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    COALESCE(SUM(cp.impressions), 0) AS total_impressions,
    COALESCE(SUM(cp.clicks), 0) AS total_clicks,
    COALESCE(SUM(cp.conversions), 0) AS total_conversions,
    COALESCE(SUM(cp.spend), 0) AS total_spend,
    COALESCE(SUM(cp.revenue), 0) AS total_revenue,
    CASE 
        WHEN SUM(cp.impressions) > 0 THEN (SUM(cp.clicks) * 100.0 / SUM(cp.impressions))
        ELSE 0 
    END AS ctr_pct,
    CASE 
        WHEN SUM(cp.clicks) > 0 THEN (SUM(cp.conversions) * 100.0 / SUM(cp.clicks))
        ELSE 0 
    END AS conversion_rate_pct,
    CASE 
        WHEN SUM(cp.conversions) > 0 THEN (SUM(cp.spend) / SUM(cp.conversions))
        ELSE 0 
    END AS cpa,
    CASE 
        WHEN SUM(cp.spend) > 0 THEN (SUM(cp.revenue) / SUM(cp.spend))
        ELSE 0 
    END AS roas,
    CASE 
        WHEN SUM(cp.spend) > 0 THEN ((SUM(cp.revenue) - SUM(cp.spend)) * 100.0 / SUM(cp.spend))
        ELSE 0 
    END AS roi_pct,
    c.status AS campaign_status
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.status;

-- Update refresh log
UPDATE mv_refresh_log 
SET end_time = NOW(), 
    rows_affected = ROW_COUNT(), 
    status = 'completed'
WHERE log_id = @log_id_campaign;

-- Update last refresh tracker
INSERT INTO mv_last_refresh (view_name, last_refresh_time, next_refresh_due, refresh_frequency_minutes)
VALUES ('mv_campaign_dashboard', NOW(), DATE_ADD(NOW(), INTERVAL 60 MINUTE), 60)
ON DUPLICATE KEY UPDATE 
    last_refresh_time = NOW(),
    next_refresh_due = DATE_ADD(NOW(), INTERVAL refresh_frequency_minutes MINUTE);

SELECT 'Campaign Dashboard refreshed' AS Status, ROW_COUNT() AS Rows_Updated;

-- ========================================
-- SECTION 8: INCREMENTAL REFRESH - TODAY'S DATA ONLY
-- ========================================
SELECT '========== INCREMENTAL REFRESH: Today''s Data ==========' AS '';

-- Refresh today's data in daily sales summary
DELETE FROM mv_daily_sales_summary WHERE summary_date = CURDATE();

INSERT INTO mv_daily_sales_summary (
    summary_date, total_orders, total_revenue, total_items_sold, 
    unique_customers, average_order_value, total_shipping_cost, 
    total_tax, cancelled_orders, refunded_orders
)
SELECT 
    CURDATE() AS summary_date,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COALESCE(SUM(o.total_amount), 0) AS total_revenue,
    COALESCE(SUM(oi.quantity), 0) AS total_items_sold,
    COUNT(DISTINCT o.customer_id) AS unique_customers,
    COALESCE(AVG(o.total_amount), 0) AS average_order_value,
    COALESCE(SUM(o.shipping_cost), 0) AS total_shipping_cost,
    COALESCE(SUM(o.tax_amount), 0) AS total_tax,
    SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    SUM(CASE WHEN o.payment_status = 'refunded' THEN 1 ELSE 0 END) AS refunded_orders
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE DATE(o.order_date) = CURDATE();

SELECT 'Today''s data incrementally refreshed' AS Status;

-- ========================================
-- SECTION 9: VIEW DEPENDENCIES CHECK
-- ========================================
SELECT '========== VIEW DEPENDENCIES CHECK ==========' AS '';

-- Check which base tables are used by which materialized views
SELECT 
    'mv_daily_sales_summary' AS Materialized_View,
    'orders, order_items' AS Dependent_Tables,
    'Daily aggregation of order data' AS Purpose
UNION ALL
SELECT 
    'mv_product_performance',
    'products, product_categories, order_items, orders, reviews, returns',
    'Product sales and profitability metrics'
UNION ALL
SELECT 
    'mv_customer_ltv',
    'customers, orders, loyalty_program, reviews, returns',
    'Customer value and segmentation'
UNION ALL
SELECT 
    'mv_inventory_alerts',
    'products, inventory, order_items, vendor_contracts',
    'Stock levels and reorder recommendations'
UNION ALL
SELECT 
    'mv_campaign_dashboard',
    'campaigns, campaign_performance',
    'Marketing campaign analytics';

-- ========================================
-- SECTION 10: REFRESH SCHEDULE STATUS
-- ========================================
SELECT '========== REFRESH SCHEDULE STATUS ==========' AS '';

-- Show when each view was last refreshed and when it's due
SELECT 
    view_name AS Materialized_View,
    last_refresh_time AS Last_Refreshed,
    next_refresh_due AS Next_Refresh_Due,
    refresh_frequency_minutes AS Frequency_Minutes,
    CASE 
        WHEN next_refresh_due < NOW() THEN 'OVERDUE'
        WHEN next_refresh_due < DATE_ADD(NOW(), INTERVAL 15 MINUTE) THEN 'DUE SOON'
        ELSE 'ON SCHEDULE'
    END AS Status
FROM mv_last_refresh
ORDER BY next_refresh_due;

-- ========================================
-- SECTION 11: REFRESH HISTORY
-- ========================================
SELECT '========== RECENT REFRESH HISTORY ==========' AS '';

-- Show last 20 refresh operations
SELECT 
    view_name AS View_Name,
    refresh_type AS Type,
    start_time AS Start_Time,
    end_time AS End_Time,
    TIMESTAMPDIFF(SECOND, start_time, end_time) AS Duration_Seconds,
    rows_affected AS Rows_Affected,
    status AS Status,
    error_message AS Error
FROM mv_refresh_log
ORDER BY start_time DESC
LIMIT 20;

-- ========================================
-- SECTION 12: PERFORMANCE METRICS
-- ========================================
SELECT '========== MATERIALIZED VIEW PERFORMANCE ==========' AS '';

-- Show size and performance of each materialized view
SELECT 
    TABLE_NAME AS View_Name,
    TABLE_ROWS AS Row_Count,
    ROUND((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024, 2) AS Size_MB,
    ROUND(DATA_LENGTH / 1024 / 1024, 2) AS Data_MB,
    ROUND(INDEX_LENGTH / 1024 / 1024, 2) AS Index_MB
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
AND TABLE_NAME LIKE 'mv_%'
ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC;

-- ========================================
-- SECTION 13: VALIDATION CHECKS
-- ========================================
SELECT '========== VALIDATION CHECKS ==========' AS '';

-- Verify data consistency between base tables and materialized views
SELECT 
    'Daily Sales Validation' AS Check_Type,
    (SELECT SUM(total_revenue) FROM mv_daily_sales_summary) AS MV_Total,
    (SELECT SUM(total_amount) FROM orders) AS Base_Total,
    CASE 
        WHEN ABS((SELECT SUM(total_revenue) FROM mv_daily_sales_summary) - (SELECT SUM(total_amount) FROM orders)) < 0.01 THEN 'PASS'
        ELSE 'FAIL'
    END AS Status;

-- ========================================
-- FINAL MESSAGE
-- ========================================
SELECT '========================================' AS '';
SELECT 'Materialized Views Refresh Completed' AS Result;
SELECT CONCAT('Total execution time: ', ROUND((SELECT SUM(TIMESTAMPDIFF(SECOND, start_time, end_time)) FROM mv_refresh_log WHERE start_time >= DATE_SUB(NOW(), INTERVAL 1 HOUR)), 2), ' seconds') AS Performance;
SELECT '========================================' AS '';