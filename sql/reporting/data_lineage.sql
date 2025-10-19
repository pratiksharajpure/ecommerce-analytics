-- ========================================
-- DATA LINEAGE TRACKING SYSTEM
-- Complete data source tracking
-- Transformation history
-- Data flow visualization
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- SECTION 1: DATA LINEAGE TABLES
-- ========================================

-- 1.1 Data Sources Registry
CREATE TABLE IF NOT EXISTS data_sources (
    source_id INT PRIMARY KEY AUTO_INCREMENT,
    source_name VARCHAR(100) NOT NULL UNIQUE,
    source_type ENUM('database', 'api', 'file', 'stream', 'manual', 'third_party') NOT NULL,
    source_description TEXT,
    connection_string VARCHAR(500),
    source_owner VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    data_retention_days INT DEFAULT 365,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_source_type (source_type),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.2 Data Entities (Tables/Views)
CREATE TABLE IF NOT EXISTS data_entities (
    entity_id INT PRIMARY KEY AUTO_INCREMENT,
    entity_name VARCHAR(100) NOT NULL,
    entity_type ENUM('table', 'view', 'materialized_view', 'staging', 'fact', 'dimension') NOT NULL,
    source_id INT,
    schema_name VARCHAR(100),
    description TEXT,
    row_count_estimate BIGINT,
    last_refresh_date TIMESTAMP,
    refresh_frequency ENUM('real_time', 'hourly', 'daily', 'weekly', 'monthly', 'manual') DEFAULT 'daily',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES data_sources(source_id) ON DELETE SET NULL,
    INDEX idx_entity_name (entity_name),
    INDEX idx_entity_type (entity_type),
    INDEX idx_source_id (source_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.3 Data Transformations
CREATE TABLE IF NOT EXISTS data_transformations (
    transformation_id INT PRIMARY KEY AUTO_INCREMENT,
    transformation_name VARCHAR(200) NOT NULL,
    transformation_type ENUM('etl', 'aggregation', 'join', 'filter', 'calculation', 'cleaning', 'enrichment') NOT NULL,
    source_entity_id INT,
    target_entity_id INT,
    transformation_logic TEXT,
    transformation_sql TEXT,
    execution_order INT,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES data_entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES data_entities(entity_id) ON DELETE CASCADE,
    INDEX idx_source_entity (source_entity_id),
    INDEX idx_target_entity (target_entity_id),
    INDEX idx_transformation_type (transformation_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.4 Data Lineage (Relationships)
CREATE TABLE IF NOT EXISTS data_lineage (
    lineage_id INT PRIMARY KEY AUTO_INCREMENT,
    source_entity_id INT NOT NULL,
    target_entity_id INT NOT NULL,
    transformation_id INT,
    lineage_type ENUM('direct', 'derived', 'aggregated', 'joined', 'filtered') NOT NULL,
    column_mapping JSON,
    dependency_level INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES data_entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES data_entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (transformation_id) REFERENCES data_transformations(transformation_id) ON DELETE SET NULL,
    INDEX idx_source_entity (source_entity_id),
    INDEX idx_target_entity (target_entity_id),
    INDEX idx_lineage_type (lineage_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.5 Transformation Execution Log
CREATE TABLE IF NOT EXISTS transformation_execution_log (
    execution_id INT PRIMARY KEY AUTO_INCREMENT,
    transformation_id INT NOT NULL,
    execution_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_end TIMESTAMP NULL,
    execution_status ENUM('running', 'success', 'failed', 'cancelled') DEFAULT 'running',
    rows_processed BIGINT DEFAULT 0,
    rows_inserted BIGINT DEFAULT 0,
    rows_updated BIGINT DEFAULT 0,
    rows_deleted BIGINT DEFAULT 0,
    error_message TEXT,
    execution_duration_seconds INT AS (TIMESTAMPDIFF(SECOND, execution_start, execution_end)) STORED,
    FOREIGN KEY (transformation_id) REFERENCES data_transformations(transformation_id) ON DELETE CASCADE,
    INDEX idx_transformation_id (transformation_id),
    INDEX idx_execution_status (execution_status),
    INDEX idx_execution_start (execution_start)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 1.6 Data Quality Checkpoints
CREATE TABLE IF NOT EXISTS data_quality_checkpoints (
    checkpoint_id INT PRIMARY KEY AUTO_INCREMENT,
    entity_id INT NOT NULL,
    checkpoint_name VARCHAR(200) NOT NULL,
    quality_rule TEXT NOT NULL,
    rule_type ENUM('completeness', 'accuracy', 'consistency', 'validity', 'uniqueness', 'timeliness') NOT NULL,
    threshold_value DECIMAL(5,2),
    actual_value DECIMAL(5,2),
    check_status ENUM('pass', 'fail', 'warning') DEFAULT 'pass',
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entity_id) REFERENCES data_entities(entity_id) ON DELETE CASCADE,
    INDEX idx_entity_id (entity_id),
    INDEX idx_check_status (check_status),
    INDEX idx_checked_at (checked_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- SECTION 2: INSERT SAMPLE DATA SOURCES
-- ========================================

INSERT INTO data_sources (source_name, source_type, source_description, source_owner, is_active) VALUES
('Production Database', 'database', 'Main production MySQL database containing live transactional data', 'DBA Team', TRUE),
('Salesforce API', 'api', 'Customer and lead data from Salesforce CRM', 'Sales Ops', TRUE),
('Google Analytics', 'third_party', 'Website traffic and conversion data', 'Marketing Team', TRUE),
('Payment Gateway', 'api', 'Payment transaction data from Stripe', 'Finance Team', TRUE),
('Product Catalog CSV', 'file', 'Daily product catalog updates from suppliers', 'Inventory Team', TRUE),
('Manual Entry', 'manual', 'Manually entered data corrections and adjustments', 'Data Team', TRUE),
('Email Campaign Platform', 'api', 'Email marketing metrics from Mailchimp', 'Marketing Team', TRUE),
('Social Media API', 'api', 'Social media engagement metrics', 'Marketing Team', TRUE);

-- ========================================
-- SECTION 3: INSERT DATA ENTITIES
-- ========================================

INSERT INTO data_entities (entity_name, entity_type, source_id, schema_name, description, refresh_frequency) VALUES
-- Source Tables
('customers', 'table', 1, 'ecommerce_analytics', 'Customer master data', 'real_time'),
('products', 'table', 1, 'ecommerce_analytics', 'Product catalog', 'daily'),
('orders', 'table', 1, 'ecommerce_analytics', 'Order transactions', 'real_time'),
('order_items', 'table', 1, 'ecommerce_analytics', 'Order line items', 'real_time'),
('inventory', 'table', 1, 'ecommerce_analytics', 'Inventory levels', 'hourly'),
('vendors', 'table', 1, 'ecommerce_analytics', 'Vendor information', 'daily'),
('campaigns', 'table', 1, 'ecommerce_analytics', 'Marketing campaigns', 'daily'),
('reviews', 'table', 1, 'ecommerce_analytics', 'Product reviews', 'daily'),
('returns', 'table', 1, 'ecommerce_analytics', 'Return transactions', 'real_time'),

-- Staging Tables
('stg_salesforce_leads', 'staging', 2, 'ecommerce_analytics', 'Staging table for Salesforce lead data', 'hourly'),
('stg_ga_sessions', 'staging', 3, 'ecommerce_analytics', 'Staging table for Google Analytics sessions', 'daily'),
('stg_payment_transactions', 'staging', 4, 'ecommerce_analytics', 'Staging table for payment data', 'hourly'),

-- Fact Tables
('fact_daily_sales', 'fact', 1, 'ecommerce_analytics', 'Daily aggregated sales metrics', 'daily'),
('fact_customer_lifetime_value', 'fact', 1, 'ecommerce_analytics', 'Customer lifetime value calculations', 'weekly'),
('fact_product_performance', 'fact', 1, 'ecommerce_analytics', 'Product sales performance metrics', 'daily'),
('fact_campaign_roi', 'fact', 1, 'ecommerce_analytics', 'Campaign return on investment', 'daily'),

-- Dimension Tables
('dim_customer_segments', 'dimension', 1, 'ecommerce_analytics', 'Customer segmentation', 'weekly'),
('dim_product_categories', 'dimension', 1, 'ecommerce_analytics', 'Product category hierarchy', 'daily'),
('dim_date', 'dimension', 1, 'ecommerce_analytics', 'Date dimension table', 'monthly'),

-- Analytics Views
('vw_customer_purchase_history', 'view', 1, 'ecommerce_analytics', 'Customer purchase history view', 'real_time'),
('vw_inventory_status', 'view', 1, 'ecommerce_analytics', 'Current inventory status', 'real_time'),
('vw_top_selling_products', 'materialized_view', 1, 'ecommerce_analytics', 'Top selling products', 'hourly');

-- ========================================
-- SECTION 4: INSERT TRANSFORMATIONS
-- ========================================

INSERT INTO data_transformations (transformation_name, transformation_type, source_entity_id, target_entity_id, transformation_logic, execution_order, is_active, created_by) VALUES
-- ETL Transformations
('Load Salesforce Leads', 'etl', 10, 1, 'Extract leads from Salesforce API, transform to customer format, load to customers table', 1, TRUE, 'ETL_Service'),
('Load GA Sessions', 'etl', 11, 1, 'Import Google Analytics session data and match to customers', 2, TRUE, 'ETL_Service'),
('Load Payment Data', 'etl', 12, 3, 'Import payment transactions and reconcile with orders', 3, TRUE, 'ETL_Service'),

-- Aggregation Transformations
('Daily Sales Aggregation', 'aggregation', 3, 13, 'Aggregate order data by day, product, and customer segment', 4, TRUE, 'Analytics_Service'),
('Customer LTV Calculation', 'aggregation', 3, 14, 'Calculate customer lifetime value based on historical orders', 5, TRUE, 'Analytics_Service'),
('Product Performance Metrics', 'aggregation', 4, 15, 'Calculate product-level performance metrics including revenue, units sold, and returns', 6, TRUE, 'Analytics_Service'),

-- Join Transformations
('Customer Purchase History', 'join', 1, 20, 'Join customers with orders and order items', 7, TRUE, 'Analytics_Service'),
('Inventory Status View', 'join', 5, 21, 'Join inventory with products for current status', 8, TRUE, 'Analytics_Service'),

-- Enrichment Transformations
('Customer Segmentation', 'enrichment', 1, 17, 'Enrich customers with RFM segmentation', 9, TRUE, 'ML_Service'),
('Campaign ROI Calculation', 'calculation', 7, 16, 'Calculate ROI for marketing campaigns based on attributed orders', 10, TRUE, 'Analytics_Service'),

-- Cleaning Transformations
('Clean Product Data', 'cleaning', 2, 2, 'Standardize product names, remove duplicates, validate prices', 11, TRUE, 'Data_Quality_Service'),
('Validate Customer Emails', 'cleaning', 1, 1, 'Validate and standardize customer email addresses', 12, TRUE, 'Data_Quality_Service');

-- ========================================
-- SECTION 5: INSERT DATA LINEAGE
-- ========================================

INSERT INTO data_lineage (source_entity_id, target_entity_id, transformation_id, lineage_type, dependency_level) VALUES
-- Direct Lineage
(10, 1, 1, 'direct', 1),  -- Salesforce to Customers
(11, 1, 2, 'direct', 1),  -- GA to Customers
(12, 3, 3, 'direct', 1),  -- Payment to Orders

-- Derived Lineage
(3, 13, 4, 'aggregated', 1),  -- Orders to Daily Sales
(3, 14, 5, 'aggregated', 1),  -- Orders to Customer LTV
(4, 15, 6, 'aggregated', 1),  -- Order Items to Product Performance
(1, 17, 9, 'derived', 1),     -- Customers to Customer Segments
(7, 16, 10, 'derived', 1),    -- Campaigns to Campaign ROI

-- Joined Lineage
(1, 20, 7, 'joined', 2),  -- Customers to Purchase History
(3, 20, 7, 'joined', 2),  -- Orders to Purchase History
(4, 20, 7, 'joined', 2),  -- Order Items to Purchase History
(5, 21, 8, 'joined', 2),  -- Inventory to Inventory Status
(2, 21, 8, 'joined', 2),  -- Products to Inventory Status

-- Multi-level Dependencies
(1, 13, 4, 'derived', 2),  -- Customers to Daily Sales (through orders)
(2, 13, 4, 'derived', 2),  -- Products to Daily Sales (through orders)
(1, 16, 10, 'derived', 2), -- Customers to Campaign ROI (through orders)
(3, 16, 10, 'derived', 2); -- Orders to Campaign ROI

-- ========================================
-- SECTION 6: DATA LINEAGE QUERIES
-- ========================================

-- 6.1 Complete Data Lineage for an Entity
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS sp_get_data_lineage(IN p_entity_name VARCHAR(100))
BEGIN
    WITH RECURSIVE lineage_tree AS (
        -- Base case: direct sources
        SELECT 
            dl.lineage_id,
            ds.entity_name AS source_entity,
            dt.entity_name AS target_entity,
            dtr.transformation_name,
            dl.lineage_type,
            dl.dependency_level,
            1 AS depth
        FROM data_lineage dl
        JOIN data_entities ds ON dl.source_entity_id = ds.entity_id
        JOIN data_entities dt ON dl.target_entity_id = dt.entity_id
        LEFT JOIN data_transformations dtr ON dl.transformation_id = dtr.transformation_id
        WHERE dt.entity_name = p_entity_name
        
        UNION ALL
        
        -- Recursive case: upstream sources
        SELECT 
            dl.lineage_id,
            ds.entity_name,
            dt.entity_name,
            dtr.transformation_name,
            dl.lineage_type,
            dl.dependency_level,
            lt.depth + 1
        FROM data_lineage dl
        JOIN data_entities ds ON dl.source_entity_id = ds.entity_id
        JOIN data_entities dt ON dl.target_entity_id = dt.entity_id
        LEFT JOIN data_transformations dtr ON dl.transformation_id = dtr.transformation_id
        JOIN lineage_tree lt ON dt.entity_name = lt.source_entity
        WHERE lt.depth < 10  -- Prevent infinite recursion
    )
    SELECT DISTINCT
        depth,
        source_entity,
        target_entity,
        transformation_name,
        lineage_type,
        dependency_level
    FROM lineage_tree
    ORDER BY depth, source_entity;
END //
DELIMITER ;

-- 6.2 Downstream Impact Analysis
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS sp_get_downstream_impact(IN p_entity_name VARCHAR(100))
BEGIN
    WITH RECURSIVE downstream_tree AS (
        -- Base case: direct targets
        SELECT 
            dl.lineage_id,
            ds.entity_name AS source_entity,
            dt.entity_name AS target_entity,
            dt.entity_type,
            dtr.transformation_name,
            dl.lineage_type,
            1 AS depth
        FROM data_lineage dl
        JOIN data_entities ds ON dl.source_entity_id = ds.entity_id
        JOIN data_entities dt ON dl.target_entity_id = dt.entity_id
        LEFT JOIN data_transformations dtr ON dl.transformation_id = dtr.transformation_id
        WHERE ds.entity_name = p_entity_name
        
        UNION ALL
        
        -- Recursive case: downstream targets
        SELECT 
            dl.lineage_id,
            ds.entity_name,
            dt.entity_name,
            dt.entity_type,
            dtr.transformation_name,
            dl.lineage_type,
            dst.depth + 1
        FROM data_lineage dl
        JOIN data_entities ds ON dl.source_entity_id = ds.entity_id
        JOIN data_entities dt ON dl.target_entity_id = dt.entity_id
        LEFT JOIN data_transformations dtr ON dl.transformation_id = dtr.transformation_id
        JOIN downstream_tree dst ON ds.entity_name = dst.target_entity
        WHERE dst.depth < 10
    )
    SELECT DISTINCT
        depth,
        source_entity,
        target_entity,
        entity_type,
        transformation_name,
        lineage_type,
        COUNT(*) OVER (PARTITION BY depth) AS entities_at_level
    FROM downstream_tree
    ORDER BY depth, target_entity;
END //
DELIMITER ;

-- 6.3 Data Flow Visualization Query
SELECT 
    CONCAT(REPEAT('  ', dl.dependency_level - 1), '└─> ', dt.entity_name) AS data_flow,
    ds.entity_name AS source,
    dt.entity_name AS target,
    dtr.transformation_name AS transformation,
    dl.lineage_type,
    dsrc.source_type AS source_type,
    dt.refresh_frequency
FROM data_lineage dl
JOIN data_entities ds ON dl.source_entity_id = ds.entity_id
JOIN data_entities dt ON dl.target_entity_id = dt.entity_id
LEFT JOIN data_transformations dtr ON dl.transformation_id = dtr.transformation_id
LEFT JOIN data_sources dsrc ON ds.source_id = dsrc.source_id
ORDER BY dl.dependency_level, dt.entity_name;

-- 6.4 Transformation Execution Summary
SELECT 
    dtr.transformation_name,
    dtr.transformation_type,
    ds.entity_name AS source_entity,
    dt.entity_name AS target_entity,
    COUNT(tel.execution_id) AS total_executions,
    SUM(CASE WHEN tel.execution_status = 'success' THEN 1 ELSE 0 END) AS successful_executions,
    SUM(CASE WHEN tel.execution_status = 'failed' THEN 1 ELSE 0 END) AS failed_executions,
    ROUND(AVG(tel.execution_duration_seconds), 2) AS avg_duration_seconds,
    MAX(tel.execution_start) AS last_execution,
    SUM(tel.rows_processed) AS total_rows_processed
FROM data_transformations dtr
LEFT JOIN data_entities ds ON dtr.source_entity_id = ds.entity_id
LEFT JOIN data_entities dt ON dtr.target_entity_id = dt.entity_id
LEFT JOIN transformation_execution_log tel ON dtr.transformation_id = tel.transformation_id
WHERE dtr.is_active = TRUE
GROUP BY dtr.transformation_id, dtr.transformation_name, dtr.transformation_type, ds.entity_name, dt.entity_name
ORDER BY last_execution DESC;

-- 6.5 Data Source Dependency Matrix
SELECT 
    dsrc.source_name,
    dsrc.source_type,
    COUNT(DISTINCT de.entity_id) AS entities_count,
    COUNT(DISTINCT dl.lineage_id) AS downstream_dependencies,
    GROUP_CONCAT(DISTINCT de.entity_name ORDER BY de.entity_name SEPARATOR ', ') AS entities
FROM data_sources dsrc
JOIN data_entities de ON dsrc.source_id = de.source_id
LEFT JOIN data_lineage dl ON de.entity_id = dl.source_entity_id
WHERE dsrc.is_active = TRUE
GROUP BY dsrc.source_id, dsrc.source_name, dsrc.source_type
ORDER BY downstream_dependencies DESC;

-- ========================================
-- SECTION 7: DATA QUALITY LINEAGE
-- ========================================

-- 7.1 Quality Check Results by Entity
SELECT 
    de.entity_name,
    de.entity_type,
    dqc.checkpoint_name,
    dqc.rule_type,
    dqc.threshold_value,
    dqc.actual_value,
    dqc.check_status,
    dqc.checked_at,
    CASE 
        WHEN dqc.actual_value >= dqc.threshold_value THEN '✓ PASS'
        WHEN dqc.actual_value >= dqc.threshold_value * 0.9 THEN '⚠ WARNING'
        ELSE '✗ FAIL'
    END AS quality_indicator
FROM data_entities de
LEFT JOIN data_quality_checkpoints dqc ON de.entity_id = dqc.entity_id
WHERE dqc.checked_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY de.entity_name, dqc.checked_at DESC;

-- 7.2 Failed Quality Checks with Impact
SELECT 
    de.entity_name AS failing_entity,
    dqc.checkpoint_name,
    dqc.check_status,
    dqc.actual_value,
    dqc.threshold_value,
    COUNT(DISTINCT dl.target_entity_id) AS affected_downstream_entities,
    GROUP_CONCAT(DISTINCT dt.entity_name SEPARATOR ', ') AS downstream_entities
FROM data_quality_checkpoints dqc
JOIN data_entities de ON dqc.entity_id = de.entity_id
LEFT JOIN data_lineage dl ON de.entity_id = dl.source_entity_id
LEFT JOIN data_entities dt ON dl.target_entity_id = dt.entity_id
WHERE dqc.check_status = 'fail'
    AND dqc.checked_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY de.entity_id, de.entity_name, dqc.checkpoint_id, dqc.checkpoint_name, dqc.check_status, dqc.actual_value, dqc.threshold_value
ORDER BY affected_downstream_entities DESC;

-- ========================================
-- SECTION 8: DATA REFRESH TIMELINE
-- ========================================

SELECT 
    de.entity_name,
    de.entity_type,
    de.refresh_frequency,
    de.last_refresh_date,
    TIMESTAMPDIFF(HOUR, de.last_refresh_date, NOW()) AS hours_since_refresh,
    CASE de.refresh_frequency
        WHEN 'real_time' THEN 1
        WHEN 'hourly' THEN 1
        WHEN 'daily' THEN 24
        WHEN 'weekly' THEN 168
        WHEN 'monthly' THEN 720
        ELSE 9999
    END AS expected_refresh_hours,
    CASE 
        WHEN TIMESTAMPDIFF(HOUR, de.last_refresh_date, NOW()) <= 
            CASE de.refresh_frequency
                WHEN 'real_time' THEN 1
                WHEN 'hourly' THEN 1
                WHEN 'daily' THEN 24
                WHEN 'weekly' THEN 168
                WHEN 'monthly' THEN 720
                ELSE 9999
            END THEN '✓ Current'
        WHEN TIMESTAMPDIFF(HOUR, de.last_refresh_date, NOW()) <= 
            CASE de.refresh_frequency
                WHEN 'real_time' THEN 2
                WHEN 'hourly' THEN 2
                WHEN 'daily' THEN 30
                WHEN 'weekly' THEN 200
                WHEN 'monthly' THEN 800
                ELSE 9999
            END THEN '⚠ Delayed'
        ELSE '✗ Stale'
    END AS refresh_status
FROM data_entities de
ORDER BY hours_since_refresh DESC;

-- ========================================
-- SECTION 9: COMPLETE LINEAGE REPORT
-- ========================================

SELECT 
    'LINEAGE SUMMARY' AS report_section,
    COUNT(DISTINCT ds.source_id) AS total_data_sources,
    COUNT(DISTINCT de.entity_id) AS total_entities,
    COUNT(DISTINCT dtr.transformation_id) AS total_transformations,
    COUNT(DISTINCT dl.lineage_id) AS total_lineage_connections,
    COUNT(DISTINCT CASE WHEN de.entity_type = 'table' THEN de.entity_id END) AS source_tables,
    COUNT(DISTINCT CASE WHEN de.entity_type = 'fact' THEN de.entity_id END) AS fact_tables,
    COUNT(DISTINCT CASE WHEN de.entity_type = 'dimension' THEN de.entity_id END) AS dimension_tables,
    COUNT(DISTINCT CASE WHEN de.entity_type = 'view' THEN de.entity_id END) AS views
FROM data_sources ds
CROSS JOIN data_entities de
CROSS JOIN data_transformations dtr
CROSS JOIN data_lineage dl;

-- Display completion message
SELECT 
    'Data Lineage System Initialized' AS Status,
    'All tables, procedures, and sample data created successfully' AS Message;