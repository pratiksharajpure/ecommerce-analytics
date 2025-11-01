-- ========================================
-- QUALITY THRESHOLDS & SCORING SYSTEM
-- E-commerce Revenue Analytics Engine
-- Quality Scores, Acceptable Ranges, Warning Levels
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for procedure creation
DELIMITER //

-- ========================================
-- QUALITY THRESHOLD TABLES
-- ========================================

-- Quality Threshold Definitions
CREATE TABLE IF NOT EXISTS quality_thresholds (
    threshold_id INT PRIMARY KEY AUTO_INCREMENT,
    metric_name VARCHAR(200) NOT NULL UNIQUE,
    metric_category ENUM('product', 'vendor', 'customer', 'financial', 'operational', 'marketing') NOT NULL,
    description TEXT,
    -- Threshold levels
    excellent_min DECIMAL(10,2),
    good_min DECIMAL(10,2),
    acceptable_min DECIMAL(10,2),
    warning_max DECIMAL(10,2),
    critical_max DECIMAL(10,2),
    -- Target and tolerance
    target_value DECIMAL(10,2),
    tolerance_percentage DECIMAL(5,2) DEFAULT 10.00,
    -- Measurement details
    measurement_unit VARCHAR(50),
    higher_is_better BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    alert_on_breach BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_metric_category (metric_category),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Quality Score History
CREATE TABLE IF NOT EXISTS quality_score_history (
    score_id INT PRIMARY KEY AUTO_INCREMENT,
    threshold_id INT NOT NULL,
    entity_type VARCHAR(50),
    entity_id INT,
    measured_value DECIMAL(10,2),
    quality_score DECIMAL(5,2),
    quality_level ENUM('excellent', 'good', 'acceptable', 'warning', 'critical') DEFAULT 'acceptable',
    deviation_from_target DECIMAL(10,2),
    deviation_percentage DECIMAL(10,2),
    measurement_date DATE,
    measurement_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (threshold_id) REFERENCES quality_thresholds(threshold_id) ON DELETE CASCADE,
    INDEX idx_threshold_id (threshold_id),
    INDEX idx_entity_type (entity_type),
    INDEX idx_entity_id (entity_id),
    INDEX idx_quality_level (quality_level),
    INDEX idx_measurement_date (measurement_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Quality Violations Log
CREATE TABLE IF NOT EXISTS quality_violations (
    violation_id INT PRIMARY KEY AUTO_INCREMENT,
    threshold_id INT NOT NULL,
    entity_type VARCHAR(50),
    entity_id INT,
    violation_type ENUM('below_minimum', 'above_maximum', 'out_of_tolerance') NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    measured_value DECIMAL(10,2),
    threshold_value DECIMAL(10,2),
    variance DECIMAL(10,2),
    violation_message TEXT,
    status ENUM('open', 'acknowledged', 'resolved', 'false_positive') DEFAULT 'open',
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP NULL,
    resolved_at TIMESTAMP NULL,
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (threshold_id) REFERENCES quality_thresholds(threshold_id) ON DELETE CASCADE,
    INDEX idx_threshold_id (threshold_id),
    INDEX idx_severity (severity),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- ========================================
-- QUALITY THRESHOLD DEFINITIONS
-- Insert default quality thresholds
-- ========================================

-- Product Quality Thresholds
INSERT INTO quality_thresholds (metric_name, metric_category, description, excellent_min, good_min, acceptable_min, warning_max, critical_max, target_value, measurement_unit, higher_is_better) VALUES
('Product Return Rate', 'product', 'Percentage of products returned', NULL, NULL, 5.00, 10.00, 15.00, 3.00, 'percentage', FALSE),
('Product Average Rating', 'product', 'Average customer rating (1-5 stars)', 4.50, 4.00, 3.50, 3.00, 2.50, 4.80, 'stars', TRUE),
('Product Review Count', 'product', 'Number of customer reviews', 50, 25, 10, 5, 0, 100, 'count', TRUE),
('Product Stock Accuracy', 'product', 'Inventory accuracy percentage', 99.00, 97.00, 95.00, 93.00, 90.00, 99.50, 'percentage', TRUE),
('Product Profit Margin', 'product', 'Gross profit margin percentage', 40.00, 30.00, 20.00, 15.00, 10.00, 45.00, 'percentage', TRUE),
('Product Defect Rate', 'product', 'Percentage of defective products', NULL, NULL, 1.00, 2.00, 5.00, 0.50, 'percentage', FALSE),

-- Vendor Quality Thresholds
('Vendor Rating', 'vendor', 'Vendor performance rating', 4.50, 4.00, 3.50, 3.00, 2.50, 4.80, 'rating', TRUE),
('Vendor On-Time Delivery', 'vendor', 'Percentage of on-time deliveries', 98.00, 95.00, 90.00, 85.00, 80.00, 99.00, 'percentage', TRUE),
('Vendor Quality Score', 'vendor', 'Overall vendor quality score', 90.00, 80.00, 70.00, 60.00, 50.00, 95.00, 'score', TRUE),
('Vendor Defect Rate', 'vendor', 'Percentage of defective items from vendor', NULL, NULL, 1.00, 2.00, 5.00, 0.50, 'percentage', FALSE),

-- Customer Quality Thresholds
('Customer Satisfaction Score', 'customer', 'CSAT score (1-100)', 90.00, 80.00, 70.00, 60.00, 50.00, 95.00, 'score', TRUE),
('Customer Churn Rate', 'customer', 'Percentage of customers churning', NULL, NULL, 5.00, 10.00, 15.00, 3.00, 'percentage', FALSE),
('Customer Lifetime Value', 'customer', 'Average CLV in dollars', 5000.00, 3000.00, 1000.00, 500.00, 0, 10000.00, 'dollars', TRUE),
('Customer Complaint Rate', 'customer', 'Complaints per 100 orders', NULL, NULL, 2.00, 5.00, 10.00, 1.00, 'rate', FALSE),

-- Financial Quality Thresholds
('Payment Success Rate', 'financial', 'Percentage of successful payments', 98.00, 95.00, 92.00, 90.00, 85.00, 99.00, 'percentage', TRUE),
('Order Fulfillment Rate', 'financial', 'Percentage of orders fulfilled', 99.00, 97.00, 95.00, 93.00, 90.00, 99.50, 'percentage', TRUE),
('Revenue Growth Rate', 'financial', 'Month-over-month revenue growth', 20.00, 10.00, 5.00, 0, -5.00, 25.00, 'percentage', TRUE),
('Gross Margin Percentage', 'financial', 'Overall gross profit margin', 45.00, 35.00, 25.00, 20.00, 15.00, 50.00, 'percentage', TRUE),
('Refund Rate', 'financial', 'Percentage of orders refunded', NULL, NULL, 3.00, 5.00, 10.00, 2.00, 'percentage', FALSE),

-- Operational Quality Thresholds
('Order Processing Time', 'operational', 'Average hours to process order', NULL, NULL, 24.00, 48.00, 72.00, 12.00, 'hours', FALSE),
('Shipping Accuracy', 'operational', 'Percentage of accurate shipments', 99.00, 97.00, 95.00, 93.00, 90.00, 99.50, 'percentage', TRUE),
('Inventory Turnover Ratio', 'operational', 'Times inventory turns per year', 12.00, 8.00, 6.00, 4.00, 2.00, 15.00, 'ratio', TRUE),
('Out of Stock Rate', 'operational', 'Percentage of products out of stock', NULL, NULL, 2.00, 5.00, 10.00, 1.00, 'percentage', FALSE),

-- Marketing Quality Thresholds
('Campaign ROI', 'marketing', 'Return on investment percentage', 300.00, 200.00, 150.00, 100.00, 50.00, 400.00, 'percentage', TRUE),
('Customer Acquisition Cost', 'marketing', 'Cost to acquire one customer', NULL, NULL, 50.00, 75.00, 100.00, 30.00, 'dollars', FALSE),
('Conversion Rate', 'marketing', 'Percentage of visitors converting', 5.00, 3.00, 2.00, 1.00, 0.50, 6.00, 'percentage', TRUE),
('Click-Through Rate', 'marketing', 'Campaign click-through rate', 5.00, 3.00, 2.00, 1.00, 0.50, 7.00, 'percentage', TRUE),
('Email Open Rate', 'marketing', 'Email campaign open rate', 30.00, 20.00, 15.00, 10.00, 5.00, 35.00, 'percentage', TRUE)//

-- ========================================
-- QUALITY ASSESSMENT FUNCTIONS
-- ========================================

-- Function to calculate quality level
CREATE FUNCTION fn_get_quality_level(
    p_measured_value DECIMAL(10,2),
    p_excellent_min DECIMAL(10,2),
    p_good_min DECIMAL(10,2),
    p_acceptable_min DECIMAL(10,2),
    p_warning_max DECIMAL(10,2),
    p_critical_max DECIMAL(10,2),
    p_higher_is_better BOOLEAN
) RETURNS VARCHAR(20)
DETERMINISTIC
BEGIN
    DECLARE v_quality_level VARCHAR(20);
    
    IF p_higher_is_better THEN
        -- Higher values are better
        IF p_measured_value >= p_excellent_min THEN
            SET v_quality_level = 'excellent';
        ELSEIF p_measured_value >= p_good_min THEN
            SET v_quality_level = 'good';
        ELSEIF p_measured_value >= p_acceptable_min THEN
            SET v_quality_level = 'acceptable';
        ELSEIF p_measured_value >= p_warning_max THEN
            SET v_quality_level = 'warning';
        ELSE
            SET v_quality_level = 'critical';
        END IF;
    ELSE
        -- Lower values are better
        IF p_measured_value <= p_excellent_min OR p_excellent_min IS NULL THEN
            IF p_acceptable_min IS NOT NULL AND p_measured_value <= p_acceptable_min THEN
                SET v_quality_level = 'excellent';
            ELSEIF p_warning_max IS NOT NULL AND p_measured_value <= p_warning_max THEN
                SET v_quality_level = 'good';
            ELSE
                SET v_quality_level = 'acceptable';
            END IF;
        ELSEIF p_warning_max IS NOT NULL AND p_measured_value <= p_warning_max THEN
            SET v_quality_level = 'warning';
        ELSEIF p_critical_max IS NOT NULL AND p_measured_value <= p_critical_max THEN
            SET v_quality_level = 'critical';
        ELSE
            SET v_quality_level = 'critical';
        END IF;
    END IF;
    
    RETURN v_quality_level;
END//

-- Function to calculate quality score (0-100)
CREATE FUNCTION fn_calculate_quality_score(
    p_measured_value DECIMAL(10,2),
    p_target_value DECIMAL(10,2),
    p_tolerance_pct DECIMAL(5,2),
    p_higher_is_better BOOLEAN
) RETURNS DECIMAL(5,2)
DETERMINISTIC
BEGIN
    DECLARE v_score DECIMAL(5,2);
    DECLARE v_deviation_pct DECIMAL(10,2);
    
    -- Calculate deviation percentage
    SET v_deviation_pct = ABS((p_measured_value - p_target_value) / NULLIF(p_target_value, 0)) * 100;
    
    -- Calculate score based on deviation
    IF v_deviation_pct <= p_tolerance_pct THEN
        -- Within tolerance: score 90-100
        SET v_score = 100 - (v_deviation_pct / p_tolerance_pct * 10);
    ELSE
        -- Outside tolerance: score decreases more rapidly
        SET v_score = 90 - LEAST((v_deviation_pct - p_tolerance_pct), 90);
    END IF;
    
    -- Ensure score is between 0 and 100
    SET v_score = GREATEST(0, LEAST(100, v_score));
    
    RETURN v_score;
END//

-- ========================================
-- QUALITY MEASUREMENT PROCEDURES
-- ========================================

-- Procedure to measure and record quality metric
CREATE PROCEDURE sp_record_quality_metric(
    IN p_metric_name VARCHAR(200),
    IN p_entity_type VARCHAR(50),
    IN p_entity_id INT,
    IN p_measured_value DECIMAL(10,2)
)
BEGIN
DECLARE v_threshold_id INT;
    DECLARE v_target_value DECIMAL(10,2);
    DECLARE v_tolerance_pct DECIMAL(5,2);
    DECLARE v_higher_is_better BOOLEAN;
    DECLARE v_excellent_min DECIMAL(10,2);
    DECLARE v_good_min DECIMAL(10,2);
    DECLARE v_acceptable_min DECIMAL(10,2);
    DECLARE v_warning_max DECIMAL(10,2);
    DECLARE v_critical_max DECIMAL(10,2);
    DECLARE v_quality_level VARCHAR(20);
    DECLARE v_quality_score DECIMAL(5,2);
    DECLARE v_deviation DECIMAL(10,2);
    DECLARE v_deviation_pct DECIMAL(10,2);
    
    -- Get threshold configuration
    SELECT 
        threshold_id,
        target_value,
        tolerance_percentage,
        higher_is_better,
        excellent_min,
        good_min,
        acceptable_min,
        warning_max,
        critical_max
    INTO 
        v_threshold_id,
        v_target_value,
        v_tolerance_pct,
        v_higher_is_better,
        v_excellent_min,
        v_good_min,
        v_acceptable_min,
        v_warning_max,
        v_critical_max
    FROM quality_thresholds
    WHERE metric_name = p_metric_name
        AND is_active = TRUE;
    
    -- Calculate quality level
    

    












SET v_quality_level = fn_get_quality_level(
        p_measured_value,
        v_excellent_min,
        v_good_min,
        v_acceptable_min,
        v_warning_max,
        v_critical_max,
        v_higher_is_better
    );
    
    -- Calculate quality score
    SET v_quality_score = fn_calculate_quality_score(
        p_measured_value,
        v_target_value,
        v_tolerance_pct,
        v_higher_is_better
    );
    
    -- Calculate deviation
    SET v_deviation = p_measured_value - v_target_value;
    SET v_deviation_pct = (v_deviation / NULLIF(v_target_value, 0)) * 100;
    
    -- Record quality score
    INSERT INTO quality_score_history (
        threshold_id,
        entity_type,
        entity_id,
        measured_value,
        quality_score,
        quality_level,
        deviation_from_target,
        deviation_percentage,
        measurement_date
    ) VALUES (
        v_threshold_id,
        p_entity_type,
        p_entity_id,
        p_measured_value,
        v_quality_score,
        v_quality_level,
        v_deviation,
        v_deviation_pct,
        CURDATE()
    );
    
    -- Log violation if quality is warning or critical
    IF v_quality_level IN ('warning', 'critical') THEN
        INSERT INTO quality_violations (
            threshold_id,
            entity_type,
            entity_id,
            violation_type,
            severity,
            measured_value,
            threshold_value,
            variance,
            violation_message,
            status
        ) VALUES (
            v_threshold_id,
            p_entity_type,
            p_entity_id,
            CASE 
                WHEN v_higher_is_better AND p_measured_value < v_acceptable_min THEN 'below_minimum'
                WHEN NOT v_higher_is_better AND p_measured_value > v_warning_max THEN 'above_maximum'
                ELSE 'out_of_tolerance'
            END,
            CASE 
                WHEN v_quality_level = 'critical' THEN 'critical'
                ELSE 'high'
            END,
            p_measured_value,
            v_target_value,
            v_deviation,
            CONCAT(p_metric_name, ' is at ', v_quality_level, ' level: ',
                   p_measured_value, ' (target: ', v_target_value, ')'),
            'open'
        );
    END IF;
END//

-- ========================================
-- COMPREHENSIVE QUALITY ASSESSMENT
-- ========================================

-- Procedure to assess product quality
CREATE PROCEDURE sp_assess_product_quality(
    IN p_product_id INT
)
BEGIN
DECLARE v_return_rate DECIMAL(10,2);
    DECLARE v_avg_rating DECIMAL(10,2);
    DECLARE v_review_count INT;
    DECLARE v_profit_margin DECIMAL(10,2);
    DECLARE v_defect_rate DECIMAL(10,2);
    
    -- Calculate return rate (last 90 days)
    SELECT 
        COALESCE(
            (COUNT(DISTINCT r.return_id) / NULLIF(COUNT(DISTINCT oi.order_id), 0)) * 100,
            0
        )
    INTO v_return_rate
    FROM order_items oi
    LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
        AND r.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
    WHERE oi.product_id = p_product_id
        AND oi.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY);
    
    -- Get average rating and review count
    SELECT 
        COALESCE(AVG(rating), 0),
        COUNT(*)
    INTO v_avg_rating, v_review_count
    FROM reviews
    WHERE product_id = p_product_id
        AND status = 'approved';
    
    -- Calculate profit margin
    SELECT 
        COALESCE(((price - cost) / NULLIF(price, 0)) * 100, 0)
    INTO v_profit_margin
    FROM products
    WHERE product_id = p_product_id;
    
    -- Calculate defect rate (returns due to defects)
    SELECT 
        COALESCE(
            (COUNT(DISTINCT CASE WHEN r.reason = 'defective' THEN r.return_id END) / 
             NULLIF(COUNT(DISTINCT oi.order_id), 0)) * 100,
            0
        )
    INTO v_defect_rate
    FROM order_items oi
    LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
        AND r.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
    WHERE oi.product_id = p_product_id
        AND oi.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY);
    
    -- Record metrics
    CALL sp_record_quality_metric('Product Return Rate', 'product', p_product_id, v_return_rate);
    CALL sp_record_quality_metric('Product Average Rating', 'product', p_product_id, v_avg_rating);
    CALL sp_record_quality_metric('Product Review Count', 'product', p_product_id, v_review_count);
    CALL sp_record_quality_metric('Product Profit Margin', 'product', p_product_id, v_profit_margin);
    CALL sp_record_quality_metric('Product Defect Rate', 'product', p_product_id, v_defect_rate);
    
    -- Return summary
    SELECT 
        p_product_id AS product_id,
        v_return_rate AS return_rate,
        v_avg_rating AS avg_rating,
        v_review_count AS review_count,
        v_profit_margin AS profit_margin,
        v_defect_rate AS defect_rate;
END//

-- Procedure to assess vendor quality
CREATE PROCEDURE sp_assess_vendor_quality(
    IN p_vendor_id INT
)
BEGIN
    DECLARE v_vendor_rating DECIMAL(10,2);
    DECLARE v_quality_score DECIMAL(10,2);
    DECLARE v_defect_rate DECIMAL(10,2);
    
    -- Get vendor rating
    SELECT rating INTO v_vendor_rating
    FROM vendors
    WHERE vendor_id = p_vendor_id;
    
    -- Calculate quality score (simplified)
    

    












SET v_quality_score = v_vendor_rating * 20; -- Convert 5-star to 100-point scale
    
    -- Calculate defect rate from returns
    SELECT 
        COALESCE(
            (COUNT(DISTINCT CASE WHEN r.reason = 'defective' THEN r.return_id END) / 
             NULLIF(COUNT(DISTINCT oi.order_id), 0)) * 100,
            0
        )
    INTO v_defect_rate
    FROM vendor_contracts vc
    JOIN order_items oi ON vc.product_id = oi.product_id
    LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
        AND r.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
    WHERE vc.vendor_id = p_vendor_id
        AND vc.status = 'active'
        AND oi.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY);
    
    -- Record metrics
    CALL sp_record_quality_metric('Vendor Rating', 'vendor', p_vendor_id, v_vendor_rating);
    CALL sp_record_quality_metric('Vendor Quality Score', 'vendor', p_vendor_id, v_quality_score);
    CALL sp_record_quality_metric('Vendor Defect Rate', 'vendor', p_vendor_id, v_defect_rate);
    
    -- Return summary
    SELECT 
        p_vendor_id AS vendor_id,
        v_vendor_rating AS vendor_rating,
        v_quality_score AS quality_score,
        v_defect_rate AS defect_rate;
END//

-- Procedure to assess financial quality
CREATE PROCEDURE sp_assess_financial_quality()
BEGIN
DECLARE v_payment_success_rate DECIMAL(10,2);
    DECLARE v_fulfillment_rate DECIMAL(10,2);
    DECLARE v_revenue_growth DECIMAL(10,2);
    DECLARE v_gross_margin DECIMAL(10,2);
    DECLARE v_refund_rate DECIMAL(10,2);
    DECLARE v_current_month_revenue DECIMAL(15,2);
    DECLARE v_previous_month_revenue DECIMAL(15,2);
    
    -- Payment success rate (last 30 days)
    SELECT 
        (SUM(CASE WHEN payment_status = 'paid' THEN 1 ELSE 0 END) / 
         NULLIF(COUNT(*), 0)) * 100
    INTO v_payment_success_rate
    FROM orders
    WHERE order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY);
    
    -- Order fulfillment rate
    SELECT 
        (SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) / 
         NULLIF(COUNT(*), 0)) * 100
    INTO v_fulfillment_rate
    FROM orders
    WHERE order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY);
    
    -- Revenue growth rate
    SELECT SUM(total_amount) INTO v_current_month_revenue
    FROM orders
    WHERE payment_status = 'paid'
        AND order_date >= DATE_FORMAT(NOW(), '%Y-%m-01');
    
    SELECT SUM(total_amount) INTO v_previous_month_revenue
    FROM orders
    WHERE payment_status = 'paid'
        AND order_date >= DATE_FORMAT(DATE_SUB(NOW(), INTERVAL 1 MONTH), '%Y-%m-01')
        AND order_date < DATE_FORMAT(NOW(), '%Y-%m-01');
    
    

    












SET v_revenue_growth = ((v_current_month_revenue - v_previous_month_revenue) / 
                            NULLIF(v_previous_month_revenue, 0)) * 100;
    
    -- Gross margin percentage
    SELECT 
        ((SUM(oi.subtotal) - SUM(oi.quantity * p.cost)) / 
         NULLIF(SUM(oi.subtotal), 0)) * 100
    INTO v_gross_margin
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY);
    
    -- Refund rate
    SELECT 
        (SUM(CASE WHEN payment_status = 'refunded' THEN 1 ELSE 0 END) / 
         NULLIF(COUNT(*), 0)) * 100
    INTO v_refund_rate
    FROM orders
    WHERE order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY);
    
    -- Record metrics
    CALL sp_record_quality_metric('Payment Success Rate', 'financial', NULL, v_payment_success_rate);
    CALL sp_record_quality_metric('Order Fulfillment Rate', 'financial', NULL, v_fulfillment_rate);
    CALL sp_record_quality_metric('Revenue Growth Rate', 'financial', NULL, v_revenue_growth);
    CALL sp_record_quality_metric('Gross Margin Percentage', 'financial', NULL, v_gross_margin);
    CALL sp_record_quality_metric('Refund Rate', 'financial', NULL, v_refund_rate);
    
    -- Return summary
    SELECT 
        v_payment_success_rate AS payment_success_rate,
        v_fulfillment_rate AS fulfillment_rate,
        v_revenue_growth AS revenue_growth_rate,
        v_gross_margin AS gross_margin_pct,
        v_refund_rate AS refund_rate;
END//

-- Procedure to run all quality assessments
CREATE PROCEDURE sp_run_all_quality_assessments()
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_product_id INT;
    DECLARE v_vendor_id INT;
    DECLARE v_products_assessed INT DEFAULT 0;
    DECLARE v_vendors_assessed INT DEFAULT 0;
    
    DECLARE product_cursor CURSOR FOR 
        SELECT product_id 
        FROM products 
        WHERE status = 'active'
        LIMIT 100; -- Limit to prevent timeout
    
    DECLARE vendor_cursor CURSOR FOR 
        SELECT vendor_id 
        FROM vendors 
        WHERE status = 'active';
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    -- Assess products
    OPEN product_cursor;
    product_loop: LOOP
        FETCH product_cursor INTO v_product_id;
        IF done THEN
            LEAVE product_loop;
        END IF;
        
        CALL sp_assess_product_quality(v_product_id);
        SET v_products_assessed = v_products_assessed + 1;
    END LOOP;
    CLOSE product_cursor;
    
    SET done = FALSE;
    
    -- Assess vendors
    OPEN vendor_cursor;
    vendor_loop: LOOP
        FETCH vendor_cursor INTO v_vendor_id;
        IF done THEN
            LEAVE vendor_loop;
        END IF;
        
        CALL sp_assess_vendor_quality(v_vendor_id);
        SET v_vendors_assessed = v_vendors_assessed + 1;
    END LOOP;
    CLOSE vendor_cursor;
    
    -- Assess financial quality
    CALL sp_assess_financial_quality();
    
    SELECT 
        'Quality Assessment Complete' AS Status,
        v_products_assessed AS Products_Assessed,
        v_vendors_assessed AS Vendors_Assessed;
END//

DELIMITER ;

-- ========================================
-- QUALITY MONITORING VIEWS
-- ========================================

-- Quality Dashboard View
CREATE OR REPLACE VIEW v_quality_dashboard AS
SELECT 
    qt.metric_name,
    qt.metric_category,
    qt.target_value,
    qt.measurement_unit,
    qsh_latest.measured_value AS current_value,
    qsh_latest.quality_level,
    qsh_latest.quality_score,
    qsh_latest.deviation_from_target,
    qsh_latest.deviation_percentage,
    qsh_latest.measurement_date,
    -- Status indicators
    CASE 
        WHEN qsh_latest.quality_level = 'excellent' THEN '✓ Excellent'
        WHEN qsh_latest.quality_level = 'good' THEN '✓ Good'
        WHEN qsh_latest.quality_level = 'acceptable' THEN '• Acceptable'
        WHEN qsh_latest.quality_level = 'warning' THEN '⚠ Warning'
        WHEN qsh_latest.quality_level = 'critical' THEN '✗ Critical'
        ELSE 'No Data'
    END AS status_indicator,
    -- Trend (compare to 7 days ago)
    (SELECT measured_value 
     FROM quality_score_history 
     WHERE threshold_id = qt.threshold_id 
     AND measurement_date = DATE_SUB(CURDATE(), INTERVAL 7 DAY)
     ORDER BY measurement_timestamp DESC 
     LIMIT 1) AS value_7_days_ago,
    CASE 
        WHEN qsh_latest.measured_value > (SELECT measured_value 
             FROM quality_score_history 
             WHERE threshold_id = qt.threshold_id 
             AND measurement_date = DATE_SUB(CURDATE(), INTERVAL 7 DAY)
             ORDER BY measurement_timestamp DESC 
             LIMIT 1) THEN '↑ Improving'
        WHEN qsh_latest.measured_value < (SELECT measured_value 
             FROM quality_score_history 
             WHERE threshold_id = qt.threshold_id 
             AND measurement_date = DATE_SUB(CURDATE(), INTERVAL 7 DAY)
             ORDER BY measurement_timestamp DESC 
             LIMIT 1) THEN '↓ Declining'
        ELSE '→ Stable'
    END AS trend_direction
FROM quality_thresholds qt
LEFT JOIN (
    SELECT 
        threshold_id,
        measured_value,
        quality_level,
        quality_score,
        deviation_from_target,
        deviation_percentage,
        measurement_date,
        measurement_timestamp,
        ROW_NUMBER() OVER (PARTITION BY threshold_id ORDER BY measurement_timestamp DESC) AS rn
    FROM quality_score_history
) qsh_latest ON qt.threshold_id = qsh_latest.threshold_id AND qsh_latest.rn = 1
WHERE qt.is_active = TRUE
ORDER BY 
    qt.metric_category,
    FIELD(qsh_latest.quality_level, 'critical', 'warning', 'acceptable', 'good', 'excellent');

-- Quality Violations Summary
CREATE OR REPLACE VIEW v_quality_violations_summary AS
SELECT 
    qt.metric_name,
    qt.metric_category,
    qv.severity,
    COUNT(*) AS violation_count,
    COUNT(CASE WHEN qv.status = 'open' THEN 1 END) AS open_violations,
    COUNT(CASE WHEN qv.status = 'acknowledged' THEN 1 END) AS acknowledged_violations,
    COUNT(CASE WHEN qv.status = 'resolved' THEN 1 END) AS resolved_violations,
    MIN(qv.created_at) AS oldest_violation,
    MAX(qv.created_at) AS newest_violation,
    AVG(qv.variance) AS avg_variance
FROM quality_violations qv
JOIN quality_thresholds qt ON qv.threshold_id = qt.threshold_id
WHERE qv.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY qt.metric_name, qt.metric_category, qv.severity
ORDER BY 
    FIELD(qv.severity, 'critical', 'high', 'medium', 'low'),
    violation_count DESC;

-- Product Quality Scorecard
CREATE OR REPLACE VIEW v_product_quality_scorecard AS
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    -- Quality metrics
    MAX(CASE WHEN qt.metric_name = 'Product Return Rate' THEN qsh.measured_value END) AS return_rate,
    MAX(CASE WHEN qt.metric_name = 'Product Average Rating' THEN qsh.measured_value END) AS avg_rating,
    MAX(CASE WHEN qt.metric_name = 'Product Review Count' THEN qsh.measured_value END) AS review_count,
    MAX(CASE WHEN qt.metric_name = 'Product Profit Margin' THEN qsh.measured_value END) AS profit_margin,
    MAX(CASE WHEN qt.metric_name = 'Product Defect Rate' THEN qsh.measured_value END) AS defect_rate,
    -- Quality levels
    MAX(CASE WHEN qt.metric_name = 'Product Return Rate' THEN qsh.quality_level END) AS return_rate_level,
    MAX(CASE WHEN qt.metric_name = 'Product Average Rating' THEN qsh.quality_level END) AS rating_level,
    MAX(CASE WHEN qt.metric_name = 'Product Profit Margin' THEN qsh.quality_level END) AS margin_level,
    -- Overall quality score (average)
    ROUND(AVG(qsh.quality_score), 2) AS overall_quality_score,
    -- Quality grade
    CASE 
        WHEN AVG(qsh.quality_score) >= 90 THEN 'A (Excellent)'
        WHEN AVG(qsh.quality_score) >= 80 THEN 'B (Good)'
        WHEN AVG(qsh.quality_score) >= 70 THEN 'C (Acceptable)'
        WHEN AVG(qsh.quality_score) >= 60 THEN 'D (Warning)'
        ELSE 'F (Critical)'
    END AS quality_grade,
    MAX(qsh.measurement_date) AS last_assessed
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN quality_score_history qsh ON p.product_id = qsh.entity_id AND qsh.entity_type = 'product'
LEFT JOIN quality_thresholds qt ON qsh.threshold_id = qt.threshold_id
WHERE p.status = 'active'
    AND qsh.measurement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name
HAVING overall_quality_score IS NOT NULL
ORDER BY overall_quality_score DESC;

-- Vendor Quality Scorecard
CREATE OR REPLACE VIEW v_vendor_quality_scorecard AS
SELECT 
    v.vendor_id,
    v.vendor_name,
    v.status,
    -- Quality metrics
    MAX(CASE WHEN qt.metric_name = 'Vendor Rating' THEN qsh.measured_value END) AS vendor_rating,
    MAX(CASE WHEN qt.metric_name = 'Vendor Quality Score' THEN qsh.measured_value END) AS quality_score,
    MAX(CASE WHEN qt.metric_name = 'Vendor Defect Rate' THEN qsh.measured_value END) AS defect_rate,
    -- Quality levels
    MAX(CASE WHEN qt.metric_name = 'Vendor Rating' THEN qsh.quality_level END) AS rating_level,
    MAX(CASE WHEN qt.metric_name = 'Vendor Quality Score' THEN qsh.quality_level END) AS score_level,
    -- Overall assessment
    ROUND(AVG(qsh.quality_score), 2) AS overall_quality_score,
    CASE 
        WHEN AVG(qsh.quality_score) >= 90 THEN 'Preferred Vendor'
        WHEN AVG(qsh.quality_score) >= 80 THEN 'Approved Vendor'
        WHEN AVG(qsh.quality_score) >= 70 THEN 'Conditional Vendor'
        ELSE 'Review Required'
    END AS vendor_status_assessment,
    MAX(qsh.measurement_date) AS last_assessed
FROM vendors v
LEFT JOIN quality_score_history qsh ON v.vendor_id = qsh.entity_id AND qsh.entity_type = 'vendor'
LEFT JOIN quality_thresholds qt ON qsh.threshold_id = qt.threshold_id
WHERE v.status = 'active'
    AND qsh.measurement_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
GROUP BY v.vendor_id, v.vendor_name, v.status
HAVING overall_quality_score IS NOT NULL
ORDER BY overall_quality_score DESC;

-- Critical Quality Issues
CREATE OR REPLACE VIEW v_critical_quality_issues AS
SELECT 
    qv.violation_id,
    qt.metric_name,
    qt.metric_category,
    qv.severity,
    qv.violation_type,
    qv.entity_type,
    qv.entity_id,
    qv.measured_value,
    qv.threshold_value,
    qv.variance,
    qv.violation_message,
    qv.status,
    qv.created_at,
    DATEDIFF(NOW(), qv.created_at) AS days_open,
    -- Entity details
    CASE 
        WHEN qv.entity_type = 'product' THEN (SELECT product_name FROM products WHERE product_id = qv.entity_id)
        WHEN qv.entity_type = 'vendor' THEN (SELECT vendor_name FROM vendors WHERE vendor_id = qv.entity_id)
        WHEN qv.entity_type = 'customer' THEN (SELECT CONCAT(first_name, ' ', last_name) FROM customers WHERE customer_id = qv.entity_id)
        ELSE NULL
    END AS entity_name
FROM quality_violations qv
JOIN quality_thresholds qt ON qv.threshold_id = qt.threshold_id
WHERE qv.status IN ('open', 'acknowledged')
    AND qv.severity IN ('critical', 'high')
ORDER BY 
    FIELD(qv.severity, 'critical', 'high'),
    qv.created_at ASC;

-- Quality Trends Over Time
CREATE OR REPLACE VIEW v_quality_trends AS
SELECT 
    qt.metric_name,
    qt.metric_category,
    qt.target_value,
    qsh.measurement_date,
    AVG(qsh.measured_value) AS avg_measured_value,
    AVG(qsh.quality_score) AS avg_quality_score,
    COUNT(CASE WHEN qsh.quality_level = 'excellent' THEN 1 END) AS excellent_count,
    COUNT(CASE WHEN qsh.quality_level = 'good' THEN 1 END) AS good_count,
    COUNT(CASE WHEN qsh.quality_level = 'acceptable' THEN 1 END) AS acceptable_count,
    COUNT(CASE WHEN qsh.quality_level = 'warning' THEN 1 END) AS warning_count,
    COUNT(CASE WHEN qsh.quality_level = 'critical' THEN 1 END) AS critical_count,
    COUNT(*) AS total_measurements
FROM quality_thresholds qt
JOIN quality_score_history qsh ON qt.threshold_id = qsh.threshold_id
WHERE qsh.measurement_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY qt.metric_name, qt.metric_category, qt.target_value, qsh.measurement_date
ORDER BY qsh.measurement_date DESC, qt.metric_category;

-- Quality Performance by Category
CREATE OR REPLACE VIEW v_quality_by_category AS
SELECT 
    qt.metric_category,
    COUNT(DISTINCT qt.threshold_id) AS total_metrics,
    -- Latest quality levels
    SUM(CASE WHEN qsh.quality_level = 'excellent' THEN 1 ELSE 0 END) AS excellent_metrics,
    SUM(CASE WHEN qsh.quality_level = 'good' THEN 1 ELSE 0 END) AS good_metrics,
    SUM(CASE WHEN qsh.quality_level = 'acceptable' THEN 1 ELSE 0 END) AS acceptable_metrics,
    SUM(CASE WHEN qsh.quality_level = 'warning' THEN 1 ELSE 0 END) AS warning_metrics,
    SUM(CASE WHEN qsh.quality_level = 'critical' THEN 1 ELSE 0 END) AS critical_metrics,
    -- Average quality score
    ROUND(AVG(qsh.quality_score), 2) AS avg_quality_score,
    -- Category health
    CASE 
        WHEN AVG(qsh.quality_score) >= 90 THEN '✓ Excellent Health'
        WHEN AVG(qsh.quality_score) >= 80 THEN '✓ Good Health'
        WHEN AVG(qsh.quality_score) >= 70 THEN '• Fair Health'
        WHEN AVG(qsh.quality_score) >= 60 THEN '⚠ Needs Attention'
        ELSE '✗ Critical Issues'
    END AS category_health
FROM quality_thresholds qt
LEFT JOIN (
    SELECT 
        threshold_id,
        quality_level,
        quality_score,
        ROW_NUMBER() OVER (PARTITION BY threshold_id ORDER BY measurement_timestamp DESC) AS rn
    FROM quality_score_history
) qsh ON qt.threshold_id = qsh.threshold_id AND qsh.rn = 1
WHERE qt.is_active = TRUE
GROUP BY qt.metric_category
ORDER BY avg_quality_score DESC;

-- ========================================
-- QUALITY MANAGEMENT PROCEDURES
-- ========================================

DELIMITER //

-- Procedure to resolve quality violation
CREATE PROCEDURE sp_resolve_quality_violation(
    IN p_violation_id INT,
    IN p_status VARCHAR(20),
    IN p_resolution_notes TEXT,
    IN p_resolved_by VARCHAR(100)
)
BEGIN
    UPDATE quality_violations
    SET 
        status = p_status,
        resolution_notes = p_resolution_notes,
        acknowledged_by = p_resolved_by,
        acknowledged_at = CASE WHEN p_status = 'acknowledged' THEN NOW() ELSE acknowledged_at END,
        resolved_at = CASE WHEN p_status = 'resolved' THEN NOW() ELSE NULL END
    WHERE violation_id = p_violation_id;
    
    SELECT CONCAT('Violation #', p_violation_id, ' marked as ', p_status) AS Result;
END//

-- Procedure to update quality threshold
CREATE PROCEDURE sp_update_quality_threshold(
    IN p_metric_name VARCHAR(200),
    IN p_target_value DECIMAL(10,2),
    IN p_tolerance_pct DECIMAL(5,2),
    IN p_excellent_min DECIMAL(10,2),
    IN p_good_min DECIMAL(10,2),
    IN p_acceptable_min DECIMAL(10,2),
    IN p_warning_max DECIMAL(10,2),
    IN p_critical_max DECIMAL(10,2)
)
BEGIN
    UPDATE quality_thresholds
    SET 
        target_value = p_target_value,
        tolerance_percentage = p_tolerance_pct,
        excellent_min = p_excellent_min,
        good_min = p_good_min,
        acceptable_min = p_acceptable_min,
        warning_max = p_warning_max,
        critical_max = p_critical_max,
        updated_at = NOW()
    WHERE metric_name = p_metric_name;
    
    SELECT CONCAT('Threshold updated for: ', p_metric_name) AS Result;
END//

-- Procedure to generate quality report
CREATE PROCEDURE sp_generate_quality_report(
    IN p_category VARCHAR(50),
    IN p_days_back INT
)
BEGIN
    -- Summary statistics
    SELECT 
        'Quality Report Summary' AS Report_Section,
        qt.metric_category,
        COUNT(DISTINCT qt.threshold_id) AS total_metrics,
        COUNT(DISTINCT qsh.entity_id) AS entities_measured,
        ROUND(AVG(qsh.quality_score), 2) AS avg_quality_score,
        SUM(CASE WHEN qsh.quality_level = 'excellent' THEN 1 ELSE 0 END) AS excellent_count,
        SUM(CASE WHEN qsh.quality_level = 'good' THEN 1 ELSE 0 END) AS good_count,
        SUM(CASE WHEN qsh.quality_level = 'acceptable' THEN 1 ELSE 0 END) AS acceptable_count,
        SUM(CASE WHEN qsh.quality_level = 'warning' THEN 1 ELSE 0 END) AS warning_count,
        SUM(CASE WHEN qsh.quality_level = 'critical' THEN 1 ELSE 0 END) AS critical_count
    FROM quality_thresholds qt
    LEFT JOIN quality_score_history qsh ON qt.threshold_id = qsh.threshold_id
        AND qsh.measurement_date >= DATE_SUB(CURDATE(), INTERVAL p_days_back DAY)
    WHERE (p_category IS NULL OR qt.metric_category = p_category)
        AND qt.is_active = TRUE
    GROUP BY qt.metric_category;
    
    -- Top quality issues
    SELECT 
        'Top Quality Issues' AS Report_Section,
        qt.metric_name,
        COUNT(*) AS issue_count,
        AVG(qv.variance) AS avg_variance,
        MAX(qv.severity) AS max_severity
    FROM quality_violations qv
    JOIN quality_thresholds qt ON qv.threshold_id = qt.threshold_id
    WHERE qv.created_at >= DATE_SUB(NOW(), INTERVAL p_days_back DAY)
        AND (p_category IS NULL OR qt.metric_category = p_category)
        AND qv.status IN ('open', 'acknowledged')
    GROUP BY qt.metric_name
    ORDER BY issue_count DESC
    LIMIT 10;
    
    -- Quality improvements
    SELECT 
        'Quality Improvements' AS Report_Section,
        qt.metric_name,
        qsh_old.measured_value AS previous_value,
        qsh_new.measured_value AS current_value,
        (qsh_new.measured_value - qsh_old.measured_value) AS improvement,
        ROUND(((qsh_new.measured_value - qsh_old.measured_value) / 
               NULLIF(qsh_old.measured_value, 0)) * 100, 2) AS improvement_pct
    FROM quality_thresholds qt
    LEFT JOIN (
        SELECT threshold_id, AVG(measured_value) AS measured_value
        FROM quality_score_history
        WHERE measurement_date = DATE_SUB(CURDATE(), INTERVAL p_days_back DAY)
        GROUP BY threshold_id
    ) qsh_old ON qt.threshold_id = qsh_old.threshold_id
    LEFT JOIN (
        SELECT threshold_id, AVG(measured_value) AS measured_value
        FROM quality_score_history
        WHERE measurement_date = CURDATE()
        GROUP BY threshold_id
    ) qsh_new ON qt.threshold_id = qsh_new.threshold_id
    WHERE (p_category IS NULL OR qt.metric_category = p_category)
        AND qt.is_active = TRUE
        AND qt.higher_is_better = TRUE
        AND qsh_new.measured_value > qsh_old.measured_value
    ORDER BY improvement_pct DESC
    LIMIT 10;
END//

-- Procedure to alert on threshold breaches
CREATE PROCEDURE sp_check_quality_thresholds()
BEGIN
    DECLARE v_violations_found INT DEFAULT 0;
    
    -- Check for new violations in the last measurement
    SELECT COUNT(*) INTO v_violations_found
    FROM quality_violations
    WHERE status = 'open'
        AND created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR);
    
    IF v_violations_found > 0 THEN
        -- Return list of new violations
        SELECT 
            qv.violation_id,
            qt.metric_name,
            qt.metric_category,
            qv.severity,
            qv.violation_message,
            qv.created_at
        FROM quality_violations qv
        JOIN quality_thresholds qt ON qv.threshold_id = qt.threshold_id
        WHERE qv.status = 'open'
            AND qv.created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
        ORDER BY FIELD(qv.severity, 'critical', 'high', 'medium', 'low');
    ELSE
        SELECT 'No new quality violations detected' AS Status;
    END IF;
END//

DELIMITER ;

-- ========================================
-- QUICK START COMMANDS
-- ========================================

-- Display completion message and examples
SELECT '============================================' AS '';
SELECT 'Quality Thresholds System Created Successfully' AS Status;
SELECT '============================================' AS '';
SELECT '' AS '';
SELECT 'QUALITY THRESHOLD RANGES:' AS '';
SELECT '-------------------------------------------' AS '';
SELECT 'Excellent: >= Excellent Min' AS '';
SELECT 'Good:      >= Good Min' AS '';
SELECT 'Acceptable: >= Acceptable Min' AS '';
SELECT 'Warning:   <= Warning Max' AS '';
SELECT 'Critical:  <= Critical Max' AS '';
SELECT '' AS '';
SELECT 'QUICK START COMMANDS:' AS '';
SELECT '-------------------------------------------' AS '';
SELECT '1. View quality dashboard:' AS '';
SELECT '   SELECT * FROM v_quality_dashboard;' AS '';
SELECT '' AS '';
SELECT '2. Assess product quality:' AS '';
SELECT '   CALL sp_assess_product_quality(1);' AS '';
SELECT '' AS '';
SELECT '3. Assess vendor quality:' AS '';
SELECT '   CALL sp_assess_vendor_quality(1);' AS '';
SELECT '' AS '';
SELECT '4. Assess financial quality:' AS '';
SELECT '   CALL sp_assess_financial_quality();' AS '';
SELECT '' AS '';
SELECT '5. Run all quality assessments:' AS '';
SELECT '   CALL sp_run_all_quality_assessments();' AS '';
SELECT '' AS '';
SELECT '6. View critical issues:' AS '';
SELECT '   SELECT * FROM v_critical_quality_issues;' AS '';
SELECT '' AS '';
SELECT '7. View product scorecard:' AS '';
SELECT '   SELECT * FROM v_product_quality_scorecard;' AS '';
SELECT '' AS '';
SELECT '8. Generate quality report:' AS '';
SELECT '   CALL sp_generate_quality_report("product", 30);' AS '';
SELECT '' AS '';
SELECT '9. Check for threshold breaches:' AS '';
SELECT '   CALL sp_check_quality_thresholds();' AS '';
SELECT '' AS '';
SELECT '10. Resolve a violation:' AS '';
SELECT '    CALL sp_resolve_quality_violation(1, "resolved", "Issue fixed", "admin");' AS '';
SELECT '' AS '';
SELECT '============================================' AS '';

-- Show configured thresholds summary
SELECT 'Configured Quality Thresholds by Category:' AS '';
SELECT 
    metric_category,
    COUNT(*) AS threshold_count,
    SUM(CASE WHEN is_active = TRUE THEN 1 ELSE 0 END) AS active_count
FROM quality_thresholds
GROUP BY metric_category
ORDER BY metric_category;