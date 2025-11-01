-- ========================================
-- DATA VALIDATION RULES ENGINE
-- E-commerce Revenue Analytics Engine
-- Business Rules & Constraint Checking
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- SECTION 1: CREATE VALIDATION INFRASTRUCTURE
-- ========================================
SELECT '========== CREATING VALIDATION INFRASTRUCTURE ==========' AS '';

-- Table to store validation rule definitions
CREATE TABLE IF NOT EXISTS validation_rules (
    rule_id INT PRIMARY KEY AUTO_INCREMENT,
    rule_name VARCHAR(100) NOT NULL,
    rule_category ENUM('data_quality', 'business_logic', 'referential_integrity', 'financial', 'compliance') NOT NULL,
    table_name VARCHAR(100),
    severity ENUM('critical', 'error', 'warning', 'info') DEFAULT 'error',
    rule_description TEXT,
    validation_query TEXT NOT NULL,
    fix_suggestion TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_rule_category (rule_category),
    INDEX idx_severity (severity),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB;

-- Table to store validation execution results
CREATE TABLE IF NOT EXISTS validation_results (
    result_id INT PRIMARY KEY AUTO_INCREMENT,
    rule_id INT NOT NULL,
    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    violations_found INT DEFAULT 0,
    status ENUM('pass', 'fail', 'error') DEFAULT 'pass',
    error_message TEXT,
    sample_violations TEXT,
    FOREIGN KEY (rule_id) REFERENCES validation_rules(rule_id) ON DELETE CASCADE,
    INDEX idx_rule_id (rule_id),
    INDEX idx_execution_time (execution_time),
    INDEX idx_status (status)
) ENGINE=InnoDB;

-- Table to store detailed violation records
CREATE TABLE IF NOT EXISTS validation_violations (
    violation_id INT PRIMARY KEY AUTO_INCREMENT,
    rule_id INT NOT NULL,
    table_name VARCHAR(100),
    record_id VARCHAR(50),
    violation_details TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP NULL,
    resolved_by VARCHAR(100),
    FOREIGN KEY (rule_id) REFERENCES validation_rules(rule_id) ON DELETE CASCADE,
    INDEX idx_rule_id (rule_id),
    INDEX idx_table_name (table_name),
    INDEX idx_resolved (resolved),
    INDEX idx_detected_at (detected_at)
) ENGINE=InnoDB;

SELECT 'Validation infrastructure created successfully' AS Status;

-- ========================================
-- SECTION 2: DEFINE DATA QUALITY RULES
-- ========================================
SELECT '========== DEFINING DATA QUALITY RULES ==========' AS '';

-- Clear existing rules (for clean reinstall)
DELETE FROM validation_rules;

-- RULE 1: No NULL emails for active customers
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'customer_email_required',
    'data_quality',
    'customers',
    'critical',
    'All active customers must have a valid email address',
    'SELECT customer_id, CONCAT(first_name, " ", last_name) AS customer_name, status FROM customers WHERE (email IS NULL OR email = "") AND status = "active"',
    'Update customer email or change status to inactive'
);

-- RULE 2: Valid email format
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'customer_email_format',
    'data_quality',
    'customers',
    'error',
    'Customer emails must be in valid format (contains @ and domain)',
    'SELECT customer_id, email FROM customers WHERE email IS NOT NULL AND email NOT LIKE "%_@__%.__%"',
    'Correct email format to include @ symbol and domain'
);

-- RULE 3: No duplicate SKUs
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'product_unique_sku',
    'data_quality',
    'products',
    'critical',
    'Product SKUs must be unique across all products',
    'SELECT sku, COUNT(*) AS duplicate_count, GROUP_CONCAT(product_id) AS product_ids FROM products WHERE sku IS NOT NULL GROUP BY sku HAVING COUNT(*) > 1',
    'Assign unique SKUs or merge duplicate products'
);

-- RULE 4: Product price must be positive
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'product_positive_price',
    'data_quality',
    'products',
    'error',
    'Product prices must be greater than zero for active products',
    'SELECT product_id, sku, product_name, price FROM products WHERE (price IS NULL OR price <= 0) AND status = "active"',
    'Set valid price or mark product as discontinued'
);

-- RULE 5: Phone number format
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'customer_phone_format',
    'data_quality',
    'customers',
    'warning',
    'Phone numbers should contain only digits, spaces, dashes, and parentheses',
    'SELECT customer_id, phone FROM customers WHERE phone IS NOT NULL AND phone REGEXP "[^0-9()\\- ]"',
    'Clean phone number to contain only valid characters'
);

-- ========================================
-- SECTION 3: DEFINE BUSINESS LOGIC RULES
-- ========================================
SELECT '========== DEFINING BUSINESS LOGIC RULES ==========' AS '';

-- RULE 6: Order total matches line items
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'order_total_accuracy',
    'business_logic',
    'orders',
    'critical',
    'Order total must equal sum of line items plus shipping and tax',
    'SELECT o.order_id, o.total_amount AS declared_total, (COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount) AS calculated_total, ABS(o.total_amount - (COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount)) AS difference FROM orders o LEFT JOIN order_items oi ON o.order_id = oi.order_id GROUP BY o.order_id, o.total_amount, o.shipping_cost, o.tax_amount HAVING ABS(o.total_amount - (COALESCE(SUM(oi.subtotal), 0) + o.shipping_cost + o.tax_amount)) > 0.01',
    'Recalculate order total or review line items'
);

-- RULE 7: Inventory cannot be negative
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'inventory_non_negative',
    'business_logic',
    'inventory',
    'critical',
    'Inventory quantities cannot be negative',
    'SELECT i.inventory_id, i.product_id, p.sku, p.product_name, i.quantity_on_hand, i.quantity_reserved FROM inventory i JOIN products p ON i.product_id = p.product_id WHERE i.quantity_on_hand < 0 OR i.quantity_reserved < 0',
    'Investigate inventory discrepancies and correct quantities'
);

-- RULE 8: Discount cannot exceed item price
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'order_item_valid_discount',
    'business_logic',
    'order_items',
    'error',
    'Discount cannot exceed the item total price',
    'SELECT oi.order_item_id, oi.order_id, p.product_name, oi.quantity, oi.unit_price, oi.discount, (oi.quantity * oi.unit_price) AS item_total FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE oi.discount > (oi.quantity * oi.unit_price)',
    'Review discount amount and adjust to valid value'
);

-- RULE 9: Order date cannot be in the future
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'order_valid_date',
    'business_logic',
    'orders',
    'error',
    'Order dates cannot be in the future',
    'SELECT order_id, customer_id, order_date, total_amount FROM orders WHERE order_date > NOW()',
    'Correct order date to current or past timestamp'
);

-- RULE 10: Product cost should not exceed price
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'product_profitable_pricing',
    'business_logic',
    'products',
    'warning',
    'Product cost should not exceed selling price (negative margin)',
    'SELECT product_id, sku, product_name, cost, price, (cost - price) AS loss_per_unit FROM products WHERE cost > price AND status = "active"',
    'Review pricing strategy or product costs'
);

-- RULE 11: Paid orders must have delivered or shipped status
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'order_payment_status_consistency',
    'business_logic',
    'orders',
    'error',
    'Orders marked as paid should not be pending or cancelled',
    'SELECT order_id, customer_id, status, payment_status, total_amount FROM orders WHERE payment_status = "paid" AND status IN ("pending", "cancelled")',
    'Update order status to match payment status'
);

-- RULE 12: Loyalty points cannot be negative
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'loyalty_positive_balance',
    'business_logic',
    'loyalty_program',
    'critical',
    'Loyalty program points balance cannot be negative',
    'SELECT lp.loyalty_id, lp.customer_id, c.email, lp.points_balance FROM loyalty_program lp JOIN customers c ON lp.customer_id = c.customer_id WHERE lp.points_balance < 0',
    'Investigate points redemption and correct balance'
);

-- ========================================
-- SECTION 4: DEFINE REFERENTIAL INTEGRITY RULES
-- ========================================
SELECT '========== DEFINING REFERENTIAL INTEGRITY RULES ==========' AS '';

-- RULE 13: Orphaned order items
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'order_items_valid_order',
    'referential_integrity',
    'order_items',
    'critical',
    'All order items must reference a valid order',
    'SELECT oi.order_item_id, oi.order_id FROM order_items oi LEFT JOIN orders o ON oi.order_id = o.order_id WHERE o.order_id IS NULL',
    'Delete orphaned order items or restore missing orders'
);

-- RULE 14: Orphaned orders (no customer)
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'orders_valid_customer',
    'referential_integrity',
    'orders',
    'error',
    'Orders should reference a valid customer',
    'SELECT o.order_id, o.customer_id, o.order_date, o.total_amount FROM orders o LEFT JOIN customers c ON o.customer_id = c.customer_id WHERE o.customer_id IS NOT NULL AND c.customer_id IS NULL',
    'Link order to correct customer or mark customer_id as NULL'
);

-- RULE 15: Orders without line items
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'orders_have_items',
    'referential_integrity',
    'orders',
    'warning',
    'Orders should contain at least one line item',
    'SELECT o.order_id, o.customer_id, o.order_date, o.total_amount FROM orders o LEFT JOIN order_items oi ON o.order_id = oi.order_id WHERE oi.order_item_id IS NULL AND o.status NOT IN ("cancelled")',
    'Add line items or cancel empty orders'
);

-- RULE 16: Products without categories
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'products_have_category',
    'referential_integrity',
    'products',
    'warning',
    'Active products should be assigned to a category',
    'SELECT product_id, sku, product_name, status FROM products WHERE category_id IS NULL AND status = "active"',
    'Assign products to appropriate categories'
);

-- ========================================
-- SECTION 5: DEFINE FINANCIAL RULES
-- ========================================
SELECT '========== DEFINING FINANCIAL RULES ==========' AS '';

-- RULE 17: Tax rate reasonability
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'order_reasonable_tax',
    'financial',
    'orders',
    'warning',
    'Tax should be between 0% and 20% of subtotal',
    'SELECT o.order_id, o.total_amount, o.tax_amount, COALESCE(SUM(oi.subtotal), 0) AS subtotal, (o.tax_amount * 100.0 / NULLIF(COALESCE(SUM(oi.subtotal), 0), 0)) AS tax_rate_pct FROM orders o LEFT JOIN order_items oi ON o.order_id = oi.order_id GROUP BY o.order_id, o.total_amount, o.tax_amount HAVING tax_rate_pct > 20 OR tax_rate_pct < 0',
    'Review tax calculation and rates'
);

-- RULE 18: Shipping cost reasonability
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'order_reasonable_shipping',
    'financial',
    'orders',
    'warning',
    'Shipping cost should not exceed order subtotal',
    'SELECT o.order_id, o.shipping_cost, COALESCE(SUM(oi.subtotal), 0) AS subtotal FROM orders o LEFT JOIN order_items oi ON o.order_id = oi.order_id GROUP BY o.order_id, o.shipping_cost HAVING o.shipping_cost > subtotal AND subtotal > 0',
    'Review shipping charges for accuracy'
);

-- RULE 19: Refund amounts match order totals
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'returns_valid_refund_amount',
    'financial',
    'returns',
    'error',
    'Refund amounts should not exceed original order item value',
    'SELECT r.return_id, r.order_id, r.refund_amount, oi.subtotal AS item_value FROM returns r LEFT JOIN order_items oi ON r.order_item_id = oi.order_item_id WHERE r.refund_amount > oi.subtotal',
    'Adjust refund amount to match original purchase value'
);

-- RULE 20: Campaign spend within budget
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'campaign_budget_compliance',
    'financial',
    'campaigns',
    'warning',
    'Campaign total spend should not exceed allocated budget',
    'SELECT c.campaign_id, c.campaign_name, c.budget, COALESCE(SUM(cp.spend), 0) AS total_spend, (COALESCE(SUM(cp.spend), 0) - c.budget) AS over_budget FROM campaigns c LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id WHERE c.budget IS NOT NULL GROUP BY c.campaign_id, c.campaign_name, c.budget HAVING total_spend > c.budget',
    'Review campaign spending and adjust budget or pause campaign'
);

-- ========================================
-- SECTION 6: DEFINE COMPLIANCE RULES
-- ========================================
SELECT '========== DEFINING COMPLIANCE RULES ==========' AS '';

-- RULE 21: Customer data completeness
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'customer_data_complete',
    'compliance',
    'customers',
    'warning',
    'Active customers should have complete address information',
    'SELECT customer_id, email, city, state, zip_code FROM customers WHERE status = "active" AND (city IS NULL OR state IS NULL OR zip_code IS NULL)',
    'Complete customer address information'
);

-- RULE 22: Payment method expiration
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'payment_method_not_expired',
    'compliance',
    'payment_methods',
    'warning',
    'Customer payment methods should not be expired',
    'SELECT pm.payment_method_id, pm.customer_id, c.email, pm.card_last_four, pm.expiry_month, pm.expiry_year FROM payment_methods pm JOIN customers c ON pm.customer_id = c.customer_id WHERE (pm.expiry_year < YEAR(NOW())) OR (pm.expiry_year = YEAR(NOW()) AND pm.expiry_month < MONTH(NOW()))',
    'Request updated payment method from customer'
);

-- RULE 23: Vendor contract expiration
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'vendor_contract_active_valid',
    'compliance',
    'vendor_contracts',
    'error',
    'Active vendor contracts should not be expired',
    'SELECT vc.contract_id, v.vendor_name, p.product_name, vc.end_date FROM vendor_contracts vc JOIN vendors v ON vc.vendor_id = v.vendor_id JOIN products p ON vc.product_id = p.product_id WHERE vc.status = "active" AND vc.end_date < CURDATE()',
    'Update contract status to expired or renew contract'
);

-- RULE 24: Review moderation timeliness
INSERT INTO validation_rules (rule_name, rule_category, table_name, severity, rule_description, validation_query, fix_suggestion)
VALUES (
    'reviews_timely_moderation',
    'compliance',
    'reviews',
    'info',
    'Pending reviews should be moderated within 7 days',
    'SELECT review_id, product_id, customer_id, rating, created_at, DATEDIFF(NOW(), created_at) AS days_pending FROM reviews WHERE status = "pending" AND DATEDIFF(NOW(), created_at) > 7',
    'Review and approve/reject pending reviews'
);

SELECT CONCAT('Total validation rules defined: ', COUNT(*), '') AS Status 
FROM validation_rules;

-- ========================================
-- SECTION 7: EXECUTE ALL VALIDATION RULES
-- ========================================
SELECT '========== EXECUTING ALL VALIDATION RULES ==========' AS '';

-- Clear recent violations (optional - keep history)
-- DELETE FROM validation_violations WHERE detected_at < DATE_SUB(NOW(), INTERVAL 30 DAY);

-- Stored procedure to execute a single validation rule
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS execute_validation_rule(IN p_rule_id INT)
BEGIN
DECLARE v_rule_name VARCHAR(100);
    DECLARE v_validation_query TEXT;
    DECLARE v_violations INT;
    DECLARE v_result_id INT;
    
    -- Get rule details
    SELECT rule_name, validation_query 
    INTO v_rule_name, v_validation_query
    FROM validation_rules 
    WHERE rule_id = p_rule_id AND is_active = TRUE;
    
    -- Insert initial result record
    INSERT INTO validation_results (rule_id, status)
    VALUES (p_rule_id, 'pass');
    

    












SET v_result_id = LAST_INSERT_ID();
    
    -- Execute validation query and count violations
    SET @sql = CONCAT('SELECT COUNT(*) INTO @violation_count FROM (', v_validation_query, ') AS violations');
    PREPARE stmt FROM @sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;
    
    SET v_violations = @violation_count;
    
    -- Update result with findings
    UPDATE validation_results
    SET violations_found = v_violations,
        status = IF(v_violations > 0, 'fail', 'pass')
    WHERE result_id = v_result_id;
    
    -- If violations found, store sample records
    IF v_violations > 0 THEN
        SET @sql = CONCAT('INSERT INTO validation_violations (rule_id, table_name, violation_details) 
                          SELECT ', p_rule_id, ', "see_query", CONCAT_WS(", ", *) 
                          FROM (', v_validation_query, ') AS violations LIMIT 10');
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END//
DELIMITER ;

-- Execute all active validation rules
DELIMITER //
CREATE PROCEDURE IF NOT EXISTS execute_all_validations()
BEGIN
DECLARE done INT DEFAULT FALSE;
    DECLARE v_rule_id INT;
    DECLARE cur CURSOR FOR SELECT rule_id FROM validation_rules WHERE is_active = TRUE ORDER BY severity DESC;
    DECLARE CONTINUE HANDLER FOR NOT FOUND 

    












SET done = TRUE;
    
    OPEN cur;
    read_loop: LOOP
        FETCH cur INTO v_rule_id;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        CALL execute_validation_rule(v_rule_id);
    END LOOP;
    CLOSE cur;
END//
DELIMITER ;

-- Run all validations
CALL execute_all_validations();

SELECT 'All validation rules executed' AS Status;

-- ========================================
-- SECTION 8: VALIDATION SUMMARY REPORT
-- ========================================
SELECT '========== VALIDATION SUMMARY REPORT ==========' AS '';

-- Overall validation summary
SELECT 
    COUNT(*) AS total_rules,
    SUM(CASE WHEN is_active = TRUE THEN 1 ELSE 0 END) AS active_rules,
    SUM(CASE WHEN is_active = FALSE THEN 1 ELSE 0 END) AS inactive_rules
FROM validation_rules;

-- Summary by category
SELECT 
    rule_category AS Category,
    COUNT(*) AS Total_Rules,
    SUM(CASE WHEN is_active = TRUE THEN 1 ELSE 0 END) AS Active_Rules,
    SUM(CASE WHEN vr.rule_id IS NOT NULL AND vr.violations_found > 0 THEN 1 ELSE 0 END) AS Rules_Failed
FROM validation_rules vl
LEFT JOIN (
    SELECT rule_id, violations_found 
    FROM validation_results 
    WHERE result_id IN (
        SELECT MAX(result_id) 
        FROM validation_results 
        GROUP BY rule_id
    )
) vr ON vl.rule_id = vr.rule_id
GROUP BY rule_category
ORDER BY Rules_Failed DESC;

-- Summary by severity
SELECT 
    severity AS Severity,
    COUNT(*) AS Total_Rules,
    SUM(CASE WHEN vr.rule_id IS NOT NULL AND vr.violations_found > 0 THEN 1 ELSE 0 END) AS Rules_Failed,
    SUM(COALESCE(vr.violations_found, 0)) AS Total_Violations
FROM validation_rules vl
LEFT JOIN (
    SELECT rule_id, violations_found 
    FROM validation_results 
    WHERE result_id IN (
        SELECT MAX(result_id) 
        FROM validation_results 
        GROUP BY rule_id
    )
) vr ON vl.rule_id = vr.rule_id
WHERE vl.is_active = TRUE
GROUP BY severity
ORDER BY FIELD(severity, 'critical', 'error', 'warning', 'info');

-- ========================================
-- SECTION 9: FAILED RULES DETAIL REPORT
-- ========================================
SELECT '========== FAILED VALIDATION RULES ==========' AS '';

SELECT 
    vl.rule_name AS Rule_Name,
    vl.rule_category AS Category,
    vl.severity AS Severity,
    vl.table_name AS Table_Name,
    vr.violations_found AS Violations_Count,
    vl.rule_description AS Description,
    vl.fix_suggestion AS Fix_Suggestion
FROM validation_rules vl
JOIN (
    SELECT rule_id, violations_found 
    FROM validation_results 
    WHERE result_id IN (
        SELECT MAX(result_id) 
        FROM validation_results 
        GROUP BY rule_id
    )
) vr ON vl.rule_id = vr.rule_id
WHERE vr.violations_found > 0
ORDER BY 
    FIELD(vl.severity, 'critical', 'error', 'warning', 'info'),
    vr.violations_found DESC;

-- ========================================
-- SECTION 10: TOP VIOLATIONS BY TABLE
-- ========================================
SELECT '========== TOP VIOLATIONS BY TABLE ==========' AS '';

SELECT 
    vl.table_name AS Table_Name,
    COUNT(DISTINCT vl.rule_id) AS Failed_Rules,
    SUM(vr.violations_found) AS Total_Violations,
    GROUP_CONCAT(DISTINCT vl.severity ORDER BY FIELD(vl.severity, 'critical', 'error', 'warning', 'info') SEPARATOR ', ') AS Severity_Levels
FROM validation_rules vl
JOIN (
    SELECT rule_id, violations_found 
    FROM validation_results 
    WHERE result_id IN (
        SELECT MAX(result_id) 
        FROM validation_results 
        GROUP BY rule_id
    )
) vr ON vl.rule_id = vr.rule_id
WHERE vr.violations_found > 0 AND vl.table_name IS NOT NULL
GROUP BY vl.table_name
ORDER BY Total_Violations DESC;

-- ========================================
-- SECTION 11: VALIDATION TRENDS
-- ========================================
SELECT '========== VALIDATION TRENDS (LAST 7 DAYS) ==========' AS '';

SELECT 
    DATE(vr.execution_time) AS Execution_Date,
    COUNT(DISTINCT vr.rule_id) AS Rules_Executed,
    SUM(CASE WHEN vr.status = 'fail' THEN 1 ELSE 0 END) AS Rules_Failed,
    SUM(vr.violations_found) AS Total_Violations,
    ROUND(AVG(vr.violations_found), 2) AS Avg_Violations_Per_Rule
FROM validation_results vr
WHERE vr.execution_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY DATE(vr.execution_time)
ORDER BY Execution_Date DESC;

-- ========================================
-- SECTION 12: CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION
-- ========================================
SELECT '========== CRITICAL ISSUES - IMMEDIATE ACTION REQUIRED ==========' AS '';

SELECT 
    vl.rule_name AS Critical_Issue,
    vl.table_name AS Affected_Table,
    vr.violations_found AS Violations,
    vl.rule_description AS Issue_Description,
    vl.fix_suggestion AS Recommended_Action
FROM validation_rules vl
JOIN (
    SELECT rule_id, violations_found 
    FROM validation_results 
    WHERE result_id IN (
        SELECT MAX(result_id) 
        FROM validation_results 
        GROUP BY rule_id
    )
) vr ON vl.rule_id = vr.rule_id
WHERE vl.severity = 'critical' 
AND vr.violations_found > 0
ORDER BY vr.violations_found DESC;

-- ========================================
-- SECTION 13: DATA QUALITY SCORE
-- ========================================
SELECT '========== OVERALL DATA QUALITY SCORE ==========' AS '';

SELECT 
    CONCAT(
        ROUND(
            (1 - (SUM(CASE WHEN vr.violations_found > 0 THEN 1 ELSE 0 END) / COUNT(*))) * 100,
            2
        ),
        '%'
    ) AS Data_Quality_Score,
    COUNT(*) AS Total_Rules_Checked,
    SUM(CASE WHEN vr.violations_found = 0 THEN 1 ELSE 0 END) AS Rules_Passed,
    SUM(CASE WHEN vr.violations_found > 0 THEN 1 ELSE 0 END) AS Rules_Failed,
    SUM(vr.violations_found) AS Total_Violations_Found
FROM validation_rules vl
JOIN (
    SELECT rule_id, violations_found 
    FROM validation_results 
    WHERE result_id IN (
        SELECT MAX(result_id) 
        FROM validation_results 
        GROUP BY rule_id
    )
) vr ON vl.rule_id = vr.rule_id
WHERE vl.is_active = TRUE;

-- ========================================
-- FINAL MESSAGE
-- ========================================
SELECT '========================================' AS '';
SELECT 'Data Validation Completed' AS Result;
SELECT 'Review reports above for detailed findings' AS Note;
SELECT 'Address CRITICAL issues immediately' AS Action;
SELECT '========================================' AS '';