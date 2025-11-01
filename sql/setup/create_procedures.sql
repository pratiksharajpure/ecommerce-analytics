-- ========================================
-- CREATE STORED PROCEDURES - MYSQL 8.0 COMPATIBLE
-- Fixed for pymysql execution
-- ========================================

USE ecommerce_analytics;

-- Drop existing procedures
DROP PROCEDURE IF EXISTS sp_process_order;
DROP PROCEDURE IF EXISTS sp_update_inventory;
DROP PROCEDURE IF EXISTS sp_calculate_customer_ltv;
DROP PROCEDURE IF EXISTS sp_generate_sales_report;
DROP PROCEDURE IF EXISTS sp_identify_duplicate_customers;
DROP PROCEDURE IF EXISTS sp_validate_email_addresses;
DROP PROCEDURE IF EXISTS sp_check_inventory_levels;
DROP PROCEDURE IF EXISTS sp_process_returns;
DROP PROCEDURE IF EXISTS sp_update_loyalty_points;
DROP PROCEDURE IF EXISTS sp_cleanup_old_data;
DROP PROCEDURE IF EXISTS sp_refresh_materialized_views;
DROP PROCEDURE IF EXISTS sp_audit_data_quality;

-- ========================================
-- PROCEDURE 1: Process Order
-- ========================================
CREATE PROCEDURE sp_process_order(
    IN p_customer_id VARCHAR(20),
    IN p_order_id VARCHAR(20)
)
BEGIN
DECLARE v_total DECIMAL(10,2);
    DECLARE v_customer_exists INT;
    
    -- Check if customer exists
    SELECT COUNT(*) INTO v_customer_exists
    FROM customers
    WHERE customer_id = p_customer_id;
    
    IF v_customer_exists = 0 THEN
        SIGNAL SQLSTATE '45000'
        

    












SET MESSAGE_TEXT = 'Customer does not exist';
    END IF;
    
    -- Calculate order total
    SELECT COALESCE(SUM(quantity * unit_price - discount), 0)
    INTO v_total
    FROM order_items
    WHERE order_id = p_order_id;
    
    -- Update order
    UPDATE orders
    SET total_amount = v_total,
        status = 'processing',
        updated_at = CURRENT_TIMESTAMP
    WHERE order_id = p_order_id;
    
    SELECT v_total AS order_total;
END;

-- ========================================
-- PROCEDURE 2: Update Inventory
-- ========================================
CREATE PROCEDURE sp_update_inventory(
    IN p_product_id VARCHAR(20),
    IN p_quantity INT
)
BEGIN
DECLARE v_current_qty INT DEFAULT 0;
    
    -- Get current quantity
    SELECT COALESCE(SUM(quantity_on_hand), 0)
    INTO v_current_qty
    FROM inventory
    WHERE product_id = p_product_id;
    
    -- Update or insert inventory
    IF v_current_qty > 0 THEN
        UPDATE inventory
        

    












SET quantity_on_hand = quantity_on_hand + p_quantity,
            last_updated = CURRENT_TIMESTAMP
        WHERE product_id = p_product_id;
    ELSE
        INSERT INTO inventory (product_id, quantity_on_hand, warehouse_id)
        VALUES (p_product_id, p_quantity, 1);
    END IF;
    
    SELECT p_product_id AS product_id, 
           quantity_on_hand AS new_quantity
    FROM inventory
    WHERE product_id = p_product_id
    LIMIT 1;
END;

-- ========================================
-- PROCEDURE 3: Calculate Customer LTV
-- ========================================
CREATE PROCEDURE sp_calculate_customer_ltv(
    IN p_customer_id VARCHAR(20),
    OUT p_ltv DECIMAL(12,2)
)
BEGIN
DECLARE v_total_records INT DEFAULT 0;
    DECLARE v_invalid_count INT DEFAULT 0;
    
    -- Count total records
    SELECT COUNT(*) INTO v_total_records
    FROM customers;
    
    -- Count invalid emails
    SELECT COUNT(*) INTO v_invalid_count
    FROM customers
    WHERE email IS NULL 
       OR email = '' 
       OR email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$';
    
    -- Return results
    SELECT 
        v_total_records AS total_customers,
        v_invalid_count AS invalid_emails,
        (v_total_records - v_invalid_count) AS valid_emails,
        ROUND((v_invalid_count / v_total_records * 100), 2) AS invalid_percentage;
END;

-- ========================================
-- PROCEDURE 7: Check Inventory Levels
-- ========================================
CREATE PROCEDURE sp_check_inventory_levels()
BEGIN
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        COALESCE(SUM(i.quantity_on_hand), 0) AS total_on_hand,
        COALESCE(SUM(i.quantity_reserved), 0) AS total_reserved,
        COALESCE(SUM(i.quantity_available), 0) AS total_available,
        p.stock_quantity AS product_stock,
        CASE 
            WHEN COALESCE(SUM(i.quantity_available), 0) = 0 THEN 'OUT_OF_STOCK'
            WHEN COALESCE(SUM(i.quantity_available), 0) < 10 THEN 'LOW_STOCK'
            ELSE 'IN_STOCK'
        END AS stock_status
    FROM products p
    LEFT JOIN inventory i ON p.product_id = i.product_id
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, p.sku, p.stock_quantity
    HAVING stock_status != 'IN_STOCK'
    ORDER BY total_available ASC;
END;

-- ========================================
-- PROCEDURE 8: Process Returns
-- ========================================
CREATE PROCEDURE sp_process_returns(
    IN p_return_id INT
)
BEGIN
    DECLARE v_order_id VARCHAR(20);
    DECLARE v_refund_amount DECIMAL(10,2);
    
    -- Get return details
    SELECT order_id, refund_amount
    INTO v_order_id, v_refund_amount
    FROM returns
    WHERE return_id = p_return_id;
    
    -- Update return status
    UPDATE returns
    

    SELECT COALESCE(SUM(total_amount), 0)
    INTO p_ltv
    FROM orders
    WHERE customer_id = p_customer_id
    AND status != 'cancelled';
END;

-- ========================================
-- PROCEDURE 4: Generate Sales Report
-- ========================================
CREATE PROCEDURE sp_generate_sales_report(
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    SELECT 
        DATE(order_date) AS sale_date,
        COUNT(DISTINCT order_id) AS total_orders,
        COUNT(DISTINCT customer_id) AS unique_customers,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_order_value,
        MAX(total_amount) AS max_order_value,
        MIN(total_amount) AS min_order_value
    FROM orders
    WHERE order_date BETWEEN p_start_date AND p_end_date
    AND status != 'cancelled'
    GROUP BY DATE(order_date)
    ORDER BY sale_date;
END;

-- ========================================
-- PROCEDURE 5: Identify Duplicate Customers
-- ========================================
CREATE PROCEDURE sp_identify_duplicate_customers()
BEGIN
    SELECT 
        email,
        COUNT(*) AS duplicate_count,
        GROUP_CONCAT(customer_id) AS customer_ids
    FROM customers
    WHERE email IS NOT NULL AND email != ''
    GROUP BY email
    HAVING COUNT(*) > 1
    ORDER BY duplicate_count DESC;
END;

-- ========================================
-- PROCEDURE 6: Validate Email Addresses
-- ========================================
CREATE PROCEDURE sp_validate_email_addresses()
BEGIN
    












SET status = 'refunded',
        updated_at = CURRENT_TIMESTAMP
    WHERE return_id = p_return_id;
    
    -- Update order status
    UPDATE orders
    SET payment_status = 'refunded',
        updated_at = CURRENT_TIMESTAMP
    WHERE order_id = v_order_id;
    
    SELECT 
        p_return_id AS return_id,
        v_order_id AS order_id,
        v_refund_amount AS refund_amount,
        'Return processed successfully' AS message;
END;

-- ========================================
-- PROCEDURE 9: Update Loyalty Points
-- ========================================
CREATE PROCEDURE sp_update_loyalty_points(
    IN p_customer_id VARCHAR(20),
    IN p_points INT
)
BEGIN
DECLARE v_exists INT DEFAULT 0;
    
    -- Check if loyalty record exists
    SELECT COUNT(*) INTO v_exists
    FROM loyalty_program
    WHERE customer_id = p_customer_id;
    
    IF v_exists > 0 THEN
        -- Update existing record
        UPDATE loyalty_program
        

    












SET points_balance = points_balance + p_points,
            points_earned_lifetime = points_earned_lifetime + IF(p_points > 0, p_points, 0),
            points_redeemed_lifetime = points_redeemed_lifetime + IF(p_points < 0, ABS(p_points), 0),
            last_activity_date = CURRENT_TIMESTAMP
        WHERE customer_id = p_customer_id;
    ELSE
        -- Insert new record
        INSERT INTO loyalty_program (customer_id, points_balance, points_earned_lifetime, joined_date)
        VALUES (p_customer_id, GREATEST(p_points, 0), GREATEST(p_points, 0), CURRENT_DATE);
    END IF;
    
    SELECT 
        customer_id,
        points_balance,
        tier
    FROM loyalty_program
    WHERE customer_id = p_customer_id;
END;

-- ========================================
-- PROCEDURE 10: Cleanup Old Data
-- ========================================
CREATE PROCEDURE sp_cleanup_old_data(
    IN p_days_old INT
)
BEGIN
DECLARE v_cutoff_date DATE;
    DECLARE v_deleted_count INT DEFAULT 0;
    
    

    












SET v_cutoff_date = DATE_SUB(CURRENT_DATE, INTERVAL p_days_old DAY);
    
    -- Delete old campaign performance data
    DELETE FROM campaign_performance
    WHERE report_date < v_cutoff_date;
    
    SET v_deleted_count = ROW_COUNT();
    
    SELECT 
        v_cutoff_date AS cutoff_date,
        v_deleted_count AS records_deleted,
        'Cleanup completed successfully' AS message;
END;

-- ========================================
-- PROCEDURE 11: Refresh Materialized Views
-- ========================================
CREATE PROCEDURE sp_refresh_materialized_views()
BEGIN
DECLARE v_start_time TIMESTAMP;
    
    

    












SET v_start_time = CURRENT_TIMESTAMP;
    
    -- Refresh customer LTV view
    TRUNCATE TABLE mv_customer_ltv;
    
    INSERT INTO mv_customer_ltv 
        (customer_id, total_orders, total_spent, avg_order_value, 
         first_order_date, last_order_date, days_as_customer, ltv)
    SELECT 
        c.customer_id,
        COUNT(o.order_id) AS total_orders,
        COALESCE(SUM(o.total_amount), 0) AS total_spent,
        COALESCE(AVG(o.total_amount), 0) AS avg_order_value,
        MIN(DATE(o.order_date)) AS first_order_date,
        MAX(DATE(o.order_date)) AS last_order_date,
        DATEDIFF(CURRENT_DATE, MIN(DATE(o.order_date))) AS days_as_customer,
        COALESCE(SUM(o.total_amount), 0) AS ltv
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status != 'cancelled'
    GROUP BY c.customer_id;
    
    -- Refresh product performance view
    TRUNCATE TABLE mv_product_performance;
    
    INSERT INTO mv_product_performance
        (product_id, total_sold, total_revenue, avg_price, times_ordered, last_sold_date)
    SELECT 
        p.product_id,
        COALESCE(SUM(oi.quantity), 0) AS total_sold,
        COALESCE(SUM(oi.subtotal), 0) AS total_revenue,
        COALESCE(AVG(oi.unit_price), 0) AS avg_price,
        COUNT(DISTINCT oi.order_id) AS times_ordered,
        MAX(DATE(o.order_date)) AS last_sold_date
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id AND o.status != 'cancelled'
    GROUP BY p.product_id;
    
    SELECT 
        'Materialized views refreshed' AS message,
        v_start_time AS started_at,
        CURRENT_TIMESTAMP AS completed_at,
        TIMESTAMPDIFF(SECOND, v_start_time, CURRENT_TIMESTAMP) AS duration_seconds;
END;

-- ========================================
-- PROCEDURE 12: Audit Data Quality
-- ========================================
CREATE PROCEDURE sp_audit_data_quality()
BEGIN
DECLARE v_order_count INT DEFAULT 0;
    DECLARE v_total_spent DECIMAL(10,2) DEFAULT 0.00;
    DECLARE v_error_msg VARCHAR(500);
    
    -- Error handler
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1 v_error_msg = MESSAGE_TEXT;
        

    -- Create temp table for results
    CREATE TEMPORARY TABLE IF NOT EXISTS temp_quality_audit (
        check_name VARCHAR(100),
        table_name VARCHAR(50),
        total_records INT,
        issues_found INT,
        quality_score DECIMAL(5,2)
    );
    
    TRUNCATE TABLE temp_quality_audit;
    
    -- Check 1: Invalid emails
    INSERT INTO temp_quality_audit
    SELECT 
        'Invalid Email Format' AS check_name,
        'customers' AS table_name,
        COUNT(*) AS total_records,
        SUM(CASE WHEN email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$' THEN 1 ELSE 0 END) AS issues_found,
        ROUND((1 - SUM(CASE WHEN email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS quality_score
    FROM customers
    WHERE email IS NOT NULL;
    
    -- Check 2: Negative prices
    INSERT INTO temp_quality_audit
    SELECT 
        'Negative Prices' AS check_name,
        'products' AS table_name,
        COUNT(*) AS total_records,
        SUM(CASE WHEN price < 0 THEN 1 ELSE 0 END) AS issues_found,
        ROUND((1 - SUM(CASE WHEN price < 0 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS quality_score
    FROM products;
    
    -- Check 3: Order integrity
    INSERT INTO temp_quality_audit
    SELECT 
        'Order-Customer Orphans' AS check_name,
        'orders' AS table_name,
        COUNT(*) AS total_records,
        SUM(CASE WHEN c.customer_id IS NULL THEN 1 ELSE 0 END) AS issues_found,
        ROUND((1 - SUM(CASE WHEN c.customer_id IS NULL THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS quality_score
    FROM orders o
    LEFT JOIN customers c ON o.customer_id = c.customer_id;
    
    -- Return results
    SELECT * FROM temp_quality_audit
    ORDER BY quality_score ASC;
    
    -- Cleanup
    DROP TEMPORARY TABLE IF EXISTS temp_quality_audit;
END;
DROP PROCEDURE IF EXISTS sp_example_template;

DELIMITER $$

CREATE PROCEDURE sp_example_template(
    IN p_customer_id INT,
    OUT p_result_message VARCHAR(500),
    OUT p_success BOOLEAN
)
BEGIN
    -- Variable declarations (NOT @variables!)
    












SET p_result_message = CONCAT('Error: ', v_error_msg);
        SET p_success = FALSE;
        ROLLBACK;
    END;
    
    -- Main logic
    START TRANSACTION;
    
    -- Get order count
    SELECT COUNT(*), COALESCE(SUM(total_amount), 0)
    INTO v_order_count, v_total_spent
    FROM orders
    WHERE customer_id = p_customer_id;
    
    -- Set output
    SET p_result_message = CONCAT(
        'Customer has ', v_order_count, ' orders totaling $', v_total_spent
    );
    SET p_success = TRUE;
    
    COMMIT;
END$$

DELIMITER ;

-- Test the procedure
CALL sp_example_template(1, @msg, @success);
SELECT @msg as Result_Message, @success as Success;

-- Display confirmation
SELECT 'All stored procedures created successfully (MySQL 8.0 compatible)' AS Status;