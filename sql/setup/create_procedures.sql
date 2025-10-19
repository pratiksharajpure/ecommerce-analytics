-- ========================================
-- CREATE STORED PROCEDURES
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for procedure creation
DELIMITER //

-- ========================================
-- PROCEDURE 1: Process Order
-- ========================================
CREATE PROCEDURE sp_process_order(
    IN p_customer_id INT,
    IN p_order_items JSON,
    OUT p_order_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
    DECLARE v_product_id INT;
    DECLARE v_quantity INT;
    DECLARE v_unit_price DECIMAL(10,2);
    DECLARE v_total DECIMAL(10,2) DEFAULT 0;
    DECLARE v_available BOOLEAN;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SET p_status_message = 'Error processing order';
        SET p_order_id = NULL;
    END;
    
    START TRANSACTION;
    
    -- Create order record
    INSERT INTO orders (customer_id, order_date, total_amount, status, payment_status)
    VALUES (p_customer_id, NOW(), 0, 'pending', 'pending');
    
    SET p_order_id = LAST_INSERT_ID();
    
    -- Process each order item (JSON parsing logic would go here)
    -- This is a simplified version
    
    -- Update order total
    UPDATE orders 
    SET total_amount = fn_calculate_order_total(p_order_id)
    WHERE order_id = p_order_id;
    
    COMMIT;
    SET p_status_message = 'Order processed successfully';
END//

-- ========================================
-- PROCEDURE 2: Update Inventory
-- ========================================
CREATE PROCEDURE sp_update_inventory(
    IN p_product_id INT,
    IN p_quantity_change INT,
    IN p_warehouse_id INT,
    OUT p_status_message VARCHAR(255)
)
BEGIN
    DECLARE v_current_qty INT;
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SET p_status_message = 'Error updating inventory';
    END;
    
    START TRANSACTION;
    
    -- Get current quantity
    SELECT quantity_on_hand INTO v_current_qty
    FROM inventory
    WHERE product_id = p_product_id AND warehouse_id = p_warehouse_id;
    
    -- Update inventory
    IF v_current_qty IS NOT NULL THEN
        UPDATE inventory
        SET quantity_on_hand = quantity_on_hand + p_quantity_change,
            last_updated = NOW()
        WHERE product_id = p_product_id AND warehouse_id = p_warehouse_id;
        
        SET p_status_message = 'Inventory updated successfully';
    ELSE
        -- Insert new inventory record
        INSERT INTO inventory (product_id, warehouse_id, quantity_on_hand, last_updated)
        VALUES (p_product_id, p_warehouse_id, p_quantity_change, NOW());
        
        SET p_status_message = 'New inventory record created';
    END IF;
    
    COMMIT;
END//

-- ========================================
-- PROCEDURE 3: Calculate Quality Score
-- ========================================
CREATE PROCEDURE sp_calculate_quality_score(
    IN p_table_name VARCHAR(50),
    OUT p_quality_score DECIMAL(5,2),
    OUT p_issues_found INT
)
BEGIN
    DECLARE v_total_records INT DEFAULT 0;
    DECLARE v_valid_records INT DEFAULT 0;
    
    -- Example for customers table
    IF p_table_name = 'customers' THEN
        SELECT COUNT(*) INTO v_total_records FROM customers;
        
        SELECT COUNT(*) INTO v_valid_records
        FROM customers
        WHERE email IS NOT NULL 
        AND email != ''
        AND fn_validate_email(email) = TRUE
        AND phone IS NOT NULL
        AND phone != '';
        
    END IF;
    
    -- Calculate score
    IF v_total_records > 0 THEN
        SET p_quality_score = (v_valid_records * 100.0 / v_total_records);
        SET p_issues_found = v_total_records - v_valid_records;
    ELSE
        SET p_quality_score = 0;
        SET p_issues_found = 0;
    END IF;
END//

-- ========================================
-- PROCEDURE 4: Generate Report
-- ========================================
CREATE PROCEDURE sp_generate_report(
    IN p_report_type VARCHAR(50),
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    IF p_report_type = 'sales' THEN
        SELECT 
            DATE(order_date) AS report_date,
            COUNT(order_id) AS total_orders,
            SUM(total_amount) AS total_revenue,
            AVG(total_amount) AS average_order_value
        FROM orders
        WHERE order_date BETWEEN p_start_date AND p_end_date
        AND status != 'cancelled'
        GROUP BY DATE(order_date)
        ORDER BY report_date;
        
    ELSEIF p_report_type = 'customers' THEN
        SELECT 
            customer_id,
            CONCAT(first_name, ' ', last_name) AS customer_name,
            email,
            fn_get_customer_lifetime_value(customer_id) AS lifetime_value,
            status
        FROM customers
        ORDER BY lifetime_value DESC;
        
    ELSE
        SELECT 'Invalid report type' AS error_message;
    END IF;
END//

-- Reset delimiter
DELIMITER ;

-- Display confirmation
SELECT 'All stored procedures created successfully' AS Status;