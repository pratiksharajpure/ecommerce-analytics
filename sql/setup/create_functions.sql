-- ========================================
-- CREATE USER-DEFINED FUNCTIONS
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for function creation
DELIMITER //

-- ========================================
-- FUNCTION 1: Calculate Order Total
-- ========================================
CREATE FUNCTION fn_calculate_order_total(p_order_id INT)
RETURNS DECIMAL(10,2)
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_total DECIMAL(10,2);
    
    SELECT COALESCE(SUM(quantity * unit_price - discount), 0)
    INTO v_total
    FROM order_items
    WHERE order_id = p_order_id;
    
    RETURN v_total;
END//

-- ========================================
-- FUNCTION 2: Get Customer Lifetime Value
-- ========================================
CREATE FUNCTION fn_get_customer_lifetime_value(p_customer_id INT)
RETURNS DECIMAL(10,2)
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_lifetime_value DECIMAL(10,2);
    
    SELECT COALESCE(SUM(total_amount), 0)
    INTO v_lifetime_value
    FROM orders
    WHERE customer_id = p_customer_id
    AND status != 'cancelled';
    
    RETURN v_lifetime_value;
END//

-- ========================================
-- FUNCTION 3: Check Inventory Availability
-- ========================================
CREATE FUNCTION fn_check_inventory_availability(p_product_id INT, p_quantity INT)
RETURNS BOOLEAN
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_available INT;
    
    SELECT COALESCE(SUM(quantity_available), 0)
    INTO v_available
    FROM inventory
    WHERE product_id = p_product_id;
    
    RETURN (v_available >= p_quantity);
END//

-- ========================================
-- FUNCTION 4: Validate Email
-- ========================================
CREATE FUNCTION fn_validate_email(p_email VARCHAR(100))
RETURNS BOOLEAN
DETERMINISTIC
NO SQL
BEGIN
    DECLARE v_is_valid BOOLEAN;
    
    SET v_is_valid = (
        p_email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'
    );
    
    RETURN v_is_valid;
END//

-- ========================================
-- FUNCTION 5: Validate Phone
-- ========================================
CREATE FUNCTION fn_validate_phone(p_phone VARCHAR(20))
RETURNS BOOLEAN
DETERMINISTIC
NO SQL
BEGIN
    DECLARE v_is_valid BOOLEAN;
    DECLARE v_clean_phone VARCHAR(20);
    
    -- Remove all non-numeric characters
    SET v_clean_phone = REGEXP_REPLACE(p_phone, '[^0-9]', '');
    
    -- Check if cleaned phone has 10 digits (US format)
    SET v_is_valid = (LENGTH(v_clean_phone) = 10);
    
    RETURN v_is_valid;
END//

-- Reset delimiter
DELIMITER ;

-- Display confirmation
SELECT 'All user-defined functions created successfully' AS Status;