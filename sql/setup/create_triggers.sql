-- ========================================
-- CREATE TRIGGERS
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- Set delimiter for trigger creation
DELIMITER //

-- ========================================
-- TRIGGER 1: Update Inventory After Order
-- ========================================
CREATE TRIGGER trg_update_inventory_after_order
AFTER INSERT ON order_items
FOR EACH ROW
BEGIN
    -- Reduce inventory quantity when order item is created
    UPDATE inventory
    SET quantity_reserved = quantity_reserved + NEW.quantity
    WHERE product_id = NEW.product_id
    LIMIT 1;
    
    -- Update product stock quantity
    UPDATE products
    SET stock_quantity = stock_quantity - NEW.quantity
    WHERE product_id = NEW.product_id;
END//

-- ========================================
-- TRIGGER 2: Audit Customer Changes
-- ========================================
-- First create audit table
CREATE TABLE IF NOT EXISTS customer_audit (
    audit_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    action_type ENUM('INSERT', 'UPDATE', 'DELETE'),
    old_email VARCHAR(100),
    new_email VARCHAR(100),
    old_phone VARCHAR(20),
    new_phone VARCHAR(20),
    old_status VARCHAR(20),
    new_status VARCHAR(20),
    changed_by VARCHAR(50),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

-- Trigger for customer updates
CREATE TRIGGER trg_audit_customer_update
AFTER UPDATE ON customers
FOR EACH ROW
BEGIN
    INSERT INTO customer_audit (
        customer_id,
        action_type,
        old_email,
        new_email,
        old_phone,
        new_phone,
        old_status,
        new_status,
        changed_by
    ) VALUES (
        NEW.customer_id,
        'UPDATE',
        OLD.email,
        NEW.email,
        OLD.phone,
        NEW.phone,
        OLD.status,
        NEW.status,
        USER()
    );
END//

-- ========================================
-- TRIGGER 3: Validate Order Total
-- ========================================
CREATE TRIGGER trg_validate_order_total
BEFORE INSERT ON orders
FOR EACH ROW
BEGIN
    -- Ensure total amount is not negative
    IF NEW.total_amount < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Order total amount cannot be negative';
    END IF;
    
    -- Ensure shipping cost is not negative
    IF NEW.shipping_cost < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Shipping cost cannot be negative';
    END IF;
    
    -- Ensure tax amount is not negative
    IF NEW.tax_amount < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Tax amount cannot be negative';
    END IF;
END//

-- ========================================
-- TRIGGER 4: Update Order Total After Item Changes
-- ========================================
CREATE TRIGGER trg_update_order_total_after_item
AFTER INSERT ON order_items
FOR EACH ROW
BEGIN
    -- Recalculate order total
    UPDATE orders
    SET total_amount = (
        SELECT COALESCE(SUM(quantity * unit_price - discount), 0)
        FROM order_items
        WHERE order_id = NEW.order_id
    )
    WHERE order_id = NEW.order_id;
END//

-- ========================================
-- TRIGGER 5: Prevent Negative Inventory
-- ========================================
CREATE TRIGGER trg_prevent_negative_inventory
BEFORE UPDATE ON inventory
FOR EACH ROW
BEGIN
    -- Prevent negative quantities
    IF NEW.quantity_on_hand < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Inventory quantity cannot be negative';
    END IF;
    
    IF NEW.quantity_reserved < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Reserved quantity cannot be negative';
    END IF;
END//

-- ========================================
-- TRIGGER 6: Update Product Rating After Review
-- ========================================
-- Add average_rating column to products if not exists
ALTER TABLE products ADD COLUMN IF NOT EXISTS average_rating DECIMAL(3,2) DEFAULT 0.00//
ALTER TABLE products ADD COLUMN IF NOT EXISTS review_count INT DEFAULT 0//

CREATE TRIGGER trg_update_product_rating_after_review
AFTER INSERT ON reviews
FOR EACH ROW
BEGIN
    UPDATE products p
    SET 
        review_count = (SELECT COUNT(*) FROM reviews WHERE product_id = NEW.product_id AND status = 'approved'),
        average_rating = (SELECT AVG(rating) FROM reviews WHERE product_id = NEW.product_id AND status = 'approved')
    WHERE p.product_id = NEW.product_id;
END//

-- ========================================
-- TRIGGER 7: Log Inventory Changes
-- ========================================
-- Create inventory log table
CREATE TABLE IF NOT EXISTS inventory_log (
    log_id INT PRIMARY KEY AUTO_INCREMENT,
    product_id INT,
    warehouse_id INT,
    old_quantity INT,
    new_quantity INT,
    change_amount INT,
    change_type VARCHAR(50),
    changed_by VARCHAR(50),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci//

CREATE TRIGGER trg_log_inventory_changes
AFTER UPDATE ON inventory
FOR EACH ROW
BEGIN
    INSERT INTO inventory_log (
        product_id,
        warehouse_id,
        old_quantity,
        new_quantity,
        change_amount,
        change_type,
        changed_by
    ) VALUES (
        NEW.product_id,
        NEW.warehouse_id,
        OLD.quantity_on_hand,
        NEW.quantity_on_hand,
        NEW.quantity_on_hand - OLD.quantity_on_hand,
        CASE 
            WHEN NEW.quantity_on_hand > OLD.quantity_on_hand THEN 'INCREASE'
            WHEN NEW.quantity_on_hand < OLD.quantity_on_hand THEN 'DECREASE'
            ELSE 'NO_CHANGE'
        END,
        USER()
    );
END//

-- Reset delimiter
DELIMITER ;

-- Display confirmation
SELECT 'All triggers created successfully' AS Status;