-- ========================================
-- CREATE TRIGGERS - MYSQL 8.0 COMPATIBLE
-- Fixed for pymysql execution
-- ========================================

USE ecommerce_analytics;

-- Drop existing triggers
DROP TRIGGER IF EXISTS trg_orders_after_insert;
DROP TRIGGER IF EXISTS trg_orders_after_update;
DROP TRIGGER IF EXISTS trg_order_items_after_insert;
DROP TRIGGER IF EXISTS trg_inventory_after_update;
DROP TRIGGER IF EXISTS trg_customers_after_insert;
DROP TRIGGER IF EXISTS trg_products_before_insert;
DROP TRIGGER IF EXISTS trg_products_before_update;
DROP TRIGGER IF EXISTS trg_reviews_after_insert;
DROP TRIGGER IF EXISTS trg_returns_after_insert;

-- ========================================
-- TRIGGER 1: Update order timestamp after insert
-- ========================================
CREATE TRIGGER trg_orders_after_insert
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    -- Update customer last activity
    UPDATE customers
    SET updated_at = CURRENT_TIMESTAMP
    WHERE customer_id = NEW.customer_id;
END;

-- ========================================
-- TRIGGER 2: Update order timestamp after update
-- ========================================
CREATE TRIGGER trg_orders_after_update
AFTER UPDATE ON orders
FOR EACH ROW
BEGIN
    -- Update customer last activity if order status changed
    IF OLD.status != NEW.status THEN
        UPDATE customers
        SET updated_at = CURRENT_TIMESTAMP
        WHERE customer_id = NEW.customer_id;
    END IF;
END;

-- ========================================
-- TRIGGER 3: Update inventory after order item insert
-- ========================================
CREATE TRIGGER trg_order_items_after_insert
AFTER INSERT ON order_items
FOR EACH ROW
BEGIN
    -- Reserve inventory
    UPDATE inventory
    SET quantity_reserved = quantity_reserved + NEW.quantity,
        last_updated = CURRENT_TIMESTAMP
    WHERE product_id = NEW.product_id
    LIMIT 1;
END;

-- ========================================
-- TRIGGER 4: Update product stock after inventory change
-- ========================================
CREATE TRIGGER trg_inventory_after_update
AFTER UPDATE ON inventory
FOR EACH ROW
BEGIN
    -- Sync product stock quantity
    UPDATE products
    SET stock_quantity = (
        SELECT COALESCE(SUM(quantity_on_hand), 0)
        FROM inventory
        WHERE product_id = NEW.product_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE product_id = NEW.product_id;
END;

-- ========================================
-- TRIGGER 5: Initialize customer after insert
-- ========================================
CREATE TRIGGER trg_customers_after_insert
AFTER INSERT ON customers
FOR EACH ROW
BEGIN
    -- Create loyalty program entry
    INSERT IGNORE INTO loyalty_program (
        customer_id,
        points_balance,
        joined_date,
        tier
    ) VALUES (
        NEW.customer_id,
        0,
        CURRENT_DATE,
        'bronze'
    );
END;

-- ========================================
-- TRIGGER 6: Validate product before insert
-- ========================================
CREATE TRIGGER trg_products_before_insert
BEFORE INSERT ON products
FOR EACH ROW
BEGIN
    -- Ensure price is not negative
    IF NEW.price < 0 THEN
        SET NEW.price = 0;
    END IF;
    
    -- Ensure cost is not negative
    IF NEW.cost < 0 THEN
        SET NEW.cost = 0;
    END IF;
    
    -- Ensure stock is not negative
    IF NEW.stock_quantity < 0 THEN
        SET NEW.stock_quantity = 0;
    END IF;
END;

-- ========================================
-- TRIGGER 7: Validate product before update
-- ========================================
CREATE TRIGGER trg_products_before_update
BEFORE UPDATE ON products
FOR EACH ROW
BEGIN
    -- Ensure price is not negative
    IF NEW.price < 0 THEN
        SET NEW.price = 0;
    END IF;
    
    -- Ensure cost is not negative
    IF NEW.cost < 0 THEN
        SET NEW.cost = 0;
    END IF;
    
    -- Ensure stock is not negative
    IF NEW.stock_quantity < 0 THEN
        SET NEW.stock_quantity = 0;
    END IF;
    
    -- Update status based on stock
    IF NEW.stock_quantity = 0 AND OLD.stock_quantity > 0 THEN
        SET NEW.status = 'out_of_stock';
    ELSEIF NEW.stock_quantity > 0 AND OLD.status = 'out_of_stock' THEN
        SET NEW.status = 'active';
    END IF;
END;

-- ========================================
-- TRIGGER 8: Update product rating after review insert
-- ========================================
CREATE TRIGGER trg_reviews_after_insert
AFTER INSERT ON reviews
FOR EACH ROW
BEGIN
    -- Update product statistics (if needed in future)
    UPDATE products
    SET updated_at = CURRENT_TIMESTAMP
    WHERE product_id = NEW.product_id;
END;

-- ========================================
-- TRIGGER 9: Process return request
-- ========================================
CREATE TRIGGER trg_returns_after_insert
AFTER INSERT ON returns
FOR EACH ROW
BEGIN
    -- Update order status if return is requested
    UPDATE orders
    SET status = CASE 
        WHEN status = 'delivered' THEN 'processing'
        ELSE status
    END,
    updated_at = CURRENT_TIMESTAMP
    WHERE order_id = NEW.order_id;
END;

-- Display confirmation
SELECT 'All triggers created successfully (MySQL 8.0 compatible)' AS Status;