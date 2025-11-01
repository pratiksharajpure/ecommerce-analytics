-- ========================================
-- MARIADB-COMPATIBLE E-COMMERCE SQL SETUP
-- Fixed for MariaDB (XAMPP) compatibility
-- Run AFTER CSV data is loaded
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- PART 1: CREATE ESSENTIAL VIEWS
-- ========================================

-- Customer Summary View
CREATE OR REPLACE VIEW v_customer_summary AS
SELECT 
    c.customer_id,
    c.customer_name,
    c.email,
    COUNT(DISTINCT o.order_id) as total_orders,
    COALESCE(SUM(o.total_amount), 0) as lifetime_value,
    MAX(o.order_date) as last_order_date,
    c.status
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name, c.email, c.status;

-- Product Performance View
CREATE OR REPLACE VIEW v_product_performance AS
SELECT 
    p.product_id,
    p.product_name,
    p.price,
    p.cost,
    (p.price - p.cost) as profit_per_unit,
    p.stock_quantity,
    COUNT(DISTINCT oi.order_id) as times_ordered,
    COALESCE(SUM(oi.quantity), 0) as total_units_sold,
    COALESCE(SUM(oi.quantity * oi.unit_price), 0) as total_revenue
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.price, p.cost, p.stock_quantity;

-- Order Summary View
CREATE OR REPLACE VIEW v_order_summary AS
SELECT 
    o.order_id,
    o.customer_id,
    c.customer_name,
    o.order_date,
    o.total_amount,
    o.status,
    o.payment_status,
    COUNT(oi.order_item_id) as item_count
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.customer_id, c.customer_name, o.order_date, 
         o.total_amount, o.status, o.payment_status;

-- Inventory Alert View
CREATE OR REPLACE VIEW v_inventory_alerts AS
SELECT 
    i.inventory_id,
    i.product_id,
    p.product_name,
    i.quantity_on_hand,
    i.quantity_reserved,
    (i.quantity_on_hand - i.quantity_reserved) as available_quantity,
    i.reorder_level,
    CASE 
        WHEN (i.quantity_on_hand - i.quantity_reserved) <= 0 THEN 'OUT_OF_STOCK'
        WHEN (i.quantity_on_hand - i.quantity_reserved) <= i.reorder_level THEN 'LOW_STOCK'
        ELSE 'IN_STOCK'
    END as stock_status
FROM inventory i
JOIN products p ON i.product_id = p.product_id;

-- Daily Sales Summary View
CREATE OR REPLACE VIEW v_daily_sales AS
SELECT 
    DATE(order_date) as sale_date,
    COUNT(DISTINCT order_id) as order_count,
    COUNT(DISTINCT customer_id) as customer_count,
    SUM(total_amount) as daily_revenue,
    AVG(total_amount) as avg_order_value
FROM orders
WHERE status != 'cancelled'
GROUP BY DATE(order_date);

-- ========================================
-- PART 2: CREATE USEFUL STORED PROCEDURES
-- (MariaDB-compatible syntax)
-- ========================================

DELIMITER //

-- Get Customer Orders
CREATE OR REPLACE PROCEDURE sp_get_customer_orders(
    IN p_customer_id VARCHAR(20)
)
BEGIN
    SELECT 
        o.order_id,
        o.order_date,
        o.total_amount,
        o.status,
        o.payment_status,
        COUNT(oi.order_item_id) as item_count
    FROM orders o
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.customer_id = p_customer_id
    GROUP BY o.order_id, o.order_date, o.total_amount, o.status, o.payment_status
    ORDER BY o.order_date DESC;
END//

-- Get Low Stock Products
CREATE OR REPLACE PROCEDURE sp_get_low_stock_products()
BEGIN
    SELECT 
        p.product_id,
        p.product_name,
        p.price,
        i.quantity_on_hand,
        i.quantity_reserved,
        (i.quantity_on_hand - i.quantity_reserved) as available,
        i.reorder_level
    FROM products p
    JOIN inventory i ON p.product_id = i.product_id
    WHERE (i.quantity_on_hand - i.quantity_reserved) <= i.reorder_level
    ORDER BY available ASC;
END//

-- Calculate Customer Lifetime Value
CREATE OR REPLACE PROCEDURE sp_calculate_customer_ltv(
    IN p_customer_id VARCHAR(20),
    OUT p_lifetime_value DECIMAL(10,2),
    OUT p_order_count INT
)
BEGIN
    SELECT 
        COALESCE(SUM(total_amount), 0),
        COUNT(order_id)
    INTO p_lifetime_value, p_order_count
    FROM orders
    WHERE customer_id = p_customer_id
    AND status != 'cancelled';
END//

-- Get Top Selling Products
CREATE OR REPLACE PROCEDURE sp_get_top_products(
    IN p_limit INT
)
BEGIN
    SELECT 
        p.product_id,
        p.product_name,
        p.price,
        COUNT(DISTINCT oi.order_id) as order_count,
        SUM(oi.quantity) as total_sold,
        SUM(oi.quantity * oi.unit_price) as revenue
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    GROUP BY p.product_id, p.product_name, p.price
    ORDER BY total_sold DESC
    LIMIT p_limit;
END//

-- Get Sales by Date Range
CREATE OR REPLACE PROCEDURE sp_sales_by_date_range(
    IN p_start_date DATE,
    IN p_end_date DATE
)
BEGIN
    SELECT 
        DATE(order_date) as sale_date,
        COUNT(order_id) as orders,
        SUM(total_amount) as revenue,
        AVG(total_amount) as avg_order_value
    FROM orders
    WHERE DATE(order_date) BETWEEN p_start_date AND p_end_date
    AND status != 'cancelled'
    GROUP BY DATE(order_date)
    ORDER BY sale_date;
END//

DELIMITER ;

-- ========================================
-- PART 3: CREATE USEFUL INDEXES
-- (Only if they don't exist)
-- ========================================

-- Check and create indexes
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory(product_id);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);

-- ========================================
-- VERIFICATION QUERIES
-- ========================================

-- Show what we created
SELECT 'Views created:' as info;
SHOW FULL TABLES WHERE Table_type = 'VIEW';

SELECT 'Procedures created:' as info;
SHOW PROCEDURE STATUS WHERE Db = 'ecommerce_analytics';

SELECT 'Indexes created:' as info;
SHOW INDEX FROM orders;

SELECT ''SUCCESS' Setup complete!' as status;
