-- ========================================
-- INSERT SAMPLE INVENTORY
-- Inventory levels for all products across warehouses
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- INVENTORY RECORDS
-- Create inventory entries for all 300 products
-- ========================================

-- Warehouse 1 (Main) - All products
INSERT INTO inventory (product_id, warehouse_id, quantity_on_hand, quantity_reserved, reorder_level)
SELECT 
    p.product_id,
    1 AS warehouse_id,
    -- Quantity based on product stock, with some variation
    CASE 
        WHEN p.stock_quantity > 300 THEN FLOOR(p.stock_quantity * (0.6 + RAND() * 0.3))
        WHEN p.stock_quantity > 100 THEN FLOOR(p.stock_quantity * (0.5 + RAND() * 0.4))
        ELSE FLOOR(p.stock_quantity * (0.4 + RAND() * 0.5))
    END AS quantity_on_hand,
    -- Reserved quantity (pending orders)
    CASE 
        WHEN p.stock_quantity > 200 THEN FLOOR(RAND() * 30)
        WHEN p.stock_quantity > 50 THEN FLOOR(RAND() * 15)
        ELSE FLOOR(RAND() * 5)
    END AS quantity_reserved,
    -- Reorder level based on popularity
    CASE 
        WHEN p.price < 50 THEN 50
        WHEN p.price < 100 THEN 30
        WHEN p.price < 300 THEN 20
        ELSE 10
    END AS reorder_level
FROM products p;

-- Warehouse 2 (East Coast) - 70% of products
INSERT INTO inventory (product_id, warehouse_id, quantity_on_hand, quantity_reserved, reorder_level)
SELECT 
    p.product_id,
    2 AS warehouse_id,
    FLOOR(p.stock_quantity * (0.2 + RAND() * 0.2)) AS quantity_on_hand,
    FLOOR(RAND() * 10) AS quantity_reserved,
    CASE 
        WHEN p.price < 50 THEN 30
        WHEN p.price < 100 THEN 20
        WHEN p.price < 300 THEN 15
        ELSE 5
    END AS reorder_level
FROM products p
WHERE p.product_id % 10 <= 6;

-- Warehouse 3 (West Coast) - 70% of products
INSERT INTO inventory (product_id, warehouse_id, quantity_on_hand, quantity_reserved, reorder_level)
SELECT 
    p.product_id,
    3 AS warehouse_id,
    FLOOR(p.stock_quantity * (0.15 + RAND() * 0.25)) AS quantity_on_hand,
    FLOOR(RAND() * 12) AS quantity_reserved,
    CASE 
        WHEN p.price < 50 THEN 30
        WHEN p.price < 100 THEN 20
        WHEN p.price < 300 THEN 15
        ELSE 5
    END AS reorder_level
FROM products p
WHERE p.product_id % 10 <= 6;

-- Warehouse 4 (Central) - 50% of products (popular items)
INSERT INTO inventory (product_id, warehouse_id, quantity_on_hand, quantity_reserved, reorder_level)
SELECT 
    p.product_id,
    4 AS warehouse_id,
    FLOOR(p.stock_quantity * (0.1 + RAND() * 0.15)) AS quantity_on_hand,
    FLOOR(RAND() * 8) AS quantity_reserved,
    CASE 
        WHEN p.price < 50 THEN 25
        WHEN p.price < 100 THEN 15
        ELSE 10
    END AS reorder_level
FROM products p
WHERE p.stock_quantity > 100;

-- ========================================
-- CREATE LOW STOCK SITUATIONS
-- Intentionally set some items to low/out of stock
-- ========================================

-- Set 5% of inventory to critical low levels (below reorder point)
UPDATE inventory 
SET quantity_on_hand = FLOOR(reorder_level * (0.3 + RAND() * 0.4))
WHERE inventory_id % 20 = 0;

-- Set 3% to out of stock
UPDATE inventory 
SET quantity_on_hand = 0,
    quantity_reserved = 0
WHERE inventory_id % 33 = 0;

-- Set some high-demand items with high reserved quantities
UPDATE inventory 
SET quantity_reserved = FLOOR(quantity_on_hand * (0.4 + RAND() * 0.3))
WHERE inventory_id % 15 = 0
AND quantity_on_hand > 50;

-- ========================================
-- UPDATE PRODUCT STOCK QUANTITIES
-- Sync products table with total inventory
-- ========================================

UPDATE products p
SET stock_quantity = (
    SELECT COALESCE(SUM(i.quantity_on_hand), 0)
    FROM inventory i
    WHERE i.product_id = p.product_id
);

-- Update product status based on stock
UPDATE products 
SET status = 'out_of_stock'
WHERE stock_quantity = 0;

UPDATE products 
SET status = 'active'
WHERE stock_quantity > 0 
AND status = 'out_of_stock';

-- ========================================
-- UPDATE TIMESTAMPS
-- Vary last_updated times for realism
-- ========================================

UPDATE inventory 
SET last_updated = DATE_SUB(NOW(), INTERVAL FLOOR(RAND() * 30) DAY)
WHERE RAND() < 0.3;

UPDATE inventory 
SET last_updated = DATE_SUB(NOW(), INTERVAL FLOOR(RAND() * 7) DAY)
WHERE RAND() < 0.5
AND last_updated = CURRENT_TIMESTAMP;

-- ========================================
-- DISPLAY CONFIRMATION & STATISTICS
-- ========================================

SELECT 'Inventory data inserted successfully!' AS Status;

SELECT 
    COUNT(*) AS total_inventory_records,
    COUNT(DISTINCT product_id) AS products_with_inventory,
    COUNT(DISTINCT warehouse_id) AS active_warehouses,
    SUM(quantity_on_hand) AS total_units_on_hand,
    SUM(quantity_reserved) AS total_units_reserved,
    SUM(quantity_available) AS total_units_available
FROM inventory;

SELECT 'Inventory health by warehouse:' AS Info;
SELECT 
    warehouse_id,
    COUNT(*) AS product_count,
    SUM(quantity_on_hand) AS total_on_hand,
    SUM(quantity_reserved) AS total_reserved,
    SUM(quantity_available) AS total_available,
    SUM(CASE WHEN quantity_available < reorder_level THEN 1 ELSE 0 END) AS items_below_reorder,
    SUM(CASE WHEN quantity_on_hand = 0 THEN 1 ELSE 0 END) AS out_of_stock_items
FROM inventory
GROUP BY warehouse_id
ORDER BY warehouse_id;

SELECT 'Top 10 products by total inventory:' AS Info;
SELECT 
    p.product_id,
    p.product_name,
    p.sku,
    SUM(i.quantity_on_hand) AS total_inventory,
    SUM(i.quantity_available) AS total_available,
    COUNT(i.inventory_id) AS warehouse_count
FROM products p
JOIN inventory i ON p.product_id = i.product_id
GROUP BY p.product_id, p.product_name, p.sku
ORDER BY total_inventory DESC
LIMIT 10;

SELECT 'Products needing restock (below reorder level):' AS Info;
SELECT 
    p.product_id,
    p.product_name,
    i.warehouse_id,
    i.quantity_available,
    i.reorder_level,
    (i.reorder_level - i.quantity_available) AS units_to_reorder
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE i.quantity_available < i.reorder_level
ORDER BY (i.reorder_level - i.quantity_available) DESC
LIMIT 20;