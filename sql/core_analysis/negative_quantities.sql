-- ========================================
-- NEGATIVE_QUANTITIES.SQL
-- Data Quality Check: Invalid Quantity Values
-- Path: sql/core_analysis/negative_quantities.sql
-- Identifies negative, zero, or unrealistic quantity values
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Order Items with Negative or Zero Quantities
-- ========================================
SELECT 
    oi.order_item_id,
    oi.order_id,
    o.order_date,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    oi.product_id,
    p.sku,
    p.product_name,
    oi.quantity,
    oi.unit_price,
    oi.discount,
    oi.subtotal,
    o.status AS order_status,
    CASE 
        WHEN oi.quantity = 0 THEN 'Zero quantity'
        WHEN oi.quantity < 0 THEN 'Negative quantity'
    END AS issue_type
FROM order_items oi
INNER JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE oi.quantity <= 0
ORDER BY oi.quantity ASC, o.order_date DESC;

-- ========================================
-- 2. Order Items with Unrealistically High Quantities
-- ========================================
WITH quantity_stats AS (
    SELECT 
        product_id,
        AVG(quantity) AS avg_quantity,
        STDDEV(quantity) AS stddev_quantity,
        MAX(quantity) AS max_quantity
    FROM order_items
    WHERE quantity > 0
    GROUP BY product_id
    HAVING COUNT(*) >= 10
)
SELECT 
    oi.order_item_id,
    oi.order_id,
    o.order_date,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    p.product_id,
    p.sku,
    p.product_name,
    oi.quantity,
    ROUND(qs.avg_quantity, 2) AS avg_quantity_for_product,
    ROUND(qs.max_quantity, 0) AS historical_max_quantity,
    ROUND((oi.quantity - qs.avg_quantity) / NULLIF(qs.stddev_quantity, 0), 2) AS std_deviations,
    oi.unit_price,
    oi.subtotal,
    o.status AS order_status,
    'Unusually high quantity' AS issue_type
FROM order_items oi
INNER JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN products p ON oi.product_id = p.product_id
INNER JOIN quantity_stats qs ON oi.product_id = qs.product_id
WHERE oi.quantity > (qs.avg_quantity + 5 * qs.stddev_quantity)
    OR oi.quantity > 1000
ORDER BY oi.quantity DESC;

-- ========================================
-- 3. Inventory with Negative Stock Quantities
-- ========================================
SELECT 
    i.inventory_id,
    i.product_id,
    p.sku,
    p.product_name,
    i.warehouse_id,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_available,
    p.stock_quantity AS product_table_stock,
    CASE 
        WHEN i.quantity_on_hand < 0 THEN 'Negative quantity on hand'
        WHEN i.quantity_reserved < 0 THEN 'Negative reserved quantity'
        WHEN i.quantity_available < 0 THEN 'Negative available quantity'
    END AS issue_type,
    i.last_updated
FROM inventory i
INNER JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_on_hand < 0 
    OR i.quantity_reserved < 0 
    OR i.quantity_available < 0
ORDER BY i.quantity_on_hand ASC;

-- ========================================
-- 4. Products with Negative Stock Quantity
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.stock_quantity,
    p.price,
    pc.category_name,
    p.status,
    COUNT(DISTINCT oi.order_id) AS recent_orders,
    SUM(oi.quantity) AS recent_quantity_sold,
    'Negative stock quantity' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id 
    AND oi.created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
WHERE p.stock_quantity < 0
GROUP BY p.product_id, p.sku, p.product_name, p.stock_quantity, 
         p.price, pc.category_name, p.status
ORDER BY p.stock_quantity ASC;

-- ========================================
-- 5. Quantity vs Subtotal Mismatches
-- ========================================
SELECT 
    oi.order_item_id,
    oi.order_id,
    o.order_date,
    p.product_id,
    p.sku,
    p.product_name,
    oi.quantity,
    oi.unit_price,
    oi.discount,
    oi.subtotal AS recorded_subtotal,
    ROUND((oi.quantity * oi.unit_price - oi.discount), 2) AS calculated_subtotal,
    ROUND(oi.subtotal - (oi.quantity * oi.unit_price - oi.discount), 2) AS discrepancy,
    o.status AS order_status,
    'Subtotal calculation error' AS issue_type
FROM order_items oi
INNER JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE ABS(oi.subtotal - (oi.quantity * oi.unit_price - oi.discount)) > 0.01
ORDER BY ABS(discrepancy) DESC;

-- ========================================
-- 6. Reserved Quantity Exceeds On-Hand Quantity
-- ========================================
SELECT 
    i.inventory_id,
    i.product_id,
    p.sku,
    p.product_name,
    i.warehouse_id,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_available,
    (i.quantity_reserved - i.quantity_on_hand) AS overcommitted_by,
    'Reserved quantity exceeds on-hand' AS issue_type,
    i.last_updated
FROM inventory i
INNER JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_reserved > i.quantity_on_hand
ORDER BY (i.quantity_reserved - i.quantity_on_hand) DESC;

-- ========================================
-- 7. Stock Quantity Discrepancies Across Tables
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.stock_quantity AS products_table_stock,
    COALESCE(SUM(i.quantity_on_hand), 0) AS inventory_table_total,
    ROUND(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0), 0) AS discrepancy,
    COUNT(i.inventory_id) AS warehouse_count,
    'Stock quantity mismatch between tables' AS issue_type
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
WHERE p.status = 'active'
GROUP BY p.product_id, p.sku, p.product_name, p.stock_quantity
HAVING ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0)) > 0
ORDER BY ABS(discrepancy) DESC;

-- ========================================
-- 8. Orders with Excessive Discounts
-- ========================================
SELECT 
    oi.order_item_id,
    oi.order_id,
    o.order_date,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    p.product_id,
    p.sku,
    p.product_name,
    oi.quantity,
    oi.unit_price,
    oi.discount,
    ROUND((oi.discount / (oi.quantity * oi.unit_price)) * 100, 2) AS discount_percentage,
    oi.subtotal,
    CASE 
        WHEN oi.discount > (oi.quantity * oi.unit_price) THEN 'Discount exceeds item value'
        WHEN oi.discount < 0 THEN 'Negative discount'
        WHEN (oi.discount / (oi.quantity * oi.unit_price)) > 0.75 THEN 'Discount over 75%'
    END AS issue_type
FROM order_items oi
INNER JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE oi.discount > (oi.quantity * oi.unit_price) 
    OR oi.discount < 0
    OR (oi.discount / NULLIF(oi.quantity * oi.unit_price, 0)) > 0.75
ORDER BY discount_percentage DESC;

-- ========================================
-- 9. Quantity Anomalies Summary Report
-- ========================================
SELECT 
    'Order Items with Zero/Negative Qty' AS metric,
    COUNT(*) AS count,
    COALESCE(SUM(oi.subtotal), 0) AS revenue_impact
FROM order_items oi
WHERE oi.quantity <= 0

UNION ALL

SELECT 
    'Products with Negative Stock' AS metric,
    COUNT(*) AS count,
    COALESCE(SUM(p.price * ABS(p.stock_quantity)), 0) AS revenue_impact
FROM products p
WHERE p.stock_quantity < 0

UNION ALL

SELECT 
    'Inventory with Negative Quantities' AS metric,
    COUNT(*) AS count,
    NULL AS revenue_impact
FROM inventory i
WHERE i.quantity_on_hand < 0 OR i.quantity_reserved < 0

UNION ALL

SELECT 
    'Reserved Qty Exceeds On-Hand' AS metric,
    COUNT(*) AS count,
    NULL AS revenue_impact
FROM inventory i
WHERE i.quantity_reserved > i.quantity_on_hand

UNION ALL

SELECT 
    'Quantity/Subtotal Mismatches' AS metric,
    COUNT(*) AS count,
    SUM(ABS(oi.subtotal - (oi.quantity * oi.unit_price - oi.discount))) AS revenue_impact
FROM order_items oi
WHERE ABS(oi.subtotal - (oi.quantity * oi.unit_price - oi.discount)) > 0.01

UNION ALL

SELECT 
    'Excessive Discounts' AS metric,
    COUNT(*) AS count,
    SUM(oi.discount) AS revenue_impact
FROM order_items oi
WHERE oi.discount > (oi.quantity * oi.unit_price) 
    OR oi.discount < 0
    OR (oi.discount / NULLIF(oi.quantity * oi.unit_price, 0)) > 0.75;

-- ========================================
-- 10. Products Oversold (Orders Exceed Stock)
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.stock_quantity AS current_stock,
    COUNT(DISTINCT oi.order_id) AS pending_orders,
    SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END) AS pending_quantity,
    (p.stock_quantity - SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END)) AS remaining_stock,
    ABS(p.stock_quantity - SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END)) AS oversold_by,
    'Oversold - insufficient stock' AS issue_type
FROM products p
INNER JOIN order_items oi ON p.product_id = oi.product_id
INNER JOIN orders o ON oi.order_id = o.order_id
WHERE o.status IN ('pending', 'processing')
GROUP BY p.product_id, p.sku, p.product_name, p.stock_quantity
HAVING p.stock_quantity < SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END)
ORDER BY oversold_by DESC;

-- ========================================
-- 11. Bulk Order Anomalies
-- ========================================
SELECT 
    oi.order_item_id,
    oi.order_id,
    o.order_date,
    o.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    p.product_id,
    p.sku,
    p.product_name,
    oi.quantity,
    p.stock_quantity AS available_stock,
    oi.unit_price,
    oi.subtotal,
    ROUND((oi.quantity * 100.0) / NULLIF(p.stock_quantity, 0), 2) AS pct_of_stock,
    'Single order depleting majority of stock' AS issue_type
FROM order_items oi
INNER JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN products p ON oi.product_id = p.product_id
WHERE oi.quantity >= 100
    AND oi.quantity >= (p.stock_quantity * 0.50)
    AND o.status IN ('pending', 'processing')
ORDER BY oi.quantity DESC;