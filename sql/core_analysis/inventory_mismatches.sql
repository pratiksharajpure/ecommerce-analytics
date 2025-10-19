-- ========================================
-- INVENTORY_MISMATCHES.SQL
-- Data Quality Check: Stock Level Discrepancies
-- Path: sql/core_analysis/inventory_mismatches.sql
-- Identifies inventory discrepancies and stock management issues
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Products vs Inventory Table Mismatches
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.stock_quantity AS products_table_stock,
    COALESCE(SUM(i.quantity_on_hand), 0) AS inventory_table_stock,
    ROUND(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0), 0) AS discrepancy,
    COUNT(i.inventory_id) AS warehouse_locations,
    p.status,
    CASE 
        WHEN SUM(i.quantity_on_hand) IS NULL THEN 'Product not in inventory system'
        WHEN p.stock_quantity > SUM(i.quantity_on_hand) THEN 'Products table shows more stock'
        WHEN p.stock_quantity < SUM(i.quantity_on_hand) THEN 'Inventory table shows more stock'
    END AS issue_type
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
WHERE p.status = 'active'
GROUP BY p.product_id, p.sku, p.product_name, p.stock_quantity, p.status
HAVING ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0)) > 0
ORDER BY ABS(discrepancy) DESC;

-- ========================================
-- 2. Oversold Products
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.stock_quantity AS current_stock,
    COALESCE(SUM(i.quantity_on_hand), 0) AS warehouse_stock,
    SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END) AS committed_quantity,
    p.stock_quantity - SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END) AS available_after_commitment,
    ABS(p.stock_quantity - SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END)) AS oversold_quantity,
    COUNT(DISTINCT CASE WHEN o.status IN ('pending', 'processing') THEN o.order_id END) AS affected_orders,
    'Product oversold' AS issue_type
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
INNER JOIN order_items oi ON p.product_id = oi.product_id
INNER JOIN orders o ON oi.order_id = o.order_id
WHERE o.status IN ('pending', 'processing')
GROUP BY p.product_id, p.sku, p.product_name, p.stock_quantity
HAVING p.stock_quantity < SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END)
ORDER BY oversold_quantity DESC;

-- ========================================
-- 3. Reserved Quantity Exceeds Available Stock
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
    (i.quantity_reserved - i.quantity_on_hand) AS over_reserved_by,
    'Reserved exceeds on-hand inventory' AS issue_type,
    i.last_updated
FROM inventory i
INNER JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_reserved > i.quantity_on_hand
ORDER BY over_reserved_by DESC;

-- ========================================
-- 4. Calculated Available Quantity Mismatch
-- ========================================
SELECT 
    i.inventory_id,
    i.product_id,
    p.sku,
    p.product_name,
    i.warehouse_id,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_available AS stored_available,
    (i.quantity_on_hand - i.quantity_reserved) AS calculated_available,
    (i.quantity_available - (i.quantity_on_hand - i.quantity_reserved)) AS discrepancy,
    'Available quantity calculation error' AS issue_type
FROM inventory i
INNER JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_available != (i.quantity_on_hand - i.quantity_reserved)
ORDER BY ABS(discrepancy) DESC;

-- ========================================
-- 5. Products Below Reorder Level
-- ========================================
SELECT 
    i.inventory_id,
    i.product_id,
    p.sku,
    p.product_name,
    p.price,
    pc.category_name,
    i.warehouse_id,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_available,
    i.reorder_level,
    (i.reorder_level - i.quantity_available) AS units_below_threshold,
    ROUND((i.reorder_level - i.quantity_available) * p.cost, 2) AS restock_cost,
    COUNT(DISTINCT oi.order_id) AS orders_last_30_days,
    SUM(CASE WHEN o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY) THEN oi.quantity ELSE 0 END) AS sold_last_30_days,
    'Below reorder level' AS issue_type
FROM inventory i
INNER JOIN products p ON i.product_id = p.product_id
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id 
    AND o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
WHERE i.quantity_available < i.reorder_level
    AND p.status = 'active'
GROUP BY i.inventory_id, i.product_id, p.sku, p.product_name, p.price, pc.category_name,
         i.warehouse_id, i.quantity_on_hand, i.quantity_reserved, i.quantity_available, 
         i.reorder_level, p.cost
ORDER BY units_below_threshold DESC;

-- ========================================
-- 6. Inventory Without Product Records
-- ========================================
SELECT 
    i.inventory_id,
    i.product_id AS orphaned_product_id,
    i.warehouse_id,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_available,
    i.last_updated,
    'Inventory record for deleted product' AS issue_type
FROM inventory i
LEFT JOIN products p ON i.product_id = p.product_id
WHERE p.product_id IS NULL
ORDER BY i.quantity_on_hand DESC;

-- ========================================
-- 7. Out of Stock Products with Active Orders
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.stock_quantity,
    p.status AS product_status,
    COUNT(DISTINCT o.order_id) AS pending_orders,
    SUM(oi.quantity) AS total_ordered_quantity,
    SUM(oi.subtotal) AS revenue_at_risk,
    MIN(o.order_date) AS oldest_pending_order,
    DATEDIFF(CURRENT_DATE, MIN(o.order_date)) AS days_pending,
    'Out of stock with pending orders' AS issue_type
FROM products p
INNER JOIN order_items oi ON p.product_id = oi.product_id
INNER JOIN orders o ON oi.order_id = o.order_id
WHERE (p.stock_quantity <= 0 OR p.status = 'out_of_stock')
    AND o.status IN ('pending', 'processing')
GROUP BY p.product_id, p.sku, p.product_name, p.stock_quantity, p.status
ORDER BY revenue_at_risk DESC;

-- ========================================
-- 8. Duplicate Inventory Records
-- ========================================
SELECT 
    i1.product_id,
    p.sku,
    p.product_name,
    i1.warehouse_id,
    COUNT(*) AS duplicate_count,
    SUM(i1.quantity_on_hand) AS total_quantity,
    'Duplicate inventory records for same product/warehouse' AS issue_type
FROM inventory i1
INNER JOIN products p ON i1.product_id = p.product_id
GROUP BY i1.product_id, p.sku, p.product_name, i1.warehouse_id
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;

-- ========================================
-- 9. Stock Discrepancy Impact Analysis
-- ========================================
SELECT 
    CASE 
        WHEN ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0)) = 0 THEN 'Perfect Match'
        WHEN ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0)) BETWEEN 1 AND 10 THEN 'Minor Discrepancy (1-10 units)'
        WHEN ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0)) BETWEEN 11 AND 50 THEN 'Moderate Discrepancy (11-50 units)'
        WHEN ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0)) BETWEEN 51 AND 100 THEN 'Major Discrepancy (51-100 units)'
        ELSE 'Critical Discrepancy (100+ units)'
    END AS discrepancy_level,
    COUNT(DISTINCT p.product_id) AS product_count,
    SUM(ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0))) AS total_unit_discrepancy,
    ROUND(AVG(p.price * ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0))), 2) AS avg_value_at_risk,
    ROUND(SUM(p.price * ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0))), 2) AS total_value_at_risk
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
WHERE p.status = 'active'
GROUP BY p.product_id, p.stock_quantity, p.price, discrepancy_level
ORDER BY 
    CASE discrepancy_level
        WHEN 'Perfect Match' THEN 1
        WHEN 'Minor Discrepancy (1-10 units)' THEN 2
        WHEN 'Moderate Discrepancy (11-50 units)' THEN 3
        WHEN 'Major Discrepancy (51-100 units)' THEN 4
        ELSE 5
    END;

-- ========================================
-- 10. Inventory Mismatch Summary Report
-- ========================================
SELECT 
    'Total Active Products' AS metric,
    COUNT(*) AS count,
    NULL AS total_discrepancy
FROM products
WHERE status = 'active'

UNION ALL

SELECT 
    'Products with Inventory Mismatch' AS metric,
    COUNT(DISTINCT p.product_id) AS count,
    SUM(ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0))) AS total_discrepancy
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
WHERE p.status = 'active'
GROUP BY p.product_id, p.stock_quantity
HAVING ABS(p.stock_quantity - COALESCE(SUM(i.quantity_on_hand), 0)) > 0

UNION ALL

SELECT 
    'Oversold Products' AS metric,
    COUNT(DISTINCT p.product_id) AS count,
    SUM(SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END) - p.stock_quantity) AS total_discrepancy
FROM products p
INNER JOIN order_items oi ON p.product_id = oi.product_id
INNER JOIN orders o ON oi.order_id = o.order_id
WHERE o.status IN ('pending', 'processing')
GROUP BY p.product_id, p.stock_quantity
HAVING p.stock_quantity < SUM(CASE WHEN o.status IN ('pending', 'processing') THEN oi.quantity ELSE 0 END)

UNION ALL

SELECT 
    'Reserved Exceeds On-Hand' AS metric,
    COUNT(*) AS count,
    SUM(i.quantity_reserved - i.quantity_on_hand) AS total_discrepancy
FROM inventory i
WHERE i.quantity_reserved > i.quantity_on_hand

UNION ALL

SELECT 
    'Products Below Reorder Level' AS metric,
    COUNT(*) AS count,
    SUM(i.reorder_level - i.quantity_available) AS total_discrepancy
FROM inventory i
INNER JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_available < i.reorder_level
    AND p.status = 'active'

UNION ALL

SELECT 
    'Orphaned Inventory Records' AS metric,
    COUNT(*) AS count,
    SUM(i.quantity_on_hand) AS total_discrepancy
FROM inventory i
LEFT JOIN products p ON i.product_id = p.product_id
WHERE p.product_id IS NULL;