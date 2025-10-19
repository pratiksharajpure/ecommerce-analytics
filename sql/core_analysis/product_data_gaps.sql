-- ========================================
-- PRODUCT DATA GAPS ANALYSIS
-- Identifies missing or incomplete product information
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. OVERALL DATA COMPLETENESS SUMMARY
-- ========================================
SELECT 
    'PRODUCT DATA COMPLETENESS OVERVIEW' AS report_title,
    COUNT(*) AS total_products,
    SUM(CASE WHEN sku IS NULL OR TRIM(sku) = '' THEN 1 ELSE 0 END) AS missing_sku,
    SUM(CASE WHEN product_name IS NULL OR TRIM(product_name) = '' THEN 1 ELSE 0 END) AS missing_name,
    SUM(CASE WHEN description IS NULL OR TRIM(description) = '' THEN 1 ELSE 0 END) AS missing_description,
    SUM(CASE WHEN category_id IS NULL THEN 1 ELSE 0 END) AS missing_category,
    SUM(CASE WHEN price IS NULL OR price <= 0 THEN 1 ELSE 0 END) AS missing_invalid_price,
    SUM(CASE WHEN cost IS NULL OR cost <= 0 THEN 1 ELSE 0 END) AS missing_invalid_cost,
    SUM(CASE WHEN stock_quantity IS NULL THEN 1 ELSE 0 END) AS missing_stock_info,
    ROUND(SUM(CASE WHEN description IS NULL OR TRIM(description) = '' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_missing_description,
    ROUND(SUM(CASE WHEN category_id IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_missing_category
FROM products;

-- ========================================
-- 2. PRODUCTS WITH MISSING DESCRIPTIONS
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.description,
    pc.category_name,
    p.price,
    p.cost,
    p.stock_quantity,
    p.status,
    p.created_at,
    COUNT(DISTINCT oi.order_item_id) AS times_ordered,
    COALESCE(SUM(oi.quantity), 0) AS total_units_sold,
    COALESCE(SUM(oi.subtotal), 0) AS total_revenue
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.description IS NULL 
   OR TRIM(p.description) = ''
   OR LENGTH(p.description) < 20  -- Description too short to be useful
GROUP BY p.product_id, p.sku, p.product_name, p.description, 
         pc.category_name, p.price, p.cost, p.stock_quantity, p.status, p.created_at
ORDER BY total_revenue DESC, times_ordered DESC;

-- ========================================
-- 3. PRODUCTS WITH MISSING CATEGORIES
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.category_id,
    p.price,
    p.stock_quantity,
    p.status,
    p.created_at,
    COUNT(DISTINCT oi.order_item_id) AS times_ordered,
    COALESCE(SUM(oi.quantity), 0) AS total_units_sold,
    COALESCE(SUM(oi.subtotal), 0) AS total_revenue
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.category_id IS NULL
GROUP BY p.product_id, p.sku, p.product_name, p.category_id, 
         p.price, p.stock_quantity, p.status, p.created_at
ORDER BY total_revenue DESC;

-- ========================================
-- 4. PRODUCTS WITH PRICING ISSUES
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    p.price,
    p.cost,
    p.price - p.cost AS profit_margin,
    CASE 
        WHEN p.price IS NULL THEN 'Price Missing'
        WHEN p.price <= 0 THEN 'Price Zero or Negative'
        WHEN p.cost IS NULL THEN 'Cost Missing'
        WHEN p.cost <= 0 THEN 'Cost Zero or Negative'
        WHEN p.price < p.cost THEN 'Price Below Cost'
        WHEN p.price = p.cost THEN 'No Profit Margin'
        ELSE 'Other Issue'
    END AS pricing_issue,
    p.stock_quantity,
    p.status,
    COUNT(DISTINCT oi.order_item_id) AS times_ordered,
    COALESCE(SUM(oi.subtotal), 0) AS total_revenue
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.price IS NULL 
   OR p.price <= 0 
   OR p.cost IS NULL 
   OR p.cost <= 0
   OR p.price < p.cost
   OR p.price = p.cost
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name,
         p.price, p.cost, p.stock_quantity, p.status
ORDER BY total_revenue DESC;

-- ========================================
-- 5. PRODUCTS WITH MISSING OR INVALID SKU
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    p.price,
    p.status,
    CASE 
        WHEN p.sku IS NULL THEN 'SKU Missing'
        WHEN TRIM(p.sku) = '' THEN 'SKU Empty'
        WHEN LENGTH(p.sku) < 5 THEN 'SKU Too Short'
        ELSE 'Other Issue'
    END AS sku_issue,
    COUNT(DISTINCT oi.order_item_id) AS times_ordered,
    COALESCE(SUM(oi.quantity), 0) AS total_units_sold
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.sku IS NULL 
   OR TRIM(p.sku) = ''
   OR LENGTH(p.sku) < 5
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name, p.price, p.status
ORDER BY times_ordered DESC;

-- ========================================
-- 6. DATA COMPLETENESS SCORE BY PRODUCT
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    p.status,
    (CASE WHEN p.sku IS NOT NULL AND TRIM(p.sku) != '' AND LENGTH(p.sku) >= 5 THEN 15 ELSE 0 END +
     CASE WHEN p.product_name IS NOT NULL AND TRIM(p.product_name) != '' THEN 15 ELSE 0 END +
     CASE WHEN p.description IS NOT NULL AND LENGTH(p.description) >= 20 THEN 20 ELSE 0 END +
     CASE WHEN p.category_id IS NOT NULL THEN 15 ELSE 0 END +
     CASE WHEN p.price IS NOT NULL AND p.price > 0 THEN 15 ELSE 0 END +
     CASE WHEN p.cost IS NOT NULL AND p.cost > 0 THEN 10 ELSE 0 END +
     CASE WHEN p.stock_quantity IS NOT NULL THEN 10 ELSE 0 END) AS completeness_score,
    COUNT(DISTINCT oi.order_item_id) AS times_ordered,
    COALESCE(SUM(oi.subtotal), 0) AS total_revenue
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name, p.status,
         p.description, p.category_id, p.price, p.cost, p.stock_quantity
HAVING completeness_score < 100
ORDER BY times_ordered DESC, completeness_score ASC
LIMIT 100;

-- ========================================
-- 7. CATEGORY-LEVEL DATA QUALITY
-- ========================================
SELECT 
    COALESCE(pc.category_name, 'Uncategorized') AS category_name,
    COUNT(p.product_id) AS total_products,
    SUM(CASE WHEN p.description IS NULL OR TRIM(p.description) = '' OR LENGTH(p.description) < 20 THEN 1 ELSE 0 END) AS missing_description,
    SUM(CASE WHEN p.sku IS NULL OR TRIM(p.sku) = '' THEN 1 ELSE 0 END) AS missing_sku,
    SUM(CASE WHEN p.price IS NULL OR p.price <= 0 THEN 1 ELSE 0 END) AS missing_price,
    SUM(CASE WHEN p.cost IS NULL OR p.cost <= 0 THEN 1 ELSE 0 END) AS missing_cost,
    ROUND(SUM(CASE WHEN p.description IS NULL OR TRIM(p.description) = '' OR LENGTH(p.description) < 20 THEN 1 ELSE 0 END) * 100.0 / COUNT(p.product_id), 2) AS pct_missing_description,
    ROUND(AVG(p.price), 2) AS avg_price,
    SUM(p.stock_quantity) AS total_stock
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
GROUP BY pc.category_name
ORDER BY total_products DESC;

-- ========================================
-- 8. HIGH-VALUE PRODUCTS WITH DATA GAPS
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    p.price,
    p.status,
    COUNT(DISTINCT oi.order_item_id) AS times_ordered,
    COALESCE(SUM(oi.quantity), 0) AS total_units_sold,
    COALESCE(SUM(oi.subtotal), 0) AS total_revenue,
    CASE 
        WHEN p.description IS NULL OR LENGTH(p.description) < 20 THEN 'Description'
        WHEN p.category_id IS NULL THEN 'Category'
        WHEN p.price IS NULL OR p.price <= 0 THEN 'Price'
        WHEN p.cost IS NULL OR p.cost <= 0 THEN 'Cost'
        ELSE 'Multiple Issues'
    END AS primary_gap
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE (p.description IS NULL OR TRIM(p.description) = '' OR LENGTH(p.description) < 20
       OR p.category_id IS NULL
       OR p.price IS NULL OR p.price <= 0
       OR p.cost IS NULL OR p.cost <= 0)
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name, 
         p.price, p.status, p.description, p.category_id, p.cost
HAVING total_revenue > 1000  -- Focus on products with significant sales
ORDER BY total_revenue DESC;

-- ========================================
-- 9. PRODUCTS WITH STOCK QUANTITY ISSUES
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    p.stock_quantity,
    p.status,
    COUNT(DISTINCT oi.order_item_id) AS times_ordered,
    COALESCE(SUM(oi.quantity), 0) AS total_units_sold,
    COALESCE(SUM(CASE WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN oi.quantity ELSE 0 END), 0) AS units_sold_last_30_days,
    CASE 
        WHEN p.stock_quantity IS NULL THEN 'Stock Data Missing'
        WHEN p.stock_quantity = 0 AND p.status = 'active' THEN 'Out of Stock but Active'
        WHEN p.stock_quantity < 0 THEN 'Negative Stock'
        WHEN p.stock_quantity < 10 AND p.status = 'active' THEN 'Low Stock'
        ELSE 'Other Issue'
    END AS stock_issue
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE p.stock_quantity IS NULL 
   OR p.stock_quantity < 0
   OR (p.stock_quantity = 0 AND p.status = 'active')
   OR (p.stock_quantity < 10 AND p.status = 'active')
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name, 
         p.stock_quantity, p.status
HAVING times_ordered > 0  -- Focus on products with demand
ORDER BY units_sold_last_30_days DESC, times_ordered DESC;

-- ========================================
-- 10. MONTHLY TREND - NEW PRODUCTS WITH DATA GAPS
-- ========================================
SELECT 
    DATE_FORMAT(p.created_at, '%Y-%m') AS month,
    COUNT(*) AS new_products_added,
    SUM(CASE WHEN p.description IS NULL OR TRIM(p.description) = '' OR LENGTH(p.description) < 20 THEN 1 ELSE 0 END) AS missing_description,
    SUM(CASE WHEN p.category_id IS NULL THEN 1 ELSE 0 END) AS missing_category,
    SUM(CASE WHEN p.price IS NULL OR p.price <= 0 THEN 1 ELSE 0 END) AS missing_price,
    ROUND(SUM(CASE WHEN p.description IS NULL OR TRIM(p.description) = '' OR LENGTH(p.description) < 20 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_missing_description,
    ROUND(SUM(CASE WHEN p.category_id IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_missing_category
FROM products p
GROUP BY DATE_FORMAT(p.created_at, '%Y-%m')
ORDER BY month DESC;

-- ========================================
-- 11. ORPHANED CATEGORIES (Categories with no products)
-- ========================================
SELECT 
    pc.category_id,
    pc.category_name,
    pc.parent_id,
    parent_cat.category_name AS parent_category_name,
    pc.created_at,
    COUNT(p.product_id) AS product_count
FROM product_categories pc
LEFT JOIN product_categories parent_cat ON pc.parent_id = parent_cat.category_id
LEFT JOIN products p ON pc.category_id = p.category_id
GROUP BY pc.category_id, pc.category_name, pc.parent_id, 
         parent_cat.category_name, pc.created_at
HAVING product_count = 0
ORDER BY pc.created_at DESC;

-- ========================================
-- Display Final Summary
-- ========================================
SELECT 
    'CRITICAL DATA GAPS REQUIRING ATTENTION' AS alert_title,
    (SELECT COUNT(*) FROM products 
     WHERE description IS NULL OR TRIM(description) = '' OR LENGTH(description) < 20) AS products_needing_description,
    (SELECT COUNT(*) FROM products WHERE category_id IS NULL) AS products_needing_category,
    (SELECT COUNT(*) FROM products 
     WHERE price IS NULL OR price <= 0 OR cost IS NULL OR cost <= 0) AS products_with_pricing_issues,
    (SELECT COUNT(*) FROM products 
     WHERE stock_quantity IS NULL OR stock_quantity < 0) AS products_with_stock_issues,
    (SELECT COUNT(*) FROM product_categories pc 
     LEFT JOIN products p ON pc.category_id = p.category_id 
     WHERE p.product_id IS NULL) AS empty_categories;