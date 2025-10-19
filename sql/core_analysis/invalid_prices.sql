-- ========================================
-- INVALID_PRICES.SQL
-- Data Quality Check: Invalid or Unrealistic Prices
-- Identifies zero, negative, or unrealistic prices
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Products with Zero or Negative Prices
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    pc.category_name,
    p.stock_quantity,
    p.status,
    CASE 
        WHEN p.price = 0 THEN 'Zero price'
        WHEN p.price < 0 THEN 'Negative price'
    END AS issue_type,
    p.created_at
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE (p.price <= 0 OR p.price IS NULL)
    AND p.status = 'active'
ORDER BY p.price ASC, p.created_at DESC;

-- ========================================
-- 2. Products with Negative or Zero Cost
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    CASE 
        WHEN p.cost IS NOT NULL THEN ROUND(((p.price - p.cost) / p.price) * 100, 2)
        ELSE NULL
    END AS margin_pct,
    pc.category_name,
    p.status,
    CASE 
        WHEN p.cost = 0 THEN 'Zero cost'
        WHEN p.cost < 0 THEN 'Negative cost'
        WHEN p.cost IS NULL THEN 'Missing cost'
    END AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE (p.cost <= 0 OR p.cost IS NULL)
    AND p.status = 'active'
    AND p.price > 0
ORDER BY p.price DESC;

-- ========================================
-- 3. Products with Cost Higher Than Price (Negative Margins)
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    ROUND(p.price - p.cost, 2) AS profit_per_unit,
    ROUND(((p.price - p.cost) / p.price) * 100, 2) AS margin_pct,
    pc.category_name,
    p.stock_quantity,
    COUNT(DISTINCT oi.order_id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity_sold,
    'Selling at a loss' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.cost > p.price
    AND p.status = 'active'
GROUP BY p.product_id, p.sku, p.product_name, p.price, p.cost, pc.category_name, p.stock_quantity
ORDER BY (p.cost - p.price) * COALESCE(SUM(oi.quantity), 0) DESC;

-- ========================================
-- 4. Unrealistically Low Prices by Category
-- (Prices below 5th percentile in category)
-- ========================================
WITH category_price_stats AS (
    SELECT 
        category_id,
        MIN(price) AS min_price,
        MAX(price) AS max_price,
        AVG(price) AS avg_price,
        STDDEV(price) AS stddev_price,
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY price) AS price_5th_percentile,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY price) AS price_95th_percentile
    FROM products
    WHERE status = 'active' AND price > 0
    GROUP BY category_id
)
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    pc.category_name,
    cps.avg_price AS category_avg_price,
    cps.price_5th_percentile AS category_5th_percentile,
    ROUND((p.price / cps.avg_price) * 100, 2) AS pct_of_category_avg,
    'Price unusually low for category' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
INNER JOIN category_price_stats cps ON p.category_id = cps.category_id
WHERE p.price < cps.price_5th_percentile
    AND p.price > 0
    AND p.status = 'active'
ORDER BY p.price / cps.avg_price ASC;

-- ========================================
-- 5. Unrealistically High Prices by Category
-- (Prices above 95th percentile in category)
-- ========================================
WITH category_price_stats AS (
    SELECT 
        category_id,
        AVG(price) AS avg_price,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY price) AS price_95th_percentile
    FROM products
    WHERE status = 'active' AND price > 0
    GROUP BY category_id
)
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    pc.category_name,
    cps.avg_price AS category_avg_price,
    cps.price_95th_percentile AS category_95th_percentile,
    ROUND((p.price / cps.avg_price) * 100, 2) AS pct_of_category_avg,
    'Price unusually high for category' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
INNER JOIN category_price_stats cps ON p.category_id = cps.category_id
WHERE p.price > cps.price_95th_percentile
    AND p.status = 'active'
ORDER BY p.price / cps.avg_price DESC;

-- ========================================
-- 6. Suspicious Price-Cost Relationships
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    ROUND(p.price - p.cost, 2) AS profit_per_unit,
    ROUND(((p.price - p.cost) / p.price) * 100, 2) AS margin_pct,
    pc.category_name,
    CASE 
        WHEN p.price = p.cost THEN 'Price equals cost (0% margin)'
        WHEN p.price < p.cost THEN 'Price below cost (negative margin)'
        WHEN ((p.price - p.cost) / p.price) * 100 > 90 THEN 'Margin over 90% (suspiciously high)'
        WHEN ((p.price - p.cost) / p.price) * 100 < 5 AND p.price > 100 THEN 'Margin under 5% (very thin)'
    END AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.status = 'active'
    AND p.price > 0
    AND p.cost > 0
    AND (
        p.price = p.cost OR
        p.price < p.cost OR
        ((p.price - p.cost) / p.price) * 100 > 90 OR
        (((p.price - p.cost) / p.price) * 100 < 5 AND p.price > 100)
    )
ORDER BY margin_pct DESC;

-- ========================================
-- 7. Prices with Too Many Decimal Places
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    pc.category_name,
    'Price has more than 2 decimal places' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.price IS NOT NULL
    AND p.price != ROUND(p.price, 2)
    AND p.status = 'active'
ORDER BY p.price DESC;

-- ========================================
-- 8. Products with Sales Despite Invalid Prices
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.cost,
    pc.category_name,
    COUNT(DISTINCT oi.order_id) AS order_count,
    SUM(oi.quantity) AS total_quantity_sold,
    SUM(oi.subtotal) AS total_revenue,
    CASE 
        WHEN p.price <= 0 THEN 'Zero/negative price with sales'
        WHEN p.cost > p.price THEN 'Selling at loss with sales'
        WHEN p.cost <= 0 THEN 'Zero/negative cost with sales'
    END AS critical_issue
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
INNER JOIN order_items oi ON p.product_id = oi.product_id
WHERE (
    p.price <= 0 OR
    p.cost > p.price OR
    p.cost <= 0
)
GROUP BY p.product_id, p.sku, p.product_name, p.price, p.cost, pc.category_name
ORDER BY total_revenue DESC;

-- ========================================
-- 9. Price Validation Summary Report
-- ========================================
SELECT 
    'Total Active Products' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE status = 'active'

UNION ALL

SELECT 
    'Zero or Negative Prices' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE (price <= 0 OR price IS NULL)
    AND status = 'active'

UNION ALL

SELECT 
    'Zero or Negative Costs' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE (cost <= 0 OR cost IS NULL)
    AND status = 'active'
    AND price > 0

UNION ALL

SELECT 
    'Negative Margins (Cost > Price)' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE cost > price
    AND status = 'active'

UNION ALL

SELECT 
    'Suspiciously High Margins (>90%)' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE status = 'active'
    AND price > 0
    AND cost > 0
    AND ((price - cost) / price) * 100 > 90

UNION ALL

SELECT 
    'Valid Price & Cost Data' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE status = 'active'
    AND price > 0
    AND cost > 0
    AND cost < price
    AND ((price - cost) / price) * 100 BETWEEN 5 AND 90;

-- ========================================
-- 10. Average Margin by Category
-- ========================================
SELECT 
    pc.category_name,
    COUNT(*) AS product_count,
    ROUND(AVG(p.price), 2) AS avg_price,
    ROUND(AVG(p.cost), 2) AS avg_cost,
    ROUND(AVG((p.price - p.cost) / p.price) * 100, 2) AS avg_margin_pct,
    ROUND(MIN((p.price - p.cost) / p.price) * 100, 2) AS min_margin_pct,
    ROUND(MAX((p.price - p.cost) / p.price) * 100, 2) AS max_margin_pct,
    SUM(CASE WHEN p.cost > p.price THEN 1 ELSE 0 END) AS negative_margin_count
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.status = 'active'
    AND p.price > 0
    AND p.cost > 0
GROUP BY pc.category_name
ORDER BY avg_margin_pct ASC;