-- ========================================
-- CATEGORY_INCONSISTENCIES.SQL
-- Data Quality Check: Category Inconsistencies
-- Identifies miscategorized products and category structure issues
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Products Without Categories
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.description,
    p.price,
    p.stock_quantity,
    p.status,
    'No category assigned' AS issue_type,
    p.created_at
FROM products p
WHERE p.category_id IS NULL
    AND p.status = 'active'
ORDER BY p.price DESC;

-- ========================================
-- 2. Categories with No Products
-- ========================================
SELECT 
    pc.category_id,
    pc.category_name,
    pc.parent_id,
    parent_cat.category_name AS parent_category_name,
    'Empty category - no products' AS issue_type,
    pc.created_at
FROM product_categories pc
LEFT JOIN product_categories parent_cat ON pc.parent_id = parent_cat.category_id
LEFT JOIN products p ON pc.category_id = p.category_id AND p.status = 'active'
WHERE p.product_id IS NULL
ORDER BY pc.created_at DESC;

-- ========================================
-- 3. Orphaned Categories (Parent Category Doesn't Exist)
-- ========================================
SELECT 
    pc.category_id,
    pc.category_name,
    pc.parent_id AS invalid_parent_id,
    pc.description,
    COUNT(p.product_id) AS product_count,
    'Orphaned category - parent missing' AS issue_type
FROM product_categories pc
LEFT JOIN product_categories parent_cat ON pc.parent_id = parent_cat.category_id
LEFT JOIN products p ON pc.category_id = p.category_id AND p.status = 'active'
WHERE pc.parent_id IS NOT NULL
    AND parent_cat.category_id IS NULL
GROUP BY pc.category_id, pc.category_name, pc.parent_id, pc.description
ORDER BY product_count DESC;

-- ========================================
-- 4. Circular Category References
-- ========================================
SELECT 
    pc1.category_id AS category_1_id,
    pc1.category_name AS category_1_name,
    pc1.parent_id AS points_to_category_2,
    pc2.category_name AS category_2_name,
    pc2.parent_id AS points_to_category_1,
    'Circular reference detected' AS issue_type
FROM product_categories pc1
INNER JOIN product_categories pc2 ON pc1.parent_id = pc2.category_id
WHERE pc2.parent_id = pc1.category_id;

-- ========================================
-- 5. Price Outliers by Category
-- (Products significantly overpriced/underpriced for their category)
-- ========================================
WITH category_price_stats AS (
    SELECT 
        category_id,
        AVG(price) AS avg_price,
        STDDEV(price) AS stddev_price,
        MIN(price) AS min_price,
        MAX(price) AS max_price,
        COUNT(*) AS product_count
    FROM products
    WHERE status = 'active' AND price > 0
    GROUP BY category_id
    HAVING COUNT(*) >= 5  -- Only categories with 5+ products
)
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    pc.category_name,
    ROUND(cps.avg_price, 2) AS category_avg_price,
    ROUND(cps.min_price, 2) AS category_min_price,
    ROUND(cps.max_price, 2) AS category_max_price,
    ROUND((p.price - cps.avg_price) / cps.stddev_price, 2) AS std_deviations_from_mean,
    CASE 
        WHEN p.price > (cps.avg_price + 3 * cps.stddev_price) THEN 'Significantly overpriced'
        WHEN p.price < (cps.avg_price - 3 * cps.stddev_price) THEN 'Significantly underpriced'
    END AS issue_type
FROM products p
INNER JOIN product_categories pc ON p.category_id = pc.category_id
INNER JOIN category_price_stats cps ON p.category_id = cps.category_id
WHERE p.status = 'active'
    AND cps.stddev_price > 0
    AND (
        p.price > (cps.avg_price + 3 * cps.stddev_price) OR
        p.price < (cps.avg_price - 3 * cps.stddev_price)
    )
ORDER BY ABS((p.price - cps.avg_price) / cps.stddev_price) DESC;

-- ========================================
-- 6. Products in Wrong Category (Based on Name Patterns)
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    pc.category_name AS current_category,
    CASE 
        WHEN LOWER(p.product_name) LIKE '%laptop%' OR LOWER(p.product_name) LIKE '%computer%' THEN 'Should be in Electronics/Computers'
        WHEN LOWER(p.product_name) LIKE '%shirt%' OR LOWER(p.product_name) LIKE '%pants%' OR LOWER(p.product_name) LIKE '%dress%' THEN 'Should be in Clothing'
        WHEN LOWER(p.product_name) LIKE '%book%' OR LOWER(p.product_name) LIKE '%novel%' THEN 'Should be in Books'
        WHEN LOWER(p.product_name) LIKE '%phone%' OR LOWER(p.product_name) LIKE '%tablet%' THEN 'Should be in Electronics/Mobile'
        WHEN LOWER(p.product_name) LIKE '%shoe%' OR LOWER(p.product_name) LIKE '%boot%' OR LOWER(p.product_name) LIKE '%sneaker%' THEN 'Should be in Footwear'
        WHEN LOWER(p.product_name) LIKE '%toy%' OR LOWER(p.product_name) LIKE '%game%' THEN 'Should be in Toys/Games'
        WHEN LOWER(p.product_name) LIKE '%furniture%' OR LOWER(p.product_name) LIKE '%chair%' OR LOWER(p.product_name) LIKE '%table%' THEN 'Should be in Furniture'
    END AS suggested_category,
    'Potential miscategorization' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.status = 'active'
    AND (
        (LOWER(p.product_name) LIKE '%laptop%' OR LOWER(p.product_name) LIKE '%computer%') OR
        (LOWER(p.product_name) LIKE '%shirt%' OR LOWER(p.product_name) LIKE '%pants%' OR LOWER(p.product_name) LIKE '%dress%') OR
        (LOWER(p.product_name) LIKE '%book%' OR LOWER(p.product_name) LIKE '%novel%') OR
        (LOWER(p.product_name) LIKE '%phone%' OR LOWER(p.product_name) LIKE '%tablet%') OR
        (LOWER(p.product_name) LIKE '%shoe%' OR LOWER(p.product_name) LIKE '%boot%' OR LOWER(p.product_name) LIKE '%sneaker%') OR
        (LOWER(p.product_name) LIKE '%toy%' OR LOWER(p.product_name) LIKE '%game%') OR
        (LOWER(p.product_name) LIKE '%furniture%' OR LOWER(p.product_name) LIKE '%chair%' OR LOWER(p.product_name) LIKE '%table%')
    )
    AND (
        pc.category_name IS NULL OR
        (LOWER(p.product_name) LIKE '%laptop%' AND LOWER(pc.category_name) NOT LIKE '%computer%' AND LOWER(pc.category_name) NOT LIKE '%electronic%') OR
        (LOWER(p.product_name) LIKE '%shirt%' AND LOWER(pc.category_name) NOT LIKE '%cloth%' AND LOWER(pc.category_name) NOT LIKE '%apparel%')
    )
ORDER BY p.price DESC;

-- ========================================
-- 7. Category Depth Analysis
-- (Find deeply nested categories that may need restructuring)
-- ========================================
WITH RECURSIVE category_hierarchy AS (
    SELECT 
        category_id,
        category_name,
        parent_id,
        1 AS depth,
        CAST(category_name AS CHAR(500)) AS path
    FROM product_categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    SELECT 
        pc.category_id,
        pc.category_name,
        pc.parent_id,
        ch.depth + 1 AS depth,
        CONCAT(ch.path, ' > ', pc.category_name) AS path
    FROM product_categories pc
    INNER JOIN category_hierarchy ch ON pc.parent_id = ch.category_id
    WHERE ch.depth < 10
)
SELECT 
    ch.category_id,
    ch.category_name,
    ch.depth,
    ch.path AS category_path,
    COUNT(p.product_id) AS product_count,
    CASE 
        WHEN ch.depth > 4 THEN 'Excessively nested category'
        WHEN ch.depth > 3 THEN 'Deeply nested category'
    END AS issue_type
FROM category_hierarchy ch
LEFT JOIN products p ON ch.category_id = p.category_id AND p.status = 'active'
WHERE ch.depth > 3
GROUP BY ch.category_id, ch.category_name, ch.depth, ch.path
ORDER BY ch.depth DESC, product_count DESC;

-- ========================================
-- 8. Categories with Inconsistent Product Count
-- ========================================
SELECT 
    pc.category_id,
    pc.category_name,
    parent_cat.category_name AS parent_category,
    COUNT(DISTINCT p.product_id) AS direct_product_count,
    COUNT(DISTINCT child_products.product_id) AS child_category_products,
    COUNT(DISTINCT p.product_id) + COUNT(DISTINCT child_products.product_id) AS total_products,
    CASE 
        WHEN COUNT(DISTINCT p.product_id) = 0 AND COUNT(DISTINCT child_products.product_id) = 0 THEN 'Empty parent category'
        WHEN COUNT(DISTINCT p.product_id) > 0 AND COUNT(DISTINCT child_products.product_id) > 0 THEN 'Mixed usage (has own products and subcategories)'
        WHEN COUNT(DISTINCT child_products.product_id) > 0 THEN 'Parent category only'
    END AS category_usage
FROM product_categories pc
LEFT JOIN product_categories parent_cat ON pc.parent_id = parent_cat.category_id
LEFT JOIN products p ON pc.category_id = p.category_id AND p.status = 'active'
LEFT JOIN product_categories child_cat ON pc.category_id = child_cat.parent_id
LEFT JOIN products child_products ON child_cat.category_id = child_products.category_id AND child_products.status = 'active'
GROUP BY pc.category_id, pc.category_name, parent_cat.category_name
HAVING COUNT(DISTINCT p.product_id) = 0 OR 
       (COUNT(DISTINCT p.product_id) > 0 AND COUNT(DISTINCT child_products.product_id) > 0)
ORDER BY total_products DESC;

-- ========================================
-- 9. Category Performance Metrics
-- ========================================
SELECT 
    pc.category_id,
    pc.category_name,
    COUNT(DISTINCT p.product_id) AS active_products,
    COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS discontinued_products,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    SUM(oi.quantity) AS total_units_sold,
    ROUND(SUM(oi.subtotal), 2) AS total_revenue,
    ROUND(AVG(p.price), 2) AS avg_product_price,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    COUNT(DISTINCT r.review_id) AS total_reviews,
    CASE 
        WHEN COUNT(DISTINCT p.product_id) = 0 THEN 'Empty category'
        WHEN COUNT(DISTINCT oi.order_id) = 0 THEN 'No sales recorded'
        WHEN COUNT(DISTINCT r.review_id) = 0 THEN 'No reviews'
        WHEN AVG(r.rating) < 3.0 THEN 'Low average rating'
    END AS potential_issue
FROM product_categories pc
LEFT JOIN products p ON pc.category_id = p.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN reviews r ON p.product_id = r.product_id AND r.status = 'approved'
GROUP BY pc.category_id, pc.category_name
HAVING COUNT(DISTINCT p.product_id) = 0 
    OR COUNT(DISTINCT oi.order_id) = 0 
    OR COUNT(DISTINCT r.review_id) = 0 
    OR AVG(r.rating) < 3.0
ORDER BY total_revenue DESC;

-- ========================================
-- 10. Duplicate Category Names
-- ========================================
SELECT 
    pc1.category_id AS category_id_1,
    pc1.category_name,
    pc1.parent_id AS parent_id_1,
    parent1.category_name AS parent_name_1,
    pc2.category_id AS category_id_2,
    pc2.parent_id AS parent_id_2,
    parent2.category_name AS parent_name_2,
    COUNT(p1.product_id) AS products_in_cat_1,
    COUNT(p2.product_id) AS products_in_cat_2,
    'Duplicate category name' AS issue_type
FROM product_categories pc1
INNER JOIN product_categories pc2 ON pc1.category_name = pc2.category_name
    AND pc1.category_id < pc2.category_id
LEFT JOIN product_categories parent1 ON pc1.parent_id = parent1.category_id
LEFT JOIN product_categories parent2 ON pc2.parent_id = parent2.category_id
LEFT JOIN products p1 ON pc1.category_id = p1.category_id AND p1.status = 'active'
LEFT JOIN products p2 ON pc2.category_id = p2.category_id AND p2.status = 'active'
GROUP BY pc1.category_id, pc1.category_name, pc1.parent_id, parent1.category_name,
         pc2.category_id, pc2.parent_id, parent2.category_name
ORDER BY pc1.category_name;

-- ========================================
-- 11. Category Summary Statistics
-- ========================================
SELECT 
    'Total Categories' AS metric,
    COUNT(*) AS count
FROM product_categories

UNION ALL

SELECT 
    'Root Categories (No Parent)' AS metric,
    COUNT(*) AS count
FROM product_categories
WHERE parent_id IS NULL

UNION ALL

SELECT 
    'Subcategories' AS metric,
    COUNT(*) AS count
FROM product_categories
WHERE parent_id IS NOT NULL

UNION ALL

SELECT 
    'Empty Categories' AS metric,
    COUNT(*) AS count
FROM product_categories pc
LEFT JOIN products p ON pc.category_id = p.category_id AND p.status = 'active'
WHERE p.product_id IS NULL

UNION ALL

SELECT 
    'Orphaned Categories' AS metric,
    COUNT(*) AS count
FROM product_categories pc
LEFT JOIN product_categories parent_cat ON pc.parent_id = parent_cat.category_id
WHERE pc.parent_id IS NOT NULL
    AND parent_cat.category_id IS NULL

UNION ALL

SELECT 
    'Products Without Category' AS metric,
    COUNT(*) AS count
FROM products
WHERE category_id IS NULL
    AND status = 'active';

-- ========================================
-- 12. Category Hierarchy Visualization
-- (Top-level categories with their subcategories)
-- ========================================
SELECT 
    COALESCE(parent.category_name, 'ROOT') AS parent_category,
    pc.category_name AS subcategory,
    COUNT(DISTINCT p.product_id) AS product_count,
    ROUND(AVG(p.price), 2) AS avg_price,
    COUNT(DISTINCT oi.order_id) AS order_count,
    ROUND(SUM(oi.subtotal), 2) AS total_revenue
FROM product_categories pc
LEFT JOIN product_categories parent ON pc.parent_id = parent.category_id
LEFT JOIN products p ON pc.category_id = p.category_id AND p.status = 'active'
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY parent.category_name, pc.category_name
ORDER BY parent.category_name, total_revenue DESC;