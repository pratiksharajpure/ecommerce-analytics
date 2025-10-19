-- ========================================
-- MISSING_DESCRIPTIONS.SQL
-- Data Quality Check: Product Description Quality
-- Identifies products without descriptions or poor quality descriptions
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Products with Completely Missing Descriptions
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    p.stock_quantity,
    pc.category_name,
    p.status,
    'No description' AS issue_type,
    p.created_at
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE (p.description IS NULL OR p.description = '')
    AND p.status = 'active'
ORDER BY p.price DESC;

-- ========================================
-- 2. Products with Very Short Descriptions
-- (Less than 50 characters)
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.description,
    LENGTH(p.description) AS description_length,
    p.price,
    pc.category_name,
    'Description too short' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.description IS NOT NULL
    AND p.description != ''
    AND LENGTH(TRIM(p.description)) < 50
    AND p.status = 'active'
ORDER BY LENGTH(p.description) ASC;

-- ========================================
-- 3. Products with Generic/Placeholder Descriptions
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.description,
    p.price,
    pc.category_name,
    'Generic/placeholder description' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.status = 'active'
    AND (
        LOWER(p.description) LIKE '%lorem ipsum%' OR
        LOWER(p.description) LIKE '%placeholder%' OR
        LOWER(p.description) LIKE '%test description%' OR
        LOWER(p.description) LIKE '%coming soon%' OR
        LOWER(p.description) LIKE '%to be updated%' OR
        LOWER(p.description) LIKE '%tbd%' OR
        LOWER(p.description) = 'description' OR
        LOWER(p.description) = 'n/a' OR
        LOWER(p.description) = 'na' OR
        p.description = '...' OR
        p.description = '---'
    )
ORDER BY p.price DESC;

-- ========================================
-- 4. Products with Sales but Missing Descriptions
-- (High priority - affecting conversion)
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    pc.category_name,
    COUNT(DISTINCT oi.order_id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity_sold,
    SUM(oi.subtotal) AS total_revenue,
    AVG(oi.unit_price) AS avg_sale_price,
    CASE 
        WHEN p.description IS NULL OR p.description = '' THEN 'No description'
        WHEN LENGTH(TRIM(p.description)) < 50 THEN 'Description too short'
    END AS issue_type,
    'High priority - has sales history' AS priority
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
INNER JOIN order_items oi ON p.product_id = oi.product_id
WHERE (
    p.description IS NULL OR 
    p.description = '' OR 
    LENGTH(TRIM(p.description)) < 50
)
GROUP BY p.product_id, p.sku, p.product_name, p.price, pc.category_name, p.description
ORDER BY total_revenue DESC;

-- ========================================
-- 5. Products with Duplicate Descriptions
-- ========================================
WITH description_counts AS (
    SELECT 
        description,
        COUNT(*) AS product_count
    FROM products
    WHERE description IS NOT NULL
        AND description != ''
        AND LENGTH(description) > 20
        AND status = 'active'
    GROUP BY description
    HAVING COUNT(*) > 1
)
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    p.price,
    pc.category_name,
    LEFT(p.description, 100) AS description_preview,
    dc.product_count AS products_with_same_description,
    'Duplicate description' AS issue_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
INNER JOIN description_counts dc ON p.description = dc.description
WHERE p.status = 'active'
ORDER BY dc.product_count DESC, p.price DESC;

-- ========================================
-- 6. Description Quality Score by Category
-- ========================================
SELECT 
    pc.category_name,
    COUNT(*) AS total_products,
    SUM(CASE WHEN p.description IS NULL OR p.description = '' THEN 1 ELSE 0 END) AS no_description,
    SUM(CASE WHEN LENGTH(TRIM(p.description)) < 50 AND p.description IS NOT NULL THEN 1 ELSE 0 END) AS short_description,
    SUM(CASE WHEN LENGTH(TRIM(p.description)) >= 50 AND LENGTH(TRIM(p.description)) < 200 THEN 1 ELSE 0 END) AS medium_description,
    SUM(CASE WHEN LENGTH(TRIM(p.description)) >= 200 THEN 1 ELSE 0 END) AS detailed_description,
    ROUND(AVG(CASE WHEN p.description IS NOT NULL THEN LENGTH(p.description) ELSE 0 END), 0) AS avg_description_length,
    ROUND(
        SUM(CASE WHEN LENGTH(TRIM(p.description)) >= 50 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) AS quality_score_pct
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.status = 'active'
GROUP BY pc.category_name
ORDER BY quality_score_pct ASC;

-- ========================================
-- 7. Products with Description Length Outliers
-- ========================================
WITH description_stats AS (
    SELECT 
        AVG(LENGTH(description)) AS avg_length,
        STDDEV(LENGTH(description)) AS stddev_length
    FROM products
    WHERE description IS NOT NULL
        AND description != ''
        AND status = 'active'
)
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    LENGTH(p.description) AS description_length,
    p.price,
    pc.category_name,
    ROUND(ds.avg_length, 0) AS avg_description_length,
    CASE 
        WHEN LENGTH(p.description) > (ds.avg_length + 3 * ds.stddev_length) THEN 'Unusually long'
        WHEN LENGTH(p.description) < (ds.avg_length - 2 * ds.stddev_length) THEN 'Unusually short'
    END AS outlier_type
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
CROSS JOIN description_stats ds
WHERE p.description IS NOT NULL
    AND p.description != ''
    AND p.status = 'active'
    AND (
        LENGTH(p.description) > (ds.avg_length + 3 * ds.stddev_length) OR
        LENGTH(p.description) < (ds.avg_length - 2 * ds.stddev_length)
    )
ORDER BY LENGTH(p.description) DESC;

-- ========================================
-- 8. Products Missing Descriptions by Price Range
-- ========================================
SELECT 
    CASE 
        WHEN p.price < 10 THEN 'Under $10'
        WHEN p.price >= 10 AND p.price < 50 THEN '$10 - $50'
        WHEN p.price >= 50 AND p.price < 100 THEN '$50 - $100'
        WHEN p.price >= 100 AND p.price < 500 THEN '$100 - $500'
        ELSE 'Over $500'
    END AS price_range,
    COUNT(*) AS total_products,
    SUM(CASE WHEN p.description IS NULL OR p.description = '' THEN 1 ELSE 0 END) AS missing_description,
    SUM(CASE WHEN LENGTH(TRIM(p.description)) < 50 AND p.description IS NOT NULL THEN 1 ELSE 0 END) AS short_description,
    SUM(CASE WHEN LENGTH(TRIM(p.description)) >= 50 THEN 1 ELSE 0 END) AS adequate_description,
    ROUND(
        SUM(CASE WHEN p.description IS NULL OR p.description = '' OR LENGTH(TRIM(p.description)) < 50 
            THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) AS poor_quality_pct
FROM products p
WHERE p.status = 'active'
GROUP BY price_range
ORDER BY 
    CASE 
        WHEN p.price < 10 THEN 1
        WHEN p.price >= 10 AND p.price < 50 THEN 2
        WHEN p.price >= 50 AND p.price < 100 THEN 3
        WHEN p.price >= 100 AND p.price < 500 THEN 4
        ELSE 5
    END;

-- ========================================
-- 9. Description Quality Impact on Reviews
-- ========================================
SELECT 
    CASE 
        WHEN p.description IS NULL OR p.description = '' THEN 'No description'
        WHEN LENGTH(TRIM(p.description)) < 50 THEN 'Short description (<50 chars)'
        WHEN LENGTH(TRIM(p.description)) < 200 THEN 'Medium description (50-200 chars)'
        ELSE 'Detailed description (200+ chars)'
    END AS description_quality,
    COUNT(DISTINCT p.product_id) AS product_count,
    COUNT(DISTINCT r.review_id) AS total_reviews,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    ROUND(COUNT(DISTINCT r.review_id) * 1.0 / COUNT(DISTINCT p.product_id), 2) AS reviews_per_product
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id AND r.status = 'approved'
WHERE p.status = 'active'
GROUP BY description_quality
ORDER BY 
    CASE 
        WHEN p.description IS NULL OR p.description = '' THEN 1
        WHEN LENGTH(TRIM(p.description)) < 50 THEN 2
        WHEN LENGTH(TRIM(p.description)) < 200 THEN 3
        ELSE 4
    END;

-- ========================================
-- 10. Summary Report - Description Quality
-- ========================================
SELECT 
    'Total Active Products' AS metric,
    COUNT(*) AS count,
    '100%' AS percentage
FROM products
WHERE status = 'active'

UNION ALL

SELECT 
    'No Description' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE (description IS NULL OR description = '')
    AND status = 'active'

UNION ALL

SELECT 
    'Short Description (<50 chars)' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE description IS NOT NULL
    AND description != ''
    AND LENGTH(TRIM(description)) < 50
    AND status = 'active'

UNION ALL

SELECT 
    'Medium Description (50-200 chars)' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE LENGTH(TRIM(description)) >= 50
    AND LENGTH(TRIM(description)) < 200
    AND status = 'active'

UNION ALL

SELECT 
    'Detailed Description (200+ chars)' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE LENGTH(TRIM(description)) >= 200
    AND status = 'active'

UNION ALL

SELECT 
    'Generic/Placeholder Text' AS metric,
    COUNT(*) AS count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2), '%') AS percentage
FROM products
WHERE status = 'active'
    AND (
        LOWER(description) LIKE '%lorem ipsum%' OR
        LOWER(description) LIKE '%placeholder%' OR
        LOWER(description) LIKE '%test description%'
    );