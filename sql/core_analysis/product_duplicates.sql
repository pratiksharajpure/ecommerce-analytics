-- ========================================
-- PRODUCT DUPLICATES ANALYSIS
-- Identifies duplicate and similar products
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. EXACT DUPLICATE SKUs
-- ========================================
SELECT 
    'DUPLICATE SKU ANALYSIS' AS report_title,
    sku,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(product_id ORDER BY product_id) AS product_ids,
    GROUP_CONCAT(product_name ORDER BY product_id SEPARATOR ' | ') AS product_names,
    GROUP_CONCAT(status ORDER BY product_id) AS statuses
FROM products
WHERE sku IS NOT NULL AND TRIM(sku) != ''
GROUP BY sku
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, sku;

-- ========================================
-- 2. EXACT DUPLICATE PRODUCT NAMES
-- ========================================
SELECT 
    'DUPLICATE PRODUCT NAME ANALYSIS' AS report_title,
    product_name,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(product_id ORDER BY product_id) AS product_ids,
    GROUP_CONCAT(sku ORDER BY product_id SEPARATOR ' | ') AS skus,
    GROUP_CONCAT(CONCAT('$', price) ORDER BY product_id SEPARATOR ' | ') AS prices,
    GROUP_CONCAT(status ORDER BY product_id) AS statuses
FROM products
WHERE product_name IS NOT NULL AND TRIM(product_name) != ''
GROUP BY product_name
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, product_name;

-- ========================================
-- 3. SIMILAR PRODUCT NAMES (Fuzzy Matching)
-- Using SOUNDEX and Levenshtein-like similarity
-- ========================================
SELECT DISTINCT
    p1.product_id AS product_1_id,
    p1.sku AS product_1_sku,
    p1.product_name AS product_1_name,
    p1.price AS product_1_price,
    p1.status AS product_1_status,
    p2.product_id AS product_2_id,
    p2.sku AS product_2_sku,
    p2.product_name AS product_2_name,
    p2.price AS product_2_price,
    p2.status AS product_2_status,
    pc.category_name,
    CASE 
        WHEN SOUNDEX(p1.product_name) = SOUNDEX(p2.product_name) THEN 'Phonetically Similar'
        WHEN LOWER(p1.product_name) LIKE CONCAT('%', LOWER(p2.product_name), '%') THEN 'Name Contains Other'
        WHEN LOWER(p2.product_name) LIKE CONCAT('%', LOWER(p1.product_name), '%') THEN 'Name Contained By Other'
        ELSE 'Similar'
    END AS similarity_type
FROM products p1
INNER JOIN products p2 ON p1.product_id < p2.product_id
LEFT JOIN product_categories pc ON p1.category_id = pc.category_id
WHERE p1.category_id = p2.category_id  -- Same category
  AND (
    -- Phonetically similar names
    SOUNDEX(p1.product_name) = SOUNDEX(p2.product_name)
    -- One name contains the other
    OR LOWER(p1.product_name) LIKE CONCAT('%', LOWER(p2.product_name), '%')
    OR LOWER(p2.product_name) LIKE CONCAT('%', LOWER(p1.product_name), '%')
    -- Very similar first words (brand/model matching)
    OR (
        SUBSTRING_INDEX(LOWER(p1.product_name), ' ', 3) = 
        SUBSTRING_INDEX(LOWER(p2.product_name), ' ', 3)
        AND LENGTH(p1.product_name) > 10
    )
  )
  AND p1.product_name != p2.product_name  -- Exclude exact matches
ORDER BY pc.category_name, p1.product_name;

-- ========================================
-- 4. PRODUCTS WITH SAME PRICE AND CATEGORY
-- ========================================
SELECT 
    pc.category_name,
    p.price,
    COUNT(*) AS product_count,
    GROUP_CONCAT(p.product_id ORDER BY p.product_id) AS product_ids,
    GROUP_CONCAT(p.product_name ORDER BY p.product_id SEPARATOR ' | ') AS product_names,
    GROUP_CONCAT(p.sku ORDER BY p.product_id SEPARATOR ' | ') AS skus
FROM products p
INNER JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.price IS NOT NULL
GROUP BY pc.category_id, pc.category_name, p.price
HAVING COUNT(*) > 1 AND COUNT(*) < 10  -- Avoid grouping truly common prices
ORDER BY product_count DESC, pc.category_name, p.price;

-- ========================================
-- 5. DUPLICATE DETECTION BY DESCRIPTION SIMILARITY
-- ========================================
SELECT DISTINCT
    p1.product_id AS product_1_id,
    p1.product_name AS product_1_name,
    p1.sku AS product_1_sku,
    p2.product_id AS product_2_id,
    p2.product_name AS product_2_name,
    p2.sku AS product_2_sku,
    pc.category_name,
    'Similar Descriptions' AS match_type,
    LENGTH(p1.description) AS desc_1_length,
    LENGTH(p2.description) AS desc_2_length
FROM products p1
INNER JOIN products p2 ON p1.product_id < p2.product_id
LEFT JOIN product_categories pc ON p1.category_id = pc.category_id
WHERE p1.category_id = p2.category_id
  AND p1.description IS NOT NULL 
  AND p2.description IS NOT NULL
  AND LENGTH(p1.description) > 20
  AND LENGTH(p2.description) > 20
  AND (
    -- Same description
    p1.description = p2.description
    -- Description starts with same text
    OR LEFT(p1.description, 50) = LEFT(p2.description, 50)
  )
ORDER BY pc.category_name, p1.product_id;

-- ========================================
-- 6. POTENTIAL DUPLICATES - COMPREHENSIVE SCORING
-- ========================================
SELECT 
    p1.product_id AS product_1_id,
    p1.sku AS sku_1,
    p1.product_name AS name_1,
    p1.price AS price_1,
    p1.status AS status_1,
    p2.product_id AS product_2_id,
    p2.sku AS sku_2,
    p2.product_name AS name_2,
    p2.price AS price_2,
    p2.status AS status_2,
    pc.category_name,
    (
        -- Name similarity (40 points max)
        CASE 
            WHEN p1.product_name = p2.product_name THEN 40
            WHEN SOUNDEX(p1.product_name) = SOUNDEX(p2.product_name) THEN 30
            WHEN SUBSTRING_INDEX(LOWER(p1.product_name), ' ', 3) = SUBSTRING_INDEX(LOWER(p2.product_name), ' ', 3) THEN 25
            WHEN LOWER(p1.product_name) LIKE CONCAT('%', LOWER(SUBSTRING_INDEX(p2.product_name, ' ', 2)), '%') THEN 15
            ELSE 0
        END +
        -- Price similarity (30 points max)
        CASE 
            WHEN p1.price = p2.price THEN 30
            WHEN ABS(p1.price - p2.price) / GREATEST(p1.price, p2.price) < 0.05 THEN 25  -- Within 5%
            WHEN ABS(p1.price - p2.price) / GREATEST(p1.price, p2.price) < 0.10 THEN 15  -- Within 10%
            ELSE 0
        END +
        -- Description similarity (30 points max)
        CASE 
            WHEN p1.description = p2.description THEN 30
            WHEN LEFT(p1.description, 50) = LEFT(p2.description, 50) THEN 20
            WHEN p1.description IS NULL AND p2.description IS NULL THEN 10
            ELSE 0
        END
    ) AS duplicate_score,
    CONCAT(
        CASE WHEN SOUNDEX(p1.product_name) = SOUNDEX(p2.product_name) THEN 'Similar Name, ' ELSE '' END,
        CASE WHEN ABS(p1.price - p2.price) / GREATEST(p1.price, p2.price) < 0.10 THEN 'Similar Price, ' ELSE '' END,
        CASE WHEN p1.description = p2.description THEN 'Same Description, ' ELSE '' END
    ) AS match_reasons
FROM products p1
INNER JOIN products p2 ON p1.product_id < p2.product_id
LEFT JOIN product_categories pc ON p1.category_id = pc.category_id
WHERE p1.category_id = p2.category_id
HAVING duplicate_score >= 40  -- Threshold for potential duplicates
ORDER BY duplicate_score DESC, pc.category_name;

-- ========================================
-- 7. DUPLICATE DETECTION WITH SALES IMPACT
-- ========================================
SELECT 
    p1.product_id AS product_1_id,
    p1.sku AS sku_1,
    p1.product_name AS name_1,
    COUNT(DISTINCT oi1.order_item_id) AS orders_1,
    COALESCE(SUM(oi1.quantity), 0) AS units_sold_1,
    COALESCE(SUM(oi1.subtotal), 0) AS revenue_1,
    p2.product_id AS product_2_id,
    p2.sku AS sku_2,
    p2.product_name AS name_2,
    COUNT(DISTINCT oi2.order_item_id) AS orders_2,
    COALESCE(SUM(oi2.quantity), 0) AS units_sold_2,
    COALESCE(SUM(oi2.subtotal), 0) AS revenue_2,
    pc.category_name,
    'Potential Duplicate' AS issue_type
FROM products p1
INNER JOIN products p2 ON p1.product_id < p2.product_id
LEFT JOIN product_categories pc ON p1.category_id = pc.category_id
LEFT JOIN order_items oi1 ON p1.product_id = oi1.product_id
LEFT JOIN order_items oi2 ON p2.product_id = oi2.product_id
WHERE p1.category_id = p2.category_id
  AND (
    -- Similar names
    SOUNDEX(p1.product_name) = SOUNDEX(p2.product_name)
    OR SUBSTRING_INDEX(LOWER(p1.product_name), ' ', 3) = SUBSTRING_INDEX(LOWER(p2.product_name), ' ', 3)
    -- Similar prices
    OR ABS(p1.price - p2.price) / GREATEST(p1.price, p2.price) < 0.05
  )
GROUP BY p1.product_id, p1.sku, p1.product_name, 
         p2.product_id, p2.sku, p2.product_name, pc.category_name
HAVING (units_sold_1 > 0 OR units_sold_2 > 0)  -- At least one has sales
ORDER BY (revenue_1 + revenue_2) DESC;

-- ========================================
-- 8. SKU PATTERN ANALYSIS (Find Inconsistent SKU Formats)
-- ========================================
SELECT 
    CASE 
        WHEN sku REGEXP '^[A-Z]{3,4}-[A-Z]{2}-[0-9]{3}$' THEN 'Standard Format (AAA-BB-123)'
        WHEN sku REGEXP '^[A-Z]+-[0-9]+$' THEN 'Simple Format (AAA-123)'
        WHEN sku REGEXP '^[0-9]+$' THEN 'Numbers Only'
        WHEN sku REGEXP '^[A-Z]+$' THEN 'Letters Only'
        ELSE 'Non-Standard Format'
    END AS sku_format,
    COUNT(*) AS product_count,
    GROUP_CONCAT(DISTINCT SUBSTRING(sku, 1, 10) ORDER BY sku LIMIT 5) AS example_skus
FROM products
WHERE sku IS NOT NULL AND TRIM(sku) != ''
GROUP BY sku_format
ORDER BY product_count DESC;

-- ========================================
-- 9. POTENTIAL VERSION DUPLICATES (v1, v2, etc.)
-- ========================================
SELECT 
    p1.product_id AS older_product_id,
    p1.sku AS older_sku,
    p1.product_name AS older_name,
    p1.price AS older_price,
    p1.status AS older_status,
    p1.created_at AS older_created,
    p2.product_id AS newer_product_id,
    p2.sku AS newer_sku,
    p2.product_name AS newer_name,
    p2.price AS newer_price,
    p2.status AS newer_status,
    p2.created_at AS newer_created,
    pc.category_name,
    'Possible Version Variants' AS duplicate_type
FROM products p1
INNER JOIN products p2 ON p1.product_id < p2.product_id AND p1.category_id = p2.category_id
LEFT JOIN product_categories pc ON p1.category_id = pc.category_id
WHERE (
    -- SKU pattern matching (base SKU with version suffix)
    p1.sku LIKE CONCAT(LEFT(p2.sku, LENGTH(p2.sku) - 3), '%')
    OR p2.sku LIKE CONCAT(LEFT(p1.sku, LENGTH(p1.sku) - 3), '%')
    -- Name contains version indicators
    OR (
        REPLACE(REPLACE(LOWER(p1.product_name), ' v1', ''), ' v2', '') = 
        REPLACE(REPLACE(LOWER(p2.product_name), ' v1', ''), ' v2', '')
    )
    OR (
        REPLACE(REPLACE(REPLACE(LOWER(p1.product_name), ' 2023', ''), ' 2024', ''), ' 2025', '') = 
        REPLACE(REPLACE(REPLACE(LOWER(p2.product_name), ' 2023', ''), ' 2024', ''), ' 2025', '')
    )
)
ORDER BY pc.category_name, p1.created_at;

-- ========================================
-- 10. CASE-SENSITIVE DUPLICATE DETECTION
-- ========================================
SELECT 
    UPPER(product_name) AS normalized_name,
    COUNT(*) AS variation_count,
    GROUP_CONCAT(product_id ORDER BY product_id) AS product_ids,
    GROUP_CONCAT(product_name ORDER BY product_id SEPARATOR ' | ') AS name_variations,
    GROUP_CONCAT(sku ORDER BY product_id SEPARATOR ' | ') AS skus,
    MIN(price) AS min_price,
    MAX(price) AS max_price
FROM products
WHERE product_name IS NOT NULL
GROUP BY UPPER(product_name)
HAVING COUNT(*) > 1
ORDER BY variation_count DESC;

-- ========================================
-- 11. PRODUCTS WITH LEADING/TRAILING SPACES
-- ========================================
SELECT 
    product_id,
    CONCAT('"', sku, '"') AS sku_with_quotes,
    CONCAT('"', product_name, '"') AS name_with_quotes,
    LENGTH(sku) AS sku_length,
    LENGTH(TRIM(sku)) AS sku_trimmed_length,
    LENGTH(product_name) AS name_length,
    LENGTH(TRIM(product_name)) AS name_trimmed_length,
    'Whitespace Issue' AS issue_type
FROM products
WHERE sku != TRIM(sku)
   OR product_name != TRIM(product_name)
ORDER BY product_id;

-- ========================================
-- 12. CROSS-CATEGORY DUPLICATES (Same product in different categories)
-- ========================================
SELECT 
    p1.product_id AS product_1_id,
    p1.product_name AS product_1_name,
    pc1.category_name AS category_1,
    p2.product_id AS product_2_id,
    p2.product_name AS product_2_name,
    pc2.category_name AS category_2,
    'Cross-Category Duplicate' AS issue_type
FROM products p1
INNER JOIN products p2 ON p1.product_id < p2.product_id
INNER JOIN product_categories pc1 ON p1.category_id = pc1.category_id
INNER JOIN product_categories pc2 ON p2.category_id = pc2.category_id
WHERE p1.category_id != p2.category_id
  AND (
    p1.sku = p2.sku
    OR p1.product_name = p2.product_name
    OR SOUNDEX(p1.product_name) = SOUNDEX(p2.product_name)
  )
ORDER BY p1.product_name;

-- ========================================
-- 13. DUPLICATE COST/PRICE COMBINATIONS
-- ========================================
SELECT 
    p.price,
    p.cost,
    p.price - p.cost AS margin,
    pc.category_name,
    COUNT(*) AS product_count,
    GROUP_CONCAT(p.product_id ORDER BY p.product_id LIMIT 10) AS sample_product_ids,
    GROUP_CONCAT(p.product_name ORDER BY p.product_id LIMIT 5 SEPARATOR ' | ') AS sample_names
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE p.price IS NOT NULL AND p.cost IS NOT NULL
GROUP BY p.price, p.cost, pc.category_id, pc.category_name
HAVING COUNT(*) > 1
ORDER BY product_count DESC, p.price DESC;

-- ========================================
-- 14. PRODUCTS WITH SAME STOCK QUANTITY (Potential Copy-Paste Error)
-- ========================================
SELECT 
    stock_quantity,
    pc.category_name,
    COUNT(*) AS product_count,
    GROUP_CONCAT(p.product_id ORDER BY p.product_id LIMIT 10) AS product_ids,
    GROUP_CONCAT(p.product_name ORDER BY p.product_id LIMIT 5 SEPARATOR ' | ') AS sample_names
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE stock_quantity IS NOT NULL
GROUP BY stock_quantity, pc.category_id, pc.category_name
HAVING COUNT(*) >= 5  -- 5 or more products with exact same stock
ORDER BY product_count DESC, stock_quantity DESC;

-- ========================================
-- 15. SUMMARY REPORT - DUPLICATE ISSUES BY TYPE
-- ========================================
SELECT 
    'DUPLICATE DETECTION SUMMARY' AS report_title,
    (SELECT COUNT(DISTINCT sku) FROM (
        SELECT sku FROM products WHERE sku IS NOT NULL GROUP BY sku HAVING COUNT(*) > 1
    ) sub) AS duplicate_skus,
    (SELECT COUNT(DISTINCT product_name) FROM (
        SELECT product_name FROM products WHERE product_name IS NOT NULL GROUP BY product_name HAVING COUNT(*) > 1
    ) sub) AS duplicate_names,
    (SELECT COUNT(*) FROM products WHERE sku != TRIM(sku) OR product_name != TRIM(product_name)) AS whitespace_issues,
    (SELECT COUNT(DISTINCT p1.product_id) FROM products p1 
     INNER JOIN products p2 ON p1.product_id < p2.product_id 
     WHERE p1.category_id = p2.category_id 
     AND SOUNDEX(p1.product_name) = SOUNDEX(p2.product_name)) AS phonetically_similar,
    (SELECT COUNT(*) FROM (
        SELECT UPPER(product_name) FROM products GROUP BY UPPER(product_name) HAVING COUNT(*) > 1
    ) sub) AS case_variation_duplicates;

-- ========================================
-- 16. RECOMMENDED MERGE CANDIDATES (High Confidence Duplicates)
-- ========================================
SELECT 
    p1.product_id AS keep_product_id,
    p1.sku AS keep_sku,
    p1.product_name AS keep_name,
    p1.status AS keep_status,
    COUNT(DISTINCT oi1.order_item_id) AS keep_orders,
    COALESCE(SUM(oi1.subtotal), 0) AS keep_revenue,
    p2.product_id AS remove_product_id,
    p2.sku AS remove_sku,
    p2.product_name AS remove_name,
    p2.status AS remove_status,
    COUNT(DISTINCT oi2.order_item_id) AS remove_orders,
    COALESCE(SUM(oi2.subtotal), 0) AS remove_revenue,
    CASE 
        WHEN p1.sku = p2.sku THEN 'Exact SKU Match'
        WHEN p1.product_name = p2.product_name THEN 'Exact Name Match'
        WHEN p1.description = p2.description AND p1.price = p2.price THEN 'Exact Description and Price Match'
        ELSE 'High Similarity'
    END AS merge_reason,
    'Consider merging these products' AS recommendation
FROM products p1
INNER JOIN products p2 ON p1.product_id < p2.product_id
LEFT JOIN order_items oi1 ON p1.product_id = oi1.product_id
LEFT JOIN order_items oi2 ON p2.product_id = oi2.product_id
WHERE (
    p1.sku = p2.sku
    OR p1.product_name = p2.product_name
    OR (p1.description = p2.description AND p1.price = p2.price AND p1.description IS NOT NULL)
)
GROUP BY p1.product_id, p1.sku, p1.product_name, p1.status,
         p2.product_id, p2.sku, p2.product_name, p2.status
ORDER BY (keep_revenue + remove_revenue) DESC;

-- Display completion message
SELECT 'Product duplicate analysis complete. Review results above to identify and resolve duplicate products.' AS Status;