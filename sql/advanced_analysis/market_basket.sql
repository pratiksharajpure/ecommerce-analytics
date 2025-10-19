-- ========================================
-- MARKET BASKET ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Product Associations & Bundle Recommendations
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. PRODUCT ASSOCIATION RULES
-- Discovers which products are frequently bought together
-- Calculates Support, Confidence, and Lift metrics
-- ========================================

WITH order_products AS (
    SELECT 
        o.order_id,
        o.customer_id,
        oi.product_id,
        p.product_name,
        pc.category_name,
        oi.quantity,
        oi.unit_price
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
),
total_orders AS (
    SELECT COUNT(DISTINCT order_id) AS total_order_count
    FROM order_products
),
product_frequency AS (
    SELECT 
        product_id,
        product_name,
        category_name,
        COUNT(DISTINCT order_id) AS order_count
    FROM order_products
    GROUP BY product_id, product_name, category_name
),
product_pairs AS (
    SELECT 
        op1.product_id AS product_a_id,
        op1.product_name AS product_a_name,
        op1.category_name AS product_a_category,
        op2.product_id AS product_b_id,
        op2.product_name AS product_b_name,
        op2.category_name AS product_b_category,
        COUNT(DISTINCT op1.order_id) AS co_occurrence_count
    FROM order_products op1
    JOIN order_products op2 ON op1.order_id = op2.order_id
        AND op1.product_id < op2.product_id  -- Avoid duplicates and self-pairs
    GROUP BY op1.product_id, op1.product_name, op1.category_name,
             op2.product_id, op2.product_name, op2.category_name
    HAVING co_occurrence_count >= 3  -- Minimum co-occurrence threshold
)
SELECT 
    pp.product_a_id,
    pp.product_a_name,
    pp.product_a_category,
    pp.product_b_id,
    pp.product_b_name,
    pp.product_b_category,
    pp.co_occurrence_count AS times_bought_together,
    -- Support: P(A AND B) - Percentage of all orders containing both items
    ROUND(pp.co_occurrence_count * 100.0 / t.total_order_count, 2) AS support_pct,
    -- Confidence: P(B|A) - If customer buys A, probability they buy B
    ROUND(pp.co_occurrence_count * 100.0 / pfa.order_count, 2) AS confidence_a_to_b_pct,
    -- Confidence: P(A|B) - If customer buys B, probability they buy A
    ROUND(pp.co_occurrence_count * 100.0 / pfb.order_count, 2) AS confidence_b_to_a_pct,
    -- Lift: Measures how much more likely B is purchased when A is purchased
    ROUND(
        (pp.co_occurrence_count * t.total_order_count * 1.0) / 
        (pfa.order_count * pfb.order_count),
        2
    ) AS lift_score,
    -- Classification
    CASE 
        WHEN (pp.co_occurrence_count * t.total_order_count * 1.0) / 
             (pfa.order_count * pfb.order_count) >= 3 THEN 'Strong Association'
        WHEN (pp.co_occurrence_count * t.total_order_count * 1.0) / 
             (pfa.order_count * pfb.order_count) >= 2 THEN 'Moderate Association'
        WHEN (pp.co_occurrence_count * t.total_order_count * 1.0) / 
             (pfa.order_count * pfb.order_count) >= 1.5 THEN 'Weak Association'
        ELSE 'No Association'
    END AS association_strength
FROM product_pairs pp
CROSS JOIN total_orders t
JOIN product_frequency pfa ON pp.product_a_id = pfa.product_id
JOIN product_frequency pfb ON pp.product_b_id = pfb.product_id
WHERE (pp.co_occurrence_count * t.total_order_count * 1.0) / 
      (pfa.order_count * pfb.order_count) >= 1.5  -- Lift >= 1.5
ORDER BY lift_score DESC, co_occurrence_count DESC
LIMIT 100;

-- ========================================
-- 2. FREQUENTLY BOUGHT TOGETHER - TOP PAIRS
-- Identifies best product pairs for cross-selling
-- ========================================

WITH recent_orders AS (
    SELECT 
        o.order_id,
        o.customer_id,
        o.order_date,
        o.total_amount
    FROM orders o
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
),
product_pairs AS (
    SELECT 
        oi1.product_id AS product_a_id,
        p1.product_name AS product_a_name,
        p1.price AS product_a_price,
        pc1.category_name AS product_a_category,
        oi2.product_id AS product_b_id,
        p2.product_name AS product_b_name,
        p2.price AS product_b_price,
        pc2.category_name AS product_b_category,
        COUNT(DISTINCT ro.order_id) AS pair_frequency,
        SUM(oi1.quantity) AS product_a_total_qty,
        SUM(oi2.quantity) AS product_b_total_qty,
        AVG(ro.total_amount) AS avg_order_value,
        COUNT(DISTINCT ro.customer_id) AS unique_customers
    FROM recent_orders ro
    JOIN order_items oi1 ON ro.order_id = oi1.order_id
    JOIN order_items oi2 ON ro.order_id = oi2.order_id
        AND oi1.product_id < oi2.product_id
    JOIN products p1 ON oi1.product_id = p1.product_id
    JOIN products p2 ON oi2.product_id = p2.product_id
    LEFT JOIN product_categories pc1 ON p1.category_id = pc1.category_id
    LEFT JOIN product_categories pc2 ON p2.category_id = pc2.category_id
    GROUP BY oi1.product_id, p1.product_name, p1.price, pc1.category_name,
             oi2.product_id, p2.product_name, p2.price, pc2.category_name
    HAVING pair_frequency >= 5
)
SELECT 
    product_a_id,
    product_a_name,
    ROUND(product_a_price, 2) AS product_a_price,
    product_a_category,
    product_b_id,
    product_b_name,
    ROUND(product_b_price, 2) AS product_b_price,
    product_b_category,
    pair_frequency AS times_bought_together,
    unique_customers,
    product_a_total_qty AS total_qty_a,
    product_b_total_qty AS total_qty_b,
    ROUND(avg_order_value, 2) AS avg_order_value_with_pair,
    ROUND(product_a_price + product_b_price, 2) AS bundle_price,
    -- Potential bundle discount
    ROUND((product_a_price + product_b_price) * 0.90, 2) AS suggested_bundle_price_10pct_off,
    -- Cross-sell recommendation priority
    CASE 
        WHEN pair_frequency >= 20 THEN 'High Priority'
        WHEN pair_frequency >= 10 THEN 'Medium Priority'
        ELSE 'Standard'
    END AS cross_sell_priority,
    -- Best placement strategy
    CASE 
        WHEN product_a_category = product_b_category THEN 'Same Category Display'
        ELSE 'Cross-Category Recommendation'
    END AS placement_strategy
FROM product_pairs
ORDER BY pair_frequency DESC, avg_order_value DESC
LIMIT 50;

-- ========================================
-- 3. PRODUCT BUNDLE RECOMMENDATIONS
-- Creates optimized 3-product bundles
-- ========================================

WITH order_products AS (
    SELECT 
        o.order_id,
        oi.product_id,
        p.product_name,
        p.price,
        pc.category_name
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
),
product_triplets AS (
    SELECT 
        op1.product_id AS product_1_id,
        op1.product_name AS product_1_name,
        op1.price AS product_1_price,
        op1.category_name AS product_1_category,
        op2.product_id AS product_2_id,
        op2.product_name AS product_2_name,
        op2.price AS product_2_price,
        op2.category_name AS product_2_category,
        op3.product_id AS product_3_id,
        op3.product_name AS product_3_name,
        op3.price AS product_3_price,
        op3.category_name AS product_3_category,
        COUNT(DISTINCT op1.order_id) AS bundle_frequency
    FROM order_products op1
    JOIN order_products op2 ON op1.order_id = op2.order_id
        AND op1.product_id < op2.product_id
    JOIN order_products op3 ON op1.order_id = op3.order_id
        AND op2.product_id < op3.product_id
    GROUP BY op1.product_id, op1.product_name, op1.price, op1.category_name,
             op2.product_id, op2.product_name, op2.price, op2.category_name,
             op3.product_id, op3.product_name, op3.price, op3.category_name
    HAVING bundle_frequency >= 3
)
SELECT 
    CONCAT('BUNDLE-', ROW_NUMBER() OVER (ORDER BY bundle_frequency DESC)) AS bundle_id,
    product_1_name,
    ROUND(product_1_price, 2) AS price_1,
    product_1_category,
    product_2_name,
    ROUND(product_2_price, 2) AS price_2,
    product_2_category,
    product_3_name,
    ROUND(product_3_price, 2) AS price_3,
    product_3_category,
    bundle_frequency AS times_purchased_together,
    ROUND(product_1_price + product_2_price + product_3_price, 2) AS total_retail_price,
    -- Bundle pricing strategies
    ROUND((product_1_price + product_2_price + product_3_price) * 0.85, 2) AS bundle_price_15pct_off,
    ROUND((product_1_price + product_2_price + product_3_price) * 0.80, 2) AS bundle_price_20pct_off,
    ROUND((product_1_price + product_2_price + product_3_price) * 0.15, 2) AS savings_15pct,
    ROUND((product_1_price + product_2_price + product_3_price) * 0.20, 2) AS savings_20pct,
    -- Bundle characteristics
    CASE 
        WHEN product_1_category = product_2_category 
             AND product_2_category = product_3_category THEN 'Single Category Bundle'
        WHEN product_1_category = product_2_category 
             OR product_2_category = product_3_category 
             OR product_1_category = product_3_category THEN 'Mixed Category Bundle'
        ELSE 'Cross-Category Bundle'
    END AS bundle_type,
    -- Marketing potential
    CASE 
        WHEN bundle_frequency >= 10 THEN 'Featured Bundle'
        WHEN bundle_frequency >= 5 THEN 'Standard Bundle'
        ELSE 'Niche Bundle'
    END AS marketing_tier
FROM product_triplets
ORDER BY bundle_frequency DESC, total_retail_price DESC
LIMIT 30;

-- ========================================
-- 4. CATEGORY AFFINITY ANALYSIS
-- Which product categories are purchased together
-- ========================================

WITH order_categories AS (
    SELECT DISTINCT
        o.order_id,
        o.customer_id,
        pc.category_id,
        pc.category_name
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
),
category_pairs AS (
    SELECT 
        oc1.category_name AS category_a,
        oc2.category_name AS category_b,
        COUNT(DISTINCT oc1.order_id) AS co_purchase_count,
        COUNT(DISTINCT oc1.customer_id) AS unique_customers
    FROM order_categories oc1
    JOIN order_categories oc2 ON oc1.order_id = oc2.order_id
        AND oc1.category_id < oc2.category_id
    GROUP BY oc1.category_name, oc2.category_name
    HAVING co_purchase_count >= 5
),
category_totals AS (
    SELECT 
        category_name,
        COUNT(DISTINCT order_id) AS total_orders
    FROM order_categories
    GROUP BY category_name
),
total_orders AS (
    SELECT COUNT(DISTINCT order_id) AS order_count
    FROM order_categories
)
SELECT 
    cp.category_a,
    cp.category_b,
    cp.co_purchase_count AS orders_with_both,
    cp.unique_customers,
    ct1.total_orders AS category_a_total_orders,
    ct2.total_orders AS category_b_total_orders,
    ROUND(cp.co_purchase_count * 100.0 / ct1.total_orders, 2) AS pct_of_category_a_orders,
    ROUND(cp.co_purchase_count * 100.0 / ct2.total_orders, 2) AS pct_of_category_b_orders,
    -- Affinity score (lift)
    ROUND(
        (cp.co_purchase_count * t.order_count * 1.0) / 
        (ct1.total_orders * ct2.total_orders),
        2
    ) AS affinity_score,
    CASE 
        WHEN (cp.co_purchase_count * t.order_count * 1.0) / 
             (ct1.total_orders * ct2.total_orders) >= 2.5 THEN 'Very Strong Affinity'
        WHEN (cp.co_purchase_count * t.order_count * 1.0) / 
             (ct1.total_orders * ct2.total_orders) >= 2.0 THEN 'Strong Affinity'
        WHEN (cp.co_purchase_count * t.order_count * 1.0) / 
             (ct1.total_orders * ct2.total_orders) >= 1.5 THEN 'Moderate Affinity'
        ELSE 'Weak Affinity'
    END AS affinity_strength
FROM category_pairs cp
CROSS JOIN total_orders t
JOIN category_totals ct1 ON cp.category_a = ct1.category_name
JOIN category_totals ct2 ON cp.category_b = ct2.category_name
ORDER BY affinity_score DESC, co_purchase_count DESC;

-- ========================================
-- 5. PERSONALIZED RECOMMENDATIONS BY CUSTOMER
-- "Customers who bought X also bought Y"
-- ========================================

WITH customer_purchases AS (
    SELECT DISTINCT
        o.customer_id,
        oi.product_id,
        p.product_name,
        MAX(o.order_date) AS last_purchase_date
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    GROUP BY o.customer_id, oi.product_id, p.product_name
),
similar_customer_purchases AS (
    SELECT 
        cp1.customer_id AS target_customer,
        cp1.product_id AS owned_product_id,
        cp1.product_name AS owned_product,
        cp2.product_id AS recommended_product_id,
        p.product_name AS recommended_product,
        p.price AS recommended_price,
        pc.category_name AS recommended_category,
        COUNT(DISTINCT cp2.customer_id) AS customers_who_also_bought,
        AVG(p.price) AS avg_price
    FROM customer_purchases cp1
    JOIN customer_purchases cp2 ON cp1.product_id = cp2.product_id
        AND cp1.customer_id != cp2.customer_id
    JOIN products p ON cp2.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE NOT EXISTS (
        -- Exclude products the target customer already owns
        SELECT 1 FROM customer_purchases cp3
        WHERE cp3.customer_id = cp1.customer_id
            AND cp3.product_id = cp2.product_id
    )
    GROUP BY cp1.customer_id, cp1.product_id, cp1.product_name,
             cp2.product_id, p.product_name, p.price, pc.category_name
    HAVING customers_who_also_bought >= 3
)
SELECT 
    target_customer AS customer_id,
    owned_product AS based_on_purchase,
    recommended_product,
    ROUND(recommended_price, 2) AS price,
    recommended_category,
    customers_who_also_bought,
    -- Recommendation strength
    CASE 
        WHEN customers_who_also_bought >= 20 THEN 'Highly Recommended'
        WHEN customers_who_also_bought >= 10 THEN 'Recommended'
        ELSE 'Suggested'
    END AS recommendation_strength,
    -- Ranking within customer's recommendations
    ROW_NUMBER() OVER (
        PARTITION BY target_customer 
        ORDER BY customers_who_also_bought DESC
    ) AS recommendation_rank
FROM similar_customer_purchases
WHERE ROW_NUMBER() OVER (
    PARTITION BY target_customer 
    ORDER BY customers_who_also_bought DESC
) <= 5  -- Top 5 recommendations per customer
ORDER BY target_customer, recommendation_rank;

-- ========================================
-- 6. COMPLEMENTARY PRODUCT SUGGESTIONS
-- Products that complete a customer's collection
-- ========================================

WITH customer_category_coverage AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        COUNT(DISTINCT pc.category_id) AS categories_purchased,
        GROUP_CONCAT(DISTINCT pc.category_name ORDER BY pc.category_name) AS purchased_categories,
        SUM(o.total_amount) AS lifetime_value
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email
),
popular_products_by_category AS (
    SELECT 
        pc.category_id,
        pc.category_name,
        p.product_id,
        p.product_name,
        p.price,
        COUNT(DISTINCT o.order_id) AS times_ordered,
        AVG(oi.quantity) AS avg_quantity_per_order,
        COUNT(DISTINCT o.customer_id) AS unique_buyers
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND p.status = 'active'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY pc.category_id, pc.category_name, p.product_id, p.product_name, p.price
)
SELECT 
    ccc.customer_id,
    ccc.customer_name,
    ccc.email,
    ccc.categories_purchased,
    ROUND(ccc.lifetime_value, 2) AS lifetime_value,
    ppbc.category_name AS suggested_category,
    ppbc.product_name AS suggested_product,
    ROUND(ppbc.price, 2) AS product_price,
    ppbc.times_ordered AS popularity_score,
    ppbc.unique_buyers,
    -- Why this suggestion
    'Completes product portfolio' AS suggestion_reason,
    -- Ranking
    ROW_NUMBER() OVER (
        PARTITION BY ccc.customer_id 
        ORDER BY ppbc.times_ordered DESC
    ) AS suggestion_rank
FROM customer_category_coverage ccc
CROSS JOIN popular_products_by_category ppbc
WHERE NOT EXISTS (
    -- Only suggest categories they haven't purchased from
    SELECT 1 FROM orders o2
    JOIN order_items oi2 ON o2.order_id = oi2.order_id
    JOIN products p2 ON oi2.product_id = p2.product_id
    WHERE o2.customer_id = ccc.customer_id
        AND p2.category_id = ppbc.category_id
)
    AND ccc.lifetime_value >= 100  -- Focus on engaged customers
    AND ROW_NUMBER() OVER (
        PARTITION BY ccc.customer_id 
        ORDER BY ppbc.times_ordered DESC
    ) <= 3  -- Top 3 suggestions per customer
ORDER BY ccc.lifetime_value DESC, ccc.customer_id, suggestion_rank;

-- ========================================
-- 7. BUNDLE PERFORMANCE METRICS
-- Tracks actual performance of product combinations
-- ========================================

WITH product_bundles AS (
    SELECT 
        o.order_id,
        o.order_date,
        o.total_amount,
        COUNT(DISTINCT oi.product_id) AS products_in_order,
        SUM(oi.subtotal) AS bundle_subtotal,
        GROUP_CONCAT(DISTINCT p.product_name ORDER BY p.product_name SEPARATOR ' + ') AS product_combination
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY o.order_id, o.order_date, o.total_amount
    HAVING products_in_order BETWEEN 2 AND 5
),
bundle_stats AS (
    SELECT 
        products_in_order AS bundle_size,
        product_combination,
        COUNT(*) AS order_frequency,
        ROUND(AVG(total_amount), 2) AS avg_order_value,
        ROUND(SUM(total_amount), 2) AS total_revenue,
        ROUND(AVG(bundle_subtotal), 2) AS avg_bundle_value,
        MIN(order_date) AS first_ordered,
        MAX(order_date) AS last_ordered
    FROM product_bundles
    GROUP BY products_in_order, product_combination
    HAVING order_frequency >= 2
)
SELECT 
    bundle_size,
    product_combination,
    order_frequency,
    avg_order_value,
    total_revenue,
    avg_bundle_value,
    first_ordered,
    last_ordered,
    DATEDIFF(last_ordered, first_ordered) AS days_active,
    ROUND(total_revenue / order_frequency, 2) AS revenue_per_occurrence,
    CASE 
        WHEN order_frequency >= 10 THEN 'High Performing Bundle'
        WHEN order_frequency >= 5 THEN 'Moderate Performing Bundle'
        ELSE 'Low Performing Bundle'
    END AS performance_tier
FROM bundle_stats
ORDER BY total_revenue DESC, order_frequency DESC
LIMIT 50;

-- ========================================
-- End of Market Basket Analysis
-- ========================================