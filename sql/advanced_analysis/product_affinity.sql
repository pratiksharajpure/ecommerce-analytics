-- ========================================
-- PRODUCT AFFINITY ANALYSIS
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. MARKET BASKET ANALYSIS
-- Products frequently bought together
-- ========================================
WITH product_pairs AS (
    SELECT 
        oi1.product_id AS product_a_id,
        oi2.product_id AS product_b_id,
        COUNT(DISTINCT oi1.order_id) AS times_bought_together,
        COUNT(DISTINCT CASE WHEN oi1.product_id < oi2.product_id THEN oi1.order_id END) AS unique_combinations
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id 
        AND oi1.product_id != oi2.product_id
    JOIN orders o ON oi1.order_id = o.order_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY oi1.product_id, oi2.product_id
    HAVING times_bought_together >= 3
),
product_stats AS (
    SELECT 
        p.product_id,
        COUNT(DISTINCT oi.order_id) AS total_orders
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY p.product_id
)
SELECT 
    p1.product_name AS product_a,
    p1.sku AS product_a_sku,
    pc1.category_name AS product_a_category,
    p2.product_name AS product_b,
    p2.sku AS product_b_sku,
    pc2.category_name AS product_b_category,
    pp.times_bought_together,
    ps1.total_orders AS product_a_total_orders,
    ps2.total_orders AS product_b_total_orders,
    -- Calculate confidence: P(B|A) = times together / times A bought
    ROUND(pp.times_bought_together * 100.0 / ps1.total_orders, 2) AS confidence_a_to_b,
    -- Calculate lift: confidence / support of B
    ROUND((pp.times_bought_together * 1.0 / ps1.total_orders) / 
          (ps2.total_orders * 1.0 / (SELECT COUNT(DISTINCT order_id) FROM orders WHERE payment_status = 'paid')), 2) AS lift
FROM product_pairs pp
JOIN products p1 ON pp.product_a_id = p1.product_id
JOIN products p2 ON pp.product_b_id = p2.product_id
JOIN product_categories pc1 ON p1.category_id = pc1.category_id
JOIN product_categories pc2 ON p2.category_id = pc2.category_id
JOIN product_stats ps1 ON pp.product_a_id = ps1.product_id
JOIN product_stats ps2 ON pp.product_b_id = ps2.product_id
WHERE pp.product_a_id < pp.product_b_id  -- Avoid duplicate pairs
    AND (pp.times_bought_together * 100.0 / ps1.total_orders) >= 20  -- Minimum 20% confidence
ORDER BY lift DESC, times_bought_together DESC
LIMIT 50;

-- ========================================
-- 2. PERSONALIZED PRODUCT RECOMMENDATIONS
-- Based on customer purchase history
-- ========================================
WITH customer_purchases AS (
    SELECT DISTINCT
        o.customer_id,
        oi.product_id,
        p.category_id
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
),
similar_customers AS (
    SELECT 
        cp1.customer_id AS target_customer,
        cp2.customer_id AS similar_customer,
        COUNT(DISTINCT cp1.product_id) AS common_products,
        COUNT(DISTINCT cp2.product_id) AS similar_customer_total_products
    FROM customer_purchases cp1
    JOIN customer_purchases cp2 ON cp1.product_id = cp2.product_id 
        AND cp1.customer_id != cp2.customer_id
    GROUP BY cp1.customer_id, cp2.customer_id
    HAVING common_products >= 2
),
recommendations AS (
    SELECT 
        sc.target_customer,
        cp.product_id,
        COUNT(DISTINCT sc.similar_customer) AS recommended_by_customers,
        AVG(sc.common_products) AS avg_similarity_score,
        SUM(sc.common_products) AS total_similarity_score
    FROM similar_customers sc
    JOIN customer_purchases cp ON sc.similar_customer = cp.customer_id
    WHERE NOT EXISTS (
        SELECT 1 FROM customer_purchases cp2 
        WHERE cp2.customer_id = sc.target_customer 
        AND cp2.product_id = cp.product_id
    )
    GROUP BY sc.target_customer, cp.product_id
)
SELECT 
    r.target_customer,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    p.product_name,
    p.sku,
    pc.category_name,
    ROUND(p.price, 2) AS price,
    r.recommended_by_customers,
    ROUND(r.avg_similarity_score, 2) AS similarity_score,
    r.total_similarity_score,
    -- Calculate recommendation strength
    ROUND((r.recommended_by_customers * r.avg_similarity_score), 2) AS recommendation_strength,
    -- Product popularity
    COUNT(DISTINCT oi.order_id) AS times_ordered_overall,
    ROUND(AVG(rev.rating), 1) AS avg_rating,
    COUNT(DISTINCT rev.review_id) AS review_count
FROM recommendations r
JOIN customers c ON r.target_customer = c.customer_id
JOIN products p ON r.product_id = p.product_id
JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN reviews rev ON p.product_id = rev.product_id AND rev.status = 'approved'
WHERE p.status = 'active'
    AND p.stock_quantity > 0
GROUP BY r.target_customer, c.first_name, c.last_name, p.product_name, p.sku, 
         pc.category_name, p.price, r.recommended_by_customers, 
         r.avg_similarity_score, r.total_similarity_score
HAVING recommendation_strength >= 5
ORDER BY r.target_customer, recommendation_strength DESC
LIMIT 100;

-- ========================================
-- 3. CROSS-SELL OPPORTUNITIES
-- Identify products to recommend at checkout
-- ========================================
WITH cart_analysis AS (
    SELECT 
        oi1.product_id AS anchor_product,
        oi2.product_id AS cross_sell_product,
        COUNT(DISTINCT oi1.order_id) AS times_bought_together,
        AVG(oi2.unit_price) AS avg_cross_sell_price,
        SUM(oi2.subtotal) AS total_cross_sell_revenue
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id 
        AND oi1.product_id != oi2.product_id
    JOIN orders o ON oi1.order_id = o.order_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY oi1.product_id, oi2.product_id
),
anchor_product_stats AS (
    SELECT 
        oi.product_id,
        COUNT(DISTINCT oi.order_id) AS total_times_ordered
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY oi.product_id
)
SELECT 
    p1.product_name AS anchor_product_name,
    p1.sku AS anchor_sku,
    pc1.category_name AS anchor_category,
    ROUND(p1.price, 2) AS anchor_price,
    p2.product_name AS cross_sell_product_name,
    p2.sku AS cross_sell_sku,
    pc2.category_name AS cross_sell_category,
    ROUND(p2.price, 2) AS cross_sell_price,
    ca.times_bought_together,
    aps.total_times_ordered AS anchor_order_count,
    -- Attachment rate: what % of anchor purchases included cross-sell
    ROUND(ca.times_bought_together * 100.0 / aps.total_times_ordered, 2) AS attachment_rate_pct,
    -- Revenue potential
    ROUND(ca.total_cross_sell_revenue, 2) AS historical_cross_sell_revenue,
    -- Potential revenue if we improved attachment rate
    ROUND((aps.total_times_ordered - ca.times_bought_together) * ca.avg_cross_sell_price, 2) AS potential_additional_revenue,
    -- Product ratings
    ROUND(AVG(r2.rating), 1) AS cross_sell_avg_rating
FROM cart_analysis ca
JOIN products p1 ON ca.anchor_product = p1.product_id
JOIN products p2 ON ca.cross_sell_product = p2.product_id
JOIN product_categories pc1 ON p1.category_id = pc1.category_id
JOIN product_categories pc2 ON p2.category_id = pc2.category_id
JOIN anchor_product_stats aps ON ca.anchor_product = aps.product_id
LEFT JOIN reviews r2 ON p2.product_id = r2.product_id AND r2.status = 'approved'
WHERE ca.times_bought_together >= 5
    AND p1.status = 'active'
    AND p2.status = 'active'
    AND p2.stock_quantity > 0
    AND (ca.times_bought_together * 100.0 / aps.total_times_ordered) >= 15  -- At least 15% attachment
GROUP BY ca.anchor_product, ca.cross_sell_product, p1.product_name, p1.sku, 
         pc1.category_name, p1.price, p2.product_name, p2.sku, pc2.category_name, 
         p2.price, ca.times_bought_together, aps.total_times_ordered, 
         ca.total_cross_sell_revenue, ca.avg_cross_sell_price
ORDER BY potential_additional_revenue DESC
LIMIT 50;

-- ========================================
-- 4. UP-SELL OPPORTUNITIES
-- Identify premium product alternatives
-- ========================================
WITH product_upgrades AS (
    SELECT 
        p1.product_id AS base_product_id,
        p1.product_name AS base_product,
        p1.price AS base_price,
        p2.product_id AS premium_product_id,
        p2.product_name AS premium_product,
        p2.price AS premium_price,
        (p2.price - p1.price) AS price_difference,
        ((p2.price - p1.price) * 100.0 / p1.price) AS price_increase_pct
    FROM products p1
    JOIN products p2 ON p1.category_id = p2.category_id 
        AND p1.product_id != p2.product_id
        AND p2.price > p1.price
        AND p2.price <= p1.price * 1.5  -- Within 50% price increase
    WHERE p1.status = 'active' 
        AND p2.status = 'active'
        AND p1.stock_quantity > 0
        AND p2.stock_quantity > 0
),
product_performance AS (
    SELECT 
        oi.product_id,
        COUNT(DISTINCT oi.order_id) AS order_count,
        SUM(oi.subtotal) AS total_revenue,
        AVG(r.rating) AS avg_rating,
        COUNT(DISTINCT r.review_id) AS review_count,
        COUNT(DISTINCT ret.return_id) AS return_count
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    LEFT JOIN reviews r ON oi.product_id = r.product_id AND r.status = 'approved'
    LEFT JOIN returns ret ON oi.order_item_id = ret.order_item_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY oi.product_id
)
SELECT 
    pu.base_product,
    pc.category_name,
    ROUND(pu.base_price, 2) AS base_price,
    pp1.order_count AS base_order_count,
    ROUND(pp1.avg_rating, 1) AS base_rating,
    pu.premium_product,
    ROUND(pu.premium_price, 2) AS premium_price,
    ROUND(pu.price_difference, 2) AS price_difference,
    ROUND(pu.price_increase_pct, 1) AS price_increase_pct,
    pp2.order_count AS premium_order_count,
    ROUND(pp2.avg_rating, 1) AS premium_rating,
    pp2.review_count AS premium_review_count,
    -- Upsell potential score
    ROUND((pp1.order_count * pu.price_difference * 
           CASE 
               WHEN pp2.avg_rating >= pp1.avg_rating + 0.5 THEN 1.5
               WHEN pp2.avg_rating >= pp1.avg_rating THEN 1.2
               ELSE 1.0
           END), 2) AS upsell_revenue_potential,
    -- Calculate success probability
    ROUND(pp2.order_count * 100.0 / (pp1.order_count + pp2.order_count), 1) AS premium_preference_pct,
    -- Return rate comparison
    ROUND(pp1.return_count * 100.0 / NULLIF(pp1.order_count, 0), 1) AS base_return_rate_pct,
    ROUND(pp2.return_count * 100.0 / NULLIF(pp2.order_count, 0), 1) AS premium_return_rate_pct
FROM product_upgrades pu
JOIN product_categories pc ON pu.base_product_id = pc.category_id
JOIN product_performance pp1 ON pu.base_product_id = pp1.product_id
JOIN product_performance pp2 ON pu.premium_product_id = pp2.product_id
WHERE pp1.order_count >= 10  -- Base product has sufficient volume
    AND (pp2.avg_rating >= pp1.avg_rating OR pp2.avg_rating >= 4.0)  -- Premium is rated well
ORDER BY upsell_revenue_potential DESC
LIMIT 50;

-- ========================================
-- 5. CATEGORY AFFINITY ANALYSIS
-- Which product categories are bought together
-- ========================================
WITH category_pairs AS (
    SELECT 
        p1.category_id AS category_a,
        p2.category_id AS category_b,
        COUNT(DISTINCT oi1.order_id) AS times_bought_together,
        SUM(oi1.subtotal + oi2.subtotal) AS combined_revenue
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id 
        AND oi1.order_item_id < oi2.order_item_id
    JOIN products p1 ON oi1.product_id = p1.product_id
    JOIN products p2 ON oi2.product_id = p2.product_id
    JOIN orders o ON oi1.order_id = o.order_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
        AND p1.category_id != p2.category_id
    GROUP BY p1.category_id, p2.category_id
),
category_stats AS (
    SELECT 
        p.category_id,
        COUNT(DISTINCT oi.order_id) AS category_order_count
    FROM products p
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY p.category_id
)
SELECT 
    pc1.category_name AS category_a_name,
    pc2.category_name AS category_b_name,
    cp.times_bought_together,
    cs1.category_order_count AS category_a_total_orders,
    cs2.category_order_count AS category_b_total_orders,
    ROUND(cp.times_bought_together * 100.0 / cs1.category_order_count, 2) AS cross_category_rate_pct,
    ROUND(cp.combined_revenue, 2) AS combined_revenue,
    ROUND(cp.combined_revenue / cp.times_bought_together, 2) AS avg_combined_order_value,
    -- Calculate category affinity score
    ROUND((cp.times_bought_together * 1.0 / cs1.category_order_count) / 
          (cs2.category_order_count * 1.0 / (SELECT COUNT(DISTINCT order_id) FROM orders WHERE payment_status = 'paid')), 2) AS affinity_score
FROM category_pairs cp
JOIN product_categories pc1 ON cp.category_a = pc1.category_id
JOIN product_categories pc2 ON cp.category_b = pc2.category_id
JOIN category_stats cs1 ON cp.category_a = cs1.category_id
JOIN category_stats cs2 ON cp.category_b = cs2.category_id
WHERE cp.times_bought_together >= 5
ORDER BY affinity_score DESC, combined_revenue DESC
LIMIT 30;

-- ========================================
-- 6. PRODUCT BUNDLE RECOMMENDATIONS
-- Suggest optimal product bundles
-- ========================================
WITH three_product_combos AS (
    SELECT 
        oi1.product_id AS product_1,
        oi2.product_id AS product_2,
        oi3.product_id AS product_3,
        COUNT(DISTINCT oi1.order_id) AS times_bought_together,
        AVG(oi1.unit_price + oi2.unit_price + oi3.unit_price) AS avg_bundle_price,
        SUM(oi1.subtotal + oi2.subtotal + oi3.subtotal) AS total_bundle_revenue
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id 
        AND oi1.product_id < oi2.product_id
    JOIN order_items oi3 ON oi2.order_id = oi3.order_id 
        AND oi2.product_id < oi3.product_id
    JOIN orders o ON oi1.order_id = o.order_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY oi1.product_id, oi2.product_id, oi3.product_id
    HAVING times_bought_together >= 3
)
SELECT 
    p1.product_name AS product_1_name,
    p1.sku AS product_1_sku,
    ROUND(p1.price, 2) AS product_1_price,
    p2.product_name AS product_2_name,
    p2.sku AS product_2_sku,
    ROUND(p2.price, 2) AS product_2_price,
    p3.product_name AS product_3_name,
    p3.sku AS product_3_sku,
    ROUND(p3.price, 2) AS product_3_price,
    tpc.times_bought_together AS bundle_frequency,
    ROUND(p1.price + p2.price + p3.price, 2) AS full_price,
    -- Suggested bundle price (10% discount)
    ROUND((p1.price + p2.price + p3.price) * 0.90, 2) AS suggested_bundle_price,
    ROUND((p1.price + p2.price + p3.price) * 0.10, 2) AS discount_amount,
    ROUND(tpc.total_bundle_revenue, 2) AS historical_revenue,
    -- Estimate potential if offered as bundle
    ROUND(tpc.times_bought_together * (p1.price + p2.price + p3.price) * 0.90, 2) AS potential_bundle_revenue
FROM three_product_combos tpc
JOIN products p1 ON tpc.product_1 = p1.product_id
JOIN products p2 ON tpc.product_2 = p2.product_id
JOIN products p3 ON tpc.product_3 = p3.product_id
WHERE p1.status = 'active' AND p1.stock_quantity > 0
    AND p2.status = 'active' AND p2.stock_quantity > 0
    AND p3.status = 'active' AND p3.stock_quantity > 0
    AND (p1.price + p2.price + p3.price) >= 50  -- Minimum bundle value
ORDER BY bundle_frequency DESC, potential_bundle_revenue DESC
LIMIT 20;

-- ========================================
-- 7. COMPLEMENTARY PRODUCT SUGGESTIONS
-- Products that solve related customer needs
-- ========================================
WITH customer_product_journey AS (
    SELECT 
        o.customer_id,
        oi.product_id,
        p.category_id,
        o.order_date,
        ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS purchase_sequence
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
),
sequential_purchases AS (
    SELECT 
        cpj1.product_id AS first_product,
        cpj2.product_id AS next_product,
        COUNT(DISTINCT cpj1.customer_id) AS customer_count,
        AVG(DATEDIFF(cpj2.order_date, cpj1.order_date)) AS avg_days_between
    FROM customer_product_journey cpj1
    JOIN customer_product_journey cpj2 ON cpj1.customer_id = cpj2.customer_id
        AND cpj1.purchase_sequence < cpj2.purchase_sequence
        AND cpj1.product_id != cpj2.product_id
    WHERE DATEDIFF(cpj2.order_date, cpj1.order_date) <= 90  -- Within 90 days
    GROUP BY cpj1.product_id, cpj2.product_id
    HAVING customer_count >= 3
)
SELECT 
    p1.product_name AS initial_purchase,
    p1.sku AS initial_sku,
    pc1.category_name AS initial_category,
    p2.product_name AS complementary_product,
    p2.sku AS complementary_sku,
    pc2.category_name AS complementary_category,
    ROUND(p2.price, 2) AS complementary_price,
    sp.customer_count AS times_purchased_sequentially,
    ROUND(sp.avg_days_between, 0) AS avg_days_until_purchase,
    -- Calculate recommendation timing
    CASE 
        WHEN sp.avg_days_between <= 7 THEN 'Immediate follow-up'
        WHEN sp.avg_days_between <= 30 THEN 'Send after 1 week'
        WHEN sp.avg_days_between <= 60 THEN 'Send after 1 month'
        ELSE 'Send after 2 months'
    END AS recommended_marketing_timing,
    -- Product ratings
    ROUND(AVG(r.rating), 1) AS complementary_avg_rating,
    COUNT(DISTINCT r.review_id) AS review_count
FROM sequential_purchases sp
JOIN products p1 ON sp.first_product = p1.product_id
JOIN products p2 ON sp.next_product = p2.product_id
JOIN product_categories pc1 ON p1.category_id = pc1.category_id
JOIN product_categories pc2 ON p2.category_id = pc2.category_id
LEFT JOIN reviews r ON p2.product_id = r.product_id AND r.status = 'approved'
WHERE p1.status = 'active'
    AND p2.status = 'active'
    AND p2.stock_quantity > 0
GROUP BY sp.first_product, sp.next_product, p1.product_name, p1.sku, 
         pc1.category_name, p2.product_name, p2.sku, pc2.category_name, 
         p2.price, sp.customer_count, sp.avg_days_between
ORDER BY sp.customer_count DESC, sp.avg_days_between ASC
LIMIT 50;