-- ========================================
-- PRICE ELASTICITY & OPTIMIZATION ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Price Sensitivity, Optimal Pricing & Demand Curves
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. PRICE SENSITIVITY ANALYSIS
-- Analyzes how quantity demanded responds to price changes
-- ========================================

WITH product_price_history AS (
    SELECT 
        p.product_id,
        p.product_name,
        pc.category_name,
        oi.unit_price AS sale_price,
        DATE_FORMAT(o.order_date, '%Y-%m') AS sale_month,
        SUM(oi.quantity) AS units_sold,
        COUNT(DISTINCT o.order_id) AS order_count,
        COUNT(DISTINCT o.customer_id) AS unique_customers,
        SUM(oi.subtotal) AS total_revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY p.product_id, p.product_name, pc.category_name, oi.unit_price, sale_month
),
price_points AS (
    SELECT 
        product_id,
        product_name,
        category_name,
        sale_price,
        SUM(units_sold) AS total_units_sold,
        SUM(order_count) AS total_orders,
        SUM(unique_customers) AS total_customers,
        SUM(total_revenue) AS total_revenue,
        AVG(units_sold) AS avg_monthly_units,
        COUNT(DISTINCT sale_month) AS months_at_price
    FROM product_price_history
    GROUP BY product_id, product_name, category_name, sale_price
    HAVING months_at_price >= 2  -- At least 2 months of data at this price
),
price_comparisons AS (
    SELECT 
        pp1.product_id,
        pp1.product_name,
        pp1.category_name,
        pp1.sale_price AS price_point_1,
        pp1.avg_monthly_units AS units_at_price_1,
        pp1.total_revenue AS revenue_at_price_1,
        pp2.sale_price AS price_point_2,
        pp2.avg_monthly_units AS units_at_price_2,
        pp2.total_revenue AS revenue_at_price_2,
        -- Calculate percentage changes
        ROUND(((pp2.sale_price - pp1.sale_price) / pp1.sale_price) * 100, 2) AS price_change_pct,
        ROUND(((pp2.avg_monthly_units - pp1.avg_monthly_units) / pp1.avg_monthly_units) * 100, 2) AS quantity_change_pct
    FROM price_points pp1
    JOIN price_points pp2 ON pp1.product_id = pp2.product_id
        AND pp1.sale_price < pp2.sale_price
    WHERE ABS(pp2.sale_price - pp1.sale_price) >= pp1.sale_price * 0.05  -- At least 5% price difference
)
SELECT 
    product_id,
    product_name,
    category_name,
    ROUND(price_point_1, 2) AS lower_price,
    ROUND(units_at_price_1, 1) AS units_at_lower_price,
    ROUND(revenue_at_price_1, 2) AS revenue_at_lower_price,
    ROUND(price_point_2, 2) AS higher_price,
    ROUND(units_at_price_2, 1) AS units_at_higher_price,
    ROUND(revenue_at_price_2, 2) AS revenue_at_higher_price,
    price_change_pct,
    quantity_change_pct,
    -- Price Elasticity of Demand (PED)
    -- PED = % change in quantity / % change in price
    ROUND(quantity_change_pct / NULLIF(price_change_pct, 0), 2) AS price_elasticity,
    CASE 
        WHEN ABS(quantity_change_pct / NULLIF(price_change_pct, 0)) > 1 THEN 'Elastic (price sensitive)'
        WHEN ABS(quantity_change_pct / NULLIF(price_change_pct, 0)) = 1 THEN 'Unit Elastic'
        WHEN ABS(quantity_change_pct / NULLIF(price_change_pct, 0)) < 1 AND 
             ABS(quantity_change_pct / NULLIF(price_change_pct, 0)) > 0 THEN 'Inelastic (price insensitive)'
        ELSE 'Perfectly Inelastic'
    END AS elasticity_classification,
    -- Revenue impact
    ROUND(revenue_at_price_2 - revenue_at_price_1, 2) AS revenue_change,
    CASE 
        WHEN revenue_at_price_2 > revenue_at_price_1 THEN 'Price increase beneficial'
        WHEN revenue_at_price_2 < revenue_at_price_1 THEN 'Price increase harmful'
        ELSE 'No revenue change'
    END AS pricing_recommendation
FROM price_comparisons
WHERE quantity_change_pct IS NOT NULL 
    AND price_change_pct != 0
ORDER BY ABS(quantity_change_pct / NULLIF(price_change_pct, 0)) DESC;

-- ========================================
-- 2. OPTIMAL PRICING ANALYSIS
-- Identifies revenue-maximizing price points
-- ========================================

WITH product_pricing_data AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.cost AS unit_cost,
        oi.unit_price,
        DATE_FORMAT(o.order_date, '%Y-%m') AS period,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS gross_revenue,
        SUM(oi.quantity * p.cost) AS total_cost,
        SUM(oi.subtotal - (oi.quantity * p.cost)) AS gross_profit,
        COUNT(DISTINCT o.order_id) AS order_count,
        AVG(oi.discount) AS avg_discount
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND p.cost > 0  -- Ensure cost data is available
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name, p.cost, oi.unit_price, period
),
aggregated_pricing AS (
    SELECT 
        product_id,
        product_name,
        sku,
        category_name,
        unit_cost,
        unit_price,
        SUM(units_sold) AS total_units_sold,
        SUM(gross_revenue) AS total_revenue,
        SUM(gross_profit) AS total_profit,
        AVG(units_sold) AS avg_monthly_units,
        COUNT(DISTINCT period) AS months_active,
        AVG(avg_discount) AS avg_discount_rate
    FROM product_pricing_data
    GROUP BY product_id, product_name, sku, category_name, unit_cost, unit_price
    HAVING months_active >= 2
),
price_performance AS (
    SELECT 
        product_id,
        product_name,
        sku,
        category_name,
        ROUND(unit_cost, 2) AS unit_cost,
        ROUND(unit_price, 2) AS price_point,
        total_units_sold,
        ROUND(total_revenue, 2) AS revenue_at_price,
        ROUND(total_profit, 2) AS profit_at_price,
        ROUND(avg_monthly_units, 1) AS avg_monthly_demand,
        ROUND((unit_price - unit_cost) / unit_price * 100, 2) AS profit_margin_pct,
        ROUND(total_profit / total_revenue * 100, 2) AS profit_margin_realized_pct,
        months_active,
        -- Revenue per month
        ROUND(total_revenue / months_active, 2) AS avg_monthly_revenue,
        ROUND(total_profit / months_active, 2) AS avg_monthly_profit
    FROM aggregated_pricing
),
optimal_price_candidates AS (
    SELECT 
        product_id,
        product_name,
        category_name,
        unit_cost,
        price_point,
        revenue_at_price,
        profit_at_price,
        avg_monthly_revenue,
        avg_monthly_profit,
        profit_margin_pct,
        total_units_sold,
        -- Rank by different metrics
        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY revenue_at_price DESC) AS revenue_rank,
        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY profit_at_price DESC) AS profit_rank,
        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY total_units_sold DESC) AS volume_rank
    FROM price_performance
)
SELECT 
    product_id,
    product_name,
    category_name,
    ROUND(unit_cost, 2) AS unit_cost,
    ROUND(MIN(price_point), 2) AS lowest_price_tested,
    ROUND(MAX(price_point), 2) AS highest_price_tested,
    -- Optimal pricing recommendations
    ROUND(MAX(CASE WHEN revenue_rank = 1 THEN price_point END), 2) AS optimal_price_for_revenue,
    ROUND(MAX(CASE WHEN revenue_rank = 1 THEN revenue_at_price END), 2) AS max_revenue_achievable,
    ROUND(MAX(CASE WHEN profit_rank = 1 THEN price_point END), 2) AS optimal_price_for_profit,
    ROUND(MAX(CASE WHEN profit_rank = 1 THEN profit_at_price END), 2) AS max_profit_achievable,
    ROUND(MAX(CASE WHEN volume_rank = 1 THEN price_point END), 2) AS price_for_max_volume,
    MAX(CASE WHEN volume_rank = 1 THEN total_units_sold END) AS max_volume_achievable,
    -- Price range analysis
    ROUND((MAX(price_point) - MIN(price_point)) / MIN(price_point) * 100, 2) AS price_range_variance_pct,
    COUNT(DISTINCT price_point) AS number_of_price_points_tested,
    -- Strategic recommendation
    CASE 
        WHEN MAX(CASE WHEN profit_rank = 1 THEN price_point END) = 
             MAX(CASE WHEN revenue_rank = 1 THEN price_point END) THEN 'Use profit-optimal price'
        WHEN MAX(CASE WHEN profit_rank = 1 THEN profit_margin_pct END) > 40 THEN 'Use profit-optimal price (high margin)'
        WHEN MAX(CASE WHEN revenue_rank = 1 THEN total_units_sold END) > 
             MAX(CASE WHEN profit_rank = 1 THEN total_units_sold END) * 1.5 THEN 'Consider revenue-optimal for market share'
        ELSE 'Balance between profit and revenue optimal'
    END AS pricing_strategy_recommendation
FROM optimal_price_candidates
GROUP BY product_id, product_name, category_name, unit_cost
HAVING number_of_price_points_tested >= 2
ORDER BY max_revenue_achievable DESC;

-- ========================================
-- 3. DEMAND CURVE ESTIMATION
-- Models relationship between price and quantity
-- ========================================

WITH price_quantity_data AS (
    SELECT 
        p.product_id,
        p.product_name,
        pc.category_name,
        oi.unit_price AS price,
        DATE_FORMAT(o.order_date, '%Y-%m') AS month,
        SUM(oi.quantity) AS quantity_sold,
        COUNT(DISTINCT o.order_id) AS order_frequency,
        SUM(oi.subtotal) AS revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 18 MONTH)
    GROUP BY p.product_id, p.product_name, pc.category_name, oi.unit_price, month
),
aggregated_demand AS (
    SELECT 
        product_id,
        product_name,
        category_name,
        price,
        AVG(quantity_sold) AS avg_monthly_quantity,
        SUM(quantity_sold) AS total_quantity,
        AVG(revenue) AS avg_monthly_revenue,
        SUM(revenue) AS total_revenue,
        COUNT(DISTINCT month) AS months_observed
    FROM price_quantity_data
    GROUP BY product_id, product_name, category_name, price
    HAVING months_observed >= 2
),
demand_curve_stats AS (
    SELECT 
        product_id,
        product_name,
        category_name,
        COUNT(DISTINCT price) AS price_points,
        MIN(price) AS min_price,
        MAX(price) AS max_price,
        AVG(price) AS avg_price,
        SUM(total_quantity) AS total_units_sold,
        SUM(total_revenue) AS total_revenue_all_prices,
        -- Linear demand curve coefficients (simplified)
        ROUND(
            (SUM(avg_monthly_quantity) * COUNT(*) - SUM(price) * SUM(avg_monthly_quantity)) /
            (SUM(price * price) * COUNT(*) - SUM(price) * SUM(price)),
            4
        ) AS demand_slope,
        ROUND(
            (SUM(avg_monthly_quantity) - 
             ((SUM(avg_monthly_quantity) * COUNT(*) - SUM(price) * SUM(avg_monthly_quantity)) /
              (SUM(price * price) * COUNT(*) - SUM(price) * SUM(price))) * SUM(price)) / COUNT(*),
            2
        ) AS demand_intercept
    FROM aggregated_demand
    GROUP BY product_id, product_name, category_name
    HAVING price_points >= 3
)
SELECT 
    ad.product_id,
    ad.product_name,
    ad.category_name,
    ROUND(ad.price, 2) AS price_point,
    ROUND(ad.avg_monthly_quantity, 1) AS observed_monthly_demand,
    ROUND(ad.total_quantity, 0) AS total_units_sold,
    ROUND(ad.total_revenue, 2) AS total_revenue,
    ad.months_observed,
    -- Demand curve stats
    ROUND(dcs.min_price, 2) AS lowest_price_observed,
    ROUND(dcs.max_price, 2) AS highest_price_observed,
    ROUND(dcs.avg_price, 2) AS average_price,
    dcs.price_points AS data_points,
    dcs.demand_slope,
    dcs.demand_intercept,
    -- Predicted demand using linear model: Q = intercept + slope * P
    ROUND(dcs.demand_intercept + (dcs.demand_slope * ad.price), 1) AS predicted_demand,
    ROUND(ad.avg_monthly_quantity - (dcs.demand_intercept + (dcs.demand_slope * ad.price)), 1) AS demand_prediction_error,
    -- Price positioning
    CASE 
        WHEN ad.price <= dcs.min_price * 1.1 THEN 'Budget Pricing'
        WHEN ad.price >= dcs.max_price * 0.9 THEN 'Premium Pricing'
        WHEN ad.price BETWEEN dcs.avg_price * 0.9 AND dcs.avg_price * 1.1 THEN 'Market Average'
        WHEN ad.price < dcs.avg_price THEN 'Below Market'
        ELSE 'Above Market'
    END AS price_positioning
FROM aggregated_demand ad
JOIN demand_curve_stats dcs ON ad.product_id = dcs.product_id
ORDER BY ad.product_id, ad.price;

-- ========================================
-- 4. COMPETITIVE PRICE POSITIONING
-- Analyzes pricing relative to category benchmarks
-- ========================================

WITH category_price_stats AS (
    SELECT 
        pc.category_name,
        AVG(p.price) AS avg_category_price,
        MIN(p.price) AS min_category_price,
        MAX(p.price) AS max_category_price,
        STDDEV(p.price) AS price_std_dev,
        COUNT(DISTINCT p.product_id) AS products_in_category
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE p.status = 'active'
        AND p.price > 0
    GROUP BY pc.category_name
),
product_performance AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price AS current_price,
        p.cost AS unit_cost,
        SUM(oi.quantity) AS total_units_sold,
        SUM(oi.subtotal) AS total_revenue,
        COUNT(DISTINCT o.order_id) AS total_orders,
        AVG(oi.quantity) AS avg_quantity_per_order
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name, p.price, p.cost
)
SELECT 
    pp.product_id,
    pp.product_name,
    pp.sku,
    pp.category_name,
    ROUND(pp.current_price, 2) AS current_price,
    ROUND(pp.unit_cost, 2) AS unit_cost,
    ROUND((pp.current_price - pp.unit_cost) / pp.current_price * 100, 2) AS profit_margin_pct,
    pp.total_units_sold,
    ROUND(pp.total_revenue, 2) AS total_revenue_6m,
    pp.total_orders,
    -- Category comparison
    ROUND(cps.avg_category_price, 2) AS category_avg_price,
    ROUND(cps.min_category_price, 2) AS category_min_price,
    ROUND(cps.max_category_price, 2) AS category_max_price,
    cps.products_in_category,
    -- Price positioning metrics
    ROUND((pp.current_price - cps.avg_category_price) / cps.avg_category_price * 100, 2) AS price_vs_category_avg_pct,
    ROUND((pp.current_price - cps.min_category_price) / (cps.max_category_price - cps.min_category_price) * 100, 2) AS position_in_range_pct,
    -- Price tier
    CASE 
        WHEN pp.current_price >= cps.avg_category_price + cps.price_std_dev THEN 'Premium Tier'
        WHEN pp.current_price >= cps.avg_category_price THEN 'Above Average'
        WHEN pp.current_price >= cps.avg_category_price - cps.price_std_dev THEN 'Below Average'
        ELSE 'Budget Tier'
    END AS price_tier,
    -- Performance relative to pricing
    CASE 
        WHEN pp.total_units_sold > 
             (SELECT AVG(total_units_sold) FROM product_performance WHERE category_name = pp.category_name) 
             AND pp.current_price >= cps.avg_category_price THEN 'Premium performer'
        WHEN pp.total_units_sold > 
             (SELECT AVG(total_units_sold) FROM product_performance WHERE category_name = pp.category_name) 
             THEN 'Volume leader'
        WHEN pp.current_price >= cps.avg_category_price THEN 'Premium but low volume'
        ELSE 'Budget with low volume'
    END AS market_position,
    -- Pricing recommendations
    CASE 
        WHEN pp.current_price < cps.avg_category_price * 0.8 
             AND pp.total_units_sold > 50 THEN 'Consider price increase - high demand'
        WHEN pp.current_price > cps.avg_category_price * 1.2 
             AND pp.total_units_sold < 10 THEN 'Consider price decrease - low demand'
        WHEN (pp.current_price - pp.unit_cost) / pp.current_price < 0.20 THEN 'Low margin - review pricing'
        ELSE 'Maintain current pricing'
    END AS pricing_recommendation
FROM product_performance pp
JOIN category_price_stats cps ON pp.category_name = cps.category_name
WHERE pp.current_price > 0
ORDER BY pp.category_name, pp.current_price DESC;

-- ========================================
-- 5. DISCOUNT EFFECTIVENESS ANALYSIS
-- Measures impact of discounts on demand
-- ========================================

WITH discount_analysis AS (
    SELECT 
        p.product_id,
        p.product_name,
        pc.category_name,
        p.price AS list_price,
        oi.unit_price AS sale_price,
        ROUND(oi.discount / oi.unit_price * 100, 2) AS discount_pct,
        CASE 
            WHEN oi.discount = 0 THEN 'No Discount'
            WHEN oi.discount / (oi.unit_price + oi.discount) < 0.10 THEN '1-9% Off'
            WHEN oi.discount / (oi.unit_price + oi.discount) < 0.20 THEN '10-19% Off'
            WHEN oi.discount / (oi.unit_price + oi.discount) < 0.30 THEN '20-29% Off'
            WHEN oi.discount / (oi.unit_price + oi.discount) < 0.40 THEN '30-39% Off'
            ELSE '40%+ Off'
        END AS discount_tier,
        SUM(oi.quantity) AS units_sold,
        COUNT(DISTINCT o.order_id) AS order_count,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity * p.cost) AS total_cost,
        SUM(oi.subtotal - (oi.quantity * p.cost)) AS gross_profit
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND p.cost > 0
    GROUP BY p.product_id, p.product_name, pc.category_name, p.price, 
             oi.unit_price, oi.discount, discount_tier
),
baseline_performance AS (
    SELECT 
        product_id,
        AVG(CASE WHEN discount_tier = 'No Discount' THEN units_sold END) AS baseline_units,
        AVG(CASE WHEN discount_tier = 'No Discount' THEN revenue END) AS baseline_revenue,
        AVG(CASE WHEN discount_tier = 'No Discount' THEN gross_profit END) AS baseline_profit
    FROM discount_analysis
    GROUP BY product_id
)
SELECT 
    da.product_id,
    da.product_name,
    da.category_name,
    ROUND(da.list_price, 2) AS list_price,
    da.discount_tier,
    AVG(da.discount_pct) AS avg_discount_pct,
    SUM(da.units_sold) AS total_units,
    SUM(da.order_count) AS total_orders,
    ROUND(SUM(da.revenue), 2) AS total_revenue,
    ROUND(SUM(da.gross_profit), 2) AS total_profit,
    ROUND(SUM(da.gross_profit) / SUM(da.revenue) * 100, 2) AS profit_margin_pct,
    -- Comparison to baseline (no discount)
    ROUND(bp.baseline_units, 1) AS baseline_units_no_discount,
    ROUND((SUM(da.units_sold) - bp.baseline_units) / NULLIF(bp.baseline_units, 0) * 100, 2) AS unit_lift_vs_baseline_pct,
    ROUND((SUM(da.revenue) - bp.baseline_revenue) / NULLIF(bp.baseline_revenue, 0) * 100, 2) AS revenue_lift_vs_baseline_pct,
    ROUND((SUM(da.gross_profit) - bp.baseline_profit) / NULLIF(bp.baseline_profit, 0) * 100, 2) AS profit_lift_vs_baseline_pct,
    -- Effectiveness rating
    CASE 
        WHEN SUM(da.gross_profit) > bp.baseline_profit * 1.2 THEN 'Highly Effective'
        WHEN SUM(da.gross_profit) > bp.baseline_profit THEN 'Effective'
        WHEN SUM(da.gross_profit) > bp.baseline_profit * 0.8 THEN 'Marginally Effective'
        ELSE 'Ineffective - Reduces Profit'
    END AS discount_effectiveness
FROM discount_analysis da
LEFT JOIN baseline_performance bp ON da.product_id = bp.product_id
WHERE da.discount_tier != 'No Discount'
GROUP BY da.product_id, da.product_name, da.category_name, da.list_price, 
         da.discount_tier, bp.baseline_units, bp.baseline_revenue, bp.baseline_profit
ORDER BY da.product_id, da.discount_tier;

-- ========================================
-- 6. PRICE OPTIMIZATION SIMULATOR
-- Estimates revenue at different price points
-- ========================================

WITH current_performance AS (
    SELECT 
        p.product_id,
        p.product_name,
        pc.category_name,
        p.price AS current_price,
        p.cost AS unit_cost,
        AVG(oi.quantity) AS avg_quantity_per_order,
        SUM(oi.quantity) / 
            NULLIF(COUNT(DISTINCT DATE_FORMAT(o.order_date, '%Y-%m')), 0) AS avg_monthly_volume,
        COUNT(DISTINCT o.order_id) AS total_orders_6m
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    WHERE p.status = 'active'
        AND p.price > 0
        AND p.cost > 0
    GROUP BY p.product_id, p.product_name, pc.category_name, p.price, p.cost
    HAVING avg_monthly_volume > 0
),
price_scenarios AS (
    SELECT 
        product_id,
        product_name,
        category_name,
        current_price,
        unit_cost,
        avg_monthly_volume,
        total_orders_6m,
        -- Scenario: -20% price
        ROUND(current_price * 0.80, 2) AS price_minus_20,
        ROUND(avg_monthly_volume * 1.30, 1) AS volume_minus_20,  -- Assume 30% volume increase
        -- Scenario: -10% price
        ROUND(current_price * 0.90, 2) AS price_minus_10,
        ROUND(avg_monthly_volume * 1.15, 1) AS volume_minus_10,  -- Assume 15% volume increase
        -- Scenario: Current price
        current_price AS price_current,
        avg_monthly_volume AS volume_current,
        -- Scenario: +10% price
        ROUND(current_price * 1.10, 2) AS price_plus_10,
        ROUND(avg_monthly_volume * 0.90, 1) AS volume_plus_10,  -- Assume 10% volume decrease
        -- Scenario: +20% price
        ROUND(current_price * 1.20, 2) AS price_plus_20,
        ROUND(avg_monthly_volume * 0.75, 1) AS volume_plus_20   -- Assume 25% volume decrease
    FROM current_performance
)
SELECT 
    product_id,
    product_name,
    category_name,
    ROUND(unit_cost, 2) AS unit_cost,
    total_orders_6m,
    -- Current state
    ROUND(price_current, 2) AS current_price,
    ROUND(volume_current, 1) AS current_monthly_volume,
    ROUND(price_current * volume_current, 2) AS current_monthly_revenue,
    ROUND((price_current - unit_cost) * volume_current, 2) AS current_monthly_profit,
    -- -20% scenario
    price_minus_20,
    volume_minus_20,
    ROUND(price_minus_20 * volume_minus_20, 2) AS revenue_minus_20,
    ROUND((price_minus_20 - unit_cost) * volume_minus_20, 2) AS profit_minus_20,
    -- -10% scenario
    price_minus_10,
    volume_minus_10,
    ROUND(price_minus_10 * volume_minus_10, 2) AS revenue_minus_10,
    ROUND((price_minus_10 - unit_cost) * volume_minus_10, 2) AS profit_minus_10,
    -- +10% scenario
    price_plus_10,
    volume_plus_10,
    ROUND(price_plus_10 * volume_plus_10, 2) AS revenue_plus_10,
    ROUND((price_plus_10 - unit_cost) * volume_plus_10, 2) AS profit_plus_10,
    -- +20% scenario
    price_plus_20,
    volume_plus_20,
    ROUND(price_plus_20 * volume_plus_20, 2) AS revenue_plus_20,
    ROUND((price_plus_20 - unit_cost) * volume_plus_20, 2) AS profit_plus_20,
    -- Best scenario identification
    CASE 
        WHEN (price_plus_20 - unit_cost) * volume_plus_20 = GREATEST(
            (price_minus_20 - unit_cost) * volume_minus_20,
            (price_minus_10 - unit_cost) * volume_minus_10,
            (price_current - unit_cost) * volume_current,
            (price_plus_10 - unit_cost) * volume_plus_10,
            (price_plus_20 - unit_cost) * volume_plus_20
        ) THEN '+20% Price'
        WHEN (price_plus_10 - unit_cost) * volume_plus_10 = GREATEST(
            (price_minus_20 - unit_cost) * volume_minus_20,
            (price_minus_10 - unit_cost) * volume_minus_10,
            (price_current - unit_cost) * volume_current,
            (price_plus_10 - unit_cost) * volume_plus_10,
            (price_plus_20 - unit_cost) * volume_plus_20
        ) THEN '+10% Price'
        WHEN (price_current - unit_cost) * volume_current = GREATEST(
            (price_minus_20 - unit_cost) * volume_minus_20,
            (price_minus_10 - unit_cost) * volume_minus_10,
            (price_current - unit_cost) * volume_current,
            (price_plus_10 - unit_cost) * volume_plus_10,
            (price_plus_20 - unit_cost) * volume_plus_20
        ) THEN 'Current Price'
        WHEN (price_minus_10 - unit_cost) * volume_minus_10 = GREATEST(
            (price_minus_20 - unit_cost) * volume_minus_20,
            (price_minus_10 - unit_cost) * volume_minus_10,
            (price_current - unit_cost) * volume_current,
            (price_plus_10 - unit_cost) * volume_plus_10,
            (price_plus_20 - unit_cost) * volume_plus_20
        ) THEN '-10% Price'
        ELSE '-20% Price'
    END AS optimal_price_scenario,
    ROUND(GREATEST(
        (price_minus_20 - unit_cost) * volume_minus_20,
        (price_minus_10 - unit_cost) * volume_minus_10,
        (price_current - unit_cost) * volume_current,
        (price_plus_10 - unit_cost) * volume_plus_10,
        (price_plus_20 - unit_cost) * volume_plus_20
    ), 2) AS max_monthly_profit_potential,
    ROUND(
        (GREATEST(
            (price_minus_20 - unit_cost) * volume_minus_20,
            (price_minus_10 - unit_cost) * volume_minus_10,
            (price_current - unit_cost) * volume_current,
            (price_plus_10 - unit_cost) * volume_plus_10,
            (price_plus_20 - unit_cost) * volume_plus_20
        ) - ((price_current - unit_cost) * volume_current)) / 
        NULLIF((price_current - unit_cost) * volume_current, 0) * 100,
        2
    ) AS profit_improvement_pct
FROM price_scenarios
ORDER BY profit_improvement_pct DESC;

-- ========================================
-- 7. DYNAMIC PRICING RECOMMENDATIONS
-- Real-time pricing suggestions based on multiple factors
-- ========================================

WITH product_metrics AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price AS current_price,
        p.cost AS unit_cost,
        p.stock_quantity,
        -- Recent sales velocity (last 30 days)
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN o.order_id 
        END) AS orders_last_30d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN oi.quantity 
            ELSE 0 
        END) AS units_sold_last_30d,
        -- Historical average (6 months)
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 180 DAY) THEN o.order_id 
        END) / 6.0 AS avg_monthly_orders,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 180 DAY) THEN oi.quantity 
            ELSE 0 
        END) / 6.0 AS avg_monthly_units,
        -- Competitive metrics
        AVG(oi.unit_price) AS avg_selling_price,
        MIN(oi.unit_price) AS lowest_price_sold,
        MAX(oi.unit_price) AS highest_price_sold,
        -- Customer engagement
        COUNT(DISTINCT o.customer_id) AS unique_customers_6m,
        AVG(oi.discount) AS avg_discount_given
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 180 DAY)
    WHERE p.status = 'active'
        AND p.price > 0
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name, 
             p.price, p.cost, p.stock_quantity
),
category_benchmarks AS (
    SELECT 
        category_name,
        AVG(current_price) AS avg_category_price,
        AVG(avg_monthly_units) AS avg_category_volume
    FROM product_metrics
    GROUP BY category_name
),
pricing_signals AS (
    SELECT 
        pm.product_id,
        pm.product_name,
        pm.sku,
        pm.category_name,
        ROUND(pm.current_price, 2) AS current_price,
        ROUND(pm.unit_cost, 2) AS unit_cost,
        pm.stock_quantity,
        ROUND(pm.units_sold_last_30d, 1) AS units_sold_last_30d,
        ROUND(pm.avg_monthly_units, 1) AS avg_monthly_units,
        pm.unique_customers_6m,
        -- Pricing signals
        CASE 
            WHEN pm.stock_quantity < pm.avg_monthly_units * 0.5 THEN 'Low Stock'
            WHEN pm.stock_quantity > pm.avg_monthly_units * 3 THEN 'Excess Stock'
            ELSE 'Normal Stock'
        END AS inventory_status,
        CASE 
            WHEN pm.units_sold_last_30d > pm.avg_monthly_units * 1.3 THEN 'Accelerating'
            WHEN pm.units_sold_last_30d > pm.avg_monthly_units * 0.7 THEN 'Stable'
            ELSE 'Declining'
        END AS demand_trend,
        CASE 
            WHEN pm.current_price > cb.avg_category_price * 1.2 THEN 'Premium'
            WHEN pm.current_price < cb.avg_category_price * 0.8 THEN 'Budget'
            ELSE 'Market Rate'
        END AS price_position,
        ROUND((pm.current_price - pm.unit_cost) / pm.current_price * 100, 2) AS current_margin_pct,
        -- Recommendation factors
        pm.avg_discount_given,
        ROUND(cb.avg_category_price, 2) AS category_avg_price
    FROM product_metrics pm
    JOIN category_benchmarks cb ON pm.category_name = cb.category_name
)
SELECT 
    product_id,
    product_name,
    sku,
    category_name,
    current_price,
    unit_cost,
    current_margin_pct,
    stock_quantity,
    inventory_status,
    demand_trend,
    price_position,
    units_sold_last_30d,
    avg_monthly_units,
    category_avg_price,
    -- Dynamic pricing recommendation
    CASE 
        -- High demand + low stock = increase price
        WHEN demand_trend = 'Accelerating' AND inventory_status = 'Low Stock' 
        THEN ROUND(current_price * 1.15, 2)
        -- High demand + normal stock = slight increase
        WHEN demand_trend = 'Accelerating' AND inventory_status = 'Normal Stock' 
        THEN ROUND(current_price * 1.08, 2)
        -- Declining demand + excess stock = decrease price
        WHEN demand_trend = 'Declining' AND inventory_status = 'Excess Stock' 
        THEN ROUND(current_price * 0.85, 2)
        -- Declining demand + normal stock = slight decrease
        WHEN demand_trend = 'Declining' AND inventory_status = 'Normal Stock' 
        THEN ROUND(current_price * 0.93, 2)
        -- Excess stock regardless of demand = clearance pricing
        WHEN inventory_status = 'Excess Stock' 
        THEN ROUND(current_price * 0.80, 2)
        -- Premium priced with declining demand = adjust to market
        WHEN price_position = 'Premium' AND demand_trend = 'Declining' 
        THEN ROUND(category_avg_price * 1.05, 2)
        -- Budget priced with high demand = increase to market
        WHEN price_position = 'Budget' AND demand_trend = 'Accelerating' 
        THEN ROUND(category_avg_price * 0.95, 2)
        -- Otherwise maintain current price
        ELSE current_price
    END AS recommended_price,
    -- Recommendation reasoning
    CASE 
        WHEN demand_trend = 'Accelerating' AND inventory_status = 'Low Stock' 
        THEN 'Increase +15%: High demand, limited supply'
        WHEN demand_trend = 'Accelerating' AND inventory_status = 'Normal Stock' 
        THEN 'Increase +8%: Strong demand momentum'
        WHEN demand_trend = 'Declining' AND inventory_status = 'Excess Stock' 
        THEN 'Decrease -15%: Clear excess inventory'
        WHEN demand_trend = 'Declining' AND inventory_status = 'Normal Stock' 
        THEN 'Decrease -7%: Stimulate demand'
        WHEN inventory_status = 'Excess Stock' 
        THEN 'Decrease -20%: Clearance pricing needed'
        WHEN price_position = 'Premium' AND demand_trend = 'Declining' 
        THEN 'Adjust to market: Premium not justified'
        WHEN price_position = 'Budget' AND demand_trend = 'Accelerating' 
        THEN 'Increase to market: Capture value'
        ELSE 'Maintain: Market equilibrium'
    END AS recommendation_rationale,
    -- Urgency level
    CASE 
        WHEN (demand_trend = 'Declining' AND inventory_status = 'Excess Stock') 
             OR (demand_trend = 'Accelerating' AND inventory_status = 'Low Stock') 
        THEN 'High'
        WHEN demand_trend IN ('Declining', 'Accelerating') 
             OR inventory_status IN ('Excess Stock', 'Low Stock') 
        THEN 'Medium'
        ELSE 'Low'
    END AS action_urgency,
    -- Expected impact
    ROUND(
        (CASE 
            WHEN demand_trend = 'Accelerating' AND inventory_status = 'Low Stock' 
            THEN current_price * 1.15
            WHEN demand_trend = 'Accelerating' AND inventory_status = 'Normal Stock' 
            THEN current_price * 1.08
            WHEN demand_trend = 'Declining' AND inventory_status = 'Excess Stock' 
            THEN current_price * 0.85
            WHEN demand_trend = 'Declining' AND inventory_status = 'Normal Stock' 
            THEN current_price * 0.93
            WHEN inventory_status = 'Excess Stock' 
            THEN current_price * 0.80
            WHEN price_position = 'Premium' AND demand_trend = 'Declining' 
            THEN category_avg_price * 1.05
            WHEN price_position = 'Budget' AND demand_trend = 'Accelerating' 
            THEN category_avg_price * 0.95
            ELSE current_price
        END - current_price) / current_price * 100,
        2
    ) AS price_change_pct
FROM pricing_signals
WHERE current_price != CASE 
    WHEN demand_trend = 'Accelerating' AND inventory_status = 'Low Stock' 
    THEN ROUND(current_price * 1.15, 2)
    WHEN demand_trend = 'Accelerating' AND inventory_status = 'Normal Stock' 
    THEN ROUND(current_price * 1.08, 2)
    WHEN demand_trend = 'Declining' AND inventory_status = 'Excess Stock' 
    THEN ROUND(current_price * 0.85, 2)
    WHEN demand_trend = 'Declining' AND inventory_status = 'Normal Stock' 
    THEN ROUND(current_price * 0.93, 2)
    WHEN inventory_status = 'Excess Stock' 
    THEN ROUND(current_price * 0.80, 2)
    WHEN price_position = 'Premium' AND demand_trend = 'Declining' 
    THEN ROUND(category_avg_price * 1.05, 2)
    WHEN price_position = 'Budget' AND demand_trend = 'Accelerating' 
    THEN ROUND(category_avg_price * 0.95, 2)
    ELSE current_price
END  -- Only show products where price change is recommended
ORDER BY 
    FIELD(action_urgency, 'High', 'Medium', 'Low'),
    ABS(price_change_pct) DESC;

-- ========================================
-- End of Price Elasticity & Optimization Analysis
-- ========================================