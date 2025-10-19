-- ========================================
-- INVENTORY TURNOVER & OPTIMIZATION ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Turnover Rates, Fast/Slow Movers, Stock Optimization
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. INVENTORY TURNOVER RATE ANALYSIS
-- Calculates how quickly inventory sells
-- ========================================

WITH inventory_metrics AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price AS current_price,
        p.cost AS unit_cost,
        p.stock_quantity AS current_stock,
        -- Sales data (last 12 months)
        SUM(oi.quantity) AS units_sold_12m,
        SUM(oi.subtotal) AS revenue_12m,
        SUM(oi.quantity * p.cost) AS cogs_12m,
        COUNT(DISTINCT o.order_id) AS order_count_12m,
        COUNT(DISTINCT o.customer_id) AS unique_customers_12m,
        -- Average inventory (simplified: current stock as proxy)
        p.stock_quantity AS avg_inventory_level,
        -- Recent velocity
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            THEN oi.quantity ELSE 0 
        END) AS units_sold_30d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN oi.quantity ELSE 0 
        END) AS units_sold_90d
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name, 
             p.price, p.cost, p.stock_quantity
)
SELECT 
    product_id,
    product_name,
    sku,
    category_name,
    ROUND(current_price, 2) AS current_price,
    ROUND(unit_cost, 2) AS unit_cost,
    current_stock,
    units_sold_12m,
    ROUND(revenue_12m, 2) AS revenue_12m,
    order_count_12m,
    unique_customers_12m,
    -- Inventory Turnover Ratio = COGS / Average Inventory Value
    ROUND(
        cogs_12m / NULLIF(avg_inventory_level * unit_cost, 0),
        2
    ) AS inventory_turnover_ratio,
    -- Days Inventory Outstanding (DIO) = 365 / Turnover Ratio
    ROUND(
        365 / NULLIF(cogs_12m / NULLIF(avg_inventory_level * unit_cost, 0), 0),
        1
    ) AS days_inventory_outstanding,
    -- Days of Supply = Current Stock / Daily Sales Rate
    ROUND(
        current_stock / NULLIF(units_sold_12m / 365.0, 0),
        1
    ) AS days_of_supply,
    -- Monthly velocity
    ROUND(units_sold_12m / 12.0, 1) AS avg_monthly_sales,
    ROUND(units_sold_90d / 3.0, 1) AS avg_monthly_sales_90d,
    units_sold_30d AS units_sold_last_month,
    -- Stock status
    CASE 
        WHEN current_stock = 0 THEN 'Out of Stock'
        WHEN current_stock <= (units_sold_12m / 12.0) * 0.5 THEN 'Critically Low'
        WHEN current_stock <= (units_sold_12m / 12.0) THEN 'Low Stock'
        WHEN current_stock <= (units_sold_12m / 12.0) * 2 THEN 'Normal'
        WHEN current_stock <= (units_sold_12m / 12.0) * 4 THEN 'High Stock'
        ELSE 'Excess Stock'
    END AS stock_status,
    -- Performance classification
    CASE 
        WHEN cogs_12m / NULLIF(avg_inventory_level * unit_cost, 0) >= 12 THEN 'Fast Mover'
        WHEN cogs_12m / NULLIF(avg_inventory_level * unit_cost, 0) >= 6 THEN 'Medium Mover'
        WHEN cogs_12m / NULLIF(avg_inventory_level * unit_cost, 0) >= 2 THEN 'Slow Mover'
        WHEN cogs_12m / NULLIF(avg_inventory_level * unit_cost, 0) > 0 THEN 'Very Slow Mover'
        ELSE 'Dead Stock'
    END AS turnover_classification,
    -- Inventory health score (0-100)
    LEAST(100, ROUND(
        (cogs_12m / NULLIF(avg_inventory_level * unit_cost, 0)) * 8.33,
        0
    )) AS inventory_health_score
FROM inventory_metrics
WHERE units_sold_12m > 0 OR current_stock > 0
ORDER BY inventory_turnover_ratio DESC;

-- ========================================
-- 2. FAST MOVERS vs SLOW MOVERS ANALYSIS
-- Identifies products by velocity
-- ========================================

WITH sales_velocity AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price,
        p.cost,
        p.stock_quantity,
        -- Sales metrics
        SUM(oi.quantity) AS total_units_sold,
        SUM(oi.subtotal) AS total_revenue,
        COUNT(DISTINCT o.order_id) AS order_frequency,
        COUNT(DISTINCT DATE(o.order_date)) AS days_with_sales,
        COUNT(DISTINCT o.customer_id) AS unique_customers,
        -- Time periods
        MIN(o.order_date) AS first_sale_date,
        MAX(o.order_date) AS last_sale_date,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) + 1 AS selling_period_days,
        -- Recent performance
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            THEN oi.quantity ELSE 0 
        END) AS units_last_30d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN oi.quantity ELSE 0 
        END) AS units_last_90d
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name,
             p.price, p.cost, p.stock_quantity
),
velocity_metrics AS (
    SELECT 
        product_id,
        product_name,
        sku,
        category_name,
        ROUND(price, 2) AS price,
        ROUND(cost, 2) AS cost,
        stock_quantity,
        total_units_sold,
        ROUND(total_revenue, 2) AS total_revenue,
        order_frequency,
        days_with_sales,
        unique_customers,
        first_sale_date,
        last_sale_date,
        units_last_30d,
        units_last_90d,
        -- Velocity calculations
        ROUND(total_units_sold / 12.0, 2) AS avg_units_per_month,
        ROUND(total_units_sold / NULLIF(selling_period_days, 0), 2) AS avg_units_per_day,
        ROUND(total_revenue / 12.0, 2) AS avg_revenue_per_month,
        ROUND(days_with_sales * 100.0 / NULLIF(selling_period_days, 0), 2) AS sales_consistency_pct,
        -- Stock coverage
        ROUND(stock_quantity / NULLIF(total_units_sold / 365.0, 0), 1) AS days_of_stock_remaining,
        -- Velocity score (composite metric)
        (
            (total_units_sold / 12.0) * 0.4 +  -- Monthly volume (40%)
            (order_frequency / 12.0) * 0.3 +    -- Order frequency (30%)
            (days_with_sales / 365.0 * 100) * 0.2 +  -- Consistency (20%)
            (unique_customers / 10.0) * 0.1     -- Customer reach (10%)
        ) AS velocity_score
    FROM sales_velocity
    WHERE total_units_sold > 0 OR stock_quantity > 0
),
velocity_classification AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY velocity_score DESC) AS velocity_quintile,
        CASE 
            WHEN velocity_score >= 50 THEN 'Fast Mover'
            WHEN velocity_score >= 30 THEN 'Medium-Fast Mover'
            WHEN velocity_score >= 15 THEN 'Medium Mover'
            WHEN velocity_score >= 5 THEN 'Slow Mover'
            ELSE 'Very Slow/Dead Stock'
        END AS velocity_category
    FROM velocity_metrics
)
SELECT 
    product_id,
    product_name,
    sku,
    category_name,
    price,
    cost,
    stock_quantity,
    total_units_sold,
    total_revenue,
    order_frequency,
    unique_customers,
    avg_units_per_month,
    avg_units_per_day,
    avg_revenue_per_month,
    units_last_30d,
    units_last_90d,
    ROUND(sales_consistency_pct, 1) AS sales_consistency_pct,
    days_of_stock_remaining,
    ROUND(velocity_score, 2) AS velocity_score,
    velocity_category,
    velocity_quintile,
    -- Recommendations
    CASE 
        WHEN velocity_category = 'Fast Mover' AND days_of_stock_remaining < 30 
            THEN 'URGENT: Reorder immediately'
        WHEN velocity_category = 'Fast Mover' AND days_of_stock_remaining < 60 
            THEN 'Schedule reorder soon'
        WHEN velocity_category IN ('Slow Mover', 'Very Slow/Dead Stock') AND stock_quantity > avg_units_per_month * 6 
            THEN 'Consider clearance/promotion'
        WHEN velocity_category IN ('Slow Mover', 'Very Slow/Dead Stock') AND units_last_90d = 0 
            THEN 'Consider discontinuing'
        WHEN velocity_category = 'Medium Mover' AND stock_quantity < avg_units_per_month * 2 
            THEN 'Monitor stock levels'
        ELSE 'Stock levels appropriate'
    END AS inventory_action
FROM velocity_classification
ORDER BY velocity_score DESC, total_revenue DESC;

-- ========================================
-- 3. ABC ANALYSIS (PARETO ANALYSIS)
-- Classifies inventory by revenue contribution
-- ========================================

WITH product_revenue AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price,
        p.cost,
        p.stock_quantity,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS total_revenue,
        SUM(oi.quantity * p.cost) AS total_cost,
        SUM(oi.subtotal - (oi.quantity * p.cost)) AS gross_profit,
        COUNT(DISTINCT o.order_id) AS order_count
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name,
             p.price, p.cost, p.stock_quantity
),
revenue_ranking AS (
    SELECT 
        *,
        SUM(total_revenue) OVER () AS total_company_revenue,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC) AS cumulative_revenue,
        ROUND(
            SUM(total_revenue) OVER (ORDER BY total_revenue DESC) * 100.0 / 
            SUM(total_revenue) OVER (),
            2
        ) AS cumulative_revenue_pct,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) AS revenue_rank,
        COUNT(*) OVER () AS total_products
    FROM product_revenue
    WHERE total_revenue > 0
),
abc_classification AS (
    SELECT 
        product_id,
        product_name,
        sku,
        category_name,
        ROUND(price, 2) AS price,
        ROUND(cost, 2) AS cost,
        stock_quantity,
        units_sold,
        ROUND(total_revenue, 2) AS total_revenue,
        ROUND(gross_profit, 2) AS gross_profit,
        ROUND(gross_profit / NULLIF(total_revenue, 0) * 100, 2) AS profit_margin_pct,
        order_count,
        revenue_rank,
        ROUND(total_revenue * 100.0 / total_company_revenue, 2) AS revenue_contribution_pct,
        cumulative_revenue_pct,
        -- ABC Classification (Pareto Principle)
        CASE 
            WHEN cumulative_revenue_pct <= 80 THEN 'A - High Value (Top 80%)'
            WHEN cumulative_revenue_pct <= 95 THEN 'B - Medium Value (80-95%)'
            ELSE 'C - Low Value (95-100%)'
        END AS abc_class,
        -- Stock value
        ROUND(stock_quantity * cost, 2) AS inventory_value,
        -- Recommendations by class
        CASE 
            WHEN cumulative_revenue_pct <= 80 THEN 'Critical - Monitor daily, never stock out'
            WHEN cumulative_revenue_pct <= 95 THEN 'Important - Monitor weekly, maintain buffer'
            ELSE 'Standard - Monitor monthly, optimize stock levels'
        END AS management_priority
    FROM revenue_ranking
)
SELECT 
    abc_class,
    COUNT(*) AS product_count,
    ROUND(SUM(total_revenue), 2) AS total_revenue,
    ROUND(SUM(gross_profit), 2) AS total_profit,
    ROUND(SUM(inventory_value), 2) AS total_inventory_value,
    ROUND(AVG(profit_margin_pct), 2) AS avg_profit_margin_pct,
    ROUND(SUM(total_revenue) * 100.0 / SUM(SUM(total_revenue)) OVER (), 2) AS revenue_pct,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS product_count_pct
FROM abc_classification
GROUP BY abc_class
ORDER BY FIELD(abc_class, 'A - High Value (Top 80%)', 'B - Medium Value (80-95%)', 'C - Low Value (95-100%)');

-- ========================================
-- 4. STOCK OPTIMIZATION RECOMMENDATIONS
-- Determines optimal stock levels
-- ========================================

WITH product_statistics AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price,
        p.cost,
        p.stock_quantity AS current_stock,
        i.quantity_reserved,
        i.quantity_available,
        i.reorder_level,
        -- Historical sales
        AVG(daily_sales.daily_quantity) AS avg_daily_sales,
        STDDEV(daily_sales.daily_quantity) AS stddev_daily_sales,
        MAX(daily_sales.daily_quantity) AS max_daily_sales,
        COUNT(DISTINCT daily_sales.sale_date) AS days_with_sales,
        -- Lead time (from vendor contracts)
        AVG(vc.lead_time_days) AS avg_lead_time_days
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN inventory i ON p.product_id = i.product_id
    LEFT JOIN vendor_contracts vc ON p.product_id = vc.product_id 
        AND vc.status = 'active'
    LEFT JOIN (
        SELECT 
            oi.product_id,
            DATE(o.order_date) AS sale_date,
            SUM(oi.quantity) AS daily_quantity
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.status IN ('delivered', 'shipped', 'processing')
            AND o.payment_status = 'paid'
            AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
        GROUP BY oi.product_id, DATE(o.order_date)
    ) daily_sales ON p.product_id = daily_sales.product_id
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name,
             p.price, p.cost, p.stock_quantity, i.quantity_reserved,
             i.quantity_available, i.reorder_level
),
optimal_stock_calc AS (
    SELECT 
        product_id,
        product_name,
        sku,
        category_name,
        ROUND(price, 2) AS price,
        ROUND(cost, 2) AS cost,
        current_stock,
        quantity_reserved,
        quantity_available,
        reorder_level AS current_reorder_level,
        ROUND(avg_daily_sales, 2) AS avg_daily_sales,
        ROUND(stddev_daily_sales, 2) AS stddev_daily_sales,
        max_daily_sales,
        COALESCE(avg_lead_time_days, 14) AS lead_time_days,
        -- Safety Stock = Z-score × StdDev × √Lead Time
        -- Using Z=1.65 for 95% service level
        ROUND(
            1.65 * COALESCE(stddev_daily_sales, avg_daily_sales * 0.3) * 
            SQRT(COALESCE(avg_lead_time_days, 14)),
            0
        ) AS recommended_safety_stock,
        -- Reorder Point = (Avg Daily Sales × Lead Time) + Safety Stock
        ROUND(
            (COALESCE(avg_daily_sales, 0) * COALESCE(avg_lead_time_days, 14)) +
            (1.65 * COALESCE(stddev_daily_sales, avg_daily_sales * 0.3) * 
             SQRT(COALESCE(avg_lead_time_days, 14))),
            0
        ) AS recommended_reorder_point,
        -- Economic Order Quantity (simplified)
        -- EOQ = SQRT((2 × Annual Demand × Order Cost) / Holding Cost)
        -- Assuming order cost = $50, holding cost = 25% of unit cost
        ROUND(
            SQRT((2 * (COALESCE(avg_daily_sales, 0) * 365) * 50) / 
                 (cost * 0.25)),
            0
        ) AS economic_order_quantity,
        -- Maximum stock level = Reorder Point + EOQ
        ROUND(
            (COALESCE(avg_daily_sales, 0) * COALESCE(avg_lead_time_days, 14)) +
            (1.65 * COALESCE(stddev_daily_sales, avg_daily_sales * 0.3) * 
             SQRT(COALESCE(avg_lead_time_days, 14))) +
            SQRT((2 * (COALESCE(avg_daily_sales, 0) * 365) * 50) / 
                 (cost * 0.25)),
            0
        ) AS recommended_max_stock
    FROM product_statistics
    WHERE avg_daily_sales > 0
)
SELECT 
    product_id,
    product_name,
    sku,
    category_name,
    price,
    cost,
    current_stock,
    quantity_available,
    quantity_reserved,
    avg_daily_sales,
    lead_time_days,
    current_reorder_level,
    recommended_safety_stock,
    recommended_reorder_point,
    economic_order_quantity,
    recommended_max_stock,
    -- Stock status
    CASE 
        WHEN quantity_available <= recommended_safety_stock THEN 'Critical - Below Safety Stock'
        WHEN quantity_available <= recommended_reorder_point THEN 'Low - Reorder Now'
        WHEN current_stock <= recommended_max_stock THEN 'Optimal'
        WHEN current_stock <= recommended_max_stock * 1.5 THEN 'High - Acceptable'
        ELSE 'Excess - Reduce Stock'
    END AS stock_status,
    -- Reorder recommendation
    CASE 
        WHEN quantity_available <= recommended_reorder_point 
            THEN economic_order_quantity
        ELSE 0
    END AS recommended_order_qty,
    -- Days until stockout (at current rate)
    ROUND(
        quantity_available / NULLIF(avg_daily_sales, 0),
        0
    ) AS days_until_stockout,
    -- Cost implications
    ROUND(recommended_safety_stock * cost, 2) AS safety_stock_investment,
    ROUND(economic_order_quantity * cost, 2) AS order_cost,
    ROUND(
        (current_stock - recommended_max_stock) * cost,
        2
    ) AS excess_inventory_value
FROM optimal_stock_calc
ORDER BY 
    CASE 
        WHEN quantity_available <= recommended_safety_stock THEN 1
        WHEN quantity_available <= recommended_reorder_point THEN 2
        ELSE 3
    END,
    avg_daily_sales DESC;

-- ========================================
-- 5. DEAD STOCK IDENTIFICATION
-- Products with no sales requiring action
-- ========================================

WITH product_activity AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price,
        p.cost,
        p.stock_quantity,
        p.created_at AS product_created_date,
        DATEDIFF(CURDATE(), p.created_at) AS product_age_days,
        -- Sales analysis
        MAX(o.order_date) AS last_sale_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_sale,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN oi.quantity ELSE 0 
        END) AS units_sold_90d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 180 DAY) 
            THEN oi.quantity ELSE 0 
        END) AS units_sold_180d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 365 DAY) 
            THEN oi.quantity ELSE 0 
        END) AS units_sold_365d,
        SUM(oi.quantity) AS total_units_sold,
        SUM(oi.subtotal) AS total_revenue,
        COUNT(DISTINCT o.order_id) AS total_orders
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    WHERE p.status = 'active'
        AND p.stock_quantity > 0
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name,
             p.price, p.cost, p.stock_quantity, p.created_at
),
dead_stock_analysis AS (
    SELECT 
        product_id,
        product_name,
        sku,
        category_name,
        ROUND(price, 2) AS price,
        ROUND(cost, 2) AS cost,
        stock_quantity,
        ROUND(stock_quantity * cost, 2) AS inventory_value,
        ROUND(product_age_days / 365.0, 1) AS product_age_years,
        last_sale_date,
        days_since_last_sale,
        units_sold_90d,
        units_sold_180d,
        units_sold_365d,
        COALESCE(total_units_sold, 0) AS lifetime_units_sold,
        ROUND(COALESCE(total_revenue, 0), 2) AS lifetime_revenue,
        -- Dead stock classification
        CASE 
            WHEN last_sale_date IS NULL AND product_age_days > 180 THEN 'Never Sold - Old Product'
            WHEN last_sale_date IS NULL THEN 'Never Sold - New Product'
            WHEN days_since_last_sale > 365 THEN 'Dead Stock (1+ year)'
            WHEN days_since_last_sale > 180 THEN 'Dormant (6-12 months)'
            WHEN units_sold_180d = 0 AND lifetime_units_sold > 0 THEN 'Recently Dormant'
            WHEN units_sold_365d < 5 THEN 'Very Slow Mover'
            ELSE 'Active But Concerning'
        END AS stock_classification,
        -- Risk level
        CASE 
            WHEN (last_sale_date IS NULL OR days_since_last_sale > 365) 
                 AND stock_quantity * cost > 1000 THEN 'High Risk'
            WHEN (last_sale_date IS NULL OR days_since_last_sale > 180)
                 AND stock_quantity * cost > 500 THEN 'Medium Risk'
            WHEN days_since_last_sale > 90 OR units_sold_365d < 5 THEN 'Low Risk'
            ELSE 'Monitor'
        END AS risk_level
    FROM product_activity
    WHERE 
        (last_sale_date IS NULL AND product_age_days > 90) OR
        (days_since_last_sale > 180) OR
        (units_sold_365d < 5 AND stock_quantity > 10)
)
SELECT 
    product_id,
    product_name,
    sku,
    category_name,
    price,
    cost,
    stock_quantity,
    inventory_value,
    product_age_years,
    last_sale_date,
    days_since_last_sale,
    units_sold_90d,
    units_sold_180d,
    units_sold_365d,
    lifetime_units_sold,
    lifetime_revenue,
    stock_classification,
    risk_level,
    -- Action recommendations
    CASE 
        WHEN risk_level = 'High Risk' AND days_since_last_sale > 365 
            THEN 'Immediate: Clearance sale 50%+ off or donate'
        WHEN risk_level = 'High Risk' 
            THEN 'Urgent: Deep discount 40-50% off'
        WHEN risk_level = 'Medium Risk' AND stock_quantity > 50 
            THEN 'Promote: 30-40% discount + bundle deals'
        WHEN risk_level = 'Medium Risk' 
            THEN 'Discount: 20-30% off'
        WHEN stock_classification = 'Very Slow Mover' 
            THEN 'Monitor: Consider 15-20% promotion'
        ELSE 'Watch: Evaluate in 30 days'
    END AS recommended_action,
    -- Potential recovery
    ROUND(
        CASE 
            WHEN days_since_last_sale > 365 THEN inventory_value * 0.30
            WHEN days_since_last_sale > 180 THEN inventory_value * 0.50
            WHEN units_sold_365d < 5 THEN inventory_value * 0.70
            ELSE inventory_value * 0.85
        END,
        2
    ) AS estimated_recovery_value,
    ROUND(
        inventory_value - 
        CASE 
            WHEN days_since_last_sale > 365 THEN inventory_value * 0.30
            WHEN days_since_last_sale > 180 THEN inventory_value * 0.50
            WHEN units_sold_365d < 5 THEN inventory_value * 0.70
            ELSE inventory_value * 0.85
        END,
        2
    ) AS estimated_loss
FROM dead_stock_analysis
ORDER BY 
    FIELD(risk_level, 'High Risk', 'Medium Risk', 'Low Risk', 'Monitor'),
    inventory_value DESC;

-- ========================================
-- 6. CATEGORY-LEVEL INVENTORY PERFORMANCE
-- Aggregated metrics by category
-- ========================================

WITH category_metrics AS (
    SELECT 
        pc.category_name,
        COUNT(DISTINCT p.product_id) AS total_products,
        SUM(p.stock_quantity) AS total_units_in_stock,
        SUM(p.stock_quantity * p.cost) AS total_inventory_value,
        -- Sales performance
        SUM(oi.quantity) AS units_sold_12m,
        SUM(oi.subtotal) AS revenue_12m,
        SUM(oi.quantity * p.cost) AS cogs_12m,
        SUM(oi.subtotal - (oi.quantity * p.cost)) AS gross_profit_12m,
        COUNT(DISTINCT o.order_id) AS order_count_12m,
        -- Stock status
        SUM(CASE WHEN p.stock_quantity = 0 THEN 1 ELSE 0 END) AS out_of_stock_count,
        SUM(CASE 
            WHEN p.stock_quantity > 0 
                 AND NOT EXISTS (
                     SELECT 1 FROM order_items oi2 
                     JOIN orders o2 ON oi2.order_id = o2.order_id
                     WHERE oi2.product_id = p.product_id
                         AND o2.order_date >= DATE_SUB(CURDATE(), INTERVAL 180 DAY)
                         AND o2.status IN ('delivered', 'shipped', 'processing')
                 )
            THEN 1 ELSE 0 
        END) AS dead_stock_count
    FROM product_categories pc
    JOIN products p ON pc.category_id = p.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE p.status = 'active'
    GROUP BY pc.category_name
)
SELECT 
    category_name,
    total_products,
    total_units_in_stock,
    ROUND(total_inventory_value, 2) AS total_inventory_value,
    units_sold_12m,
    ROUND(revenue_12m, 2) AS revenue_12m,
    ROUND(gross_profit_12m, 2) AS gross_profit_12m,
    ROUND(gross_profit_12m / NULLIF(revenue_12m, 0) * 100, 2) AS profit_margin_pct,
    order_count_12m,
    -- Inventory turnover
    ROUND(
        cogs_12m / NULLIF(total_inventory_value, 0),
        2
    ) AS inventory_turnover_ratio,
    ROUND(
        365 / NULLIF(cogs_12m / NULLIF(total_inventory_value, 0), 0),
        1
    ) AS days_inventory_outstanding,
    -- Stock health
    out_of_stock_count,
    ROUND(out_of_stock_count * 100.0 / NULLIF(total_products, 0), 2) AS out_of_stock_pct,
    dead_stock_count,
    ROUND(dead_stock_count * 100.0 / NULLIF(total_products, 0), 2) AS dead_stock_pct,
    -- Performance metrics
    ROUND(revenue_12m / NULLIF(total_inventory_value, 0), 2) AS revenue_to_inventory_ratio,
    ROUND(units_sold_12m / NULLIF(total_units_in_stock, 0), 2) AS stock_efficiency_ratio,
    -- Category health score (0-100)
    ROUND(
        LEAST(100,
            (cogs_12m / NULLIF(total_inventory_value, 0) * 8.33) * 0.40 +  -- Turnover (40%)
            ((100 - (out_of_stock_count * 100.0 / NULLIF(total_products, 0)))) * 0.30 +  -- Availability (30%)
            ((100 - (dead_stock_count * 100.0 / NULLIF(total_products, 0)))) * 0.20 +  -- Freshness (20%)
            (gross_profit_12m / NULLIF(revenue_12m, 0) * 100) * 0.10  -- Profitability (10%)
        ),
        0
    ) AS category_health_score,
    -- Performance classification
    CASE 
        WHEN cogs_12m / NULLIF(total_inventory_value, 0) >= 8 
             AND out_of_stock_count * 100.0 / NULLIF(total_products, 0) < 10 THEN 'Excellent'
        WHEN cogs_12m / NULLIF(total_inventory_value, 0) >= 6 THEN 'Good'
        WHEN cogs_12m / NULLIF(total_inventory_value, 0) >= 4 THEN 'Fair'
        WHEN cogs_12m / NULLIF(total_inventory_value, 0) >= 2 THEN 'Poor'
        ELSE 'Critical'
    END AS category_performance,
    -- Strategic recommendations
    CASE 
        WHEN out_of_stock_count * 100.0 / NULLIF(total_products, 0) > 15 
            THEN 'Increase inventory levels - high stockout rate'
        WHEN dead_stock_count * 100.0 / NULLIF(total_products, 0) > 20 
            THEN 'Reduce SKU count - too much dead stock'
        WHEN cogs_12m / NULLIF(total_inventory_value, 0) < 3 
            THEN 'Improve turnover - slow moving category'
        WHEN cogs_12m / NULLIF(total_inventory_value, 0) > 10 
             AND out_of_stock_count > 0 
            THEN 'Optimize stock levels - high demand'
        ELSE 'Maintain current strategy'
    END AS strategic_recommendation
FROM category_metrics
ORDER BY revenue_12m DESC;

-- ========================================
-- 7. INVENTORY AGING ANALYSIS
-- Shows how long inventory has been sitting
-- ========================================

WITH product_aging AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price,
        p.cost,
        p.stock_quantity,
        p.created_at AS product_listed_date,
        DATEDIFF(CURDATE(), p.created_at) AS product_age_days,
        -- Last activity
        MAX(o.order_date) AS last_sale_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_sale,
        -- Sales velocity
        SUM(oi.quantity) AS lifetime_units_sold,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN oi.quantity ELSE 0 
        END) AS units_sold_90d,
        -- Average days to sell based on current rate
        CASE 
            WHEN SUM(CASE 
                WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
                THEN oi.quantity ELSE 0 
            END) > 0 
            THEN p.stock_quantity / (SUM(CASE 
                WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
                THEN oi.quantity ELSE 0 
            END) / 90.0)
            ELSE NULL
        END AS estimated_days_to_sell_stock
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    WHERE p.status = 'active'
        AND p.stock_quantity > 0
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name,
             p.price, p.cost, p.stock_quantity, p.created_at
)
SELECT 
    product_id,
    product_name,
    sku,
    category_name,
    ROUND(price, 2) AS price,
    ROUND(cost, 2) AS cost,
    stock_quantity,
    ROUND(stock_quantity * cost, 2) AS inventory_value,
    product_listed_date,
    ROUND(product_age_days / 30.0, 1) AS product_age_months,
    last_sale_date,
    days_since_last_sale,
    units_sold_90d,
    lifetime_units_sold,
    ROUND(estimated_days_to_sell_stock, 0) AS days_to_sell_current_stock,
    -- Aging buckets
    CASE 
        WHEN days_since_last_sale IS NULL THEN 'Never Sold'
        WHEN days_since_last_sale <= 30 THEN '0-30 days'
        WHEN days_since_last_sale <= 60 THEN '31-60 days'
        WHEN days_since_last_sale <= 90 THEN '61-90 days'
        WHEN days_since_last_sale <= 180 THEN '91-180 days'
        WHEN days_since_last_sale <= 365 THEN '181-365 days'
        ELSE '365+ days'
    END AS aging_bucket,
    -- Risk assessment
    CASE 
        WHEN days_since_last_sale IS NULL AND product_age_days > 90 
            THEN 'High Risk - Never sold'
        WHEN days_since_last_sale > 365 
            THEN 'High Risk - Very stale'
        WHEN days_since_last_sale > 180 
            THEN 'Medium Risk - Aging'
        WHEN days_since_last_sale > 90 AND estimated_days_to_sell_stock > 180 
            THEN 'Medium Risk - Slow mover'
        WHEN estimated_days_to_sell_stock > 365 
            THEN 'Low Risk - But overstocked'
        ELSE 'Low Risk - Fresh'
    END AS aging_risk,
    -- Markdown recommendations
    CASE 
        WHEN days_since_last_sale > 365 OR (days_since_last_sale IS NULL AND product_age_days > 180)
            THEN ROUND(price * 0.50, 2)  -- 50% off
        WHEN days_since_last_sale > 180 
            THEN ROUND(price * 0.70, 2)  -- 30% off
        WHEN days_since_last_sale > 90 AND estimated_days_to_sell_stock > 180
            THEN ROUND(price * 0.80, 2)  -- 20% off
        WHEN estimated_days_to_sell_stock > 365
            THEN ROUND(price * 0.85, 2)  -- 15% off
        ELSE price  -- No discount
    END AS suggested_clearance_price,
    -- Potential loss if cleared at discount
    ROUND(
        stock_quantity * (cost - 
            CASE 
                WHEN days_since_last_sale > 365 OR (days_since_last_sale IS NULL AND product_age_days > 180)
                    THEN price * 0.50
                WHEN days_since_last_sale > 180 
                    THEN price * 0.70
                WHEN days_since_last_sale > 90 AND estimated_days_to_sell_stock > 180
                    THEN price * 0.80
                WHEN estimated_days_to_sell_stock > 365
                    THEN price * 0.85
                ELSE price
            END
        ),
        2
    ) AS potential_loss_at_clearance
FROM product_aging
ORDER BY 
    CASE 
        WHEN days_since_last_sale IS NULL AND product_age_days > 90 THEN 1
        WHEN days_since_last_sale > 365 THEN 2
        WHEN days_since_last_sale > 180 THEN 3
        WHEN days_since_last_sale > 90 THEN 4
        ELSE 5
    END,
    inventory_value DESC;

-- ========================================
-- 8. REORDER ALERT DASHBOARD
-- Products requiring immediate action
-- ========================================

WITH reorder_metrics AS (
    SELECT 
        p.product_id,
        p.product_name,
        p.sku,
        pc.category_name,
        p.price,
        p.cost,
        p.stock_quantity,
        i.quantity_reserved,
        i.quantity_available,
        i.reorder_level,
        -- Sales velocity (last 30 days)
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            THEN oi.quantity ELSE 0 
        END) / 30.0 AS daily_sales_rate,
        -- Vendor info
        MIN(vc.lead_time_days) AS shortest_lead_time,
        MIN(vc.minimum_order_quantity) AS min_order_qty,
        MIN(vc.cost_per_unit) AS vendor_cost,
        -- Recent demand
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) 
            THEN o.order_id 
        END) AS orders_last_7_days
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN inventory i ON p.product_id = i.product_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
    LEFT JOIN vendor_contracts vc ON p.product_id = vc.product_id
        AND vc.status = 'active'
    WHERE p.status = 'active'
    GROUP BY p.product_id, p.product_name, p.sku, pc.category_name,
             p.price, p.cost, p.stock_quantity, i.quantity_reserved,
             i.quantity_available, i.reorder_level
)
SELECT 
    product_id,
    product_name,
    sku,
    category_name,
    ROUND(price, 2) AS price,
    ROUND(cost, 2) AS cost,
    stock_quantity,
    quantity_reserved,
    quantity_available,
    reorder_level,
    ROUND(daily_sales_rate, 2) AS avg_daily_sales,
    orders_last_7_days,
    shortest_lead_time AS lead_time_days,
    -- Days until stockout
    ROUND(
        quantity_available / NULLIF(daily_sales_rate, 0),
        0
    ) AS days_until_stockout,
    -- Recommended order quantity
    GREATEST(
        CEIL(daily_sales_rate * (shortest_lead_time + 30)),  -- Lead time + 1 month buffer
        COALESCE(min_order_qty, 0)
    ) AS recommended_order_qty,
    -- Order cost
    ROUND(
        GREATEST(
            CEIL(daily_sales_rate * (shortest_lead_time + 30)),
            COALESCE(min_order_qty, 0)
        ) * COALESCE(vendor_cost, cost),
        2
    ) AS estimated_order_cost,
    -- Alert priority
    CASE 
        WHEN quantity_available = 0 THEN 'CRITICAL - Out of Stock'
        WHEN quantity_available / NULLIF(daily_sales_rate, 0) < 7 THEN 'URGENT - Less than 1 week'
        WHEN quantity_available <= reorder_level THEN 'HIGH - At reorder point'
        WHEN quantity_available / NULLIF(daily_sales_rate, 0) < 14 THEN 'MEDIUM - Less than 2 weeks'
        WHEN quantity_available / NULLIF(daily_sales_rate, 0) < shortest_lead_time THEN 'MEDIUM - Below lead time'
        ELSE 'LOW - Monitor'
    END AS reorder_priority,
    -- Action needed
    CASE 
        WHEN quantity_available = 0 THEN 'Rush order immediately'
        WHEN quantity_available / NULLIF(daily_sales_rate, 0) < 7 THEN 'Order today'
        WHEN quantity_available <= reorder_level THEN 'Place order this week'
        WHEN quantity_available / NULLIF(daily_sales_rate, 0) < 14 THEN 'Schedule order soon'
        ELSE 'Monitor stock levels'
    END AS action_required,
    -- Revenue at risk
    ROUND(
        CASE 
            WHEN quantity_available = 0 THEN daily_sales_rate * price * 7
            WHEN quantity_available / NULLIF(daily_sales_rate, 0) < 7 
                THEN daily_sales_rate * price * (7 - quantity_available / NULLIF(daily_sales_rate, 0))
            ELSE 0
        END,
        2
    ) AS revenue_at_risk_7days
FROM reorder_metrics
WHERE 
    quantity_available = 0 
    OR quantity_available <= reorder_level
    OR quantity_available / NULLIF(daily_sales_rate, 0) < shortest_lead_time
ORDER BY 
    FIELD(reorder_priority, 'CRITICAL - Out of Stock', 'URGENT - Less than 1 week', 
          'HIGH - At reorder point', 'MEDIUM - Less than 2 weeks', 
          'MEDIUM - Below lead time', 'LOW - Monitor'),
    revenue_at_risk_7days DESC;

-- ========================================
-- End of Inventory Turnover & Optimization Analysis
-- ========================================