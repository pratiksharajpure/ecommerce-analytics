-- ========================================
-- ROI CALCULATIONS & VALUE METRICS
-- E-commerce Revenue Analytics Engine
-- Return on Investment, Cost-Benefit Analysis
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. MARKETING CAMPAIGN ROI
-- Calculate ROI for each marketing campaign
-- ========================================
SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.budget,
    SUM(cp.spend) AS total_spend,
    SUM(cp.revenue) AS total_revenue,
    SUM(cp.conversions) AS total_conversions,
    SUM(cp.clicks) AS total_clicks,
    SUM(cp.impressions) AS total_impressions,
    -- ROI Calculation
    ROUND(((SUM(cp.revenue) - SUM(cp.spend)) / NULLIF(SUM(cp.spend), 0)) * 100, 2) AS roi_percentage,
    ROUND(SUM(cp.revenue) - SUM(cp.spend), 2) AS net_profit,
    -- Cost per metrics
    ROUND(SUM(cp.spend) / NULLIF(SUM(cp.clicks), 0), 2) AS cost_per_click,
    ROUND(SUM(cp.spend) / NULLIF(SUM(cp.conversions), 0), 2) AS cost_per_conversion,
    ROUND(SUM(cp.revenue) / NULLIF(SUM(cp.conversions), 0), 2) AS revenue_per_conversion,
    -- Efficiency metrics
    ROUND((SUM(cp.clicks) / NULLIF(SUM(cp.impressions), 0)) * 100, 2) AS click_through_rate,
    ROUND((SUM(cp.conversions) / NULLIF(SUM(cp.clicks), 0)) * 100, 2) AS conversion_rate,
    -- Budget utilization
    ROUND((SUM(cp.spend) / NULLIF(c.budget, 0)) * 100, 2) AS budget_utilization_pct
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
WHERE c.status IN ('active', 'completed')
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.budget
ORDER BY roi_percentage DESC;

-- ========================================
-- 2. PRODUCT PROFITABILITY ANALYSIS
-- Calculate profit margins and ROI per product
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    p.price AS selling_price,
    p.cost AS product_cost,
    -- Profit margins
    ROUND(p.price - p.cost, 2) AS gross_profit_per_unit,
    ROUND(((p.price - p.cost) / NULLIF(p.price, 0)) * 100, 2) AS gross_margin_percentage,
    -- Sales metrics
    COUNT(DISTINCT oi.order_id) AS total_orders,
    SUM(oi.quantity) AS total_units_sold,
    SUM(oi.subtotal) AS total_revenue,
    SUM(oi.quantity * p.cost) AS total_cost,
    -- Profitability
    ROUND(SUM(oi.subtotal) - SUM(oi.quantity * p.cost), 2) AS total_gross_profit,
    ROUND(((SUM(oi.subtotal) - SUM(oi.quantity * p.cost)) / NULLIF(SUM(oi.quantity * p.cost), 0)) * 100, 2) AS product_roi_percentage,
    -- Average metrics
    ROUND(AVG(oi.quantity), 2) AS avg_quantity_per_order,
    ROUND(AVG(oi.subtotal), 2) AS avg_revenue_per_order
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.status = 'active'
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name, p.price, p.cost
HAVING total_units_sold > 0
ORDER BY total_gross_profit DESC;

-- ========================================
-- 3. CUSTOMER LIFETIME VALUE (CLV) & ROI
-- Calculate CLV and customer acquisition cost
-- ========================================
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.status,
    -- Order metrics
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_spent,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value,
    MIN(o.order_date) AS first_order_date,
    MAX(o.order_date) AS last_order_date,
    DATEDIFF(MAX(o.order_date), MIN(o.order_date)) AS customer_lifespan_days,
    -- CLV calculation
    ROUND(SUM(o.total_amount) / NULLIF(DATEDIFF(MAX(o.order_date), MIN(o.order_date)), 0) * 365, 2) AS annualized_clv,
    -- Customer value tier
    CASE 
        WHEN SUM(o.total_amount) >= 10000 THEN 'VIP'
        WHEN SUM(o.total_amount) >= 5000 THEN 'High Value'
        WHEN SUM(o.total_amount) >= 1000 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS customer_tier,
    -- Loyalty metrics
    lp.points_balance,
    lp.tier AS loyalty_tier,
    lp.points_earned_lifetime
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
WHERE o.payment_status = 'paid'
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.status, 
         lp.points_balance, lp.tier, lp.points_earned_lifetime
HAVING total_orders > 0
ORDER BY total_spent DESC;

-- ========================================
-- 4. VENDOR COST-BENEFIT ANALYSIS
-- Evaluate vendor performance and cost efficiency
-- ========================================
SELECT 
    v.vendor_id,
    v.vendor_name,
    v.rating,
    v.status,
    -- Contract metrics
    COUNT(DISTINCT vc.contract_id) AS total_contracts,
    COUNT(DISTINCT vc.product_id) AS products_supplied,
    ROUND(AVG(vc.cost_per_unit), 2) AS avg_cost_per_unit,
    AVG(vc.lead_time_days) AS avg_lead_time_days,
    -- Active contracts
    SUM(CASE WHEN vc.status = 'active' THEN 1 ELSE 0 END) AS active_contracts,
    -- Product performance with this vendor
    SUM(oi.quantity) AS total_units_ordered,
    ROUND(SUM(oi.quantity * vc.cost_per_unit), 2) AS total_vendor_cost,
    ROUND(SUM(oi.subtotal), 2) AS total_revenue_generated,
    -- Value metrics
    ROUND(SUM(oi.subtotal) - SUM(oi.quantity * vc.cost_per_unit), 2) AS total_margin_contribution,
    ROUND(((SUM(oi.subtotal) - SUM(oi.quantity * vc.cost_per_unit)) / NULLIF(SUM(oi.quantity * vc.cost_per_unit), 0)) * 100, 2) AS vendor_roi_percentage,
    -- Efficiency score (combines rating, lead time, and ROI)
    ROUND(
        (v.rating * 20) + 
        (100 - AVG(vc.lead_time_days)) / 2 + 
        LEAST(((SUM(oi.subtotal) - SUM(oi.quantity * vc.cost_per_unit)) / NULLIF(SUM(oi.quantity * vc.cost_per_unit), 0)) * 10, 50),
    2) AS vendor_efficiency_score
FROM vendors v
LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
LEFT JOIN order_items oi ON vc.product_id = oi.product_id
WHERE v.status = 'active'
GROUP BY v.vendor_id, v.vendor_name, v.rating, v.status
HAVING total_units_ordered > 0
ORDER BY vendor_efficiency_score DESC;

-- ========================================
-- 5. INVENTORY HOLDING COST ANALYSIS
-- Calculate inventory carrying costs and turnover
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.quantity_available,
    p.cost AS unit_cost,
    -- Inventory value
    ROUND(i.quantity_on_hand * p.cost, 2) AS total_inventory_value,
    ROUND(i.quantity_available * p.cost, 2) AS available_inventory_value,
    -- Sales metrics
    COALESCE(SUM(oi.quantity), 0) AS units_sold_last_90_days,
    -- Inventory turnover
    ROUND(COALESCE(SUM(oi.quantity), 0) / NULLIF(i.quantity_on_hand, 0), 2) AS inventory_turnover_ratio,
    ROUND(90 / NULLIF(COALESCE(SUM(oi.quantity), 0) / NULLIF(i.quantity_on_hand, 0), 0), 0) AS days_to_sell_inventory,
    -- Holding cost estimation (assuming 25% annual holding cost)
    ROUND((i.quantity_on_hand * p.cost * 0.25) / 365 * 90, 2) AS estimated_holding_cost_90_days,
    -- Stock status
    CASE 
        WHEN i.quantity_available <= 0 THEN 'Out of Stock'
        WHEN i.quantity_available <= i.reorder_level THEN 'Reorder Needed'
        WHEN i.quantity_available > i.reorder_level * 5 THEN 'Overstocked'
        ELSE 'Optimal'
    END AS stock_status
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN inventory i ON p.product_id = i.product_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id 
    AND oi.created_at >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 90 DAY)
WHERE p.status = 'active' AND i.inventory_id IS NOT NULL
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name, 
         i.quantity_on_hand, i.quantity_reserved, i.quantity_available, 
         p.cost, i.reorder_level
ORDER BY total_inventory_value DESC;

-- ========================================
-- 6. RETURN RATE & COST ANALYSIS
-- Calculate return rates and associated costs
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    -- Sales metrics
    COUNT(DISTINCT oi.order_id) AS total_orders,
    SUM(oi.quantity) AS total_units_sold,
    ROUND(SUM(oi.subtotal), 2) AS total_revenue,
    -- Return metrics
    COUNT(DISTINCT r.return_id) AS total_returns,
    SUM(CASE WHEN r.status IN ('approved', 'received', 'refunded') THEN 1 ELSE 0 END) AS approved_returns,
    ROUND(SUM(r.refund_amount), 2) AS total_refund_amount,
    -- Return rate calculation
    ROUND((COUNT(DISTINCT r.return_id) / NULLIF(COUNT(DISTINCT oi.order_id), 0)) * 100, 2) AS return_rate_percentage,
    ROUND((SUM(r.refund_amount) / NULLIF(SUM(oi.subtotal), 0)) * 100, 2) AS refund_rate_percentage,
    -- Net revenue after returns
    ROUND(SUM(oi.subtotal) - COALESCE(SUM(r.refund_amount), 0), 2) AS net_revenue,
    -- Return reasons breakdown
    SUM(CASE WHEN r.reason = 'defective' THEN 1 ELSE 0 END) AS defective_returns,
    SUM(CASE WHEN r.reason = 'wrong_item' THEN 1 ELSE 0 END) AS wrong_item_returns,
    SUM(CASE WHEN r.reason = 'not_as_described' THEN 1 ELSE 0 END) AS not_as_described_returns,
    SUM(CASE WHEN r.reason = 'changed_mind' THEN 1 ELSE 0 END) AS changed_mind_returns,
    -- Quality score (inverse of return rate)
    ROUND(100 - ((COUNT(DISTINCT r.return_id) / NULLIF(COUNT(DISTINCT oi.order_id), 0)) * 100), 2) AS quality_score
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
WHERE oi.created_at >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 180 DAY)
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name
HAVING total_units_sold > 10
ORDER BY return_rate_percentage DESC;

-- ========================================
-- 7. OVERALL BUSINESS ROI DASHBOARD
-- High-level business performance metrics
-- ========================================
SELECT 
    'Overall Business Metrics' AS metric_category,
    -- Revenue metrics
    ROUND(SUM(o.total_amount), 2) AS total_revenue,
    ROUND(SUM(oi.quantity * p.cost), 2) AS total_cogs,
    ROUND(SUM(o.total_amount) - SUM(oi.quantity * p.cost), 2) AS gross_profit,
    ROUND(((SUM(o.total_amount) - SUM(oi.quantity * p.cost)) / NULLIF(SUM(oi.quantity * p.cost), 0)) * 100, 2) AS gross_margin_percentage,
    -- Order metrics
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value,
    -- Customer metrics
    COUNT(DISTINCT o.customer_id) AS total_customers,
    ROUND(SUM(o.total_amount) / NULLIF(COUNT(DISTINCT o.customer_id), 0), 2) AS revenue_per_customer,
    -- Product metrics
    COUNT(DISTINCT oi.product_id) AS products_sold,
    SUM(oi.quantity) AS total_units_sold,
    -- Marketing ROI
    (SELECT ROUND(SUM(cp.revenue) - SUM(cp.spend), 2) 
     FROM campaign_performance cp) AS marketing_net_profit,
    (SELECT ROUND(((SUM(cp.revenue) - SUM(cp.spend)) / NULLIF(SUM(cp.spend), 0)) * 100, 2)
     FROM campaign_performance cp) AS marketing_roi_percentage,
    -- Return impact
    (SELECT COUNT(*) FROM returns WHERE status IN ('approved', 'received', 'refunded')) AS total_returns,
    (SELECT ROUND(SUM(refund_amount), 2) FROM returns WHERE status = 'refunded') AS total_refunds,
    -- Net profit estimation (gross profit - marketing spend - refunds)
    ROUND(
        (SUM(o.total_amount) - SUM(oi.quantity * p.cost)) - 
        COALESCE((SELECT SUM(cp.spend) FROM campaign_performance cp), 0) -
        COALESCE((SELECT SUM(refund_amount) FROM returns WHERE status = 'refunded'), 0),
    2) AS estimated_net_profit
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.payment_status = 'paid'
  AND o.order_date >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 12 MONTH);

-- ========================================
-- 8. CATEGORY-LEVEL ROI ANALYSIS
-- Performance breakdown by product category
-- ========================================
SELECT 
    pc.category_id,
    pc.category_name,
    -- Sales metrics
    COUNT(DISTINCT oi.order_id) AS total_orders,
    SUM(oi.quantity) AS total_units_sold,
    ROUND(SUM(oi.subtotal), 2) AS total_revenue,
    ROUND(AVG(oi.subtotal), 2) AS avg_revenue_per_order,
    -- Cost and profitability
    ROUND(SUM(oi.quantity * p.cost), 2) AS total_cost,
    ROUND(SUM(oi.subtotal) - SUM(oi.quantity * p.cost), 2) AS gross_profit,
    ROUND(((SUM(oi.subtotal) - SUM(oi.quantity * p.cost)) / NULLIF(SUM(oi.quantity * p.cost), 0)) * 100, 2) AS roi_percentage,
    ROUND(((SUM(oi.subtotal) - SUM(oi.quantity * p.cost)) / NULLIF(SUM(oi.subtotal), 0)) * 100, 2) AS profit_margin_percentage,
    -- Product variety
    COUNT(DISTINCT p.product_id) AS unique_products,
    -- Market share
    ROUND((SUM(oi.subtotal) / (SELECT SUM(subtotal) FROM order_items) * 100), 2) AS revenue_share_percentage
FROM product_categories pc
JOIN products p ON pc.category_id = p.category_id
JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.created_at >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 6 MONTH)
GROUP BY pc.category_id, pc.category_name
ORDER BY gross_profit DESC;

-- Display completion message
SELECT 'ROI Calculations and Value Metrics Analysis Complete' AS Status;