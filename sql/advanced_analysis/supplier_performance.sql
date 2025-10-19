-- ========================================
-- SUPPLIER PERFORMANCE ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Vendor Scorecards, Delivery Performance, Quality Ratings
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. COMPREHENSIVE VENDOR SCORECARD
-- 360-degree supplier performance metrics
-- ========================================

WITH vendor_product_sales AS (
    SELECT 
        v.vendor_id,
        v.vendor_name,
        vc.product_id,
        p.product_name,
        COUNT(DISTINCT o.order_id) AS orders_containing_product,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS revenue_generated,
        SUM(oi.quantity * p.cost) AS cogs,
        SUM(oi.subtotal - (oi.quantity * p.cost)) AS gross_profit,
        -- Returns
        COUNT(DISTINCT r.return_id) AS return_count,
        SUM(CASE WHEN r.reason = 'defective' THEN 1 ELSE 0 END) AS defective_returns,
        SUM(CASE WHEN r.reason = 'wrong_item' THEN 1 ELSE 0 END) AS wrong_item_returns,
        -- Reviews
        COUNT(DISTINCT rev.review_id) AS review_count,
        AVG(CASE WHEN rev.rating IS NOT NULL THEN rev.rating END) AS avg_product_rating,
        SUM(CASE WHEN rev.rating <= 2 THEN 1 ELSE 0 END) AS negative_reviews
    FROM vendors v
    JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    JOIN products p ON vc.product_id = p.product_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN returns r ON o.order_id = r.order_id
        AND r.status IN ('approved', 'refunded')
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    WHERE v.status = 'active'
    GROUP BY v.vendor_id, v.vendor_name, vc.product_id, p.product_name
),
vendor_aggregates AS (
    SELECT 
        vendor_id,
        vendor_name,
        COUNT(DISTINCT product_id) AS active_products,
        SUM(units_sold) AS total_units_sold,
        SUM(revenue_generated) AS total_revenue,
        SUM(gross_profit) AS total_profit,
        AVG(avg_product_rating) AS avg_vendor_rating,
        SUM(return_count) AS total_returns,
        SUM(defective_returns) AS defective_returns,
        SUM(wrong_item_returns) AS wrong_item_returns,
        SUM(review_count) AS total_reviews,
        SUM(negative_reviews) AS negative_reviews
    FROM vendor_product_sales
    GROUP BY vendor_id, vendor_name
),
vendor_contracts_info AS (
    SELECT 
        v.vendor_id,
        COUNT(DISTINCT vc.contract_id) AS active_contracts,
        AVG(vc.cost_per_unit) AS avg_unit_cost,
        AVG(vc.lead_time_days) AS avg_lead_time,
        MIN(vc.lead_time_days) AS min_lead_time,
        MAX(vc.lead_time_days) AS max_lead_time,
        COUNT(DISTINCT CASE WHEN vc.status = 'expired' THEN vc.contract_id END) AS expired_contracts
    FROM vendors v
    JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    WHERE v.status = 'active'
    GROUP BY v.vendor_id
)
SELECT 
    v.vendor_id,
    v.vendor_name,
    v.contact_person,
    v.email,
    v.phone,
    v.city,
    v.state,
    ROUND(v.rating, 2) AS vendor_rating,
    va.active_products,
    vci.active_contracts,
    vci.expired_contracts,
    -- Financial metrics
    ROUND(va.total_revenue, 2) AS total_revenue_12m,
    ROUND(va.total_profit, 2) AS total_gross_profit_12m,
    ROUND(va.total_profit / NULLIF(va.total_revenue, 0) * 100, 2) AS profit_margin_pct,
    va.total_units_sold,
    ROUND(va.total_revenue / NULLIF(va.active_products, 0), 2) AS avg_revenue_per_product,
    -- Quality metrics
    ROUND(va.avg_vendor_rating, 2) AS avg_product_rating,
    va.total_reviews,
    va.negative_reviews,
    ROUND(va.negative_reviews * 100.0 / NULLIF(va.total_reviews, 0), 2) AS negative_review_pct,
    -- Returns & defects
    va.total_returns,
    ROUND(va.total_returns * 100.0 / NULLIF(va.total_units_sold, 0), 2) AS return_rate_pct,
    va.defective_returns,
    ROUND(va.defective_returns * 100.0 / NULLIF(va.total_returns, 0), 2) AS defect_rate_pct,
    va.wrong_item_returns,
    -- Lead time metrics
    ROUND(vci.avg_lead_time, 1) AS avg_lead_time_days,
    vci.min_lead_time AS min_lead_time_days,
    vci.max_lead_time AS max_lead_time_days,
    -- Overall performance score (0-100)
    ROUND(
        LEAST(100,
            -- Revenue performance (25%)
            (va.total_revenue / 10000 * 25) * 0.25 +
            -- Quality rating (25%)
            (COALESCE(va.avg_vendor_rating, 3) / 5 * 100) * 0.25 +
            -- Low return rate (25%) - inverse scoring
            (100 - LEAST(100, (va.total_returns * 100.0 / NULLIF(va.total_units_sold, 0)) * 10)) * 0.25 +
            -- Lead time performance (25%) - shorter is better
            (100 - LEAST(100, vci.avg_lead_time * 2)) * 0.25
        ),
        0
    ) AS overall_performance_score,
    -- Performance tier
    CASE 
        WHEN LEAST(100,
            (va.total_revenue / 10000 * 25) * 0.25 +
            (COALESCE(va.avg_vendor_rating, 3) / 5 * 100) * 0.25 +
            (100 - LEAST(100, (va.total_returns * 100.0 / NULLIF(va.total_units_sold, 0)) * 10)) * 0.25 +
            (100 - LEAST(100, vci.avg_lead_time * 2)) * 0.25
        ) >= 80 THEN 'A - Excellent'
        WHEN LEAST(100,
            (va.total_revenue / 10000 * 25) * 0.25 +
            (COALESCE(va.avg_vendor_rating, 3) / 5 * 100) * 0.25 +
            (100 - LEAST(100, (va.total_returns * 100.0 / NULLIF(va.total_units_sold, 0)) * 10)) * 0.25 +
            (100 - LEAST(100, vci.avg_lead_time * 2)) * 0.25
        ) >= 65 THEN 'B - Good'
        WHEN LEAST(100,
            (va.total_revenue / 10000 * 25) * 0.25 +
            (COALESCE(va.avg_vendor_rating, 3) / 5 * 100) * 0.25 +
            (100 - LEAST(100, (va.total_returns * 100.0 / NULLIF(va.total_units_sold, 0)) * 10)) * 0.25 +
            (100 - LEAST(100, vci.avg_lead_time * 2)) * 0.25
        ) >= 50 THEN 'C - Average'
        WHEN LEAST(100,
            (va.total_revenue / 10000 * 25) * 0.25 +
            (COALESCE(va.avg_vendor_rating, 3) / 5 * 100) * 0.25 +
            (100 - LEAST(100, (va.total_returns * 100.0 / NULLIF(va.total_units_sold, 0)) * 10)) * 0.25 +
            (100 - LEAST(100, vci.avg_lead_time * 2)) * 0.25
        ) >= 35 THEN 'D - Below Average'
        ELSE 'F - Poor'
    END AS performance_tier,
    -- Strategic recommendation
    CASE 
        WHEN va.total_revenue >= 50000 
             AND va.avg_vendor_rating >= 4.0 
             AND va.total_returns * 100.0 / NULLIF(va.total_units_sold, 0) < 5 
            THEN 'Strategic Partner - Expand relationship'
        WHEN va.total_revenue >= 20000 
             AND va.avg_vendor_rating >= 3.5 
            THEN 'Preferred Vendor - Maintain relationship'
        WHEN va.total_returns * 100.0 / NULLIF(va.total_units_sold, 0) > 15 
             OR va.avg_vendor_rating < 2.5 
            THEN 'Performance Review Required'
        WHEN va.defective_returns > 10 
            THEN 'Quality Improvement Plan Needed'
        WHEN vci.avg_lead_time > 30 
            THEN 'Negotiate Better Lead Times'
        ELSE 'Continue Monitoring'
    END AS strategic_action
FROM vendors v
LEFT JOIN vendor_aggregates va ON v.vendor_id = va.vendor_id
LEFT JOIN vendor_contracts_info vci ON v.vendor_id = vci.vendor_id
WHERE v.status = 'active'
ORDER BY overall_performance_score DESC, total_revenue_12m DESC;

-- ========================================
-- 2. DELIVERY PERFORMANCE ANALYSIS
-- On-time delivery and lead time tracking
-- ========================================

WITH order_fulfillment AS (
    SELECT 
        v.vendor_id,
        v.vendor_name,
        p.product_id,
        p.product_name,
        o.order_id,
        o.order_date,
        o.status AS order_status,
        vc.lead_time_days AS promised_lead_time,
        -- Calculate actual delivery time (using order status transitions)
        CASE 
            WHEN o.status = 'delivered' THEN DATEDIFF(o.updated_at, o.order_date)
            WHEN o.status = 'shipped' THEN DATEDIFF(o.updated_at, o.order_date)
            ELSE NULL
        END AS actual_delivery_days,
        oi.quantity,
        oi.subtotal,
        -- Delivery performance
        CASE 
            WHEN o.status = 'delivered' 
                 AND DATEDIFF(o.updated_at, o.order_date) <= vc.lead_time_days 
                THEN 1 ELSE 0 
        END AS on_time_delivery,
        CASE 
            WHEN o.status = 'delivered' 
                 AND DATEDIFF(o.updated_at, o.order_date) <= vc.lead_time_days * 0.9 
                THEN 1 ELSE 0 
        END AS early_delivery,
        CASE 
            WHEN o.status = 'delivered' 
                 AND DATEDIFF(o.updated_at, o.order_date) > vc.lead_time_days 
                THEN 1 ELSE 0 
        END AS late_delivery
    FROM vendors v
    JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    JOIN products p ON vc.product_id = p.product_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND v.status = 'active'
        AND o.status IN ('delivered', 'shipped', 'processing')
),
delivery_metrics AS (
    SELECT 
        vendor_id,
        vendor_name,
        COUNT(DISTINCT order_id) AS total_orders,
        COUNT(DISTINCT product_id) AS products_supplied,
        SUM(quantity) AS total_units,
        ROUND(SUM(subtotal), 2) AS total_order_value,
        -- Delivery performance
        SUM(on_time_delivery) AS on_time_deliveries,
        SUM(early_delivery) AS early_deliveries,
        SUM(late_delivery) AS late_deliveries,
        ROUND(AVG(actual_delivery_days), 1) AS avg_actual_delivery_days,
        ROUND(AVG(promised_lead_time), 1) AS avg_promised_lead_time,
        ROUND(MIN(actual_delivery_days), 1) AS best_delivery_time,
        ROUND(MAX(actual_delivery_days), 1) AS worst_delivery_time,
        ROUND(STDDEV(actual_delivery_days), 1) AS delivery_time_variance
    FROM order_fulfillment
    WHERE actual_delivery_days IS NOT NULL
    GROUP BY vendor_id, vendor_name
)
SELECT 
    vendor_id,
    vendor_name,
    total_orders,
    products_supplied,
    total_units,
    total_order_value,
    -- Delivery metrics
    on_time_deliveries,
    early_deliveries,
    late_deliveries,
    ROUND(on_time_deliveries * 100.0 / NULLIF(total_orders, 0), 2) AS on_time_delivery_rate_pct,
    ROUND(early_deliveries * 100.0 / NULLIF(total_orders, 0), 2) AS early_delivery_rate_pct,
    ROUND(late_deliveries * 100.0 / NULLIF(total_orders, 0), 2) AS late_delivery_rate_pct,
    -- Lead time analysis
    avg_promised_lead_time,
    avg_actual_delivery_days,
    ROUND(avg_actual_delivery_days - avg_promised_lead_time, 1) AS avg_delivery_variance_days,
    best_delivery_time,
    worst_delivery_time,
    delivery_time_variance,
    -- Reliability score
    ROUND(
        (on_time_deliveries * 100.0 / NULLIF(total_orders, 0)) * 0.60 +  -- On-time weight (60%)
        (100 - LEAST(100, delivery_time_variance * 5)) * 0.25 +           -- Consistency (25%)
        (CASE WHEN avg_actual_delivery_days <= avg_promised_lead_time 
              THEN 100 ELSE 50 END) * 0.15,                               -- Promise keeping (15%)
        0
    ) AS delivery_reliability_score,
    -- Performance classification
    CASE 
        WHEN on_time_deliveries * 100.0 / NULLIF(total_orders, 0) >= 95 THEN 'Excellent'
        WHEN on_time_deliveries * 100.0 / NULLIF(total_orders, 0) >= 85 THEN 'Good'
        WHEN on_time_deliveries * 100.0 / NULLIF(total_orders, 0) >= 70 THEN 'Fair'
        WHEN on_time_deliveries * 100.0 / NULLIF(total_orders, 0) >= 50 THEN 'Poor'
        ELSE 'Unacceptable'
    END AS delivery_performance,
    -- Action recommendation
    CASE 
        WHEN late_deliveries * 100.0 / NULLIF(total_orders, 0) > 30 
            THEN 'Critical: Renegotiate lead times or find alternative'
        WHEN late_deliveries * 100.0 / NULLIF(total_orders, 0) > 15 
            THEN 'Warning: Address delivery issues immediately'
        WHEN delivery_time_variance > 5 
            THEN 'Action: Improve delivery consistency'
        WHEN on_time_deliveries * 100.0 / NULLIF(total_orders, 0) >= 95 
            THEN 'Excellent: Consider preferred vendor status'
        ELSE 'Monitor: Maintain current relationship'
    END AS action_recommendation
FROM delivery_metrics
ORDER BY on_time_delivery_rate_pct DESC, total_order_value DESC;

-- ========================================
-- 3. QUALITY RATINGS & DEFECT ANALYSIS
-- Product quality and customer satisfaction
-- ========================================

WITH product_quality_metrics AS (
    SELECT 
        v.vendor_id,
        v.vendor_name,
        p.product_id,
        p.product_name,
        pc.category_name,
        -- Sales volume
        COUNT(DISTINCT o.order_id) AS orders_with_product,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS revenue,
        -- Customer reviews
        COUNT(DISTINCT rev.review_id) AS total_reviews,
        AVG(CASE WHEN rev.rating IS NOT NULL THEN rev.rating END) AS avg_rating,
        SUM(CASE WHEN rev.rating = 5 THEN 1 ELSE 0 END) AS five_star_reviews,
        SUM(CASE WHEN rev.rating = 4 THEN 1 ELSE 0 END) AS four_star_reviews,
        SUM(CASE WHEN rev.rating = 3 THEN 1 ELSE 0 END) AS three_star_reviews,
        SUM(CASE WHEN rev.rating = 2 THEN 1 ELSE 0 END) AS two_star_reviews,
        SUM(CASE WHEN rev.rating = 1 THEN 1 ELSE 0 END) AS one_star_reviews,
        -- Returns analysis
        COUNT(DISTINCT r.return_id) AS total_returns,
        SUM(CASE WHEN r.reason = 'defective' THEN 1 ELSE 0 END) AS defective_count,
        SUM(CASE WHEN r.reason = 'wrong_item' THEN 1 ELSE 0 END) AS wrong_item_count,
        SUM(CASE WHEN r.reason = 'not_as_described' THEN 1 ELSE 0 END) AS not_as_described_count,
        SUM(CASE WHEN r.reason IN ('defective', 'wrong_item', 'not_as_described') 
            THEN 1 ELSE 0 END) AS quality_related_returns
    FROM vendors v
    JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    JOIN products p ON vc.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    LEFT JOIN returns r ON o.order_id = r.order_id
        AND r.status IN ('approved', 'refunded')
    WHERE v.status = 'active'
    GROUP BY v.vendor_id, v.vendor_name, p.product_id, p.product_name, pc.category_name
),
vendor_quality_summary AS (
    SELECT 
        vendor_id,
        vendor_name,
        COUNT(DISTINCT product_id) AS products_count,
        SUM(units_sold) AS total_units_sold,
        ROUND(SUM(revenue), 2) AS total_revenue,
        -- Review metrics
        SUM(total_reviews) AS total_reviews,
        ROUND(AVG(avg_rating), 2) AS avg_vendor_rating,
        SUM(five_star_reviews) AS five_star_count,
        SUM(one_star_reviews + two_star_reviews) AS negative_reviews_count,
        -- Defect metrics
        SUM(total_returns) AS total_returns,
        SUM(quality_related_returns) AS quality_returns,
        SUM(defective_count) AS defective_returns,
        SUM(wrong_item_count) AS wrong_item_returns,
        SUM(not_as_described_count) AS not_as_described_returns
    FROM product_quality_metrics
    GROUP BY vendor_id, vendor_name
)
SELECT 
    vendor_id,
    vendor_name,
    products_count,
    total_units_sold,
    total_revenue,
    -- Rating analysis
    total_reviews,
    avg_vendor_rating,
    five_star_count,
    ROUND(five_star_count * 100.0 / NULLIF(total_reviews, 0), 2) AS five_star_pct,
    negative_reviews_count,
    ROUND(negative_reviews_count * 100.0 / NULLIF(total_reviews, 0), 2) AS negative_review_pct,
    -- Quality metrics
    total_returns,
    ROUND(total_returns * 100.0 / NULLIF(total_units_sold, 0), 2) AS overall_return_rate_pct,
    quality_returns,
    ROUND(quality_returns * 100.0 / NULLIF(total_returns, 0), 2) AS quality_related_return_pct,
    defective_returns,
    ROUND(defective_returns * 100.0 / NULLIF(total_units_sold, 0), 2) AS defect_rate_pct,
    wrong_item_returns,
    not_as_described_returns,
    -- Quality score (0-100)
    ROUND(
        LEAST(100,
            -- High rating (40%)
            (COALESCE(avg_vendor_rating, 3) / 5 * 100) * 0.40 +
            -- Low return rate (35%)
            (100 - LEAST(100, (total_returns * 100.0 / NULLIF(total_units_sold, 0)) * 10)) * 0.35 +
            -- Low defect rate (25%)
            (100 - LEAST(100, (defective_returns * 100.0 / NULLIF(total_units_sold, 0)) * 20)) * 0.25
        ),
        0
    ) AS quality_score,
    -- Quality classification
    CASE 
        WHEN avg_vendor_rating >= 4.5 
             AND total_returns * 100.0 / NULLIF(total_units_sold, 0) < 3 
             AND defective_returns * 100.0 / NULLIF(total_units_sold, 0) < 1 
            THEN 'Premium Quality'
        WHEN avg_vendor_rating >= 4.0 
             AND total_returns * 100.0 / NULLIF(total_units_sold, 0) < 5 
            THEN 'High Quality'
        WHEN avg_vendor_rating >= 3.5 
             AND total_returns * 100.0 / NULLIF(total_units_sold, 0) < 10 
            THEN 'Good Quality'
        WHEN avg_vendor_rating >= 3.0 
            THEN 'Acceptable Quality'
        WHEN defective_returns * 100.0 / NULLIF(total_units_sold, 0) > 5 
            THEN 'Quality Issues - High Defects'
        WHEN total_returns * 100.0 / NULLIF(total_units_sold, 0) > 15 
            THEN 'Quality Issues - High Returns'
        ELSE 'Poor Quality'
    END AS quality_classification,
    -- Action items
    CASE 
        WHEN defective_returns * 100.0 / NULLIF(total_units_sold, 0) > 5 
            THEN 'URGENT: Implement quality control process'
        WHEN avg_vendor_rating < 3.0 
            THEN 'Critical: Product quality review required'
        WHEN wrong_item_returns > 10 
            THEN 'Action: Improve order accuracy/packaging'
        WHEN not_as_described_returns > 10 
            THEN 'Action: Review product descriptions/images'
        WHEN total_returns * 100.0 / NULLIF(total_units_sold, 0) > 10 
            THEN 'Warning: Investigate return causes'
        WHEN avg_vendor_rating >= 4.5 
            THEN 'Excellence: Feature in marketing'
        ELSE 'Monitor: Continue tracking'
    END AS quality_action_plan
FROM vendor_quality_summary
ORDER BY quality_score DESC, total_revenue DESC;

-- ========================================
-- 4. VENDOR COST COMPETITIVENESS
-- Price comparison and cost analysis
-- ========================================

WITH vendor_pricing AS (
    SELECT 
        v.vendor_id,
        v.vendor_name,
        vc.product_id,
        p.product_name,
        pc.category_name,
        vc.cost_per_unit AS vendor_cost,
        p.price AS retail_price,
        vc.minimum_order_quantity AS moq,
        vc.lead_time_days,
        -- Market comparison
        (SELECT AVG(vc2.cost_per_unit)
         FROM vendor_contracts vc2
         WHERE vc2.product_id = vc.product_id
           AND vc2.status = 'active') AS avg_market_cost,
        (SELECT MIN(vc2.cost_per_unit)
         FROM vendor_contracts vc2
         WHERE vc2.product_id = vc.product_id
           AND vc2.status = 'active') AS lowest_market_cost,
        -- Sales performance
        SUM(oi.quantity) AS units_sold_12m,
        SUM(oi.subtotal) AS revenue_12m
    FROM vendors v
    JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    JOIN products p ON vc.product_id = p.product_id
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE v.status = 'active'
        AND vc.status = 'active'
    GROUP BY v.vendor_id, v.vendor_name, vc.product_id, p.product_name, 
             pc.category_name, vc.cost_per_unit, p.price, vc.minimum_order_quantity,
             vc.lead_time_days
),
vendor_cost_summary AS (
    SELECT 
        vendor_id,
        vendor_name,
        COUNT(DISTINCT product_id) AS products_supplied,
        ROUND(AVG(vendor_cost), 2) AS avg_cost_per_unit,
        ROUND(AVG(retail_price), 2) AS avg_retail_price,
        ROUND(AVG((retail_price - vendor_cost) / NULLIF(retail_price, 0) * 100), 2) AS avg_margin_pct,
        AVG(moq) AS avg_moq,
        ROUND(AVG(lead_time_days), 1) AS avg_lead_time,
        SUM(units_sold_12m) AS total_units_sold,
        ROUND(SUM(revenue_12m), 2) AS total_revenue,
        -- Cost competitiveness
        ROUND(AVG((vendor_cost - lowest_market_cost) / NULLIF(lowest_market_cost, 0) * 100), 2) AS avg_premium_vs_lowest_pct,
        ROUND(AVG((vendor_cost - avg_market_cost) / NULLIF(avg_market_cost, 0) * 100), 2) AS avg_vs_market_pct,
        SUM(CASE WHEN vendor_cost <= lowest_market_cost THEN 1 ELSE 0 END) AS products_with_best_price,
        SUM(CASE WHEN vendor_cost <= avg_market_cost THEN 1 ELSE 0 END) AS products_below_market_avg
    FROM vendor_pricing
    GROUP BY vendor_id, vendor_name
)
SELECT 
    vendor_id,
    vendor_name,
    products_supplied,
    avg_cost_per_unit,
    avg_retail_price,
    avg_margin_pct,
    ROUND(avg_moq, 0) AS avg_minimum_order_qty,
    avg_lead_time,
    total_units_sold,
    total_revenue,
    -- Competitiveness metrics
    avg_vs_market_pct,
    products_with_best_price,
    products_below_market_avg,
    ROUND(products_below_market_avg * 100.0 / NULLIF(products_supplied, 0), 2) AS competitive_pricing_pct,
    -- Cost competitiveness score
    ROUND(
        LEAST(100,
            -- Pricing (50%)
            (100 - LEAST(100, ABS(avg_vs_market_pct) * 2)) * 0.50 +
            -- Margin support (30%)
            (avg_margin_pct * 2) * 0.30 +
            -- Flexibility - low MOQ (20%)
            (100 - LEAST(100, avg_moq)) * 0.20
        ),
        0
    ) AS cost_competitiveness_score,
    -- Classification
    CASE 
        WHEN avg_vs_market_pct <= -10 THEN 'Highly Competitive'
        WHEN avg_vs_market_pct <= 0 THEN 'Competitive'
        WHEN avg_vs_market_pct <= 10 THEN 'Market Rate'
        WHEN avg_vs_market_pct <= 20 THEN 'Above Market'
        ELSE 'Premium Pricing'
    END AS pricing_tier,
    -- Recommendations
    CASE 
        WHEN avg_vs_market_pct > 20 AND total_revenue < 10000 
            THEN 'Renegotiate or source alternative - High cost, low volume'
        WHEN avg_vs_market_pct > 20 
            THEN 'Negotiate better pricing - Above market rate'
        WHEN avg_vs_market_pct <= -10 AND avg_margin_pct >= 40
        THEN 'Strategic partner - Excellent value'
        WHEN products_with_best_price >= products_supplied * 0.5 
            THEN 'Preferred vendor - Best pricing'
        WHEN avg_moq > 100 
            THEN 'Negotiate lower MOQ for flexibility'
        ELSE 'Maintain relationship - Fair pricing'
    END AS cost_recommendation
FROM vendor_cost_summary
ORDER BY cost_competitiveness_score DESC, total_revenue DESC;

-- ========================================
-- 5. VENDOR RELIABILITY & RISK ASSESSMENT
-- Comprehensive risk scoring
-- ========================================

WITH vendor_history AS (
    SELECT 
        v.vendor_id,
        v.vendor_name,
        v.status,
        DATEDIFF(CURDATE(), v.created_at) AS relationship_days,
        -- Contract history
        COUNT(DISTINCT vc.contract_id) AS total_contracts,
        SUM(CASE WHEN vc.status = 'active' THEN 1 ELSE 0 END) AS active_contracts,
        SUM(CASE WHEN vc.status = 'expired' THEN 1 ELSE 0 END) AS expired_contracts,
        SUM(CASE WHEN vc.status = 'terminated' THEN 1 ELSE 0 END) AS terminated_contracts,
        -- Product diversity
        COUNT(DISTINCT p.category_id) AS categories_supplied,
        COUNT(DISTINCT vc.product_id) AS products_supplied,
        -- Financial metrics
        SUM(oi.quantity * vc.cost_per_unit) AS total_purchase_value_12m,
        SUM(oi.subtotal) AS total_sales_value_12m,
        -- Performance indicators
        AVG(CASE WHEN rev.rating IS NOT NULL THEN rev.rating END) AS avg_product_rating,
        COUNT(DISTINCT CASE 
            WHEN r.reason IN ('defective', 'wrong_item', 'not_as_described') 
            THEN r.return_id 
        END) AS quality_issues,
        COUNT(DISTINCT o.order_id) AS order_count
    FROM vendors v
    LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    LEFT JOIN products p ON vc.product_id = p.product_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    LEFT JOIN returns r ON o.order_id = r.order_id
        AND r.status IN ('approved', 'refunded')
    WHERE v.status = 'active'
    GROUP BY v.vendor_id, v.vendor_name, v.status, v.created_at
),
risk_scoring AS (
    SELECT 
        vendor_id,
        vendor_name,
        ROUND(relationship_days / 365.0, 1) AS relationship_years,
        total_contracts,
        active_contracts,
        expired_contracts,
        terminated_contracts,
        categories_supplied,
        products_supplied,
        ROUND(total_purchase_value_12m, 2) AS annual_purchase_value,
        ROUND(total_sales_value_12m, 2) AS annual_sales_value,
        ROUND(avg_product_rating, 2) AS avg_rating,
        quality_issues,
        order_count,
        -- Risk factors
        CASE WHEN terminated_contracts > 0 THEN 1 ELSE 0 END AS has_terminated_contracts,
        CASE WHEN avg_product_rating < 3.0 THEN 1 ELSE 0 END AS low_quality_rating,
        CASE WHEN quality_issues > 20 THEN 1 ELSE 0 END AS high_quality_issues,
        CASE WHEN active_contracts = 0 THEN 1 ELSE 0 END AS no_active_contracts,
        CASE WHEN products_supplied <= 2 THEN 1 ELSE 0 END AS limited_product_range,
        -- Risk score components
        -- Relationship stability (25%)
        LEAST(100, (relationship_days / 365.0) * 20) * 0.25 AS stability_score,
        -- Contract health (20%)
        (active_contracts * 100.0 / NULLIF(total_contracts, 0)) * 0.20 AS contract_score,
        -- Quality (25%)
        (COALESCE(avg_product_rating, 3) / 5 * 100) * 0.25 AS quality_score,
        -- Volume/diversification (15%)
        LEAST(100, products_supplied * 10) * 0.15 AS diversification_score,
        -- Issue management (15%)
        (100 - LEAST(100, quality_issues * 5)) * 0.15 AS issue_score
    FROM vendor_history
)
SELECT 
    vendor_id,
    vendor_name,
    relationship_years,
    total_contracts,
    active_contracts,
    expired_contracts,
    terminated_contracts,
    categories_supplied,
    products_supplied,
    annual_purchase_value,
    annual_sales_value,
    avg_rating,
    quality_issues,
    order_count,
    -- Risk flags
    has_terminated_contracts,
    low_quality_rating,
    high_quality_issues,
    no_active_contracts,
    limited_product_range,
    (has_terminated_contracts + low_quality_rating + high_quality_issues + 
     no_active_contracts + limited_product_range) AS total_risk_flags,
    -- Overall reliability score (0-100)
    ROUND(
        stability_score + contract_score + quality_score + 
        diversification_score + issue_score,
        0
    ) AS reliability_score,
    -- Risk classification
    CASE 
        WHEN (has_terminated_contracts + low_quality_rating + high_quality_issues + 
              no_active_contracts + limited_product_range) >= 3 THEN 'High Risk'
        WHEN (has_terminated_contracts + low_quality_rating + high_quality_issues + 
              no_active_contracts + limited_product_range) >= 2 THEN 'Medium Risk'
        WHEN (has_terminated_contracts + low_quality_rating + high_quality_issues + 
              no_active_contracts + limited_product_range) = 1 THEN 'Low Risk'
        ELSE 'Minimal Risk'
    END AS risk_category,
    -- Strategic importance
    CASE 
        WHEN annual_purchase_value >= 50000 THEN 'Critical'
        WHEN annual_purchase_value >= 20000 THEN 'Important'
        WHEN annual_purchase_value >= 5000 THEN 'Standard'
        ELSE 'Low Volume'
    END AS strategic_importance,
    -- Risk management action
    CASE 
        WHEN (has_terminated_contracts + low_quality_rating + high_quality_issues + 
              no_active_contracts + limited_product_range) >= 3 
             AND annual_purchase_value >= 20000 
            THEN 'CRITICAL: Develop contingency plan, source alternatives'
        WHEN (has_terminated_contracts + low_quality_rating + high_quality_issues + 
              no_active_contracts + limited_product_range) >= 3 
            THEN 'HIGH: Performance improvement plan or phase out'
        WHEN has_terminated_contracts = 1 AND active_contracts = 0 
            THEN 'WARNING: Relationship at risk, re-engage or exit'
        WHEN high_quality_issues = 1 
            THEN 'ACTION: Quality audit and corrective action required'
        WHEN low_quality_rating = 1 
            THEN 'REVIEW: Address customer satisfaction issues'
        WHEN limited_product_range = 1 AND annual_purchase_value < 5000 
            THEN 'CONSIDER: Consolidate to larger vendors'
        ELSE 'MAINTAIN: Continue monitoring'
    END AS risk_mitigation_action
FROM risk_scoring
ORDER BY 
    CASE 
        WHEN (has_terminated_contracts + low_quality_rating + high_quality_issues + 
              no_active_contracts + limited_product_range) >= 3 THEN 1
        WHEN (has_terminated_contracts + low_quality_rating + high_quality_issues + 
              no_active_contracts + limited_product_range) >= 2 THEN 2
        ELSE 3
    END,
    annual_purchase_value DESC;

-- ========================================
-- 6. VENDOR COMPARISON MATRIX
-- Side-by-side comparison of key metrics
-- ========================================

WITH vendor_metrics_summary AS (
    SELECT 
        v.vendor_id,
        v.vendor_name,
        v.rating AS vendor_rating,
        COUNT(DISTINCT vc.product_id) AS products,
        COUNT(DISTINCT p.category_id) AS categories,
        -- Financial
        SUM(oi.subtotal) AS revenue_12m,
        SUM(oi.quantity * p.cost) AS cogs_12m,
        -- Quality
        AVG(rev.rating) AS avg_product_rating,
        COUNT(DISTINCT rev.review_id) AS review_count,
        -- Returns
        COUNT(DISTINCT r.return_id) AS return_count,
        SUM(oi.quantity) AS units_sold,
        -- Lead time
        AVG(vc.lead_time_days) AS avg_lead_time,
        AVG(vc.cost_per_unit) AS avg_unit_cost
    FROM vendors v
    LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    LEFT JOIN products p ON vc.product_id = p.product_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    LEFT JOIN returns r ON o.order_id = r.order_id
        AND r.status IN ('approved', 'refunded')
    WHERE v.status = 'active'
    GROUP BY v.vendor_id, v.vendor_name, v.rating
)
SELECT 
    vendor_name,
    ROUND(vendor_rating, 2) AS vendor_rating,
    products AS total_products,
    categories AS total_categories,
    -- Financial metrics
    ROUND(revenue_12m, 2) AS annual_revenue,
    ROUND((revenue_12m - cogs_12m) / NULLIF(revenue_12m, 0) * 100, 2) AS gross_margin_pct,
    -- Quality metrics
    ROUND(avg_product_rating, 2) AS avg_product_rating,
    review_count,
    -- Performance metrics
    ROUND(return_count * 100.0 / NULLIF(units_sold, 0), 2) AS return_rate_pct,
    ROUND(avg_lead_time, 1) AS avg_lead_time_days,
    -- Comparative rankings
    RANK() OVER (ORDER BY revenue_12m DESC) AS revenue_rank,
    RANK() OVER (ORDER BY avg_product_rating DESC) AS quality_rank,
    RANK() OVER (ORDER BY return_count * 100.0 / NULLIF(units_sold, 0)) AS returns_rank,
    RANK() OVER (ORDER BY avg_lead_time) AS lead_time_rank,
    -- Overall rank (lower is better)
    ROUND(
        (RANK() OVER (ORDER BY revenue_12m DESC) * 0.30 +
         RANK() OVER (ORDER BY avg_product_rating DESC) * 0.30 +
         RANK() OVER (ORDER BY return_count * 100.0 / NULLIF(units_sold, 0)) * 0.25 +
         RANK() OVER (ORDER BY avg_lead_time) * 0.15),
        2
    ) AS composite_rank_score
FROM vendor_metrics_summary
ORDER BY composite_rank_score;

-- ========================================
-- 7. VENDOR PERFORMANCE TRENDS
-- Month-over-month performance tracking
-- ========================================

WITH monthly_vendor_performance AS (
    SELECT 
        v.vendor_id,
        v.vendor_name,
        DATE_FORMAT(o.order_date, '%Y-%m') AS performance_month,
        COUNT(DISTINCT o.order_id) AS monthly_orders,
        SUM(oi.quantity) AS monthly_units,
        SUM(oi.subtotal) AS monthly_revenue,
        COUNT(DISTINCT r.return_id) AS monthly_returns,
        AVG(rev.rating) AS monthly_avg_rating
    FROM vendors v
    JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    JOIN products p ON vc.product_id = p.product_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN returns r ON o.order_id = r.order_id
        AND r.status IN ('approved', 'refunded')
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
        AND DATE_FORMAT(rev.created_at, '%Y-%m') = DATE_FORMAT(o.order_date, '%Y-%m')
    WHERE v.status = 'active'
    GROUP BY v.vendor_id, v.vendor_name, DATE_FORMAT(o.order_date, '%Y-%m')
),
performance_trends AS (
    SELECT 
        vendor_id,
        vendor_name,
        performance_month,
        monthly_orders,
        monthly_units,
        ROUND(monthly_revenue, 2) AS monthly_revenue,
        monthly_returns,
        ROUND(monthly_returns * 100.0 / NULLIF(monthly_units, 0), 2) AS return_rate_pct,
        ROUND(monthly_avg_rating, 2) AS avg_rating,
        -- Previous month comparison
        LAG(monthly_revenue) OVER (PARTITION BY vendor_id ORDER BY performance_month) AS prev_month_revenue,
        LAG(monthly_returns) OVER (PARTITION BY vendor_id ORDER BY performance_month) AS prev_month_returns,
        LAG(monthly_avg_rating) OVER (PARTITION BY vendor_id ORDER BY performance_month) AS prev_month_rating,
        -- 3-month moving average
        AVG(monthly_revenue) OVER (
            PARTITION BY vendor_id 
            ORDER BY performance_month 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS revenue_3mo_avg
    FROM monthly_vendor_performance
)
SELECT 
    vendor_name,
    performance_month,
    monthly_orders,
    monthly_units,
    monthly_revenue,
    ROUND(revenue_3mo_avg, 2) AS revenue_3mo_moving_avg,
    monthly_returns,
    return_rate_pct,
    avg_rating,
    -- Month-over-month changes
    ROUND((monthly_revenue - prev_month_revenue) / NULLIF(prev_month_revenue, 0) * 100, 2) AS revenue_mom_change_pct,
    ROUND((monthly_returns - prev_month_returns) / NULLIF(prev_month_returns, 0) * 100, 2) AS returns_mom_change_pct,
    ROUND(avg_rating - prev_month_rating, 2) AS rating_mom_change,
    -- Trend indicator
    CASE 
        WHEN monthly_revenue > prev_month_revenue * 1.1 AND monthly_returns < prev_month_returns 
            THEN 'Improving'
        WHEN monthly_revenue > prev_month_revenue AND avg_rating > prev_month_rating 
            THEN 'Positive'
        WHEN monthly_revenue < prev_month_revenue * 0.9 OR monthly_returns > prev_month_returns * 1.5 
            THEN 'Declining'
        WHEN monthly_revenue < prev_month_revenue 
            THEN 'Weakening'
        ELSE 'Stable'
    END AS trend_status
FROM performance_trends
WHERE performance_month >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 6 MONTH), '%Y-%m')
ORDER BY vendor_name, performance_month DESC;

-- ========================================
-- 8. TOP PERFORMERS & UNDERPERFORMERS
-- Best and worst vendors summary
-- ========================================

WITH vendor_overall_metrics AS (
    SELECT 
        v.vendor_id,
        v.vendor_name,
        SUM(oi.subtotal) AS total_revenue,
        AVG(rev.rating) AS avg_rating,
        COUNT(DISTINCT r.return_id) * 100.0 / NULLIF(SUM(oi.quantity), 0) AS return_rate,
        AVG(vc.lead_time_days) AS avg_lead_time,
        COUNT(DISTINCT vc.product_id) AS products_count,
        -- Composite score
        (
            (SUM(oi.subtotal) / 10000) * 0.25 +
            (COALESCE(AVG(rev.rating), 3) / 5 * 100) * 0.35 +
            (100 - LEAST(100, (COUNT(DISTINCT r.return_id) * 100.0 / NULLIF(SUM(oi.quantity), 0)) * 10)) * 0.25 +
            (100 - LEAST(100, AVG(vc.lead_time_days) * 2)) * 0.15
        ) AS performance_score
    FROM vendors v
    LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    LEFT JOIN products p ON vc.product_id = p.product_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    LEFT JOIN returns r ON o.order_id = r.order_id
        AND r.status IN ('approved', 'refunded')
    WHERE v.status = 'active'
    GROUP BY v.vendor_id, v.vendor_name
)
(
    -- Top 10 performers
    SELECT 
        'Top Performer' AS category,
        vendor_name,
        ROUND(total_revenue, 2) AS revenue,
        ROUND(avg_rating, 2) AS avg_rating,
        ROUND(return_rate, 2) AS return_rate_pct,
        ROUND(avg_lead_time, 1) AS lead_time_days,
        products_count,
        ROUND(performance_score, 0) AS score
    FROM vendor_overall_metrics
    ORDER BY performance_score DESC
    LIMIT 10
)
UNION ALL
(
    -- Bottom 10 performers
    SELECT 
        'Underperformer' AS category,
        vendor_name,
        ROUND(total_revenue, 2) AS revenue,
        ROUND(avg_rating, 2) AS avg_rating,
        ROUND(return_rate, 2) AS return_rate_pct,
        ROUND(avg_lead_time, 1) AS lead_time_days,
        products_count,
        ROUND(performance_score, 0) AS score
    FROM vendor_overall_metrics
    ORDER BY performance_score ASC
    LIMIT 10
)
ORDER BY category DESC, score DESC;

-- ========================================
-- End of Supplier Performance Analysis
-- ========================================