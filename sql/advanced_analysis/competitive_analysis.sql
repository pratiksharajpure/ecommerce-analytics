-- ========================================
-- COMPETITIVE ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Market Position, Competitive Pricing, Market Share
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. MARKET POSITION ANALYSIS
-- Our performance vs. market benchmarks
-- ========================================

WITH our_performance AS (
    SELECT 
        COUNT(DISTINCT c.customer_id) AS our_customers,
        COUNT(DISTINCT o.order_id) AS our_orders,
        SUM(o.total_amount) AS our_revenue,
        AVG(o.total_amount) AS our_avg_order_value,
        SUM(o.total_amount) / COUNT(DISTINCT c.customer_id) AS our_revenue_per_customer,
        COUNT(DISTINCT o.order_id) / COUNT(DISTINCT c.customer_id) AS our_orders_per_customer,
        COUNT(DISTINCT p.product_id) AS our_product_count,
        COUNT(DISTINCT pc.category_id) AS our_category_count,
        AVG(rev.rating) AS our_avg_rating,
        COUNT(DISTINCT rev.review_id) AS our_review_count
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    WHERE c.status = 'active'
),
-- Simulated competitor data (in reality, would come from market research/external sources)
market_benchmarks AS (
    SELECT 
        'Industry Average' AS benchmark_type,
        10000 AS market_customers,
        50000 AS market_orders,
        5000000 AS market_revenue,
        100 AS market_avg_order_value,
        500 AS market_revenue_per_customer,
        5.0 AS market_orders_per_customer,
        500 AS market_product_count,
        20 AS market_category_count,
        4.2 AS market_avg_rating,
        5000 AS market_review_count
    UNION ALL
    SELECT 
        'Top Quartile',
        15000, 80000, 8000000, 100, 533, 5.3, 600, 25, 4.5, 8000
    UNION ALL
    SELECT 
        'Market Leader',
        50000, 300000, 30000000, 100, 600, 6.0, 1000, 30, 4.6, 25000
)
SELECT 
    mb.benchmark_type,
    -- Our metrics
    op.our_customers,
    op.our_orders,
    ROUND(op.our_revenue, 2) AS our_revenue,
    ROUND(op.our_avg_order_value, 2) AS our_avg_order_value,
    ROUND(op.our_revenue_per_customer, 2) AS our_revenue_per_customer,
    ROUND(op.our_orders_per_customer, 2) AS our_orders_per_customer,
    op.our_product_count,
    ROUND(op.our_avg_rating, 2) AS our_avg_rating,
    -- Market benchmarks
    mb.market_customers,
    mb.market_revenue,
    mb.market_avg_order_value,
    mb.market_revenue_per_customer,
    mb.market_orders_per_customer,
    mb.market_product_count,
    mb.market_avg_rating,
    -- Competitive gaps
    ROUND((op.our_customers - mb.market_customers) * 100.0 / mb.market_customers, 2) AS customer_gap_pct,
    ROUND((op.our_revenue - mb.market_revenue) * 100.0 / mb.market_revenue, 2) AS revenue_gap_pct,
    ROUND((op.our_avg_order_value - mb.market_avg_order_value) * 100.0 / mb.market_avg_order_value, 2) AS aov_gap_pct,
    ROUND((op.our_orders_per_customer - mb.market_orders_per_customer) * 100.0 / mb.market_orders_per_customer, 2) AS frequency_gap_pct,
    ROUND(op.our_avg_rating - mb.market_avg_rating, 2) AS rating_gap,
    -- Competitive position
    CASE 
        WHEN op.our_revenue >= mb.market_revenue * 1.5 THEN 'Market Leader'
        WHEN op.our_revenue >= mb.market_revenue THEN 'Above Market'
        WHEN op.our_revenue >= mb.market_revenue * 0.75 THEN 'At Market'
        WHEN op.our_revenue >= mb.market_revenue * 0.50 THEN 'Below Market'
        ELSE 'Challenger'
    END AS market_position,
    -- Strategic priorities
    CASE 
        WHEN (op.our_customers - mb.market_customers) * 100.0 / mb.market_customers < -30 
            THEN 'Focus: Customer Acquisition'
        WHEN (op.our_orders_per_customer - mb.market_orders_per_customer) * 100.0 / mb.market_orders_per_customer < -20 
            THEN 'Focus: Customer Retention & Frequency'
        WHEN (op.our_avg_order_value - mb.market_avg_order_value) * 100.0 / mb.market_avg_order_value < -15 
            THEN 'Focus: Average Order Value'
        WHEN op.our_avg_rating < mb.market_avg_rating - 0.3 
            THEN 'Focus: Product Quality & Satisfaction'
        ELSE 'Focus: Market Share Growth'
    END AS strategic_priority
FROM our_performance op
CROSS JOIN market_benchmarks mb;

-- ========================================
-- 2. COMPETITIVE PRICING ANALYSIS
-- Price positioning vs. market
-- ========================================

WITH our_pricing AS (
    SELECT 
        pc.category_name,
        p.product_id,
        p.product_name,
        p.price AS our_price,
        p.cost AS our_cost,
        (p.price - p.cost) / p.price * 100 AS our_margin_pct,
        COUNT(DISTINCT o.order_id) AS units_sold,
        SUM(oi.subtotal) AS revenue_generated
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE p.status = 'active'
    GROUP BY pc.category_name, p.product_id, p.product_name, p.price, p.cost
),
-- Simulated competitor pricing (would come from price monitoring tools)
competitor_pricing AS (
    SELECT 
        category_name,
        AVG(our_price) * 1.10 AS competitor_avg_price,
        MIN(our_price) * 0.85 AS competitor_min_price,
        MAX(our_price) * 1.25 AS competitor_max_price,
        AVG(our_margin_pct) AS market_avg_margin
    FROM our_pricing
    GROUP BY category_name
)
SELECT 
    op.category_name,
    op.product_name,
    op.product_id,
    ROUND(op.our_price, 2) AS our_price,
    ROUND(op.our_cost, 2) AS our_cost,
    ROUND(op.our_margin_pct, 2) AS our_margin_pct,
    op.units_sold,
    ROUND(op.revenue_generated, 2) AS revenue_generated,
    -- Market pricing
    ROUND(cp.competitor_avg_price, 2) AS market_avg_price,
    ROUND(cp.competitor_min_price, 2) AS market_min_price,
    ROUND(cp.competitor_max_price, 2) AS market_max_price,
    -- Price positioning
    ROUND((op.our_price - cp.competitor_avg_price) / cp.competitor_avg_price * 100, 2) AS price_vs_market_pct,
    ROUND((op.our_price - cp.competitor_min_price) / (cp.competitor_max_price - cp.competitor_min_price) * 100, 2) AS price_percentile,
    -- Competitive position
    CASE 
        WHEN op.our_price <= cp.competitor_min_price * 1.05 THEN 'Price Leader (Lowest)'
        WHEN op.our_price <= cp.competitor_avg_price * 0.90 THEN 'Discount Player'
        WHEN op.our_price <= cp.competitor_avg_price * 1.10 THEN 'Market Rate'
        WHEN op.our_price <= cp.competitor_avg_price * 1.25 THEN 'Premium'
        ELSE 'Ultra-Premium'
    END AS price_positioning,
    -- Pricing recommendation
    CASE 
        WHEN op.our_price > cp.competitor_avg_price * 1.15 AND op.units_sold < 10 
            THEN 'Consider price reduction - Limited sales at premium'
        WHEN op.our_price < cp.competitor_avg_price * 0.85 AND op.our_margin_pct < 20 
            THEN 'Increase price - Margin too thin'
        WHEN op.our_price <= cp.competitor_min_price AND op.units_sold > 100 
            THEN 'Strong position - Consider small increase'
        WHEN op.our_price > cp.competitor_max_price 
            THEN 'Overpriced - Align with market'
        ELSE 'Maintain current pricing'
    END AS pricing_recommendation,
    -- Competitive advantage
    CASE 
        WHEN op.our_price < cp.competitor_avg_price AND op.our_margin_pct > cp.market_avg_margin 
            THEN 'Cost Advantage'
        WHEN op.our_price > cp.competitor_avg_price AND op.units_sold > 50 
            THEN 'Brand/Quality Premium'
        WHEN op.our_price < cp.competitor_avg_price * 0.90 
            THEN 'Price Competitive'
        ELSE 'At Market'
    END AS competitive_advantage
FROM our_pricing op
JOIN competitor_pricing cp ON op.category_name = cp.category_name
WHERE op.revenue_generated > 0
ORDER BY op.revenue_generated DESC;

-- ========================================
-- 3. MARKET SHARE ESTIMATION
-- Share of wallet and market penetration
-- ========================================

WITH category_performance AS (
    SELECT 
        pc.category_name,
        COUNT(DISTINCT c.customer_id) AS our_customers,
        COUNT(DISTINCT o.order_id) AS our_orders,
        SUM(o.total_amount) AS our_revenue,
        SUM(oi.quantity) AS our_units_sold,
        AVG(o.total_amount) AS our_avg_order_value
    FROM product_categories pc
    JOIN products p ON pc.category_id = p.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY pc.category_name
),
-- Estimated total market size by category (would come from market research)
market_size_estimates AS (
    SELECT 
        category_name,
        our_revenue * 5 AS estimated_total_market_revenue,
        our_customers * 10 AS estimated_total_market_customers,
        our_orders * 6 AS estimated_total_market_orders
    FROM category_performance
)
SELECT 
    cp.category_name,
    cp.our_customers,
    cp.our_orders,
    ROUND(cp.our_revenue, 2) AS our_revenue,
    cp.our_units_sold,
    ROUND(cp.our_avg_order_value, 2) AS our_avg_order_value,
    -- Market estimates
    mse.estimated_total_market_customers,
    ROUND(mse.estimated_total_market_revenue, 2) AS estimated_market_size,
    mse.estimated_total_market_orders,
    -- Market share calculations
    ROUND(cp.our_customers * 100.0 / mse.estimated_total_market_customers, 2) AS estimated_customer_share_pct,
    ROUND(cp.our_revenue * 100.0 / mse.estimated_total_market_revenue, 2) AS estimated_revenue_share_pct,
    ROUND(cp.our_orders * 100.0 / mse.estimated_total_market_orders, 2) AS estimated_order_share_pct,
    -- Opportunity sizing
    ROUND(mse.estimated_total_market_revenue - cp.our_revenue, 2) AS addressable_revenue_opportunity,
    mse.estimated_total_market_customers - cp.our_customers AS addressable_customers,
    -- Market position
    CASE 
        WHEN cp.our_revenue * 100.0 / mse.estimated_total_market_revenue >= 30 THEN 'Market Leader (30%+)'
        WHEN cp.our_revenue * 100.0 / mse.estimated_total_market_revenue >= 15 THEN 'Major Player (15-30%)'
        WHEN cp.our_revenue * 100.0 / mse.estimated_total_market_revenue >= 5 THEN 'Established Player (5-15%)'
        WHEN cp.our_revenue * 100.0 / mse.estimated_total_market_revenue >= 2 THEN 'Niche Player (2-5%)'
        ELSE 'Emerging (<2%)'
    END AS market_position,
    -- Growth strategy
    CASE 
        WHEN cp.our_revenue * 100.0 / mse.estimated_total_market_revenue < 5 
            THEN 'Aggressive expansion - Large untapped market'
        WHEN cp.our_revenue * 100.0 / mse.estimated_total_market_revenue < 15 
            THEN 'Growth focus - Capture market share'
        WHEN cp.our_revenue * 100.0 / mse.estimated_total_market_revenue < 30 
            THEN 'Competitive defense - Protect position'
        ELSE 'Market leadership - Innovate and expand'
    END AS growth_strategy
FROM category_performance cp
JOIN market_size_estimates mse ON cp.category_name = mse.category_name
ORDER BY cp.our_revenue DESC;

-- ========================================
-- 4. COMPETITIVE PRODUCT COMPARISON
-- Feature and quality comparison
-- ========================================

WITH our_product_metrics AS (
    SELECT 
        pc.category_name,
        COUNT(DISTINCT p.product_id) AS our_product_count,
        AVG(p.price) AS our_avg_price,
        AVG(rev.rating) AS our_avg_rating,
        COUNT(DISTINCT rev.review_id) AS our_review_count,
        SUM(CASE WHEN rev.rating >= 4 THEN 1 ELSE 0 END) * 100.0 / 
            NULLIF(COUNT(DISTINCT rev.review_id), 0) AS our_positive_review_pct,
        COUNT(DISTINCT r.return_id) * 100.0 / 
            NULLIF(COUNT(DISTINCT o.order_id), 0) AS our_return_rate_pct,
        AVG(p.stock_quantity) AS our_avg_stock_level
    FROM product_categories pc
    JOIN products p ON pc.category_id = pc.category_id
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE p.status = 'active'
    GROUP BY pc.category_name
),
-- Market benchmarks (from competitive intelligence)
market_benchmarks AS (
    SELECT 
        category_name,
        our_product_count * 2 AS market_avg_product_count,
        our_avg_price * 1.05 AS market_avg_price,
        4.3 AS market_avg_rating,
        our_review_count * 1.5 AS market_avg_reviews,
        85.0 AS market_positive_review_pct,
        8.0 AS market_return_rate_pct
    FROM our_product_metrics
)
SELECT 
    opm.category_name,
    opm.our_product_count,
    ROUND(opm.our_avg_price, 2) AS our_avg_price,
    ROUND(opm.our_avg_rating, 2) AS our_avg_rating,
    opm.our_review_count,
    ROUND(opm.our_positive_review_pct, 2) AS our_positive_review_pct,
    ROUND(opm.our_return_rate_pct, 2) AS our_return_rate_pct,
    -- Market benchmarks
    mb.market_avg_product_count,
    ROUND(mb.market_avg_price, 2) AS market_avg_price,
    mb.market_avg_rating,
    ROUND(mb.market_avg_reviews, 0) AS market_avg_reviews,
    mb.market_positive_review_pct,
    mb.market_return_rate_pct,
    -- Competitive gaps
    opm.our_product_count - mb.market_avg_product_count AS product_range_gap,
    ROUND(opm.our_avg_rating - mb.market_avg_rating, 2) AS rating_gap,
    ROUND(opm.our_return_rate_pct - mb.market_return_rate_pct, 2) AS return_rate_gap,
    -- Quality position
    CASE 
        WHEN opm.our_avg_rating >= 4.5 AND opm.our_return_rate_pct < 5 THEN 'Premium Quality Leader'
        WHEN opm.our_avg_rating >= mb.market_avg_rating AND opm.our_return_rate_pct <= mb.market_return_rate_pct THEN 'Above Market Quality'
        WHEN opm.our_avg_rating >= mb.market_avg_rating - 0.2 THEN 'Market Quality'
        WHEN opm.our_return_rate_pct > mb.market_return_rate_pct * 1.5 THEN 'Quality Issues - High Returns'
        ELSE 'Below Market Quality'
    END AS quality_position,
    -- Strategic recommendations
    CASE 
        WHEN opm.our_product_count < mb.market_avg_product_count * 0.7 
            THEN 'Expand product range'
        WHEN opm.our_avg_rating < mb.market_avg_rating - 0.3 
            THEN 'Focus on quality improvement'
        WHEN opm.our_return_rate_pct > mb.market_return_rate_pct * 1.5 
            THEN 'Urgent: Address quality/description issues'
        WHEN opm.our_review_count < mb.market_avg_reviews * 0.5 
            THEN 'Increase review collection'
        WHEN opm.our_avg_rating >= 4.5 
            THEN 'Leverage quality in marketing'
        ELSE 'Maintain current standards'
    END AS quality_action
FROM our_product_metrics opm
JOIN market_benchmarks mb ON opm.category_name = mb.category_name
ORDER BY opm.our_avg_rating DESC;

-- ========================================
-- 5. CUSTOMER ACQUISITION COMPETITIVENESS
-- How we compare in attracting customers
-- ========================================

WITH customer_acquisition AS (
    SELECT 
        DATE_FORMAT(c.created_at, '%Y-%m') AS cohort_month,
        COUNT(DISTINCT c.customer_id) AS new_customers,
        SUM(o.total_amount) AS first_month_revenue,
        AVG(o.total_amount) AS avg_first_order_value,
        COUNT(DISTINCT o.order_id) AS first_month_orders,
        -- Calculate activation rate (made first purchase within 30 days)
        COUNT(DISTINCT CASE 
            WHEN DATEDIFF(o.order_date, c.created_at) <= 30 
            THEN c.customer_id 
        END) * 100.0 / COUNT(DISTINCT c.customer_id) AS activation_rate_pct
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND DATEDIFF(o.order_date, c.created_at) <= 30
    WHERE c.created_at >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY DATE_FORMAT(c.created_at, '%Y-%m')
),
market_benchmarks AS (
    SELECT 
        65.0 AS market_activation_rate,
        85.0 AS market_avg_first_order_value,
        1.8 AS market_first_month_frequency
    FROM DUAL
)
SELECT 
    ca.cohort_month,
    ca.new_customers,
    ROUND(ca.first_month_revenue, 2) AS first_month_revenue,
    ROUND(ca.avg_first_order_value, 2) AS avg_first_order_value,
    ROUND(ca.first_month_orders * 1.0 / ca.new_customers, 2) AS first_month_frequency,
    ROUND(ca.activation_rate_pct, 2) AS activation_rate_pct,
    -- Market comparison
    mb.market_activation_rate,
    mb.market_avg_first_order_value,
    mb.market_first_month_frequency,
    -- Gaps
    ROUND(ca.activation_rate_pct - mb.market_activation_rate, 2) AS activation_gap_pct,
    ROUND(ca.avg_first_order_value - mb.market_avg_first_order_value, 2) AS aov_gap,
    -- Performance vs market
    CASE 
        WHEN ca.activation_rate_pct >= mb.market_activation_rate * 1.1 THEN 'Above Market'
        WHEN ca.activation_rate_pct >= mb.market_activation_rate * 0.9 THEN 'At Market'
        ELSE 'Below Market'
    END AS activation_performance,
    -- Growth trend
    ROUND(
        (ca.new_customers - LAG(ca.new_customers) OVER (ORDER BY ca.cohort_month)) * 100.0 /
        NULLIF(LAG(ca.new_customers) OVER (ORDER BY ca.cohort_month), 0),
        2
    ) AS mom_customer_growth_pct,
    -- Recommendation
    CASE 
        WHEN ca.activation_rate_pct < mb.market_activation_rate * 0.8 
            THEN 'Critical: Improve onboarding and first purchase experience'
        WHEN ca.avg_first_order_value < mb.market_avg_first_order_value * 0.85 
            THEN 'Focus: Increase initial basket size'
        WHEN ca.activation_rate_pct >= mb.market_activation_rate * 1.1 
            THEN 'Strength: Scale acquisition efforts'
        ELSE 'Maintain current acquisition strategy'
    END AS acquisition_recommendation
FROM customer_acquisition ca
CROSS JOIN market_benchmarks mb
ORDER BY ca.cohort_month DESC;

-- ========================================
-- 6. COMPETITIVE DASHBOARD SUMMARY
-- Executive view of competitive position
-- ========================================

WITH our_metrics AS (
    SELECT 
        COUNT(DISTINCT c.customer_id) AS total_customers,
        SUM(o.total_amount) AS total_revenue,
        COUNT(DISTINCT o.order_id) AS total_orders,
        AVG(o.total_amount) AS avg_order_value,
        AVG(rev.rating) AS avg_rating,
        COUNT(DISTINCT p.product_id) AS product_count,
        SUM(o.total_amount) / COUNT(DISTINCT c.customer_id) AS revenue_per_customer
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    WHERE c.status = 'active'
),
market_position AS (
    SELECT 
        'Our Company' AS company,
        total_customers AS customers,
        total_revenue AS revenue,
        total_orders AS orders,
        avg_order_value AS aov,
        avg_rating AS rating,
        product_count AS products,
        revenue_per_customer AS cltv
    FROM our_metrics
    UNION ALL
    SELECT 
        'Market Average',
        total_customers * 2,
        total_revenue * 2,
        total_orders * 2,
        avg_order_value * 1.0,
        4.2,
        product_count * 1.5,
        revenue_per_customer * 0.95
    FROM our_metrics
    UNION ALL
    SELECT 
        'Market Leader',
        total_customers * 5,
        total_revenue * 6,
        total_orders * 5.5,
        avg_order_value * 1.1,
        4.6,
        product_count * 2,
        revenue_per_customer * 1.2
    FROM our_metrics
)
SELECT 
    company,
    customers,
    ROUND(revenue, 2) AS annual_revenue,
    orders,
    ROUND(aov, 2) AS avg_order_value,
    ROUND(rating, 2) AS avg_rating,
    products AS product_count,
    ROUND(cltv, 2) AS customer_ltv,
    -- Market share estimates
    ROUND(revenue * 100.0 / (SELECT SUM(revenue) FROM market_position), 2) AS estimated_market_share_pct,
    -- Ranking
    RANK() OVER (ORDER BY revenue DESC) AS revenue_rank,
    RANK() OVER (ORDER BY customers DESC) AS customer_rank,
    RANK() OVER (ORDER BY rating DESC) AS quality_rank,
    -- Competitive status
    CASE 
        WHEN company = 'Our Company' THEN
            CASE 
                WHEN revenue >= (SELECT revenue FROM market_position WHERE company = 'Market Leader') * 0.8 
                    THEN '🥇 Near Leader'
                WHEN revenue >= (SELECT revenue FROM market_position WHERE company = 'Market Average') * 1.2 
                    THEN ''TRENDING_UP' Above Market'
                WHEN revenue >= (SELECT revenue FROM market_position WHERE company = 'Market Average') * 0.8 
                    THEN '⚖️ At Market'
                ELSE ''CHART' Below Market'
            END
        ELSE ''
    END AS competitive_status
FROM market_position
ORDER BY revenue DESC;

-- ========================================
-- 7. SWOT ANALYSIS DATA
-- Strengths, Weaknesses, Opportunities, Threats
-- ========================================

WITH swot_metrics AS (
    SELECT 
        -- Strengths indicators
        AVG(rev.rating) AS avg_rating,
        COUNT(DISTINCT rev.review_id) AS review_volume,
        AVG(o.total_amount) AS avg_order_value,
        SUM(o.total_amount) / COUNT(DISTINCT c.customer_id) AS customer_ltv,
        -- Weaknesses indicators
        COUNT(DISTINCT r.return_id) * 100.0 / NULLIF(COUNT(DISTINCT o.order_id), 0) AS return_rate,
        DATEDIFF(AVG(o.updated_at), AVG(o.order_date)) AS avg_delivery_days,
        COUNT(DISTINCT p.product_id) AS product_range,
        -- Opportunities indicators
        COUNT(DISTINCT c.customer_id) AS current_customers,
        (SELECT COUNT(*) FROM customers WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)) AS new_customers_90d,
        -- Threats indicators
        STDDEV(o.total_amount) / NULLIF(AVG(o.total_amount), 0) AS revenue_volatility
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN reviews rev ON c.customer_id = rev.customer_id
        AND rev.status = 'approved'
    LEFT JOIN returns r ON o.order_id = r.order_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    WHERE c.status = 'active'
)
SELECT 
    'STRENGTHS' AS category,
    CASE 
        WHEN avg_rating >= 4.5 THEN '✓ Excellent product ratings (4.5+)'
        WHEN avg_rating >= 4.0 THEN '✓ Good product ratings (4.0+)'
        ELSE '✗ Ratings need improvement'
    END AS factor_1,
    CASE 
        WHEN customer_ltv >= 500 THEN '✓ High customer lifetime value'
        WHEN customer_ltv >= 300 THEN '✓ Solid customer value'
        ELSE '○ Opportunity to increase LTV'
    END AS factor_2,
    CASE 
        WHEN review_volume >= 1000 THEN '✓ Strong social proof'
        WHEN review_volume >= 500 THEN '✓ Good review volume'
        ELSE '○ Build more social proof'
    END AS factor_3
FROM swot_metrics

UNION ALL

SELECT 
    'WEAKNESSES' AS category,
    CASE 
        WHEN return_rate > 15 THEN '✗ High return rate (>15%)'
        WHEN return_rate > 10 THEN '○ Elevated return rate (10-15%)'
        ELSE '✓ Acceptable return rate (<10%)'
    END AS factor_1,
    CASE 
        WHEN avg_delivery_days > 7 THEN '✗ Slow delivery (>7 days)'
        WHEN avg_delivery_days > 5 THEN '○ Average delivery speed'
        ELSE '✓ Fast delivery (<5 days)'
    END AS factor_2,
    CASE 
        WHEN product_range < 100 THEN '○ Limited product range'
        WHEN product_range < 500 THEN '✓ Moderate product range'
        ELSE '✓ Extensive product range'
    END AS factor_3
FROM swot_metrics

UNION ALL

SELECT 
    'OPPORTUNITIES' AS category,
    '✓ Market expansion potential' AS factor_1,
    CASE 
        WHEN new_customers_90d > current_customers * 0.05 THEN '✓ Strong customer acquisition momentum'
        ELSE '○ Opportunity to accelerate acquisition'
    END AS factor_2,
    '✓ Digital marketing channel growth' AS factor_3
FROM swot_metrics

UNION ALL

SELECT 
    'THREATS' AS category,
    '○ Increasing market competition' AS factor_1,
    '○ Price pressure from competitors' AS factor_2,
    CASE 
        WHEN revenue_volatility > 0.5 THEN '✗ High revenue volatility'
        WHEN revenue_volatility > 0.3 THEN '○ Moderate volatility'
        ELSE '✓ Stable revenue stream'
    END AS factor_3
FROM swot_metrics;

-- ========================================
-- 8. COMPETITIVE INTELLIGENCE SUMMARY
-- Executive dashboard with key competitive metrics
-- ========================================

WITH our_performance_summary AS (
    SELECT 
        COUNT(DISTINCT c.customer_id) AS total_customers,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        AVG(p.price) AS avg_product_price,
        AVG(rev.rating) AS avg_rating,
        COUNT(DISTINCT p.product_id) AS product_count
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN reviews rev ON p.product_id = rev.product_id
        AND rev.status = 'approved'
    WHERE c.status = 'active'
),
market_estimates AS (
    SELECT 
        ops.total_customers * 5 AS estimated_market_customers,
        ops.total_revenue * 5 AS estimated_market_revenue,
        ops.avg_product_price * 1.05 AS market_avg_price,
        4.2 AS market_avg_rating
    FROM our_performance_summary ops
),
competitive_metrics AS (
    SELECT 
        'Market Share' AS metric_name,
        ROUND(ops.total_revenue * 100.0 / me.estimated_market_revenue, 2) AS our_metric,
        20.0 AS market_benchmark,
        '%' AS unit_type,
        'Revenue-based market share estimate' AS description
    FROM our_performance_summary ops
    CROSS JOIN market_estimates me
    
    UNION ALL
    
    SELECT 
        'Customer Share',
        ROUND(ops.total_customers * 100.0 / me.estimated_market_customers, 2),
        20.0,
        '%',
        'Customer base market penetration'
    FROM our_performance_summary ops
    CROSS JOIN market_estimates me
    
    UNION ALL
    
    SELECT 
        'Price Position',
        ROUND((ops.avg_product_price - me.market_avg_price) / me.market_avg_price * 100, 2),
        0.0,
        '%',
        'Average price vs market (0 = at market)'
    FROM our_performance_summary ops
    CROSS JOIN market_estimates me
    
    UNION ALL
    
    SELECT 
        'Quality Rating',
        ROUND(ops.avg_rating, 2),
        me.market_avg_rating,
        '/5',
        'Customer satisfaction rating'
    FROM our_performance_summary ops
    CROSS JOIN market_estimates me
    
    UNION ALL
    
    SELECT 
        'Product Range',
        ops.product_count,
        ops.product_count * 1.5,
        'SKUs',
        'Total product offerings'
    FROM our_performance_summary ops
    
    UNION ALL
    
    SELECT 
        'Avg Order Value',
        ROUND(ops.avg_order_value, 2),
        100.0,
        ',
        'Average transaction size'
    FROM our_performance_summary ops
)
SELECT 
    metric_name,
    our_metric AS our_value,
    market_benchmark AS benchmark_value,
    unit_type,
    ROUND((our_metric - market_benchmark) / NULLIF(market_benchmark, 0) * 100, 2) AS variance_pct,
    CASE 
        WHEN metric_name = 'Market Share' AND our_metric >= 15 THEN ''GREEN' Strong'
        WHEN metric_name = 'Market Share' AND our_metric >= 10 THEN ''YELLOW' Growing'
        WHEN metric_name = 'Market Share' AND our_metric >= 5 THEN ''YELLOW' Established'
        WHEN metric_name = 'Market Share' THEN ''RED' Emerging'
        WHEN metric_name = 'Price Position' AND ABS(our_metric) <= 5 THEN ''GREEN' Competitive'
        WHEN metric_name = 'Price Position' AND our_metric > 5 THEN ''YELLOW' Premium'
        WHEN metric_name = 'Price Position' THEN ''YELLOW' Discount'
        WHEN metric_name = 'Quality Rating' AND our_metric >= 4.5 THEN ''GREEN' Excellent'
        WHEN metric_name = 'Quality Rating' AND our_metric >= 4.0 THEN ''GREEN' Good'
        WHEN metric_name = 'Quality Rating' AND our_metric >= 3.5 THEN ''YELLOW' Average'
        WHEN metric_name = 'Quality Rating' THEN ''RED' Below Average'
        WHEN metric_name = 'Avg Order Value' AND our_metric >= market_benchmark THEN ''GREEN' Above Market'
        WHEN metric_name = 'Avg Order Value' THEN ''YELLOW' Below Market'
        WHEN our_metric >= market_benchmark THEN ''GREEN' Leading'
        ELSE ''YELLOW' Following'
    END AS status,
    description
FROM competitive_metrics
ORDER BY FIELD(metric_name, 'Market Share', 'Customer Share', 'Price Position', 'Quality Rating', 'Product Range', 'Avg Order Value');

-- ========================================
-- End of Competitive Analysis
-- ========================================