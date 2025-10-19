-- ========================================
-- CUSTOMER LIFETIME VALUE (CLV) ANALYSIS
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. HISTORICAL CLV CALCULATION
-- Actual customer lifetime value to date
-- ========================================

WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        c.status,
        c.created_at AS signup_date,
        DATEDIFF(CURDATE(), c.created_at) AS customer_age_days,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        MIN(o.order_date) AS first_order_date,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) AS purchase_span_days,
        SUM(o.total_amount - o.shipping_cost - o.tax_amount) AS net_revenue
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.status, c.created_at
)
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    signup_date,
    customer_age_days,
    ROUND(customer_age_days / 365.25, 1) AS customer_age_years,
    total_orders,
    ROUND(total_revenue, 2) AS historical_clv,
    ROUND(net_revenue, 2) AS net_clv,
    ROUND(avg_order_value, 2) AS avg_order_value,
    first_order_date,
    last_order_date,
    purchase_span_days,
    CASE 
        WHEN total_orders = 0 THEN 0
        WHEN purchase_span_days = 0 THEN total_orders
        ELSE ROUND(total_orders * 365.25 / purchase_span_days, 2)
    END AS purchase_frequency_per_year,
    DATEDIFF(CURDATE(), last_order_date) AS days_since_last_order
FROM customer_metrics
WHERE total_orders > 0
ORDER BY historical_clv DESC;

-- ========================================
-- 2. CUSTOMER VALUE SEGMENTATION (RFM MODEL)
-- Recency, Frequency, Monetary segmentation
-- ========================================

WITH customer_rfm AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS recency_days,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(o.total_amount) AS monetary_value,
        AVG(o.total_amount) AS avg_order_value
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email
),
rfm_scores AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        recency_days,
        frequency,
        ROUND(monetary_value, 2) AS monetary_value,
        ROUND(avg_order_value, 2) AS avg_order_value,
        -- RFM Scoring (1-5 scale, 5 being best)
        CASE 
            WHEN recency_days <= 30 THEN 5
            WHEN recency_days <= 60 THEN 4
            WHEN recency_days <= 90 THEN 3
            WHEN recency_days <= 180 THEN 2
            ELSE 1
        END AS recency_score,
        CASE 
            WHEN frequency >= 20 THEN 5
            WHEN frequency >= 10 THEN 4
            WHEN frequency >= 5 THEN 3
            WHEN frequency >= 2 THEN 2
            ELSE 1
        END AS frequency_score,
        CASE 
            WHEN monetary_value >= 5000 THEN 5
            WHEN monetary_value >= 2000 THEN 4
            WHEN monetary_value >= 1000 THEN 3
            WHEN monetary_value >= 500 THEN 2
            ELSE 1
        END AS monetary_score
    FROM customer_rfm
)
SELECT 
    customer_id,
    customer_name,
    email,
    recency_days,
    frequency,
    monetary_value,
    avg_order_value,
    recency_score,
    frequency_score,
    monetary_score,
    (recency_score + frequency_score + monetary_score) AS total_rfm_score,
    CASE 
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
        WHEN recency_score >= 4 AND frequency_score <= 2 THEN 'New Customers'
        WHEN recency_score <= 2 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'At Risk VIP'
        WHEN recency_score <= 2 AND frequency_score >= 2 THEN 'At Risk'
        WHEN recency_score >= 3 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'Promising'
        WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost Customers'
        ELSE 'Potential Loyalists'
    END AS customer_segment
FROM rfm_scores
ORDER BY total_rfm_score DESC, monetary_value DESC;

-- ========================================
-- 3. CUSTOMER VALUE TIERS
-- Simple high/medium/low value classification
-- ========================================

WITH customer_revenue AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        c.created_at AS signup_date,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        MAX(o.order_date) AS last_order_date
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.created_at
),
revenue_percentiles AS (
    SELECT 
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_revenue) AS p75,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_revenue) AS p25
    FROM customer_revenue
)
SELECT 
    cr.customer_id,
    cr.customer_name,
    cr.email,
    cr.signup_date,
    cr.total_orders,
    ROUND(cr.total_revenue, 2) AS lifetime_value,
    ROUND(cr.avg_order_value, 2) AS avg_order_value,
    cr.last_order_date,
    DATEDIFF(CURDATE(), cr.last_order_date) AS days_since_last_order,
    CASE 
        WHEN cr.total_revenue >= rp.p75 THEN 'High Value'
        WHEN cr.total_revenue >= rp.p25 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS value_tier,
    CASE 
        WHEN DATEDIFF(CURDATE(), cr.last_order_date) <= 30 THEN 'Active'
        WHEN DATEDIFF(CURDATE(), cr.last_order_date) <= 90 THEN 'Cooling'
        WHEN DATEDIFF(CURDATE(), cr.last_order_date) <= 180 THEN 'At Risk'
        ELSE 'Dormant'
    END AS activity_status
FROM customer_revenue cr
CROSS JOIN revenue_percentiles rp
ORDER BY cr.total_revenue DESC;

-- ========================================
-- 4. PREDICTED LIFETIME VALUE (SIMPLE MODEL)
-- Based on early behavior patterns
-- ========================================

WITH customer_early_behavior AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        c.created_at AS signup_date,
        DATEDIFF(CURDATE(), c.created_at) AS customer_age_days,
        -- First 90 days metrics
        COUNT(DISTINCT CASE 
            WHEN DATEDIFF(o.order_date, c.created_at) <= 90 THEN o.order_id 
        END) AS orders_first_90_days,
        SUM(CASE 
            WHEN DATEDIFF(o.order_date, c.created_at) <= 90 THEN o.total_amount 
            ELSE 0 
        END) AS revenue_first_90_days,
        -- Total metrics
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        MAX(o.order_date) AS last_order_date
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.created_at
),
prediction_model AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        signup_date,
        customer_age_days,
        orders_first_90_days,
        ROUND(revenue_first_90_days, 2) AS revenue_first_90_days,
        total_orders,
        ROUND(total_revenue, 2) AS actual_ltv,
        ROUND(avg_order_value, 2) AS avg_order_value,
        last_order_date,
        -- Simple prediction: extrapolate first 90 days to 3 years
        CASE 
            WHEN customer_age_days >= 90 THEN
                ROUND(revenue_first_90_days * (1095.0 / 90.0), 2)  -- 3 years = 1095 days
            ELSE 
                ROUND(revenue_first_90_days * (1095.0 / GREATEST(customer_age_days, 1)), 2)
        END AS predicted_3yr_ltv,
        -- Confidence based on data maturity
        CASE 
            WHEN customer_age_days >= 365 THEN 'High'
            WHEN customer_age_days >= 180 THEN 'Medium'
            WHEN customer_age_days >= 90 THEN 'Low'
            ELSE 'Very Low'
        END AS prediction_confidence
    FROM customer_early_behavior
    WHERE orders_first_90_days > 0
)
SELECT 
    customer_id,
    customer_name,
    email,
    signup_date,
    customer_age_days,
    ROUND(customer_age_days / 365.25, 1) AS customer_age_years,
    orders_first_90_days,
    revenue_first_90_days,
    total_orders,
    actual_ltv,
    predicted_3yr_ltv,
    prediction_confidence,
    CASE 
        WHEN customer_age_days >= 1095 THEN 
            ROUND((actual_ltv - predicted_3yr_ltv) / NULLIF(predicted_3yr_ltv, 0) * 100, 2)
        ELSE NULL
    END AS prediction_accuracy_pct,
    CASE 
        WHEN predicted_3yr_ltv >= 3000 THEN 'VIP Potential'
        WHEN predicted_3yr_ltv >= 1500 THEN 'High Potential'
        WHEN predicted_3yr_ltv >= 750 THEN 'Medium Potential'
        ELSE 'Standard'
    END AS predicted_value_tier
FROM prediction_model
ORDER BY predicted_3yr_ltv DESC;

-- ========================================
-- 5. COHORT-BASED CLV BENCHMARKING
-- Compare individual CLV against cohort averages
-- ========================================

WITH customer_cohorts AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        DATE_FORMAT(c.created_at, '%Y-%m') AS cohort_month,
        c.created_at AS signup_date,
        SUM(o.total_amount) AS customer_ltv,
        COUNT(DISTINCT o.order_id) AS total_orders
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, cohort_month, c.created_at
),
cohort_averages AS (
    SELECT 
        cohort_month,
        AVG(customer_ltv) AS avg_cohort_ltv,
        AVG(total_orders) AS avg_cohort_orders,
        COUNT(DISTINCT customer_id) AS cohort_size
    FROM customer_cohorts
    GROUP BY cohort_month
)
SELECT 
    cc.customer_id,
    cc.customer_name,
    cc.cohort_month,
    cc.signup_date,
    ROUND(cc.customer_ltv, 2) AS customer_ltv,
    cc.total_orders,
    ca.cohort_size,
    ROUND(ca.avg_cohort_ltv, 2) AS avg_cohort_ltv,
    ROUND(ca.avg_cohort_orders, 1) AS avg_cohort_orders,
    ROUND((cc.customer_ltv - ca.avg_cohort_ltv) / NULLIF(ca.avg_cohort_ltv, 0) * 100, 2) AS ltv_vs_cohort_pct,
    CASE 
        WHEN cc.customer_ltv >= ca.avg_cohort_ltv * 1.5 THEN 'Top Performer'
        WHEN cc.customer_ltv >= ca.avg_cohort_ltv THEN 'Above Average'
        WHEN cc.customer_ltv >= ca.avg_cohort_ltv * 0.5 THEN 'Below Average'
        ELSE 'Bottom Performer'
    END AS cohort_performance
FROM customer_cohorts cc
JOIN cohort_averages ca ON cc.cohort_month = ca.cohort_month
WHERE cc.customer_ltv > 0
ORDER BY cc.cohort_month DESC, cc.customer_ltv DESC;

-- ========================================
-- 6. CLV WITH PRODUCT PROFITABILITY
-- CLV adjusted for product margins
-- ========================================

WITH customer_profitability AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        SUM(oi.quantity * p.cost) AS total_cogs,
        SUM(o.shipping_cost) AS total_shipping_cost,
        SUM(o.total_amount - (oi.quantity * p.cost) - o.shipping_cost) AS gross_profit
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email
)
SELECT 
    customer_id,
    customer_name,
    email,
    total_orders,
    ROUND(total_revenue, 2) AS lifetime_revenue,
    ROUND(total_cogs, 2) AS lifetime_cogs,
    ROUND(total_shipping_cost, 2) AS lifetime_shipping,
    ROUND(gross_profit, 2) AS lifetime_gross_profit,
    ROUND((gross_profit / NULLIF(total_revenue, 0)) * 100, 2) AS profit_margin_pct,
    ROUND(gross_profit / total_orders, 2) AS avg_profit_per_order,
    CASE 
        WHEN gross_profit >= 1000 THEN 'Highly Profitable'
        WHEN gross_profit >= 500 THEN 'Profitable'
        WHEN gross_profit >= 100 THEN 'Moderately Profitable'
        WHEN gross_profit > 0 THEN 'Low Profit'
        ELSE 'Unprofitable'
    END AS profitability_tier
FROM customer_profitability
ORDER BY lifetime_gross_profit DESC;

-- ========================================
-- 7. CLV SUMMARY BY SEGMENT
-- Aggregate CLV metrics by customer segment
-- ========================================

WITH customer_segments AS (
    SELECT 
        c.customer_id,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS recency_days,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(o.total_amount) AS monetary_value,
        CASE 
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) <= 30 
                 AND COUNT(DISTINCT o.order_id) >= 5 
                 AND SUM(o.total_amount) >= 1000 THEN 'Champions'
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) <= 60 
                 AND COUNT(DISTINCT o.order_id) >= 3 THEN 'Loyal'
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) <= 30 
                 AND COUNT(DISTINCT o.order_id) <= 2 THEN 'New'
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) > 180 
                 AND SUM(o.total_amount) >= 1000 THEN 'At Risk VIP'
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) > 180 THEN 'Lost'
            ELSE 'Regular'
        END AS segment
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND c.status = 'active'
    GROUP BY c.customer_id
)
SELECT 
    segment,
    COUNT(DISTINCT customer_id) AS customer_count,
    ROUND(AVG(monetary_value), 2) AS avg_ltv,
    ROUND(SUM(monetary_value), 2) AS total_ltv,
    ROUND(AVG(frequency), 1) AS avg_orders,
    ROUND(AVG(recency_days), 0) AS avg_days_since_order,
    ROUND(AVG(monetary_value / NULLIF(frequency, 0)), 2) AS avg_order_value,
    ROUND(SUM(monetary_value) / SUM(SUM(monetary_value)) OVER () * 100, 2) AS revenue_contribution_pct
FROM customer_segments
GROUP BY segment
ORDER BY total_ltv DESC;

-- ========================================
-- End of Customer Lifetime Value Analysis
-- ========================================