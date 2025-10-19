-- ========================================
-- CHURN PREDICTION & AT-RISK ANALYSIS
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CHURN INDICATORS & RISK FACTORS
-- Identifies key behavioral signals of churn
-- ========================================

WITH customer_behavior AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        c.created_at AS signup_date,
        DATEDIFF(CURDATE(), c.created_at) AS customer_age_days,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        MIN(o.order_date) AS first_order_date,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) AS purchase_span_days,
        -- Last 90 days activity
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) THEN o.order_id 
        END) AS orders_last_90_days,
        -- Last 30 days activity
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN o.order_id 
        END) AS orders_last_30_days,
        -- Returns and issues
        COUNT(DISTINCT r.return_id) AS total_returns,
        COUNT(DISTINCT CASE 
            WHEN r.status IN ('approved', 'refunded') THEN r.return_id 
        END) AS approved_returns
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.created_at
),
churn_indicators AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        signup_date,
        customer_age_days,
        total_orders,
        ROUND(total_revenue, 2) AS total_revenue,
        ROUND(avg_order_value, 2) AS avg_order_value,
        last_order_date,
        days_since_last_order,
        orders_last_90_days,
        orders_last_30_days,
        total_returns,
        approved_returns,
        -- Calculate average days between orders
        CASE 
            WHEN total_orders <= 1 THEN NULL
            ELSE ROUND(purchase_span_days * 1.0 / (total_orders - 1), 0)
        END AS avg_days_between_orders,
        -- Return rate
        ROUND((approved_returns * 100.0) / NULLIF(total_orders, 0), 2) AS return_rate_pct,
        -- Churn indicators (binary flags)
        CASE WHEN days_since_last_order > 90 THEN 1 ELSE 0 END AS inactive_90_days,
        CASE WHEN days_since_last_order > 180 THEN 1 ELSE 0 END AS inactive_180_days,
        CASE WHEN orders_last_90_days = 0 AND total_orders > 0 THEN 1 ELSE 0 END AS no_recent_orders,
        CASE WHEN (approved_returns * 100.0) / NULLIF(total_orders, 0) > 20 THEN 1 ELSE 0 END AS high_return_rate,
        CASE 
            WHEN total_orders > 1 AND days_since_last_order > (purchase_span_days * 2.0 / (total_orders - 1)) 
            THEN 1 ELSE 0 
        END AS exceeded_normal_gap
    FROM customer_behavior
    WHERE total_orders > 0
)
SELECT 
    customer_id,
    customer_name,
    email,
    total_orders,
    total_revenue,
    avg_order_value,
    last_order_date,
    days_since_last_order,
    avg_days_between_orders,
    orders_last_90_days,
    orders_last_30_days,
    return_rate_pct,
    -- Churn indicators
    inactive_90_days AS is_inactive_90d,
    inactive_180_days AS is_inactive_180d,
    no_recent_orders AS no_orders_90d,
    high_return_rate AS has_high_returns,
    exceeded_normal_gap AS exceeded_purchase_cycle,
    -- Total risk flags
    (inactive_90_days + no_recent_orders + high_return_rate + exceeded_normal_gap) AS total_risk_flags,
    CASE 
        WHEN inactive_180_days = 1 THEN 'Likely Churned'
        WHEN (inactive_90_days + no_recent_orders + high_return_rate + exceeded_normal_gap) >= 3 THEN 'Critical Risk'
        WHEN (inactive_90_days + no_recent_orders + high_return_rate + exceeded_normal_gap) >= 2 THEN 'High Risk'
        WHEN (inactive_90_days + no_recent_orders + high_return_rate + exceeded_normal_gap) = 1 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS churn_risk_level
FROM churn_indicators
ORDER BY total_risk_flags DESC, total_revenue DESC;

-- ========================================
-- 2. AT-RISK CUSTOMERS PRIORITIZATION
-- High-value customers showing churn signals
-- ========================================

WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        c.phone,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS lifetime_value,
        AVG(o.total_amount) AS avg_order_value,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order,
        -- Calculate expected return based on history
        AVG(DATEDIFF(
            LEAD(o.order_date) OVER (PARTITION BY c.customer_id ORDER BY o.order_date),
            o.order_date
        )) AS avg_days_between_orders,
        -- Recent engagement
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) THEN o.order_id 
        END) AS orders_last_90_days,
        -- Loyalty program
        lp.tier AS loyalty_tier,
        lp.points_balance
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.phone, 
             lp.tier, lp.points_balance
    HAVING total_orders > 0
),
at_risk_scoring AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        phone,
        total_orders,
        ROUND(lifetime_value, 2) AS lifetime_value,
        ROUND(avg_order_value, 2) AS avg_order_value,
        last_order_date,
        days_since_last_order,
        ROUND(avg_days_between_orders, 0) AS expected_order_interval,
        orders_last_90_days,
        loyalty_tier,
        COALESCE(points_balance, 0) AS loyalty_points,
        -- Risk calculation
        CASE 
            WHEN avg_days_between_orders IS NULL THEN 0
            WHEN days_since_last_order > avg_days_between_orders * 2 THEN 3
            WHEN days_since_last_order > avg_days_between_orders * 1.5 THEN 2
            WHEN days_since_last_order > avg_days_between_orders THEN 1
            ELSE 0
        END AS overdue_score,
        CASE 
            WHEN lifetime_value >= 2000 THEN 5
            WHEN lifetime_value >= 1000 THEN 4
            WHEN lifetime_value >= 500 THEN 3
            WHEN lifetime_value >= 250 THEN 2
            ELSE 1
        END AS value_score,
        CASE 
            WHEN total_orders >= 10 THEN 5
            WHEN total_orders >= 5 THEN 4
            WHEN total_orders >= 3 THEN 3
            WHEN total_orders >= 2 THEN 2
            ELSE 1
        END AS frequency_score
    FROM customer_metrics
),
priority_ranking AS (
    SELECT 
        *,
        -- Priority = Risk × Value
        (overdue_score * value_score * frequency_score) AS priority_score,
        CASE 
            WHEN overdue_score >= 2 AND value_score >= 4 THEN 'Urgent - High Value at Risk'
            WHEN overdue_score >= 2 AND value_score >= 3 THEN 'High Priority - Medium Value at Risk'
            WHEN overdue_score >= 2 THEN 'Medium Priority - At Risk'
            WHEN overdue_score = 1 AND value_score >= 4 THEN 'Watch List - High Value'
            ELSE 'Monitor'
        END AS priority_category
    FROM at_risk_scoring
)
SELECT 
    customer_id,
    customer_name,
    email,
    phone,
    total_orders,
    lifetime_value,
    avg_order_value,
    last_order_date,
    days_since_last_order,
    expected_order_interval,
    ROUND(days_since_last_order * 1.0 / NULLIF(expected_order_interval, 0), 2) AS days_overdue_ratio,
    orders_last_90_days,
    loyalty_tier,
    loyalty_points,
    priority_score,
    priority_category,
    -- Recommended action
    CASE 
        WHEN priority_category = 'Urgent - High Value at Risk' THEN 'Immediate outreach + VIP offer'
        WHEN priority_category = 'High Priority - Medium Value at Risk' THEN 'Personalized email + discount'
        WHEN priority_category = 'Medium Priority - At Risk' THEN 'Re-engagement campaign'
        WHEN priority_category = 'Watch List - High Value' THEN 'Proactive check-in'
        ELSE 'Standard monitoring'
    END AS recommended_action
FROM priority_ranking
WHERE overdue_score >= 1  -- Only show customers showing risk signals
ORDER BY priority_score DESC, lifetime_value DESC
LIMIT 500;

-- ========================================
-- 3. CHURN PROBABILITY SCORING MODEL
-- Statistical scoring based on multiple factors
-- ========================================

WITH customer_features AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        DATEDIFF(CURDATE(), c.created_at) AS customer_age_days,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS recency_days,
        -- Average time between orders
        CASE 
            WHEN COUNT(DISTINCT o.order_id) <= 1 THEN NULL
            ELSE DATEDIFF(MAX(o.order_date), MIN(o.order_date)) * 1.0 / (COUNT(DISTINCT o.order_id) - 1)
        END AS avg_days_between_orders,
        -- Trend indicators
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) THEN o.order_id 
        END) AS orders_last_90d,
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 180 DAY) 
                 AND o.order_date < DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN o.order_id 
        END) AS orders_prev_90d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) THEN o.total_amount 
            ELSE 0 
        END) AS revenue_last_90d,
        SUM(CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 180 DAY) 
                 AND o.order_date < DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN o.total_amount 
            ELSE 0 
        END) AS revenue_prev_90d,
        -- Product diversity
        COUNT(DISTINCT oi.product_id) AS unique_products_purchased,
        -- Returns and negative experiences
        COUNT(DISTINCT r.return_id) AS total_returns,
        COUNT(DISTINCT CASE 
            WHEN rev.rating <= 2 THEN rev.review_id 
        END) AS negative_reviews
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN returns r ON o.order_id = r.order_id
    LEFT JOIN reviews rev ON c.customer_id = rev.customer_id
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.created_at
    HAVING total_orders > 0
),
churn_scores AS (
    SELECT 
        customer_id,
        customer_name,
        email,
        customer_age_days,
        total_orders,
        ROUND(total_revenue, 2) AS total_revenue,
        last_order_date,
        recency_days,
        ROUND(avg_days_between_orders, 1) AS avg_days_between_orders,
        orders_last_90d,
        orders_prev_90d,
        ROUND(revenue_last_90d, 2) AS revenue_last_90d,
        ROUND(revenue_prev_90d, 2) AS revenue_prev_90d,
        unique_products_purchased,
        total_returns,
        negative_reviews,
        -- SCORING COMPONENTS (0-20 points each)
        -- 1. Recency Score (20 points max, lower is better)
        CASE 
            WHEN recency_days <= 30 THEN 0
            WHEN recency_days <= 60 THEN 5
            WHEN recency_days <= 90 THEN 10
            WHEN recency_days <= 180 THEN 15
            ELSE 20
        END AS recency_risk_score,
        -- 2. Order Gap Score (20 points max)
        CASE 
            WHEN avg_days_between_orders IS NULL THEN 10
            WHEN recency_days <= avg_days_between_orders THEN 0
            WHEN recency_days <= avg_days_between_orders * 1.5 THEN 5
            WHEN recency_days <= avg_days_between_orders * 2 THEN 10
            WHEN recency_days <= avg_days_between_orders * 3 THEN 15
            ELSE 20
        END AS order_gap_risk_score,
        -- 3. Activity Decline Score (20 points max)
        CASE 
            WHEN orders_prev_90d = 0 THEN 0
            WHEN orders_last_90d > orders_prev_90d THEN 0
            WHEN orders_last_90d = orders_prev_90d THEN 5
            WHEN orders_last_90d >= orders_prev_90d * 0.5 THEN 10
            WHEN orders_last_90d > 0 THEN 15
            ELSE 20
        END AS activity_decline_score,
        -- 4. Revenue Decline Score (20 points max)
        CASE 
            WHEN revenue_prev_90d = 0 THEN 0
            WHEN revenue_last_90d > revenue_prev_90d THEN 0
            WHEN revenue_last_90d >= revenue_prev_90d * 0.75 THEN 5
            WHEN revenue_last_90d >= revenue_prev_90d * 0.5 THEN 10
            WHEN revenue_last_90d > 0 THEN 15
            ELSE 20
        END AS revenue_decline_score,
        -- 5. Negative Experience Score (20 points max)
        LEAST(20, (total_returns * 5) + (negative_reviews * 10)) AS negative_experience_score
    FROM customer_features
)
SELECT 
    customer_id,
    customer_name,
    email,
    customer_age_days,
    total_orders,
    total_revenue,
    last_order_date,
    recency_days,
    avg_days_between_orders,
    orders_last_90d,
    orders_prev_90d,
    revenue_last_90d,
    revenue_prev_90d,
    -- Individual score components
    recency_risk_score,
    order_gap_risk_score,
    activity_decline_score,
    revenue_decline_score,
    negative_experience_score,
    -- Total churn score (0-100)
    (recency_risk_score + order_gap_risk_score + activity_decline_score + 
     revenue_decline_score + negative_experience_score) AS total_churn_score,
    -- Churn probability (normalized to percentage)
    ROUND((recency_risk_score + order_gap_risk_score + activity_decline_score + 
           revenue_decline_score + negative_experience_score), 0) AS churn_probability_score,
    -- Risk classification
    CASE 
        WHEN (recency_risk_score + order_gap_risk_score + activity_decline_score + 
              revenue_decline_score + negative_experience_score) >= 70 THEN 'Critical (70-100)'
        WHEN (recency_risk_score + order_gap_risk_score + activity_decline_score + 
              revenue_decline_score + negative_experience_score) >= 50 THEN 'High (50-69)'
        WHEN (recency_risk_score + order_gap_risk_score + activity_decline_score + 
              revenue_decline_score + negative_experience_score) >= 30 THEN 'Medium (30-49)'
        WHEN (recency_risk_score + order_gap_risk_score + activity_decline_score + 
              revenue_decline_score + negative_experience_score) >= 15 THEN 'Low (15-29)'
        ELSE 'Minimal (0-14)'
    END AS churn_risk_category,
    -- Value-risk matrix
    CASE 
        WHEN total_revenue >= 1000 THEN 'High Value'
        WHEN total_revenue >= 500 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS customer_value_tier
FROM churn_scores
ORDER BY total_churn_score DESC, total_revenue DESC;

-- ========================================
-- 4. CHURN PREDICTION BY CUSTOMER SEGMENT
-- Aggregate churn risk across segments
-- ========================================

WITH customer_churn_data AS (
    SELECT 
        c.customer_id,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS recency_days,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(o.total_amount) AS monetary_value,
        lp.tier AS loyalty_tier,
        CASE 
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) <= 30 THEN 'Active'
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) <= 90 THEN 'Cooling'
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) <= 180 THEN 'At Risk'
            ELSE 'Churned'
        END AS activity_status,
        CASE 
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) > 180 THEN 1
            ELSE 0
        END AS is_churned
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
    WHERE c.status = 'active'
    GROUP BY c.customer_id, lp.tier
    HAVING COUNT(DISTINCT o.order_id) > 0
)
SELECT 
    activity_status,
    COALESCE(loyalty_tier, 'No Tier') AS loyalty_tier,
    COUNT(*) AS customer_count,
    SUM(is_churned) AS churned_customers,
    ROUND(AVG(monetary_value), 2) AS avg_ltv,
    ROUND(AVG(frequency), 1) AS avg_orders,
    ROUND(AVG(recency_days), 0) AS avg_days_since_order,
    ROUND(SUM(is_churned) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(monetary_value), 2) AS total_revenue_at_risk
FROM customer_churn_data
GROUP BY activity_status, loyalty_tier
ORDER BY 
    FIELD(activity_status, 'Active', 'Cooling', 'At Risk', 'Churned'),
    FIELD(loyalty_tier, 'platinum', 'gold', 'silver', 'bronze', 'No Tier');

-- ========================================
-- 5. WIN-BACK OPPORTUNITY ANALYSIS
-- Recently churned customers worth recovering
-- ========================================

WITH churned_customers AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.email,
        c.phone,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS lifetime_value,
        AVG(o.total_amount) AS avg_order_value,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_churned,
        MIN(o.order_date) AS first_order_date,
        DATEDIFF(MAX(o.order_date), MIN(o.order_date)) AS active_period_days,
        -- Calculate their most purchased category
        (SELECT pc.category_name 
         FROM order_items oi2
         JOIN products p ON oi2.product_id = p.product_id
         JOIN product_categories pc ON p.category_id = pc.category_id
         WHERE oi2.order_id IN (
             SELECT order_id FROM orders WHERE customer_id = c.customer_id
         )
         GROUP BY pc.category_name
         ORDER BY COUNT(*) DESC
         LIMIT 1
        ) AS favorite_category,
        -- Returns
        COUNT(DISTINCT r.return_id) AS total_returns,
        -- Loyalty
        lp.points_balance,
        lp.tier AS loyalty_tier
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    LEFT JOIN returns r ON o.order_id = r.order_id
    LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
    WHERE c.status = 'active'
        AND DATEDIFF(CURDATE(), (SELECT MAX(order_date) 
                                  FROM orders 
                                  WHERE customer_id = c.customer_id 
                                    AND status IN ('delivered', 'shipped', 'processing')
                                    AND payment_status = 'paid')) BETWEEN 180 AND 365
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.phone,
             lp.points_balance, lp.tier
),
win_back_scoring AS (
    SELECT 
        *,
        -- Win-back potential score
        CASE 
            WHEN lifetime_value >= 2000 THEN 40
            WHEN lifetime_value >= 1000 THEN 30
            WHEN lifetime_value >= 500 THEN 20
            ELSE 10
        END AS value_score,
        CASE 
            WHEN total_orders >= 10 THEN 30
            WHEN total_orders >= 5 THEN 20
            WHEN total_orders >= 3 THEN 10
            ELSE 5
        END AS loyalty_score,
        CASE 
            WHEN days_churned <= 210 THEN 20
            WHEN days_churned <= 270 THEN 15
            WHEN days_churned <= 330 THEN 10
            ELSE 5
        END AS recency_score,
        CASE 
            WHEN total_returns = 0 THEN 10
            WHEN (total_returns * 100.0 / total_orders) < 10 THEN 5
            ELSE 0
        END AS satisfaction_score
    FROM churned_customers
)
SELECT 
    customer_id,
    customer_name,
    email,
    phone,
    total_orders,
    ROUND(lifetime_value, 2) AS lifetime_value,
    ROUND(avg_order_value, 2) AS avg_order_value,
    last_order_date,
    days_churned,
    favorite_category,
    COALESCE(loyalty_tier, 'None') AS loyalty_tier,
    COALESCE(points_balance, 0) AS unused_loyalty_points,
    (value_score + loyalty_score + recency_score + satisfaction_score) AS win_back_score,
    CASE 
        WHEN (value_score + loyalty_score + recency_score + satisfaction_score) >= 80 THEN 'Tier 1 - High Priority'
        WHEN (value_score + loyalty_score + recency_score + satisfaction_score) >= 60 THEN 'Tier 2 - Medium Priority'
        WHEN (value_score + loyalty_score + recency_score + satisfaction_score) >= 40 THEN 'Tier 3 - Low Priority'
        ELSE 'Tier 4 - Monitor Only'
    END AS win_back_priority,
    -- Recommended win-back offer
    CASE 
        WHEN lifetime_value >= 1000 AND points_balance > 0 THEN 'VIP: 30% off + bonus points'
        WHEN lifetime_value >= 500 THEN '25% off next order'
        WHEN total_orders >= 5 THEN '20% off + free shipping'
        ELSE '15% off welcome back offer'
    END AS recommended_offer
FROM win_back_scoring
WHERE (value_score + loyalty_score + recency_score + satisfaction_score) >= 40
ORDER BY win_back_score DESC, lifetime_value DESC
LIMIT 200;

-- ========================================
-- End of Churn Prediction Analysis
-- ========================================