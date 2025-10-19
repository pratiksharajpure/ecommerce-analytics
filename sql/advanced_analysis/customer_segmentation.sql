-- ========================================
-- CUSTOMER SEGMENTATION QUERIES
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. RFM SEGMENTATION ANALYSIS
-- Recency, Frequency, Monetary Value
-- ========================================

-- Calculate RFM Scores for all customers
WITH customer_rfm AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        -- Recency: Days since last purchase
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS recency_days,
        -- Frequency: Number of orders
        COUNT(DISTINCT o.order_id) AS frequency,
        -- Monetary: Total spend
        COALESCE(SUM(o.total_amount), 0) AS monetary_value,
        -- Average order value
        COALESCE(AVG(o.total_amount), 0) AS avg_order_value
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id 
        AND o.payment_status = 'paid'
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email
),
rfm_percentiles AS (
    SELECT 
        customer_id,
        first_name,
        last_name,
        email,
        recency_days,
        frequency,
        monetary_value,
        avg_order_value,
        -- Calculate percentile ranks (1-5 scale)
        NTILE(5) OVER (ORDER BY recency_days ASC) AS recency_score,
        NTILE(5) OVER (ORDER BY frequency DESC) AS frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value DESC) AS monetary_score
    FROM customer_rfm
),
rfm_segments AS (
    SELECT 
        *,
        -- Create combined RFM score
        CONCAT(recency_score, frequency_score, monetary_score) AS rfm_score,
        (recency_score + frequency_score + monetary_score) AS rfm_total,
        -- Assign customer segments
        CASE 
            WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
            WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
            WHEN recency_score >= 4 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'New Customers'
            WHEN recency_score >= 3 AND frequency_score <= 2 AND monetary_score <= 3 THEN 'Potential Loyalists'
            WHEN recency_score <= 2 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'At Risk'
            WHEN recency_score <= 2 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Cant Lose Them'
            WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'Hibernating High Value'
            WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'Lost'
            WHEN recency_score >= 3 AND monetary_score >= 3 THEN 'Promising'
            ELSE 'Needs Attention'
        END AS customer_segment
    FROM rfm_percentiles
)
SELECT 
    customer_segment,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS segment_percentage,
    ROUND(AVG(recency_days), 1) AS avg_recency_days,
    ROUND(AVG(frequency), 1) AS avg_frequency,
    ROUND(AVG(monetary_value), 2) AS avg_monetary_value,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(SUM(monetary_value), 2) AS total_segment_revenue,
    ROUND(SUM(monetary_value) * 100.0 / SUM(SUM(monetary_value)) OVER (), 2) AS revenue_percentage
FROM rfm_segments
GROUP BY customer_segment
ORDER BY total_segment_revenue DESC;

-- ========================================
-- 2. DETAILED RFM CUSTOMER LIST
-- Get individual customer details with segments
-- ========================================
WITH customer_rfm AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        c.city,
        c.state,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS recency_days,
        COUNT(DISTINCT o.order_id) AS frequency,
        COALESCE(SUM(o.total_amount), 0) AS monetary_value,
        MAX(o.order_date) AS last_order_date,
        MIN(o.order_date) AS first_order_date
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id 
        AND o.payment_status = 'paid'
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.city, c.state
),
rfm_scores AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY recency_days ASC) AS recency_score,
        NTILE(5) OVER (ORDER BY frequency DESC) AS frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value DESC) AS monetary_score
    FROM customer_rfm
)
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    city,
    state,
    recency_days,
    frequency,
    ROUND(monetary_value, 2) AS total_spent,
    recency_score,
    frequency_score,
    monetary_score,
    CASE 
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
        WHEN recency_score >= 4 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'New Customers'
        WHEN recency_score >= 3 AND frequency_score <= 2 AND monetary_score <= 3 THEN 'Potential Loyalists'
        WHEN recency_score <= 2 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Cant Lose Them'
        WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'Hibernating High Value'
        WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'Lost'
        WHEN recency_score >= 3 AND monetary_score >= 3 THEN 'Promising'
        ELSE 'Needs Attention'
    END AS customer_segment,
    last_order_date,
    first_order_date,
    DATEDIFF(last_order_date, first_order_date) AS customer_lifetime_days
FROM rfm_scores
ORDER BY monetary_value DESC
LIMIT 100;

-- ========================================
-- 3. BEHAVIORAL SEGMENTS
-- Purchase behavior patterns
-- ========================================
WITH customer_behavior AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        COUNT(DISTINCT o.order_id) AS total_orders,
        COUNT(DISTINCT DATE_FORMAT(o.order_date, '%Y-%m')) AS active_months,
        COALESCE(SUM(o.total_amount), 0) AS total_revenue,
        COALESCE(AVG(o.total_amount), 0) AS avg_order_value,
        COUNT(DISTINCT oi.product_id) AS unique_products_purchased,
        COALESCE(AVG(oi.quantity), 0) AS avg_items_per_order,
        MAX(o.order_date) AS last_purchase_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_purchase,
        -- Weekend vs weekday preference
        SUM(CASE WHEN DAYOFWEEK(o.order_date) IN (1,7) THEN 1 ELSE 0 END) AS weekend_orders,
        -- Discount sensitivity
        AVG(oi.discount) AS avg_discount_per_item,
        -- Return behavior
        COUNT(DISTINCT r.return_id) AS total_returns
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email
)
SELECT 
    CASE 
        WHEN total_orders >= 10 AND avg_order_value >= 150 THEN 'High Value Frequent'
        WHEN total_orders >= 10 AND avg_order_value < 150 THEN 'Frequent Bargain Hunter'
        WHEN total_orders >= 5 AND total_orders < 10 AND avg_order_value >= 150 THEN 'Medium Value Regular'
        WHEN total_orders >= 5 AND avg_order_value < 150 THEN 'Regular Shopper'
        WHEN total_orders >= 2 AND total_orders < 5 THEN 'Occasional Buyer'
        WHEN total_orders = 1 THEN 'One-Time Buyer'
        ELSE 'Window Shopper'
    END AS behavior_segment,
    COUNT(*) AS customer_count,
    ROUND(AVG(total_orders), 1) AS avg_orders,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(AVG(unique_products_purchased), 1) AS avg_unique_products,
    ROUND(AVG(days_since_last_purchase), 1) AS avg_days_since_purchase,
    ROUND(SUM(total_revenue), 2) AS segment_total_revenue,
    ROUND(AVG(weekend_orders * 100.0 / NULLIF(total_orders, 0)), 1) AS weekend_order_percentage,
    ROUND(AVG(total_returns * 100.0 / NULLIF(total_orders, 0)), 1) AS return_rate_percentage
FROM customer_behavior
WHERE total_orders > 0
GROUP BY behavior_segment
ORDER BY segment_total_revenue DESC;

-- ========================================
-- 4. DEMOGRAPHIC SEGMENTS
-- Geographic and demographic analysis
-- ========================================
WITH customer_demographics AS (
    SELECT 
        c.customer_id,
        c.state,
        c.city,
        c.country,
        COUNT(DISTINCT o.order_id) AS order_count,
        COALESCE(SUM(o.total_amount), 0) AS total_spent,
        COALESCE(AVG(o.total_amount), 0) AS avg_order_value,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(CURDATE(), c.created_at) AS customer_age_days,
        -- Loyalty program membership
        lp.tier AS loyalty_tier,
        lp.points_balance
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.state, c.city, c.country, c.created_at, lp.tier, lp.points_balance
)
SELECT 
    state,
    COUNT(DISTINCT customer_id) AS customer_count,
    SUM(order_count) AS total_orders,
    ROUND(SUM(total_spent), 2) AS total_revenue,
    ROUND(AVG(total_spent), 2) AS avg_customer_value,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(AVG(order_count), 1) AS avg_orders_per_customer,
    COUNT(CASE WHEN loyalty_tier IS NOT NULL THEN 1 END) AS loyalty_members,
    ROUND(COUNT(CASE WHEN loyalty_tier IS NOT NULL THEN 1 END) * 100.0 / COUNT(*), 1) AS loyalty_penetration_pct
FROM customer_demographics
GROUP BY state
HAVING customer_count >= 5
ORDER BY total_revenue DESC
LIMIT 20;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE SEGMENTS
-- CLV-based segmentation
-- ========================================
WITH customer_ltv AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        COALESCE(SUM(o.total_amount), 0) AS historical_value,
        COUNT(DISTINCT o.order_id) AS order_count,
        DATEDIFF(CURDATE(), MIN(o.order_date)) AS customer_lifetime_days,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order,
        -- Projected CLV (simple model: historical value * 2 if active, * 0.5 if churning)
        CASE 
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) <= 90 THEN COALESCE(SUM(o.total_amount), 0) * 2.0
            WHEN DATEDIFF(CURDATE(), MAX(o.order_date)) <= 180 THEN COALESCE(SUM(o.total_amount), 0) * 1.2
            ELSE COALESCE(SUM(o.total_amount), 0) * 0.5
        END AS projected_ltv
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email
)
SELECT 
    CASE 
        WHEN projected_ltv >= 2000 THEN 'Very High Value (VIP)'
        WHEN projected_ltv >= 1000 THEN 'High Value'
        WHEN projected_ltv >= 500 THEN 'Medium Value'
        WHEN projected_ltv >= 100 THEN 'Low Value'
        ELSE 'Minimal Value'
    END AS ltv_segment,
    COUNT(*) AS customer_count,
    ROUND(AVG(historical_value), 2) AS avg_historical_value,
    ROUND(AVG(projected_ltv), 2) AS avg_projected_ltv,
    ROUND(AVG(order_count), 1) AS avg_orders,
    ROUND(AVG(customer_lifetime_days), 0) AS avg_lifetime_days,
    ROUND(SUM(historical_value), 2) AS segment_historical_revenue,
    ROUND(SUM(projected_ltv), 2) AS segment_projected_ltv
FROM customer_ltv
GROUP BY ltv_segment
ORDER BY avg_projected_ltv DESC;

-- ========================================
-- 6. CHURN RISK SEGMENTATION
-- Identify customers at risk of churning
-- ========================================
WITH customer_churn_risk AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        COUNT(DISTINCT o.order_id) AS total_orders,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order,
        COALESCE(SUM(o.total_amount), 0) AS total_spent,
        AVG(DATEDIFF(o.order_date, LAG(o.order_date) OVER (PARTITION BY c.customer_id ORDER BY o.order_date))) AS avg_days_between_orders,
        -- Returns and complaints
        COUNT(DISTINCT r.return_id) AS return_count
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE c.status = 'active'
    GROUP BY c.customer_id, c.first_name, c.last_name, c.email
)
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    total_orders,
    ROUND(total_spent, 2) AS total_spent,
    last_order_date,
    days_since_last_order,
    ROUND(avg_days_between_orders, 0) AS avg_days_between_orders,
    return_count,
    CASE 
        WHEN days_since_last_order > COALESCE(avg_days_between_orders * 2, 90) AND total_spent >= 500 THEN 'Critical - High Value'
        WHEN days_since_last_order > COALESCE(avg_days_between_orders * 2, 90) THEN 'High Risk'
        WHEN days_since_last_order > COALESCE(avg_days_between_orders * 1.5, 60) THEN 'Medium Risk'
        WHEN days_since_last_order > COALESCE(avg_days_between_orders, 30) THEN 'Low Risk'
        ELSE 'Active'
    END AS churn_risk_level,
    CASE 
        WHEN days_since_last_order > COALESCE(avg_days_between_orders * 2, 90) THEN 'Send win-back campaign with discount'
        WHEN days_since_last_order > COALESCE(avg_days_between_orders * 1.5, 60) THEN 'Send re-engagement email'
        WHEN days_since_last_order > COALESCE(avg_days_between_orders, 30) THEN 'Send reminder about new products'
        ELSE 'Continue regular marketing'
    END AS recommended_action
FROM customer_churn_risk
WHERE total_orders > 0
ORDER BY 
    CASE 
        WHEN days_since_last_order > COALESCE(avg_days_between_orders * 2, 90) AND total_spent >= 500 THEN 1
        WHEN days_since_last_order > COALESCE(avg_days_between_orders * 2, 90) THEN 2
        WHEN days_since_last_order > COALESCE(avg_days_between_orders * 1.5, 60) THEN 3
        ELSE 4
    END,
    total_spent DESC
LIMIT 100;