-- ========================================
-- CHANNEL ANALYSIS & ATTRIBUTION
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CAMPAIGN PERFORMANCE OVERVIEW
-- Comprehensive campaign metrics
-- ========================================
SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.status,
    c.start_date,
    c.end_date,
    DATEDIFF(COALESCE(c.end_date, CURDATE()), c.start_date) AS campaign_duration_days,
    ROUND(c.budget, 2) AS budget,
    -- Performance metrics
    SUM(cp.impressions) AS total_impressions,
    SUM(cp.clicks) AS total_clicks,
    SUM(cp.conversions) AS total_conversions,
    ROUND(SUM(cp.spend), 2) AS total_spend,
    ROUND(SUM(cp.revenue), 2) AS total_revenue,
    -- Calculate KPIs
    ROUND(SUM(cp.clicks) * 100.0 / NULLIF(SUM(cp.impressions), 0), 2) AS ctr_pct,
    ROUND(SUM(cp.conversions) * 100.0 / NULLIF(SUM(cp.clicks), 0), 2) AS conversion_rate_pct,
    ROUND(SUM(cp.spend) / NULLIF(SUM(cp.clicks), 0), 2) AS cpc,
    ROUND(SUM(cp.spend) / NULLIF(SUM(cp.conversions), 0), 2) AS cpa,
    ROUND(SUM(cp.revenue) / NULLIF(SUM(cp.spend), 0), 2) AS roas,
    ROUND(SUM(cp.revenue) - SUM(cp.spend), 2) AS net_profit,
    ROUND((SUM(cp.revenue) - SUM(cp.spend)) * 100.0 / NULLIF(SUM(cp.spend), 0), 2) AS roi_pct,
    -- Budget utilization
    ROUND(SUM(cp.spend) * 100.0 / NULLIF(c.budget, 0), 2) AS budget_utilization_pct,
    ROUND(c.budget - SUM(cp.spend), 2) AS remaining_budget
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
WHERE c.created_at >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.status, 
         c.start_date, c.end_date, c.budget
ORDER BY total_revenue DESC;

-- ========================================
-- 2. CAMPAIGN TYPE COMPARISON
-- Compare performance across campaign types
-- ========================================
WITH campaign_metrics AS (
    SELECT 
        c.campaign_type,
        COUNT(DISTINCT c.campaign_id) AS campaign_count,
        SUM(cp.impressions) AS total_impressions,
        SUM(cp.clicks) AS total_clicks,
        SUM(cp.conversions) AS total_conversions,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE c.start_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY c.campaign_type
)
SELECT 
    campaign_type,
    campaign_count,
    total_impressions,
    total_clicks,
    total_conversions,
    ROUND(total_spend, 2) AS total_spend,
    ROUND(total_revenue, 2) AS total_revenue,
    -- Average metrics per campaign
    ROUND(total_spend / campaign_count, 2) AS avg_spend_per_campaign,
    ROUND(total_revenue / campaign_count, 2) AS avg_revenue_per_campaign,
    -- Performance ratios
    ROUND(total_clicks * 100.0 / NULLIF(total_impressions, 0), 2) AS ctr_pct,
    ROUND(total_conversions * 100.0 / NULLIF(total_clicks, 0), 2) AS conversion_rate_pct,
    ROUND(total_spend / NULLIF(total_clicks, 0), 2) AS avg_cpc,
    ROUND(total_spend / NULLIF(total_conversions, 0), 2) AS avg_cpa,
    ROUND(total_revenue / NULLIF(total_spend, 0), 2) AS roas,
    ROUND((total_revenue - total_spend) * 100.0 / NULLIF(total_spend, 0), 2) AS roi_pct,
    -- Share of metrics
    ROUND(total_spend * 100.0 / SUM(total_spend) OVER (), 2) AS spend_share_pct,
    ROUND(total_revenue * 100.0 / SUM(total_revenue) OVER (), 2) AS revenue_share_pct
FROM campaign_metrics
ORDER BY total_revenue DESC;

-- ========================================
-- 3. DAILY CAMPAIGN PERFORMANCE TRENDS
-- Track campaign performance over time
-- ========================================
WITH daily_performance AS (
    SELECT 
        cp.report_date,
        DATE_FORMAT(cp.report_date, '%Y-%m') AS year_month,
        DAYNAME(cp.report_date) AS day_of_week,
        c.campaign_type,
        SUM(cp.impressions) AS impressions,
        SUM(cp.clicks) AS clicks,
        SUM(cp.conversions) AS conversions,
        SUM(cp.spend) AS spend,
        SUM(cp.revenue) AS revenue
    FROM campaign_performance cp
    JOIN campaigns c ON cp.campaign_id = c.campaign_id
    WHERE cp.report_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
    GROUP BY cp.report_date, year_month, day_of_week, c.campaign_type
)
SELECT 
    report_date,
    year_month,
    day_of_week,
    campaign_type,
    impressions,
    clicks,
    conversions,
    ROUND(spend, 2) AS spend,
    ROUND(revenue, 2) AS revenue,
    ROUND(clicks * 100.0 / NULLIF(impressions, 0), 2) AS ctr_pct,
    ROUND(conversions * 100.0 / NULLIF(clicks, 0), 2) AS conversion_rate_pct,
    ROUND(revenue / NULLIF(spend, 0), 2) AS roas,
    -- Calculate moving averages (7-day)
    ROUND(AVG(revenue) OVER (PARTITION BY campaign_type ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS revenue_7day_ma,
    ROUND(AVG(spend) OVER (PARTITION BY campaign_type ORDER BY report_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS spend_7day_ma
FROM daily_performance
ORDER BY report_date DESC, campaign_type;

-- ========================================
-- 4. CHANNEL ATTRIBUTION ANALYSIS
-- First-touch, last-touch, and multi-touch attribution
-- ========================================
WITH customer_campaign_touchpoints AS (
    SELECT 
        o.customer_id,
        o.order_id,
        o.order_date,
        o.total_amount,
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        cp.report_date AS touchpoint_date,
        ROW_NUMBER() OVER (PARTITION BY o.customer_id, o.order_id ORDER BY cp.report_date ASC) AS first_touch_rank,
        ROW_NUMBER() OVER (PARTITION BY o.customer_id, o.order_id ORDER BY cp.report_date DESC) AS last_touch_rank,
        COUNT(*) OVER (PARTITION BY o.customer_id, o.order_id) AS total_touchpoints
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN campaign_performance cp ON DATE(o.order_date) >= cp.report_date 
        AND DATE(o.order_date) <= DATE_ADD(cp.report_date, INTERVAL 30 DAY)
    JOIN campaigns camp ON cp.campaign_id = camp.campaign_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
),
attribution_models AS (
    SELECT 
        campaign_id,
        campaign_name,
        campaign_type,
        -- First-touch attribution
        COUNT(DISTINCT CASE WHEN first_touch_rank = 1 THEN order_id END) AS first_touch_orders,
        SUM(CASE WHEN first_touch_rank = 1 THEN total_amount ELSE 0 END) AS first_touch_revenue,
        -- Last-touch attribution
        COUNT(DISTINCT CASE WHEN last_touch_rank = 1 THEN order_id END) AS last_touch_orders,
        SUM(CASE WHEN last_touch_rank = 1 THEN total_amount ELSE 0 END) AS last_touch_revenue,
        -- Linear attribution (equal credit to all touchpoints)
        COUNT(DISTINCT order_id) AS assisted_orders,
        SUM(total_amount / total_touchpoints) AS linear_attribution_revenue,
        -- Total unique customers influenced
        COUNT(DISTINCT customer_id) AS influenced_customers
    FROM customer_campaign_touchpoints
    GROUP BY campaign_id, campaign_name, campaign_type
)
SELECT 
    campaign_name,
    campaign_type,
    influenced_customers,
    -- First-touch metrics
    first_touch_orders,
    ROUND(first_touch_revenue, 2) AS first_touch_revenue,
    ROUND(first_touch_revenue / NULLIF(first_touch_orders, 0), 2) AS first_touch_aov,
    -- Last-touch metrics
    last_touch_orders,
    ROUND(last_touch_revenue, 2) AS last_touch_revenue,
    ROUND(last_touch_revenue / NULLIF(last_touch_orders, 0), 2) AS last_touch_aov,
    -- Linear attribution metrics
    assisted_orders,
    ROUND(linear_attribution_revenue, 2) AS linear_revenue,
    ROUND(linear_attribution_revenue / NULLIF(assisted_orders, 0), 2) AS linear_aov,
    -- Compare attribution models
    ROUND((first_touch_revenue - last_touch_revenue) * 100.0 / NULLIF(last_touch_revenue, 0), 1) AS first_vs_last_diff_pct
FROM attribution_models
ORDER BY linear_revenue DESC;

-- ========================================
-- 5. MULTI-CHANNEL CUSTOMER BEHAVIOR
-- Customers who interact with multiple channels
-- ========================================
WITH customer_channel_engagement AS (
    SELECT 
        o.customer_id,
        COUNT(DISTINCT c.campaign_type) AS channels_used,
        STRING_AGG(DISTINCT c.campaign_type ORDER BY c.campaign_type SEPARATOR ', ') AS channel_mix,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        MIN(o.order_date) AS first_order_date,
        MAX(o.order_date) AS last_order_date
    FROM orders o
    JOIN campaign_performance cp ON DATE(o.order_date) >= cp.report_date 
        AND DATE(o.order_date) <= DATE_ADD(cp.report_date, INTERVAL 30 DAY)
    JOIN campaigns c ON cp.campaign_id = c.campaign_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY o.customer_id
)
SELECT 
    channels_used AS number_of_channels,
    COUNT(DISTINCT customer_id) AS customer_count,
    SUM(total_orders) AS total_orders,
    ROUND(SUM(total_revenue), 2) AS total_revenue,
    ROUND(AVG(total_revenue), 2) AS avg_customer_ltv,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(AVG(total_orders), 1) AS avg_orders_per_customer,
    -- Revenue share
    ROUND(SUM(total_revenue) * 100.0 / SUM(SUM(total_revenue)) OVER (), 2) AS revenue_share_pct,
    -- Top channel combinations for this group
    (SELECT channel_mix FROM customer_channel_engagement cce2 
     WHERE cce2.channels_used = cce.channels_used 
     GROUP BY channel_mix ORDER BY COUNT(*) DESC LIMIT 1) AS most_common_channel_mix
FROM customer_channel_engagement cce
GROUP BY channels_used
ORDER BY channels_used DESC;

-- ========================================
-- 6. CAMPAIGN EFFICIENCY ANALYSIS
-- Identify best and worst performing campaigns
-- ========================================
WITH campaign_efficiency AS (
    SELECT 
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        c.status,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        SUM(cp.clicks) AS total_clicks,
        DATEDIFF(COALESCE(c.end_date, CURDATE()), c.start_date) AS campaign_duration
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE c.start_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.status, c.start_date, c.end_date
    HAVING total_spend > 0
)
SELECT 
    campaign_name,
    campaign_type,
    status,
    campaign_duration AS duration_days,
    total_conversions,
    ROUND(total_spend, 2) AS total_spend,
    ROUND(total_revenue, 2) AS total_revenue,
    ROUND(total_revenue - total_spend, 2) AS net_profit,
    ROUND(total_revenue / NULLIF(total_spend, 0), 2) AS roas,
    ROUND((total_revenue - total_spend) * 100.0 / NULLIF(total_spend, 0), 2) AS roi_pct,
    ROUND(total_spend / NULLIF(total_conversions, 0), 2) AS cpa,
    -- Performance rating
    CASE 
        WHEN (total_revenue / NULLIF(total_spend, 0)) >= 4.0 THEN 'Excellent'
        WHEN (total_revenue / NULLIF(total_spend, 0)) >= 2.5 THEN 'Good'
        WHEN (total_revenue / NULLIF(total_spend, 0)) >= 1.5 THEN 'Average'
        WHEN (total_revenue / NULLIF(total_spend, 0)) >= 1.0 THEN 'Below Average'
        ELSE 'Poor'
    END AS performance_rating,
    -- Recommendations
    CASE 
        WHEN (total_revenue / NULLIF(total_spend, 0)) >= 3.0 AND status = 'active' THEN 'Increase budget'
        WHEN (total_revenue / NULLIF(total_spend, 0)) >= 2.0 AND status = 'paused' THEN 'Reactivate campaign'
        WHEN (total_revenue / NULLIF(total_spend, 0)) < 1.0 THEN 'Pause or terminate'
        WHEN (total_revenue / NULLIF(total_spend, 0)) < 1.5 THEN 'Optimize or reduce budget'
        ELSE 'Monitor performance'
    END AS recommendation
FROM campaign_efficiency
ORDER BY roas DESC;

-- ========================================
-- 7. CHANNEL CUSTOMER ACQUISITION COST
-- Compare acquisition costs across channels
-- ========================================
WITH new_customers_by_channel AS (
    SELECT 
        c.campaign_type,
        o.customer_id,
        MIN(o.order_date) AS first_order_date,
        MIN(o.total_amount) AS first_order_value
    FROM orders o
    JOIN campaign_performance cp ON DATE(o.order_date) >= cp.report_date 
        AND DATE(o.order_date) <= DATE_ADD(cp.report_date, INTERVAL 7 DAY)
    JOIN campaigns c ON cp.campaign_id = c.campaign_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY c.campaign_type, o.customer_id
    HAVING first_order_date = MIN(o.order_date)
),
channel_spend AS (
    SELECT 
        c.campaign_type,
        SUM(cp.spend) AS total_spend,
        SUM(cp.conversions) AS total_conversions
    FROM campaigns c
    JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE cp.report_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY c.campaign_type
)
SELECT 
    cs.campaign_type,
    COUNT(DISTINCT nc.customer_id) AS new_customers_acquired,
    ROUND(cs.total_spend, 2) AS total_spend,
    cs.total_conversions,
    ROUND(cs.total_spend / NULLIF(COUNT(DISTINCT nc.customer_id), 0), 2) AS cac,
    ROUND(AVG(nc.first_order_value), 2) AS avg_first_order_value,
    ROUND(SUM(nc.first_order_value), 2) AS total_first_order_revenue,
    -- CAC payback analysis
    ROUND(AVG(nc.first_order_value) / NULLIF(cs.total_spend / NULLIF(COUNT(DISTINCT nc.customer_id), 0), 0), 2) AS first_order_cac_ratio,
    CASE 
        WHEN AVG(nc.first_order_value) >= (cs.total_spend / NULLIF(COUNT(DISTINCT nc.customer_id), 0)) THEN 'Profitable on first order'
        WHEN AVG(nc.first_order_value) >= (cs.total_spend / NULLIF(COUNT(DISTINCT nc.customer_id), 0)) * 0.5 THEN 'Break-even in 2 orders'
        ELSE 'Requires 3+ orders to break even'
    END AS payback_analysis
FROM channel_spend cs
LEFT JOIN new_customers_by_channel nc ON cs.campaign_type = nc.campaign_type
GROUP BY cs.campaign_type, cs.total_spend, cs.total_conversions
ORDER BY cac ASC;

-- ========================================
-- 8. SEASONAL CHANNEL PERFORMANCE
-- How different channels perform by season/month
-- ========================================
SELECT 
    DATE_FORMAT(cp.report_date, '%Y-%m') AS year_month,
    MONTHNAME(cp.report_date) AS month_name,
    QUARTER(cp.report_date) AS quarter,
    c.campaign_type,
    COUNT(DISTINCT c.campaign_id) AS active_campaigns,
    SUM(cp.impressions) AS impressions,
    SUM(cp.clicks) AS clicks,
    SUM(cp.conversions) AS conversions,
    ROUND(SUM(cp.spend), 2) AS spend,
    ROUND(SUM(cp.revenue), 2) AS revenue,
    ROUND(SUM(cp.clicks) * 100.0 / NULLIF(SUM(cp.impressions), 0), 2) AS ctr_pct,
    ROUND(SUM(cp.conversions) * 100.0 / NULLIF(SUM(cp.clicks), 0), 2) AS conversion_rate_pct,
    ROUND(SUM(cp.revenue) / NULLIF(SUM(cp.spend), 0), 2) AS roas,
    -- Compare to average
    ROUND((SUM(cp.revenue) / NULLIF(SUM(cp.spend), 0)) - 
          AVG(SUM(cp.revenue) / NULLIF(SUM(cp.spend), 0)) OVER (PARTITION BY c.campaign_type), 2) AS roas_vs_avg
FROM campaign_performance cp
JOIN campaigns c ON cp.campaign_id = c.campaign_id
WHERE cp.report_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY year_month, month_name, quarter, c.campaign_type
ORDER BY cp.report_date DESC, c.campaign_type;

-- ========================================
-- 9. CAMPAIGN SATURATION ANALYSIS
-- Identify diminishing returns in campaigns
-- ========================================
WITH weekly_campaign_performance AS (
    SELECT 
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        YEARWEEK(cp.report_date) AS year_week,
        SUM(cp.spend) AS weekly_spend,
        SUM(cp.revenue) AS weekly_revenue,
        SUM(cp.conversions) AS weekly_conversions,
        SUM(cp.clicks) AS weekly_clicks
    FROM campaigns c
    JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE c.status IN ('active', 'completed')
        AND cp.report_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
    GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, year_week
),
campaign_trends AS (
    SELECT 
        campaign_id,
        campaign_name,
        campaign_type,
        year_week,
        weekly_spend,
        weekly_revenue,
        weekly_conversions,
        ROUND(weekly_revenue / NULLIF(weekly_spend, 0), 2) AS weekly_roas,
        LAG(weekly_revenue / NULLIF(weekly_spend, 0)) OVER (PARTITION BY campaign_id ORDER BY year_week) AS prev_week_roas,
        AVG(weekly_revenue / NULLIF(weekly_spend, 0)) OVER (PARTITION BY campaign_id ORDER BY year_week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS roas_4week_ma
    FROM weekly_campaign_performance
)
SELECT 
    campaign_name,
    campaign_type,
    COUNT(*) AS weeks_active,
    ROUND(AVG(weekly_spend), 2) AS avg_weekly_spend,
    ROUND(AVG(weekly_revenue), 2) AS avg_weekly_revenue,
    ROUND(AVG(weekly_roas), 2) AS avg_roas,
    ROUND(MIN(weekly_roas), 2) AS min_roas,
    ROUND(MAX(weekly_roas), 2) AS max_roas,
    ROUND(STDDEV(weekly_roas), 2) AS roas_volatility,
    -- Trend detection
    CASE 
        WHEN AVG(CASE WHEN weekly_roas < prev_week_roas THEN 1 ELSE 0 END) > 0.6 THEN 'Declining performance'
        WHEN AVG(CASE WHEN weekly_roas > prev_week_roas THEN 1 ELSE 0 END) > 0.6 THEN 'Improving performance'
        WHEN STDDEV(weekly_roas) > 1.0 THEN 'Volatile performance'
        ELSE 'Stable performance'
    END AS performance_trend,
    -- Saturation indicator
    CASE 
        WHEN AVG(weekly_roas) < 1.5 AND AVG(CASE WHEN weekly_roas < prev_week_roas THEN 1 ELSE 0 END) > 0.5 THEN 'High saturation risk'
        WHEN AVG(weekly_roas) < 2.0 AND STDDEV(weekly_roas) < 0.3 THEN 'Possible saturation'
        ELSE 'Healthy'
    END AS saturation_status
FROM campaign_trends
WHERE weekly_roas IS NOT NULL
GROUP BY campaign_id, campaign_name, campaign_type
HAVING weeks_active >= 4
ORDER BY avg_roas DESC;

-- ========================================
-- 10. CROSS-CHANNEL SYNERGY ANALYSIS
-- How channels work together
-- ========================================
WITH customer_channel_sequence AS (
    SELECT 
        o.customer_id,
        o.order_id,
        c.campaign_type,
        cp.report_date,
        ROW_NUMBER() OVER (PARTITION BY o.customer_id, o.order_id ORDER BY cp.report_date) AS touch_sequence,
        COUNT(*) OVER (PARTITION BY o.customer_id, o.order_id) AS total_touches,
        o.total_amount
    FROM orders o
    JOIN campaign_performance cp ON DATE(o.order_date) >= cp.report_date 
        AND DATE(o.order_date) <= DATE_ADD(cp.report_date, INTERVAL 30 DAY)
    JOIN campaigns c ON cp.campaign_id = c.campaign_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
),
channel_sequences AS (
    SELECT 
        GROUP_CONCAT(campaign_type ORDER BY touch_sequence SEPARATOR ' -> ') AS channel_path,
        COUNT(DISTINCT order_id) AS conversions,
        COUNT(DISTINCT customer_id) AS unique_customers,
        AVG(total_amount) AS avg_order_value,
        SUM(total_amount) AS total_revenue,
        AVG(total_touches) AS avg_touchpoints
    FROM customer_channel_sequence
    WHERE total_touches BETWEEN 2 AND 5  -- Focus on multi-touch journeys
    GROUP BY channel_path
    HAVING conversions >= 3
)
SELECT 
    channel_path,
    conversions,
    unique_customers,
    ROUND(avg_touchpoints, 1) AS avg_touchpoints,
    ROUND(avg_order_value, 2) AS avg_order_value,
    ROUND(total_revenue, 2) AS total_revenue,
    ROUND(total_revenue / conversions, 2) AS revenue_per_conversion,
    -- Calculate path value
    ROUND(total_revenue * conversions / SUM(total_revenue * conversions) OVER () * 100, 2) AS path_value_score
FROM channel_sequences
ORDER BY total_revenue DESC
LIMIT 30;