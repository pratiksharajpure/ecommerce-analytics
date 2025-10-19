-- ========================================
-- CAMPAIGN ATTRIBUTION ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Marketing Attribution Models, Multi-Touch, Channel Effectiveness
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. SINGLE-TOUCH ATTRIBUTION MODELS
-- First-touch, Last-touch, and comparison
-- ========================================

WITH customer_campaign_journey AS (
    SELECT 
        c.customer_id,
        c.created_at AS customer_acquisition_date,
        o.order_id,
        o.order_date,
        o.total_amount,
        -- Simulate campaign touchpoints (in real scenario, would come from tracking data)
        -- Using campaign performance data as proxy
        (SELECT cp.campaign_id 
         FROM campaign_performance cp 
         JOIN campaigns cam ON cp.campaign_id = cam.campaign_id
         WHERE cp.report_date <= DATE(c.created_at)
         ORDER BY cp.report_date DESC
         LIMIT 1) AS first_touch_campaign,
        (SELECT cp.campaign_id 
         FROM campaign_performance cp 
         JOIN campaigns cam ON cp.campaign_id = cam.campaign_id
         WHERE cp.report_date <= DATE(o.order_date)
         ORDER BY cp.report_date DESC
         LIMIT 1) AS last_touch_campaign
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    WHERE c.created_at >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
),
attribution_summary AS (
    SELECT 
        -- First-touch attribution
        ft.campaign_id AS campaign_id,
        ft.campaign_name,
        ft.campaign_type,
        COUNT(DISTINCT CASE WHEN ccj.first_touch_campaign = ft.campaign_id 
              THEN ccj.customer_id END) AS first_touch_customers,
        COUNT(DISTINCT CASE WHEN ccj.first_touch_campaign = ft.campaign_id 
              THEN ccj.order_id END) AS first_touch_orders,
        SUM(CASE WHEN ccj.first_touch_campaign = ft.campaign_id 
            THEN ccj.total_amount ELSE 0 END) AS first_touch_revenue,
        -- Last-touch attribution
        COUNT(DISTINCT CASE WHEN ccj.last_touch_campaign = ft.campaign_id 
              THEN ccj.customer_id END) AS last_touch_customers,
        COUNT(DISTINCT CASE WHEN ccj.last_touch_campaign = ft.campaign_id 
              THEN ccj.order_id END) AS last_touch_orders,
        SUM(CASE WHEN ccj.last_touch_campaign = ft.campaign_id 
            THEN ccj.total_amount ELSE 0 END) AS last_touch_revenue,
        -- Campaign spend
        SUM(DISTINCT cp.spend) AS total_spend
    FROM campaigns ft
    LEFT JOIN customer_campaign_journey ccj ON 1=1
    LEFT JOIN campaign_performance cp ON ft.campaign_id = cp.campaign_id
    WHERE ft.status IN ('active', 'completed')
        AND ft.start_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY ft.campaign_id, ft.campaign_name, ft.campaign_type
)
SELECT 
    campaign_id,
    campaign_name,
    campaign_type,
    -- First-touch metrics
    first_touch_customers,
    first_touch_orders,
    ROUND(first_touch_revenue, 2) AS first_touch_revenue,
    ROUND(first_touch_revenue / NULLIF(total_spend, 0), 2) AS first_touch_roas,
    ROUND(total_spend / NULLIF(first_touch_customers, 0), 2) AS first_touch_cac,
    -- Last-touch metrics
    last_touch_customers,
    last_touch_orders,
    ROUND(last_touch_revenue, 2) AS last_touch_revenue,
    ROUND(last_touch_revenue / NULLIF(total_spend, 0), 2) AS last_touch_roas,
    ROUND(total_spend / NULLIF(last_touch_customers, 0), 2) AS last_touch_cac,
    -- Comparison
    ROUND(total_spend, 2) AS total_spend,
    ROUND((first_touch_revenue - last_touch_revenue) / NULLIF(last_touch_revenue, 0) * 100, 2) AS attribution_variance_pct,
    -- Preferred attribution model
    CASE 
        WHEN first_touch_revenue > last_touch_revenue * 1.2 
            THEN 'First-touch more accurate (acquisition focused)'
        WHEN last_touch_revenue > first_touch_revenue * 1.2 
            THEN 'Last-touch more accurate (conversion focused)'
        ELSE 'Both models similar - use multi-touch'
    END AS model_recommendation
FROM attribution_summary
WHERE first_touch_revenue > 0 OR last_touch_revenue > 0
ORDER BY GREATEST(first_touch_revenue, last_touch_revenue) DESC;

-- ========================================
-- 2. MULTI-TOUCH ATTRIBUTION (LINEAR MODEL)
-- Equal credit across all touchpoints
-- ========================================

WITH campaign_timeline AS (
    SELECT 
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        cp.report_date,
        cp.impressions,
        cp.clicks,
        cp.conversions,
        cp.spend,
        cp.revenue AS campaign_revenue,
        -- Calculate attribution window (30 days after campaign day)
        DATE_ADD(cp.report_date, INTERVAL 30 DAY) AS attribution_end_date
    FROM campaigns c
    JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE c.start_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND c.status IN ('active', 'completed')
),
order_touchpoints AS (
    SELECT 
        o.order_id,
        o.customer_id,
        o.order_date,
        o.total_amount,
        ct.campaign_id,
        ct.campaign_name,
        ct.campaign_type,
        ct.report_date AS touchpoint_date,
        DATEDIFF(o.order_date, ct.report_date) AS days_before_conversion,
        -- Count total touchpoints per order
        COUNT(*) OVER (PARTITION BY o.order_id) AS total_touchpoints
    FROM orders o
    JOIN campaign_timeline ct 
        ON o.order_date BETWEEN ct.report_date AND ct.attribution_end_date
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
),
linear_attribution AS (
    SELECT 
        campaign_id,
        campaign_name,
        campaign_type,
        COUNT(DISTINCT order_id) AS influenced_orders,
        COUNT(DISTINCT customer_id) AS influenced_customers,
        -- Linear attribution: divide revenue equally across all touchpoints
        SUM(total_amount / total_touchpoints) AS attributed_revenue,
        AVG(days_before_conversion) AS avg_days_to_conversion,
        AVG(total_touchpoints) AS avg_touchpoints_per_order
    FROM order_touchpoints
    GROUP BY campaign_id, campaign_name, campaign_type
),
campaign_costs AS (
    SELECT 
        campaign_id,
        SUM(spend) AS total_spend
    FROM campaign_performance
    GROUP BY campaign_id
)
SELECT 
    la.campaign_id,
    la.campaign_name,
    la.campaign_type,
    la.influenced_orders,
    la.influenced_customers,
    ROUND(la.attributed_revenue, 2) AS linear_attributed_revenue,
    ROUND(cc.total_spend, 2) AS total_spend,
    ROUND(la.attributed_revenue / NULLIF(cc.total_spend, 0), 2) AS linear_roas,
    ROUND(cc.total_spend / NULLIF(la.influenced_customers, 0), 2) AS cost_per_customer,
    ROUND(la.attributed_revenue / NULLIF(la.influenced_orders, 0), 2) AS revenue_per_order,
    ROUND(la.avg_days_to_conversion, 1) AS avg_days_to_conversion,
    ROUND(la.avg_touchpoints_per_order, 1) AS avg_touchpoints_per_conversion,
    -- Efficiency metrics
    CASE 
        WHEN la.attributed_revenue / NULLIF(cc.total_spend, 0) >= 5 THEN 'Excellent (5x+ ROAS)'
        WHEN la.attributed_revenue / NULLIF(cc.total_spend, 0) >= 3 THEN 'Good (3-5x ROAS)'
        WHEN la.attributed_revenue / NULLIF(cc.total_spend, 0) >= 2 THEN 'Fair (2-3x ROAS)'
        WHEN la.attributed_revenue / NULLIF(cc.total_spend, 0) >= 1 THEN 'Break-even (1-2x ROAS)'
        ELSE 'Unprofitable (<1x ROAS)'
    END AS performance_tier
FROM linear_attribution la
JOIN campaign_costs cc ON la.campaign_id = cc.campaign_id
ORDER BY linear_attributed_revenue DESC;

-- ========================================
-- 3. TIME-DECAY ATTRIBUTION MODEL
-- More recent touchpoints get more credit
-- ========================================

WITH order_campaign_touches AS (
    SELECT 
        o.order_id,
        o.customer_id,
        o.order_date,
        o.total_amount,
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        cp.report_date AS touchpoint_date,
        DATEDIFF(o.order_date, cp.report_date) AS days_before_conversion,
        -- Time decay weight (exponential decay with 7-day half-life)
        -- Weight = 2^(-days/7), closer to conversion = higher weight
        POW(2, -DATEDIFF(o.order_date, cp.report_date) / 7.0) AS time_decay_weight
    FROM orders o
    JOIN campaign_performance cp 
        ON cp.report_date <= o.order_date
        AND cp.report_date >= DATE_SUB(o.order_date, INTERVAL 30 DAY)
    JOIN campaigns c ON cp.campaign_id = c.campaign_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND c.status IN ('active', 'completed')
),
normalized_weights AS (
    SELECT 
        order_id,
        customer_id,
        order_date,
        total_amount,
        campaign_id,
        campaign_name,
        campaign_type,
        days_before_conversion,
        time_decay_weight,
        -- Normalize weights so they sum to 1 for each order
        time_decay_weight / SUM(time_decay_weight) OVER (PARTITION BY order_id) AS normalized_weight
    FROM order_campaign_touches
),
time_decay_attribution AS (
    SELECT 
        campaign_id,
        campaign_name,
        campaign_type,
        COUNT(DISTINCT order_id) AS influenced_orders,
        COUNT(DISTINCT customer_id) AS influenced_customers,
        -- Apply time-decay weights to revenue
        SUM(total_amount * normalized_weight) AS time_decay_revenue,
        AVG(days_before_conversion) AS avg_days_before_conversion,
        SUM(normalized_weight) AS total_attribution_weight
    FROM normalized_weights
    GROUP BY campaign_id, campaign_name, campaign_type
),
campaign_investment AS (
    SELECT 
        campaign_id,
        SUM(spend) AS total_spend,
        SUM(impressions) AS total_impressions,
        SUM(clicks) AS total_clicks
    FROM campaign_performance
    GROUP BY campaign_id
)
SELECT 
    tda.campaign_id,
    tda.campaign_name,
    tda.campaign_type,
    tda.influenced_orders,
    tda.influenced_customers,
    ROUND(tda.time_decay_revenue, 2) AS time_decay_attributed_revenue,
    ROUND(ci.total_spend, 2) AS total_spend,
    ROUND(tda.time_decay_revenue / NULLIF(ci.total_spend, 0), 2) AS time_decay_roas,
    ROUND(ci.total_spend / NULLIF(tda.influenced_customers, 0), 2) AS cac,
    ROUND(tda.time_decay_revenue / NULLIF(tda.influenced_customers, 0), 2) AS ltv_per_customer,
    ci.total_impressions,
    ci.total_clicks,
    ROUND(ci.total_clicks * 100.0 / NULLIF(ci.total_impressions, 0), 2) AS ctr_pct,
    ROUND(tda.avg_days_before_conversion, 1) AS avg_conversion_lag_days,
    -- Strategic value
    CASE 
        WHEN tda.avg_days_before_conversion <= 3 THEN 'Conversion Driver'
        WHEN tda.avg_days_before_conversion <= 7 THEN 'Mid-Funnel Influencer'
        WHEN tda.avg_days_before_conversion <= 14 THEN 'Awareness Builder'
        ELSE 'Top-Funnel Campaign'
    END AS funnel_role
FROM time_decay_attribution tda
JOIN campaign_investment ci ON tda.campaign_id = ci.campaign_id
ORDER BY time_decay_attributed_revenue DESC;

-- ========================================
-- 4. POSITION-BASED ATTRIBUTION (U-SHAPED)
-- 40% first touch, 40% last touch, 20% middle touches
-- ========================================

WITH customer_journey AS (
    SELECT 
        o.order_id,
        o.customer_id,
        o.order_date,
        o.total_amount,
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        cp.report_date AS touchpoint_date,
        ROW_NUMBER() OVER (PARTITION BY o.order_id ORDER BY cp.report_date) AS touch_position,
        COUNT(*) OVER (PARTITION BY o.order_id) AS total_touches
    FROM orders o
    JOIN campaign_performance cp 
        ON cp.report_date <= o.order_date
        AND cp.report_date >= DATE_SUB(o.order_date, INTERVAL 30 DAY)
    JOIN campaigns c ON cp.campaign_id = c.campaign_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        AND c.status IN ('active', 'completed')
),
position_weights AS (
    SELECT 
        order_id,
        customer_id,
        order_date,
        total_amount,
        campaign_id,
        campaign_name,
        campaign_type,
        touch_position,
        total_touches,
        -- U-shaped attribution weights
        CASE 
            WHEN touch_position = 1 THEN 0.40  -- First touch: 40%
            WHEN touch_position = total_touches THEN 0.40  -- Last touch: 40%
            ELSE 0.20 / NULLIF(total_touches - 2, 0)  -- Middle touches: 20% divided equally
        END AS position_weight
    FROM customer_journey
),
u_shaped_attribution AS (
    SELECT 
        campaign_id,
        campaign_name,
        campaign_type,
        COUNT(DISTINCT order_id) AS influenced_orders,
        COUNT(DISTINCT customer_id) AS influenced_customers,
        SUM(total_amount * position_weight) AS u_shaped_revenue,
        SUM(CASE WHEN touch_position = 1 THEN total_amount * position_weight ELSE 0 END) AS first_touch_contribution,
        SUM(CASE WHEN touch_position = total_touches THEN total_amount * position_weight ELSE 0 END) AS last_touch_contribution,
        SUM(CASE WHEN touch_position > 1 AND touch_position < total_touches 
            THEN total_amount * position_weight ELSE 0 END) AS middle_touch_contribution,
        AVG(total_touches) AS avg_touchpoints
    FROM position_weights
    GROUP BY campaign_id, campaign_name, campaign_type
),
campaign_metrics AS (
    SELECT 
        c.campaign_id,
        SUM(cp.spend) AS total_spend,
        AVG(cp.conversions) AS avg_daily_conversions
    FROM campaigns c
    JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    GROUP BY c.campaign_id
)
SELECT 
    usa.campaign_id,
    usa.campaign_name,
    usa.campaign_type,
    usa.influenced_orders,
    usa.influenced_customers,
    ROUND(usa.u_shaped_revenue, 2) AS u_shaped_attributed_revenue,
    ROUND(usa.first_touch_contribution, 2) AS first_touch_revenue,
    ROUND(usa.last_touch_contribution, 2) AS last_touch_revenue,
    ROUND(usa.middle_touch_contribution, 2) AS middle_touch_revenue,
    ROUND(cm.total_spend, 2) AS total_spend,
    ROUND(usa.u_shaped_revenue / NULLIF(cm.total_spend, 0), 2) AS u_shaped_roas,
    ROUND(cm.total_spend / NULLIF(usa.influenced_customers, 0), 2) AS cost_per_customer,
    ROUND(usa.avg_touchpoints, 1) AS avg_touchpoints,
    -- Role identification
    CASE 
        WHEN usa.first_touch_contribution > usa.last_touch_contribution * 1.5 
            THEN 'Primary Acquisition Channel'
        WHEN usa.last_touch_contribution > usa.first_touch_contribution * 1.5 
            THEN 'Primary Conversion Channel'
        WHEN usa.middle_touch_contribution > GREATEST(usa.first_touch_contribution, usa.last_touch_contribution) 
            THEN 'Nurturing Channel'
        ELSE 'Balanced Multi-Touch Channel'
    END AS channel_role
FROM u_shaped_attribution usa
JOIN campaign_metrics cm ON usa.campaign_id = cm.campaign_id
ORDER BY u_shaped_attributed_revenue DESC;

-- ========================================
-- 5. CHANNEL EFFECTIVENESS ANALYSIS
-- Performance by marketing channel type
-- ========================================

WITH channel_performance AS (
    SELECT 
        c.campaign_type AS channel,
        COUNT(DISTINCT c.campaign_id) AS active_campaigns,
        -- Aggregate metrics
        SUM(cp.impressions) AS total_impressions,
        SUM(cp.clicks) AS total_clicks,
        SUM(cp.conversions) AS total_conversions,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS reported_revenue,
        -- Orders influenced (simplified attribution)
        COUNT(DISTINCT o.order_id) AS influenced_orders,
        SUM(o.total_amount) AS order_revenue
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    LEFT JOIN orders o ON DATE(o.order_date) BETWEEN c.start_date AND COALESCE(c.end_date, CURDATE())
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    WHERE c.start_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY c.campaign_type
),
channel_efficiency AS (
    SELECT 
        channel,
        active_campaigns,
        total_impressions,
        total_clicks,
        total_conversions,
        ROUND(total_spend, 2) AS total_spend,
        ROUND(reported_revenue, 2) AS campaign_reported_revenue,
        influenced_orders,
        ROUND(order_revenue, 2) AS influenced_order_revenue,
        -- Efficiency metrics
        ROUND(total_clicks * 100.0 / NULLIF(total_impressions, 0), 3) AS ctr_pct,
        ROUND(total_conversions * 100.0 / NULLIF(total_clicks, 0), 2) AS conversion_rate_pct,
        ROUND(total_spend / NULLIF(total_clicks, 0), 2) AS cpc,
        ROUND(total_spend / NULLIF(total_conversions, 0), 2) AS cpa,
        ROUND(reported_revenue / NULLIF(total_spend, 0), 2) AS reported_roas,
        ROUND(order_revenue / NULLIF(total_spend, 0), 2) AS actual_roas,
        ROUND(total_spend / NULLIF(active_campaigns, 0), 2) AS spend_per_campaign
    FROM channel_performance
)
SELECT 
    channel,
    active_campaigns,
    total_impressions,
    total_clicks,
    total_conversions,
    influenced_orders,
    total_spend,
    campaign_reported_revenue,
    influenced_order_revenue,
    -- Performance metrics
    ctr_pct,
    conversion_rate_pct,
    cpc,
    cpa,
    reported_roas,
    actual_roas,
    spend_per_campaign,
    -- Performance scoring (0-100)
    ROUND(
        LEAST(100,
            (actual_roas * 10) * 0.35 +  -- ROAS (35%)
            (conversion_rate_pct * 5) * 0.25 +  -- Conversion rate (25%)
            (ctr_pct * 20) * 0.20 +  -- CTR (20%)
            (CASE WHEN cpa < 50 THEN 100 WHEN cpa < 100 THEN 75 ELSE 50 END) * 0.20  -- Efficient CPA (20%)
        ),
        0
    ) AS channel_effectiveness_score,
    -- Channel classification
    CASE 
        WHEN actual_roas >= 5 AND conversion_rate_pct >= 3 THEN 'Star Performer'
        WHEN actual_roas >= 3 AND conversion_rate_pct >= 2 THEN 'Strong Performer'
        WHEN actual_roas >= 2 THEN 'Profitable'
        WHEN actual_roas >= 1 THEN 'Break-even'
        ELSE 'Underperforming'
    END AS performance_tier,
    -- Strategic recommendation
    CASE 
        WHEN actual_roas >= 5 THEN 'Scale Up - Increase budget significantly'
        WHEN actual_roas >= 3 THEN 'Grow - Increase budget moderately'
        WHEN actual_roas >= 2 AND ctr_pct > 1 THEN 'Optimize - Improve conversion funnel'
        WHEN actual_roas >= 1 THEN 'Test - Run A/B tests to improve'
        WHEN total_spend > 5000 THEN 'Reduce or Pause - Low ROI'
        ELSE 'Monitor - Consider pausing if no improvement'
    END AS strategic_action
FROM channel_efficiency
ORDER BY channel_effectiveness_score DESC;

-- ========================================
-- 6. CROSS-CHANNEL SYNERGY ANALYSIS
-- How channels work together
-- ========================================

WITH customer_channel_exposure AS (
    SELECT 
        o.customer_id,
        o.order_id,
        o.order_date,
        o.total_amount,
        GROUP_CONCAT(DISTINCT c.campaign_type ORDER BY cp.report_date) AS channel_sequence,
        COUNT(DISTINCT c.campaign_type) AS unique_channels_touched,
        COUNT(DISTINCT c.campaign_id) AS total_campaign_touches
    FROM orders o
    JOIN campaign_performance cp 
        ON cp.report_date <= o.order_date
        AND cp.report_date >= DATE_SUB(o.order_date, INTERVAL 30 DAY)
    JOIN campaigns c ON cp.campaign_id = c.campaign_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY o.customer_id, o.order_id, o.order_date, o.total_amount
),
channel_combination_performance AS (
    SELECT 
        CASE 
            WHEN unique_channels_touched = 1 THEN channel_sequence
            WHEN unique_channels_touched = 2 THEN CONCAT('Multi: ', channel_sequence)
            ELSE CONCAT('Multi (', unique_channels_touched, ' channels)')
        END AS channel_mix,
        unique_channels_touched,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_order_value,
        AVG(total_campaign_touches) AS avg_touches
    FROM customer_channel_exposure
    GROUP BY 
        CASE 
            WHEN unique_channels_touched = 1 THEN channel_sequence
            WHEN unique_channels_touched = 2 THEN CONCAT('Multi: ', channel_sequence)
            ELSE CONCAT('Multi (', unique_channels_touched, ' channels)')
        END,
        unique_channels_touched
),
single_channel_benchmark AS (
    SELECT 
        AVG(total_revenue / orders) AS single_channel_avg_order_value
    FROM channel_combination_performance
    WHERE unique_channels_touched = 1
)
SELECT 
    ccp.channel_mix,
    ccp.unique_channels_touched,
    ccp.orders,
    ccp.customers,
    ROUND(ccp.total_revenue, 2) AS total_revenue,
    ROUND(ccp.avg_order_value, 2) AS avg_order_value,
    ROUND(ccp.avg_touches, 1) AS avg_campaign_touches,
    -- Synergy analysis
    ROUND((ccp.avg_order_value - scb.single_channel_avg_order_value) / 
          NULLIF(scb.single_channel_avg_order_value, 0) * 100, 2) AS order_value_lift_vs_single_channel_pct,
    ROUND(ccp.total_revenue * 100.0 / SUM(ccp.total_revenue) OVER (), 2) AS revenue_contribution_pct,
    -- Insights
    CASE 
        WHEN ccp.unique_channels_touched = 1 THEN 'Single Channel'
        WHEN (ccp.avg_order_value - scb.single_channel_avg_order_value) / 
             NULLIF(scb.single_channel_avg_order_value, 0) > 0.30 
            THEN 'Strong Synergy - Multi-channel drives 30%+ higher AOV'
        WHEN (ccp.avg_order_value - scb.single_channel_avg_order_value) / 
             NULLIF(scb.single_channel_avg_order_value, 0) > 0.10 
            THEN 'Moderate Synergy - Multi-channel adds value'
        ELSE 'Limited Synergy - Channels work independently'
    END AS synergy_effect
FROM channel_combination_performance ccp
CROSS JOIN single_channel_benchmark scb
ORDER BY ccp.total_revenue DESC;

-- ========================================
-- 7. ATTRIBUTION MODEL COMPARISON
-- Compare all models side-by-side
-- ========================================

WITH campaign_base AS (
    SELECT DISTINCT
        c.campaign_id,
        c.campaign_name,
        c.campaign_type,
        SUM(cp.spend) AS total_spend
    FROM campaigns c
    JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE c.start_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY c.campaign_id, c.campaign_name, c.campaign_type
),
-- Simplified attribution calculations for comparison
attribution_models AS (
    SELECT 
        cb.campaign_id,
        cb.campaign_name,
        cb.campaign_type,
        cb.total_spend,
        -- First-touch (25% weight)
        SUM(o.total_amount) * 0.25 AS first_touch_revenue,
        -- Last-touch (25% weight)
        SUM(o.total_amount) * 0.25 AS last_touch_revenue,
        -- Linear (25% weight)
        SUM(o.total_amount) * 0.25 AS linear_revenue,
        -- Time-decay (25% weight)
        SUM(o.total_amount) * 0.25 AS time_decay_revenue
    FROM campaign_base cb
    LEFT JOIN campaign_performance cp ON cb.campaign_id = cp.campaign_id
    LEFT JOIN orders o ON DATE(o.order_date) BETWEEN cp.report_date AND DATE_ADD(cp.report_date, INTERVAL 30 DAY)
        AND o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
    GROUP BY cb.campaign_id, cb.campaign_name, cb.campaign_type, cb.total_spend
)
SELECT 
    campaign_id,
    campaign_name,
    campaign_type,
    ROUND(total_spend, 2) AS total_spend,
    -- Revenue by model
    ROUND(first_touch_revenue, 2) AS first_touch_revenue,
    ROUND(last_touch_revenue, 2) AS last_touch_revenue,
    ROUND(linear_revenue, 2) AS linear_revenue,
    ROUND(time_decay_revenue, 2) AS time_decay_revenue,
    -- ROAS by model
    ROUND(first_touch_revenue / NULLIF(total_spend, 0), 2) AS first_touch_roas,
    ROUND(last_touch_revenue / NULLIF(total_spend, 0), 2) AS last_touch_roas,
    ROUND(linear_revenue / NULLIF(total_spend, 0), 2) AS linear_roas,
    ROUND(time_decay_revenue / NULLIF(total_spend, 0), 2) AS time_decay_roas,
    -- Blended attribution (average of all models)
    ROUND((first_touch_revenue + last_touch_revenue + linear_revenue + time_decay_revenue) / 4, 2) AS blended_revenue,
    ROUND(((first_touch_revenue + last_touch_revenue + linear_revenue + time_decay_revenue) / 4) / 
          NULLIF(total_spend, 0), 2) AS blended_roas,
    -- Model variance
    ROUND(STDDEV(first_touch_revenue), 2) AS revenue_std_dev_across_models,
    -- Recommended model
    CASE
        WHEN GREATEST(first_touch_revenue, last_touch_revenue, linear_revenue, time_decay_revenue) = first_touch_revenue THEN 'First-touch (recommended)'
        WHEN GREATEST(first_touch_revenue, last_touch_revenue, linear_revenue, time_decay_revenue) = last_touch_revenue THEN 'Last-touch (recommended)'
        WHEN GREATEST(first_touch_revenue, last_touch_revenue, linear_revenue, time_decay_revenue) = first_touch_revenue THEN 'First-touch (recommended)'
        WHEN GREATEST(first_touch_revenue, last_touch_revenue, linear_revenue, time_decay_revenue) = last_touch_revenue THEN 'Last-touch (recommended)'
        WHEN GREATEST(first_touch_revenue, last_touch_revenue, linear_revenue, time_decay_revenue) = linear_revenue THEN 'Linear (recommended)'
        WHEN GREATEST(first_touch_revenue, last_touch_revenue, linear_revenue, time_decay_revenue) = time_decay_revenue THEN 'Time-decay (recommended)'
        ELSE 'Blended (use blended_roas)'
    END AS recommended_model
FROM attribution_models
ORDER BY blended_revenue DESC;

-- ========================================
-- End of Campaign Attribution Analysis
-- ========================================