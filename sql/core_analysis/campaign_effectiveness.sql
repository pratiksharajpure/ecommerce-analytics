-- ========================================
-- CAMPAIGN EFFECTIVENESS ANALYSIS
-- Day 6-7: Order & Transaction Queries
-- Data Validation & Performance Metrics
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CAMPAIGN DATA QUALITY VALIDATION
-- ========================================

-- Check for invalid date ranges
SELECT 
    campaign_id,
    campaign_name,
    campaign_type,
    start_date,
    end_date,
    DATEDIFF(end_date, start_date) AS duration_days,
    status,
    budget
FROM campaigns
WHERE end_date < start_date
   OR start_date IS NULL
   OR end_date IS NULL
   OR DATEDIFF(end_date, start_date) > 365
ORDER BY campaign_id;

-- ========================================
-- 2. CAMPAIGNS WITH MISSING PERFORMANCE DATA
-- ========================================

SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.start_date,
    c.end_date,
    c.status,
    COUNT(cp.performance_id) AS performance_records,
    DATEDIFF(COALESCE(c.end_date, CURDATE()), c.start_date) AS expected_days
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
WHERE c.status IN ('active', 'completed')
  AND c.start_date <= CURDATE()
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.start_date, c.end_date, c.status
HAVING COUNT(cp.performance_id) = 0
   OR COUNT(cp.performance_id) < DATEDIFF(COALESCE(c.end_date, CURDATE()), c.start_date);

-- ========================================
-- 3. SUSPICIOUS CAMPAIGN METRICS
-- ========================================

-- Campaigns with impossible click-through rates or conversion rates
SELECT 
    cp.campaign_id,
    c.campaign_name,
    cp.report_date,
    cp.impressions,
    cp.clicks,
    cp.conversions,
    CASE 
        WHEN cp.impressions > 0 THEN ROUND((cp.clicks / cp.impressions) * 100, 2)
        ELSE 0 
    END AS ctr_percent,
    CASE 
        WHEN cp.clicks > 0 THEN ROUND((cp.conversions / cp.clicks) * 100, 2)
        ELSE 0 
    END AS conversion_rate_percent,
    cp.spend,
    cp.revenue,
    CASE 
        WHEN cp.spend > 0 THEN ROUND((cp.revenue / cp.spend), 2)
        ELSE 0 
    END AS roi_ratio
FROM campaign_performance cp
JOIN campaigns c ON cp.campaign_id = c.campaign_id
WHERE cp.clicks > cp.impressions  -- Impossible: more clicks than impressions
   OR cp.conversions > cp.clicks   -- Impossible: more conversions than clicks
   OR (cp.impressions > 0 AND (cp.clicks / cp.impressions) > 0.5)  -- CTR > 50% suspicious
   OR (cp.clicks > 0 AND (cp.conversions / cp.clicks) > 0.8)       -- Conv rate > 80% suspicious
   OR cp.spend < 0
   OR cp.revenue < 0
ORDER BY cp.report_date DESC, cp.campaign_id;

-- ========================================
-- 4. BUDGET OVERRUN DETECTION
-- ========================================

SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.budget AS allocated_budget,
    SUM(cp.spend) AS total_spent,
    c.budget - SUM(cp.spend) AS remaining_budget,
    ROUND((SUM(cp.spend) / NULLIF(c.budget, 0)) * 100, 2) AS budget_utilization_percent,
    CASE 
        WHEN SUM(cp.spend) > c.budget THEN 'OVERRUN'
        WHEN SUM(cp.spend) > c.budget * 0.9 THEN 'WARNING'
        ELSE 'OK'
    END AS budget_status,
    c.status AS campaign_status
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
WHERE c.budget IS NOT NULL
  AND c.budget > 0
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.budget, c.status
HAVING SUM(cp.spend) > c.budget * 0.85  -- Show campaigns using 85%+ of budget
ORDER BY budget_utilization_percent DESC;

-- ========================================
-- 5. CAMPAIGN PERFORMANCE ANOMALIES
-- ========================================

-- Daily performance with significant deviations from campaign average
WITH campaign_averages AS (
    SELECT 
        campaign_id,
        AVG(clicks) AS avg_clicks,
        STDDEV(clicks) AS stddev_clicks,
        AVG(conversions) AS avg_conversions,
        STDDEV(conversions) AS stddev_conversions,
        AVG(spend) AS avg_spend,
        STDDEV(spend) AS stddev_spend
    FROM campaign_performance
    GROUP BY campaign_id
)
SELECT 
    cp.campaign_id,
    c.campaign_name,
    cp.report_date,
    cp.clicks,
    ca.avg_clicks,
    ROUND((cp.clicks - ca.avg_clicks) / NULLIF(ca.stddev_clicks, 0), 2) AS clicks_z_score,
    cp.conversions,
    ca.avg_conversions,
    ROUND((cp.conversions - ca.avg_conversions) / NULLIF(ca.stddev_conversions, 0), 2) AS conversions_z_score,
    cp.spend,
    ca.avg_spend,
    ROUND((cp.spend - ca.avg_spend) / NULLIF(ca.stddev_spend, 0), 2) AS spend_z_score
FROM campaign_performance cp
JOIN campaigns c ON cp.campaign_id = c.campaign_id
JOIN campaign_averages ca ON cp.campaign_id = ca.campaign_id
WHERE ABS((cp.clicks - ca.avg_clicks) / NULLIF(ca.stddev_clicks, 0)) > 2
   OR ABS((cp.conversions - ca.avg_conversions) / NULLIF(ca.stddev_conversions, 0)) > 2
   OR ABS((cp.spend - ca.avg_spend) / NULLIF(ca.stddev_spend, 0)) > 2
ORDER BY cp.report_date DESC, cp.campaign_id;

-- ========================================
-- 6. ZERO PERFORMANCE CAMPAIGNS
-- ========================================

SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.start_date,
    c.end_date,
    c.budget,
    c.status,
    COUNT(cp.performance_id) AS total_reports,
    SUM(cp.impressions) AS total_impressions,
    SUM(cp.clicks) AS total_clicks,
    SUM(cp.conversions) AS total_conversions,
    SUM(cp.spend) AS total_spend,
    SUM(cp.revenue) AS total_revenue
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
WHERE c.status IN ('active', 'completed')
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.start_date, c.end_date, c.budget, c.status
HAVING (SUM(cp.impressions) = 0 OR SUM(cp.impressions) IS NULL)
   AND (SUM(cp.clicks) = 0 OR SUM(cp.clicks) IS NULL)
   AND DATEDIFF(COALESCE(c.end_date, CURDATE()), c.start_date) > 7  -- Running for more than 7 days
ORDER BY c.start_date DESC;

-- ========================================
-- 7. CAMPAIGN ROI ANALYSIS WITH VALIDATION
-- ========================================

SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    SUM(cp.spend) AS total_spend,
    SUM(cp.revenue) AS total_revenue,
    SUM(cp.revenue) - SUM(cp.spend) AS net_profit,
    CASE 
        WHEN SUM(cp.spend) > 0 THEN ROUND(((SUM(cp.revenue) - SUM(cp.spend)) / SUM(cp.spend)) * 100, 2)
        ELSE NULL
    END AS roi_percent,
    SUM(cp.impressions) AS total_impressions,
    SUM(cp.clicks) AS total_clicks,
    SUM(cp.conversions) AS total_conversions,
    CASE 
        WHEN SUM(cp.impressions) > 0 THEN ROUND((SUM(cp.clicks) / SUM(cp.impressions)) * 100, 2)
        ELSE 0
    END AS ctr_percent,
    CASE 
        WHEN SUM(cp.clicks) > 0 THEN ROUND((SUM(cp.conversions) / SUM(cp.clicks)) * 100, 2)
        ELSE 0
    END AS conversion_rate_percent,
    CASE 
        WHEN SUM(cp.conversions) > 0 THEN ROUND(SUM(cp.spend) / SUM(cp.conversions), 2)
        ELSE NULL
    END AS cost_per_conversion,
    c.status
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.status
HAVING SUM(cp.spend) > 0
ORDER BY roi_percent DESC;

-- ========================================
-- 8. DUPLICATE PERFORMANCE RECORDS CHECK
-- ========================================

SELECT 
    campaign_id,
    report_date,
    COUNT(*) AS duplicate_count,
    GROUP_CONCAT(performance_id ORDER BY performance_id) AS performance_ids
FROM campaign_performance
GROUP BY campaign_id, report_date
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, campaign_id, report_date;

-- ========================================
-- 9. CAMPAIGN TYPE EFFECTIVENESS COMPARISON
-- ========================================

SELECT 
    c.campaign_type,
    COUNT(DISTINCT c.campaign_id) AS total_campaigns,
    SUM(cp.spend) AS total_spend,
    SUM(cp.revenue) AS total_revenue,
    ROUND(AVG(CASE WHEN cp.spend > 0 THEN (cp.revenue - cp.spend) / cp.spend * 100 END), 2) AS avg_roi_percent,
    SUM(cp.impressions) AS total_impressions,
    SUM(cp.clicks) AS total_clicks,
    SUM(cp.conversions) AS total_conversions,
    ROUND(AVG(CASE WHEN cp.impressions > 0 THEN (cp.clicks / cp.impressions) * 100 END), 2) AS avg_ctr_percent,
    ROUND(AVG(CASE WHEN cp.clicks > 0 THEN (cp.conversions / cp.clicks) * 100 END), 2) AS avg_conversion_rate_percent
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
WHERE c.status IN ('active', 'completed')
GROUP BY c.campaign_type
ORDER BY total_revenue DESC;

-- ========================================
-- 10. RECENT CAMPAIGNS WITH NO RECENT DATA
-- ========================================

SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.status,
    c.start_date,
    c.end_date,
    MAX(cp.report_date) AS last_report_date,
    DATEDIFF(CURDATE(), MAX(cp.report_date)) AS days_since_last_report
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
WHERE c.status = 'active'
  AND c.start_date <= CURDATE()
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.status, c.start_date, c.end_date
HAVING MAX(cp.report_date) IS NULL 
    OR DATEDIFF(CURDATE(), MAX(cp.report_date)) > 7
ORDER BY days_since_last_report DESC;

-- ========================================
-- END OF CAMPAIGN EFFECTIVENESS ANALYSIS
-- ========================================