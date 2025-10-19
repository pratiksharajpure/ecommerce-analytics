-- ========================================
-- INSERT SAMPLE CAMPAIGNS & PERFORMANCE DATA
-- 20 marketing campaigns with daily performance metrics
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- CAMPAIGNS (20 records)
-- ========================================

INSERT INTO campaigns (campaign_name, campaign_type, start_date, end_date, budget, status, created_at) VALUES
('Spring Sale 2024', 'email', '2024-03-01', '2024-03-31', 15000.00, 'completed', '2024-02-15'),
('Summer Electronics Blowout', 'ppc', '2024-06-01', '2024-06-30', 25000.00, 'completed', '2024-05-20'),
('Back to School Campaign', 'social_media', '2024-08-01', '2024-08-31', 18000.00, 'completed', '2024-07-15'),
('Black Friday Mega Sale', 'ppc', '2024-11-15', '2024-11-30', 50000.00, 'completed', '2024-10-01'),
('Cyber Monday Deals', 'email', '2024-11-28', '2024-12-02', 30000.00, 'completed', '2024-11-01'),
('Holiday Gift Guide', 'social_media', '2024-12-01', '2024-12-24', 35000.00, 'completed', '2024-11-10'),
('New Year Fitness Resolution', 'ppc', '2024-12-26', '2025-01-31', 22000.00, 'active', '2024-12-15'),
('Valentine''s Day Special', 'email', '2025-02-01', '2025-02-14', 12000.00, 'active', '2025-01-15'),
('Spring Fashion Week', 'display', '2024-04-01', '2024-04-30', 20000.00, 'completed', '2024-03-10'),
('Summer Outdoor Adventure', 'affiliate', '2024-05-15', '2024-07-15', 15000.00, 'completed', '2024-04-25'),
('Fall Home Refresh', 'social_media', '2024-09-01', '2024-09-30', 16000.00, 'completed', '2024-08-15'),
('Tech Tuesday Weekly Deals', 'email', '2024-01-02', '2024-12-31', 40000.00, 'active', '2023-12-15'),
('Instagram Influencer Partnership', 'social_media', '2024-07-01', '2024-12-31', 28000.00, 'active', '2024-06-10'),
('Google Shopping Feed Optimization', 'ppc', '2024-01-01', '2024-12-31', 45000.00, 'active', '2023-12-20'),
('Loyalty Program Launch', 'email', '2024-02-01', '2024-02-29', 8000.00, 'completed', '2024-01-20'),
('Mobile App Download Campaign', 'display', '2024-03-15', '2024-05-15', 25000.00, 'completed', '2024-03-01'),
('Customer Win-Back Email Series', 'email', '2024-06-01', '2024-08-31', 10000.00, 'completed', '2024-05-15'),
('TikTok Brand Awareness', 'social_media', '2024-08-15', '2024-10-15', 20000.00, 'completed', '2024-08-01'),
('Holiday Retargeting Campaign', 'display', '2024-11-01', '2024-12-31', 32000.00, 'active', '2024-10-20'),
('Q1 2025 Product Launch', 'ppc', '2025-01-15', '2025-03-31', 35000.00, 'active', '2025-01-05');

-- ========================================
-- CAMPAIGN PERFORMANCE DATA
-- Generate daily performance metrics for completed campaigns
-- ========================================

-- Spring Sale 2024 (31 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    1 AS campaign_id,
    DATE_ADD('2024-03-01', INTERVAL seq DAY) AS report_date,
    FLOOR(8000 + (RAND() * 4000)) AS impressions,
    FLOOR(400 + (RAND() * 300)) AS clicks,
    FLOOR(15 + (RAND() * 25)) AS conversions,
    ROUND(400 + (RAND() * 200), 2) AS spend,
    ROUND(1500 + (RAND() * 2500), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT @row := -1) r
) numbers
WHERE seq < 29;

-- Mobile App Download Campaign (62 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    16 AS campaign_id,
    DATE_ADD('2024-03-15', INTERVAL seq DAY) AS report_date,
    FLOOR(20000 + (RAND() * 10000)) AS impressions,
    FLOOR(800 + (RAND() * 500)) AS clicks,
    FLOOR(30 + (RAND() * 40)) AS conversions,
    ROUND(400 + (RAND() * 200), 2) AS spend,
    ROUND(2500 + (RAND() * 3500), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 62;

-- Customer Win-Back Email Series (92 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    17 AS campaign_id,
    DATE_ADD('2024-06-01', INTERVAL seq DAY) AS report_date,
    FLOOR(4000 + (RAND() * 2000)) AS impressions,
    FLOOR(250 + (RAND() * 150)) AS clicks,
    FLOOR(15 + (RAND() * 20)) AS conversions,
    ROUND(100 + (RAND() * 60), 2) AS spend,
    ROUND(1200 + (RAND() * 1800), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
          UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 92;

-- TikTok Brand Awareness (62 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    18 AS campaign_id,
    DATE_ADD('2024-08-15', INTERVAL seq DAY) AS report_date,
    FLOOR(25000 + (RAND() * 15000)) AS impressions,
    FLOOR(600 + (RAND() * 400)) AS clicks,
    FLOOR(12 + (RAND() * 18)) AS conversions,
    ROUND(310 + (RAND() * 190), 2) AS spend,
    ROUND(1500 + (RAND() * 2500), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 62;

-- ========================================
-- ACTIVE CAMPAIGN PERFORMANCE (Recent data)
-- ========================================

-- New Year Fitness Resolution (Campaign 7) - Recent 30 days
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    7 AS campaign_id,
    DATE_SUB(CURDATE(), INTERVAL (29 - seq) DAY) AS report_date,
    FLOOR(13000 + (RAND() * 7000)) AS impressions,
    FLOOR(650 + (RAND() * 450)) AS clicks,
    FLOOR(28 + (RAND() * 32)) AS conversions,
    ROUND(700 + (RAND() * 350), 2) AS spend,
    ROUND(3500 + (RAND() * 4500), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 30;

-- Tech Tuesday Weekly Deals (Campaign 12) - Ongoing, recent 60 days
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    12 AS campaign_id,
    DATE_SUB(CURDATE(), INTERVAL (59 - seq) DAY) AS report_date,
    FLOOR(7000 + (RAND() * 3000)) AS impressions,
    FLOOR(350 + (RAND() * 200)) AS clicks,
    FLOOR(18 + (RAND() * 22)) AS conversions,
    ROUND(110 + (RAND() * 60), 2) AS spend,
    ROUND(1800 + (RAND() * 2200), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 60;

-- Instagram Influencer Partnership (Campaign 13) - Recent 45 days
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    13 AS campaign_id,
    DATE_SUB(CURDATE(), INTERVAL (44 - seq) DAY) AS report_date,
    FLOOR(18000 + (RAND() * 10000)) AS impressions,
    FLOOR(550 + (RAND() * 350)) AS clicks,
    FLOOR(14 + (RAND() * 18)) AS conversions,
    ROUND(600 + (RAND() * 300), 2) AS spend,
    ROUND(2200 + (RAND() * 2800), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 45;

-- Google Shopping Feed Optimization (Campaign 14) - Recent 60 days
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    14 AS campaign_id,
    DATE_SUB(CURDATE(), INTERVAL (59 - seq) DAY) AS report_date,
    FLOOR(22000 + (RAND() * 12000)) AS impressions,
    FLOOR(1100 + (RAND() * 700)) AS clicks,
    FLOOR(55 + (RAND() * 65)) AS conversions,
    ROUND(1400 + (RAND() * 700), 2) AS spend,
    ROUND(6500 + (RAND() * 8500), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 60;

-- Holiday Retargeting Campaign (Campaign 19) - Recent 30 days
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    19 AS campaign_id,
    DATE_SUB(CURDATE(), INTERVAL (29 - seq) DAY) AS report_date,
    FLOOR(16000 + (RAND() * 8000)) AS impressions,
    FLOOR(900 + (RAND() * 600)) AS clicks,
    FLOOR(42 + (RAND() * 48)) AS conversions,
    ROUND(1000 + (RAND() * 500), 2) AS spend,
    ROUND(5200 + (RAND() * 6800), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 30;

-- Q1 2025 Product Launch (Campaign 20) - Recent 15 days
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    20 AS campaign_id,
    DATE_SUB(CURDATE(), INTERVAL (14 - seq) DAY) AS report_date,
    FLOOR(19000 + (RAND() * 11000)) AS impressions,
    FLOOR(1000 + (RAND() * 800)) AS clicks,
    FLOOR(48 + (RAND() * 62)) AS conversions,
    ROUND(1600 + (RAND() * 800), 2) AS spend,
    ROUND(6800 + (RAND() * 9200), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 15;

-- ========================================
-- DISPLAY CONFIRMATION & STATISTICS
-- ========================================

SELECT 'Campaigns and performance data inserted successfully!' AS Status;

SELECT 
    COUNT(*) AS total_campaigns,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_campaigns,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_campaigns,
    SUM(CASE WHEN status = 'paused' THEN 1 ELSE 0 END) AS paused_campaigns,
    ROUND(SUM(budget), 2) AS total_budget
FROM campaigns;

SELECT 'Campaign performance summary:' AS Info;
SELECT 
    COUNT(*) AS total_performance_records,
    SUM(impressions) AS total_impressions,
    SUM(clicks) AS total_clicks,
    SUM(conversions) AS total_conversions,
    ROUND(SUM(spend), 2) AS total_spend,
    ROUND(SUM(revenue), 2) AS total_revenue,
    ROUND((SUM(revenue) - SUM(spend)) / SUM(spend) * 100, 2) AS roi_percentage,
    ROUND(SUM(clicks) / SUM(impressions) * 100, 2) AS avg_ctr,
    ROUND(SUM(conversions) / SUM(clicks) * 100, 2) AS avg_conversion_rate
FROM campaign_performance;

SELECT 'Top 5 campaigns by ROI:' AS Info;
SELECT 
    c.campaign_name,
    c.campaign_type,
    ROUND(SUM(cp.revenue), 2) AS total_revenue,
    ROUND(SUM(cp.spend), 2) AS total_spend,
    ROUND((SUM(cp.revenue) - SUM(cp.spend)) / SUM(cp.spend) * 100, 2) AS roi_percentage,
    SUM(cp.conversions) AS total_conversions
FROM campaigns c
JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type
HAVING total_spend > 0
ORDER BY roi_percentage DESC
LIMIT 5;

SELECT 'Campaign performance by type:' AS Info;
SELECT 
    c.campaign_type,
    COUNT(DISTINCT c.campaign_id) AS campaign_count,
    ROUND(AVG(cp.clicks / cp.impressions * 100), 2) AS avg_ctr,
    ROUND(AVG(cp.conversions / cp.clicks * 100), 2) AS avg_conversion_rate,
    ROUND(SUM(cp.revenue) / SUM(cp.spend), 2) AS roas
FROM campaigns c
JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
GROUP BY c.campaign_type
ORDER BY roas DESC;SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 31;

-- Summer Electronics Blowout (30 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    2 AS campaign_id,
    DATE_ADD('2024-06-01', INTERVAL seq DAY) AS report_date,
    FLOOR(15000 + (RAND() * 8000)) AS impressions,
    FLOOR(750 + (RAND() * 500)) AS clicks,
    FLOOR(35 + (RAND() * 40)) AS conversions,
    ROUND(750 + (RAND() * 350), 2) AS spend,
    ROUND(3500 + (RAND() * 4500), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 30;

-- Back to School Campaign (31 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    3 AS campaign_id,
    DATE_ADD('2024-08-01', INTERVAL seq DAY) AS report_date,
    FLOOR(12000 + (RAND() * 6000)) AS impressions,
    FLOOR(500 + (RAND() * 400)) AS clicks,
    FLOOR(20 + (RAND() * 30)) AS conversions,
    ROUND(550 + (RAND() * 250), 2) AS spend,
    ROUND(2200 + (RAND() * 3300), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 31;

-- Black Friday Mega Sale (16 days) - Higher performance
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    4 AS campaign_id,
    DATE_ADD('2024-11-15', INTERVAL seq DAY) AS report_date,
    FLOOR(35000 + (RAND() * 15000) + IF(seq IN (9, 10, 11), 20000, 0)) AS impressions,
    FLOOR(2500 + (RAND() * 1500) + IF(seq IN (9, 10, 11), 3000, 0)) AS clicks,
    FLOOR(120 + (RAND() * 100) + IF(seq IN (9, 10, 11), 150, 0)) AS conversions,
    ROUND(2800 + (RAND() * 1200), 2) AS spend,
    ROUND(15000 + (RAND() * 12000) + IF(seq IN (9, 10, 11), 20000, 0), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 16;

-- Cyber Monday Deals (5 days) - Very high performance
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    5 AS campaign_id,
    DATE_ADD('2024-11-28', INTERVAL seq DAY) AS report_date,
    FLOOR(45000 + (RAND() * 20000)) AS impressions,
    FLOOR(3500 + (RAND() * 2000)) AS clicks,
    FLOOR(180 + (RAND() * 150)) AS conversions,
    ROUND(5500 + (RAND() * 1500), 2) AS spend,
    ROUND(22000 + (RAND() * 15000), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4) t1,
         (SELECT @row := -1) r
) numbers
WHERE seq < 5;

-- Holiday Gift Guide (24 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    6 AS campaign_id,
    DATE_ADD('2024-12-01', INTERVAL seq DAY) AS report_date,
    FLOOR(18000 + (RAND() * 9000)) AS impressions,
    FLOOR(900 + (RAND() * 600)) AS clicks,
    FLOOR(45 + (RAND() * 55)) AS conversions,
    ROUND(1400 + (RAND() * 600), 2) AS spend,
    ROUND(5500 + (RAND() * 6500), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 24;

-- Spring Fashion Week (30 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    9 AS campaign_id,
    DATE_ADD('2024-04-01', INTERVAL seq DAY) AS report_date,
    FLOOR(10000 + (RAND() * 5000)) AS impressions,
    FLOOR(350 + (RAND() * 250)) AS clicks,
    FLOOR(18 + (RAND() * 22)) AS conversions,
    ROUND(650 + (RAND() * 300), 2) AS spend,
    ROUND(2800 + (RAND() * 3200), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 30;

-- Summer Outdoor Adventure (62 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    10 AS campaign_id,
    DATE_ADD('2024-05-15', INTERVAL seq DAY) AS report_date,
    FLOOR(6000 + (RAND() * 3000)) AS impressions,
    FLOOR(200 + (RAND() * 150)) AS clicks,
    FLOOR(8 + (RAND() * 12)) AS conversions,
    ROUND(230 + (RAND() * 120), 2) AS spend,
    ROUND(1200 + (RAND() * 1800), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 62;

-- Fall Home Refresh (30 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    11 AS campaign_id,
    DATE_ADD('2024-09-01', INTERVAL seq DAY) AS report_date,
    FLOOR(11000 + (RAND() * 5000)) AS impressions,
    FLOOR(450 + (RAND() * 300)) AS clicks,
    FLOOR(22 + (RAND() * 28)) AS conversions,
    ROUND(520 + (RAND() * 280), 2) AS spend,
    ROUND(3000 + (RAND() * 3500), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (SELECT @row := -1) r
) numbers
WHERE seq < 30;

-- Loyalty Program Launch (29 days)
INSERT INTO campaign_performance (campaign_id, report_date, impressions, clicks, conversions, spend, revenue)
SELECT 
    15 AS campaign_id,
    DATE_ADD('2024-02-01', INTERVAL seq DAY) AS report_date,
    FLOOR(5000 + (RAND() * 2000)) AS impressions,
    FLOOR(300 + (RAND() * 200)) AS clicks,
    FLOOR(25 + (RAND() * 35)) AS conversions,
    ROUND(270 + (RAND() * 130), 2) AS spend,
    ROUND(1800 + (RAND() * 2200), 2) AS revenue
FROM (
    SELECT @row := @row + 1 as seq
    FROM (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
         (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) t2,
         (