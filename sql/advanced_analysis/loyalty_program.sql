-- ========================================
-- LOYALTY PROGRAM ANALYSIS
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. LOYALTY PROGRAM OVERVIEW
-- Key metrics and program health
-- ========================================
WITH member_stats AS (
    SELECT 
        lp.tier,
        COUNT(DISTINCT lp.customer_id) AS member_count,
        SUM(lp.points_balance) AS total_points_outstanding,
        SUM(lp.points_earned_lifetime) AS total_points_earned,
        SUM(lp.points_redeemed_lifetime) AS total_points_redeemed,
        AVG(lp.points_balance) AS avg_points_balance,
        AVG(DATEDIFF(CURDATE(), lp.joined_date)) AS avg_membership_days
    FROM loyalty_program lp
    JOIN customers c ON lp.customer_id = c.customer_id
    WHERE c.status = 'active'
    GROUP BY lp.tier
),
member_revenue AS (
    SELECT 
        lp.tier,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value
    FROM loyalty_program lp
    JOIN orders o ON lp.customer_id = o.customer_id
    WHERE o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY lp.tier
)
SELECT 
    ms.tier,
    ms.member_count,
    ROUND(ms.member_count * 100.0 / SUM(ms.member_count) OVER (), 2) AS member_pct,
    ROUND(ms.avg_membership_days, 0) AS avg_membership_days,
    mr.total_orders,
    ROUND(mr.total_revenue, 2) AS total_revenue,
    ROUND(mr.total_revenue * 100.0 / SUM(mr.total_revenue) OVER (), 2) AS revenue_pct,
    ROUND(mr.avg_order_value, 2) AS avg_order_value,
    ROUND(mr.total_orders * 1.0 / ms.member_count, 1) AS avg_orders_per_member,
    ROUND(mr.total_revenue / ms.member_count, 2) AS revenue_per_member,
    ROUND(ms.avg_points_balance, 0) AS avg_points_balance,
    ms.total_points_outstanding,
    ms.total_points_earned,
    ms.total_points_redeemed,
    ROUND(ms.total_points_redeemed * 100.0 / NULLIF(ms.total_points_earned, 0), 2) AS redemption_rate_pct
FROM member_stats ms
LEFT JOIN member_revenue mr ON ms.tier = mr.tier
ORDER BY 
    CASE ms.tier 
        WHEN 'platinum' THEN 1 
        WHEN 'gold' THEN 2 
        WHEN 'silver' THEN 3 
        WHEN 'bronze' THEN 4 
    END;

-- ========================================
-- 2. LOYALTY MEMBER VS NON-MEMBER COMPARISON
-- Compare behavior and value
-- ========================================
WITH loyalty_comparison AS (
    SELECT 
        c.customer_id,
        CASE WHEN lp.customer_id IS NOT NULL THEN 'Loyalty Member' ELSE 'Non-Member' END AS membership_status,
        lp.tier,
        COUNT(DISTINCT o.order_id) AS order_count,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        MIN(o.order_date) AS first_order_date,
        MAX(o.order_date) AS last_order_date,
        COUNT(DISTINCT DATE_FORMAT(o.order_date, '%Y-%m')) AS active_months,
        COUNT(DISTINCT r.return_id) AS return_count
    FROM customers c
    LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    LEFT JOIN returns r ON o.order_id = r.order_id
    WHERE c.status = 'active'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY c.customer_id, membership_status, lp.tier
)
SELECT 
    membership_status,
    COUNT(DISTINCT customer_id) AS customer_count,
    SUM(order_count) AS total_orders,
    ROUND(SUM(total_revenue), 2) AS total_revenue,
    ROUND(AVG(total_revenue), 2) AS avg_customer_ltv,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(AVG(order_count), 1) AS avg_orders_per_customer,
    ROUND(AVG(active_months), 1) AS avg_active_months,
    ROUND(AVG(DATEDIFF(last_order_date, first_order_date)), 0) AS avg_customer_lifespan_days,
    ROUND(SUM(return_count) * 100.0 / NULLIF(SUM(order_count), 0), 2) AS return_rate_pct,
    -- Revenue per customer comparison
    ROUND(SUM(total_revenue) / COUNT(DISTINCT customer_id), 2) AS revenue_per_customer
FROM loyalty_comparison
GROUP BY membership_status
ORDER BY revenue_per_customer DESC;

-- ========================================
-- 3. LOYALTY TIER PROGRESSION ANALYSIS
-- Track customer movement between tiers
-- ========================================
WITH tier_history AS (
    SELECT 
        lp.customer_id,
        lp.tier AS current_tier,
        lp.tier_start_date,
        lp.joined_date,
        DATEDIFF(CURDATE(), lp.tier_start_date) AS days_in_current_tier,
        DATEDIFF(lp.tier_start_date, lp.joined_date) AS days_to_current_tier,
        lp.points_earned_lifetime,
        lp.points_balance,
        COUNT(DISTINCT o.order_id) AS orders_since_tier_start,
        SUM(o.total_amount) AS revenue_since_tier_start
    FROM loyalty_program lp
    LEFT JOIN orders o ON lp.customer_id = o.customer_id 
        AND o.payment_status = 'paid'
        AND o.order_date >= lp.tier_start_date
    GROUP BY lp.customer_id, lp.tier, lp.tier_start_date, lp.joined_date, 
             lp.points_earned_lifetime, lp.points_balance
)
SELECT 
    current_tier,
    COUNT(*) AS member_count,
    ROUND(AVG(days_in_current_tier), 0) AS avg_days_in_tier,
    ROUND(AVG(days_to_current_tier), 0) AS avg_days_to_reach_tier,
    ROUND(AVG(points_earned_lifetime), 0) AS avg_lifetime_points,
    ROUND(AVG(points_balance), 0) AS avg_current_balance,
    ROUND(AVG(orders_since_tier_start), 1) AS avg_orders_in_tier,
    ROUND(AVG(revenue_since_tier_start), 2) AS avg_revenue_in_tier,
    -- Tier engagement
    COUNT(CASE WHEN days_in_current_tier <= 90 THEN 1 END) AS new_to_tier_90days,
    COUNT(CASE WHEN days_in_current_tier > 365 THEN 1 END) AS in_tier_over_1year,
    -- Tier progression readiness
    COUNT(CASE 
        WHEN current_tier = 'bronze' AND points_balance >= 5000 THEN 1
        WHEN current_tier = 'silver' AND points_balance >= 10000 THEN 1
        WHEN current_tier = 'gold' AND points_balance >= 20000 THEN 1
    END) AS ready_for_upgrade
FROM tier_history
GROUP BY current_tier
ORDER BY 
    CASE current_tier 
        WHEN 'platinum' THEN 1 
        WHEN 'gold' THEN 2 
        WHEN 'silver' THEN 3 
        WHEN 'bronze' THEN 4 
    END;

-- ========================================
-- 4. POINTS ECONOMICS ANALYSIS
-- Understand points earning and redemption
-- ========================================
WITH points_activity AS (
    SELECT 
        lp.customer_id,
        lp.tier,
        lp.points_balance,
        lp.points_earned_lifetime,
        lp.points_redeemed_lifetime,
        lp.joined_date,
        DATEDIFF(CURDATE(), lp.joined_date) AS membership_days,
        SUM(o.total_amount) AS total_spent,
        COUNT(DISTINCT o.order_id) AS order_count
    FROM loyalty_program lp
    LEFT JOIN orders o ON lp.customer_id = o.customer_id 
        AND o.payment_status = 'paid'
    GROUP BY lp.customer_id, lp.tier, lp.points_balance, lp.points_earned_lifetime,
             lp.points_redeemed_lifetime, lp.joined_date
)
SELECT 
    tier,
    COUNT(*) AS member_count,
    -- Points per member
    ROUND(AVG(points_balance), 0) AS avg_points_balance,
    ROUND(AVG(points_earned_lifetime), 0) AS avg_points_earned,
    ROUND(AVG(points_redeemed_lifetime), 0) AS avg_points_redeemed,
    -- Points velocity
    ROUND(AVG(points_earned_lifetime * 365.0 / NULLIF(membership_days, 0)), 0) AS avg_points_earned_per_year,
    ROUND(AVG(points_redeemed_lifetime * 365.0 / NULLIF(membership_days, 0)), 0) AS avg_points_redeemed_per_year,
    -- Redemption behavior
    ROUND(AVG(points_redeemed_lifetime * 100.0 / NULLIF(points_earned_lifetime, 0)), 2) AS avg_redemption_rate_pct,
    COUNT(CASE WHEN points_redeemed_lifetime = 0 THEN 1 END) AS never_redeemed_count,
    ROUND(COUNT(CASE WHEN points_redeemed_lifetime = 0 THEN 1 END) * 100.0 / COUNT(*), 2) AS never_redeemed_pct,
    -- Economic metrics (assuming 1 point = $0.01)
    ROUND(SUM(points_balance) * 0.01, 2) AS total_liability_dollars,
    ROUND(AVG(total_spent), 2) AS avg_total_spent,
    ROUND(AVG(points_earned_lifetime * 0.01 * 100.0 / NULLIF(total_spent, 0)), 2) AS avg_points_earn_rate_pct,
    -- Engagement
    COUNT(CASE WHEN points_balance > 0 AND DATEDIFF(CURDATE(), joined_date) > 180 
               AND points_redeemed_lifetime = 0 THEN 1 END) AS dormant_balances
FROM points_activity
GROUP BY tier
ORDER BY 
    CASE tier 
        WHEN 'platinum' THEN 1 
        WHEN 'gold' THEN 2 
        WHEN 'silver' THEN 3 
        WHEN 'bronze' THEN 4 
    END;

-- ========================================
-- 5. MEMBER ENGAGEMENT AND ACTIVITY
-- Track active vs inactive members
-- ========================================
WITH member_activity AS (
    SELECT 
        lp.customer_id,
        lp.tier,
        lp.joined_date,
        lp.last_activity_date,
        DATEDIFF(CURDATE(), lp.last_activity_date) AS days_since_activity,
        DATEDIFF(CURDATE(), lp.joined_date) AS membership_duration_days,
        COUNT(DISTINCT o.order_id) AS total_orders,
        COUNT(DISTINCT CASE WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
              THEN o.order_id END) AS orders_last_90days,
        SUM(o.total_amount) AS lifetime_value,
        SUM(CASE WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN o.total_amount ELSE 0 END) AS revenue_last_90days
    FROM loyalty_program lp
    LEFT JOIN orders o ON lp.customer_id = o.customer_id AND o.payment_status = 'paid'
    GROUP BY lp.customer_id, lp.tier, lp.joined_date, lp.last_activity_date
)
SELECT 
    tier,
    CASE 
        WHEN days_since_activity <= 30 THEN 'Highly Active (0-30 days)'
        WHEN days_since_activity <= 90 THEN 'Active (31-90 days)'
        WHEN days_since_activity <= 180 THEN 'At Risk (91-180 days)'
        WHEN days_since_activity <= 365 THEN 'Dormant (181-365 days)'
        ELSE 'Inactive (365+ days)'
    END AS activity_segment,
    COUNT(*) AS member_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY tier), 2) AS pct_of_tier,
    ROUND(AVG(membership_duration_days), 0) AS avg_membership_days,
    ROUND(AVG(total_orders), 1) AS avg_lifetime_orders,
    ROUND(AVG(orders_last_90days), 1) AS avg_orders_last_90days,
    ROUND(AVG(lifetime_value), 2) AS avg_lifetime_value,
    ROUND(SUM(lifetime_value), 2) AS segment_total_value,
    -- Reactivation priority
    CASE 
        WHEN days_since_activity BETWEEN 91 AND 180 THEN 'High Priority - Send Reactivation'
        WHEN days_since_activity BETWEEN 181 AND 365 THEN 'Medium Priority - Win-back Campaign'
        WHEN days_since_activity > 365 THEN 'Low Priority - Last Chance Offer'
        ELSE 'Active - Continue Engagement'
    END AS recommended_action
FROM member_activity
GROUP BY tier, activity_segment
ORDER BY tier, 
    CASE activity_segment
        WHEN 'Highly Active (0-30 days)' THEN 1
        WHEN 'Active (31-90 days)' THEN 2
        WHEN 'At Risk (91-180 days)' THEN 3
        WHEN 'Dormant (181-365 days)' THEN 4
        ELSE 5
    END;

-- ========================================
-- 6. LOYALTY PROGRAM ROI ANALYSIS
-- Calculate program effectiveness
-- ========================================
WITH member_comparison AS (
    SELECT 
        c.customer_id,
        CASE WHEN lp.customer_id IS NOT NULL THEN 1 ELSE 0 END AS is_member,
        lp.tier,
        lp.joined_date,
        COUNT(DISTINCT o.order_id) AS order_count,
        SUM(o.total_amount) AS total_revenue,
        AVG(o.total_amount) AS avg_order_value,
        -- Pre-membership revenue (for members)
        SUM(CASE WHEN o.order_date < COALESCE(lp.joined_date, '2099-12-31') 
            THEN o.total_amount ELSE 0 END) AS pre_membership_revenue,
        -- Post-membership revenue (for members)
        SUM(CASE WHEN o.order_date >= COALESCE(lp.joined_date, '2099-12-31') 
            THEN o.total_amount ELSE 0 END) AS post_membership_revenue,
        COUNT(DISTINCT CASE WHEN o.order_date < COALESCE(lp.joined_date, '2099-12-31') 
              THEN o.order_id END) AS pre_membership_orders,
        COUNT(DISTINCT CASE WHEN o.order_date >= COALESCE(lp.joined_date, '2099-12-31') 
              THEN o.order_id END) AS post_membership_orders
    FROM customers c
    LEFT JOIN loyalty_program lp ON c.customer_id = lp.customer_id
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    WHERE c.status = 'active'
        AND c.created_at >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY c.customer_id, is_member, lp.tier, lp.joined_date
)
SELECT 
    CASE WHEN is_member = 1 THEN 'Loyalty Members' ELSE 'Non-Members' END AS customer_type,
    COUNT(*) AS customer_count,
    ROUND(AVG(order_count), 1) AS avg_orders,
    ROUND(AVG(total_revenue), 2) AS avg_customer_value,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    -- Member-specific metrics
    ROUND(AVG(CASE WHEN is_member = 1 THEN pre_membership_orders END), 1) AS avg_orders_before_joining,
    ROUND(AVG(CASE WHEN is_member = 1 THEN post_membership_orders END), 1) AS avg_orders_after_joining,
    ROUND(AVG(CASE WHEN is_member = 1 THEN post_membership_revenue END), 2) AS avg_revenue_after_joining,
    -- Calculate lift
    ROUND((AVG(CASE WHEN is_member = 1 THEN post_membership_orders END) - 
           AVG(CASE WHEN is_member = 1 THEN pre_membership_orders END)) * 100.0 /
          NULLIF(AVG(CASE WHEN is_member = 1 THEN pre_membership_orders END), 0), 2) AS order_frequency_lift_pct,
    -- Total value
    ROUND(SUM(total_revenue), 2) AS total_revenue
FROM member_comparison
GROUP BY customer_type
ORDER BY avg_customer_value DESC;

-- ========================================
-- 7. TOP LOYALTY MEMBERS (VIPs)
-- Identify most valuable loyalty members
-- ========================================
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    lp.tier,
    lp.joined_date,
    DATEDIFF(CURDATE(), lp.joined_date) AS membership_days,
    lp.points_balance,
    lp.points_earned_lifetime,
    lp.points_redeemed_lifetime,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(o.total_amount), 2) AS lifetime_revenue,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value,
    ROUND(SUM(o.total_amount) / NULLIF(DATEDIFF(CURDATE(), lp.joined_date), 0) * 365, 2) AS annualized_revenue,
    MAX(o.order_date) AS last_order_date,
    DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order,
    COUNT(DISTINCT DATE_FORMAT(o.order_date, '%Y-%m')) AS active_months,
    -- VIP scoring
    ROUND((SUM(o.total_amount) * 0.4) + 
          (COUNT(DISTINCT o.order_id) * 50 * 0.3) + 
          (lp.points_balance * 0.01 * 0.3), 2) AS vip_score
FROM loyalty_program lp
JOIN customers c ON lp.customer_id = c.customer_id
LEFT JOIN orders o ON lp.customer_id = o.customer_id AND o.payment_status = 'paid'
WHERE c.status = 'active'
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, lp.tier, 
         lp.joined_date, lp.points_balance, lp.points_earned_lifetime, 
         lp.points_redeemed_lifetime
HAVING lifetime_revenue >= 500
ORDER BY vip_score DESC
LIMIT 100;

-- ========================================
-- 8. LOYALTY MEMBER RETENTION COHORT
-- Track retention by join cohort
-- ========================================
WITH cohorts AS (
    SELECT 
        lp.customer_id,
        DATE_FORMAT(lp.joined_date, '%Y-%m') AS join_cohort,
        lp.joined_date,
        PERIOD_DIFF(DATE_FORMAT(CURDATE(), '%Y%m'), DATE_FORMAT(lp.joined_date, '%Y%m')) AS months_since_join
    FROM loyalty_program lp
    WHERE lp.joined_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
),
cohort_activity AS (
    SELECT 
        c.join_cohort,
        c.months_since_join,
        COUNT(DISTINCT c.customer_id) AS cohort_size,
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            THEN c.customer_id 
        END) AS active_last_30days,
        COUNT(DISTINCT CASE 
            WHEN o.order_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY) 
            THEN c.customer_id 
        END) AS active_last_90days,
        SUM(o.total_amount) AS cohort_revenue,
        AVG(o.total_amount) AS avg_order_value
    FROM cohorts c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    GROUP BY c.join_cohort, c.months_since_join
)
SELECT 
    join_cohort,
    months_since_join,
    cohort_size,
    active_last_30days,
    active_last_90days,
    ROUND(active_last_30days * 100.0 / cohort_size, 2) AS retention_30day_pct,
    ROUND(active_last_90days * 100.0 / cohort_size, 2) AS retention_90day_pct,
    ROUND(cohort_revenue, 2) AS cohort_revenue,
    ROUND(cohort_revenue / cohort_size, 2) AS revenue_per_member,
    ROUND(avg_order_value, 2) AS avg_order_value
FROM cohort_activity
ORDER BY join_cohort DESC, months_since_join;

-- ========================================
-- 9. LOYALTY POINTS BREAKAGE ANALYSIS
-- Identify unused points and potential savings
-- ========================================
WITH points_aging AS (
    SELECT 
        lp.customer_id,
        lp.tier,
        lp.points_balance,
        lp.points_earned_lifetime,
        lp.points_redeemed_lifetime,
        lp.last_activity_date,
        DATEDIFF(CURDATE(), lp.last_activity_date) AS days_inactive,
        DATEDIFF(CURDATE(), lp.joined_date) AS membership_days,
        COUNT(DISTINCT o.order_id) AS total_orders,
        MAX(o.order_date) AS last_order_date
    FROM loyalty_program lp
    LEFT JOIN orders o ON lp.customer_id = o.customer_id AND o.payment_status = 'paid'
    GROUP BY lp.customer_id, lp.tier, lp.points_balance, lp.points_earned_lifetime,
             lp.points_redeemed_lifetime, lp.last_activity_date, lp.joined_date
)
SELECT 
    CASE 
        WHEN days_inactive <= 90 THEN 'Active (0-90 days)'
        WHEN days_inactive <= 180 THEN 'Moderately Inactive (91-180 days)'
        WHEN days_inactive <= 365 THEN 'Highly Inactive (181-365 days)'
        ELSE 'Dormant (365+ days)'
    END AS activity_status,
    tier,
    COUNT(*) AS member_count,
    SUM(points_balance) AS total_points,
    ROUND(AVG(points_balance), 0) AS avg_points_balance,
    -- Liability calculation (assuming 1 point = $0.01)
    ROUND(SUM(points_balance) * 0.01, 2) AS liability_amount,
    -- Breakage probability
    CASE 
        WHEN days_inactive <= 90 THEN '10%'
        WHEN days_inactive <= 180 THEN '30%'
        WHEN days_inactive <= 365 THEN '60%'
        ELSE '85%'
    END AS estimated_breakage_rate,
    -- Potential breakage savings
    ROUND(SUM(points_balance) * 0.01 * 
        CASE 
            WHEN days_inactive <= 90 THEN 0.10
            WHEN days_inactive <= 180 THEN 0.30
            WHEN days_inactive <= 365 THEN 0.60
            ELSE 0.85
        END, 2) AS potential_breakage_value
FROM points_aging
WHERE points_balance > 0
GROUP BY activity_status, tier
ORDER BY 
    CASE activity_status
        WHEN 'Active (0-90 days)' THEN 1
        WHEN 'Moderately Inactive (91-180 days)' THEN 2
        WHEN 'Highly Inactive (181-365 days)' THEN 3
        ELSE 4
    END,
    CASE tier 
        WHEN 'platinum' THEN 1 
        WHEN 'gold' THEN 2 
        WHEN 'silver' THEN 3 
        WHEN 'bronze' THEN 4 
    END;

-- ========================================
-- 10. LOYALTY TIER UPGRADE CANDIDATES
-- Identify members close to tier upgrades
-- ========================================
WITH member_metrics AS (
    SELECT 
        lp.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        lp.tier AS current_tier,
        lp.points_balance,
        lp.points_earned_lifetime,
        lp.tier_start_date,
        COUNT(DISTINCT o.order_id) AS orders_last_6months,
        SUM(o.total_amount) AS revenue_last_6months,
        AVG(o.total_amount) AS avg_order_value,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order
    FROM loyalty_program lp
    JOIN customers c ON lp.customer_id = c.customer_id
    LEFT JOIN orders o ON lp.customer_id = o.customer_id 
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    WHERE c.status = 'active'
    GROUP BY lp.customer_id, c.first_name, c.last_name, c.email, lp.tier, 
             lp.points_balance, lp.points_earned_lifetime, lp.tier_start_date
)
SELECT 
    customer_id,
    CONCAT(first_name, ' ', last_name) AS customer_name,
    email,
    current_tier,
    CASE 
        WHEN current_tier = 'bronze' THEN 'silver'
        WHEN current_tier = 'silver' THEN 'gold'
        WHEN current_tier = 'gold' THEN 'platinum'
        ELSE 'MAX'
    END AS next_tier,
    points_balance AS current_points,
    CASE 
        WHEN current_tier = 'bronze' THEN 5000 - points_balance
        WHEN current_tier = 'silver' THEN 10000 - points_balance
        WHEN current_tier = 'gold' THEN 20000 - points_balance
        ELSE 0
    END AS points_to_next_tier,
    orders_last_6months,
    ROUND(revenue_last_6months, 2) AS revenue_last_6months,
    ROUND(avg_order_value, 2) AS avg_order_value,
    last_order_date,
    days_since_last_order,
    -- Upgrade probability score
    CASE 
        WHEN current_tier = 'bronze' AND points_balance >= 4000 THEN 'Very High'
        WHEN current_tier = 'bronze' AND points_balance >= 3000 THEN 'High'
        WHEN current_tier = 'silver' AND points_balance >= 8000 THEN 'Very High'
        WHEN current_tier = 'silver' AND points_balance >= 6000 THEN 'High'
        WHEN current_tier = 'gold' AND points_balance >= 15000 THEN 'Very High'
        WHEN current_tier = 'gold' AND points_balance >= 12000 THEN 'High'
        ELSE 'Medium'
    END AS upgrade_likelihood,
    -- Recommended action
    CASE 
        WHEN current_tier = 'bronze' AND points_balance >= 4500 
            THEN CONCAT('Only ', 5000 - points_balance, ' points to Silver! Send upgrade incentive')
        WHEN current_tier = 'silver' AND points_balance >= 9000 
            THEN CONCAT('Only ', 10000 - points_balance, ' points to Gold! Send upgrade incentive')
        WHEN current_tier = 'gold' AND points_balance >= 18000 
            THEN CONCAT('Only ', 20000 - points_balance, ' points to Platinum! Send upgrade incentive')
        WHEN days_since_last_order > 60 
            THEN 'Send re-engagement campaign with bonus points'
        ELSE 'Continue regular engagement'
    END AS recommended_action
FROM member_metrics
WHERE current_tier != 'platinum'
    AND (
        (current_tier = 'bronze' AND points_balance >= 3000) OR
        (current_tier = 'silver' AND points_balance >= 6000) OR
        (current_tier = 'gold' AND points_balance >= 12000)
    )
ORDER BY 
    CASE upgrade_likelihood
        WHEN 'Very High' THEN 1
        WHEN 'High' THEN 2
        ELSE 3
    END,
    points_balance DESC
LIMIT 100;