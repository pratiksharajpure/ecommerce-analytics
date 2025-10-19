-- ========================================
-- SEASONAL TRENDS ANALYSIS
-- E-commerce Revenue Analytics Engine
-- Seasonal Patterns, Trend Decomposition, Forecasting
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. SEASONAL REVENUE PATTERNS
-- Identify recurring seasonal patterns
-- ========================================

WITH daily_sales AS (
    SELECT 
        DATE(order_date) AS sale_date,
        YEAR(order_date) AS year,
        MONTH(order_date) AS month,
        DAYOFWEEK(order_date) AS day_of_week,
        WEEK(order_date) AS week_of_year,
        QUARTER(order_date) AS quarter,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        SUM(total_amount) AS revenue,
        AVG(total_amount) AS avg_order_value
    FROM orders
    WHERE status IN ('delivered', 'shipped', 'processing')
        AND payment_status = 'paid'
        AND order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY DATE(order_date), YEAR(order_date), MONTH(order_date), 
             DAYOFWEEK(order_date), WEEK(order_date), QUARTER(order_date)
),
seasonal_aggregates AS (
    SELECT 
        month,
        CASE month
            WHEN 1 THEN 'January'
            WHEN 2 THEN 'February'
            WHEN 3 THEN 'March'
            WHEN 4 THEN 'April'
            WHEN 5 THEN 'May'
            WHEN 6 THEN 'June'
            WHEN 7 THEN 'July'
            WHEN 8 THEN 'August'
            WHEN 9 THEN 'September'
            WHEN 10 THEN 'October'
            WHEN 11 THEN 'November'
            WHEN 12 THEN 'December'
        END AS month_name,
        quarter,
        day_of_week,
        CASE day_of_week
            WHEN 1 THEN 'Sunday'
            WHEN 2 THEN 'Monday'
            WHEN 3 THEN 'Tuesday'
            WHEN 4 THEN 'Wednesday'
            WHEN 5 THEN 'Thursday'
            WHEN 6 THEN 'Friday'
            WHEN 7 THEN 'Saturday'
        END AS day_name,
        AVG(revenue) AS avg_daily_revenue,
        AVG(orders) AS avg_daily_orders,
        AVG(customers) AS avg_daily_customers,
        AVG(avg_order_value) AS avg_order_value,
        STDDEV(revenue) AS revenue_stddev,
        MIN(revenue) AS min_revenue,
        MAX(revenue) AS max_revenue
    FROM daily_sales
    GROUP BY month, quarter, day_of_week
)
SELECT 
    month,
    month_name,
    quarter,
    CONCAT('Q', quarter) AS quarter_label,
    ROUND(SUM(avg_daily_revenue), 2) AS monthly_avg_revenue,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(SUM(avg_daily_orders), 0) AS monthly_avg_orders,
    ROUND(SUM(avg_daily_customers), 0) AS monthly_avg_customers,
    -- Seasonality index (compared to annual average)
    ROUND(
        SUM(avg_daily_revenue) / 
        (SELECT AVG(avg_daily_revenue) FROM seasonal_aggregates) * 100,
        2
    ) AS seasonality_index,
    -- Day of week patterns
    MAX(CASE WHEN day_of_week = 1 THEN ROUND(avg_daily_revenue, 2) END) AS sunday_avg,
    MAX(CASE WHEN day_of_week = 2 THEN ROUND(avg_daily_revenue, 2) END) AS monday_avg,
    MAX(CASE WHEN day_of_week = 3 THEN ROUND(avg_daily_revenue, 2) END) AS tuesday_avg,
    MAX(CASE WHEN day_of_week = 4 THEN ROUND(avg_daily_revenue, 2) END) AS wednesday_avg,
    MAX(CASE WHEN day_of_week = 5 THEN ROUND(avg_daily_revenue, 2) END) AS thursday_avg,
    MAX(CASE WHEN day_of_week = 6 THEN ROUND(avg_daily_revenue, 2) END) AS friday_avg,
    MAX(CASE WHEN day_of_week = 7 THEN ROUND(avg_daily_revenue, 2) END) AS saturday_avg,
    -- Peak day identification
    CASE 
        WHEN MAX(avg_daily_revenue) = MAX(CASE WHEN day_of_week = 1 THEN avg_daily_revenue END) THEN 'Sunday'
        WHEN MAX(avg_daily_revenue) = MAX(CASE WHEN day_of_week = 2 THEN avg_daily_revenue END) THEN 'Monday'
        WHEN MAX(avg_daily_revenue) = MAX(CASE WHEN day_of_week = 3 THEN avg_daily_revenue END) THEN 'Tuesday'
        WHEN MAX(avg_daily_revenue) = MAX(CASE WHEN day_of_week = 4 THEN avg_daily_revenue END) THEN 'Wednesday'
        WHEN MAX(avg_daily_revenue) = MAX(CASE WHEN day_of_week = 5 THEN avg_daily_revenue END) THEN 'Thursday'
        WHEN MAX(avg_daily_revenue) = MAX(CASE WHEN day_of_week = 6 THEN avg_daily_revenue END) THEN 'Friday'
        ELSE 'Saturday'
    END AS peak_day_of_week,
    -- Season classification
    CASE 
        WHEN month IN (11, 12) THEN 'Holiday Season (High)'
        WHEN month IN (1, 2) THEN 'Post-Holiday (Low)'
        WHEN month IN (3, 4, 5) THEN 'Spring (Medium)'
        WHEN month IN (6, 7, 8) THEN 'Summer (Variable)'
        ELSE 'Fall (Building)'
    END AS seasonal_period
FROM seasonal_aggregates
GROUP BY month, month_name, quarter
ORDER BY month;

-- ========================================
-- 2. TREND DECOMPOSITION
-- Separate trend, seasonal, and random components
-- ========================================

WITH monthly_data AS (
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m-01') AS month_start,
        YEAR(order_date) AS year,
        MONTH(order_date) AS month,
        COUNT(DISTINCT order_id) AS orders,
        SUM(total_amount) AS revenue,
        AVG(total_amount) AS avg_order_value
    FROM orders
    WHERE status IN ('delivered', 'shipped', 'processing')
        AND payment_status = 'paid'
        AND order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY DATE_FORMAT(order_date, '%Y-%m-01'), YEAR(order_date), MONTH(order_date)
),
moving_averages AS (
    SELECT 
        month_start,
        year,
        month,
        revenue,
        orders,
        -- 12-month moving average (trend component)
        AVG(revenue) OVER (
            ORDER BY month_start 
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) AS trend_12mo,
        -- 3-month moving average (smoothed)
        AVG(revenue) OVER (
            ORDER BY month_start 
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS trend_3mo
    FROM monthly_data
),
seasonal_component AS (
    SELECT 
        month_start,
        year,
        month,
        revenue AS actual_revenue,
        ROUND(trend_12mo, 2) AS trend_component,
        ROUND(trend_3mo, 2) AS smoothed_trend,
        -- Seasonal component = Actual - Trend
        ROUND(revenue - trend_12mo, 2) AS seasonal_component,
        -- Seasonal index
        ROUND((revenue / NULLIF(trend_12mo, 0)) * 100, 2) AS seasonal_index,
        -- Random/irregular component
        ROUND(revenue - trend_3mo, 2) AS irregular_component
    FROM moving_averages
)
SELECT 
    month_start,
    year,
    month,
    CASE month
        WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
    END AS month_abbr,
    ROUND(actual_revenue, 2) AS actual_revenue,
    trend_component,
    smoothed_trend,
    seasonal_component,
    seasonal_index,
    irregular_component,
    -- Strength of seasonal effect
    CASE 
        WHEN ABS(seasonal_component / NULLIF(trend_component, 0)) > 0.30 THEN 'Strong Seasonal Effect'
        WHEN ABS(seasonal_component / NULLIF(trend_component, 0)) > 0.15 THEN 'Moderate Seasonal Effect'
        WHEN ABS(seasonal_component / NULLIF(trend_component, 0)) > 0.05 THEN 'Weak Seasonal Effect'
        ELSE 'No Seasonal Effect'
    END AS seasonality_strength,
    -- Trend direction
    CASE 
        WHEN trend_component > LAG(trend_component) OVER (ORDER BY month_start) * 1.05 THEN 'Growing'
        WHEN trend_component < LAG(trend_component) OVER (ORDER BY month_start) * 0.95 THEN 'Declining'
        ELSE 'Stable'
    END AS trend_direction
FROM seasonal_component
ORDER BY month_start DESC;

-- ========================================
-- 3. SEASONAL FORECASTING
-- Predict future seasonal patterns
-- ========================================

WITH historical_seasonal_factors AS (
    SELECT 
        MONTH(order_date) AS month,
        YEAR(order_date) AS year,
        SUM(total_amount) AS monthly_revenue
    FROM orders
    WHERE status IN ('delivered', 'shipped', 'processing')
        AND payment_status = 'paid'
        AND order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY MONTH(order_date), YEAR(order_date)
),
average_seasonal_factors AS (
    SELECT 
        month,
        AVG(monthly_revenue) AS avg_monthly_revenue,
        STDDEV(monthly_revenue) AS stddev_monthly_revenue,
        COUNT(*) AS years_of_data
    FROM historical_seasonal_factors
    GROUP BY month
),
baseline_trend AS (
    SELECT 
        AVG(avg_monthly_revenue) AS annual_baseline
    FROM average_seasonal_factors
),
seasonal_indices AS (
    SELECT 
        asf.month,
        asf.avg_monthly_revenue,
        asf.stddev_monthly_revenue,
        bt.annual_baseline,
        -- Seasonal index (1.0 = average, >1.0 = above average, <1.0 = below average)
        asf.avg_monthly_revenue / NULLIF(bt.annual_baseline, 0) AS seasonal_index,
        asf.years_of_data
    FROM average_seasonal_factors asf
    CROSS JOIN baseline_trend bt
),
growth_rate AS (
    SELECT 
        ((SELECT SUM(total_amount) FROM orders 
          WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
            AND status IN ('delivered', 'shipped', 'processing')) /
         (SELECT SUM(total_amount) FROM orders 
          WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
            AND order_date < DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
            AND status IN ('delivered', 'shipped', 'processing')) - 1) AS recent_growth_rate
)
SELECT 
    si.month,
    CASE si.month
        WHEN 1 THEN 'January' WHEN 2 THEN 'February' WHEN 3 THEN 'March'
        WHEN 4 THEN 'April' WHEN 5 THEN 'May' WHEN 6 THEN 'June'
        WHEN 7 THEN 'July' WHEN 8 THEN 'August' WHEN 9 THEN 'September'
        WHEN 10 THEN 'October' WHEN 11 THEN 'November' WHEN 12 THEN 'December'
    END AS month_name,
    ROUND(si.avg_monthly_revenue, 2) AS historical_avg_revenue,
    ROUND(si.seasonal_index, 3) AS seasonal_index,
    -- Forecast for next occurrence of this month
    ROUND(si.annual_baseline * (1 + gr.recent_growth_rate) * si.seasonal_index, 2) AS forecasted_revenue,
    -- Confidence interval (±1 standard deviation)
    ROUND(si.annual_baseline * (1 + gr.recent_growth_rate) * si.seasonal_index - 
          si.stddev_monthly_revenue, 2) AS forecast_lower_bound,
    ROUND(si.annual_baseline * (1 + gr.recent_growth_rate) * si.seasonal_index + 
          si.stddev_monthly_revenue, 2) AS forecast_upper_bound,
    ROUND(gr.recent_growth_rate * 100, 2) AS assumed_growth_rate_pct,
    -- Peak/Trough identification
    CASE 
        WHEN si.seasonal_index >= 1.20 THEN 'Peak Season'
        WHEN si.seasonal_index >= 1.10 THEN 'High Season'
        WHEN si.seasonal_index >= 0.95 THEN 'Average Season'
        WHEN si.seasonal_index >= 0.85 THEN 'Low Season'
        ELSE 'Trough Season'
    END AS season_classification,
    -- Planning recommendations
    CASE 
        WHEN si.seasonal_index >= 1.20 THEN 'Max inventory, full staffing, peak marketing'
        WHEN si.seasonal_index >= 1.10 THEN 'High inventory, increased staffing'
        WHEN si.seasonal_index >= 0.95 THEN 'Standard operations'
        WHEN si.seasonal_index >= 0.85 THEN 'Reduce inventory, promotional focus'
        ELSE 'Minimal inventory, cost reduction focus'
    END AS operational_recommendation
FROM seasonal_indices si
CROSS JOIN growth_rate gr
ORDER BY si.month;

-- ========================================
-- 4. HOLIDAY & EVENT IMPACT ANALYSIS
-- Measure impact of specific holidays and events
-- ========================================

WITH holiday_calendar AS (
    -- Major holidays (would typically come from external calendar)
    SELECT '2024-01-01' AS holiday_date, 'New Year' AS holiday_name
    UNION ALL SELECT '2024-02-14', 'Valentine''s Day'
    UNION ALL SELECT '2024-03-17', 'St Patrick''s Day'
    UNION ALL SELECT '2024-05-27', 'Memorial Day'
    UNION ALL SELECT '2024-07-04', 'Independence Day'
    UNION ALL SELECT '2024-09-02', 'Labor Day'
    UNION ALL SELECT '2024-10-31', 'Halloween'
    UNION ALL SELECT '2024-11-28', 'Thanksgiving'
    UNION ALL SELECT '2024-11-29', 'Black Friday'
    UNION ALL SELECT '2024-12-02', 'Cyber Monday'
    UNION ALL SELECT '2024-12-25', 'Christmas'
    UNION ALL SELECT '2023-11-24', 'Black Friday 2023'
    UNION ALL SELECT '2023-11-27', 'Cyber Monday 2023'
    UNION ALL SELECT '2023-12-25', 'Christmas 2023'
),
daily_sales_with_holidays AS (
    SELECT 
        DATE(o.order_date) AS sale_date,
        hc.holiday_name,
        CASE 
            WHEN hc.holiday_date IS NOT NULL THEN 'Holiday'
            WHEN DATE(o.order_date) BETWEEN DATE_SUB(hc.holiday_date, INTERVAL 7 DAY) 
                 AND DATE_SUB(hc.holiday_date, INTERVAL 1 DAY) THEN 'Pre-Holiday'
            WHEN DATE(o.order_date) BETWEEN DATE_ADD(hc.holiday_date, INTERVAL 1 DAY) 
                 AND DATE_ADD(hc.holiday_date, INTERVAL 7 DAY) THEN 'Post-Holiday'
            ELSE 'Regular'
        END AS period_type,
        COUNT(DISTINCT o.order_id) AS orders,
        SUM(o.total_amount) AS revenue,
        COUNT(DISTINCT o.customer_id) AS customers,
        AVG(o.total_amount) AS avg_order_value
    FROM orders o
    LEFT JOIN holiday_calendar hc 
        ON DATE(o.order_date) BETWEEN DATE_SUB(hc.holiday_date, INTERVAL 7 DAY)
        AND DATE_ADD(hc.holiday_date, INTERVAL 7 DAY)
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY DATE(o.order_date), hc.holiday_name, period_type
),
baseline_metrics AS (
    SELECT 
        AVG(revenue) AS avg_regular_day_revenue,
        AVG(orders) AS avg_regular_day_orders,
        AVG(avg_order_value) AS avg_regular_order_value
    FROM daily_sales_with_holidays
    WHERE period_type = 'Regular'
)
SELECT 
    dsh.holiday_name,
    dsh.sale_date,
    dsh.period_type,
    dsh.orders,
    ROUND(dsh.revenue, 2) AS revenue,
    dsh.customers,
    ROUND(dsh.avg_order_value, 2) AS avg_order_value,
    -- Baseline comparison
    ROUND(bm.avg_regular_day_revenue, 2) AS baseline_revenue,
    ROUND((dsh.revenue - bm.avg_regular_day_revenue) / NULLIF(bm.avg_regular_day_revenue, 0) * 100, 2) AS revenue_lift_pct,
    ROUND((dsh.orders - bm.avg_regular_day_orders) / NULLIF(bm.avg_regular_day_orders, 0) * 100, 2) AS order_lift_pct,
    -- Impact classification
    CASE 
        WHEN (dsh.revenue - bm.avg_regular_day_revenue) / NULLIF(bm.avg_regular_day_revenue, 0) > 2.0 THEN 'Massive Impact (200%+)'
        WHEN (dsh.revenue - bm.avg_regular_day_revenue) / NULLIF(bm.avg_regular_day_revenue, 0) > 1.0 THEN 'Major Impact (100-200%)'
        WHEN (dsh.revenue - bm.avg_regular_day_revenue) / NULLIF(bm.avg_regular_day_revenue, 0) > 0.5 THEN 'Significant Impact (50-100%)'
        WHEN (dsh.revenue - bm.avg_regular_day_revenue) / NULLIF(bm.avg_regular_day_revenue, 0) > 0.2 THEN 'Moderate Impact (20-50%)'
        WHEN (dsh.revenue - bm.avg_regular_day_revenue) / NULLIF(bm.avg_regular_day_revenue, 0) > 0 THEN 'Minor Impact (0-20%)'
        ELSE 'Negative/No Impact'
    END AS impact_level
FROM daily_sales_with_holidays dsh
CROSS JOIN baseline_metrics bm
WHERE dsh.holiday_name IS NOT NULL
ORDER BY dsh.sale_date DESC, dsh.revenue DESC;

-- ========================================
-- 5. PRODUCT SEASONALITY
-- Which products are seasonal vs. evergreen
-- ========================================

WITH monthly_product_sales AS (
    SELECT 
        p.product_id,
        p.product_name,
        pc.category_name,
        MONTH(o.order_date) AS month,
        SUM(oi.quantity) AS units_sold,
        SUM(oi.subtotal) AS revenue
    FROM products p
    JOIN product_categories pc ON p.category_id = pc.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.status IN ('delivered', 'shipped', 'processing')
        AND o.payment_status = 'paid'
        AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
    GROUP BY p.product_id, p.product_name, pc.category_name, MONTH(o.order_date)
),
product_stats AS (
    SELECT 
        product_id,
        product_name,
        category_name,
        AVG(revenue) AS avg_monthly_revenue,
        STDDEV(revenue) AS stddev_monthly_revenue,
        MIN(revenue) AS min_monthly_revenue,
        MAX(revenue) AS max_monthly_revenue,
        -- Coefficient of variation (CV) - measure of seasonality
        STDDEV(revenue) / NULLIF(AVG(revenue), 0) AS coefficient_of_variation
    FROM monthly_product_sales
    GROUP BY product_id, product_name, category_name
),
peak_months AS (
    SELECT 
        mps.product_id,
        GROUP_CONCAT(
            CASE mps.month
                WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
                WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
                WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
                WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
            END
            ORDER BY mps.revenue DESC
            SEPARATOR ', '
        ) AS top_months
    FROM monthly_product_sales mps
    JOIN (
        SELECT product_id, MAX(revenue) AS max_rev
        FROM monthly_product_sales
        GROUP BY product_id
    ) max_sales ON mps.product_id = max_sales.product_id 
        AND mps.revenue >= max_sales.max_rev * 0.8
    GROUP BY mps.product_id
)
SELECT 
    ps.product_id,
    ps.product_name,
    ps.category_name,
    ROUND(ps.avg_monthly_revenue, 2) AS avg_monthly_revenue,
    ROUND(ps.min_monthly_revenue, 2) AS min_monthly_revenue,
    ROUND(ps.max_monthly_revenue, 2) AS max_monthly_revenue,
    ROUND(ps.coefficient_of_variation, 3) AS seasonality_coefficient,
    pm.top_months AS peak_selling_months,
    -- Seasonality classification
    CASE 
        WHEN ps.coefficient_of_variation >= 0.75 THEN 'Highly Seasonal'
        WHEN ps.coefficient_of_variation >= 0.50 THEN 'Moderately Seasonal'
        WHEN ps.coefficient_of_variation >= 0.30 THEN 'Slightly Seasonal'
        ELSE 'Evergreen/Stable'
    END AS seasonality_type,
    -- Inventory strategy
    CASE 
        WHEN ps.coefficient_of_variation >= 0.75 THEN 'Just-in-time for peak, minimal off-season'
        WHEN ps.coefficient_of_variation >= 0.50 THEN 'Build inventory before peak months'
        WHEN ps.coefficient_of_variation >= 0.30 THEN 'Maintain buffer stock year-round'
        ELSE 'Consistent inventory levels'
    END AS inventory_recommendation
FROM product_stats ps
LEFT JOIN peak_months pm ON ps.product_id = pm.product_id
WHERE ps.avg_monthly_revenue > 0
ORDER BY ps.coefficient_of_variation DESC, ps.avg_monthly_revenue DESC
LIMIT 100;

-- ========================================
-- End of Seasonal Trends Analysis
-- ========================================