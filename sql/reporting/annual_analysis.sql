-- ========================================
-- ANNUAL ANALYSIS & YEAR-OVER-YEAR REVIEW
-- Long-term patterns and trend analysis
-- Multi-year comparative metrics
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. DEFINE YEAR PERIODS
-- ========================================
SET @current_year = YEAR(CURDATE());
SET @current_year_start = CONCAT(@current_year, '-01-01');
SET @current_year_end = CONCAT(@current_year, '-12-31');
SET @previous_year = @current_year - 1;
SET @previous_year_start = CONCAT(@previous_year, '-01-01');
SET @previous_year_end = CONCAT(@previous_year, '-12-31');
SET @two_years_ago = @current_year - 2;
SET @two_years_ago_start = CONCAT(@two_years_ago, '-01-01');
SET @two_years_ago_end = CONCAT(@two_years_ago, '-12-31');

-- Display analysis period
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'ANNUAL BUSINESS ANALYSIS & TRENDS' AS 'Report Title';
SELECT CONCAT('Analysis Period: ', @two_years_ago, ' - ', @current_year) AS 'Period';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    @two_years_ago AS 'Year 1',
    @previous_year AS 'Year 2',
    @current_year AS 'Year 3 (Current)',
    CONCAT(DATEDIFF(CURDATE(), @current_year_start), ' days completed') AS 'Current Year Progress';

-- ========================================
-- 2. ANNUAL REVENUE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'ANNUAL REVENUE ANALYSIS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH annual_revenue AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(DISTINCT order_id) AS total_orders,
        COUNT(DISTINCT customer_id) AS unique_customers,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_order_value,
        SUM(total_amount) / COUNT(DISTINCT customer_id) AS revenue_per_customer,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    FORMAT(unique_customers, 0) AS 'Unique Customers',
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(total_refunds, 2)) AS 'Total Refunds',
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(avg_refund, 2)) AS 'Avg Refund Amount',
    CONCAT(FORMAT(return_rate, 2), '%') AS 'Return Rate',
    CASE
        WHEN return_rate <= 5 THEN '✅ Excellent'
        WHEN return_rate <= 10 THEN '✓ Good'
        WHEN return_rate <= 15 THEN '⚠️ Fair'
        ELSE '❌ High'
    END AS 'Return Performance'
FROM returns_annual
ORDER BY year;

-- ========================================
-- 10. INVENTORY TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'INVENTORY MANAGEMENT TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH inventory_snapshots AS (
    SELECT
        YEAR(i.last_updated) AS year,
        COUNT(DISTINCT i.product_id) AS total_skus,
        SUM(i.quantity_on_hand) AS total_units,
        SUM(i.quantity_on_hand * p.cost) AS total_inventory_value,
        AVG(i.quantity_on_hand) AS avg_stock_per_sku,
        SUM(CASE WHEN i.quantity_on_hand = 0 THEN 1 ELSE 0 END) AS out_of_stock_count,
        SUM(CASE WHEN i.quantity_on_hand < i.reorder_level THEN 1 ELSE 0 END) AS below_reorder_count,
        SUM(CASE WHEN i.quantity_reserved > i.quantity_on_hand THEN 1 ELSE 0 END) AS overbooked_count
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    WHERE YEAR(i.last_updated) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(i.last_updated)
)
SELECT
    year AS 'Year',
    FORMAT(total_skus, 0) AS 'Total SKUs',
    FORMAT(total_units, 0) AS 'Total Units',
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(total_inventory_value, 2)) AS 'Inventory Value',
    FORMAT(avg_stock_per_sku, 1) AS 'Avg Stock/SKU',
    FORMAT(out_of_stock_count, 0) AS 'Out of Stock',
    CONCAT(FORMAT((out_of_stock_count * 100.0 / total_skus), 1), '%') AS 'Stockout Rate',
    FORMAT(below_reorder_count, 0) AS 'Below Reorder',
    FORMAT(overbooked_count, 0) AS 'Overbooked',
    CASE
        WHEN (out_of_stock_count * 100.0 / total_skus) <= 5 THEN '✅ Excellent'
        WHEN (out_of_stock_count * 100.0 / total_skus) <= 10 THEN '✓ Good'
        WHEN (out_of_stock_count * 100.0 / total_skus) <= 15 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Inventory Health'
FROM inventory_snapshots
ORDER BY year;

-- ========================================
-- 11. VENDOR PERFORMANCE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'VENDOR PERFORMANCE OVER TIME' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH vendor_annual AS (
    SELECT
        YEAR(v.created_at) AS year,
        COUNT(DISTINCT v.vendor_id) AS total_vendors,
        COUNT(DISTINCT CASE WHEN v.status = 'active' THEN v.vendor_id END) AS active_vendors,
        COUNT(DISTINCT CASE WHEN v.status = 'blacklisted' THEN v.vendor_id END) AS blacklisted_vendors,
        AVG(v.rating) AS avg_rating,
        AVG(vc.cost_per_unit) AS avg_cost_per_unit,
        AVG(vc.lead_time_days) AS avg_lead_time
    FROM vendors v
    LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
    WHERE YEAR(v.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(v.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_vendors, 0) AS 'Total Vendors',
    FORMAT(active_vendors, 0) AS 'Active',
    FORMAT(blacklisted_vendors, 0) AS 'Blacklisted',
    FORMAT(avg_rating, 2) AS 'Avg Rating',
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(avg_cost_per_unit, 2)) AS 'Avg Cost/Unit',
    CONCAT(FORMAT(avg_lead_time, 0), ' days') AS 'Avg Lead Time',
    CASE
        WHEN avg_rating >= 4.5 THEN '⭐ Excellent'
        WHEN avg_rating >= 4.0 THEN '✓ Good'
        WHEN avg_rating >= 3.5 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Vendor Quality'
FROM vendor_annual
ORDER BY year;

-- ========================================
-- 12. CUSTOMER DEMOGRAPHICS EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER BASE DEMOGRAPHICS TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH geographic_trends AS (
    SELECT
        YEAR(created_at) AS year,
        state,
        COUNT(*) AS customer_count,
        SUM(customer_revenue) AS total_revenue
    FROM (
        SELECT
            c.created_at,
            c.state,
            c.customer_id,
            COALESCE(SUM(o.total_amount), 0) AS customer_revenue
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
        WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
        GROUP BY c.customer_id, c.created_at, c.state
    ) customer_data
    GROUP BY YEAR(created_at), state
)
SELECT
    state AS 'State',
    MAX(CASE WHEN year = @two_years_ago THEN customer_count END) AS CONCAT(@two_years_ago, ' Customers'),
    MAX(CASE WHEN year = @previous_year THEN customer_count END) AS CONCAT(@previous_year, ' Customers'),
    MAX(CASE WHEN year = @current_year THEN customer_count END) AS CONCAT(@current_year, ' Customers'),
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Year Revenue',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN customer_count END) > 
             MAX(CASE WHEN year = @previous_year THEN customer_count END) THEN '📈 Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN customer_count END) < 
             MAX(CASE WHEN year = @previous_year THEN customer_count END) THEN '📉 Declining'
        ELSE '→ Stable'
    END AS 'Trend'
FROM geographic_trends
WHERE state IS NOT NULL
GROUP BY state
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC
LIMIT 15;

-- ========================================
-- 13. DATA QUALITY EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'DATA QUALITY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quality_trends AS (
    SELECT
        @two_years_ago AS year,
        'Customer Data' AS category,
        COUNT(*) AS total_records,
        ROUND(100 * (1 - SUM(CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
    FROM customers
    WHERE YEAR(created_at) = @two_years_ago
    
    UNION ALL
    
    SELECT
        @previous_year,
        'Customer Data',
        COUNT(*),
        ROUND(100 * (1 - SUM(CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END) / COUNT(*)), 1)
    FROM customers
    WHERE YEAR(created_at) = @previous_year
    
    UNION ALL
    
    SELECT
        @current_year,
        'Customer Data',
        COUNT(*),
        ROUND(100 * (1 - SUM(CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END) / COUNT(*)), 1)
    FROM customers
    WHERE YEAR(created_at) = @current_year
    
    UNION ALL
    
    SELECT
        @two_years_ago,
        'Product Data',
        COUNT(*),
        ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
    FROM products
    WHERE YEAR(created_at) = @two_years_ago
    
    UNION ALL
    
    SELECT
        @previous_year,
        'Product Data',
        COUNT(*),
        ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
    FROM products
    WHERE YEAR(created_at) = @previous_year
    
    UNION ALL
    
    SELECT
        @current_year,
        'Product Data',
        COUNT(*),
        ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
    FROM products
    WHERE YEAR(created_at) = @current_year
)
SELECT
    category AS 'Data Category',
    MAX(CASE WHEN year = @two_years_ago THEN CONCAT(quality_score, '%') END) AS CONCAT(@two_years_ago, ' Score'),
    MAX(CASE WHEN year = @previous_year THEN CONCAT(quality_score, '%') END) AS CONCAT(@previous_year, ' Score'),
    MAX(CASE WHEN year = @current_year THEN CONCAT(quality_score, '%') END) AS CONCAT(@current_year, ' Score'),
    CONCAT(FORMAT(
        MAX(CASE WHEN year = @current_year THEN quality_score END) - 
        MAX(CASE WHEN year = @previous_year THEN quality_score END), 1), '%'
    ) AS 'YoY Change',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN quality_score END) > 
             MAX(CASE WHEN year = @previous_year THEN quality_score END) THEN '📈 Improving'
        WHEN MAX(CASE WHEN year = @current_year THEN quality_score END) < 
             MAX(CASE WHEN year = @previous_year THEN quality_score END) THEN '📉 Declining'
        ELSE '→ Stable'
    END AS 'Trend'
FROM quality_trends
GROUP BY category;

-- ========================================
-- 14. COMPOUND ANNUAL GROWTH RATE (CAGR)
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'COMPOUND ANNUAL GROWTH RATE (CAGR) ANALYSIS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH cagr_metrics AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS starting_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS ending_revenue,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @two_years_ago) AS starting_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS ending_customers,
        (SELECT AVG(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS starting_aov,
        (SELECT AVG(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS ending_aov
)
SELECT
    'Revenue' AS 'Metric',
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(starting_revenue, 2)) AS 'Starting Value',
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(ending_revenue, 2)) AS 'Ending Value',
    CONCAT(FORMAT(
        (POWER(ending_revenue / NULLIF(starting_revenue, 0), 1.0/2) - 1) * 100, 1
    ), '%') AS 'CAGR',
    CASE
        WHEN (POWER(ending_revenue / NULLIF(starting_revenue, 0), 1.0/2) - 1) * 100 >= 30 THEN '🚀 Hyper Growth'
        WHEN (POWER(ending_revenue / NULLIF(starting_revenue, 0), 1.0/2) - 1) * 100 >= 20 THEN '📈 High Growth'
        WHEN (POWER(ending_revenue / NULLIF(starting_revenue, 0), 1.0/2) - 1) * 100 >= 10 THEN '✓ Steady Growth'
        WHEN (POWER(ending_revenue / NULLIF(starting_revenue, 0), 1.0/2) - 1) * 100 >= 0 THEN '→ Slow Growth'
        ELSE '📉 Declining'
    END AS 'Growth Category'
FROM cagr_metrics

UNION ALL

SELECT
    'Customer Base',
    FORMAT(starting_customers, 0),
    FORMAT(ending_customers, 0),
    CONCAT(FORMAT(
        (POWER(ending_customers / NULLIF(starting_customers, 0), 1.0/2) - 1) * 100, 1
    ), '%'),
    CASE
        WHEN (POWER(ending_customers / NULLIF(starting_customers, 0), 1.0/2) - 1) * 100 >= 25 THEN '🚀 Hyper Growth'
        WHEN (POWER(ending_customers / NULLIF(starting_customers, 0), 1.0/2) - 1) * 100 >= 15 THEN '📈 High Growth'
        WHEN (POWER(ending_customers / NULLIF(starting_customers, 0), 1.0/2) - 1) * 100 >= 5 THEN '✓ Steady Growth'
        WHEN (POWER(ending_customers / NULLIF(starting_customers, 0), 1.0/2) - 1) * 100 >= 0 THEN '→ Slow Growth'
        ELSE '📉 Declining'
    END
FROM cagr_metrics

UNION ALL

SELECT
    'Average Order Value',
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(starting_aov, 2)),
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(ending_aov, 2)),
    CONCAT(FORMAT(
        (POWER(ending_aov / NULLIF(starting_aov, 0), 1.0/2) - 1) * 100, 1
    ), '%'),
    CASE
        WHEN (POWER(ending_aov / NULLIF(starting_aov, 0), 1.0/2) - 1) * 100 >= 15 THEN '🚀 Hyper Growth'
        WHEN (POWER(ending_aov / NULLIF(starting_aov, 0), 1.0/2) - 1) * 100 >= 10 THEN '📈 High Growth'
        WHEN (POWER(ending_aov / NULLIF(starting_aov, 0), 1.0/2) - 1) * 100 >= 5 THEN '✓ Steady Growth'
        WHEN (POWER(ending_aov / NULLIF(starting_aov, 0), 1.0/2) - 1) * 100 >= 0 THEN '→ Slow Growth'
        ELSE '📉 Declining'
    END
FROM cagr_metrics;

-- ========================================
-- 15. LONG-TERM STRATEGIC INSIGHTS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'STRATEGIC INSIGHTS & PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    'Business Scale' AS 'Insight Category',
    CONCAT(
        'Revenue has ',
        CASE
            WHEN (SELECT SUM(total_amount) FROM orders WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') >
                 (SELECT SUM(total_amount) FROM orders WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') * 1.5
            THEN 'grown by over 50%'
            WHEN (SELECT SUM(total_amount) FROM orders WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') >
                 (SELECT SUM(total_amount) FROM orders WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') * 1.2
            THEN 'grown by 20-50%'
            WHEN (SELECT SUM(total_amount) FROM orders WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') >
                 (SELECT SUM(total_amount) FROM orders WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')
            THEN 'grown steadily'
            ELSE 'declined or remained flat'
        END,
        ' year-over-year'
    ) AS 'Key Finding',
    'Focus on sustaining growth momentum and scaling operations' AS 'Strategic Recommendation'

UNION ALL

SELECT
    'Customer Retention',
    CONCAT(
        'Retention patterns show ',
        CASE
            WHEN (SELECT AVG(order_count) FROM (
                SELECT customer_id, COUNT(*) AS order_count 
                FROM orders 
                WHERE YEAR(order_date) = @current_year 
                GROUP BY customer_id
            ) cust) > 2.5 THEN 'strong loyalty with repeat purchases'
            WHEN (SELECT AVG(order_count) FROM (
                SELECT customer_id, COUNT(*) AS order_count 
                FROM orders 
                WHERE YEAR(order_date) = @current_year 
                GROUP BY customer_id
            ) cust) > 1.5 THEN 'moderate repeat purchase behavior'
            ELSE 'primarily one-time purchases'
        END
    ),
    'Implement loyalty programs and personalized engagement strategies'

UNION ALL

SELECT
    'Product Portfolio',
    CONCAT(
        'Portfolio has ',
        CASE
            WHEN (SELECT COUNT(*) FROM products WHERE status = 'active') >
                 (SELECT COUNT(*) FROM products WHERE YEAR(created_at) = @previous_year) * 1.2
            THEN 'expanded significantly'
            WHEN (SELECT COUNT(*) FROM products WHERE status = 'active') >
                 (SELECT COUNT(*) FROM products WHERE YEAR(created_at) = @previous_year)
            THEN 'grown modestly'
            ELSE 'remained stable'
        END,
        ' with focus on ',
        (SELECT category_name FROM product_categories pc 
         JOIN products p ON pc.category_id = p.category_id 
         GROUP BY pc.category_name 
         ORDER BY COUNT(*) DESC LIMIT 1),
        ' category'
    ),
    'Diversify offerings and optimize SKU mix based on performance'

UNION ALL

SELECT
    'Seasonal Patterns',
    CONCAT(
        'Peak season identified as ',
        (SELECT MONTHNAME(order_date) FROM orders 
         WHERE YEAR(order_date) = @current_year 
         GROUP BY MONTH(order_date), MONTHNAME(order_date)
         ORDER BY SUM(total_amount) DESC LIMIT 1),
        ' with revenue spikes during holiday periods'
    ),
    'Optimize inventory and marketing spend around seasonal peaks'

UNION ALL

SELECT
    'Operational Efficiency',
    CONCAT(
        'Fulfillment time has ',
        CASE
            WHEN (SELECT AVG(DATEDIFF(updated_at, order_date)) FROM orders 
                  WHERE YEAR(order_date) = @current_year) <
                 (SELECT AVG(DATEDIFF(updated_at, order_date)) FROM orders 
                  WHERE YEAR(order_date) = @previous_year)
            THEN 'improved'
            ELSE 'remained consistent or declined'
        END,
        ' while order volume increased'
    ),
    'Continue process optimization and automation initiatives'

UNION ALL

SELECT
    'Market Position',
    CONCAT(
        'Customer base expanded by ',
        FORMAT(
            (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @current_year) -
            (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year), 0
        ),
        ' new customers, indicating ',
        CASE
            WHEN (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @current_year) >
                 (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year) * 1.2
            THEN 'strong market penetration'
            ELSE 'steady market presence'
        END
    ),
    'Invest in brand awareness and customer acquisition channels';

-- ========================================
-- 16. EXECUTIVE SUMMARY
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'EXECUTIVE SUMMARY - ANNUAL PERFORMANCE' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    CONCAT(@two_years_ago, ' - ', @current_year) AS 'Analysis Period',
    
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid'), 2)
    ) AS 'Current Year Revenue',
    
    CONCAT(FORMAT(
        ((SELECT SUM(total_amount) FROM orders 
          WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
         (SELECT SUM(total_amount) FROM orders 
          WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) * 100.0 /
         NULLIF((SELECT SUM(total_amount) FROM orders 
          WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0), 1
    ), '%') AS 'YoY Growth',
    
    FORMAT(
        (SELECT COUNT(DISTINCT customer_id) FROM customers), 0
    ) AS 'Total Customer Base',
    
    CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(
        (SELECT AVG(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid'), 2)
    ) AS 'Current AOV',
    
    CONCAT(FORMAT(
        (POWER(
            (SELECT SUM(total_amount) FROM orders 
             WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') /
            NULLIF((SELECT SUM(total_amount) FROM orders 
             WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid'), 0),
            1.0/2) - 1) * 100, 1
    ), '%') AS '2-Year CAGR',
    
    CASE
        WHEN (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') >
             (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') * 1.2
        THEN '🚀 High Growth Trajectory'
        WHEN (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') >
             (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')
        THEN '📈 Positive Growth'
        ELSE '→ Stable/Flat'
    END AS 'Business Health';

-- ========================================
-- 17. KEY PERFORMANCE MILESTONES
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'KEY PERFORMANCE MILESTONES' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    1 AS 'Rank',
    CONCAT('Achieved total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid'), 2
    )) AS 'Milestone',
    @current_year AS 'Year',
    'Revenue' AS 'Category',
    '✅' AS 'Status'

UNION ALL

SELECT
    2,
    CONCAT('Grew customer base to ', FORMAT(
        (SELECT COUNT(*) FROM customers), 0
    ), ' total customers'),
    @current_year,
    'Customer Growth',
    '✅'

UNION ALL

SELECT
    3,
    CONCAT('Maintained ', FORMAT(
        (SELECT COUNT(*) FROM products WHERE status = 'active'), 0
    ), ' active products'),
    @current_year,
    'Product Portfolio',
    '✅'

UNION ALL

SELECT
    4,
    CONCAT('Achieved ', FORMAT(
        (SELECT AVG(CASE 
            WHEN status = 'delivered' THEN 100.0 
            ELSE 0 
        END) FROM orders WHERE YEAR(order_date) = @current_year), 1
    ), '% delivery rate'),
    @current_year,
    'Operations',
    CASE
        WHEN (SELECT AVG(CASE WHEN status = 'delivered' THEN 100.0 ELSE 0 END) 
              FROM orders WHERE YEAR(order_date) = @current_year) >= 90 THEN '✅'
        ELSE '⚠️'
    END

UNION ALL

SELECT
    5,
    CONCAT('Processed ', FORMAT(
        (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = @current_year), 0
    ), ' orders in ', @current_year),
    @current_year,
    'Order Volume',
    '✅'

UNION ALL

SELECT
    6,
    CONCAT('Expanded product portfolio by ', FORMAT(
        (SELECT COUNT(*) FROM products 
         WHERE YEAR(created_at) = @current_year), 0
    ), ' new SKUs'),
    @current_year,
    'Product Expansion',
    '✅';

-- ========================================
-- 18. STRATEGIC RECOMMENDATIONS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'STRATEGIC RECOMMENDATIONS FOR NEXT YEAR' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    1 AS 'Priority',
    'Revenue Optimization' AS 'Focus Area',
    CASE
        WHEN (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') >
             (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') * 1.2
        THEN 'Sustain high growth momentum through market expansion and new customer acquisition'
        WHEN (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') >
             (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')
        THEN 'Accelerate growth by optimizing conversion rates and increasing AOV'
        ELSE 'Revitalize revenue through aggressive marketing and product innovation'
    END AS 'Recommendation',
    'High' AS 'Impact',
    'Immediate' AS 'Timeline'

UNION ALL

SELECT
    2,
    'Customer Retention',
    CONCAT('Implement advanced retention programs targeting the ', 
        FORMAT((SELECT COUNT(DISTINCT customer_id) FROM orders 
                WHERE YEAR(order_date) = @current_year), 0),
        ' active customers with personalized engagement'
    ),
    'High',
    'Q1 Next Year'

UNION ALL

SELECT
    3,
    'Product Strategy',
    CONCAT('Focus on top-performing categories (', 
        (SELECT category_name FROM product_categories pc 
         JOIN products p ON pc.category_id = p.category_id 
         JOIN order_items oi ON p.product_id = oi.product_id
         JOIN orders o ON oi.order_id = o.order_id
         WHERE YEAR(o.order_date) = @current_year
         GROUP BY pc.category_name 
         ORDER BY SUM(oi.subtotal) DESC LIMIT 1),
        ') and phase out underperforming SKUs'
    ),
    'Medium',
    'Q1-Q2 Next Year'

UNION ALL

SELECT
    4,
    'Operational Excellence',
    'Invest in automation and process optimization to handle increased volume while reducing costs',
    'Medium',
    'Ongoing'

UNION ALL

SELECT
    5,
    'Market Expansion',
    CONCAT('Expand into high-growth regions beyond current top markets, targeting ',
        (SELECT COUNT(DISTINCT state) FROM customers) + 5,
        ' total states'
    ),
    'Medium',
    'Q2-Q3 Next Year'

UNION ALL

SELECT
    6,
    'Technology & Data',
    'Enhance data quality initiatives to achieve 95%+ quality scores across all data categories',
    'High',
    'Q1 Next Year'

UNION ALL

SELECT
    7,
    'Customer Experience',
    'Reduce return rates and improve fulfillment speed to enhance customer satisfaction',
    'Medium',
    'Ongoing'

UNION ALL

SELECT
    8,
    'Financial Planning',
    CONCAT('Set revenue target of total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(,
        FORMAT((SELECT SUM(total_amount) FROM orders 
                WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') * 1.25, 0),
        ' (25% growth) for next year'
    ),
    'High',
    'Annual Planning'

ORDER BY Priority;

-- ========================================
-- 19. RISK ASSESSMENT - LONG-TERM VIEW
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'LONG-TERM RISK ASSESSMENT' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    'Customer Concentration' AS 'Risk Factor',
    CASE
        WHEN (SELECT SUM(revenue_contribution) FROM (
            SELECT SUM(o.total_amount) AS revenue_contribution
            FROM orders o
            WHERE YEAR(o.order_date) = @current_year
              AND o.payment_status = 'paid'
            GROUP BY o.customer_id
            ORDER BY revenue_contribution DESC
            LIMIT 10
        ) top_10) > 
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') * 0.3
        THEN 'High - Top customers represent >30% of revenue'
        ELSE 'Low - Revenue well distributed'
    END AS 'Assessment',
    'Medium' AS 'Severity',
    'Diversify customer base and reduce dependency on key accounts' AS 'Mitigation Strategy'

UNION ALL

SELECT
    'Growth Sustainability',
    CASE
        WHEN (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') >
             (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') * 1.5
        THEN 'Monitor - Rapid growth may strain operations'
        WHEN (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') <
             (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')
        THEN 'High - Declining revenue trend'
        ELSE 'Low - Healthy sustainable growth'
    END,
    CASE
        WHEN (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') <
             (SELECT SUM(total_amount) FROM orders 
              WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')
        THEN 'High'
        ELSE 'Low'
    END,
    'Ensure infrastructure and team capacity match growth trajectory'

UNION ALL

SELECT
    'Product Portfolio Health',
    CONCAT(
        (SELECT COUNT(*) FROM products WHERE status = 'discontinued'),
        ' discontinued products may indicate portfolio challenges'
    ),
    'Low',
    'Regular portfolio reviews and product lifecycle management'

UNION ALL

SELECT
    'Seasonal Dependency',
    CONCAT('Revenue concentrated in ',
        (SELECT MONTHNAME(order_date) FROM orders 
         WHERE YEAR(order_date) = @current_year 
         GROUP BY MONTH(order_date), MONTHNAME(order_date)
         ORDER BY SUM(total_amount) DESC LIMIT 1),
        ' - diversification needed'
    ),
    'Medium',
    'Develop off-season promotions and expand product offerings'

UNION ALL

SELECT
    'Data Quality',
    CASE
        WHEN (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) scores) < 80
        THEN 'High - Quality below acceptable threshold'
        WHEN (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) scores) < 90
        THEN 'Medium - Quality needs improvement'
        ELSE 'Low - Good data quality'
    END,
    CASE
        WHEN (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) scores) < 80
        THEN 'High'
        ELSE 'Medium'
    END,
    'Implement automated data validation and regular quality audits';

-- ========================================
-- 20. YEAR-OVER-YEAR COMPARISON SUMMARY
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'COMPREHENSIVE YOY COMPARISON SUMMARY' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH yoy_summary AS (
    SELECT
        'Revenue' AS metric,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS prev_value,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS curr_value,
        'currency' AS type_format
    
    UNION ALL
    
    SELECT
        'Orders',
        (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = @previous_year),
        (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = @current_year),
        'count'
    
    UNION ALL
    
    SELECT
        'New Customers',
        (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year),
        (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @current_year),
        'count'
    
    UNION ALL
    
    SELECT
        'Avg Order Value',
        (SELECT AVG(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'),
        (SELECT AVG(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid'),
        'currency'
    
    UNION ALL
    
    SELECT
        'Active Products',
        (SELECT COUNT(*) FROM products 
         WHERE YEAR(created_at) <= @previous_year AND status = 'active'),
        (SELECT COUNT(*) FROM products WHERE status = 'active'),
        'count'
    
    UNION ALL
    
    SELECT
        'Delivery Rate (%)',
        (SELECT COUNT(CASE WHEN status = 'delivered' THEN 1 END) * 100.0 / COUNT(*)
         FROM orders WHERE YEAR(order_date) = @previous_year),
        (SELECT COUNT(CASE WHEN status = 'delivered' THEN 1 END) * 100.0 / COUNT(*)
         FROM orders WHERE YEAR(order_date) = @current_year),
        'percentage'
    
    UNION ALL
    
    SELECT
        'Return Rate (%)',
        (SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = @previous_year)
         FROM returns WHERE YEAR(created_at) = @previous_year),
        (SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = @current_year)
         FROM returns WHERE YEAR(created_at) = @current_year),
        'percentage'
)
SELECT
    metric AS 'Metric',
    CASE 
        WHEN type_format = 'currency' THEN CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(prev_value, 2))
        WHEN type_format = 'percentage' THEN CONCAT(FORMAT(prev_value, 2), '%')
        ELSE FORMAT(prev_value, 0)
    END AS CONCAT(@previous_year, ' Value'),
    CASE 
        WHEN type_format = 'currency' THEN CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(curr_value, 2))
        WHEN type_format = 'percentage' THEN CONCAT(FORMAT(curr_value, 2), '%')
        ELSE FORMAT(curr_value, 0)
    END AS CONCAT(@current_year, ' Value'),
    CASE 
        WHEN type_format = 'currency' THEN CONCAT('total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;

-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(, FORMAT(curr_value - prev_value, 2))
        WHEN type_format = 'percentage' THEN CONCAT(FORMAT(curr_value - prev_value, 2), ' pp')
        ELSE FORMAT(curr_value - prev_value, 0)
    END AS 'Absolute Change',
    CONCAT(FORMAT(((curr_value - prev_value) / NULLIF(prev_value, 0) * 100), 1), '%') AS 'Percent Change',
    CASE
        WHEN ((curr_value - prev_value) / NULLIF(prev_value, 0) * 100) > 20 THEN '🚀 Exceptional'
        WHEN ((curr_value - prev_value) / NULLIF(prev_value, 0) * 100) > 10 THEN '📈 Strong Growth'
        WHEN ((curr_value - prev_value) / NULLIF(prev_value, 0) * 100) > 5 THEN '✓ Good Growth'
        WHEN ((curr_value - prev_value) / NULLIF(prev_value, 0) * 100) > 0 THEN '→ Slight Growth'
        WHEN ((curr_value - prev_value) / NULLIF(prev_value, 0) * 100) = 0 THEN '→ Flat'
        WHEN ((curr_value - prev_value) / NULLIF(prev_value, 0) * 100) > -5 THEN '↘️ Slight Decline'
        WHEN ((curr_value - prev_value) / NULLIF(prev_value, 0) * 100) > -10 THEN '📉 Declining'
        ELSE '❌ Significant Decline'
    END AS 'Status'
FROM yoy_summary;

-- ========================================
-- 21. FINAL SCORECARD
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'ANNUAL PERFORMANCE SCORECARD' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    'Revenue Growth' AS 'Performance Area',
    CASE
        WHEN ((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
              (SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
              NULLIF((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 30 THEN 'A+'
        WHEN ((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
              (SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
              NULLIF((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 20 THEN 'A'
        WHEN ((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
              (SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
              NULLIF((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 10 THEN 'B'
        WHEN ((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
              (SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
              NULLIF((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 5 THEN 'C'
        WHEN ((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
              (SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
              NULLIF((SELECT SUM(total_amount) FROM orders 
               WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 0 THEN 'D'
        ELSE 'F'
    END AS 'Grade',
    CONCAT(FORMAT(
        ((SELECT SUM(total_amount) FROM orders 
          WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
         (SELECT SUM(total_amount) FROM orders 
          WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
         NULLIF((SELECT SUM(total_amount) FROM orders 
          WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100, 1
    ), '% YoY') AS 'Score',
    25.0 AS 'Weight %',
    'Critical for business sustainability' AS 'Importance'

UNION ALL

SELECT
    'Customer Acquisition',
    CASE
        WHEN (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @current_year) >
             (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year) * 1.25 THEN 'A+'
        WHEN (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @current_year) >
             (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year) * 1.15 THEN 'A'
        WHEN (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @current_year) >
             (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year) THEN 'B'
        WHEN (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @current_year) >=
             (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year) * 0.9 THEN 'C'
        ELSE 'F'
    END,
    CONCAT(FORMAT(
        ((SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @current_year) -
         (SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year)) /
         NULLIF((SELECT COUNT(*) FROM customers WHERE YEAR(created_at) = @previous_year), 0) * 100, 1
    ), '% YoY'),
    20.0,
    'Drives future revenue growth'

UNION ALL

SELECT
    'Operational Excellence',
    CASE
        WHEN (SELECT AVG(CASE WHEN status = 'delivered' THEN 100.0 ELSE 0 END)
              FROM orders WHERE YEAR(order_date) = @current_year) >= 95 THEN 'A+'
        WHEN (SELECT AVG(CASE WHEN status = 'delivered' THEN 100.0 ELSE 0 END)
              FROM orders WHERE YEAR(order_date) = @current_year) >= 90 THEN 'A'
        WHEN (SELECT AVG(CASE WHEN status = 'delivered' THEN 100.0 ELSE 0 END)
              FROM orders WHERE YEAR(order_date) = @current_year) >= 85 THEN 'B'
        WHEN (SELECT AVG(CASE WHEN status = 'delivered' THEN 100.0 ELSE 0 END)
              FROM orders WHERE YEAR(order_date) = @current_year) >= 80 THEN 'C'
        ELSE 'F'
    END,
    CONCAT(FORMAT(
        (SELECT AVG(CASE WHEN status = 'delivered' THEN 100.0 ELSE 0 END)
         FROM orders WHERE YEAR(order_date) = @current_year), 1
    ), '% delivery'),
    20.0,
    'Impacts customer satisfaction'

UNION ALL

SELECT
    'Product Performance',
    CASE
        WHEN (SELECT COUNT(*) FROM products WHERE status = 'active') >
             (SELECT COUNT(*) FROM products WHERE YEAR(created_at) = @previous_year) * 1.2 THEN 'A+'
        WHEN (SELECT COUNT(*) FROM products WHERE status = 'active') >
             (SELECT COUNT(*) FROM products WHERE YEAR(created_at) = @previous_year) THEN 'A'
        ELSE 'B'
    END,
    CONCAT(FORMAT((SELECT COUNT(*) FROM products WHERE status = 'active'), 0), ' active SKUs'),
    15.0,
    'Enables market differentiation'

UNION ALL

SELECT
    'Data Quality',
    CASE
        WHEN (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) scores) >= 95 THEN 'A+'
        WHEN (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) scores) >= 90 THEN 'A'
        WHEN (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) scores) >= 85 THEN 'B'
        ELSE 'C'
    END,
    CONCAT(FORMAT(
        (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) scores), 1
    ), '% quality'),
    20.0,
    'Foundation for decision making';

-- Overall Grade Calculation
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'OVERALL ANNUAL GRADE' AS 'Summary';

SELECT
    CASE
        WHEN AVG(grade_value) >= 95 THEN 'A+ (Exceptional Performance)'
        WHEN AVG(grade_value) >= 90 THEN 'A (Excellent Performance)'
        WHEN AVG(grade_value) >= 85 THEN 'B+ (Very Good Performance)'
        WHEN AVG(grade_value) >= 80 THEN 'B (Good Performance)'
        WHEN AVG(grade_value) >= 75 THEN 'C+ (Satisfactory Performance)'
        WHEN AVG(grade_value) >= 70 THEN 'C (Fair Performance)'
        ELSE 'D (Needs Improvement)'
    END AS 'Overall Grade',
    CONCAT(FORMAT(AVG(grade_value), 1), '%') AS 'Weighted Score',
    CASE
        WHEN AVG(grade_value) >= 90 THEN '✅ Excellent - Continue current trajectory'
        WHEN AVG(grade_value) >= 80 THEN '✓ Good - Minor improvements needed'
        WHEN AVG(grade_value) >= 70 THEN '⚠️ Fair - Strategic changes required'
        ELSE '❌ Poor - Immediate action needed'
    END AS 'Assessment'
FROM (
    SELECT
        CASE
            WHEN ((SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
                  (SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
                  NULLIF((SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 30 THEN 98
            WHEN ((SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
                  (SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
                  NULLIF((SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 20 THEN 93
            WHEN ((SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
                  (SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
                  NULLIF((SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 10 THEN 85
            WHEN ((SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') -
                  (SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid')) /
                  NULLIF((SELECT SUM(total_amount) FROM orders 
                   WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid'), 0) * 100 >= 5 THEN 75
            ELSE 60
        END AS grade_value
) grades;

-- ========================================
-- REPORT FOOTER
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT CONCAT('Annual Analysis Report Generated: ', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')) AS 'Timestamp';
SELECT CONCAT('Analysis Period: ', @two_years_ago, ' through ', @current_year) AS 'Report Scope';
SELECT CONCAT('Total Years Analyzed: 3 years (', @two_years_ago, ', ', @previous_year, ', ', @current_year, ')') AS 'Coverage';
SELECT 'End of Annual Analysis Report' AS 'Report Status';
SELECT '══════════════════════════════════════════════════════════' AS Separator;total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders * 100.0 / total_orders), 1), '%') AS 'Cancellation Rate'
FROM annual_revenue
ORDER BY year;

-- Year-over-Year Growth Rates
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'YEAR-OVER-YEAR GROWTH RATES' AS 'Sub-Section';

WITH yoy_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @current_year AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @previous_year AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE YEAR(order_date) = @two_years_ago AND payment_status = 'paid') AS two_years_revenue,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @current_year) AS current_orders,
        (SELECT COUNT(DISTINCT order_id) FROM orders 
         WHERE YEAR(order_date) = @previous_year) AS previous_orders,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @current_year) AS current_new_customers,
        (SELECT COUNT(DISTINCT customer_id) FROM customers 
         WHERE YEAR(created_at) = @previous_year) AS previous_new_customers
)
SELECT
    'Revenue Growth' AS 'Metric',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS CONCAT(@previous_year, ' Value'),
    CONCAT('$', FORMAT(current_revenue, 2)) AS CONCAT(@current_year, ' Value'),
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'YoY Growth',
    CONCAT(FORMAT(((previous_revenue - two_years_revenue) / two_years_revenue * 100), 1), '%') AS 'Previous YoY',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) > 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📈 Accelerating'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) < 
             ((previous_revenue - two_years_revenue) / two_years_revenue * 100) THEN '📉 Decelerating'
        ELSE '→ Stable'
    END AS 'Trend'
FROM yoy_comparison

UNION ALL

SELECT
    'Order Volume',
    FORMAT(previous_orders, 0),
    FORMAT(current_orders, 0),
    CONCAT(FORMAT(((current_orders - previous_orders) / previous_orders * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_orders > previous_orders THEN '📈 Growing'
        WHEN current_orders < previous_orders THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison

UNION ALL

SELECT
    'New Customer Acquisition',
    FORMAT(previous_new_customers, 0),
    FORMAT(current_new_customers, 0),
    CONCAT(FORMAT(((current_new_customers - previous_new_customers) / previous_new_customers * 100), 1), '%'),
    'N/A',
    CASE
        WHEN current_new_customers > previous_new_customers THEN '📈 Growing'
        WHEN current_new_customers < previous_new_customers THEN '📉 Declining'
        ELSE '→ Flat'
    END
FROM yoy_comparison;

-- ========================================
-- 3. MONTHLY TRENDS ACROSS YEARS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MONTHLY REVENUE TRENDS - MULTI-YEAR COMPARISON' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH monthly_trends AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth %',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) > 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↗️'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) < 
             MAX(CASE WHEN year = @previous_year THEN revenue END) THEN '↘️'
        ELSE '→'
    END AS 'Trend'
FROM monthly_trends
GROUP BY month_name, month_num
ORDER BY month_num;

-- ========================================
-- 4. QUARTERLY PERFORMANCE TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY PERFORMANCE - MULTI-YEAR VIEW' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH quarterly_trends AS (
    SELECT
        CONCAT('Q', QUARTER(order_date)) AS quarter,
        QUARTER(order_date) AS quarter_num,
        YEAR(order_date) AS year,
        SUM(total_amount) AS revenue,
        COUNT(DISTINCT order_id) AS orders,
        COUNT(DISTINCT customer_id) AS customers,
        AVG(total_amount) AS aov
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND payment_status = 'paid'
      AND status NOT IN ('cancelled')
    GROUP BY YEAR(order_date), QUARTER(order_date)
)
SELECT
    quarter AS 'Quarter',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN orders END), 0) AS 'Current Orders',
    FORMAT(MAX(CASE WHEN year = @current_year THEN customers END), 0) AS 'Current Customers',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN aov END), 2)) AS 'Current AOV'
FROM quarterly_trends
GROUP BY quarter, quarter_num
ORDER BY quarter_num;

-- ========================================
-- 5. CUSTOMER LIFETIME VALUE EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_cohorts AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS customers_acquired,
        AVG(customer_stats.total_spent) AS avg_clv,
        AVG(customer_stats.order_count) AS avg_orders,
        AVG(customer_stats.customer_lifetime_days) AS avg_lifetime_days,
        SUM(customer_stats.total_spent) AS cohort_total_revenue
    FROM customers c
    LEFT JOIN (
        SELECT
            customer_id,
            SUM(total_amount) AS total_spent,
            COUNT(DISTINCT order_id) AS order_count,
            DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifetime_days
        FROM orders
        WHERE payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY customer_id
    ) customer_stats ON c.customer_id = customer_stats.customer_id
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(customers_acquired, 0) AS 'Customers Acquired',
    CONCAT('$', FORMAT(avg_clv, 2)) AS 'Avg Customer LTV',
    FORMAT(avg_orders, 1) AS 'Avg Orders per Customer',
    CONCAT(FORMAT(avg_lifetime_days, 0), ' days') AS 'Avg Customer Lifespan',
    CONCAT('$', FORMAT(cohort_total_revenue, 2)) AS 'Total Cohort Revenue',
    CONCAT('$', FORMAT(cohort_total_revenue / customers_acquired, 2)) AS 'Revenue per Acquisition'
FROM customer_cohorts
ORDER BY cohort_year;

-- Customer Retention by Cohort
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'COHORT RETENTION ANALYSIS' AS 'Sub-Section';

WITH cohort_retention AS (
    SELECT
        YEAR(c.created_at) AS cohort_year,
        COUNT(DISTINCT c.customer_id) AS total_customers,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 1
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year1,
        COUNT(DISTINCT CASE 
            WHEN EXISTS (
                SELECT 1 FROM orders o 
                WHERE o.customer_id = c.customer_id 
                AND YEAR(o.order_date) = YEAR(c.created_at) + 2
                AND o.payment_status = 'paid'
            ) THEN c.customer_id 
        END) AS retained_year2
    FROM customers c
    WHERE YEAR(c.created_at) IN (@two_years_ago, @previous_year)
    GROUP BY YEAR(c.created_at)
)
SELECT
    cohort_year AS 'Cohort Year',
    FORMAT(total_customers, 0) AS 'Initial Customers',
    FORMAT(retained_year1, 0) AS 'Retained Year 1',
    CONCAT(FORMAT((retained_year1 * 100.0 / total_customers), 1), '%') AS 'Year 1 Retention',
    FORMAT(retained_year2, 0) AS 'Retained Year 2',
    CONCAT(FORMAT((retained_year2 * 100.0 / total_customers), 1), '%') AS 'Year 2 Retention',
    CASE
        WHEN (retained_year1 * 100.0 / total_customers) >= 60 THEN '✅ Excellent'
        WHEN (retained_year1 * 100.0 / total_customers) >= 40 THEN '✓ Good'
        WHEN (retained_year1 * 100.0 / total_customers) >= 25 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Retention Health'
FROM cohort_retention
ORDER BY cohort_year;

-- ========================================
-- 6. PRODUCT PORTFOLIO EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PORTFOLIO TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH product_trends AS (
    SELECT
        YEAR(p.created_at) AS year,
        COUNT(DISTINCT p.product_id) AS products_added,
        COUNT(DISTINCT CASE WHEN p.status = 'discontinued' THEN p.product_id END) AS products_discontinued,
        COUNT(DISTINCT CASE WHEN p.status = 'active' THEN p.product_id END) AS active_products,
        AVG(p.price) AS avg_price,
        AVG(p.cost) AS avg_cost,
        AVG((p.price - p.cost) / p.price * 100) AS avg_margin_pct
    FROM products p
    WHERE YEAR(p.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(p.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(products_added, 0) AS 'New Products',
    FORMAT(products_discontinued, 0) AS 'Discontinued',
    FORMAT(active_products, 0) AS 'Active Products',
    CONCAT('$', FORMAT(avg_price, 2)) AS 'Avg Price',
    CONCAT('$', FORMAT(avg_cost, 2)) AS 'Avg Cost',
    CONCAT(FORMAT(avg_margin_pct, 1), '%') AS 'Avg Margin'
FROM product_trends
ORDER BY year;

-- Category Performance Evolution
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE OVER TIME' AS 'Sub-Section';

WITH category_annual AS (
    SELECT
        pc.category_name,
        YEAR(o.order_date) AS year,
        SUM(oi.subtotal) AS revenue,
        SUM(oi.quantity) AS units_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    WHERE YEAR(o.order_date) IN (@two_years_ago, @previous_year, @current_year)
      AND o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
    GROUP BY pc.category_name, YEAR(o.order_date)
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @two_years_ago THEN revenue END), 2)) AS CONCAT(@two_years_ago, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @previous_year THEN revenue END), 2)) AS CONCAT(@previous_year, ' Revenue'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN revenue END), 2)) AS CONCAT(@current_year, ' Revenue'),
    FORMAT(MAX(CASE WHEN year = @current_year THEN units_sold END), 0) AS 'Current Year Units',
    CONCAT(FORMAT(
        ((MAX(CASE WHEN year = @current_year THEN revenue END) - 
          MAX(CASE WHEN year = @previous_year THEN revenue END)) / 
          NULLIF(MAX(CASE WHEN year = @previous_year THEN revenue END), 0) * 100), 1), '%'
    ) AS 'YoY Growth',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @two_years_ago THEN revenue END) IS NOT NULL THEN 'Consistent'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NOT NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Growing'
        WHEN MAX(CASE WHEN year = @current_year THEN revenue END) IS NULL AND
             MAX(CASE WHEN year = @previous_year THEN revenue END) IS NOT NULL THEN 'Declining'
        ELSE 'New'
    END AS 'Trend Pattern'
FROM category_annual
GROUP BY category_name
ORDER BY MAX(CASE WHEN year = @current_year THEN revenue END) DESC
LIMIT 15;

-- ========================================
-- 7. SEASONALITY PATTERNS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'SEASONALITY & CYCLICAL PATTERNS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH seasonal_patterns AS (
    SELECT
        MONTHNAME(order_date) AS month_name,
        MONTH(order_date) AS month_num,
        AVG(total_amount) AS avg_monthly_revenue,
        AVG(order_count) AS avg_monthly_orders,
        STDDEV(total_amount) AS revenue_volatility
    FROM (
        SELECT
            order_date,
            SUM(total_amount) AS total_amount,
            COUNT(*) AS order_count
        FROM orders
        WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
          AND payment_status = 'paid'
          AND status NOT IN ('cancelled')
        GROUP BY order_date
    ) daily_data
    GROUP BY MONTH(order_date), MONTHNAME(order_date)
)
SELECT
    month_name AS 'Month',
    CONCAT('$', FORMAT(avg_monthly_revenue, 2)) AS 'Avg Daily Revenue',
    FORMAT(avg_monthly_orders, 0) AS 'Avg Daily Orders',
    CONCAT('$', FORMAT(revenue_volatility, 2)) AS 'Revenue Volatility',
    CASE
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.2 FROM seasonal_patterns) THEN '🔥 Peak Season'
        WHEN avg_monthly_revenue > (SELECT AVG(avg_monthly_revenue) * 1.1 FROM seasonal_patterns) THEN '📈 High Season'
        WHEN avg_monthly_revenue < (SELECT AVG(avg_monthly_revenue) * 0.8 FROM seasonal_patterns) THEN '❄️ Low Season'
        ELSE '→ Normal'
    END AS 'Season Type',
    CONCAT(FORMAT(
        (avg_monthly_revenue / (SELECT AVG(avg_monthly_revenue) FROM seasonal_patterns) * 100), 0), '%'
    ) AS 'vs Annual Avg'
FROM seasonal_patterns
ORDER BY month_num;

-- Day of Week Patterns
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'DAY OF WEEK PERFORMANCE PATTERNS' AS 'Sub-Section';

SELECT
    DAYNAME(order_date) AS 'Day of Week',
    DAYOFWEEK(order_date) AS day_num,
    FORMAT(COUNT(*), 0) AS 'Total Orders',
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS 'Avg Order Value',
    CONCAT(FORMAT(
        (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders 
                              WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
                              AND payment_status = 'paid')), 1), '%'
    ) AS '% of Orders',
    CASE
        WHEN COUNT(*) > (SELECT AVG(order_count) * 1.15 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📈 High Traffic'
        WHEN COUNT(*) < (SELECT AVG(order_count) * 0.85 FROM (
            SELECT COUNT(*) AS order_count FROM orders 
            WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
            AND payment_status = 'paid'
            GROUP BY DAYOFWEEK(order_date)
        ) daily_avg) THEN '📉 Low Traffic'
        ELSE '→ Average'
    END AS 'Traffic Pattern'
FROM orders
WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
  AND payment_status = 'paid'
  AND status NOT IN ('cancelled')
GROUP BY DAYNAME(order_date), DAYOFWEEK(order_date)
ORDER BY day_num;

-- ========================================
-- 8. MARKETING CHANNEL EVOLUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CHANNEL PERFORMANCE TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH channel_annual AS (
    SELECT
        c.campaign_type AS channel,
        YEAR(c.start_date) AS year,
        COUNT(DISTINCT c.campaign_id) AS campaigns,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue,
        SUM(cp.conversions) AS total_conversions,
        AVG((cp.revenue - cp.spend) / NULLIF(cp.spend, 0) * 100) AS avg_roi
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE YEAR(c.start_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY c.campaign_type, YEAR(c.start_date)
)
SELECT
    channel AS 'Channel',
    MAX(CASE WHEN year = @two_years_ago THEN campaigns END) AS CONCAT(@two_years_ago, ' Campaigns'),
    MAX(CASE WHEN year = @previous_year THEN campaigns END) AS CONCAT(@previous_year, ' Campaigns'),
    MAX(CASE WHEN year = @current_year THEN campaigns END) AS CONCAT(@current_year, ' Campaigns'),
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_spend END), 2)) AS 'Current Spend',
    CONCAT('$', FORMAT(MAX(CASE WHEN year = @current_year THEN total_revenue END), 2)) AS 'Current Revenue',
    CONCAT(FORMAT(MAX(CASE WHEN year = @current_year THEN avg_roi END), 1), '%') AS 'Current ROI',
    CASE
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 200 THEN '🏆 Excellent'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 100 THEN '⭐ Good'
        WHEN MAX(CASE WHEN year = @current_year THEN avg_roi END) >= 50 THEN '✓ Fair'
        ELSE '⚠️ Poor'
    END AS 'Performance'
FROM channel_annual
GROUP BY channel
ORDER BY MAX(CASE WHEN year = @current_year THEN total_revenue END) DESC;

-- ========================================
-- 9. OPERATIONAL METRICS TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_annual AS (
    SELECT
        YEAR(order_date) AS year,
        COUNT(*) AS total_orders,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS delivery_rate,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS cancellation_rate,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS payment_failure_rate,
        AVG(shipping_cost) AS avg_shipping_cost,
        SUM(shipping_cost) AS total_shipping_cost
    FROM orders
    WHERE YEAR(order_date) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(order_date)
)
SELECT
    year AS 'Year',
    FORMAT(total_orders, 0) AS 'Total Orders',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT(FORMAT(delivery_rate, 1), '%') AS 'Delivery Rate',
    CONCAT(FORMAT(cancellation_rate, 1), '%') AS 'Cancellation Rate',
    CONCAT(FORMAT(payment_failure_rate, 1), '%') AS 'Payment Failure Rate',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CASE
        WHEN delivery_rate >= 95 AND cancellation_rate <= 5 THEN '✅ Excellent'
        WHEN delivery_rate >= 90 AND cancellation_rate <= 8 THEN '✓ Good'
        WHEN delivery_rate >= 85 AND cancellation_rate <= 12 THEN '⚠️ Fair'
        ELSE '❌ Needs Improvement'
    END AS 'Operational Health'
FROM operational_annual
ORDER BY year;
-- Returns & Refunds Trends
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS TRENDS' AS 'Sub-Section';

WITH returns_annual AS (
    SELECT
        YEAR(r.created_at) AS year,
        COUNT(*) AS total_returns,
        SUM(r.refund_amount) AS total_refunds,
        AVG(r.refund_amount) AS avg_refund,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE YEAR(order_date) = YEAR(r.created_at)) AS return_rate
    FROM returns r
    WHERE YEAR(r.created_at) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(r.created_at)
)
SELECT
    year AS 'Year',
    FORMAT(total_returns, 0) AS 'Total Returns',
    CONCAT('$', FORMAT(total_refunds, 2)) AS 'Total Refunds',
    CONCAT('$', FORMAT(avg_refund, 2)) AS 'Avg Refund Amount',
    CONCAT(FORMAT(return_rate, 2), '%') AS 'Return Rate',
    CASE
        WHEN return_rate <= 5 THEN '✅ Excellent'
        WHEN return_rate <= 10 THEN '✓ Good'
        WHEN return_rate <= 15 THEN '⚠️ Fair'
        ELSE '❌ High'
    END AS 'Return Performance'
FROM returns_annual
ORDER BY year;

-- ========================================
-- 10. INVENTORY TRENDS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'INVENTORY MANAGEMENT TRENDS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH inventory_snapshots AS (
    SELECT
        YEAR(i.last_updated) AS year,
        COUNT(DISTINCT i.product_id) AS total_skus,
        SUM(i.quantity_on_hand) AS total_units,
        SUM(i.quantity_on_hand * p.cost) AS total_inventory_value,
        AVG(i.quantity_on_hand) AS avg_stock_per_sku,
        SUM(CASE WHEN i.quantity_on_hand = 0 THEN 1 ELSE 0 END) AS out_of_stock_count,
        SUM(CASE WHEN i.quantity_on_hand < i.reorder_level THEN 1 ELSE 0 END) AS below_reorder_count,
        SUM(CASE WHEN i.quantity_reserved > i.quantity_on_hand THEN 1 ELSE 0 END) AS overbooked_count
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    WHERE YEAR(i.last_updated) IN (@two_years_ago, @previous_year, @current_year)
    GROUP BY YEAR(i.last_updated)
)
SELECT
    year AS 'Year',
    FORMAT(total_skus, 0) AS 'Total SKUs',
    FORMAT(total_units, 0) AS 'Total Units',
    CONCAT('$', FORMAT(total_inventory_value, 2)) AS 'Inventory Value',
    FORMAT(avg_stock_per_sku, 1) AS 'Avg Stock/SKU',
    FORMAT(out_of_stock_count, 0) AS 'Out of Stock',
    CONCAT(FORMAT((out_of_stock_count * 100.0 / total_skus), 1), '%') AS 'Stockout Rate',
    FORMAT(below_reorder_count, 0) AS 'Below Reorder',
    FORMAT(overbooked_count, 0) AS 'Overbooked',
    CASE
        WHEN (out_of_stock_count * 100.0 / total_skus) <= 5 THEN '✅ Excellent'
        WHEN (out_of_stock_count * 100.0 / total_skus) <= 10 THEN '✓ Good'
        WHEN (out_of_stock_count * 100.0 / total_skus) <= 15 THEN '⚠️ Fair'
        ELSE '❌ Poor'
    END AS 'Inventory Health'
FROM inventory_snapshots
ORDER BY year;

-- ========================================
-- REPORT COMPLETE
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT CONCAT('Annual Analysis Report Generated: ', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')) AS 'Timestamp';
SELECT CONCAT('Analysis Period: ', @two_years_ago, ' through ', @current_year) AS 'Report Scope';
SELECT CONCAT('Total Years Analyzed: 3 years') AS 'Coverage';
SELECT 'End of Annual Analysis Report' AS 'Report Status';
SELECT '══════════════════════════════════════════════════════════' AS Separator;