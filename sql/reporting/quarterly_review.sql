-- ========================================
-- QUARTERLY BUSINESS REVIEW (QBR)
-- Strategic metrics and goal tracking
-- Comprehensive quarterly performance analysis
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. DEFINE QUARTER PERIODS
-- ========================================
SET @current_quarter_start = DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL MOD(MONTH(CURDATE())-1, 3) MONTH), '%Y-%m-01');
SET @current_quarter_end = LAST_DAY(DATE_ADD(@current_quarter_start, INTERVAL 2 MONTH));
SET @previous_quarter_start = DATE_SUB(@current_quarter_start, INTERVAL 3 MONTH);
SET @previous_quarter_end = LAST_DAY(DATE_ADD(@previous_quarter_start, INTERVAL 2 MONTH));
SET @year_ago_quarter_start = DATE_SUB(@current_quarter_start, INTERVAL 12 MONTH);
SET @year_ago_quarter_end = LAST_DAY(DATE_ADD(@year_ago_quarter_start, INTERVAL 2 MONTH));

-- Display quarter information
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY BUSINESS REVIEW' AS 'Report Title';
SELECT CONCAT('Q', QUARTER(CURDATE()), ' ', YEAR(CURDATE())) AS 'Current Quarter';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    'Current Quarter' AS Period,
    @current_quarter_start AS 'Start Date',
    @current_quarter_end AS 'End Date'
UNION ALL
SELECT
    'Previous Quarter',
    @previous_quarter_start,
    @previous_quarter_end
UNION ALL
SELECT
    'Year Ago Quarter',
    @year_ago_quarter_start,
    @year_ago_quarter_end;

-- ========================================
-- 2. REVENUE METRICS & GOALS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'REVENUE PERFORMANCE' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH revenue_metrics AS (
    SELECT
        'Current Quarter' AS period,
        COUNT(DISTINCT order_id) AS total_orders,
        COUNT(DISTINCT customer_id) AS unique_customers,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_order_value,
        SUM(total_amount) / COUNT(DISTINCT customer_id) AS revenue_per_customer
    FROM orders
    WHERE order_date >= @current_quarter_start 
      AND order_date <= @current_quarter_end
      AND status NOT IN ('cancelled')
      AND payment_status = 'paid'
    
    UNION ALL
    
    SELECT
        'Previous Quarter',
        COUNT(DISTINCT order_id),
        COUNT(DISTINCT customer_id),
        SUM(total_amount),
        AVG(total_amount),
        SUM(total_amount) / COUNT(DISTINCT customer_id)
    FROM orders
    WHERE order_date >= @previous_quarter_start 
      AND order_date <= @previous_quarter_end
      AND status NOT IN ('cancelled')
      AND payment_status = 'paid'
    
    UNION ALL
    
    SELECT
        'Year Ago Quarter',
        COUNT(DISTINCT order_id),
        COUNT(DISTINCT customer_id),
        SUM(total_amount),
        AVG(total_amount),
        SUM(total_amount) / COUNT(DISTINCT customer_id)
    FROM orders
    WHERE order_date >= @year_ago_quarter_start 
      AND order_date <= @year_ago_quarter_end
      AND status NOT IN ('cancelled')
      AND payment_status = 'paid'
)
SELECT
    period AS 'Period',
    FORMAT(total_orders, 0) AS 'Total Orders',
    FORMAT(unique_customers, 0) AS 'Unique Customers',
    CONCAT('$', FORMAT(total_revenue, 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(avg_order_value, 2)) AS 'Avg Order Value',
    CONCAT('$', FORMAT(revenue_per_customer, 2)) AS 'Revenue/Customer'
FROM revenue_metrics
ORDER BY FIELD(period, 'Current Quarter', 'Previous Quarter', 'Year Ago Quarter');

-- Revenue Growth Analysis
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'REVENUE GROWTH ANALYSIS' AS 'Sub-Section';

WITH revenue_comparison AS (
    SELECT
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
         AND status NOT IN ('cancelled') AND payment_status = 'paid') AS current_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= @previous_quarter_start AND order_date <= @previous_quarter_end
         AND status NOT IN ('cancelled') AND payment_status = 'paid') AS previous_revenue,
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= @year_ago_quarter_start AND order_date <= @year_ago_quarter_end
         AND status NOT IN ('cancelled') AND payment_status = 'paid') AS yoy_revenue
)
SELECT
    CONCAT('$', FORMAT(current_revenue, 2)) AS 'Current Quarter Revenue',
    CONCAT('$', FORMAT(previous_revenue, 2)) AS 'Previous Quarter Revenue',
    CONCAT(FORMAT(((current_revenue - previous_revenue) / previous_revenue * 100), 1), '%') AS 'QoQ Growth',
    CONCAT('$', FORMAT(yoy_revenue, 2)) AS 'Year Ago Revenue',
    CONCAT(FORMAT(((current_revenue - yoy_revenue) / yoy_revenue * 100), 1), '%') AS 'YoY Growth',
    CASE
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) >= 20 THEN '🚀 Exceptional'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) >= 10 THEN ''TRENDING_UP' Strong'
        WHEN ((current_revenue - previous_revenue) / previous_quarter * 100) >= 5 THEN '✓ Good'
        WHEN ((current_revenue - previous_revenue) / previous_revenue * 100) >= 0 THEN '→ Stable'
        ELSE ''WARNING' Declining'
    END AS 'Performance Status'
FROM revenue_comparison;

-- ========================================
-- 3. CUSTOMER ACQUISITION & RETENTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER METRICS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_metrics AS (
    -- Current Quarter
    SELECT
        'Current Quarter' AS period,
        COUNT(DISTINCT CASE WHEN created_at >= @current_quarter_start THEN customer_id END) AS new_customers,
        COUNT(DISTINCT CASE 
            WHEN created_at < @current_quarter_start 
            AND customer_id IN (
                SELECT DISTINCT customer_id FROM orders 
                WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
            ) THEN customer_id 
        END) AS returning_customers,
        COUNT(DISTINCT customer_id) AS total_active_customers
    FROM customers
    WHERE created_at <= @current_quarter_end
    
    UNION ALL
    
    -- Previous Quarter
    SELECT
        'Previous Quarter',
        COUNT(DISTINCT CASE WHEN created_at >= @previous_quarter_start AND created_at <= @previous_quarter_end THEN customer_id END),
        COUNT(DISTINCT CASE 
            WHEN created_at < @previous_quarter_start 
            AND customer_id IN (
                SELECT DISTINCT customer_id FROM orders 
                WHERE order_date >= @previous_quarter_start AND order_date <= @previous_quarter_end
            ) THEN customer_id 
        END),
        COUNT(DISTINCT CASE WHEN created_at <= @previous_quarter_end THEN customer_id END)
    FROM customers
    WHERE created_at <= @previous_quarter_end
    
    UNION ALL
    
    -- Year Ago Quarter
    SELECT
        'Year Ago Quarter',
        COUNT(DISTINCT CASE WHEN created_at >= @year_ago_quarter_start AND created_at <= @year_ago_quarter_end THEN customer_id END),
        COUNT(DISTINCT CASE 
            WHEN created_at < @year_ago_quarter_start 
            AND customer_id IN (
                SELECT DISTINCT customer_id FROM orders 
                WHERE order_date >= @year_ago_quarter_start AND order_date <= @year_ago_quarter_end
            ) THEN customer_id 
        END),
        COUNT(DISTINCT CASE WHEN created_at <= @year_ago_quarter_end THEN customer_id END)
    FROM customers
    WHERE created_at <= @year_ago_quarter_end
)
SELECT
    period AS 'Period',
    FORMAT(new_customers, 0) AS 'New Customers',
    FORMAT(returning_customers, 0) AS 'Returning Customers',
    CONCAT(FORMAT((returning_customers / (new_customers + returning_customers) * 100), 1), '%') AS 'Retention Rate',
    FORMAT(total_active_customers, 0) AS 'Total Active Customers'
FROM customer_metrics
ORDER BY FIELD(period, 'Current Quarter', 'Previous Quarter', 'Year Ago Quarter');

-- Customer Lifetime Value (CLV)
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CUSTOMER LIFETIME VALUE ANALYSIS' AS 'Sub-Section';

SELECT
    CONCAT('$', FORMAT(AVG(customer_value), 2)) AS 'Average CLV',
    CONCAT('$', FORMAT(MIN(customer_value), 2)) AS 'Minimum CLV',
    CONCAT('$', FORMAT(MAX(customer_value), 2)) AS 'Maximum CLV',
    CONCAT('$', FORMAT(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY customer_value), 2)) AS 'Median CLV',
    FORMAT(AVG(order_count), 1) AS 'Avg Orders per Customer',
    FORMAT(AVG(DATEDIFF(@current_quarter_end, first_order_date)), 0) AS 'Avg Customer Age (Days)'
FROM (
    SELECT
        c.customer_id,
        COUNT(DISTINCT o.order_id) AS order_count,
        SUM(o.total_amount) AS customer_value,
        MIN(o.order_date) AS first_order_date
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
      AND o.order_date <= @current_quarter_end
    GROUP BY c.customer_id
) customer_stats;

-- ========================================
-- 4. PRODUCT PERFORMANCE
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'PRODUCT PERFORMANCE' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

-- Top Products by Revenue
SELECT
    p.product_name AS 'Product',
    pc.category_name AS 'Category',
    FORMAT(SUM(oi.quantity), 0) AS 'Units Sold',
    CONCAT('$', FORMAT(SUM(oi.subtotal), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(oi.unit_price), 2)) AS 'Avg Price',
    CONCAT(FORMAT((SUM(oi.subtotal) / (SELECT SUM(oi2.subtotal) 
                   FROM order_items oi2 
                   JOIN orders o2 ON oi2.order_id = o2.order_id
                   WHERE o2.order_date >= @current_quarter_start 
                   AND o2.order_date <= @current_quarter_end
                   AND o2.payment_status = 'paid') * 100), 1), '%') AS '% of Revenue'
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN products p ON oi.product_id = p.product_id
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE o.order_date >= @current_quarter_start 
  AND o.order_date <= @current_quarter_end
  AND o.payment_status = 'paid'
  AND o.status NOT IN ('cancelled')
GROUP BY p.product_id, p.product_name, pc.category_name
ORDER BY SUM(oi.subtotal) DESC
LIMIT 10;

-- Category Performance
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'CATEGORY PERFORMANCE' AS 'Sub-Section';

SELECT
    COALESCE(pc.category_name, 'Uncategorized') AS 'Category',
    COUNT(DISTINCT p.product_id) AS 'Products',
    FORMAT(SUM(oi.quantity), 0) AS 'Units Sold',
    CONCAT('$', FORMAT(SUM(oi.subtotal), 2)) AS 'Total Revenue',
    CONCAT('$', FORMAT(AVG(oi.subtotal), 2)) AS 'Avg Transaction Value',
    CONCAT(FORMAT((SUM(oi.subtotal) / (SELECT SUM(oi2.subtotal) 
                   FROM order_items oi2 
                   JOIN orders o2 ON oi2.order_id = o2.order_id
                   WHERE o2.order_date >= @current_quarter_start 
                   AND o2.order_date <= @current_quarter_end
                   AND o2.payment_status = 'paid') * 100), 1), '%') AS '% of Revenue'
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN products p ON oi.product_id = p.product_id
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
WHERE o.order_date >= @current_quarter_start 
  AND o.order_date <= @current_quarter_end
  AND o.payment_status = 'paid'
  AND o.status NOT IN ('cancelled')
GROUP BY pc.category_name
ORDER BY SUM(oi.subtotal) DESC;

-- ========================================
-- 5. OPERATIONAL EFFICIENCY
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OPERATIONAL EFFICIENCY' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH operational_metrics AS (
    SELECT
        COUNT(*) AS total_orders,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered_orders,
        SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
        SUM(CASE WHEN payment_status = 'failed' THEN 1 ELSE 0 END) AS failed_payments,
        AVG(DATEDIFF(updated_at, order_date)) AS avg_fulfillment_days,
        SUM(shipping_cost) AS total_shipping_cost,
        AVG(shipping_cost) AS avg_shipping_cost
    FROM orders
    WHERE order_date >= @current_quarter_start 
      AND order_date <= @current_quarter_end
)
SELECT
    FORMAT(total_orders, 0) AS 'Total Orders',
    FORMAT(delivered_orders, 0) AS 'Delivered Orders',
    CONCAT(FORMAT((delivered_orders / total_orders * 100), 1), '%') AS 'Delivery Rate',
    FORMAT(cancelled_orders, 0) AS 'Cancelled Orders',
    CONCAT(FORMAT((cancelled_orders / total_orders * 100), 1), '%') AS 'Cancellation Rate',
    FORMAT(failed_payments, 0) AS 'Failed Payments',
    CONCAT(FORMAT((failed_payments / total_orders * 100), 1), '%') AS 'Payment Failure Rate',
    CONCAT(FORMAT(avg_fulfillment_days, 1), ' days') AS 'Avg Fulfillment Time',
    CONCAT('$', FORMAT(total_shipping_cost, 2)) AS 'Total Shipping Cost',
    CONCAT('$', FORMAT(avg_shipping_cost, 2)) AS 'Avg Shipping Cost'
FROM operational_metrics;

-- Returns & Refunds Analysis
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'RETURNS & REFUNDS ANALYSIS' AS 'Sub-Section';

SELECT
    FORMAT(COUNT(*), 0) AS 'Total Returns',
    FORMAT(SUM(CASE WHEN status = 'refunded' THEN 1 ELSE 0 END), 0) AS 'Refunded',
    CONCAT('$', FORMAT(SUM(refund_amount), 2)) AS 'Total Refund Amount',
    CONCAT('$', FORMAT(AVG(refund_amount), 2)) AS 'Avg Refund Amount',
    CONCAT(FORMAT((COUNT(*) / (SELECT COUNT(*) FROM orders 
                                WHERE order_date >= @current_quarter_start 
                                AND order_date <= @current_quarter_end) * 100), 1), '%') AS 'Return Rate',
    reason AS 'Top Return Reason',
    COUNT(*) AS 'Occurrences'
FROM returns
WHERE created_at >= @current_quarter_start 
  AND created_at <= @current_quarter_end
GROUP BY reason
ORDER BY COUNT(*) DESC
LIMIT 1;

-- ========================================
-- 6. MARKETING CAMPAIGN PERFORMANCE
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MARKETING CAMPAIGN PERFORMANCE' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH campaign_summary AS (
    SELECT
        c.campaign_name,
        c.campaign_type,
        c.budget,
        SUM(cp.impressions) AS total_impressions,
        SUM(cp.clicks) AS total_clicks,
        SUM(cp.conversions) AS total_conversions,
        SUM(cp.spend) AS total_spend,
        SUM(cp.revenue) AS total_revenue
    FROM campaigns c
    LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
    WHERE c.start_date >= @current_quarter_start
      AND c.start_date <= @current_quarter_end
    GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.budget
)
SELECT
    campaign_name AS 'Campaign',
    campaign_type AS 'Type',
    CONCAT('$', FORMAT(budget, 2)) AS 'Budget',
    CONCAT('$', FORMAT(total_spend, 2)) AS 'Actual Spend',
    CONCAT(FORMAT((total_spend / budget * 100), 1), '%') AS 'Budget Utilization',
    FORMAT(total_impressions, 0) AS 'Impressions',
    FORMAT(total_clicks, 0) AS 'Clicks',
    CONCAT(FORMAT((total_clicks / total_impressions * 100), 2), '%') AS 'CTR',
    FORMAT(total_conversions, 0) AS 'Conversions',
    CONCAT(FORMAT((total_conversions / total_clicks * 100), 2), '%') AS 'Conversion Rate',
    CONCAT('$', FORMAT(total_revenue, 2)) AS 'Revenue',
    CONCAT(FORMAT(((total_revenue - total_spend) / total_spend * 100), 1), '%') AS 'ROI'
FROM campaign_summary
ORDER BY total_revenue DESC
LIMIT 10;

-- Marketing ROI Summary
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'MARKETING ROI SUMMARY' AS 'Sub-Section';

SELECT
    CONCAT('$', FORMAT(SUM(cp.spend), 2)) AS 'Total Marketing Spend',
    CONCAT('$', FORMAT(SUM(cp.revenue), 2)) AS 'Total Revenue Generated',
    CONCAT('$', FORMAT(SUM(cp.revenue) - SUM(cp.spend), 2)) AS 'Net Profit',
    CONCAT(FORMAT(((SUM(cp.revenue) - SUM(cp.spend)) / SUM(cp.spend) * 100), 1), '%') AS 'Overall ROI',
    CONCAT('$', FORMAT(SUM(cp.spend) / SUM(cp.conversions), 2)) AS 'Cost Per Acquisition',
    FORMAT(AVG((cp.clicks / cp.impressions * 100)), 2) AS 'Avg CTR (%)',
    FORMAT(AVG((cp.conversions / cp.clicks * 100)), 2) AS 'Avg Conversion Rate (%)'
FROM campaign_performance cp
JOIN campaigns c ON cp.campaign_id = c.campaign_id
WHERE c.start_date >= @current_quarter_start
  AND c.start_date <= @current_quarter_end;

-- ========================================
-- 7. INVENTORY & SUPPLY CHAIN
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'INVENTORY & SUPPLY CHAIN' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    FORMAT(COUNT(DISTINCT p.product_id), 0) AS 'Total SKUs',
    FORMAT(SUM(i.quantity_on_hand), 0) AS 'Total Units in Stock',
    CONCAT('$', FORMAT(SUM(i.quantity_on_hand * p.cost), 2)) AS 'Total Inventory Value',
    FORMAT(SUM(CASE WHEN i.quantity_on_hand = 0 THEN 1 ELSE 0 END), 0) AS 'Out of Stock Items',
    CONCAT(FORMAT((SUM(CASE WHEN i.quantity_on_hand = 0 THEN 1 ELSE 0 END) / COUNT(*) * 100), 1), '%') AS 'Stockout Rate',
    FORMAT(SUM(CASE WHEN i.quantity_on_hand < i.reorder_level THEN 1 ELSE 0 END), 0) AS 'Below Reorder Level',
    FORMAT(SUM(CASE WHEN i.quantity_reserved > i.quantity_on_hand THEN 1 ELSE 0 END), 0) AS 'Overbooked Items',
    CONCAT('$', FORMAT(AVG(p.price), 2)) AS 'Avg Product Price',
    CONCAT('$', FORMAT(AVG(p.cost), 2)) AS 'Avg Product Cost',
    CONCAT(FORMAT(AVG((p.price - p.cost) / p.price * 100), 1), '%') AS 'Avg Margin'
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
WHERE p.status = 'active';

-- Inventory Turnover
SELECT '──────────────────────────────────────────────────────────' AS Separator;
SELECT 'INVENTORY TURNOVER ANALYSIS' AS 'Sub-Section';

WITH inventory_turnover AS (
    SELECT
        pc.category_name,
        SUM(oi.quantity) AS units_sold,
        AVG(i.quantity_on_hand) AS avg_inventory,
        SUM(oi.quantity) / NULLIF(AVG(i.quantity_on_hand), 0) AS turnover_ratio,
        90 / NULLIF(SUM(oi.quantity) / NULLIF(AVG(i.quantity_on_hand), 0), 0) AS days_to_sell
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_categories pc ON p.category_id = pc.category_id
    LEFT JOIN inventory i ON p.product_id = i.product_id
    WHERE o.order_date >= @current_quarter_start
      AND o.order_date <= @current_quarter_end
      AND o.payment_status = 'paid'
    GROUP BY pc.category_name
)
SELECT
    COALESCE(category_name, 'Uncategorized') AS 'Category',
    FORMAT(units_sold, 0) AS 'Units Sold',
    FORMAT(avg_inventory, 0) AS 'Avg Inventory',
    FORMAT(turnover_ratio, 2) AS 'Turnover Ratio',
    CONCAT(FORMAT(days_to_sell, 0), ' days') AS 'Days to Sell',
    CASE
        WHEN turnover_ratio >= 4 THEN ''SUCCESS' Excellent'
        WHEN turnover_ratio >= 2 THEN '✓ Good'
        WHEN turnover_ratio >= 1 THEN ''WARNING' Fair'
        ELSE ''ERROR' Poor'
    END AS 'Performance'
FROM inventory_turnover
ORDER BY turnover_ratio DESC;

-- ========================================
-- 8. VENDOR PERFORMANCE
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'VENDOR PERFORMANCE' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    v.vendor_name AS 'Vendor',
    FORMAT(COUNT(DISTINCT vc.product_id), 0) AS 'Products Supplied',
    CONCAT('$', FORMAT(AVG(vc.cost_per_unit), 2)) AS 'Avg Cost Per Unit',
    FORMAT(AVG(vc.lead_time_days), 0) AS 'Avg Lead Time (Days)',
    FORMAT(v.rating, 2) AS 'Vendor Rating',
    v.status AS 'Status',
    CASE
        WHEN v.rating >= 4.5 THEN '⭐ Excellent'
        WHEN v.rating >= 4.0 THEN '✓ Good'
        WHEN v.rating >= 3.0 THEN ''WARNING' Fair'
        ELSE ''ERROR' Poor'
    END AS 'Performance'
FROM vendors v
LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
WHERE v.status = 'active'
  AND (vc.status = 'active' OR vc.status IS NULL)
GROUP BY v.vendor_id, v.vendor_name, v.rating, v.status
ORDER BY v.rating DESC
LIMIT 15;

-- ========================================
-- 9. DATA QUALITY SCORECARD
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'DATA QUALITY SCORECARD' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH data_quality AS (
    SELECT
        'Customer Data' AS category,
        COUNT(*) AS total_records,
        SUM(CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END) AS missing_critical_fields,
        ROUND(100 * (1 - SUM(CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
    FROM customers
    
    UNION ALL
    
    SELECT
        'Product Data',
        COUNT(*),
        SUM(CASE WHEN (sku IS NULL OR price IS NULL OR product_name IS NULL) THEN 1 ELSE 0 END),
        ROUND(100 * (1 - SUM(CASE WHEN (sku IS NULL OR price IS NULL OR product_name IS NULL) THEN 1 ELSE 0 END) / COUNT(*)), 1)
    FROM products
    
    UNION ALL
    
    SELECT
        'Order Data',
        COUNT(*),
        SUM(CASE WHEN (customer_id IS NULL OR total_amount IS NULL) THEN 1 ELSE 0 END),
        ROUND(100 * (1 - SUM(CASE WHEN (customer_id IS NULL OR total_amount IS NULL) THEN 1 ELSE 0 END) / COUNT(*)), 1)
    FROM orders
    WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
    
    UNION ALL
    
    SELECT
        'Inventory Data',
        COUNT(*),
        SUM(CASE WHEN (quantity_on_hand < 0 OR quantity_reserved < 0) THEN 1 ELSE 0 END),
        ROUND(100 * (1 - SUM(CASE WHEN (quantity_on_hand < 0 OR quantity_reserved < 0) THEN 1 ELSE 0 END) / COUNT(*)), 1)
    FROM inventory
)
SELECT
    category AS 'Data Category',
    FORMAT(total_records, 0) AS 'Total Records',
    FORMAT(missing_critical_fields, 0) AS 'Quality Issues',
    CONCAT(quality_score, '%') AS 'Quality Score',
    CASE
        WHEN quality_score >= 95 THEN 'A+ 'SUCCESS''
        WHEN quality_score >= 90 THEN 'A 'SUCCESS''
        WHEN quality_score >= 85 THEN 'B+ ✓'
        WHEN quality_score >= 80 THEN 'B ✓'
        WHEN quality_score >= 75 THEN 'C+ 'WARNING''
        WHEN quality_score >= 70 THEN 'C 'WARNING''
        ELSE 'F 'ERROR''
    END AS 'Grade'
FROM data_quality
ORDER BY quality_score DESC;

-- ========================================
-- 10. QUARTERLY GOALS & TARGETS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'QUARTERLY GOALS & TARGET ACHIEVEMENT' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH goals AS (
    SELECT
        'Revenue Target' AS goal,
        5000000 AS target_value,
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
         AND payment_status = 'paid' AND status NOT IN ('cancelled')) AS actual_value,
        'currency' AS value_type
    
    UNION ALL
    
    SELECT
        'New Customer Acquisition',
        1000,
        (SELECT COUNT(*) FROM customers 
         WHERE created_at >= @current_quarter_start AND created_at <= @current_quarter_end),
        'count'
    
    UNION ALL
    
    SELECT
        'Average Order Value',
        150,
        (SELECT AVG(total_amount) FROM orders 
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
         AND payment_status = 'paid' AND status NOT IN ('cancelled')),
        'currency'
    
    UNION ALL
    
    SELECT
        'Customer Retention Rate',
        75,
        (SELECT COUNT(DISTINCT o.customer_id) * 100.0 / 
                (SELECT COUNT(DISTINCT customer_id) FROM customers WHERE created_at < @current_quarter_start)
         FROM orders o
         WHERE o.customer_id IN (SELECT customer_id FROM customers WHERE created_at < @current_quarter_start)
         AND o.order_date >= @current_quarter_start AND o.order_date <= @current_quarter_end),
        'percentage'
    
    UNION ALL
    
    SELECT
        'Conversion Rate',
        3.5,
        (SELECT COUNT(DISTINCT customer_id) * 100.0 / 
                (SELECT COUNT(*) FROM (SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3) AS dummy_visitors LIMIT 10000)
         FROM orders 
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
         AND payment_status = 'paid'),
        'percentage'
    
    UNION ALL
    
    SELECT
        'Marketing ROI',
        250,
        (SELECT (SUM(revenue) - SUM(spend)) * 100.0 / NULLIF(SUM(spend), 0)
         FROM campaign_performance cp
         JOIN campaigns c ON cp.campaign_id = c.campaign_id
         WHERE c.start_date >= @current_quarter_start AND c.start_date <= @current_quarter_end),
        'percentage'
    
    UNION ALL
    
    SELECT
        'Inventory Turnover Ratio',
        3.0,
        (SELECT SUM(oi.quantity) / NULLIF(AVG(i.quantity_on_hand), 0)
         FROM order_items oi
         JOIN orders o ON oi.order_id = o.order_id
         JOIN inventory i ON oi.product_id = i.product_id
         WHERE o.order_date >= @current_quarter_start AND o.order_date <= @current_quarter_end
         AND o.payment_status = 'paid'),
        'ratio'
    
    UNION ALL
    
    SELECT
        'Order Fulfillment Rate',
        95,
        (SELECT COUNT(CASE WHEN status = 'delivered' THEN 1 END) * 100.0 / COUNT(*)
         FROM orders
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end),
        'percentage'
    
    UNION ALL
    
    SELECT
        'Data Quality Score',
        90,
        (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) AS scores),
        'percentage'
)
SELECT
    goal AS 'Goal / Metric',
    CASE 
        WHEN value_type = 'currency' THEN CONCAT(', FORMAT(target_value, 2))
        WHEN value_type = 'percentage' THEN CONCAT(FORMAT(target_value, 1), '%')
        WHEN value_type = 'ratio' THEN FORMAT(target_value, 2)
        ELSE FORMAT(target_value, 0)
    END AS 'Target',
    CASE 
        WHEN value_type = 'currency' THEN CONCAT(', FORMAT(actual_value, 2))
        WHEN value_type = 'percentage' THEN CONCAT(FORMAT(actual_value, 1), '%')
        WHEN value_type = 'ratio' THEN FORMAT(actual_value, 2)
        ELSE FORMAT(actual_value, 0)
    END AS 'Actual',
    CONCAT(FORMAT((actual_value / target_value * 100), 1), '%') AS 'Achievement',
    CASE
        WHEN actual_value >= target_value THEN ''SUCCESS' Achieved'
        WHEN actual_value >= target_value * 0.9 THEN '↗️ Nearly There'
        WHEN actual_value >= target_value * 0.75 THEN ''WARNING' Below Target'
        ELSE ''ERROR' Missed'
    END AS 'Status',
    CASE
        WHEN actual_value >= target_value * 1.2 THEN '🏆 Exceptional'
        WHEN actual_value >= target_value * 1.1 THEN '⭐ Outstanding'
        WHEN actual_value >= target_value THEN '✓ Good'
        WHEN actual_value >= target_value * 0.9 THEN '→ Fair'
        ELSE '↓ Needs Improvement'
    END AS 'Performance'
FROM goals
ORDER BY (actual_value / target_value) DESC;

-- ========================================
-- 11. STRATEGIC INITIATIVES PROGRESS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'STRATEGIC INITIATIVES & PROJECTS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    'Customer Experience Enhancement' AS 'Initiative',
    'Improve customer satisfaction and reduce churn' AS 'Objective',
    'In Progress' AS 'Status',
    '75%' AS 'Completion',
    CONCAT(
        'Retention: ', 
        (SELECT CONCAT(FORMAT(COUNT(DISTINCT o.customer_id) * 100.0 / 
                (SELECT COUNT(DISTINCT customer_id) FROM customers WHERE created_at < @current_quarter_start), 1), '%')
         FROM orders o
         WHERE o.customer_id IN (SELECT customer_id FROM customers WHERE created_at < @current_quarter_start)
         AND o.order_date >= @current_quarter_start AND o.order_date <= @current_quarter_end)
    ) AS 'Key Metric'

UNION ALL

SELECT
    'Digital Marketing Optimization',
    'Increase ROI and reduce customer acquisition cost',
    'In Progress',
    '60%',
    CONCAT('ROI: ',
        (SELECT CONCAT(FORMAT(((SUM(revenue) - SUM(spend)) / NULLIF(SUM(spend), 0) * 100), 1), '%')
         FROM campaign_performance cp
         JOIN campaigns c ON cp.campaign_id = c.campaign_id
         WHERE c.start_date >= @current_quarter_start AND c.start_date <= @current_quarter_end)
    )

UNION ALL

SELECT
    'Inventory Optimization',
    'Reduce stockouts and improve turnover',
    'On Track',
    '80%',
    CONCAT('Stockout Rate: ',
        (SELECT CONCAT(FORMAT((SUM(CASE WHEN quantity_on_hand = 0 THEN 1 ELSE 0 END) / COUNT(*) * 100), 1), '%')
         FROM inventory)
    )

UNION ALL

SELECT
    'Data Quality Improvement',
    'Achieve 95% data quality across all systems',
    'In Progress',
    '85%',
    CONCAT('Overall Score: ',
        (SELECT CONCAT(FORMAT(AVG(quality_score), 1), '%') FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) AS scores)
    )

UNION ALL

SELECT
    'Product Portfolio Expansion',
    'Launch new product categories and SKUs',
    'Completed',
    '100%',
    CONCAT('New Products: ',
        (SELECT FORMAT(COUNT(*), 0)
         FROM products
         WHERE created_at >= @current_quarter_start AND created_at <= @current_quarter_end)
    );

-- ========================================
-- 12. RISK & OPPORTUNITIES
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'RISKS & OPPORTUNITIES' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    ''WARNING' High Cancellation Rate' AS 'Risk/Opportunity',
    'Risk' AS 'Type',
    'Medium' AS 'Priority',
    CONCAT(
        (SELECT FORMAT((SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*) * 100), 1)
         FROM orders
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end),
        '% cancellation rate'
    ) AS 'Impact',
    'Review checkout process and payment options' AS 'Action Required'

UNION ALL

SELECT
    ''TRENDING_UP' Growing Product Categories',
    'Opportunity',
    'High',
    CONCAT(
        (SELECT COUNT(*) FROM product_categories pc
         WHERE (SELECT SUM(oi.subtotal) 
                FROM order_items oi 
                JOIN products p ON oi.product_id = p.product_id
                JOIN orders o ON oi.order_id = o.order_id
                WHERE p.category_id = pc.category_id 
                AND o.order_date >= @current_quarter_start 
                AND o.order_date <= @current_quarter_end) > 
               (SELECT AVG(cat_revenue) FROM (
                   SELECT SUM(oi2.subtotal) AS cat_revenue
                   FROM order_items oi2
                   JOIN products p2 ON oi2.product_id = p2.product_id
                   JOIN orders o2 ON oi2.order_id = o2.order_id
                   WHERE o2.order_date >= @current_quarter_start 
                   AND o2.order_date <= @current_quarter_end
                   GROUP BY p2.category_id
               ) AS avg_calc)),
        ' high-performing categories'
    ),
    'Increase inventory and marketing for top categories'

UNION ALL

SELECT
    ''WARNING' Inventory Overstock',
    'Risk',
    'Medium',
    CONCAT(
        (SELECT FORMAT(SUM(quantity_on_hand * p.cost), 2)
         FROM inventory i
         JOIN products p ON i.product_id = p.product_id
         WHERE i.quantity_on_hand > i.reorder_level * 3),
        ' in excess inventory'
    ),
    'Implement clearance promotions for slow-moving items'

UNION ALL

SELECT
    ''CHART' Data Quality Improvements',
    'Opportunity',
    'High',
    CONCAT(
        (SELECT CONCAT(FORMAT(100 - AVG(quality_score), 1), '%')
         FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
         ) AS scores),
        ' potential improvement'
    ),
    'Implement automated data validation and cleanup'

UNION ALL

SELECT
    ''MONEY' Upselling Opportunities',
    'Opportunity',
    'Medium',
    CONCAT(',
        (SELECT FORMAT(AVG(total_amount) * 0.15, 2)
         FROM orders
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
         AND payment_status = 'paid'),
        ' potential per order'
    ),
    'Develop cross-sell and upsell recommendations';

-- ========================================
-- 13. CUSTOMER SEGMENTATION INSIGHTS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CUSTOMER SEGMENTATION INSIGHTS' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

WITH customer_segments AS (
    SELECT
        c.customer_id,
        COUNT(DISTINCT o.order_id) AS order_count,
        SUM(o.total_amount) AS total_spent,
        MAX(o.order_date) AS last_order_date,
        DATEDIFF(@current_quarter_end, MAX(o.order_date)) AS days_since_last_order
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.payment_status = 'paid'
      AND o.status NOT IN ('cancelled')
      AND o.order_date <= @current_quarter_end
    GROUP BY c.customer_id
),
rfm_segments AS (
    SELECT
        CASE
            WHEN days_since_last_order <= 30 AND order_count >= 5 AND total_spent >= 1000 THEN 'VIP Champions'
            WHEN days_since_last_order <= 60 AND order_count >= 3 AND total_spent >= 500 THEN 'Loyal Customers'
            WHEN days_since_last_order <= 90 AND order_count >= 2 THEN 'Potential Loyalists'
            WHEN days_since_last_order <= 30 AND order_count = 1 THEN 'New Customers'
            WHEN days_since_last_order > 180 AND order_count >= 3 THEN 'At Risk'
            WHEN days_since_last_order > 365 THEN 'Churned'
            ELSE 'Occasional Buyers'
        END AS segment,
        COUNT(*) AS customer_count,
        AVG(total_spent) AS avg_spent,
        AVG(order_count) AS avg_orders,
        AVG(days_since_last_order) AS avg_days_since_last_order
    FROM customer_segments
    GROUP BY segment
)
SELECT
    segment AS 'Customer Segment',
    FORMAT(customer_count, 0) AS 'Customers',
    CONCAT(FORMAT((customer_count * 100.0 / (SELECT SUM(customer_count) FROM rfm_segments)), 1), '%') AS '% of Base',
    CONCAT(', FORMAT(avg_spent, 2)) AS 'Avg Lifetime Value',
    FORMAT(avg_orders, 1) AS 'Avg Orders',
    CONCAT(FORMAT(avg_days_since_last_order, 0), ' days') AS 'Avg Days Since Last Order',
    CASE
        WHEN segment = 'VIP Champions' THEN '🏆 High Priority - Retain & Reward'
        WHEN segment = 'Loyal Customers' THEN '⭐ High Priority - Nurture & Upsell'
        WHEN segment = 'Potential Loyalists' THEN '✓ Medium Priority - Convert to Loyal'
        WHEN segment = 'New Customers' THEN '🆕 Medium Priority - Engage & Convert'
        WHEN segment = 'At Risk' THEN ''WARNING' High Priority - Re-engage Campaign'
        WHEN segment = 'Churned' THEN ''ERROR' Low Priority - Win-back if High Value'
        ELSE '→ Medium Priority - Regular Engagement'
    END AS 'Strategy'
FROM rfm_segments
ORDER BY avg_spent DESC;

-- ========================================
-- 14. EXECUTIVE SUMMARY
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'EXECUTIVE SUMMARY' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    CONCAT('Q', QUARTER(CURDATE()), ' ', YEAR(CURDATE())) AS 'Quarter',
    
    CONCAT(', FORMAT(
        (SELECT SUM(total_amount) FROM orders 
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
         AND payment_status = 'paid' AND status NOT IN ('cancelled')), 2)
    ) AS 'Total Revenue',
    
    CONCAT(FORMAT(
        ((SELECT SUM(total_amount) FROM orders 
          WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
          AND payment_status = 'paid') - 
         (SELECT SUM(total_amount) FROM orders 
          WHERE order_date >= @previous_quarter_start AND order_date <= @previous_quarter_end
          AND payment_status = 'paid')) * 100.0 /
         (SELECT SUM(total_amount) FROM orders 
          WHERE order_date >= @previous_quarter_start AND order_date <= @previous_quarter_end
          AND payment_status = 'paid'), 1), '%')
    ) AS 'QoQ Growth',
    
    FORMAT(
        (SELECT COUNT(*) FROM customers 
         WHERE created_at >= @current_quarter_start AND created_at <= @current_quarter_end), 0
    ) AS 'New Customers',
    
    CONCAT(', FORMAT(
        (SELECT AVG(total_amount) FROM orders 
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end
         AND payment_status = 'paid'), 2)
    ) AS 'Avg Order Value',
    
    FORMAT(
        (SELECT COUNT(*) FROM orders 
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end), 0
    ) AS 'Total Orders',
    
    CONCAT(FORMAT(
        (SELECT COUNT(CASE WHEN status = 'delivered' THEN 1 END) * 100.0 / COUNT(*)
         FROM orders
         WHERE order_date >= @current_quarter_start AND order_date <= @current_quarter_end), 1), '%'
    ) AS 'Fulfillment Rate',
    
    CONCAT(FORMAT(
        (SELECT AVG(quality_score) FROM (
            SELECT ROUND(100 * (1 - SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1) AS quality_score
            FROM customers
            UNION ALL
            SELECT ROUND(100 * (1 - SUM(CASE WHEN sku IS NULL OR price IS NULL THEN 1 ELSE 0 END) / COUNT(*)), 1)
            FROM products
        ) AS scores), 1), '%'
    ) AS 'Data Quality Score';

-- ========================================
-- 15. KEY RECOMMENDATIONS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'KEY RECOMMENDATIONS FOR NEXT QUARTER' AS 'Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    1 AS 'Priority',
    'Revenue Growth' AS 'Focus Area',
    'Launch targeted campaigns for high-performing product categories' AS 'Recommendation',
    'Marketing & Sales' AS 'Department',
    'High' AS 'Impact'

UNION ALL

SELECT
    2,
    'Customer Retention',
    'Implement loyalty program enhancements and re-engagement campaigns for at-risk customers',
    'Customer Success',
    'High'

UNION ALL

SELECT
    3,
    'Operational Efficiency',
    'Optimize inventory management to reduce stockouts and overstock situations',
    'Operations',
    'Medium'

UNION ALL

SELECT
    4,
    'Data Quality',
    'Deploy automated data validation rules and conduct data cleanup initiatives',
    'IT & Data',
    'Medium'

UNION ALL

SELECT
    5,
    'Marketing ROI',
    'Reallocate budget from underperforming campaigns to high-ROI channels',
    'Marketing',
    'High'

UNION ALL

SELECT
    6,
    'Product Portfolio',
    'Expand inventory for top-performing categories and phase out slow-moving SKUs',
    'Product & Merchandising',
    'Medium'

UNION ALL

SELECT
    7,
    'Customer Experience',
    'Reduce order cancellation rate by improving checkout process and payment options',
    'Customer Experience',
    'Medium'

UNION ALL

SELECT
    8,
    'Supply Chain',
    'Strengthen relationships with top-performing vendors and negotiate better terms',
    'Procurement',
    'Low'

ORDER BY Priority;

-- ========================================
-- REPORT FOOTER
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT CONCAT('Report Generated: ', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')) AS 'Timestamp';
SELECT 'End of Quarterly Business Review' AS 'Report Status';
SELECT '══════════════════════════════════════════════════════════' AS Separator;