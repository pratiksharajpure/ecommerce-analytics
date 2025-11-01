-- ========================================
-- MONTHLY QUALITY SCORECARD
-- Comprehensive monthly data quality assessment
-- Includes MoM comparison and performance grades
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CUSTOMER DATA QUALITY - CURRENT MONTH
-- ========================================
WITH customer_metrics_current AS (
    SELECT 
        DATE_FORMAT(CURRENT_DATE, '%Y-%m') AS report_month,
        COUNT(*) AS total_customers,
        COUNT(DISTINCT email) AS unique_emails,
        COUNT(*) - COUNT(email) AS missing_emails,
        SUM(CASE WHEN email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$' THEN 1 ELSE 0 END) AS invalid_emails,
        COUNT(*) - COUNT(phone) AS missing_phones,
        COUNT(*) - COUNT(address_line1) AS missing_addresses,
        COUNT(*) - COUNT(city) AS missing_cities,
        COUNT(*) - COUNT(state) AS missing_states,
        COUNT(*) - COUNT(zip_code) AS missing_zip_codes,
        COUNT(*) - COUNT(DISTINCT CONCAT(LOWER(TRIM(first_name)), '|', LOWER(TRIM(last_name)), '|', LOWER(TRIM(email)))) AS potential_duplicates
    FROM customers
    WHERE created_at >= DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

customer_metrics_previous AS (
    SELECT 
        DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m') AS report_month,
        COUNT(*) AS total_customers,
        COUNT(DISTINCT email) AS unique_emails,
        COUNT(*) - COUNT(email) AS missing_emails,
        SUM(CASE WHEN email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$' THEN 1 ELSE 0 END) AS invalid_emails,
        COUNT(*) - COUNT(phone) AS missing_phones,
        COUNT(*) - COUNT(address_line1) AS missing_addresses,
        COUNT(*) - COUNT(city) AS missing_cities,
        COUNT(*) - COUNT(state) AS missing_states,
        COUNT(*) - COUNT(zip_code) AS missing_zip_codes,
        COUNT(*) - COUNT(DISTINCT CONCAT(LOWER(TRIM(first_name)), '|', LOWER(TRIM(last_name)), '|', LOWER(TRIM(email)))) AS potential_duplicates
    FROM customers
    WHERE created_at >= DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m-01')
      AND created_at < DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

-- ========================================
-- 2. PRODUCT DATA QUALITY METRICS
-- ========================================
product_metrics_current AS (
    SELECT 
        DATE_FORMAT(CURRENT_DATE, '%Y-%m') AS report_month,
        COUNT(*) AS total_products,
        COUNT(*) - COUNT(sku) AS missing_skus,
        COUNT(*) - COUNT(DISTINCT sku) AS duplicate_skus,
        COUNT(*) - COUNT(product_name) AS missing_names,
        COUNT(*) - COUNT(description) AS missing_descriptions,
        COUNT(*) - COUNT(category_id) AS missing_categories,
        SUM(CASE WHEN price IS NULL OR price <= 0 THEN 1 ELSE 0 END) AS invalid_prices,
        SUM(CASE WHEN cost IS NULL OR cost <= 0 THEN 1 ELSE 0 END) AS invalid_costs,
        SUM(CASE WHEN price < cost THEN 1 ELSE 0 END) AS negative_margin_products,
        SUM(CASE WHEN stock_quantity IS NULL OR stock_quantity < 0 THEN 1 ELSE 0 END) AS invalid_stock
    FROM products
    WHERE created_at >= DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

product_metrics_previous AS (
    SELECT 
        DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m') AS report_month,
        COUNT(*) AS total_products,
        COUNT(*) - COUNT(sku) AS missing_skus,
        COUNT(*) - COUNT(DISTINCT sku) AS duplicate_skus,
        COUNT(*) - COUNT(product_name) AS missing_names,
        COUNT(*) - COUNT(description) AS missing_descriptions,
        COUNT(*) - COUNT(category_id) AS missing_categories,
        SUM(CASE WHEN price IS NULL OR price <= 0 THEN 1 ELSE 0 END) AS invalid_prices,
        SUM(CASE WHEN cost IS NULL OR cost <= 0 THEN 1 ELSE 0 END) AS invalid_costs,
        SUM(CASE WHEN price < cost THEN 1 ELSE 0 END) AS negative_margin_products,
        SUM(CASE WHEN stock_quantity IS NULL OR stock_quantity < 0 THEN 1 ELSE 0 END) AS invalid_stock
    FROM products
    WHERE created_at >= DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m-01')
      AND created_at < DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

-- ========================================
-- 3. ORDER DATA QUALITY METRICS
-- ========================================
order_metrics_current AS (
    SELECT 
        DATE_FORMAT(CURRENT_DATE, '%Y-%m') AS report_month,
        COUNT(*) AS total_orders,
        COUNT(*) - COUNT(customer_id) AS orphaned_orders,
        SUM(CASE WHEN total_amount IS NULL OR total_amount < 0 THEN 1 ELSE 0 END) AS invalid_amounts,
        SUM(CASE WHEN order_date > CURRENT_TIMESTAMP THEN 1 ELSE 0 END) AS future_dates,
        SUM(CASE WHEN status IS NULL THEN 1 ELSE 0 END) AS missing_status,
        SUM(CASE WHEN payment_status = 'paid' AND status = 'cancelled' THEN 1 ELSE 0 END) AS payment_status_mismatch
    FROM orders
    WHERE order_date >= DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

order_metrics_previous AS (
    SELECT 
        DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m') AS report_month,
        COUNT(*) AS total_orders,
        COUNT(*) - COUNT(customer_id) AS orphaned_orders,
        SUM(CASE WHEN total_amount IS NULL OR total_amount < 0 THEN 1 ELSE 0 END) AS invalid_amounts,
        SUM(CASE WHEN order_date > CURRENT_TIMESTAMP THEN 1 ELSE 0 END) AS future_dates,
        SUM(CASE WHEN status IS NULL THEN 1 ELSE 0 END) AS missing_status,
        SUM(CASE WHEN payment_status = 'paid' AND status = 'cancelled' THEN 1 ELSE 0 END) AS payment_status_mismatch
    FROM orders
    WHERE order_date >= DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m-01')
      AND order_date < DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

-- ========================================
-- 4. INVENTORY DATA QUALITY METRICS
-- ========================================
inventory_metrics_current AS (
    SELECT 
        DATE_FORMAT(CURRENT_DATE, '%Y-%m') AS report_month,
        COUNT(*) AS total_inventory_records,
        SUM(CASE WHEN quantity_on_hand < 0 THEN 1 ELSE 0 END) AS negative_quantities,
        SUM(CASE WHEN quantity_reserved < 0 THEN 1 ELSE 0 END) AS negative_reserved,
        SUM(CASE WHEN quantity_reserved > quantity_on_hand THEN 1 ELSE 0 END) AS overbooked_inventory,
        SUM(CASE WHEN quantity_on_hand = 0 AND quantity_reserved > 0 THEN 1 ELSE 0 END) AS impossible_reservations
    FROM inventory
    WHERE last_updated >= DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

inventory_metrics_previous AS (
    SELECT 
        DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m') AS report_month,
        COUNT(*) AS total_inventory_records,
        SUM(CASE WHEN quantity_on_hand < 0 THEN 1 ELSE 0 END) AS negative_quantities,
        SUM(CASE WHEN quantity_reserved < 0 THEN 1 ELSE 0 END) AS negative_reserved,
        SUM(CASE WHEN quantity_reserved > quantity_on_hand THEN 1 ELSE 0 END) AS overbooked_inventory,
        SUM(CASE WHEN quantity_on_hand = 0 AND quantity_reserved > 0 THEN 1 ELSE 0 END) AS impossible_reservations
    FROM inventory
    WHERE last_updated >= DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m-01')
      AND last_updated < DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

-- ========================================
-- 5. CAMPAIGN DATA QUALITY METRICS
-- ========================================
campaign_metrics_current AS (
    SELECT 
        DATE_FORMAT(CURRENT_DATE, '%Y-%m') AS report_month,
        COUNT(*) AS total_campaigns,
        COUNT(*) - COUNT(campaign_name) AS missing_names,
        SUM(CASE WHEN start_date IS NULL THEN 1 ELSE 0 END) AS missing_start_dates,
        SUM(CASE WHEN end_date IS NULL THEN 1 ELSE 0 END) AS missing_end_dates,
        SUM(CASE WHEN end_date < start_date THEN 1 ELSE 0 END) AS invalid_date_ranges,
        SUM(CASE WHEN budget IS NULL OR budget <= 0 THEN 1 ELSE 0 END) AS invalid_budgets
    FROM campaigns
    WHERE created_at >= DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

campaign_metrics_previous AS (
    SELECT 
        DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m') AS report_month,
        COUNT(*) AS total_campaigns,
        COUNT(*) - COUNT(campaign_name) AS missing_names,
        SUM(CASE WHEN start_date IS NULL THEN 1 ELSE 0 END) AS missing_start_dates,
        SUM(CASE WHEN end_date IS NULL THEN 1 ELSE 0 END) AS missing_end_dates,
        SUM(CASE WHEN end_date < start_date THEN 1 ELSE 0 END) AS invalid_date_ranges,
        SUM(CASE WHEN budget IS NULL OR budget <= 0 THEN 1 ELSE 0 END) AS invalid_budgets
    FROM campaigns
    WHERE created_at >= DATE_FORMAT(DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH), '%Y-%m-01')
      AND created_at < DATE_FORMAT(CURRENT_DATE, '%Y-%m-01')
),

-- ========================================
-- 6. CALCULATE QUALITY SCORES & GRADES
-- ========================================
quality_scores AS (
    -- CUSTOMER QUALITY SCORE
    SELECT
        'Customer Data Quality' AS metric_category,
        curr.report_month AS current_month,
        prev.report_month AS previous_month,
        ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                         curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) AS current_score,
        ROUND(100 * (1 - (prev.missing_emails + prev.invalid_emails + prev.missing_phones + 
                         prev.missing_addresses + prev.potential_duplicates) / NULLIF(prev.total_customers * 5, 0)), 2) AS previous_score,
        ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                         curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) -
        ROUND(100 * (1 - (prev.missing_emails + prev.invalid_emails + prev.missing_phones + 
                         prev.missing_addresses + prev.potential_duplicates) / NULLIF(prev.total_customers * 5, 0)), 2) AS mom_change,
        CASE 
            WHEN ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                                  curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) >= 95 THEN 'A+'
            WHEN ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                                  curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) >= 90 THEN 'A'
            WHEN ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                                  curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) >= 85 THEN 'B+'
            WHEN ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                                  curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) >= 80 THEN 'B'
            WHEN ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                                  curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) >= 75 THEN 'C+'
            WHEN ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                                  curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) >= 70 THEN 'C'
            WHEN ROUND(100 * (1 - (curr.missing_emails + curr.invalid_emails + curr.missing_phones + 
                                  curr.missing_addresses + curr.potential_duplicates) / NULLIF(curr.total_customers * 5, 0)), 2) >= 60 THEN 'D'
            ELSE 'F'
        END AS current_grade,
        curr.total_customers AS current_record_count,
        prev.total_customers AS previous_record_count
    FROM customer_metrics_current curr
    CROSS JOIN customer_metrics_previous prev
    
    UNION ALL
    
    -- PRODUCT QUALITY SCORE
    SELECT
        'Product Data Quality' AS metric_category,
        curr.report_month,
        prev.report_month,
        ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                         curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                         curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2),
        ROUND(100 * (1 - (prev.missing_skus + prev.duplicate_skus + prev.missing_names + 
                         prev.missing_descriptions + prev.invalid_prices + prev.invalid_costs + 
                         prev.negative_margin_products) / NULLIF(prev.total_products * 7, 0)), 2),
        ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                         curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                         curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2) -
        ROUND(100 * (1 - (prev.missing_skus + prev.duplicate_skus + prev.missing_names + 
                         prev.missing_descriptions + prev.invalid_prices + prev.invalid_costs + 
                         prev.negative_margin_products) / NULLIF(prev.total_products * 7, 0)), 2),
        CASE 
            WHEN ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                                  curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                                  curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2) >= 95 THEN 'A+'
            WHEN ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                                  curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                                  curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2) >= 90 THEN 'A'
            WHEN ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                                  curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                                  curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2) >= 85 THEN 'B+'
            WHEN ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                                  curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                                  curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2) >= 80 THEN 'B'
            WHEN ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                                  curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                                  curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2) >= 75 THEN 'C+'
            WHEN ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                                  curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                                  curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2) >= 70 THEN 'C'
            WHEN ROUND(100 * (1 - (curr.missing_skus + curr.duplicate_skus + curr.missing_names + 
                                  curr.missing_descriptions + curr.invalid_prices + curr.invalid_costs + 
                                  curr.negative_margin_products) / NULLIF(curr.total_products * 7, 0)), 2) >= 60 THEN 'D'
            ELSE 'F'
        END,
        curr.total_products,
        prev.total_products
    FROM product_metrics_current curr
    CROSS JOIN product_metrics_previous prev
    
    UNION ALL
    
    -- ORDER QUALITY SCORE
    SELECT
        'Order Data Quality',
        curr.report_month,
        prev.report_month,
        ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                         curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2),
        ROUND(100 * (1 - (prev.orphaned_orders + prev.invalid_amounts + prev.future_dates + 
                         prev.missing_status + prev.payment_status_mismatch) / NULLIF(prev.total_orders * 5, 0)), 2),
        ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                         curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2) -
        ROUND(100 * (1 - (prev.orphaned_orders + prev.invalid_amounts + prev.future_dates + 
                         prev.missing_status + prev.payment_status_mismatch) / NULLIF(prev.total_orders * 5, 0)), 2),
        CASE 
            WHEN ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                                  curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2) >= 95 THEN 'A+'
            WHEN ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                                  curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2) >= 90 THEN 'A'
            WHEN ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                                  curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2) >= 85 THEN 'B+'
            WHEN ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                                  curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2) >= 80 THEN 'B'
            WHEN ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                                  curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2) >= 75 THEN 'C+'
            WHEN ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                                  curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2) >= 70 THEN 'C'
            WHEN ROUND(100 * (1 - (curr.orphaned_orders + curr.invalid_amounts + curr.future_dates + 
                                  curr.missing_status + curr.payment_status_mismatch) / NULLIF(curr.total_orders * 5, 0)), 2) >= 60 THEN 'D'
            ELSE 'F'
        END,
        curr.total_orders,
        prev.total_orders
    FROM order_metrics_current curr
    CROSS JOIN order_metrics_previous prev
    
    UNION ALL
    
    -- INVENTORY QUALITY SCORE
    SELECT
        'Inventory Data Quality',
        curr.report_month,
        prev.report_month,
        ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                         curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2),
        ROUND(100 * (1 - (prev.negative_quantities + prev.negative_reserved + prev.overbooked_inventory + 
                         prev.impossible_reservations) / NULLIF(prev.total_inventory_records * 4, 0)), 2),
        ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                         curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2) -
        ROUND(100 * (1 - (prev.negative_quantities + prev.negative_reserved + prev.overbooked_inventory + 
                         prev.impossible_reservations) / NULLIF(prev.total_inventory_records * 4, 0)), 2),
        CASE 
            WHEN ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                                  curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2) >= 95 THEN 'A+'
            WHEN ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                                  curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2) >= 90 THEN 'A'
            WHEN ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                                  curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2) >= 85 THEN 'B+'
            WHEN ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                                  curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2) >= 80 THEN 'B'
            WHEN ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                                  curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2) >= 75 THEN 'C+'
            WHEN ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                                  curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2) >= 70 THEN 'C'
            WHEN ROUND(100 * (1 - (curr.negative_quantities + curr.negative_reserved + curr.overbooked_inventory + 
                                  curr.impossible_reservations) / NULLIF(curr.total_inventory_records * 4, 0)), 2) >= 60 THEN 'D'
            ELSE 'F'
        END,
        curr.total_inventory_records,
        prev.total_inventory_records
    FROM inventory_metrics_current curr
    CROSS JOIN inventory_metrics_previous prev
    
    UNION ALL
    
    -- CAMPAIGN QUALITY SCORE
    SELECT
        'Campaign Data Quality',
        curr.report_month,
        prev.report_month,
        ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                         curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2),
        ROUND(100 * (1 - (prev.missing_names + prev.missing_start_dates + prev.missing_end_dates + 
                         prev.invalid_date_ranges + prev.invalid_budgets) / NULLIF(prev.total_campaigns * 5, 0)), 2),
        ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                         curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2) -
        ROUND(100 * (1 - (prev.missing_names + prev.missing_start_dates + prev.missing_end_dates + 
                         prev.invalid_date_ranges + prev.invalid_budgets) / NULLIF(prev.total_campaigns * 5, 0)), 2),
        CASE 
            WHEN ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                                  curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2) >= 95 THEN 'A+'
            WHEN ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                                  curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2) >= 90 THEN 'A'
            WHEN ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                                  curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2) >= 85 THEN 'B+'
            WHEN ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                                  curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2) >= 80 THEN 'B'
            WHEN ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                                  curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2) >= 75 THEN 'C+'
            WHEN ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                                  curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2) >= 70 THEN 'C'
            WHEN ROUND(100 * (1 - (curr.missing_names + curr.missing_start_dates + curr.missing_end_dates + 
                                  curr.invalid_date_ranges + curr.invalid_budgets) / NULLIF(curr.total_campaigns * 5, 0)), 2) >= 60 THEN 'D'
            ELSE 'F'
        END,
        curr.total_campaigns,
        prev.total_campaigns
    FROM campaign_metrics_current curr
    CROSS JOIN campaign_metrics_previous prev
)

-- ========================================
-- MAIN SCORECARD OUTPUT
-- ========================================
SELECT
    metric_category AS 'Category',
    current_month AS 'Current',
    previous_month AS 'Previous',
    CONCAT(current_score, '%') AS 'Current Score',
    CONCAT(previous_score, '%') AS 'Previous Score',
    CONCAT(
        CASE WHEN mom_change > 0 THEN '+' ELSE '' END,
        mom_change, '%'
    ) AS 'MoM Change',
    current_grade AS 'Grade',
    CASE 
        WHEN mom_change > 5 THEN ''TRENDING_UP' Significant Improvement'
        WHEN mom_change > 0 THEN '↗️ Improving'
        WHEN mom_change = 0 THEN '→ Stable'
        WHEN mom_change > -5 THEN '↘️ Declining'
        ELSE ''TRENDING_DOWN' Needs Attention'
    END AS 'Trend',
    current_record_count AS 'Current Records',
    previous_record_count AS 'Previous Records',
    CASE 
        WHEN current_grade IN ('A+', 'A') THEN ''SUCCESS' Excellent'
        WHEN current_grade IN ('B+', 'B') THEN '✓ Good'
        WHEN current_grade IN ('C+', 'C') THEN ''WARNING' Fair'
        WHEN current_grade = 'D' THEN ''WARNING' Poor'
        ELSE ''ERROR' Critical'
    END AS 'Status'
FROM quality_scores
ORDER BY current_score DESC;

-- ========================================
-- OVERALL QUALITY SUMMARY
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'OVERALL QUALITY SUMMARY' AS 'Report Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    CONCAT(ROUND(AVG(current_score), 2), '%') AS 'Overall Quality Score',
    CONCAT(ROUND(AVG(previous_score), 2), '%') AS 'Previous Month Score',
    CONCAT(
        CASE WHEN AVG(mom_change) > 0 THEN '+' ELSE '' END,
        ROUND(AVG(mom_change), 2), '%'
    ) AS 'Average MoM Change',
    CASE 
        WHEN AVG(current_score) >= 95 THEN 'A+'
        WHEN AVG(current_score) >= 90 THEN 'A'
        WHEN AVG(current_score) >= 85 THEN 'B+'
        WHEN AVG(current_score) >= 80 THEN 'B'
        WHEN AVG(current_score) >= 75 THEN 'C+'
        WHEN AVG(current_score) >= 70 THEN 'C'
        WHEN AVG(current_score) >= 60 THEN 'D'
        ELSE 'F'
    END AS 'Overall Grade',
    SUM(current_record_count) AS 'Total Records Analyzed'
FROM quality_scores;

-- ========================================
-- TOP PERFORMERS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'TOP PERFORMING CATEGORIES' AS 'Report Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    metric_category AS 'Category',
    CONCAT(current_score, '%') AS 'Score',
    current_grade AS 'Grade',
    CONCAT(CASE WHEN mom_change > 0 THEN '+' ELSE '' END, mom_change, '%') AS 'MoM Change'
FROM quality_scores
ORDER BY current_score DESC
LIMIT 3;

-- ========================================
-- CATEGORIES NEEDING ATTENTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'CATEGORIES NEEDING ATTENTION' AS 'Report Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    metric_category AS 'Category',
    CONCAT(current_score, '%') AS 'Score',
    current_grade AS 'Grade',
    CONCAT(CASE WHEN mom_change > 0 THEN '+' ELSE '' END, mom_change, '%') AS 'MoM Change',
    CASE 
        WHEN current_grade IN ('D', 'F') THEN '🚨 Urgent Action Required'
        WHEN current_grade IN ('C', 'C+') THEN ''WARNING' Improvement Needed'
        ELSE '✓ Monitor'
    END AS 'Action Required'
FROM quality_scores
WHERE current_score < 80
ORDER BY current_score ASC;

-- ========================================
-- MOST IMPROVED CATEGORIES
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'MOST IMPROVED CATEGORIES' AS 'Report Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    metric_category AS 'Category',
    CONCAT(previous_score, '%') AS 'Previous Score',
    CONCAT(current_score, '%') AS 'Current Score',
    CONCAT('+', mom_change, '%') AS 'Improvement',
    CASE 
        WHEN mom_change > 10 THEN '🏆 Outstanding'
        WHEN mom_change > 5 THEN '⭐ Excellent'
        ELSE '✓ Good'
    END AS 'Recognition'
FROM quality_scores
WHERE mom_change > 0
ORDER BY mom_change DESC
LIMIT 3;

-- ========================================
-- DECLINING CATEGORIES
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'DECLINING CATEGORIES' AS 'Report Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    metric_category AS 'Category',
    CONCAT(previous_score, '%') AS 'Previous Score',
    CONCAT(current_score, '%') AS 'Current Score',
    CONCAT(mom_change, '%') AS 'Decline',
    CASE 
        WHEN mom_change < -10 THEN '🚨 Critical Decline'
        WHEN mom_change < -5 THEN ''WARNING' Significant Decline'
        ELSE ''WARNING' Minor Decline'
    END AS 'Severity'
FROM quality_scores
WHERE mom_change < 0
ORDER BY mom_change ASC;

-- ========================================
-- GRADE DISTRIBUTION
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'GRADE DISTRIBUTION' AS 'Report Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    current_grade AS 'Grade',
    COUNT(*) AS 'Number of Categories',
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM quality_scores), 1), '%') AS 'Percentage',
    CASE 
        WHEN current_grade IN ('A+', 'A') THEN ''SUCCESS' Excellent Performance'
        WHEN current_grade IN ('B+', 'B') THEN '✓ Good Performance'
        WHEN current_grade IN ('C+', 'C') THEN ''WARNING' Needs Improvement'
        ELSE ''ERROR' Critical Issues'
    END AS 'Performance Level'
FROM quality_scores
GROUP BY current_grade
ORDER BY 
    CASE current_grade
        WHEN 'A+' THEN 1
        WHEN 'A' THEN 2
        WHEN 'B+' THEN 3
        WHEN 'B' THEN 4
        WHEN 'C+' THEN 5
        WHEN 'C' THEN 6
        WHEN 'D' THEN 7
        WHEN 'F' THEN 8
    END;

-- ========================================
-- EXECUTIVE SUMMARY
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'EXECUTIVE SUMMARY' AS 'Report Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    DATE_FORMAT(CURRENT_DATE, '%Y-%m') AS 'Reporting Period',
    COUNT(*) AS 'Categories Evaluated',
    SUM(CASE WHEN current_grade IN ('A+', 'A') THEN 1 ELSE 0 END) AS 'Excellent',
    SUM(CASE WHEN current_grade IN ('B+', 'B') THEN 1 ELSE 0 END) AS 'Good',
    SUM(CASE WHEN current_grade IN ('C+', 'C') THEN 1 ELSE 0 END) AS 'Fair',
    SUM(CASE WHEN current_grade IN ('D', 'F') THEN 1 ELSE 0 END) AS 'Poor',
    SUM(CASE WHEN mom_change > 0 THEN 1 ELSE 0 END) AS 'Improving',
    SUM(CASE WHEN mom_change < 0 THEN 1 ELSE 0 END) AS 'Declining',
    SUM(CASE WHEN mom_change = 0 THEN 1 ELSE 0 END) AS 'Stable',
    CONCAT(ROUND(AVG(current_score), 1), '%') AS 'Avg Quality Score'
FROM quality_scores;

-- ========================================
-- KEY RECOMMENDATIONS
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT 'KEY RECOMMENDATIONS' AS 'Report Section';
SELECT '══════════════════════════════════════════════════════════' AS Separator;

SELECT
    ROW_NUMBER() OVER (ORDER BY current_score ASC) AS 'Priority',
    metric_category AS 'Category',
    current_grade AS 'Grade',
    CASE 
        WHEN current_grade = 'F' THEN 'Immediate data quality intervention required'
        WHEN current_grade = 'D' THEN 'Implement data validation rules and cleanup procedures'
        WHEN current_grade IN ('C', 'C+') THEN 'Review data entry processes and training'
        WHEN mom_change < -5 THEN 'Investigate recent changes causing quality decline'
        ELSE 'Continue monitoring and maintain current standards'
    END AS 'Recommended Action'
FROM quality_scores
WHERE current_grade NOT IN ('A+', 'A') OR mom_change < 0
ORDER BY 
    CASE 
        WHEN current_grade = 'F' THEN 1
        WHEN current_grade = 'D' THEN 2
        WHEN mom_change < -5 THEN 3
        ELSE 4
    END,
    current_score ASC;

-- ========================================
-- REPORT TIMESTAMP
-- ========================================
SELECT '══════════════════════════════════════════════════════════' AS Separator;
SELECT CONCAT('Report Generated: ', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')) AS 'Timestamp';
SELECT '══════════════════════════════════════════════════════════' AS Separator;