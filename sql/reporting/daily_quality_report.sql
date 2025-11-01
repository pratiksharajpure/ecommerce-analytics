-- ========================================
-- DAILY DATA QUALITY REPORT
-- E-commerce Revenue Analytics Engine
-- Daily Quality Summary, Issues Detection, Quality Trends
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- REPORT HEADER - EXECUTIVE SUMMARY
-- ========================================

SELECT 
    '═══════════════════════════════════════════════════════════' AS separator,
    'DAILY DATA QUALITY REPORT' AS title,
    DATE_FORMAT(CURDATE(), '%A, %B %d, %Y') AS report_date,
    DATE_FORMAT(NOW(), '%H:%i:%s') AS generated_at,
    '═══════════════════════════════════════════════════════════' AS separator2;

-- ========================================
-- 1. OVERALL QUALITY SCORE
-- Aggregate data quality metrics
-- ========================================

WITH quality_metrics AS (
    SELECT 
        -- Customer data quality
        COUNT(*) AS total_customers,
        SUM(CASE WHEN email IS NULL OR email = '' OR email NOT LIKE '%@%.%' THEN 1 ELSE 0 END) AS invalid_emails,
        SUM(CASE WHEN phone IS NULL OR phone = '' THEN 1 ELSE 0 END) AS missing_phones,
        SUM(CASE WHEN first_name IS NULL OR first_name = '' THEN 1 ELSE 0 END) AS missing_first_names,
        SUM(CASE WHEN last_name IS NULL OR last_name = '' THEN 1 ELSE 0 END) AS missing_last_names,
        SUM(CASE WHEN address_line1 IS NULL OR address_line1 = '' THEN 1 ELSE 0 END) AS missing_addresses,
        SUM(CASE WHEN city IS NULL OR city = '' THEN 1 ELSE 0 END) AS missing_cities,
        SUM(CASE WHEN state IS NULL OR state = '' THEN 1 ELSE 0 END) AS missing_states,
        SUM(CASE WHEN zip_code IS NULL OR zip_code = '' THEN 1 ELSE 0 END) AS missing_zip_codes
    FROM customers
),
product_quality AS (
    SELECT 
        COUNT(*) AS total_products,
        SUM(CASE WHEN product_name IS NULL OR product_name = '' THEN 1 ELSE 0 END) AS missing_names,
        SUM(CASE WHEN description IS NULL OR description = '' THEN 1 ELSE 0 END) AS missing_descriptions,
        SUM(CASE WHEN price IS NULL OR price <= 0 THEN 1 ELSE 0 END) AS invalid_prices,
        SUM(CASE WHEN cost IS NULL OR cost < 0 THEN 1 ELSE 0 END) AS invalid_costs,
        SUM(CASE WHEN category_id IS NULL THEN 1 ELSE 0 END) AS missing_categories,
        SUM(CASE WHEN sku IS NULL OR sku = '' THEN 1 ELSE 0 END) AS missing_skus,
        SUM(CASE WHEN stock_quantity < 0 THEN 1 ELSE 0 END) AS negative_stock
    FROM products
),
order_quality AS (
    SELECT 
        COUNT(*) AS total_orders,
        SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS orphaned_orders,
        SUM(CASE WHEN total_amount IS NULL OR total_amount <= 0 THEN 1 ELSE 0 END) AS invalid_amounts,
        SUM(CASE WHEN order_date IS NULL THEN 1 ELSE 0 END) AS missing_dates,
        SUM(CASE WHEN order_date > NOW() THEN 1 ELSE 0 END) AS future_dates,
        SUM(CASE WHEN status NOT IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled') THEN 1 ELSE 0 END) AS invalid_status
    FROM orders
)
SELECT 
    'DATA QUALITY OVERVIEW' AS section,
    ROUND(100 - (
        (qm.invalid_emails + qm.missing_phones + qm.missing_first_names + qm.missing_last_names + 
         qm.missing_addresses + qm.missing_cities + qm.missing_states + qm.missing_zip_codes) * 100.0 / 
        (qm.total_customers * 8)
    ), 2) AS customer_quality_score,
    ROUND(100 - (
        (pq.missing_names + pq.missing_descriptions + pq.invalid_prices + pq.invalid_costs + 
         pq.missing_categories + pq.missing_skus + pq.negative_stock) * 100.0 / 
        (pq.total_products * 7)
    ), 2) AS product_quality_score,
    ROUND(100 - (
        (oq.orphaned_orders + oq.invalid_amounts + oq.missing_dates + oq.future_dates + oq.invalid_status) * 100.0 / 
        (oq.total_orders * 5)
    ), 2) AS order_quality_score,
    ROUND((
        (100 - (qm.invalid_emails + qm.missing_phones + qm.missing_first_names + qm.missing_last_names + 
         qm.missing_addresses + qm.missing_cities + qm.missing_states + qm.missing_zip_codes) * 100.0 / 
        (qm.total_customers * 8)) +
        (100 - (pq.missing_names + pq.missing_descriptions + pq.invalid_prices + pq.invalid_costs + 
         pq.missing_categories + pq.missing_skus + pq.negative_stock) * 100.0 / 
        (pq.total_products * 7)) +
        (100 - (oq.orphaned_orders + oq.invalid_amounts + oq.missing_dates + oq.future_dates + oq.invalid_status) * 100.0 / 
        (oq.total_orders * 5))
    ) / 3, 2) AS overall_quality_score,
    CASE 
        WHEN ROUND((
            (100 - (qm.invalid_emails + qm.missing_phones + qm.missing_first_names + qm.missing_last_names + 
             qm.missing_addresses + qm.missing_cities + qm.missing_states + qm.missing_zip_codes) * 100.0 / 
            (qm.total_customers * 8)) +
            (100 - (pq.missing_names + pq.missing_descriptions + pq.invalid_prices + pq.invalid_costs + 
             pq.missing_categories + pq.missing_skus + pq.negative_stock) * 100.0 / 
            (pq.total_products * 7)) +
            (100 - (oq.orphaned_orders + oq.invalid_amounts + oq.missing_dates + oq.future_dates + oq.invalid_status) * 100.0 / 
            (oq.total_orders * 5))
        ) / 3, 2) >= 95 THEN ''GREEN' Excellent'
        WHEN ROUND((
            (100 - (qm.invalid_emails + qm.missing_phones + qm.missing_first_names + qm.missing_last_names + 
             qm.missing_addresses + qm.missing_cities + qm.missing_states + qm.missing_zip_codes) * 100.0 / 
            (qm.total_customers * 8)) +
            (100 - (pq.missing_names + pq.missing_descriptions + pq.invalid_prices + pq.invalid_costs + 
             pq.missing_categories + pq.missing_skus + pq.negative_stock) * 100.0 / 
            (pq.total_products * 7)) +
            (100 - (oq.orphaned_orders + oq.invalid_amounts + oq.missing_dates + oq.future_dates + oq.invalid_status) * 100.0 / 
            (oq.total_orders * 5))
        ) / 3, 2) >= 85 THEN ''YELLOW' Good'
        WHEN ROUND((
            (100 - (qm.invalid_emails + qm.missing_phones + qm.missing_first_names + qm.missing_last_names + 
             qm.missing_addresses + qm.missing_cities + qm.missing_states + qm.missing_zip_codes) * 100.0 / 
            (qm.total_customers * 8)) +
            (100 - (pq.missing_names + pq.missing_descriptions + pq.invalid_prices + pq.invalid_costs + 
             pq.missing_categories + pq.missing_skus + pq.negative_stock) * 100.0 / 
            (pq.total_products * 7)) +
            (100 - (oq.orphaned_orders + oq.invalid_amounts + oq.missing_dates + oq.future_dates + oq.invalid_status) * 100.0 / 
            (oq.total_orders * 5))
        ) / 3, 2) >= 70 THEN '🟠 Fair'
        ELSE ''RED' Poor'
    END AS quality_grade
FROM quality_metrics qm, product_quality pq, order_quality oq;

-- ========================================
-- 2. CUSTOMER DATA QUALITY - DETAILED
-- Issues detected today in customer data
-- ========================================

SELECT 
    ''SEARCH' CUSTOMER DATA QUALITY ISSUES' AS section,
    '' AS spacer;

-- Invalid Email Addresses
SELECT 
    'Invalid Email Addresses' AS issue_type,
    COUNT(*) AS issue_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage,
    'High' AS severity,
    ''RED'' AS priority,
    'Validate and update email formats' AS recommended_action
FROM customers
WHERE email IS NULL OR email = '' OR email NOT LIKE '%@%.%'
HAVING COUNT(*) > 0

UNION ALL

-- Missing Phone Numbers
SELECT 
    'Missing Phone Numbers',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2),
    'Medium',
    ''YELLOW'',
    'Request phone numbers for customer records'
FROM customers
WHERE phone IS NULL OR phone = ''
HAVING COUNT(*) > 0

UNION ALL

-- Incomplete Names
SELECT 
    'Missing First or Last Name',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2),
    'Medium',
    ''YELLOW'',
    'Update customer profile information'
FROM customers
WHERE first_name IS NULL OR first_name = '' OR last_name IS NULL OR last_name = ''
HAVING COUNT(*) > 0

UNION ALL

-- Incomplete Address
SELECT 
    'Incomplete Address Information',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2),
    'High',
    ''RED'',
    'Validate shipping addresses before orders'
FROM customers
WHERE address_line1 IS NULL OR address_line1 = '' 
   OR city IS NULL OR city = '' 
   OR state IS NULL OR state = ''
   OR zip_code IS NULL OR zip_code = ''
HAVING COUNT(*) > 0

UNION ALL

-- Duplicate Email Addresses
SELECT 
    'Duplicate Email Addresses',
    COUNT(*) - COUNT(DISTINCT email),
    ROUND((COUNT(*) - COUNT(DISTINCT email)) * 100.0 / COUNT(*), 2),
    'High',
    ''RED'',
    'Merge or deactivate duplicate customer records'
FROM customers
WHERE email IS NOT NULL AND email != ''
HAVING COUNT(*) - COUNT(DISTINCT email) > 0

UNION ALL

-- Inactive Customers with Recent Activity
SELECT 
    'Inactive Customers with Orders',
    COUNT(DISTINCT c.customer_id),
    ROUND(COUNT(DISTINCT c.customer_id) * 100.0 / (SELECT COUNT(*) FROM customers), 2),
    'Low',
    ''GREEN'',
    'Update customer status to active'
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.status = 'inactive'
  AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
HAVING COUNT(DISTINCT c.customer_id) > 0;

-- ========================================
-- 3. PRODUCT DATA QUALITY - DETAILED
-- Issues detected today in product data
-- ========================================

SELECT 
    '📦 PRODUCT DATA QUALITY ISSUES' AS section,
    '' AS spacer;

-- Missing Product Names
SELECT 
    'Missing Product Names' AS issue_type,
    COUNT(*) AS issue_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products), 2) AS percentage,
    'Critical' AS severity,
    ''RED'' AS priority,
    'Add product names immediately' AS recommended_action
FROM products
WHERE product_name IS NULL OR product_name = ''
HAVING COUNT(*) > 0

UNION ALL

-- Missing Descriptions
SELECT 
    'Missing Product Descriptions',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products), 2),
    'Medium',
    ''YELLOW'',
    'Add detailed product descriptions for SEO'
FROM products
WHERE description IS NULL OR description = ''
HAVING COUNT(*) > 0

UNION ALL

-- Invalid Prices
SELECT 
    'Invalid or Zero Prices',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products), 2),
    'Critical',
    ''RED'',
    'Update pricing immediately - products unsellable'
FROM products
WHERE price IS NULL OR price <= 0
HAVING COUNT(*) > 0

UNION ALL

-- Invalid Costs
SELECT 
    'Invalid Cost Data',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products), 2),
    'High',
    ''RED'',
    'Update cost data for margin calculations'
FROM products
WHERE cost IS NULL OR cost < 0
HAVING COUNT(*) > 0

UNION ALL

-- Negative Margins
SELECT 
    'Products Selling Below Cost',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products), 2),
    'High',
    ''RED'',
    'Review pricing strategy - negative margins'
FROM products
WHERE price < cost AND price > 0 AND cost > 0
HAVING COUNT(*) > 0

UNION ALL

-- Missing Categories
SELECT 
    'Missing Product Categories',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products), 2),
    'Medium',
    ''YELLOW'',
    'Assign products to appropriate categories'
FROM products
WHERE category_id IS NULL
HAVING COUNT(*) > 0

UNION ALL

-- Missing SKUs
SELECT 
    'Missing SKU Codes',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products), 2),
    'High',
    ''RED'',
    'Generate unique SKU codes for tracking'
FROM products
WHERE sku IS NULL OR sku = ''
HAVING COUNT(*) > 0

UNION ALL

-- Duplicate SKUs
SELECT 
    'Duplicate SKU Codes',
    COUNT(*) - COUNT(DISTINCT sku),
    ROUND((COUNT(*) - COUNT(DISTINCT sku)) * 100.0 / COUNT(*), 2),
    'Critical',
    ''RED'',
    'Resolve SKU conflicts immediately'
FROM products
WHERE sku IS NOT NULL AND sku != ''
HAVING COUNT(*) - COUNT(DISTINCT sku) > 0

UNION ALL

-- Negative Stock Quantities
SELECT 
    'Negative Stock Quantities',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products), 2),
    'Critical',
    ''RED'',
    'Correct inventory discrepancies'
FROM products
WHERE stock_quantity < 0
HAVING COUNT(*) > 0

UNION ALL

-- Active Products Out of Stock
SELECT 
    'Active Products with Zero Stock',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2),
    'Medium',
    ''YELLOW'',
    'Restock or change status to out_of_stock'
FROM products
WHERE status = 'active' AND stock_quantity = 0
HAVING COUNT(*) > 0;

-- ========================================
-- 4. ORDER DATA QUALITY - DETAILED
-- Issues detected today in order data
-- ========================================

SELECT 
    '🛒 ORDER DATA QUALITY ISSUES' AS section,
    '' AS spacer;

-- Orphaned Orders
SELECT 
    'Orders without Valid Customer' AS issue_type,
    COUNT(*) AS issue_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2) AS percentage,
    'Critical' AS severity,
    ''RED'' AS priority,
    'Link orders to customer records' AS recommended_action
FROM orders
WHERE customer_id IS NULL 
   OR customer_id NOT IN (SELECT customer_id FROM customers)
HAVING COUNT(*) > 0

UNION ALL

-- Invalid Order Amounts
SELECT 
    'Invalid Order Amounts',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2),
    'Critical',
    ''RED'',
    'Recalculate order totals'
FROM orders
WHERE total_amount IS NULL OR total_amount <= 0
HAVING COUNT(*) > 0

UNION ALL

-- Missing Order Dates
SELECT 
    'Missing Order Dates',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2),
    'Critical',
    ''RED'',
    'Add order date timestamps'
FROM orders
WHERE order_date IS NULL
HAVING COUNT(*) > 0

UNION ALL

-- Future Dated Orders
SELECT 
    'Orders with Future Dates',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2),
    'High',
    ''RED'',
    'Correct order date timestamps'
FROM orders
WHERE order_date > NOW()
HAVING COUNT(*) > 0

UNION ALL

-- Orders without Items
SELECT 
    'Orders with No Line Items',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2),
    'Critical',
    ''RED'',
    'Add order items or cancel orders'
FROM orders o
WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.order_id = o.order_id)
HAVING COUNT(*) > 0

UNION ALL

-- Paid Orders Still Pending
SELECT 
    'Paid Orders Still in Pending Status',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE payment_status = 'paid'), 2),
    'High',
    ''RED'',
    'Update order status to processing'
FROM orders
WHERE payment_status = 'paid' 
  AND status = 'pending'
  AND order_date < DATE_SUB(NOW(), INTERVAL 24 HOUR)
HAVING COUNT(*) > 0

UNION ALL

-- Orders Stuck in Processing
SELECT 
    'Orders Stuck in Processing (7+ days)',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders WHERE status = 'processing'), 2),
    'High',
    ''RED'',
    'Investigate fulfillment delays'
FROM orders
WHERE status = 'processing'
  AND order_date < DATE_SUB(CURDATE(), INTERVAL 7 DAY)
HAVING COUNT(*) > 0

UNION ALL

-- Mismatched Order Totals
SELECT 
    'Order Total Mismatch with Line Items',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM orders), 2),
    'Critical',
    ''RED'',
    'Recalculate and update order totals'
FROM orders o
JOIN (
    SELECT order_id, SUM(subtotal) AS calculated_total
    FROM order_items
    GROUP BY order_id
) oi ON o.order_id = oi.order_id
WHERE ABS(o.total_amount - oi.calculated_total) > 0.01
HAVING COUNT(*) > 0;

-- ========================================
-- 5. INVENTORY DATA QUALITY
-- Stock level and tracking issues
-- ========================================

SELECT 
    ''CHART' INVENTORY DATA QUALITY ISSUES' AS section,
    '' AS spacer;

-- Products without Inventory Records
SELECT 
    'Products Missing Inventory Records' AS issue_type,
    COUNT(*) AS issue_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products WHERE status = 'active'), 2) AS percentage,
    'High' AS severity,
    ''RED'' AS priority,
    'Create inventory records for all active products' AS recommended_action
FROM products p
WHERE p.status = 'active'
  AND NOT EXISTS (SELECT 1 FROM inventory i WHERE i.product_id = p.product_id)
HAVING COUNT(*) > 0

UNION ALL

-- Stock Quantity Mismatch
SELECT 
    'Stock Quantity Mismatch (Product vs Inventory)',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM products p JOIN inventory i ON p.product_id = i.product_id), 2),
    'High',
    ''RED'',
    'Reconcile stock quantities between tables'
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.stock_quantity != i.quantity_on_hand
HAVING COUNT(*) > 0

UNION ALL

-- Reserved Quantity Exceeds On-Hand
SELECT 
    'Reserved Quantity Exceeds Available Stock',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM inventory), 2),
    'Critical',
    ''RED'',
    'Audit and correct reserved quantities'
FROM inventory
WHERE quantity_reserved > quantity_on_hand
HAVING COUNT(*) > 0

UNION ALL

-- Products Below Reorder Level
SELECT 
    'Products Below Reorder Level',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM inventory), 2),
    'Medium',
    ''YELLOW'',
    'Generate purchase orders for low stock items'
FROM inventory
WHERE quantity_available < reorder_level
HAVING COUNT(*) > 0;

-- ========================================
-- 6. VENDOR DATA QUALITY
-- Supplier data issues
-- ========================================

SELECT 
    '🏭 VENDOR DATA QUALITY ISSUES' AS section,
    '' AS spacer;

-- Missing Vendor Contact Info
SELECT 
    'Vendors Missing Contact Information' AS issue_type,
    COUNT(*) AS issue_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM vendors), 2) AS percentage,
    'Medium' AS severity,
    ''YELLOW'' AS priority,
    'Update vendor contact details' AS recommended_action
FROM vendors
WHERE (email IS NULL OR email = '') 
   OR (phone IS NULL OR phone = '')
   OR (contact_person IS NULL OR contact_person = '')
HAVING COUNT(*) > 0

UNION ALL

-- Invalid Vendor Ratings
SELECT 
    'Invalid Vendor Ratings',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM vendors), 2),
    'Low',
    ''GREEN'',
    'Validate rating values (0-5 scale)'
FROM vendors
WHERE rating IS NOT NULL AND (rating < 0 OR rating > 5)
HAVING COUNT(*) > 0

UNION ALL

-- Vendors without Active Contracts
SELECT 
    'Active Vendors without Contracts',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM vendors WHERE status = 'active'), 2),
    'Medium',
    ''YELLOW'',
    'Review vendor relationships and create contracts'
FROM vendors v
WHERE v.status = 'active'
  AND NOT EXISTS (
      SELECT 1 FROM vendor_contracts vc 
      WHERE vc.vendor_id = v.vendor_id AND vc.status = 'active'
  )
HAVING COUNT(*) > 0

UNION ALL

-- Expired Vendor Contracts
SELECT 
    'Expired Vendor Contracts Not Updated',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM vendor_contracts), 2),
    'High',
    ''RED'',
    'Update expired contracts or create new ones'
FROM vendor_contracts
WHERE end_date < CURDATE() AND status = 'active'
HAVING COUNT(*) > 0;

-- ========================================
-- 7. REVIEW DATA QUALITY
-- Customer review issues
-- ========================================

SELECT 
    '⭐ REVIEW DATA QUALITY ISSUES' AS section,
    '' AS spacer;

-- Reviews without Ratings
SELECT 
    'Reviews Missing Rating Scores' AS issue_type,
    COUNT(*) AS issue_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) AS percentage,
    'High' AS severity,
    ''RED'' AS priority,
    'Add rating scores to reviews' AS recommended_action
FROM reviews
WHERE rating IS NULL
HAVING COUNT(*) > 0

UNION ALL

-- Invalid Rating Values
SELECT 
    'Invalid Rating Values (Not 1-5)',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2),
    'High',
    ''RED'',
    'Correct rating values to 1-5 scale'
FROM reviews
WHERE rating IS NOT NULL AND (rating < 1 OR rating > 5)
HAVING COUNT(*) > 0

UNION ALL

-- Reviews Pending Too Long
SELECT 
    'Reviews Pending Moderation (7+ days)',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews WHERE status = 'pending'), 2),
    'Medium',
    ''YELLOW'',
    'Expedite review moderation process'
FROM reviews
WHERE status = 'pending'
  AND created_at < DATE_SUB(CURDATE(), INTERVAL 7 DAY)
HAVING COUNT(*) > 0

UNION ALL

-- Reviews for Non-existent Products
SELECT 
    'Reviews for Deleted/Invalid Products',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2),
    'High',
    ''RED'',
    'Archive or delete orphaned reviews'
FROM reviews r
WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = r.product_id)
HAVING COUNT(*) > 0;

-- ========================================
-- 8. CAMPAIGN DATA QUALITY
-- Marketing campaign data issues
-- ========================================

SELECT 
    '📢 CAMPAIGN DATA QUALITY ISSUES' AS section,
    '' AS spacer;

-- Campaigns Missing Budget
SELECT 
    'Campaigns without Budget Allocation' AS issue_type,
    COUNT(*) AS issue_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM campaigns), 2) AS percentage,
    'Medium' AS severity,
    ''YELLOW'' AS priority,
    'Assign budget to active campaigns' AS recommended_action
FROM campaigns
WHERE budget IS NULL OR budget <= 0
HAVING COUNT(*) > 0

UNION ALL

-- Active Campaigns Past End Date
SELECT 
    'Active Campaigns Past End Date',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM campaigns WHERE status = 'active'), 2),
    'Medium',
    ''YELLOW'',
    'Update campaign status to completed'
FROM campaigns
WHERE status = 'active' AND end_date < CURDATE()
HAVING COUNT(*) > 0

UNION ALL

-- Campaigns without Performance Data
SELECT 
    'Active Campaigns Missing Performance Metrics',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM campaigns WHERE status = 'active'), 2),
    'Low',
    ''GREEN'',
    'Track and record campaign performance'
FROM campaigns c
WHERE c.status = 'active'
  AND c.start_date <= CURDATE()
  AND NOT EXISTS (
      SELECT 1 FROM campaign_performance cp 
      WHERE cp.campaign_id = c.campaign_id
  )
HAVING COUNT(*) > 0;

-- ========================================
-- 9. RETURN DATA QUALITY
-- Returns and refunds issues
-- ========================================

SELECT 
    '🔄 RETURN DATA QUALITY ISSUES' AS section,
    '' AS spacer;

-- Returns without Refund Amounts
SELECT 
    'Approved Returns Missing Refund Amount' AS issue_type,
    COUNT(*) AS issue_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM returns WHERE status = 'approved'), 2) AS percentage,
    'High' AS severity,
    ''RED'' AS priority,
    'Calculate and add refund amounts' AS recommended_action
FROM returns
WHERE status IN ('approved', 'refunded') 
  AND (refund_amount IS NULL OR refund_amount <= 0)
HAVING COUNT(*) > 0

UNION ALL

-- Old Pending Returns
SELECT 
    'Returns Pending for 14+ Days',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM returns WHERE status = 'requested'), 2),
    'High',
    ''RED'',
    'Process pending return requests'
FROM returns
WHERE status = 'requested'
  AND created_at < DATE_SUB(CURDATE(), INTERVAL 14 DAY)
HAVING COUNT(*) > 0

UNION ALL

-- Returns for Invalid Orders
SELECT 
    'Returns for Non-existent Orders',
    COUNT(*),
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM returns), 2),
    'Critical',
    ''RED'',
    'Clean up orphaned return records'
FROM returns r
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = r.order_id)
HAVING COUNT(*) > 0;

-- ========================================
-- 10. QUALITY TRENDS - 7 DAY COMPARISON
-- Track quality improvements or degradation
-- ========================================

SELECT 
    ''TRENDING_UP' QUALITY TRENDS (7-Day Comparison)' AS section,
    '' AS spacer;

WITH today_quality AS (
    SELECT 
        (SELECT COUNT(*) FROM customers 
         WHERE DATE(created_at) = CURDATE() 
         AND (email IS NULL OR email = '' OR email NOT LIKE '%@%.%')) AS invalid_emails_today,
        (SELECT COUNT(*) FROM products 
         WHERE DATE(created_at) = CURDATE() 
         AND (price IS NULL OR price <= 0)) AS invalid_prices_today,
        (SELECT COUNT(*) FROM orders 
         WHERE DATE(order_date) = CURDATE() 
         AND customer_id IS NULL) AS orphaned_orders_today,
        (SELECT COUNT(*) FROM products 
         WHERE status = 'active' AND stock_quantity = 0) AS out_of_stock_today
),
week_ago_quality AS (
    SELECT 
        (SELECT COUNT(*) FROM customers 
         WHERE DATE(created_at) = DATE_SUB(CURDATE(), INTERVAL 7 DAY)
         AND (email IS NULL OR email = '' OR email NOT LIKE '%@%.%')) AS invalid_emails_7d,
        (SELECT COUNT(*) FROM products 
         WHERE DATE(created_at) = DATE_SUB(CURDATE(), INTERVAL 7 DAY)
         AND (price IS NULL OR price <= 0)) AS invalid_prices_7d,
        (SELECT COUNT(*) FROM orders 
         WHERE DATE(order_date) = DATE_SUB(CURDATE(), INTERVAL 7 DAY)
         AND customer_id IS NULL) AS orphaned_orders_7d,
        (SELECT COUNT(*) FROM products 
         WHERE status = 'active' AND stock_quantity = 0) AS out_of_stock_7d
)
SELECT 
    'Invalid Customer Emails' AS quality_metric,
    tq.invalid_emails_today AS today_count,
    wq.invalid_emails_7d AS week_ago_count,
    tq.invalid_emails_today - wq.invalid_emails_7d AS change,
    CASE 
        WHEN tq.invalid_emails_today < wq.invalid_emails_7d THEN ''TRENDING_DOWN' Improving'
        WHEN tq.invalid_emails_today > wq.invalid_emails_7d THEN ''TRENDING_UP' Degrading'
        ELSE '➡️ Stable'
    END AS trend,
    CASE 
        WHEN tq.invalid_emails_today < wq.invalid_emails_7d THEN ''GREEN''
        WHEN tq.invalid_emails_today > wq.invalid_emails_7d THEN ''RED''
        ELSE ''YELLOW''
    END AS status_icon
FROM today_quality tq, week_ago_quality wq

UNION ALL

SELECT 
    'Invalid Product Prices',
    tq.invalid_prices_today,
    wq.invalid_prices_7d,
    tq.invalid_prices_today - wq.invalid_prices_7d,
    CASE 
        WHEN tq.invalid_prices_today < wq.invalid_prices_7d THEN ''TRENDING_DOWN' Improving'
        WHEN tq.invalid_prices_today > wq.invalid_prices_7d THEN ''TRENDING_UP' Degrading'
        ELSE '➡️ Stable'
    END,
    CASE 
        WHEN tq.invalid_prices_today < wq.invalid_prices_7d THEN ''GREEN''
        WHEN tq.invalid_prices_today > wq.invalid_prices_7d THEN ''RED''
        ELSE ''YELLOW''
    END
FROM today_quality tq, week_ago_quality wq

UNION ALL

SELECT 
    'Orphaned Orders',
    tq.orphaned_orders_today,
    wq.orphaned_orders_7d,
    tq.orphaned_orders_today - wq.orphaned_orders_7d,
    CASE 
        WHEN tq.orphaned_orders_today < wq.orphaned_orders_7d THEN ''TRENDING_DOWN' Improving'
        WHEN tq.orphaned_orders_today > wq.orphaned_orders_7d THEN ''TRENDING_UP' Degrading'
        ELSE '➡️ Stable'
    END,
    CASE 
        WHEN tq.orphaned_orders_today < wq.orphaned_orders_7d THEN ''GREEN''
        WHEN tq.orphaned_orders_today > wq.orphaned_orders_7d THEN ''RED''
        ELSE ''YELLOW''
    END
FROM today_quality tq, week_ago_quality wq

UNION ALL

SELECT 
    'Out of Stock Products',
    tq.out_of_stock_today,
    wq.out_of_stock_7d,
    tq.out_of_stock_today - wq.out_of_stock_7d,
    CASE 
        WHEN tq.out_of_stock_today < wq.out_of_stock_7d THEN ''TRENDING_DOWN' Improving'
        WHEN tq.out_of_stock_today > wq.out_of_stock_7d THEN ''TRENDING_UP' Degrading'
        ELSE '➡️ Stable'
    END,
    CASE 
        WHEN tq.out_of_stock_today < wq.out_of_stock_7d THEN ''GREEN''
        WHEN tq.out_of_stock_today > wq.out_of_stock_7d THEN ''RED''
        ELSE ''YELLOW''
    END
FROM today_quality tq, week_ago_quality wq;

-- ========================================
-- 11. TOP 10 DATA QUALITY ISSUES
-- Prioritized list of most critical issues
-- ========================================

SELECT 
    '🚨 TOP 10 CRITICAL DATA QUALITY ISSUES' AS section,
    '' AS spacer;

WITH all_issues AS (
    SELECT 'Invalid Email Addresses' AS issue,
           COUNT(*) AS count,
           'Customer' AS category,
           'Critical' AS severity,
           1 AS priority_order
    FROM customers
    WHERE email IS NULL OR email = '' OR email NOT LIKE '%@%.%'
    
    UNION ALL
    
    SELECT 'Invalid Product Prices',
           COUNT(*),
           'Product',
           'Critical',
           1
    FROM products
    WHERE price IS NULL OR price <= 0
    
    UNION ALL
    
    SELECT 'Products Selling Below Cost',
           COUNT(*),
           'Product',
           'Critical',
           1
    FROM products
    WHERE price < cost AND price > 0 AND cost > 0
    
    UNION ALL
    
    SELECT 'Orphaned Orders',
           COUNT(*),
           'Order',
           'Critical',
           1
    FROM orders
    WHERE customer_id IS NULL OR customer_id NOT IN (SELECT customer_id FROM customers)
    
    UNION ALL
    
    SELECT 'Duplicate SKU Codes',
           COUNT(*) - COUNT(DISTINCT sku),
           'Product',
           'Critical',
           1
    FROM products
    WHERE sku IS NOT NULL AND sku != ''
    
    UNION ALL
    
    SELECT 'Negative Stock Quantities',
           COUNT(*),
           'Inventory',
           'Critical',
           1
    FROM products
    WHERE stock_quantity < 0
    
    UNION ALL
    
    SELECT 'Orders Stuck in Processing',
           COUNT(*),
           'Order',
           'High',
           2
    FROM orders
    WHERE status = 'processing' AND order_date < DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    
    UNION ALL
    
    SELECT 'Missing Product Descriptions',
           COUNT(*),
           'Product',
           'Medium',
           3
    FROM products
    WHERE description IS NULL OR description = ''
    
    UNION ALL
    
    SELECT 'Incomplete Customer Addresses',
           COUNT(*),
           'Customer',
           'High',
           2
    FROM customers
    WHERE address_line1 IS NULL OR address_line1 = '' 
       OR city IS NULL OR city = ''
    
    UNION ALL
    
    SELECT 'Active Products Out of Stock',
           COUNT(*),
           'Inventory',
           'High',
           2
    FROM products
    WHERE status = 'active' AND stock_quantity = 0
)
SELECT 
    ROW_NUMBER() OVER (ORDER BY priority_order, count DESC) AS rank,
    issue,
    count AS affected_records,
    category,
    severity,
    CASE severity
        WHEN 'Critical' THEN ''RED''
        WHEN 'High' THEN '🟠'
        WHEN 'Medium' THEN ''YELLOW''
        ELSE ''GREEN''
    END AS severity_icon,
    CASE 
        WHEN count = 0 THEN 'No Action Needed'
        WHEN count < 10 THEN 'Minor - Address Soon'
        WHEN count < 50 THEN 'Moderate - Address Today'
        WHEN count < 100 THEN 'Significant - Urgent Action'
        ELSE 'Critical - Immediate Action Required'
    END AS action_required
FROM all_issues
WHERE count > 0
ORDER BY priority_order, count DESC
LIMIT 10;

-- ========================================
-- 12. DATA COMPLETENESS SCORECARD
-- Field-level completeness analysis
-- ========================================

SELECT 
    '📋 DATA COMPLETENESS SCORECARD' AS section,
    '' AS spacer;

SELECT 
    'Customers' AS table_name,
    'Email' AS field_name,
    COUNT(*) AS total_records,
    COUNT(email) AS populated_records,
    COUNT(*) - COUNT(email) AS missing_records,
    ROUND(COUNT(email) * 100.0 / COUNT(*), 2) AS completeness_pct,
    CASE 
        WHEN ROUND(COUNT(email) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
        WHEN ROUND(COUNT(email) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
        WHEN ROUND(COUNT(email) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
        ELSE ''RED' Poor'
    END AS grade
FROM customers

UNION ALL

SELECT 'Customers', 'Phone',
       COUNT(*), COUNT(phone), COUNT(*) - COUNT(phone),
       ROUND(COUNT(phone) * 100.0 / COUNT(*), 2),
       CASE 
           WHEN ROUND(COUNT(phone) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
           WHEN ROUND(COUNT(phone) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
           WHEN ROUND(COUNT(phone) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
           ELSE ''RED' Poor'
       END
FROM customers

UNION ALL

SELECT 'Customers', 'Address',
       COUNT(*), COUNT(address_line1), COUNT(*) - COUNT(address_line1),
       ROUND(COUNT(address_line1) * 100.0 / COUNT(*), 2),
       CASE 
           WHEN ROUND(COUNT(address_line1) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
           WHEN ROUND(COUNT(address_line1) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
           WHEN ROUND(COUNT(address_line1) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
           ELSE ''RED' Poor'
       END
FROM customers

UNION ALL

SELECT 'Products', 'Description',
       COUNT(*), COUNT(description), COUNT(*) - COUNT(description),
       ROUND(COUNT(description) * 100.0 / COUNT(*), 2),
       CASE 
           WHEN ROUND(COUNT(description) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
           WHEN ROUND(COUNT(description) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
           WHEN ROUND(COUNT(description) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
           ELSE ''RED' Poor'
       END
FROM products

UNION ALL

SELECT 'Products', 'Price',
       COUNT(*), COUNT(price), COUNT(*) - COUNT(price),
       ROUND(COUNT(price) * 100.0 / COUNT(*), 2),
       CASE 
           WHEN ROUND(COUNT(price) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
           WHEN ROUND(COUNT(price) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
           WHEN ROUND(COUNT(price) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
           ELSE ''RED' Poor'
       END
FROM products

UNION ALL

SELECT 'Products', 'Category',
       COUNT(*), COUNT(category_id), COUNT(*) - COUNT(category_id),
       ROUND(COUNT(category_id) * 100.0 / COUNT(*), 2),
       CASE 
           WHEN ROUND(COUNT(category_id) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
           WHEN ROUND(COUNT(category_id) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
           WHEN ROUND(COUNT(category_id) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
           ELSE ''RED' Poor'
       END
FROM products

UNION ALL

SELECT 'Products', 'SKU',
       COUNT(*), COUNT(sku), COUNT(*) - COUNT(sku),
       ROUND(COUNT(sku) * 100.0 / COUNT(*), 2),
       CASE 
           WHEN ROUND(COUNT(sku) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
           WHEN ROUND(COUNT(sku) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
           WHEN ROUND(COUNT(sku) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
           ELSE ''RED' Poor'
       END
FROM products

UNION ALL

SELECT 'Orders', 'Customer ID',
       COUNT(*), COUNT(customer_id), COUNT(*) - COUNT(customer_id),
       ROUND(COUNT(customer_id) * 100.0 / COUNT(*), 2),
       CASE 
           WHEN ROUND(COUNT(customer_id) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
           WHEN ROUND(COUNT(customer_id) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
           WHEN ROUND(COUNT(customer_id) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
           ELSE ''RED' Poor'
       END
FROM orders

UNION ALL

SELECT 'Orders', 'Order Date',
       COUNT(*), COUNT(order_date), COUNT(*) - COUNT(order_date),
       ROUND(COUNT(order_date) * 100.0 / COUNT(*), 2),
       CASE 
           WHEN ROUND(COUNT(order_date) * 100.0 / COUNT(*), 2) >= 95 THEN ''GREEN' Excellent'
           WHEN ROUND(COUNT(order_date) * 100.0 / COUNT(*), 2) >= 80 THEN ''YELLOW' Good'
           WHEN ROUND(COUNT(order_date) * 100.0 / COUNT(*), 2) >= 60 THEN '🟠 Fair'
           ELSE ''RED' Poor'
       END
FROM orders;

-- ========================================
-- 13. DATA VALIDATION SUMMARY
-- Business rule validation results
-- ========================================

SELECT 
    ''SUCCESS' DATA VALIDATION SUMMARY' AS section,
    '' AS spacer;

SELECT 
    'Price Validation' AS validation_rule,
    'All products must have price > 0' AS rule_description,
    COUNT(*) AS total_checked,
    COUNT(*) - COUNT(CASE WHEN price > 0 THEN 1 END) AS violations,
    ROUND((COUNT(*) - COUNT(CASE WHEN price > 0 THEN 1 END)) * 100.0 / COUNT(*), 2) AS violation_rate,
    CASE 
        WHEN COUNT(*) - COUNT(CASE WHEN price > 0 THEN 1 END) = 0 THEN ''SUCCESS' Pass'
        WHEN COUNT(*) - COUNT(CASE WHEN price > 0 THEN 1 END) < 10 THEN ''WARNING' Warning'
        ELSE ''ERROR' Fail'
    END AS validation_status
FROM products

UNION ALL

SELECT 
    'Margin Validation',
    'Products should sell above cost',
    COUNT(*),
    COUNT(CASE WHEN price > 0 AND cost > 0 AND price < cost THEN 1 END),
    ROUND(COUNT(CASE WHEN price > 0 AND cost > 0 AND price < cost THEN 1 END) * 100.0 / COUNT(*), 2),
    CASE 
        WHEN COUNT(CASE WHEN price > 0 AND cost > 0 AND price < cost THEN 1 END) = 0 THEN ''SUCCESS' Pass'
        WHEN COUNT(CASE WHEN price > 0 AND cost > 0 AND price < cost THEN 1 END) < 5 THEN ''WARNING' Warning'
        ELSE ''ERROR' Fail'
    END
FROM products

UNION ALL

SELECT 
    'Email Validation',
    'Customer emails must be valid format',
    COUNT(*),
    COUNT(CASE WHEN email IS NOT NULL AND email NOT LIKE '%@%.%' THEN 1 END),
    ROUND(COUNT(CASE WHEN email IS NOT NULL AND email NOT LIKE '%@%.%' THEN 1 END) * 100.0 / COUNT(*), 2),
    CASE 
        WHEN COUNT(CASE WHEN email IS NOT NULL AND email NOT LIKE '%@%.%' THEN 1 END) = 0 THEN ''SUCCESS' Pass'
        WHEN COUNT(CASE WHEN email IS NOT NULL AND email NOT LIKE '%@%.%' THEN 1 END) < 20 THEN ''WARNING' Warning'
        ELSE ''ERROR' Fail'
    END
FROM customers

UNION ALL

SELECT 
    'Order Total Validation',
    'Order totals must match line items',
    COUNT(DISTINCT o.order_id),
    COUNT(DISTINCT CASE WHEN ABS(o.total_amount - COALESCE(oi.total, 0)) > 0.01 THEN o.order_id END),
    ROUND(COUNT(DISTINCT CASE WHEN ABS(o.total_amount - COALESCE(oi.total, 0)) > 0.01 THEN o.order_id END) * 100.0 / COUNT(DISTINCT o.order_id), 2),
    CASE 
        WHEN COUNT(DISTINCT CASE WHEN ABS(o.total_amount - COALESCE(oi.total, 0)) > 0.01 THEN o.order_id END) = 0 THEN ''SUCCESS' Pass'
        WHEN COUNT(DISTINCT CASE WHEN ABS(o.total_amount - COALESCE(oi.total, 0)) > 0.01 THEN o.order_id END) < 5 THEN ''WARNING' Warning'
        ELSE ''ERROR' Fail'
    END
FROM orders o
LEFT JOIN (
    SELECT order_id, SUM(subtotal) AS total
    FROM order_items
    GROUP BY order_id
) oi ON o.order_id = oi.order_id

UNION ALL

SELECT 
    'Stock Validation',
    'Stock quantities cannot be negative',
    COUNT(*),
    COUNT(CASE WHEN stock_quantity < 0 THEN 1 END),
    ROUND(COUNT(CASE WHEN stock_quantity < 0 THEN 1 END) * 100.0 / COUNT(*), 2),
    CASE 
        WHEN COUNT(CASE WHEN stock_quantity < 0 THEN 1 END) = 0 THEN ''SUCCESS' Pass'
        WHEN COUNT(CASE WHEN stock_quantity < 0 THEN 1 END) < 5 THEN ''WARNING' Warning'
        ELSE ''ERROR' Fail'
    END
FROM products

UNION ALL

SELECT 
    'Date Validation',
    'Order dates cannot be in the future',
    COUNT(*),
    COUNT(CASE WHEN order_date > NOW() THEN 1 END),
    ROUND(COUNT(CASE WHEN order_date > NOW() THEN 1 END) * 100.0 / COUNT(*), 2),
    CASE 
        WHEN COUNT(CASE WHEN order_date > NOW() THEN 1 END) = 0 THEN ''SUCCESS' Pass'
        ELSE ''ERROR' Fail'
    END
FROM orders

UNION ALL

SELECT 
    'Rating Validation',
    'Review ratings must be between 1-5',
    COUNT(*),
    COUNT(CASE WHEN rating IS NOT NULL AND (rating < 1 OR rating > 5) THEN 1 END),
    ROUND(COUNT(CASE WHEN rating IS NOT NULL AND (rating < 1 OR rating > 5) THEN 1 END) * 100.0 / COUNT(*), 2),
    CASE 
        WHEN COUNT(CASE WHEN rating IS NOT NULL AND (rating < 1 OR rating > 5) THEN 1 END) = 0 THEN ''SUCCESS' Pass'
        ELSE ''ERROR' Fail'
    END
FROM reviews;

-- ========================================
-- 14. RECOMMENDED ACTIONS
-- Prioritized action items
-- ========================================

SELECT 
    '🎯 RECOMMENDED ACTIONS' AS section,
    '' AS spacer;

WITH issue_summary AS (
    SELECT 
        (SELECT COUNT(*) FROM customers WHERE email IS NULL OR email = '' OR email NOT LIKE '%@%.%') AS invalid_emails,
        (SELECT COUNT(*) FROM products WHERE price IS NULL OR price <= 0) AS invalid_prices,
        (SELECT COUNT(*) FROM orders WHERE customer_id IS NULL) AS orphaned_orders,
        (SELECT COUNT(*) FROM products WHERE stock_quantity < 0) AS negative_stock,
        (SELECT COUNT(*) FROM products WHERE status = 'active' AND stock_quantity = 0) AS out_of_stock,
        (SELECT COUNT(*) FROM orders WHERE status = 'processing' AND order_date < DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS stuck_orders,
        (SELECT COUNT(*) FROM reviews WHERE status = 'pending' AND created_at < DATE_SUB(CURDATE(), INTERVAL 7 DAY)) AS old_pending_reviews,
        (SELECT COUNT(*) FROM returns WHERE status = 'requested' AND created_at < DATE_SUB(CURDATE(), INTERVAL 14 DAY)) AS old_pending_returns
)
SELECT 
    1 AS priority,
    ''RED' CRITICAL' AS urgency,
    'Fix Invalid Product Prices' AS action,
    CONCAT(invalid_prices, ' products') AS scope,
    'Products cannot be sold without valid pricing' AS impact,
    'Update pricing immediately' AS next_steps
FROM issue_summary
WHERE invalid_prices > 0

UNION ALL

SELECT 2, ''RED' CRITICAL', 'Correct Negative Stock Quantities',
       CONCAT(negative_stock, ' products'), 
       'Inventory tracking is unreliable',
       'Audit and correct stock levels'
FROM issue_summary
WHERE negative_stock > 0

UNION ALL

SELECT 3, ''RED' CRITICAL', 'Link Orphaned Orders to Customers',
       CONCAT(orphaned_orders, ' orders'),
       'Revenue tracking and customer analytics affected',
       'Match orders to customer records'
FROM issue_summary
WHERE orphaned_orders > 0

UNION ALL

SELECT 4, '🟠 HIGH', 'Validate Customer Emails',
       CONCAT(invalid_emails, ' customers'),
       'Cannot communicate with customers',
       'Request email updates or validate formats'
FROM issue_summary
WHERE invalid_emails > 0

UNION ALL

SELECT 5, '🟠 HIGH', 'Process Stuck Orders',
       CONCAT(stuck_orders, ' orders'),
       'Poor customer experience and revenue delays',
       'Investigate and expedite fulfillment'
FROM issue_summary
WHERE stuck_orders > 0

UNION ALL

SELECT 6, ''YELLOW' MEDIUM', 'Restock Out-of-Stock Products',
       CONCAT(out_of_stock, ' products'),
       'Lost sales opportunities',
       'Generate purchase orders or update status'
FROM issue_summary
WHERE out_of_stock > 0

UNION ALL

SELECT 7, ''YELLOW' MEDIUM', 'Moderate Pending Reviews',
       CONCAT(old_pending_reviews, ' reviews'),
       'Customer feedback not visible',
       'Expedite review moderation queue'
FROM issue_summary
WHERE old_pending_reviews > 0

UNION ALL

SELECT 8, ''YELLOW' MEDIUM', 'Process Pending Returns',
       CONCAT(old_pending_returns, ' returns'),
       'Poor customer service experience',
       'Review and process return requests'
FROM issue_summary
WHERE old_pending_returns > 0

ORDER BY priority;

-- ========================================
-- REPORT FOOTER - SUMMARY
-- ========================================

SELECT 
    '═══════════════════════════════════════════════════════════' AS separator,
    CONCAT('Total Issues Found: ', 
           (SELECT COUNT(*) FROM (
               SELECT 1 FROM customers WHERE email IS NULL OR email = '' OR email NOT LIKE '%@%.%'
               UNION ALL SELECT 1 FROM products WHERE price IS NULL OR price <= 0
               UNION ALL SELECT 1 FROM orders WHERE customer_id IS NULL
               UNION ALL SELECT 1 FROM products WHERE stock_quantity < 0
           ) AS all_issues)
    ) AS summary,
    CONCAT('Report Generated: ', DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s')) AS timestamp,
    ''SUCCESS' Daily Data Quality Report Complete' AS status;