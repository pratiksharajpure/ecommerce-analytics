-- ========================================
-- MYSQL 8 SCHEMA FIX & DIAGNOSTIC SCRIPT
-- ========================================

USE ecommerce_analytics;

-- Verify MySQL version
SELECT VERSION() as mysql_version,
       @@version_comment as server_type,
       CASE 
           WHEN VERSION() LIKE '8%' THEN ''SUCCESS' MySQL 8 Detected'
           ELSE ''WARNING' Not MySQL 8'
       END as status;

-- ========================================
-- FIX 1: ADD ALL MISSING COLUMNS
-- ========================================

-- Countries table
ALTER TABLE countries 
ADD COLUMN IF NOT EXISTS country_code VARCHAR(3) AFTER country_id;

ALTER TABLE countries
ADD UNIQUE INDEX IF NOT EXISTS uk_country_code (country_code);

UPDATE countries SET country_code = LEFT(country_id, 3) WHERE country_code IS NULL;

-- Categories table
ALTER TABLE categories
ADD COLUMN IF NOT EXISTS category_code VARCHAR(20) AFTER category_id;

ALTER TABLE categories
ADD UNIQUE INDEX IF NOT EXISTS uk_category_code (category_code);

UPDATE categories 
SET category_code = UPPER(REPLACE(REPLACE(category_name, ' ', '_'), '-', '_'))
WHERE category_code IS NULL;

-- Shipping table
ALTER TABLE shipping
ADD COLUMN IF NOT EXISTS method_code VARCHAR(20) AFTER shipping_method;

ALTER TABLE shipping
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE AFTER tracking_number;

UPDATE shipping 
SET method_code = CASE 
    WHEN LOWER(shipping_method) LIKE '%standard%' THEN 'STANDARD'
    WHEN LOWER(shipping_method) LIKE '%express%' THEN 'EXPRESS'
    WHEN LOWER(shipping_method) LIKE '%overnight%' THEN 'OVERNIGHT'
    WHEN LOWER(shipping_method) LIKE '%free%' THEN 'FREE'
    ELSE 'STANDARD'
END
WHERE method_code IS NULL;

-- Orders table
ALTER TABLE orders
ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending' AFTER total_amount;

UPDATE orders SET status = 'delivered' WHERE status IS NULL AND order_date < DATE_SUB(NOW(), INTERVAL 7 DAY);
UPDATE orders SET status = 'pending' WHERE status IS NULL;

-- Reviews table
ALTER TABLE reviews
ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'approved' AFTER verified_purchase;

-- Vendors table
ALTER TABLE vendors
ADD COLUMN IF NOT EXISTS category_code VARCHAR(20) AFTER contact_email;

-- Campaign_performance table
USE ecommerce_analytics;

-- First, let's see what columns you currently have
DESCRIBE campaign_performance;

-- Add conversion_rate column
ALTER TABLE campaign_performance
ADD COLUMN conversion_rate DECIMAL(5,2) AFTER conversions;

-- Then add min_orders column


-- Verify columns were added
SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'ecommerce_analytics' 
  AND TABLE_NAME = 'campaign_performance'
ORDER BY ORDINAL_POSITION;

-- Payment_methods table (ensure card_brand exists)
ALTER TABLE payment_methods
ADD COLUMN IF NOT EXISTS card_brand VARCHAR(20) AFTER card_type;

UPDATE payment_methods 
SET card_brand = card_type 
WHERE card_brand IS NULL;

-- ========================================
-- FIX 2: CREATE MISSING TABLES
-- ========================================

-- Customer segment assignments
CREATE TABLE IF NOT EXISTS customer_segment_assignments (
    assignment_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    segment_code VARCHAR(20) NOT NULL,
    assigned_date DATE NOT NULL DEFAULT (CURRENT_DATE),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (segment_code) REFERENCES customer_segments(segment_code),
    UNIQUE KEY uk_customer_segment (customer_id, segment_code, assigned_date),
    INDEX idx_active (is_active),
    INDEX idx_segment (segment_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ========================================
-- FIX 3: VERIFY TABLE STRUCTURE
-- ========================================

-- Check all tables exist
SELECT 
    TABLE_NAME,
    TABLE_ROWS,
    ROUND(DATA_LENGTH / 1024 / 1024, 2) as 'Size_MB'
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'ecommerce_analytics'
    AND TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_ROWS DESC;

-- Check for missing foreign key parent records
SELECT 'Checking foreign key integrity...' as status;

-- Orders without customers
SELECT COUNT(*) as orders_without_customers
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;

-- Order items without products
SELECT COUNT(*) as order_items_without_products
FROM order_items oi
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE p.product_id IS NULL;

-- Order items without orders
SELECT COUNT(*) as order_items_without_orders
FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;

-- ========================================
-- FIX 4: CHECK FOR PROBLEMATIC DATA
-- ========================================

-- Check for empty primary keys (causing duplicate entry errors)
SELECT 'customers' as table_name, COUNT(*) as empty_pk_count
FROM customers WHERE customer_id = '' OR customer_id IS NULL
UNION ALL
SELECT 'products', COUNT(*) FROM products WHERE product_id = '' OR product_id IS NULL
UNION ALL
SELECT 'orders', COUNT(*) FROM orders WHERE order_id = '' OR order_id IS NULL
UNION ALL
SELECT 'vendors', COUNT(*) FROM vendors WHERE vendor_id = '' OR vendor_id IS NULL;

-- ========================================
-- FIX 5: CLEAN UP ORPHANED RECORDS
-- ========================================

-- Remove order_items for non-existent products
DELETE FROM order_items 
WHERE product_id NOT IN (SELECT product_id FROM products);

-- Remove order_items for non-existent orders
DELETE FROM order_items 
WHERE order_id NOT IN (SELECT order_id FROM orders);

-- Remove orders for non-existent customers
DELETE FROM orders 
WHERE customer_id NOT IN (SELECT customer_id FROM customers);

-- ========================================
-- FIX 6: UPDATE REFERENCE DATA
-- ========================================

-- Ensure vendor categories are populated
UPDATE vendors v
LEFT JOIN vendor_categories vc ON v.category_code = vc.category_code
SET v.category_code = 'DISTRIBUTOR'
WHERE v.category_code IS NULL OR vc.category_code IS NULL;

-- ========================================
-- DIAGNOSTIC OUTPUT
-- ========================================

SELECT ''SUCCESS' Schema fixes applied!' as Status;

-- Show which tables have data
SELECT 
    'Data Summary' as report_type,
    (SELECT COUNT(*) FROM customers) as customers,
    (SELECT COUNT(*) FROM products) as products,
    (SELECT COUNT(*) FROM orders) as orders,
    (SELECT COUNT(*) FROM order_items) as order_items,
    (SELECT COUNT(*) FROM reviews) as reviews,
    (SELECT COUNT(*) FROM vendors) as vendors;

-- Show which lookup tables are populated
SELECT 
    'Lookup Tables' as report_type,
    (SELECT COUNT(*) FROM states) as states,
    (SELECT COUNT(*) FROM product_conditions) as conditions,
    (SELECT COUNT(*) FROM order_statuses) as order_statuses,
    (SELECT COUNT(*) FROM shipping_methods) as shipping_methods,
    (SELECT COUNT(*) FROM customer_segments) as customer_segments,
    (SELECT COUNT(*) FROM campaign_types) as campaign_types,
    (SELECT COUNT(*) FROM vendor_categories) as vendor_categories;