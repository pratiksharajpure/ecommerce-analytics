-- ========================================
-- PURGE DUPLICATES & DATA CLEANUP SCRIPT
-- E-commerce Revenue Analytics Engine
-- Remove duplicates, merge records, clean data
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- SAFETY: CREATE BACKUP TABLES
-- ========================================
SELECT '========== CREATING BACKUP TABLES ==========' AS '';

-- Backup customers before cleanup
DROP TABLE IF EXISTS customers_backup;
CREATE TABLE customers_backup AS SELECT * FROM customers;
SELECT 'Customers backed up' AS Status, COUNT(*) AS Records FROM customers_backup;

-- Backup products before cleanup
DROP TABLE IF EXISTS products_backup;
CREATE TABLE products_backup AS SELECT * FROM products;
SELECT 'Products backed up' AS Status, COUNT(*) AS Records FROM products_backup;

-- Backup orders before cleanup
DROP TABLE IF EXISTS orders_backup;
CREATE TABLE orders_backup AS SELECT * FROM orders;
SELECT 'Orders backed up' AS Status, COUNT(*) AS Records FROM orders_backup;

-- ========================================
-- SECTION 1: IDENTIFY DUPLICATE CUSTOMERS
-- ========================================
SELECT '========== IDENTIFYING DUPLICATE CUSTOMERS ==========' AS '';

-- Find duplicate customers by email
DROP TEMPORARY TABLE IF EXISTS duplicate_customer_emails;
CREATE TEMPORARY TABLE duplicate_customer_emails AS
SELECT 
    email,
    COUNT(*) AS duplicate_count,
    MIN(customer_id) AS keep_customer_id,
    GROUP_CONCAT(customer_id ORDER BY customer_id) AS all_customer_ids,
    GROUP_CONCAT(CONCAT(first_name, ' ', last_name) ORDER BY customer_id SEPARATOR ' | ') AS all_names
FROM customers
WHERE email IS NOT NULL AND email != ''
GROUP BY email
HAVING COUNT(*) > 1;

SELECT 
    'Duplicate Customer Emails Found' AS Check_Type,
    COUNT(*) AS Duplicate_Email_Count,
    SUM(duplicate_count) AS Total_Duplicate_Records
FROM duplicate_customer_emails;

-- Display duplicate customer details
SELECT 
    email,
    duplicate_count AS Duplicates,
    keep_customer_id AS Keep_ID,
    all_customer_ids AS All_IDs,
    all_names AS All_Names
FROM duplicate_customer_emails
ORDER BY duplicate_count DESC;

-- ========================================
-- SECTION 2: MERGE DUPLICATE CUSTOMERS
-- ========================================
SELECT '========== MERGING DUPLICATE CUSTOMERS ==========' AS '';

-- Create temp table with customer IDs to merge and keep
DROP TEMPORARY TABLE IF EXISTS customer_merge_map;
CREATE TEMPORARY TABLE customer_merge_map AS
SELECT 
    c.customer_id AS old_customer_id,
    d.keep_customer_id AS new_customer_id
FROM customers c
JOIN duplicate_customer_emails d ON c.email = d.email
WHERE c.customer_id != d.keep_customer_id;

SELECT 
    'Customer Merge Map Created' AS Status,
    COUNT(*) AS Records_To_Merge
FROM customer_merge_map;

-- Update orders to point to kept customer
UPDATE orders o
JOIN customer_merge_map m ON o.customer_id = m.old_customer_id
SET o.customer_id = m.new_customer_id;

SELECT 'Orders updated with merged customer IDs' AS Status;

-- Update shipping addresses
UPDATE shipping_addresses sa
JOIN customer_merge_map m ON sa.customer_id = m.old_customer_id
SET sa.customer_id = m.new_customer_id;

SELECT 'Shipping addresses updated with merged customer IDs' AS Status;

-- Update payment methods
UPDATE payment_methods pm
JOIN customer_merge_map m ON pm.customer_id = m.old_customer_id
SET pm.customer_id = m.new_customer_id;

SELECT 'Payment methods updated with merged customer IDs' AS Status;

-- Update reviews
UPDATE reviews r
JOIN customer_merge_map m ON r.customer_id = m.old_customer_id
SET r.customer_id = m.new_customer_id;

SELECT 'Reviews updated with merged customer IDs' AS Status;

-- Update loyalty program
UPDATE loyalty_program lp
JOIN customer_merge_map m ON lp.customer_id = m.old_customer_id
SET lp.customer_id = m.new_customer_id
ON DUPLICATE KEY UPDATE
    points_balance = points_balance + VALUES(points_balance),
    points_earned_lifetime = points_earned_lifetime + VALUES(points_earned_lifetime),
    points_redeemed_lifetime = points_redeemed_lifetime + VALUES(points_redeemed_lifetime);

SELECT 'Loyalty program updated with merged customer IDs' AS Status;

-- Delete duplicate customer records
DELETE c FROM customers c
JOIN customer_merge_map m ON c.customer_id = m.old_customer_id;

SELECT 
    'Duplicate Customers Deleted' AS Status,
    ROW_COUNT() AS Records_Deleted;

-- ========================================
-- SECTION 3: IDENTIFY DUPLICATE PRODUCTS
-- ========================================
SELECT '========== IDENTIFYING DUPLICATE PRODUCTS ==========' AS '';

-- Find duplicate products by SKU
DROP TEMPORARY TABLE IF EXISTS duplicate_product_skus;
CREATE TEMPORARY TABLE duplicate_product_skus AS
SELECT 
    sku,
    COUNT(*) AS duplicate_count,
    MIN(product_id) AS keep_product_id,
    GROUP_CONCAT(product_id ORDER BY product_id) AS all_product_ids,
    GROUP_CONCAT(product_name ORDER BY product_id SEPARATOR ' | ') AS all_names
FROM products
WHERE sku IS NOT NULL AND sku != ''
GROUP BY sku
HAVING COUNT(*) > 1;

SELECT 
    'Duplicate Product SKUs Found' AS Check_Type,
    COUNT(*) AS Duplicate_SKU_Count,
    SUM(duplicate_count) AS Total_Duplicate_Records
FROM duplicate_product_skus;

-- Display duplicate product details
SELECT 
    sku,
    duplicate_count AS Duplicates,
    keep_product_id AS Keep_ID,
    all_product_ids AS All_IDs,
    all_names AS All_Names
FROM duplicate_product_skus
ORDER BY duplicate_count DESC;

-- Find duplicate products by exact name match
DROP TEMPORARY TABLE IF EXISTS duplicate_product_names;
CREATE TEMPORARY TABLE duplicate_product_names AS
SELECT 
    product_name,
    COUNT(*) AS duplicate_count,
    MIN(product_id) AS keep_product_id,
    GROUP_CONCAT(product_id ORDER BY product_id) AS all_product_ids,
    GROUP_CONCAT(sku ORDER BY product_id SEPARATOR ' | ') AS all_skus
FROM products
WHERE product_name IS NOT NULL
GROUP BY product_name
HAVING COUNT(*) > 1;

SELECT 
    'Duplicate Product Names Found' AS Check_Type,
    COUNT(*) AS Duplicate_Name_Count
FROM duplicate_product_names;

-- ========================================
-- SECTION 4: MERGE DUPLICATE PRODUCTS
-- ========================================
SELECT '========== MERGING DUPLICATE PRODUCTS ==========' AS '';

-- Create temp table with product IDs to merge (based on SKU)
DROP TEMPORARY TABLE IF EXISTS product_merge_map;
CREATE TEMPORARY TABLE product_merge_map AS
SELECT 
    p.product_id AS old_product_id,
    d.keep_product_id AS new_product_id
FROM products p
JOIN duplicate_product_skus d ON p.sku = d.sku
WHERE p.product_id != d.keep_product_id;

SELECT 
    'Product Merge Map Created' AS Status,
    COUNT(*) AS Records_To_Merge
FROM product_merge_map;

-- Update order items to point to kept product
UPDATE order_items oi
JOIN product_merge_map m ON oi.product_id = m.old_product_id
SET oi.product_id = m.new_product_id;

SELECT 'Order items updated with merged product IDs' AS Status;

-- Merge inventory data (sum quantities)
INSERT INTO inventory (product_id, warehouse_id, quantity_on_hand, quantity_reserved, reorder_level)
SELECT 
    m.new_product_id,
    i.warehouse_id,
    i.quantity_on_hand,
    i.quantity_reserved,
    i.reorder_level
FROM inventory i
JOIN product_merge_map m ON i.product_id = m.old_product_id
ON DUPLICATE KEY UPDATE
    quantity_on_hand = quantity_on_hand + VALUES(quantity_on_hand),
    quantity_reserved = quantity_reserved + VALUES(quantity_reserved);

-- Delete old inventory records
DELETE i FROM inventory i
JOIN product_merge_map m ON i.product_id = m.old_product_id;

SELECT 'Inventory merged and updated' AS Status;

-- Update vendor contracts
UPDATE vendor_contracts vc
JOIN product_merge_map m ON vc.product_id = m.old_product_id
SET vc.product_id = m.new_product_id;

SELECT 'Vendor contracts updated with merged product IDs' AS Status;

-- Update reviews
UPDATE reviews r
JOIN product_merge_map m ON r.product_id = m.old_product_id
SET r.product_id = m.new_product_id;

SELECT 'Reviews updated with merged product IDs' AS Status;

-- Delete duplicate product records
DELETE p FROM products p
JOIN product_merge_map m ON p.product_id = m.old_product_id;

SELECT 
    'Duplicate Products Deleted' AS Status,
    ROW_COUNT() AS Records_Deleted;

-- ========================================
-- SECTION 5: CLEAN DUPLICATE ORDER ITEMS
-- ========================================
SELECT '========== CLEANING DUPLICATE ORDER ITEMS ==========' AS '';

-- Find and remove exact duplicate order items (same order_id and product_id)
DROP TEMPORARY TABLE IF EXISTS duplicate_order_items;
CREATE TEMPORARY TABLE duplicate_order_items AS
SELECT 
    order_id,
    product_id,
    COUNT(*) AS duplicate_count,
    MIN(order_item_id) AS keep_order_item_id,
    SUM(quantity) AS total_quantity,
    AVG(unit_price) AS avg_unit_price,
    SUM(discount) AS total_discount
FROM order_items
GROUP BY order_id, product_id
HAVING COUNT(*) > 1;

SELECT 
    'Duplicate Order Items Found' AS Check_Type,
    COUNT(*) AS Duplicate_Combinations,
    SUM(duplicate_count) AS Total_Duplicate_Records
FROM duplicate_order_items;

-- Update the kept order item with consolidated data
UPDATE order_items oi
JOIN duplicate_order_items doi ON oi.order_item_id = doi.keep_order_item_id
SET 
    oi.quantity = doi.total_quantity,
    oi.unit_price = doi.avg_unit_price,
    oi.discount = doi.total_discount;

SELECT 'Consolidated order items updated' AS Status;

-- Delete duplicate order items (keep only the first one per order+product)
DELETE oi FROM order_items oi
JOIN duplicate_order_items doi ON oi.order_id = doi.order_id AND oi.product_id = doi.product_id
WHERE oi.order_item_id != doi.keep_order_item_id;

SELECT 
    'Duplicate Order Items Deleted' AS Status,
    ROW_COUNT() AS Records_Deleted;

-- ========================================
-- SECTION 6: CLEAN DUPLICATE REVIEWS
-- ========================================
SELECT '========== CLEANING DUPLICATE REVIEWS ==========' AS '';

-- Find duplicate reviews (same customer and product)
DROP TEMPORARY TABLE IF EXISTS duplicate_reviews;
CREATE TEMPORARY TABLE duplicate_reviews AS
SELECT 
    customer_id,
    product_id,
    COUNT(*) AS duplicate_count,
    MAX(review_id) AS keep_review_id,
    AVG(rating) AS avg_rating
FROM reviews
WHERE customer_id IS NOT NULL
GROUP BY customer_id, product_id
HAVING COUNT(*) > 1;

SELECT 
    'Duplicate Reviews Found' AS Check_Type,
    COUNT(*) AS Duplicate_Combinations,
    SUM(duplicate_count) AS Total_Duplicate_Records
FROM duplicate_reviews;

-- Keep the most recent review per customer+product
DELETE r FROM reviews r
JOIN duplicate_reviews dr ON r.customer_id = dr.customer_id AND r.product_id = dr.product_id
WHERE r.review_id != dr.keep_review_id;

SELECT 
    'Duplicate Reviews Deleted' AS Status,
    ROW_COUNT() AS Records_Deleted;

-- ========================================
-- SECTION 7: CLEAN DUPLICATE SHIPPING ADDRESSES
-- ========================================
SELECT '========== CLEANING DUPLICATE SHIPPING ADDRESSES ==========' AS '';

-- Find duplicate shipping addresses (exact match on all fields)
DROP TEMPORARY TABLE IF EXISTS duplicate_addresses;
CREATE TEMPORARY TABLE duplicate_addresses AS
SELECT 
    customer_id,
    address_line1,
    city,
    state,
    zip_code,
    COUNT(*) AS duplicate_count,
    MIN(address_id) AS keep_address_id
FROM shipping_addresses
GROUP BY customer_id, address_line1, city, state, zip_code
HAVING COUNT(*) > 1;

SELECT 
    'Duplicate Shipping Addresses Found' AS Check_Type,
    COUNT(*) AS Duplicate_Combinations,
    SUM(duplicate_count) AS Total_Duplicate_Records
FROM duplicate_addresses;

-- Delete duplicate addresses
DELETE sa FROM shipping_addresses sa
JOIN duplicate_addresses da ON 
    sa.customer_id = da.customer_id 
    AND sa.address_line1 = da.address_line1
    AND sa.city = da.city
    AND sa.state = da.state
    AND sa.zip_code = da.zip_code
WHERE sa.address_id != da.keep_address_id;

SELECT 
    'Duplicate Shipping Addresses Deleted' AS Status,
    ROW_COUNT() AS Records_Deleted;

-- ========================================
-- SECTION 8: CLEAN DUPLICATE PAYMENT METHODS
-- ========================================
SELECT '========== CLEANING DUPLICATE PAYMENT METHODS ==========' AS '';

-- Find duplicate payment methods (same card details)
DROP TEMPORARY TABLE IF EXISTS duplicate_payments;
CREATE TEMPORARY TABLE duplicate_payments AS
SELECT 
    customer_id,
    payment_type,
    card_last_four,
    card_brand,
    COUNT(*) AS duplicate_count,
    MIN(payment_method_id) AS keep_payment_id
FROM payment_methods
WHERE card_last_four IS NOT NULL
GROUP BY customer_id, payment_type, card_last_four, card_brand
HAVING COUNT(*) > 1;

SELECT 
    'Duplicate Payment Methods Found' AS Check_Type,
    COUNT(*) AS Duplicate_Combinations,
    SUM(duplicate_count) AS Total_Duplicate_Records
FROM duplicate_payments;

-- Delete duplicate payment methods
DELETE pm FROM payment_methods pm
JOIN duplicate_payments dp ON 
    pm.customer_id = dp.customer_id 
    AND pm.payment_type = dp.payment_type
    AND pm.card_last_four = dp.card_last_four
    AND COALESCE(pm.card_brand, '') = COALESCE(dp.card_brand, '')
WHERE pm.payment_method_id != dp.keep_payment_id;

SELECT 
    'Duplicate Payment Methods Deleted' AS Status,
    ROW_COUNT() AS Records_Deleted;

-- ========================================
-- SECTION 9: CLEAN DUPLICATE VENDOR CONTRACTS
-- ========================================
SELECT '========== CLEANING DUPLICATE VENDOR CONTRACTS ==========' AS '';

-- Find overlapping active contracts for same vendor+product
DROP TEMPORARY TABLE IF EXISTS duplicate_contracts;
CREATE TEMPORARY TABLE duplicate_contracts AS
SELECT 
    vendor_id,
    product_id,
    COUNT(*) AS duplicate_count,
    MAX(contract_id) AS keep_contract_id,
    MIN(cost_per_unit) AS best_cost
FROM vendor_contracts
WHERE status = 'active'
GROUP BY vendor_id, product_id
HAVING COUNT(*) > 1;

SELECT 
    'Duplicate Active Vendor Contracts Found' AS Check_Type,
    COUNT(*) AS Duplicate_Combinations,
    SUM(duplicate_count) AS Total_Duplicate_Records
FROM duplicate_contracts;

-- Keep the contract with the best (lowest) cost, mark others as terminated
UPDATE vendor_contracts vc
JOIN duplicate_contracts dc ON vc.vendor_id = dc.vendor_id AND vc.product_id = dc.product_id
SET vc.status = 'terminated'
WHERE vc.contract_id != dc.keep_contract_id AND vc.status = 'active';

SELECT 
    'Duplicate Vendor Contracts Terminated' AS Status,
    ROW_COUNT() AS Records_Updated;

-- ========================================
-- SECTION 10: CLEAN NULL/EMPTY VALUES
-- ========================================
SELECT '========== CLEANING NULL/EMPTY VALUES ==========' AS '';

-- Remove customers with NULL email and no orders
DELETE c FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE (c.email IS NULL OR c.email = '')
AND o.order_id IS NULL;

SELECT 
    'Customers With NULL Email (No Orders) Deleted' AS Status,
    ROW_COUNT() AS Records_Deleted;

-- Remove products with NULL SKU and never ordered
DELETE p FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE (p.sku IS NULL OR p.sku = '')
AND oi.order_item_id IS NULL;

SELECT 
    'Products With NULL SKU (Never Ordered) Deleted' AS Status,
    ROW_COUNT() AS Records_Deleted;

-- ========================================
-- SECTION 11: REVALIDATE DATA INTEGRITY
-- ========================================
SELECT '========== REVALIDATING DATA INTEGRITY ==========' AS '';

-- Check for remaining duplicates
SELECT 
    'Remaining Duplicate Customer Emails' AS Check_Type,
    COUNT(*) AS Count
FROM (
    SELECT email
    FROM customers
    WHERE email IS NOT NULL
    GROUP BY email
    HAVING COUNT(*) > 1
) AS remaining_dupes;

SELECT 
    'Remaining Duplicate Product SKUs' AS Check_Type,
    COUNT(*) AS Count
FROM (
    SELECT sku
    FROM products
    WHERE sku IS NOT NULL
    GROUP BY sku
    HAVING COUNT(*) > 1
) AS remaining_dupes;

-- ========================================
-- SECTION 12: CLEANUP SUMMARY
-- ========================================
SELECT '========== CLEANUP SUMMARY ==========' AS '';

SELECT 
    'Customers After Cleanup' AS Metric,
    COUNT(*) AS Current_Count,
    (SELECT COUNT(*) FROM customers_backup) AS Before_Count,
    (SELECT COUNT(*) FROM customers_backup) - COUNT(*) AS Removed
FROM customers;

SELECT 
    'Products After Cleanup' AS Metric,
    COUNT(*) AS Current_Count,
    (SELECT COUNT(*) FROM products_backup) AS Before_Count,
    (SELECT COUNT(*) FROM products_backup) - COUNT(*) AS Removed
FROM products;

SELECT 
    'Orders After Cleanup' AS Metric,
    COUNT(*) AS Current_Count,
    (SELECT COUNT(*) FROM orders_backup) AS Before_Count,
    (SELECT COUNT(*) FROM orders_backup) - COUNT(*) AS Removed
FROM orders;

-- ========================================
-- FINAL MESSAGE
-- ========================================
SELECT '========================================' AS '';
SELECT 'Duplicate Purge Completed Successfully' AS Result;
SELECT 'Backup tables retained: customers_backup, products_backup, orders_backup' AS Note;
SELECT 'Review summary above and test application thoroughly' AS Recommendation;
SELECT '========================================' AS '';