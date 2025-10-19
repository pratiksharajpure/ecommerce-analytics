-- ========================================
-- SEED DATA WITH INTENTIONAL ISSUES
-- For testing data quality, validation, and cleaning
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. DUPLICATE CUSTOMERS (50 records)
-- Customers with slightly different info but likely the same person
-- ========================================

SELECT 'Inserting duplicate customers...' AS Status;

-- Exact duplicates with different customer_id
INSERT INTO customers (first_name, last_name, email, phone, address_line1, city, state, zip_code, status, created_at)
SELECT 
    first_name,
    last_name,
    email,
    phone,
    address_line1,
    city,
    state,
    zip_code,
    status,
    DATE_ADD(created_at, INTERVAL 1 DAY) AS created_at
FROM customers
WHERE customer_id BETWEEN 1 AND 25
LIMIT 25;

-- Similar customers with slight variations (typos in name, missing data)
INSERT INTO customers (first_name, last_name, email, phone, address_line1, city, state, zip_code, status, created_at)
SELECT 
    CONCAT(first_name, ' ') AS first_name,  -- Extra space
    last_name,
    LOWER(email) AS email,  -- Different case
    REPLACE(phone, '555-', '555') AS phone,  -- Different format
    address_line1,
    city,
    state,
    zip_code,
    status,
    DATE_ADD(created_at, INTERVAL 2 DAY) AS created_at
FROM customers
WHERE customer_id BETWEEN 26 AND 50
LIMIT 25;

-- ========================================
-- 2. INVALID EMAILS (30 records)
-- Email addresses that don't follow standard format
-- ========================================

SELECT 'Inserting customers with invalid emails...' AS Status;

INSERT INTO customers (first_name, last_name, email, phone, address_line1, city, state, zip_code, status, created_at) VALUES
('John', 'InvalidEmail1', 'notanemail', '555-9001', '100 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-01'),
('Jane', 'InvalidEmail2', 'missing@domain', '555-9002', '101 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-02'),
('Bob', 'InvalidEmail3', '@nodomain.com', '555-9003', '102 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-03'),
('Alice', 'InvalidEmail4', 'double@@email.com', '555-9004', '103 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-04'),
('Charlie', 'InvalidEmail5', 'spaces in@email.com', '555-9005', '104 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-05'),
('David', 'InvalidEmail6', 'no.domain@', '555-9006', '105 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-06'),
('Eve', 'InvalidEmail7', '.startdot@email.com', '555-9007', '106 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-07'),
('Frank', 'InvalidEmail8', 'enddot.@email.com', '555-9008', '107 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-08'),
('Grace', 'InvalidEmail9', 'special!char@email.com', '555-9009', '108 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-09'),
('Henry', 'InvalidEmail10', 'nodomain', '555-9010', '109 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-10'),
('Ivy', 'InvalidEmail11', 'toolong' || REPEAT('x', 100) || '@email.com', '555-9011', '110 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-11'),
('Jack', 'InvalidEmail12', '', '555-9012', '111 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-12'),
('Kate', 'InvalidEmail13', NULL, '555-9013', '112 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-13'),
('Leo', 'InvalidEmail14', 'user@.com', '555-9014', '113 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-14'),
('Mary', 'InvalidEmail15', 'user@domain.', '555-9015', '114 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-15'),
('Nick', 'InvalidEmail16', 'user..double@email.com', '555-9016', '115 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-16'),
('Olivia', 'InvalidEmail17', 'user@domain..com', '555-9017', '116 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-17'),
('Paul', 'InvalidEmail18', 'user name@email.com', '555-9018', '117 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-18'),
('Quinn', 'InvalidEmail19', 'user@domain space.com', '555-9019', '118 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-19'),
('Rachel', 'InvalidEmail20', 'user#email.com', '555-9020', '119 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-20'),
('Sam', 'InvalidEmail21', 'user$email.com', '555-9021', '120 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-21'),
('Tina', 'InvalidEmail22', '@', '555-9022', '121 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-22'),
('Uma', 'InvalidEmail23', 'user@', '555-9023', '122 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-23'),
('Victor', 'InvalidEmail24', '@domain.com', '555-9024', '123 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-24'),
('Wendy', 'InvalidEmail25', 'user@@domain.com', '555-9025', '124 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-25'),
('Xander', 'InvalidEmail26', 'user@domain@com', '555-9026', '125 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-26'),
('Yara', 'InvalidEmail27', 'user@domain,com', '555-9027', '126 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-27'),
('Zack', 'InvalidEmail28', 'user@domain com', '555-9028', '127 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-28'),
('Amy', 'InvalidEmail29', 'user(at)domain.com', '555-9029', '128 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-29'),
('Ben', 'InvalidEmail30', 'user[at]domain.com', '555-9030', '129 Test St', 'Boston', 'MA', '02101', 'active', '2024-06-30');

-- ========================================
-- 3. MISSING PRODUCT DESCRIPTIONS (40 records)
-- Products without descriptions or with minimal info
-- ========================================

SELECT 'Inserting products with missing descriptions...' AS Status;

INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('BAD-001', 'Mystery Product 1', NULL, 10, 99.99, 50.00, 10, 'active', '2024-07-01'),
('BAD-002', 'Mystery Product 2', '', 11, 149.99, 75.00, 5, 'active', '2024-07-02'),
('BAD-003', 'Mystery Product 3', '   ', 12, 199.99, 100.00, 8, 'active', '2024-07-03'),
('BAD-004', 'Unknown Item', NULL, 15, 49.99, 25.00, 20, 'active', '2024-07-04'),
('BAD-005', 'Product', '', 16, 79.99, 40.00, 15, 'active', '2024-07-05'),
('BAD-006', 'Item', NULL, 17, 29.99, 15.00, 50, 'active', '2024-07-06'),
('BAD-007', 'Thing', '', 19, 399.99, 200.00, 3, 'active', '2024-07-07'),
('BAD-008', 'Stuff', NULL, 20, 599.99, 300.00, 2, 'active', '2024-07-08'),
('BAD-009', 'Widget', '', 23, 89.99, 45.00, 12, 'active', '2024-07-09'),
('BAD-010', 'Gadget', NULL, 24, 129.99, 65.00, 7, 'active', '2024-07-10'),
('BAD-011', 'Test Product A', 'N/A', 10, 99.99, 50.00, 10, 'active', '2024-07-11'),
('BAD-012', 'Test Product B', 'TBD', 11, 149.99, 75.00, 5, 'active', '2024-07-12'),
('BAD-013', 'Test Product C', 'Coming Soon', 12, 199.99, 100.00, 0, 'active', '2024-07-13'),
('BAD-014', 'Sample Item 1', 'See website', 15, 49.99, 25.00, 20, 'active', '2024-07-14'),
('BAD-015', 'Sample Item 2', 'No description', 16, 79.99, 40.00, 15, 'active', '2024-07-15'),
('BAD-016', 'Placeholder Product', 'Lorem ipsum', 17, 29.99, 15.00, 50, 'active', '2024-07-16'),
('BAD-017', 'Generic Product', 'Product', 19, 399.99, 200.00, 3, 'active', '2024-07-17'),
('BAD-018', 'Basic Item', 'Item', 20, 599.99, 300.00, 2, 'active', '2024-07-18'),
('BAD-019', 'Simple Thing', 'Thing', 23, 89.99, 45.00, 12, 'active', '2024-07-19'),
('BAD-020', 'Plain Product', '.', 24, 129.99, 65.00, 7, 'active', '2024-07-20'),
('BAD-021', 'Product X', '-', 10, 99.99, 50.00, 10, 'active', '2024-07-21'),
('BAD-022', 'Product Y', '...', 11, 149.99, 75.00, 5, 'active', '2024-07-22'),
('BAD-023', 'Product Z', 'xyz', 12, 199.99, 100.00, 8, 'active', '2024-07-23'),
('BAD-024', 'Unnamed Product', 'TODO: Add description', 15, 49.99, 25.00, 20, 'active', '2024-07-24'),
('BAD-025', 'New Product', 'Under development', 16, 79.99, 40.00, 0, 'active', '2024-07-25'),
('BAD-026', 'Old Product', 'Discontinued', 17, 29.99, 15.00, 0, 'discontinued', '2024-07-26'),
('BAD-027', 'Random Item 1', NULL, 19, 399.99, 200.00, 3, 'active', '2024-07-27'),
('BAD-028', 'Random Item 2', '', 20, 599.99, 300.00, 2, 'active', '2024-07-28'),
('BAD-029', 'Random Item 3', NULL, 23, 89.99, 45.00, 12, 'active', '2024-07-29'),
('BAD-030', 'Random Item 4', '', 24, 129.99, 65.00, 7, 'active', '2024-07-30'),
('BAD-031', 'Misc Product 1', NULL, 10, 99.99, 50.00, 10, 'active', '2024-07-31'),
('BAD-032', 'Misc Product 2', '', 11, 149.99, 75.00, 5, 'active', '2024-08-01'),
('BAD-033', 'Misc Product 3', NULL, 12, 199.99, 100.00, 8, 'active', '2024-08-02'),
('BAD-034', 'Misc Product 4', '', 15, 49.99, 25.00, 20, 'active', '2024-08-03'),
('BAD-035', 'Misc Product 5', NULL, 16, 79.99, 40.00, 15, 'active', '2024-08-04'),
('BAD-036', 'Misc Product 6', '', 17, 29.99, 15.00, 50, 'active', '2024-08-05'),
('BAD-037', 'Misc Product 7', NULL, 19, 399.99, 200.00, 3, 'active', '2024-08-06'),
('BAD-038', 'Misc Product 8', '', 20, 599.99, 300.00, 2, 'active', '2024-08-07'),
('BAD-039', 'Misc Product 9', NULL, 23, 89.99, 45.00, 12, 'active', '2024-08-08'),
('BAD-040', 'Misc Product 10', '', 24, 129.99, 65.00, 7, 'active', '2024-08-09');

-- ========================================
-- 4. ZERO OR NEGATIVE PRICES (20 records)
-- Products with invalid pricing
-- ========================================

SELECT 'Inserting products with invalid prices...' AS Status;

INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('PRICE-001', 'Free Product', 'This should not be free', 10, 0.00, 50.00, 10, 'active', '2024-08-10'),
('PRICE-002', 'Negative Price Item', 'Price is negative', 11, -10.00, 75.00, 5, 'active', '2024-08-11'),
('PRICE-003', 'Zero Cost Product', 'Cost is zero but price is set', 12, 100.00, 0.00, 8, 'active', '2024-08-12'),
('PRICE-004', 'Both Zero', 'Both price and cost are zero', 15, 0.00, 0.00, 20, 'active', '2024-08-13'),
('PRICE-005', 'Negative Cost', 'Cost should not be negative', 16, 50.00, -25.00, 15, 'active', '2024-08-14'),
('PRICE-006', 'Loss Leader', 'Price is less than cost (big loss)', 17, 10.00, 100.00, 50, 'active', '2024-08-15'),
('PRICE-007', 'Penny Product', 'Unrealistically low price', 19, 0.01, 50.00, 3, 'active', '2024-08-16'),
('PRICE-008', 'Negative Both', 'Both negative', 20, -50.00, -25.00, 2, 'active', '2024-08-17'),
('PRICE-009', 'Free With Cost', 'Free but has cost', 23, 0.00, 100.00, 12, 'active', '2024-08-18'),
('PRICE-010', 'Huge Negative', 'Very negative price', 24, -999.99, 50.00, 7, 'active', '2024-08-19'),
('PRICE-011', 'Zero Price Premium', 'Premium product for free', 10, 0.00, 500.00, 1, 'active', '2024-08-20'),
('PRICE-012', 'Negative Electronics', 'Electronics with negative price', 11, -149.99, 200.00, 3, 'active', '2024-08-21'),
('PRICE-013', 'Backwards Pricing 1', 'Cost more than double price', 12, 50.00, 150.00, 10, 'active', '2024-08-22'),
('PRICE-014', 'Backwards Pricing 2', 'Cost 10x the price', 15, 10.00, 100.00, 25, 'active', '2024-08-23'),
('PRICE-015', 'NULL Price Product', 'Price is NULL', 16, NULL, 40.00, 15, 'active', '2024-08-24'),
('PRICE-016', 'NULL Cost Product', 'Cost is NULL', 17, 99.99, NULL, 50, 'active', '2024-08-25'),
('PRICE-017', 'Both NULL', 'Both price and cost are NULL', 19, NULL, NULL, 3, 'active', '2024-08-26'),
('PRICE-018', 'Unrealistic High Price', 'Price is unrealistically high', 20, 999999.99, 10.00, 1, 'active', '2024-08-27'),
('PRICE-019', 'Penny Cost', 'Cost is one penny', 23, 100.00, 0.01, 12, 'active', '2024-08-28'),
('PRICE-020', 'Mixed Up Values', 'Price and cost might be swapped', 24, 25.00, 150.00, 7, 'active', '2024-08-29');

-- ========================================
-- 5. ORPHANED ORDERS (15 records)
-- Orders referencing non-existent customers
-- ========================================

SELECT 'Inserting orphaned orders...' AS Status;

-- Temporarily disable foreign key checks to insert orphaned records
SET FOREIGN_KEY_CHECKS = 0;

INSERT INTO orders (customer_id, order_date, total_amount, status, payment_status, shipping_cost, tax_amount, created_at) VALUES
(99999, '2024-08-01', 150.00, 'pending', 'pending', 5.99, 12.00, '2024-08-01'),
(99998, '2024-08-02', 250.00, 'processing', 'paid', 9.99, 20.00, '2024-08-02'),
(99997, '2024-08-03', 350.00, 'shipped', 'paid', 15.99, 28.00, '2024-08-03'),
(99996, '2024-08-04', 450.00, 'delivered', 'paid', 0.00, 36.00, '2024-08-04'),
(99995, '2024-08-05', 550.00, 'cancelled', 'refunded', 5.99, 44.00, '2024-08-05'),
(99994, '2024-08-06', 75.00, 'pending', 'pending', 5.99, 6.00, '2024-08-06'),
(99993, '2024-08-07', 125.00, 'processing', 'paid', 9.99, 10.00, '2024-08-07'),
(99992, '2024-08-08', 200.00, 'shipped', 'paid', 15.99, 16.00, '2024-08-08'),
(99991, '2024-08-09', 300.00, 'delivered', 'paid', 0.00, 24.00, '2024-08-09'),
(99990, '2024-08-10', 400.00, 'cancelled', 'failed', 5.99, 32.00, '2024-08-10'),
(88888, '2024-08-11', 100.00, 'pending', 'pending', 5.99, 8.00, '2024-08-11'),
(77777, '2024-08-12', 175.00, 'processing', 'paid', 9.99, 14.00, '2024-08-12'),
(66666, '2024-08-13', 225.00, 'shipped', 'paid', 15.99, 18.00, '2024-08-13'),
(55555, '2024-08-14', 325.00, 'delivered', 'paid', 0.00, 26.00, '2024-08-14'),
(44444, '2024-08-15', 500.00, 'cancelled', 'refunded', 5.99, 40.00, '2024-08-15');

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

-- ========================================
-- 6. INVENTORY MISMATCHES (25 records)
-- Products with inventory issues
-- ========================================

SELECT 'Creating inventory mismatches...' AS Status;

-- Products with negative inventory
UPDATE inventory 
SET quantity_on_hand = -10
WHERE product_id IN (SELECT product_id FROM products ORDER BY RAND() LIMIT 5);

-- Products with reserved quantity > on_hand
UPDATE inventory 
SET quantity_reserved = quantity_on_hand + 50
WHERE product_id IN (SELECT product_id FROM products ORDER BY RAND() LIMIT 5);

-- Products with stock_quantity not matching inventory totals
UPDATE products p
SET stock_quantity = 1000
WHERE product_id IN (
    SELECT product_id 
    FROM inventory 
    GROUP BY product_id 
    HAVING SUM(quantity_on_hand) < 100
    LIMIT 5
);

-- Products marked as out_of_stock but have inventory
UPDATE products 
SET status = 'out_of_stock'
WHERE product_id IN (
    SELECT DISTINCT product_id 
    FROM inventory 
    WHERE quantity_on_hand > 50
    LIMIT 5
);

-- Products marked as active but zero inventory
UPDATE inventory 
SET quantity_on_hand = 0,
    quantity_reserved = 0
WHERE product_id IN (
    SELECT product_id 
    FROM products 
    WHERE status = 'active' 
    AND stock_quantity > 100
    LIMIT 5
);

-- ========================================
-- DISPLAY SUMMARY OF ISSUES
-- ========================================

SELECT '====== DATA QUALITY ISSUES SUMMARY ======' AS '';

SELECT 'Duplicate Customers:' AS Issue, 
       COUNT(*) AS Count
FROM (
    SELECT email, COUNT(*) as cnt
    FROM customers
    GROUP BY email
    HAVING cnt > 1
) AS dupes;

SELECT 'Invalid Emails:' AS Issue,
       COUNT(*) AS Count
FROM customers
WHERE email NOT REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'
   OR email IS NULL
   OR email = '';

SELECT 'Missing Product Descriptions:' AS Issue,
       COUNT(*) AS Count
FROM products
WHERE description IS NULL 
   OR TRIM(description) = '' 
   OR LENGTH(TRIM(description)) < 10;

SELECT 'Products with Zero/Negative Prices:' AS Issue,
       COUNT(*) AS Count
FROM products
WHERE price <= 0 OR cost <= 0 OR price IS NULL OR cost IS NULL;

SELECT 'Products with Price < Cost (Loss):' AS Issue,
       COUNT(*) AS Count
FROM products
WHERE price < cost AND price > 0 AND cost > 0;

SELECT 'Orphaned Orders (Invalid Customer):' AS Issue,
       COUNT(*) AS Count
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;

SELECT 'Negative Inventory:' AS Issue,
       COUNT(*) AS Count
FROM inventory
WHERE quantity_on_hand < 0;

SELECT 'Reserved > On Hand:' AS Issue,
       COUNT(*) AS Count
FROM inventory
WHERE quantity_reserved > quantity_on_hand;

SELECT 'Stock Quantity Mismatches:' AS Issue,
       COUNT(*) AS Count
FROM products p
JOIN (
    SELECT product_id, SUM(quantity_on_hand) as total_inventory
    FROM inventory
    GROUP BY product_id
) i ON p.product_id = i.product_id
WHERE p.stock_quantity != i.total_inventory;

SELECT 'Out of Stock Products with Inventory:' AS Issue,
       COUNT(*) AS Count
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.status = 'out_of_stock' 
  AND i.quantity_on_hand > 0;

SELECT '====== END OF SUMMARY ======' AS '';