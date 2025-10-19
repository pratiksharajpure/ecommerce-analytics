-- ========================================
-- INSERT SAMPLE ORDERS & ORDER ITEMS
-- 1000 realistic order records with line items
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- ORDERS (1000 records)
-- Distributed across 2024 with realistic patterns
-- ========================================

-- Generate orders using customer IDs 1-500 and dates throughout 2024
INSERT INTO orders (customer_id, order_date, total_amount, status, payment_status, shipping_cost, tax_amount, created_at)
SELECT 
    -- Unit price from products table (will be updated with actual prices)
    0.00 AS unit_price,
    -- Discount: 90% no discount, 10% have discount
    CASE 
        WHEN RAND() < 0.9 THEN 0.00
        ELSE ROUND(5 + (RAND() * 45), 2)
    END AS discount
FROM orders o;

-- Update unit prices from products table for first items
UPDATE order_items oi
JOIN products p ON oi.product_id = p.product_id
SET oi.unit_price = p.price
WHERE oi.order_item_id <= 1000;

-- Second item for 70% of orders
INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount)
SELECT 
    o.order_id,
    FLOOR(1 + (RAND() * 300)) AS product_id,
    CASE 
        WHEN RAND() < 0.6 THEN 1
        WHEN RAND() < 0.9 THEN 2
        ELSE FLOOR(1 + (RAND() * 3))
    END AS quantity,
    0.00 AS unit_price,
    CASE 
        WHEN RAND() < 0.92 THEN 0.00
        ELSE ROUND(5 + (RAND() * 35), 2)
    END AS discount
FROM orders o
WHERE RAND() < 0.7;

-- Update unit prices for second items
UPDATE order_items oi
JOIN products p ON oi.product_id = p.product_id
SET oi.unit_price = p.price
WHERE oi.order_item_id > 1000 AND oi.unit_price = 0;

-- Third item for 30% of orders
INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount)
SELECT 
    o.order_id,
    FLOOR(1 + (RAND() * 300)) AS product_id,
    CASE 
        WHEN RAND() < 0.7 THEN 1
        ELSE 2
    END AS quantity,
    0.00 AS unit_price,
    CASE 
        WHEN RAND() < 0.95 THEN 0.00
        ELSE ROUND(5 + (RAND() * 25), 2)
    END AS discount
FROM orders o
WHERE RAND() < 0.3;

-- Update unit prices for third items
UPDATE order_items oi
JOIN products p ON oi.product_id = p.product_id
SET oi.unit_price = p.price
WHERE oi.unit_price = 0;

-- ========================================
-- UPDATE ORDER TOTALS
-- Calculate total_amount and tax_amount based on order items
-- ========================================

-- Update tax_amount (8% of subtotal)
UPDATE orders o
SET tax_amount = ROUND((
    SELECT SUM(oi.subtotal) * 0.08
    FROM order_items oi
    WHERE oi.order_id = o.order_id
), 2);

-- Update total_amount (subtotal + tax + shipping)
UPDATE orders o
SET total_amount = ROUND((
    SELECT COALESCE(SUM(oi.subtotal), 0)
    FROM order_items oi
    WHERE oi.order_id = o.order_id
) + o.tax_amount + o.shipping_cost, 2);

-- Handle any orders with NULL amounts
UPDATE orders 
SET total_amount = shipping_cost + tax_amount 
WHERE total_amount IS NULL OR total_amount = 0;

-- ========================================
-- REALISTIC DATE ADJUSTMENTS
-- Make order patterns more realistic (more recent orders, seasonal peaks)
-- ========================================

-- Add more orders in Q4 (holiday shopping season)
UPDATE orders 
SET order_date = DATE_ADD('2024-10-01', INTERVAL FLOOR(RAND() * 92) DAY)
WHERE order_id % 5 = 0 AND MONTH(order_date) < 10;

-- Add more orders on weekends
UPDATE orders 
SET order_date = DATE_ADD(order_date, INTERVAL ((RAND() * 2) - 1) DAY)
WHERE DAYOFWEEK(order_date) IN (1, 7);

-- Recent orders are more likely to be pending/processing
UPDATE orders 
SET status = 'processing', payment_status = 'paid'
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
AND status NOT IN ('delivered', 'cancelled');

UPDATE orders 
SET status = 'pending', payment_status = 'pending'
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 3 DAY)
AND status NOT IN ('cancelled');

-- Older orders are more likely delivered
UPDATE orders 
SET status = 'delivered', payment_status = 'paid'
WHERE order_date < DATE_SUB(CURDATE(), INTERVAL 30 DAY)
AND status NOT IN ('cancelled', 'processing');

-- Match updated_at with status changes
UPDATE orders 
SET updated_at = CASE 
    WHEN status = 'delivered' THEN DATE_ADD(order_date, INTERVAL 5 + FLOOR(RAND() * 10) DAY)
    WHEN status = 'shipped' THEN DATE_ADD(order_date, INTERVAL 2 + FLOOR(RAND() * 3) DAY)
    WHEN status = 'processing' THEN DATE_ADD(order_date, INTERVAL FLOOR(RAND() * 2) DAY)
    ELSE order_date
END;

-- ========================================
-- SHIPPING ADDRESSES
-- Create shipping addresses for orders
-- ========================================

INSERT INTO shipping_addresses (customer_id, address_label, address_line1, city, state, zip_code, is_default)
SELECT DISTINCT
    o.customer_id,
    'Home' AS address_label,
    c.address_line1,
    c.city,
    c.state,
    c.zip_code,
    TRUE AS is_default
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE NOT EXISTS (
    SELECT 1 FROM shipping_addresses sa 
    WHERE sa.customer_id = o.customer_id
)
LIMIT 500;

-- Add work addresses for 30% of customers
INSERT INTO shipping_addresses (customer_id, address_label, address_line1, city, state, zip_code, is_default)
SELECT 
    c.customer_id,
    'Work' AS address_label,
    CONCAT(c.customer_id * 100, ' Office Park Dr') AS address_line1,
    c.city,
    c.state,
    c.zip_code,
    FALSE AS is_default
FROM customers c
WHERE c.customer_id % 3 = 0
AND c.customer_id <= 150;

-- ========================================
-- PAYMENT METHODS
-- Create payment methods for customers who have ordered
-- ========================================

INSERT INTO payment_methods (customer_id, payment_type, card_last_four, card_brand, expiry_month, expiry_year, is_default)
SELECT DISTINCT
    o.customer_id,
    ELT(MOD(o.customer_id, 4) + 1, 'credit_card', 'credit_card', 'debit_card', 'paypal') AS payment_type,
    CASE 
        WHEN MOD(o.customer_id, 4) = 3 THEN NULL
        ELSE LPAD(FLOOR(1000 + (RAND() * 9000)), 4, '0')
    END AS card_last_four,
    CASE 
        WHEN MOD(o.customer_id, 4) = 3 THEN NULL
        ELSE ELT(MOD(o.customer_id, 3) + 1, 'Visa', 'Mastercard', 'Amex')
    END AS card_brand,
    CASE 
        WHEN MOD(o.customer_id, 4) = 3 THEN NULL
        ELSE FLOOR(1 + (RAND() * 12))
    END AS expiry_month,
    CASE 
        WHEN MOD(o.customer_id, 4) = 3 THEN NULL
        ELSE YEAR(CURDATE()) + FLOOR(1 + (RAND() * 5))
    END AS expiry_year,
    TRUE AS is_default
FROM orders o
WHERE NOT EXISTS (
    SELECT 1 FROM payment_methods pm 
    WHERE pm.customer_id = o.customer_id
)
LIMIT 500;

-- Add second payment method for 20% of customers
INSERT INTO payment_methods (customer_id, payment_type, card_last_four, card_brand, expiry_month, expiry_year, is_default)
SELECT 
    customer_id,
    'credit_card' AS payment_type,
    LPAD(FLOOR(1000 + (RAND() * 9000)), 4, '0') AS card_last_four,
    'Discover' AS card_brand,
    FLOOR(1 + (RAND() * 12)) AS expiry_month,
    YEAR(CURDATE()) + FLOOR(1 + (RAND() * 5)) AS expiry_year,
    FALSE AS is_default
FROM customers
WHERE customer_id % 5 = 0
AND customer_id <= 100;

-- ========================================
-- DISPLAY CONFIRMATION & STATISTICS
-- ========================================

SELECT 'Orders and order items inserted successfully!' AS Status;

SELECT 
    COUNT(*) AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered_orders,
    SUM(CASE WHEN status = 'shipped' THEN 1 ELSE 0 END) AS shipped_orders,
    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing_orders,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_orders,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
    ROUND(SUM(total_amount), 2) AS total_revenue,
    ROUND(AVG(total_amount), 2) AS avg_order_value
FROM orders;

SELECT 'Order items summary:' AS Info;
SELECT 
    COUNT(*) AS total_order_items,
    ROUND(AVG(quantity), 2) AS avg_quantity_per_item,
    ROUND(AVG(unit_price), 2) AS avg_unit_price,
    ROUND(SUM(discount), 2) AS total_discounts_given
FROM order_items;

SELECT 'Top 10 customers by order count:' AS Info;
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    COUNT(o.order_id) AS order_count,
    ROUND(SUM(o.total_amount), 2) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY order_count DESC
LIMIT 10;

SELECT 'Monthly order trends:' AS Info;
SELECT 
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    COUNT(*) AS orders,
    ROUND(SUM(total_amount), 2) AS revenue
FROM orders
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month; Customer ID (1-500, some customers order multiple times)
    FLOOR(1 + (RAND() * 500)) AS customer_id,
    -- Order dates spread across 2024, with more recent orders
    DATE_ADD('2024-01-01', INTERVAL FLOOR(RAND() * 365) DAY) + INTERVAL FLOOR(RAND() * 86400) SECOND AS order_date,
    -- Total amount will be calculated, placeholder for now
    0.00 AS total_amount,
    -- Status distribution: 40% delivered, 25% shipped, 20% processing, 10% pending, 5% cancelled
    ELT(
        CASE 
            WHEN seq % 20 = 0 THEN 5
            WHEN seq % 10 = 0 THEN 1
            WHEN seq % 5 = 0 THEN 2
            WHEN seq % 4 IN (0,1) THEN 3
            ELSE 4
        END,
        'pending', 'processing', 'shipped', 'delivered', 'cancelled'
    ) AS status,
    -- Payment status: 85% paid, 10% pending, 4% failed, 1% refunded
    ELT(
        CASE 
            WHEN seq % 100 = 0 THEN 4
            WHEN seq % 25 = 0 THEN 3
            WHEN seq % 10 = 0 THEN 1
            ELSE 2
        END,
        'pending', 'paid', 'failed', 'refunded'
    ) AS payment_status,
    -- Shipping cost: 0 (free), 5.99, 9.99, or 15.99
    ELT(MOD(seq, 4) + 1, 0.00, 5.99, 9.99, 15.99) AS shipping_cost,
    -- Tax will be calculated as 8% of subtotal
    0.00 AS tax_amount,
    DATE_ADD('2024-01-01', INTERVAL FLOOR(RAND() * 365) DAY) AS created_at
FROM (
    SELECT @row := @row + 1 as seq
    FROM 
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t2,
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t3,
        (SELECT @row := 0) r
) numbers
WHERE seq < 1000;

-- ========================================
-- ORDER ITEMS (2000-3000 items across 1000 orders)
-- Average 2-3 items per order
-- ========================================

-- First item for each order
INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount)
SELECT 
    o.order_id,
    -- Random product ID (1-300)
    FLOOR(1 + (RAND() * 300)) AS product_id,
    -- Quantity (1-5, most orders have 1-2 items)
    CASE 
        WHEN RAND() < 0.5 THEN 1
        WHEN RAND() < 0.8 THEN 2
        WHEN RAND() < 0.95 THEN 3
        ELSE FLOOR(1 + (RAND() * 5))
    END AS quantity,
    --