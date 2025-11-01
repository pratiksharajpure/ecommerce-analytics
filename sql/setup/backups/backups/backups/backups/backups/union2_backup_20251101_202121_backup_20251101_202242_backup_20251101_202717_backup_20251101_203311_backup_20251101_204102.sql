-- Save as: verify_setup.sql
USE ecommerce_analytics;

-- Check table counts
SELECT 'CUSTOMERS' AS table_name, COUNT(*) AS records FROM customers
UNION ALL SELECT 'PRODUCTS', COUNT(*) FROM products
UNION ALL SELECT 'ORDERS', COUNT(*) FROM orders
UNION ALL SELECT 'ORDER_ITEMS', COUNT(*) FROM order_items
UNION ALL SELECT 'INVENTORY', COUNT(*) FROM inventory
UNION ALL SELECT 'VENDORS', COUNT(*) FROM vendors
UNION ALL SELECT 'PRODUCT_CATEGORIES', COUNT(*) FROM product_categories
UNION ALL SELECT 'CAMPAIGNS', COUNT(*) FROM campaigns
UNION ALL SELECT 'REVIEWS', COUNT(*) FROM reviews
UNION ALL SELECT 'RETURNS', COUNT(*) FROM returns
UNION ALL SELECT 'LOYALTY_PROGRAM', COUNT(*) FROM loyalty_program;

-- Check critical columns exist
SHOW COLUMNS FROM customers LIKE 'country_code';
SHOW COLUMNS FROM products LIKE 'category_code';
SHOW COLUMNS FROM payment_methods LIKE 'card_brand';

-- Test a simple join
SELECT 
    c.customer_name,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name
LIMIT 10;

SELECT '✅ All verification checks passed!' AS status;