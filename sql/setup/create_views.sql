-- ========================================
-- CREATE DATABASE VIEWS
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- VIEW 1: CUSTOMER SUMMARY
-- ========================================
CREATE OR REPLACE VIEW vw_customer_summary AS
SELECT 
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    c.phone,
    c.status AS customer_status,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COALESCE(SUM(o.total_amount), 0) AS lifetime_value,
    COALESCE(AVG(o.total_amount), 0) AS average_order_value,
    MAX(o.order_date) AS last_order_date,
    c.created_at AS customer_since
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.phone, c.status, c.created_at;

-- ========================================
-- VIEW 2: PRODUCT INVENTORY STATUS
-- ========================================
CREATE OR REPLACE VIEW vw_product_inventory AS
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    pc.category_name,
    p.price,
    p.cost,
    (p.price - p.cost) AS profit_margin,
    p.stock_quantity,
    COALESCE(SUM(i.quantity_on_hand), 0) AS warehouse_quantity,
    COALESCE(SUM(i.quantity_reserved), 0) AS reserved_quantity,
    COALESCE(SUM(i.quantity_available), 0) AS available_quantity,
    p.status AS product_status,
    CASE 
        WHEN p.stock_quantity <= 0 THEN 'Out of Stock'
        WHEN p.stock_quantity < 10 THEN 'Low Stock'
        ELSE 'In Stock'
    END AS stock_status
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN inventory i ON p.product_id = i.product_id
GROUP BY p.product_id, p.sku, p.product_name, pc.category_name, p.price, p.cost, p.stock_quantity, p.status;

-- ========================================
-- VIEW 3: ORDER DETAILS
-- ========================================
CREATE OR REPLACE VIEW vw_order_details AS
SELECT 
    o.order_id,
    o.order_date,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email AS customer_email,
    COUNT(oi.order_item_id) AS items_count,
    SUM(oi.quantity) AS total_quantity,
    o.total_amount,
    o.shipping_cost,
    o.tax_amount,
    o.status AS order_status,
    o.payment_status,
    DATEDIFF(CURRENT_DATE, o.order_date) AS days_since_order
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.order_date, c.first_name, c.last_name, c.email, 
         o.total_amount, o.shipping_cost, o.tax_amount, o.status, o.payment_status;

-- ========================================
-- VIEW 4: CAMPAIGN ROI
-- ========================================
CREATE OR REPLACE VIEW vw_campaign_roi AS
SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.budget,
    c.status,
    c.start_date,
    c.end_date,
    COALESCE(SUM(cp.impressions), 0) AS total_impressions,
    COALESCE(SUM(cp.clicks), 0) AS total_clicks,
    COALESCE(SUM(cp.conversions), 0) AS total_conversions,
    COALESCE(SUM(cp.spend), 0) AS total_spend,
    COALESCE(SUM(cp.revenue), 0) AS total_revenue,
    CASE 
        WHEN SUM(cp.impressions) > 0 THEN (SUM(cp.clicks) * 100.0 / SUM(cp.impressions))
        ELSE 0 
    END AS click_through_rate,
    CASE 
        WHEN SUM(cp.clicks) > 0 THEN (SUM(cp.conversions) * 100.0 / SUM(cp.clicks))
        ELSE 0 
    END AS conversion_rate,
    CASE 
        WHEN SUM(cp.spend) > 0 THEN ((SUM(cp.revenue) - SUM(cp.spend)) * 100.0 / SUM(cp.spend))
        ELSE 0 
    END AS roi_percentage
FROM campaigns c
LEFT JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.budget, c.status, c.start_date, c.end_date;

-- Display confirmation
SELECT 'All database views created successfully' AS Status;