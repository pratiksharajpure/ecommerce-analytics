-- ========================================
-- CREATE COMPREHENSIVE PERFORMANCE AND GENERAL INDEXES
-- E-commerce Revenue Analytics Engine
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- CUSTOMERS TABLE
-- ========================================
-- General indexes
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_status ON customers(status);
CREATE INDEX idx_customers_created_at ON customers(created_at);

-- Performance / composite indexes
CREATE INDEX idx_customer_name ON customers(last_name, first_name);
CREATE INDEX idx_customer_city_state ON customers(city, state);
CREATE INDEX idx_customer_created_date ON customers(created_at);

-- ========================================
-- PRODUCTS TABLE
-- ========================================
-- General indexes
CREATE INDEX idx_products_sku ON products(sku);
CREATE INDEX idx_products_name ON products(name);
CREATE INDEX idx_products_category_id ON products(category_id);
CREATE INDEX idx_products_status ON products(status);

-- Performance indexes
CREATE INDEX idx_product_price_range ON products(price);
CREATE INDEX idx_product_stock ON products(stock_quantity);
CREATE INDEX idx_product_category_status ON products(category_id, status);
CREATE FULLTEXT INDEX idx_product_search ON products(product_name, description);

-- ========================================
-- PRODUCT CATEGORIES TABLE
-- ========================================
CREATE INDEX idx_product_categories_parent_id ON product_categories(parent_id);

-- ========================================
-- ORDERS TABLE
-- ========================================
-- General indexes
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_payment_status ON orders(payment_status);

-- Performance / composite indexes
CREATE INDEX idx_order_total_amount ON orders(total_amount);
CREATE INDEX idx_order_date_status ON orders(order_date, status);
CREATE INDEX idx_order_customer_date ON orders(customer_id, order_date);

-- ========================================
-- ORDER ITEMS TABLE
-- ========================================
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- Performance indexes
CREATE INDEX idx_orderitem_product_order ON order_items(product_id, order_id);
CREATE INDEX idx_orderitem_unit_price ON order_items(unit_price);

-- ========================================
-- INVENTORY TABLE
-- ========================================
CREATE INDEX idx_inventory_product_id ON inventory(product_id);
CREATE INDEX idx_inventory_warehouse_id ON inventory(warehouse_id);
CREATE INDEX idx_inventory_last_updated ON inventory(last_updated);

-- Performance indexes
CREATE INDEX idx_inventory_availability ON inventory(quantity_available);
CREATE INDEX idx_inventory_reorder ON inventory(reorder_level);

-- ========================================
-- VENDORS TABLE
-- ========================================
CREATE INDEX idx_vendors_email ON vendors(email);
CREATE INDEX idx_vendors_status ON vendors(status);

-- Performance indexes
CREATE INDEX idx_vendor_rating_status ON vendors(rating, status);

-- ========================================
-- VENDOR CONTRACTS TABLE
-- ========================================
CREATE INDEX idx_vendor_contracts_vendor_id ON vendor_contracts(vendor_id);
CREATE INDEX idx_vendor_contracts_product_id ON vendor_contracts(product_id);

-- ========================================
-- SHIPPING ADDRESSES TABLE
-- ========================================
CREATE INDEX idx_shipping_addresses_customer_id ON shipping_addresses(customer_id);

-- ========================================
-- PAYMENT METHODS TABLE
-- ========================================
CREATE INDEX idx_payment_methods_customer_id ON payment_methods(customer_id);
CREATE INDEX idx_payment_methods_type ON payment_methods(payment_type);

-- ========================================
-- CAMPAIGNS TABLE
-- ========================================
CREATE INDEX idx_campaigns_type ON campaigns(campaign_type);
CREATE INDEX idx_campaigns_start_date ON campaigns(start_date);
CREATE INDEX idx_campaigns_end_date ON campaigns(end_date);
CREATE INDEX idx_campaigns_status ON campaigns(status);

-- Performance indexes
CREATE INDEX idx_campaign_date_range ON campaigns(start_date, end_date);
CREATE INDEX idx_campaign_type_status ON campaigns(campaign_type, status);

-- ========================================
-- CAMPAIGN PERFORMANCE TABLE
-- ========================================
CREATE INDEX idx_campaign_performance_campaign_id ON campaign_performance(campaign_id);

-- ========================================
-- REVIEWS TABLE
-- ========================================
CREATE INDEX idx_reviews_product_id ON reviews(product_id);
CREATE INDEX idx_reviews_customer_id ON reviews(customer_id);
CREATE INDEX idx_reviews_created_at ON reviews(created_at);

-- Performance indexes
CREATE INDEX idx_review_product_rating ON reviews(product_id, rating);
CREATE INDEX idx_review_verified ON reviews(is_verified_purchase);

-- ========================================
-- RETURNS TABLE
-- ========================================
CREATE INDEX idx_returns_order_id ON returns(order_id);
CREATE INDEX idx_returns_status ON returns(status);
CREATE INDEX idx_returns_created_at ON returns(created_at);

-- Performance indexes
CREATE INDEX idx_return_reason ON returns(reason);
CREATE INDEX idx_return_date_status ON returns(created_at, status);

-- ========================================
-- LOYALTY PROGRAM TABLE
-- ========================================
CREATE INDEX idx_loyalty_program_customer_id ON loyalty_program(customer_id);
CREATE INDEX idx_loyalty_program_joined_date ON loyalty_program(joined_date);
CREATE INDEX idx_loyalty_program_tier ON loyalty_program(tier);

-- ========================================
-- Confirmation message
-- ========================================
SELECT 'All indexes (general + performance) created successfully' AS Status;
