-- ============================================================================
-- E-COMMERCE ANALYTICS INDEX CREATION SCRIPT
-- Database: ecommerce_analytics
-- MySQL Version: 8.0.43
-- Purpose: Create indexes for optimized query performance
-- ============================================================================

USE ecommerce_analytics;

-- ============================================================================
-- CUSTOMERS TABLE INDEXES
-- ============================================================================

-- Index on email for faster customer lookups and duplicate detection
CREATE INDEX idx_customers_email ON customers(email);

-- Index on registration_date for temporal queries
CREATE INDEX idx_customers_registration_date ON customers(registration_date);

-- Index on name for search functionality
CREATE INDEX idx_customers_name ON customers(name);

-- Composite index for location-based queries
CREATE INDEX idx_customers_location ON customers(city, state, country);

-- ============================================================================
-- PRODUCTS TABLE INDEXES
-- ============================================================================

-- Unique index on SKU for fast product lookups
CREATE UNIQUE INDEX idx_products_sku ON products(sku);

-- Index on category_id for category-based filtering
CREATE INDEX idx_products_category ON products(category_id);

-- Index on name for product search
CREATE INDEX idx_products_name ON products(name);

-- Index on created_date for temporal analysis
CREATE INDEX idx_products_created_date ON products(created_date);

-- Index on price for price-based queries and sorting
CREATE INDEX idx_products_price ON products(price);

-- Composite index for inventory status queries
CREATE INDEX idx_products_stock ON products(stock_quantity, is_active);

-- ============================================================================
-- ORDERS TABLE INDEXES
-- ============================================================================

-- Index on customer_id for customer order history
CREATE INDEX idx_orders_customer_id ON orders(customer_id);

-- Index on order_date for temporal queries and reporting
CREATE INDEX idx_orders_order_date ON orders(order_date);

-- Index on status for filtering orders by status
CREATE INDEX idx_orders_status ON orders(status);

-- Index on payment_status for payment tracking
CREATE INDEX idx_orders_payment_status ON orders(payment_status);

-- Index on shipping_status for shipment tracking
CREATE INDEX idx_orders_shipping_status ON orders(shipping_status);

-- Composite index for date range and status queries
CREATE INDEX idx_orders_date_status ON orders(order_date, status);

-- ============================================================================
-- ORDER_ITEMS TABLE INDEXES
-- ============================================================================

-- Index on order_id for order detail lookups
CREATE INDEX idx_order_items_order_id ON order_items(order_id);

-- Index on product_id for product sales analysis
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- Composite index for product-order queries
CREATE INDEX idx_order_items_product_order ON order_items(product_id, order_id);

-- ============================================================================
-- INVENTORY TABLE INDEXES
-- ============================================================================

-- Index on product_id for product inventory lookups
CREATE INDEX idx_inventory_product_id ON inventory(product_id);

-- Index on warehouse_id for warehouse inventory queries
CREATE INDEX idx_inventory_warehouse_id ON inventory(warehouse_id);

-- Index on last_updated for tracking recent changes
CREATE INDEX idx_inventory_last_updated ON inventory(last_updated);

-- Index for low stock alerts
CREATE INDEX idx_inventory_low_stock ON inventory(quantity_available, reorder_point);

-- ============================================================================
-- VENDORS TABLE INDEXES
-- ============================================================================

-- Index on vendor_name for vendor search
CREATE INDEX idx_vendors_name ON vendors(vendor_name);

-- Index on status for filtering active vendors
CREATE INDEX idx_vendors_status ON vendors(status);

-- Index on country for geographic queries
CREATE INDEX idx_vendors_country ON vendors(country);

-- Index on rating for vendor performance queries
CREATE INDEX idx_vendors_rating ON vendors(rating);

-- Index on contact_email for communication
CREATE INDEX idx_vendors_email ON vendors(contact_email);

-- ============================================================================
-- VENDOR_CONTRACTS TABLE INDEXES
-- ============================================================================

-- Index on vendor_id for vendor contract lookups
CREATE INDEX idx_vendor_contracts_vendor_id ON vendor_contracts(vendor_id);

-- Index on status for active contract filtering
CREATE INDEX idx_vendor_contracts_status ON vendor_contracts(status);

-- Composite index for date range queries
CREATE INDEX idx_vendor_contracts_dates ON vendor_contracts(start_date, end_date);

-- ============================================================================
-- CAMPAIGNS TABLE INDEXES
-- ============================================================================

-- Index on campaign_name for campaign search
CREATE INDEX idx_campaigns_name ON campaigns(campaign_name);

-- Index on channel for channel-based analysis
CREATE INDEX idx_campaigns_channel ON campaigns(channel);

-- Index on status for filtering active campaigns
CREATE INDEX idx_campaigns_status ON campaigns(status);

-- Composite index for date range queries
CREATE INDEX idx_campaigns_dates ON campaigns(start_date, end_date);

-- ============================================================================
-- CAMPAIGN_PERFORMANCE TABLE INDEXES
-- ============================================================================

-- Index on campaign_id for campaign performance lookups
CREATE INDEX idx_campaign_performance_campaign_id ON campaign_performance(campaign_id);

-- Index on date for temporal analysis
CREATE INDEX idx_campaign_performance_date ON campaign_performance(date);

-- Composite index for campaign date range queries
CREATE INDEX idx_campaign_performance_campaign_date ON campaign_performance(campaign_id, date);

-- ============================================================================
-- RETURNS TABLE INDEXES
-- ============================================================================

-- Index on order_id for order return lookups
CREATE INDEX idx_returns_order_id ON returns(order_id);

-- Index on customer_id for customer return history
CREATE INDEX idx_returns_customer_id ON returns(customer_id);

-- Index on return_date for temporal queries
CREATE INDEX idx_returns_return_date ON returns(return_date);

-- Index on status for return processing
CREATE INDEX idx_returns_status ON returns(status);

-- Index on reason for return reason analysis
CREATE INDEX idx_returns_reason ON returns(reason);

-- ============================================================================
-- REFUNDS TABLE INDEXES
-- ============================================================================

-- Index on return_id for return-refund relationship
CREATE INDEX idx_refunds_return_id ON refunds(return_id);

-- Index on order_id for order refund lookups
CREATE INDEX idx_refunds_order_id ON refunds(order_id);

-- Index on initiated_date for temporal queries
CREATE INDEX idx_refunds_initiated_date ON refunds(initiated_date);

-- Index on status for refund processing
CREATE INDEX idx_refunds_status ON refunds(status);

-- Index on transaction_id for transaction tracking
CREATE INDEX idx_refunds_transaction_id ON refunds(transaction_id);

-- ============================================================================
-- PAYMENTS TABLE INDEXES
-- ============================================================================

-- Index on order_id for order payment lookups
CREATE INDEX idx_payments_order_id ON payments(order_id);

-- Index on payment_date for temporal queries and reporting
CREATE INDEX idx_payments_payment_date ON payments(payment_date);

-- Index on status for payment status filtering
CREATE INDEX idx_payments_status ON payments(status);

-- Index on payment_method for payment method analysis
CREATE INDEX idx_payments_payment_method ON payments(payment_method);

-- Index on transaction_id for transaction tracking
CREATE INDEX idx_payments_transaction_id ON payments(transaction_id);

-- Composite index for date and status queries
CREATE INDEX idx_payments_date_status ON payments(payment_date, status);

-- ============================================================================
-- SHIPPING TABLE INDEXES
-- ============================================================================

-- Index on order_id for order shipment lookups
CREATE INDEX idx_shipping_order_id ON shipping(order_id);

-- Index on tracking_number for shipment tracking
CREATE INDEX idx_shipping_tracking_number ON shipping(tracking_number);

-- Index on shipped_date for temporal queries
CREATE INDEX idx_shipping_shipped_date ON shipping(shipped_date);

-- Index on delivered_date for delivery tracking
CREATE INDEX idx_shipping_delivered_date ON shipping(delivered_date);

-- Index on status for shipment status filtering
CREATE INDEX idx_shipping_status ON shipping(status);

-- Index on carrier for carrier analysis
CREATE INDEX idx_shipping_carrier ON shipping(carrier);

-- Index on address_validated for address validation checks
CREATE INDEX idx_shipping_address_validated ON shipping(address_validated);

-- ============================================================================
-- REVIEWS TABLE INDEXES
-- ============================================================================

-- Index on product_id for product review lookups
CREATE INDEX idx_reviews_product_id ON reviews(product_id);

-- Index on customer_id for customer review history
CREATE INDEX idx_reviews_customer_id ON reviews(customer_id);

-- Index on order_id for order review lookups
CREATE INDEX idx_reviews_order_id ON reviews(order_id);

-- Index on rating for rating-based filtering
CREATE INDEX idx_reviews_rating ON reviews(rating);

-- Index on review_date for temporal queries
CREATE INDEX idx_reviews_review_date ON reviews(review_date);

-- Index on verified_purchase for verified review filtering
CREATE INDEX idx_reviews_verified_purchase ON reviews(verified_purchase);

-- Composite index for product rating queries
CREATE INDEX idx_reviews_product_rating ON reviews(product_id, rating);

-- ============================================================================
-- LOYALTY_PROGRAM TABLE INDEXES
-- ============================================================================

-- Index on customer_id for customer loyalty lookups (already UNIQUE)
-- CREATE UNIQUE INDEX idx_loyalty_customer_id ON loyalty_program(customer_id);

-- Index on tier for tier-based queries
CREATE INDEX idx_loyalty_tier ON loyalty_program(tier);

-- Index on status for active member filtering
CREATE INDEX idx_loyalty_status ON loyalty_program(status);

-- Index on join_date for temporal analysis
CREATE INDEX idx_loyalty_join_date ON loyalty_program(join_date);

-- Index on points_balance for points-based queries
CREATE INDEX idx_loyalty_points_balance ON loyalty_program(points_balance);

-- ============================================================================
-- WAREHOUSES TABLE INDEXES
-- ============================================================================

-- Index on warehouse_name for warehouse search
CREATE INDEX idx_warehouses_name ON warehouses(warehouse_name);

-- Index on location for geographic queries
CREATE INDEX idx_warehouses_location ON warehouses(city, state);

-- Index on is_active for active warehouse filtering
CREATE INDEX idx_warehouses_is_active ON warehouses(is_active);

-- ============================================================================
-- PRODUCT_CATEGORIES TABLE INDEXES
-- ============================================================================

-- Index on parent_category_id for category hierarchy queries
CREATE INDEX idx_product_categories_parent ON product_categories(parent_category_id);

-- Index on category_name is already UNIQUE
-- CREATE UNIQUE INDEX idx_product_categories_name ON product_categories(category_name);

-- ============================================================================
-- DISPLAY CONFIRMATION
-- ============================================================================

SELECT 'All indexes created successfully!' AS Status;

-- Display index count per table
SELECT 
    table_name AS 'Table Name',
    COUNT(*) AS 'Number of Indexes',
    GROUP_CONCAT(index_name SEPARATOR ', ') AS 'Index Names'
FROM information_schema.statistics 
WHERE table_schema = 'ecommerce_analytics'
    AND index_name != 'PRIMARY'
GROUP BY table_name
ORDER BY table_name;

-- Display total index count
SELECT COUNT(DISTINCT index_name) AS 'Total Indexes Created'
FROM information_schema.statistics 
WHERE table_schema = 'ecommerce_analytics'
    AND index_name != 'PRIMARY';
