-- ========================================
-- FIX MISSING TABLES AND COLUMNS
-- Run this FIRST before everything else
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. CREATE MISSING LOOKUP TABLES
-- ========================================

-- States lookup table
CREATE TABLE IF NOT EXISTS states (
    state_code VARCHAR(2) PRIMARY KEY,
    state_name VARCHAR(50) NOT NULL,
    region VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Product conditions lookup
CREATE TABLE IF NOT EXISTS product_conditions (
    condition_code VARCHAR(20) PRIMARY KEY,
    condition_name VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Order statuses lookup
CREATE TABLE IF NOT EXISTS order_statuses (
    status_code VARCHAR(20) PRIMARY KEY,
    status_name VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Shipping methods lookup
CREATE TABLE IF NOT EXISTS shipping_methods (
    method_code VARCHAR(20) PRIMARY KEY,
    method_name VARCHAR(50) NOT NULL,
    estimated_days INT,
    base_cost DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Customer segments lookup
CREATE TABLE IF NOT EXISTS customer_segments (
    segment_code VARCHAR(20) PRIMARY KEY,
    segment_name VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Campaign types lookup
CREATE TABLE IF NOT EXISTS campaign_types (
    type_code VARCHAR(20) PRIMARY KEY,
    type_name VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Vendor categories lookup
CREATE TABLE IF NOT EXISTS vendor_categories (
    category_code VARCHAR(20) PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Quality scores tracking
CREATE TABLE IF NOT EXISTS quality_scores (
    score_id INT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100),
    score DECIMAL(5,2),
    total_records INT,
    issues_found INT,
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_table_metric (table_name, metric_name),
    INDEX idx_measured_at (measured_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Data retention policy
CREATE TABLE IF NOT EXISTS data_retention_policy (
    policy_id INT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    retention_days INT NOT NULL,
    archive_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_table (table_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
-- ========================================
-- CREATE PAYMENT_METHODS TABLE
-- ========================================


-- Archive log
CREATE TABLE IF NOT EXISTS archive_log (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    records_archived INT DEFAULT 0,
    archive_date DATE NOT NULL,
    status ENUM('success', 'failed', 'in_progress') DEFAULT 'in_progress',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_table_date (table_name, archive_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Materialized views metadata
CREATE TABLE IF NOT EXISTS mv_refresh_log (
    refresh_id INT AUTO_INCREMENT PRIMARY KEY,
    view_name VARCHAR(50) NOT NULL,
    refresh_started TIMESTAMP NULL,
    refresh_completed TIMESTAMP NULL,
    rows_affected INT,
    status ENUM('running', 'completed', 'failed') DEFAULT 'running',
    error_message TEXT,
    INDEX idx_view_status (view_name, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Materialized view: Customer LTV
CREATE TABLE IF NOT EXISTS mv_customer_ltv (
    customer_id VARCHAR(20) PRIMARY KEY,
    total_orders INT,
    total_spent DECIMAL(12,2),
    avg_order_value DECIMAL(10,2),
    first_order_date DATE,
    last_order_date DATE,
    days_as_customer INT,
    ltv DECIMAL(12,2),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Materialized view: Product performance
CREATE TABLE IF NOT EXISTS mv_product_performance (
    product_id VARCHAR(20) PRIMARY KEY,
    total_sold INT,
    total_revenue DECIMAL(12,2),
    avg_price DECIMAL(10,2),
    times_ordered INT,
    last_sold_date DATE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Countries table
CREATE TABLE IF NOT EXISTS countries (
    country_id VARCHAR(3) PRIMARY KEY,
    country_code VARCHAR(3) UNIQUE NOT NULL,
    country_name VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_region (region)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



-- Categories (self-referencing, no external dependencies)
CREATE TABLE categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_code VARCHAR(20) UNIQUE,
    category_name VARCHAR(100) NOT NULL,
    parent_category_id INT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_parent (parent_category_id),
    INDEX idx_code (category_code),
    INDEX idx_name (category_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
ALTER TABLE categories
ADD CONSTRAINT fk_parent_category 
FOREIGN KEY (parent_category_id) REFERENCES categories(category_id) ON DELETE SET NULL;


-- ========================================
-- CREATE SHIPPING AND CAMPAIGN_PERFORMANCE TABLES
-- ========================================

USE ecommerce_analytics;

SET FOREIGN_KEY_CHECKS = 0;

-- ========================================
-- SHIPPING TABLE
-- ========================================

CREATE TABLE IF NOT EXISTS shipping (
    shipping_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id VARCHAR(20) NOT NULL,
    shipping_method VARCHAR(50),
    method_code VARCHAR(20),
    shipping_cost DECIMAL(10,2),
    tracking_number VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    shipped_date DATE,
    delivered_date DATE,
    estimated_delivery DATE,
    carrier VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (method_code) REFERENCES shipping_methods(method_code) ON DELETE SET NULL,
    INDEX idx_order (order_id),
    INDEX idx_tracking (tracking_number),
    INDEX idx_method (method_code),
    INDEX idx_shipped_date (shipped_date),
    INDEX idx_delivered_date (delivered_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;



-- ========================================
-- 2. ADD MISSING COLUMNS TO EXISTING TABLES
-- ========================================

-- Add card_brand as alias for card_type (for compatibility)
ALTER TABLE payment_methods 
ADD COLUMN IF NOT EXISTS card_brand VARCHAR(20) AFTER card_type;

-- Update card_brand to match card_type
UPDATE payment_methods SET card_brand = card_type WHERE card_brand IS NULL;

-- ========================================
-- 3. INSERT REFERENCE DATA
-- ========================================

-- Insert US States
INSERT IGNORE INTO states (state_code, state_name, region) VALUES
('AL', 'Alabama', 'South'),
('AK', 'Alaska', 'West'),
('AZ', 'Arizona', 'West'),
('AR', 'Arkansas', 'South'),
('CA', 'California', 'West'),
('CO', 'Colorado', 'West'),
('CT', 'Connecticut', 'Northeast'),
('DE', 'Delaware', 'Northeast'),
('FL', 'Florida', 'South'),
('GA', 'Georgia', 'South'),
('HI', 'Hawaii', 'West'),
('ID', 'Idaho', 'West'),
('IL', 'Illinois', 'Midwest'),
('IN', 'Indiana', 'Midwest'),
('IA', 'Iowa', 'Midwest'),
('KS', 'Kansas', 'Midwest'),
('KY', 'Kentucky', 'South'),
('LA', 'Louisiana', 'South'),
('ME', 'Maine', 'Northeast'),
('MD', 'Maryland', 'Northeast'),
('MA', 'Massachusetts', 'Northeast'),
('MI', 'Michigan', 'Midwest'),
('MN', 'Minnesota', 'Midwest'),
('MS', 'Mississippi', 'South'),
('MO', 'Missouri', 'Midwest'),
('MT', 'Montana', 'West'),
('NE', 'Nebraska', 'Midwest'),
('NV', 'Nevada', 'West'),
('NH', 'New Hampshire', 'Northeast'),
('NJ', 'New Jersey', 'Northeast'),
('NM', 'New Mexico', 'West'),
('NY', 'New York', 'Northeast'),
('NC', 'North Carolina', 'South'),
('ND', 'North Dakota', 'Midwest'),
('OH', 'Ohio', 'Midwest'),
('OK', 'Oklahoma', 'South'),
('OR', 'Oregon', 'West'),
('PA', 'Pennsylvania', 'Northeast'),
('RI', 'Rhode Island', 'Northeast'),
('SC', 'South Carolina', 'South'),
('SD', 'South Dakota', 'Midwest'),
('TN', 'Tennessee', 'South'),
('TX', 'Texas', 'South'),
('UT', 'Utah', 'West'),
('VT', 'Vermont', 'Northeast'),
('VA', 'Virginia', 'South'),
('WA', 'Washington', 'West'),
('WV', 'West Virginia', 'South'),
('WI', 'Wisconsin', 'Midwest'),
('WY', 'Wyoming', 'West');

-- Insert product conditions
INSERT IGNORE INTO product_conditions (condition_code, condition_name, description) VALUES
('NEW', 'New', 'Brand new, unused product'),
('REFURB', 'Refurbished', 'Professionally restored to like-new condition'),
('USED_LIKE_NEW', 'Used - Like New', 'Gently used, excellent condition'),
('USED_GOOD', 'Used - Good', 'Previously used, good working condition'),
('USED_ACCEPTABLE', 'Used - Acceptable', 'Shows wear, fully functional');

-- Insert order statuses
INSERT IGNORE INTO order_statuses (status_code, status_name, description) VALUES
('pending', 'Pending', 'Order received, awaiting processing'),
('processing', 'Processing', 'Order is being prepared'),
('shipped', 'Shipped', 'Order has been shipped'),
('delivered', 'Delivered', 'Order delivered to customer'),
('cancelled', 'Cancelled', 'Order was cancelled');

-- Insert shipping methods
INSERT IGNORE INTO shipping_methods (method_code, method_name, estimated_days, base_cost) VALUES
('STANDARD', 'Standard Shipping', 5, 5.99),
('EXPRESS', 'Express Shipping', 2, 12.99),
('OVERNIGHT', 'Overnight Shipping', 1, 24.99),
('FREE', 'Free Shipping', 7, 0.00),
('PICKUP', 'Store Pickup', 0, 0.00);

-- Insert customer segments
INSERT IGNORE INTO customer_segments (segment_code, segment_name, description) VALUES
('VIP', 'VIP', 'High-value customers'),
('REGULAR', 'Regular', 'Standard customers'),
('NEW', 'New', 'First-time customers'),
('AT_RISK', 'At Risk', 'Customers who may churn'),
('LOST', 'Lost', 'Inactive customers');

-- Insert campaign types
INSERT IGNORE INTO campaign_types (type_code, type_name, description) VALUES
('EMAIL', 'Email Marketing', 'Email campaigns'),
('SOCIAL', 'Social Media', 'Social media advertising'),
('PPC', 'Pay-Per-Click', 'Search engine advertising'),
('DISPLAY', 'Display Ads', 'Banner and display advertising'),
('AFFILIATE', 'Affiliate Marketing', 'Partner marketing programs');

-- Insert vendor categories
INSERT IGNORE INTO vendor_categories (category_code, category_name, description) VALUES
('MANUFACTURER', 'Manufacturer', 'Direct manufacturers'),
('DISTRIBUTOR', 'Distributor', 'Distribution companies'),
('WHOLESALER', 'Wholesaler', 'Wholesale suppliers'),
('DROPSHIP', 'Dropshipper', 'Dropshipping partners'),
('SERVICE', 'Service Provider', 'Service vendors');

-- Insert default retention policies
INSERT IGNORE INTO data_retention_policy (table_name, retention_days, archive_enabled) VALUES
('orders', 2555, TRUE),  -- 7 years
('order_items', 2555, TRUE),
('returns', 1095, TRUE),  -- 3 years
('reviews', 1825, FALSE), -- 5 years, no archive
('campaign_performance', 730, TRUE); -- 2 years

SELECT 'All missing tables and reference data created successfully!' AS Status;
