-- ========================================
-- MASTER FIX SCRIPT - RUN THIS FIRST
-- Fixes: Missing columns, duplicate indexes, MariaDB issues
-- ========================================

USE ecommerce_analytics;

SET FOREIGN_KEY_CHECKS = 0;

-- ========================================
-- 1. FIX MISSING COLUMNS
-- ========================================

-- Add country_code to customers
ALTER TABLE customers 
ADD COLUMN IF NOT EXISTS country_code VARCHAR(3) DEFAULT 'USA' AFTER country;

-- Add category_code to products
ALTER TABLE products
ADD COLUMN IF NOT EXISTS category_code VARCHAR(20) AFTER category_id;

-- Ensure card_brand exists in payment_methods (it should from fix_missing_tables.sql)
ALTER TABLE payment_methods
ADD COLUMN IF NOT EXISTS card_brand VARCHAR(20) AFTER card_type;

-- Update card_brand with card_type values
UPDATE payment_methods SET card_brand = card_type WHERE card_brand IS NULL;

-- ========================================
-- 2. FIX DUPLICATE INDEXES
-- ========================================

-- Drop all potentially duplicate indexes first
DROP INDEX IF EXISTS idx_customers_email ON customers;
DROP INDEX IF EXISTS idx_customers_status ON customers;
DROP INDEX IF EXISTS idx_customers_created_at ON customers;
DROP INDEX IF EXISTS idx_customer_name ON customers;
DROP INDEX IF EXISTS idx_customer_city_state ON customers;
DROP INDEX IF EXISTS idx_customer_created_date ON customers;

DROP INDEX IF EXISTS idx_products_name ON products;
DROP INDEX IF EXISTS idx_product_search ON products;

DROP INDEX IF EXISTS idx_loyalty_program_customer_id ON loyalty_program;
DROP INDEX IF EXISTS idx_loyalty_program_joined_date ON loyalty_program;
DROP INDEX IF EXISTS idx_loyalty_program_tier ON loyalty_program;

-- Now recreate only necessary indexes
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_status ON customers(status);
CREATE INDEX idx_customer_name ON customers(last_name, first_name);

-- ========================================
-- 3. CREATE MISSING SUPPORT TABLES
-- ========================================

CREATE TABLE IF NOT EXISTS mv_last_refresh (
    view_name VARCHAR(50) PRIMARY KEY,
    last_refresh_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    refresh_type ENUM('full', 'incremental') DEFAULT 'full',
    rows_affected INT,
    duration_seconds INT,
    start_time TIMESTAMP NULL,
    INDEX idx_refresh_time(last_refresh_time)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS notification_preferences (
    preference_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    notification_type VARCHAR(50),
    channel ENUM('email', 'sms', 'push') DEFAULT 'email',
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user(user_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS data_quality_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_category VARCHAR(50),
    current_score DECIMAL(5,2),
    current_grade VARCHAR(2),
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category(metric_category),
    INDEX idx_measured(measured_at)
) ENGINE=InnoDB;

-- ========================================
-- 4. POPULATE CATEGORY CODES
-- ========================================

-- Update product category codes based on category_id
UPDATE products p
JOIN product_categories pc ON p.category_id = pc.category_id
SET p.category_code = pc.category_id
WHERE p.category_code IS NULL;

SET FOREIGN_KEY_CHECKS = 1;

-- ========================================
-- 5. VERIFICATION
-- ========================================

SELECT 'Fix script completed successfully!' AS Status;

-- Verify critical columns exist
SELECT 
    'Customers table' AS check_item,
    COUNT(*) AS record_count,
    SUM(CASE WHEN country_code IS NOT NULL THEN 1 ELSE 0 END) AS country_code_filled
FROM customers
UNION ALL
SELECT 
    'Products table',
    COUNT(*),
    SUM(CASE WHEN category_code IS NOT NULL THEN 1 ELSE 0 END)
FROM products
UNION ALL
SELECT 
    'Payment methods table',
    COUNT(*),
    SUM(CASE WHEN card_brand IS NOT NULL THEN 1 ELSE 0 END)
FROM payment_methods;


-- ========================================
-- MASTER FIX - Run IMMEDIATELY
-- ========================================

USE ecommerce_analytics;
SET FOREIGN_KEY_CHECKS = 0;

-- Fix missing columns causing errors
ALTER TABLE customers ADD COLUMN IF NOT EXISTS country_code VARCHAR(3) DEFAULT 'USA';
ALTER TABLE products ADD COLUMN IF NOT EXISTS category_code VARCHAR(20);
ALTER TABLE payment_methods ADD COLUMN IF NOT EXISTS card_brand VARCHAR(20);

-- Remove duplicate indexes causing errors
DROP INDEX IF EXISTS idx_customers_email ON customers;
DROP INDEX IF EXISTS idx_customers_status ON customers;
DROP INDEX IF EXISTS idx_customer_name ON customers;
DROP INDEX IF EXISTS idx_products_name ON products;

-- Recreate indexes properly
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_status ON customers(status);
CREATE INDEX idx_customer_name ON customers(last_name, first_name);

-- Create missing tables
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_category VARCHAR(50),
    current_score DECIMAL(5,2),
    current_grade VARCHAR(2),
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SET FOREIGN_KEY_CHECKS = 1;
SELECT ''SUCCESS' Master fix completed!' AS Status;