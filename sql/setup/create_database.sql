-- ========================================
-- CREATE DATABASE
-- E-commerce Revenue Analytics Engine
-- ========================================

-- Drop database if exists (for clean setup)
DROP DATABASE IF EXISTS ecommerce_analytics;

-- Create database with proper character set and collation
CREATE DATABASE ecommerce_analytics
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

-- Use the database
USE ecommerce_analytics;

-- Display confirmation
SELECT 'Database ecommerce_analytics created successfully' AS Status;
