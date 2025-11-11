-- ============================================================================
-- E-COMMERCE ANALYTICS DATABASE CREATION SCRIPT
-- Database: ecommerce_analytics
-- MySQL Version: 8.0.43
-- Purpose: Create the main database with proper configuration
-- ============================================================================

-- Drop database if it exists (use with caution in production)
DROP DATABASE IF EXISTS ecommerce_analytics;

-- Create the database with UTF-8 encoding
CREATE DATABASE ecommerce_analytics
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Use the database
USE ecommerce_analytics;

-- Display confirmation message
SELECT 'Database ecommerce_analytics created successfully!' AS Status;

-- Set default storage engine to InnoDB for transaction support
SET default_storage_engine=InnoDB;

-- Set SQL mode for strict data validation
SET sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- Display database configuration
SELECT 
    @@character_set_database AS 'Character Set',
    @@collation_database AS 'Collation',
    @@default_storage_engine AS 'Storage Engine',
    @@sql_mode AS 'SQL Mode';
