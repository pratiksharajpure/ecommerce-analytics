-- ============================================================================
-- E-COMMERCE ANALYTICS TABLE CREATION SCRIPT
-- Database: ecommerce_analytics
-- MySQL Version: 8.0.43
-- Purpose: Create all tables in proper order (parent tables before child tables)
-- ============================================================================

USE ecommerce_analytics;


-- Drop tables if they exist (in reverse order of dependencies)
DROP TABLE IF EXISTS loyalty_program CASCADE;
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS shipping CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS refunds CASCADE;
DROP TABLE IF EXISTS returns CASCADE;
DROP TABLE IF EXISTS campaign_performance CASCADE;
DROP TABLE IF EXISTS campaigns CASCADE;
DROP TABLE IF EXISTS vendor_contracts CASCADE;
DROP TABLE IF EXISTS vendors CASCADE;
DROP TABLE IF EXISTS inventory CASCADE;
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS product_categories CASCADE;
DROP TABLE IF EXISTS warehouses CASCADE;
DROP TABLE IF EXISTS customers CASCADE;


-- ============================================================================
-- CORE BUSINESS TABLES
-- ============================================================================

-- Table: customers
-- Purpose: Store customer information
CREATE TABLE customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    zip_code VARCHAR(20),
    country VARCHAR(100) DEFAULT 'USA',
    registration_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: product_categories
-- Purpose: Store product category hierarchy
CREATE TABLE product_categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL UNIQUE,
    parent_category_id INT DEFAULT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_category_id) REFERENCES product_categories(category_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: products
-- Purpose: Store product catalog information
CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    sku VARCHAR(50) NOT NULL,
    name VARCHAR(255),
    category_id INT,
    price DECIMAL(10, 2),
    cost DECIMAL(10, 2),
    description TEXT,
    image_url VARCHAR(500),
    stock_quantity INT DEFAULT 0,
    weight DECIMAL(10, 2),
    dimensions VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES product_categories(category_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: warehouses
-- Purpose: Store warehouse location information
CREATE TABLE warehouses (
    warehouse_id INT AUTO_INCREMENT PRIMARY KEY,
    warehouse_name VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100) DEFAULT 'USA',
    capacity INT,
    current_utilization INT DEFAULT 0,
    manager_name VARCHAR(255),
    manager_contact VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: orders
-- Purpose: Store order header information
CREATE TABLE orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    order_date DATETIME NOT NULL,
    status ENUM('pending', 'processing', 'shipped', 'completed', 'cancelled') DEFAULT 'pending',
    total_amount DECIMAL(10, 2) NOT NULL,
    payment_status ENUM('unpaid', 'paid', 'failed', 'refunded', 'partial') DEFAULT 'unpaid',
    shipping_status ENUM('pending', 'processing', 'shipped', 'in_transit', 'delivered', 'exception') DEFAULT 'pending',
    shipping_address TEXT,
    billing_address TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: order_items
-- Purpose: Store individual line items for each order
CREATE TABLE order_items (
    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    total_price DECIMAL(10, 2) NOT NULL,
    discount DECIMAL(10, 2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: inventory
-- Purpose: Track inventory levels across warehouses
CREATE TABLE inventory (
    inventory_id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    warehouse_id INT NOT NULL,
    quantity_available INT DEFAULT 0,
    quantity_reserved INT DEFAULT 0,
    reorder_point INT DEFAULT 10,
    reorder_quantity INT DEFAULT 50,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_count_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id) ON DELETE CASCADE,
    UNIQUE KEY unique_product_warehouse (product_id, warehouse_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- SUPPORTING TABLES
-- ============================================================================

-- Table: vendors
-- Purpose: Store vendor/supplier information
CREATE TABLE vendors (
    vendor_id INT AUTO_INCREMENT PRIMARY KEY,
    vendor_name VARCHAR(255) NOT NULL,
    contact_name VARCHAR(255),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    category VARCHAR(100),
    rating DECIMAL(3, 2),
    status ENUM('active', 'inactive', 'pending', 'suspended') DEFAULT 'active',
    total_orders INT DEFAULT 0,
    on_time_delivery_pct DECIMAL(5, 2),
    quality_score DECIMAL(5, 2),
    response_time_hours INT,
    created_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: vendor_contracts
-- Purpose: Store vendor contract information
CREATE TABLE vendor_contracts (
    contract_id INT AUTO_INCREMENT PRIMARY KEY,
    vendor_id INT NOT NULL,
    contract_name VARCHAR(255),
    start_date DATE NOT NULL,
    end_date DATE,
    contract_value DECIMAL(12, 2),
    payment_terms VARCHAR(100),
    status ENUM('active', 'expired', 'terminated', 'pending') DEFAULT 'active',
    renewal_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: campaigns
-- Purpose: Store marketing campaign information
CREATE TABLE campaigns (
    campaign_id INT AUTO_INCREMENT PRIMARY KEY,
    campaign_name VARCHAR(255) NOT NULL,
    campaign_type VARCHAR(100),
    channel ENUM('Email', 'Social', 'Display', 'Search', 'Influencer', 'Other') NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    budget DECIMAL(12, 2),
    spent DECIMAL(12, 2) DEFAULT 0.00,
    target_audience TEXT,
    status ENUM('draft', 'active', 'paused', 'completed', 'cancelled') DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: campaign_performance
-- Purpose: Track daily performance metrics for campaigns
CREATE TABLE campaign_performance (
    performance_id INT AUTO_INCREMENT PRIMARY KEY,
    campaign_id INT NOT NULL,
    date DATE NOT NULL,
    impressions INT DEFAULT 0,
    clicks INT DEFAULT 0,
    conversions INT DEFAULT 0,
    revenue DECIMAL(12, 2) DEFAULT 0.00,
    cost DECIMAL(12, 2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id) ON DELETE CASCADE,
    UNIQUE KEY unique_campaign_date (campaign_id, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: returns
-- Purpose: Store product return information
CREATE TABLE returns (
    return_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    customer_id INT,
    return_date DATE NOT NULL,
    reason VARCHAR(255),
    status ENUM('pending', 'approved', 'processing', 'completed', 'rejected') DEFAULT 'pending',
    refund_amount DECIMAL(10, 2),
    condition_received VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: refunds
-- Purpose: Store refund transaction details
CREATE TABLE refunds (
    refund_id INT AUTO_INCREMENT PRIMARY KEY,
    return_id INT NOT NULL,
    order_id INT,
    refund_amount DECIMAL(10, 2) NOT NULL,
    refund_method VARCHAR(100),
    initiated_date DATETIME NOT NULL,
    completed_date DATETIME,
    processing_days INT,
    status ENUM('pending', 'processing', 'completed', 'failed', 'cancelled') DEFAULT 'pending',
    transaction_id VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (return_id) REFERENCES returns(return_id) ON DELETE CASCADE,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: payments
-- Purpose: Store payment transaction information
CREATE TABLE payments (
    payment_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    payment_processor VARCHAR(100),
    amount DECIMAL(10, 2) NOT NULL,
    transaction_id VARCHAR(100),
    payment_date DATETIME NOT NULL,
    status ENUM('pending', 'completed', 'failed', 'refunded', 'cancelled') DEFAULT 'pending',
    failure_reason VARCHAR(255),
    attempts INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: shipping
-- Purpose: Store shipment tracking information
CREATE TABLE shipping (
    shipment_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    carrier VARCHAR(100),
    tracking_number VARCHAR(100),
    shipped_date DATETIME,
    estimated_delivery_date DATE,
    delivered_date DATETIME,
    destination VARCHAR(255),
    shipping_address TEXT,
    shipping_cost DECIMAL(10, 2),
    weight DECIMAL(10, 2),
    dimensions VARCHAR(100),
    status ENUM('pending', 'picked_up', 'in_transit', 'out_for_delivery', 'delivered', 'delayed', 'exception') DEFAULT 'pending',
    address_validated BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: reviews
-- Purpose: Store product reviews from customers
CREATE TABLE reviews (
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    customer_id INT,
    order_id INT,
    rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_title VARCHAR(255),
    review_text TEXT,
    helpful_votes INT DEFAULT 0,
    review_date DATE NOT NULL,
    verified_purchase BOOLEAN DEFAULT FALSE,
    is_approved BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table: loyalty_program
-- Purpose: Store customer loyalty program membership details
CREATE TABLE loyalty_program (
    member_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL UNIQUE,
    tier ENUM('Bronze', 'Silver', 'Gold', 'Platinum') DEFAULT 'Bronze',
    points_balance INT DEFAULT 0,
    points_earned_lifetime INT DEFAULT 0,
    points_redeemed INT DEFAULT 0,
    join_date DATE NOT NULL,
    last_activity_date DATE,
    status ENUM('active', 'inactive', 'suspended') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- DISPLAY CONFIRMATION
-- ============================================================================

SELECT 'All tables created successfully!' AS Status;

-- Display table count
SELECT COUNT(*) AS 'Total Tables Created'
FROM information_schema.tables 
WHERE table_schema = 'ecommerce_analytics';

-- Display all tables
SELECT table_name AS 'Table Name', 
       table_rows AS 'Estimated Rows',
       ROUND((data_length + index_length) / 1024 / 1024, 2) AS 'Size (MB)'
FROM information_schema.tables 
WHERE table_schema = 'ecommerce_analytics'
ORDER BY table_name;
