-- =============================================
-- Reference Data - Lookup Tables
-- insert_lookup_data.sql
-- =============================================

-- Countries
INSERT INTO countries (country_code, country_name, continent) VALUES
('US', 'United States', 'North America'),
('CA', 'Canada', 'North America'),
('MX', 'Mexico', 'North America'),
('GB', 'United Kingdom', 'Europe'),
('DE', 'Germany', 'Europe'),
('FR', 'France', 'Europe'),
('IT', 'Italy', 'Europe'),
('ES', 'Spain', 'Europe'),
('AU', 'Australia', 'Oceania'),
('NZ', 'New Zealand', 'Oceania'),
('JP', 'Japan', 'Asia'),
('CN', 'China', 'Asia'),
('IN', 'India', 'Asia'),
('BR', 'Brazil', 'South America'),
('AR', 'Argentina', 'South America');

-- US States
INSERT INTO states (state_code, state_name, country_code) VALUES
('AL', 'Alabama', 'US'),
('AK', 'Alaska', 'US'),
('AZ', 'Arizona', 'US'),
('AR', 'Arkansas', 'US'),
('CA', 'California', 'US'),
('CO', 'Colorado', 'US'),
('CT', 'Connecticut', 'US'),
('DE', 'Delaware', 'US'),
('FL', 'Florida', 'US'),
('GA', 'Georgia', 'US'),
('HI', 'Hawaii', 'US'),
('ID', 'Idaho', 'US'),
('IL', 'Illinois', 'US'),
('IN', 'Indiana', 'US'),
('IA', 'Iowa', 'US'),
('KS', 'Kansas', 'US'),
('KY', 'Kentucky', 'US'),
('LA', 'Louisiana', 'US'),
('ME', 'Maine', 'US'),
('MD', 'Maryland', 'US'),
('MA', 'Massachusetts', 'US'),
('MI', 'Michigan', 'US'),
('MN', 'Minnesota', 'US'),
('MS', 'Mississippi', 'US'),
('MO', 'Missouri', 'US'),
('MT', 'Montana', 'US'),
('NE', 'Nebraska', 'US'),
('NV', 'Nevada', 'US'),
('NH', 'New Hampshire', 'US'),
('NJ', 'New Jersey', 'US'),
('NM', 'New Mexico', 'US'),
('NY', 'New York', 'US'),
('NC', 'North Carolina', 'US'),
('ND', 'North Dakota', 'US'),
('OH', 'Ohio', 'US'),
('OK', 'Oklahoma', 'US'),
('OR', 'Oregon', 'US'),
('PA', 'Pennsylvania', 'US'),
('RI', 'Rhode Island', 'US'),
('SC', 'South Carolina', 'US'),
('SD', 'South Dakota', 'US'),
('TN', 'Tennessee', 'US'),
('TX', 'Texas', 'US'),
('UT', 'Utah', 'US'),
('VT', 'Vermont', 'US'),
('VA', 'Virginia', 'US'),
('WA', 'Washington', 'US'),
('WV', 'West Virginia', 'US'),
('WI', 'Wisconsin', 'US'),
('WY', 'Wyoming', 'US');

-- Canadian Provinces
INSERT INTO states (state_code, state_name, country_code) VALUES
('AB', 'Alberta', 'CA'),
('BC', 'British Columbia', 'CA'),
('MB', 'Manitoba', 'CA'),
('NB', 'New Brunswick', 'CA'),
('NL', 'Newfoundland and Labrador', 'CA'),
('NS', 'Nova Scotia', 'CA'),
('ON', 'Ontario', 'CA'),
('PE', 'Prince Edward Island', 'CA'),
('QC', 'Quebec', 'CA'),
('SK', 'Saskatchewan', 'CA');

-- Product Categories
INSERT INTO product_categories (category_code, category_name, parent_category_id, description) VALUES
-- Level 1 - Main Categories
('ELEC', 'Electronics', NULL, 'Electronic devices and accessories'),
('CLOTH', 'Clothing', NULL, 'Apparel and fashion items'),
('HOME', 'Home & Garden', NULL, 'Home improvement and garden supplies'),
('SPORT', 'Sports & Outdoors', NULL, 'Sports equipment and outdoor gear'),
('BOOK', 'Books & Media', NULL, 'Books, music, movies, and games'),
('FOOD', 'Food & Grocery', NULL, 'Food items and groceries'),
('HEALTH', 'Health & Beauty', NULL, 'Healthcare and beauty products'),
('TOY', 'Toys & Games', NULL, 'Toys and gaming products'),
('AUTO', 'Automotive', NULL, 'Auto parts and accessories'),
('PET', 'Pet Supplies', NULL, 'Pet food and accessories');

-- Level 2 - Electronics Subcategories
INSERT INTO product_categories (category_code, category_name, parent_category_id, description) VALUES
('ELEC-COMP', 'Computers', (SELECT category_id FROM product_categories WHERE category_code = 'ELEC'), 'Desktop and laptop computers'),
('ELEC-PHON', 'Phones & Tablets', (SELECT category_id FROM product_categories WHERE category_code = 'ELEC'), 'Smartphones and tablets'),
('ELEC-AUDI', 'Audio', (SELECT category_id FROM product_categories WHERE category_code = 'ELEC'), 'Headphones, speakers, and audio equipment'),
('ELEC-CAME', 'Cameras', (SELECT category_id FROM product_categories WHERE category_code = 'ELEC'), 'Digital cameras and photography equipment'),
('ELEC-ACCE', 'Accessories', (SELECT category_id FROM product_categories WHERE category_code = 'ELEC'), 'Cables, chargers, and accessories');

-- Level 2 - Clothing Subcategories
INSERT INTO product_categories (category_code, category_name, parent_category_id, description) VALUES
('CLOTH-MEN', 'Mens Clothing', (SELECT category_id FROM product_categories WHERE category_code = 'CLOTH'), 'Clothing for men'),
('CLOTH-WOM', 'Womens Clothing', (SELECT category_id FROM product_categories WHERE category_code = 'CLOTH'), 'Clothing for women'),
('CLOTH-KIDS', 'Kids Clothing', (SELECT category_id FROM product_categories WHERE category_code = 'CLOTH'), 'Clothing for children'),
('CLOTH-SHOE', 'Shoes', (SELECT category_id FROM product_categories WHERE category_code = 'CLOTH'), 'Footwear for all ages'),
('CLOTH-ACCE', 'Accessories', (SELECT category_id FROM product_categories WHERE category_code = 'CLOTH'), 'Fashion accessories');

-- Product Conditions
INSERT INTO product_conditions (condition_code, condition_name, description) VALUES
('NEW', 'New', 'Brand new, unused product'),
('REFURB', 'Refurbished', 'Professionally restored to working condition'),
('USED-GOOD', 'Used - Good', 'Previously used, in good condition'),
('USED-FAIR', 'Used - Fair', 'Previously used, shows wear'),
('DAMAGED', 'Damaged', 'Has damage but may be functional');

-- Order Status Types
INSERT INTO order_statuses (status_code, status_name, description, is_active) VALUES
('PENDING', 'Pending', 'Order received, awaiting processing', TRUE),
('PROCESSING', 'Processing', 'Order is being prepared', TRUE),
('SHIPPED', 'Shipped', 'Order has been shipped', TRUE),
('DELIVERED', 'Delivered', 'Order has been delivered', FALSE),
('CANCELLED', 'Cancelled', 'Order was cancelled', FALSE),
('REFUNDED', 'Refunded', 'Order was refunded', FALSE),
('ON-HOLD', 'On Hold', 'Order is temporarily on hold', TRUE),
('FAILED', 'Failed', 'Order processing failed', FALSE);

-- Payment Methods
INSERT INTO payment_methods (method_code, method_name, is_active) VALUES
('CREDIT', 'Credit Card', TRUE),
('DEBIT', 'Debit Card', TRUE),
('PAYPAL', 'PayPal', TRUE),
('APPLE-PAY', 'Apple Pay', TRUE),
('GOOGLE-PAY', 'Google Pay', TRUE),
('BANK', 'Bank Transfer', TRUE),
('COD', 'Cash on Delivery', TRUE),
('CRYPTO', 'Cryptocurrency', FALSE);

-- Shipping Methods
INSERT INTO shipping_methods (method_code, method_name, estimated_days, base_cost) VALUES
('STANDARD', 'Standard Shipping', 7, 5.99),
('EXPRESS', 'Express Shipping', 3, 12.99),
('OVERNIGHT', 'Overnight', 1, 24.99),
('TWO-DAY', 'Two Day Shipping', 2, 15.99),
('ECONOMY', 'Economy Shipping', 10, 3.99),
('PICKUP', 'Store Pickup', 0, 0.00);

-- Customer Segments
INSERT INTO customer_segments (segment_code, segment_name, description, min_orders, min_revenue) VALUES
('NEW', 'New Customer', 'Less than 2 orders', 0, 0.00),
('REGULAR', 'Regular Customer', '2-10 orders', 2, 100.00),
('LOYAL', 'Loyal Customer', '11-50 orders', 11, 1000.00),
('VIP', 'VIP Customer', 'Over 50 orders or $5000+ revenue', 51, 5000.00),
('INACTIVE', 'Inactive Customer', 'No orders in 12+ months', 0, 0.00),
('AT-RISK', 'At Risk', 'Declining purchase frequency', 0, 0.00);

-- Campaign Types
INSERT INTO campaign_types (type_code, type_name, description) VALUES
('EMAIL', 'Email Campaign', 'Marketing via email'),
('SMS', 'SMS Campaign', 'Marketing via text message'),
('SOCIAL', 'Social Media', 'Marketing via social media platforms'),
('DISPLAY', 'Display Ads', 'Banner and display advertising'),
('SEARCH', 'Search Ads', 'Search engine advertising'),
('AFFILIATE', 'Affiliate', 'Affiliate marketing programs'),
('REFERRAL', 'Referral', 'Customer referral programs');

-- Vendor Categories
INSERT INTO vendor_categories (category_code, category_name, description) VALUES
('MANUFACTURER', 'Manufacturer', 'Direct product manufacturers'),
('WHOLESALER', 'Wholesaler', 'Bulk product distributors'),
('DISTRIBUTOR', 'Distributor', 'Regional product distributors'),
('DROPSHIP', 'Dropshipper', 'Direct shipping vendors'),
('SUPPLIER', 'Supplier', 'General suppliers');


-- Display confirmation
SELECT 'Lookup data inserted successfully' AS Status;
SELECT COUNT(*) AS total_categories FROM product_categories;

COMMIT;