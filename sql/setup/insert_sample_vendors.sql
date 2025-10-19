-- ========================================
-- INSERT SAMPLE VENDORS & CONTRACTS
-- 50 vendor records with contracts
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- VENDORS (50 records)
-- ========================================

INSERT INTO vendors (vendor_name, contact_person, email, phone, address, city, state, zip_code, rating, status, created_at) VALUES
('TechSupply Global', 'John Martinez', 'john.m@techsupply.com', '555-1001', '1500 Tech Parkway', 'San Jose', 'CA', '95110', 4.75, 'active', '2023-01-15'),
('ElectroWholesale Inc', 'Sarah Chen', 'schen@electrowholesale.com', '555-1002', '2100 Commerce Dr', 'Seattle', 'WA', '98101', 4.50, 'active', '2023-01-20'),
('Fashion Forward Distributors', 'Michael Thompson', 'mthompson@fashionforward.com', '555-1003', '350 Fashion Ave', 'New York', 'NY', '10018', 4.25, 'active', '2023-02-01'),
('Home Essentials Supply', 'Emily Rodriguez', 'erodriguez@homeessentials.com', '555-1004', '4500 Industrial Blvd', 'Chicago', 'IL', '60601', 4.60, 'active', '2023-02-10'),
('SportGear Wholesale', 'David Kim', 'dkim@sportgear.com', '555-1005', '875 Athletic Way', 'Portland', 'OR', '97201', 4.40, 'active', '2023-02-15'),
('Beauty Products International', 'Lisa Wang', 'lwang@beautyprod.com', '555-1006', '225 Cosmetics Lane', 'Los Angeles', 'CA', '90015', 4.80, 'active', '2023-03-01'),
('AudioTech Suppliers', 'James Anderson', 'janderson@audiotech.com', '555-1007', '1800 Sound Circle', 'Nashville', 'TN', '37201', 4.55, 'active', '2023-03-10'),
('Footwear Direct', 'Maria Garcia', 'mgarcia@footweardirect.com', '555-1008', '950 Shoe District', 'Boston', 'MA', '02101', 4.35, 'active', '2023-03-15'),
('Kitchen Innovations Ltd', 'Robert Lee', 'rlee@kitcheninnovations.com', '555-1009', '3200 Culinary Dr', 'Dallas', 'TX', '75201', 4.70, 'active', '2023-04-01'),
('Outdoor Adventure Supply', 'Jennifer White', 'jwhite@outdooradv.com', '555-1010', '680 Trail Head Rd', 'Denver', 'CO', '80201', 4.45, 'active', '2023-04-10'),
('Global Apparel Sourcing', 'Kevin Brown', 'kbrown@globalapparel.com', '555-1011', '1250 Textile Way', 'Atlanta', 'GA', '30301', 4.20, 'active', '2023-04-20'),
('Premium Electronics Corp', 'Amanda Miller', 'amiller@premiumelec.com', '555-1012', '4100 Circuit Ave', 'Austin', 'TX', '78701', 4.85, 'active', '2023-05-01'),
('Fitness Equipment Pros', 'Daniel Taylor', 'dtaylor@fitnesspros.com', '555-1013', '725 Wellness Blvd', 'Phoenix', 'AZ', '85001', 4.30, 'active', '2023-05-15'),
('Luxury Home Furnishings', 'Michelle Johnson', 'mjohnson@luxuryhome.com', '555-1014', '1900 Designer St', 'Miami', 'FL', '33101', 4.65, 'active', '2023-06-01'),
('Smart Gadgets Warehouse', 'Christopher Davis', 'cdavis@smartgadgets.com', '555-1015', '3500 Innovation Dr', 'San Francisco', 'CA', '94101', 4.75, 'active', '2023-06-10'),
('Clothing Manufacturers United', 'Jessica Wilson', 'jwilson@clothingmfg.com', '555-1016', '800 Garment District', 'New York', 'NY', '10001', 4.15, 'active', '2023-06-20'),
('Health & Wellness Distributors', 'Matthew Moore', 'mmoore@healthwellness.com', '555-1017', '2750 Vitality Lane', 'Minneapolis', 'MN', '55401', 4.50, 'active', '2023-07-01'),
('Tech Accessories Plus', 'Ashley Thomas', 'athomas@techaccessories.com', '555-1018', '1400 Gadget Plaza', 'Seattle', 'WA', '98102', 4.40, 'active', '2023-07-15'),
('Sustainable Products Co', 'Ryan Jackson', 'rjackson@sustainable.com', '555-1019', '550 Green Commerce Way', 'Portland', 'OR', '97202', 4.90, 'active', '2023-08-01'),
('International Home Goods', 'Nicole Martin', 'nmartin@inthomegoods.com', '555-1020', '4200 Import Blvd', 'Long Beach', 'CA', '90801', 4.25, 'active', '2023-08-10'),
('Premium Leather Goods', 'Brandon Lee', 'blee@premiumleather.com', '555-1021', '1100 Tannery Rd', 'Nashville', 'TN', '37202', 4.55, 'active', '2023-08-20'),
('Camera & Photo Supply', 'Stephanie Harris', 'sharris@cameraphoto.com', '555-1022', '2200 Lens Ave', 'Denver', 'CO', '80202', 4.70, 'active', '2023-09-01'),
('Quality Kitchenware', 'Justin Clark', 'jclark@qualitykitchen.com', '555-1023', '900 Cookware Circle', 'Chicago', 'IL', '60602', 4.60, 'active', '2023-09-10'),
('Active Lifestyle Products', 'Megan Lewis', 'mlewis@activelifestyle.com', '555-1024', '1650 Sports Complex Dr', 'Phoenix', 'AZ', '85002', 4.35, 'active', '2023-09-20'),
('Digital Solutions Wholesale', 'Tyler Walker', 'twalker@digitalsolutions.com', '555-1025', '3300 Tech Hub Pkwy', 'Austin', 'TX', '78702', 4.80, 'active', '2023-10-01'),
('Comfort Footwear Suppliers', 'Rachel Hall', 'rhall@comfortfootwear.com', '555-1026', '750 Stride St', 'Boston', 'MA', '02102', 4.45, 'active', '2023-10-10'),
('Modern Home Decor', 'Eric Allen', 'eallen@modernhomedecor.com', '555-1027', '2900 Design District', 'Los Angeles', 'CA', '90016', 4.50, 'active', '2023-10-20'),
('Performance Sports Gear', 'Amber Young', 'ayoung@performsports.com', '555-1028', '1800 Athletic Complex', 'Seattle', 'WA', '98103', 4.40, 'active', '2023-11-01'),
('Vintage Apparel Collective', 'Jordan King', 'jking@vintageapparel.com', '555-1029', '425 Retro Plaza', 'Portland', 'OR', '97203', 3.95, 'active', '2023-11-10'),
('Eco-Friendly Products Hub', 'Taylor Wright', 'twright@ecofriendly.com', '555-1030', '1575 Sustainability Ave', 'San Francisco', 'CA', '94102', 4.85, 'active', '2023-11-20'),
('Professional Office Supply', 'Morgan Scott', 'mscott@prooffice.com', '555-1031', '3100 Corporate Dr', 'Dallas', 'TX', '75202', 4.30, 'active', '2023-12-01'),
('Urban Living Essentials', 'Casey Green', 'cgreen@urbanliving.com', '555-1032', '2400 Metropolitan Blvd', 'Miami', 'FL', '33102', 4.20, 'active', '2023-12-10'),
('Advanced Audio Systems', 'Jordan Baker', 'jbaker@advancedaudio.com', '555-1033', '875 Acoustic Way', 'Nashville', 'TN', '37203', 4.75, 'active', '2023-12-15'),
('Global Fashion Imports', 'Alex Adams', 'aadams@globalfashion.com', '555-1034', '1950 Import Plaza', 'New York', 'NY', '10002', 4.10, 'active', '2024-01-05'),
('Wellness & Beauty Direct', 'Jamie Nelson', 'jnelson@wellnessbeauty.com', '555-1035', '650 Spa District', 'Los Angeles', 'CA', '90017', 4.65, 'active', '2024-01-15'),
('Innovative Kitchen Tech', 'Drew Carter', 'dcarter@innovkitchen.com', '555-1036', '2100 Culinary Tech Dr', 'Chicago', 'IL', '60603', 4.55, 'active', '2024-01-25'),
('Outdoor Exploration Supply', 'Riley Mitchell', 'rmitchell@outdoorexplore.com', '555-1037', '1300 Wilderness Rd', 'Denver', 'CO', '80203', 4.70, 'active', '2024-02-01'),
('Smart Home Technologies', 'Avery Perez', 'aperez@smarthometech.com', '555-1038', '3800 Automation Pkwy', 'San Jose', 'CA', '95111', 4.90, 'active', '2024-02-10'),
('Classic Furniture Makers', 'Cameron Roberts', 'croberts@classicfurniture.com', '555-1039', '950 Craftsman Ave', 'Atlanta', 'GA', '30302', 4.25, 'active', '2024-02-15'),
('Athletic Performance Co', 'Morgan Turner', 'mturner@athleticperf.com', '555-1040', '2600 Training Facility Rd', 'Phoenix', 'AZ', '85003', 4.45, 'active', '2024-02-20'),
('Premium Textiles Group', 'Skylar Phillips', 'sphillips@premiumtextiles.com', '555-1041', '1750 Fabric Row', 'Boston', 'MA', '02103', 4.35, 'active', '2024-03-01'),
('Modern Electronics Hub', 'Quinn Campbell', 'qcampbell@modernelec.com', '555-1042', '4500 Circuit Plaza', 'Austin', 'TX', '78703', 4.80, 'active', '2024-03-10'),
('Lifestyle Brands Collective', 'Sage Parker', 'sparker@lifestylebrands.com', '555-1043', '820 Boutique District', 'Miami', 'FL', '33103', 4.15, 'active', '2024-03-15'),
('Industrial Supply Partners', 'Reese Evans', 'reevans@industrialsupply.com', '555-1044', '3400 Warehouse Way', 'Seattle', 'WA', '98104', 4.50, 'active', '2024-03-20'),
('Gourmet Kitchen Solutions', 'Harper Edwards', 'hedwards@gourmetkitchen.com', '555-1045', '1200 Chef''s Plaza', 'San Francisco', 'CA', '94103', 4.75, 'active', '2024-03-25'),
('Travel Gear Specialists', 'Dakota Collins', 'dcollins@travelgear.com', '555-1046', '2850 Luggage Lane', 'Denver', 'CO', '80204', 4.40, 'active', '2024-04-01'),
('Luxury Living Imports', 'River Stewart', 'rstewart@luxuryliving.com', '555-1047', '1500 Elite Ave', 'Los Angeles', 'CA', '90018', 4.60, 'active', '2024-04-10'),
('Tech Innovations Group', 'Phoenix Morris', 'pmorris@techinnovations.com', '555-1048', '3950 Future Tech Blvd', 'San Jose', 'CA', '95112', 4.85, 'active', '2024-04-15'),
('Budget Home Essentials', 'Charlie Rogers', 'crogers@budgethome.com', '555-1049', '700 Value Plaza', 'Chicago', 'IL', '60604', 3.85, 'active', '2024-04-20'),
('Premium Beauty Brands', 'Rowan Reed', 'rreed@premiumbeauty.com', '555-1050', '1050 Luxury Cosmetics Dr', 'New York', 'NY', '10003', 4.70, 'active', '2024-04-25');

-- ========================================
-- VENDOR CONTRACTS
-- Create 150-200 contracts linking vendors to products
-- ========================================

-- Electronics vendors (Vendors 1, 2, 12, 15, 18, 25, 42, 48)
INSERT INTO vendor_contracts (vendor_id, product_id, cost_per_unit, minimum_order_quantity, lead_time_days, contract_terms, start_date, end_date, status)
SELECT 
    ELT(MOD(p.product_id, 8) + 1, 1, 2, 12, 15, 18, 25, 42, 48) AS vendor_id,
    p.product_id,
    p.cost AS cost_per_unit,
    CASE 
        WHEN p.price > 1000 THEN 5
        WHEN p.price > 500 THEN 10
        WHEN p.price > 100 THEN 25
        ELSE 50
    END AS minimum_order_quantity,
    FLOOR(7 + (RAND() * 21)) AS lead_time_days,
    'Standard wholesale agreement with volume discounts available' AS contract_terms,
    DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 365) DAY) AS start_date,
    DATE_ADD(CURDATE(), INTERVAL 180 + FLOOR(RAND() * 365) DAY) AS end_date,
    'active' AS status
FROM products p
WHERE p.category_id IN (10, 11, 12, 13, 14);

-- Clothing vendors (Vendors 3, 11, 16, 29, 34, 41)
INSERT INTO vendor_contracts (vendor_id, product_id, cost_per_unit, minimum_order_quantity, lead_time_days, contract_terms, start_date, end_date, status)
SELECT 
    ELT(MOD(p.product_id, 6) + 1, 3, 11, 16, 29, 34, 41) AS vendor_id,
    p.product_id,
    p.cost AS cost_per_unit,
    CASE 
        WHEN p.price > 200 THEN 10
        WHEN p.price > 100 THEN 20
        ELSE 50
    END AS minimum_order_quantity,
    FLOOR(14 + (RAND() * 28)) AS lead_time_days,
    'Seasonal collection agreement with flexible order quantities' AS contract_terms,
    DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 365) DAY) AS start_date,
    DATE_ADD(CURDATE(), INTERVAL 180 + FLOOR(RAND() * 365) DAY) AS end_date,
    'active' AS status
FROM products p
WHERE p.category_id IN (15, 16, 17, 18);

-- Home & Kitchen vendors (Vendors 4, 9, 14, 20, 23, 27, 36, 39, 45, 49)
INSERT INTO vendor_contracts (vendor_id, product_id, cost_per_unit, minimum_order_quantity, lead_time_days, contract_terms, start_date, end_date, status)
SELECT 
    ELT(MOD(p.product_id, 10) + 1, 4, 9, 14, 20, 23, 27, 36, 39, 45, 49) AS vendor_id,
    p.product_id,
    p.cost AS cost_per_unit,
    CASE 
        WHEN p.price > 500 THEN 5
        WHEN p.price > 200 THEN 15
        WHEN p.price > 100 THEN 25
        ELSE 40
    END AS minimum_order_quantity,
    FLOOR(10 + (RAND() * 25)) AS lead_time_days,
    'Annual supply agreement with warranty coverage' AS contract_terms,
    DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 365) DAY) AS start_date,
    DATE_ADD(CURDATE(), INTERVAL 180 + FLOOR(RAND() * 365) DAY) AS end_date,
    'active' AS status
FROM products p
WHERE p.category_id IN (19, 20, 21, 22);

-- Sports & Fitness vendors (Vendors 5, 10, 13, 24, 28, 37, 40, 46)
INSERT INTO vendor_contracts (vendor_id, product_id, cost_per_unit, minimum_order_quantity, lead_time_days, contract_terms, start_date, end_date, status)
SELECT 
    ELT(MOD(p.product_id, 8) + 1, 5, 10, 13, 24, 28, 37, 40, 46) AS vendor_id,
    p.product_id,
    p.cost AS cost_per_unit,
    CASE 
        WHEN p.price > 1000 THEN 3
        WHEN p.price > 300 THEN 10
        WHEN p.price > 100 THEN 20
        ELSE 30
    END AS minimum_order_quantity,
    FLOOR(7 + (RAND() * 21)) AS lead_time_days,
    'Performance equipment supply contract with quality guarantees' AS contract_terms,
    DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 365) DAY) AS start_date,
    DATE_ADD(CURDATE(), INTERVAL 180 + FLOOR(RAND() * 365) DAY) AS end_date,
    'active' AS status
FROM products p
WHERE p.category_id IN (23, 24, 25);

-- Health & Beauty vendors (Vendors 6, 17, 35, 47, 50)
INSERT INTO vendor_contracts (vendor_id, product_id, cost_per_unit, minimum_order_quantity, lead_time_days, contract_terms, start_date, end_date, status)
SELECT 
    ELT(MOD(p.product_id, 5) + 1, 6, 17, 35, 47, 50) AS vendor_id,
    p.product_id,
    p.cost AS cost_per_unit,
    CASE 
        WHEN p.price > 300 THEN 5
        WHEN p.price > 100 THEN 15
        WHEN p.price > 50 THEN 30
        ELSE 60
    END AS minimum_order_quantity,
    FLOOR(5 + (RAND() * 15)) AS lead_time_days,
    'Beauty and wellness products supply agreement' AS contract_terms,
    DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 365) DAY) AS start_date,
    DATE_ADD(CURDATE(), INTERVAL 180 + FLOOR(RAND() * 365) DAY) AS end_date,
    'active' AS status
FROM products p
WHERE p.category_id = 6;

-- Create some expired contracts (5%)
UPDATE vendor_contracts 
SET status = 'expired',
    end_date = DATE_SUB(CURDATE(), INTERVAL FLOOR(1 + RAND() * 90) DAY)
WHERE contract_id % 20 = 0;

-- Create some terminated contracts (2%)
UPDATE vendor_contracts 
SET status = 'terminated',
    end_date = DATE_SUB(CURDATE(), INTERVAL FLOOR(1 + RAND() * 180) DAY)
WHERE contract_id % 50 = 0;

-- Set a few vendors to inactive (3 vendors)
UPDATE vendors 
SET status = 'inactive'
WHERE vendor_id IN (29, 43, 49);

-- Set one vendor to blacklisted
UPDATE vendors 
SET status = 'blacklisted',
    rating = 2.5
WHERE vendor_id = 49;

-- ========================================
-- DISPLAY CONFIRMATION & STATISTICS
-- ========================================

SELECT 'Vendors and contracts inserted successfully!' AS Status;

SELECT 
    COUNT(*) AS total_vendors,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_vendors,
    SUM(CASE WHEN status = 'inactive' THEN 1 ELSE 0 END) AS inactive_vendors,
    SUM(CASE WHEN status = 'blacklisted' THEN 1 ELSE 0 END) AS blacklisted_vendors,
    ROUND(AVG(rating), 2) AS avg_vendor_rating
FROM vendors;

SELECT 'Vendor contracts summary:' AS Info;
SELECT 
    COUNT(*) AS total_contracts,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_contracts,
    SUM(CASE WHEN status = 'expired' THEN 1 ELSE 0 END) AS expired_contracts,
    SUM(CASE WHEN status = 'terminated' THEN 1 ELSE 0 END) AS terminated_contracts,
    ROUND(AVG(cost_per_unit), 2) AS avg_cost_per_unit,
    ROUND(AVG(lead_time_days), 0) AS avg_lead_time_days
FROM vendor_contracts;

SELECT 'Top 10 vendors by contract count:' AS Info;
SELECT 
    v.vendor_id,
    v.vendor_name,
    v.rating,
    v.status,
    COUNT(vc.contract_id) AS active_contracts
FROM vendors v
LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id AND vc.status = 'active'
GROUP BY v.vendor_id, v.vendor_name, v.rating, v.status
ORDER BY active_contracts DESC
LIMIT 10;

SELECT 'Vendors by rating distribution:' AS Info;
SELECT 
    CASE 
        WHEN rating >= 4.5 THEN '4.5 - 5.0 (Excellent)'
        WHEN rating >= 4.0 THEN '4.0 - 4.49 (Very Good)'
        WHEN rating >= 3.5 THEN '3.5 - 3.99 (Good)'
        ELSE 'Below 3.5 (Poor)'
    END AS rating_range,
    COUNT(*) AS vendor_count
FROM vendors
GROUP BY 
    CASE 
        WHEN rating >= 4.5 THEN '4.5 - 5.0 (Excellent)'
        WHEN rating >= 4.0 THEN '4.0 - 4.49 (Very Good)'
        WHEN rating >= 3.5 THEN '3.5 - 3.99 (Good)'
        ELSE 'Below 3.5 (Poor)'
    END
ORDER BY MIN(rating) DESC;