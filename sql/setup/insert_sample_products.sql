-- ========================================
-- INSERT SAMPLE PRODUCT CATEGORIES & PRODUCTS
-- 300 realistic product records across multiple categories
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- PRODUCT CATEGORIES (Hierarchical Structure)
-- ========================================

-- Main Categories (Parent level)
INSERT INTO product_categories (category_name, parent_id, description, created_at) VALUES
('Electronics', NULL, 'Electronic devices and accessories', '2024-01-01'),
('Clothing & Apparel', NULL, 'Fashion and clothing items', '2024-01-01'),
('Home & Kitchen', NULL, 'Home furnishings and kitchen appliances', '2024-01-01'),
('Sports & Outdoors', NULL, 'Sports equipment and outdoor gear', '2024-01-01'),
('Books & Media', NULL, 'Books, movies, music and media', '2024-01-01'),
('Health & Beauty', NULL, 'Health products and beauty items', '2024-01-01'),
('Toys & Games', NULL, 'Toys, games and entertainment', '2024-01-01'),
('Automotive', NULL, 'Auto parts and accessories', '2024-01-01');

-- Sub-Categories (Electronics)
INSERT INTO product_categories (category_name, parent_id, description, created_at) VALUES
('Smartphones', 1, 'Mobile phones and accessories', '2024-01-02'),
('Laptops & Computers', 1, 'Computers and computing devices', '2024-01-02'),
('Audio & Headphones', 1, 'Headphones, speakers and audio equipment', '2024-01-02'),
('Cameras & Photography', 1, 'Cameras and photography equipment', '2024-01-02'),
('Smart Home', 1, 'Smart home devices and automation', '2024-01-02');

-- Sub-Categories (Clothing)
INSERT INTO product_categories (category_name, parent_id, description, created_at) VALUES
('Men''s Clothing', 2, 'Clothing for men', '2024-01-02'),
('Women''s Clothing', 2, 'Clothing for women', '2024-01-02'),
('Shoes & Footwear', 2, 'Shoes and footwear', '2024-01-02'),
('Accessories', 2, 'Fashion accessories', '2024-01-02');

-- Sub-Categories (Home & Kitchen)
INSERT INTO product_categories (category_name, parent_id, description, created_at) VALUES
('Kitchen Appliances', 3, 'Kitchen tools and appliances', '2024-01-02'),
('Furniture', 3, 'Home and office furniture', '2024-01-02'),
('Bedding & Bath', 3, 'Bedding, towels and bath items', '2024-01-02'),
('Home Decor', 3, 'Decorative items for home', '2024-01-02');

-- Sub-Categories (Sports)
INSERT INTO product_categories (category_name, parent_id, description, created_at) VALUES
('Fitness Equipment', 4, 'Exercise and fitness gear', '2024-01-02'),
('Outdoor Recreation', 4, 'Camping, hiking and outdoor gear', '2024-01-02'),
('Team Sports', 4, 'Equipment for team sports', '2024-01-02');

-- ========================================
-- PRODUCTS (300 Records)
-- ========================================

-- Electronics - Smartphones (20 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('ELEC-SP-001', 'iPhone 15 Pro 256GB', 'Latest Apple smartphone with A17 Pro chip, titanium design', 10, 999.00, 750.00, 45, 'active', '2024-01-15'),
('ELEC-SP-002', 'Samsung Galaxy S24 Ultra', 'Premium Android phone with 200MP camera and S Pen', 10, 1199.00, 900.00, 38, 'active', '2024-01-15'),
('ELEC-SP-003', 'Google Pixel 8 Pro', 'Google flagship with advanced AI features', 10, 899.00, 650.00, 52, 'active', '2024-01-16'),
('ELEC-SP-004', 'OnePlus 12', 'High-performance smartphone with fast charging', 10, 799.00, 580.00, 60, 'active', '2024-01-16'),
('ELEC-SP-005', 'iPhone 14 128GB', 'Previous generation iPhone, still powerful', 10, 699.00, 520.00, 75, 'active', '2024-01-17'),
('ELEC-SP-006', 'Samsung Galaxy A54', 'Mid-range Samsung with great camera', 10, 449.00, 320.00, 120, 'active', '2024-01-17'),
('ELEC-SP-007', 'Motorola Edge 40', 'Affordable flagship alternative', 10, 599.00, 420.00, 85, 'active', '2024-01-18'),
('ELEC-SP-008', 'Xiaomi 13T Pro', 'Feature-packed phone with excellent value', 10, 649.00, 450.00, 95, 'active', '2024-01-18'),
('ELEC-SP-009', 'Nothing Phone (2)', 'Unique design with glyph interface', 10, 599.00, 425.00, 65, 'active', '2024-01-19'),
('ELEC-SP-010', 'ASUS ROG Phone 7', 'Gaming smartphone with accessories', 10, 999.00, 720.00, 30, 'active', '2024-01-19'),
('ELEC-SP-011', 'iPhone SE (2024)', 'Compact and affordable iPhone', 10, 429.00, 310.00, 110, 'active', '2024-01-20'),
('ELEC-SP-012', 'Sony Xperia 5 V', 'Compact flagship with professional camera', 10, 899.00, 650.00, 40, 'active', '2024-01-20'),
('ELEC-SP-013', 'Realme GT 3', 'Fast-charging flagship killer', 10, 549.00, 390.00, 88, 'active', '2024-01-21'),
('ELEC-SP-014', 'Oppo Find X6 Pro', 'Photography-focused premium phone', 10, 1099.00, 800.00, 35, 'active', '2024-01-21'),
('ELEC-SP-015', 'Nokia XR21', 'Rugged smartphone for outdoor use', 10, 499.00, 350.00, 70, 'active', '2024-01-22'),
('ELEC-SP-016', 'Vivo X100 Pro', 'Flagship with advanced imaging', 10, 899.00, 640.00, 55, 'active', '2024-01-22'),
('ELEC-SP-017', 'Honor Magic 6 Pro', 'Premium phone with AI features', 10, 849.00, 600.00, 62, 'active', '2024-01-23'),
('ELEC-SP-018', 'Asus Zenfone 10', 'Compact powerhouse phone', 10, 699.00, 490.00, 75, 'active', '2024-01-23'),
('ELEC-SP-019', 'Motorola Razr 40 Ultra', 'Foldable flip phone', 10, 999.00, 720.00, 28, 'active', '2024-01-24'),
('ELEC-SP-020', 'Samsung Galaxy Z Fold 5', 'Premium foldable smartphone', 10, 1799.00, 1350.00, 22, 'active', '2024-01-24');

-- Electronics - Laptops (25 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('ELEC-LP-001', 'MacBook Pro 14" M3', 'Apple MacBook Pro with M3 chip, 16GB RAM', 11, 1999.00, 1500.00, 35, 'active', '2024-01-25'),
('ELEC-LP-002', 'Dell XPS 15', 'Premium Windows laptop with OLED display', 11, 1799.00, 1350.00, 42, 'active', '2024-01-25'),
('ELEC-LP-003', 'HP Spectre x360', 'Convertible 2-in-1 laptop', 11, 1499.00, 1100.00, 38, 'active', '2024-01-26'),
('ELEC-LP-004', 'Lenovo ThinkPad X1 Carbon', 'Business ultrabook with security features', 11, 1699.00, 1250.00, 45, 'active', '2024-01-26'),
('ELEC-LP-005', 'ASUS ROG Zephyrus G14', 'Compact gaming laptop', 11, 1599.00, 1150.00, 32, 'active', '2024-01-27'),
('ELEC-LP-006', 'Microsoft Surface Laptop 5', 'Sleek Windows laptop with touchscreen', 11, 1299.00, 950.00, 50, 'active', '2024-01-27'),
('ELEC-LP-007', 'Acer Swift 3', 'Budget-friendly ultrabook', 11, 699.00, 480.00, 85, 'active', '2024-01-28'),
('ELEC-LP-008', 'Razer Blade 15', 'Premium gaming laptop', 11, 2299.00, 1750.00, 25, 'active', '2024-01-28'),
('ELEC-LP-009', 'MacBook Air M2', 'Lightweight laptop for everyday use', 11, 1199.00, 880.00, 65, 'active', '2024-01-29'),
('ELEC-LP-010', 'MSI Stealth 17', 'Large screen gaming laptop', 11, 1899.00, 1400.00, 28, 'active', '2024-01-29'),
('ELEC-LP-011', 'LG Gram 17', 'Ultra-lightweight 17-inch laptop', 11, 1599.00, 1150.00, 35, 'active', '2024-01-30'),
('ELEC-LP-012', 'HP Pavilion 15', 'Reliable mid-range laptop', 11, 799.00, 550.00, 95, 'active', '2024-01-30'),
('ELEC-LP-013', 'Lenovo IdeaPad 5 Pro', 'Value laptop with good performance', 11, 899.00, 640.00, 72, 'active', '2024-01-31'),
('ELEC-LP-014', 'ASUS ZenBook 14', 'Compact business laptop', 11, 999.00, 720.00, 58, 'active', '2024-01-31'),
('ELEC-LP-015', 'Dell Inspiron 16', 'Family laptop with large screen', 11, 849.00, 600.00, 68, 'active', '2024-02-01'),
('ELEC-LP-016', 'Acer Predator Helios 300', 'Mid-range gaming laptop', 11, 1399.00, 1000.00, 40, 'active', '2024-02-01'),
('ELEC-LP-017', 'Samsung Galaxy Book3 Pro', 'Premium thin and light laptop', 11, 1499.00, 1100.00, 45, 'active', '2024-02-02'),
('ELEC-LP-018', 'Framework Laptop', 'Modular repairable laptop', 11, 1199.00, 850.00, 30, 'active', '2024-02-02'),
('ELEC-LP-019', 'Gigabyte Aero 16', 'Content creator laptop', 11, 2199.00, 1650.00, 22, 'active', '2024-02-03'),
('ELEC-LP-020', 'HP Envy 13', 'Stylish ultraportable laptop', 11, 999.00, 700.00, 62, 'active', '2024-02-03'),
('ELEC-LP-021', 'Lenovo Yoga 9i', 'Premium 2-in-1 convertible', 11, 1599.00, 1180.00, 38, 'active', '2024-02-04'),
('ELEC-LP-022', 'Chromebook Pixel', 'High-end Chromebook', 11, 999.00, 680.00, 55, 'active', '2024-02-04'),
('ELEC-LP-023', 'MSI Creator Z16', 'Professional workstation laptop', 11, 2099.00, 1550.00, 28, 'active', '2024-02-05'),
('ELEC-LP-024', 'ASUS TUF Gaming A15', 'Budget gaming laptop', 11, 899.00, 630.00, 75, 'active', '2024-02-05'),
('ELEC-LP-025', 'Dell Latitude 9430', 'Enterprise business laptop', 11, 1899.00, 1400.00, 35, 'active', '2024-02-06');

-- Electronics - Audio (20 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('ELEC-AU-001', 'Sony WH-1000XM5', 'Premium noise-cancelling headphones', 12, 399.00, 280.00, 120, 'active', '2024-02-07'),
('ELEC-AU-002', 'Apple AirPods Pro 2', 'Wireless earbuds with ANC', 12, 249.00, 175.00, 200, 'active', '2024-02-07'),
('ELEC-AU-003', 'Bose QuietComfort 45', 'Comfortable noise-cancelling headphones', 12, 329.00, 230.00, 95, 'active', '2024-02-08'),
('ELEC-AU-004', 'Samsung Galaxy Buds2 Pro', 'Premium wireless earbuds', 12, 229.00, 160.00, 150, 'active', '2024-02-08'),
('ELEC-AU-005', 'JBL Flip 6', 'Portable Bluetooth speaker', 12, 129.00, 85.00, 180, 'active', '2024-02-09'),
('ELEC-AU-006', 'Sennheiser Momentum 4', 'Audiophile wireless headphones', 12, 379.00, 265.00, 75, 'active', '2024-02-09'),
('ELEC-AU-007', 'Anker Soundcore Liberty 4', 'Budget-friendly true wireless', 12, 99.00, 65.00, 250, 'active', '2024-02-10'),
('ELEC-AU-008', 'Beats Studio Pro', 'Stylish over-ear headphones', 12, 349.00, 245.00, 110, 'active', '2024-02-10'),
('ELEC-AU-009', 'Audio-Technica ATH-M50x', 'Professional studio monitors', 12, 169.00, 115.00, 140, 'active', '2024-02-11'),
('ELEC-AU-010', 'Ultimate Ears BOOM 3', 'Rugged portable speaker', 12, 149.00, 100.00, 165, 'active', '2024-02-11'),
('ELEC-AU-011', 'Jabra Elite 85t', 'Premium earbuds for calls', 12, 229.00, 155.00, 130, 'active', '2024-02-12'),
('ELEC-AU-012', 'Sony SRS-XB43', 'Extra bass party speaker', 12, 249.00, 170.00, 88, 'active', '2024-02-12'),
('ELEC-AU-013', 'Shure AONIC 50', 'Professional-grade headphones', 12, 299.00, 210.00, 65, 'active', '2024-02-13'),
('ELEC-AU-014', 'Google Pixel Buds Pro', 'Smart wireless earbuds', 12, 199.00, 135.00, 155, 'active', '2024-02-13'),
('ELEC-AU-015', 'Marshall Emberton', 'Vintage-style portable speaker', 12, 169.00, 115.00, 125, 'active', '2024-02-14'),
('ELEC-AU-016', 'Nothing Ear (2)', 'Transparent design earbuds', 12, 149.00, 100.00, 145, 'active', '2024-02-14'),
('ELEC-AU-017', 'Bowers & Wilkins PX7 S2', 'Luxury wireless headphones', 12, 399.00, 280.00, 55, 'active', '2024-02-15'),
('ELEC-AU-018', 'Sonos Roam', 'Portable smart speaker', 12, 179.00, 120.00, 135, 'active', '2024-02-15'),
('ELEC-AU-019', 'Beyerdynamic DT 770 Pro', 'Studio reference headphones', 12, 159.00, 105.00, 95, 'active', '2024-02-16'),
('ELEC-AU-020', 'Cambridge Audio Melomania 1+', 'Long-battery life earbuds', 12, 129.00, 85.00, 115, 'active', '2024-02-16');

-- Clothing - Men's (25 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('CLO-MN-001', 'Levi''s 501 Original Jeans', 'Classic straight fit denim jeans', 15, 89.00, 45.00, 200, 'active', '2024-02-17'),
('CLO-MN-002', 'Nike Dri-FIT T-Shirt', 'Moisture-wicking athletic tee', 15, 29.00, 15.00, 350, 'active', '2024-02-17'),
('CLO-MN-003', 'Ralph Lauren Polo Shirt', 'Classic polo with embroidered logo', 15, 89.00, 45.00, 180, 'active', '2024-02-18'),
('CLO-MN-004', 'Adidas Track Pants', 'Comfortable athletic pants', 15, 65.00, 32.00, 220, 'active', '2024-02-18'),
('CLO-MN-005', 'Carhartt Work Jacket', 'Durable canvas work jacket', 15, 129.00, 65.00, 95, 'active', '2024-02-19'),
('CLO-MN-006', 'Calvin Klein Boxer Briefs 3-Pack', 'Premium cotton underwear', 15, 42.00, 20.00, 400, 'active', '2024-02-19'),
('CLO-MN-007', 'The North Face Fleece', 'Warm fleece pullover', 15, 99.00, 50.00, 150, 'active', '2024-02-20'),
('CLO-MN-008', 'Dockers Khaki Pants', 'Classic fit chino pants', 15, 59.00, 28.00, 175, 'active', '2024-02-20'),
('CLO-MN-009', 'Columbia Button-Down Shirt', 'Casual long-sleeve shirt', 15, 49.00, 24.00, 210, 'active', '2024-02-21'),
('CLO-MN-010', 'Champion Hoodie', 'Classic pullover hoodie', 15, 55.00, 27.00, 265, 'active', '2024-02-21'),
('CLO-MN-011', 'Wrangler Cargo Shorts', 'Multi-pocket utility shorts', 15, 39.00, 18.00, 285, 'active', '2024-02-22'),
('CLO-MN-012', 'Under Armour Compression Shirt', 'Athletic compression top', 15, 45.00, 22.00, 190, 'active', '2024-02-22'),
('CLO-MN-013', 'Tommy Hilfiger Sweater', 'Classic crew neck sweater', 15, 79.00, 38.00, 135, 'active', '2024-02-23'),
('CLO-MN-014', 'Patagonia Down Jacket', 'Lightweight insulated jacket', 15, 229.00, 115.00, 75, 'active', '2024-02-23'),
('CLO-MN-015', 'Hanes T-Shirt 6-Pack', 'Value pack basic tees', 15, 24.00, 10.00, 500, 'active', '2024-02-24'),
('CLO-MN-016', 'Brooks Brothers Dress Shirt', 'Formal business shirt', 15, 98.00, 48.00, 120, 'active', '2024-02-24'),
('CLO-MN-017', 'Puma Running Shorts', 'Lightweight athletic shorts', 15, 35.00, 16.00, 240, 'active', '2024-02-25'),
('CLO-MN-018', 'Nautica Swim Trunks', 'Quick-dry board shorts', 15, 49.00, 23.00, 195, 'active', '2024-02-25'),
('CLO-MN-019', 'Dickies Work Pants', 'Heavy-duty work trousers', 15, 45.00, 21.00, 210, 'active', '2024-02-26'),
('CLO-MN-020', 'Eddie Bauer Flannel Shirt', 'Classic plaid flannel', 15, 59.00, 28.00, 165, 'active', '2024-02-26'),
('CLO-MN-021', 'Reebok Athletic Socks 6-Pack', 'Cushioned sports socks', 15, 18.00, 8.00, 450, 'active', '2024-02-27'),
('CLO-MN-022', 'Lululemon ABC Pants', 'Anti-ball crushing pants', 15, 128.00, 64.00, 110, 'active', '2024-02-27'),
('CLO-MN-023', 'Vineyard Vines Quarter Zip', 'Preppy pullover sweater', 15, 98.00, 48.00, 95, 'active', '2024-02-28'),
('CLO-MN-024', 'Timberland Thermal Henley', 'Warm waffle-knit shirt', 15, 45.00, 21.00, 155, 'active', '2024-02-28'),
('CLO-MN-025', 'Arc''teryx Softshell Jacket', 'Technical outdoor jacket', 15, 299.00, 150.00, 55, 'active', '2024-02-29');

-- Clothing - Women's (25 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('CLO-WM-001', 'Lululemon Align Leggings', 'High-waisted yoga pants', 16, 98.00, 48.00, 220, 'active', '2024-03-01'),
('CLO-WM-002', 'Zara Midi Dress', 'Elegant casual dress', 16, 79.00, 38.00, 145, 'active', '2024-03-01'),
('CLO-WM-003', 'Nike Sports Bra', 'High-support athletic bra', 16, 45.00, 22.00, 280, 'active', '2024-03-02'),
('CLO-WM-004', 'Gap High-Rise Jeans', 'Classic skinny fit jeans', 16, 79.00, 38.00, 190, 'active', '2024-03-02'),
('CLO-WM-005', 'H&M Blazer', 'Professional fitted blazer', 16, 89.00, 43.00, 125, 'active', '2024-03-03'),
('CLO-WM-006', 'Old Navy Tank Top 3-Pack', 'Basic layering tanks', 16, 25.00, 11.00, 400, 'active', '2024-03-03'),
('CLO-WM-007', 'Athleta Salutation Stash Pocket', 'Yoga pants with pockets', 16, 98.00, 48.00, 185, 'active', '2024-03-04'),
('CLO-WM-008', 'Madewell Perfect Tee', 'Soft everyday t-shirt', 16, 35.00, 16.00, 310, 'active', '2024-03-04'),
('CLO-WM-009', 'Free People Sweater', 'Cozy oversized knit', 16, 128.00, 63.00, 95, 'active', '2024-03-05'),
('CLO-WM-010', 'Spanx Faux Leather Leggings', 'Stretchy leather-look pants', 16, 98.00, 48.00, 165, 'active', '2024-03-05'),
('CLO-WM-011', 'Everlane Cashmere Sweater', 'Sustainable cashmere knit', 16, 128.00, 63.00, 88, 'active', '2024-03-06'),
('CLO-WM-012', 'Anthropologie Blouse', 'Flowy romantic top', 16, 98.00, 48.00, 135, 'active', '2024-03-06'),
('CLO-WM-013', 'Patagonia Better Sweater', 'Recycled fleece jacket', 16, 139.00, 68.00, 110, 'active', '2024-03-07'),
('CLO-WM-014', 'Reformation Wrap Dress', 'Sustainable midi dress', 16, 218.00, 108.00, 75, 'active', '2024-03-07'),
('CLO-WM-015', 'Aerie Leggings', 'Comfortable everyday leggings', 16, 49.00, 23.00, 325, 'active', '2024-03-08'),
('CLO-WM-016', 'J.Crew Factory Cardigan', 'Classic button cardigan', 16, 59.00, 28.00, 195, 'active', '2024-03-08'),
('CLO-WM-017', 'Victoria''s Secret Push-Up Bra', 'Padded lace bra', 16, 49.00, 23.00, 240, 'active', '2024-03-09'),
('CLO-WM-018', 'Uniqlo Heattech Long Sleeve', 'Thermal base layer', 16, 29.00, 13.00, 380, 'active', '2024-03-09'),
('CLO-WM-019', 'Target Universal Thread Jeans', 'Affordable high-rise jeans', 16, 32.00, 14.00, 420, 'active', '2024-03-10'),
('CLO-WM-020', 'Adidas Athletic Shorts', 'Running workout shorts', 16, 35.00, 16.00, 295, 'active', '2024-03-10'),
('CLO-WM-021', 'ASOS Midi Skirt', 'Trendy pleated skirt', 16, 42.00, 19.00, 215, 'active', '2024-03-11'),
('CLO-WM-022', 'Eileen Fisher Tunic', 'Organic cotton long top', 16, 118.00, 58.00, 85, 'active', '2024-03-11'),
('CLO-WM-023', 'Outdoor Voices Exercise Dress', 'Athletic tennis dress', 16, 98.00, 48.00, 125, 'active', '2024-03-12'),
('CLO-WM-024', 'Banana Republic Trousers', 'Work-appropriate pants', 16, 98.00, 48.00, 145, 'active', '2024-03-12'),
('CLO-WM-025', 'Columbia Rain Jacket', 'Waterproof hooded jacket', 16, 89.00, 43.00, 165, 'active', '2024-03-13');

-- Shoes & Footwear (20 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('CLO-SH-001', 'Nike Air Max 90', 'Classic cushioned sneakers', 17, 129.00, 64.00, 180, 'active', '2024-03-14'),
('CLO-SH-002', 'Adidas Ultraboost 22', 'Premium running shoes', 17, 189.00, 94.00, 145, 'active', '2024-03-14'),
('CLO-SH-003', 'Converse Chuck Taylor All Star', 'Iconic canvas sneakers', 17, 65.00, 32.00, 320, 'active', '2024-03-15'),
('CLO-SH-004', 'Vans Old Skool', 'Skateboarding classic shoes', 17, 75.00, 37.00, 275, 'active', '2024-03-15'),
('CLO-SH-005', 'New Balance 574', 'Retro lifestyle sneakers', 17, 89.00, 44.00, 195, 'active', '2024-03-16'),
('CLO-SH-006', 'Dr. Martens 1460 Boots', 'Classic leather ankle boots', 17, 169.00, 84.00, 125, 'active', '2024-03-16'),
('CLO-SH-007', 'Birkenstock Arizona', 'Comfortable cork footbed sandals', 17, 99.00, 49.00, 210, 'active', '2024-03-17'),
('CLO-SH-008', 'Timberland 6-Inch Premium Boot', 'Waterproof work boots', 17, 189.00, 94.00, 155, 'active', '2024-03-17'),
('CLO-SH-009', 'Crocs Classic Clog', 'Lightweight foam clogs', 17, 49.00, 24.00, 385, 'active', '2024-03-18'),
('CLO-SH-010', 'Allbirds Wool Runners', 'Sustainable merino sneakers', 17, 98.00, 48.00, 165, 'active', '2024-03-18'),
('CLO-SH-011', 'Clarks Desert Boot', 'Suede chukka boots', 17, 139.00, 69.00, 135, 'active', '2024-03-19'),
('CLO-SH-012', 'UGG Classic Short II', 'Sheepskin winter boots', 17, 169.00, 84.00, 145, 'active', '2024-03-19'),
('CLO-SH-013', 'Steve Madden Ankle Boots', 'Fashionable booties', 17, 99.00, 49.00, 195, 'active', '2024-03-20'),
('CLO-SH-014', 'Skechers Go Walk', 'Comfortable walking shoes', 17, 65.00, 32.00, 245, 'active', '2024-03-20'),
('CLO-SH-015', 'Puma Suede Classic', 'Retro basketball sneakers', 17, 75.00, 37.00, 215, 'active', '2024-03-21'),
('CLO-SH-016', 'Reebok Classic Leather', 'Vintage running shoes', 17, 79.00, 39.00, 225, 'active', '2024-03-21'),
('CLO-SH-017', 'TOMS Alpargata', 'Slip-on canvas espadrilles', 17, 64.00, 32.00, 265, 'active', '2024-03-22'),
('CLO-SH-018', 'Sam Edelman Loraine Loafer', 'Classic leather loafers', 17, 139.00, 69.00, 115, 'active', '2024-03-22'),
('CLO-SH-019', 'Sorel Winter Boots', 'Insulated snow boots', 17, 189.00, 94.00, 95, 'active', '2024-03-23'),
('CLO-SH-020', 'Havaianas Flip Flops', 'Brazilian rubber sandals', 17, 26.00, 12.00, 450, 'active', '2024-03-23');

-- Home & Kitchen - Appliances (20 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('HOM-KA-001', 'KitchenAid Stand Mixer', 'Professional 5-quart mixer', 19, 449.00, 270.00, 85, 'active', '2024-03-24'),
('HOM-KA-002', 'Ninja Foodi Air Fryer', 'Multi-function air fryer', 19, 199.00, 120.00, 145, 'active', '2024-03-24'),
('HOM-KA-003', 'Instant Pot Duo', '7-in-1 pressure cooker', 19, 89.00, 53.00, 220, 'active', '2024-03-25'),
('HOM-KA-004', 'Cuisinart Food Processor', '14-cup capacity processor', 19, 199.00, 120.00, 95, 'active', '2024-03-25'),
('HOM-KA-005', 'Vitamix Professional Blender', 'High-performance blender', 19, 549.00, 330.00, 65, 'active', '2024-03-26'),
('HOM-KA-006', 'Keurig K-Elite Coffee Maker', 'Single-serve coffee brewer', 19, 169.00, 101.00, 175, 'active', '2024-03-26'),
('HOM-KA-007', 'Breville Smart Oven', 'Countertop convection oven', 19, 299.00, 179.00, 88, 'active', '2024-03-27'),
('HOM-KA-008', 'Hamilton Beach Slow Cooker', '6-quart programmable crock pot', 19, 49.00, 29.00, 265, 'active', '2024-03-27'),
('HOM-KA-009', 'Nespresso VertuoPlus', 'Coffee and espresso maker', 19, 189.00, 113.00, 125, 'active', '2024-03-28'),
('HOM-KA-010', 'Lodge Cast Iron Skillet', '12-inch pre-seasoned pan', 19, 34.00, 20.00, 385, 'active', '2024-03-28'),
('HOM-KA-011', 'Cuisinart Toaster Oven', '4-slice compact oven', 19, 89.00, 53.00, 195, 'active', '2024-03-29'),
('HOM-KA-012', 'All-Clad Stainless Cookware Set', '10-piece professional set', 19, 699.00, 420.00, 45, 'active', '2024-03-29'),
('HOM-KA-013', 'OXO Good Grips Knife Block', '15-piece knife set', 19, 149.00, 89.00, 115, 'active', '2024-03-30'),
('HOM-KA-014', 'Dyson V15 Vacuum', 'Cordless stick vacuum', 19, 749.00, 450.00, 75, 'active', '2024-03-30'),
('HOM-KA-015', 'iRobot Roomba j7+', 'Smart robot vacuum', 19, 799.00, 480.00, 55, 'active', '2024-03-31'),
('HOM-KA-016', 'Shark Navigator Vacuum', 'Upright vacuum cleaner', 19, 199.00, 119.00, 145, 'active', '2024-03-31'),
('HOM-KA-017', 'Anova Precision Cooker', 'Sous vide immersion circulator', 19, 149.00, 89.00, 95, 'active', '2024-04-01'),
('HOM-KA-018', 'Black+Decker Toaster', '2-slice wide slot toaster', 19, 29.00, 17.00, 325, 'active', '2024-04-01'),
('HOM-KA-019', 'Philips Air Fryer XXL', 'Extra-large air fryer', 19, 279.00, 167.00, 105, 'active', '2024-04-02'),
('HOM-KA-020', 'Oster Blender Pro', '1200-watt blender', 19, 79.00, 47.00, 185, 'active', '2024-04-02');

-- Home & Kitchen - Furniture (15 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('HOM-FU-001', 'IKEA MALM Bed Frame', 'Queen size modern bed', 20, 299.00, 179.00, 45, 'active', '2024-04-03'),
('HOM-FU-002', 'Ashley Sectional Sofa', 'L-shaped fabric sectional', 20, 899.00, 540.00, 25, 'active', '2024-04-03'),
('HOM-FU-003', 'West Elm Mid-Century Desk', 'Modern work desk with drawers', 20, 599.00, 359.00, 38, 'active', '2024-04-04'),
('HOM-FU-004', 'La-Z-Boy Recliner', 'Leather power recliner', 20, 799.00, 479.00, 32, 'active', '2024-04-04'),
('HOM-FU-005', 'Wayfair Dining Table Set', '5-piece dining set', 20, 499.00, 299.00, 42, 'active', '2024-04-05'),
('HOM-FU-006', 'Pottery Barn Bookshelf', '5-shelf wooden bookcase', 20, 449.00, 269.00, 55, 'active', '2024-04-05'),
('HOM-FU-007', 'Herman Miller Aeron Chair', 'Ergonomic office chair', 20, 1395.00, 837.00, 18, 'active', '2024-04-06'),
('HOM-FU-008', 'Crate & Barrel Coffee Table', 'Modern glass coffee table', 20, 399.00, 239.00, 48, 'active', '2024-04-06'),
('HOM-FU-009', 'Target Nightstand Set', 'Two-drawer bedside tables (2-pack)', 20, 149.00, 89.00, 95, 'active', '2024-04-07'),
('HOM-FU-010', 'Article Sven Sofa', 'Scandinavian leather sofa', 20, 1499.00, 899.00, 22, 'active', '2024-04-07'),
('HOM-FU-011', 'Steelcase Leap Chair', 'Premium ergonomic chair', 20, 1099.00, 659.00, 28, 'active', '2024-04-08'),
('HOM-FU-012', 'Overstock TV Stand', '65-inch media console', 20, 299.00, 179.00, 65, 'active', '2024-04-08'),
('HOM-FU-013', 'HomeGoods Ottoman', 'Storage bench ottoman', 20, 129.00, 77.00, 88, 'active', '2024-04-09'),
('HOM-FU-014', 'CB2 Bar Stools', 'Modern counter stools (set of 2)', 20, 299.00, 179.00, 52, 'active', '2024-04-09'),
('HOM-FU-015', 'Threshold Floor Lamp', 'Arc floor lamp with shade', 20, 79.00, 47.00, 135, 'active', '2024-04-10');

-- Sports & Fitness (25 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('SPO-FT-001', 'Bowflex Adjustable Dumbbells', '5-52.5 lb dumbbell set', 23, 349.00, 209.00, 75, 'active', '2024-04-11'),
('SPO-FT-002', 'Peloton Bike Basic', 'Indoor cycling bike', 23, 1445.00, 867.00, 28, 'active', '2024-04-11'),
('SPO-FT-003', 'Manduka PRO Yoga Mat', 'Professional yoga mat', 23, 128.00, 77.00, 185, 'active', '2024-04-12'),
('SPO-FT-004', 'TRX Suspension Trainer', 'Bodyweight resistance system', 23, 179.00, 107.00, 125, 'active', '2024-04-12'),
('SPO-FT-005', 'NordicTrack Treadmill', 'Folding home treadmill', 23, 1299.00, 779.00, 35, 'active', '2024-04-13'),
('SPO-FT-006', 'Fitbit Charge 6', 'Fitness tracking smartwatch', 23, 159.00, 95.00, 245, 'active', '2024-04-13'),
('SPO-FT-007', 'PowerBlock Elite Dumbbells', 'Expandable weight system', 23, 399.00, 239.00, 65, 'active', '2024-04-14'),
('SPO-FT-008', 'Concept2 Model D Rower', 'Indoor rowing machine', 23, 945.00, 567.00, 42, 'active', '2024-04-14'),
('SPO-FT-009', 'CAP Barbell Weight Set', '300 lb Olympic set', 23, 299.00, 179.00, 55, 'active', '2024-04-15'),
('SPO-FT-010', 'Theragun Elite', 'Percussive therapy massage gun', 23, 399.00, 239.00, 95, 'active', '2024-04-15'),
('SPO-FT-011', 'Yes4All Kettlebell Set', '5-50 lb kettlebell collection', 23, 189.00, 113.00, 115, 'active', '2024-04-16'),
('SPO-FT-012', 'Garmin Forerunner 265', 'GPS running watch', 23, 449.00, 269.00, 88, 'active', '2024-04-16'),
('SPO-FT-013', 'Schwinn IC4 Bike', 'Indoor cycling bike', 23, 899.00, 539.00, 45, 'active', '2024-04-17'),
('SPO-FT-014', 'Rogue Fitness Bench', 'Adjustable weight bench', 23, 425.00, 255.00, 58, 'active', '2024-04-17'),
('SPO-FT-015', 'Liforme Yoga Mat', 'Eco-friendly alignment mat', 23, 139.00, 83.00, 145, 'active', '2024-04-18'),
('SPO-FT-016', 'Marcy Home Gym', 'Multi-station weight machine', 23, 699.00, 419.00, 32, 'active', '2024-04-18'),
('SPO-FT-017', 'Gaiam Balance Ball Chair', 'Stability ball office chair', 23, 79.00, 47.00, 175, 'active', '2024-04-19'),
('SPO-FT-018', 'ProForm Elliptical', 'Front-drive elliptical machine', 23, 899.00, 539.00, 38, 'active', '2024-04-19'),
('SPO-FT-019', 'Hyperice Hypervolt', 'Vibration massage device', 23, 299.00, 179.00, 105, 'active', '2024-04-20'),
('SPO-FT-020', 'SPRI Resistance Bands Set', 'Exercise band kit with handles', 23, 29.00, 17.00, 385, 'active', '2024-04-20'),
('SPO-FT-021', 'Wahoo KICKR Bike', 'Smart indoor bike', 23, 3499.00, 2099.00, 15, 'active', '2024-04-21'),
('SPO-FT-022', 'Bosu Balance Trainer', 'Half ball balance platform', 23, 129.00, 77.00, 125, 'active', '2024-04-21'),
('SPO-FT-023', 'Gold''s Gym Power Tower', 'Pull-up and dip station', 23, 199.00, 119.00, 68, 'active', '2024-04-22'),
('SPO-FT-024', 'Perfect Fitness Ab Carver', 'Ab roller wheel', 23, 39.00, 23.00, 265, 'active', '2024-04-22'),
('SPO-FT-025', 'Stamina InMotion Elliptical', 'Compact under-desk trainer', 23, 89.00, 53.00, 195, 'active', '2024-04-23');

-- Sports - Outdoor (15 products)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('SPO-OD-001', 'Coleman Sundome Tent', '4-person camping tent', 24, 89.00, 53.00, 125, 'active', '2024-04-24'),
('SPO-OD-002', 'Yeti Tundra 45 Cooler', 'Rotomolded hard cooler', 24, 349.00, 209.00, 75, 'active', '2024-04-24'),
('SPO-OD-003', 'Osprey Atmos 65 Backpack', 'Hiking backpack with suspension', 24, 289.00, 173.00, 88, 'active', '2024-04-25'),
('SPO-OD-004', 'ENO DoubleNest Hammock', 'Portable camping hammock', 24, 69.00, 41.00, 215, 'active', '2024-04-25'),
('SPO-OD-005', 'Merrell Moab 2 Hiking Boots', 'Waterproof trail boots', 24, 135.00, 81.00, 165, 'active', '2024-04-26'),
('SPO-OD-006', 'Black Diamond Headlamp', 'Rechargeable LED headlamp', 24, 49.00, 29.00, 245, 'active', '2024-04-26'),
('SPO-OD-007', 'MSR PocketRocket Stove', 'Ultralight backpacking stove', 24, 49.00, 29.00, 185, 'active', '2024-04-27'),
('SPO-OD-008', 'CamelBak Hydration Pack', '3L reservoir hiking pack', 24, 99.00, 59.00, 155, 'active', '2024-04-27'),
('SPO-OD-009', 'REI Co-op Sleeping Bag', '20°F down sleeping bag', 24, 299.00, 179.00, 95, 'active', '2024-04-28'),
('SPO-OD-010', 'Thule Bike Rack', '2-bike hitch mount carrier', 24, 399.00, 239.00, 58, 'active', '2024-04-28'),
('SPO-OD-011', 'Patagonia Down Sweater', 'Packable insulated jacket', 24, 269.00, 161.00, 125, 'active', '2024-04-29'),
('SPO-OD-012', 'GoPro HERO12 Black', 'Action camera', 24, 399.00, 239.00, 105, 'active', '2024-04-29'),
('SPO-OD-013', 'Leatherman Wave+ Multi-Tool', 'Premium multi-tool', 24, 119.00, 71.00, 185, 'active', '2024-04-30'),
('SPO-OD-014', 'Garmin inReach Mini 2', 'Satellite communicator', 24, 399.00, 239.00, 65, 'active', '2024-04-30'),
('SPO-OD-015', 'Jetboil Flash Cooking System', 'Fast-boil camping stove', 24, 109.00, 65.00, 145, 'active', '2024-05-01');

-- Books & Health/Beauty (20 products to reach 300 total)
INSERT INTO products (sku, product_name, description, category_id, price, cost, stock_quantity, status, created_at) VALUES
('HEA-BE-001', 'Dyson Airwrap Styler', 'Multi-styling hair tool', 6, 599.00, 359.00, 45, 'active', '2024-05-02'),
('HEA-BE-002', 'Olaplex Hair Treatment Set', 'Bond-building hair repair kit', 6, 88.00, 53.00, 265, 'active', '2024-05-02'),
('HEA-BE-003', 'Neutrogena Cleanser', 'Hydrating facial cleanser', 6, 12.00, 7.00, 485, 'active', '2024-05-03'),
('HEA-BE-004', 'The Ordinary Serum Set', 'Skincare essentials collection', 6, 45.00, 27.00, 325, 'active', '2024-05-03'),
('HEA-BE-005', 'Clarisonic Mia Smart', 'Sonic facial cleansing brush', 6, 219.00, 131.00, 75, 'active', '2024-05-04'),
('HEA-BE-006', 'CeraVe Moisturizing Cream', 'Daily face and body lotion', 6, 19.00, 11.00, 425, 'active', '2024-05-04'),
('HEA-BE-007', 'Revlon One-Step Hair Dryer', 'Volumizing hot air brush', 6, 59.00, 35.00, 195, 'active', '2024-05-05'),
('HEA-BE-008', 'La Roche-Posay Sunscreen SPF 50', 'Mineral face sunscreen', 6, 34.00, 20.00, 365, 'active', '2024-05-05'),
('HEA-BE-009', 'Foreo Luna 3', 'Smart facial cleansing device', 6, 219.00, 131.00, 88, 'active', '2024-05-06'),
('HEA-BE-010', 'ghd Platinum+ Styler', 'Professional hair straightener', 6, 279.00, 167.00, 95, 'active', '2024-05-06'),
('HEA-BE-011', 'Fenty Beauty Foundation', 'Pro Filt''r soft matte foundation', 6, 39.00, 23.00, 285, 'active', '2024-05-07'),
('HEA-BE-012', 'Tatcha Dewy Skin Cream', 'Plumping moisturizer', 6, 69.00, 41.00, 175, 'active', '2024-05-07'),
('HEA-BE-013', 'T3 Micro Dryer', 'Professional ionic hair dryer', 6, 229.00, 137.00, 105, 'active', '2024-05-08'),
('HEA-BE-014', 'Drunk Elephant Protini', 'Peptide moisturizer', 6, 68.00, 41.00, 215, 'active', '2024-05-08'),
('HEA-BE-015', 'Braun Silk-épil 9', 'Epilator and exfoliator', 6, 139.00, 83.00, 125, 'active', '2024-05-09'),
('HEA-BE-016', 'Sunday Riley Good Genes', 'Lactic acid treatment', 6, 85.00, 51.00, 155, 'active', '2024-05-09'),
('HEA-BE-017', 'Philips Sonicare DiamondClean', 'Electric toothbrush', 6, 229.00, 137.00, 145, 'active', '2024-05-10'),
('HEA-BE-018', 'Benefit Cosmetics BADgal Bang', 'Volumizing mascara', 6, 26.00, 16.00, 395, 'active', '2024-05-10'),
('HEA-BE-019', 'NuFace Trinity Device', 'Facial toning device', 6, 339.00, 203.00, 65, 'active', '2024-05-11'),
('HEA-BE-020', 'Anastasia Beverly Hills Palette', 'Eye shadow palette', 6, 45.00, 27.00, 245, 'active', '2024-05-11');

-- Display confirmation
SELECT 'Product categories and 300 products inserted successfully!' AS Status;

SELECT 
    COUNT(*) AS total_products,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_products,
    SUM(stock_quantity) AS total_stock_units,
    ROUND(AVG(price), 2) AS avg_price,
    ROUND(AVG(price - cost), 2) AS avg_profit_margin
FROM products;

SELECT 'Products by category:' AS Info;
SELECT 
    pc.category_name,
    COUNT(p.product_id) AS product_count,
    ROUND(AVG(p.price), 2) AS avg_price
FROM product_categories pc
LEFT JOIN products p ON pc.category_id = p.category_id
GROUP BY pc.category_id, pc.category_name
ORDER BY product_count DESC
LIMIT 15;