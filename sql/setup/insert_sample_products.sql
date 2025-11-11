-- SAMPLE PRODUCT DATA INSERTION SCRIPT
-- Database: ecommerce_analytics
-- Target: 400 product records
-- Data Quality Issues (intentional for testing):
--   - 9% invalid/zero prices (36 records) - price = 0 or NULL or negative
--   - 26% missing descriptions (104 records) - NULL description
--   - 10% missing categories (40 records) - NULL category_id
--   - 15% no images (60 records) - NULL image_url
--   - 2% missing names (8 records) - NULL name
-- Price Range: $5-$500
-- Categories: Electronics, Clothing, Home & Kitchen, Books, Sports, Toys, Food & Beverage, Beauty
-- Stock: Mix of in-stock (0-500) and out-of-stock (0) items
-- Date Range: Last 2 years (2023-11-06 to 2025-11-06)
-- ============================================================================

USE ecommerce_analytics;

-- Disable foreign key checks temporarily for faster insertion
SET FOREIGN_KEY_CHECKS = 0;

-- ============================================================================
-- INSERT PRODUCT CATEGORIES FIRST
-- ============================================================================

INSERT INTO product_categories (category_name, parent_category_id, description) VALUES
('Electronics', NULL, 'Electronic devices and accessories'),
('Clothing', NULL, 'Apparel and fashion items'),
('Home & Kitchen', NULL, 'Home goods and kitchen appliances'),
('Books', NULL, 'Books and publications'),
('Sports', NULL, 'Sports equipment and fitness'),
('Toys', NULL, 'Toys and games'),
('Food & Beverage', NULL, 'Food and drink products'),
('Beauty', NULL, 'Beauty and personal care products');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 1: ELECTRONICS (Good Quality: 1-50)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('ELEC-001', 'Wireless Bluetooth Headphones', 1, 79.99, 45.00, 'Premium wireless headphones with noise cancellation and 30-hour battery life', 'https://images.example.com/headphones-001.jpg', 145, 0.5, '7x6x3', '2024-01-15'),
('ELEC-002', 'Smartphone 128GB', 1, 499.99, 350.00, 'Latest generation smartphone with 6.5 inch display and triple camera system', 'https://images.example.com/phone-002.jpg', 87, 0.4, '6x3x0.3', '2024-02-20'),
('ELEC-003', 'Laptop 15.6 inch i5 8GB RAM', 1, 649.99, 450.00, 'Powerful laptop for work and entertainment with SSD storage', 'https://images.example.com/laptop-003.jpg', 42, 4.5, '14x10x1', '2024-01-10'),
('ELEC-004', '4K Smart TV 55 inch', 1, 449.99, 300.00, 'Ultra HD smart TV with HDR and streaming apps built-in', 'https://images.example.com/tv-004.jpg', 28, 35.0, '49x29x3', '2024-03-05'),
('ELEC-005', 'Wireless Mouse', 1, 24.99, 12.00, 'Ergonomic wireless mouse with precision tracking', 'https://images.example.com/mouse-005.jpg', 234, 0.2, '4x2x1', '2024-02-14'),
('ELEC-006', 'USB-C Charging Cable 6ft', 1, 14.99, 5.00, 'Fast charging cable compatible with most devices', 'https://images.example.com/cable-006.jpg', 456, 0.1, '6x1x0.5', '2024-01-20'),
('ELEC-007', 'Portable Power Bank 20000mAh', 1, 39.99, 20.00, 'High capacity power bank with dual USB ports and fast charging', 'https://images.example.com/powerbank-007.jpg', 178, 0.8, '6x3x1', '2024-03-12'),
('ELEC-008', 'Wireless Keyboard', 1, 44.99, 25.00, 'Slim wireless keyboard with quiet keys and long battery life', 'https://images.example.com/keyboard-008.jpg', 156, 1.2, '17x5x1', '2024-02-08'),
('ELEC-009', 'Smart Watch Fitness Tracker', 1, 159.99, 90.00, 'Advanced fitness tracker with heart rate monitor and GPS', 'https://images.example.com/watch-009.jpg', 93, 0.3, '2x2x0.5', '2024-01-25'),
('ELEC-010', 'Bluetooth Speaker Waterproof', 1, 59.99, 30.00, 'Portable waterproof speaker with 360-degree sound', 'https://images.example.com/speaker-010.jpg', 201, 1.5, '8x3x3', '2024-03-18'),
('ELEC-011', 'Tablet 10.1 inch 64GB', 1, 199.99, 120.00, 'Lightweight tablet perfect for reading and browsing', 'https://images.example.com/tablet-011.jpg', 67, 1.2, '10x7x0.3', '2024-02-22'),
('ELEC-012', 'Webcam 1080p HD', 1, 49.99, 25.00, 'High definition webcam with built-in microphone', 'https://images.example.com/webcam-012.jpg', 134, 0.4, '4x3x3', '2024-01-30'),
('ELEC-013', 'Gaming Headset RGB', 1, 89.99, 50.00, 'Immersive gaming headset with surround sound and RGB lighting', 'https://images.example.com/headset-013.jpg', 112, 0.7, '8x7x4', '2024-03-08'),
('ELEC-014', 'External Hard Drive 2TB', 1, 79.99, 45.00, 'Portable external storage with USB 3.0 for fast transfers', 'https://images.example.com/hdd-014.jpg', 89, 0.5, '5x3x0.5', '2024-02-15'),
('ELEC-015', 'Wireless Earbuds', 1, 69.99, 35.00, 'True wireless earbuds with charging case and touch controls', 'https://images.example.com/earbuds-015.jpg', 267, 0.1, '3x2x1', '2024-01-18'),
('ELEC-016', 'HDMI Cable 10ft', 1, 12.99, 5.00, 'High-speed HDMI cable supports 4K and HDR', 'https://images.example.com/hdmi-016.jpg', 387, 0.3, '10x1x0.5', '2024-03-01'),
('ELEC-017', 'USB Flash Drive 128GB', 1, 19.99, 8.00, 'High-speed USB 3.1 flash drive for portable storage', 'https://images.example.com/usb-017.jpg', 298, 0.05, '2x1x0.3', '2024-02-10'),
('ELEC-018', 'Computer Monitor 27 inch', 1, 249.99, 150.00, 'Full HD monitor with IPS panel and thin bezels', 'https://images.example.com/monitor-018.jpg', 56, 12.0, '24x16x2', '2024-01-22'),
('ELEC-019', 'Router WiFi 6', 1, 119.99, 70.00, 'Next-gen WiFi router with dual-band and gigabit speeds', 'https://images.example.com/router-019.jpg', 73, 1.8, '10x7x2', '2024-03-14'),
('ELEC-020', 'Security Camera Wireless', 1, 89.99, 50.00, 'HD security camera with night vision and motion detection', 'https://images.example.com/camera-020.jpg', 124, 0.6, '4x3x3', '2024-02-05'),
('ELEC-021', 'Action Camera 4K', 1, 179.99, 100.00, 'Rugged action camera with waterproof case and image stabilization', 'https://images.example.com/action-021.jpg', 81, 0.4, '3x2x2', '2024-01-28'),
('ELEC-022', 'Drone with Camera', 1, 299.99, 180.00, 'Beginner-friendly drone with 1080p camera and GPS', 'https://images.example.com/drone-022.jpg', 34, 2.5, '12x10x5', '2024-03-20'),
('ELEC-023', 'Smart Light Bulb LED', 1, 19.99, 10.00, 'WiFi-enabled color-changing LED bulb', 'https://images.example.com/bulb-023.jpg', 456, 0.2, '3x3x5', '2024-02-18'),
('ELEC-024', 'Video Doorbell', 1, 149.99, 85.00, 'Smart doorbell with HD video and two-way audio', 'https://images.example.com/doorbell-024.jpg', 92, 0.8, '5x2x1', '2024-01-12'),
('ELEC-025', 'Soundbar with Subwoofer', 1, 199.99, 120.00, 'Premium soundbar system with wireless subwoofer for home theater', 'https://images.example.com/soundbar-025.jpg', 47, 8.0, '36x4x3', '2024-03-10'),
('ELEC-026', 'Electric Toothbrush', 1, 59.99, 30.00, 'Rechargeable electric toothbrush with multiple brush heads', 'https://images.example.com/toothbrush-026.jpg', 189, 0.4, '9x2x2', '2024-02-25'),
('ELEC-027', 'Hair Dryer Professional', 1, 79.99, 40.00, 'Ionic hair dryer with multiple heat settings', 'https://images.example.com/dryer-027.jpg', 143, 1.5, '10x9x4', '2024-01-16'),
('ELEC-028', 'Coffee Maker Programmable', 1, 89.99, 50.00, '12-cup programmable coffee maker with auto-brew timer', 'https://images.example.com/coffee-028.jpg', 98, 5.0, '12x8x13', '2024-03-06'),
('ELEC-029', 'Air Fryer 5 Quart', 1, 109.99, 65.00, 'Digital air fryer for healthier cooking with multiple presets', 'https://images.example.com/airfryer-029.jpg', 76, 10.0, '13x13x14', '2024-02-12'),
('ELEC-030', 'Robot Vacuum Cleaner', 1, 249.99, 150.00, 'Smart robot vacuum with mapping and scheduling', 'https://images.example.com/vacuum-030.jpg', 54, 7.5, '14x14x4', '2024-01-08'),
('ELEC-031', 'Digital Photo Frame', 1, 79.99, 45.00, '10-inch WiFi digital photo frame with touchscreen', 'https://images.example.com/frame-031.jpg', 112, 1.2, '10x8x1', '2024-03-16'),
('ELEC-032', 'Instant Print Camera', 1, 89.99, 50.00, 'Fun instant camera that prints photos instantly', 'https://images.example.com/instacam-032.jpg', 134, 0.8, '5x4x2', '2024-02-20'),
('ELEC-033', 'Portable Projector', 1, 199.99, 120.00, 'Mini projector with WiFi and Bluetooth connectivity', 'https://images.example.com/projector-033.jpg', 63, 2.0, '8x6x3', '2024-01-24'),
('ELEC-034', 'Car Phone Mount', 1, 19.99, 8.00, 'Universal car mount for smartphones with strong grip', 'https://images.example.com/mount-034.jpg', 287, 0.3, '5x4x3', '2024-03-11'),
('ELEC-035', 'Dash Cam 1080p', 1, 69.99, 35.00, 'Dashboard camera with loop recording and G-sensor', 'https://images.example.com/dashcam-035.jpg', 156, 0.4, '4x2x2', '2024-02-07'),
('ELEC-036', 'Surge Protector 12 Outlet', 1, 29.99, 15.00, 'Power strip with surge protection and USB charging ports', 'https://images.example.com/surge-036.jpg', 234, 1.5, '12x3x2', '2024-01-19'),
('ELEC-037', 'Laptop Stand Adjustable', 1, 34.99, 18.00, 'Ergonomic laptop stand with adjustable height and angle', 'https://images.example.com/stand-037.jpg', 178, 1.0, '11x9x2', '2024-03-13'),
('ELEC-038', 'Phone Screen Protector', 1, 9.99, 3.00, 'Tempered glass screen protector with easy installation', 'https://images.example.com/protector-038.jpg', 543, 0.05, '6x3x0.1', '2024-02-16'),
('ELEC-039', 'Selfie Ring Light', 1, 24.99, 12.00, 'LED ring light for perfect selfies and video calls', 'https://images.example.com/ringlight-039.jpg', 267, 0.6, '10x10x2', '2024-01-11'),
('ELEC-040', 'Smart Thermostat', 1, 179.99, 100.00, 'WiFi-enabled smart thermostat with energy-saving features', 'https://images.example.com/thermostat-040.jpg', 89, 0.5, '4x4x1', '2024-03-09'),
('ELEC-041', 'Microphone USB Condenser', 1, 59.99, 30.00, 'Professional USB microphone for podcasting and streaming', 'https://images.example.com/mic-041.jpg', 145, 1.2, '7x3x3', '2024-02-23'),
('ELEC-042', 'Graphics Tablet Drawing', 1, 79.99, 45.00, 'Digital drawing tablet with pressure-sensitive pen', 'https://images.example.com/tablet-042.jpg', 92, 1.5, '13x9x0.5', '2024-01-14'),
('ELEC-043', 'VR Headset', 1, 299.99, 180.00, 'Virtual reality headset with immersive 3D experience', 'https://images.example.com/vr-043.jpg', 43, 1.8, '8x6x5', '2024-03-17'),
('ELEC-044', 'Digital Alarm Clock', 1, 29.99, 15.00, 'LED alarm clock with USB charging ports and dimmer', 'https://images.example.com/clock-044.jpg', 198, 0.7, '6x3x3', '2024-02-11'),
('ELEC-045', 'Electric Kettle 1.7L', 1, 34.99, 18.00, 'Fast-boiling electric kettle with auto shut-off', 'https://images.example.com/kettle-045.jpg', 167, 2.0, '8x6x10', '2024-01-27'),
('ELEC-046', 'Blender High Speed', 1, 69.99, 35.00, 'Powerful blender for smoothies and food prep', 'https://images.example.com/blender-046.jpg', 124, 6.0, '8x8x16', '2024-03-15'),
('ELEC-047', 'Slow Cooker 6 Quart', 1, 49.99, 25.00, 'Programmable slow cooker with digital timer', 'https://images.example.com/slowcooker-047.jpg', 89, 9.0, '13x10x12', '2024-02-19'),
('ELEC-048', 'Pressure Cooker Electric', 1, 89.99, 50.00, 'Multi-function pressure cooker with 10 cooking programs', 'https://images.example.com/pressure-048.jpg', 76, 12.0, '13x12x13', '2024-01-21'),
('ELEC-049', 'Stand Mixer 5 Quart', 1, 199.99, 120.00, 'Professional stand mixer with multiple attachments', 'https://images.example.com/mixer-049.jpg', 52, 20.0, '14x9x14', '2024-03-19'),
('ELEC-050', 'Toaster 4 Slice', 1, 44.99, 22.00, 'Extra-wide slot toaster with bagel and defrost settings', 'https://images.example.com/toaster-050.jpg', 143, 4.5, '13x7x8', '2024-02-13');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 2: CLOTHING (Good Quality: 51-100)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('CLTH-051', 'Men Cotton T-Shirt Blue', 2, 19.99, 8.00, 'Comfortable cotton t-shirt in classic blue color', 'https://images.example.com/tshirt-051.jpg', 234, 0.3, '12x9x1', '2024-01-10'),
('CLTH-052', 'Women Yoga Pants Black', 2, 34.99, 15.00, 'High-waisted yoga pants with moisture-wicking fabric', 'https://images.example.com/yogapants-052.jpg', 187, 0.4, '12x10x2', '2024-02-15'),
('CLTH-053', 'Men Jeans Slim Fit', 2, 49.99, 25.00, 'Classic slim-fit jeans in dark wash denim', 'https://images.example.com/jeans-053.jpg', 156, 0.9, '14x12x2', '2024-01-20'),
('CLTH-054', 'Women Summer Dress', 2, 39.99, 18.00, 'Flowy summer dress with floral print', 'https://images.example.com/dress-054.jpg', 143, 0.4, '14x10x2', '2024-03-05'),
('CLTH-055', 'Men Hoodie Gray', 2, 44.99, 22.00, 'Comfortable pullover hoodie with kangaroo pocket', 'https://images.example.com/hoodie-055.jpg', 167, 0.7, '13x11x3', '2024-02-10'),
('CLTH-056', 'Women Leggings', 2, 24.99, 12.00, 'Stretchy athletic leggings for workout or casual wear', 'https://images.example.com/leggings-056.jpg', 298, 0.3, '11x9x1', '2024-01-25'),
('CLTH-057', 'Men Polo Shirt', 2, 29.99, 14.00, 'Classic polo shirt with breathable cotton fabric', 'https://images.example.com/polo-057.jpg', 189, 0.4, '12x9x1', '2024-03-12'),
('CLTH-058', 'Women Cardigan Sweater', 2, 54.99, 28.00, 'Cozy cardigan sweater perfect for layering', 'https://images.example.com/cardigan-058.jpg', 124, 0.6, '14x11x3', '2024-02-20'),
('CLTH-059', 'Men Shorts Cargo', 2, 34.99, 16.00, 'Cargo shorts with multiple pockets and adjustable waist', 'https://images.example.com/shorts-059.jpg', 178, 0.5, '12x10x2', '2024-01-15'),
('CLTH-060', 'Women Blouse White', 2, 32.99, 15.00, 'Elegant white blouse for office or casual wear', 'https://images.example.com/blouse-060.jpg', 156, 0.3, '12x9x2', '2024-03-18'),
('CLTH-061', 'Men Jacket Winter', 2, 89.99, 45.00, 'Insulated winter jacket with water-resistant shell', 'https://images.example.com/jacket-061.jpg', 89, 1.5, '16x13x4', '2024-02-08'),
('CLTH-062', 'Women Skirt Midi', 2, 36.99, 18.00, 'Midi-length skirt with elastic waistband', 'https://images.example.com/skirt-062.jpg', 134, 0.4, '13x10x2', '2024-01-30'),
('CLTH-063', 'Men Athletic Shorts', 2, 27.99, 13.00, 'Lightweight athletic shorts with mesh lining', 'https://images.example.com/athletic-063.jpg', 212, 0.3, '11x9x1', '2024-03-08'),
('CLTH-064', 'Women Tank Top', 2, 16.99, 7.00, 'Racerback tank top for active lifestyle', 'https://images.example.com/tank-064.jpg', 267, 0.2, '10x8x1', '2024-02-25'),
('CLTH-065', 'Men Sweatpants', 2, 39.99, 20.00, 'Comfortable sweatpants with drawstring waist', 'https://images.example.com/sweatpants-065.jpg', 145, 0.6, '13x11x2', '2024-01-18'),
('CLTH-066', 'Women Sports Bra', 2, 29.99, 14.00, 'High-support sports bra for intense workouts', 'https://images.example.com/sportsbra-066.jpg', 198, 0.2, '10x8x1', '2024-03-14'),
('CLTH-067', 'Men Button-Down Shirt', 2, 42.99, 21.00, 'Professional button-down shirt for business casual', 'https://images.example.com/buttondown-067.jpg', 167, 0.4, '13x10x2', '2024-02-05'),
('CLTH-068', 'Women Maxi Dress', 2, 49.99, 25.00, 'Long maxi dress perfect for summer events', 'https://images.example.com/maxidress-068.jpg', 112, 0.5, '15x11x2', '2024-01-28'),
('CLTH-069', 'Men Running Shoes', 2, 69.99, 35.00, 'Lightweight running shoes with cushioned sole', 'https://images.example.com/runshoes-069.jpg', 134, 1.2, '13x5x5', '2024-03-20'),
('CLTH-070', 'Women Sneakers White', 2, 59.99, 30.00, 'Classic white sneakers for everyday wear', 'https://images.example.com/sneakers-070.jpg', 156, 1.0, '12x5x5', '2024-02-18'),
('CLTH-071', 'Men Dress Shoes', 2, 79.99, 40.00, 'Formal dress shoes with leather upper', 'https://images.example.com/dressshoes-071.jpg', 98, 1.5, '13x5x5', '2024-01-12'),
('CLTH-072', 'Women Sandals', 2, 34.99, 17.00, 'Comfortable sandals with adjustable straps', 'https://images.example.com/sandals-072.jpg', 189, 0.6, '11x4x3', '2024-03-10'),
('CLTH-073', 'Men Socks 6-Pack', 2, 14.99, 6.00, 'Cotton blend socks in assorted colors', 'https://images.example.com/socks-073.jpg', 345, 0.4, '10x6x2', '2024-02-22'),
('CLTH-074', 'Women Handbag Leather', 2, 89.99, 45.00, 'Stylish leather handbag with multiple compartments', 'https://images.example.com/handbag-074.jpg', 87, 1.2, '14x10x6', '2024-01-16'),
('CLTH-075', 'Men Wallet Bifold', 2, 24.99, 12.00, 'Slim bifold wallet with RFID protection', 'https://images.example.com/wallet-075.jpg', 256, 0.2, '5x4x1', '2024-03-06'),
('CLTH-076', 'Women Sunglasses', 2, 29.99, 14.00, 'UV protection sunglasses with stylish frames', 'https://images.example.com/sunglasses-076.jpg', 178, 0.1, '6x2x2', '2024-02-12'),
('CLTH-077', 'Men Baseball Cap', 2, 19.99, 9.00, 'Adjustable baseball cap with embroidered logo', 'https://images.example.com/cap-077.jpg', 234, 0.2, '8x8x4', '2024-01-08'),
('CLTH-078', 'Women Scarf', 2, 22.99, 11.00, 'Soft wool scarf in multiple colors', 'https://images.example.com/scarf-078.jpg', 156, 0.3, '70x12x1', '2024-03-16'),
('CLTH-079', 'Men Belt Leather', 2, 32.99, 16.00, 'Genuine leather belt with reversible buckle', 'https://images.example.com/belt-079.jpg', 189, 0.4, '45x2x0.5', '2024-02-20'),
('CLTH-080', 'Women Boots Ankle', 2, 69.99, 35.00, 'Fashionable ankle boots with side zipper', 'https://images.example.com/boots-080.jpg', 98, 1.8, '12x6x6', '2024-01-24'),
('CLTH-081', 'Men Boxers 3-Pack', 2, 24.99, 12.00, 'Comfortable cotton boxers in assorted patterns', 'https://images.example.com/boxers-081.jpg', 267, 0.3, '12x8x2', '2024-03-11'),
('CLTH-082', 'Women Pajama Set', 2, 39.99, 20.00, 'Cozy pajama set with top and pants', 'https://images.example.com/pajamas-082.jpg', 145, 0.5, '13x10x2', '2024-02-07'),
('CLTH-083', 'Men Swim Trunks', 2, 29.99, 15.00, 'Quick-dry swim trunks with mesh lining', 'https://images.example.com/swimtrunks-083.jpg', 167, 0.3, '11x9x1', '2024-01-19'),
('CLTH-084', 'Women Swimsuit One-Piece', 2, 49.99, 25.00, 'Flattering one-piece swimsuit with tummy control', 'https://images.example.com/swimsuit-084.jpg', 112, 0.3, '11x9x1', '2024-03-13'),
('CLTH-085', 'Men Rain Jacket', 2, 54.99, 28.00, 'Waterproof rain jacket with adjustable hood', 'https://images.example.com/rainjacket-085.jpg', 89, 0.8, '15x12x3', '2024-02-16'),
('CLTH-086', 'Women Yoga Mat', 2, 29.99, 15.00, 'Non-slip yoga mat with carrying strap', 'https://images.example.com/yogamat-086.jpg', 178, 2.0, '72x24x0.5', '2024-01-11'),
('CLTH-087', 'Men Compression Shirt', 2, 34.99, 17.00, 'Athletic compression shirt for workouts', 'https://images.example.com/compression-087.jpg', 145, 0.3, '12x9x1', '2024-03-09'),
('CLTH-088', 'Women Athletic Top', 2, 32.99, 16.00, 'Moisture-wicking athletic top with mesh panels', 'https://images.example.com/athletictop-088.jpg', 167, 0.3, '11x9x1', '2024-02-23'),
('CLTH-089', 'Men Thermal Underwear', 2, 39.99, 20.00, 'Base layer thermal underwear for cold weather', 'https://images.example.com/thermal-089.jpg', 124, 0.4, '12x10x2', '2024-01-14'),
('CLTH-090', 'Women Robe Bathrobe', 2, 44.99, 22.00, 'Plush bathrobe with tie belt and pockets', 'https://images.example.com/robe-090.jpg', 134, 1.0, '14x11x3', '2024-03-17'),
('CLTH-091', 'Men Work Gloves', 2, 16.99, 8.00, 'Durable work gloves with reinforced palms', 'https://images.example.com/gloves-091.jpg', 198, 0.3, '10x5x2', '2024-02-11'),
('CLTH-092', 'Women Fashion Gloves', 2, 24.99, 12.00, 'Touchscreen-compatible fashion gloves', 'https://images.example.com/fashiongloves-092.jpg', 156, 0.2, '9x4x1', '2024-01-27'),
('CLTH-093', 'Men Backpack', 2, 49.99, 25.00, 'Durable backpack with laptop compartment', 'https://images.example.com/backpack-093.jpg', 145, 1.5, '18x12x8', '2024-03-15'),
('CLTH-094', 'Women Crossbody Bag', 2, 39.99, 20.00, 'Compact crossbody bag with adjustable strap', 'https://images.example.com/crossbody-094.jpg', 167, 0.6, '10x8x3', '2024-02-19'),
('CLTH-095', 'Men Tie Silk', 2, 19.99, 10.00, 'Classic silk tie for formal occasions', 'https://images.example.com/tie-095.jpg', 234, 0.1, '58x4x0.5', '2024-01-21'),
('CLTH-096', 'Women Hair Accessories Set', 2, 14.99, 7.00, 'Set of hair clips, bands, and scrunchies', 'https://images.example.com/hairacc-096.jpg', 287, 0.2, '8x6x2', '2024-03-19'),
('CLTH-097', 'Men Slippers', 2, 24.99, 12.00, 'Comfortable indoor slippers with memory foam', 'https://images.example.com/slippers-097.jpg', 189, 0.7, '12x5x4', '2024-02-13'),
('CLTH-098', 'Women Jewelry Set', 2, 34.99, 17.00, 'Matching necklace and earring set', 'https://images.example.com/jewelry-098.jpg', 134, 0.1, '6x5x2', '2024-01-07'),
('CLTH-099', 'Men Watch Analog', 2, 89.99, 45.00, 'Stylish analog watch with leather strap', 'https://images.example.com/watch-099.jpg', 98, 0.3, '3x3x1', '2024-03-21'),
('CLTH-100', 'Women Tote Bag', 2, 44.99, 22.00, 'Large canvas tote bag for shopping or beach', 'https://images.example.com/tote-100.jpg', 178, 0.8, '16x14x6', '2024-02-17');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 3: HOME & KITCHEN (Good Quality: 101-150)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('HOME-101', 'Bed Sheets Queen Set', 3, 39.99, 20.00, 'Soft microfiber bed sheet set includes fitted and flat sheets', 'https://images.example.com/sheets-101.jpg', 156, 2.5, '14x11x4', '2024-01-10'),
('HOME-102', 'Pillow Set 2-Pack', 3, 29.99, 15.00, 'Hypoallergenic pillows with adjustable fill', 'https://images.example.com/pillows-102.jpg', 198, 3.0, '20x14x8', '2024-02-15'),
('HOME-103', 'Comforter King Size', 3, 79.99, 40.00, 'All-season comforter with down alternative fill', 'https://images.example.com/comforter-103.jpg', 89, 5.0, '16x13x8', '2024-01-20'),
('HOME-104', 'Bath Towel Set 6-Piece', 3, 34.99, 17.00, 'Absorbent cotton towel set in multiple colors', 'https://images.example.com/towels-104.jpg', 167, 3.5, '15x12x6', '2024-03-05'),
('HOME-105', 'Shower Curtain', 3, 19.99, 10.00, 'Water-resistant shower curtain with decorative design', 'https://images.example.com/curtain-105.jpg', 234, 1.0, '72x72x1', '2024-02-10'),
('HOME-106', 'Bathroom Mat Non-Slip', 3, 16.99, 8.00, 'Memory foam bathroom mat with rubber backing', 'https://images.example.com/bathmat-106.jpg', 267, 1.2, '24x17x2', '2024-01-25'),
('HOME-107', 'Kitchen Knife Set', 3, 49.99, 25.00, '15-piece knife set with wooden block', 'https://images.example.com/knives-107.jpg', 112, 4.0, '12x8x10', '2024-03-12'),
('HOME-108', 'Cookware Set 10-Piece', 3, 129.99, 65.00, 'Non-stick cookware set with glass lids', 'https://images.example.com/cookware-108.jpg', 76, 15.0, '16x14x12', '2024-02-20'),
('HOME-109', 'Dinnerware Set 16-Piece', 3, 59.99, 30.00, 'Ceramic dinnerware set service for 4', 'https://images.example.com/dinnerware-109.jpg', 98, 18.0, '14x14x10', '2024-01-15'),
('HOME-110', 'Glassware Set 12-Piece', 3, 24.99, 12.00, 'Durable drinking glasses in various sizes', 'https://images.example.com/glassware-110.jpg', 178, 6.0, '12x10x8', '2024-03-18'),
('HOME-111', 'Flatware Set 20-Piece', 3, 34.99, 17.00, 'Stainless steel silverware set service for 4', 'https://images.example.com/flatware-111.jpg', 145, 2.5, '12x9x3', '2024-02-08'),
('HOME-112', 'Storage Containers 10-Pack', 3, 29.99, 15.00, 'BPA-free food storage containers with lids', 'https://images.example.com/containers-112.jpg', 234, 2.0, '12x10x6', '2024-01-30'),
('HOME-113', 'Trash Can 13 Gallon', 3, 39.99, 20.00, 'Step-on trash can with liner lock', 'https://images.example.com/trashcan-113.jpg', 134, 6.0, '16x13x24', '2024-03-08'),
('HOME-114', 'Vacuum Cleaner Upright', 3, 149.99, 75.00, 'Powerful upright vacuum with HEPA filter', 'https://images.example.com/vacuum-114.jpg', 67, 18.0, '14x12x42', '2024-02-25'),
('HOME-115', 'Mop and Bucket Set', 3, 34.99, 17.00, 'Spin mop with bucket and reusable pads', 'https://images.example.com/mop-115.jpg', 156, 5.0, '12x12x10', '2024-01-18'),
('HOME-116', 'Iron Steam', 3, 44.99, 22.00, 'Steam iron with anti-drip and auto shut-off', 'https://images.example.com/iron-116.jpg', 123, 3.0, '12x6x6', '2024-03-14'),
('HOME-117', 'Ironing Board', 3, 39.99, 20.00, 'Adjustable ironing board with iron rest', 'https://images.example.com/ironboard-117.jpg', 98, 10.0, '54x14x3', '2024-02-05'),
('HOME-118', 'Laundry Basket', 3, 19.99, 10.00, 'Collapsible laundry basket with handles', 'https://images.example.com/basket-118.jpg', 189, 1.5, '22x15x12', '2024-01-28'),
('HOME-119', 'Clothes Hangers 50-Pack', 3, 16.99, 8.00, 'Velvet hangers with non-slip surface', 'https://images.example.com/hangers-119.jpg', 287, 3.0, '17x1x9', '2024-03-20'),
('HOME-120', 'Area Rug 5x7', 3, 89.99, 45.00, 'Modern area rug for living room or bedroom', 'https://images.example.com/rug-120.jpg', 76, 12.0, '84x60x1', '2024-02-18'),
('HOME-121', 'Curtains Blackout 2-Panel', 3, 34.99, 17.00, 'Room-darkening curtains with grommet top', 'https://images.example.com/curtains-121.jpg', 145, 3.0, '84x52x2', '2024-01-12'),
('HOME-122', 'Wall Clock', 3, 24.99, 12.00, 'Silent wall clock with modern design', 'https://images.example.com/clock-122.jpg', 198, 1.5, '12x12x2', '2024-03-10'),
('HOME-123', 'Picture Frame Set 5-Piece', 3, 29.99, 15.00, 'Gallery wall frame set in various sizes', 'https://images.example.com/frames-123.jpg', 167, 4.0, '16x12x4', '2024-02-22'),
('HOME-124', 'Table Lamp', 3, 39.99, 20.00, 'Bedside table lamp with fabric shade', 'https://images.example.com/lamp-124.jpg', 134, 3.5, '10x10x18', '2024-01-16'),
('HOME-125', 'Floor Lamp', 3, 69.99, 35.00, 'Standing floor lamp with adjustable head', 'https://images.example.com/floorlamp-125.jpg', 87, 8.0, '10x10x65', '2024-03-06'),
('HOME-126', 'Candles Scented 3-Pack', 3, 24.99, 12.00, 'Soy wax candles in relaxing scents', 'https://images.example.com/candles-126.jpg', 234, 2.0, '10x8x4', '2024-02-12'),
('HOME-127', 'Throw Blanket', 3, 29.99, 15.00, 'Soft fleece throw blanket for couch', 'https://images.example.com/blanket-127.jpg', 189, 2.0, '60x50x3', '2024-01-08'),
('HOME-128', 'Decorative Pillows 2-Pack', 3, 26.99, 13.00, 'Accent pillows with removable covers', 'https://images.example.com/decpillows-128.jpg', 156, 1.5, '18x18x6', '2024-03-16'),
('HOME-129', 'Vase Ceramic', 3, 19.99, 10.00, 'Modern ceramic vase for flowers', 'https://images.example.com/vase-129.jpg', 178, 2.0, '8x8x12', '2024-02-20'),
('HOME-130', 'Mirror Wall Mounted', 3, 49.99, 25.00, 'Large wall mirror with decorative frame', 'https://images.example.com/mirror-130.jpg', 98, 10.0, '30x24x2', '2024-01-24'),
('HOME-131', 'Coat Rack Standing', 3, 44.99, 22.00, 'Freestanding coat rack with multiple hooks', 'https://images.example.com/coatrack-131.jpg', 112, 7.0, '18x18x70', '2024-03-11'),
('HOME-132', 'Shoe Rack 3-Tier', 3, 29.99, 15.00, 'Stackable shoe organizer holds 12 pairs', 'https://images.example.com/shoerack-132.jpg', 145, 5.0, '36x12x20', '2024-02-07'),
('HOME-133', 'Bookshelf 5-Shelf', 3, 79.99, 40.00, 'Wood bookshelf with adjustable shelves', 'https://images.example.com/bookshelf-133.jpg', 76, 35.0, '36x12x72', '2024-01-19'),
('HOME-134', 'Desk Organizer', 3, 19.99, 10.00, 'Desktop organizer with multiple compartments', 'https://images.example.com/organizer-134.jpg', 234, 1.0, '12x6x5', '2024-03-13'),
('HOME-135', 'Office Chair', 3, 129.99, 65.00, 'Ergonomic office chair with lumbar support', 'https://images.example.com/chair-135.jpg', 54, 25.0, '26x26x40', '2024-02-16'),
('HOME-136', 'File Cabinet 2-Drawer', 3, 89.99, 45.00, 'Locking file cabinet for documents', 'https://images.example.com/filecab-136.jpg', 67, 40.0, '18x24x28', '2024-01-11'),
('HOME-137', 'Desk Lamp LED', 3, 34.99, 17.00, 'Adjustable LED desk lamp with USB port', 'https://images.example.com/desklamp-137.jpg', 167, 1.5, '8x6x18', '2024-03-09'),
('HOME-138', 'Whiteboard 24x36', 3, 29.99, 15.00, 'Magnetic dry erase whiteboard with marker tray', 'https://images.example.com/whiteboard-138.jpg', 134, 5.0, '36x24x1', '2024-02-23'),
('HOME-139', 'Plant Pot Set 3-Piece', 3, 24.99, 12.00, 'Ceramic plant pots with drainage holes', 'https://images.example.com/pots-139.jpg', 198, 4.0, '12x10x8', '2024-01-14'),
('HOME-140', 'Garden Tools Set', 3, 39.99, 20.00, 'Essential garden tools with carrying case', 'https://images.example.com/gardentools-140.jpg', 123, 6.0, '14x10x4', '2024-03-17'),
('HOME-141', 'Watering Can', 3, 16.99, 8.00, '1-gallon watering can with long spout', 'https://images.example.com/watercan-141.jpg', 178, 1.0, '14x8x12', '2024-02-11'),
('HOME-142', 'Outdoor Thermometer', 3, 14.99, 7.00, 'Weather-resistant outdoor thermometer', 'https://images.example.com/thermometer-142.jpg', 189, 0.5, '12x3x1', '2024-01-27'),
('HOME-143', 'Door Mat', 3, 19.99, 10.00, 'Weather-resistant welcome mat', 'https://images.example.com/doormat-143.jpg', 234, 2.0, '30x18x1', '2024-03-15'),
('HOME-144', 'Umbrella Stand', 3, 29.99, 15.00, 'Metal umbrella stand with drip tray', 'https://images.example.com/umbrellastand-144.jpg', 145, 5.0, '10x10x22', '2024-02-19'),
('HOME-145', 'Key Holder Wall Mount', 3, 16.99, 8.00, 'Decorative key holder with hooks', 'https://images.example.com/keyholder-145.jpg', 267, 0.8, '12x4x3', '2024-01-21'),
('HOME-146', 'Magazine Rack', 3, 24.99, 12.00, 'Modern magazine rack for living room', 'https://images.example.com/magazinerack-146.jpg', 156, 3.0, '14x8x16', '2024-03-19'),
('HOME-147', 'Trash Bags 100-Count', 3, 12.99, 6.00, 'Heavy-duty trash bags with drawstring', 'https://images.example.com/trashbags-147.jpg', 345, 2.5, '12x8x4', '2024-02-13'),
('HOME-148', 'Paper Towels 12-Pack', 3, 19.99, 10.00, 'Absorbent paper towel rolls bulk pack', 'https://images.example.com/papertowels-148.jpg', 289, 8.0, '18x12x12', '2024-01-07'),
('HOME-149', 'Dish Soap 3-Pack', 3, 9.99, 5.00, 'Concentrated dish soap with grease-cutting power', 'https://images.example.com/dishsoap-149.jpg', 398, 3.0, '10x8x6', '2024-03-21'),
('HOME-150', 'Sponges 24-Pack', 3, 11.99, 6.00, 'Non-scratch scrub sponges for dishes', 'https://images.example.com/sponges-150.jpg', 456, 1.0, '12x8x4', '2024-02-17');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 4: BOOKS, SPORTS, TOYS (Good Quality: 151-200)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('BOOK-151', 'Fiction Novel Bestseller', 4, 14.99, 7.00, 'Gripping fiction novel by award-winning author', 'https://images.example.com/book-151.jpg', 234, 0.8, '9x6x1', '2024-01-10'),
('BOOK-152', 'Self-Help Book', 4, 16.99, 8.00, 'Transform your life with proven strategies', 'https://images.example.com/book-152.jpg', 189, 0.9, '9x6x1', '2024-02-15'),
('BOOK-153', 'Cookbook Italian', 4, 24.99, 12.00, 'Authentic Italian recipes for home cooking', 'https://images.example.com/cookbook-153.jpg', 156, 1.5, '10x8x1', '2024-01-20'),
('BOOK-154', 'Mystery Thriller', 4, 15.99, 8.00, 'Page-turning mystery with unexpected twists', 'https://images.example.com/mystery-154.jpg', 178, 0.8, '9x6x1', '2024-03-05'),
('BOOK-155', 'Biography Historical', 4, 19.99, 10.00, 'Inspiring biography of influential leader', 'https://images.example.com/biography-155.jpg', 145, 1.0, '9x6x1.5', '2024-02-10'),
('BOOK-156', 'Childrens Picture Book', 4, 12.99, 6.00, 'Colorful picture book for ages 3-7', 'https://images.example.com/kidsbook-156.jpg', 267, 0.6, '11x9x0.5', '2024-01-25'),
('BOOK-157', 'Science Fiction Novel', 4, 16.99, 8.00, 'Epic sci-fi adventure set in distant future', 'https://images.example.com/scifi-157.jpg', 167, 0.9, '9x6x1', '2024-03-12'),
('BOOK-158', 'Romance Novel', 4, 13.99, 7.00, 'Heartwarming romance with happy ending', 'https://images.example.com/romance-158.jpg', 198, 0.7, '9x6x1', '2024-02-20'),
('BOOK-159', 'History Book', 4, 22.99, 11.00, 'Comprehensive history of ancient civilizations', 'https://images.example.com/history-159.jpg', 123, 1.3, '10x7x1.5', '2024-01-15'),
('BOOK-160', 'Poetry Collection', 4, 14.99, 7.00, 'Beautiful poetry collection from modern poets', 'https://images.example.com/poetry-160.jpg', 134, 0.6, '8x5x0.8', '2024-03-18'),
('SPRT-161', 'Yoga Mat Premium', 5, 39.99, 20.00, 'Extra-thick yoga mat with alignment marks', 'https://images.example.com/yogamat-161.jpg', 156, 3.0, '72x24x0.6', '2024-02-08'),
('SPRT-162', 'Dumbbells Set 20lb', 5, 49.99, 25.00, 'Pair of rubber-coated dumbbells', 'https://images.example.com/dumbbells-162.jpg', 98, 20.0, '12x6x6', '2024-01-30'),
('SPRT-163', 'Resistance Bands Set', 5, 19.99, 10.00, '5-piece resistance band set with handles', 'https://images.example.com/bands-163.jpg', 234, 1.0, '10x8x3', '2024-03-08'),
('SPRT-164', 'Jump Rope', 5, 9.99, 5.00, 'Adjustable speed jump rope for cardio', 'https://images.example.com/jumprope-164.jpg', 345, 0.3, '10x4x2', '2024-02-25'),
('SPRT-165', 'Exercise Ball 65cm', 5, 24.99, 12.00, 'Anti-burst exercise ball with pump', 'https://images.example.com/ball-165.jpg', 145, 2.5, '26x10x10', '2024-01-18'),
('SPRT-166', 'Foam Roller', 5, 29.99, 15.00, 'High-density foam roller for muscle recovery', 'https://images.example.com/roller-166.jpg', 178, 1.5, '36x6x6', '2024-03-14'),
('SPRT-167', 'Kettlebell 15lb', 5, 34.99, 17.00, 'Cast iron kettlebell with wide handle', 'https://images.example.com/kettlebell-167.jpg', 123, 15.0, '10x8x8', '2024-02-05'),
('SPRT-168', 'Bike Lock Heavy Duty', 5, 29.99, 15.00, 'U-lock with mounting bracket', 'https://images.example.com/bikelock-168.jpg', 189, 2.0, '12x8x3', '2024-01-28'),
('SPRT-169', 'Water Bottle 32oz', 5, 16.99, 8.00, 'Insulated stainless steel water bottle', 'https://images.example.com/bottle-169.jpg', 298, 0.8, '11x3x3', '2024-03-20'),
('SPRT-170', 'Gym Bag Duffel', 5, 39.99, 20.00, 'Large duffel bag with shoe compartment', 'https://images.example.com/gymbag-170.jpg', 167, 1.5, '24x12x12', '2024-02-18'),
('SPRT-171', 'Tennis Racket', 5, 79.99, 40.00, 'Lightweight tennis racket for intermediate players', 'https://images.example.com/racket-171.jpg', 87, 0.9, '27x11x2', '2024-01-12'),
('SPRT-172', 'Soccer Ball Official Size', 5, 24.99, 12.00, 'Regulation size 5 soccer ball', 'https://images.example.com/soccer-172.jpg', 156, 1.2, '9x9x9', '2024-03-10'),
('SPRT-173', 'Basketball Indoor/Outdoor', 5, 29.99, 15.00, 'Durable basketball for all surfaces', 'https://images.example.com/basketball-173.jpg', 134, 1.5, '10x10x10', '2024-02-22'),
('SPRT-174', 'Baseball Glove', 5, 49.99, 25.00, 'Leather baseball glove break-in ready', 'https://images.example.com/glove-174.jpg', 112, 1.0, '12x8x4', '2024-01-16'),
('SPRT-175', 'Golf Balls 12-Pack', 5, 24.99, 12.00, 'High-performance golf balls for distance', 'https://images.example.com/golfballs-175.jpg', 189, 1.5, '8x6x4', '2024-03-06'),
('SPRT-176', 'Skateboard Complete', 5, 69.99, 35.00, 'Complete skateboard ready to ride', 'https://images.example.com/skateboard-176.jpg', 98, 5.0, '32x8x6', '2024-02-12'),
('SPRT-177', 'Boxing Gloves 12oz', 5, 44.99, 22.00, 'Training boxing gloves with wrist support', 'https://images.example.com/boxing-177.jpg', 134, 1.5, '12x8x8', '2024-01-08'),
('SPRT-178', 'Swim Goggles', 5, 14.99, 7.00, 'Anti-fog swim goggles with UV protection', 'https://images.example.com/goggles-178.jpg', 267, 0.2, '7x4x3', '2024-03-16'),
('SPRT-179', 'Camping Tent 4-Person', 5, 129.99, 65.00, 'Waterproof camping tent with easy setup', 'https://images.example.com/tent-179.jpg', 67, 15.0, '24x12x12', '2024-02-20'),
('SPRT-180', 'Sleeping Bag', 5, 49.99, 25.00, 'Mummy sleeping bag rated for 40°F', 'https://images.example.com/sleepbag-180.jpg', 123, 4.0, '16x10x10', '2024-01-24'),
('TOY-181', 'Building Blocks 500-Piece', 6, 34.99, 17.00, 'Creative building blocks compatible with major brands', 'https://images.example.com/blocks-181.jpg', 189, 3.0, '14x10x8', '2024-03-11'),
('TOY-182', 'Remote Control Car', 6, 49.99, 25.00, 'Fast RC car with rechargeable battery', 'https://images.example.com/rccar-182.jpg', 134, 2.5, '16x10x8', '2024-02-07'),
('TOY-183', 'Board Game Family', 6, 29.99, 15.00, 'Classic board game for family game night', 'https://images.example.com/boardgame-183.jpg', 167, 2.0, '16x11x3', '2024-01-19'),
('TOY-184', 'Puzzle 1000-Piece', 6, 16.99, 8.00, 'Challenging jigsaw puzzle with beautiful image', 'https://images.example.com/puzzle-184.jpg', 234, 1.5, '14x10x2', '2024-03-13'),
('TOY-185', 'Action Figure Set', 6, 24.99, 12.00, 'Poseable action figures with accessories', 'https://images.example.com/actionfigure-185.jpg', 198, 0.8, '12x8x3', '2024-02-16'),
('TOY-186', 'Doll with Accessories', 6, 34.99, 17.00, 'Fashion doll with outfit and accessories', 'https://images.example.com/doll-186.jpg', 145, 1.0, '14x6x3', '2024-01-11'),
('TOY-187', 'Art Set Kids', 6, 19.99, 10.00, 'Complete art set with crayons, markers, and paper', 'https://images.example.com/artset-187.jpg', 223, 2.0, '14x10x3', '2024-03-09'),
('TOY-188', 'Stuffed Animal Teddy Bear', 6, 19.99, 10.00, 'Soft and cuddly teddy bear', 'https://images.example.com/teddy-188.jpg', 267, 0.8, '14x10x8', '2024-01-14'),
('TOY-189', 'STEM Learning Kit', 6, 44.99, 22.00, 'Educational science kit for kids', 'https://images.example.com/stem-189.jpg', 134, 3.0, '12x10x6', '2024-03-17'),
('TOY-190', 'Musical Instrument Toy', 6, 29.99, 15.00, 'Kids musical keyboard with sound effects', 'https://images.example.com/music-190.jpg', 156, 2.0, '18x10x4', '2024-02-11'),
('TOY-191', 'Playdough Set', 6, 14.99, 7.00, '10-color playdough set with tools', 'https://images.example.com/playdough-191.jpg', 298, 2.5, '12x10x4', '2024-01-27'),
('TOY-192', 'Toy Train Set', 6, 59.99, 30.00, 'Electric train set with track and accessories', 'https://images.example.com/train-192.jpg', 98, 5.0, '20x16x6', '2024-03-15'),
('TOY-193', 'Construction Vehicle Set', 6, 39.99, 20.00, 'Set of 5 construction vehicle toys', 'https://images.example.com/construction-193.jpg', 167, 3.0, '16x12x6', '2024-02-19'),
('TOY-194', 'Outdoor Play Set', 6, 79.99, 40.00, 'Backyard play equipment for kids', 'https://images.example.com/outdoor-194.jpg', 54, 25.0, '36x24x12', '2024-01-21'),
('TOY-195', 'Card Game Family', 6, 12.99, 6.00, 'Fun card game for all ages', 'https://images.example.com/cardgame-195.jpg', 345, 0.5, '6x4x2', '2024-03-19'),
('TOY-196', 'Robot Building Kit', 6, 69.99, 35.00, 'Build and program your own robot', 'https://images.example.com/robot-196.jpg', 89, 4.0, '14x10x8', '2024-02-13'),
('TOY-197', 'Bubble Machine', 6, 24.99, 12.00, 'Automatic bubble maker with solution', 'https://images.example.com/bubbles-197.jpg', 178, 1.5, '10x8x8', '2024-01-07'),
('TOY-198', 'Water Blaster', 6, 16.99, 8.00, 'Super soaker water gun for summer fun', 'https://images.example.com/watergun-198.jpg', 234, 1.0, '20x8x4', '2024-03-21'),
('TOY-199', 'Coloring Book Set', 6, 9.99, 5.00, 'Activity coloring books with stickers', 'https://images.example.com/coloring-199.jpg', 389, 1.0, '11x9x1', '2024-02-17'),
('TOY-200', 'Sandbox Toys Set', 6, 14.99, 7.00, 'Beach and sandbox toy set with bucket', 'https://images.example.com/sandbox-200.jpg', 234, 1.5, '12x10x8', '2024-01-05');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 5: FOOD & BEVERAGE, BEAUTY (Good Quality: 201-230)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('FOOD-201', 'Organic Coffee Beans 1lb', 7, 15.99, 8.00, 'Premium organic coffee beans medium roast', 'https://images.example.com/coffee-201.jpg', 234, 1.0, '10x6x3', '2024-01-10'),
('FOOD-202', 'Green Tea 100 Bags', 7, 12.99, 6.00, 'Organic green tea with antioxidants', 'https://images.example.com/tea-202.jpg', 298, 0.5, '8x6x4', '2024-02-15'),
('FOOD-203', 'Protein Powder Vanilla 2lb', 7, 39.99, 20.00, 'Whey protein powder for muscle building', 'https://images.example.com/protein-203.jpg', 145, 2.5, '8x6x8', '2024-01-20'),
('FOOD-204', 'Granola Bars 18-Pack', 7, 11.99, 6.00, 'Healthy granola bars with nuts and honey', 'https://images.example.com/granola-204.jpg', 367, 1.5, '10x8x6', '2024-03-05'),
('FOOD-205', 'Trail Mix Variety Pack', 7, 14.99, 7.00, 'Mixed nuts and dried fruit snack packs', 'https://images.example.com/trailmix-205.jpg', 278, 2.0, '12x8x6', '2024-02-10'),
('FOOD-206', 'Olive Oil Extra Virgin', 7, 19.99, 10.00, 'Cold-pressed extra virgin olive oil', 'https://images.example.com/oliveoil-206.jpg', 189, 3.0, '10x4x12', '2024-01-25'),
('FOOD-207', 'Pasta Variety Pack', 7, 9.99, 5.00, 'Assorted pasta shapes 6-pack', 'https://images.example.com/pasta-207.jpg', 345, 4.0, '12x10x8', '2024-03-12'),
('FOOD-208', 'Tomato Sauce 6-Pack', 7, 13.99, 7.00, 'Traditional Italian tomato sauce', 'https://images.example.com/sauce-208.jpg', 256, 6.0, '14x10x8', '2024-02-20'),
('FOOD-209', 'Honey Organic 16oz', 7, 12.99, 6.00, 'Pure organic honey from wildflowers', 'https://images.example.com/honey-209.jpg', 198, 1.5, '6x4x6', '2024-01-15'),
('FOOD-210', 'Peanut Butter 28oz', 7, 7.99, 4.00, 'Creamy peanut butter all-natural', 'https://images.example.com/peanutbutter-210.jpg', 389, 2.0, '6x4x6', '2024-03-18'),
('BEAUTY-211', 'Face Moisturizer', 8, 24.99, 12.00, 'Hydrating face cream with SPF 30', 'https://images.example.com/moisturizer-211.jpg', 234, 0.5, '6x3x3', '2024-02-08'),
('BEAUTY-212', 'Shampoo and Conditioner Set', 8, 19.99, 10.00, 'Sulfate-free hair care duo', 'https://images.example.com/shampoo-212.jpg', 298, 2.0, '10x4x8', '2024-01-30'),
('BEAUTY-213', 'Facial Cleanser', 8, 16.99, 8.00, 'Gentle foaming face wash for all skin types', 'https://images.example.com/cleanser-213.jpg', 267, 0.6, '6x3x8', '2024-03-08'),
('BEAUTY-214', 'Body Lotion 16oz', 8, 14.99, 7.00, 'Nourishing body lotion with shea butter', 'https://images.example.com/lotion-214.jpg', 345, 1.2, '8x4x8', '2024-02-25'),
('BEAUTY-215', 'Lip Balm 3-Pack', 8, 8.99, 4.00, 'Moisturizing lip balm in assorted flavors', 'https://images.example.com/lipbalm-215.jpg', 456, 0.2, '4x3x2', '2024-01-18'),
('BEAUTY-216', 'Makeup Remover Wipes', 8, 9.99, 5.00, 'Gentle makeup remover wipes 50-count', 'https://images.example.com/wipes-216.jpg', 389, 0.8, '8x6x3', '2024-03-14'),
('BEAUTY-217', 'Face Mask Sheet 10-Pack', 8, 12.99, 6.00, 'Hydrating sheet masks with hyaluronic acid', 'https://images.example.com/facemask-217.jpg', 278, 0.6, '8x6x2', '2024-02-05'),
('BEAUTY-218', 'Nail Polish Set 6-Pack', 8, 16.99, 8.00, 'Long-lasting nail polish in trendy colors', 'https://images.example.com/nailpolish-218.jpg', 234, 0.8, '8x6x4', '2024-01-28'),
('BEAUTY-219', 'Makeup Brush Set', 8, 29.99, 15.00, 'Professional makeup brushes 12-piece set', 'https://images.example.com/brushes-219.jpg', 189, 0.8, '10x6x3', '2024-03-20'),
('BEAUTY-220', 'Perfume Eau de Parfum', 8, 49.99, 25.00, 'Elegant floral fragrance 50ml', 'https://images.example.com/perfume-220.jpg', 145, 0.5, '5x3x6', '2024-02-18'),
('BEAUTY-221', 'Face Serum Anti-Aging', 8, 34.99, 17.00, 'Vitamin C serum for bright skin', 'https://images.example.com/serum-221.jpg', 198, 0.4, '5x2x5', '2024-01-12'),
('BEAUTY-222', 'Hair Mask Treatment', 8, 19.99, 10.00, 'Deep conditioning hair mask', 'https://images.example.com/hairmask-222.jpg', 234, 1.0, '6x4x4', '2024-03-10'),
('BEAUTY-223', 'Deodorant Natural', 8, 9.99, 5.00, 'Aluminum-free natural deodorant', 'https://images.example.com/deodorant-223.jpg', 367, 0.4, '6x2x2', '2024-02-22'),
('BEAUTY-224', 'Sunscreen SPF 50', 8, 14.99, 7.00, 'Broad-spectrum sunscreen water-resistant', 'https://images.example.com/sunscreen-224.jpg', 289, 0.8, '7x3x2', '2024-01-16'),
('BEAUTY-225', 'Bath Bombs Set 6-Pack', 8, 16.99, 8.00, 'Fizzy bath bombs with essential oils', 'https://images.example.com/bathbombs-225.jpg', 256, 1.2, '10x8x4', '2024-03-06'),
('BEAUTY-226', 'Body Wash Moisturizing', 8, 11.99, 6.00, 'Gentle body wash with aloe vera', 'https://images.example.com/bodywash-226.jpg', 345, 1.5, '8x4x10', '2024-02-12'),
('BEAUTY-227', 'Face Toner', 8, 18.99, 9.00, 'Alcohol-free toner with witch hazel', 'https://images.example.com/toner-227.jpg', 234, 0.8, '6x3x8', '2024-01-08'),
('BEAUTY-228', 'Eye Cream Dark Circles', 8, 27.99, 14.00, 'Brightening eye cream reduces puffiness', 'https://images.example.com/eyecream-228.jpg', 189, 0.3, '4x2x4', '2024-03-16'),
('BEAUTY-229', 'Mascara Waterproof', 8, 12.99, 6.00, 'Lengthening mascara long-lasting formula', 'https://images.example.com/mascara-229.jpg', 298, 0.2, '6x2x2', '2024-02-20'),
('BEAUTY-230', 'Foundation Liquid', 8, 24.99, 12.00, 'Full-coverage liquid foundation 8 shades', 'https://images.example.com/foundation-230.jpg', 167, 0.5, '5x2x6', '2024-01-24');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 6: Products with INVALID/ZERO PRICES (36 products = 9%)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('ELEC-231', 'Smart Speaker Voice Assistant', 1, 0.00, 40.00, 'Voice-controlled smart speaker with AI', 'https://images.example.com/speaker-231.jpg', 87, 1.8, '6x6x6', '2024-03-11'),
('CLTH-232', 'Men Running Jacket', 2, NULL, 30.00, 'Lightweight running jacket with reflective strips', 'https://images.example.com/jacket-232.jpg', 134, 0.7, '14x12x3', '2024-02-07'),
('HOME-233', 'Coffee Grinder Electric', 3, -15.99, 25.00, 'Burr coffee grinder with multiple settings', 'https://images.example.com/grinder-233.jpg', 98, 3.5, '8x6x10', '2024-01-19'),
('BOOK-234', 'Business Strategy Guide', 4, 0.00, 10.00, 'Comprehensive guide to business planning', 'https://images.example.com/business-234.jpg', 156, 1.0, '9x6x1', '2024-03-13'),
('SPRT-235', 'Bike Helmet Adult', 5, NULL, 25.00, 'Safety helmet with adjustable fit', 'https://images.example.com/helmet-235.jpg', 123, 1.2, '12x10x9', '2024-02-16'),
('TOY-236', 'Educational Tablet Kids', 6, 0.00, 35.00, 'Learning tablet with games and activities', 'https://images.example.com/tablet-236.jpg', 89, 1.5, '10x8x1', '2024-01-11'),
('FOOD-237', 'Chocolate Bar Organic', 7, -3.99, 2.00, 'Dark chocolate 70% cacao organic', 'https://images.example.com/chocolate-237.jpg', 445, 0.3, '6x3x0.5', '2024-03-09'),
('BEAUTY-238', 'Hair Styling Gel', 8, NULL, 6.00, 'Strong hold hair gel alcohol-free', 'https://images.example.com/gel-238.jpg', 267, 0.8, '6x3x6', '2024-02-23'),
('ELEC-239', 'Fitness Tracker Band', 1, 0.00, 35.00, 'Activity tracker with heart rate monitor', 'https://images.example.com/tracker-239.jpg', 145, 0.2, '10x2x1', '2024-01-14'),
('CLTH-240', 'Women Winter Coat', 2, NULL, 60.00, 'Warm puffer coat with fur-lined hood', 'https://images.example.com/coat-240.jpg', 76, 2.5, '16x14x6', '2024-03-17'),
('HOME-241', 'Electric Can Opener', 3, -9.99, 12.00, 'Automatic can opener hands-free operation', 'https://images.example.com/canopener-241.jpg', 178, 1.5, '8x6x6', '2024-02-11'),
('BOOK-242', 'Travel Guide Europe', 4, 0.00, 12.00, 'Complete travel guide to European cities', 'https://images.example.com/travel-242.jpg', 134, 1.2, '9x7x1', '2024-01-27'),
('SPRT-243', 'Fishing Rod Combo', 5, NULL, 40.00, 'Fishing rod and reel combo for beginners', 'https://images.example.com/fishing-243.jpg', 67, 3.0, '72x4x4', '2024-03-15'),
('TOY-244', 'Remote Control Helicopter', 6, 0.00, 30.00, 'Easy-to-fly RC helicopter with lights', 'https://images.example.com/heli-244.jpg', 98, 1.8, '14x10x6', '2024-02-19'),
('FOOD-245', 'Protein Bars 12-Pack', 7, -19.99, 10.00, 'High-protein snack bars assorted flavors', 'https://images.example.com/proteinbar-245.jpg', 234, 1.5, '10x8x4', '2024-01-21'),
('BEAUTY-246', 'Body Scrub Exfoliating', 8, NULL, 8.00, 'Sugar scrub with coconut oil', 'https://images.example.com/scrub-246.jpg', 189, 1.0, '6x4x4', '2024-03-19'),
('ELEC-247', 'Digital Kitchen Scale', 1, 0.00, 12.00, 'Precise kitchen scale with tare function', 'https://images.example.com/scale-247.jpg', 167, 1.2, '8x6x2', '2024-02-13'),
('CLTH-248', 'Kids Rain Boots', 2, NULL, 15.00, 'Waterproof rain boots for children', 'https://images.example.com/boots-248.jpg', 198, 1.5, '12x6x8', '2024-01-07'),
('HOME-249', 'Wall Shelf Floating', 3, -12.99, 8.00, 'Modern floating shelf for home decor', 'https://images.example.com/shelf-249.jpg', 145, 3.0, '24x8x2', '2024-03-21'),
('BOOK-250', 'Gardening Handbook', 4, 0.00, 9.00, 'Complete guide to organic gardening', 'https://images.example.com/garden-250.jpg', 167, 0.9, '9x6x1', '2024-02-17'),
('SPRT-251', 'Camping Backpack 50L', 5, NULL, 45.00, 'Large hiking backpack with rain cover', 'https://images.example.com/backpack-251.jpg', 89, 4.0, '26x14x10', '2024-01-05'),
('TOY-252', 'Magic Trick Set', 6, 0.00, 12.00, 'Beginner magic kit with instructions', 'https://images.example.com/magic-252.jpg', 156, 1.5, '12x10x4', '2024-03-23'),
('FOOD-253', 'Multivitamin Gummies', 7, -14.99, 8.00, 'Daily vitamins in gummy form 90-count', 'https://images.example.com/vitamin-253.jpg', 278, 0.8, '6x4x6', '2024-02-09'),
('BEAUTY-254', 'Tweezers Precision', 8, NULL, 5.00, 'Stainless steel tweezers for eyebrows', 'https://images.example.com/tweezers-254.jpg', 345, 0.1, '5x1x0.5', '2024-01-13'),
('ELEC-255', 'Extension Cord 25ft', 1, 0.00, 10.00, 'Heavy-duty extension cord indoor/outdoor', 'https://images.example.com/cord-255.jpg', 234, 2.5, '25x1x1', '2024-03-25'),
('CLTH-256', 'Baby Clothes 5-Pack', 2, NULL, 18.00, 'Soft cotton baby onesies assorted colors', 'https://images.example.com/baby-256.jpg', 189, 0.6, '12x10x4', '2024-02-21'),
('HOME-257', 'Dish Drying Rack', 3, -8.99, 12.00, 'Stainless steel dish rack with drainboard', 'https://images.example.com/dishrack-257.jpg', 167, 3.0, '18x14x6', '2024-01-17'),
('BOOK-258', 'Photography Basics', 4, 0.00, 13.00, 'Learn photography fundamentals', 'https://images.example.com/photo-258.jpg', 134, 1.1, '10x8x1', '2024-03-27'),
('SPRT-259', 'Badminton Set', 5, NULL, 20.00, 'Complete badminton set with net and rackets', 'https://images.example.com/badminton-259.jpg', 112, 4.0, '26x10x4', '2024-02-15'),
('TOY-260', 'Wooden Puzzle 3D', 6, 0.00, 15.00, '3D wooden puzzle mechanical model', 'https://images.example.com/woodpuzzle-260.jpg', 145, 1.2, '12x10x3', '2024-01-09'),
('FOOD-261', 'Spice Rack with Spices', 7, -24.99, 15.00, 'Spice organizer with 20 filled jars', 'https://images.example.com/spices-261.jpg', 123, 4.0, '14x8x8', '2024-03-29'),
('BEAUTY-262', 'Nail File Set', 8, NULL, 4.00, 'Professional nail files and buffer', 'https://images.example.com/nailfile-262.jpg', 389, 0.2, '8x4x1', '2024-02-25'),
('ELEC-263', 'Night Light LED', 1, 0.00, 6.00, 'Plug-in LED night light with sensor', 'https://images.example.com/nightlight-263.jpg', 456, 0.3, '4x3x2', '2024-01-19'),
('CLTH-264', 'Compression Socks', 2, NULL, 10.00, 'Medical compression socks 3-pack', 'https://images.example.com/socks-264.jpg', 234, 0.4, '10x6x2', '2024-03-31'),
('HOME-265', 'Drawer Organizer Set', 3, -11.99, 8.00, 'Adjustable drawer dividers 6-piece', 'https://images.example.com/organizer-265.jpg', 198, 1.5, '16x12x4', '2024-02-27'),
('BOOK-266', 'Meditation Guide', 4, 0.00, 8.00, 'Mindfulness and meditation practices', 'https://images.example.com/meditation-266.jpg', 178, 0.7, '8x5x0.8', '2024-01-23');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 7: Products with MISSING DESCRIPTIONS (104 products = 26%)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('ELEC-267', 'Wireless Charger Pad', 1, 29.99, 15.00, NULL, 'https://images.example.com/charger-267.jpg', 234, 0.5, '4x4x1', '2024-04-02'),
('CLTH-268', 'Women Ankle Socks 6-Pack', 2, 12.99, 6.00, NULL, 'https://images.example.com/socks-268.jpg', 345, 0.4, '10x8x4', '2024-03-01'),
('HOME-269', 'Measuring Cups Set', 3, 14.99, 7.00, NULL, 'https://images.example.com/measuring-269.jpg', 267, 1.0, '8x6x6', '2024-02-05'),
('BOOK-270', 'Coloring Book Adult', 4, 9.99, 5.00, NULL, 'https://images.example.com/coloring-270.jpg', 298, 0.6, '11x9x0.5', '2024-01-11'),
('SPRT-271', 'Bike Water Bottle Holder', 5, 8.99, 4.00, NULL, 'https://images.example.com/holder-271.jpg', 389, 0.3, '6x4x8', '2024-04-04'),
('TOY-272', 'Slime Making Kit', 6, 16.99, 8.00, NULL, 'https://images.example.com/slime-272.jpg', 234, 1.5, '10x8x4', '2024-03-07'),
('FOOD-273', 'Popcorn Kernels 2lb', 7, 6.99, 3.00, NULL, 'https://images.example.com/popcorn-273.jpg', 456, 2.0, '10x6x3', '2024-02-11'),
('BEAUTY-274', 'Cotton Swabs 500-Pack', 8, 5.99, 3.00, NULL, 'https://images.example.com/swabs-274.jpg', 498, 0.5, '6x4x3', '2024-01-15'),
('ELEC-275', 'USB Hub 4-Port', 1, 16.99, 8.00, NULL, 'https://images.example.com/hub-275.jpg', 267, 0.3, '4x2x1', '2024-04-06'),
('CLTH-276', 'Beach Towel Large', 2, 24.99, 12.00, NULL, 'https://images.example.com/beachtowel-276.jpg', 198, 1.5, '70x40x2', '2024-03-13'),
('HOME-277', 'Can Opener Manual', 3, 7.99, 4.00, NULL, 'https://images.example.com/opener-277.jpg', 345, 0.4, '8x3x2', '2024-02-17'),
('BOOK-278', 'Crossword Puzzle Book', 4, 8.99, 4.00, NULL, 'https://images.example.com/crossword-278.jpg', 289, 0.7, '9x7x1', '2024-01-21'),
('SPRT-279', 'Hand Grip Strengthener', 5, 11.99, 6.00, NULL, 'https://images.example.com/grip-279.jpg', 234, 0.5, '6x4x2', '2024-04-08'),
('TOY-280', 'Finger Puppets Set', 6, 9.99, 5.00, NULL, 'https://images.example.com/puppets-280.jpg', 278, 0.3, '8x6x3', '2024-03-19'),
('FOOD-281', 'Hot Sauce Variety Pack', 7, 18.99, 9.00, NULL, 'https://images.example.com/hotsauce-281.jpg', 234, 2.5, '12x8x6', '2024-02-23'),
('BEAUTY-282', 'Lip Gloss Set 6-Pack', 8, 14.99, 7.00, NULL, 'https://images.example.com/lipgloss-282.jpg', 267, 0.6, '8x6x3', '2024-01-27'),
('ELEC-283', 'Battery Charger AA/AAA', 1, 19.99, 10.00, NULL, 'https://images.example.com/batterycharger-283.jpg', 189, 0.8, '6x4x3', '2024-04-10'),
('CLTH-284', 'Neck Gaiter', 2, 12.99, 6.00, NULL, 'https://images.example.com/gaiter-284.jpg', 256, 0.2, '10x4x1', '2024-03-25'),
('HOME-285', 'Ice Cube Trays 3-Pack', 3, 9.99, 5.00, NULL, 'https://images.example.com/icetray-285.jpg', 378, 0.8, '12x6x2', '2024-03-01'),
('BOOK-286', 'Sudoku Puzzle Book', 4, 7.99, 4.00, NULL, 'https://images.example.com/sudoku-286.jpg', 298, 0.6, '9x7x0.8', '2024-02-05'),
('SPRT-287', 'Ab Roller Wheel', 5, 14.99, 7.00, NULL, 'https://images.example.com/abroller-287.jpg', 167, 1.5, '10x8x8', '2024-01-09'),
('TOY-288', 'Sidewalk Chalk 24-Pack', 6, 6.99, 3.00, NULL, 'https://images.example.com/chalk-288.jpg', 445, 1.0, '8x6x4', '2024-04-12'),
('FOOD-289', 'Dried Fruit Mix 2lb', 7, 16.99, 8.00, NULL, 'https://images.example.com/driedfruit-289.jpg', 234, 2.0, '10x8x4', '2024-03-31'),
('BEAUTY-290', 'Eyelash Curler', 8, 8.99, 4.00, NULL, 'https://images.example.com/curler-290.jpg', 298, 0.2, '5x2x1', '2024-02-09'),
('ELEC-291', 'Phone Ring Holder', 1, 5.99, 3.00, NULL, 'https://images.example.com/ringholder-291.jpg', 567, 0.1, '3x3x0.5', '2024-01-13'),
('CLTH-292', 'Sleep Mask', 2, 9.99, 5.00, NULL, 'https://images.example.com/sleepmask-292.jpg', 345, 0.1, '8x4x1', '2024-04-14'),
('HOME-293', 'Chip Clips 12-Pack', 3, 7.99, 4.00, NULL, 'https://images.example.com/clips-293.jpg', 456, 0.3, '8x6x2', '2024-04-06'),
('BOOK-294', 'Word Search Book', 4, 7.99, 4.00, NULL, 'https://images.example.com/wordsearch-294.jpg', 312, 0.6, '9x7x0.8', '2024-03-13'),
('SPRT-295', 'Sports Armband Phone', 5, 11.99, 6.00, NULL, 'https://images.example.com/armband-295.jpg', 234, 0.3, '8x6x2', '2024-02-17'),
('TOY-296', 'Sticker Book Kids', 6, 8.99, 4.00, NULL, 'https://images.example.com/stickers-296.jpg', 389, 0.5, '10x8x1', '2024-01-21'),
('FOOD-297', 'Cereal Variety 10-Pack', 7, 19.99, 10.00, NULL, 'https://images.example.com/cereal-297.jpg', 234, 3.5, '14x12x10', '2024-04-16'),
('BEAUTY-298', 'Shower Cap', 8, 6.99, 3.00, NULL, 'https://images.example.com/showercap-298.jpg', 445, 0.2, '6x6x2', '2024-03-19'),
('ELEC-299', 'Cable Ties 100-Pack', 1, 8.99, 4.00, NULL, 'https://images.example.com/cableties-299.jpg', 389, 0.5, '10x6x2', '2024-02-23'),
('CLTH-300', 'Arm Sleeves UV Protection', 2, 13.99, 7.00, NULL, 'https://images.example.com/armsleeves-300.jpg', 267, 0.2, '12x6x1', '2024-01-27'),
('HOME-301', 'Soap Dispenser', 3, 12.99, 6.00, NULL, 'https://images.example.com/soapdispenser-301.jpg', 298, 0.8, '6x4x8', '2024-04-18'),
('BOOK-302', 'Comic Book Collection', 4, 19.99, 10.00, NULL, 'https://images.example.com/comic-302.jpg', 178, 1.2, '11x8x2', '2024-03-25'),
('SPRT-303', 'Sweat Headband 3-Pack', 5, 9.99, 5.00, NULL, 'https://images.example.com/headband-303.jpg', 345, 0.2, '10x8x2', '2024-03-01'),
('TOY-304', 'Bouncy Balls 12-Pack', 6, 7.99, 4.00, NULL, 'https://images.example.com/balls-304.jpg', 456, 0.6, '8x6x4', '2024-02-05'),
('FOOD-305', 'Tea Sampler 20 Flavors', 7, 17.99, 9.00, NULL, 'https://images.example.com/teasampler-305.jpg', 234, 1.0, '12x8x4', '2024-01-09'),
('BEAUTY-306', 'Face Roller Jade', 8, 16.99, 8.00, NULL, 'https://images.example.com/roller-306.jpg', 189, 0.4, '6x4x2', '2024-04-20'),
('ELEC-307', 'LED Strip Lights 16ft', 1, 24.99, 12.00, NULL, 'https://images.example.com/ledstrip-307.jpg', 234, 1.2, '12x8x3', '2024-03-31'),
('CLTH-308', 'Headband Sports', 2, 8.99, 4.00, NULL, 'https://images.example.com/headband-308.jpg', 378, 0.1, '8x2x1', '2024-03-07'),
('HOME-309', 'Wine Bottle Opener', 3, 11.99, 6.00, NULL, 'https://images.example.com/opener-309.jpg', 234, 0.5, '7x4x2', '2024-02-11'),
('BOOK-310', 'Journal Lined 200 Pages', 4, 11.99, 6.00, NULL, 'https://images.example.com/journal-310.jpg', 289, 1.0, '9x6x1', '2024-01-15'),
('SPRT-311', 'Golf Glove', 5, 14.99, 7.00, NULL, 'https://images.example.com/golfglove-311.jpg', 178, 0.2, '10x6x1', '2024-04-22'),
('TOY-312', 'Yo-Yo Professional', 6, 12.99, 6.00, NULL, 'https://images.example.com/yoyo-312.jpg', 234, 0.3, '4x4x2', '2024-04-06'),
('FOOD-313', 'Jerky Variety Pack', 7, 22.99, 11.00, NULL, 'https://images.example.com/jerky-313.jpg', 189, 1.5, '10x8x4', '2024-03-13'),
('BEAUTY-314', 'Pore Strips 14-Pack', 8, 9.99, 5.00, NULL, 'https://images.example.com/porestrips-314.jpg', 345, 0.3, '6x4x2', '2024-02-17'),
('ELEC-315', 'Phone Case Protective', 1, 12.99, 6.00, NULL, 'https://images.example.com/case-315.jpg', 456, 0.3, '7x4x1', '2024-01-21'),
('CLTH-316', 'Waist Bag Fanny Pack', 2, 18.99, 9.00, NULL, 'https://images.example.com/fanny-316.jpg', 267, 0.4, '12x6x3', '2024-04-24'),
('HOME-317', 'Cooling Rack Wire', 3, 13.99, 7.00, NULL, 'https://images.example.com/coolingrack-317.jpg', 234, 1.5, '16x10x1', '2024-03-19'),
('BOOK-318', 'Calendar 2025 Wall', 4, 12.99, 6.00, NULL, 'https://images.example.com/calendar-318.jpg', 298, 1.2, '12x12x0.5', '2024-02-23'),
('SPRT-319', 'Ankle Weights 2lb Pair', 5, 19.99, 10.00, NULL, 'https://images.example.com/ankleweights-319.jpg', 156, 2.0, '12x6x3', '2024-01-27'),
('TOY-320', 'Modeling Clay 24 Colors', 6, 16.99, 8.00, NULL, 'https://images.example.com/clay-320.jpg', 234, 2.0, '12x10x4', '2024-04-26'),
('FOOD-321', 'Salad Dressing Set 6-Pack', 7, 24.99, 12.00, NULL, 'https://images.example.com/dressing-321.jpg', 189, 4.0, '12x8x8', '2024-03-25'),
('BEAUTY-322', 'Facial Roller Quartz', 8, 22.99, 11.00, NULL, 'https://images.example.com/quartz-322.jpg', 167, 0.5, '6x4x2', '2024-03-01'),
('ELEC-323', 'Computer Mouse Pad', 1, 9.99, 5.00, NULL, 'https://images.example.com/mousepad-323.jpg', 389, 0.4, '12x10x0.2', '2024-02-05'),
('CLTH-324', 'Toe Socks 3-Pack', 2, 14.99, 7.00, NULL, 'https://images.example.com/toesocks-324.jpg', 234, 0.4, '10x8x3', '2024-01-09'),
('HOME-325', 'Pot Holders Set 4-Pack', 3, 11.99, 6.00, NULL, 'https://images.example.com/potholders-325.jpg', 298, 0.6, '10x8x2', '2024-04-28'),
('BOOK-326', 'Planner Daily 2025', 4, 16.99, 8.00, NULL, 'https://images.example.com/planner-326.jpg', 234, 1.3, '9x7x1.5', '2024-03-31'),
('SPRT-327', 'Whistle Sports Coach', 5, 6.99, 3.00, NULL, 'https://images.example.com/whistle-327.jpg', 456, 0.1, '4x2x1', '2024-03-07'),
('TOY-328', 'Spin Top Metal', 6, 8.99, 4.00, NULL, 'https://images.example.com/top-328.jpg', 345, 0.3, '4x4x2', '2024-02-11'),
('FOOD-329', 'Baking Mix Variety', 7, 18.99, 9.00, NULL, 'https://images.example.com/baking-329.jpg', 234, 3.0, '12x10x6', '2024-01-15'),
('BEAUTY-330', 'Cuticle Oil Pen', 8, 7.99, 4.00, NULL, 'https://images.example.com/cuticle-330.jpg', 378, 0.1, '5x1x1', '2024-04-30'),
('ELEC-331', 'Earphone Case', 1, 8.99, 4.00, NULL, 'https://images.example.com/earcase-331.jpg', 445, 0.1, '4x3x2', '2024-04-06'),
('CLTH-332', 'Hair Ties 100-Pack', 2, 9.99, 5.00, NULL, 'https://images.example.com/hairties-332.jpg', 456, 0.3, '8x6x2', '2024-03-13'),
('HOME-333', 'Coasters Set 8-Pack', 3, 12.99, 6.00, NULL, 'https://images.example.com/coasters-333.jpg', 298, 0.8, '8x6x2', '2024-02-17'),
('BOOK-334', 'Notebook Set 5-Pack', 4, 14.99, 7.00, NULL, 'https://images.example.com/notebook-334.jpg', 345, 1.5, '10x8x2', '2024-01-21'),
('SPRT-335', 'Shin Guards Soccer', 5, 16.99, 8.00, NULL, 'https://images.example.com/shinguards-335.jpg', 167, 0.8, '12x6x4', '2024-05-02'),
('TOY-336', 'Marbles Glass 50-Pack', 6, 11.99, 6.00, NULL, 'https://images.example.com/marbles-336.jpg', 289, 1.2, '8x6x4', '2024-03-19'),
('FOOD-337', 'Instant Noodles 12-Pack', 7, 14.99, 7.00, NULL, 'https://images.example.com/noodles-337.jpg', 389, 2.5, '12x10x8', '2024-02-23'),
('BEAUTY-338', 'Loofah 4-Pack', 8, 8.99, 4.00, NULL, 'https://images.example.com/loofah-338.jpg', 456, 0.4, '8x6x4', '2024-01-27'),
('ELEC-339', 'Screen Cleaner Kit', 1, 11.99, 6.00, NULL, 'https://images.example.com/cleaner-339.jpg', 298, 0.5, '6x4x2', '2024-05-04'),
('CLTH-340', 'Bobby Pins 300-Pack', 2, 6.99, 3.00, NULL, 'https://images.example.com/bobby-340.jpg', 567, 0.3, '6x4x2', '2024-03-25'),
('HOME-341', 'Silicone Baking Mat', 3, 14.99, 7.00, NULL, 'https://images.example.com/bakingmat-341.jpg', 267, 0.6, '17x12x0.2', '2024-03-01'),
('BOOK-342', 'Sketchbook 100 Pages', 4, 10.99, 5.00, NULL, 'https://images.example.com/sketch-342.jpg', 234, 1.0, '11x9x0.8', '2024-02-05'),
('SPRT-343', 'Mouthguard Sports', 5, 12.99, 6.00, NULL, 'https://images.example.com/mouthguard-343.jpg', 234, 0.2, '6x4x2', '2024-01-09'),
('TOY-344', 'Kinetic Sand 2lb', 6, 14.99, 7.00, NULL, 'https://images.example.com/sand-344.jpg', 234, 2.0, '10x8x4', '2024-05-06'),
('FOOD-345', 'Cookie Mix 6-Pack', 7, 16.99, 8.00, NULL, 'https://images.example.com/cookie-345.jpg', 278, 3.0, '12x10x8', '2024-03-31'),
('BEAUTY-346', 'Blotting Papers 100 Sheets', 8, 6.99, 3.00, NULL, 'https://images.example.com/blotting-346.jpg', 389, 0.2, '4x3x1', '2024-03-07'),
('ELEC-347', 'Laptop Sleeve 15 inch', 1, 19.99, 10.00, NULL, 'https://images.example.com/sleeve-347.jpg', 234, 0.8, '16x11x1', '2024-02-11'),
('CLTH-348', 'Shoe Inserts Comfort', 2, 14.99, 7.00, NULL, 'https://images.example.com/inserts-348.jpg', 267, 0.5, '12x5x2', '2024-01-15'),
('HOME-349', 'Plunger Toilet', 3, 12.99, 6.00, NULL, 'https://images.example.com/plunger-349.jpg', 198, 2.0, '24x6x6', '2024-05-08'),
('BOOK-350', 'Map World Wall', 4, 18.99, 9.00, NULL, 'https://images.example.com/map-350.jpg', 156, 0.5, '36x24x0.1', '2024-04-06'),
('SPRT-351', 'Headband Sweatband', 5, 7.99, 4.00, NULL, 'https://images.example.com/sweatband-351.jpg', 445, 0.1, '8x2x1', '2024-03-13'),
('TOY-352', 'Dominoes Game Set', 6, 13.99, 7.00, NULL, 'https://images.example.com/dominoes-352.jpg', 198, 1.5, '10x8x3', '2024-02-17'),
('FOOD-353', 'Maple Syrup Organic', 7, 14.99, 7.00, NULL, 'https://images.example.com/syrup-353.jpg', 234, 2.0, '8x4x10', '2024-01-21'),
('BEAUTY-354', 'Pumice Stone', 8, 5.99, 3.00, NULL, 'https://images.example.com/pumice-354.jpg', 456, 0.3, '4x3x2', '2024-05-10'),
('ELEC-355', 'Cable Organizer Box', 1, 16.99, 8.00, NULL, 'https://images.example.com/cablebox-355.jpg', 234, 1.2, '10x6x4', '2024-03-19'),
('CLTH-356', 'Luggage Tag 4-Pack', 2, 9.99, 5.00, NULL, 'https://images.example.com/luggagetag-356.jpg', 367, 0.3, '8x6x1', '2024-02-23'),
('HOME-357', 'Garlic Press', 3, 11.99, 6.00, NULL, 'https://images.example.com/garlic-357.jpg', 234, 0.6, '7x4x3', '2024-01-27'),
('BOOK-358', 'Sticker Book Adult', 4, 12.99, 6.00, NULL, 'https://images.example.com/adultsticker-358.jpg', 234, 0.8, '10x8x1', '2024-05-12'),
('SPRT-359', 'Ping Pong Balls 24-Pack', 5, 9.99, 5.00, NULL, 'https://images.example.com/pingpong-359.jpg', 345, 0.3, '8x6x4', '2024-03-25'),
('TOY-360', 'Play Dough 20-Pack', 6, 19.99, 10.00, NULL, 'https://images.example.com/playdough-360.jpg', 267, 3.0, '14x12x6', '2024-03-01'),
('FOOD-361', 'Seasoning Mix Variety', 7, 13.99, 7.00, NULL, 'https://images.example.com/seasoning-361.jpg', 298, 1.5, '10x8x4', '2024-02-05'),
('BEAUTY-362', 'Shaving Cream 10oz', 8, 7.99, 4.00, NULL, 'https://images.example.com/shavingcream-362.jpg', 389, 0.8, '6x3x8', '2024-01-09'),
('ELEC-363', 'Phone Tripod', 1, 22.99, 11.00, NULL, 'https://images.example.com/tripod-363.jpg', 189, 1.2, '12x4x4', '2024-05-14'),
('CLTH-364', 'Travel Pillow Neck', 2, 19.99, 10.00, NULL, 'https://images.example.com/neckpillow-364.jpg', 234, 0.8, '12x10x6', '2024-03-31'),
('HOME-365', 'Vegetable Peeler Set', 3, 9.99, 5.00, NULL, 'https://images.example.com/peeler-365.jpg', 345, 0.4, '8x6x2', '2024-03-07'),
('BOOK-366', 'Bookmark Set 20-Pack', 4, 8.99, 4.00, NULL, 'https://images.example.com/bookmark-366.jpg', 456, 0.2, '8x2x0.5', '2024-02-11'),
('SPRT-367', 'Sports Whistle Lanyard', 5, 8.99, 4.00, NULL, 'https://images.example.com/lanyard-367.jpg', 398, 0.2, '18x1x0.5', '2024-01-15'),
('TOY-368', 'Nerf Foam Darts 100-Pack', 6, 14.99, 7.00, NULL, 'https://images.example.com/darts-368.jpg', 298, 0.8, '10x8x4', '2024-05-16'),
('FOOD-369', 'Energy Bars 15-Pack', 7, 21.99, 11.00, NULL, 'https://images.example.com/energybar-369.jpg', 234, 1.8, '10x8x6', '2024-04-06'),
('BEAUTY-370', 'Makeup Sponges 10-Pack', 8, 11.99, 6.00, NULL, 'https://images.example.com/sponges-370.jpg', 345, 0.3, '6x4x3', '2024-03-13');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 8: Products with MISSING CATEGORIES (40 products = 10%)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('MISC-371', 'Universal Remote Control', NULL, 29.99, 15.00, 'Programmable remote for multiple devices', 'https://images.example.com/remote-371.jpg', 156, 0.5, '8x4x2', '2024-02-17'),
('MISC-372', 'Garden Hose 50ft', NULL, 34.99, 17.00, 'Durable garden hose with spray nozzle', 'https://images.example.com/hose-372.jpg', 123, 8.0, '14x14x6', '2024-01-21'),
('MISC-373', 'Flashlight Rechargeable', NULL, 24.99, 12.00, 'LED flashlight with USB charging', 'https://images.example.com/flashlight-373.jpg', 234, 0.6, '6x2x2', '2024-05-18'),
('MISC-374', 'First Aid Kit 100-Piece', NULL, 22.99, 11.00, 'Complete first aid supplies for home', 'https://images.example.com/firstaid-374.jpg', 189, 2.0, '10x8x4', '2024-03-19'),
('MISC-375', 'Measuring Tape 25ft', NULL, 11.99, 6.00, 'Retractable measuring tape with lock', 'https://images.example.com/tape-375.jpg', 298, 0.4, '4x4x2', '2024-02-23'),
('MISC-376', 'Scissors Set 3-Pack', NULL, 14.99, 7.00, 'Multipurpose scissors for office and crafts', 'https://images.example.com/scissors-376.jpg', 267, 0.6, '10x6x2', '2024-01-27'),
('MISC-377', 'Duct Tape Heavy Duty', NULL, 8.99, 4.00, 'Strong adhesive tape for repairs', 'https://images.example.com/ducttape-377.jpg', 445, 0.8, '6x6x2', '2024-05-20'),
('MISC-378', 'Safety Gloves Work', NULL, 12.99, 6.00, 'Cut-resistant work gloves large', 'https://images.example.com/workgloves-378.jpg', 234, 0.5, '10x6x2', '2024-03-25'),
('MISC-379', 'Tool Box Portable', NULL, 39.99, 20.00, 'Plastic toolbox with organizers', 'https://images.example.com/toolbox-379.jpg', 98, 4.0, '20x10x10', '2024-03-01'),
('MISC-380', 'Ladder Step Stool', NULL, 44.99, 22.00, 'Two-step ladder with non-slip surface', 'https://images.example.com/stool-380.jpg', 87, 10.0, '18x12x36', '2024-02-05'),
('MISC-381', 'Level Tool 12 inch', NULL, 13.99, 7.00, 'Magnetic level for precise measurements', 'https://images.example.com/level-381.jpg', 178, 0.8, '12x3x2', '2024-01-09'),
('MISC-382', 'Utility Knife', NULL, 9.99, 5.00, 'Retractable utility knife with extra blades', 'https://images.example.com/knife-382.jpg', 298, 0.3, '6x2x1', '2024-05-22'),
('MISC-383', 'Bungee Cords 10-Pack', NULL, 16.99, 8.00, 'Assorted size bungee cords with hooks', 'https://images.example.com/bungee-383.jpg', 234, 1.5, '12x10x4', '2024-03-31'),
('MISC-384', 'Screwdriver Set 10-Piece', NULL, 19.99, 10.00, 'Phillips and flat head screwdriver set', 'https://images.example.com/screwdrivers-384.jpg', 189, 1.5, '12x8x3', '2024-03-07'),
('MISC-385', 'Work Light LED', NULL, 29.99, 15.00, 'Portable LED work light with stand', 'https://images.example.com/worklight-385.jpg', 145, 2.0, '10x8x12', '2024-02-11'),
('MISC-386', 'Wire Cutters', NULL, 14.99, 7.00, 'Professional wire cutting pliers', 'https://images.example.com/wirecutters-386.jpg', 198, 0.6, '8x4x2', '2024-01-15'),
('MISC-387', 'Paint Roller Set', NULL, 18.99, 9.00, 'Paint roller with tray and extra covers', 'https://images.example.com/roller-387.jpg', 156, 2.0, '14x10x4', '2024-05-24'),
('MISC-388', 'Painter Tape 6-Pack', NULL, 16.99, 8.00, 'Blue painter tape multiple widths', 'https://images.example.com/paintertape-388.jpg', 267, 1.2, '12x8x6', '2024-04-06'),
('MISC-389', 'Drop Cloth Canvas', NULL, 22.99, 11.00, 'Heavy-duty canvas drop cloth 9x12', 'https://images.example.com/dropcloth-389.jpg', 134, 5.0, '14x10x4', '2024-03-13'),
('MISC-390', 'Adjustable Wrench Set', NULL, 24.99, 12.00, '3-piece adjustable wrench set', 'https://images.example.com/wrench-390.jpg', 178, 2.5, '12x8x3', '2024-02-17'),
('MISC-391', 'Socket Set 40-Piece', NULL, 39.99, 20.00, 'Complete socket set with ratchet', 'https://images.example.com/socket-391.jpg', 123, 6.0, '14x10x4', '2024-01-21'),
('MISC-392', 'Rubber Mallet', NULL, 16.99, 8.00, 'Soft face rubber mallet 16oz', 'https://images.example.com/mallet-392.jpg', 189, 1.2, '12x5x3', '2024-05-26'),
('MISC-393', 'Hammer Claw', NULL, 19.99, 10.00, '16oz claw hammer with fiberglass handle', 'https://images.example.com/hammer-393.jpg', 198, 1.5, '13x6x2', '2024-03-19'),
('MISC-394', 'Pliers Set 3-Piece', NULL, 21.99, 11.00, 'Needle nose, slip joint, and groove pliers', 'https://images.example.com/pliers-394.jpg', 167, 1.8, '12x8x3', '2024-02-23'),
('MISC-395', 'Hex Key Set Metric', NULL, 12.99, 6.00, 'L-shaped hex key set 15-piece', 'https://images.example.com/hexkey-395.jpg', 234, 0.6, '8x6x2', '2024-01-27'),
('MISC-396', 'Safety Glasses Clear', NULL, 9.99, 5.00, 'Impact-resistant safety eyewear', 'https://images.example.com/safetyglasses-396.jpg', 345, 0.2, '7x3x2', '2024-05-28'),
('MISC-397', 'Ear Protection Muffs', NULL, 18.99, 9.00, 'Noise reduction ear muffs 32dB', 'https://images.example.com/earmuffs-397.jpg', 167, 0.8, '8x8x4', '2024-03-25'),
('NOCAT-469', 'Multi-Tool Pocket', NULL, 24.99, 12.00, 'Compact multi-tool with 15 functions', 'https://images.example.com/multitool-469.jpg', 189, 0.5, '5x2x1', '2024-03-10'),
('NOCAT-470', 'Reusable Straws Set', NULL, 11.99, 6.00, 'Stainless steel straws with cleaning brush', 'https://images.example.com/straws-470.jpg', 345, 0.3, '10x6x2', '2024-02-15'),
('NOCAT-471', 'Travel Adapter Universal', NULL, 19.99, 10.00, 'All-in-one travel adapter for 150+ countries', 'https://images.example.com/adapter-471.jpg', 234, 0.4, '4x3x3', '2024-01-20'),
('NOCAT-472', 'Waterproof Phone Pouch', NULL, 14.99, 7.00, 'Clear waterproof case for beach and pool', 'https://images.example.com/pouch-472.jpg', 267, 0.2, '8x5x1', '2024-03-25'),
('NOCAT-473', 'Carabiner Clips 12-Pack', NULL, 12.99, 6.00, 'Heavy-duty carabiner clips assorted sizes', 'https://images.example.com/carabiner-473.jpg', 389, 0.6, '8x6x2', '2024-02-10'),
('NOCAT-474', 'Lint Roller 5-Pack', NULL, 9.99, 5.00, 'Extra-sticky lint rollers with refills', 'https://images.example.com/lint-474.jpg', 456, 0.8, '10x6x4', '2024-01-15'),
('NOCAT-475', 'Portable Fan USB', NULL, 16.99, 8.00, 'Rechargeable mini fan with 3 speeds', 'https://images.example.com/fan-475.jpg', 234, 0.4, '6x6x2', '2024-03-30'),
('NOCAT-476', 'Ice Pack Reusable 4-Pack', NULL, 13.99, 7.00, 'Gel ice packs for lunch boxes', 'https://images.example.com/icepack-476.jpg', 298, 1.2, '8x6x3', '2024-02-25'),
('NOCAT-477', 'Magnet Hooks 20-Pack', NULL, 11.99, 6.00, 'Strong magnetic hooks for metal surfaces', 'https://images.example.com/magnets-477.jpg', 367, 0.5, '6x4x2', '2024-01-30'),
('NOCAT-478', 'Pill Organizer Weekly', NULL, 8.99, 4.00, '7-day pill organizer with AM/PM', 'https://images.example.com/pills-478.jpg', 445, 0.3, '8x4x2', '2024-03-15'),
('NOCAT-479', 'Name Tags Self-Adhesive', NULL, 6.99, 3.00, 'Name tag stickers 100-count', 'https://images.example.com/nametags-479.jpg', 498, 0.2, '6x4x1', '2024-02-05'),
('NOCAT-480', 'Velcro Strips Adhesive', NULL, 14.99, 7.00, 'Heavy-duty velcro strips 15ft', 'https://images.example.com/velcro-480.jpg', 289, 0.6, '15x2x1', '2024-01-10'),
('NOCAT-481', 'Command Strips 24-Pack', NULL, 12.99, 6.00, 'Damage-free hanging strips assorted', 'https://images.example.com/command-481.jpg', 378, 0.4, '8x6x2', '2024-03-20');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 9: Products with MISSING IMAGES (60 products = 15%)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('IMG-401', 'Wireless Earphones Pro', 1, 89.99, 45.00, 'Premium wireless earphones with active noise cancellation', NULL, 145, 0.3, '4x3x2', '2024-01-05'),
('IMG-402', 'Men Polo Shirt Navy', 2, 32.99, 16.00, 'Classic navy polo shirt 100% cotton', NULL, 198, 0.4, '12x9x1', '2024-02-10'),
('IMG-403', 'Kitchen Knife Chef 8 inch', 3, 44.99, 22.00, 'Professional chef knife with ergonomic handle', NULL, 123, 0.8, '14x3x1', '2024-01-15'),
('IMG-404', 'Fantasy Novel Epic', 4, 18.99, 9.00, 'Epic fantasy novel bestselling series', NULL, 234, 1.0, '9x6x1.5', '2024-03-20'),
('IMG-405', 'Resistance Bands Heavy', 5, 24.99, 12.00, 'Heavy resistance bands for strength training', NULL, 167, 1.2, '10x8x3', '2024-02-25'),
('IMG-406', 'Building Set 1000 Pieces', 6, 49.99, 25.00, 'Large building block set with storage box', NULL, 134, 4.0, '16x12x8', '2024-01-30'),
('IMG-407', 'Organic Nuts Mix 2lb', 7, 21.99, 11.00, 'Mixed raw organic nuts bulk pack', NULL, 267, 2.0, '10x8x4', '2024-03-05'),
('IMG-408', 'Face Cream Night', 8, 29.99, 15.00, 'Anti-aging night cream with retinol', NULL, 189, 0.6, '6x4x4', '2024-02-15'),
('IMG-409', 'Tablet Stand Adjustable', 1, 19.99, 10.00, 'Universal tablet stand with 360 rotation', NULL, 234, 0.8, '8x6x4', '2024-01-10'),
('IMG-410', 'Women Jacket Denim', 2, 59.99, 30.00, 'Classic denim jacket with stretch', NULL, 156, 1.2, '14x12x4', '2024-03-25'),
('IMG-411', 'Cutting Board Bamboo', 3, 24.99, 12.00, 'Large bamboo cutting board with juice groove', NULL, 198, 2.5, '18x12x1', '2024-02-20'),
('IMG-412', 'Science Fiction Anthology', 4, 16.99, 8.00, 'Collection of classic sci-fi short stories', NULL, 178, 0.9, '9x6x1', '2024-01-25'),
('IMG-413', 'Yoga Block Set 2-Pack', 5, 16.99, 8.00, 'High-density foam yoga blocks', NULL, 234, 1.0, '9x6x4', '2024-03-15'),
('IMG-414', 'Toy Kitchen Playset', 6, 69.99, 35.00, 'Kids pretend play kitchen with accessories', NULL, 98, 8.0, '24x18x12', '2024-02-05'),
('IMG-415', 'Protein Shake Mix Chocolate', 7, 34.99, 17.00, 'Whey protein shake mix 30 servings', NULL, 189, 2.8, '8x6x9', '2024-01-20'),
('IMG-416', 'Hair Straightener Ceramic', 8, 39.99, 20.00, 'Professional ceramic hair straightener', NULL, 167, 1.0, '12x4x3', '2024-03-10'),
('IMG-417', 'Smart Plug WiFi 4-Pack', 1, 29.99, 15.00, 'WiFi smart plugs with app control', NULL, 278, 0.6, '8x6x4', '2024-02-28'),
('IMG-418', 'Men Dress Pants', 2, 49.99, 25.00, 'Formal dress pants wrinkle-resistant', NULL, 134, 0.8, '14x12x2', '2024-01-15'),
('IMG-419', 'Baking Sheet Set 3-Pack', 3, 29.99, 15.00, 'Non-stick baking sheets multiple sizes', NULL, 189, 3.5, '18x13x2', '2024-03-20'),
('IMG-420', 'Horror Novel Collection', 4, 22.99, 11.00, 'Classic horror stories anthology', NULL, 156, 1.1, '9x6x1.5', '2024-02-10'),
('IMG-421', 'Camping Chair Folding', 5, 34.99, 17.00, 'Portable folding chair with cup holder', NULL, 145, 5.0, '22x8x8', '2024-01-05'),
('IMG-422', 'Dollhouse Wooden', 6, 89.99, 45.00, 'Large wooden dollhouse with furniture', NULL, 76, 15.0, '28x16x24', '2024-03-25'),
('IMG-423', 'Vitamin C Tablets 100-Count', 7, 12.99, 6.00, 'Vitamin C 1000mg immune support', NULL, 345, 0.5, '6x4x4', '2024-02-15'),
('IMG-424', 'Makeup Palette Eyeshadow', 8, 26.99, 13.00, '24-color eyeshadow palette with mirror', NULL, 198, 0.6, '8x6x1', '2024-01-30'),
('IMG-425', 'Bluetooth Headphones Over-Ear', 1, 69.99, 35.00, 'Over-ear headphones with deep bass', NULL, 167, 0.9, '8x7x4', '2024-03-05'),
('IMG-426', 'Women Cardigan Knit', 2, 44.99, 22.00, 'Soft knit cardigan with pockets', NULL, 189, 0.7, '14x11x3', '2024-02-20'),
('IMG-427', 'Mixing Bowl Set Stainless', 3, 34.99, 17.00, 'Stainless steel mixing bowls 5-piece', NULL, 156, 3.0, '14x12x8', '2024-01-25'),
('IMG-428', 'Mystery Box Set 3 Books', 4, 29.99, 15.00, 'Mystery thriller book bundle', NULL, 178, 2.5, '9x6x4', '2024-03-15'),
('IMG-429', 'Bike Helmet Kids', 5, 29.99, 15.00, 'Kids safety helmet with adjustable straps', NULL, 198, 0.9, '10x9x8', '2024-02-05'),
('IMG-430', 'Board Game Strategy', 6, 39.99, 20.00, 'Award-winning strategy board game', NULL, 134, 2.5, '12x10x3', '2024-01-20'),
('IMG-431', 'Quinoa Organic 2lb', 7, 14.99, 7.00, 'Organic tri-color quinoa bulk bag', NULL, 289, 2.0, '10x8x3', '2024-03-10'),
('IMG-432', 'Bath Salt Lavender', 8, 16.99, 8.00, 'Relaxing lavender bath salts 32oz', NULL, 234, 2.5, '8x6x8', '2024-02-28'),
('IMG-433', 'Camera Tripod Portable', 1, 39.99, 20.00, 'Lightweight tripod for cameras and phones', NULL, 178, 2.0, '18x4x4', '2024-01-15'),
('IMG-434', 'Men Athletic Pants', 2, 39.99, 20.00, 'Moisture-wicking athletic pants with zipper pockets', NULL, 167, 0.6, '13x11x2', '2024-03-20'),
('IMG-435', 'Spice Grinder Electric', 3, 29.99, 15.00, 'Electric spice and coffee grinder', NULL, 189, 1.5, '6x5x8', '2024-02-10'),
('IMG-436', 'Romance Novel Series Set', 4, 34.99, 17.00, 'Complete romance series 4 books', NULL, 145, 3.0, '9x6x5', '2024-01-05'),
('IMG-437', 'Yoga Strap 8ft', 5, 11.99, 6.00, 'Durable yoga strap with D-ring buckle', NULL, 267, 0.3, '96x2x0.2', '2024-03-25'),
('IMG-438', 'Toy Car Race Track', 6, 54.99, 27.00, 'Electric race track with 2 cars', NULL, 112, 4.5, '20x16x6', '2024-02-15'),
('IMG-439', 'Dark Chocolate Bar 12-Pack', 7, 19.99, 10.00, 'Premium dark chocolate 85% cacao', NULL, 234, 1.5, '10x8x4', '2024-01-30'),
('IMG-440', 'Facial Cleanser Foam', 8, 18.99, 9.00, 'Gentle foaming cleanser for sensitive skin', NULL, 198, 0.7, '6x3x8', '2024-03-05'),
('IMG-441', 'Wireless Mouse Ergonomic', 1, 29.99, 15.00, 'Ergonomic vertical wireless mouse', NULL, 189, 0.4, '5x4x3', '2024-02-20'),
('IMG-442', 'Women Tank Top Pack 3', 2, 24.99, 12.00, 'Cotton tank tops 3-pack assorted colors', NULL, 256, 0.5, '12x10x3', '2024-01-25'),
('IMG-443', 'Colander Stainless Steel', 3, 19.99, 10.00, 'Large colander with stable base', NULL, 178, 1.5, '12x12x6', '2024-03-15'),
('IMG-444', 'Graphic Novel Collection', 4, 27.99, 14.00, 'Award-winning graphic novel series', NULL, 167, 1.8, '11x8x2', '2024-02-05'),
('IMG-445', 'Hand Weights Set 5lb', 5, 29.99, 15.00, 'Neoprene coated hand weights pair', NULL, 145, 10.0, '8x4x4', '2024-01-20'),
('IMG-446', 'Toy Dinosaur Set', 6, 24.99, 12.00, 'Realistic dinosaur figures 12-piece', NULL, 198, 2.0, '14x10x6', '2024-03-10'),
('IMG-447', 'Almond Butter Organic', 7, 16.99, 8.00, 'Smooth almond butter no added sugar', NULL, 234, 1.8, '6x4x6', '2024-02-28'),
('IMG-448', 'Body Butter Shea', 8, 19.99, 10.00, 'Whipped shea body butter 8oz', NULL, 189, 0.9, '6x4x4', '2024-01-15'),
('IMG-449', 'USB-C Hub Multi-Port', 1, 34.99, 17.00, '7-in-1 USB-C hub with HDMI', NULL, 167, 0.4, '5x3x1', '2024-03-20'),
('IMG-450', 'Men Underwear 6-Pack', 2, 29.99, 15.00, 'Cotton blend boxer briefs assorted', NULL, 234, 0.6, '12x10x4', '2024-02-10'),
('IMG-451', 'Measuring Spoon Set', 3, 12.99, 6.00, 'Stainless steel measuring spoons 6-piece', NULL, 298, 0.4, '8x4x2', '2024-01-05'),
('IMG-452', 'Self-Help Book Success', 4, 17.99, 9.00, 'Achieve your goals practical guide', NULL, 189, 0.9, '9x6x1', '2024-03-25'),
('IMG-453', 'Skipping Rope Weighted', 5, 14.99, 7.00, 'Weighted jump rope for cardio', NULL, 234, 0.6, '10x6x2', '2024-02-15'),
('IMG-454', 'Craft Kit Kids Art', 6, 32.99, 16.00, 'Complete craft supplies for children', NULL, 178, 3.0, '14x12x6', '2024-01-30'),
('IMG-455', 'Coconut Oil Organic', 7, 13.99, 7.00, 'Virgin coconut oil for cooking 16oz', NULL, 267, 1.5, '6x4x6', '2024-03-05'),
('IMG-456', 'Hair Serum Anti-Frizz', 8, 22.99, 11.00, 'Smoothing hair serum with argan oil', NULL, 198, 0.5, '5x3x7', '2024-02-20'),
('IMG-457', 'Phone Grip Stand', 1, 9.99, 5.00, 'Collapsible phone grip and stand', NULL, 456, 0.1, '3x3x1', '2024-01-25'),
('IMG-458', 'Women Leggings High-Waist', 2, 32.99, 16.00, 'High-waisted leggings with pockets', NULL, 234, 0.4, '12x10x2', '2024-03-15'),
('IMG-459', 'Oven Mitt Set 2-Pack', 3, 16.99, 8.00, 'Heat-resistant oven mitts quilted', NULL, 267, 0.6, '12x8x4', '2024-02-05'),
('IMG-460', 'Biography Inspiring Leaders', 4, 19.99, 10.00, 'Stories of world-changing leaders', NULL, 178, 1.0, '9x6x1.5', '2024-01-20');

-- ============================================================================
-- INSERT PRODUCT DATA - BATCH 10: Products with MISSING NAMES (8 products = 2%)
-- ============================================================================

INSERT INTO products (sku, name, category_id, price, cost, description, image_url, stock_quantity, weight, dimensions, created_date) VALUES
('NAME-461', NULL, 1, 79.99, 40.00, 'High-quality electronic device with multiple features', 'https://images.example.com/device-461.jpg', 123, 1.5, '10x8x4', '2024-01-10'),
('NAME-462', NULL, 2, 44.99, 22.00, 'Comfortable and stylish clothing item', 'https://images.example.com/clothing-462.jpg', 198, 0.6, '12x10x2', '2024-02-15'),
('NAME-463', NULL, 3, 34.99, 17.00, 'Essential kitchen item for everyday use', 'https://images.example.com/kitchen-463.jpg', 167, 2.0, '12x8x6', '2024-03-20'),
('NAME-464', NULL, 4, 24.99, 12.00, 'Entertaining and informative reading material', 'https://images.example.com/book-464.jpg', 234, 1.2, '9x6x1', '2024-01-25'),
('NAME-465', NULL, 5, 39.99, 20.00, 'Sports equipment for active lifestyle', 'https://images.example.com/sports-465.jpg', 145, 3.0, '14x10x8', '2024-02-28'),
('NAME-466', NULL, 6, 29.99, 15.00, 'Fun and educational toy for children', 'https://images.example.com/toy-466.jpg', 189, 2.5, '12x10x6', '2024-03-05'),
('NAME-467', NULL, 7, 16.99, 8.00, 'Nutritious food product for healthy eating', 'https://images.example.com/food-467.jpg', 278, 1.5, '10x8x4', '2024-01-15'),
('NAME-468', NULL, 8, 26.99, 13.00, 'Beauty product for self-care routine', 'https://images.example.com/beauty-468.jpg', 234, 0.8, '6x4x6', '2024-02-20');

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Count total products
SELECT COUNT(*) as total_products FROM products;

-- Count products with data quality issues
SELECT 
  COUNT(CASE WHEN price = 0 OR price IS NULL THEN 1 END) as invalid_prices,
  COUNT(CASE WHEN description IS NULL THEN 1 END) as missing_descriptions,
 COUNT(CASE WHEN category_id IS NULL THEN 1 END) as missing_categories,
 COUNT(CASE WHEN image_url IS NULL THEN 1 END) as missing_images,
COUNT(CASE WHEN name IS NULL THEN 1 END) as missing_names
FROM products;

-- Find duplicate SKUs
SELECT sku, COUNT(*) as count 
FROM products 
GROUP BY sku 
HAVING count > 1;

-- ============================================================================
-- END OF SAMPLE DATA INSERTION SCRIPT
-- ============================================================================