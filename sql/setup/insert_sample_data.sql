-- Active: 1759082837172@@127.0.0.1@3307@ecommerce_analytics
USE ecommerce_analytics;

-- Insert sample customers (with intentional quality issues)
INSERT INTO customers (first_name, last_name, email, phone, address, city, state, zip_code, created_date) VALUES
('John', 'Smith', 'john.smith@email.com', '555-0001', '123 Main St', 'Anytown', 'CA', '12345', '2024-01-15'),
('Jane', 'Doe', 'jane.doe@gmail.com', '555-0002', '456 Oak Ave', 'Springfield', 'NY', '67890', '2024-01-20'),
('John', 'Smith', 'johnsmith@email.com', '555-0001', '123 Main Street', 'Anytown', 'CA', '12345', '2024-02-01'),
('Bob', 'Johnson', 'invalid-email', '555-0003', '789 Pine St', 'Madison', 'WI', '54321', '2024-01-25'),
('Alice', 'Brown', 'alice@domain.com', '', '321 Elm St', 'Portland', 'OR', '98765', '2024-02-05'),
('Mike', 'Davis', 'mike.davis@yahoo.com', '555-0004', NULL, NULL, 'TX', NULL, '2024-01-30'),
('Jane', 'Doe', 'jane.doe@gmail.com', '555-0002', '456 Oak Avenue', 'Springfield', 'NY', '67890', '2024-02-10'),
('Carol', 'Wilson', 'carol@', '555-0005', '654 Maple Dr', 'Denver', 'CO', '80202', '2024-02-15'),
('Tom', 'Miller', 'tom.miller@hotmail.com', '555-0006', '987 Cedar Ln', 'Seattle', 'WA', '98101', '2024-02-20'),
('Sarah', '', 'sarah@example.com', '555-0007', '147 Birch St', 'Miami', 'FL', '33101', '2024-02-25');

-- Insert sample products (with quality issues)
INSERT INTO products (product_name, description, category, price, stock_quantity, created_date) VALUES
('Wireless Headphones', 'High-quality wireless headphones with noise cancellation', 'Electronics', 199.99, 50, '2024-01-01'),
('Running Shoes', '', 'Footwear', 89.99, 75, '2024-01-05'),
('Coffee Maker', 'Automatic drip coffee maker with programmable timer', 'Kitchen', 129.99, 30, '2024-01-10'),
('Laptop Stand', 'Adjustable aluminum laptop stand', 'Electronics', -15.99, 25, '2024-01-15'),
('Water Bottle', 'Stainless steel insulated water bottle', 'Sports', 24.99, 0, '2024-01-20'),
('', 'Comfortable ergonomic office chair', 'Furniture', 299.99, 15, '2024-01-25'),
('Bluetooth Speaker', 'Portable wireless speaker with bass boost', NULL, 79.99, 40, '2024-01-30'),
('Desk Lamp', 'LED desk lamp with adjustable brightness', 'Office', 0.00, 60, '2024-02-01'),
('Phone Case', 'Protective case for smartphones', 'Electronics', 19.99, -5, '2024-02-05'),
('Yoga Mat', 'Non-slip exercise yoga mat', 'Sports', 39.99, 100, '2024-02-10');


-- Insert sample orders (valid customer IDs only)
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(1, '2024-02-01', 199.99, 'completed'),
(2, '2024-02-02', 89.99, 'shipped'),
(3, '2024-02-04', 24.99, 'completed'),
(4, '2024-02-05', -15.99, 'completed'),
(5, NULL, 79.99, 'processing'),
(6, '2024-02-07', 0.00, 'completed'),
(1, '2024-02-08', 39.99, 'invalid_status'),
(2, '2024-02-09', 299.99, 'cancelled'),
(7, '2024-02-10', 19.99, 'completed');

SET FOREIGN_KEY_CHECKS = 0;
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES (999, '2024-02-03', 129.99, 'pending');
SET FOREIGN_KEY_CHECKS = 1;