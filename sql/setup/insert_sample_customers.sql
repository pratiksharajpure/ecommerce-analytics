-- ========================================
-- INSERT 500 SAMPLE CUSTOMERS
-- Realistic customer data for ecommerce_analytics database
-- ========================================

USE ecommerce_analytics;

-- First 30 customers with realistic, hand-crafted data
INSERT INTO customers (first_name, last_name, email, phone, address_line1, city, state, zip_code, status, created_at) VALUES
('James', 'Anderson', 'james.anderson@email.com', '555-0101', '123 Main St', 'New York', 'NY', '10001', 'active', '2024-01-15'),
('Mary', 'Johnson', 'mary.johnson@gmail.com', '555-0102', '456 Oak Ave', 'Los Angeles', 'CA', '90001', 'active', '2024-01-16'),
('Robert', 'Williams', 'robert.williams@yahoo.com', '555-0103', '789 Pine St', 'Chicago', 'IL', '60601', 'active', '2024-01-17'),
('Jennifer', 'Brown', 'jennifer.brown@hotmail.com', '555-0104', '321 Elm St', 'Houston', 'TX', '77001', 'active', '2024-01-18'),
('Michael', 'Davis', 'michael.davis@email.com', '555-0105', '654 Maple Dr', 'Phoenix', 'AZ', '85001', 'active', '2024-01-19'),
('Linda', 'Miller', 'linda.miller@gmail.com', '555-0106', '987 Cedar Ln', 'Philadelphia', 'PA', '19101', 'active', '2024-01-20'),
('William', 'Wilson', 'william.wilson@yahoo.com', '555-0107', '147 Birch St', 'San Antonio', 'TX', '78201', 'active', '2024-01-21'),
('Elizabeth', 'Moore', 'elizabeth.moore@email.com', '555-0108', '258 Spruce Ave', 'San Diego', 'CA', '92101', 'active', '2024-01-22'),
('David', 'Taylor', 'david.taylor@hotmail.com', '555-0109', '369 Willow Rd', 'Dallas', 'TX', '75201', 'active', '2024-01-23'),
('Barbara', 'Anderson', 'barbara.anderson@gmail.com', '555-0110', '741 Ash Ct', 'San Jose', 'CA', '95101', 'active', '2024-01-24'),
('Richard', 'Thomas', 'richard.thomas@email.com', '555-0111', '852 Poplar Way', 'Austin', 'TX', '78701', 'active', '2024-01-25'),
('Susan', 'Jackson', 'susan.jackson@yahoo.com', '555-0112', '963 Walnut Blvd', 'Jacksonville', 'FL', '32099', 'active', '2024-01-26'),
('Joseph', 'White', 'joseph.white@gmail.com', '555-0113', '159 Cherry Ln', 'Fort Worth', 'TX', '76101', 'active', '2024-01-27'),
('Jessica', 'Harris', 'jessica.harris@hotmail.com', '555-0114', '357 Beech Dr', 'Columbus', 'OH', '43085', 'active', '2024-01-28'),
('Thomas', 'Martin', 'thomas.martin@email.com', '555-0115', '486 Hickory Rd', 'Charlotte', 'NC', '28201', 'active', '2024-01-29'),
('Sarah', 'Thompson', 'sarah.thompson@gmail.com', '555-0116', '753 Sycamore St', 'San Francisco', 'CA', '94101', 'active', '2024-01-30'),
('Charles', 'Garcia', 'charles.garcia@yahoo.com', '555-0117', '951 Magnolia Ave', 'Indianapolis', 'IN', '46201', 'active', '2024-01-31'),
('Karen', 'Martinez', 'karen.martinez@email.com', '555-0118', '842 Dogwood Ct', 'Seattle', 'WA', '98101', 'active', '2024-02-01'),
('Daniel', 'Robinson', 'daniel.robinson@hotmail.com', '555-0119', '624 Redwood Ln', 'Denver', 'CO', '80201', 'active', '2024-02-02'),
('Nancy', 'Clark', 'nancy.clark@gmail.com', '555-0120', '735 Juniper Way', 'Washington', 'DC', '20001', 'active', '2024-02-03'),
('Matthew', 'Rodriguez', 'matthew.rodriguez@email.com', '555-0121', '846 Palm Dr', 'Boston', 'MA', '02101', 'active', '2024-02-04'),
('Betty', 'Lewis', 'betty.lewis@yahoo.com', '555-0122', '957 Oak Ridge Rd', 'Nashville', 'TN', '37201', 'active', '2024-02-05'),
('Mark', 'Lee', 'mark.lee@gmail.com', '555-0123', '135 Pine Valley Dr', 'El Paso', 'TX', '79901', 'active', '2024-02-06'),
('Helen', 'Walker', 'helen.walker@hotmail.com', '555-0124', '246 Cedar Creek Ln', 'Detroit', 'MI', '48201', 'active', '2024-02-07'),
('Donald', 'Hall', 'donald.hall@email.com', '555-0125', '579 Maple Heights Ct', 'Portland', 'OR', '97201', 'active', '2024-02-08'),
('Dorothy', 'Allen', 'dorothy.allen@gmail.com', '555-0126', '680 Elm Grove Ave', 'Las Vegas', 'NV', '89101', 'active', '2024-02-09'),
('Steven', 'Young', 'steven.young@yahoo.com', '555-0127', '791 Birch Forest Way', 'Memphis', 'TN', '37501', 'active', '2024-02-10'),
('Sandra', 'Hernandez', 'sandra.hernandez@email.com', '555-0128', '802 Willow Creek Dr', 'Louisville', 'KY', '40201', 'active', '2024-02-11'),
('Paul', 'King', 'paul.king@hotmail.com', '555-0129', '913 Spruce Mountain Rd', 'Baltimore', 'MD', '21201', 'active', '2024-02-12'),
('Ashley', 'Wright', 'ashley.wright@gmail.com', '555-0130', '124 Ash Valley Ln', 'Milwaukee', 'WI', '53201', 'active', '2024-02-13');

-- Generate remaining 470 customers (31-500) with varied, realistic patterns
INSERT INTO customers (first_name, last_name, email, phone, address_line1, city, state, zip_code, status, created_at)
SELECT 
    -- Vary first names for more realism
    ELT(MOD(seq, 20) + 1, 
        'Alex', 'Emma', 'Noah', 'Olivia', 'Liam', 
        'Ava', 'Mason', 'Sophia', 'Lucas', 'Isabella',
        'Ethan', 'Mia', 'Logan', 'Charlotte', 'Oliver',
        'Amelia', 'Elijah', 'Harper', 'Jackson', 'Evelyn') AS first_name,
    -- Vary last names for more realism
    ELT(MOD(seq, 25) + 1,
        'Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
        'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
        'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
        'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
        'Lee', 'Perez', 'Thompson', 'White', 'Harris') AS last_name,
    CONCAT('customer', LPAD(seq, 3, '0'), '@', 
        ELT(MOD(seq, 5) + 1, 'gmail.com', 'yahoo.com', 'hotmail.com', 'email.com', 'outlook.com')) AS email,
    CONCAT('555-', LPAD(seq, 4, '0')) AS phone,
    CONCAT(seq * 10, ' ', 
        ELT(MOD(seq, 15) + 1, 'Main', 'Oak', 'Pine', 'Maple', 'Cedar',
            'Elm', 'Washington', 'Park', 'Lake', 'Hill',
            'Valley', 'River', 'Forest', 'Mountain', 'Meadow'),
        ' ',
        ELT(MOD(seq, 8) + 1, 'St', 'Ave', 'Blvd', 'Dr', 'Ln', 'Ct', 'Way', 'Rd')) AS address_line1,
    -- 50 major US cities for variety
    ELT(MOD(seq, 50) + 1, 
        'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
        'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
        'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
        'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington',
        'Boston', 'Nashville', 'El Paso', 'Detroit', 'Portland',
        'Las Vegas', 'Memphis', 'Louisville', 'Baltimore', 'Milwaukee',
        'Albuquerque', 'Tucson', 'Fresno', 'Sacramento', 'Kansas City',
        'Mesa', 'Atlanta', 'Omaha', 'Colorado Springs', 'Raleigh',
        'Miami', 'Cleveland', 'Tulsa', 'Oakland', 'Minneapolis',
        'Wichita', 'Arlington', 'Tampa', 'New Orleans', 'Bakersfield') AS city,
    -- Corresponding state codes
    ELT(MOD(seq, 50) + 1,
        'NY', 'CA', 'IL', 'TX', 'AZ',
        'PA', 'TX', 'CA', 'TX', 'CA',
        'TX', 'FL', 'TX', 'OH', 'NC',
        'CA', 'IN', 'WA', 'CO', 'DC',
        'MA', 'TN', 'TX', 'MI', 'OR',
        'NV', 'TN', 'KY', 'MD', 'WI',
        'NM', 'AZ', 'CA', 'CA', 'MO',
        'AZ', 'GA', 'NE', 'CO', 'NC',
        'FL', 'OH', 'OK', 'CA', 'MN',
        'KS', 'TX', 'FL', 'LA', 'CA') AS state,
    -- Generate realistic zip codes based on city pattern
    CONCAT(
        ELT(MOD(seq, 50) + 1,
            '100', '900', '606', '770', '850',
            '191', '782', '921', '752', '951',
            '787', '320', '761', '430', '282',
            '941', '462', '981', '802', '200',
            '021', '372', '799', '482', '972',
            '891', '375', '402', '212', '532',
            '871', '857', '937', '958', '641',
            '852', '303', '681', '809', '276',
            '331', '441', '741', '946', '554',
            '672', '760', '336', '701', '933'),
        LPAD(MOD(seq, 100), 2, '0')) AS zip_code,
    -- Status distribution: 80% active, 15% inactive, 5% suspended
    CASE 
        WHEN MOD(seq, 20) = 0 THEN 'suspended'
        WHEN MOD(seq, 7) = 0 THEN 'inactive'
        ELSE 'active'
    END AS status,
    -- Spread creation dates across 2024
    DATE_ADD('2024-02-14', INTERVAL seq DAY) AS created_at
FROM (
    SELECT @row := @row + 1 as seq
    FROM 
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t2,
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t3,
        (SELECT @row := 30) r
) numbers
WHERE seq <= 470;

-- Display confirmation and statistics
SELECT 'Sample customers inserted successfully!' AS Status;

SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_customers,
    SUM(CASE WHEN status = 'inactive' THEN 1 ELSE 0 END) AS inactive_customers,
    SUM(CASE WHEN status = 'suspended' THEN 1 ELSE 0 END) AS suspended_customers
FROM customers;

SELECT 'Customer distribution by state (top 10):' AS Info;
SELECT state, COUNT(*) as customer_count 
FROM customers 
GROUP BY state 
ORDER BY customer_count DESC 
LIMIT 10;