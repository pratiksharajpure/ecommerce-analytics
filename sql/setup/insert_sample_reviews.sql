-- ========================================
-- INSERT SAMPLE PRODUCT REVIEWS
-- 500 realistic product reviews from customers
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- REVIEWS (500 records)
-- ========================================

-- Generate reviews for products with varied ratings
INSERT INTO reviews (product_id, customer_id, rating, review_title, review_comment, is_verified_purchase, helpful_count, status, created_at)
SELECT 
    -- Product ID (1-300, some products get more reviews)
    CASE 
        WHEN seq % 3 = 0 THEN FLOOR(1 + (RAND() * 50))  -- Popular products get more reviews
        ELSE FLOOR(1 + (RAND() * 300))
    END AS product_id,
    -- Customer ID (only customers who have ordered)
    FLOOR(1 + (RAND() * 500)) AS customer_id,
    -- Rating distribution: 60% are 4-5 stars, 25% are 3 stars, 15% are 1-2 stars
    CASE 
        WHEN RAND() < 0.35 THEN 5
        WHEN RAND() < 0.60 THEN 4
        WHEN RAND() < 0.85 THEN 3
        WHEN RAND() < 0.95 THEN 2
        ELSE 1
    END AS rating,
    -- Review title based on rating
    CASE FLOOR(1 + (RAND() * 5))
        WHEN 1 THEN 'Great product!'
        WHEN 2 THEN 'Excellent quality'
        WHEN 3 THEN 'Good value for money'
        WHEN 4 THEN 'Very satisfied'
        ELSE 'Highly recommend'
    END AS review_title,
    -- Review comment
    'Product exceeded my expectations. Quality is excellent and delivery was fast.' AS review_comment,
    -- 85% are verified purchases
    RAND() < 0.85 AS is_verified_purchase,
    -- Helpful count (0-50, higher for older reviews)
    FLOOR(RAND() * 50) AS helpful_count,
    -- 95% approved, 4% pending, 1% rejected
    CASE 
        WHEN RAND() < 0.95 THEN 'approved'
        WHEN RAND() < 0.99 THEN 'pending'
        ELSE 'rejected'
    END AS status,
    -- Created date in 2024
    DATE_ADD('2024-01-01', INTERVAL FLOOR(RAND() * 365) DAY) AS created_at
FROM (
    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected_reviews,
    SUM(CASE WHEN is_verified_purchase THEN 1 ELSE 0 END) AS verified_purchases,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(helpful_count), 2) AS avg_helpful_count
FROM reviews;

SELECT 'Reviews by rating distribution:' AS Info;
SELECT 
    rating,
    COUNT(*) AS review_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) AS percentage
FROM reviews
GROUP BY rating
ORDER BY rating DESC;

SELECT 'Top 10 most reviewed products:' AS Info;
SELECT 
    p.product_id,
    p.product_name,
    COUNT(r.review_id) AS review_count,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    SUM(r.helpful_count) AS total_helpful_votes
FROM products p
JOIN reviews r ON p.product_id = r.product_id
WHERE r.status = 'approved'
GROUP BY p.product_id, p.product_name
ORDER BY review_count DESC
LIMIT 10;

SELECT 'Top 10 highest rated products (min 5 reviews):' AS Info;
SELECT 
    p.product_id,
    p.product_name,
    COUNT(r.review_id) AS review_count,
    ROUND(AVG(r.rating), 2) AS avg_rating
FROM products p
JOIN reviews r ON p.product_id = r.product_id
WHERE r.status = 'approved'
GROUP BY p.product_id, p.product_name
HAVING review_count >= 5
ORDER BY avg_rating DESC, review_count DESC
LIMIT 10;

SELECT 'Loyalty program statistics:' AS Info;
SELECT 
    COUNT(*) AS total_members,
    SUM(CASE WHEN tier = 'platinum' THEN 1 ELSE 0 END) AS platinum_members,
    SUM(CASE WHEN tier = 'gold' THEN 1 ELSE 0 END) AS gold_members,
    SUM(CASE WHEN tier = 'silver' THEN 1 ELSE 0 END) AS silver_members,
    SUM(CASE WHEN tier = 'bronze' THEN 1 ELSE 0 END) AS bronze_members,
    ROUND(AVG(points_balance), 0) AS avg_points_balance,
    SUM(points_earned_lifetime) AS total_points_earned,
    SUM(points_redeemed_lifetime) AS total_points_redeemed
FROM loyalty_program;ELECT @row := @row + 1 as seq
    FROM 
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t1,
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t2,
        (SELECT 0 UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 
         UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) t3,
        (SELECT @row := 0) r
) numbers
WHERE seq < 500;

-- Update review titles and comments based on rating
UPDATE reviews SET 
    review_title = CASE rating
        WHEN 5 THEN ELT(FLOOR(1 + RAND() * 10),
            'Absolutely love it!',
            'Perfect in every way',
            'Best purchase ever',
            'Exceeded all expectations',
            'Cannot recommend enough',
            'Five stars all the way',
            'Outstanding quality',
            'Worth every penny',
            'Amazing product',
            'Flawless and fantastic')
        WHEN 4 THEN ELT(FLOOR(1 + RAND() * 10),
            'Great product overall',
            'Very happy with purchase',
            'Good quality item',
            'Solid choice',
            'Meets expectations',
            'Pretty good product',
            'Nice quality',
            'Happy customer',
            'Would buy again',
            'Satisfied with quality')
        WHEN 3 THEN ELT(FLOOR(1 + RAND() * 10),
            'It''s okay',
            'Average product',
            'Does the job',
            'Decent for the price',
            'Neither great nor terrible',
            'Acceptable quality',
            'Could be better',
            'Mixed feelings',
            'Not bad, not great',
            'Fair value')
        WHEN 2 THEN ELT(FLOOR(1 + RAND() * 10),
            'Disappointed',
            'Not what I expected',
            'Below average',
            'Quality issues',
            'Could be much better',
            'Not impressed',
            'Had problems',
            'Somewhat dissatisfied',
            'Not worth the price',
            'Expected more')
        ELSE ELT(FLOOR(1 + RAND() * 10),
            'Very disappointed',
            'Do not buy',
            'Terrible quality',
            'Waste of money',
            'Returned immediately',
            'Awful experience',
            'Poor quality',
            'Complete letdown',
            'Not recommended',
            'Save your money')
    END,
    review_comment = CASE rating
        WHEN 5 THEN ELT(FLOOR(1 + RAND() * 15),
            'This product is absolutely fantastic! The quality exceeded my expectations and it works perfectly. Shipping was fast and packaging was great. Highly recommend to anyone looking for this type of item.',
            'I am so impressed with this purchase. The build quality is excellent, and it does exactly what it promises. Best decision I made this year. Will definitely order more products from this seller.',
            'Outstanding product! Everything about it is perfect - from the design to functionality. Customer service was also top-notch. Cannot say enough good things about this item.',
            'Exactly what I was looking for and more. The attention to detail is remarkable and you can tell this is a premium product. Worth every single penny. Five stars all the way!',
            'Amazing quality and great value. This product has made my life so much easier. The features work flawlessly and I love using it every day. Couldn''t be happier with this purchase.',
            'Best product in its category that I have ever used. The performance is exceptional and it feels very well-made. Delivery was quick and the item arrived in perfect condition.',
            'I researched many similar products before choosing this one, and I am so glad I did. Superior quality and excellent functionality. This company really knows what they''re doing.',
            'Phenomenal product! Surpassed all my expectations in terms of quality and performance. The price point is very reasonable for what you get. Highly recommend without reservation.',
            'Love everything about this item. The quality is top-tier and it works better than advertised. Great purchase that I would make again in a heartbeat.',
            'Exceptional product that delivers on all promises. The craftsmanship is evident and it feels like a premium item. Very satisfied with this investment.',
            'Perfect purchase! This product is exactly as described and the quality is outstanding. Fast shipping and excellent packaging. Will be a repeat customer for sure.',
            'Couldn''t ask for anything better. This product has exceeded my expectations in every way possible. The functionality is superb and it looks great too.',
            'Absolutely thrilled with this purchase! The quality is fantastic and it works beautifully. Great value for money. Highly recommend this to everyone.',
            'This is hands down the best product of its kind. The attention to detail and quality of materials is impressive. Very happy customer here!',
            'Incredible product that works flawlessly. The design is thoughtful and the execution is perfect. One of my best purchases this year.')
        WHEN 4 THEN ELT(FLOOR(1 + RAND() * 12),
            'Very good product overall. The quality is solid and it works as expected. Only minor issue is that it could have slightly better packaging, but the product itself is great.',
            'Happy with this purchase. Good quality and does what it''s supposed to do. Delivery was on time and the item was well-protected. Would recommend to others.',
            'Pleased with the quality and functionality. It meets all my needs and the price is fair. Not perfect but definitely a good buy. Four stars from me.',
            'Good product for the price. Works well and seems durable. Wish it had a few more features but overall I am satisfied with my purchase.',
            'Pretty satisfied with this item. The quality is decent and it arrived quickly. A few small improvements could make it perfect, but it''s a solid choice.',
            'Nice product that delivers on its promises. Good build quality and functions as advertised. Would buy again if I needed another one.',
            'Solid purchase. The product is well-made and works reliably. Customer service was responsive. Only giving four stars because there''s always room for improvement.',
            'Good value for money. The product quality is above average and it does the job well. Delivery was prompt and hassle-free.',
            'Very pleased with this product. It''s well-designed and functional. Only minor complaint is the instructions could be clearer, but otherwise great.',
            'Quality product that works as described. Happy with the performance and the price point is reasonable. Would recommend to friends.',
            'Bought this based on reviews and I''m not disappointed. Good quality and reliable performance. Just a couple of small issues that prevent it from being perfect.',
            'Overall a good purchase. The product meets expectations and the quality is satisfactory. Fast shipping and good customer service.')
        WHEN 3 THEN ELT(FLOOR(1 + RAND() * 10),
            'It''s an okay product. Does what it''s supposed to but nothing special. Quality is average for the price. Might look for alternatives next time.',
            'Average product that gets the job done. Not particularly impressive but not terrible either. Expected a bit more based on the price.',
            'Decent product but has some issues. Works most of the time but quality could be better. For the price, I expected slightly higher quality.',
            'It''s alright. The product functions adequately but there are some quality concerns. Not sure if I would purchase again.',
            'Mixed feelings about this purchase. Some aspects are good while others are disappointing. Average quality overall.',
            'The product is acceptable but not outstanding. It works but I''ve had some minor issues. Considering the price, it''s just okay.',
            'Fair value for money. The product does its job but don''t expect premium quality. It''s serviceable but not exceptional.',
            'Somewhat satisfied with this purchase. Quality is mediocre and there''s definitely room for improvement. It''s functional though.',
            'Product is okay for basic needs. Nothing fancy but it works. Quality is about what you''d expect at this price point.',
            'Neither impressed nor disappointed. The product is functional but unremarkable. There are probably better options available.')
        WHEN 2 THEN ELT(FLOOR(1 + RAND() * 8),
            'Disappointed with this purchase. The quality is below what I expected based on the description. It works but not very well. Would not recommend.',
            'Not happy with this product. Several quality issues right out of the box. Expected much better for the price. Consider other options.',
            'Below average product. Had problems from the start and quality is questionable. Customer service was slow to respond. Not satisfied.',
            'The product doesn''t live up to the description. Quality is poor and it feels cheaply made. Wish I had read more reviews before buying.',
            'Pretty disappointed overall. The item works but not reliably. Quality control seems lacking. Would not purchase again.',
            'Not what I expected at all. Multiple issues with functionality and the build quality is subpar. Regret this purchase.',
            'Quality is definitely not as advertised. The product feels cheap and doesn''t work as well as it should. Not worth the money.',
            'Unsatisfied with this purchase. Product arrived with defects and performance is mediocre at best. Expected better.')
        ELSE ELT(FLOOR(1 + RAND() * 6),
            'Terrible product, do not waste your money. Quality is awful and it barely works. Already returning it. Extremely disappointed.',
            'Complete waste of money. Product broke within days of use and quality is horrible. Save yourself the trouble and buy something else.',
            'Worst purchase I''ve made in a long time. Poor quality, doesn''t work as advertised, and customer service is unhelpful. Avoid at all costs.',
            'Very poor quality product. Didn''t work properly from day one. Returning this immediately. Would give zero stars if I could.',
            'Absolutely terrible. The product is cheaply made and doesn''t function correctly. False advertising in my opinion. Very upset with this purchase.',
            'Do not buy this product! Quality is atrocious and it''s not worth even half the price. Returning and will never order from here again.')
    END;

-- Add variety to verified purchases (make recent orders more likely to be verified)
UPDATE reviews 
SET is_verified_purchase = TRUE 
WHERE created_at > DATE_SUB(CURDATE(), INTERVAL 90 DAY)
AND RAND() < 0.95;

-- Increase helpful count for older, high-rated reviews
UPDATE reviews 
SET helpful_count = helpful_count + FLOOR(RAND() * 100)
WHERE rating >= 4 
AND created_at < DATE_SUB(CURDATE(), INTERVAL 180 DAY)
AND RAND() < 0.3;

-- Make some recent reviews pending
UPDATE reviews 
SET status = 'pending'
WHERE created_at > DATE_SUB(CURDATE(), INTERVAL 7 DAY)
AND RAND() < 0.15;

-- Reject some very negative or suspicious reviews
UPDATE reviews 
SET status = 'rejected'
WHERE (rating = 1 OR helpful_count = 0)
AND RAND() < 0.05;

-- ========================================
-- LOYALTY PROGRAM ENROLLMENT
-- Enroll 40% of customers who have placed orders
-- ========================================

INSERT INTO loyalty_program (customer_id, points_balance, points_earned_lifetime, points_redeemed_lifetime, tier, tier_start_date, joined_date)
SELECT DISTINCT
    o.customer_id,
    -- Points balance based on order history
    FLOOR(100 + (RAND() * 900)) AS points_balance,
    -- Lifetime points earned
    FLOOR(500 + (RAND() * 4500)) AS points_earned_lifetime,
    -- Lifetime points redeemed
    FLOOR(RAND() * 500) AS points_redeemed_lifetime,
    -- Tier based on activity
    CASE 
        WHEN RAND() < 0.10 THEN 'platinum'
        WHEN RAND() < 0.30 THEN 'gold'
        WHEN RAND() < 0.60 THEN 'silver'
        ELSE 'bronze'
    END AS tier,
    -- Tier start date
    DATE_SUB(CURDATE(), INTERVAL FLOOR(RAND() * 180) DAY) AS tier_start_date,
    -- Joined date (before their first order)
    DATE_SUB((SELECT MIN(order_date) FROM orders o2 WHERE o2.customer_id = o.customer_id), INTERVAL FLOOR(1 + RAND() * 30) DAY) AS joined_date
FROM orders o
WHERE o.customer_id % 5 <= 1  -- 40% of customers
AND o.customer_id <= 200
GROUP BY o.customer_id;

-- Update points balance to be realistic (earned - redeemed)
UPDATE loyalty_program 
SET points_balance = GREATEST(0, points_earned_lifetime - points_redeemed_lifetime - FLOOR(RAND() * 200));

-- ========================================
-- DISPLAY CONFIRMATION & STATISTICS
-- ========================================

SELECT 'Reviews and loyalty program data inserted successfully!' AS Status;

SELECT 
    COUNT(*) AS total_reviews,
    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) AS approved_reviews,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_reviews,
    S