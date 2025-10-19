-- ========================================
-- REVIEW_AUTHENTICITY.SQL
-- Data Quality Check: Suspicious Reviews Detection
-- Path: sql/core_analysis/review_authenticity.sql
-- Identifies fake, suspicious, or low-quality reviews
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Reviews Without Verified Purchase
-- ========================================
SELECT 
    r.review_id,
    r.product_id,
    p.product_name,
    r.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    r.rating,
    r.review_title,
    LEFT(r.review_comment, 100) AS review_preview,
    r.is_verified_purchase,
    r.created_at,
    COUNT(oi.order_item_id) AS customer_purchases,
    'Review without verified purchase' AS issue_type
FROM reviews r
LEFT JOIN products p ON r.product_id = p.product_id
LEFT JOIN customers c ON r.customer_id = c.customer_id
LEFT JOIN order_items oi ON r.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id AND o.customer_id = r.customer_id
WHERE r.is_verified_purchase = FALSE
    AND r.status = 'approved'
GROUP BY r.review_id, r.product_id, p.product_name, r.customer_id, 
         c.first_name, c.last_name, r.rating, r.review_title, r.review_comment, 
         r.is_verified_purchase, r.created_at
ORDER BY r.created_at DESC;

-- ========================================
-- 2. Multiple Reviews from Same Customer for Same Product
-- ========================================
SELECT 
    r.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    r.product_id,
    p.product_name,
    COUNT(r.review_id) AS review_count,
    GROUP_CONCAT(r.rating ORDER BY r.created_at) AS ratings,
    MIN(r.created_at) AS first_review,
    MAX(r.created_at) AS latest_review,
    'Multiple reviews for same product' AS issue_type
FROM reviews r
LEFT JOIN customers c ON r.customer_id = c.customer_id
LEFT JOIN products p ON r.product_id = p.product_id
GROUP BY r.customer_id, c.first_name, c.last_name, r.product_id, p.product_name
HAVING COUNT(r.review_id) > 1
ORDER BY review_count DESC;

-- ========================================
-- 3. Reviews with Identical or Very Similar Text
-- ========================================
SELECT 
    r1.review_id AS review_id_1,
    r2.review_id AS review_id_2,
    r1.product_id AS product_1,
    r2.product_id AS product_2,
    p1.product_name AS product_name_1,
    p2.product_name AS product_name_2,
    r1.customer_id AS customer_1,
    r2.customer_id AS customer_2,
    r1.rating AS rating_1,
    r2.rating AS rating_2,
    LEFT(r1.review_comment, 100) AS comment_preview,
    r1.created_at AS date_1,
    r2.created_at AS date_2,
    'Duplicate/suspicious review text' AS issue_type
FROM reviews r1
INNER JOIN reviews r2 ON r1.review_id < r2.review_id
    AND r1.review_comment = r2.review_comment
LEFT JOIN products p1 ON r1.product_id = p1.product_id
LEFT JOIN products p2 ON r2.product_id = p2.product_id
WHERE r1.review_comment IS NOT NULL
    AND LENGTH(r1.review_comment) > 20
    AND r1.status = 'approved'
    AND r2.status = 'approved'
ORDER BY r1.created_at DESC;

-- ========================================
-- 4. Suspicious Review Velocity
-- ========================================
SELECT 
    DATE(r.created_at) AS review_date,
    r.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    COUNT(r.review_id) AS reviews_posted,
    GROUP_CONCAT(DISTINCT p.product_name ORDER BY r.created_at SEPARATOR '; ') AS products_reviewed,
    'Multiple reviews in single day' AS issue_type
FROM reviews r
LEFT JOIN customers c ON r.customer_id = c.customer_id
LEFT JOIN products p ON r.product_id = p.product_id
GROUP BY DATE(r.created_at), r.customer_id, c.first_name, c.last_name, c.email
HAVING COUNT(r.review_id) >= 5
ORDER BY reviews_posted DESC, review_date DESC;

-- ========================================
-- 5. Reviews with Extreme Ratings Only
-- ========================================
SELECT 
    r.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    COUNT(r.review_id) AS total_reviews,
    SUM(CASE WHEN r.rating = 5 THEN 1 ELSE 0 END) AS five_star_reviews,
    SUM(CASE WHEN r.rating = 1 THEN 1 ELSE 0 END) AS one_star_reviews,
    SUM(CASE WHEN r.rating IN (2,3,4) THEN 1 ELSE 0 END) AS moderate_reviews,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    'Only extreme ratings (1 or 5 stars)' AS issue_type
FROM reviews r
LEFT JOIN customers c ON r.customer_id = c.customer_id
WHERE r.status = 'approved'
GROUP BY r.customer_id, c.first_name, c.last_name, c.email
HAVING COUNT(r.review_id) >= 5
    AND SUM(CASE WHEN r.rating IN (2,3,4) THEN 1 ELSE 0 END) = 0
ORDER BY total_reviews DESC;

-- ========================================
-- 6. Reviews with Missing or Poor Quality Content
-- ========================================
SELECT 
    r.review_id,
    r.product_id,
    p.product_name,
    r.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    r.rating,
    r.review_title,
    r.review_comment,
    LENGTH(COALESCE(r.review_comment, '')) AS comment_length,
    r.status,
    CASE 
        WHEN r.review_comment IS NULL OR r.review_comment = '' THEN 'No review text'
        WHEN LENGTH(r.review_comment) < 10 THEN 'Very short review (<10 chars)'
        WHEN r.review_title IS NULL OR r.review_title = '' THEN 'No review title'
    END AS issue_type
FROM reviews r
LEFT JOIN products p ON r.product_id = p.product_id
LEFT JOIN customers c ON r.customer_id = c.customer_id
WHERE r.status = 'approved'
    AND (
        r.review_comment IS NULL OR 
        r.review_comment = '' OR
        LENGTH(r.review_comment) < 10 OR
        r.review_title IS NULL OR
        r.review_title = ''
    )
ORDER BY r.created_at DESC;

-- ========================================
-- 7. Reviews for Products Customer Never Purchased
-- ========================================
SELECT 
    r.review_id,
    r.product_id,
    p.product_name,
    r.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    c.email,
    r.rating,
    r.is_verified_purchase,
    r.created_at,
    COUNT(o.order_id) AS customer_total_orders,
    'Review for never-purchased product' AS issue_type
FROM reviews r
LEFT JOIN products p ON r.product_id = p.product_id
LEFT JOIN customers c ON r.customer_id = c.customer_id
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id AND r.product_id = oi.product_id
WHERE oi.order_item_id IS NULL
    AND r.customer_id IS NOT NULL
GROUP BY r.review_id, r.product_id, p.product_name, r.customer_id, 
         c.first_name, c.last_name, c.email, r.rating, r.is_verified_purchase, r.created_at
ORDER BY r.created_at DESC;

-- ========================================
-- 8. Suspicious Helpful Vote Patterns
-- ========================================
SELECT 
    r.review_id,
    r.product_id,
    p.product_name,
    r.customer_id,
    r.rating,
    r.helpful_count,
    DATEDIFF(CURRENT_DATE, r.created_at) AS days_since_posted,
    ROUND(r.helpful_count / NULLIF(DATEDIFF(CURRENT_DATE, r.created_at), 0), 2) AS helpful_votes_per_day,
    r.status,
    'Unusually high helpful vote count' AS issue_type
FROM reviews r
LEFT JOIN products p ON r.product_id = p.product_id
WHERE r.helpful_count >= 100
    OR (r.helpful_count > 50 AND DATEDIFF(CURRENT_DATE, r.created_at) < 7)
ORDER BY helpful_votes_per_day DESC;

-- ========================================
-- 9. Product Review Discrepancies
-- ========================================
SELECT 
    p.product_id,
    p.sku,
    p.product_name,
    COUNT(r.review_id) AS total_reviews,
    COUNT(CASE WHEN r.status = 'approved' THEN 1 END) AS approved_reviews,
    COUNT(CASE WHEN r.status = 'rejected' THEN 1 END) AS rejected_reviews,
    COUNT(CASE WHEN r.status = 'pending' THEN 1 END) AS pending_reviews,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    ROUND((COUNT(CASE WHEN r.status = 'rejected' THEN 1 END) * 100.0) / NULLIF(COUNT(r.review_id), 0), 2) AS rejection_rate,
    COUNT(DISTINCT oi.order_id) AS times_ordered,
    CASE 
        WHEN COUNT(CASE WHEN r.status = 'rejected' THEN 1 END) * 100.0 / NULLIF(COUNT(r.review_id), 0) > 30 THEN 'High rejection rate'
        WHEN COUNT(CASE WHEN r.status = 'pending' THEN 1 END) > 20 THEN 'Many pending reviews'
    END AS issue_type
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.status = 'active'
GROUP BY p.product_id, p.sku, p.product_name
HAVING (COUNT(CASE WHEN r.status = 'rejected' THEN 1 END) * 100.0 / NULLIF(COUNT(r.review_id), 0) > 30)
    OR COUNT(CASE WHEN r.status = 'pending' THEN 1 END) > 20
ORDER BY rejection_rate DESC;

-- ========================================
-- 10. Review Authenticity Summary
-- ========================================
SELECT 
    'Total Reviews' AS metric,
    COUNT(*) AS count
FROM reviews

UNION ALL

SELECT 
    'Approved Reviews' AS metric,
    COUNT(*) AS count
FROM reviews
WHERE status = 'approved'

UNION ALL

SELECT 
    'Non-Verified Purchase Reviews' AS metric,
    COUNT(*) AS count
FROM reviews
WHERE is_verified_purchase = FALSE AND status = 'approved'

UNION ALL

SELECT 
    'Duplicate Review Text' AS metric,
    COUNT(DISTINCT r1.review_id) AS count
FROM reviews r1
INNER JOIN reviews r2 ON r1.review_id < r2.review_id AND r1.review_comment = r2.review_comment
WHERE r1.review_comment IS NOT NULL AND LENGTH(r1.review_comment) > 20

UNION ALL

SELECT 
    'Reviews for Never-Purchased Products' AS metric,
    COUNT(DISTINCT r.review_id) AS count
FROM reviews r
LEFT JOIN order_items oi ON r.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id AND o.customer_id = r.customer_id
WHERE oi.order_item_id IS NULL AND r.customer_id IS NOT NULL

UNION ALL

SELECT 
    'Reviews with No Comment' AS metric,
    COUNT(*) AS count
FROM reviews
WHERE (review_comment IS NULL OR review_comment = '') AND status = 'approved'

UNION ALL

SELECT 
    'Customers with Only Extreme Ratings' AS metric,
    COUNT(DISTINCT customer_id) AS count
FROM (
    SELECT customer_id
    FROM reviews
    WHERE status = 'approved'
    GROUP BY customer_id
    HAVING COUNT(*) >= 5
        AND SUM(CASE WHEN rating IN (2,3,4) THEN 1 ELSE 0 END) = 0
) AS extreme_raters;