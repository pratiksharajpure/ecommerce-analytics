-- ========================================
-- PERFORMANCE METRICS & OPTIMIZATION
-- E-commerce Revenue Analytics Engine
-- Query Performance, System Monitoring, Optimization
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. DATABASE SIZE & TABLE STATISTICS
-- Analyze storage usage and table sizes
-- ========================================
SELECT 
    table_name,
    ROUND((data_length + index_length) / 1024 / 1024, 2) AS total_size_mb,
    ROUND(data_length / 1024 / 1024, 2) AS data_size_mb,
    ROUND(index_length / 1024 / 1024, 2) AS index_size_mb,
    table_rows,
    ROUND(index_length / NULLIF(data_length, 0) * 100, 2) AS index_to_data_ratio,
    engine,
    table_collation,
    create_time,
    update_time,
    CASE 
        WHEN index_length > data_length * 2 THEN 'Over-indexed'
        WHEN index_length < data_length * 0.1 THEN 'Under-indexed'
        ELSE 'Balanced'
    END AS index_status
FROM information_schema.tables
WHERE table_schema = 'ecommerce_analytics'
  AND table_type = 'BASE TABLE'
ORDER BY (data_length + index_length) DESC;

-- ========================================
-- 2. INDEX USAGE ANALYSIS
-- Identify unused and redundant indexes
-- ========================================
SELECT 
    t.table_schema,
    t.table_name,
    s.index_name,
    GROUP_CONCAT(s.column_name ORDER BY s.seq_in_index) AS indexed_columns,
    s.non_unique,
    s.cardinality,
    CASE 
        WHEN s.index_name = 'PRIMARY' THEN 'Primary Key'
        WHEN s.non_unique = 0 THEN 'Unique Index'
        ELSE 'Regular Index'
    END AS index_type,
    -- Check if index is being used (requires MySQL 5.7+)
    COALESCE(us.table_rows, 0) AS table_rows_scanned,
    -- Recommendations
    CASE 
        WHEN s.cardinality IS NULL OR s.cardinality = 0 THEN 'Low Cardinality - Consider Removing'
        WHEN s.cardinality < 10 THEN 'Very Low Selectivity'
        ELSE 'Good Selectivity'
    END AS index_health
FROM information_schema.statistics s
JOIN information_schema.tables t 
    ON s.table_schema = t.table_schema 
    AND s.table_name = t.table_name
LEFT JOIN (
    SELECT table_schema, table_name, SUM(table_rows) as table_rows
    FROM information_schema.tables
    GROUP BY table_schema, table_name
) us ON s.table_schema = us.table_schema AND s.table_name = us.table_name
WHERE s.table_schema = 'ecommerce_analytics'
GROUP BY t.table_schema, t.table_name, s.index_name, s.non_unique, s.cardinality, us.table_rows
ORDER BY t.table_name, s.index_name;

-- ========================================
-- 3. TABLE SCAN PERFORMANCE
-- Identify tables with full table scans
-- ========================================
SELECT 
    t.table_name,
    t.table_rows,
    t.avg_row_length,
    ROUND((t.data_length + t.index_length) / 1024 / 1024, 2) AS total_size_mb,
    t.data_free AS fragmented_space,
    ROUND(t.data_free / NULLIF(t.data_length, 0) * 100, 2) AS fragmentation_percentage,
    -- Performance indicators
    CASE 
        WHEN t.table_rows > 100000 AND t.avg_row_length > 500 THEN 'Large & Wide - Optimize Queries'
        WHEN t.table_rows > 1000000 THEN 'Very Large - Consider Partitioning'
        WHEN t.data_free / NULLIF(t.data_length, 0) > 0.2 THEN 'High Fragmentation - Run OPTIMIZE'
        ELSE 'Performance OK'
    END AS performance_recommendation
FROM information_schema.tables t
WHERE t.table_schema = 'ecommerce_analytics'
  AND t.table_type = 'BASE TABLE'
ORDER BY t.table_rows DESC;

-- ========================================
-- 4. SLOW QUERY CANDIDATES
-- Identify potentially slow queries based on table joins
-- ========================================
-- Orders with many items (potential slow aggregations)
SELECT 
    'Orders with High Item Count' AS query_type,
    o.order_id,
    o.customer_id,
    o.order_date,
    COUNT(oi.order_item_id) AS item_count,
    ROUND(o.total_amount, 2) AS total_amount,
    'Consider pagination or caching' AS optimization_tip
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.order_id, o.customer_id, o.order_date, o.total_amount
HAVING item_count > 20
ORDER BY item_count DESC
LIMIT 10;

-- ========================================
-- 5. QUERY EXECUTION TIME ESTIMATES
-- Analyze complex query patterns
-- ========================================
-- Customer order history complexity
SELECT 
    'Customer Order History Complexity' AS analysis_type,
    c.customer_id,
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT oi.order_item_id) AS total_items,
    COUNT(DISTINCT r.return_id) AS total_returns,
    COUNT(DISTINCT rev.review_id) AS total_reviews,
    -- Complexity score (higher = more complex queries)
    (COUNT(DISTINCT o.order_id) * 2 + 
     COUNT(DISTINCT oi.order_item_id) + 
     COUNT(DISTINCT r.return_id) * 3 + 
     COUNT(DISTINCT rev.review_id)) AS query_complexity_score,
    CASE 
        WHEN (COUNT(DISTINCT o.order_id) * 2 + COUNT(DISTINCT oi.order_item_id)) > 500 
            THEN 'High - Use Indexed Views or Materialized Results'
        WHEN (COUNT(DISTINCT o.order_id) * 2 + COUNT(DISTINCT oi.order_item_id)) > 200 
            THEN 'Medium - Monitor Performance'
        ELSE 'Low - No Action Needed'
    END AS optimization_priority
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN returns r ON o.order_id = r.order_id
LEFT JOIN reviews rev ON c.customer_id = rev.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING query_complexity_score > 100
ORDER BY query_complexity_score DESC
LIMIT 20;

-- ========================================
-- 6. JOIN PERFORMANCE ANALYSIS
-- Identify potentially problematic joins
-- ========================================
SELECT 
    'Product-Order Join Analysis' AS analysis_type,
    p.product_id,
    p.product_name,
    COUNT(DISTINCT oi.order_id) AS orders_count,
    COUNT(oi.order_item_id) AS order_items_count,
    AVG(oi.quantity) AS avg_quantity,
    -- Join efficiency metrics
    COUNT(oi.order_item_id) / NULLIF(COUNT(DISTINCT oi.order_id), 0) AS items_per_order_ratio,
    CASE 
        WHEN COUNT(oi.order_item_id) > 10000 THEN 'Very High Traffic - Index Critical'
        WHEN COUNT(oi.order_item_id) > 1000 THEN 'High Traffic - Monitor Indexes'
        ELSE 'Normal Traffic'
    END AS traffic_level,
    CASE 
        WHEN COUNT(oi.order_item_id) / NULLIF(COUNT(DISTINCT oi.order_id), 0) > 5 
            THEN 'Multiple Items Per Order - Good Bundling'
        ELSE 'Single Item Orders - Marketing Opportunity'
    END AS order_pattern
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE p.status = 'active'
GROUP BY p.product_id, p.product_name
HAVING orders_count > 0
ORDER BY order_items_count DESC
LIMIT 20;

-- ========================================
-- 7. MISSING INDEX RECOMMENDATIONS
-- Suggest indexes based on query patterns
-- ========================================
-- Check for foreign keys without indexes
SELECT 
    'Missing Index Recommendations' AS recommendation_type,
    kcu.table_name,
    kcu.column_name,
    kcu.referenced_table_name,
    kcu.referenced_column_name,
    CONCAT('CREATE INDEX idx_', kcu.table_name, '_', kcu.column_name, 
           ' ON ', kcu.table_name, '(', kcu.column_name, ');') AS suggested_index
FROM information_schema.key_column_usage kcu
LEFT JOIN information_schema.statistics s 
    ON kcu.table_schema = s.table_schema
    AND kcu.table_name = s.table_name
    AND kcu.column_name = s.column_name
    AND s.seq_in_index = 1
WHERE kcu.table_schema = 'ecommerce_analytics'
  AND kcu.referenced_table_name IS NOT NULL
  AND s.index_name IS NULL
ORDER BY kcu.table_name, kcu.column_name;

-- ========================================
-- 8. QUERY CACHE OPPORTUNITIES
-- Identify frequently accessed data
-- ========================================
-- Most accessed products (cache candidates)
SELECT 
    'Cache Candidate - Hot Products' AS cache_type,
    p.product_id,
    p.sku,
    p.product_name,
    COUNT(DISTINCT o.order_id) AS order_frequency,
    COUNT(oi.order_item_id) AS total_line_items,
    SUM(oi.quantity) AS total_units_sold,
    ROUND(AVG(oi.unit_price), 2) AS avg_selling_price,
    COUNT(DISTINCT DATE(oi.created_at)) AS days_with_sales,
    -- Access pattern
    ROUND(COUNT(DISTINCT o.order_id) / NULLIF(COUNT(DISTINCT DATE(oi.created_at)), 0), 2) AS avg_orders_per_day,
    CASE 
        WHEN COUNT(DISTINCT o.order_id) > 100 THEN 'High Priority Cache'
        WHEN COUNT(DISTINCT o.order_id) > 50 THEN 'Medium Priority Cache'
        ELSE 'Low Priority Cache'
    END AS cache_priority
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE oi.created_at >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 30 DAY)
GROUP BY p.product_id, p.sku, p.product_name
HAVING order_frequency > 10
ORDER BY order_frequency DESC
LIMIT 50;

-- ========================================
-- 9. DATA DISTRIBUTION ANALYSIS
-- Check for skewed data that affects performance
-- ========================================
-- Order distribution by customer
SELECT 
    'Order Distribution Analysis' AS metric_type,
    CASE 
        WHEN order_count = 1 THEN '1 Order'
        WHEN order_count BETWEEN 2 AND 5 THEN '2-5 Orders'
        WHEN order_count BETWEEN 6 AND 10 THEN '6-10 Orders'
        WHEN order_count BETWEEN 11 AND 20 THEN '11-20 Orders'
        WHEN order_count > 20 THEN '20+ Orders'
    END AS order_bucket,
    COUNT(*) AS customer_count,
    ROUND(AVG(total_spent), 2) AS avg_spent_per_customer,
    ROUND(SUM(total_spent), 2) AS bucket_total_revenue,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS customer_percentage,
    ROUND(SUM(total_spent) * 100.0 / SUM(SUM(total_spent)) OVER (), 2) AS revenue_percentage
FROM (
    SELECT 
        c.customer_id,
        COUNT(o.order_id) AS order_count,
        SUM(o.total_amount) AS total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.payment_status = 'paid'
    GROUP BY c.customer_id
) customer_orders
GROUP BY order_bucket
ORDER BY MIN(order_count);

-- ========================================
-- 10. TEMPORAL PERFORMANCE PATTERNS
-- Identify time-based query patterns
-- ========================================
SELECT 
    'Temporal Query Pattern Analysis' AS analysis_type,
    HOUR(o.order_date) AS hour_of_day,
    COUNT(DISTINCT o.order_id) AS order_count,
    COUNT(oi.order_item_id) AS item_count,
    ROUND(AVG(o.total_amount), 2) AS avg_order_value,
    -- Performance implications
    CASE 
        WHEN HOUR(o.order_date) BETWEEN 8 AND 17 THEN 'Business Hours - High Load'
        WHEN HOUR(o.order_date) BETWEEN 18 AND 23 THEN 'Evening - Medium Load'
        ELSE 'Night - Low Load - Maintenance Window'
    END AS load_period,
    CASE 
        WHEN COUNT(DISTINCT o.order_id) > 100 THEN 'Peak Time - Scale Up'
        WHEN COUNT(DISTINCT o.order_id) > 50 THEN 'Moderate - Monitor'
        ELSE 'Low Traffic - Good for Maintenance'
    END AS scaling_recommendation
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 30 DAY)
GROUP BY HOUR(o.order_date)
ORDER BY HOUR(o.order_date);

-- ========================================
-- 11. COMPOSITE INDEX RECOMMENDATIONS
-- Suggest multi-column indexes for common queries
-- ========================================
SELECT 
    'Composite Index Recommendations' AS recommendation_type,
    'orders' AS table_name,
    'customer_id, order_date, status' AS suggested_columns,
    'CREATE INDEX idx_orders_customer_date_status ON orders(customer_id, order_date, status);' AS sql_command,
    'Optimizes customer order history queries with filters' AS benefit
UNION ALL
SELECT 
    'Composite Index Recommendations',
    'order_items',
    'order_id, product_id',
    'CREATE INDEX idx_order_items_order_product ON order_items(order_id, product_id);',
    'Optimizes order detail lookups'
UNION ALL
SELECT 
    'Composite Index Recommendations',
    'products',
    'category_id, status, price',
    'CREATE INDEX idx_products_category_status_price ON products(category_id, status, price);',
    'Optimizes category browsing with filters'
UNION ALL
SELECT 
    'Composite Index Recommendations',
    'reviews',
    'product_id, status, rating',
    'CREATE INDEX idx_reviews_product_status_rating ON reviews(product_id, status, rating);',
    'Optimizes product review displays'
UNION ALL
SELECT 
    'Composite Index Recommendations',
    'inventory',
    'product_id, quantity_available',
    'CREATE INDEX idx_inventory_product_available ON inventory(product_id, quantity_available);',
    'Optimizes stock availability checks';

-- ========================================
-- 12. PARTITION RECOMMENDATIONS
-- Suggest table partitioning opportunities
-- ========================================
SELECT 
    'Partition Recommendations' AS recommendation_type,
    table_name,
    table_rows,
    ROUND((data_length + index_length) / 1024 / 1024, 2) AS size_mb,
    CASE 
        WHEN table_name = 'orders' THEN 'RANGE PARTITION BY order_date (monthly or quarterly)'
        WHEN table_name = 'order_items' THEN 'RANGE PARTITION BY created_at (monthly)'
        WHEN table_name = 'campaign_performance' THEN 'RANGE PARTITION BY report_date (monthly)'
        WHEN table_name = 'reviews' THEN 'RANGE PARTITION BY created_at (quarterly)'
        ELSE 'No partitioning needed'
    END AS partition_strategy,
    CASE 
        WHEN table_rows > 1000000 THEN 'High Priority'
        WHEN table_rows > 500000 THEN 'Medium Priority'
        ELSE 'Low Priority'
    END AS partition_priority
FROM information_schema.tables
WHERE table_schema = 'ecommerce_analytics'
  AND table_type = 'BASE TABLE'
  AND table_rows > 100000
ORDER BY table_rows DESC;

-- ========================================
-- 13. QUERY OPTIMIZATION CHECKLIST
-- Performance optimization summary
-- ========================================
SELECT 
    'Performance Optimization Summary' AS summary_type,
    (SELECT COUNT(*) FROM information_schema.tables 
     WHERE table_schema = 'ecommerce_analytics' AND table_type = 'BASE TABLE') AS total_tables,
    (SELECT COUNT(DISTINCT table_name) FROM information_schema.statistics 
     WHERE table_schema = 'ecommerce_analytics') AS tables_with_indexes,
    (SELECT COUNT(*) FROM information_schema.statistics 
     WHERE table_schema = 'ecommerce_analytics' AND index_name != 'PRIMARY') AS total_indexes,
    (SELECT SUM(table_rows) FROM information_schema.tables 
     WHERE table_schema = 'ecommerce_analytics') AS total_rows,
    (SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) 
     FROM information_schema.tables 
     WHERE table_schema = 'ecommerce_analytics') AS total_size_mb,
    (SELECT COUNT(*) FROM information_schema.tables 
     WHERE table_schema = 'ecommerce_analytics' 
     AND data_free / NULLIF(data_length, 0) > 0.2) AS tables_needing_optimization,
    'Run ANALYZE TABLE and OPTIMIZE TABLE periodically' AS maintenance_tip;

-- ========================================
-- 14. REAL-TIME MONITORING QUERY
-- Current system performance snapshot
-- ========================================
SELECT 
    'Current Performance Snapshot' AS snapshot_type,
    (SELECT COUNT(*) FROM orders 
     WHERE order_date >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 1 HOUR)) AS orders_last_hour,
    (SELECT COUNT(*) FROM orders 
     WHERE order_date >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 24 HOUR)) AS orders_last_24h,
    (SELECT COUNT(*) FROM customers 
     WHERE created_at >= DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 24 HOUR)) AS new_customers_24h,
    (SELECT COUNT(*) FROM reviews 
     WHERE status = 'pending') AS pending_reviews,
    (SELECT COUNT(*) FROM products 
     WHERE status = 'out_of_stock') AS out_of_stock_products,
    (SELECT COUNT(*) FROM returns 
     WHERE status = 'requested') AS pending_returns,
    CURRENT_TIMESTAMP AS snapshot_time;

-- Display completion message
SELECT 'Performance Metrics and Optimization Analysis Complete' AS Status,
       'Review recommendations and implement suggested indexes/optimizations' AS Next_Steps;