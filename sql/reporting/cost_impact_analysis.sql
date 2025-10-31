-- ========================================
-- COST IMPACT ANALYSIS - E-COMMERCE ANALYTICS
-- Financial Impact of Data Issues & ROI Calculations
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. EXECUTIVE SUMMARY - TOTAL COST OF POOR QUALITY
-- ========================================
SELECT 
    'EXECUTIVE SUMMARY' AS report_section,
    '==================' AS separator;

-- Calculate total cost of poor data quality
SELECT 
    'Total Cost of Poor Quality (Last 90 Days)' AS metric,
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(total_amount) FROM orders 
                  WHERE status = 'cancelled' 
                  AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0) +
        COALESCE((SELECT SUM(refund_amount) FROM returns 
                  WHERE status = 'refunded' 
                  AND created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0) +
        COALESCE((SELECT SUM((p.price - p.cost) * oi.quantity) 
                  FROM order_items oi
                  JOIN orders o ON oi.order_id = o.order_id
                  JOIN products p ON oi.product_id = p.product_id
                  WHERE p.status = 'out_of_stock'
                  AND o.status = 'cancelled'
                  AND o.order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    )) AS total_cost_impact,
    CONCAT(
        ROUND(
            (COALESCE((SELECT SUM(total_amount) FROM orders 
                      WHERE status = 'cancelled' 
                      AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0) +
            COALESCE((SELECT SUM(refund_amount) FROM returns 
                      WHERE status = 'refunded' 
                      AND created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0)) * 100.0 /
            NULLIF((SELECT SUM(total_amount) FROM orders 
                    WHERE order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
            2
        ),
        '%'
    ) AS percentage_of_revenue;

-- ========================================
-- 2. CANCELLED ORDERS COST ANALYSIS
-- ========================================
SELECT 
    '' AS blank_line,
    'CANCELLED ORDERS IMPACT' AS report_section,
    '=======================' AS separator;

-- Detailed cancellation cost breakdown
SELECT 
    'Cancelled Orders Cost Breakdown' AS metric,
    COUNT(*) AS cancelled_order_count,
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS lost_revenue,
    CONCAT('$', FORMAT(AVG(total_amount), 2)) AS avg_cancelled_order_value,
    CONCAT('$', FORMAT(SUM(shipping_cost), 2)) AS wasted_shipping_costs,
    CONCAT('$', FORMAT(
        SUM(total_amount) + SUM(shipping_cost),
        2
    )) AS total_cancellation_cost
FROM orders
WHERE status = 'cancelled'
    AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY);

-- Cancellation reasons and impact (estimated based on payment status)
SELECT 
    'Cancellation Cost by Payment Status' AS metric,
    payment_status,
    COUNT(*) AS order_count,
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS revenue_impact,
    CONCAT(
        ROUND(
            COUNT(*) * 100.0 / 
            (SELECT COUNT(*) FROM orders 
             WHERE status = 'cancelled' 
             AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)),
            2
        ),
        '%'
    ) AS percentage_of_cancellations,
    CASE 
        WHEN payment_status = 'paid' THEN 'High - Refund Processing Cost'
        WHEN payment_status = 'failed' THEN 'Medium - Failed Transaction Cost'
        ELSE 'Low - Pre-payment Cancellation'
    END AS cost_severity
FROM orders
WHERE status = 'cancelled'
    AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY payment_status
ORDER BY SUM(total_amount) DESC;

-- ========================================
-- 3. RETURN & REFUND COST ANALYSIS
-- ========================================
SELECT 
    '' AS blank_line,
    'RETURN & REFUND COSTS' AS report_section,
    '=====================' AS separator;

-- Return processing costs
SELECT 
    'Return Financial Impact' AS metric,
    COUNT(*) AS total_returns,
    CONCAT('$', FORMAT(SUM(refund_amount), 2)) AS total_refunds,
    CONCAT('$', FORMAT(AVG(refund_amount), 2)) AS avg_refund_amount,
    -- Estimate processing cost at $15 per return
    CONCAT('$', FORMAT(COUNT(*) * 15, 2)) AS estimated_processing_cost,
    CONCAT('$', FORMAT(SUM(refund_amount) + (COUNT(*) * 15), 2)) AS total_return_cost
FROM returns
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY);

-- Cost by return reason
SELECT 
    'Cost by Return Reason' AS metric,
    reason,
    COUNT(*) AS return_count,
    CONCAT('$', FORMAT(SUM(refund_amount), 2)) AS total_refund_cost,
    CONCAT('$', FORMAT(AVG(refund_amount), 2)) AS avg_refund,
    CONCAT(
        ROUND(
            COUNT(*) * 100.0 / 
            (SELECT COUNT(*) FROM returns 
             WHERE created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)),
            2
        ),
        '%'
    ) AS percentage_of_returns,
    CASE 
        WHEN reason = 'defective' THEN 'Critical - Quality Control Issue'
        WHEN reason = 'wrong_item' THEN 'Critical - Fulfillment Error'
        WHEN reason = 'not_as_described' THEN 'High - Data Quality Issue'
        WHEN reason = 'changed_mind' THEN 'Medium - Customer Service Issue'
        ELSE 'Variable'
    END AS issue_severity
FROM returns
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY reason
ORDER BY SUM(refund_amount) DESC;

-- ========================================
-- 4. INVENTORY COST ANALYSIS
-- ========================================
SELECT 
    '' AS blank_line,
    'INVENTORY DATA QUALITY COSTS' AS report_section,
    '============================' AS separator;

-- Out of stock cost (lost sales opportunity)
SELECT 
    'Out of Stock Impact' AS metric,
    COUNT(DISTINCT p.product_id) AS out_of_stock_products,
    COUNT(DISTINCT o.order_id) AS affected_orders,
    CONCAT('$', FORMAT(
        SUM((p.price - p.cost) * oi.quantity),
        2
    )) AS lost_profit_margin,
    CONCAT('$', FORMAT(
        SUM(p.price * oi.quantity),
        2
    )) AS lost_revenue_opportunity
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE p.status = 'out_of_stock'
    AND o.status = 'cancelled'
    AND o.order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY);

-- Negative inventory cost (data quality issue)
SELECT 
    'Negative Inventory Issues' AS metric,
    COUNT(*) AS affected_products,
    CONCAT('$', FORMAT(
        SUM(ABS(i.quantity_available) * p.cost),
        2
    )) AS unaccounted_inventory_value,
    'Critical Data Quality Issue' AS risk_level,
    'Immediate Audit Required' AS recommended_action
FROM inventory i
JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_available < 0;

-- Overstocking cost (capital tied up)
SELECT 
    'Overstocking Analysis' AS metric,
    COUNT(*) AS overstocked_items,
    CONCAT('$', FORMAT(
        SUM((i.quantity_on_hand - i.reorder_level * 3) * p.cost),
        2
    )) AS excess_inventory_value,
    -- Assume 10% annual carrying cost, prorated to 90 days
    CONCAT('$', FORMAT(
        SUM((i.quantity_on_hand - i.reorder_level * 3) * p.cost) * 0.10 * (90.0/365),
        2
    )) AS carrying_cost_90_days
FROM inventory i
JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_on_hand > (i.reorder_level * 3)
    AND i.reorder_level > 0;

-- ========================================
-- 5. CUSTOMER DATA QUALITY COSTS
-- ========================================
SELECT 
    '' AS blank_line,
    'CUSTOMER DATA QUALITY COSTS' AS report_section,
    '===========================' AS separator;

-- Incomplete customer data impact
SELECT 
    'Incomplete Customer Data Impact' AS metric,
    COUNT(*) AS customers_with_missing_data,
    -- Estimate $50 cost per incomplete record for marketing inefficiency
    CONCAT('$', FORMAT(COUNT(*) * 50, 2)) AS estimated_marketing_waste,
    CONCAT(
        ROUND(
            COUNT(*) * 100.0 / 
            (SELECT COUNT(*) FROM customers WHERE status = 'active'),
            2
        ),
        '%'
    ) AS percentage_of_active_customers
FROM customers
WHERE status = 'active'
    AND (email IS NULL OR email = '' OR phone IS NULL OR phone = '' OR address_line1 IS NULL);

-- Duplicate customer impact
SELECT 
    'Potential Duplicate Customers' AS metric,
    COUNT(*) AS potential_duplicates,
    -- Estimate $25 cost per duplicate for wasted communications
    CONCAT('$', FORMAT(COUNT(*) * 25, 2)) AS estimated_duplicate_cost
FROM (
    SELECT email, COUNT(*) as count
    FROM customers
    WHERE email IS NOT NULL AND email != ''
    GROUP BY email
    HAVING COUNT(*) > 1
) duplicates;

-- ========================================
-- 6. PAYMENT & TRANSACTION FAILURE COSTS
-- ========================================
SELECT 
    '' AS blank_line,
    'PAYMENT FAILURE COSTS' AS report_section,
    '=====================' AS separator;

-- Failed payment analysis
SELECT 
    'Failed Payment Impact' AS metric,
    COUNT(*) AS failed_payment_count,
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS lost_revenue,
    -- Estimate $5 processing cost per failed transaction
    CONCAT('$', FORMAT(COUNT(*) * 5, 2)) AS transaction_processing_waste,
    CONCAT('$', FORMAT(
        SUM(total_amount) + (COUNT(*) * 5),
        2
    )) AS total_payment_failure_cost,
    CONCAT(
        ROUND(
            COUNT(*) * 100.0 / 
            (SELECT COUNT(*) FROM orders 
             WHERE order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)),
            2
        ),
        '%'
    ) AS failure_rate
FROM orders
WHERE payment_status = 'failed'
    AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY);

-- ========================================
-- 7. MARKETING CAMPAIGN ROI ANALYSIS
-- ========================================
SELECT 
    '' AS blank_line,
    'MARKETING CAMPAIGN ROI' AS report_section,
    '======================' AS separator;

-- Campaign performance and ROI
SELECT 
    'Campaign ROI Analysis' AS metric,
    c.campaign_name,
    c.campaign_type,
    CONCAT('$', FORMAT(SUM(cp.spend), 2)) AS total_spend,
    CONCAT('$', FORMAT(SUM(cp.revenue), 2)) AS total_revenue,
    CONCAT('$', FORMAT(SUM(cp.revenue) - SUM(cp.spend), 2)) AS net_profit,
    CONCAT(
        ROUND(
            ((SUM(cp.revenue) - SUM(cp.spend)) / NULLIF(SUM(cp.spend), 0)) * 100,
            2
        ),
        '%'
    ) AS roi_percentage,
    SUM(cp.conversions) AS total_conversions,
    CONCAT('$', FORMAT(
        SUM(cp.spend) / NULLIF(SUM(cp.conversions), 0),
        2
    )) AS cost_per_conversion,
    CASE 
        WHEN ((SUM(cp.revenue) - SUM(cp.spend)) / NULLIF(SUM(cp.spend), 0)) >= 3.0 THEN 'Excellent ROI'
        WHEN ((SUM(cp.revenue) - SUM(cp.spend)) / NULLIF(SUM(cp.spend), 0)) >= 1.0 THEN 'Good ROI'
        WHEN ((SUM(cp.revenue) - SUM(cp.spend)) / NULLIF(SUM(cp.spend), 0)) >= 0 THEN 'Break Even'
        ELSE 'Negative ROI - Review Campaign'
    END AS roi_assessment
FROM campaigns c
JOIN campaign_performance cp ON c.campaign_id = cp.campaign_id
WHERE cp.report_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY c.campaign_id, c.campaign_name, c.campaign_type
ORDER BY ((SUM(cp.revenue) - SUM(cp.spend)) / NULLIF(SUM(cp.spend), 0)) DESC;

-- ========================================
-- 8. PRODUCT QUALITY COST ANALYSIS
-- ========================================
SELECT 
    '' AS blank_line,
    'PRODUCT QUALITY COSTS' AS report_section,
    '=====================' AS separator;

-- Products with high return rates
SELECT 
    'High Return Rate Products' AS metric,
    p.product_name,
    pc.category_name,
    COUNT(DISTINCT r.return_id) AS return_count,
    COUNT(DISTINCT oi.order_item_id) AS total_sales,
    CONCAT(
        ROUND(
            COUNT(DISTINCT r.return_id) * 100.0 / 
            NULLIF(COUNT(DISTINCT oi.order_item_id), 0),
            2
        ),
        '%'
    ) AS return_rate,
    CONCAT('$', FORMAT(SUM(r.refund_amount), 2)) AS total_refund_cost,
    CASE 
        WHEN COUNT(DISTINCT r.return_id) * 100.0 / NULLIF(COUNT(DISTINCT oi.order_item_id), 0) > 15 
        THEN 'Critical - Investigate Quality Issues'
        WHEN COUNT(DISTINCT r.return_id) * 100.0 / NULLIF(COUNT(DISTINCT oi.order_item_id), 0) > 10 
        THEN 'High - Review Product'
        WHEN COUNT(DISTINCT r.return_id) * 100.0 / NULLIF(COUNT(DISTINCT oi.order_item_id), 0) > 5 
        THEN 'Moderate - Monitor'
        ELSE 'Normal'
    END AS quality_status
FROM products p
LEFT JOIN product_categories pc ON p.category_id = pc.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
WHERE oi.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY p.product_id, p.product_name, pc.category_name
HAVING COUNT(DISTINCT r.return_id) > 0
ORDER BY return_rate DESC
LIMIT 10;

-- ========================================
-- 9. VENDOR QUALITY COST IMPACT
-- ========================================
SELECT 
    '' AS blank_line,
    'VENDOR QUALITY COSTS' AS report_section,
    '====================' AS separator;

-- Vendor-related quality costs
SELECT 
    'Vendor Quality Impact' AS metric,
    v.vendor_name,
    v.rating,
    COUNT(DISTINCT p.product_id) AS products_supplied,
    COUNT(DISTINCT r.return_id) AS returns_from_vendor,
    CONCAT('$', FORMAT(SUM(r.refund_amount), 2)) AS total_return_cost,
    CONCAT('$', FORMAT(
        AVG(vc.cost_per_unit) * SUM(r.refund_amount) / NULLIF(AVG(p.price), 0),
        2
    )) AS estimated_cogs_impact,
    CASE 
        WHEN v.rating < 3.0 THEN 'High Risk - Consider Alternative Vendors'
        WHEN v.rating < 4.0 THEN 'Medium Risk - Monitor Closely'
        ELSE 'Low Risk - Acceptable Performance'
    END AS vendor_risk_level
FROM vendors v
LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
LEFT JOIN products p ON vc.product_id = p.product_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN returns r ON oi.order_item_id = r.order_item_id
WHERE r.created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
    AND vc.status = 'active'
GROUP BY v.vendor_id, v.vendor_name, v.rating
HAVING COUNT(DISTINCT r.return_id) > 0
ORDER BY total_return_cost DESC;

-- ========================================
-- 10. COMPREHENSIVE COST SUMMARY & ROI OPPORTUNITIES
-- ========================================
SELECT 
    '' AS blank_line,
    'COMPREHENSIVE COST SUMMARY' AS report_section,
    '==========================' AS separator;

-- Total cost breakdown
SELECT 
    'Cost Category Summary' AS analysis_type,
    'Cancelled Orders' AS cost_category,
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(total_amount) + SUM(shipping_cost) 
                  FROM orders 
                  WHERE status = 'cancelled' 
                  AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    )) AS cost_amount,
    'High' AS priority
UNION ALL
SELECT 
    'Cost Category Summary',
    'Returns & Refunds',
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(refund_amount) + (COUNT(*) * 15) 
                  FROM returns 
                  WHERE created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    )),
    'High'
UNION ALL
SELECT 
    'Cost Category Summary',
    'Payment Failures',
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(total_amount) + (COUNT(*) * 5) 
                  FROM orders 
                  WHERE payment_status = 'failed' 
                  AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    )),
    'Medium'
UNION ALL
SELECT 
    'Cost Category Summary',
    'Inventory Issues',
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM((quantity_on_hand - reorder_level * 3) * p.cost) * 0.10 * (90.0/365)
                  FROM inventory i
                  JOIN products p ON i.product_id = p.product_id
                  WHERE i.quantity_on_hand > (i.reorder_level * 3) 
                  AND i.reorder_level > 0), 0),
        2
    )),
    'Medium'
UNION ALL
SELECT 
    'Cost Category Summary',
    'Customer Data Quality',
    CONCAT('$', FORMAT(
        COALESCE((SELECT COUNT(*) * 50 
                  FROM customers 
                  WHERE status = 'active' 
                  AND (email IS NULL OR phone IS NULL OR address_line1 IS NULL)), 0),
        2
    )),
    'Low';

-- ROI improvement opportunities
SELECT 
    '' AS blank_line,
    'ROI IMPROVEMENT OPPORTUNITIES' AS report_section,
    '=============================' AS separator;

SELECT 
    'Recommended Actions' AS category,
    'Reduce Cancellation Rate by 25%' AS improvement_opportunity,
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(total_amount) * 0.25 
                  FROM orders 
                  WHERE status = 'cancelled' 
                  AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    )) AS potential_savings_90_days,
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(total_amount) * 0.25 * 4 
                  FROM orders 
                  WHERE status = 'cancelled' 
                  AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    )) AS annual_savings_projection
UNION ALL
SELECT 
    'Recommended Actions',
    'Reduce Return Rate by 20%',
    CONCAT('$', FORMAT(
        COALESCE((SELECT (SUM(refund_amount) + COUNT(*) * 15) * 0.20 
                  FROM returns 
                  WHERE created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    )),
    CONCAT('$', FORMAT(
        COALESCE((SELECT (SUM(refund_amount) + COUNT(*) * 15) * 0.20 * 4 
                  FROM returns 
                  WHERE created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    ))
UNION ALL
SELECT 
    'Recommended Actions',
    'Improve Payment Success Rate by 15%',
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(total_amount) * 0.15 
                  FROM orders 
                  WHERE payment_status = 'failed' 
                  AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    )),
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM(total_amount) * 0.15 * 4 
                  FROM orders 
                  WHERE payment_status = 'failed' 
                  AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)), 0),
        2
    ))
UNION ALL
SELECT 
    'Recommended Actions',
    'Optimize Inventory Management',
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM((quantity_on_hand - reorder_level * 3) * p.cost) * 0.10 * (90.0/365)
                  FROM inventory i
                  JOIN products p ON i.product_id = p.product_id
                  WHERE i.quantity_on_hand > (i.reorder_level * 3)), 0),
        2
    )),
    CONCAT('$', FORMAT(
        COALESCE((SELECT SUM((quantity_on_hand - reorder_level * 3) * p.cost) * 0.10
                  FROM inventory i
                  JOIN products p ON i.product_id = p.product_id
                  WHERE i.quantity_on_hand > (i.reorder_level * 3)), 0),
        2
    ));

-- Display completion message
SELECT 
    '' AS blank_line,
    'COST IMPACT ANALYSIS COMPLETE' AS status,
    'Review action items for maximum ROI' AS recommendation,
    NOW() AS generated_at;