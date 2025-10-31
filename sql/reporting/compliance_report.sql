-- ========================================
-- COMPLIANCE REPORT - E-COMMERCE ANALYTICS
-- Regulatory Compliance, Scores & Audit Readiness
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. DATA QUALITY & COMPLETENESS SCORE
-- ========================================
SELECT 
    'DATA QUALITY ASSESSMENT' AS report_section,
    '========================' AS separator;

-- Customer data completeness
SELECT 
    'Customer Data Completeness' AS metric,
    CONCAT(
        ROUND(
            (COUNT(CASE WHEN email IS NOT NULL AND email != '' THEN 1 END) * 100.0 / COUNT(*)), 
            2
        ), 
        '%'
    ) AS email_completeness,
    CONCAT(
        ROUND(
            (COUNT(CASE WHEN phone IS NOT NULL AND phone != '' THEN 1 END) * 100.0 / COUNT(*)), 
            2
        ), 
        '%'
    ) AS phone_completeness,
    CONCAT(
        ROUND(
            (COUNT(CASE WHEN address_line1 IS NOT NULL THEN 1 END) * 100.0 / COUNT(*)), 
            2
        ), 
        '%'
    ) AS address_completeness,
    CASE 
        WHEN AVG(
            CASE WHEN email IS NOT NULL AND phone IS NOT NULL AND address_line1 IS NOT NULL 
            THEN 100 ELSE 0 END
        ) >= 95 THEN 'EXCELLENT'
        WHEN AVG(
            CASE WHEN email IS NOT NULL AND phone IS NOT NULL AND address_line1 IS NOT NULL 
            THEN 100 ELSE 0 END
        ) >= 80 THEN 'GOOD'
        WHEN AVG(
            CASE WHEN email IS NOT NULL AND phone IS NOT NULL AND address_line1 IS NOT NULL 
            THEN 100 ELSE 0 END
        ) >= 60 THEN 'NEEDS IMPROVEMENT'
        ELSE 'CRITICAL'
    END AS compliance_status
FROM customers
WHERE status = 'active';

-- ========================================
-- 2. GDPR COMPLIANCE CHECK
-- ========================================
SELECT 
    '' AS blank_line,
    'GDPR COMPLIANCE METRICS' AS report_section,
    '=======================' AS separator;

-- Data retention compliance
SELECT 
    'Data Retention Analysis' AS metric,
    COUNT(*) AS total_customers,
    COUNT(CASE WHEN DATEDIFF(NOW(), created_at) > 2555 THEN 1 END) AS customers_7_years_old,
    COUNT(CASE WHEN status = 'inactive' AND DATEDIFF(NOW(), updated_at) > 730 THEN 1 END) AS inactive_2plus_years,
    CASE 
        WHEN COUNT(CASE WHEN status = 'inactive' AND DATEDIFF(NOW(), updated_at) > 730 THEN 1 END) = 0 
        THEN 'COMPLIANT' 
        ELSE 'REVIEW REQUIRED' 
    END AS retention_compliance
FROM customers;

-- Customer consent tracking (simulated - assumes consent is implicit in active status)
SELECT 
    'Customer Consent Status' AS metric,
    status,
    COUNT(*) AS customer_count,
    CONCAT(ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2), '%') AS percentage,
    CASE 
        WHEN status = 'active' THEN 'Active Consent'
        WHEN status = 'inactive' THEN 'Consent Review Needed'
        ELSE 'No Consent'
    END AS consent_status
FROM customers
GROUP BY status
ORDER BY customer_count DESC;

-- ========================================
-- 3. PCI DSS COMPLIANCE - PAYMENT SECURITY
-- ========================================
SELECT 
    '' AS blank_line,
    'PCI DSS COMPLIANCE' AS report_section,
    '==================' AS separator;

-- Payment method security check
SELECT 
    'Payment Security Assessment' AS metric,
    COUNT(*) AS total_payment_methods,
    COUNT(CASE WHEN card_last_four IS NOT NULL AND LENGTH(card_last_four) = 4 THEN 1 END) AS properly_masked_cards,
    CONCAT(
        ROUND(
            COUNT(CASE WHEN card_last_four IS NOT NULL AND LENGTH(card_last_four) = 4 THEN 1 END) * 100.0 / 
            NULLIF(COUNT(*), 0), 
            2
        ), 
        '%'
    ) AS masking_compliance,
    CASE 
        WHEN COUNT(CASE WHEN card_last_four IS NOT NULL AND LENGTH(card_last_four) = 4 THEN 1 END) = COUNT(*) 
        THEN 'COMPLIANT' 
        ELSE 'NON-COMPLIANT' 
    END AS pci_status
FROM payment_methods
WHERE payment_type IN ('credit_card', 'debit_card');

-- Payment transaction security
SELECT 
    'Payment Transaction Analysis' AS metric,
    COUNT(*) AS total_orders,
    COUNT(CASE WHEN payment_status = 'paid' THEN 1 END) AS successful_payments,
    COUNT(CASE WHEN payment_status = 'failed' THEN 1 END) AS failed_payments,
    COUNT(CASE WHEN payment_status = 'refunded' THEN 1 END) AS refunded_payments,
    CONCAT(
        ROUND(
            COUNT(CASE WHEN payment_status IN ('paid', 'refunded') THEN 1 END) * 100.0 / 
            NULLIF(COUNT(*), 0), 
            2
        ), 
        '%'
    ) AS transaction_success_rate
FROM orders
WHERE order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY);

-- ========================================
-- 4. FINANCIAL COMPLIANCE METRICS
-- ========================================
SELECT 
    '' AS blank_line,
    'FINANCIAL COMPLIANCE' AS report_section,
    '====================' AS separator;

-- Revenue recognition compliance
SELECT 
    'Revenue Recognition Status' AS metric,
    status AS order_status,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_revenue,
    CASE 
        WHEN status IN ('delivered', 'cancelled') THEN 'Revenue Recognized'
        WHEN status IN ('shipped', 'processing') THEN 'Revenue Pending'
        ELSE 'Revenue Deferred'
    END AS recognition_status
FROM orders
WHERE order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY status, recognition_status
ORDER BY total_revenue DESC;

-- Tax compliance check
SELECT 
    'Tax Collection Analysis' AS metric,
    COUNT(*) AS total_orders,
    COUNT(CASE WHEN tax_amount > 0 THEN 1 END) AS orders_with_tax,
    CONCAT(
        ROUND(
            COUNT(CASE WHEN tax_amount > 0 THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 
            2
        ), 
        '%'
    ) AS tax_collection_rate,
    SUM(tax_amount) AS total_tax_collected,
    CASE 
        WHEN COUNT(CASE WHEN tax_amount > 0 THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0) >= 95 
        THEN 'COMPLIANT' 
        ELSE 'REVIEW REQUIRED' 
    END AS tax_compliance
FROM orders
WHERE order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)
    AND status != 'cancelled';

-- ========================================
-- 5. RETURN & REFUND COMPLIANCE
-- ========================================
SELECT 
    '' AS blank_line,
    'RETURN & REFUND COMPLIANCE' AS report_section,
    '===========================' AS separator;

-- Return processing timeliness
SELECT 
    'Return Processing Metrics' AS metric,
    status,
    COUNT(*) AS return_count,
    AVG(DATEDIFF(updated_at, created_at)) AS avg_processing_days,
    CASE 
        WHEN status = 'refunded' AND AVG(DATEDIFF(updated_at, created_at)) <= 14 THEN 'COMPLIANT'
        WHEN status = 'refunded' AND AVG(DATEDIFF(updated_at, created_at)) > 14 THEN 'DELAYED'
        ELSE 'IN PROGRESS'
    END AS processing_compliance
FROM returns
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY)
GROUP BY status
ORDER BY return_count DESC;

-- Refund accuracy
SELECT 
    'Refund Accuracy Check' AS metric,
    COUNT(*) AS total_refunds,
    SUM(refund_amount) AS total_refund_amount,
    AVG(refund_amount) AS avg_refund_amount,
    COUNT(CASE WHEN refund_amount > 0 THEN 1 END) AS processed_refunds,
    CONCAT(
        ROUND(
            COUNT(CASE WHEN refund_amount > 0 THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0), 
            2
        ), 
        '%'
    ) AS refund_completion_rate
FROM returns
WHERE status IN ('refunded', 'approved')
    AND created_at >= DATE_SUB(NOW(), INTERVAL 90 DAY);

-- ========================================
-- 6. INVENTORY COMPLIANCE
-- ========================================
SELECT 
    '' AS blank_line,
    'INVENTORY COMPLIANCE' AS report_section,
    '====================' AS separator;

-- Inventory accuracy
SELECT 
    'Inventory Accuracy Metrics' AS metric,
    COUNT(*) AS total_products,
    COUNT(CASE WHEN quantity_available < 0 THEN 1 END) AS negative_inventory_items,
    COUNT(CASE WHEN quantity_available < reorder_level THEN 1 END) AS items_below_reorder,
    CONCAT(
        ROUND(
            (COUNT(*) - COUNT(CASE WHEN quantity_available < 0 THEN 1 END)) * 100.0 / 
            NULLIF(COUNT(*), 0), 
            2
        ), 
        '%'
    ) AS inventory_accuracy,
    CASE 
        WHEN COUNT(CASE WHEN quantity_available < 0 THEN 1 END) = 0 THEN 'COMPLIANT'
        ELSE 'AUDIT REQUIRED'
    END AS compliance_status
FROM inventory;

-- ========================================
-- 7. VENDOR COMPLIANCE SCORING
-- ========================================
SELECT 
    '' AS blank_line,
    'VENDOR COMPLIANCE' AS report_section,
    '==================' AS separator;

-- Vendor performance and compliance
SELECT 
    'Vendor Compliance Scores' AS metric,
    v.vendor_name,
    v.status,
    v.rating,
    COUNT(vc.contract_id) AS active_contracts,
    COUNT(CASE WHEN vc.status = 'expired' THEN 1 END) AS expired_contracts,
    CASE 
        WHEN v.rating >= 4.5 AND v.status = 'active' THEN 'PREFERRED'
        WHEN v.rating >= 3.5 AND v.status = 'active' THEN 'APPROVED'
        WHEN v.rating < 3.5 OR v.status != 'active' THEN 'REVIEW REQUIRED'
        ELSE 'UNRATED'
    END AS vendor_compliance_tier
FROM vendors v
LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
GROUP BY v.vendor_id, v.vendor_name, v.status, v.rating
HAVING active_contracts > 0
ORDER BY v.rating DESC;

-- ========================================
-- 8. OVERALL COMPLIANCE SCORECARD
-- ========================================
SELECT 
    '' AS blank_line,
    'OVERALL COMPLIANCE SCORECARD' AS report_section,
    '============================' AS separator;

-- Comprehensive compliance score
SELECT 
    'Overall Compliance Score' AS metric,
    ROUND(
        (
            -- Data Quality (25%)
            (SELECT COUNT(CASE WHEN email IS NOT NULL AND phone IS NOT NULL THEN 1 END) * 25.0 / COUNT(*) 
             FROM customers WHERE status = 'active') +
            
            -- Payment Security (25%)
            (SELECT COUNT(CASE WHEN LENGTH(card_last_four) = 4 THEN 1 END) * 25.0 / NULLIF(COUNT(*), 0) 
             FROM payment_methods WHERE payment_type IN ('credit_card', 'debit_card')) +
            
            -- Tax Compliance (25%)
            (SELECT COUNT(CASE WHEN tax_amount > 0 THEN 1 END) * 25.0 / NULLIF(COUNT(*), 0) 
             FROM orders WHERE status != 'cancelled' AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)) +
            
            -- Inventory Accuracy (25%)
            (SELECT (COUNT(*) - COUNT(CASE WHEN quantity_available < 0 THEN 1 END)) * 25.0 / NULLIF(COUNT(*), 0) 
             FROM inventory)
        ), 
        2
    ) AS compliance_score_percentage,
    CASE 
        WHEN ROUND(
            (
                (SELECT COUNT(CASE WHEN email IS NOT NULL AND phone IS NOT NULL THEN 1 END) * 25.0 / COUNT(*) 
                 FROM customers WHERE status = 'active') +
                (SELECT COUNT(CASE WHEN LENGTH(card_last_four) = 4 THEN 1 END) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM payment_methods WHERE payment_type IN ('credit_card', 'debit_card')) +
                (SELECT COUNT(CASE WHEN tax_amount > 0 THEN 1 END) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM orders WHERE status != 'cancelled' AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)) +
                (SELECT (COUNT(*) - COUNT(CASE WHEN quantity_available < 0 THEN 1 END)) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM inventory)
            ), 
            2
        ) >= 95 THEN 'EXCELLENT - AUDIT READY'
        WHEN ROUND(
            (
                (SELECT COUNT(CASE WHEN email IS NOT NULL AND phone IS NOT NULL THEN 1 END) * 25.0 / COUNT(*) 
                 FROM customers WHERE status = 'active') +
                (SELECT COUNT(CASE WHEN LENGTH(card_last_four) = 4 THEN 1 END) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM payment_methods WHERE payment_type IN ('credit_card', 'debit_card')) +
                (SELECT COUNT(CASE WHEN tax_amount > 0 THEN 1 END) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM orders WHERE status != 'cancelled' AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)) +
                (SELECT (COUNT(*) - COUNT(CASE WHEN quantity_available < 0 THEN 1 END)) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM inventory)
            ), 
            2
        ) >= 85 THEN 'GOOD - MINOR IMPROVEMENTS NEEDED'
        WHEN ROUND(
            (
                (SELECT COUNT(CASE WHEN email IS NOT NULL AND phone IS NOT NULL THEN 1 END) * 25.0 / COUNT(*) 
                 FROM customers WHERE status = 'active') +
                (SELECT COUNT(CASE WHEN LENGTH(card_last_four) = 4 THEN 1 END) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM payment_methods WHERE payment_type IN ('credit_card', 'debit_card')) +
                (SELECT COUNT(CASE WHEN tax_amount > 0 THEN 1 END) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM orders WHERE status != 'cancelled' AND order_date >= DATE_SUB(NOW(), INTERVAL 90 DAY)) +
                (SELECT (COUNT(*) - COUNT(CASE WHEN quantity_available < 0 THEN 1 END)) * 25.0 / NULLIF(COUNT(*), 0) 
                 FROM inventory)
            ), 
            2
        ) >= 70 THEN 'FAIR - SIGNIFICANT IMPROVEMENTS REQUIRED'
        ELSE 'CRITICAL - IMMEDIATE ACTION REQUIRED'
    END AS audit_readiness;

-- ========================================
-- 9. AUDIT TRAIL RECOMMENDATIONS
-- ========================================
SELECT 
    '' AS blank_line,
    'AUDIT RECOMMENDATIONS' AS report_section,
    '=====================' AS separator;

SELECT 
    'Critical Action Items' AS category,
    CONCAT(
        CASE WHEN (SELECT COUNT(*) FROM customers WHERE status = 'inactive' AND DATEDIFF(NOW(), updated_at) > 730) > 0
        THEN '1. Review and archive inactive customers (GDPR compliance)\n' ELSE '' END,
        CASE WHEN (SELECT COUNT(*) FROM inventory WHERE quantity_available < 0) > 0
        THEN '2. Correct negative inventory values\n' ELSE '' END,
        CASE WHEN (SELECT AVG(DATEDIFF(updated_at, created_at)) FROM returns WHERE status = 'refunded') > 14
        THEN '3. Improve return processing time\n' ELSE '' END,
        CASE WHEN (SELECT COUNT(*) FROM orders WHERE status != 'cancelled' AND tax_amount = 0 
                   AND order_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)) > 0
        THEN '4. Review tax collection on recent orders\n' ELSE '' END,
        CASE WHEN (SELECT COUNT(*) FROM vendors WHERE status = 'active' AND rating < 3.5) > 0
        THEN '5. Review low-rated vendor relationships\n' ELSE '' END
    ) AS recommendations;

-- Display completion message
SELECT 
    '' AS blank_line,
    'COMPLIANCE REPORT COMPLETE' AS status,
    NOW() AS generated_at;