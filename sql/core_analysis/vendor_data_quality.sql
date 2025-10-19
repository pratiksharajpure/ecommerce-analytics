-- ========================================
-- VENDOR_DATA_QUALITY.SQL
-- Data Quality Check: Vendor Data Validation
-- Path: sql/core_analysis/vendor_data_quality.sql
-- Validates vendor information and contract integrity
-- ========================================

USE ecommerce_analytics;

-- ========================================
-- 1. Vendors with Missing Contact Information
-- ========================================
SELECT 
    v.vendor_id,
    v.vendor_name,
    v.contact_person,
    v.email,
    v.phone,
    v.rating,
    v.status,
    COUNT(DISTINCT vc.contract_id) AS active_contracts,
    CASE 
        WHEN (v.email IS NULL OR v.email = '') AND (v.phone IS NULL OR v.phone = '') THEN 'No contact information'
        WHEN (v.email IS NULL OR v.email = '') THEN 'Missing email'
        WHEN (v.phone IS NULL OR v.phone = '') THEN 'Missing phone'
        WHEN (v.contact_person IS NULL OR v.contact_person = '') THEN 'Missing contact person'
    END AS issue_type
FROM vendors v
LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id AND vc.status = 'active'
WHERE v.status = 'active'
    AND (
        (v.email IS NULL OR v.email = '') OR
        (v.phone IS NULL OR v.phone = '') OR
        (v.contact_person IS NULL OR v.contact_person = '')
    )
GROUP BY v.vendor_id, v.vendor_name, v.contact_person, v.email, v.phone, v.rating, v.status
ORDER BY active_contracts DESC;

-- ========================================
-- 2. Vendors with Missing Address Information
-- ========================================
SELECT 
    v.vendor_id,
    v.vendor_name,
    v.address,
    v.city,
    v.state,
    v.zip_code,
    v.rating,
    v.status,
    COUNT(DISTINCT vc.contract_id) AS total_contracts,
    'Incomplete address' AS issue_type
FROM vendors v
LEFT JOIN vendor_contracts vc ON v.vendor_id = vc.vendor_id
WHERE v.status = 'active'
    AND (
        (v.address IS NULL OR v.address = '') OR
        (v.city IS NULL OR v.city = '') OR
        (v.state IS NULL OR v.state = '') OR
        (v.zip_code IS NULL OR v.zip_code = '')
    )
GROUP BY v.vendor_id, v.vendor_name, v.address, v.city, v.state, v.zip_code, v.rating, v.status
ORDER BY total_contracts DESC;

-- ========================================
-- 3. Vendor Contracts Without Products
-- ========================================
SELECT 
    vc.contract_id,
    vc.vendor_id,
    v.vendor_name,
    vc.product_id AS invalid_product_id,
    vc.cost_per_unit,
    vc.start_date,
    vc.end_date,
    vc.status,
    'Contract references deleted product' AS issue_type
FROM vendor_contracts vc
INNER JOIN vendors v ON vc.vendor_id = v.vendor_id
LEFT JOIN products p ON vc.product_id = p.product_id
WHERE p.product_id IS NULL
ORDER BY vc.cost_per_unit DESC;

-- ========================================
-- 4. Active Contracts with Expired Dates
-- ========================================
SELECT 
    vc.contract_id,
    vc.vendor_id,
    v.vendor_name,
    v.email,
    v.phone,
    vc.product_id,
    p.product_name,
    vc.start_date,
    vc.end_date,
    DATEDIFF(CURRENT_DATE, vc.end_date) AS days_expired,
    vc.cost_per_unit,
    vc.status,
    'Active contract past end date' AS issue_type
FROM vendor_contracts vc
INNER JOIN vendors v ON vc.vendor_id = v.vendor_id
LEFT JOIN products p ON vc.product_id = p.product_id
WHERE vc.status = 'active'