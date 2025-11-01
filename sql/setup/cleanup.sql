-- ============================================================
-- COMPLETE DATABASE CLEANUP & RESTART KIT
-- MariaDB Compatible Version
-- ============================================================
-- Purpose: Clean slate for ecommerce_analytics database
-- Usage: Run this FIRST before re-executing your setup scripts
-- ============================================================

-- ============================================================
-- STEP 1: COMPLETE CLEANUP
-- ============================================================

USE ecommerce_analytics;

-- Disable foreign key checks for cleanup
SET FOREIGN_KEY_CHECKS = 0;

-- ============================================================
-- Drop all triggers
-- ============================================================
DROP TRIGGER IF EXISTS trg_audit_customer_changes;
DROP TRIGGER IF EXISTS trg_audit_order_changes;
DROP TRIGGER IF EXISTS trg_audit_product_changes;
DROP TRIGGER IF EXISTS trg_update_customer_updated_at;
DROP TRIGGER IF EXISTS trg_update_order_updated_at;
DROP TRIGGER IF EXISTS trg_update_product_updated_at;

-- ============================================================
-- Drop all events
-- ============================================================
DROP EVENT IF EXISTS evt_daily_cleanup;
DROP EVENT IF EXISTS evt_refresh_materialized_views;
DROP EVENT IF EXISTS evt_archive_old_data;
DROP EVENT IF EXISTS evt_update_statistics;

-- ============================================================
-- Drop all procedures
-- ============================================================
DROP PROCEDURE IF EXISTS sp_escalate_alert;
DROP PROCEDURE IF EXISTS sp_start_batch_job;
DROP PROCEDURE IF EXISTS sp_complete_batch_job;
DROP PROCEDURE IF EXISTS sp_log_data_quality_issue;
DROP PROCEDURE IF EXISTS sp_etl_customer_metrics;
DROP PROCEDURE IF EXISTS sp_etl_product_performance;
DROP PROCEDURE IF EXISTS sp_etl_data_quality_check;
DROP PROCEDURE IF EXISTS sp_etl_cleanup_old_data;
DROP PROCEDURE IF EXISTS sp_run_daily_etl_pipeline;
DROP PROCEDURE IF EXISTS sp_transform_normalize_customers;
DROP PROCEDURE IF EXISTS sp_transform_enrich_orders;
DROP PROCEDURE IF EXISTS sp_validate_data_quality;
DROP PROCEDURE IF EXISTS sp_escalate_incident;
DROP PROCEDURE IF EXISTS sp_auto_escalate_incidents;
DROP PROCEDURE IF EXISTS sp_check_critical_thresholds;
DROP PROCEDURE IF EXISTS sp_send_notification;
DROP PROCEDURE IF EXISTS sp_resolve_incident;
DROP PROCEDURE IF EXISTS sp_generate_escalation_report;
DROP PROCEDURE IF EXISTS sp_register_integration;
DROP PROCEDURE IF EXISTS sp_import_data;
DROP PROCEDURE IF EXISTS sp_export_data;
DROP PROCEDURE IF EXISTS sp_log_api_request;
DROP PROCEDURE IF EXISTS sp_sync_customer_to_crm;
DROP PROCEDURE IF EXISTS sp_sync_order_to_fulfillment;
DROP PROCEDURE IF EXISTS sp_cleanup_integration_data;
DROP PROCEDURE IF EXISTS sp_queue_notification;
DROP PROCEDURE IF EXISTS sp_process_notification_queue;
DROP PROCEDURE IF EXISTS sp_send_order_notification;
DROP PROCEDURE IF EXISTS sp_send_alert_to_subscribers;
DROP PROCEDURE IF EXISTS sp_send_abandoned_cart_reminders;
DROP PROCEDURE IF EXISTS sp_generate_notification_report;
DROP PROCEDURE IF EXISTS sp_calculate_data_quality_score;
DROP PROCEDURE IF EXISTS sp_assess_customer_data_quality;
DROP PROCEDURE IF EXISTS sp_assess_product_data_quality;
DROP PROCEDURE IF EXISTS sp_assess_order_data_quality;
DROP PROCEDURE IF EXISTS sp_assess_vendor_data_quality;
DROP PROCEDURE IF EXISTS sp_assess_financial_quality;
DROP PROCEDURE IF EXISTS sp_generate_quality_report;
DROP PROCEDURE IF EXISTS sp_monitor_data_quality;
DROP PROCEDURE IF EXISTS sp_start_audit_execution;
DROP PROCEDURE IF EXISTS sp_complete_audit_execution;
DROP PROCEDURE IF EXISTS sp_log_audit_finding;
DROP PROCEDURE IF EXISTS sp_audit_data_integrity;
DROP PROCEDURE IF EXISTS sp_audit_data_quality;
DROP PROCEDURE IF EXISTS sp_audit_referential_integrity;
DROP PROCEDURE IF EXISTS sp_audit_business_rules;
DROP PROCEDURE IF EXISTS sp_audit_performance;
DROP PROCEDURE IF EXISTS sp_run_daily_audits;
DROP PROCEDURE IF EXISTS sp_resolve_audit_finding;
DROP PROCEDURE IF EXISTS sp_cleanup_old_audit_data;
DROP PROCEDURE IF EXISTS sp_export_audit_findings;
DROP PROCEDURE IF EXISTS sp_create_workflow;
DROP PROCEDURE IF EXISTS sp_execute_workflow;
DROP PROCEDURE IF EXISTS sp_add_workflow_step;
DROP PROCEDURE IF EXISTS sp_schedule_task;
DROP PROCEDURE IF EXISTS sp_execute_task;
DROP PROCEDURE IF EXISTS sp_add_to_process_queue;
DROP PROCEDURE IF EXISTS sp_process_queue_items;
DROP PROCEDURE IF EXISTS sp_execute_automation_rules;
DROP PROCEDURE IF EXISTS sp_reset_daily_counters;
DROP PROCEDURE IF EXISTS sp_archive_orders;
DROP PROCEDURE IF EXISTS sp_archive_customers;
DROP PROCEDURE IF EXISTS sp_archive_products;
DROP PROCEDURE IF EXISTS sp_archive_audit_logs;
DROP PROCEDURE IF EXISTS sp_purge_old_archives;
DROP PROCEDURE IF EXISTS sp_restore_from_archive;
DROP PROCEDURE IF EXISTS sp_cleanup_temp_tables;
DROP PROCEDURE IF EXISTS sp_rebuild_fragmented_indexes;
DROP PROCEDURE IF EXISTS sp_analyze_tables;
DROP PROCEDURE IF EXISTS sp_identify_unused_indexes;
DROP PROCEDURE IF EXISTS sp_drop_unused_indexes;
DROP PROCEDURE IF EXISTS sp_auto_index_maintenance;
DROP PROCEDURE IF EXISTS sp_optimize_table;
DROP PROCEDURE IF EXISTS sp_analyze_query_performance;
DROP PROCEDURE IF EXISTS sp_batch_analyze_critical_queries;
DROP PROCEDURE IF EXISTS sp_refresh_metadata;
DROP PROCEDURE IF EXISTS sp_auto_statistics_maintenance;
DROP PROCEDURE IF EXISTS sp_compare_statistics;
DROP PROCEDURE IF EXISTS sp_refresh_customer_summary;
DROP PROCEDURE IF EXISTS sp_refresh_product_summary;
DROP PROCEDURE IF EXISTS sp_refresh_all_materialized_views;
DROP PROCEDURE IF EXISTS sp_restore_full_database;
DROP PROCEDURE IF EXISTS sp_point_in_time_recovery;
DROP PROCEDURE IF EXISTS sp_restore_with_validation;
DROP PROCEDURE IF EXISTS sp_restore_incremental;
DROP PROCEDURE IF EXISTS sp_restore_date_range;
DROP PROCEDURE IF EXISTS sp_emergency_restore;

-- ============================================================
-- Drop all functions
-- ============================================================
DROP FUNCTION IF EXISTS fn_calculate_order_total;
DROP FUNCTION IF EXISTS fn_get_customer_lifetime_value;
DROP FUNCTION IF EXISTS fn_check_inventory_availability;
DROP FUNCTION IF EXISTS fn_validate_email;
DROP FUNCTION IF EXISTS fn_validate_phone;
DROP FUNCTION IF EXISTS fn_check_notification_preference;
DROP FUNCTION IF EXISTS fn_get_external_id;
DROP FUNCTION IF EXISTS fn_check_rate_limit;

-- ============================================================
-- Drop all views
-- ============================================================
DROP VIEW IF EXISTS vw_customer_order_summary;
DROP VIEW IF EXISTS vw_product_performance;
DROP VIEW IF EXISTS vw_daily_sales;
DROP VIEW IF EXISTS vw_inventory_status;
DROP VIEW IF EXISTS vw_customer_metrics;

-- ============================================================
-- Drop all temporary and archived tables
-- ============================================================
DROP TABLE IF EXISTS tmp_customer_segments;
DROP TABLE IF EXISTS tmp_fragmentation_report;
DROP TABLE IF EXISTS tmp_unused_indexes;
DROP TABLE IF EXISTS tmp_stale_stats;
DROP TABLE IF EXISTS temp_customer_segments;

-- Archive tables
DROP TABLE IF EXISTS archive_orders;
DROP TABLE IF EXISTS archive_customers;
DROP TABLE IF EXISTS archive_products;
DROP TABLE IF EXISTS archive_audit_logs;

-- ============================================================
-- Drop all regular tables (in correct order)
-- ============================================================

-- Drop child tables first (have foreign keys)
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS campaign_interactions;
DROP TABLE IF EXISTS customer_addresses;
DROP TABLE IF EXISTS product_images;
DROP TABLE IF EXISTS inventory_transactions;
DROP TABLE IF EXISTS loyalty_transactions;
DROP TABLE IF EXISTS notification_queue;
DROP TABLE IF EXISTS notification_log;
DROP TABLE IF EXISTS integration_sync_log;
DROP TABLE IF EXISTS integration_field_mappings;
DROP TABLE IF EXISTS webhook_events;
DROP TABLE IF EXISTS api_request_log;
DROP TABLE IF EXISTS audit_trail;
DROP TABLE IF EXISTS audit_findings;
DROP TABLE IF EXISTS data_quality_scores;
DROP TABLE IF EXISTS data_quality_violations;
DROP TABLE IF EXISTS batch_job_log;
DROP TABLE IF EXISTS etl_execution_log;
DROP TABLE IF EXISTS workflow_execution_log;
DROP TABLE IF EXISTS workflow_steps;
DROP TABLE IF EXISTS scheduled_tasks;
DROP TABLE IF EXISTS process_queue;
DROP TABLE IF EXISTS automation_rules;
DROP TABLE IF EXISTS alert_subscriptions;
DROP TABLE IF EXISTS alert_log;
DROP TABLE IF EXISTS incidents;

-- Drop intermediate tables
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS vendor_products;
DROP TABLE IF EXISTS loyalty_program;
DROP TABLE IF EXISTS integration_systems;
DROP TABLE IF EXISTS notification_templates;

-- Drop parent tables
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS vendors;
DROP TABLE IF EXISTS campaigns;
DROP TABLE IF EXISTS users;

-- Drop lookup tables last
DROP TABLE IF EXISTS countries;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS order_status;
DROP TABLE IF EXISTS payment_methods;
DROP TABLE IF EXISTS shipping_methods;
DROP TABLE IF EXISTS loyalty_tiers;
DROP TABLE IF EXISTS notification_channels;
DROP TABLE IF EXISTS vendor_types;

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

-- ============================================================
-- STEP 2: VERIFICATION
-- ============================================================

-- Check that all tables are dropped
SELECT 
    COUNT(*) as remaining_tables,
    'Tables' as object_type
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'ecommerce_analytics'

UNION ALL

SELECT 
    COUNT(*) as remaining_objects,
    'Procedures' as object_type
FROM INFORMATION_SCHEMA.ROUTINES 
WHERE ROUTINE_SCHEMA = 'ecommerce_analytics' 
AND ROUTINE_TYPE = 'PROCEDURE'

UNION ALL

SELECT 
    COUNT(*) as remaining_objects,
    'Functions' as object_type
FROM INFORMATION_SCHEMA.ROUTINES 
WHERE ROUTINE_SCHEMA = 'ecommerce_analytics' 
AND ROUTINE_TYPE = 'FUNCTION'

UNION ALL

SELECT 
    COUNT(*) as remaining_objects,
    'Views' as object_type
FROM INFORMATION_SCHEMA.VIEWS 
WHERE TABLE_SCHEMA = 'ecommerce_analytics'

UNION ALL

SELECT 
    COUNT(*) as remaining_objects,
    'Triggers' as object_type
FROM INFORMATION_SCHEMA.TRIGGERS 
WHERE TRIGGER_SCHEMA = 'ecommerce_analytics';

-- ============================================================
-- STEP 3: READY FOR FRESH START
-- ============================================================

SELECT 
    'Database cleaned successfully!' as Status,
    'Ready for setup scripts' as Next_Step,
    NOW() as Timestamp;

-- ============================================================
-- RECOMMENDED EXECUTION ORDER FOR YOUR SETUP SCRIPTS:
-- ============================================================
/*
1.  create_database.sql        (if needed)
2.  create_tables.sql           (with fixes below)
3.  create_indexes.sql          (with fixes below)
4.  create_views.sql
5.  create_functions.sql        (add DROP IF EXISTS)
6.  insert_lookup_data.sql      (fix column names)
7.  insert_sample_vendors.sql
8.  insert_sample_customers.sql
9.  insert_sample_products.sql  (ensure foreign keys exist)
10. insert_sample_orders.sql
11. insert_sample_inventory.sql
12. insert_sample_campaigns.sql
13. insert_sample_reviews.sql

Then proceed with analysis and reporting scripts.
*/

-- ============================================================
-- CRITICAL FIXES NEEDED IN YOUR SCRIPTS:
-- ============================================================

/*
1. CREATE_TABLES.SQL:
   - Add DROP TABLE IF EXISTS before each CREATE TABLE
   - Verify all column definitions match insert scripts
   - Check for missing columns: country_code, category_code, is_active, status

2. CREATE_INDEXES.SQL:
   - Remove duplicate index names
   - Ensure all referenced columns exist
   - Example fix:
     DROP INDEX IF EXISTS idx_customers_status ON customers;
     CREATE INDEX idx_customers_status ON customers(status);

3. CREATE_FUNCTIONS.SQL:
   - Add this before each function:
     DROP FUNCTION IF EXISTS fn_name;
   - Change delimiter:
     DELIMITER $$
     CREATE FUNCTION fn_name(...) RETURNS type
     BEGIN
       -- function body
     END$$
     DELIMITER ;

4. INSERT_LOOKUP_DATA.SQL:
   - Match column names exactly with schema
   - Add INSERT IGNORE or ON DUPLICATE KEY UPDATE
   - Example:
     INSERT IGNORE INTO countries (country_id, country_name) 
     VALUES (1, 'United States');

5. INSERT_SAMPLE_PRODUCTS.SQL:
   - Verify vendor_id exists in vendors table
   - Verify category_id exists in categories table
   - Add error handling:
     INSERT INTO products (...)
     SELECT ... 
     WHERE EXISTS (SELECT 1 FROM vendors WHERE vendor_id = ...)
       AND EXISTS (SELECT 1 FROM categories WHERE category_id = ...);

6. ALL STORED PROCEDURES:
   - Replace system variables (@var) with declared variables (DECLARE var)
   - Add proper error handlers
   - Add DROP PROCEDURE IF EXISTS
   - Template:
     DROP PROCEDURE IF EXISTS sp_name;
     DELIMITER $$
     CREATE PROCEDURE sp_name(IN param1 INT)
     BEGIN
       DECLARE v_local INT DEFAULT 0;
       DECLARE EXIT HANDLER FOR SQLEXCEPTION
       BEGIN
         ROLLBACK;
         -- error handling
       END;
       
       START TRANSACTION;
       -- procedure body
       COMMIT;
     END$$
     DELIMITER ;
*/

-- ============================================================
-- END OF CLEANUP SCRIPT
-- ============================================================