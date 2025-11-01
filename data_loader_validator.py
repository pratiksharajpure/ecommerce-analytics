"""
E-commerce Data Loader with Validation
Handles complete directory structure with 59 CSV files
"""

import pandas as pd
import pymysql
from sqlalchemy import create_engine, text
import os
from pathlib import Path

class DataValidator:
    """Validates foreign key relationships before loading"""
    
    def __init__(self, engine):
        self.engine = engine
        self.validation_errors = []
    
    def get_existing_ids(self, table, id_column):
        """Get all existing IDs from a table"""
        query = f"SELECT DISTINCT {id_column} FROM {table} WHERE {id_column} IS NOT NULL"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                return set(row[0] for row in result)
        except:
            return set()
    
    def validate_foreign_keys(self, df, column, parent_table, parent_column, csv_name):
        """Validate that all foreign key values exist in parent table"""
        if column not in df.columns:
            return True
        
        # Get non-null foreign key values from CSV
        fk_values = set(df[column].dropna().unique())
        
        if not fk_values:
            return True
        
        # Get existing parent IDs
        existing_ids = self.get_existing_ids(parent_table, parent_column)
        
        # Find missing references
        missing = fk_values - existing_ids
        
        if missing:
            error_msg = f"❌ {csv_name}: {len(missing)} missing {column} references"
            print(f"   Missing IDs in {parent_table}: {sorted(list(missing))[:10]}...")
            self.validation_errors.append(error_msg)
            return False
        
        print(f"   ✅ All {column} references valid ({len(fk_values)} unique values)")
        return True


class DataLoader:
    """Loads CSV data into MySQL database with validation"""
    
    def __init__(self, host, user, password, database, csv_base_directory):
        self.connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        self.engine = create_engine(self.connection_string)
        self.csv_base_dir = Path(csv_base_directory)
        self.validator = DataValidator(self.engine)
        
        # Define loading order with foreign key dependencies
        self.load_phases = {
            'core_data': [
                {'csv': 'customers', 'table': 'customers', 'fk_checks': []},
                {'csv': 'product_categories', 'table': 'product_categories', 'fk_checks': []},
                {'csv': 'vendors', 'table': 'vendors', 'fk_checks': []},
                
                {'csv': 'products', 'table': 'products', 'fk_checks': [
                    ('category_id', 'product_categories', 'category_id')
                ]},
                {'csv': 'shipping_addresses', 'table': 'shipping_addresses', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
                {'csv': 'payment_methods', 'table': 'payment_methods', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
                
                {'csv': 'orders', 'table': 'orders', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id'),
                    ('shipping_address_id', 'shipping_addresses', 'address_id')
                ]},
                
                {'csv': 'order_items', 'table': 'order_items', 'fk_checks': [
                    ('order_id', 'orders', 'order_id'),
                    ('product_id', 'products', 'product_id')
                ]},
                
                {'csv': 'inventory', 'table': 'inventory', 'fk_checks': [
                    ('product_id', 'products', 'product_id')
                ]},
                {'csv': 'vendor_contracts', 'table': 'vendor_contracts', 'fk_checks': [
                    ('vendor_id', 'vendors', 'vendor_id'),
                    ('product_id', 'products', 'product_id')
                ]},
                
                # Quality check tables
                {'csv': 'customers_duplicates', 'table': 'customers_duplicates', 'fk_checks': []},
                {'csv': 'customers_invalid_emails', 'table': 'customers_invalid_emails', 'fk_checks': []},
                {'csv': 'products_missing_info', 'table': 'products_missing_info', 'fk_checks': []},
                {'csv': 'orders_invalid', 'table': 'orders_invalid', 'fk_checks': []},
                {'csv': 'inventory_discrepancies', 'table': 'inventory_discrepancies', 'fk_checks': []},
            ],
            
            'marketing_data': [
                {'csv': 'campaigns', 'table': 'campaigns', 'fk_checks': []},
                {'csv': 'campaign_performance', 'table': 'campaign_performance', 'fk_checks': [
                    ('campaign_id', 'campaigns', 'campaign_id')
                ]},
                {'csv': 'email_campaigns', 'table': 'email_campaigns', 'fk_checks': []},
                {'csv': 'social_media_campaigns', 'table': 'social_media_campaigns', 'fk_checks': []},
                {'csv': 'affiliate_tracking', 'table': 'affiliate_tracking', 'fk_checks': []},
                {'csv': 'promotional_codes', 'table': 'promotional_codes', 'fk_checks': []},
                {'csv': 'customer_segments', 'table': 'customer_segments', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
                {'csv': 'loyalty_program', 'table': 'loyalty_program', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
                {'csv': 'referral_program', 'table': 'referral_program', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
                {'csv': 'marketing_attribution', 'table': 'marketing_attribution', 'fk_checks': []},
            ],
            
            'operational_data': [
                {'csv': 'returns', 'table': 'returns', 'fk_checks': [
                    ('order_id', 'orders', 'order_id')
                ]},
                {'csv': 'refunds', 'table': 'refunds', 'fk_checks': []},
                {'csv': 'exchanges', 'table': 'exchanges', 'fk_checks': []},
                {'csv': 'customer_service', 'table': 'customer_service', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
                {'csv': 'reviews', 'table': 'reviews', 'fk_checks': [
                    ('product_id', 'products', 'product_id'),
                    ('customer_id', 'customers', 'customer_id')
                ]},
                {'csv': 'ratings', 'table': 'ratings', 'fk_checks': []},
                {'csv': 'wishlists', 'table': 'wishlists', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id'),
                    ('product_id', 'products', 'product_id')
                ]},
                {'csv': 'shopping_carts', 'table': 'shopping_carts', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
                {'csv': 'browsing_history', 'table': 'browsing_history', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
                {'csv': 'search_queries', 'table': 'search_queries', 'fk_checks': []},
                {'csv': 'page_views', 'table': 'page_views', 'fk_checks': []},
                {'csv': 'conversion_events', 'table': 'conversion_events', 'fk_checks': []},
                {'csv': 'ab_test_results', 'table': 'ab_test_results', 'fk_checks': []},
                {'csv': 'user_sessions', 'table': 'user_sessions', 'fk_checks': [
                    ('customer_id', 'customers', 'customer_id')
                ]},
            ],
            
            'financial_data': [
                {'csv': 'transactions', 'table': 'transactions', 'fk_checks': [
                    ('order_id', 'orders', 'order_id')
                ]},
                {'csv': 'payment_failures', 'table': 'payment_failures', 'fk_checks': []},
                {'csv': 'chargebacks', 'table': 'chargebacks', 'fk_checks': []},
                {'csv': 'tax_calculations', 'table': 'tax_calculations', 'fk_checks': []},
                {'csv': 'shipping_costs', 'table': 'shipping_costs', 'fk_checks': []},
                {'csv': 'discount_usage', 'table': 'discount_usage', 'fk_checks': []},
                {'csv': 'revenue_recognition', 'table': 'revenue_recognition', 'fk_checks': []},
                {'csv': 'cost_of_goods', 'table': 'cost_of_goods', 'fk_checks': []},
                {'csv': 'profit_margins', 'table': 'profit_margins', 'fk_checks': []},
                {'csv': 'financial_reconciliation', 'table': 'financial_reconciliation', 'fk_checks': []},
            ],
            
            'external_data': [
                {'csv': 'competitor_prices', 'table': 'competitor_prices', 'fk_checks': []},
                {'csv': 'market_trends', 'table': 'market_trends', 'fk_checks': []},
                {'csv': 'economic_indicators', 'table': 'economic_indicators', 'fk_checks': []},
                {'csv': 'weather_data', 'table': 'weather_data', 'fk_checks': []},
                {'csv': 'holiday_calendar', 'table': 'holiday_calendar', 'fk_checks': []},
                {'csv': 'demographics', 'table': 'demographics', 'fk_checks': []},
                {'csv': 'geographic_data', 'table': 'geographic_data', 'fk_checks': []},
                {'csv': 'supplier_ratings', 'table': 'supplier_ratings', 'fk_checks': []},
                {'csv': 'industry_benchmarks', 'table': 'industry_benchmarks', 'fk_checks': []},
                {'csv': 'regulatory_data', 'table': 'regulatory_data', 'fk_checks': []},
            ]
        }
    
    def apply_schema_fixes(self):
        """Apply schema fixes before loading data"""
        print("\n" + "="*60)
        print("APPLYING SCHEMA FIXES")
        print("="*60)
        
        with self.engine.connect() as conn:
            # Fix 1: Add stock_quantity to inventory
            try:
                conn.execute(text("""
                    ALTER TABLE inventory 
                    ADD COLUMN stock_quantity INT DEFAULT 0 
                    AFTER quantity_available
                """))
                conn.commit()
                print("✅ Added stock_quantity to inventory table")
            except Exception as e:
                if "Duplicate column" in str(e):
                    print("⚠️  stock_quantity already exists in inventory")
                else:
                    print(f"ℹ️  Inventory table fix skipped: {str(e)[:50]}")
            
            # Fix 2: Add shipping_address_id to orders
            try:
                conn.execute(text("""
                    ALTER TABLE orders 
                    ADD COLUMN shipping_address_id INT 
                    AFTER customer_id
                """))
                conn.commit()
                print("✅ Added shipping_address_id to orders table")
            except Exception as e:
                if "Duplicate column" in str(e):
                    print("⚠️  shipping_address_id already exists in orders")
                else:
                    print(f"ℹ️  Orders table fix skipped: {str(e)[:50]}")
            
            # Fix 3: Rename card_brand to card_type in payment_methods
            try:
                conn.execute(text("""
                    ALTER TABLE payment_methods 
                    CHANGE COLUMN card_brand card_type VARCHAR(20)
                """))
                conn.commit()
                print("✅ Renamed card_brand to card_type in payment_methods")
            except Exception as e:
                if "Unknown column" in str(e) and "card_brand" in str(e):
                    print("⚠️  card_brand already renamed or doesn't exist")
                else:
                    print(f"ℹ️  Payment methods fix skipped: {str(e)[:50]}")
    
    def validate_csv(self, csv_file, table_name, fk_checks):
        """Validate CSV data before loading"""
        if not csv_file.exists():
            return None, "CSV file not found"
        
        try:
            df = pd.read_csv(csv_file)
            
            if df.empty:
                return None, "CSV is empty"
            
            print(f"\n📋 Validating {csv_file.name} ({len(df)} rows)")
            
            # Validate foreign keys
            for fk_column, parent_table, parent_id_column in fk_checks:
                self.validator.validate_foreign_keys(
                    df, fk_column, parent_table, parent_id_column, csv_file.name
                )
            
            return df, None
            
        except Exception as e:
            return None, str(e)
    
    def load_table(self, table_name, df):
        """Load data into table"""
        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists='append',
                index=False,
                chunksize=1000
            )
            return True, None
        except Exception as e:
            return False, str(e)
    
    def run(self, phases_to_load=None):
        """Execute the complete data loading process"""
        print("\n" + "="*70)
        print("E-COMMERCE DATA LOADER WITH VALIDATION")
        print("="*70)
        
        # Step 1: Apply schema fixes
        self.apply_schema_fixes()
        
        # Determine which phases to load
        if phases_to_load is None:
            phases_to_load = list(self.load_phases.keys())
        
        # Statistics
        total_stats = {
            'loaded': 0,
            'skipped': 0,
            'failed': 0,
            'not_found': 0
        }
        
        # Load each phase
        for phase_name in phases_to_load:
            if phase_name not in self.load_phases:
                print(f"\n⚠️  Unknown phase: {phase_name}")
                continue
            
            print("\n" + "="*70)
            print(f"PHASE: {phase_name.upper().replace('_', ' ')}")
            print("="*70)
            
            phase_config = self.load_phases[phase_name]
            csv_dir = self.csv_base_dir / phase_name
            
            for item in phase_config:
                csv_name = item['csv']
                table_name = item['table']
                fk_checks = item['fk_checks']
                
                csv_file = csv_dir / f"{csv_name}.csv"
                
                # Validate CSV
                df, error = self.validate_csv(csv_file, table_name, fk_checks)
                
                if df is None:
                    if error == "CSV file not found":
                        print(f"⏭️  {csv_name}: File not found at {csv_file}")
                        total_stats['not_found'] += 1
                    else:
                        print(f"❌ {csv_name}: Validation failed - {error}")
                        total_stats['failed'] += 1
                    continue
                
                # Load data if validation passed
                if not self.validator.validation_errors:
                    success, error = self.load_table(table_name, df)
                    
                    if success:
                        print(f"✅ {csv_name} → {table_name}: Loaded {len(df)} rows")
                        total_stats['loaded'] += 1
                    else:
                        if "Duplicate entry" in str(error):
                            print(f"⚠️  {csv_name}: Already loaded, skipping")
                            total_stats['skipped'] += 1
                        elif "doesn't exist" in str(error):
                            print(f"⚠️  {csv_name}: Table '{table_name}' doesn't exist (create it first)")
                            total_stats['not_found'] += 1
                        else:
                            print(f"❌ {csv_name}: Load failed - {str(error)[:100]}")
                            total_stats['failed'] += 1
                else:
                    print(f"⏭️  {csv_name}: Skipped due to validation errors")
                    total_stats['failed'] += 1
                    self.validator.validation_errors = []  # Reset for next file
        
        # Summary
        print("\n" + "="*70)
        print("LOADING SUMMARY")
        print("="*70)
        print(f"✅ Successfully loaded: {total_stats['loaded']} files")
        print(f"⏭️  Skipped (already loaded): {total_stats['skipped']} files")
        print(f"❌ Failed validation/load: {total_stats['failed']} files")
        print(f"📁 Files not found: {total_stats['not_found']} files")
        print(f"📊 Total processed: {sum(total_stats.values())} files")


# ========================================
# USAGE EXAMPLES
# ========================================

if __name__ == "__main__":
    # Configuration - UPDATED WITH CORRECT PORT
    config = {
        'host': 'localhost:3307',  # Added port 3307
        'user': 'root',
        'password': '',  # Empty password as per your config
        'database': 'ecommerce_analytics',
        'csv_base_directory': r'E:\ecommerce-analytics\sample_data'
    }
    
    # Create loader
    loader = DataLoader(**config)
    
    # Load only core data (recommended to start)
    loader.run(phases_to_load=['core_data'])
    
    print("\n✨ Data loading process complete!")