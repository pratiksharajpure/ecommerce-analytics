"""
Quick Diagnostic Script to Identify Data Issues
Updated to work with subdirectory structure (core_data, marketing_data, etc.)
"""

import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

class DataDiagnostic:
    def __init__(self, host, user, password, database, csv_base_directory):
        self.connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
        self.engine = create_engine(self.connection_string)
        self.csv_base_dir = Path(csv_base_directory)
    
    def get_table_count(self, table_name):
        """Get current row count in table"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.fetchone()[0]
        except:
            return 0
    
    def get_existing_ids(self, table, id_column):
        """Get existing IDs from database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT DISTINCT {id_column} FROM {table}"))
                return set(row[0] for row in result)
        except:
            return set()
    
    def diagnose_csv(self, csv_path, csv_name, checks):
        """Run diagnostics on a CSV file"""
        
        if not csv_path.exists():
            print(f"⏭️  {csv_name}.csv - FILE NOT FOUND at {csv_path}")
            return
        
        try:
            df = pd.read_csv(csv_path)
            print(f"\n📊 {csv_name}.csv")
            print(f"   Location: {csv_path}")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {', '.join(list(df.columns)[:5])}{'...' if len(df.columns) > 5 else ''}")
            
            # Check database status
            db_count = self.get_table_count(csv_name)
            if db_count > 0:
                print(f"   ✅ Database: {db_count:,} rows currently loaded")
            else:
                print(f"   ❌ Database: Empty (0 rows)")
            
            # Run foreign key checks
            for fk_col, parent_table, parent_col in checks:
                if fk_col not in df.columns:
                    print(f"   ⚠️  Column '{fk_col}' not in CSV")
                    continue
                
                csv_ids = set(df[fk_col].dropna().unique())
                if not csv_ids:
                    print(f"   ℹ️  {fk_col}: All NULL values")
                    continue
                
                db_ids = self.get_existing_ids(parent_table, parent_col)
                
                if not db_ids:
                    print(f"   ⚠️  {fk_col}: Parent table '{parent_table}' is empty! Load it first.")
                    continue
                
                missing = csv_ids - db_ids
                
                if missing:
                    print(f"   ❌ {fk_col}: {len(missing)} missing references to {parent_table}.{parent_col}")
                    print(f"      Sample missing IDs: {sorted(list(missing))[:10]}")
                else:
                    print(f"   ✅ {fk_col}: All {len(csv_ids)} references valid")
        
        except Exception as e:
            print(f"❌ Error reading {csv_name}.csv: {e}")
    
    def run_diagnostics(self):
        """Run complete diagnostic check"""
        print("="*70)
        print("E-COMMERCE DATA DIAGNOSTIC REPORT")
        print("="*70)
        
        # Check what's already loaded
        print("\n📦 CURRENTLY LOADED TABLES:")
        tables_to_check = [
            'customers', 'product_categories', 'vendors', 'campaigns',
            'products', 'shipping_addresses', 'payment_methods',
            'orders', 'order_items', 'inventory', 'vendor_contracts'
        ]
        
        for table in tables_to_check:
            count = self.get_table_count(table)
            status = "✅" if count > 0 else "❌"
            print(f"   {status} {table}: {count:,} rows")
        
        # Diagnose core data tables with proper subdirectory paths
        print("\n" + "="*70)
        print("DETAILED DIAGNOSTICS - CORE DATA")
        print("="*70)
        
        core_diagnostics = [
            ('core_data', 'customers', []),
            ('core_data', 'product_categories', []),
            ('core_data', 'vendors', []),
            ('core_data', 'products', [
                ('category_id', 'product_categories', 'category_id')
            ]),
            ('core_data', 'shipping_addresses', [
                ('customer_id', 'customers', 'customer_id')
            ]),
            ('core_data', 'payment_methods', [
                ('customer_id', 'customers', 'customer_id')
            ]),
            ('core_data', 'orders', [
                ('customer_id', 'customers', 'customer_id'),
                ('shipping_address_id', 'shipping_addresses', 'address_id')
            ]),
            ('core_data', 'order_items', [
                ('order_id', 'orders', 'order_id'),
                ('product_id', 'products', 'product_id')
            ]),
            ('core_data', 'inventory', [
                ('product_id', 'products', 'product_id')
            ]),
            ('core_data', 'vendor_contracts', [
                ('vendor_id', 'vendors', 'vendor_id'),
                ('product_id', 'products', 'product_id')
            ])
        ]
        
        for subdirectory, csv_name, checks in core_diagnostics:
            csv_path = self.csv_base_dir / subdirectory / f"{csv_name}.csv"
            self.diagnose_csv(csv_path, csv_name, checks)
        
        # Check for common data quality issues
        print("\n" + "="*70)
        print("DATA QUALITY CHECKS")
        print("="*70)
        
        self.check_data_quality()
        
        # Loading order recommendation
        print("\n" + "="*70)
        print("RECOMMENDED LOADING ORDER")
        print("="*70)
        print("1. ✅ customers (independent)")
        print("2. ✅ product_categories (independent)")
        print("3. ✅ vendors (independent)")
        print("4. 📦 products (needs: product_categories)")
        print("5. 📦 shipping_addresses (needs: customers)")
        print("6. 📦 payment_methods (needs: customers)")
        print("7. 📦 orders (needs: customers, shipping_addresses)")
        print("8. 📦 order_items (needs: orders, products)")
        print("9. 📦 inventory (needs: products)")
        print("10. 📦 vendor_contracts (needs: vendors, products)")
    
    def check_data_quality(self):
        """Check for common data quality issues"""
        quality_checks = [
            ('core_data', 'products', 'product_id'),
            ('core_data', 'orders', 'order_id'),
            ('core_data', 'customers', 'customer_id')
        ]
        
        for subdirectory, csv_name, id_col in quality_checks:
            csv_path = self.csv_base_dir / subdirectory / f"{csv_name}.csv"
            
            if not csv_path.exists():
                continue
            
            try:
                df = pd.read_csv(csv_path)
                
                # Check for duplicate IDs
                if id_col in df.columns:
                    dupes = df[id_col].duplicated().sum()
                    if dupes > 0:
                        print(f"   ⚠️  {csv_name}: {dupes} duplicate {id_col} values")
                    else:
                        print(f"   ✅ {csv_name}: No duplicate IDs")
                
                # Check for nulls in critical columns
                critical_cols = [col for col in df.columns if 'id' in col.lower() and not col.endswith('_id')]
                if id_col in df.columns:
                    null_ids = df[id_col].isnull().sum()
                    if null_ids > 0:
                        print(f"   ⚠️  {csv_name}: {null_ids} NULL values in {id_col}")
            
            except Exception as e:
                print(f"   ⚠️  Could not check {csv_name}: {e}")


# ========================================
# USAGE
# ========================================
if __name__ == "__main__":
    diagnostic = DataDiagnostic(
        host='localhost',
        user='root',
        password='',  # Update this
        database='ecommerce_analytics',
        csv_base_directory='sample_data'  # Base directory with subdirectories
    )
    
    diagnostic.run_diagnostics()