"""
SQL Error Auto-Fixer for E-commerce Analytics Database
Automatically detects and fixes common SQL errors in MariaDB
Works with pymysql (already installed) or mysql-connector-python
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Try to import available MySQL library
try:
    import pymysql as mysql_lib
    USE_PYMYSQL = True
    print("✓ Using pymysql")
except ImportError:
    try:
        import mysql.connector as mysql_lib
        USE_PYMYSQL = False
        print("✓ Using mysql-connector-python")
    except ImportError:
        print("\n❌ No MySQL library found!")
        print("\nPlease install one of the following:")
        print("  pip install pymysql")
        print("  OR")
        print("  pip install mysql-connector-python")
        sys.exit(1)

class SQLErrorFixer:
    def __init__(self, host='localhost', port=3306, user='root', password='', database='ecommerce_analytics'):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
        self.fixes_applied = []
        self.use_pymysql = USE_PYMYSQL
        
    def connect(self):
        """Connect to the database"""
        try:
            if self.use_pymysql:
                self.connection = mysql_lib.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    charset='utf8mb4'
                )
            else:
                self.connection = mysql_lib.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database
                )
            self.cursor = self.connection.cursor()
            print("✅ Connected to database successfully\n")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def execute_safe(self, sql, description=""):
        """Execute SQL with error handling"""
        try:
            self.cursor.execute(sql)
            self.connection.commit()
            print(f"  ✅ {description}")
            self.fixes_applied.append(description)
            return True
        except Exception as e:
            print(f"  ⚠️  {description}: {str(e)[:100]}")
            return False
    
    def table_exists(self, table_name):
        """Check if table exists"""
        try:
            self.cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            return self.cursor.fetchone() is not None
        except:
            return False
    
    def column_exists(self, table_name, column_name):
        """Check if column exists in table"""
        try:
            self.cursor.execute(f"""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = '{self.database}' 
                AND TABLE_NAME = '{table_name}' 
                AND COLUMN_NAME = '{column_name}'
            """)
            result = self.cursor.fetchone()
            return result[0] > 0 if result else False
        except:
            return False
    
    # ====================
    # FIX #1: Add Missing Columns
    # ====================
    def fix_missing_columns(self):
        """Add all missing columns identified in error log"""
        print("\n" + "="*60)
        print("🔧 FIXING MISSING COLUMNS")
        print("="*60)
        
        missing_columns = [
            # Lookup tables
            ('order_status', 'country_code', 'VARCHAR(10)'),
            ('order_status', 'is_active', 'BOOLEAN DEFAULT TRUE'),
            
            # Categories
            ('categories', 'category_code', 'VARCHAR(50)'),
            ('categories', 'is_active', 'BOOLEAN DEFAULT TRUE'),
            
            # Payment methods
            ('payment_methods', 'method_code', 'VARCHAR(50)'),
            ('payment_methods', 'is_active', 'BOOLEAN DEFAULT TRUE'),
            
            # Shipping methods
            ('shipping_methods', 'method_code', 'VARCHAR(50)'),
            ('shipping_methods', 'is_active', 'BOOLEAN DEFAULT TRUE'),
            
            # Loyalty program
            ('loyalty_program', 'min_orders', 'INT DEFAULT 0'),
            
            # Reviews
            ('reviews', 'status', "ENUM('pending', 'approved', 'rejected') DEFAULT 'pending'"),
            
            # Customer segments
            ('customer_segments', 'customer_name', 'VARCHAR(255)'),
            
            # Orders
            ('orders', 'campaign_id', 'INT'),
            
            # Inventory
            ('inventory', 'days_of_stock', 'DECIMAL(10,2)'),
        ]
        
        # Check if materialized view registry exists and add columns
        if self.table_exists('materialized_view_registry'):
            missing_columns.extend([
                ('materialized_view_registry', 'refresh_type', "ENUM('full', 'incremental', 'delta') DEFAULT 'full'"),
                ('materialized_view_registry', 'next_refresh_due', 'DATETIME'),
                ('materialized_view_registry', 'start_time', 'DATETIME'),
            ])
        
        for table, column, datatype in missing_columns:
            if not self.table_exists(table):
                print(f"  ⚠️  Table {table} does not exist, skipping...")
                continue
                
            if not self.column_exists(table, column):
                sql = f"ALTER TABLE {table} ADD COLUMN {column} {datatype}"
                self.execute_safe(sql, f"Added {table}.{column}")
            else:
                print(f"  ✓ {table}.{column} already exists")
    
    # ====================
    # FIX #2: Create Missing Tables
    # ====================
    def fix_missing_tables(self):
        """Create missing tables"""
        print("\n" + "="*60)
        print("🔧 CREATING MISSING TABLES")
        print("="*60)
        
        # Materialized view registry
        if not self.table_exists('materialized_view_registry'):
            sql = """
            CREATE TABLE IF NOT EXISTS materialized_view_registry (
                view_id INT PRIMARY KEY AUTO_INCREMENT,
                view_name VARCHAR(100) UNIQUE NOT NULL,
                refresh_type ENUM('full', 'incremental', 'delta') DEFAULT 'full',
                last_refresh DATETIME,
                next_refresh_due DATETIME,
                start_time DATETIME,
                status ENUM('active', 'inactive') DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.execute_safe(sql, "Created materialized_view_registry table")
        else:
            print("  ✓ materialized_view_registry already exists")
        
        # Campaign tracking if missing
        if not self.table_exists('campaigns'):
            sql = """
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id INT PRIMARY KEY AUTO_INCREMENT,
                campaign_name VARCHAR(200) NOT NULL,
                campaign_type VARCHAR(50),
                start_date DATE,
                end_date DATE,
                budget DECIMAL(10,2),
                status ENUM('active', 'paused', 'completed') DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.execute_safe(sql, "Created campaigns table")
        else:
            print("  ✓ campaigns already exists")
    
    # ====================
    # FIX #3: Clean Duplicate Data
    # ====================
    def fix_duplicate_data(self):
        """Remove duplicate entries before re-insertion"""
        print("\n" + "="*60)
        print("🔧 CLEANING DUPLICATE DATA")
        print("="*60)
        
        # Common duplicates found in logs
        duplicates_to_clean = [
            ("order_status", "DELETE FROM order_status WHERE status_code IN ('NEW', 'PENDING', 'PROCESSING')", "order_status duplicates"),
            ("shipping_methods", "DELETE FROM shipping_methods WHERE shipping_method_id IN ('STANDARD', 'EXPRESS')", "shipping_methods duplicates"),
            ("notification_templates", "DELETE FROM notification_templates WHERE template_name = 'order_confirmation'", "notification template duplicates"),
            ("notification_triggers", "DELETE FROM notification_triggers WHERE trigger_name = 'order_confirmation_trigger'", "notification trigger duplicates"),
        ]
        
        for table, sql, description in duplicates_to_clean:
            if self.table_exists(table):
                self.execute_safe(sql, f"Cleaned {description}")
            else:
                print(f"  ⚠️  Table {table} does not exist, skipping...")
    
    # ====================
    # FIX #4: Update Foreign Keys
    # ====================
    def fix_foreign_keys(self):
        """Fix foreign key constraints"""
        print("\n" + "="*60)
        print("🔧 FIXING FOREIGN KEY CONSTRAINTS")
        print("="*60)
        
        # Add campaign_id foreign key if campaigns table exists
        if self.table_exists('campaigns') and self.table_exists('orders'):
            if not self.column_exists('orders', 'campaign_id'):
                sql = "ALTER TABLE orders ADD COLUMN campaign_id INT"
                self.execute_safe(sql, "Added orders.campaign_id column")
            
            # Try to add foreign key constraint
            try:
                # First check if constraint already exists
                self.cursor.execute("""
                    SELECT COUNT(*) 
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
                    WHERE CONSTRAINT_SCHEMA = %s 
                    AND CONSTRAINT_NAME = 'fk_orders_campaign'
                """, (self.database,))
                
                if self.cursor.fetchone()[0] == 0:
                    sql = """
                    ALTER TABLE orders 
                    ADD CONSTRAINT fk_orders_campaign 
                    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id) ON DELETE SET NULL
                    """
                    self.execute_safe(sql, "Added foreign key orders -> campaigns")
                else:
                    print("  ✓ Foreign key fk_orders_campaign already exists")
            except Exception as e:
                print(f"  ⚠️  Could not add foreign key: {str(e)[:100]}")
    
    # ====================
    # FIX #5: Update Lookup Data Schema
    # ====================
    def fix_lookup_data_schema(self):
        """Fix issues with lookup data tables"""
        print("\n" + "="*60)
        print("🔧 FIXING LOOKUP DATA SCHEMA")
        print("="*60)
        
        # Add category codes for existing categories
        if self.table_exists('categories') and self.column_exists('categories', 'category_code'):
            sql = """
            UPDATE categories 
            SET category_code = UPPER(REPLACE(category_name, ' ', '_'))
            WHERE category_code IS NULL OR category_code = ''
            """
            self.execute_safe(sql, "Generated category codes")
        
        # Add method codes for payment methods
        if self.table_exists('payment_methods') and self.column_exists('payment_methods', 'method_code'):
            sql = """
            UPDATE payment_methods 
            SET method_code = UPPER(REPLACE(method_name, ' ', '_'))
            WHERE method_code IS NULL OR method_code = ''
            """
            self.execute_safe(sql, "Generated payment method codes")
        
        # Add method codes for shipping methods
        if self.table_exists('shipping_methods') and self.column_exists('shipping_methods', 'method_code'):
            sql = """
            UPDATE shipping_methods 
            SET method_code = UPPER(REPLACE(method_name, ' ', '_'))
            WHERE method_code IS NULL OR method_code = ''
            """
            self.execute_safe(sql, "Generated shipping method codes")
    
    # ====================
    # FIX #6: Validate Schema
    # ====================
    def validate_schema(self):
        """Validate database schema"""
        print("\n" + "="*60)
        print("🔍 VALIDATING SCHEMA")
        print("="*60)
        
        required_tables = [
            'customers', 'orders', 'order_items', 'products', 
            'categories', 'vendors', 'reviews', 'inventory'
        ]
        
        missing_tables = []
        existing_tables = []
        
        for table in required_tables:
            if self.table_exists(table):
                print(f"  ✓ {table} exists")
                existing_tables.append(table)
            else:
                print(f"  ✗ {table} MISSING")
                missing_tables.append(table)
        
        if missing_tables:
            print(f"\n  ⚠️  Missing tables: {', '.join(missing_tables)}")
        else:
            print("\n  ✅ All required tables exist")
        
        return len(missing_tables) == 0
    
    # ====================
    # FIX #7: Generate Fix Report
    # ====================
    def generate_report(self):
        """Generate a report of all fixes applied"""
        print("\n" + "="*60)
        print("📊 FIX REPORT")
        print("="*60)
        
        print(f"\nTotal fixes applied: {len(self.fixes_applied)}")
        
        if self.fixes_applied:
            print("\nFixes:")
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"  {i}. {fix}")
        else:
            print("\nNo fixes were needed or all tables already have correct schema.")
        
        # Save report to file
        report_file = f"fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write("SQL ERROR FIX REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Total fixes: {len(self.fixes_applied)}\n\n")
            f.write("Fixes Applied:\n")
            for i, fix in enumerate(self.fixes_applied, 1):
                f.write(f"{i}. {fix}\n")
        
        print(f"\n📄 Report saved to: {report_file}")
    
    # ====================
    # MAIN EXECUTION
    # ====================
    def run_all_fixes(self):
        """Run all fixes in order"""
        if not self.connect():
            return
        
        print("\n" + "="*60)
        print("🚀 STARTING AUTOMATED SQL ERROR FIXES")
        print("="*60)
        
        try:
            # Phase 1: Schema fixes
            self.fix_missing_tables()
            self.fix_missing_columns()
            
            # Phase 2: Data fixes
            self.fix_lookup_data_schema()
            self.fix_duplicate_data()
            self.fix_foreign_keys()
            
            # Phase 3: Validation
            schema_valid = self.validate_schema()
            
            # Phase 4: Report
            self.generate_report()
            
            print("\n" + "="*60)
            print("✅ ALL FIXES COMPLETED")
            print("="*60)
            
            if schema_valid:
                print("\n✓ Database schema is valid!")
            else:
                print("\n⚠️  Some tables are still missing. Run sql_executor_fix.py to create them.")
            
        except Exception as e:
            print(f"\n❌ Error during fix process: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.connection:
                self.connection.close()
                print("\n✅ Database connection closed")


# ====================
# USAGE
# ====================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║   E-COMMERCE ANALYTICS DATABASE ERROR FIXER              ║
║   Automatically fixes common SQL errors                  ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configure your database connection
    DB_CONFIG = {
        'host': 'localhost',
        'port': 3307,  # ← CHANGED to match your setup
        'user': 'root',
        'password': '',
        'database': 'ecommerce_analytics'
    }
    
    # Check if config file exists
    if os.path.exists('db_config.py'):
        try:
            from db_config import DB_CONFIG as imported_config
            DB_CONFIG.update(imported_config)
            print("✓ Loaded configuration from db_config.py\n")
        except:
            pass
    
    # If no password configured, prompt for it
    if DB_CONFIG.get('password') is None:
        import getpass
        print("Database connection required.")
        print(f"Host: {DB_CONFIG['host']}")
        print(f"Port: {DB_CONFIG.get('port', 3306)}")
        print(f"User: {DB_CONFIG['user']}")
        print(f"Database: {DB_CONFIG['database']}")
        password = getpass.getpass("Enter MySQL password (or press Enter if none): ")
        DB_CONFIG['password'] = password
        print()  # Add blank line for readability
    elif DB_CONFIG.get('password') == '':
        # Empty string is valid (no password)
        pass
    
    fixer = SQLErrorFixer(**DB_CONFIG)
    
    # Run all fixes
    fixer.run_all_fixes()
    
    print("\n" + "="*60)
    print("💡 NEXT STEPS:")
    print("="*60)
    print("1. Review the fix report generated")
    print("2. Re-run your sql_executor_fix.py script")
    print("3. Check for remaining errors")
    print("4. Run this fixer again if needed")
    print("="*60)