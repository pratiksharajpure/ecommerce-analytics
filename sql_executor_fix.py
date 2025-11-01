"""
Complete SQL Executor for E-commerce Analytics Project
Handles DELIMITER syntax and executes all SQL files in correct order
"""

import pymysql
import re
import os
from pathlib import Path

class SQLExecutor:
    def __init__(self, config):
        """Initialize with database configuration"""
        self.config = config
        self.connection = None
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = pymysql.connect(
                host=self.config['host'],
                port=self.config.get('port', 3306),
                user=self.config['user'],
                password=self.config['password'],
                database=self.config.get('database', ''),
                charset='utf8mb4'
            )
            print("✅ Connected to database successfully")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def fix_sql_content(self, sql_content):
        """Convert SQL with DELIMITER syntax to pymysql-compatible format"""
        # Remove DELIMITER commands
        sql_content = re.sub(r'DELIMITER\s*//', '', sql_content, flags=re.IGNORECASE | re.MULTILINE)
        sql_content = re.sub(r'DELIMITER\s*;', '', sql_content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Replace // with ;
        sql_content = sql_content.replace('//', ';')
        
        return sql_content
    
    def parse_statements(self, sql_content):
        """Parse SQL content into executable statements"""
        statements = []
        current_statement = []
        in_block = False
        block_depth = 0
        
        lines = sql_content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and standalone comments
            if not stripped or (stripped.startswith('--') and not any(x in stripped.upper() for x in ['CREATE', 'DROP', 'INSERT', 'UPDATE'])):
                continue
            
            # Track BEGIN/END blocks
            if re.search(r'\bBEGIN\b', stripped, re.IGNORECASE):
                in_block = True
                block_depth += 1
            
            if re.search(r'\bEND\b', stripped, re.IGNORECASE):
                block_depth -= 1
                if block_depth <= 0:
                    in_block = False
                    block_depth = 0
            
            # Add line to current statement
            current_statement.append(line)
            
            # Statement complete if ends with ; and not in block
            if stripped.endswith(';') and not in_block:
                stmt = '\n'.join(current_statement).strip()
                if stmt and not stmt.startswith('--'):
                    statements.append(stmt)
                current_statement = []
        
        # Add any remaining statement
        if current_statement:
            stmt = '\n'.join(current_statement).strip()
            if stmt and not stmt.startswith('--'):
                statements.append(stmt)
        
        return statements
    
    def execute_statement(self, cursor, statement, statement_num):
        """Execute a single SQL statement"""
        try:
            cursor.execute(statement)
            self.connection.commit()
            return True
        except Exception as e:
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + '...'
            
            print(f"   ⚠️  Statement {statement_num} error: {error_msg}")
            
            # Try to rollback
            try:
                self.connection.rollback()
            except:
                pass
            
            return False
    
    def execute_file(self, filepath):
        """Execute a single SQL file"""
        filename = os.path.basename(filepath)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            self.stats['failed'] += 1
            return False
        
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Fix DELIMITER syntax
            sql_content = self.fix_sql_content(sql_content)
            
            # Parse into statements
            statements = self.parse_statements(sql_content)
            
            if not statements:
                print(f"⚠️  {filename}: No statements found")
                return True
            
            # Execute statements
            cursor = self.connection.cursor()
            executed = 0
            failed = 0
            
            for i, statement in enumerate(statements, 1):
                if self.execute_statement(cursor, statement, i):
                    executed += 1
                else:
                    failed += 1
            
            cursor.close()
            
            if failed == 0:
                print(f"✅ {filename}: Executed {executed} statements")
                self.stats['successful'] += 1
                return True
            else:
                print(f"⚠️  {filename}: {executed} succeeded, {failed} failed")
                self.stats['failed'] += 1
                return False
                
        except Exception as e:
            print(f"❌ {filename}: Fatal error - {str(e)[:100]}")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{filename}: {str(e)}")
            return False
    
    def clear_data(self):
        """Clear all data from tables"""
        print("\n🧹 Clearing existing data...")
        cursor = self.connection.cursor()
        
        try:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            
            tables = [
                'loyalty_program', 'returns', 'reviews', 'campaign_performance',
                'order_items', 'orders', 'inventory', 'vendor_contracts',
                'payment_methods', 'shipping_addresses', 'products',
                'product_categories', 'vendors', 'customers', 'campaigns'
            ]
            
            for table in tables:
                try:
                    cursor.execute(f"TRUNCATE TABLE {table}")
                except:
                    pass  # Table might not exist yet
            
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            self.connection.commit()
            print("✅ Data cleared successfully\n")
            
        except Exception as e:
            print(f"⚠️  Could not clear data: {e}\n")
        finally:
            cursor.close()
    
    def execute_directory(self, directory, file_pattern=None):
        """Execute all SQL files in a directory"""
        if not os.path.exists(directory):
            print(f"⚠️  Directory not found: {directory}")
            return
        
        files = sorted([f for f in os.listdir(directory) if f.endswith('.sql')])
        
        if file_pattern:
            files = [f for f in files if file_pattern in f]
        
        if not files:
            print(f"⚠️  No SQL files found in {directory}")
            return
        
        print(f"\n📁 Executing files from: {os.path.basename(directory)}")
        print("=" * 60)
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            self.stats['total_files'] += 1
            self.execute_file(filepath)
    
    def execute_all(self, sql_root_dir, clear_data=False, skip_categories=None):
        """Execute all SQL files in proper order"""
        print("\n" + "=" * 60)
        print("🚀 E-COMMERCE ANALYTICS DATABASE SETUP")
        print("=" * 60)
        
        if skip_categories is None:
            skip_categories = []
        
        # Clear data if requested
        if clear_data:
            self.clear_data()
        
        # Define execution order with option to skip categories
        execution_order = [
            ('setup', [
                'create_database.sql',
                'create_tables.sql',
                '00_fix_missing_tables.sql',  # Creates missing tables
                'create_indexes.sql',
                'create_views.sql',
                'create_functions.sql',
                        'fix_missing_tables.sql',
'master.sql',
'union2.sql',
'mariadb_fix',
'mariadb-compatible-setup.sql',
                'create_procedures_fixed.sql',  # Use fixed version
                'create_triggers_fixed.sql',    # Use fixed version
                'insert_lookup_data.sql',
                'insert_sample_customers.sql',
                'insert_sample_products.sql',
                'insert_sample_orders.sql',
                'insert_sample_inventory.sql',
                'insert_sample_vendors.sql',
                'insert_sample_campaigns.sql',
                'insert_sample_reviews.sql',
                # 'seed_data_with_issues.sql'
            ]),
            # Skip problematic categories initially
     ('core_analysis', None),
     ('advanced_analysis', None),
     ('reporting', None),
     ('maintenance', None),
    ('automation', None)
        ]
        
        # Execute in order
        for folder, files in execution_order:
            # Skip if in skip list
            if folder in skip_categories:
                print(f"\n⏭️  Skipping: {folder}/")
                continue
                
            folder_path = os.path.join(sql_root_dir, folder)
            
            if files:
                # Execute specific files in order
                print(f"\n📁 Executing: {folder}/")
                print("=" * 60)
                for filename in files:
                    filepath = os.path.join(folder_path, filename)
                    if os.path.exists(filepath):
                        self.stats['total_files'] += 1
                        self.execute_file(filepath)
            else:
                # Execute all files in folder
                self.execute_directory(folder_path)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print execution summary"""
        print("\n" + "=" * 60)
        print("📊 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Total files: {self.stats['total_files']}")
        print(f"✅ Successful: {self.stats['successful']}")
        print(f"❌ Failed: {self.stats['failed']}")
        
        if self.stats['errors']:
            print(f"\n⚠️  Critical Errors:")
            for error in self.stats['errors'][:5]:  # Show first 5
                print(f"   • {error}")
        
        success_rate = (self.stats['successful'] / self.stats['total_files'] * 100) if self.stats['total_files'] > 0 else 0
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        print("=" * 60 + "\n")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")


# Main execution
if __name__ == "__main__":
    # Your configuration - FIXED
    config = {
        'host': 'localhost',      # ✅ Fixed: removed port from host
        'port': 3307,             # ✅ Fixed: added port parameter
        'user': 'root',
        'password': '',
        'database': 'ecommerce_analytics'
    }
    
    # Path to your SQL files
    sql_root_dir = r'E:\ecommerce-analytics\sql'  # Adjust this to your actual path
    
    # Create executor
    executor = SQLExecutor(config)
    
    # Connect to database
    if executor.connect():
        # Execute all files
        # Set clear_data=True if you want to clear existing data first
        executor.execute_all(sql_root_dir, clear_data=True)
        
        # Close connection
        executor.close()
    else:
        print("❌ Could not connect to database. Please check your configuration.")
    
    