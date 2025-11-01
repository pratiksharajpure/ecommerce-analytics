#!/usr/bin/env python3
"""
Complete Automated Database Setup Script - WINDOWS FIXED VERSION
Handles schema creation, data loading, and error recovery
"""

import pymysql
import pandas as pd
from pathlib import Path
import logging
import sys
from datetime import datetime

# Setup logging WITHOUT EMOJIS for Windows
log_filename = f"setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatabaseSetup:
    """Complete database setup and data loading"""
    
    def __init__(self, host='localhost', port=3306, user='root', password='', database='ecommerce_analytics'):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.conn = None
        self.stats = {
            'tables_created': 0,
            'tables_cleared': 0,
            'files_loaded': 0,
            'rows_loaded': 0,
            'errors': []
        }
        
        # Correct loading order
        self.load_order = [
            'product_categories', 'customers', 'vendors', 'products',
            'shipping_addresses', 'payment_methods', 'campaigns',
            'orders', 'order_items', 'inventory', 'vendor_contracts',
            'reviews', 'returns', 'loyalty_program', 'campaign_performance',
            'customers_duplicates', 'customers_invalid_emails',
            'products_missing_info', 'inventory_discrepancies', 'orders_invalid'
        ]
    
    def connect(self, use_db=True):
        """Connect to MySQL"""
        try:
            params = {
                'host': self.host,
                'port': self.port,
                'user': self.user,
                'password': self.password,
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor
            }
            if use_db:
                params['database'] = self.database
            
            self.conn = pymysql.connect(**params)
            logger.info(f"[OK] Connected to MySQL{' database: ' + self.database if use_db else ''}")
            return True
        except pymysql.err.OperationalError as e:
            if '2003' in str(e):
                logger.error(f"[ERROR] Cannot connect to MySQL server")
                logger.error(f"Connection details: {self.host}:{self.port}")
                logger.error(f"Please check:")
                logger.error(f"  1. MySQL service is running")
                logger.error(f"  2. Host and port are correct (default: localhost:3306)")
                logger.error(f"  3. Username and password are correct")
                logger.error(f"  4. Firewall is not blocking the connection")
            else:
                logger.error(f"[ERROR] Connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Connection failed: {e}")
            return False
    
    def execute_sql_file(self, filepath, continue_on_error=False):
        """Execute SQL file with proper error handling"""
        if self.conn is None:
            logger.error(f"[ERROR] No database connection available")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Remove comments
            lines = []
            for line in sql_content.split('\n'):
                # Remove inline comments but keep the rest of the line
                if '--' in line:
                    line = line[:line.index('--')]
                if line.strip():
                    lines.append(line)
            
            sql_content = '\n'.join(lines)
            
            # Split by semicolons but handle multi-line statements
            statements = []
            current_stmt = []
            
            for line in sql_content.split('\n'):
                current_stmt.append(line)
                if ';' in line:
                    stmt = '\n'.join(current_stmt).strip()
                    if stmt and not stmt.startswith('--'):
                        statements.append(stmt.rstrip(';'))
                    current_stmt = []
            
            cursor = self.conn.cursor()
            success = 0
            failed = 0
            
            for stmt in statements:
                if not stmt.strip():
                    continue
                
                try:
                    cursor.execute(stmt)
                    success += 1
                    # Show what was executed for debugging
                    if 'CREATE TABLE' in stmt.upper():
                        table_name = stmt.split('CREATE TABLE')[1].split('(')[0].strip()
                        logger.info(f"  Created table: {table_name}")
                except Exception as e:
                    failed += 1
                    error_msg = f"Statement failed: {str(e)[:100]}"
                    logger.warning(f"  {error_msg}")
                    
                    if not continue_on_error:
                        self.stats['errors'].append(error_msg)
            
            self.conn.commit()
            logger.info(f"[OK] Executed {filepath.name}: {success} statements succeeded, {failed} failed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to execute {filepath}: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def create_database(self):
        """Create database if it doesn't exist"""
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Creating Database")
        logger.info("="*60)
        
        try:
            if not self.connect(use_db=False):
                return False
            
            cursor = self.conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            cursor.execute(f"USE {self.database}")
            self.conn.commit()
            
            logger.info(f"[OK] Database '{self.database}' ready")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Database creation failed: {e}")
            return False
    
    def create_tables(self, sql_file='sql/setup/create_tables.sql'):
        """Create all tables from SQL file"""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Creating Tables")
        logger.info("="*60)
        
        if self.conn is None:
            logger.error("[ERROR] No database connection. Skipping table creation.")
            return False
        
        sql_path = Path(sql_file)
        if not sql_path.exists():
            logger.error(f"[ERROR] Table creation file not found: {sql_path}")
            return False
        
        if self.execute_sql_file(sql_path):
            self.stats['tables_created'] += 1
            return True
        return False
    
    def create_quality_tables(self):
        """Create quality check tables"""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Creating Quality Check Tables")
        logger.info("="*60)
        
        if self.conn is None:
            logger.error("[ERROR] No database connection. Skipping quality table creation.")
            return False
        
        quality_tables_sql = """
        CREATE TABLE IF NOT EXISTS customers_duplicates (
            duplicate_id INT PRIMARY KEY AUTO_INCREMENT,
            customer_id VARCHAR(20),
            email VARCHAR(100),
            duplicate_count INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_email (email)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        
        CREATE TABLE IF NOT EXISTS customers_invalid_emails (
            invalid_email_id INT PRIMARY KEY AUTO_INCREMENT,
            customer_id VARCHAR(20),
            email VARCHAR(100),
            issue_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_customer_id (customer_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        
        CREATE TABLE IF NOT EXISTS products_missing_info (
            missing_info_id INT PRIMARY KEY AUTO_INCREMENT,
            product_id VARCHAR(20),
            product_name VARCHAR(200),
            missing_field VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_product_id (product_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        
        CREATE TABLE IF NOT EXISTS inventory_discrepancies (
            discrepancy_id INT PRIMARY KEY AUTO_INCREMENT,
            product_id VARCHAR(20),
            expected_quantity INT,
            actual_quantity INT,
            difference INT,
            discrepancy_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_product_id (product_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        
        CREATE TABLE IF NOT EXISTS orders_invalid (
            invalid_order_id INT PRIMARY KEY AUTO_INCREMENT,
            order_id VARCHAR(20),
            customer_id VARCHAR(20),
            issue_type VARCHAR(50),
            issue_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_order_id (order_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        try:
            cursor = self.conn.cursor()
            for statement in quality_tables_sql.split(';'):
                if statement.strip():
                    cursor.execute(statement)
            self.conn.commit()
            logger.info("[OK] Quality check tables created")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Quality table creation failed: {e}")
            return False
    
    def clear_all_tables(self):
        """Clear all existing data"""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Clearing Existing Data")
        logger.info("="*60)
        
        if self.conn is None:
            logger.error("[ERROR] No database connection. Skipping data clearing.")
            return False
        
        tables_to_clear = [
            'loyalty_program', 'returns', 'reviews', 'campaign_performance',
            'vendor_contracts', 'payment_methods', 'shipping_addresses',
            'inventory', 'order_items', 'orders', 'campaigns',
            'products', 'product_categories', 'vendors', 'customers',
            'customers_duplicates', 'customers_invalid_emails',
            'products_missing_info', 'inventory_discrepancies', 'orders_invalid'
        ]
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            
            for table in tables_to_clear:
                try:
                    cursor.execute(f"TRUNCATE TABLE {table}")
                    self.stats['tables_cleared'] += 1
                    logger.info(f"  [OK] Cleared {table}")
                except:
                    pass  # Table might not exist
            
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            self.conn.commit()
            logger.info("[OK] All tables cleared")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Clear tables failed: {e}")
            return False
    
    def load_csv_data(self, csv_directory='sample_data/core_data'):
        """Load CSV files in correct order"""
        logger.info("\n" + "="*60)
        logger.info(f"STEP 5: Loading CSV Data from {csv_directory}")
        logger.info("="*60)
        
        if self.conn is None:
            logger.error("[ERROR] No database connection. Skipping data loading.")
            return False
        
        csv_dir = Path(csv_directory)
        if not csv_dir.exists():
            logger.warning(f"[WARN] Directory not found: {csv_dir}")
            return True  # Not a critical error
        
        cursor = self.conn.cursor()
        
        try:
            # Disable constraints for loading
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            cursor.execute("SET SQL_MODE = 'NO_AUTO_VALUE_ON_ZERO'")
            
            for table_name in self.load_order:
                csv_file = csv_dir / f"{table_name}.csv"
                
                if not csv_file.exists():
                    continue
                
                try:
                    # Read CSV
                    df = pd.read_csv(csv_file, low_memory=False)
                    df.columns = df.columns.str.strip()
                    
                    # Get table columns
                    cursor.execute(f"DESCRIBE {table_name}")
                    table_cols = {row['Field'] for row in cursor.fetchall()}
                    
                    # Filter matching columns
                    matching_cols = [c for c in df.columns if c in table_cols]
                    if not matching_cols:
                        logger.warning(f"  [WARN] {table_name}: No matching columns")
                        continue
                    
                    df_clean = df[matching_cols].where(pd.notnull(df), None)
                    
                    # Build and execute insert
                    if len(df_clean) > 0:
                        cols = ', '.join(matching_cols)
                        placeholders = ', '.join(['%s'] * len(matching_cols))
                        sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
                        
                        data = [tuple(row) for row in df_clean.values]
                        cursor.executemany(sql, data)
                        self.conn.commit()
                        
                        self.stats['files_loaded'] += 1
                        self.stats['rows_loaded'] += len(data)
                        logger.info(f"  [OK] {table_name}: {len(data)} rows")
                
                except Exception as e:
                    logger.warning(f"  [ERROR] {table_name}: {str(e)[:100]}")
                    self.stats['errors'].append(f"{table_name}: {str(e)[:100]}")
                    self.conn.rollback()
            
            # Re-enable constraints
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            self.conn.commit()
            
            logger.info(f"[OK] Data loading completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Data loading failed: {e}")
            return False
    
    def verify_setup(self):
        """Verify the setup was successful"""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: Verification")
        logger.info("="*60)
        
        if self.conn is None:
            logger.error("[ERROR] No database connection. Skipping verification.")
            return False
        
        cursor = self.conn.cursor()
        
        # Count tables and rows
        cursor.execute("""
            SELECT table_name, table_rows
            FROM information_schema.tables
            WHERE table_schema = %s
            ORDER BY table_name
        """, (self.database,))
        
        results = cursor.fetchall()
        
        logger.info(f"\nDatabase: {self.database}")
        logger.info(f"Total Tables: {len(results)}")
        logger.info("\nTable Row Counts:")
        
        for row in results:
            logger.info(f"  {row['table_name']}: {row['table_rows']} rows")
        
        return True
    
    def print_summary(self):
        """Print setup summary"""
        logger.info("\n" + "="*60)
        logger.info("SETUP SUMMARY")
        logger.info("="*60)
        logger.info(f"Tables Created: {self.stats['tables_created']}")
        logger.info(f"Tables Cleared: {self.stats['tables_cleared']}")
        logger.info(f"CSV Files Loaded: {self.stats['files_loaded']}")
        logger.info(f"Total Rows Loaded: {self.stats['rows_loaded']}")
        logger.info(f"Errors: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            logger.info("\nErrors encountered:")
            for err in self.stats['errors'][:10]:
                logger.info(f"  - {err}")
            if len(self.stats['errors']) > 10:
                logger.info(f"  ... and {len(self.stats['errors'])-10} more")
        
        logger.info(f"\nLog saved to: {log_filename}")
        logger.info("="*60)
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        logger.info("="*60)
        logger.info("E-COMMERCE DATABASE COMPLETE SETUP")
        logger.info("="*60)
        
        steps = [
            (self.create_database, "Create Database"),
            (self.create_tables, "Create Tables"),
            (self.create_quality_tables, "Create Quality Tables"),
            (self.clear_all_tables, "Clear Existing Data"),
            (lambda: self.load_csv_data('sample_data/core_data'), "Load Core Data"),
            (lambda: self.load_csv_data('sample_data/marketing_data'), "Load Marketing Data"),
            (lambda: self.load_csv_data('sample_data/operational_data'), "Load Operational Data"),
            (self.verify_setup, "Verify Setup"),
        ]
        
        for step_func, step_name in steps:
            try:
                if not step_func():
                    logger.warning(f"[WARN] {step_name} had issues but continuing...")
            except Exception as e:
                logger.error(f"[ERROR] {step_name} failed: {e}")
                self.stats['errors'].append(f"{step_name}: {str(e)}")
        
        self.print_summary()
        
        if self.conn:
            self.conn.close()
            logger.info("[OK] Database connection closed")


def detect_mysql_config():
    """Try to detect MySQL configuration"""
    common_configs = [
        {'host': 'localhost', 'port': 3306},
        {'host': '127.0.0.1', 'port': 3306},
        {'host': 'localhost', 'port': 3307},
        {'host': '127.0.0.1', 'port': 3307},
    ]
    
    print("\n" + "="*60)
    print("TESTING MYSQL CONNECTIONS")
    print("="*60)
    
    for config in common_configs:
        try:
            print(f"\nTrying {config['host']}:{config['port']}...", end=" ")
            conn = pymysql.connect(
                host=config['host'],
                port=config['port'],
                user='root',
                password='',
                connect_timeout=2
            )
            conn.close()
            print("[OK] Connection successful!")
            return config
        except:
            print("[FAILED]")
    
    return None


def main():
    """Main entry point"""
    print("="*60)
    print("E-COMMERCE DATABASE SETUP")
    print("="*60)
    
    # Configuration - DIRECT SETUP FOR PORT 3307
    # If you know your MySQL port, update it here:
    config = {
        'host': 'localhost',      # Host WITHOUT port
        'port': 3307,             # Port as separate parameter (change to 3306 if needed)
        'user': 'root',
        'password': '',           # Update if you have a password
        'database': 'ecommerce_analytics'
    }
    
    print(f"\nUsing configuration:")
    print(f"  Host: {config['host']}")
    print(f"  Port: {config['port']}")
    print(f"  User: {config['user']}")
    print(f"  Database: {config['database']}")
    print(f"  Password: {'(set)' if config['password'] else '(empty)'}")
    
    # Option to change settings
    change = input("\nUse these settings? (Y/n): ").strip().lower()
    if change == 'n':
        config['host'] = input(f"Host [{config['host']}]: ").strip() or config['host']
        port_input = input(f"Port [{config['port']}]: ").strip()
        config['port'] = int(port_input) if port_input else config['port']
        config['user'] = input(f"User [{config['user']}]: ").strip() or config['user']
        password = input("Password (press Enter if none): ").strip()
        if password:
            config['password'] = password
    
    # Run setup
    setup = DatabaseSetup(**config)
    setup.run_complete_setup()


if __name__ == "__main__":
    main()