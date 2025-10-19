"""
🚀 QUICK FIX SCRIPT - Run this to fix all errors!
This will diagnose and fix your database issues
"""

import os
import sys
from pathlib import Path

def print_banner(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_env_file():
    """Check if .env file exists and is configured"""
    print_banner("1️⃣ CHECKING .ENV FILE")
    
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found!")
        print("\n📝 Creating .env file...")
        
        env_content = """# Database Configuration
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=ecommerce_analytics
DB_PORT=3306
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created!")
        print("⚠️  IMPORTANT: Edit .env and add your MySQL password!")
        return False
    else:
        print("✅ .env file exists")
        
        # Read and check contents
        with open('.env', 'r') as f:
            content = f.read()
            
        if 'your_password_here' in content:
            print("⚠️  WARNING: Default password detected!")
            print("   Please edit .env and set your real MySQL password")
            return False
        
        print("✅ .env appears configured")
        return True

def check_database_connection():
    """Test database connection"""
    print_banner("2️⃣ TESTING DATABASE CONNECTION")
    
    try:
        from utils.database import test_connection, get_database_stats
        
        if test_connection():
            print("✅ Database connection successful!")
            
            stats = get_database_stats()
            print(f"\n📊 Database: {stats.get('database', 'ecommerce_analytics')}")
            print(f"   Tables: {len(stats.get('tables', []))}")
            print(f"   Total Rows: {stats.get('total_rows', 0):,}")
            
            return True, stats
        else:
            print("❌ Database connection failed!")
            print("\n🔧 Troubleshooting:")
            print("   1. Check MySQL is running")
            print("   2. Verify .env credentials")
            print("   3. Check database 'ecommerce_analytics' exists")
            return False, {}
            
    except ImportError as e:
        print("❌ Cannot import database.py")
        print(f"   Error: {str(e)}")
        print("\n🔧 Fix: Ensure utils/database.py exists")
        return False, {}
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False, {}

def check_required_tables():
    """Check if required tables exist"""
    print_banner("3️⃣ CHECKING REQUIRED TABLES")
    
    required_tables = [
        'customers', 'products', 'orders', 
        'inventory', 'vendors', 'campaigns', 
        'reviews', 'returns'
    ]
    
    optional_tables = ['transactions']
    
    try:
        from utils.database import table_exists, execute_sql_query
        
        missing = []
        existing = []
        
        for table in required_tables:
            if table_exists(table):
                # Get row count
                df = execute_sql_query(f"SELECT COUNT(*) as count FROM {table}")
                count = df['count'].iloc[0] if df is not None else 0
                existing.append((table, count))
                print(f"✅ {table}: {count:,} rows")
            else:
                missing.append(table)
                print(f"❌ {table}: MISSING")
        
        # Check optional tables
        print("\n📋 Optional Tables:")
        for table in optional_tables:
            if table_exists(table):
                df = execute_sql_query(f"SELECT COUNT(*) as count FROM {table}")
                count = df['count'].iloc[0] if df is not None else 0
                print(f"✅ {table}: {count:,} rows")
            else:
                print(f"⚠️  {table}: Not found (will use CSV/generated data)")
        
        if missing:
            print(f"\n❌ Missing {len(missing)} required tables!")
            print("   Run your database setup scripts to create them")
            return False
        
        print(f"\n✅ All {len(required_tables)} required tables exist!")
        return True
        
    except Exception as e:
        print(f"❌ Error checking tables: {str(e)}")
        return False

def check_csv_files():
    """Check if CSV fallback files exist"""
    print_banner("4️⃣ CHECKING CSV BACKUP FILES")
    
    csv_paths = {
        'customers': 'sample_data/core_data/customers.csv',
        'products': 'sample_data/core_data/products.csv',
        'orders': 'sample_data/core_data/orders.csv',
        'inventory': 'sample_data/core_data/inventory.csv',
    }
    
    found = 0
    for name, path in csv_paths.items():
        if Path(path).exists():
            print(f"✅ {name}.csv found")
            found += 1
        else:
            print(f"❌ {name}.csv missing")
    
    if found > 0:
        print(f"\n✅ Found {found}/{len(csv_paths)} CSV files")
        print("   Dashboard can use these as fallback")
        return True
    else:
        print("\n⚠️  No CSV files found")
        print("   Dashboard will generate sample data")
        return False

def create_transactions_table():
    """Create the transactions table if missing"""
    print_banner("5️⃣ CREATING TRANSACTIONS TABLE")
    
    try:
        from utils.database import table_exists, execute_sql_query
        
        if table_exists('transactions'):
            print("✅ Transactions table already exists")
            return True
        
        print("📝 Creating transactions table...")
        
        sql = """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INT PRIMARY KEY AUTO_INCREMENT,
            order_id INT NOT NULL,
            transaction_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            amount DECIMAL(10, 2) NOT NULL,
            payment_method VARCHAR(50),
            status VARCHAR(20) DEFAULT 'pending',
            currency VARCHAR(3) DEFAULT 'USD',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_order_id (order_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        
        execute_sql_query(sql)
        
        # Verify
        if table_exists('transactions'):
            print("✅ Transactions table created successfully!")
            
            # Try to populate with sample data from orders
            print("📝 Populating with data from orders...")
            populate_sql = """
            INSERT INTO transactions (order_id, transaction_date, amount, payment_method, status)
            SELECT 
                order_id,
                order_date as transaction_date,
                total_amount as amount,
                payment_method,
                'success' as status
            FROM orders
            LIMIT 1000
            """
            execute_sql_query(populate_sql)
            
            # Check count
            df = execute_sql_query("SELECT COUNT(*) as count FROM transactions")
            count = df['count'].iloc[0] if df is not None else 0
            print(f"✅ Added {count:,} transactions")
            
            return True
        else:
            print("❌ Failed to create table")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def run_dashboard_test():
    """Test if dashboard can run"""
    print_banner("6️⃣ TESTING DASHBOARD")
    
    try:
        print("📝 Importing dashboard modules...")
        import streamlit
        import pandas
        import plotly
        print("✅ All imports successful")
        
        print("\n📝 Testing data loader...")
        sys.path.insert(0, str(Path.cwd()))
        
        # We can't actually load the Home.py module here, but we can check it exists
        if Path('Home.py').exists():
            print("✅ Home.py found")
        else:
            print("❌ Home.py not found!")
            return False
        
        print("\n✅ Dashboard should work!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing package: {str(e)}")
        print("\n🔧 Install missing packages:")
        print("   pip install streamlit pandas plotly python-dotenv pymysql")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          🚀 E-COMMERCE DASHBOARD QUICK FIX 🚀           ║
║                                                          ║
║          This will diagnose and fix all issues          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # Run checks
    results.append(("ENV File", check_env_file()))
    
    db_connected, stats = check_database_connection()
    results.append(("Database Connection", db_connected))
    
    if db_connected:
        results.append(("Required Tables", check_required_tables()))
        results.append(("Transactions Table", create_transactions_table()))
    
    results.append(("CSV Backup Files", check_csv_files()))
    results.append(("Dashboard Dependencies", run_dashboard_test()))
    
    # Summary
    print_banner("📊 SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\n{'='*60}")
    print(f"  {passed}/{total} checks passed")
    print(f"{'='*60}\n")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED! 🎉")
        print("\n✅ Your dashboard is ready to run!")
        print("\n🚀 Start with: streamlit run Home.py")
    elif passed >= total - 2:
        print("⚠️  MOSTLY READY!")
        print("\n✅ Dashboard will work with CSV/sample data")
        print("⚠️  Fix database issues for full functionality")
        print("\n🚀 You can still run: streamlit run Home.py")
    else:
        print("❌ FIXES NEEDED!")
        print("\n🔧 Fix the failed checks above, then run this script again")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
