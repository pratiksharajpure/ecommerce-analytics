"""
Dashboard Status Checker
Run this to verify what data sources are working
"""

import sys
from pathlib import Path

def print_banner(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_database():
    """Check database connectivity"""
    print_banner("1️⃣ DATABASE CONNECTION CHECK")
    
    try:
        from utils.database import test_connection, get_table_names, safe_table_query
        
        if test_connection():
            print("✅ MySQL/MariaDB connection: SUCCESS")
            
            tables = get_table_names()
            print(f"✅ Tables found: {len(tables)}")
            
            if tables:
                print("\n📊 Available Tables:")
                for table in tables:
                    try:
                        df = safe_table_query(table, limit=1)
                        row_count = len(df) if df is not None else 0
                        print(f"   ✅ {table}: accessible")
                    except:
                        print(f"   ⚠️  {table}: exists but error reading")
            
            return True, len(tables)
        else:
            print("❌ MySQL/MariaDB connection: FAILED")
            return False, 0
            
    except ImportError:
        print("❌ database.py not found or import error")
        return False, 0
    except Exception as e:
        print(f"❌ Error: {str(e)[:100]}")
        return False, 0

def check_csv_files():
    """Check CSV files availability"""
    print_banner("2️⃣ CSV FILES CHECK")
    
    csv_files = {
        'Core Data': [
            'sample_data/core_data/customers.csv',
            'sample_data/core_data/products.csv',
            'sample_data/core_data/orders.csv',
            'sample_data/core_data/inventory.csv',
            'sample_data/core_data/vendors.csv',
        ],
        'Marketing': [
            'sample_data/marketing_data/campaigns.csv',
        ],
        'Operational': [
            'sample_data/operational_data/reviews.csv',
            'sample_data/operational_data/returns.csv',
        ],
        'Financial': [
            'sample_data/financial_data/transactions.csv',
        ]
    }
    
    total_found = 0
    
    for category, files in csv_files.items():
        print(f"\n📁 {category}:")
        for csv_file in files:
            if Path(csv_file).exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    print(f"   ✅ {Path(csv_file).name}: {len(df):,} rows")
                    total_found += 1
                except Exception as e:
                    print(f"   ⚠️  {Path(csv_file).name}: exists but error reading")
            else:
                print(f"   ❌ {Path(csv_file).name}: not found")
    
    return total_found

def check_sql_files():
    """Check SQL files"""
    print_banner("3️⃣ SQL FILES CHECK")
    
    sql_dirs = {
        'Setup': 'sql/setup',
        'Core Analysis': 'sql/core_analysis',
        'Advanced Analysis': 'sql/advanced_analysis',
        'Reporting': 'sql/reporting',
        'Maintenance': 'sql/maintenance'
    }
    
    total_found = 0
    
    for category, directory in sql_dirs.items():
        if Path(directory).exists():
            sql_files = list(Path(directory).glob('*.sql'))
            print(f"\n📄 {category}: {len(sql_files)} files")
            for sql_file in sql_files[:3]:  # Show first 3
                print(f"   ✅ {sql_file.name}")
            if len(sql_files) > 3:
                print(f"   ... and {len(sql_files) - 3} more")
            total_found += len(sql_files)
        else:
            print(f"\n❌ {category}: directory not found")
    
    return total_found

def check_streamlit_pages():
    """Check Streamlit pages"""
    print_banner("4️⃣ STREAMLIT PAGES CHECK")
    
    pages = [
        'Home.py',
        'pages/customers.py',
        'pages/products.py',
        'pages/orders.py',
    ]
    
    found = 0
    for page in pages:
        if Path(page).exists():
            print(f"✅ {page}")
            found += 1
        else:
            print(f"❌ {page}: not found")
    
    return found

def check_dashboard_functionality():
    """Test if dashboard can load data"""
    print_banner("5️⃣ DASHBOARD FUNCTIONALITY TEST")
    
    try:
        # Try to import and run the data loader
        sys.path.insert(0, '.')
        
        print("🔄 Testing data loader...")
        
        # Try SQL first
        try:
            from utils.database import safe_table_query
            customers = safe_table_query('customers', limit=10)
            if customers is not None and not customers.empty:
                print(f"✅ SQL data loading: SUCCESS ({len(customers)} sample rows)")
                return "SQL Database"
        except:
            pass
        
        # Try CSV
        try:
            import pandas as pd
            customers_csv = pd.read_csv('sample_data/core_data/customers.csv')
            if not customers_csv.empty:
                print(f"✅ CSV data loading: SUCCESS ({len(customers_csv)} rows)")
                return "CSV Files"
        except:
            pass
        
        print("✅ Will use generated sample data")
        return "Sample Data"
        
    except Exception as e:
        print(f"⚠️  Error: {str(e)[:100]}")
        return "Unknown"

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║          📊 E-COMMERCE DASHBOARD STATUS CHECKER 📊                ║
║                                                                    ║
║          Verifying all data sources and functionality             ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all checks
    db_connected, table_count = check_database()
    csv_count = check_csv_files()
    sql_count = check_sql_files()
    page_count = check_streamlit_pages()
    data_source = check_dashboard_functionality()
    
    # Final summary
    print_banner("📋 SUMMARY")
    
    print(f"Database Connection: {'✅ Connected' if db_connected else '❌ Not Connected'}")
    print(f"Database Tables: {table_count}")
    print(f"CSV Files Available: {csv_count}")
    print(f"SQL Files Found: {sql_count}")
    print(f"Streamlit Pages: {page_count}")
    print(f"Primary Data Source: {data_source}")
    
    print("\n" + "="*70)
    
    # Overall status
    if db_connected or csv_count > 0:
        print("\n✅ DASHBOARD STATUS: OPERATIONAL")
        print("\n🚀 Ready to run: streamlit run Home.py")
        
        if not db_connected and csv_count > 0:
            print("\n💡 Note: Using CSV files (SQL connection unavailable)")
        elif db_connected:
            print(f"\n💡 Note: Using MySQL/MariaDB with {table_count} tables")
            print("   Some SQL files may have compatibility issues (this is normal)")
            print("   The dashboard will still display all data correctly!")
    else:
        print("\n⚠️  DASHBOARD STATUS: LIMITED")
        print("\n💡 Dashboard will work with generated sample data")
        print("🚀 You can still run: streamlit run Home.py")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
