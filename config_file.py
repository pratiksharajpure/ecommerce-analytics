"""
Centralized Configuration for E-commerce Data Loader
Update your database credentials here once, use everywhere!
"""

# ========================================
# DATABASE CONFIGURATION
# ========================================
DB_CONFIG = {
    'host': 'localhost',      # Your MySQL server (usually localhost)
    'user': 'root',           # Your MySQL username
    'password': '',  # ⚠️ CHANGE THIS to your MySQL password
    'database': 'ecommerce_analytics',  # Database name
}

# ========================================
# FILE PATHS CONFIGURATION
# ========================================
# Choose ONE of the following (comment out the others):

# Option 1: Relative path (RECOMMENDED - works from E:\ecommerce-analytics\)
CSV_BASE_DIRECTORY = 'sample_data'

# Option 2: Absolute path with raw string (if relative doesn't work)
# CSV_BASE_DIRECTORY = r'E:\ecommerce-analytics\sample_data'

# Option 3: Absolute path with forward slashes
# CSV_BASE_DIRECTORY = 'E:/ecommerce-analytics/sample_data'


# ========================================
# LOADING PREFERENCES
# ========================================
# Which data phases to load (comment out what you don't want)
PHASES_TO_LOAD = [
    'core_data',           # ✅ Essential data (customers, products, orders)
    'marketing_data',      # 📊 Marketing campaigns and performance
    'operational_data',    # 🔄 Returns, reviews, browsing history
    'financial_data',      # 💰 Transactions, payments, profit margins
    'external_data',       # 🌍 Market trends, competitors, demographics
]

# Set to True to load only specific phases, False to load all
LOAD_SPECIFIC_PHASES = True  # Set to False to load everything

# ========================================
# ADVANCED OPTIONS
# ========================================
CHUNK_SIZE = 1000  # Number of rows to insert at once
SKIP_ALREADY_LOADED = True  # Skip tables that already have data


# ========================================
# HELPER FUNCTION
# ========================================
def get_full_config():
    """Get complete configuration dictionary"""
    return {
        **DB_CONFIG,
        'csv_base_directory': CSV_BASE_DIRECTORY
    }


# ========================================
# DISPLAY CURRENT CONFIGURATION
# ========================================
if __name__ == "__main__":
    print("="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(f"Database Host: {DB_CONFIG['host']}")
    print(f"Database Name: {DB_CONFIG['database']}")
    print(f"Database User: {DB_CONFIG['user']}")
    print(f"Password Set: {'Yes' if DB_CONFIG['password'] != 'your_password' else '⚠️  NO - UPDATE THIS!'}")
    print(f"\nCSV Directory: {CSV_BASE_DIRECTORY}")
    print(f"\nPhases to Load: {', '.join(PHASES_TO_LOAD) if LOAD_SPECIFIC_PHASES else 'ALL'}")
    print("="*60)
