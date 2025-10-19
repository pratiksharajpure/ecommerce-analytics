"""
Debug script to identify why the import is failing
Run this in the same directory as Home.py
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("IMPORT DEBUG SCRIPT")
print("=" * 60)

# 1. Check current directory
print(f"\n📁 Current Directory: {os.getcwd()}")

# 2. Check Python path
print(f"\n🐍 Python Path:")
for idx, path in enumerate(sys.path, 1):
    print(f"   {idx}. {path}")

# 3. Check if utils folder exists
print(f"\n📂 Checking 'utils' folder:")
utils_path = Path("utils")
if utils_path.exists():
    print(f"   ✅ utils/ folder EXISTS")
    
    # Check __init__.py
    init_file = utils_path / "__init__.py"
    if init_file.exists():
        print(f"   ✅ utils/__init__.py EXISTS")
    else:
        print(f"   ❌ utils/__init__.py MISSING")
    
    # Check database.py
    db_file = utils_path / "database.py"
    if db_file.exists():
        print(f"   ✅ utils/database.py EXISTS")
    else:
        print(f"   ❌ utils/database.py MISSING")
    
    # List all files in utils
    print(f"\n   📋 Files in utils/:")
    for file in utils_path.iterdir():
        print(f"      - {file.name}")
else:
    print(f"   ❌ utils/ folder DOES NOT EXIST")

# 4. Check if database.py exists in current directory
print(f"\n📄 Checking current directory for database.py:")
db_current = Path("database.py")
if db_current.exists():
    print(f"   ⚠️  database.py found in CURRENT directory (should be in utils/)")
else:
    print(f"   ✅ database.py not in current directory (good)")

# 5. Try the actual import
print(f"\n🔧 Attempting import:")
try:
    from utils.database import safe_table_query, table_exists, execute_sql_file, test_connection
    print(f"   ✅ SUCCESS! Import worked!")
    
    # Try to test connection
    print(f"\n🔌 Testing database connection:")
    try:
        if test_connection():
            print(f"   ✅ Database connection successful!")
        else:
            print(f"   ❌ Database connection failed")
    except Exception as e:
        print(f"   ❌ Connection test error: {str(e)}")
        
except ImportError as e:
    print(f"   ❌ IMPORT ERROR: {str(e)}")
    print(f"\n💡 This is why you're seeing the warning!")
    
except Exception as e:
    print(f"   ❌ OTHER ERROR: {type(e).__name__}: {str(e)}")

# 6. Check dependencies
print(f"\n📦 Checking required packages:")
required_packages = ['pymysql', 'pandas', 'python-dotenv']
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"   ✅ {package} installed")
    except ImportError:
        print(f"   ❌ {package} NOT installed")

# 7. Check .env file
print(f"\n⚙️  Checking .env file:")
env_file = Path(".env")
if env_file.exists():
    print(f"   ✅ .env file EXISTS")
    print(f"\n   📋 .env contents:")
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Hide password value
                if 'PASSWORD' in line:
                    key = line.split('=')[0]
                    print(f"      {key}=***")
                else:
                    print(f"      {line}")
else:
    print(f"   ⚠️  .env file MISSING")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
