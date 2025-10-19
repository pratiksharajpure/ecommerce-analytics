#!/bin/bash
# Setup script to fix database.py import issues
# Run this script in your project root directory

echo "================================================"
echo "🔧 Fixing Database Import Structure"
echo "================================================"

# Check if database.py exists
if [ ! -f "database.py" ] && [ ! -f "utils/database.py" ]; then
    echo "❌ Error: database.py not found!"
    echo "Please make sure database.py exists in your project"
    exit 1
fi

# Create utils directory if it doesn't exist
if [ ! -d "utils" ]; then
    echo "📁 Creating utils/ directory..."
    mkdir -p utils
    echo "✅ utils/ created"
else
    echo "✅ utils/ already exists"
fi

# Create __init__.py if it doesn't exist
if [ ! -f "utils/__init__.py" ]; then
    echo "📝 Creating utils/__init__.py..."
    touch utils/__init__.py
    echo "✅ utils/__init__.py created"
else
    echo "✅ utils/__init__.py already exists"
fi

# Move database.py if it's in root
if [ -f "database.py" ] && [ ! -f "utils/database.py" ]; then
    echo "📦 Moving database.py to utils/..."
    mv database.py utils/
    echo "✅ database.py moved to utils/"
elif [ -f "utils/database.py" ]; then
    echo "✅ utils/database.py already in correct location"
fi

# Verify structure
echo ""
echo "📋 Verifying project structure:"
echo "---"

if [ -d "utils" ]; then
    echo "✅ utils/ folder exists"
else
    echo "❌ utils/ folder missing"
fi

if [ -f "utils/__init__.py" ]; then
    echo "✅ utils/__init__.py exists"
else
    echo "❌ utils/__init__.py missing"
fi

if [ -f "utils/database.py" ]; then
    echo "✅ utils/database.py exists"
else
    echo "❌ utils/database.py missing"
fi

# Check Python packages
echo ""
echo "📦 Checking Python packages:"
echo "---"

python3 -c "import pymysql" 2>/dev/null && echo "✅ pymysql installed" || echo "❌ pymysql not installed - run: pip install pymysql"
python3 -c "import pandas" 2>/dev/null && echo "✅ pandas installed" || echo "❌ pandas not installed - run: pip install pandas"
python3 -c "import dotenv" 2>/dev/null && echo "✅ python-dotenv installed" || echo "❌ python-dotenv not installed - run: pip install python-dotenv"
python3 -c "import streamlit" 2>/dev/null && echo "✅ streamlit installed" || echo "❌ streamlit not installed - run: pip install streamlit"

# Test import
echo ""
echo "🧪 Testing import:"
echo "---"

python3 -c "from utils.database import test_connection; print('✅ Import successful!')" 2>/dev/null && import_ok=1 || import_ok=0

if [ $import_ok -eq 1 ]; then
    echo "✅ Database module imports correctly!"
    
    # Test connection
    echo ""
    echo "🔌 Testing database connection:"
    echo "---"
    python3 -c "from utils.database import test_connection; result = test_connection(); print('✅ Connection successful!' if result else '❌ Connection failed - check .env file')" 2>/dev/null || echo "❌ Connection test failed"
else
    echo "❌ Import failed - there may be other issues"
fi

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Make sure your .env file has correct database credentials"
echo "2. Run: streamlit run Home.py"
echo ""