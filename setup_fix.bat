@echo off
REM Setup script to fix database.py import issues (Windows)
REM Run this script in your project root directory

echo ================================================
echo 🔧 Fixing Database Import Structure
echo ================================================
echo.

REM Check if database.py exists
if not exist "database.py" if not exist "utils\database.py" (
    echo ❌ Error: database.py not found!
    echo Please make sure database.py exists in your project
    pause
    exit /b 1
)

REM Create utils directory if it doesn't exist
if not exist "utils" (
    echo 📁 Creating utils\ directory...
    mkdir utils
    echo ✅ utils\ created
) else (
    echo ✅ utils\ already exists
)

REM Create __init__.py if it doesn't exist
if not exist "utils\__init__.py" (
    echo 📝 Creating utils\__init__.py...
    type nul > utils\__init__.py
    echo ✅ utils\__init__.py created
) else (
    echo ✅ utils\__init__.py already exists
)

REM Move database.py if it's in root
if exist "database.py" if not exist "utils\database.py" (
    echo 📦 Moving database.py to utils\...
    move database.py utils\
    echo ✅ database.py moved to utils\
) else if exist "utils\database.py" (
    echo ✅ utils\database.py already in correct location
)

REM Verify structure
echo.
echo 📋 Verifying project structure:
echo ---

if exist "utils\" (
    echo ✅ utils\ folder exists
) else (
    echo ❌ utils\ folder missing
)

if exist "utils\__init__.py" (
    echo ✅ utils\__init__.py exists
) else (
    echo ❌ utils\__init__.py missing
)

if exist "utils\database.py" (
    echo ✅ utils\database.py exists
) else (
    echo ❌ utils\database.py missing
)

REM Check Python packages
echo.
echo 📦 Checking Python packages:
echo ---

python -c "import pymysql" 2>nul && echo ✅ pymysql installed || echo ❌ pymysql not installed - run: pip install pymysql
python -c "import pandas" 2>nul && echo ✅ pandas installed || echo ❌ pandas not installed - run: pip install pandas
python -c "import dotenv" 2>nul && echo ✅ python-dotenv installed || echo ❌ python-dotenv not installed - run: pip install python-dotenv
python -c "import streamlit" 2>nul && echo ✅ streamlit installed || echo ❌ streamlit not installed - run: pip install streamlit

REM Test import
echo.
echo 🧪 Testing import:
echo ---

python -c "from utils.database import test_connection; print('✅ Import successful!')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Database module imports correctly!
    
    REM Test connection
    echo.
    echo 🔌 Testing database connection:
    echo ---
    python -c "from utils.database import test_connection; result = test_connection(); print('✅ Connection successful!' if result else '❌ Connection failed - check .env file')" 2>nul
) else (
    echo ❌ Import failed - there may be other issues
)

echo.
echo ================================================
echo ✅ Setup Complete!
echo ================================================
echo.
echo Next steps:
echo 1. Make sure your .env file has correct database credentials
echo 2. Run: streamlit run Home.py
echo.

pause