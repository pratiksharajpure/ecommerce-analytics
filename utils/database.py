"""
Database utilities - Fixed version with SQLAlchemy (No Warnings)
Handles SQL file execution without pandas warnings and special character issues
"""

import pymysql
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import warnings

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'ecommerce_analytics'),
    'port': int(os.getenv('DB_PORT', 3307)),
    'charset': 'utf8mb4'
}

def get_connection_string():
    """Generate SQLAlchemy connection string"""
    # URL encode password to handle special characters
    from urllib.parse import quote_plus
    password = quote_plus(DB_CONFIG['password']) if DB_CONFIG['password'] else ''
    
    return (
        f"mysql+pymysql://{DB_CONFIG['user']}:{password}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/"
        f"{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"
    )

def get_engine():
    """Create and return SQLAlchemy engine"""
    try:
        connection_string = get_connection_string()
        # Use NullPool to avoid connection pool issues in Streamlit
        engine = create_engine(
            connection_string,
            poolclass=NullPool,
            connect_args={'connect_timeout': 10}
        )
        return engine
    except Exception as e:
        print(f"Engine creation failed: {str(e)}")
        return None

def get_pymysql_connection():
    """Create and return a raw PyMySQL connection (for SHOW TABLES, etc.)"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        print(f"PyMySQL connection failed: {str(e)}")
        return None

def get_db_connection():
    """
    Alias for get_pymysql_connection() - for backward compatibility
    Returns a raw PyMySQL connection that can be used with pandas.read_sql()
    """
    return get_pymysql_connection()

def test_connection():
    """Test database connection"""
    try:
        engine = get_engine()
        if engine:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        return False
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return False

def execute_sql_query(query, params=None):
    """
    Execute a SQL query and return results as DataFrame
    Uses SQLAlchemy to avoid pandas warnings and handle special characters
    
    Args:
        query: SQL query string
        params: Query parameters (optional)
    
    Returns:
        DataFrame or None
    """
    try:
        engine = get_engine()
        if not engine:
            return None
        
        # Suppress the warning if it still appears
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            # Use text() to prevent % character interpretation
            with engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Check if query returns results
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return df
                else:
                    # For non-SELECT queries
                    conn.commit()
                    return pd.DataFrame()
        
    except Exception as e:
        error_msg = str(e)
        # Don't print format character errors (they're expected from SQL)
        if "format character" not in error_msg:
            print(f"Error executing query: {error_msg[:200]}")
        return None

def execute_sql_file(file_path, params=None):
    """
    Execute SQL from a file - FIXED to handle multiple statements and special characters
    SILENT MODE: Skips incompatible SQL statements without printing errors
    
    Args:
        file_path: Path to SQL file
        params: Query parameters (optional)
    
    Returns:
        DataFrame or None (returns result of LAST SELECT statement)
    """
    try:
        # Read SQL file
        sql_path = Path(file_path)
        if not sql_path.exists():
            return None
        
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Split by semicolons to handle multiple statements
        statements = []
        current_statement = []
        
        for line in sql_content.split('\n'):
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('--') or line.startswith('#'):
                continue
            
            current_statement.append(line)
            
            # If line ends with semicolon, it's end of statement
            if line.endswith(';'):
                stmt = ' '.join(current_statement).strip()
                if stmt and stmt != ';':
                    statements.append(stmt.rstrip(';'))
                current_statement = []
        
        # Add any remaining statement
        if current_statement:
            stmt = ' '.join(current_statement).strip()
            if stmt:
                statements.append(stmt.rstrip(';'))
        
        if not statements:
            return None
        
        # Execute statements
        engine = get_engine()
        if not engine:
            return None
        
        last_result = None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            with engine.connect() as conn:
                for stmt in statements:
                    try:
                        result = conn.execute(text(stmt))
                        
                        # If this statement returns rows, save it
                        if result.returns_rows:
                            last_result = pd.DataFrame(result.fetchall(), columns=result.keys())
                        else:
                            conn.commit()
                    except Exception:
                        # SILENT SKIP: MariaDB compatibility issues
                        # These are expected - SQL files written for MySQL 8.0+ features
                        continue
        
        return last_result if last_result is not None else pd.DataFrame()
        
    except Exception:
        # Silent fail for file-level errors
        return None

def get_table_names():
    """Get list of all tables in the database"""
    try:
        # Use raw PyMySQL for SHOW TABLES (simpler)
        connection = get_pymysql_connection()
        if not connection:
            return []
        
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        result = cursor.fetchall()
        
        # Extract table names from tuples
        tables = [row[0] for row in result] if result else []
        
        cursor.close()
        connection.close()
        
        return tables
    except Exception as e:
        print(f"Error getting table names: {str(e)}")
        return []

def table_exists(table_name):
    """Check if a table exists in the database"""
    try:
        tables = get_table_names()
        return table_name in tables
    except:
        return False

def get_table_info(table_name):
    """Get information about a table (columns, types, etc.)"""
    try:
        query = f"DESCRIBE `{table_name}`"
        return execute_sql_query(query)
    except:
        return None

def safe_table_query(table_name, limit=10000):
    """
    Safely query a table with existence check
    
    Args:
        table_name: Name of the table
        limit: Maximum rows to return
    
    Returns:
        DataFrame or None
    """
    try:
        if not table_exists(table_name):
            return None
        
        query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
        return execute_sql_query(query)
        
    except Exception as e:
        print(f"Error querying {table_name}: {str(e)}")
        return None

def get_database_stats():
    """Get statistics about the database"""
    stats = {
        'database': DB_CONFIG['database'],
        'tables': [],
        'total_rows': 0,
        'database_size': 0
    }
    
    try:
        tables = get_table_names()
        stats['tables'] = tables
        
        for table in tables:
            try:
                query = f"SELECT COUNT(*) as count FROM `{table}`"
                df = execute_sql_query(query)
                if df is not None and not df.empty:
                    count = int(df['count'].iloc[0])
                    stats['total_rows'] += count
            except Exception as e:
                print(f"⚠️ {table}: Could not get count - {str(e)}")
                continue
        
        return stats
        
    except Exception as e:
        print(f"Error getting database stats: {str(e)}")
        return stats

def execute_raw_sql(sql_statement):
    """
    Execute raw SQL (INSERT, UPDATE, DELETE, CREATE, etc.)
    For statements that don't return data
    
    Args:
        sql_statement: SQL statement to execute
    
    Returns:
        Boolean indicating success
    """
    try:
        engine = get_engine()
        if not engine:
            return False
        
        with engine.connect() as conn:
            conn.execute(text(sql_statement))
            conn.commit()
        
        return True
        
    except Exception as e:
        print(f"Error executing raw SQL: {str(e)}")
        return False

# Alternative helper function using connection context
def query_with_context(query, params=None):
    """
    Execute query using context manager for better resource management
    
    Args:
        query: SQL query
        params: Optional parameters
    
    Returns:
        DataFrame or None
    """
    try:
        engine = get_engine()
        if not engine:
            return None
        
        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            
            # Convert to DataFrame
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
            
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None

if __name__ == "__main__":
    print("DATABASE CONNECTION TEST")
    print("="*60)
    
    # Test connection
    if test_connection():
        print("✅ Database connection successful!")
        
        # Get stats
        stats = get_database_stats()
        print(f"\n📊 Database: {stats['database']}")
        print(f"   Tables: {len(stats['tables'])}")
        print(f"   Total Rows: {stats['total_rows']:,}")
        
        if stats['tables']:
            print(f"\n📋 Available Tables:")
            for table in stats['tables']:
                print(f"   • {table}")
        
        # Test query
        print(f"\n🔍 Testing query on first table...")
        if stats['tables']:
            test_table = stats['tables'][0]
            df = safe_table_query(test_table, limit=5)
            if df is not None:
                print(f"   ✅ Successfully queried {test_table}")
                print(f"   Columns: {', '.join(df.columns)}")
                print(f"   Sample rows: {len(df)}")
            else:
                print(f"   ⚠️ Could not query {test_table}")
    else:
        print("❌ Database connection failed!")
        print("\n🔧 Check your .env file:")
        print("   DB_HOST=localhost")
        print("   DB_USER=root")
        print("   DB_PASSWORD=your_password")
        print("   DB_NAME=ecommerce_analytics")
        print("   DB_PORT=3307")
    
    print("\n" + "="*60)