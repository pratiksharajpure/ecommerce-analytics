"""
CSV Data Diagnostic Tool
Run this to check what's wrong with your CSV files
"""

import pandas as pd
from pathlib import Path

# CSV file locations
csv_files = {
    'customers': 'sample_data/core_data/customers.csv',
    'products': 'sample_data/core_data/products.csv',
    'orders': 'sample_data/core_data/orders.csv',
    'inventory': 'sample_data/core_data/inventory.csv',
    'vendors': 'sample_data/core_data/vendors.csv',
    'campaigns': 'sample_data/marketing_data/campaigns.csv',
    'reviews': 'sample_data/operational_data/reviews.csv',
    'returns': 'sample_data/operational_data/returns.csv',
    'transactions': 'sample_data/financial_data/transactions.csv',
}

print("=" * 60)
print("CSV FILE DIAGNOSTIC REPORT")
print("=" * 60)

for name, path in csv_files.items():
    print(f"\n📁 Checking: {name}")
    print(f"   Path: {path}")
    
    # Check if file exists
    if not Path(path).exists():
        print("   ❌ FILE NOT FOUND")
        continue
    
    try:
        # Try to read the CSV
        df = pd.read_csv(path)
        
        print(f"   ✅ File loaded successfully")
        print(f"   📊 Rows: {len(df):,} | Columns: {len(df.columns)}")
        print(f"   📋 Columns: {', '.join(df.columns[:5])}" + ("..." if len(df.columns) > 5 else ""))
        
        # Check for issues
        issues = []
        
        # Check if completely empty
        if len(df) == 0:
            issues.append("⚠️  EMPTY FILE - No data rows")
        
        # Check for all NaN columns
        nan_cols = df.columns[df.isna().all()].tolist()
        if nan_cols:
            issues.append(f"⚠️  Empty columns: {', '.join(nan_cols)}")
        
        # Check first few rows for sample
        if len(df) > 0:
            print(f"\n   First 3 rows preview:")
            print(df.head(3).to_string(index=False, max_cols=5, max_colwidth=20))
        
        # Special checks per file type
        if name == 'orders':
            if 'order_date' in df.columns or 'created_at' in df.columns or 'date' in df.columns:
                date_col = next((c for c in ['order_date', 'created_at', 'date'] if c in df.columns), None)
                print(f"\n   📅 Date column: {date_col}")
                print(f"      Sample values: {df[date_col].head(3).tolist()}")
            
            if 'total_amount' in df.columns or 'amount' in df.columns:
                amt_col = next((c for c in ['total_amount', 'amount'] if c in df.columns), None)
                print(f"   💰 Amount column: {amt_col}")
                print(f"      Sample values: {df[amt_col].head(3).tolist()}")
                print(f"      Data type: {df[amt_col].dtype}")
        
        if issues:
            print(f"\n   ⚠️  ISSUES FOUND:")
            for issue in issues:
                print(f"      {issue}")
    
    except Exception as e:
        print(f"   ❌ ERROR READING FILE: {str(e)[:100]}")
        print(f"      This might be corrupted or wrong format")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
print("\n💡 Common Issues:")
print("   1. CSV files are empty (0 rows)")
print("   2. Wrong column names")
print("   3. Date columns in wrong format")
print("   4. Amount/price columns as text instead of numbers")
print("   5. Files corrupted or wrong encoding")
print("\n📝 Next Steps:")
print("   - If files are empty, regenerate them with proper data")
print("   - If column names wrong, update COLUMN_MAPPINGS in Home.py")
print("   - If dates wrong format, use: YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
