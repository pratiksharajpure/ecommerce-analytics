"""
Advanced SQL File Syntax Fixer
Fixes MariaDB compatibility issues in SQL files
"""

import re
import os
from pathlib import Path
from datetime import datetime

class SQLFileFixer:
    def __init__(self, base_directory='.'):
        self.base_directory = base_directory
        self.fixes_log = []
        self.files_processed = 0
        self.files_modified = 0
    
    def fix_qualify_clause(self, content):
        """Convert QUALIFY clause to subquery WHERE clause"""
        # Pattern: anything with QUALIFY
        pattern = r'(SELECT.*?)(QUALIFY\s+[^;]+)(;)'
        
        def replace_qualify(match):
            select_part = match.group(1)
            qualify_part = match.group(2)
            semicolon = match.group(3)
            
            # Extract the condition from QUALIFY
            condition = re.sub(r'QUALIFY\s+', '', qualify_part)
            
            # Wrap in subquery
            return f"SELECT * FROM (\n{select_part}\n) qualified\nWHERE {condition}{semicolon}"
        
        new_content = re.sub(pattern, replace_qualify, content, flags=re.DOTALL | re.IGNORECASE)
        
        if new_content != content:
            self.fixes_log.append("Fixed QUALIFY clause")
            return new_content, True
        return content, False
    
    def fix_emoji_characters(self, content):
        """Replace emoji with text equivalents"""
        replacements = {
            '🟢': "'GREEN'",
            '🟡': "'YELLOW'", 
            '🔴': "'RED'",
            '✅': "'SUCCESS'",
            '⚠️': "'WARNING'",
            '❌': "'ERROR'",
            '📊': "'CHART'",
            '🔍': "'SEARCH'",
            '💰': "'MONEY'",
            '📈': "'TRENDING_UP'",
            '📉': "'TRENDING_DOWN'"
        }
        
        modified = False
        for emoji, text in replacements.items():
            if emoji in content:
                content = content.replace(emoji, text)
                modified = True
        
        if modified:
            self.fixes_log.append("Replaced emoji characters")
        
        return content, modified
    
    def fix_date_trunc(self, content):
        """Replace DATE_TRUNC with MariaDB equivalent"""
        patterns = [
            (r"DATE_TRUNC\('year',\s*([^)]+)\)", r"DATE_FORMAT(\1, '%Y-01-01')"),
            (r"DATE_TRUNC\('quarter',\s*([^)]+)\)", r"DATE_FORMAT(\1, '%Y-%m-01')"),
            (r"DATE_TRUNC\('month',\s*([^)]+)\)", r"DATE_FORMAT(\1, '%Y-%m-01')"),
            (r"DATE_TRUNC\('week',\s*([^)]+)\)", r"DATE_FORMAT(\1, '%Y-%m-%d')"),
            (r"DATE_TRUNC\('day',\s*([^)]+)\)", r"DATE(\1)"),
        ]
        
        modified = False
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            if new_content != content:
                content = new_content
                modified = True
        
        if modified:
            self.fixes_log.append("Fixed DATE_TRUNC function")
        
        return content, modified
    
    def fix_interval_syntax(self, content):
        """Fix INTERVAL syntax for MariaDB"""
        # PostgreSQL style: INTERVAL '5 days'
        # MariaDB style: INTERVAL 5 DAY
        
        pattern = r"INTERVAL\s+'(\d+)\s+(\w+)'"
        replacement = r"INTERVAL \1 \2"
        
        new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        if new_content != content:
            self.fixes_log.append("Fixed INTERVAL syntax")
            return new_content, True
        
        return content, False
    
    def fix_window_functions_in_having(self, content):
        """Move window functions out of HAVING clause"""
        # This is complex - log for manual review
        if re.search(r'HAVING.*?(ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|FIRST_VALUE|LAST_VALUE)\s*\(', 
                     content, re.IGNORECASE):
            self.fixes_log.append("⚠️ Window function in HAVING - needs manual review")
        
        return content, False
    
    def fix_variable_declarations(self, content):
        """Fix variable declaration issues in procedures"""
        # Find procedures with SET before DECLARE
        pattern = r'(CREATE.*?PROCEDURE.*?BEGIN)(.*?)(DECLARE.*?)(SET\s+@?\w+\s*=)'
        
        def reorder_declarations(match):
            header = match.group(1)
            middle = match.group(2)
            declares = match.group(3)
            set_stmt = match.group(4)
            
            # Move DECLARE before SET
            return f"{header}\n{declares}\n{middle}\n{set_stmt}"
        
        new_content = re.sub(pattern, reorder_declarations, content, flags=re.DOTALL | re.IGNORECASE)
        
        if new_content != content:
            self.fixes_log.append("Fixed variable declaration order")
            return new_content, True
        
        return content, False
    
    def fix_regexp_patterns(self, content):
        """Fix invalid regex patterns"""
        # Common issue: unescaped characters
        invalid_patterns = [
            (r"REGEXP '\*'", r"REGEXP '\\*'"),
            (r"REGEXP '\+'", r"REGEXP '\\+'"),
            (r"REGEXP '\?'", r"REGEXP '\\?'"),
        ]
        
        modified = False
        for pattern, replacement in invalid_patterns:
            new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            if new_content != content:
                content = new_content
                modified = True
        
        if modified:
            self.fixes_log.append("Fixed REGEXP patterns")
        
        return content, modified
    
    def fix_column_count_mismatch(self, content):
        """Detect and log column count mismatches in UNION"""
        unions = re.finditer(r'(SELECT.*?)\s+UNION\s+(SELECT.*?)(?:;|\s+UNION)', 
                            content, re.IGNORECASE | re.DOTALL)
        
        for match in unions:
            select1 = match.group(1)
            select2 = match.group(2)
            
            # Simple column count
            cols1 = select1.count(',') + 1
            cols2 = select2.count(',') + 1
            
            if cols1 != cols2:
                self.fixes_log.append(f"⚠️ UNION column mismatch: {cols1} vs {cols2} - needs manual review")
        
        return content, False
    
    def add_missing_delimiters(self, content):
        """Add DELIMITER statements for procedures/functions"""
        # Check if content has procedures but no delimiter
        has_procedure = re.search(r'CREATE.*?(PROCEDURE|FUNCTION)', content, re.IGNORECASE)
        has_delimiter = re.search(r'DELIMITER', content, re.IGNORECASE)
        
        if has_procedure and not has_delimiter:
            # Add delimiter at start
            content = "DELIMITER $$\n\n" + content
            
            # Replace ; with $$ in procedure bodies
            content = re.sub(
                r'(CREATE.*?(?:PROCEDURE|FUNCTION).*?END);',
                r'\1$$',
                content,
                flags=re.DOTALL | re.IGNORECASE
            )
            
            # Add delimiter reset at end
            content = content + "\n\nDELIMITER ;\n"
            
            self.fixes_log.append("Added DELIMITER statements")
            return content, True
        
        return content, False
    
    def fix_default_timestamp(self, content):
        """Fix invalid default timestamp values"""
        # Invalid: DEFAULT '0000-00-00 00:00:00'
        # Valid: DEFAULT CURRENT_TIMESTAMP or NULL
        
        pattern = r"DEFAULT\s+'0000-00-00.*?'"
        replacement = "DEFAULT NULL"
        
        new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        if new_content != content:
            self.fixes_log.append("Fixed invalid default timestamps")
            return new_content, True
        
        return content, False
    
    def process_file(self, filepath):
        """Process a single SQL file"""
        self.fixes_log = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            any_modified = False
            
            # Apply all fixes
            fixes = [
                self.fix_emoji_characters,
                self.fix_date_trunc,
                self.fix_interval_syntax,
                self.fix_qualify_clause,
                self.fix_regexp_patterns,
                self.fix_default_timestamp,
                self.fix_variable_declarations,
                self.fix_window_functions_in_having,
                self.fix_column_count_mismatch,
            ]
            
            for fix_func in fixes:
                content, modified = fix_func(content)
                if modified:
                    any_modified = True
            
            # Save if modified
            if any_modified and content != original_content:
                # Create backup
                backup_dir = Path(filepath).parent / 'backups'
                backup_dir.mkdir(exist_ok=True)
                
                backup_path = backup_dir / f"{Path(filepath).stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_modified += 1
                return True, self.fixes_log
            
            return False, self.fixes_log
            
        except Exception as e:
            return False, [f"Error: {e}"]
    
    def process_directory(self, directory):
        """Process all SQL files in directory"""
        print("\n" + "="*60)
        print(f"🔧 PROCESSING SQL FILES IN: {directory}")
        print("="*60)
        
        sql_files = list(Path(directory).rglob('*.sql'))
        
        if not sql_files:
            print(f"  ⚠️  No SQL files found in {directory}")
            return
        
        results = {}
        
        for filepath in sql_files:
            self.files_processed += 1
            modified, fixes = self.process_file(filepath)
            
            if modified:
                print(f"\n  ✅ {filepath.name}")
                for fix in fixes:
                    print(f"     • {fix}")
                results[str(filepath)] = fixes
            else:
                if fixes:  # Has warnings/notes
                    print(f"\n  ℹ️  {filepath.name}")
                    for fix in fixes:
                        print(f"     • {fix}")
        
        return results
    
    def generate_report(self, results):
        """Generate detailed report"""
        print("\n" + "="*60)
        print("📊 PROCESSING REPORT")
        print("="*60)
        
        print(f"\nFiles processed: {self.files_processed}")
        print(f"Files modified: {self.files_modified}")
        print(f"Files unchanged: {self.files_processed - self.files_modified}")
        
        # Save detailed report
        report_file = f"sql_fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # FIX: Add encoding='utf-8' here
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SQL FILE FIX REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Files processed: {self.files_processed}\n")
            f.write(f"Files modified: {self.files_modified}\n\n")
            
            f.write("DETAILS:\n")
            f.write("-" * 60 + "\n")
            for filepath, fixes in results.items():
                f.write(f"\n{filepath}:\n")
                for fix in fixes:
                    f.write(f"  • {fix}\n")
        
        print(f"\n📄 Detailed report saved to: {report_file}")

# ====================
# MAIN EXECUTION
# ====================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║   SQL FILE SYNTAX FIXER                                  ║
║   Fixes MariaDB compatibility issues                     ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    fixer = SQLFileFixer()
    
    # Define the SQL root directory
    sql_root = r'E:\ecommerce-analytics\sql'
    
    # Check if the sql directory exists
    if not os.path.exists(sql_root):
        print(f"❌ SQL directory not found: {sql_root}")
        print("\nPlease update the sql_root variable in the script to match your directory structure.")
        sys.exit(1)
    
    # Process subdirectories
    directories = [
        os.path.join(sql_root, 'setup'),
        os.path.join(sql_root, 'core_analysis'),
        os.path.join(sql_root, 'advanced_analysis'),
        os.path.join(sql_root, 'reporting'),
        os.path.join(sql_root, 'maintenance'),
        os.path.join(sql_root, 'automation')
    ]
    
    all_results = {}
    
    # Also check if sql root itself has SQL files
    print(f"\n🔍 Scanning SQL directory: {sql_root}")
    
    for directory in directories:
        if os.path.exists(directory):
            results = fixer.process_directory(directory)
            if results:
                all_results.update(results)
        else:
            # Try without the subdirectory - maybe files are directly in sql/
            print(f"  ℹ️  Subdirectory not found: {directory}")
    
    # If no subdirectories found, process the sql root directly
    if not all_results and os.path.exists(sql_root):
        print(f"\n🔍 Processing SQL files directly in: {sql_root}")
        results = fixer.process_directory(sql_root)
        if results:
            all_results.update(results)
    
    # Generate report
    fixer.generate_report(all_results)
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)
    print("\n💡 Backups created in 'backups' subdirectories")
    print("💡 Review the report for manual fixes needed")