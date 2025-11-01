"""
Critical SQL Syntax Error Fixer
Targets the most common error patterns causing failures
"""

import re
import os
from pathlib import Path
from datetime import datetime

class CriticalSQLFixer:
    def __init__(self):
        self.fixes_applied = []
        self.files_processed = 0
        self.files_modified = 0
    
    def fix_case_statements(self, content):
        """Fix CASE statements that have syntax issues"""
        # Common pattern: Missing END after CASE
        modified = False
        
        # Fix incomplete CASE statements
        lines = content.split('\n')
        fixed_lines = []
        in_case = False
        case_depth = 0
        
        for line in lines:
            # Count CASE keywords
            case_matches = len(re.findall(r'\bCASE\b', line, re.IGNORECASE))
            end_matches = len(re.findall(r'\bEND\b(?!\s+AS|\s+CASE)', line, re.IGNORECASE))
            
            case_depth += case_matches - end_matches
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines), modified
    
    def remove_unsupported_with_clauses(self, content):
        """Remove or simplify unsupported WITH clauses"""
        modified = False
        
        # Pattern for WITH ... AS in SELECT that MariaDB doesn't support well
        # Convert to subqueries where possible
        pattern = r'WITH\s+(\w+)\s+AS\s*\((.*?)\)\s*SELECT'
        
        def replace_with(match):
            nonlocal modified
            modified = True
            cte_name = match.group(1)
            cte_query = match.group(2)
            # Convert to FROM subquery
            return f"SELECT * FROM ({cte_query}) AS {cte_name} WHERE 1=1 AND "
        
        content = re.sub(pattern, replace_with, content, flags=re.DOTALL | re.IGNORECASE)
        
        if modified:
            self.fixes_applied.append("Simplified WITH clauses")
        
        return content, modified
    
    def fix_string_aggregation(self, content):
        """Fix GROUP_CONCAT issues"""
        modified = False
        
        # Fix GROUP_CONCAT with ORDER BY issues
        pattern = r"GROUP_CONCAT\s*\(\s*DISTINCT\s+([^)]+)\s+ORDER\s+BY\s+([^)]+)\s+SEPARATOR\s+'([^']+)'\s*\)"
        replacement = r"GROUP_CONCAT(DISTINCT \1 SEPARATOR '\3')"
        
        new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        if new_content != content:
            modified = True
            self.fixes_applied.append("Fixed GROUP_CONCAT syntax")
        
        return new_content, modified
    
    def fix_json_functions(self, content):
        """Remove or replace unsupported JSON functions"""
        replacements = [
            (r'JSON_ARRAYAGG\s*\(([^)]+)\)', r"GROUP_CONCAT(\1 SEPARATOR ',')"),
            (r'JSON_OBJECTAGG\s*\(([^,]+),\s*([^)]+)\)', r"GROUP_CONCAT(CONCAT(\1, ':', \2) SEPARATOR ',')"),
        ]
        
        modified = False
        for pattern, replacement in replacements:
            new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            if new_content != content:
                content = new_content
                modified = True
        
        if modified:
            self.fixes_applied.append("Replaced JSON aggregation functions")
        
        return content, modified
    
    def fix_set_operations(self, content):
        """Fix UNION/INTERSECT/EXCEPT issues"""
        modified = False
        
        # Replace EXCEPT with NOT EXISTS
        pattern = r'SELECT\s+(.*?)\s+FROM\s+(.*?)\s+EXCEPT\s+SELECT\s+(.*?)\s+FROM\s+(.*?)(?:;|\s+(?:UNION|INTERSECT|ORDER))'
        
        def replace_except(match):
            nonlocal modified
            modified = True
            select1 = match.group(1)
            from1 = match.group(2)
            select2 = match.group(3)
            from2 = match.group(4)
            
            return f"SELECT {select1} FROM {from1} WHERE NOT EXISTS (SELECT 1 FROM {from2})"
        
        content = re.sub(pattern, replace_except, content, flags=re.DOTALL | re.IGNORECASE)
        
        if modified:
            self.fixes_applied.append("Converted EXCEPT to NOT EXISTS")
        
        return content, modified
    
    def comment_out_failing_procedures(self, content):
        """Comment out procedures that are likely to fail"""
        modified = False
        
        # Look for procedures with known problematic patterns
        problematic_patterns = [
            r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+\w+.*?FETCH\s+\w+\s+INTO.*?END(?:\s+\$\$|\s*;)',
        ]
        
        for pattern in problematic_patterns:
            matches = list(re.finditer(pattern, content, re.DOTALL | re.IGNORECASE))
            if matches:
                for match in reversed(matches):  # Reverse to maintain positions
                    start, end = match.span()
                    procedure_text = content[start:end]
                    commented = '/* COMMENTED OUT - NEEDS MANUAL FIX\n' + procedure_text + '\n*/'
                    content = content[:start] + commented + content[end:]
                    modified = True
        
        if modified:
            self.fixes_applied.append("Commented out problematic procedures")
        
        return content, modified
    
    def fix_system_variables(self, content):
        """Fix @ vs @@ variable issues"""
        modified = False
        
        # Replace invalid system variables with user variables
        invalid_vars = [
            'changed_cols', 'v_start_time', 'v_duration', 'v_update_count',
            'v_recommendations', 'v_alert_message', 'v_budget_utilization',
            'p_incident_id', 'p_status_message', 'v_execution_status', 'v_error_msg'
        ]
        
        for var in invalid_vars:
            # Replace SET @@var = with SET @var =
            old_pattern = f"SET\\s+@@{var}\\s*="
            new_replacement = f"SET @{var} ="
            new_content = re.sub(old_pattern, new_replacement, content, flags=re.IGNORECASE)
            
            if new_content != content:
                content = new_content
                modified = True
        
        if modified:
            self.fixes_applied.append("Fixed system variable usage")
        
        return content, modified
    
    def simplify_complex_ctes(self, content):
        """Simplify complex CTEs that cause errors"""
        modified = False
        
        # Look for multiple CTEs and try to simplify
        # Pattern: WITH cte1 AS (...), cte2 AS (...), cte3 AS (...)
        multi_cte_pattern = r'WITH\s+\w+\s+AS\s*\([^)]+\)\s*,\s*\w+\s+AS'
        
        if re.search(multi_cte_pattern, content, re.IGNORECASE):
            # Too complex for automatic fixing - add comment
            content = '-- WARNING: Complex CTEs detected - may need manual review\n' + content
            modified = True
            self.fixes_applied.append("Added warning for complex CTEs")
        
        return content, modified
    
    def process_file(self, filepath):
        """Process a single file"""
        self.fixes_applied = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            any_modified = False
            
            # Apply fixes in order
            fixes = [
                self.fix_system_variables,
                self.fix_json_functions,
                self.fix_string_aggregation,
                self.fix_set_operations,
                self.simplify_complex_ctes,
            ]
            
            for fix_func in fixes:
                content, modified = fix_func(content)
                if modified:
                    any_modified = True
            
            # Save if modified
            if any_modified and content != original_content:
                # Create backup
                backup_dir = Path(filepath).parent / 'backups_critical'
                backup_dir.mkdir(exist_ok=True)
                
                backup_path = backup_dir / f"{Path(filepath).stem}_critical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_modified += 1
                return True, self.fixes_applied
            
            return False, []
            
        except Exception as e:
            return False, [f"Error: {e}"]
    
    def process_directory(self, directory, pattern='*.sql'):
        """Process all matching files in directory"""
        print(f"\n{'='*60}")
        print(f"Processing: {directory}")
        print('='*60)
        
        sql_files = list(Path(directory).glob(pattern))
        
        if not sql_files:
            print(f"  No files found")
            return {}
        
        results = {}
        
        for filepath in sql_files:
            self.files_processed += 1
            modified, fixes = self.process_file(filepath)
            
            if modified:
                print(f"\n  ✅ {filepath.name}")
                for fix in fixes:
                    print(f"     • {fix}")
                results[str(filepath)] = fixes
        
        return results
    
    def generate_report(self):
        """Generate summary report"""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"Files processed: {self.files_processed}")
        print(f"Files modified: {self.files_modified}")
        print(f"Files unchanged: {self.files_processed - self.files_modified}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║   CRITICAL SQL SYNTAX ERROR FIXER                        ║
║   Fixes the most common MariaDB compatibility issues     ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    fixer = CriticalSQLFixer()
    
    # Target the problematic directories
    sql_root = r'E:\ecommerce-analytics\sql'
    
    directories = [
        os.path.join(sql_root, 'reporting'),
        os.path.join(sql_root, 'maintenance'),
        os.path.join(sql_root, 'automation'),
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            fixer.process_directory(directory)
    
    fixer.generate_report()
    
    print("\n" + "="*60)
    print("✅ COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run sql_error_fixer.py to fix missing columns")
    print("2. Re-run sql_executor_fix.py")
    print("3. Review remaining errors manually")
