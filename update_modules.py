#!/usr/bin/env python3
"""
Script to update all modules to use safe_sympify instead of sp.sympify
"""

import os
import re
from pathlib import Path

def update_module_file(file_path):
    """Update a single module file to use safe_sympify"""
    print(f"Updating {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Add import for safe_sympify if not present
    if 'from utils.safe_sympify import safe_sympify' not in content:
        # Find the imports section and add the import
        import_pattern = r'(from utils\.display_utils import[^\n]*\n)'
        if re.search(import_pattern, content):
            content = re.sub(import_pattern, r'\1from utils.safe_sympify import safe_sympify\n', content)
        else:
            # Find any utils import and add after it
            utils_import_pattern = r'(from utils\.[^\n]*\n)'
            if re.search(utils_import_pattern, content):
                content = re.sub(utils_import_pattern, r'\1from utils.safe_sympify import safe_sympify\n', content, count=1)
            else:
                # Add after other imports
                import_section_pattern = r'(import [^\n]*\nfrom [^\n]*\n)'
                content = re.sub(import_section_pattern, r'\1from utils.safe_sympify import safe_sympify\n', content, count=1)
    
    # Replace sp.sympify calls with safe_sympify
    # Pattern 1: Simple sp.sympify(expression)
    content = re.sub(r'sp\.sympify\(([^)]+)\)', r'safe_sympify(\1)', content)
    
    # Pattern 2: sp.sympify in list comprehensions
    content = re.sub(r'\[sp\.sympify\(([^)]+)\) for ([^]]+)\]', r'[safe_sympify(\1) for \2]', content)
    
    if content != original_content:
        # Write back the updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Updated {file_path}")
        return True
    else:
        print(f"  ⏭️  No changes needed for {file_path}")
        return False

def main():
    """Update all module files"""
    modules_dir = Path("src/modules")
    
    if not modules_dir.exists():
        print("Error: src/modules directory not found")
        return
    
    python_files = list(modules_dir.glob("*.py"))
    updated_files = []
    
    print(f"Found {len(python_files)} Python files in src/modules/")
    print("=" * 50)
    
    for file_path in python_files:
        if file_path.name == "__init__.py":
            continue
            
        try:
            if update_module_file(file_path):
                updated_files.append(file_path.name)
        except Exception as e:
            print(f"  ❌ Error updating {file_path}: {e}")
    
    print("=" * 50)
    print(f"Updated {len(updated_files)} files:")
    for file_name in updated_files:
        print(f"  - {file_name}")
    
    print("\n🎉 All modules updated to use safe_sympify!")

if __name__ == "__main__":
    main()
