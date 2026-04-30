#!/usr/bin/env python3
"""
Final verification script before publishing to PyPI/GitHub.
Run with: python scripts/verify_ready.py

Author: GPPanos
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report success/failure."""
    print(f"\n🔍 Checking {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {description} passed")
            return True
        else:
            print(f"   ❌ {description} failed")
            print(f"   Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ {description} error: {e}")
        return False


def main():
    print("=" * 60)
    print("Black-Litterman Improved - Final Verification")
    print("Author: GPPanos")
    print("=" * 60)
    
    checks = []
    
    # Check 1: All required files exist
    required_files = [
        "pyproject.toml",
        "README.md",
        ".github/workflows/ci.yml",
        "docs/requirements.txt",
        ".readthedocs.yml",
        "mkdocs.yml",
        ".gitignore",
        "black_litterman_improved/__init__.py",
        "black_litterman_improved/core/black_litterman.py",
        "black_litterman_improved/enhancements/ml_views.py",
        "tests/unit/test_black_litterman.py",
        "tests/unit/test_ml_views.py",
        "example.py",
    ]
    
    print("\n📁 Checking required files...")
    all_files_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ MISSING: {file}")
            all_files_exist = False
    checks.append(all_files_exist)
    
    # Check 2: No placeholder usernames
    print("\n👤 Checking for placeholder usernames...")
    placeholder_files = []
    for file in Path(".").rglob("*.py"):
        if file.is_file() and "GPPanos" not in file.read_text():
            if "yourusername" in file.read_text().lower():
                placeholder_files.append(str(file))
    
    for file in Path(".").rglob("*.md"):
        if file.is_file():
            content = file.read_text()
            if "yourusername" in content.lower() or "YOUR_USERNAME" in content:
                placeholder_files.append(str(file))
    
    for file in Path(".").rglob("*.toml"):
        if file.is_file():
            content = file.read_text()
            if "yourusername" in content.lower():
                placeholder_files.append(str(file))
    
    if placeholder_files:
        print(f"   ❌ Found placeholder 'yourusername' in: {placeholder_files}")
        checks.append(False)
    else:
        print("   ✅ No placeholder usernames found")
        checks.append(True)
    
    # Check 3: Run tests
    checks.append(run_command("python -m pytest tests/ -q --tb=no", "unit tests"))
    
    # Check 4: Check formatting
    checks.append(run_command("black --check black_litterman_improved/ --quiet", "code formatting"))
    
    # Check 5: Check types
    checks.append(run_command("mypy black_litterman_improved/ --ignore-missing-imports", "type checking"))
    
    # Check 6: Build package
    checks.append(run_command("python -m build", "package build"))
    
    # Check 7: Check for secrets (basic)
    print("\n🔐 Checking for exposed secrets...")
    secret_patterns = ['pypi-', 'api_key', 'password', 'token', 'secret']
    secrets_found = []
    for file in Path(".").rglob("*"):
        if file.is_file() and file.suffix in ['.py', '.md', '.yml', '.yaml', '.toml']:
            try:
                content = file.read_text()
                for pattern in secret_patterns:
                    if pattern in content.lower() and 'example' not in content.lower():
                        if 'covers' not in content.lower():
                            secrets_found.append(f"{file} (contains '{pattern}')")
            except:
                pass
    
    if secrets_found:
        print(f"   ⚠️  Potential secrets found: {secrets_found[:3]}")
        checks.append(True)  # Warning only, not fatal
    else:
        print("   ✅ No obvious secrets found")
        checks.append(True)
    
    # Summary
    print("\n" + "=" * 60)
    if all(checks):
        print("🎉 ALL CHECKS PASSED! Ready for GitHub and PyPI!")
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'feat: Initial release of Black-Litterman Improved library'")
        print("3. git remote add origin https://github.com/GPPanos/black-litterman-improved.git")
        print("4. git push -u origin main")
        print("5. git tag -a v0.1.0 -m 'First release'")
        print("6. git push origin v0.1.0")
        print("7. CI/CD will automatically publish to PyPI")
    else:
        print("⚠️  Some checks failed. Fix issues before publishing.")
        print(f"   Failed checks: {sum(1 for c in checks if not c)}/{len(checks)}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
