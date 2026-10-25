#!/usr/bin/env python
"""
Diagnostic script to verify MTG Card Hash Visualization setup
Run this to check if everything is installed and configured correctly
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_mark(success, message):
    symbol = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {message}")
    return success

def test_imports():
    """Test if all required packages are installed"""
    print_header("Testing Python Packages")
    
    packages = [
        ('flask', 'Flask (web framework)'),
        ('flask_cors', 'Flask-CORS (cross-origin support)'),
        ('cv2', 'OpenCV (image processing)'),
        ('numpy', 'NumPy (numerical computing)'),
        ('imagehash', 'ImageHash (perceptual hashing)'),
        ('PIL', 'Pillow (image library)'),
        ('matplotlib', 'Matplotlib (visualization)'),
        ('annoy', 'Annoy (nearest neighbor search)'),
    ]
    
    all_good = True
    for package_name, display_name in packages:
        try:
            __import__(package_name)
            check_mark(True, f"{display_name} is installed")
        except ImportError:
            check_mark(False, f"{display_name} is NOT installed")
            all_good = False
    
    return all_good

def test_files():
    """Test if required files exist"""
    print_header("Checking Files")
    
    files = [
        ('scryfall_phash_db_sample.json', 'Card database'),
        ('templates/index.html', 'Web interface template'),
        ('static/style.css', 'CSS styling'),
        ('static/script.js', 'JavaScript frontend'),
        ('app.py', 'Flask application'),
    ]
    
    all_good = True
    for filename, description in files:
        exists = os.path.exists(filename)
        check_mark(exists, f"{filename} - {description}")
        all_good = all_good and exists
    
    return all_good

def test_directories():
    """Test if required directories exist"""
    print_header("Checking Directories")
    
    directories = [
        ('Captured_Cards', 'Card image storage'),
        ('templates', 'HTML templates'),
        ('static', 'Static files (CSS, JS)'),
    ]
    
    all_good = True
    for dirname, description in directories:
        exists = os.path.isdir(dirname)
        check_mark(exists, f"{dirname}/ - {description}")
        if not exists:
            print(f"  → Would be created automatically when app starts")
        all_good = all_good and exists
    
    return all_good

def test_mtg_imports():
    """Test if MTG-specific modules can be imported"""
    print_header("Testing MTG Card Recognition Modules")
    
    try:
        from Excecutable_File_Final import (
            load_or_build_annoy_index,
            resize_to_scryfall_normal,
            compute_phash,
        )
        check_mark(True, "Excecutable_File_Final module found")
        return True
    except ImportError as e:
        check_mark(False, f"Excecutable_File_Final module not found: {e}")
        print("  Note: This is used but will fall back to defaults if missing")
        return False

def test_flask_app():
    """Test if Flask app can import successfully"""
    print_header("Testing Flask App Creation")
    
    try:
        import app as flask_app
        check_mark(True, "Flask app imports successfully")
        return True
    except Exception as e:
        check_mark(False, f"Flask app has errors: {e}")
        return False

def test_database():
    """Test if database can be loaded"""
    print_header("Testing Database")
    
    try:
        import json
        with open("scryfall_phash_db_sample.json", "r", encoding="utf-8") as f:
            db = json.load(f)
        
        num_cards = len(db)
        check_mark(True, f"Database loaded: {num_cards} cards found")
        
        # Check database structure
        if num_cards > 0:
            first_card = db[0] if isinstance(db, list) else list(db.values())[0]
            required_fields = ['name', 'phash']
            has_fields = all(field in first_card for field in required_fields)
            check_mark(has_fields, "Database has required fields (name, phash)")
        
        return True
    except Exception as e:
        check_mark(False, f"Database error: {e}")
        return False

def test_sample_images():
    """Test if sample images exist"""
    print_header("Testing Sample Images")
    
    if not os.path.isdir("Captured_Cards"):
        check_mark(False, "Captured_Cards/ directory not found")
        return False
    
    try:
        images = [f for f in os.listdir("Captured_Cards") 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if images:
            check_mark(True, f"Found {len(images)} sample image(s)")
            if len(images) <= 5:
                for img in images:
                    print(f"  • {img}")
            else:
                for img in images[:5]:
                    print(f"  • {img}")
                print(f"  ... and {len(images)-5} more")
            return True
        else:
            check_mark(False, "No sample images in Captured_Cards/")
            print("  Note: You can still upload images via the web interface")
            return True  # Not a critical error
    except Exception as e:
        check_mark(False, f"Error checking sample images: {e}")
        return False

def main():
    print("\n" + "🃏 " * 20)
    print("\nMTG Card Hash Visualization - Diagnostic Report")
    print("\n" + "🃏 " * 20 + "\n")
    
    results = {
        'packages': test_imports(),
        'files': test_files(),
        'directories': test_directories(),
        'mtg_modules': test_mtg_imports(),
        'database': test_database(),
        'samples': test_sample_images(),
    }
    
    print_header("Diagnostic Summary")
    
    all_tests = all(results.values())
    
    if all_tests:
        print("✓ All systems go! Ready to start the web server.\n")
        print("Next step: Run 'python app.py' to start the server")
        print("Then open: http://localhost:5000\n")
        return 0
    else:
        print("⚠ Some issues found. Here's what to fix:\n")
        
        if not results['packages']:
            print("1. Install missing packages:")
            print("   pip install -r requirements.txt\n")
        
        if not results['files']:
            print("2. Verify required files exist in current directory\n")
        
        if not results['database']:
            print("3. Check scryfall_phash_db_sample.json file\n")
        
        print("\nRun this diagnostic again after fixing issues.")
        print("Questions? Check WEBSITE_SETUP_GUIDE.md or README_WEB_INTERFACE.md\n")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n✗ Fatal error: {e}\n")
        sys.exit(1)
