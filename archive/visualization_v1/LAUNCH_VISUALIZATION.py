"""
MTG Card Recognition - Complete Visualization System Launcher
Starts Flask server and provides easy card processing
"""

import subprocess
import sys
import time
import webbrowser
import os
import shutil
from pathlib import Path

print("=" * 70)
print("  MTG CARD RECOGNITION - FLASK VISUALIZATION SYSTEM")
print("=" * 70)
print()

# Get the script directory
SCRIPT_DIR = Path(__file__).parent
FLASK_DIR = SCRIPT_DIR / "Visualization"
FLASK_APP = FLASK_DIR / "app.py"

def reset_flask_environment():
    """Clean up old Flask processes and cached files"""
    print("[0/3] Cleaning up old processes and cache...")
    print("-" * 70)
    
    # Kill any existing Python processes (Flask servers)
    try:
        if sys.platform == "win32":
            # Windows: Use taskkill to stop Python processes
            os.system("taskkill /F /IM python.exe /T 2>nul")
        else:
            # Unix: Use pkill
            os.system("pkill -f 'python.*app.py' 2>/dev/null")
        print("✓ Old Flask processes terminated")
    except Exception as e:
        print(f"⚠ Could not terminate old processes: {e}")
    
    # Clear Python cache (__pycache__)
    try:
        pycache_dir = FLASK_DIR / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir)
            print("✓ Python cache cleared")
        else:
            print("✓ No cache to clear")
    except Exception as e:
        print(f"⚠ Could not clear cache: {e}")
    
    print()

def start_flask_server():
    """Start the Flask visualization server in background"""
    print("[1/3] Starting Flask Visualization Server...")
    print("-" * 70)
    
    if not FLASK_APP.exists():
        print(f"ERROR: Flask app not found at {FLASK_APP}")
        return None
    
    # Start Flask in a new process
    if sys.platform == "win32":
        # Windows: Use CREATE_NEW_CONSOLE to open in new window
        process = subprocess.Popen(
            [sys.executable, str(FLASK_APP)],
            cwd=str(FLASK_DIR),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Unix-like: Start in background
        process = subprocess.Popen(
            [sys.executable, str(FLASK_APP)],
            cwd=str(FLASK_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    print("✓ Flask server started in separate window")
    print()
    
    return process

def main():
    # Reset Flask environment (clean up old processes and cache)
    reset_flask_environment()
    
    # Start Flask server
    flask_process = start_flask_server()
    
    if not flask_process:
        print("Failed to start Flask server")
        input("Press Enter to exit...")
        return
    
    # Wait for Flask to initialize
    print("Waiting for Flask server to initialize (3 seconds)...")
    time.sleep(3)
    
    print()
    print("[2/3] Opening Visualization in Browser...")
    print("-" * 70)
    
    # Open browser
    print("[3/3] Opening Browser...")
    print("-" * 70)
    try:
        webbrowser.open("http://localhost:5000")
        print("✓ Browser opened to http://localhost:5000")
    except:
        print("⚠ Could not auto-open browser. Please open manually:")
        print("  → http://localhost:5000")
    
    print()
    print("=" * 70)
    print("  SYSTEM READY!")
    print("=" * 70)
    print()
    print("Visualization Server: http://localhost:5000")
    print()
    print("Now you can run:")
    print("  • python OhMyOhMy.py          - Process test card image")
    print("  • python Excecutable_File_Final.py - Full motor control system")
    print()
    print("Both scripts automatically send data to the visualization server.")
    print("Click 'Refresh Data' button in browser after processing each card.")
    print()
    print("=" * 70)
    print()
    print("The Flask server is running in a separate window.")
    print("Close that window to stop the server.")
    print()
    input("Press Enter to exit this launcher (Flask server will keep running)...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nLauncher stopped by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        input("Press Enter to exit...")
