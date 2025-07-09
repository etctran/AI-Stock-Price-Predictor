"""
Simple Setup and Run Script for AI Stock Price Predictor

This script helps beginners set up and run the project easily.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def run_dashboard():
    """Run the Streamlit dashboard."""
    print("🚀 Starting Streamlit dashboard...")
    print("📱 Your browser should open automatically")
    print("🌐 If not, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped!")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def run_command_line():
    """Run the command line version."""
    print("💻 Starting command line version...")
    try:
        subprocess.run([sys.executable, "src/main.py"])
    except Exception as e:
        print(f"❌ Error running command line version: {e}")

def main():
    """Main setup function."""
    print("🤖 AI Stock Price Predictor Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Please run this from the ai-stock-predictor directory")
        return
    
    print("Choose an option:")
    print("1. 📦 Install packages and run dashboard")
    print("2. 🚀 Run dashboard only")
    print("3. 💻 Run command line version")
    print("4. 📦 Install packages only")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if install_requirements():
            run_dashboard()
    elif choice == "2":
        run_dashboard()
    elif choice == "3":
        run_command_line()
    elif choice == "4":
        install_requirements()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
