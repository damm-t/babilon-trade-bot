#!/usr/bin/env python3
"""
Startup script for Babilon Trade Bot
Runs both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
import threading
import webbrowser
from pathlib import Path

def run_api():
    """Run the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        subprocess.run([sys.executable, "api.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error running API server: {e}")

def run_streamlit():
    """Run the Streamlit app"""
    print("🚀 Starting Streamlit app...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_enhanced.py", "--server.port", "8501"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")

def run_web_server():
    """Run a simple HTTP server for the web app"""
    print("🚀 Starting web server...")
    try:
        os.chdir("web_app")
        subprocess.run([sys.executable, "-m", "http.server", "8080"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Web server stopped")
    except Exception as e:
        print(f"❌ Error running web server: {e}")

def main():
    print("🎯 Babilon Trade Bot - Starting all services...")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ["api.py", "streamlit_enhanced.py", "web_app/index.html"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file not found: {file}")
            sys.exit(1)
    
    print("✅ All required files found")
    print("\n📋 Available interfaces:")
    print("   • FastAPI Backend: http://localhost:8000")
    print("   • Streamlit UI: http://localhost:8501")
    print("   • Web App: http://localhost:8080")
    print("\n🔧 API Documentation: http://localhost:8000/docs")
    print("\n" + "=" * 50)
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # Start Streamlit in a separate thread
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Wait a moment for Streamlit to start
    time.sleep(3)
    
    # Start web server in a separate thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    # Wait a moment for web server to start
    time.sleep(2)
    
    print("\n🌐 Opening web interfaces...")
    
    # Open browsers (optional)
    try:
        webbrowser.open("http://localhost:8501")  # Streamlit
        time.sleep(1)
        webbrowser.open("http://localhost:8080")  # Web app
    except Exception as e:
        print(f"⚠️  Could not open browsers automatically: {e}")
    
    print("\n✅ All services started!")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
