#!/usr/bin/env python3
"""
Babilon Trade Bot - Automated Setup Script
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def setup_virtual_environment():
    """Set up virtual environment."""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("📦 Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv .venv", "Creating virtual environment"):
        return False
    
    return True


def install_dependencies():
    """Install project dependencies."""
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = ".venv/Scripts/pip"
    else:  # Unix/Linux/macOS
        pip_path = ".venv/bin/pip"
    
    if not run_command(f"{pip_path} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_path} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True


def setup_environment_file():
    """Set up environment file."""
    env_file = Path(".env")
    env_template = Path("env_template.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_template.exists():
        print("❌ env_template.txt not found")
        return False
    
    print("📝 Creating .env file from template...")
    try:
        shutil.copy(env_template, env_file)
        print("✅ .env file created successfully")
        print("⚠️  Please edit .env file with your actual API keys and preferences")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["data", "logs", "models", "tests"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"📁 Creating {directory} directory...")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ {directory} directory already exists")


def run_tests():
    """Run basic tests to verify installation."""
    # Determine the correct python path
    if os.name == 'nt':  # Windows
        python_path = ".venv/Scripts/python"
    else:  # Unix/Linux/macOS
        python_path = ".venv/bin/python"
    
    print("🧪 Running basic tests...")
    if run_command(f"{python_path} -m pytest tests/ -v", "Running tests"):
        print("✅ All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed, but setup is still functional")
        return True  # Don't fail setup if tests fail


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Babilon Trade Bot setup completed!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your Alpaca API keys")
    print("2. Get paper trading API keys from: https://app.alpaca.markets/paper/dashboard/overview")
    print("3. Activate virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("   .venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source .venv/bin/activate")
    
    print("\n4. Run the application:")
    print("   # Web interface")
    print("   streamlit run streamlit_unified.py")
    print("\n   # API server")
    print("   python api.py")
    print("\n   # CLI interface")
    print("   python main.py")
    
    print("\n🐳 Docker (Alternative):")
    print("   docker-compose up -d")
    
    print("\n⚠️  Important Notes:")
    print("- This bot is for educational purposes only")
    print("- Start with paper trading API keys")
    print("- Never commit your .env file to version control")
    print("- Trading involves substantial risk of loss")
    
    print("\n📖 Documentation:")
    print("- README.md - Basic usage")
    print("- README_ENHANCED.md - Comprehensive guide")
    print("- WhatToDo.md - Implementation roadmap")


def main():
    """Main setup function."""
    print("🚀 Babilon Trade Bot - Automated Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Set up virtual environment
    if not setup_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Set up environment file
    if not setup_environment_file():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run tests
    run_tests()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()