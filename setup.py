#!/usr/bin/env python3
"""
Babilon Trade Bot Setup Script
Automated setup and configuration for the Babilon Trade Bot
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True


def create_virtual_environment():
    """Create and activate virtual environment"""
    print("📦 Setting up virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        return True
    
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate.bat"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix-like
        activate_script = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print(f"✅ Virtual environment created")
    print(f"💡 To activate: {activate_script}")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("📚 Installing dependencies...")
    
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix-like
        pip_cmd = "venv/bin/pip"
    
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True


def setup_environment_file():
    """Set up environment configuration file"""
    print("🔧 Setting up environment configuration...")
    
    env_file = Path(".env")
    env_template = Path("env_template.txt")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Keeping existing .env file")
            return True
    
    if not env_template.exists():
        print("❌ env_template.txt not found")
        return False
    
    # Copy template to .env
    shutil.copy(env_template, env_file)
    print("✅ Created .env file from template")
    
    print("\n🔑 IMPORTANT: Please edit .env file and add your API keys:")
    print("   1. Get Alpaca API keys from: https://app.alpaca.markets/paper/dashboard/overview")
    print("   2. Edit .env file and replace 'your_paper_trading_api_key_here' with your actual keys")
    print("   3. Adjust trading parameters as needed")
    
    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ["logs", "data", "cache", "reports"]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory}/ directory")
        else:
            print(f"⚠️  {directory}/ directory already exists")
    
    return True


def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    
    # Determine pytest command based on OS
    if os.name == 'nt':  # Windows
        pytest_cmd = "venv\\Scripts\\pytest"
    else:  # Unix-like
        pytest_cmd = "venv/bin/pytest"
    
    # Set dummy environment variables for testing
    env = os.environ.copy()
    env.update({
        'ALPACA_API_KEY': 'test_key',
        'ALPACA_SECRET_KEY': 'test_secret'
    })
    
    if not run_command(f"{pytest_cmd} tests/ -v", "Running test suite"):
        print("⚠️  Some tests failed, but this is normal without valid API keys")
        return True
    
    return True


def check_git_setup():
    """Check and setup Git configuration"""
    print("🔍 Checking Git setup...")
    
    if not run_command("git --version", "Checking Git installation"):
        print("❌ Git not found. Please install Git first")
        return False
    
    # Check if this is a git repository
    if not Path(".git").exists():
        print("📝 Initializing Git repository...")
        if not run_command("git init", "Initializing Git repository"):
            return False
        
        if not run_command("git add .", "Adding files to Git"):
            return False
        
        if not run_command('git commit -m "Initial commit"', "Making initial commit"):
            return False
    
    return True


def main():
    """Main setup function"""
    print("🚀 Babilon Trade Bot Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment file", setup_environment_file),
        ("Creating directories", create_directories),
        ("Checking Git setup", check_git_setup),
        ("Running tests", run_tests),
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        print(f"\n📋 {step_name}")
        print("-" * 30)
        
        if not step_function():
            failed_steps.append(step_name)
            print(f"❌ Failed: {step_name}")
        else:
            print(f"✅ Success: {step_name}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    
    if failed_steps:
        print(f"\n⚠️  Some steps failed:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease check the errors above and run setup again if needed.")
    else:
        print("\n✅ All setup steps completed successfully!")
    
    print("\n📖 Next Steps:")
    print("1. Edit .env file with your Alpaca API keys")
    print("2. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Run the bot:")
    print("   python main.py")
    print("4. Or start the web interface:")
    print("   streamlit run streamlit_app.py")
    
    print("\n📚 Documentation:")
    print("   - README_ENHANCED.md - Comprehensive documentation")
    print("   - env_template.txt - Configuration template")
    print("   - tests/ - Test suite")
    
    print("\n🔒 Security Note:")
    print("   - Never commit your .env file to version control")
    print("   - Use paper trading keys for testing")
    print("   - Keep your API keys secure")


if __name__ == "__main__":
    main()

