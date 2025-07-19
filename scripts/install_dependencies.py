#!/usr/bin/env python3
"""
Installation script for PyTorch ABSA VLSP 2018 dependencies
Handles different installation scenarios and checks
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path


def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"Running: {command}")
    if description:
        print(f"  -> {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"  ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("=== Checking Python Version ===")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("✗ Python 3.8+ is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def check_conda_environment():
    """Check if running in conda environment"""
    print("\n=== Checking Environment ===")
    
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env_name = os.environ['CONDA_DEFAULT_ENV']
        print(f"✓ Running in conda environment: {env_name}")
        return True
    
    if sys.prefix != sys.base_prefix:
        print("✓ Running in virtual environment")
        return True
    
    print("⚠ Not running in virtual environment")
    print("  Recommendation: Create a conda/venv environment first")
    return False


def install_pytorch():
    """Install PyTorch with appropriate configuration"""
    print("\n=== Installing PyTorch ===")
    
    # Check if PyTorch is already installed
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} already installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠ CUDA not available, using CPU version")
        
        return True
    except ImportError:
        pass
    
    print("Installing PyTorch...")
    
    # Try to detect CUDA and install appropriate version
    cuda_available = False
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            cuda_available = True
            print("✓ NVIDIA GPU detected, installing CUDA version")
        else:
            print("⚠ No NVIDIA GPU detected, installing CPU version")
    except FileNotFoundError:
        print("⚠ nvidia-smi not found, installing CPU version")
    
    if cuda_available:
        # Install CUDA version
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        # Install CPU version
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(command, "Installing PyTorch")


def install_transformers():
    """Install Transformers and related packages"""
    print("\n=== Installing Transformers ===")
    
    packages = [
        "transformers>=4.30.0",
        "datasets>=2.12.0", 
        "accelerate>=0.20.0",
        "tokenizers>=0.13.0",
        "sentencepiece>=0.1.99"
    ]
    
    for package in packages:
        command = f"pip install {package}"
        if not run_command(command, f"Installing {package}"):
            return False
    
    return True


def install_data_processing():
    """Install data processing packages"""
    print("\n=== Installing Data Processing Packages ===")
    
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "regex>=2023.0.0",
        "emoji>=2.2.0",
        "vncorenlp>=1.0.3"
    ]
    
    for package in packages:
        command = f"pip install {package}"
        if not run_command(command, f"Installing {package}"):
            return False
    
    return True


def install_utilities():
    """Install utility packages"""
    print("\n=== Installing Utility Packages ===")
    
    packages = [
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "ipywidgets>=8.0.0",
        "python-dotenv>=1.0.0"
    ]
    
    for package in packages:
        command = f"pip install {package}"
        if not run_command(command, f"Installing {package}"):
            print(f"⚠ Failed to install {package}, continuing...")
    
    return True


def install_optional_packages():
    """Install optional packages for better performance"""
    print("\n=== Installing Optional Packages ===")
    
    optional_packages = [
        ("plotly>=5.14.0", "Interactive plotting"),
        ("wandb>=0.15.0", "Experiment tracking"),
        ("protobuf>=4.21.0", "Protocol buffers"),
        ("pytest>=7.3.0", "Testing framework"),
        ("pytest-cov>=4.1.0", "Test coverage")
    ]
    
    for package, description in optional_packages:
        command = f"pip install {package}"
        print(f"Installing {package} ({description})...")
        if not run_command(command):
            print(f"⚠ Failed to install {package}, skipping...")
    
    return True


def verify_installation():
    """Verify that key packages are installed correctly"""
    print("\n=== Verifying Installation ===")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'pandas', 
        'numpy', 'sklearn', 'emoji', 'vncorenlp', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        return False
    
    print("\n✓ All required packages installed successfully!")
    return True


def setup_java_for_vncorenlp():
    """Setup Java for VnCoreNLP"""
    print("\n=== Checking Java for VnCoreNLP ===")
    
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Java is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠ Java not found")
    print("VnCoreNLP requires Java 8+. Please install Java:")
    print("  Ubuntu/Debian: sudo apt install openjdk-8-jdk")
    print("  MacOS: brew install openjdk@8")
    print("  Windows: Download from Oracle JDK website")
    
    return False


def install_from_requirements():
    """Install from requirements.txt if available"""
    print("\n=== Installing from requirements.txt ===")
    
    requirements_path = Path(__file__).parent.parent / 'requirements.txt'
    
    if requirements_path.exists():
        command = f"pip install -r {requirements_path}"
        return run_command(command, "Installing from requirements.txt")
    else:
        print("⚠ requirements.txt not found, installing packages individually")
        return False


def main():
    """Main installation routine"""
    print("PyTorch ABSA VLSP 2018 - Dependency Installation")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        return
    
    check_conda_environment()
    
    # Try requirements.txt first
    if install_from_requirements():
        print("✓ Installed from requirements.txt")
    else:
        # Install packages step by step
        print("Installing packages individually...")
        
        if not install_pytorch():
            print("❌ Failed to install PyTorch")
            return
        
        if not install_transformers():
            print("❌ Failed to install Transformers")
            return
        
        if not install_data_processing():
            print("❌ Failed to install data processing packages")
            return
        
        install_utilities()
        install_optional_packages()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Installation completed successfully!")
    else:
        print("\n❌ Installation completed with errors")
        return
    
    # Check Java for VnCoreNLP
    setup_java_for_vncorenlp()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Test installation: python scripts/demo.py")
    print("2. Prepare data: python scripts/prepare_data.py")
    print("3. Start training: python scripts/train.py --config configs/hotel_multitask.yaml")
    print("\nIf you encounter issues:")
    print("- Check the project README.md")
    print("- Ensure you're in the correct conda environment")
    print("- Try installing packages individually if needed")


if __name__ == '__main__':
    main() 