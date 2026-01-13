#!/bin/bash

# ============================================================================
# Secure Federated Learning Framework - Quick Setup Script
# ============================================================================
# This script automates the installation process
# Usage: bash setup.sh

set -e  # Exit on error

echo "============================================"
echo "Secure Federated Learning - Setup Script"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    print_error "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_success "Found Python $PYTHON_VERSION"

# Check if version is 3.8 or higher
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 8 ]); then
    print_error "Python 3.8 or higher is required. You have Python $PYTHON_VERSION"
    exit 1
fi

# Check if virtual environment exists
ENV_NAME="fl_env"

if [ -d "$ENV_NAME" ]; then
    print_info "Virtual environment '$ENV_NAME' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$ENV_NAME"
    else
        print_info "Using existing virtual environment."
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
    print_info "Creating virtual environment '$ENV_NAME'..."
    $PYTHON_CMD -m venv "$ENV_NAME"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$ENV_NAME/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Ask user which installation type
echo ""
echo "Select installation type:"
echo "1) Minimal (essential packages only)"
echo "2) Full (includes optional packages)"
echo "3) GPU Support (CUDA-enabled PyTorch)"
read -p "Enter choice [1-3]: " INSTALL_CHOICE

case $INSTALL_CHOICE in
    1)
        print_info "Installing minimal requirements..."
        pip install -r requirements-minimal.txt
        ;;
    2)
        print_info "Installing full requirements..."
        pip install -r requirements.txt
        ;;
    3)
        print_info "Please enter your CUDA version (e.g., 11.8, 12.1) or 'cpu' for CPU-only:"
        read -p "CUDA version: " CUDA_VERSION
        
        print_info "Installing PyTorch with CUDA $CUDA_VERSION..."
        if [ "$CUDA_VERSION" == "cpu" ]; then
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        else
            CUDA_SHORT=$(echo $CUDA_VERSION | tr -d '.')
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu${CUDA_SHORT}
        fi
        
        print_info "Installing remaining packages..."
        pip install opacus phe networkx scikit-learn matplotlib seaborn numpy scipy tqdm
        ;;
    *)
        print_error "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

print_success "All packages installed"

# Verify installation
echo ""
print_info "Verifying installation..."

$PYTHON_CMD << EOF
try:
    import torch
    import torchvision
    import opacus
    import phe
    import networkx as nx
    import numpy as np
    import matplotlib
    import seaborn
    import sklearn
    
    print("\n${GREEN}Installation Verification:${NC}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Torchvision: {torchvision.__version__}")
    print(f"  Opacus: {opacus.__version__}")
    print(f"  phe: {phe.__version__}")
    print(f"  NetworkX: {nx.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    
    print(f"\n${GREEN}✓ All packages successfully installed!${NC}")
    
except ImportError as e:
    print(f"${RED}✗ Import error: {e}${NC}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Verification complete"
else
    print_error "Verification failed. Please check error messages above."
    exit 1
fi

# Summary
echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment in the future:"
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "To run the framework:"
echo "  python secure_federated_learning.py"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""
print_success "Happy federated learning!"
