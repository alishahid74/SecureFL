# Installation Guide for Secure Federated Learning Framework

## Quick Start

### Option 1: Install with pip (Recommended)

```bash
# Create a virtual environment (recommended)
python -m venv fl_env
source fl_env/bin/activate  # On Windows: fl_env\Scripts\activate

# Install minimal requirements
pip install -r requirements-minimal.txt

# Or install full requirements (includes optional dependencies)
pip install -r requirements.txt
```

### Option 2: Install with conda

```bash
# Create conda environment
conda create -n fl_env python=3.10 -y
conda activate fl_env

# Install PyTorch (adjust for your CUDA version if needed)
conda install pytorch torchvision -c pytorch

# Install remaining packages
pip install opacus phe networkx scikit-learn matplotlib seaborn
```

### Option 3: Install with GPU Support (CUDA)

```bash
# Create virtual environment
python -m venv fl_env
source fl_env/bin/activate

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining packages
pip install opacus phe networkx scikit-learn matplotlib seaborn numpy scipy
```

## Detailed Installation Steps

### 1. Prerequisites

- **Python**: 3.8 or higher (3.10 recommended)
- **pip**: Latest version
- **Git**: For cloning repositories (optional)

Check your Python version:
```bash
python --version  # Should show 3.8 or higher
pip --version
```

### 2. Set Up Virtual Environment

Using a virtual environment is highly recommended to avoid dependency conflicts.

**Option A: venv (built-in)**
```bash
python -m venv fl_env
source fl_env/bin/activate  # On Windows: fl_env\Scripts\activate
```

**Option B: conda**
```bash
conda create -n fl_env python=3.10
conda activate fl_env
```

### 3. Install Core Dependencies

#### Essential Packages (Required)

```bash
# Deep Learning Framework
pip install torch>=2.0.0 torchvision>=0.15.0

# Differential Privacy
pip install opacus>=1.4.0

# Homomorphic Encryption
pip install phe>=1.5.0

# Graph Processing
pip install networkx>=3.0

# Scientific Computing
pip install numpy>=1.24.0 scipy>=1.10.0

# Machine Learning
pip install scikit-learn>=1.3.0

# Visualization
pip install matplotlib>=3.7.0 seaborn>=0.12.0
```

#### Optional Packages (Recommended)

```bash
# For ECG/biosignal processing (if using heman3.py features)
pip install neurokit2>=0.2.0 pandas>=2.0.0

# For progress bars during training
pip install tqdm>=4.65.0

# For Jupyter notebook development
pip install jupyter ipykernel ipywidgets
```

### 4. Verify Installation

Create a test script to verify all packages are installed correctly:

```bash
python -c "
import torch
import torchvision
import opacus
import phe
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

print('✓ PyTorch:', torch.__version__)
print('✓ Torchvision:', torchvision.__version__)
print('✓ Opacus:', opacus.__version__)
print('✓ phe:', phe.__version__)
print('✓ NetworkX:', nx.__version__)
print('✓ NumPy:', np.__version__)
print('✓ All packages installed successfully!')
print('✓ CUDA available:', torch.cuda.is_available())
"
```

### 5. Download and Run the Framework

```bash
# Run the secure federated learning framework
python secure_federated_learning.py
```

## Platform-Specific Instructions

### Windows

```bash
# Install Visual C++ redistributable if needed
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Create virtual environment
python -m venv fl_env
fl_env\Scripts\activate

# Install packages
pip install -r requirements-minimal.txt
```

### macOS (Apple Silicon M1/M2)

```bash
# Create virtual environment
python3 -m venv fl_env
source fl_env/bin/activate

# Install PyTorch for Apple Silicon
pip install torch torchvision

# Install remaining packages
pip install opacus phe networkx scikit-learn matplotlib seaborn numpy scipy
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv python3-dev

# Create virtual environment
python3 -m venv fl_env
source fl_env/bin/activate

# Install packages
pip install -r requirements-minimal.txt
```

## GPU Support (NVIDIA CUDA)

### Check CUDA Version

```bash
nvidia-smi  # Check your CUDA version
```

### Install PyTorch with CUDA

Visit https://pytorch.org/get-started/locally/ and select your configuration, or use:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Ensure PyTorch is installed: `pip install torch torchvision`

### Issue: "phe module not found"
**Solution**: Install phe: `pip install phe`

### Issue: CUDA out of memory
**Solution**: Reduce batch size in FLConfig or use CPU: `device = torch.device('cpu')`

### Issue: Import error with Opacus
**Solution**: Ensure compatible versions:
```bash
pip install torch>=2.0.0 opacus>=1.4.0
```

### Issue: Matplotlib backend errors
**Solution**: 
```bash
# On Linux servers without display
export MPLBACKEND=Agg

# Or in Python
import matplotlib
matplotlib.use('Agg')
```

## Testing Your Installation

Run a quick test:

```python
# test_installation.py
import torch
import numpy as np
from secure_federated_learning import FLConfig, run_synthetic_experiment

# Create minimal config
config = FLConfig(
    num_clients=5,
    num_rounds=2,
    use_dp=True,
    use_homomorphic_encryption=False
)

print("Running test experiment...")
server = run_synthetic_experiment(config, n_features=10)
print("✓ Installation successful! Framework is working.")
```

## Dependency Versions (Tested)

The framework has been tested with:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.12 | Minimum 3.8 |
| PyTorch | 2.1.0 | GPU/CPU compatible |
| Opacus | 1.4.1 | Differential Privacy |
| phe | 1.5.0 | Homomorphic Encryption |
| NetworkX | 3.2.1 | Graph algorithms |
| NumPy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.2 | ML utilities |
| Matplotlib | 3.8.0 | Visualization |

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify all dependencies are installed: `pip list`
3. Try reinstalling problematic packages: `pip install --upgrade <package>`
4. Use `pip install --no-cache-dir <package>` if cache issues occur
5. Check package compatibility: `pip check`

## Development Setup (Optional)

For contributing or development:

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest tests/

# Format code
black secure_federated_learning.py

# Type checking
mypy secure_federated_learning.py
```

## Updating Packages

Keep your environment up to date:

```bash
# Upgrade all packages
pip install --upgrade -r requirements.txt

# Or upgrade specific packages
pip install --upgrade torch torchvision opacus
```

## Uninstallation

To remove the environment:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment directory
rm -rf fl_env  # On Windows: rmdir /s fl_env
```
