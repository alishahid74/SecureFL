# Secure Federated Learning Framework - Installation Files

This package contains everything you need to install and run the Secure Federated Learning Framework.

## 📦 Files Included

| File | Purpose |
|------|---------|
| `secure_federated_learning.py` | Main framework code |
| `requirements.txt` | Full dependencies (includes optional packages) |
| `requirements-minimal.txt` | Essential dependencies only |
| `INSTALLATION.md` | Detailed installation guide |
| `setup.sh` | Automated setup script for Linux/macOS |
| `setup.bat` | Automated setup script for Windows |

## 🚀 Quick Start

### Automated Installation (Recommended)

**Linux/macOS:**
```bash
bash setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Manual Installation

```bash
# 1. Create virtual environment
python -m venv fl_env
source fl_env/bin/activate  # Windows: fl_env\Scripts\activate

# 2. Install packages
pip install -r requirements-minimal.txt

# 3. Run the framework
python secure_federated_learning.py
```

## 📋 Core Requirements

The minimal installation requires:

```
torch>=2.0.0          # Deep learning framework
torchvision>=0.15.0   # Computer vision datasets
opacus>=1.4.0         # Differential privacy
phe>=1.5.0            # Homomorphic encryption
networkx>=3.0         # Graph algorithms
numpy>=1.24.0         # Numerical computing
scipy>=1.10.0         # Scientific computing
scikit-learn>=1.3.0   # Machine learning utilities
matplotlib>=3.7.0     # Plotting
seaborn>=0.12.0       # Statistical visualization
```

## 🔧 Installation Options

### Option 1: Minimal (Essential only - ~2GB)
```bash
pip install -r requirements-minimal.txt
```

### Option 2: Full (Includes optional packages - ~3GB)
```bash
pip install -r requirements.txt
```

### Option 3: GPU Support (CUDA)
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other packages
pip install opacus phe networkx scikit-learn matplotlib seaborn
```

## 🖥️ System Requirements

**Minimum:**
- Python 3.8 or higher
- 4GB RAM
- 5GB disk space

**Recommended:**
- Python 3.10 or higher
- 8GB RAM
- 10GB disk space
- NVIDIA GPU with CUDA support (optional, for faster training)

## 📊 What the Framework Does

This framework implements secure federated learning with:

- ✅ **Graph-based topology** - Clients connected via network graph
- ✅ **Multi-KRUM selection** - Byzantine-fault-tolerant aggregation
- ✅ **Adaptive Trust Algorithm** - Dynamic client trust scoring
- ✅ **Differential Privacy** - Trust-aware noise injection
- ✅ **Homomorphic Encryption** - Privacy-preserving computation
- ✅ **Secure Aggregation** - Cryptographic update masking
- ✅ **Attack Simulations** - Poison, backdoor, inference attacks

## 🏃 Running the Framework

After installation:

```bash
# Activate virtual environment
source fl_env/bin/activate  # Windows: fl_env\Scripts\activate

# Run with default settings
python secure_federated_learning.py

# The framework will:
# 1. Download MNIST dataset (if torchvision available)
# 2. Run 3 experiments with different security configurations
# 3. Generate visualizations and comparison plots
# 4. Save results as PNG files
```

## 📈 Expected Output

The framework will run 3 experiments:

1. **No Defense** - Baseline FL without security
2. **Base Defense** - Multi-KRUM + Differential Privacy
3. **Full Defense** - All security mechanisms enabled

Each experiment produces:
- Console output with accuracy per round
- Visualization plots (6-panel dashboard)
- Comparison bar chart
- Performance metrics

## 🐛 Troubleshooting

### Common Issues

**"No module named 'torch'"**
```bash
pip install torch torchvision
```

**"phe not found"**
```bash
pip install phe
```

**CUDA out of memory**
- Reduce `batch_size` in `FLConfig`
- Or use CPU: `device = torch.device('cpu')`

**Import error with Opacus**
```bash
pip install --upgrade torch opacus
```

### Verify Installation

```bash
python -c "import torch, opacus, phe, networkx; print('✓ All core packages installed')"
```

## 📖 Documentation

For detailed information, see:
- `INSTALLATION.md` - Complete installation guide
- Code comments in `secure_federated_learning.py`
- Inline docstrings for all classes and methods

## 🔄 Updating

To update to the latest package versions:

```bash
pip install --upgrade -r requirements.txt
```

## 🗑️ Uninstalling

To remove the environment:

```bash
deactivate
rm -rf fl_env  # Windows: rmdir /s fl_env
```

## 💡 Tips

1. **Use a virtual environment** - Avoids dependency conflicts
2. **Start with minimal requirements** - Add optional packages as needed
3. **Use GPU if available** - Significantly faster training
4. **Check CUDA version** - Must match PyTorch installation
5. **Monitor memory usage** - Reduce batch size if needed

## 🆘 Getting Help

If you encounter issues:

1. Check error messages carefully
2. Verify Python version: `python --version`
3. List installed packages: `pip list`
4. Check package compatibility: `pip check`
5. Try reinstalling: `pip install --force-reinstall <package>`

## 📝 Customization

To customize the framework, edit `FLConfig` in the code:

```python
config = FLConfig(
    num_clients=20,           # Number of FL clients
    num_rounds=15,            # Training rounds
    malicious_ratio=0.3,      # Fraction of malicious clients
    use_dp=True,              # Enable differential privacy
    dp_epsilon=1.0,           # Privacy budget
    selection_method=SelectionMethod.MULTI_KRUM,
    # ... and more options
)
```

## 🎓 Learning Resources

- **Federated Learning**: [Google AI Blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- **Differential Privacy**: [OpenMined Tutorials](https://blog.openmined.org/tag/differential-privacy/)
- **Byzantine Fault Tolerance**: [Multi-KRUM Paper](https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**License**: MIT  
**Author**: Hunter (via Claude)

For questions or contributions, please refer to the main documentation.
