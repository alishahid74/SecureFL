@echo off
REM ============================================================================
REM Secure Federated Learning Framework - Windows Setup Script
REM ============================================================================
REM Usage: setup.bat

echo ============================================
echo Secure Federated Learning - Setup Script
echo ============================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment
set ENV_NAME=fl_env

if exist %ENV_NAME% (
    echo Virtual environment '%ENV_NAME%' already exists.
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q %ENV_NAME%
    )
)

if not exist %ENV_NAME% (
    echo Creating virtual environment '%ENV_NAME%'...
    python -m venv %ENV_NAME%
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call %ENV_NAME%\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] pip upgraded
echo.

REM Installation menu
echo Select installation type:
echo 1) Minimal (essential packages only)
echo 2) Full (includes optional packages)
echo 3) GPU Support (CUDA-enabled PyTorch)
set /p INSTALL_CHOICE="Enter choice [1-3]: "

if "%INSTALL_CHOICE%"=="1" (
    echo Installing minimal requirements...
    pip install -r requirements-minimal.txt
) else if "%INSTALL_CHOICE%"=="2" (
    echo Installing full requirements...
    pip install -r requirements.txt
) else if "%INSTALL_CHOICE%"=="3" (
    set /p CUDA_VERSION="Enter your CUDA version (e.g., 11.8, 12.1) or 'cpu': "
    
    echo Installing PyTorch with CUDA !CUDA_VERSION!...
    if "!CUDA_VERSION!"=="cpu" (
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ) else (
        set CUDA_SHORT=!CUDA_VERSION:.=!
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu!CUDA_SHORT!
    )
    
    echo Installing remaining packages...
    pip install opacus phe networkx scikit-learn matplotlib seaborn numpy scipy tqdm
) else (
    echo [ERROR] Invalid choice. Please run the script again.
    pause
    exit /b 1
)

echo [OK] All packages installed
echo.

REM Verify installation
echo Verifying installation...
python -c "import torch, torchvision, opacus, phe, networkx as nx, numpy as np, matplotlib, seaborn, sklearn; print('\n[OK] All packages successfully installed!'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

if errorlevel 1 (
    echo [ERROR] Verification failed. Please check error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To activate the environment in the future:
echo   %ENV_NAME%\Scripts\activate.bat
echo.
echo To run the framework:
echo   python secure_federated_learning.py
echo.
echo To deactivate the environment:
echo   deactivate
echo.
echo Happy federated learning!
pause
