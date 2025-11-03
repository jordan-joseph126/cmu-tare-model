# Environment Setup Guide

## Overview

This guide walks you through setting up the `cmu-tare-model` Python environment for the TARE model codebase. The environment uses **Python 3.11.13** and includes all required packages for data analysis, visualization, and modeling.

**Environment Configuration**:
- **Name**: `cmu-tare-model` (defined in the .yml file)
- **Python Version**: 3.11.13
- **Configuration File**: `environment-cmu-tare-model.yml`
- **Project Location**: `C:\Users\14128\Research\cmu-tare-model`

---

## Prerequisites

Before starting, ensure you have:
- Anaconda or Miniconda installed
- The project repository cloned to your local machine
- Access to Anaconda Prompt (Windows) or Terminal (Mac/Linux)

---

## First-Time Setup

Follow these steps in order when setting up the environment for the first time:

### Step 1: Create the Conda Environment

```bash
# Navigate to project directory
cd C:\Users\14128\Research\cmu-tare-model

# Create environment from .yml file
# Note: The environment name 'cmu-tare-model' is specified inside the .yml file
conda env create -f environment-cmu-tare-model.yml
```

**What this does**: Creates a new isolated Python environment with all dependencies specified in the configuration file. The environment name is defined in the first line of the .yml file.

**Expected output**: Conda will download and install all packages. This may take 5-10 minutes.

### Step 2: Activate the Environment

```bash
conda activate cmu-tare-model
```

**What this does**: Switches your terminal session to use this environment's Python interpreter and packages.

**Success indicator**: Your command prompt should now show `(cmu-tare-model)` at the beginning of the line.

### Step 3: Install Project Package in Editable Mode

```bash
pip install -e .
```

**What this does**: Installs your project as a package so Python can find your custom modules (like `config` and `cmu_tare_model`). The `-e` flag means "editable" - changes to your code are immediately available without reinstalling.

**Why this matters**: Without this step, Python won't be able to import your project modules, causing `ModuleNotFoundError` even though the environment is activated.

### Step 4: Register Jupyter Kernel

```bash
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

**What this does**: Makes this environment available as a kernel option in Jupyter notebooks and VSCode.

**Why this matters**: VSCode needs to know this kernel exists to use it when running notebooks.

### Step 5: Verify Installation

```bash
# Check Python version
python --version
# Should output: Python 3.11.13

# Test core packages
python -c "import pandas; import numpy; import matplotlib; print('Core packages OK!')"

# Test project imports
python -c "from config import PROJECT_ROOT; print(f'PROJECT_ROOT: {PROJECT_ROOT}')"
python -c "import cmu_tare_model; print('Project package OK!')"
```

**Success indicators**: All commands should complete without errors and display the expected output.

---

## Daily Usage

### Opening the Project

**Recommended Method** (ensures proper environment detection):

```bash
# 1. Open Anaconda Prompt
# 2. Activate environment
conda activate cmu-tare-model

# 3. Navigate to project
cd C:\Users\14128\Research\cmu-tare-model

# 4. Launch VSCode
code .
```

**Why launch from Anaconda Prompt?** VSCode inherits the conda environment variables, ensuring it correctly detects and uses your environment.

### Selecting Kernel in VSCode

When opening a notebook:
1. Click the kernel selector in the top-right corner
2. Select **"Python 3.11.13 (cmu-tare-model)"**
3. Verify `(cmu-tare-model)` appears in the kernel indicator

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'config'` or `'cmu_tare_model'`

**Cause**: Project package not installed in editable mode.

**Solution**:
```bash
conda activate cmu-tare-model
cd C:\Users\14128\Research\cmu-tare-model
pip install -e .
```

**Then restart your Jupyter kernel** in VSCode:
- Click kernel indicator → Restart Kernel

### Issue: Wrong Python Version (e.g., 3.12.x instead of 3.11.13)

**Cause**: Environment wasn't created correctly or wrong environment is active.

**Solution**: Recreate the environment
```bash
# Deactivate current environment
conda deactivate

# Remove problematic environment
conda env remove -n cmu-tare-model

# Recreate from .yml file (start from Step 1 above)
conda env create -f environment-cmu-tare-model.yml
```

### Issue: Jupyter Kernel Not Available in VSCode

**Solution**: Re-register the kernel
```bash
conda activate cmu-tare-model
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

Then refresh VSCode's kernel list:
- Press `Ctrl+Shift+P`
- Run: **"Jupyter: Select Interpreter to Start Jupyter Server"**
- Choose the cmu-tare-model environment

### Issue: Kernel Crashes or Keeps Restarting

**Solutions**:
1. Clear all kernels:
   - Press `Ctrl+Shift+P`
   - Run: **"Jupyter: Clear All Kernels"**
2. Close all notebooks in VSCode
3. Restart VSCode
4. Reopen notebook and select kernel

### Issue: VSCode Doesn't Detect Conda Environment

**Solution**: Launch VSCode from Anaconda Prompt (see Daily Usage above)

**Alternative** (if needed):
1. Press `Ctrl+Shift+P`
2. Run: **"Python: Select Interpreter"**
3. Manually select the environment path:
   ```
   C:\Users\14128\anaconda3\envs\cmu-tare-model\python.exe
   ```

---

## Key Packages Included

Your environment includes:
- **Data Processing**: pandas, numpy, openpyxl
- **Visualization**: matplotlib, seaborn, plotly
- **Scientific Computing**: scipy, statsmodels, scikit-learn
- **Jupyter**: jupyterlab, notebook, ipykernel, ipywidgets
- **Web**: requests, httpx

See `environment-cmu-tare-model.yml` for the complete package list.

---

## Maintenance Tasks

### Adding New Packages

```bash
# Activate environment
conda activate cmu-tare-model

# Install package
conda install package-name
# Or: pip install package-name

# Update environment file to reflect changes
conda env export --no-builds > environment-cmu-tare-model-updated.yml
```

### Updating All Packages

```bash
conda activate cmu-tare-model
conda update --all
```

⚠️ **Warning**: This may cause version conflicts. Test thoroughly after updating.

### Exporting Current Environment

To create a snapshot of your current environment:

```bash
conda activate cmu-tare-model

# Cross-platform (recommended for sharing)
conda env export --no-builds > environment-clean.yml

# Platform-specific (exact reproduction on same OS)
conda env export > environment-exact.yml
```

---

## Quick Reference

**Main Model Notebook**: `tare_model_main_v2_1.ipynb`

**Scenario Notebooks**: Located in `cmu_tare_model/model_scenarios/`

**When to Run `pip install -e .`**:
- After first creating the environment
- After pulling code changes that modify package structure
- If you get `ModuleNotFoundError` for project modules

```
cmu-tare-model/
├── cmu_tare_model/                          # Main model package
│   ├── adoption_potential/                  # Adoption decision calculations
│   ├── data/                                # Input data (download separately)
│   │   ├── ami_calculations_data/           # Used for rebate eligibility
│   │   ├── euss_data/                       # Household annual energy consumption and metadata
│   │   ├── inflation_data/                  # Adjusting for inflation using BLS CPI-U
│   │   ├── fuel_prices/                     # Energy price projections
│   │   ├── marginal_social_costs/           # SCC and health damage values
│   │   ├── projections/                     # Projecting future fuel prices and grid emissions intensity, Approximating LRMER for CAPs
│   │   └── retrofit_costs/                  # Equipment and installation costs (mostly REMDB pre-2024 update)
│   ├── energy_consumption_and_metadata/     # EUSS data processing
│   ├── model_scenarios/                     # Scenario notebooks
│   │   ├── tare_baseline_v2_1.ipynb
│   │   ├── tare_basic_v2_1.ipynb
│   │   ├── tare_moderate_v2_1.ipynb
│   │   ├── tare_advanced_v2_1.ipynb
│   │   └── tare_run_simulation_v2_1.ipynb
│   ├── output_results/                      # Model outputs (created on run)
│   ├── private_impact/                      # Private cost/benefit calculations
│   ├── public_impact/                       # Climate and health impact calculations
│   ├── utils/                               # Utility and visualization functions
│   ├── tare_model_main_v2_1.ipynb           # **MAIN NOTEBOOK - START HERE**
│   └── constants.py                         # Constants used throughout the codebase
├── environment-cmu-tare-model.yml           # Conda environment (see ENVIRONMENT_SETUP.md)
├── setup.py                                 # Package installation script
├── config.py                                # Project configuration
├── ENVIRONMENT_SETUP.md                     # Detailed environment setup reference
└── TARE_MODEL_USER_GUIDE.md                 # User guide and technical documentation
```

---

**Last Updated**: 2025-10-23  
**Status**: ✅ Working - Python 3.11.13, all packages verified
