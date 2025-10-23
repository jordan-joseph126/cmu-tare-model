# Environment Setup Guide

## Environment Information

- **Environment Name**: `cmu-tare-model`
- **Python Version**: 3.11.13
- **Location**: `C:\Users\14128\anaconda3\envs\cmu-tare-model`
- **Environment Files**: `environment.yml` and `environment-clean.yml`

## Quick Start

### Method 1: Launch VSCode from Anaconda Prompt (Recommended)

```bash
# Open Anaconda Prompt
conda activate cmu-tare-model
cd C:\Users\14128\Research\cmu-tare-model

# Install project package in editable mode (do this once after environment creation)
pip install -e .

# Launch VSCode
code .
```

This ensures VSCode inherits the correct conda environment variables.

**Note**: The `pip install -e .` command only needs to be run once after creating the environment, or after pulling code changes that modify the package structure.

### Method 2: Direct VSCode Launch (Requires Configuration)

If you want to launch VSCode directly without Anaconda Prompt, configure VSCode settings:

1. Open VSCode Settings (`Ctrl+,`)
2. Search for `python.condaPath`
3. Set to: `C:\Users\14128\anaconda3\Scripts\conda.exe`

Or add to your `settings.json`:

```json
{
    "python.condaPath": "C:\\Users\\14128\\anaconda3\\Scripts\\conda.exe",
    "python.terminal.activateEnvironment": true,
    "python.defaultInterpreterPath": "C:\\Users\\14128\\anaconda3\\envs\\cmu-tare-model\\python.exe"
}
```

## Environment Creation/Recreation

### If You Need to Recreate the Environment

```bash
# Remove existing environment
conda deactivate
conda env remove -n cmu-tare-model

# Recreate from environment file
conda env create -f environment-clean.yml

# Activate the environment
conda activate cmu-tare-model

# Verify installation
python --version  # Should show Python 3.11.13
python -c "import pandas; import numpy; import matplotlib; print('Packages OK')"

# Install the project package in editable mode (IMPORTANT!)
pip install -e .

# Verify project imports work
python -c "from config import PROJECT_ROOT; print(f'PROJECT_ROOT: {PROJECT_ROOT}')"
python -c "import cmu_tare_model; print('Package imported successfully!')"

# Register Jupyter kernel
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

## Troubleshooting

### Issue: ModuleNotFoundError for 'config' or Project Modules

**Symptoms**:
- `ModuleNotFoundError: No module named 'config'`
- `ModuleNotFoundError: No module named 'cmu_tare_model'`
- Imports fail in notebooks even though environment is activated

**Solution**: Install the project package in editable mode
```bash
conda activate cmu-tare-model
cd C:\Users\14128\Research\cmu-tare-model
pip install -e .

# Verify it works
python -c "from config import PROJECT_ROOT; print('Success!')"
```

Then **restart your Jupyter kernel** in VSCode before running notebook cells again.

### Issue: Wrong Python Version or Missing Packages

**Symptoms**:
- `python --version` shows wrong version (e.g., 3.12.x instead of 3.11.13)
- `ModuleNotFoundError` when importing pandas, numpy, etc.

**Solution**: Recreate the environment (see above)

### Issue: Jupyter Kernel Not Found in VSCode

**Solution**:
```bash
conda activate cmu-tare-model
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

Then in VSCode:
1. Open a notebook
2. Click kernel selector (top right)
3. Select "Python 3.11.13 (cmu-tare-model)"

### Issue: Kernel Keeps Restarting

**Solution**:
1. Close all notebooks in VSCode
2. Press `Ctrl+Shift+P`
3. Run: `Jupyter: Clear All Kernels`
4. Reopen notebook and select kernel again

### Issue: Environment Not Visible in VSCode

**Solutions**:
1. Launch VSCode from Anaconda Prompt (Method 1 above)
2. Configure `python.condaPath` in VSCode settings
3. Restart VSCode after activating environment
4. Use "Python: Select Interpreter" command (`Ctrl+Shift+P`) and manually enter the path

## Verification Steps

After setting up the environment, verify it works:

```bash
# In Anaconda Prompt with environment activated
python --version
python -c "import pandas; import numpy; import matplotlib; import scipy; import seaborn; import plotly; import statsmodels; print('All key packages OK!')"
```

## Key Packages Included

- Data Processing: pandas, numpy, openpyxl
- Visualization: matplotlib, seaborn, plotly
- Scientific Computing: scipy, statsmodels, scikit-learn
- Jupyter: jupyterlab, notebook, ipykernel, ipywidgets
- Web: requests, httpx
- And many more (see environment.yml for full list)

## Main Model Notebooks

- `tare_model_main_v2_1.ipynb` - Latest version (v2.1)
- `tare_model_main_v2.ipynb` - Version 2
- `tare_model_main_v1.2.0_downSelect.ipynb` - Version 1.2.0
- Scenario notebooks in `cmu_tare_model/model_scenarios/`

## Best Practices

1. Always activate the environment before working: `conda activate cmu-tare-model`
2. Launch VSCode from Anaconda Prompt to ensure proper environment detection
3. Use `environment-clean.yml` for recreating the environment (more portable)
4. Keep environment files updated when adding new packages
5. Test package imports after recreating the environment

## Updating the Environment

### Add New Package

```bash
conda activate cmu-tare-model
conda install package-name

# Update environment file
conda env export --no-builds > environment-clean.yml
```

### Update Existing Packages

```bash
conda activate cmu-tare-model
conda update --all
```

## Export Current Environment

To create a new environment file snapshot:

```bash
conda activate cmu-tare-model

# With builds (platform-specific)
conda env export > environment.yml

# Without builds (more portable)
conda env export --no-builds > environment-clean.yml
```

---

**Last Updated**: 2025-10-23
**Environment Status**: Working - Python 3.11.13, all packages installed
