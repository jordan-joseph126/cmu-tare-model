# cmu-tare-model
The Tradeoff Analysis of residential Retrofits for energy Equity (TARE) Model v2.1

<img width="8000" height="4500" alt="Figure_1" src="https://github.com/user-attachments/assets/f554ea06-0b03-4aab-823c-346b8f684c00" />

---

# Table of Contents
## Table of Contents
1. [Section 1: Installation and Setup](#section-1-installation-and-setup)  
  1. [1.0 Prerequisites and Conda Health Check](#10-prerequisites-and-conda-health-check)  
  2. [1.1 Software Installation](#11-software-installation)  
    - [Git for Windows](#git-for-windows)  
    - [Anaconda Navigator](#anaconda-navigator)  
    - [Visual Studio Code](#visual-studio-code)  
    - [Fix PATH Integration (Critical Step)](#fix-path-integration-critical-step)  
    - [Install VS Code Extensions](#install-vs-code-extensions)  
  3. [1.2 Repository Access](#12-repository-access)  
  4. [1.3 Repository Structure](#13-repository-structure)  
  5. [1.4 Data Download](#14-data-download)  
  6. [1.5 Environment Setup](#15-environment-setup)  
    - [First-Time Setup](#first-time-setup)  
    - [Register Jupyter Kernel](#register-jupyter-kernel)  
  7. [1.6 Daily Usage](#16-daily-usage)  
  8. [1.7 Troubleshooting](#17-troubleshooting)  
  9. [1.8 Environment Maintenance](#18-environment-maintenance)  

2. [Section 2: Version Information and Attribution](#section-2-version-information-and-attribution)  
  1. [2.1 Version Information](#21-version-information)  
  2. [2.2 Licensing and Attribution](#22-licensing-and-attribution)  

3. [Support and Questions](#support-and-questions)  
4. [Last Updated](#last-updated)

# Section 1: Installation and Setup

## 1.0 Prerequisites and Conda Health Check

**IMPORTANT:** Before proceeding with software installation or environment setup, verify that you have a working Anaconda/Miniconda installation and that conda's package resolver is functioning correctly.

### Why This Matters

Conda uses a "solver" to figure out which package versions can coexist when creating environments. If the solver has version conflicts (common after Anaconda Navigator updates), environment creation may fail or produce errors. Fixing this BEFORE setup saves significant troubleshooting time.

### Quick Health Check

If you already have Anaconda or Miniconda installed, run this test:

```bash
# Open Anaconda Prompt (Windows) or Terminal (Mac/Linux)
conda --version
```

**✅ Good Output (Healthy):**
```
conda 25.9.1
```
Just the version number with no errors.

**❌ Problem Output (Needs Fixing):**
```
Error while loading conda entry point: conda-libmamba-solver 
(module 'libmambapy' has no attribute 'QueryFormat')
conda 24.11.3
```
If you see error messages before the version number, continue to the fix below.

### Fixing Conda Solver Errors

If you saw errors in the health check, fix them now:

**Step 1: Ensure base environment is active**
```bash
conda activate base
```

**Step 2: Update solver components together**
```bash
conda update -n base conda conda-libmamba-solver libmambapy
```

This downloads and installs compatible versions of all three components. Takes 2-5 minutes.

**Step 3: Verify the fix**
```bash
conda --version
```

Should now display only the version number with **NO error messages**.

**If the update fails or you prefer maximum stability:**

Switch to the classic (older, slower but rock-solid) solver:
```bash
conda config --set solver classic
```

You can always switch back to the faster libmamba solver later:
```bash
conda config --set solver libmamba
```

### Understanding the Issue

**What's happening:** Anaconda Navigator updates can cause version mismatches between:
- `conda` itself (the package manager)
- `conda-libmamba-solver` (the fast dependency resolver plugin)
- `libmambapy` (the underlying library)

**Why it matters for this project:** When you run `conda env create -f environment-cmu-tare-model.yml`, conda needs a working solver to figure out which versions of 150+ packages can work together. A broken solver means:
- ❌ Environment creation fails
- ❌ Confusing error messages
- ❌ Wasted setup time

**How the fix works:** The `conda update` command forces conda to find compatible versions of all three components at once, resolving the mismatch.

### If You Don't Have Anaconda Yet

Skip this health check for now and proceed to Section 1.1 to install Anaconda. After installation, return here and run the health check before attempting environment creation.

---

## 1.1 Software Installation

Install the following software in order before setting up the project environment:

### Git for Windows
**Download:** https://git-scm.com/download/win

**Installation settings:**
- Destination: Keep default (`C:\Program Files\Git`)
- Components: Enable Git LFS, associate .git* files, associate .sh files
- Default editor: **Visual Studio Code** (or Nano or another preferred program)
- Initial branch name: **Override to `main`**
- PATH environment: **Git from the command line and also from 3rd-party software** (Option 2)
- SSH executable: Use bundled OpenSSH
- HTTPS transport: Use OpenSSL library
- Line endings: **Checkout Windows-style, commit Unix-style** (Option 1)
- Terminal emulator: Use MinTTY
- `git pull` behavior: Default (fast-forward or merge)
- Credential helper: Git Credential Manager
- Extra options: Enable file system caching only (disable symbolic links)
- Experimental options: Leave all unchecked

**Configure Git identity: USE THE NAME AND EMAIL ASSOCIATED WITH YOUR GITHUB ACCOUNT**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Verify:**
```bash
git --version
git config --global --list
```

### Anaconda Navigator
**Download:** https://www.anaconda.com/download

**Installation settings:**
- Install type: **Just Me (recommended)**
- Destination: Keep default (`C:\Users\YourName\anaconda3`)
- Advanced options:
  - ❌ **DO NOT** check "Add Anaconda3 to my PATH environment variable"
  - ✅ **CHECK** "Register Anaconda3 as my default Python"

**Why not add to PATH?** Keeps Anaconda isolated, prevents conflicts with other software, and follows Anaconda's recommended best practice.

**Verify:**
```bash
# In regular Command Prompt - should fail
conda --version  # Expected: 'conda' is not recognized

# In Anaconda Prompt - should work
conda --version  # Expected: conda 25.X.X (with no errors)
python --version # Expected: Python 3.12.X or 3.13.X (base environment)
```

**⚠️ IMPORTANT: After installation, go back to [Section 1.0](#10-prerequisites-and-conda-health-check) and run the conda health check before proceeding.**

### Visual Studio Code
**Download:** https://code.visualstudio.com/download

**Installation settings:**
- Install type: User Installer (recommended)
- Destination: Keep default
- Additional tasks:
  - ✅ Add "Open with Code" action to file context menu
  - ✅ Add "Open with Code" action to directory context menu
  - ✅ Register Code as an editor for supported file types
  - ✅ **Add to PATH (requires shell restart)** ← CRITICAL for `code .` command

**Verify:**
```bash
# In new Command Prompt
code --version
```

### Fix PATH Integration (Critical Step)

After installing VS Code, Anaconda Prompt may not recognize the `code` command. Fix this:

```bash
# In Anaconda Prompt, run:
conda init powershell
```

Close and reopen Anaconda Prompt, then verify:
```bash
code --version  # Should now work
git --version   # Should also work
```

**Why this fix is needed:** Anaconda Prompt needs proper PowerShell initialization to preserve system PATH entries (including VS Code and Git) while adding conda directories.

### Install VS Code Extensions

Open Anaconda Prompt and run:
```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.jupyter-renderers
code --install-extension ms-toolsai.vscode-jupyter-cell-tags
code --install-extension ms-toolsai.jupyter-keymap
code --install-extension mechatroner.rainbow-csv
```

Or install via VS Code GUI: Extensions sidebar (`Ctrl+Shift+X`) → Search for each extension → Install.

## 1.2 Repository Access
**Repository Location:** https://github.com/jordan-joseph126/cmu-tare-model  
**Status:** Currently private (public release planned after documentation completion)  
**Getting Access:** Contact jordanjo@andrew.cmu.edu or jordanjoseph53@gmail.com for collaborator access

**Access Options:**

**Option 1 - Git Clone (recommended):**
```bash
# Navigate to where you want the project folder
cd C:\Users\YourName\Documents\Research

# Clone repository (creates 'cmu-tare-model' folder)
git clone https://github.com/jordan-joseph126/cmu-tare-model.git
cd cmu-tare-model
```

**Advantages:** Easy updates (`git pull`), track changes, version history, simplified collaboration

**Option 2 - Download ZIP:**
1. Navigate to repository URL
2. Click "Code" → "Download ZIP"
3. Extract to your preferred location

Best for: One-time use or if you don't have Git installed

## 1.3 Repository Structure

```
cmu-tare-model/
├── cmu_tare_model/                          # Main model package
│   ├── model_scenarios/                     # SCENARIO NOTEBOOKS (START HERE)
│   │   ├── tare_baseline_v2_1.ipynb
│   │   ├── tare_basic_v2_1.ipynb
│   │   ├── tare_moderate_v2_1.ipynb
│   │   ├── tare_advanced_v2_1.ipynb
│   │   └── tare_run_simulation_v2_1.ipynb
│   ├── private_impact/                      # Private cost/benefit calculations
│   ├── public_impact/                       # Climate and health impacts
│   ├── adoption_potential/                  # Technology adoption analysis
│   ├── energy_consumption_and_metadata/     # EUSS data processing
│   ├── utils/                               # Utility functions
│   ├── data/                                # Input data (download separately)
│   │   ├── euss_data/                       # NREL energy consumption data
│   │   ├── fuel_prices/                     # EIA price data and projections
│   │   ├── retrofit_costs/                  # REMDB equipment costs
│   │   ├── projections/                     # Cambium, NEI projections
│   │   ├── marginal_social_costs/           # SCC, health damages
│   │   └── [additional data folders...]
│   ├── output_results/                      # Model outputs (created on run)
│   ├── tare_model_main_v2_1.ipynb           # MAIN ENTRY POINT
│   └── constants.py                         # Model constants
├── environment-cmu-tare-model.yml           # Conda environment specification
├── setup.py                                 # Package installation script
├── config.py                                # Project configuration
└── README.md                                # This file
```

**Key Entry Points:**
- **Main Analysis:** `tare_model_main_v2_1.ipynb` - Start here
- **Individual Scenarios:** `model_scenarios/` folder
- **Functions/Modules:** Navigate to respective directories (e.g., `public_impact/`)

## 1.4 Data Download

The model requires input data hosted separately on Zenodo:

1. **Download data:** https://zenodo.org/records/17509167
2. **Extract All Files**
3. Unzip the data folder and extract all contents into `cmu-tare-model/cmu_tare_model/` to create the path: `cmu-tare-model/cmu_tare_model/data/`

This provides the required EUSS data, fuel prices, retrofit costs, projections, and social cost data.

## 1.5 Environment Setup

**⚠️ PREREQUISITE:** Before proceeding, ensure you completed the [conda health check in Section 1.0](#10-prerequisites-and-conda-health-check). If `conda --version` shows errors, fix them first.

### First-Time Setup

**Step 1: Create the Conda Environment**

```bash
# Navigate to project directory
cd /path/to/cmu-tare-model

# Create environment from .yml file
conda env create -f environment-cmu-tare-model.yml
```

**What this does:** Creates an isolated Python 3.11.13 environment with all required packages (pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, jupyterlab, and more).

**Expected behavior:**
- Downloads and installs 150+ packages
- Takes 5-10 minutes
- Should complete with **no error messages**

**If you see errors during creation:** Your conda solver may still have issues. Return to [Section 1.0](#10-prerequisites-and-conda-health-check) and verify the fix worked.

**Step 2: Activate the Environment**

```bash
conda activate cmu-tare-model
```

Your prompt should now show `(cmu-tare-model)`.

**Step 3: Install Project Package**

```bash
pip install -e .
```

**What this does:** Installs your project as a Python package so you can import modules like `config` and `cmu_tare_model` from any notebook.

**Why `-e` (editable mode)?** Changes to your code are immediately available without reinstalling. Critical for development and experimentation.

### Register Jupyter Kernel

```bash
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

**What this does:** Makes this environment available in VS Code's kernel selector.

**You only need to do this once** unless you recreate the environment.

### Verify Installation

```bash
# Check Python version
python --version  # Should output: Python 3.11.13

# Test core packages
python -c "import pandas; import numpy; import matplotlib; print('Core packages OK!')"

# Test project imports
python -c "from config import PROJECT_ROOT; print(f'PROJECT_ROOT: {PROJECT_ROOT}')"
python -c "import cmu_tare_model; print('Project package OK!')"
```

**Success indicators:** All commands complete without errors and display the expected output.

## 1.6 Daily Usage

### Launching the Project

**Recommended method** (ensures proper environment detection):

```bash
# 1. Open Anaconda Prompt
# 2. Activate environment
conda activate cmu-tare-model

# 3. Navigate to project
cd /path/to/cmu-tare-model

# 4. Launch VS Code
code .
```

**Why from Anaconda Prompt?** VS Code inherits conda environment variables, ensuring correct environment detection.

### Running Notebooks

1. Open `cmu_tare_model/tare_model_main_v2_1.ipynb`
2. Click kernel selector (top-right corner)
3. Select **"Python 3.11.13 (cmu-tare-model)"**
4. Verify `(cmu-tare-model)` appears in kernel indicator
5. Run cells sequentially from top to bottom

## 1.7 Troubleshooting

### Conda Health Issues (Solver Errors)

**Symptoms:**
- `conda env create` fails or shows warnings
- Messages about "conda entry point" or "libmambapy"
- Package resolution takes extremely long

**Solution:** See [Section 1.0: Prerequisites and Conda Health Check](#10-prerequisites-and-conda-health-check)

This issue affects conda itself (base environment), not your project environment. Fix it in the base environment before creating project environments.

### `code` Command Not Working in Anaconda Prompt

**Symptoms:** `code --version` works in Command Prompt but fails in Anaconda Prompt with "'code' is not recognized"

**Solution:**
```bash
# In Anaconda Prompt:
conda init powershell
```

Close and reopen Anaconda Prompt. The `code` command should now work.

**Why this happens:** Anaconda Prompt's initialization may not preserve system PATH entries. Running `conda init powershell` creates a proper PowerShell profile that preserves VS Code's PATH entry while adding conda directories.

### `ModuleNotFoundError: No module named 'config'` or `'cmu_tare_model'`

**Cause:** Project package not installed in editable mode.

**Solution:**
```bash
conda activate cmu-tare-model
cd /path/to/cmu-tare-model
pip install -e .
```

Then **restart the Jupyter kernel** in VS Code: Click kernel indicator → Restart Kernel

**Why this happens:** Python needs to know where to find your project modules. `pip install -e .` tells Python "this directory contains importable packages."

### Wrong Python Version (e.g., 3.12.x or 3.13.x instead of 3.11.13)

**Cause:** Environment wasn't created correctly or wrong environment is active.

**Check which environment is active:**
```bash
conda env list
# Look for the * symbol showing active environment
```

**Solution:** Recreate the environment
```bash
conda deactivate
conda env remove -n cmu-tare-model
conda env create -f environment-cmu-tare-model.yml
```

Then repeat Steps 2-4 from Section 1.5.

### Jupyter Kernel Not Available in VS Code

**Solution:** Re-register the kernel
```bash
conda activate cmu-tare-model
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

Then refresh VS Code's kernel list:
- Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
- Run: **"Jupyter: Select Interpreter to Start Jupyter Server"**
- Choose the cmu-tare-model environment

### Kernel Crashes or Keeps Restarting

**Solutions:**
1. Clear all kernels: `Ctrl+Shift+P` → "Jupyter: Clear All Kernels"
2. Close all notebooks in VS Code
3. Restart VS Code
4. Reopen notebook and select kernel

### VS Code Doesn't Detect Conda Environment

**Primary solution:** Launch VS Code from Anaconda Prompt (see Section 1.6)

**Alternative:**
1. Press `Ctrl+Shift+P`
2. Run: **"Python: Select Interpreter"**
3. Manually select: `[path-to-anaconda]/envs/cmu-tare-model/python.exe`

**Typical paths:**
- Windows: `C:\Users\YourName\anaconda3\envs\cmu-tare-model\python.exe`
- Mac: `/Users/YourName/anaconda3/envs/cmu-tare-model/bin/python`
- Linux: `/home/YourName/anaconda3/envs/cmu-tare-model/bin/python`

### Environment Creation Hangs or Takes Very Long

**Cause:** Conda solver is working but struggling with dependency resolution.

**Solutions:**

1. **Switch to classic solver** (slower but more reliable):
   ```bash
   conda config --set solver classic
   conda env create -f environment-cmu-tare-model.yml
   ```

2. **Use mamba** (faster alternative):
   ```bash
   conda install -n base mamba
   mamba env create -f environment-cmu-tare-model.yml
   ```

3. **Create with explicit channels**:
   ```bash
   conda env create -f environment-cmu-tare-model.yml --channel defaults
   ```

## 1.8 Environment Maintenance

### Adding New Packages

```bash
conda activate cmu-tare-model
conda install package-name
# Or: pip install package-name
```

To update the environment file:
```bash
conda env export --no-builds > environment-cmu-tare-model.yml
```

**Best practice:** Use `conda install` for packages available through conda channels (faster, better dependency resolution). Use `pip install` only for packages not available through conda.

### Updating All Packages

```bash
conda activate cmu-tare-model
conda update --all
```

**⚠️ Warning:** May cause version conflicts or break compatibility with notebooks. Always test thoroughly after updating.

**Safer approach:** Update specific packages individually and test between updates.

### Recreating Environment from Scratch

If your environment becomes corrupted or you want a clean slate:

```bash
# Remove old environment
conda env remove -n cmu-tare-model

# Recreate from .yml file
conda env create -f environment-cmu-tare-model.yml

# Reinstall project package
conda activate cmu-tare-model
pip install -e .

# Re-register kernel
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

### Exporting Your Environment

To share your exact environment with collaborators:

**Cross-platform (recommended):**
```bash
conda env export --no-builds > environment-shared.yml
```

**Platform-specific (exact reproduction):**
```bash
conda env export > environment-exact.yml
```

---

# Section 2: Version Information and Attribution

## 2.1 Version Information

**Current Version:** 2.1

**Development Status:** Production/Stable

**Update Frequency:** Research-driven updates as model improvements are developed

**Checking for Updates:**
```bash
git fetch origin
git pull origin main
python setup.py --version
```

**Version History:**
- **v2.1** (Current): Comprehensive documentation, environment standardization
- **v2.0**: Major refactoring and modularization
- **v1.x**: Initial development versions

## 2.2 Licensing and Attribution

**License:** MIT License (planned; to be finalized before public release)

**Author:** Jordan Joseph  
**Affiliation:** Carnegie Mellon University  
**Contact:** jordanjo@andrew.cmu.edu, jordanjoseph53@gmail.com

**Citation (Planned):**
```
Joseph, J. (2025). TARE Model: Tradeoff Analysis of Residential Retrofits for Energy Equity. 
Carnegie Mellon University. https://github.com/jordan-joseph126/cmu-tare-model
```

**Intended Usage:**
- Research and academic use
- Modification and extension for research purposes
- Integration into other research projects
- Commercial use permissions to be specified in final license

**Attribution Requirements:**
- Cite the TARE model in publications using the tool
- Reference the GitHub repository in code documentation
- Acknowledge Carnegie Mellon University as institutional affiliation

**Data Sources and Acknowledgments:**

This model uses publicly available data from:
- **NREL ResStock/EUSS**: Residential building characteristics and energy consumption
- **EIA**: Energy price data and projections
- **REMDB**: Residential equipment costs
- **EPA**: Grid emissions factors
- **Cambium**: Future grid scenarios
- **EPA/EASIUR**: Social cost estimates

## Support and Questions

- **Primary Contact:** jordanjo@andrew.cmu.edu, jordanjoseph53@gmail.com
- **Repository Issues:** GitHub Issues (once public)
- **Documentation:** This README and inline code documentation (Google-style docstrings, type hints, comments)

**Common Support Topics:**
- Environment setup troubleshooting
- Data download and integration
- Model usage and interpretation
- Extension and customization
- Collaboration opportunities

---

# Appendices

## Appendix 1: EUSS Data IntegriTy Issues (Release 1 --> Release 1.1)

The NREL national EUSS CSV file downloaded from NREL's website is **missing 40+ critical metadata columns** that are present in the individual state-level CSV files. 

These missing columns include:
- HVAC sizing parameters (`size_heating_system_primary_k_btu_h`, etc.)
- Peak load variables (`peak_when_cooling.kw`, `peak_when_heating.kw`)
- Envelope surface areas (roof, walls, windows, ducts)
- Hot water end-use disaggregation
- Infiltration and ventilation parameters
- Unmet heating and cooling hours data

**Using the national CSV directly would break the TARE model's cost estimation functions, which require these parameters for:**
- Equipment sizing and replacement cost calculations (Tables S1-S2)
- Enclosure upgrade cost calculations (Table S3)
- Peak load analysis

### Initial Workaround Implemented in TARE (using Release 1 Data)
**Step by step solution outline:**
1. Verifies all state-level CSV files are present
2. Combines state-level files into a complete dataset
3. Compares column counts to document the issue
4. Identifies specific missing columns
5. Validates the combined dataset

### Release 1.1 Resolved this issue 
Release 1.1 resolved the issue, so we now use NREL's national files directly for baseline and upgrade data without any reconstruction step.

```
RELEASE 1.1

BASELINE DATA:
    Baseline National Columns: 260
    Baseline PA Columns: 260
    Missing Columns in National vs PA: 0

MEASURE PACKAGE 8 DATA:
    Upgrade08 National Columns: 348
    Upgrade08 PA Columns: 348
    Missing Columns in National vs PA: 0

MEASURE PACKAGE 9 DATA:
    Upgrade09 National Columns: 352
    Upgrade09 PA Columns: 352
    Missing Columns in National vs PA: 0

MEASURE PACKAGE 10 DATA:
    Upgrade10 National Columns: 355
    Upgrade10 PA Columns: 355
    Missing Columns in National vs PA: 0
```

---

**Last Updated:** 2025-11-27  
