# cmu-tare-model
The Tradeoff Analysis of residential Retrofits for energy Equity (TARE) Model v2.1

<img width="8000" height="4500" alt="Figure_1" src="https://github.com/user-attachments/assets/f554ea06-0b03-4aab-823c-346b8f684c00" />

---

# Section 1: Installation and Setup

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
conda --version  # Expected: conda 25.X.X
python --version # Expected: Python 3.13.X
```

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

### First-Time Setup

**Step 1: Create the Conda Environment**

```bash
# Navigate to project directory
cd /path/to/cmu-tare-model

# Create environment from .yml file
conda env create -f environment-cmu-tare-model.yml
```

This creates an isolated Python 3.11.13 environment with all required packages (pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, jupyterlab, and more). Installation takes 5-10 minutes.

**Step 2: Activate the Environment**

```bash
conda activate cmu-tare-model
```

Your prompt should now show `(cmu-tare-model)`.

**Step 3: Install Project Package**

```bash
pip install -e .
```

The `-e` flag installs in "editable" mode, allowing Python to import your project modules (`config`, `cmu_tare_model`) without reinstalling after code changes.

**Step 4: Register Jupyter Kernel**

```bash
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

This makes the environment available in VS Code's kernel selector.

**Step 5: Verify Installation**

```bash
# Check Python version
python --version  # Should output: Python 3.11.13

# Test core packages
python -c "import pandas; import numpy; import matplotlib; print('Core packages OK!')"

# Test project imports
python -c "from config import PROJECT_ROOT; print(f'PROJECT_ROOT: {PROJECT_ROOT}')"
python -c "import cmu_tare_model; print('Project package OK!')"
```

All commands should complete without errors.

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

### Wrong Python Version (e.g., 3.12.x or 3.13.x instead of 3.11.13)

**Cause:** Environment wasn't created correctly or wrong environment is active.

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

### Updating All Packages

```bash
conda activate cmu-tare-model
conda update --all
```

**Warning:** May cause version conflicts. Test thoroughly after updating.

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

---

## Support and Questions

- **Primary Contact:** jordanjo@andrew.cmu.edu, jordanjoseph53@gmail.com
- **Repository Issues:** GitHub Issues (once public)
- **Documentation:** This README and inline code documentation (Google-style docstrings, type hints, comments)

---

**Last Updated:** 2025-11-12
