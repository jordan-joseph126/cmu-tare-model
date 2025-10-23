# TARE Model User Guide & Technical Documentation

**Version 2.1**
**Last Updated**: 2025-10-23

## About This Guide

This guide is designed to help researchers, policymakers, and practitioners with basic programming experience download, set up, and run the TARE (Tradeoff Analysis of residential Retrofits for energy Equity) Model. The TARE Model evaluates the economic viability and environmental impacts of residential building electrification and energy efficiency retrofits across the United States.

### What You'll Learn
- How to download and install all necessary software
- How to set up your Python environment
- How to run the TARE model and interpret results
- How to troubleshoot common issues

### Time Commitment
- Software installation: 30-60 minutes
- Environment setup: 15-30 minutes
- First model run: 1-2 hours (varies by geographic scope)

### Prerequisites
- Basic familiarity with command-line interfaces
- At least 10 GB of free disk space
- Stable internet connection for downloading data and software

---

# Table of Contents

## Part 1: Getting Started
- [Section 1.1: Downloading from GitHub](#section-11-downloading-from-github)
- [Section 1.2: Software Installation](#section-12-software-installation)
- [Section 1.3: Environment Setup](#section-13-environment-setup)

## Part 2: Running the Model
- [Section 2.1: Data Setup](#section-21-data-setup)
- [Section 2.2: Opening and Running the Jupyter Notebook](#section-22-opening-and-running-the-jupyter-notebook)
- [Section 2.3: Understanding Output](#section-23-understanding-output)

## Appendices
- [Appendix A: Troubleshooting Common Issues](#appendix-a-troubleshooting-common-issues)
- [Appendix B: Understanding Model Scenarios](#appendix-b-understanding-model-scenarios)
- [Appendix C: Glossary of Terms](#appendix-c-glossary-of-terms)

---

# PART 1: GETTING STARTED

## Section 1.1: Downloading from GitHub

### What is GitHub?

GitHub is a platform for hosting and sharing code. The TARE Model is hosted on GitHub, making it easy for you to download and stay updated with the latest versions.

### Repository Overview

The TARE Model repository contains:
- **Model code**: Python scripts and Jupyter notebooks for running simulations
- **Configuration files**: Environment setup files for managing dependencies
- **Data processing tools**: Scripts for analyzing and visualizing results
- **Documentation**: This guide and technical references

### Step-by-Step Download Instructions

#### Option 1: Download as ZIP (Recommended for Beginners)

1. **Navigate to the repository**:
   - Open your web browser
   - Go to: `[GITHUB-URL]` *(To be provided)*

2. **Download the code**:
   - Look for the green **"Code"** button near the top right of the page
   - Click the **"Code"** button
   - Select **"Download ZIP"** from the dropdown menu

3. **Extract the files**:
   - **Windows**: Right-click the downloaded ZIP file → Select "Extract All..." → Choose a location (e.g., `C:\Users\YourName\Research\`) → Click "Extract"
   - **Mac**: Double-click the ZIP file (it will extract automatically to the same folder)
   - **Linux**: Right-click the ZIP file → Select "Extract Here" or use terminal: `unzip cmu-tare-model-main.zip`

4. **Remember your location**:
   - Note where you extracted the files (you'll need this path later)
   - Example: `C:\Users\YourName\Research\cmu-tare-model`

#### Option 2: Clone Using Git (For Users Familiar with Git)

If you have Git installed:

```bash
# Navigate to where you want to store the project
cd C:\Users\YourName\Research

# Clone the repository
git clone [GITHUB-URL]

# Navigate into the project directory
cd cmu-tare-model
```

**Benefits of Git Cloning**:
- Easy to update to the latest version (`git pull`)
- Track your own changes
- Contribute improvements back to the project

### File Structure Explanation

After downloading, you should see this structure:

```
cmu-tare-model/
├── cmu_tare_model/                    # Main model package
│   ├── adoption_potential/            # Adoption decision analysis
│   ├── data/                          # Input data (empty until you download data)
│   │   ├── equity_data/
│   │   ├── fuel_prices/
│   │   ├── inflation_data/
│   │   ├── marginal_social_costs/
│   │   ├── projections/
│   │   └── retrofit_costs/
│   ├── energy_consumption_and_metadata/ # Energy use data processing
│   ├── model_scenarios/               # Simulation notebooks
│   │   ├── tare_baseline_v2_1.ipynb
│   │   ├── tare_basic_v2_1.ipynb
│   │   ├── tare_moderate_v2_1.ipynb
│   │   ├── tare_advanced_v2_1.ipynb
│   │   └── tare_run_simulation_v2_1.ipynb
│   ├── output_results/                # Model outputs go here
│   ├── private_impact/                # Private cost calculations
│   ├── public_impact/                 # Public benefit calculations
│   ├── tests/                         # Test suites
│   ├── utils/                         # Utility functions
│   ├── tare_model_main_v2_1.ipynb    # **MAIN NOTEBOOK - START HERE**
│   └── docs/                          # Additional documentation
├── reference_material/                # Background documentation
├── environment.yml                    # Conda environment specification
├── environment-clean.yml             # Cross-platform environment file
├── setup.py                          # Package installation script
├── config.py                         # Project configuration
├── README.md                         # Project overview
├── ENVIRONMENT_SETUP.md              # Environment setup guide
└── TARE_MODEL_USER_GUIDE.md         # This file
```

### Key Files to Know

| File | Purpose |
|------|---------|
| `tare_model_main_v2_1.ipynb` | **Main entry point** - Run this to start the model |
| `environment-clean.yml` | Contains list of required software packages |
| `config.py` | Sets up project paths automatically |
| `setup.py` | Installs the TARE model as a Python package |
| `ENVIRONMENT_SETUP.md` | Detailed environment configuration instructions |

### What's Next?

Now that you have the code downloaded, the next step is to install the necessary software (Python, Anaconda Navigator, and VS Code).

---

## Section 1.2: Software Installation

To run the TARE Model, you need three software tools:
1. **Anaconda Navigator**: Manages Python and packages
2. **VS Code**: Code editor and notebook interface
3. **Python 3.11.13**: Programming language (installed automatically with Anaconda)

### Why These Tools?

- **Anaconda Navigator**: Makes it easy to manage different Python environments and ensures all required packages are installed
- **VS Code**: Provides a user-friendly interface for running Jupyter notebooks and viewing code
- **Python 3.11.13**: The specific Python version tested with this model

### Installation Instructions by Operating System

---

#### Windows Installation

##### Step 1: Install Anaconda Navigator

1. **Download Anaconda**:
   - Go to: https://www.anaconda.com/download
   - Click **"Download"** button (should automatically detect Windows)
   - The file will be named something like `Anaconda3-2025.XX-Windows-x86_64.exe`

2. **Run the installer**:
   - Double-click the downloaded `.exe` file
   - Click **"Next"** through the welcome screens
   - **License Agreement**: Click "I Agree"
   - **Installation Type**: Select "Just Me" (recommended)
   - **Installation Location**: Use the default or choose your preferred location
     - Default: `C:\Users\YourName\anaconda3`
   - **Advanced Options**:
     - ✅ Check "Add Anaconda3 to my PATH environment variable" (makes life easier)
     - ✅ Check "Register Anaconda3 as my default Python 3.11"
   - Click **"Install"** (this takes 5-10 minutes)
   - Click **"Finish"** when complete

3. **Verify installation**:
   - Press `Windows Key` → Type "Anaconda Prompt"
   - Open **Anaconda Prompt**
   - Type: `conda --version`
   - You should see something like: `conda 24.X.X`

##### Step 2: Install VS Code

1. **Download VS Code**:
   - Go to: https://code.visualstudio.com/
   - Click **"Download for Windows"**
   - File will be named: `VSCodeUserSetup-x64-X.XX.X.exe`

2. **Run the installer**:
   - Double-click the downloaded `.exe` file
   - Accept the license agreement
   - Choose installation location (default is fine)
   - **Important**: Check these boxes:
     - ✅ "Add to PATH"
     - ✅ "Create a desktop icon"
     - ✅ "Add 'Open with Code' action to Windows Explorer context menu"
   - Click **"Install"**
   - Click **"Finish"** to launch VS Code

3. **Install Python extension**:
   - Open VS Code
   - Click the **Extensions** icon on the left sidebar (looks like four squares)
   - Search for "Python"
   - Find the extension by **Microsoft** (should be the first result)
   - Click **"Install"**
   - Also install **"Jupyter"** extension by Microsoft

4. **Verify installation**:
   - Press `Windows Key` → Type "cmd"
   - Open **Command Prompt**
   - Type: `code --version`
   - You should see the version number

---

#### macOS Installation

##### Step 1: Install Anaconda Navigator

1. **Download Anaconda**:
   - Go to: https://www.anaconda.com/download
   - Click **"Download"** (automatically detects Mac)
   - Choose the appropriate version:
     - **Intel Mac**: Select "64-Bit (x86) Installer"
     - **M1/M2/M3 Mac**: Select "64-Bit (Apple Silicon) Installer"
   - File will be named: `Anaconda3-2025.XX-MacOSX-*.pkg`

2. **Run the installer**:
   - Double-click the downloaded `.pkg` file
   - Click **"Continue"** through the introduction
   - **License**: Click "Continue" → "Agree"
   - **Installation Type**: "Install for me only" (default)
   - Click **"Install"**
   - Enter your Mac password when prompted
   - Wait 5-10 minutes for installation
   - Click **"Close"** when finished

3. **Verify installation**:
   - Open **Terminal** (Applications → Utilities → Terminal)
   - Type: `conda --version`
   - If you see an error, close and reopen Terminal, then try again
   - You should see: `conda 24.X.X`

##### Step 2: Install VS Code

1. **Download VS Code**:
   - Go to: https://code.visualstudio.com/
   - Click **"Download Mac Universal"**
   - File will be: `VSCode-darwin-universal.zip`

2. **Install VS Code**:
   - Double-click the downloaded ZIP (it will extract automatically)
   - Drag **"Visual Studio Code.app"** to your **Applications** folder
   - Open **Applications** folder
   - Double-click **"Visual Studio Code"**
   - If you see a security warning, click **"Open"**

3. **Install Python extension**:
   - In VS Code, click the **Extensions** icon (left sidebar)
   - Search for "Python"
   - Install the **Python** extension by Microsoft
   - Also install **Jupyter** extension by Microsoft

4. **Add to PATH** (makes opening VS Code from Terminal easier):
   - Open VS Code
   - Press `Cmd+Shift+P` to open Command Palette
   - Type: "shell command"
   - Select **"Shell Command: Install 'code' command in PATH"**
   - You should see: "Shell command 'code' successfully installed"

5. **Verify installation**:
   - Open **Terminal**
   - Type: `code --version`
   - You should see the version number

---

#### Linux Installation (Ubuntu/Debian)

##### Step 1: Install Anaconda Navigator

1. **Download Anaconda**:
   - Go to: https://www.anaconda.com/download
   - Click **"Download"** for Linux
   - Or use Terminal:
   ```bash
   cd ~/Downloads
   wget https://repo.anaconda.com/archive/Anaconda3-2025.XX-Linux-x86_64.sh
   ```

2. **Install Anaconda**:
   ```bash
   # Make the installer executable
   chmod +x Anaconda3-2025.XX-Linux-x86_64.sh

   # Run the installer
   bash Anaconda3-2025.XX-Linux-x86_64.sh

   # Follow the prompts:
   # - Press ENTER to review license
   # - Type 'yes' to accept license
   # - Press ENTER to confirm installation location
   # - Type 'yes' when asked to initialize Anaconda
   ```

3. **Activate Anaconda**:
   ```bash
   # Close and reopen your terminal, or run:
   source ~/.bashrc
   ```

4. **Verify installation**:
   ```bash
   conda --version
   # Should output: conda 24.X.X
   ```

##### Step 2: Install VS Code

1. **Install using package manager** (easiest):
   ```bash
   # Update package list
   sudo apt update

   # Install dependencies
   sudo apt install software-properties-common apt-transport-https wget

   # Add Microsoft GPG key
   wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -

   # Add VS Code repository
   sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"

   # Install VS Code
   sudo apt update
   sudo apt install code
   ```

2. **Verify installation**:
   ```bash
   code --version
   ```

3. **Install Python extensions**:
   - Open VS Code: `code`
   - Click Extensions icon (left sidebar)
   - Install **Python** extension by Microsoft
   - Install **Jupyter** extension by Microsoft

---

### Verification Steps (All Operating Systems)

After installing both Anaconda and VS Code, verify everything works:

1. **Check Anaconda**:
   - Windows: Open "Anaconda Prompt"
   - Mac/Linux: Open "Terminal"
   - Run: `conda --version`
   - Run: `python --version`
   - Both should return version numbers

2. **Check VS Code**:
   - Open VS Code
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type: "Python: Select Interpreter"
   - You should see at least one Python interpreter listed

3. **Check Integration**:
   - Open Anaconda Navigator (find it in your applications)
   - You should see a grid of applications including Jupyter, VS Code, and others
   - This confirms Anaconda is properly installed

### What's Next?

Now that you have all the software installed, you're ready to set up your Python environment specifically for the TARE Model.

---

## Section 1.3: Environment Setup

### What is a Python Environment?

Think of a Python environment as a self-contained workspace for a specific project. It includes:
- A specific Python version (3.11.13 for TARE)
- All required packages (pandas, numpy, matplotlib, etc.)
- Isolated from other Python projects to prevent conflicts

**Why do we need this?**
- Different projects may need different package versions
- Keeps your system Python installation clean
- Makes it easy to share your setup with others
- Ensures reproducibility

### Understanding the Environment Files

The TARE Model includes two environment configuration files:

| File | Purpose | When to Use |
|------|---------|-------------|
| `environment-clean.yml` | Cross-platform, no build numbers | **Recommended** - Works on Windows, Mac, and Linux |
| `environment.yml` | Windows-specific with exact build numbers | Use if `environment-clean.yml` has issues on Windows |

---

### Creating the Environment

#### Step 1: Open the Right Terminal

Different operating systems use different terminals for conda commands:

**Windows:**
1. Press `Windows Key`
2. Type: "Anaconda Prompt"
3. Click **"Anaconda Prompt"** (NOT "Command Prompt")

**Mac:**
1. Press `Cmd+Space` (opens Spotlight)
2. Type: "Terminal"
3. Press `Enter`

**Linux:**
1. Press `Ctrl+Alt+T`
2. Or find Terminal in your applications

#### Step 2: Navigate to Your Project Directory

Use the `cd` (change directory) command to navigate to where you downloaded the TARE Model:

**Windows:**
```bash
# Example - adjust to your actual path
cd C:\Users\YourName\Research\cmu-tare-model
```

**Mac/Linux:**
```bash
# Example - adjust to your actual path
cd ~/Research/cmu-tare-model
```

**Tip**: You can drag and drop the folder into the terminal window on most systems to auto-fill the path!

#### Step 3: Create the Environment

Now we'll create the conda environment using the YAML file:

```bash
# This command creates an environment named 'cmu-tare-model'
conda env create -f environment-clean.yml
```

**What happens next:**
- Conda will read the `environment-clean.yml` file
- It will download Python 3.11.13
- It will download and install ~200 packages
- This takes **10-20 minutes** depending on your internet speed
- You'll see progress updates as packages download and install

**Expected output:**
```
Collecting package metadata (repodata.json): done
Solving environment: done
Downloading and Extracting Packages:
...
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate cmu-tare-model
#
```

**If you encounter errors:**
- Try the Windows-specific file: `conda env create -f environment.yml`
- See [Troubleshooting](#troubleshooting-environment-creation) section below

#### Step 4: Activate the Environment

Once creation is complete, activate the environment:

```bash
conda activate cmu-tare-model
```

**How to tell it worked:**
- Your terminal prompt should change
- You'll see `(cmu-tare-model)` at the beginning of your command line
- Example: `(cmu-tare-model) C:\Users\YourName\Research\cmu-tare-model>`

#### Step 5: Verify the Environment

Check that Python and packages are correctly installed:

```bash
# Check Python version (should be 3.11.13)
python --version

# Test key packages
python -c "import pandas; import numpy; import matplotlib; print('Packages OK!')"
```

**Expected output:**
```
Python 3.11.13
Packages OK!
```

---

### Installing the TARE Package

The TARE Model needs to be installed as a Python package so that all modules can find each other.

#### Step 6: Install in Editable Mode

With your environment still activated:

```bash
# Install the TARE package in editable/development mode
pip install -e .
```

**The dot (.) is important!** It means "current directory"

**Expected output:**
```
Obtaining file:///C:/Users/YourName/Research/cmu-tare-model
  Preparing metadata (setup.py) ... done
Installing collected packages: cmu-tare-model
  Running setup.py develop for cmu-tare-model
Successfully installed cmu-tare-model-2.0
```

**What does `-e` mean?**
- `-e` stands for "editable mode"
- Changes you make to code files are immediately reflected
- No need to reinstall after modifying code

#### Step 7: Verify Package Installation

Test that the package is correctly installed:

```bash
# Test importing the config module
python -c "from config import PROJECT_ROOT; print(f'PROJECT_ROOT: {PROJECT_ROOT}')"

# Test importing the main package
python -c "import cmu_tare_model; print('Package imported successfully!')"
```

**Expected output:**
```
Project root directory: C:\Users\YourName\Research\cmu-tare-model
PROJECT_ROOT: C:\Users\YourName\Research\cmu-tare-model
Package imported successfully!
```

---

### Setting Up Jupyter Kernel

To use this environment in Jupyter notebooks:

```bash
# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

**Expected output:**
```
Installed kernelspec cmu-tare-model in C:\Users\YourName\AppData\Roaming\jupyter\kernels\cmu-tare-model
```

This makes your environment available as a kernel option when you run Jupyter notebooks in VS Code.

---

### Launching VS Code with the Environment

For the best experience, launch VS Code from the Anaconda Prompt while the environment is activated:

```bash
# Make sure environment is activated
conda activate cmu-tare-model

# Navigate to project directory (if not already there)
cd C:\Users\YourName\Research\cmu-tare-model

# Launch VS Code
code .
```

**The dot (.) means "open the current directory"**

**What this does:**
- VS Code opens with the project folder
- Environment variables are properly set
- VS Code will automatically detect the `cmu-tare-model` environment

---

### Troubleshooting Environment Creation

#### Issue: "Solving environment: failed"

**Possible causes:**
- Network connectivity issues
- Conflicts in package versions
- Corrupted package cache

**Solutions:**
```bash
# Clear conda cache and try again
conda clean --all
conda env create -f environment-clean.yml

# Or try with more relaxed solving
conda env create -f environment-clean.yml --force
```

#### Issue: Wrong Python Version Installed

**Check current version:**
```bash
python --version
```

**If it's not 3.11.13:**
```bash
# Remove the environment
conda deactivate
conda env remove -n cmu-tare-model

# Recreate it
conda env create -f environment-clean.yml
```

#### Issue: Package Import Errors

**If you see `ModuleNotFoundError`:**

1. **Verify environment is activated:**
   ```bash
   # You should see (cmu-tare-model) in your prompt
   conda env list
   # cmu-tare-model should have an asterisk (*)
   ```

2. **Reinstall the package:**
   ```bash
   pip install -e . --force-reinstall
   ```

3. **Check specific package:**
   ```bash
   # List all installed packages
   conda list

   # Install missing package
   conda install package-name
   ```

#### Issue: "setup.py egg_info failed" During pip install

**This is usually an encoding issue.**

**Solution:**
```bash
# The setup.py file should already have the fix, but if not:
# Edit setup.py and change line 22 from:
# long_description=open("README.md").read(),
# to:
# long_description=open("README.md", encoding="utf-8").read(),
```

---

### OS-Specific Command Differences

| Task | Windows (Anaconda Prompt) | Mac/Linux (Terminal) |
|------|---------------------------|----------------------|
| Activate environment | `conda activate cmu-tare-model` | `conda activate cmu-tare-model` |
| Navigate directories | `cd C:\path\to\folder` | `cd /path/to/folder` |
| List files | `dir` | `ls` |
| Show current directory | `cd` | `pwd` |
| Clear screen | `cls` | `clear` |

---

### Environment Management Commands

**Useful commands to know:**

```bash
# List all environments
conda env list

# Deactivate current environment
conda deactivate

# Activate the TARE environment
conda activate cmu-tare-model

# Update all packages in environment
conda update --all

# Export environment to file
conda env export --no-builds > my-environment.yml

# Remove environment (if needed)
conda deactivate
conda env remove -n cmu-tare-model
```

---

### What's Next?

Now that your environment is set up and working, you're ready to:
1. Download the required data files
2. Open the main notebook
3. Run your first simulation!

Continue to **Part 2: Running the Model** →


---

# PART 2: RUNNING THE MODEL

## Section 2.1: Data Setup

### Understanding the Data Requirements

The TARE Model requires external data files that are too large to store in the GitHub repository. These files contain:
- **Energy use data** from the U.S. Energy Information Administration
- **Fuel price projections** from the Annual Energy Outlook
- **Emissions data** from various air quality models
- **Economic data** including inflation rates and discount factors
- **Retrofit cost data** for different building improvements

### Data Download Location

All required data files are hosted on Zenodo, a research data repository:

**Zenodo Link**: `[ZENODO-URL]` *(To be provided)*

**Data Package Size**: Approximately 5-8 GB (compressed)

---

### Step-by-Step Data Download

#### Step 1: Access the Zenodo Repository

1. Open your web browser
2. Navigate to the Zenodo link provided above
3. You should see a page titled "TARE Model Data Files v2.1"

#### Step 2: Download the Data Files

**Option A: Download All Files as ZIP (Recommended)**

1. Look for a "Download all" or "Download ZIP" button
2. Click to download the complete data package
3. Wait for the download to complete (5-20 minutes depending on internet speed)
4. Note the download location (usually your Downloads folder)

**Option B: Download Individual Files**

If you only need specific data files:
1. Scroll down to the "Files" section
2. Click on individual files to download:
   - `equity_data.zip`
   - `fuel_prices.zip`
   - `marginal_social_costs.zip`
   - `projections.zip`
   - `retrofit_costs.zip`

#### Step 3: Extract the Data Files

**Windows:**
1. Navigate to your Downloads folder
2. Find the downloaded ZIP file
3. Right-click → Select "Extract All..."
4. Choose destination: Your project's `cmu_tare_model/data/` folder
   - Example: `C:\Users\YourName\Research\cmu-tare-model\cmu_tare_model\data\`
5. Click "Extract"

**Mac:**
1. Navigate to your Downloads folder
2. Double-click the ZIP file (it extracts automatically)
3. Move the extracted folders to: `~/Research/cmu-tare-model/cmu_tare_model/data/`

**Linux:**
```bash
# Navigate to downloads
cd ~/Downloads

# Extract to project data directory
unzip tare-model-data.zip -d ~/Research/cmu-tare-model/cmu_tare_model/data/

# Or use file manager to extract
```

---

### Verifying Data Files Are in Correct Locations

After extraction, your data directory should look like this:

```
cmu_tare_model/data/
├── equity_data/
│   ├── income_distributions.csv
│   ├── housing_tenure.csv
│   └── ... (additional equity files)
├── fuel_prices/
│   ├── ng_prices_aeo2023.csv
│   ├── electricity_prices_aeo2023.csv
│   └── ... (additional price files)
├── inflation_data/
│   ├── cpi_data.csv
│   └── ... (inflation files)
├── marginal_social_costs/
│   ├── scc_estimates.csv
│   ├── health_damages/
│   │   ├── ap2_damages.csv
│   │   ├── easiur_damages.csv
│   │   └── inmap_damages.csv
│   └── ... (additional cost files)
├── projections/
│   ├── hdd_projections/
│   ├── energy_projections/
│   └── ... (projection files)
└── retrofit_costs/
    ├── equipment_costs.csv
    ├── installation_costs.csv
    └── ... (cost files)
```

### Quick Verification

To verify all data files are correctly placed:

**Using VS Code:**
1. Open VS Code in your project folder
2. Look at the Explorer pane on the left
3. Navigate to `cmu_tare_model` → `data`
4. You should see all the subdirectories listed above

**Using Terminal/Command Prompt:**

**Windows (Anaconda Prompt):**
```bash
cd C:\Users\YourName\Research\cmu-tare-model\cmu_tare_model\data
dir
```

**Mac/Linux (Terminal):**
```bash
cd ~/Research/cmu-tare-model/cmu_tare_model/data
ls -la
```

**You should see:**
```
equity_data/
fuel_prices/
inflation_data/
marginal_social_costs/
projections/
retrofit_costs/
```

---

### Troubleshooting Data Setup

#### Issue: "Data directory is empty"

**Symptoms**: The `data` folder exists but has no subdirectories

**Causes**:
- Data files weren't extracted to the right location
- ZIP file was double-extracted (creating nested folders)

**Solution**:
1. Find your extracted data files
2. Look inside the extracted folder - you might see another folder inside
3. Copy the subdirectories (equity_data, fuel_prices, etc.) directly into `cmu_tare_model/data/`

#### Issue: "Some data files are missing"

**Symptoms**: Some subdirectories exist but others don't

**Solution**:
1. Re-download the missing data ZIP files from Zenodo
2. Extract them to the correct location
3. Or download the complete ZIP package

#### Issue: "Permission denied" when extracting

**Windows Solution**:
- Right-click the ZIP file → Properties → Unblock → Apply
- Try extracting again

**Mac Solution**:
```bash
# Give yourself permissions
sudo chown -R $USER ~/Research/cmu-tare-model/cmu_tare_model/data
```

**Linux Solution**:
```bash
# Give yourself permissions
sudo chown -R $USER:$USER ~/Research/cmu-tare-model/cmu_tare_model/data
chmod -R 755 ~/Research/cmu-tare-model/cmu_tare_model/data
```

---

### What's Next?

With your data files in place, you're ready to open and run the main Jupyter notebook!

---

## Section 2.2: Opening and Running the Jupyter Notebook

### Overview

The TARE Model uses Jupyter notebooks as its primary interface. Notebooks combine:
- **Code**: Python commands that run the model
- **Markdown**: Text explanations of what each section does
- **Output**: Results, tables, and visualizations

### The Main Notebook

The primary entry point for the TARE Model is:
- **File**: `tare_model_main_v2_1.ipynb`
- **Location**: `cmu_tare_model/tare_model_main_v2_1.ipynb`

---

### Launching the Notebook

#### Method 1: Launch from VS Code (Recommended)

1. **Open VS Code with the correct environment**:
   - Open **Anaconda Prompt** (Windows) or **Terminal** (Mac/Linux)
   - Activate the environment:
     ```bash
     conda activate cmu-tare-model
     ```
   - Navigate to your project:
     ```bash
     cd C:\Users\YourName\Research\cmu-tare-model
     ```
   - Launch VS Code:
     ```bash
     code .
     ```

2. **Open the main notebook**:
   - In VS Code's Explorer pane (left sidebar)
   - Navigate to: `cmu_tare_model` → `tare_model_main_v2_1.ipynb`
   - Click on the file to open it

3. **Select the correct kernel**:
   - Look at the top-right corner of the notebook
   - You should see a kernel selector (shows Python version)
   - Click on it
   - Select: **"Python 3.11.13 (cmu-tare-model)"**
   - If you don't see this option:
     - Click "Select Another Kernel..."
     - Choose "Python Environments..."
     - Select `cmu-tare-model`

#### Method 2: Launch from Anaconda Navigator

1. **Open Anaconda Navigator**:
   - Windows: Start Menu → "Anaconda Navigator"
   - Mac: Applications → "Anaconda Navigator"
   - Linux: Terminal → `anaconda-navigator`

2. **Switch to the correct environment**:
   - Look for the "Applications on" dropdown (top of window)
   - Select `cmu-tare-model` from the list

3. **Launch Jupyter**:
   - Find the "Jupyter Notebook" or "JupyterLab" tile
   - Click "Launch"
   - A browser window will open

4. **Navigate to the notebook**:
   - In the browser, navigate to: `cmu_tare_model/`
   - Click on: `tare_model_main_v2_1.ipynb`

#### Method 3: Command Line Launch

```bash
# Activate environment
conda activate cmu-tare-model

# Navigate to project
cd C:\Users\YourName\Research\cmu-tare-model

# Launch Jupyter Lab
jupyter lab

# OR launch classic Jupyter Notebook
jupyter notebook
```

---

### Understanding the Notebook Structure

When you open `tare_model_main_v2_1.ipynb`, you'll see several sections:

#### 1. **Header Section (Cells 1-5)**
- Import statements
- Project configuration
- Library setup (matplotlib, pandas, seaborn)

#### 2. **Model Run Selection (Cell 6)**
**This is the first interactive part!**

```python
Would you like to begin a new simulation or visualize output results from a previous model run?
Y. I'd like to start a new model run.
N. I'd like to visualize output results from a previous model run.
```

**For your first run**: Type `Y` and press Enter

#### 3. **Scenario Selection**
After selecting `Y`, the model will ask you to specify:
- **Geographic scope**: Which state(s) or regions to analyze
- **Retrofit scenario**: Basic, Moderate, or Advanced
- **Policy scenario**: Pre-IRA or IRA-Reference

#### 4. **Model Execution**
The notebook will run the simulation (this can take 30 minutes to 2 hours)

#### 5. **Results Loading and Visualization**
After the simulation completes, subsequent cells load and visualize results

---

### Running Cells Step-by-Step

#### Understanding Cell Types

**Code Cells** (grey background):
- Contain Python code
- Click the ▶ play button to run
- Or press `Shift+Enter` to run and move to next cell

**Markdown Cells** (white background):
- Contain formatted text and explanations
- No need to run these
- Read them to understand what the next code cell does

#### Execution Order

**Important**: Cells must be run in order from top to bottom!

1. **Start at the top** (Cell 1)
2. **Run each cell** by clicking the play button or pressing `Shift+Enter`
3. **Wait for completion** before running the next cell
   - Running cells show a `[*]` indicator
   - Completed cells show a number like `[1]`, `[2]`, etc.
4. **Read the output** before continuing

#### Example Walkthrough

**Cell 1** - Import PROJECT_ROOT:
```python
from config import PROJECT_ROOT
print(f"Imported PROJECT_ROOT from config.py: {PROJECT_ROOT}")
```
**Expected Output:**
```
Project root directory: C:\Users\YourName\Research\cmu-tare-model
Imported PROJECT_ROOT from config.py: C:\Users\YourName\Research\cmu-tare-model
```

**Cell 2** - Import libraries:
```python
import pandas as pd
import matplotlib.pyplot as plt
...
```
**Expected Output:** (Usually no output, just loads libraries)

**Cell 6** - Model run selection:
```python
start_new_model_run = str(input(...))
```
**What to do:**
1. A text input box will appear below the cell
2. Type `Y` (for new run) or `N` (for visualization only)
3. Press `Enter`

---

### Model Execution Details

#### Starting a New Model Run (Type `Y`)

When you select `Y` to start a new model run:

1. **The notebook will load parameters**:
   - You'll see console output showing configuration
   - Project paths are established
   - Output folder is created

2. **Geographic filter prompt**:
   ```
   Enter state abbreviations (comma-separated) or 'ALL' for entire US:
   Example: PA,NY,CA or ALL
   ```
   **Examples:**
   - `PA` - Pennsylvania only (fastest, ~30-45 minutes)
   - `PA,NY` - Multiple states (~1-2 hours)
   - `ALL` - Entire United States (~4-8 hours)

3. **Scenario selection** (depends on the simulation notebook):
   - The main notebook runs predefined scenarios
   - Advanced users can modify scenario parameters

4. **Model execution begins**:
   ```
   Running scenario: Basic Retrofit (MP8)
   Processing: Baseline consumption...
   Calculating: Private costs...
   Calculating: Climate impacts...
   Calculating: Health impacts...
   Determining: Adoption potential...
   ```

5. **Progress indicators**:
   - You'll see progress bars or percentage completion
   - Estimated time remaining
   - Current processing step

---

### Expected Runtime and Progress Indicators

#### Runtime Estimates

| Geographic Scope | Approximate Runtime |
|------------------|---------------------|
| Single small state (e.g., DE, RI) | 20-30 minutes |
| Single medium state (e.g., PA, WA) | 30-60 minutes |
| Single large state (e.g., CA, TX, FL) | 1-2 hours |
| Multiple states (5-10) | 2-4 hours |
| Full United States | 4-8 hours |

**Factors affecting runtime:**
- Your computer's processing power
- Number of CPU cores
- Available RAM (16GB+ recommended)
- Hard drive speed (SSD recommended)

#### Monitoring Progress

**What you'll see:**

1. **Import and Setup Phase** (1-2 minutes):
   ```
   Loading libraries... ✓
   Setting up project paths... ✓
   Verifying data files... ✓
   ```

2. **Data Loading Phase** (2-5 minutes):
   ```
   Loading EUSS data for PA... ✓
   Loading fuel price projections... ✓
   Loading emissions data... ✓
   ```

3. **Calculation Phase** (bulk of runtime):
   ```
   Processing: Baseline scenario...
   Processing: Basic retrofit (MP8)...
     - Calculating space heating impacts... [####      ] 45%
     - Calculating water heating impacts... [##        ] 25%
     ...
   ```

4. **Export Phase** (1-2 minutes):
   ```
   Exporting results to CSV...
   Creating summary files...
   Saving outputs to: .../output_results/2025-10-23_14-30/
   ```

#### Is It Still Running?

**Signs the model is working correctly:**
- ✅ CPU usage is high (50-100%)
- ✅ You see progress messages updating
- ✅ Output folder size is growing
- ✅ Cell shows `[*]` indicator

**Signs something might be wrong:**
- ❌ No new output for >10 minutes
- ❌ CPU usage is 0%
- ❌ Error messages in red
- ❌ Memory usage fills up completely

**If stuck:**
1. Wait at least 10 minutes (some steps are slow but not frozen)
2. Check the last progress message
3. Look for error messages
4. If truly frozen:
   - Click the ◼ stop button in VS Code
   - Restart the kernel (Ctrl+Shift+P → "Restart Kernel")
   - Run cells again from the top

---

### Troubleshooting Notebook Execution

#### Issue: "No module named 'config'"

**Cause**: Project package not installed

**Solution**:
```bash
# In Anaconda Prompt with environment activated
pip install -e .

# Then restart your Jupyter kernel in VS Code
```

#### Issue: "FileNotFoundError: [Errno 2] No such file or directory"

**Cause**: Data files not in correct location

**Solution**:
1. Verify data files are in `cmu_tare_model/data/`
2. Check the error message for which file is missing
3. Re-download that data file from Zenodo

#### Issue: "Kernel appears to have died"

**Causes**:
- Out of memory
- Code error causing crash
- Environment issues

**Solutions**:
1. **Restart kernel**: Click "Restart" in VS Code
2. **Check memory**: Close other applications
3. **Run cells again** from the top
4. **Reduce scope**: Try a smaller state if running "ALL"

#### Issue: Cells run but output looks wrong

**Symptoms**: Empty dataframes, NaN values, zero results

**Possible causes:**
1. **Skipped cells**: Must run cells in order
2. **Wrong kernel**: Not using the cmu-tare-model environment
3. **Data file issues**: Incomplete or corrupted data

**Solutions**:
1. **Restart and run all cells in order**:
   - Click "Restart" button
   - Run cells from Cell 1 sequentially
2. **Verify kernel**: Check top-right shows "Python 3.11.13 (cmu-tare-model)"
3. **Re-verify data files**

---

### What's Next?

After the model completes successfully, you'll have results in the `output_results` folder. The next section explains how to understand and interpret these outputs.

---

## Section 2.3: Understanding Output

### Output Folder Structure

When the model completes, it creates a timestamped folder in:
```
cmu_tare_model/output_results/YYYY-MM-DD_HH-MM/
```

Example: `output_results/2025-10-23_14-30/`

#### What's Inside

```
output_results/2025-10-23_14-30/
├── baseline_summary/
│   └── summary_baseline/
│       ├── summary_Whole-Home_baseline_results_2025-10-23_14-30.csv
│       └── ... (additional baseline files)
├── retrofit_basic_summary/
│   ├── summary_basic_ap2/
│   │   ├── summary_Whole-Home_basic_results_2025-10-23_14-30.csv
│   │   ├── summary_Heating_basic_results_2025-10-23_14-30.csv
│   │   ├── summary_WaterHeating_basic_results_2025-10-23_14-30.csv
│   │   ├── summary_ClothesDrying_basic_results_2025-10-23_14-30.csv
│   │   └── summary_Cooking_basic_results_2025-10-23_14-30.csv
│   ├── summary_basic_easiur/
│   │   └── ... (same structure with EASIUR health model)
│   └── summary_basic_inmap/
│       └── ... (same structure with InMAP health model)
├── retrofit_moderate_summary/
│   ├── summary_moderate_ap2/
│   ├── summary_moderate_easiur/
│   └── summary_moderate_inmap/
└── retrofit_advanced_summary/
    ├── summary_advanced_ap2/
    ├── summary_advanced_easiur/
    └── summary_advanced_inmap/
```

---

### Understanding Output Files

#### File Naming Convention

```
summary_[Equipment-Type]_[scenario]_results_[timestamp].csv
```

**Examples:**
- `summary_Whole-Home_basic_results_2025-10-23_14-30.csv`
  - Equipment: Whole-Home (all systems combined)
  - Scenario: Basic retrofit
  - Timestamp: October 23, 2025 at 2:30 PM

- `summary_Heating_moderate_results_2025-10-23_14-30.csv`
  - Equipment: Heating only
  - Scenario: Moderate retrofit
  - Timestamp: Same run

#### Equipment Types

| File Prefix | What It Contains |
|-------------|------------------|
| `Whole-Home` | Complete house retrofit results (all equipment) |
| `Heating` | Space heating system only (ASHP) |
| `WaterHeating` | Water heater only (HPWH) |
| `ClothesDrying` | Clothes dryer only (HPCD) |
| `Cooking` | Cooking equipment only (Electric Range) |

---

### Key Output Columns

Each CSV file contains hundreds of columns. Here are the most important ones:

#### Identification Columns

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| `building_id` | Unique identifier for each dwelling | 1, 2, 3, ... |
| `state` | State abbreviation | PA, NY, CA |
| `county_name` | County name | Allegheny County |
| `urbanicity` | Urban/Rural classification | Urban, Rural |

#### Baseline Characteristics

| Column Name | Description | Example Values |
|-------------|-------------|----------------|
| `base_heating_fuel` | Current heating fuel type | Natural Gas, Electricity, Fuel Oil, Propane |
| `tenure` | Homeowner or renter | Owner-occupied, Renter-occupied |
| `income_level` | Income bracket | 0-30k, 30-60k, 60-100k, 100k+ |
| `lmi_or_mui` | Income classification | LMI (Low-to-Moderate), MUI (Middle-to-Upper) |

#### Economic Results

| Column Name | Description | Units |
|-------------|-------------|-------|
| `preIRA_mp8_heating_private_npv_moreWTP` | Private NPV without IRA (more willing to pay) | 2023 USD |
| `iraRef_mp8_heating_private_npv_moreWTP` | Private NPV with IRA rebates | 2023 USD |
| `preIRA_mp8_heating_upfront_cost` | Initial equipment + installation cost | 2023 USD |
| `preIRA_mp8_heating_lifetime_fuel_savings` | Fuel cost savings over equipment life | 2023 USD |

#### Environmental Results

| Column Name | Description | Units |
|-------------|-------------|-------|
| `preIRA_mp8_heating_climate_npv_central` | Climate benefits (central SCC estimate) | 2023 USD |
| `preIRA_mp8_heating_health_npv_ap2_acs` | Health benefits (AP2 model, ACS CR function) | 2023 USD |
| `preIRA_mp8_heating_total_npv_central_ap2_acs` | Total societal NPV (private + climate + health) | 2023 USD |

#### Adoption Decision Columns

| Column Name | Description | Values |
|-------------|-------------|--------|
| `preIRA_mp8_heating_adoption_central_ap2_acs` | Adoption decision based on total NPV | 1 (adopt), 0 (don't adopt) |
| `preIRA_mp8_heating_adoption_tier` | Adoption tier classification | Tier 1, Tier 2, Tier 3, None |

**Adoption Tiers Explained:**
- **Tier 1**: Positive private NPV (economically attractive without subsidies)
- **Tier 2**: Negative private but positive private+climate NPV
- **Tier 3**: Requires climate AND health benefits for positive NPV
- **None**: Negative NPV even with all benefits included

---

### Basic Interpretation of Results

#### Example 1: Individual Household Analysis

Let's look at one row from a results file:

```
building_id: 12345
state: PA
base_heating_fuel: Natural Gas
income_level: 60-100k
preIRA_mp8_heating_private_npv_moreWTP: -$2,500
iraRef_mp8_heating_private_npv_moreWTP: +$1,200
preIRA_mp8_heating_total_npv_central_ap2_acs: +$8,400
preIRA_mp8_heating_adoption_central_ap2_acs: 1
preIRA_mp8_heating_adoption_tier: Tier 3
```

**What this tells us:**
1. This is a natural gas-heated home in PA
2. Middle-income household (60-100k)
3. **Without IRA**: Private NPV is -$2,500 (not financially attractive)
4. **With IRA**: Private NPV becomes +$1,200 (IRA rebates make it profitable)
5. **Total societal NPV**: +$8,400 (includes climate & health benefits)
6. **Adoption decision**: Would adopt (adoption = 1)
7. **Tier 3**: Requires health benefits to be cost-effective (climate alone isn't enough)

#### Example 2: Aggregate Analysis

To analyze adoption rates for a state:

1. **Open the CSV in Excel or Python**:
   ```python
   import pandas as pd
   df = pd.read_csv('summary_Whole-Home_basic_results_2025-10-23_14-30.csv')
   ```

2. **Calculate adoption rate**:
   ```python
   # Overall adoption rate
   adoption_rate = df['preIRA_mp8_heating_adoption_central_ap2_acs'].mean() * 100
   print(f"Adoption rate: {adoption_rate:.1f}%")
   
   # Adoption rate by fuel type
   by_fuel = df.groupby('base_heating_fuel')['preIRA_mp8_heating_adoption_central_ap2_acs'].mean() * 100
   print(by_fuel)
   ```

3. **Analyze by income**:
   ```python
   by_income = df.groupby('lmi_or_mui')['iraRef_mp8_heating_adoption_central_ap2_acs'].mean() * 100
   print(by_income)
   ```

---

### Visualizations

After running the full notebook, you'll see various visualizations:

#### 1. **Adoption Potential Bar Charts**
- Shows adoption rates by fuel type and income group
- Compares Pre-IRA vs. IRA-Reference scenarios
- Separate charts for each equipment type

**How to read:**
- X-axis: Fuel type + Income group (e.g., "Natural Gas - LMI")
- Y-axis: Adoption potential (0-100%)
- Colors: Different fuel types

#### 2. **Box Plots (Uncertainty Analysis)**
- Shows distribution of NPV values
- Compares different sensitivity scenarios

**How to read:**
- Box: 25th to 75th percentile (middle 50% of data)
- Line in box: Median value
- Whiskers: Extend to min/max (or 1.5× IQR)
- Dots: Outliers

#### 3. **Histograms (Distribution Analysis)**
- Shows how many homes fall into different NPV ranges
- Vertical line at $0 shows break-even point

**How to read:**
- X-axis: NPV value
- Y-axis: Number of dwelling units
- Area left of $0: Homes with negative NPV (not cost-effective)
- Area right of $0: Homes with positive NPV (cost-effective)

---

### Common Questions About Results

#### Q: Why are some NPV values negative?

**A:** Negative NPV means the retrofit costs more over its lifetime than it saves:
- Upfront costs are too high
- Fuel savings are too small
- Electricity prices are high relative to current fuel
- Current heating system is already efficient

#### Q: What does "More WTP" vs "Less WTP" mean?

**A:** Willingness to Pay (WTP) scenarios:
- **More WTP**: Homeowner values comfort, reliability, and other non-economic benefits
  - Uses longer time horizon
  - Lower discount rate
- **Less WTP**: Homeowner focuses primarily on economics
  - Uses shorter time horizon
  - Higher discount rate

#### Q: Why are there multiple health model results (AP2, EASIUR, InMAP)?

**A:** These are different air quality models that estimate health damages from emissions:
- **AP2**: Air Pollution Prediction model
- **EASIUR**: Estimating Air pollution Social Impact Using Regression
- **InMAP**: Intervention Model for Air Pollution

Different models = different estimates = uncertainty analysis

#### Q: How do I know which scenario to focus on?

**A:** Depends on your research question:
- **Policy analysis**: Compare Pre-IRA vs. IRA-Reference
- **Technology comparison**: Compare Basic vs. Moderate vs. Advanced
- **Equity analysis**: Focus on LMI vs. MUI and fuel type differences
- **Uncertainty**: Look at the range across health models and SCC estimates

---

### Exporting and Sharing Results

#### For Publications/Reports

**Option 1: Export specific visualizations**:
- Right-click on a chart in the notebook → "Save Image As..."
- Save as PNG (for presentations) or SVG (for publications)

**Option 2: Create publication-quality figures**:
```python
# In notebook cell:
import matplotlib.pyplot as plt

# Create your plot
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...

# Save high-resolution image
plt.savefig('adoption_rates_PA.png', dpi=300, bbox_inches='tight')
plt.savefig('adoption_rates_PA.pdf', bbox_inches='tight')  # Vector format
```

#### For Collaborators

**Share the output folder**:
1. Zip the timestamped results folder
2. Include a README with:
   - Date of run
   - Geographic scope
   - Scenarios included
   - Any custom parameters used

#### For Further Analysis

**The CSV files can be opened in:**
- **Excel**: For basic analysis and pivot tables
- **R**: For advanced statistical analysis
- **Python/Pandas**: For custom analysis and visualization
- **Stata/SAS**: For econometric analysis

---

### What's Next?

Now that you understand the outputs, you might want to:
- Run different geographic scopes
- Modify scenario parameters (advanced)
- Create custom visualizations
- Perform deeper statistical analysis

For advanced customization, see the [Technical Documentation](#) *(to be created)*

---

---

# APPENDICES

## Appendix A: Troubleshooting Common Issues

### Environment Issues

#### "conda: command not found"

**Cause**: Anaconda not added to PATH or terminal not restarted

**Solution**:
- **Windows**: Use "Anaconda Prompt" instead of regular Command Prompt
- **Mac/Linux**: 
  ```bash
  # Add to PATH manually
  export PATH="$HOME/anaconda3/bin:$PATH"
  # Or restart terminal after installation
  ```

#### "Environment already exists"

**Cause**: Trying to create an environment that already exists

**Solution**:
```bash
# Option 1: Remove and recreate
conda env remove -n cmu-tare-model
conda env create -f environment-clean.yml

# Option 2: Update existing environment
conda env update -f environment-clean.yml --prune
```

#### Packages won't install / Solving environment takes forever

**Cause**: Package conflicts or slow conda solver

**Solutions**:
```bash
# Solution 1: Use mamba (faster conda solver)
conda install -n base mamba -c conda-forge
mamba env create -f environment-clean.yml

# Solution 2: Clear conda cache
conda clean --all
conda env create -f environment-clean.yml

# Solution 3: Try the platform-specific file
conda env create -f environment.yml
```

---

### Jupyter Kernel Issues

#### Kernel not found in VS Code

**Solution**:
```bash
# Reinstall ipykernel
conda activate cmu-tare-model
pip install ipykernel --force-reinstall

# Re-register kernel
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

#### "Dead kernel" / Kernel keeps crashing

**Possible causes**:
1. Out of memory
2. Code error
3. Corrupted kernel

**Solutions**:
```bash
# 1. Check memory usage
# Close other applications
# Restart computer if necessary

# 2. Reinstall jupyter packages
conda activate cmu-tare-model
conda install jupyter jupyterlab notebook --force-reinstall

# 3. Clear Jupyter cache
jupyter --paths  # Shows where files are
# Manually delete kernel folders if needed
```

#### Wrong kernel selected / Can't find cmu-tare-model kernel

**Solution in VS Code**:
1. Open notebook
2. Click kernel selector (top right)
3. Click "Select Another Kernel..."
4. Choose "Python Environments..."
5. Look for `cmu-tare-model`
6. If not visible:
   - Reload VS Code window: `Ctrl+Shift+P` → "Reload Window"
   - Re-register kernel (see above)

---

### Import Errors

#### "ModuleNotFoundError: No module named 'config'"

**Cause**: Project not installed as package

**Solution**:
```bash
conda activate cmu-tare-model
cd /path/to/cmu-tare-model
pip install -e .
```

#### "ModuleNotFoundError: No module named 'pandas'" (or other package)

**Cause**: Wrong environment activated or package not installed

**Solution**:
```bash
# Check which environment is active
conda env list
# Should show * next to cmu-tare-model

# If wrong environment:
conda activate cmu-tare-model

# If package really missing:
conda install pandas
```

#### ImportError with C extensions (Windows)

**Error example**: "ImportError: DLL load failed"

**Cause**: Missing Visual C++ redistributables

**Solution**:
- Download and install: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Restart computer
- Try importing again

---

### Data File Issues

#### "FileNotFoundError" when loading data

**Diagnosis**:
```python
# Check what path the model is looking for
import os
from config import PROJECT_ROOT
data_path = os.path.join(PROJECT_ROOT, 'cmu_tare_model', 'data')
print(f"Looking for data at: {data_path}")
print(f"Directory exists: {os.path.exists(data_path)}")
if os.path.exists(data_path):
    print(f"Contents: {os.listdir(data_path)}")
```

**Common issues**:
1. **Data extracted to wrong location**:
   - Move subdirectories to `cmu_tare_model/data/`
2. **Nested folders**: 
   - Check inside extracted folder for another folder
   - Move the innermost folders to correct location
3. **Case sensitivity (Linux/Mac)**:
   - File names must match exactly
   - `Equity_Data` ≠ `equity_data`

---

### Performance Issues

#### Model runs very slowly

**Optimizations**:
```python
# 1. Reduce geographic scope
# Instead of: 'ALL'
# Try: Single state like 'PA'

# 2. Check CPU usage
# If low, you may have parallelization disabled
# Ensure numpy/scipy use multiple cores

# 3. Close other applications
# Model needs significant RAM

# 4. Use SSD instead of HDD if possible
# Significant I/O during data loading
```

#### Out of memory errors

**Solutions**:
1. **Reduce scope**: Analyze fewer states
2. **Close applications**: Free up RAM
3. **Increase virtual memory**:
   - Windows: System Properties → Advanced → Performance Settings → Advanced → Virtual Memory
4. **Process in batches**: Run states one at a time

---

### VS Code Issues

#### VS Code doesn't detect conda environments

**Solution**:
```bash
# 1. Install Python extension
# Extensions → Search "Python" → Install

# 2. Set conda path in VS Code settings
# Ctrl+, → Search "conda path"
# Set to: C:\Users\YourName\anaconda3\Scripts\conda.exe

# 3. Reload window
# Ctrl+Shift+P → "Reload Window"
```

#### Output not showing in notebooks

**Solutions**:
1. **Check output mode**: Ensure cell isn't collapsed
2. **Restart kernel**: Sometimes output buffer gets stuck
3. **Re-run cell**: Clear output and run again

---

### Git Issues (If Using Version Control)

#### Merge conflicts in environment.yml

**Solution**:
```bash
# Accept their version (from repository)
git checkout --theirs environment.yml
git add environment.yml

# Or accept your version
git checkout --ours environment.yml
git add environment.yml

# Then recreate environment
conda env remove -n cmu-tare-model
conda env create -f environment.yml
```

#### Large output files in git

**Prevention**:
- The `.gitignore` should already exclude `output_results/`
- If committed by mistake:
  ```bash
  git rm -r --cached cmu_tare_model/output_results
  git commit -m "Remove output files from git"
  ```

---

## Appendix B: Understanding Model Scenarios

### Retrofit Scenarios

#### Baseline Scenario (MP0)
**Description**: Current housing stock with existing equipment

**Characteristics**:
- No retrofits applied
- Existing heating/cooling systems
- Current fuel types (natural gas, fuel oil, propane, electricity)
- Used as comparison baseline for retrofit scenarios

**Purpose**: Establish baseline energy consumption, costs, and emissions

---

#### Basic Retrofit (MP8)
**Description**: High-efficiency equipment electrification

**Equipment Upgrades**:
- **Space Heating**: Air-Source Heat Pump (ASHP) replacing existing system
- **Water Heating**: Heat Pump Water Heater (HPWH) replacing existing
- **Clothes Drying**: Heat Pump Clothes Dryer (HPCD) replacing electric/gas dryer
- **Cooking**: Electric resistance range replacing gas range
- **No envelope improvements**

**Typical Applications**:
- Existing homes with adequate insulation
- Moderate climates
- Equipment replacement at end-of-life

**Cost Range**: $15,000 - $25,000 (varies by home size and location)

---

#### Moderate Retrofit (MP9)
**Description**: ASHP + Basic envelope improvements

**Equipment Upgrades**:
- **Space Heating**: Air-Source Heat Pump (ASHP)
- **Plus basic enclosure upgrades**:
  - Air sealing (15% reduction in infiltration)
  - Attic insulation improvement
  - Basic window upgrades in some cases

**Typical Applications**:
- Older homes (pre-1980)
- Homes with inadequate insulation
- Cold climates where heat pumps need support

**Cost Range**: $20,000 - $35,000

**Benefits vs. Basic**:
- Reduced heating loads → smaller/cheaper ASHP
- Better ASHP performance (less extreme temperatures)
- Additional energy savings

---

#### Advanced Retrofit (MP10)
**Description**: ASHP + Enhanced envelope improvements

**Equipment Upgrades**:
- **Space Heating**: Air-Source Heat Pump (ASHP)
- **Plus enhanced enclosure upgrades**:
  - Comprehensive air sealing (30% reduction in infiltration)
  - Wall insulation (added or improved)
  - Attic/roof insulation to high R-value
  - Window replacement (high-performance)
  - Foundation/basement insulation

**Typical Applications**:
- Deep energy retrofits
- Very old homes (pre-1960)
- Extremely cold climates
- Net-zero energy goals

**Cost Range**: $35,000 - $60,000+

**Benefits vs. Moderate**:
- Maximum energy savings
- Enhanced comfort
- Better long-term value
- Enables smaller ASHP equipment

---

### Policy Scenarios

#### Pre-IRA / No IRA Scenario
**Description**: Baseline economic conditions without Inflation Reduction Act incentives

**Characteristics**:
- **No IRA rebates** for equipment
- **No IRA tax credits** for energy efficiency
- **Pre-IRA fuel price projections** (AEO2023 No IRA case)
- **Pre-IRA electricity grid** (Cambium 2021 MidCase)

**Energy Projections**:
- Slower clean energy adoption
- Higher grid emissions intensity
- More conservative fuel price forecasts

**Purpose**: Establish baseline without recent policy interventions

---

#### IRA-Reference Scenario
**Description**: Economic conditions with Inflation Reduction Act incentives

**Characteristics**:
- **IRA rebates** (HOMES and HEEHRA programs):
  - Up to $8,000 for heat pumps (space heating)
  - Up to $1,750 for heat pump water heaters
  - Up to $840 for electric stoves
  - Income-based eligibility (focused on LMI households)
- **IRA tax credits** (25C and 25D)
- **IRA-Reference fuel prices** (AEO2023 Reference Case)
- **IRA-Reference electricity grid** (Cambium 2022/2023 MidCase)

**Grid Changes**:
- Faster renewable energy adoption
- Lower grid emissions intensity (cleaner electricity)
- More aggressive decarbonization

**Equity Focus**:
- Higher rebates for LMI households (80-100% of costs)
- Lower rebates for MUI households (30-50% of costs)

**Purpose**: Evaluate policy effectiveness and equity impacts

---

### Sensitivity Parameters

#### Social Cost of Carbon (SCC)

**What it is**: Monetary value of climate damages from CO₂ emissions

**Three estimates used**:
- **Lower Bound SCC**: ~$50/ton CO₂ (conservative estimate)
- **Central SCC**: ~$190/ton CO₂ (central estimate, EPA 2023)
- **Upper Bound SCC**: ~$340/ton CO₂ (high-impact estimate)

**Why it matters**:
- Higher SCC → Greater climate benefits of electrification
- Affects adoption potential (Tier 2 adopters)
- Represents uncertainty in climate damage estimates

---

#### Health Impact Models (Reduced Complexity Models)

**Three models used for air quality health impacts**:

**1. AP2 (Air Pollution Prediction Model)**
- Developed by EPA
- Moderate spatial resolution
- Generally middle-range estimates

**2. EASIUR (Estimating Air pollution Social Impact Using Regression)**
- Regression-based approach
- Often higher damage estimates
- Detailed by source location

**3. InMAP (Intervention Model for Air Pollution)**
- High-resolution model
- Often lower than EASIUR
- Detailed urban/rural differences

**Why three models?**
- Quantify uncertainty in health damage estimates
- Different models have different strengths
- Provides range of possible health benefits

---

#### Concentration-Response Functions

**Two CR functions used**:

**1. ACS (American Cancer Society Study)**
- More conservative
- Lower mortality estimates
- Based on long-term cohort study

**2. H6C (Harvard Six Cities Study)**  
- Higher mortality estimates
- Based on seminal six-city study
- Often considered upper bound

**Why both?**
- Represents uncertainty in dose-response relationship
- ACS generally lower health benefits
- H6C generally higher health benefits
- Together: Bracket range of health impacts

---

### Adoption Tiers

#### Tier 1: Privately Cost-Effective
**Definition**: Positive Private NPV (saves homeowner money over equipment lifetime)

**Characteristics**:
- No subsidies needed for adoption
- Market-driven adoption likely
- Usually:
  - High fuel costs in baseline
  - Low electricity prices
  - Efficient heat pump performance

**Policy implication**: Focus on information/awareness campaigns

---

#### Tier 2: Climate Benefit Required
**Definition**: Negative Private NPV, but positive when climate benefits included

**Characteristics**:
- Needs policy intervention (carbon pricing or subsidies)
- Climate benefits justify societal cost
- Moderately cost-effective

**Policy implication**: Carbon pricing or climate-focused subsidies

---

#### Tier 3: Health Benefit Required  
**Definition**: Negative Private+Climate NPV, but positive when health benefits included

**Characteristics**:
- Requires both climate AND health benefits for cost-effectiveness
- Least cost-effective economically
- Often in areas with:
  - Clean electricity grids (low climate benefit)
  - High health impact potential (urban areas, fossil fuel heating)

**Policy implication**: Comprehensive subsidies justified by total social benefits

---

#### None: Not Cost-Effective
**Definition**: Negative NPV even with all benefits (private + climate + health)

**Characteristics**:
- Retrofit not recommended under current assumptions
- May become viable with:
  - Technology cost reductions
  - Cleaner grid
  - Higher fuel prices
  - Different discount rates

**Policy implication**: Wait for better technology or conditions

---

## Appendix C: Glossary of Terms

### Model-Specific Terms

**ASHP (Air-Source Heat Pump)**  
Electric heating/cooling system that transfers heat between indoor and outdoor air. Provides both heating and cooling. More efficient than resistance heating in moderate climates.

**Adoption Potential**  
Percentage of homes where retrofit is cost-effective based on total NPV (private + climate + health benefits).

**Baseline Scenario**  
Current housing stock with existing equipment and fuel types. Used as comparison point for retrofit scenarios.

**CR Function (Concentration-Response Function)**  
Mathematical relationship between air pollution concentration and health impacts. Used to estimate mortality and morbidity from emissions changes.

**EUSS (Energy Use and Savings Simulation)**  
Underlying housing dataset with building characteristics and energy use estimates.

**HOMES**  
High-Efficiency Electric Home Rebate Program (IRA program providing rebates for whole-home efficiency improvements).

**HEEHRA**  
High-Efficiency Electric Home Rebate Act (IRA program for appliance rebates).

**HPWH (Heat Pump Water Heater)**  
Electric water heater using heat pump technology. 2-3× more efficient than resistance water heaters.

**HPCD (Heat Pump Clothes Dryer)**  
Electric clothes dryer using heat pump technology instead of resistance heating. More efficient but slower drying time.

**IRA**  
Inflation Reduction Act of 2022. Major U.S. climate and clean energy legislation including rebates and tax credits for building electrification.

**LMI (Low-to-Moderate Income)**  
Households with income ≤80% of Area Median Income. Eligible for higher IRA rebates.

**Measure Package (MP)**  
Predefined set of building retrofits:
- MP0: Baseline (no changes)
- MP8: Basic (equipment only)
- MP9: Moderate (equipment + basic envelope)
- MP10: Advanced (equipment + enhanced envelope)

**MUI (Middle-to-Upper Income)**  
Households with income >80% of Area Median Income. Eligible for lower IRA rebates.

**NPV (Net Present Value)**  
Sum of all costs and benefits over equipment lifetime, discounted to present value. Positive NPV = cost-effective, Negative NPV = not cost-effective.

**RCM (Reduced Complexity Model)**  
Simplified air quality model for estimating health damages from emissions. Examples: AP2, EASIUR, InMAP.

**SCC (Social Cost of Carbon)**  
Monetary estimate of damages from one ton of CO₂ emissions. Used to value climate benefits of emission reductions.

**Tenure**  
Housing occupancy type: Owner-occupied or Renter-occupied.

**WTP (Willingness to Pay)**  
Non-economic factors affecting adoption decisions:
- More WTP: Values comfort, reliability (longer time horizon, lower discount rate)
- Less WTP: Focused on economics (shorter time horizon, higher discount rate)

---

### Economic Terms

**Discount Rate**  
Annual rate used to convert future costs/benefits to present value. Higher rate = less value on future benefits.
- More WTP: 3% discount rate
- Less WTP: 7% discount rate

**Levelized Cost**  
Average cost per unit of energy over equipment lifetime, accounting for upfront costs and annual costs.

**Time Horizon**  
Period over which costs and benefits are evaluated:
- More WTP: 15-20 years
- Less WTP: 7-10 years

**Upfront Cost**  
Initial cost of equipment purchase and installation, before any rebates or tax credits.

---

### Energy Terms

**AEO (Annual Energy Outlook)**  
U.S. Energy Information Administration's annual projection of energy supply, demand, and prices.

**Cambium**  
NREL's hourly grid emissions dataset showing current and projected electricity emissions by region.

**COP (Coefficient of Performance)**  
Heat pump efficiency metric. COP of 3.0 means 3 units of heat per 1 unit of electricity. Higher is better.

**HDD (Heating Degree Days)**  
Measure of heating demand based on outdoor temperature. More HDD = more heating needed.

**HVAC**  
Heating, Ventilation, and Air Conditioning systems.

**kWh (Kilowatt-hour)**  
Unit of energy. 1 kWh = amount of energy used by a 1000-watt appliance running for 1 hour.

**MMBtu (Million British Thermal Units)**  
Unit of energy content for natural gas, fuel oil, and propane.

**R-Value**  
Measure of insulation thermal resistance. Higher R-value = better insulation.

---

### Environmental Terms

**CO₂ (Carbon Dioxide)**  
Primary greenhouse gas from fossil fuel combustion. Major driver of climate change.

**Embodied Emissions**  
Greenhouse gas emissions from manufacturing, transport, and installation of equipment.

**Marginal Emissions**  
Emissions from the power plant that responds to changes in electricity demand (usually fossil fuel plant).

**NOₓ (Nitrogen Oxides)**  
Air pollutant from combustion. Causes respiratory health impacts and contributes to ozone formation.

**PM₂.₅ (Particulate Matter ≤2.5 microns)**  
Fine particles from combustion and other sources. Major health hazard, penetrates deep into lungs.

**SO₂ (Sulfur Dioxide)**  
Air pollutant from combustion of sulfur-containing fuels. Causes respiratory issues and acid rain.

**VOCs (Volatile Organic Compounds)**  
Organic chemicals that evaporate easily. Some are toxic, contribute to ozone formation.

**VSL (Value of Statistical Life)**  
Monetary value used in policy analysis for preventing a premature death. Used to monetize health benefits. Typically ~$10-12 million in 2023 dollars.

---

### Statistical Terms

**Median**  
Middle value when data is sorted. 50% of values above, 50% below. Less affected by outliers than mean.

**Percentile**  
Value below which a given percentage of data falls:
- 25th percentile (Q1): 25% of data below this value
- 50th percentile: Median
- 75th percentile (Q3): 75% of data below this value

**IQR (Interquartile Range)**  
Range between 25th and 75th percentiles. Contains middle 50% of data. Used to identify outliers.

**Standard Deviation**  
Measure of spread/variability in data. Larger standard deviation = more variation.

**Correlation**  
Measure of relationship between two variables. Ranges from -1 to +1:
- +1: Perfect positive correlation
- 0: No correlation
- -1: Perfect negative correlation

---

## Appendix D: Quick Reference Commands

### Environment Management

```bash
# Create environment
conda env create -f environment-clean.yml

# Activate environment
conda activate cmu-tare-model

# Deactivate environment
conda deactivate

# List all environments
conda env list

# Update environment
conda env update -f environment-clean.yml --prune

# Remove environment
conda env remove -n cmu-tare-model

# Export environment
conda env export --no-builds > my-environment.yml
```

---

### Package Management

```bash
# Install project package
pip install -e .

# Install additional package
conda install package-name

# Update all packages
conda update --all

# List installed packages
conda list

# Search for package
conda search package-name
```

---

### Jupyter Kernel Management

```bash
# Register kernel
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"

# List kernels
jupyter kernelspec list

# Remove kernel
jupyter kernelspec uninstall cmu-tare-model

# Reinstall jupyter
conda install jupyter jupyterlab notebook --force-reinstall
```

---

### VS Code Launch

```bash
# Launch VS Code from command line
code .

# Launch with specific folder
code /path/to/cmu-tare-model

# Launch specific file
code cmu_tare_model/tare_model_main_v2_1.ipynb
```

---

### File Navigation

**Windows (Anaconda Prompt):**
```bash
# Change directory
cd C:\Users\YourName\Research\cmu-tare-model

# List files
dir

# Show current directory
cd

# Create directory
mkdir new_folder
```

**Mac/Linux (Terminal):**
```bash
# Change directory
cd ~/Research/cmu-tare-model

# List files
ls -la

# Show current directory
pwd

# Create directory
mkdir new_folder
```

---

### Git Commands (If Using)

```bash
# Clone repository
git clone [GITHUB-URL]

# Check status
git status

# Pull latest changes
git pull

# Create new branch
git checkout -b my-branch-name

# Commit changes
git add .
git commit -m "Description of changes"

# Push changes
git push origin my-branch-name
```

---

## Appendix E: Additional Resources

### Documentation

- **TARE Model GitHub**: `[GITHUB-URL]` *(To be provided)*
- **Data Repository (Zenodo)**: `[ZENODO-URL]` *(To be provided)*
- **Project Website**: *(To be provided)*

### Software Documentation

- **Anaconda Documentation**: https://docs.anaconda.com/
- **VS Code Documentation**: https://code.visualstudio.com/docs
- **Jupyter Documentation**: https://jupyter.org/documentation
- **Python Documentation**: https://docs.python.org/3.11/

### Related Tools

- **pandas Documentation**: https://pandas.pydata.org/docs/
- **matplotlib Documentation**: https://matplotlib.org/stable/contents.html
- **seaborn Documentation**: https://seaborn.pydata.org/
- **numpy Documentation**: https://numpy.org/doc/

### Policy Resources

- **Inflation Reduction Act Summary**: https://www.whitehouse.gov/cleanenergy/inflation-reduction-act-guidebook/
- **HOMES Rebate Program**: https://www.energy.gov/scep/home-energy-rebate-programs
- **Social Cost of Carbon (EPA)**: https://www.epa.gov/environmental-economics/social-cost-carbon

### Data Sources

- **EUSS (DOE)**: https://www.energy.gov/eere/buildings/energy-use-analysis
- **AEO (EIA)**: https://www.eia.gov/outlooks/aeo/
- **Cambium (NREL)**: https://www.nrel.gov/analysis/cambium.html

---

## Appendix F: Contact and Support

### Getting Help

**For technical issues with the model:**
1. Check this User Guide
2. Search existing GitHub Issues
3. Post a new GitHub Issue with:
   - Description of problem
   - Error messages (full text)
   - Steps to reproduce
   - Your operating system
   - Python and package versions

**For research/methodology questions:**
- Refer to published papers *(to be added)*
- Contact research team *(contact info to be added)*

### Contributing

We welcome contributions! See `CONTRIBUTING.md` in the repository *(to be created)* for guidelines on:
- Reporting bugs
- Suggesting enhancements
- Contributing code
- Improving documentation

### Citation

If you use the TARE Model in your research, please cite:

```
[Citation to be added after publication]
```

---

## Document Information

**Version**: 2.1  
**Last Updated**: 2025-10-23  
**Authors**: Jordan Joseph, CMU TARE Model Development Team  
**License**: *(To be specified)*  
**Changelog**:
- v2.1 (2025-10-23): Initial comprehensive user guide

---

**End of TARE Model User Guide**

For the latest version of this guide, visit: `[GITHUB-URL]/TARE_MODEL_USER_GUIDE.md`

