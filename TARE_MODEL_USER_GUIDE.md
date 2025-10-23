# TARE Model User Guide

**Version 2.1**
**Last Updated**: 2025-10-23

---

## About the TARE Model

The **TARE (Tradeoff Analysis of residential Retrofits for energy Equity) Model** evaluates the economic viability and environmental impacts of residential building electrification and energy efficiency retrofits across the United States.

**Key Features:**
- Analyzes private costs, climate benefits, and health benefits of retrofits
- Evaluates equity impacts across income groups
- Compares policy scenarios (Pre-IRA vs. IRA with rebates)
- Provides household-level results aggregated by geography

**Time Commitment:**
- Initial setup: 1-2 hours
- Single-state run: 30-60 minutes
- Full U.S. run: 4-8 hours

**Prerequisites:**
- Basic Python/Jupyter notebook experience
- 16 GB RAM recommended
- 10 GB free disk space

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Data Setup](#data-setup)
5. [Running the Model](#running-the-model)
6. [Understanding Outputs](#understanding-outputs)
7. [Model Scenarios Explained](#model-scenarios-explained)
8. [Troubleshooting](#troubleshooting)
9. [Glossary](#glossary)

---

## Quick Start

### 1. Install Required Software

**You will need:**
- **Anaconda Navigator** (includes Python): https://www.anaconda.com/download
- **VS Code** (code editor): https://code.visualstudio.com/

See their respective documentation for installation instructions for your operating system.

### 2. Download the Repository

**Option 1 - Download ZIP:**
- Go to: `[GITHUB-URL]`
- Click "Code" → "Download ZIP"
- Extract to your preferred location

**Option 2 - Git Clone:**
```bash
git clone [GITHUB-URL]
cd cmu-tare-model
```

### 3. Set Up Environment

Open **Anaconda Prompt** (Windows) or **Terminal** (Mac/Linux):

```bash
# Navigate to project directory
cd /path/to/cmu-tare-model

# Create conda environment
conda env create -f environment-clean.yml

# Activate environment
conda activate cmu-tare-model

# Install TARE package
pip install -e .

# Register Jupyter kernel
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

### 4. Download Data Files

Download data from Zenodo: `[ZENODO-URL]`

Extract to: `cmu_tare_model/data/`

### 5. Launch and Run

```bash
# From Anaconda Prompt with environment activated
conda activate cmu-tare-model
cd /path/to/cmu-tare-model
code .
```

In VS Code:
1. Open `cmu_tare_model/tare_model_main_v2_1.ipynb`
2. Select kernel: "Python 3.11.13 (cmu-tare-model)"
3. Run cells sequentially from top to bottom

---

## Repository Structure

```
cmu-tare-model/
├── cmu_tare_model/                    # Main model package
│   ├── adoption_potential/            # Adoption decision calculations
│   ├── data/                          # Input data (download separately)
│   │   ├── equity_data/               # Income, tenure, demographics
│   │   ├── fuel_prices/               # Energy price projections
│   │   ├── marginal_social_costs/     # SCC and health damage values
│   │   ├── projections/               # Climate and energy projections
│   │   └── retrofit_costs/            # Equipment and installation costs
│   ├── energy_consumption_and_metadata/ # EUSS data processing
│   ├── model_scenarios/               # Scenario notebooks
│   │   ├── tare_baseline_v2_1.ipynb
│   │   ├── tare_basic_v2_1.ipynb
│   │   ├── tare_moderate_v2_1.ipynb
│   │   ├── tare_advanced_v2_1.ipynb
│   │   └── tare_run_simulation_v2_1.ipynb
│   ├── output_results/                # Model outputs (created on run)
│   ├── private_impact/                # Private cost/benefit calculations
│   ├── public_impact/                 # Climate and health impact calculations
│   ├── utils/                         # Utility and visualization functions
│   └── tare_model_main_v2_1.ipynb    # **MAIN NOTEBOOK - START HERE**
├── environment.yml                    # Conda environment (Windows-specific)
├── environment-clean.yml              # Conda environment (cross-platform)
├── setup.py                           # Package installation script
├── config.py                          # Project configuration
├── ENVIRONMENT_SETUP.md               # Detailed environment setup reference
└── TARE_MODEL_USER_GUIDE.md          # This guide
```

### Key Files

| File | Purpose |
|------|---------|
| `tare_model_main_v2_1.ipynb` | **Main entry point** - Loads and visualizes results |
| `tare_run_simulation_v2_1.ipynb` | Runs the full simulation (called by main notebook) |
| `environment-clean.yml` | Environment specification (cross-platform) |
| `config.py` | Sets PROJECT_ROOT path automatically |

---

## Environment Setup

### Creating the Environment

The TARE model requires Python 3.11.13 with ~200 packages. The `environment-clean.yml` file specifies all dependencies.

```bash
# In Anaconda Prompt or Terminal
cd /path/to/cmu-tare-model
conda env create -f environment-clean.yml
```

This takes 10-20 minutes depending on internet speed.

### Activating the Environment

**Always activate the environment before working:**

```bash
conda activate cmu-tare-model
```

You'll see `(cmu-tare-model)` at the beginning of your command prompt.

### Installing the TARE Package

The model must be installed as a Python package so modules can import each other:

```bash
# With environment activated
pip install -e .
```

The `-e` flag installs in "editable" mode - code changes are immediately reflected.

**Verify installation:**
```bash
python -c "from config import PROJECT_ROOT; print(PROJECT_ROOT)"
python -c "import cmu_tare_model; print('Success!')"
```

### Jupyter Kernel Setup

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"
```

### Launching VS Code

**Best practice:** Launch VS Code from Anaconda Prompt while environment is activated:

```bash
conda activate cmu-tare-model
cd /path/to/cmu-tare-model
code .
```

This ensures VS Code inherits the correct environment variables.

**In VS Code:**
- Open any `.ipynb` file
- Click kernel selector (top right)
- Select "Python 3.11.13 (cmu-tare-model)"

---

## Data Setup

### Required Data Files

The TARE model requires external data files (~5-8 GB compressed) hosted on Zenodo:

**Zenodo Link**: `[ZENODO-URL]`

**Data files include:**
- Energy use data from EUSS
- Fuel price projections (AEO 2023)
- Grid emissions data (Cambium 2021, 2022, 2023)
- Health damage values (AP2, EASIUR, InMAP)
- Climate damage values (Social Cost of Carbon)
- Retrofit cost data
- Demographic and economic data

### Downloading and Extracting Data

1. **Download** the complete data package from Zenodo
2. **Extract** to `cmu_tare_model/data/`

**Verify data structure:**

```
cmu_tare_model/data/
├── equity_data/
├── fuel_prices/
├── inflation_data/
├── marginal_social_costs/
│   └── health_damages/
│       ├── ap2_damages.csv
│       ├── easiur_damages.csv
│       └── inmap_damages.csv
├── projections/
│   ├── hdd_projections/
│   └── energy_projections/
└── retrofit_costs/
```

**Quick check:**
```bash
# In project directory
ls cmu_tare_model/data/
# Should show: equity_data, fuel_prices, inflation_data, marginal_social_costs, projections, retrofit_costs
```

---

## Running the Model

### Main Notebook: tare_model_main_v2_1.ipynb

This notebook is the primary interface for running the TARE model.

**Location:** `cmu_tare_model/tare_model_main_v2_1.ipynb`

### Execution Flow

1. **Import and setup** (Cells 1-5)
   - Loads libraries and sets up project paths
   - Configures visualization settings

2. **Run selection** (Cell 6)
   ```
   Would you like to begin a new simulation or visualize output results from a previous model run?
   Y. I'd like to start a new model run.
   N. I'd like to visualize output results from a previous model run.
   ```

   - **Type `Y`** for new simulation
   - **Type `N`** to visualize existing results

3. **Geographic scope selection** (if Y selected)
   ```
   Enter state abbreviations (comma-separated) or 'ALL' for entire US:
   Example: PA,NY,CA or ALL
   ```

   **Examples:**
   - `PA` - Pennsylvania only (~30-60 min)
   - `PA,NY,CA` - Multiple states (~1-3 hours)
   - `ALL` - Full United States (~4-8 hours)

4. **Model execution**
   - Calls `tare_run_simulation_v2_1.ipynb` which runs all scenarios
   - Progress updates shown in output
   - Results exported to timestamped folder

5. **Results loading and visualization**
   - Loads results from output folder
   - Creates adoption potential dataframes
   - Generates visualizations (bar charts, box plots, histograms)

### Expected Runtime

| Geographic Scope | Runtime (approx.) |
|------------------|-------------------|
| Small state (DE, RI) | 20-30 min |
| Medium state (PA, WA) | 30-60 min |
| Large state (CA, TX, FL) | 1-2 hours |
| 5-10 states | 2-4 hours |
| Full U.S. (ALL) | 4-8 hours |

**Factors affecting runtime:**
- CPU cores (model uses parallelization)
- RAM (16 GB+ recommended)
- Storage speed (SSD recommended)

### Running Cells

**Important:** Cells must be run in order from top to bottom.

- **Run cell**: Click ▶ button or press `Shift+Enter`
- **Running indicator**: `[*]` appears while cell is executing
- **Completed**: Shows number like `[1]`, `[2]`, etc.

**If errors occur:**
1. Read the error message carefully
2. Check that all previous cells ran successfully
3. Verify environment and data setup
4. See [Troubleshooting](#troubleshooting) section

---

## Understanding Outputs

### Output Folder Structure

Results are saved in timestamped folders:

```
cmu_tare_model/output_results/YYYY-MM-DD_HH-MM/
├── baseline_summary/
│   └── summary_baseline/
├── retrofit_basic_summary/
│   ├── summary_basic_ap2/
│   ├── summary_basic_easiur/
│   └── summary_basic_inmap/
├── retrofit_moderate_summary/
│   ├── summary_moderate_ap2/
│   ├── summary_moderate_easiur/
│   └── summary_moderate_inmap/
└── retrofit_advanced_summary/
    ├── summary_advanced_ap2/
    ├── summary_advanced_easiur/
    └── summary_advanced_inmap/
```

### File Naming Convention

```
summary_[Equipment]_[scenario]_results_[timestamp].csv
```

**Examples:**
- `summary_Whole-Home_basic_results_2025-10-23_14-30.csv`
- `summary_Heating_moderate_results_2025-10-23_14-30.csv`

### Equipment Types

| File Prefix | Equipment |
|-------------|-----------|
| `Whole-Home` | All equipment combined |
| `Heating` | Space heating (ASHP) only |
| `WaterHeating` | Water heater (HPWH) only |
| `ClothesDrying` | Clothes dryer (HPCD) only |
| `Cooking` | Electric range only |

### Key Output Columns

#### Identification
- `building_id`: Unique dwelling identifier
- `state`, `county_name`: Geographic location
- `urbanicity`: Urban/Rural classification

#### Baseline Characteristics
- `base_heating_fuel`: Current fuel (Natural Gas, Electricity, Fuel Oil, Propane)
- `tenure`: Owner-occupied / Renter-occupied
- `income_level`: Income bracket (0-30k, 30-60k, 60-100k, 100k+)
- `lmi_or_mui`: LMI (≤80% AMI) or MUI (>80% AMI)

#### Economic Results
- `preIRA_mp8_heating_private_npv_moreWTP`: Private NPV without IRA
- `iraRef_mp8_heating_private_npv_moreWTP`: Private NPV with IRA rebates
- `preIRA_mp8_heating_upfront_cost`: Initial equipment + installation cost
- `preIRA_mp8_heating_lifetime_fuel_savings`: Fuel savings over equipment life

#### Environmental Results
- `preIRA_mp8_heating_climate_npv_central`: Climate benefits (central SCC)
- `preIRA_mp8_heating_health_npv_ap2_acs`: Health benefits (AP2 model, ACS CR function)
- `preIRA_mp8_heating_total_npv_central_ap2_acs`: Total NPV (private + climate + health)

#### Adoption Decision
- `preIRA_mp8_heating_adoption_central_ap2_acs`: Adoption decision (1 = adopt, 0 = don't adopt)
- `preIRA_mp8_heating_adoption_tier`: Tier 1, Tier 2, Tier 3, or None

### Example Interpretation

**Sample Row:**
```
building_id: 12345
state: PA
base_heating_fuel: Natural Gas
lmi_or_mui: LMI
preIRA_mp8_heating_private_npv_moreWTP: -$2,500
iraRef_mp8_heating_private_npv_moreWTP: +$1,200
preIRA_mp8_heating_total_npv_central_ap2_acs: +$8,400
preIRA_mp8_heating_adoption_central_ap2_acs: 1
preIRA_mp8_heating_adoption_tier: Tier 3
```

**Interpretation:**
- Natural gas-heated home in PA
- Low-to-moderate income household
- **Without IRA:** NPV = -$2,500 (not cost-effective privately)
- **With IRA:** NPV = +$1,200 (rebates make it profitable)
- **Total societal NPV:** +$8,400 (including climate & health)
- **Decision:** Would adopt (adoption = 1)
- **Tier 3:** Requires health benefits for positive NPV

### Aggregate Analysis

**Example - Calculate adoption rates in Python:**

```python
import pandas as pd

# Load results
df = pd.read_csv('summary_Whole-Home_basic_results_2025-10-23_14-30.csv')

# Overall adoption rate
adoption_rate = df['preIRA_mp8_heating_adoption_central_ap2_acs'].mean() * 100
print(f"Adoption rate: {adoption_rate:.1f}%")

# By fuel type
by_fuel = df.groupby('base_heating_fuel')['preIRA_mp8_heating_adoption_central_ap2_acs'].mean() * 100
print(by_fuel)

# By income group
by_income = df.groupby('lmi_or_mui')['iraRef_mp8_heating_adoption_central_ap2_acs'].mean() * 100
print(by_income)

# Compare pre-IRA vs IRA
print(f"Pre-IRA:  {df['preIRA_mp8_heating_adoption_central_ap2_acs'].mean()*100:.1f}%")
print(f"IRA-Ref:  {df['iraRef_mp8_heating_adoption_central_ap2_acs'].mean()*100:.1f}%")
```

---

## Model Scenarios Explained

### Retrofit Scenarios

#### Baseline (MP0)
Current housing stock with existing equipment. No retrofits applied.

#### Basic Retrofit (MP8)
**Equipment only** - No envelope improvements
- **Space Heating:** Air-Source Heat Pump (ASHP)
- **Water Heating:** Heat Pump Water Heater (HPWH)
- **Clothes Drying:** Heat Pump Clothes Dryer (HPCD)
- **Cooking:** Electric Resistance Range

**Cost range:** $15,000 - $25,000

#### Moderate Retrofit (MP9)
**ASHP + Basic envelope**
- Air-Source Heat Pump
- Air sealing (15% infiltration reduction)
- Attic insulation improvement
- Basic window upgrades (some cases)

**Cost range:** $20,000 - $35,000

#### Advanced Retrofit (MP10)
**ASHP + Enhanced envelope**
- Air-Source Heat Pump
- Comprehensive air sealing (30% reduction)
- Wall insulation added/improved
- High R-value attic/roof insulation
- High-performance window replacement
- Foundation/basement insulation

**Cost range:** $35,000 - $60,000+

### Policy Scenarios

#### Pre-IRA / No IRA
- No IRA rebates or tax credits
- AEO2023 "No IRA" fuel price projections
- Cambium 2021 MidCase grid emissions

#### IRA-Reference
- **HOMES rebates** (up to $8,000 for heat pumps)
- **HEEHRA rebates** (up to $1,750 for HPWHs)
- Income-based: LMI gets 80-100%, MUI gets 30-50%
- AEO2023 Reference Case fuel prices
- Cambium 2022/2023 MidCase (cleaner grid)

### Sensitivity Parameters

#### Social Cost of Carbon (SCC)
Monetary estimate of climate damages per ton CO₂:
- **Lower:** ~$50/ton CO₂
- **Central:** ~$190/ton CO₂ (EPA 2023)
- **Upper:** ~$340/ton CO₂

#### Health Impact Models (RCM)
Three reduced complexity air quality models:
- **AP2:** EPA model, moderate estimates
- **EASIUR:** Regression-based, often higher damages
- **InMAP:** High-resolution, detailed urban/rural

#### Concentration-Response Functions
- **ACS:** American Cancer Society study (conservative)
- **H6C:** Harvard Six Cities study (higher estimates)

### Adoption Tiers

#### Tier 1: Privately Cost-Effective
Positive Private NPV - saves homeowner money over lifetime.
- No subsidies needed
- Market-driven adoption likely

#### Tier 2: Climate Benefit Required
Negative Private NPV, but positive with climate benefits.
- Needs carbon pricing or climate-focused subsidies
- Societally beneficial

#### Tier 3: Health Benefit Required
Negative Private+Climate NPV, but positive with health benefits.
- Requires comprehensive subsidies
- Justified by total social benefits (climate + health)

#### None: Not Cost-Effective
Negative NPV even with all benefits included.
- Not recommended under current assumptions
- May become viable with technology improvements

---

## Troubleshooting

### Environment Issues

**Problem:** `ModuleNotFoundError: No module named 'config'`

**Solution:**
```bash
conda activate cmu-tare-model
cd /path/to/cmu-tare-model
pip install -e .
# Restart Jupyter kernel in VS Code
```

---

**Problem:** `ModuleNotFoundError: No module named 'pandas'` (or other package)

**Solution:**
```bash
# Check environment is activated
conda env list
# Should show * next to cmu-tare-model

# If wrong environment
conda activate cmu-tare-model

# If package missing
conda install pandas
```

---

**Problem:** Environment won't create / "Solving environment: failed"

**Solutions:**
```bash
# Try 1: Clear cache and retry
conda clean --all
conda env create -f environment-clean.yml

# Try 2: Use mamba (faster solver)
conda install -n base mamba -c conda-forge
mamba env create -f environment-clean.yml

# Try 3: Use Windows-specific file
conda env create -f environment.yml
```

---

### Data Issues

**Problem:** `FileNotFoundError` when loading data

**Check data location:**
```python
import os
from config import PROJECT_ROOT
data_path = os.path.join(PROJECT_ROOT, 'cmu_tare_model', 'data')
print(f"Looking for data at: {data_path}")
print(f"Contents: {os.listdir(data_path)}")
```

**Solution:** Verify data subdirectories are directly in `cmu_tare_model/data/`, not nested deeper.

---

### Jupyter Kernel Issues

**Problem:** Kernel not found or "Dead kernel"

**Solutions:**
```bash
# Reinstall ipykernel
conda activate cmu-tare-model
pip install ipykernel --force-reinstall

# Re-register kernel
python -m ipykernel install --user --name=cmu-tare-model --display-name "Python 3.11.13 (cmu-tare-model)"

# In VS Code: Reload window
# Ctrl+Shift+P → "Developer: Reload Window"
```

---

### Performance Issues

**Problem:** Model runs very slowly

**Optimizations:**
1. **Reduce geographic scope:** Run single state instead of ALL
2. **Close other applications:** Free up RAM
3. **Check CPU usage:** Should be 50-100% during model run
4. **Use SSD:** Faster I/O for data loading

---

**Problem:** Out of memory errors

**Solutions:**
1. Reduce scope (analyze fewer states)
2. Close other applications
3. Increase virtual memory (OS settings)
4. Process states in batches

---

### Common Errors

**Problem:** `UnicodeDecodeError` during `pip install -e .`

**Solution:** Ensure `setup.py` line 22 has:
```python
long_description=open("README.md", encoding="utf-8").read(),
```

---

**Problem:** Results show all NaN or zeros

**Causes:**
- Cells run out of order
- Wrong kernel selected
- Incomplete data files

**Solution:**
1. Restart kernel
2. Verify kernel is "Python 3.11.13 (cmu-tare-model)"
3. Run all cells in order from top
4. Check data files are complete

---

## Glossary

### Model Terms

**ASHP (Air-Source Heat Pump)** - Electric heating/cooling system. Provides both heating and cooling. More efficient than resistance heating.

**Adoption Potential** - Percentage of homes where retrofit is cost-effective based on total NPV.

**CR Function (Concentration-Response Function)** - Relationship between air pollution and health impacts. Used to estimate mortality/morbidity.

**EUSS** - Energy Use and Savings Simulation dataset (underlying housing data).

**HOMES/HEEHRA** - IRA rebate programs for building electrification.

**HPWH (Heat Pump Water Heater)** - Electric water heater using heat pump technology. 2-3× more efficient than resistance.

**HPCD (Heat Pump Clothes Dryer)** - Electric dryer using heat pump technology instead of resistance heating.

**LMI (Low-to-Moderate Income)** - Households ≤80% Area Median Income. Eligible for higher IRA rebates.

**Measure Package (MP)** - Predefined retrofit set:
- MP0: Baseline
- MP8: Basic (equipment only)
- MP9: Moderate (equipment + basic envelope)
- MP10: Advanced (equipment + enhanced envelope)

**MUI (Middle-to-Upper Income)** - Households >80% Area Median Income.

**NPV (Net Present Value)** - Sum of costs and benefits over equipment lifetime, discounted to present value.

**RCM (Reduced Complexity Model)** - Simplified air quality model (AP2, EASIUR, InMAP).

**SCC (Social Cost of Carbon)** - Monetary estimate of damages from one ton of CO₂ emissions.

**WTP (Willingness to Pay)** - Non-economic factors in adoption:
- More WTP: Longer time horizon, lower discount rate (3%)
- Less WTP: Shorter time horizon, higher discount rate (7%)

---

### Economic Terms

**Discount Rate** - Rate to convert future costs/benefits to present value. Model uses 3% (more WTP) or 7% (less WTP).

**Levelized Cost** - Average cost per unit of energy over equipment lifetime.

**Upfront Cost** - Initial equipment + installation cost before rebates.

---

### Energy Terms

**AEO (Annual Energy Outlook)** - EIA's annual energy projections.

**Cambium** - NREL's hourly grid emissions dataset.

**COP (Coefficient of Performance)** - Heat pump efficiency. COP of 3.0 = 3 units heat per 1 unit electricity.

**HDD (Heating Degree Days)** - Measure of heating demand based on temperature.

---

### Environmental Terms

**CO₂ (Carbon Dioxide)** - Primary greenhouse gas from fossil fuels.

**NOₓ (Nitrogen Oxides)** - Air pollutant causing respiratory issues.

**PM₂.₅ (Particulate Matter)** - Fine particles ≤2.5 microns. Major health hazard.

**VSL (Value of Statistical Life)** - Monetary value for preventing premature death. ~$10-12 million (2023 USD).

---

## Additional Resources

### Documentation
- **GitHub Repository:** `[GITHUB-URL]`
- **Data Repository:** `[ZENODO-URL]`
- **Detailed Environment Setup:** See `ENVIRONMENT_SETUP.md`

### Software Documentation
- **Anaconda:** https://docs.anaconda.com/
- **VS Code:** https://code.visualstudio.com/docs
- **Jupyter:** https://jupyter.org/documentation
- **pandas:** https://pandas.pydata.org/docs/

### Policy Resources
- **IRA Summary:** https://www.whitehouse.gov/cleanenergy/inflation-reduction-act-guidebook/
- **HOMES Rebates:** https://www.energy.gov/scep/home-energy-rebate-programs
- **Social Cost of Carbon:** https://www.epa.gov/environmental-economics/social-cost-carbon

---

## Citation

If you use the TARE Model in your research, please cite:

```
[Citation to be added after publication]
```

---

## Support

**For technical issues:**
- Check this guide and `ENVIRONMENT_SETUP.md`
- Search existing GitHub Issues: `[GITHUB-URL]/issues`
- Post new issue with error details and system info

**For research questions:**
- Contact: [Contact information to be added]

---

**Document Version:** 2.1
**Last Updated:** 2025-10-23
**Authors:** Jordan Joseph, CMU TARE Model Development Team

For the latest version: `[GITHUB-URL]/TARE_MODEL_USER_GUIDE.md`
