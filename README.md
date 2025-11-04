# cmu-tare-model
The Tradeoff Analysis of residential Retrofits for energy Equity (TARE) Model v2


<img width="8000" height="4500" alt="Figure_1" src="https://github.com/user-attachments/assets/f554ea06-0b03-4aab-823c-346b8f684c00" />

# Section 1: Tool Access and Repository

## 1.1 Accessing the TARE Model Repository

**Repository Location:** https://github.com/jordan-joseph126/cmu-tare-model

**Repository Status:** Currently private (public release planned after documentation completion)

**Recommended Access Method:** 
- **For Current Development:** Contact the repository owner for collaborator access
- **For Future Public Access:** Clone or download directly from GitHub once made public

**Getting Repository Access:**
```bash
# Once you have access, clone the repository:
git clone https://github.com/jordan-joseph126/cmu-tare-model.git
cd cmu-tare-model
```

## 1.2 Repository Structure Overview

The TARE model is organized into a clear, modular structure designed for research reproducibility and ease of use:

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

### **Main Package Directory: `cmu_tare_model/`**

**Core Analysis Modules:**
- `model_scenarios/` - **Primary execution scripts and Jupyter notebooks**
  - `tare_baseline_v2_1.ipynb` - Baseline scenario analysis (active)
  - `tare_basic_v2_1.ipynb` - Basic retrofit scenario analysis (active)
  - `tare_run_simulation_v2_1.ipynb` - Main script to run all scenarios
  - Additional scenario scripts for moderate and advanced retrofits

**Functional Modules:**
- `private_impact/` - Private cost-benefit calculations and equipment costs
- `public_impact/` - Climate and health impact calculations
- `adoption_potential/` - Technology adoption potential analysis
- `energy_consumption_and_metadata/` - Energy usage data processing
- `utils/` - Utility functions and shared calculations

**Data Directory: `data/`**
- `euss_data/` - End-use savings shapes database from NREL
- `fuel_prices/` - Regional fuel price data and projections (EIA)
- `retrofit_costs/` - Equipment and installation cost data (NREL REMDB, pre-v4.0.0)
- `projections/` - Future energy and emissions projections (Cambium, NEI)
- `marginal_social_costs/` - Marginal social costs for carbon (SCC) and health-related pollutants (PM2.5, SOx, NOx)

**Configuration and Setup:**
- `config.py` - Project configuration and path management
- `constants.py` - Model constants
- `requirements.txt` - Python package dependencies
- `setup.py` - Package installation configuration

### **Key Entry Points:**
- **Main Analysis:** Start with the main model run notebook `tare_model_main_v2_1.ipynb`
- **To work with individual scenario files:** Navigate to the `model_scenarios/` folder
- **Individual Functions:** Specific modules can be located within their respective directories (e.g., the `public_impact/` folder contains the climate, health, and public impact/NPV scripts and other modules in the `calculations/` and `data_processing/` subfolders)

## 1.3 Version Information and Updates

**Current Version:** 2.0 (as specified in setup.py)

**Development Status:** Production/Stable (5 - Production/Stable in setup.py)

**Update Frequency:** Research-driven updates as model improvements are developed

**Accessing Updates:**
```bash
# Check for updates (once public):
git fetch origin
git pull origin main

# View version information:
python setup.py --version
```

**Version History:** Currently managed through git commits; formal releases planned for public version

## 1.4 Licensing and Attribution

**License Status:** License file not yet established (planned before public release)

**Planned License Type:** MIT License (as indicated in setup.py classifiers)

**Author Information:**
- **Primary Author:** Jordan Joseph
- **Affiliation:** Carnegie Mellon University
- **Contact:** jordanjo@andrew.cmu.edu

**Intended Usage:** Open source scientific research tool for energy efficiency and equity analysis

**Citation Requirements (Planned):**
```
Joseph, J. (2025). TARE Model: Tradeoff Analysis of Residential retrofits for Energy equity. 
Carnegie Mellon University. https://github.com/jordan-joseph126/cmu-tare-model
```

**Usage Permissions (Under Development):**
- Research and academic use
- Modification and extension for research purposes  
- Integration into other research projects
- Commercial use permissions to be specified in final license

**Attribution Requirements:**
- Cite the TARE model in publications using the tool
- Reference the GitHub repository in code documentation
- Acknowledge Carnegie Mellon University as the institutional affiliation

## 1.5 Getting Started

**Prerequisites for Access:**
- Git installed on your system
- Python 3.11 or compatible version
- Conda or similar environment management tool

**First Steps After Repository Access:**
1. **Clone the repository** using the git command above
2. **Review the repository structure** to understand the organization
3. **Proceed to Section 2** for detailed installation and setup instructions
4. **Check Section 3** for data requirements and acquisition

**Support and Questions:**
- **Primary Contact:** jordanjo@andrew.cmu.edu, jordanjoseph53@gmail.com
- **Repository Issues:** GitHub Issues (once public)
- **Documentation:** This user guide and inline code documentation (google style docstrings, type hints, comments)

---

**Next Step:** Continue to Section 2 for complete installation and environment setup instructions.
