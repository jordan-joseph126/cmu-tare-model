# CMU TARE Model Structure (Abbreviated Approximation)

cmu-tare-model/
├── .gitignore
├── CHANGELOG.md
├── config.py                        # Project configuration including PROJECT_ROOT
├── README.md                        # Basic project documentation
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation configuration
├── __pycache__/                     # Python bytecode cache
│   ├── config.cpython-311.pyc
│   └── tare_model_functions_v1_5_1.cpython-311.pyc
├── .ipynb_checkpoints/              # Jupyter notebook checkpoints
├── .pytest_cache/                   # pytest cache for testing
├── archived_files/                  # Storage for older file versions
├── cmu_tare_model/                  # **Main package directory**
│   ├── __pycache__/
│   ├── constants.py                 # **Model constants like SCC values**
│   ├── model_scenarios/             # **Jupyter notebooks and Python scripts for different**
│   │   ├── tare_baseline_v2_1.ipynb # **Baseline scenario (active file)**
│   │   ├── tare_baseline_v2_1.py
│   │   ├── tare_baseline_v2.py
│   │   ├── tare_basic_v2_1.ipynb    # **Basic retrofit scenario (active file)**
│   │   ├── tare_basic_v2_1.py
│   │   ├── tare_basic_v2.py
│   │   ├── tare_advanced_v2.py      # Advanced retrofit scenario
│   │   ├── tare_moderate_v2.py      # Moderate retrofit scenario
│   │   └── tare_run_simulation_v2.py # Main script to run all scenarios
│   ├── data/                        # **Input data files**
│   │   └── ami_calculations_data/
│   │   ├── euss_data/               # Energy usage shapes data
│   │   │   └── resstock_amy2018_release_1.1/
│   │   │       └── state/
│   │   │           ├── baseline_metadata_and_annual_results.csv
│   │   │           ├── upgrade07_metadata_and_annual_results.csv
│   │   │           └── upgrade08_metadata_and_annual_results.csv
│   │   │           └── upgrade09_metadata_and_annual_results.csv
│   │   │           └── upgrade10_metadata_and_annual_results.csv
│   │   └── fuel_prices/
│   │   └── inflation_data/
│   │   └── marginal_social_costs/
│   │   └── projections/
│   │   │   └── schmitt_ev_study/
│   │   │   └── aeo_projections_2022_2050.xlsx
│   │   │   └── cambium21_midCase_annual_gea.xlsx
│   │   │   └── cambium22_allScenarios_annual_gea.xlsx
│   │   │   └── ef_pollutants_egrid.csv
│   │   │   └── fuel_price_projection_factors.xlsx
│   │   └── retrofit_costs/
│   │       └── tare_retrofit_costs_cpi.xlsx
│   ├── energy_consumption_and_metadata/
│   │   └── mp7_electric_resistance_range.py
│   │   └── process_euss_data.py
│   │   └── project_future_energy_consumption.py
│   │   └── user_input_geographic_filter.py
│   ├── public_impact/              # **Modules for calculating public impacts**
│   │   ├── calculate_lifetime_climate_impacts_sensitivity.py
│   │   ├── calculate_lifetime_health_impacts_sensitivity.py
│   │   ├── calculate_lifetime_public_impact_sensitivity.py
│   │   └── calculations/
│   │   └── data_processing/
│   ├── private_impact/             # **Modules for calculating private impacts**
│   │   ├── calculate_lifetime_fuel_costs.py
│   │   ├── calculate_lifetime_private_impact.py
│   │   ├── calculations/
│   │   │   ├── calculate_equipment_installation_costs.py
│   │   │   └── calculate_equipment_replacement_costs.py
│   │   │   └── calculate_enclosure_upgrade_costs.py
│   │   └── data_processing/
│   │       └── determine_rebate_eligibility_and_amount.py
│   │       └── process_income_data_for_rebates.py
│   ├── adoption_potential/         # **Modules for adoption potential analysis**
│   │   ├── determine_adoption_potential_sensitivity.py
│   │   └── data_processing/
│   │       └── visuals_adoption_potential_utils.py
│   ├── utils/                      # **Utility functions**
│   │   ├── create_sample_df.py
│   │   └── inflation_adjustment.py
│   │   └── modeling_params.py
│   │   └── calculation_utils.py
│   │   └── validation_framework.py
│   │   └── ...
│   ├── tests/                      # **Test files**
│   │   └── testing_documentation.md
│   │   └── ...
│   └── output_results/             # Directory for output files
└── cmu_tare_model.egg-info/        # Package metadata for installation