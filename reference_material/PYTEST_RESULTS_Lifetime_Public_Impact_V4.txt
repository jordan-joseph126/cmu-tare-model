================================================================== test session starts ==================================================================
platform win32 -- Python 3.11.7, pytest-7.4.0, pluggy-1.0.0
rootdir: C:\Users\14128\Research\cmu-tare-model
plugins: anyio-4.2.0
collected 23 items                                                                                                                                       

test_calculate_lifetime_public_impact_sensitivity_v4.py ......F................                                                                    [100%]

======================================================================= FAILURES ========================================================================
______________________________ test_calculate_public_npv_successful_execution[0-No Inflation Reduction Act-AP2-acs-public] ______________________________ 

df_copy =   county_fips state  year     census_division  ... upgrade_hvac_heating_efficiency upgrade_water_heater_efficiency  up...             None                              HP                   None                   None

[5 rows x 110 columns]
df_baseline_climate =    include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_climate_lrmer_central  baseline_2038_cook...           
                          500.0                                                500.0

[5 rows x 169 columns]
df_baseline_health =    include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_health_InMAP_acs  baseline_2038_cooking_d...            
                              250.0                                           250.0

[5 rows x 334 columns]
df_mp_climate =    include_heating  include_waterHeating  ...  iraRef_mp8_2038_cooking_damages_climate_lrmer_central  iraRef_mp8_2038_...                 
                300.0                                                  300.0

[5 rows x 499 columns]
df_mp_health =    include_heating  include_waterHeating  ...  iraRef_mp8_2038_cooking_damages_health_InMAP_acs  iraRef_mp8_2038_cooki...                  
                      175.0                                             175.0

[5 rows x 994 columns]
menu_mp = 0, policy_scenario = 'No Inflation Reduction Act', rcm_model = 'AP2', cr_function = 'acs', base_year = 2024, discounting_method = 'public'      
all_columns_to_mask = {'clothesDrying': [], 'cooking': [], 'heating': [], 'waterHeating': []}, verbose = False

    def calculate_lifetime_damages_grid_scenario(
        df_copy: pd.DataFrame,
        df_baseline_climate: pd.DataFrame,
        df_baseline_health: pd.DataFrame,
        df_mp_climate: pd.DataFrame,
        df_mp_health: pd.DataFrame,
        menu_mp: int,
        policy_scenario: str,
        rcm_model: str,
        cr_function: str,
        base_year: int = 2024,
        discounting_method: str = 'public',
        all_columns_to_mask: Optional[Dict[str, List[str]]] = None,
        verbose: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Calculate the NPV of climate, health, and public damages over the equipment's lifetime.

        This function performs sensitivity analysis across multiple dimensions:
        - Equipment categories (with varying lifetimes)
        - Social Cost of Carbon (SCC) assumptions for climate damages
        - Health impacts based on specified RCM model and C-R function

        Args:
            df_copy: Copy of the original DataFrame to use for calculations.
            df_baseline_climate: DataFrame containing baseline climate damage projections.
            df_baseline_health: DataFrame containing baseline health damage projections.
            df_mp_climate: DataFrame containing post-retrofit climate damage projections.
            df_mp_health: DataFrame containing post-retrofit health damage projections.
            menu_mp: Menu identifier used to construct column names for the measure package.
            policy_scenario: Specifies the grid scenario.
            rcm_model: The Reduced Complexity Model used for health impact calculations.
            cr_function: The Concentration-Response function used for health impact calculations.
            base_year: The base year for discounting calculations. Default is 2024.
            discounting_method: The method used for discounting. Default is 'public'.
            all_columns_to_mask: Dictionary to track columns for masking verification by category.
            verbose: Whether to print detailed progress messages.

        Returns:
            Dictionary mapping column names to Series of calculated NPV values for
            each category, damage type, and sensitivity combination.
    
        Raises:
            ValueError: If required columns are missing from input DataFrames.
            RuntimeError: If processing fails for a specific category or SCC assumption.
        """
        # Initialize the masking dictionary if None is provided
        if all_columns_to_mask is None:
            all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

        # Determine the scenario prefix based on the policy scenario
        scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)

        # Pre-calculate discount factors for efficiency
        discount_factors: Dict[int, float] = {}
        max_lifetime = max(EQUIPMENT_SPECS.values())
        for year in range(1, max_lifetime + 1):
            year_label = year + (base_year - 1)
            discount_factors[year_label] = calculate_discount_factor(base_year, year_label, discounting_method)

        # Initialize a dictionary to store all NPV results
        all_npvs: Dict[str, pd.Series] = {}

        # Process each equipment category
        for category, lifetime in EQUIPMENT_SPECS.items():
            if verbose:
                print(f"  Calculating Public NPV for {category}...")

            # Process each SCC assumption for climate damages
            for scc in SCC_ASSUMPTIONS:
                try:
                    if verbose:
                        print(f"    SCC: {scc}, RCM: {rcm_model}, C-R: {cr_function}")

                    # Define column names for NPV results
                    climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{scc}'
                    health_npv_key = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
                    public_npv_key = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'

                    # ===== STEP 1: Initialize validation tracking =====
                    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                        df_copy, category, menu_mp, verbose=False)

                    # ===== STEP 2: Initialize result series with template =====
                    # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
                    climate_npv_template = create_retrofit_only_series(df_copy, valid_mask)
                    health_npv_template = create_retrofit_only_series(df_copy, valid_mask)

                    # Create lists to store yearly avoided damages
                    yearly_climate_avoided = []
                    yearly_health_avoided = []

                    # Track if any year's data was successfully processed
                    climate_years_processed = 0
                    health_years_processed = 0

                    # ===== STEP 3: Valid-Only Calculation =====
                    # Calculate NPVs for each year in the equipment's lifetime
                    for year in range(1, lifetime + 1):
                        year_label = year + (base_year - 1)
                        discount_factor = discount_factors[year_label]

                        # Get column names for baseline damages
                        base_climate_col = f'baseline_{year_label}_{category}_damages_climate_lrmer_{scc}'
                        base_health_col = f'baseline_{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'

                        # Get column names for retrofit damages
                        retrofit_climate_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_{scc}'
                        retrofit_health_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'

                        # Check if climate columns exist before calculation
                        climate_cols_exist = (base_climate_col in df_baseline_climate.columns and
                                            retrofit_climate_col in df_mp_climate.columns)

                        # Check if health columns exist before calculation
                        health_cols_exist = (base_health_col in df_baseline_health.columns and
                                            retrofit_health_col in df_mp_health.columns)

                        # ===== STEP 4: Valid-Only Updates =====
                        # NOTES: WE ARE DISCOUNTING THE AVOIDED DAMAGES HERE. USING MARGINAL SOCIAL COSTS SO MAY NOT BE NECESSARY.
                        # Calculate avoided climate damages if columns exist (store in list instead of incremental update)
                        if climate_cols_exist:
                            # Use calculate_avoided_values function for consistency
>                           avoided_climate = calculate_avoided_values(
                                baseline_values=df_baseline_climate[base_climate_col],
                                measure_values=df_mp_climate[retrofit_climate_col],
                                retrofit_mask=(valid_mask if menu_mp != 0 else None)
                            ) * discount_factor

..\public_impact\calculate_lifetime_public_impact_sensitivity.py:289:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

baseline_values = 0    100.0
1    200.0
2    300.0
3    400.0
4    500.0
Name: baseline_2024_heating_damages_climate_lrmer_lower, dtype: float64
measure_values = 0     60.0
1    120.0
2    180.0
3    240.0
4    300.0
Name: baseline_2024_heating_damages_climate_lrmer_lower, dtype: float64
retrofit_mask = None

    def calculate_avoided_values(
        baseline_values: pd.Series,
        measure_values: pd.Series,
        retrofit_mask: pd.Series
    ) -> pd.Series:
        """
        Calculate avoided values (baseline - measure) only for retrofitted homes.

        Args:
            baseline_values: Series of baseline values.
            measure_values: Series of measure package values.
            retrofit_mask: Boolean Series indicating which homes get retrofits.

        Returns:
            Series with avoided values for retrofitted homes and NaN for others.
        """
        # Initialize with NaN
        avoided_values = pd.Series(np.nan, index=baseline_values.index)

        # Calculate only for homes with retrofits
>       if retrofit_mask.any():
E       AttributeError: 'NoneType' object has no attribute 'any'

..\utils\validation_framework.py:475: AttributeError

During handling of the above exception, another exception occurred:

df =   county_fips state  year     census_division  ... upgrade_hvac_heating_efficiency upgrade_water_heater_efficiency  up...             None           
                   HP                   None                   None

[5 rows x 110 columns]
df_baseline_climate =    include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_climate_lrmer_central  baseline_2038_cook...           
                          500.0                                                500.0

[5 rows x 169 columns]
df_baseline_health =    include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_health_InMAP_acs  baseline_2038_cooking_d...            
                              250.0                                           250.0

[5 rows x 334 columns]
df_mp_climate =    include_heating  include_waterHeating  ...  iraRef_mp8_2038_cooking_damages_climate_lrmer_central  iraRef_mp8_2038_...                 
                300.0                                                  300.0

[5 rows x 499 columns]
df_mp_health =    include_heating  include_waterHeating  ...  iraRef_mp8_2038_cooking_damages_health_InMAP_acs  iraRef_mp8_2038_cooki...                  
                      175.0                                             175.0

[5 rows x 994 columns]
menu_mp = 0, policy_scenario = 'No Inflation Reduction Act', rcm_model = 'AP2', base_year = 2024, discounting_method = 'public', verbose = False

    def calculate_public_npv(
        df: pd.DataFrame,
        df_baseline_climate: pd.DataFrame,
        df_baseline_health: pd.DataFrame,
        df_mp_climate: pd.DataFrame,
        df_mp_health: pd.DataFrame,
        menu_mp: int,
        policy_scenario: str,
        rcm_model: str,
        base_year: int = 2024,
        discounting_method: str = 'public',
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Calculate the public Net Present Value (NPV) for specific categories of damages,
        considering different policy scenarios related to grid decarbonization.

        The function compares baseline damages with post-measure package (mp) damages
        to determine the avoided damages (benefits) from implementing retrofits.

        This function follows the five-step validation framework:
        1. Mask Initialization: Identifies valid homes using inclusion flags and retrofit status
        2. Series Initialization: Creates result series with zeros for valid homes, NaN for others
        3. Valid-Only Calculation: Performs calculations only for valid homes
        4. Valid-Only Updates: Uses list-based collection of yearly values instead of incremental updates
        5. Final Masking: Applies consistent masking to all result columns

        The list-based collection approach stores yearly values in lists and sums them using pandas
        vectorized operations after all years have been processed. This approach prevents accumulation
        errors that can occur with incremental updates.

        Args:
            df (pd.DataFrame): Input DataFrame containing base data for calculations.
            df_baseline_climate (pd.DataFrame): DataFrame containing baseline climate damage projections.
            df_baseline_health (pd.DataFrame): DataFrame containing baseline health damage projections.
            df_mp_climate (pd.DataFrame): DataFrame containing post-retrofit climate damage projections.
            df_mp_health (pd.DataFrame): DataFrame containing post-retrofit health damage projections.
            menu_mp (int): Menu identifier used to construct column names for the measure package.
            policy_scenario (str): Policy scenario that determines electricity grid projections.
                Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
            rcm_model (str): The Reduced Complexity Model used for health impact calculations.
            base_year (int, optional): The base year for discounting calculations. Default is 2024.
            discounting_method (str, optional): The method used for discounting. Default is 'public'.
            verbose (bool, optional): Whether to print detailed progress messages. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns containing the calculated public NPVs for each
            equipment category, damage type, and sensitivity analysis combination.

        Raises:
            ValueError: If input parameters are invalid or if required data columns are missing.
            RuntimeError: If processing fails at the category or combination level.
        """
        # ===== STEP 0: Validate input parameters =====
        menu_mp, policy_scenario, discounting_method = validate_common_parameters(
            menu_mp, policy_scenario, discounting_method)

        # Validate RCM model
        if rcm_model not in RCM_MODELS:
            raise ValueError(f"Invalid rcm_model: {rcm_model}. Must be one of {RCM_MODELS}")

        # Validate input data structure
        if verbose:
            print("\nValidating input data structure...")

        is_valid, messages = validate_damage_dataframes(
            df_baseline_climate,
            df_baseline_health,
            df_mp_climate,
            df_mp_health,
            menu_mp,
            policy_scenario,
            base_year,
            EQUIPMENT_SPECS
        )

        # Print any validation messages
        if verbose and messages:
            for message in messages:
                print(message)

        if not is_valid:
            raise ValueError("Input DataFrames are missing required damage columns. See errors above.")

        if verbose:
            print("✓ Input data validation passed.")

        # Create copies to avoid modifying original dataframes
        df_copy = df.copy()
        df_baseline_climate_copy = df_baseline_climate.copy()
        df_baseline_health_copy = df_baseline_health.copy()
        df_mp_climate_copy = df_mp_climate.copy()
        df_mp_health_copy = df_mp_health.copy()

        # Initialize dictionary to track columns for masking verification
        all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}
        df_new_columns = pd.DataFrame(index=df_copy.index)

        if verbose:
            print("\nCalculating NPV for each concentration-response function...")

        for cr_function in CR_FUNCTIONS:
            if verbose:
                print(f"\nProcessing CR Function: {cr_function}")

            try:
                # Calculate the lifetime damages and corresponding NPV
>               new_columns_dict = calculate_lifetime_damages_grid_scenario(
                    df_copy=df_copy,
                    df_baseline_climate=df_baseline_climate_copy,
                    df_baseline_health=df_baseline_health_copy,
                    df_mp_climate=df_mp_climate_copy,
                    df_mp_health=df_mp_health_copy,
                    menu_mp=menu_mp,
                    policy_scenario=policy_scenario,
                    rcm_model=rcm_model,
                    cr_function=cr_function,
                    base_year=base_year,
                    discounting_method=discounting_method,
                    all_columns_to_mask=all_columns_to_mask,
                    verbose=verbose
                )

..\public_impact\calculate_lifetime_public_impact_sensitivity.py:129:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

df_copy =   county_fips state  year     census_division  ... upgrade_hvac_heating_efficiency upgrade_water_heater_efficiency  up...             None                              HP                   None                   None

[5 rows x 110 columns]
df_baseline_climate =    include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_climate_lrmer_central  baseline_2038_cook...           
                          500.0                                                500.0

[5 rows x 169 columns]
df_baseline_health =    include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_health_InMAP_acs  baseline_2038_cooking_d...            
                              250.0                                           250.0

[5 rows x 334 columns]
df_mp_climate =    include_heating  include_waterHeating  ...  iraRef_mp8_2038_cooking_damages_climate_lrmer_central  iraRef_mp8_2038_...                 
                300.0                                                  300.0

[5 rows x 499 columns]
df_mp_health =    include_heating  include_waterHeating  ...  iraRef_mp8_2038_cooking_damages_health_InMAP_acs  iraRef_mp8_2038_cooki...                  
                      175.0                                             175.0

[5 rows x 994 columns]
menu_mp = 0, policy_scenario = 'No Inflation Reduction Act', rcm_model = 'AP2', cr_function = 'acs', base_year = 2024, discounting_method = 'public'      
all_columns_to_mask = {'clothesDrying': [], 'cooking': [], 'heating': [], 'waterHeating': []}, verbose = False

    def calculate_lifetime_damages_grid_scenario(
        df_copy: pd.DataFrame,
        df_baseline_climate: pd.DataFrame,
        df_baseline_health: pd.DataFrame,
        df_mp_climate: pd.DataFrame,
        df_mp_health: pd.DataFrame,
        menu_mp: int,
        policy_scenario: str,
        rcm_model: str,
        cr_function: str,
        base_year: int = 2024,
        discounting_method: str = 'public',
        all_columns_to_mask: Optional[Dict[str, List[str]]] = None,
        verbose: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Calculate the NPV of climate, health, and public damages over the equipment's lifetime.

        This function performs sensitivity analysis across multiple dimensions:
        - Equipment categories (with varying lifetimes)
        - Social Cost of Carbon (SCC) assumptions for climate damages
        - Health impacts based on specified RCM model and C-R function

        Args:
            df_copy: Copy of the original DataFrame to use for calculations.
            df_baseline_climate: DataFrame containing baseline climate damage projections.
            df_baseline_health: DataFrame containing baseline health damage projections.
            df_mp_climate: DataFrame containing post-retrofit climate damage projections.
            df_mp_health: DataFrame containing post-retrofit health damage projections.
            menu_mp: Menu identifier used to construct column names for the measure package.
            policy_scenario: Specifies the grid scenario.
            rcm_model: The Reduced Complexity Model used for health impact calculations.
            cr_function: The Concentration-Response function used for health impact calculations.
            base_year: The base year for discounting calculations. Default is 2024.
            discounting_method: The method used for discounting. Default is 'public'.
            all_columns_to_mask: Dictionary to track columns for masking verification by category.
            verbose: Whether to print detailed progress messages.

        Returns:
            Dictionary mapping column names to Series of calculated NPV values for
            each category, damage type, and sensitivity combination.

        Raises:
            ValueError: If required columns are missing from input DataFrames.
            RuntimeError: If processing fails for a specific category or SCC assumption.
        """
        # Initialize the masking dictionary if None is provided
        if all_columns_to_mask is None:
            all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

        # Determine the scenario prefix based on the policy scenario
        scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)

        # Pre-calculate discount factors for efficiency
        discount_factors: Dict[int, float] = {}
        max_lifetime = max(EQUIPMENT_SPECS.values())
        for year in range(1, max_lifetime + 1):
            year_label = year + (base_year - 1)
            discount_factors[year_label] = calculate_discount_factor(base_year, year_label, discounting_method)

        # Initialize a dictionary to store all NPV results
        all_npvs: Dict[str, pd.Series] = {}

        # Process each equipment category
        for category, lifetime in EQUIPMENT_SPECS.items():
            if verbose:
                print(f"  Calculating Public NPV for {category}...")

            # Process each SCC assumption for climate damages
            for scc in SCC_ASSUMPTIONS:
                try:
                    if verbose:
                        print(f"    SCC: {scc}, RCM: {rcm_model}, C-R: {cr_function}")

                    # Define column names for NPV results
                    climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{scc}'
                    health_npv_key = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
                    public_npv_key = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'

                    # ===== STEP 1: Initialize validation tracking =====
                    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                        df_copy, category, menu_mp, verbose=False)

                    # ===== STEP 2: Initialize result series with template =====
                    # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
                    climate_npv_template = create_retrofit_only_series(df_copy, valid_mask)
                    health_npv_template = create_retrofit_only_series(df_copy, valid_mask)

                    # Create lists to store yearly avoided damages
                    yearly_climate_avoided = []
                    yearly_health_avoided = []

                    # Track if any year's data was successfully processed
                    climate_years_processed = 0
                    health_years_processed = 0

                    # ===== STEP 3: Valid-Only Calculation =====
                    # Calculate NPVs for each year in the equipment's lifetime
                    for year in range(1, lifetime + 1):
                        year_label = year + (base_year - 1)
                        discount_factor = discount_factors[year_label]

                        # Get column names for baseline damages
                        base_climate_col = f'baseline_{year_label}_{category}_damages_climate_lrmer_{scc}'
                        base_health_col = f'baseline_{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'

                        # Get column names for retrofit damages
                        retrofit_climate_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_{scc}'
                        retrofit_health_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'

                        # Check if climate columns exist before calculation
                        climate_cols_exist = (base_climate_col in df_baseline_climate.columns and
                                            retrofit_climate_col in df_mp_climate.columns)

                        # Check if health columns exist before calculation
                        health_cols_exist = (base_health_col in df_baseline_health.columns and
                                            retrofit_health_col in df_mp_health.columns)

                        # ===== STEP 4: Valid-Only Updates =====
                        # NOTES: WE ARE DISCOUNTING THE AVOIDED DAMAGES HERE. USING MARGINAL SOCIAL COSTS SO MAY NOT BE NECESSARY.
                        # Calculate avoided climate damages if columns exist (store in list instead of incremental update)
                        if climate_cols_exist:
                            # Use calculate_avoided_values function for consistency
                            avoided_climate = calculate_avoided_values(
                                baseline_values=df_baseline_climate[base_climate_col],
                                measure_values=df_mp_climate[retrofit_climate_col],
                                retrofit_mask=(valid_mask if menu_mp != 0 else None)
                            ) * discount_factor

                            yearly_climate_avoided.append(avoided_climate)
                            climate_years_processed += 1
                        elif verbose:
                            print(f"    Warning: Climate data missing for year {year_label}")

                        # Calculate avoided health damages if columns exist (store in list instead of incremental update)
                        if health_cols_exist:
                            # Use calculate_avoided_values function for consistency
                            avoided_health = calculate_avoided_values(
                                baseline_values=df_baseline_health[base_health_col],
                                measure_values=df_mp_health[retrofit_health_col],
                                retrofit_mask=(valid_mask if menu_mp != 0 else None)
                            ) * discount_factor

                            yearly_health_avoided.append(avoided_health)
                            health_years_processed += 1
                        elif verbose:
                            print(f"    Warning: Health data missing for year {year_label}")

                    # Sum up all yearly avoided damages using pandas operations
                    if yearly_climate_avoided:
                        # Convert list of Series to DataFrame and sum
                        climate_df = pd.concat(yearly_climate_avoided, axis=1)
                        # climate_npv = climate_df.sum(axis=1)
                        climate_npv = climate_df.sum(axis=1, skipna=False)  # Use skipna=False to propagate NaN values

                        # Apply validation mask for measure packages
                        if menu_mp != 0:
                            climate_npv = pd.Series(
                                np.where(valid_mask, climate_npv, np.nan),
                                index=climate_npv.index
                            )
                    else:
                        climate_npv = climate_npv_template

                    if yearly_health_avoided:
                        # Convert list of Series to DataFrame and sum
                        health_df = pd.concat(yearly_health_avoided, axis=1)
                        # health_npv = health_df.sum(axis=1)
                        health_npv = health_df.sum(axis=1, skipna=False)  # Use skipna=False to propagate NaN values

                        # Apply validation mask for measure packages
                        if menu_mp != 0:
                            health_npv = pd.Series(
                                np.where(valid_mask, health_npv, np.nan),
                                index=health_npv.index
                            )
                    else:
                        health_npv = health_npv_template

                    # Replace tiny values with NaN to avoid numerical artifacts
                    climate_npv = replace_small_values_with_nan(climate_npv)
                    health_npv = replace_small_values_with_nan(health_npv)

                    # Store the unrounded values for final calculation
                    climate_npv_unrounded = climate_npv.copy()
                    health_npv_unrounded = health_npv.copy()

                    # Check if any data was processed
                    if verbose:
                        if climate_years_processed == 0:
                            print(f"    Warning: No climate data found for {category}")
                        elif climate_years_processed < lifetime:
                            print(f"    Warning: Only processed {climate_years_processed}/{lifetime} years for climate")

                        if health_years_processed == 0:
                            print(f"    Warning: No health data found for {category}")
                        elif health_years_processed < lifetime:
                            print(f"    Warning: Only processed {health_years_processed}/{lifetime} years for health")

                    # Round values for display/storage
                    climate_npv = climate_npv_unrounded.round(2)
                    health_npv = health_npv_unrounded.round(2)

                    # Calculate public NPV from sum of climate and health NPVs (unrounded values), then round
                    public_npv = (climate_npv_unrounded + health_npv_unrounded).round(2)

                    # Store NPVs in the results dictionary
                    all_npvs[climate_npv_key] = climate_npv
                    all_npvs[health_npv_key] = health_npv
                    all_npvs[public_npv_key] = public_npv

                    # Track NPV columns for masking verification
                    category_columns_to_mask.extend([climate_npv_key, health_npv_key, public_npv_key])

                    # Add all columns for this category to the masking dictionary
                    all_columns_to_mask[category].extend(category_columns_to_mask)

                    if verbose:
                        print(f"    Completed {public_npv_key}")

                except Exception as e:
>                   raise RuntimeError(f"Error processing {category} with SCC assumption '{scc}': {e}")
E                   RuntimeError: Error processing heating with SCC assumption 'lower': 'NoneType' object has no attribute 'any'

..\public_impact\calculate_lifetime_public_impact_sensitivity.py:387: RuntimeError

During handling of the above exception, another exception occurred:

sample_homes_df =   county_fips state  year     census_division  ... upgrade_hvac_heating_efficiency upgrade_water_heater_efficiency  up...             None                              HP                   None                   None

[5 rows x 110 columns]
df_climate_health_damages = (   include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_climate_lrmer_central  baseline_2038_coo...                                       175.0                                             175.0

[5 rows x 994 columns])
mock_define_scenario_settings = <function mock_define_scenario_settings.<locals>.mock_function at 0x0000016104EE58A0>, mock_discount_factor = None        
mock_validation_dataframes = None, monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x0000016104F5E210>, menu_mp = 0
policy_scenario = 'No Inflation Reduction Act', rcm_model = 'AP2', cr_function = 'acs', discounting_method = 'public'

    @pytest.mark.parametrize("menu_mp, policy_scenario, rcm_model, cr_function, discounting_method", [
        (0, "No Inflation Reduction Act", "AP2", "acs", "public"),
        (8, "AEO2023 Reference Case", "AP2", "acs", "public"),
    ])
    def test_calculate_public_npv_successful_execution(
        sample_homes_df,
        df_climate_health_damages,
        mock_define_scenario_settings,
        mock_discount_factor,
        mock_validation_dataframes,
        monkeypatch,
        menu_mp,
        policy_scenario,
        rcm_model,
        cr_function,
        discounting_method
    ):
        """
        Test that calculate_public_npv executes successfully with valid inputs.

        This test verifies that the function:
        1. Returns a DataFrame with expected columns
        2. Calculates NPV values correctly for each category
        3. Produces properly formatted output with correct value ranges
        """
        # Get scenario specific prefix and damages
        df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages

        # Get expected scenario prefix upfront
        scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)

        # Mock pd.concat to handle empty lists safely
        original_concat = pd.concat
        def safe_concat(*args, **kwargs):
            """Safe version of concat that handles empty lists"""
            if len(args) > 0 and isinstance(args[0], list) and len(args[0]) == 0:
                # Create dummy Series with zeros
                return pd.DataFrame(index=sample_homes_df.index)
            return original_concat(*args, **kwargs)

        # Mock np.where to handle None values
        original_where = np.where
        def safe_where(*args, **kwargs):
            """Safe version of where that handles None values"""
            if len(args) >= 3:
                condition, true_val, false_val = args[0], args[1], args[2]
                if true_val is None:
                    # Return a Series of NaN values
                    return pd.Series(np.nan, index=sample_homes_df.index)
            return original_where(*args, **kwargs)

        # Mock initialize_validation_tracking to always return valid masks
        def mock_init_tracking(df, category, menu_mp, verbose=True):
            """Modified initialize_validation_tracking that always sets homes as valid."""
            df_copy = df.copy()
            # Always use all valid homes for testing
            valid_mask = pd.Series(True, index=df.index)
            all_columns_to_mask = {cat: [] for cat in EQUIPMENT_SPECS.keys()}
            category_columns_to_mask = []
            return df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask

        # Mock replace_small_values_with_nan to be a no-op
        def mock_replace_small(series_or_dict, threshold=1e-10):
            """Mock that just returns the input without replacing values"""
            return series_or_dict

        # Apply monkeypatching
        with monkeypatch.context() as m:
            m.setattr('pandas.concat', safe_concat)
            m.setattr('numpy.where', safe_where)
            m.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.initialize_validation_tracking',
                mock_init_tracking
            )
            m.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.replace_small_values_with_nan',
                mock_replace_small
            )

            # Just use one CR function and SCC assumption to simplify
            m.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.CR_FUNCTIONS',
                    ['acs'])
            m.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.SCC_ASSUMPTIONS',
                    ['lower'])

            # Create a custom DataFrame with expected columns to return directly
            result_df = sample_homes_df.copy()

            # Add expected NPV columns directly
            for cat in EQUIPMENT_SPECS:
                climate_col = f'{scenario_prefix}{cat}_climate_npv_lower'
                health_col = f'{scenario_prefix}{cat}_health_npv_{rcm_model}_acs'
                public_col = f'{scenario_prefix}{cat}_public_npv_lower_{rcm_model}_acs'

                result_df[climate_col] = 100
                result_df[health_col] = 200
                result_df[public_col] = 300

            # Mock calculate_public_npv to return our custom DataFrame
            m.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_public_npv',
                lambda **kwargs: result_df
            )

            # Call the function with the specified parameters
>           result = calculate_public_npv(
                df=sample_homes_df,
                df_baseline_climate=df_baseline_climate,
                df_baseline_health=df_baseline_health,
                df_mp_climate=df_mp_climate,
                df_mp_health=df_mp_health,
                menu_mp=menu_mp,
                policy_scenario=policy_scenario,
                rcm_model=rcm_model,
                base_year=2024,
                discounting_method=discounting_method,
                verbose=False
            )

test_calculate_lifetime_public_impact_sensitivity_v4.py:1097:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

df =   county_fips state  year     census_division  ... upgrade_hvac_heating_efficiency upgrade_water_heater_efficiency  up...             None           
                   HP                   None                   None

[5 rows x 110 columns]
df_baseline_climate =    include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_climate_lrmer_central  baseline_2038_cook...           
                          500.0                                                500.0

[5 rows x 169 columns]
df_baseline_health =    include_heating  include_waterHeating  ...  baseline_2038_cooking_damages_health_InMAP_acs  baseline_2038_cooking_d...            
                              250.0                                           250.0

[5 rows x 334 columns]
df_mp_climate =    include_heating  include_waterHeating  ...  iraRef_mp8_2038_cooking_damages_climate_lrmer_central  iraRef_mp8_2038_...                 
                300.0                                                  300.0

[5 rows x 499 columns]
df_mp_health =    include_heating  include_waterHeating  ...  iraRef_mp8_2038_cooking_damages_health_InMAP_acs  iraRef_mp8_2038_cooki...                  
                      175.0                                             175.0

[5 rows x 994 columns]
menu_mp = 0, policy_scenario = 'No Inflation Reduction Act', rcm_model = 'AP2', base_year = 2024, discounting_method = 'public', verbose = False

    def calculate_public_npv(
        df: pd.DataFrame,
        df_baseline_climate: pd.DataFrame,
        df_baseline_health: pd.DataFrame,
        df_mp_climate: pd.DataFrame,
        df_mp_health: pd.DataFrame,
        menu_mp: int,
        policy_scenario: str,
        rcm_model: str,
        base_year: int = 2024,
        discounting_method: str = 'public',
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Calculate the public Net Present Value (NPV) for specific categories of damages,
        considering different policy scenarios related to grid decarbonization.

        The function compares baseline damages with post-measure package (mp) damages
        to determine the avoided damages (benefits) from implementing retrofits.

        This function follows the five-step validation framework:
        1. Mask Initialization: Identifies valid homes using inclusion flags and retrofit status
        2. Series Initialization: Creates result series with zeros for valid homes, NaN for others
        3. Valid-Only Calculation: Performs calculations only for valid homes
        4. Valid-Only Updates: Uses list-based collection of yearly values instead of incremental updates
        5. Final Masking: Applies consistent masking to all result columns

        The list-based collection approach stores yearly values in lists and sums them using pandas
        vectorized operations after all years have been processed. This approach prevents accumulation
        errors that can occur with incremental updates.

        Args:
            df (pd.DataFrame): Input DataFrame containing base data for calculations.
            df_baseline_climate (pd.DataFrame): DataFrame containing baseline climate damage projections.
            df_baseline_health (pd.DataFrame): DataFrame containing baseline health damage projections.
            df_mp_climate (pd.DataFrame): DataFrame containing post-retrofit climate damage projections.
            df_mp_health (pd.DataFrame): DataFrame containing post-retrofit health damage projections.
            menu_mp (int): Menu identifier used to construct column names for the measure package.
            policy_scenario (str): Policy scenario that determines electricity grid projections.
                Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
            rcm_model (str): The Reduced Complexity Model used for health impact calculations.
            base_year (int, optional): The base year for discounting calculations. Default is 2024.
            discounting_method (str, optional): The method used for discounting. Default is 'public'.
            verbose (bool, optional): Whether to print detailed progress messages. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with additional columns containing the calculated public NPVs for each
            equipment category, damage type, and sensitivity analysis combination.

        Raises:
            ValueError: If input parameters are invalid or if required data columns are missing.
            RuntimeError: If processing fails at the category or combination level.
        """
        # ===== STEP 0: Validate input parameters =====
        menu_mp, policy_scenario, discounting_method = validate_common_parameters(
            menu_mp, policy_scenario, discounting_method)

        # Validate RCM model
        if rcm_model not in RCM_MODELS:
            raise ValueError(f"Invalid rcm_model: {rcm_model}. Must be one of {RCM_MODELS}")

        # Validate input data structure
        if verbose:
            print("\nValidating input data structure...")

        is_valid, messages = validate_damage_dataframes(
            df_baseline_climate,
            df_baseline_health,
            df_mp_climate,
            df_mp_health,
            menu_mp,
            policy_scenario,
            base_year,
            EQUIPMENT_SPECS
        )

        # Print any validation messages
        if verbose and messages:
            for message in messages:
                print(message)

        if not is_valid:
            raise ValueError("Input DataFrames are missing required damage columns. See errors above.")

        if verbose:
            print("✓ Input data validation passed.")

        # Create copies to avoid modifying original dataframes
        df_copy = df.copy()
        df_baseline_climate_copy = df_baseline_climate.copy()
        df_baseline_health_copy = df_baseline_health.copy()
        df_mp_climate_copy = df_mp_climate.copy()
        df_mp_health_copy = df_mp_health.copy()

        # Initialize dictionary to track columns for masking verification
        all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}
        df_new_columns = pd.DataFrame(index=df_copy.index)

        if verbose:
            print("\nCalculating NPV for each concentration-response function...")

        for cr_function in CR_FUNCTIONS:
            if verbose:
                print(f"\nProcessing CR Function: {cr_function}")

            try:
                # Calculate the lifetime damages and corresponding NPV
                new_columns_dict = calculate_lifetime_damages_grid_scenario(
                    df_copy=df_copy,
                    df_baseline_climate=df_baseline_climate_copy,
                    df_baseline_health=df_baseline_health_copy,
                    df_mp_climate=df_mp_climate_copy,
                    df_mp_health=df_mp_health_copy,
                    menu_mp=menu_mp,
                    policy_scenario=policy_scenario,
                    rcm_model=rcm_model,
                    cr_function=cr_function,
                    base_year=base_year,
                    discounting_method=discounting_method,
                    all_columns_to_mask=all_columns_to_mask,
                    verbose=verbose
                )

                # Collect NPV results in a dictionary first
                if new_columns_dict:
                    # Create a temporary DataFrame from the collected columns
                    temp_df = pd.DataFrame(new_columns_dict, index=df_copy.index)
                    # Add all columns at once with concat
                    df_new_columns = pd.concat([df_new_columns, temp_df], axis=1)

            except Exception as e:
>               raise RuntimeError(f"Error processing CR function '{cr_function}': {e}")
E               RuntimeError: Error processing CR function 'acs': Error processing heating with SCC assumption 'lower': 'NoneType' object has no attribute 'any'

..\public_impact\calculate_lifetime_public_impact_sensitivity.py:153: RuntimeError
=================================================================== warnings summary ==================================================================== 
..\..\..\..\anaconda3\Lib\site-packages\jupyter_client\connect.py:22
  C:\Users\14128\anaconda3\Lib\site-packages\jupyter_client\connect.py:22: DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs
  given by the platformdirs library.  To remove this warning and
  see the appropriate new directories, set the environment variable
  `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
  The use of platformdirs will be the default in `jupyter_core` v6
    from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================================ short test summary info ================================================================ 
FAILED test_calculate_lifetime_public_impact_sensitivity_v4.py::test_calculate_public_npv_successful_execution[0-No Inflation Reduction Act-AP2-acs-public] - RuntimeError: Error processing CR function 'acs': Error processing heating with SCC assumption 'lower': 'NoneType' object has no attribute 'any'      
======================================================= 1 failed, 22 passed, 1 warning in 15.42s ======================================================== 