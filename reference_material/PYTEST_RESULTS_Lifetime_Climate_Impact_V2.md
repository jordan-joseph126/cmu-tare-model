# ========================================================================================================================================================================================================================================================================
# TEST RESULTS FOR test_calculate_lifetime_climate_impacts_sensitivity_v2.py
# ========================================================================================================================================================================================================================================================================

============================================================== test session starts ===============================================================
platform win32 -- Python 3.11.7, pytest-7.4.0, pluggy-1.0.0
rootdir: C:\Users\14128\Research\cmu-tare-model
plugins: anyio-4.2.0
collected 31 items                                                                                                                                

test_calculate_lifetime_climate_impacts_sensitivity_v2.py ..F..F.............F...........                                                   [100%]

==================================================================== FAILURES ====================================================================
___________________________________________________ test_valid_only_calculation_implementation ___________________________________________________ 

args = ()
kwargs = {'adjusted_hdd_factor': 0    1.0
1    1.0
2    1.0
3    1.0
4    1.0
dtype: float64, 'category': 'heating', 'df':   co... 0.0005, 'pm25': 0.0002, 'so2': 0.0001}, 'propane': {'co2e': 0.06, 'nox': 0.0004, 'pm25': 0.0001, 'so2': 0.0002}}, ...}
retrofit_mask = 0     True
1     True
2    False
3     True
4    False
dtype: bool

    def mock_fossil_fuel_emissions(*args, **kwargs):
        """Mock to track if valid_mask is used for calculations."""
        nonlocal valid_mask_used
        retrofit_mask = kwargs.get('retrofit_mask')
        if retrofit_mask is not None:
            valid_mask_used = True

        # Return mock data with ALL required pollutants
        # Key fix: Make sure the result format EXACTLY matches what the source code expects
        return {
>           'so2': pd.Series(0.001, index=args[0].index),
            'nox': pd.Series(0.002, index=args[0].index),
            'pm25': pd.Series(0.003, index=args[0].index),
            'co2e': pd.Series(0.1, index=args[0].index)
        }
E       IndexError: tuple index out of range

test_calculate_lifetime_climate_impacts_sensitivity_v2.py:484: IndexError

During handling of the above exception, another exception occurred:

df =   county_fips state  year  ... upgrade_water_heater_efficiency upgrade_clothes_dryer upgrade_cooking_range
0       010...05    IL  2023  ...                              HP                  None                  None

[5 rows x 110 columns]
menu_mp = 8, policy_scenario = 'AEO2023 Reference Case', base_year = 2024, df_baseline_damages = None, verbose = False

    def calculate_lifetime_climate_impacts(
            df: pd.DataFrame,
            menu_mp: int,
            policy_scenario: str,
            base_year: int = 2024,
            df_baseline_damages: Optional[pd.DataFrame] = None,
            verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate lifetime climate impacts (CO2e emissions and climate damages) for each
        equipment category across all (mer_type, scc_value) combinations.

        This function processes each equipment category over its lifetime, computing annual
        and lifetime climate emissions/damages. Results are combined into two DataFrames:
        a main summary (df_main) and a detailed annual breakdown (df_detailed).

        This function follows the five-STEP validation framework:
        1. Mask Initialization: Identifies valid homes using inclusion flags and retrofit status
        2. Series Initialization: Creates result series with zeros for valid homes, NaN for others
        3. Valid-Only Calculation: Performs calculations only for valid homes
        4. Valid-Only Updates: Uses list-based collection of yearly values instead of incremental updates
        5. Final Masking: Applies consistent masking to all result columns

        The list-based collection approach stores yearly values in lists and sums them using pandas
        vectorized operations after all years have been processed. This approach prevents accumulation
        errors that can occur with incremental updates.

        Args:
            df (pd.DataFrame): Input DataFrame containing equipment consumption data, region info, etc.
            menu_mp (int): Measure package identifier (0 for baseline, nonzero for different scenarios).
            policy_scenario (str): Determines emissions scenario inputs (e.g., 'No Inflation Reduction Act' or 'AEO2023 Reference Case').
            base_year (int, optional): Base year for calculations. Defaults to 2024.
            df_baseline_damages (pd.DataFrame, optional): Baseline damages for computing avoided emissions/damages.
            verbose (bool, optional): Whether to print detailed progress messages. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - df_main: Main summary of lifetime climate impacts (rounded to 2 decimals).
                - df_detailed: Detailed annual and lifetime results (rounded to 2 decimals).

        Raises:
            ValueError: If menu_mp or policy_scenario is invalid.
            RuntimeError: If processing fails at the category or year level (e.g., missing data or key lookups).
        """
        # ===== STEP 0: Validate input parameters =====
        menu_mp, policy_scenario, _ = validate_common_parameters(
            menu_mp, policy_scenario, None)

        # Create a copy of the input df
        df_copy = df.copy()

        # Initialize the detailed DataFrame with the same index as df_copy
        df_detailed = pd.DataFrame(index=df_copy.index)

        # Copy inclusion flags and validation columns from df_copy to df_detailed
        validation_prefixes = ["include_", "valid_tech_", "valid_fuel_"]
        validation_cols = []
        for prefix in validation_prefixes:
            validation_cols.extend([col for col in df_copy.columns if col.startswith(prefix)])

        for col in validation_cols:
            df_detailed[col] = df_copy[col]

        # Initialize a dictionary to store lifetime climate impacts columns
        lifetime_columns_data = {}

        # Retrieve scenario-specific params for electricity/fossil-fuel emissions
        scenario_prefix, cambium_scenario, lookup_emissions_fossil_fuel, lookup_emissions_electricity_climate, _, _ = define_scenario_params(menu_mp, policy_scenario)

        # Precompute HDD adjustment factors by region and year
        hdd_factors_per_year = precompute_hdd_factors(df_copy)

        # Initialize dictionary to track columns for masking verification by category
        all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

        # Loop over each equipment category and its lifetime
        for category, lifetime in EQUIPMENT_SPECS.items():
            try:
                if verbose:
                    print(f"Calculating Climate Emissions and Damages from 2024 to {2024 + lifetime} for {category}")

                # ===== STEP 1: Initialize validation tracking for this category =====
                df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                    df_copy, category, menu_mp, verbose=verbose)

                # ===== STEP 2: Initialize result series for emissions and damages =====
                # Create templates for emissions and damages (for initialization only)
                lifetime_emissions_templates = {
                    mer_type: create_retrofit_only_series(df_copy, valid_mask)
                    for mer_type in MER_TYPES
                }

                lifetime_damages_templates = {
                    (mer_type, scc_value): create_retrofit_only_series(df_copy, valid_mask)
                    for mer_type in MER_TYPES
                    for scc_value in SCC_ASSUMPTIONS
                }

                # Create dictionaries to store yearly emissions and damages as lists
                yearly_emissions_lists = {mer_type: [] for mer_type in MER_TYPES}
                yearly_damages_lists = {
                    (mer_type, scc_value): []
                    for mer_type in MER_TYPES
                    for scc_value in SCC_ASSUMPTIONS
                }

                # Loop over each year in the equipment's lifetime
                for year in range(1, lifetime + 1):
                    try:
                        # Calculate the calendar year label (e.g., 2024, 2025, etc.)
                        year_label = year + (base_year - 1)

                        # Retrieve HDD factor for the current year; raise exception if missing
                        if year_label not in hdd_factors_per_year:
                            raise KeyError(f"HDD factor for year {year_label} not found.")
                        hdd_factor = hdd_factors_per_year[year_label]

                        # The adjusted HDD factor only applies to heating/waterHeating categories
                        # For other categories, use a default value of 1.0
                        adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)       

                        # Calculate fossil fuel emissions for the current category and year
                        total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
                            df=df_copy,
                            category=category,
                            adjusted_hdd_factor=adjusted_hdd_factor,
                            lookup_emissions_fossil_fuel=lookup_emissions_fossil_fuel,
                            menu_mp=menu_mp,
                            retrofit_mask=valid_mask,
                            verbose=verbose
                        )

                        # Compute climate emissions and damages with scc_value sensitivities
                        climate_results, annual_emissions, annual_damages = calculate_climate_emissions_and_damages(
                            df=df_copy,
                            category=category,
                            year_label=year_label,
                            adjusted_hdd_factor=adjusted_hdd_factor,
                            lookup_emissions_electricity_climate=lookup_emissions_electricity_climate,
                            cambium_scenario=cambium_scenario,
                            total_fossil_fuel_emissions=total_fossil_fuel_emissions,
                            scenario_prefix=scenario_prefix,
                            menu_mp=menu_mp
                        )

                        # ===== STEP 3 & 4: Store annual emissions and damages in lists =====
                        for mer_type in MER_TYPES:
                            emissions_values = annual_emissions.get(mer_type, 0.0).copy()
                            # Apply validation mask for measure packages
                            if menu_mp != 0:
                                emissions_values.loc[~valid_mask] = 0.0
                            yearly_emissions_lists[mer_type].append(emissions_values)

                        # Store annual damages in lists
                        for key, value in annual_damages.items():
                            damages_values = value.copy()
                            # Apply validation mask for measure packages
                            if menu_mp != 0:
                                damages_values.loc[~valid_mask] = 0.0
                            yearly_damages_lists[key].append(damages_values)

                        # # Add columns to detailed DataFrame
                        # for col_name, values in climate_results.items():
                        #     df_detailed[col_name] = values
                        #     category_columns_to_mask.append(col_name)

                        # Store annual results in a temporary dictionary
                        annual_detailed_columns = {}
                        for col_name, values in climate_results.items():
                            annual_detailed_columns[col_name] = values
                            category_columns_to_mask.append(col_name)

                        # Add to df_detailed with a single concat operation
                        if annual_detailed_columns:
                            annual_df = pd.DataFrame(annual_detailed_columns, index=df_copy.index)
                            df_detailed = pd.concat([df_detailed, annual_df], axis=1)

                    except Exception as e:
                        # Convert any exception into a RuntimeError with additional context
>                       raise RuntimeError(f"Error processing year {year_label} for category '{category}': {e}")
E                       RuntimeError: Error processing year 2024 for category 'heating': tuple index out of range

..\public_impact\calculate_lifetime_climate_impacts_sensitivity.py:202: RuntimeError

During handling of the above exception, another exception occurred:

sample_homes_df =   county_fips state  year  ... upgrade_water_heater_efficiency upgrade_clothes_dryer upgrade_cooking_range
0       010...05    IL  2023  ...                              HP                  None                  None

[5 rows x 110 columns]
dummy_define_scenario_settings = ('baseline_', 'MidCase', {'fuelOil': {'co2e': 0.07, 'nox': 0.0007, 'pm25': 0.0003, 'so2': 0.0008}, 'naturalGas': {'co2...rmer_mt_per_kWh_co2e': 0.03}, 2026: {'lrmer_mt_per_kWh_co2e': 0.02, 'srmer_mt_per_kWh_co2e': 0.03}, ...}, ...}, {}, {})
mock_precompute_hdd_factors = None, monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x000001CE990DCE10>

    def test_valid_only_calculation_implementation(
            sample_homes_df,
            dummy_define_scenario_settings,
            mock_precompute_hdd_factors,
            monkeypatch):
        """
        Test Step 3: Valid-only calculation for qualifying homes.
        """
        # Define a mock fossil fuel emissions function to track valid_mask usage
        valid_mask_used = False

        def mock_fossil_fuel_emissions(*args, **kwargs):
            """Mock to track if valid_mask is used for calculations."""
            nonlocal valid_mask_used
            retrofit_mask = kwargs.get('retrofit_mask')
            if retrofit_mask is not None:
                valid_mask_used = True

            # Return mock data with ALL required pollutants
            # Key fix: Make sure the result format EXACTLY matches what the source code expects
            return {
                'so2': pd.Series(0.001, index=args[0].index),
                'nox': pd.Series(0.002, index=args[0].index),
                'pm25': pd.Series(0.003, index=args[0].index),
                'co2e': pd.Series(0.1, index=args[0].index)
            }

        # Apply monkeypatching
        monkeypatch.setattr(
            'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_fossil_fuel_emissions',
            mock_fossil_fuel_emissions
        )

        # Call the main function
        menu_mp = 8
        policy_scenario = 'AEO2023 Reference Case'

>       df_main, _ = calculate_lifetime_climate_impacts(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )

test_calculate_lifetime_climate_impacts_sensitivity_v2.py:500:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

df =   county_fips state  year  ... upgrade_water_heater_efficiency upgrade_clothes_dryer upgrade_cooking_range
0       010...05    IL  2023  ...                              HP                  None                  None

[5 rows x 110 columns]
menu_mp = 8, policy_scenario = 'AEO2023 Reference Case', base_year = 2024, df_baseline_damages = None, verbose = False

    def calculate_lifetime_climate_impacts(
            df: pd.DataFrame,
            menu_mp: int,
            policy_scenario: str,
            base_year: int = 2024,
            df_baseline_damages: Optional[pd.DataFrame] = None,
            verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate lifetime climate impacts (CO2e emissions and climate damages) for each
        equipment category across all (mer_type, scc_value) combinations.

        This function processes each equipment category over its lifetime, computing annual
        and lifetime climate emissions/damages. Results are combined into two DataFrames:
        a main summary (df_main) and a detailed annual breakdown (df_detailed).

        This function follows the five-STEP validation framework:
        1. Mask Initialization: Identifies valid homes using inclusion flags and retrofit status
        2. Series Initialization: Creates result series with zeros for valid homes, NaN for others
        3. Valid-Only Calculation: Performs calculations only for valid homes
        4. Valid-Only Updates: Uses list-based collection of yearly values instead of incremental updates
        5. Final Masking: Applies consistent masking to all result columns

        The list-based collection approach stores yearly values in lists and sums them using pandas
        vectorized operations after all years have been processed. This approach prevents accumulation
        errors that can occur with incremental updates.

        Args:
            df (pd.DataFrame): Input DataFrame containing equipment consumption data, region info, etc.
            menu_mp (int): Measure package identifier (0 for baseline, nonzero for different scenarios).
            policy_scenario (str): Determines emissions scenario inputs (e.g., 'No Inflation Reduction Act' or 'AEO2023 Reference Case').
            base_year (int, optional): Base year for calculations. Defaults to 2024.
            df_baseline_damages (pd.DataFrame, optional): Baseline damages for computing avoided emissions/damages.
            verbose (bool, optional): Whether to print detailed progress messages. Defaults to False.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - df_main: Main summary of lifetime climate impacts (rounded to 2 decimals).
                - df_detailed: Detailed annual and lifetime results (rounded to 2 decimals).

        Raises:
            ValueError: If menu_mp or policy_scenario is invalid.
            RuntimeError: If processing fails at the category or year level (e.g., missing data or key lookups).
        """
        # ===== STEP 0: Validate input parameters =====
        menu_mp, policy_scenario, _ = validate_common_parameters(
            menu_mp, policy_scenario, None)

        # Create a copy of the input df
        df_copy = df.copy()

        # Initialize the detailed DataFrame with the same index as df_copy
        df_detailed = pd.DataFrame(index=df_copy.index)

        # Copy inclusion flags and validation columns from df_copy to df_detailed
        validation_prefixes = ["include_", "valid_tech_", "valid_fuel_"]
        validation_cols = []
        for prefix in validation_prefixes:
            validation_cols.extend([col for col in df_copy.columns if col.startswith(prefix)])

        for col in validation_cols:
            df_detailed[col] = df_copy[col]

        # Initialize a dictionary to store lifetime climate impacts columns
        lifetime_columns_data = {}

        # Retrieve scenario-specific params for electricity/fossil-fuel emissions
        scenario_prefix, cambium_scenario, lookup_emissions_fossil_fuel, lookup_emissions_electricity_climate, _, _ = define_scenario_params(menu_mp, policy_scenario)

        # Precompute HDD adjustment factors by region and year
        hdd_factors_per_year = precompute_hdd_factors(df_copy)

        # Initialize dictionary to track columns for masking verification by category
        all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

        # Loop over each equipment category and its lifetime
        for category, lifetime in EQUIPMENT_SPECS.items():
            try:
                if verbose:
                    print(f"Calculating Climate Emissions and Damages from 2024 to {2024 + lifetime} for {category}")

                # ===== STEP 1: Initialize validation tracking for this category =====
                df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                    df_copy, category, menu_mp, verbose=verbose)

                # ===== STEP 2: Initialize result series for emissions and damages =====
                # Create templates for emissions and damages (for initialization only)
                lifetime_emissions_templates = {
                    mer_type: create_retrofit_only_series(df_copy, valid_mask)
                    for mer_type in MER_TYPES
                }

                lifetime_damages_templates = {
                    (mer_type, scc_value): create_retrofit_only_series(df_copy, valid_mask)
                    for mer_type in MER_TYPES
                    for scc_value in SCC_ASSUMPTIONS
                }

                # Create dictionaries to store yearly emissions and damages as lists
                yearly_emissions_lists = {mer_type: [] for mer_type in MER_TYPES}
                yearly_damages_lists = {
                    (mer_type, scc_value): []
                    for mer_type in MER_TYPES
                    for scc_value in SCC_ASSUMPTIONS
                }

                # Loop over each year in the equipment's lifetime
                for year in range(1, lifetime + 1):
                    try:
                        # Calculate the calendar year label (e.g., 2024, 2025, etc.)
                        year_label = year + (base_year - 1)

                        # Retrieve HDD factor for the current year; raise exception if missing
                        if year_label not in hdd_factors_per_year:
                            raise KeyError(f"HDD factor for year {year_label} not found.")
                        hdd_factor = hdd_factors_per_year[year_label]

                        # The adjusted HDD factor only applies to heating/waterHeating categories
                        # For other categories, use a default value of 1.0
                        adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)       

                        # Calculate fossil fuel emissions for the current category and year
                        total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
                            df=df_copy,
                            category=category,
                            adjusted_hdd_factor=adjusted_hdd_factor,
                            lookup_emissions_fossil_fuel=lookup_emissions_fossil_fuel,
                            menu_mp=menu_mp,
                            retrofit_mask=valid_mask,
                            verbose=verbose
                        )

                        # Compute climate emissions and damages with scc_value sensitivities
                        climate_results, annual_emissions, annual_damages = calculate_climate_emissions_and_damages(
                            df=df_copy,
                            category=category,
                            year_label=year_label,
                            adjusted_hdd_factor=adjusted_hdd_factor,
                            lookup_emissions_electricity_climate=lookup_emissions_electricity_climate,
                            cambium_scenario=cambium_scenario,
                            total_fossil_fuel_emissions=total_fossil_fuel_emissions,
                            scenario_prefix=scenario_prefix,
                            menu_mp=menu_mp
                        )

                        # ===== STEP 3 & 4: Store annual emissions and damages in lists =====
                        for mer_type in MER_TYPES:
                            emissions_values = annual_emissions.get(mer_type, 0.0).copy()
                            # Apply validation mask for measure packages
                            if menu_mp != 0:
                                emissions_values.loc[~valid_mask] = 0.0
                            yearly_emissions_lists[mer_type].append(emissions_values)

                        # Store annual damages in lists
                        for key, value in annual_damages.items():
                            damages_values = value.copy()
                            # Apply validation mask for measure packages
                            if menu_mp != 0:
                                damages_values.loc[~valid_mask] = 0.0
                            yearly_damages_lists[key].append(damages_values)

                        # # Add columns to detailed DataFrame
                        # for col_name, values in climate_results.items():
                        #     df_detailed[col_name] = values
                        #     category_columns_to_mask.append(col_name)

                        # Store annual results in a temporary dictionary
                        annual_detailed_columns = {}
                        for col_name, values in climate_results.items():
                            annual_detailed_columns[col_name] = values
                            category_columns_to_mask.append(col_name)

                        # Add to df_detailed with a single concat operation
                        if annual_detailed_columns:
                            annual_df = pd.DataFrame(annual_detailed_columns, index=df_copy.index)
                            df_detailed = pd.concat([df_detailed, annual_df], axis=1)

                    except Exception as e:
                        # Convert any exception into a RuntimeError with additional context
                        raise RuntimeError(f"Error processing year {year_label} for category '{category}': {e}")

                # Sum up lifetime emissions and damages using pandas operations
                lifetime_climate_emissions = {}
                for mer_type in MER_TYPES:
                    if yearly_emissions_lists[mer_type]:
                        # Convert list of Series to DataFrame and sum
                        emissions_df = pd.concat(yearly_emissions_lists[mer_type], axis=1)
                        # total_emissions = emissions_df.sum(axis=1)
                        total_emissions = emissions_df.sum(axis=1, skipna=False)  # Use skipna=False to propagate NaN values

                        # Apply validation mask for measure packages
                        if menu_mp != 0:
                            total_emissions = pd.Series(
                                np.where(valid_mask, total_emissions, np.nan),
                                index=total_emissions.index
                            )
                        lifetime_climate_emissions[mer_type] = total_emissions
                    else:
                        lifetime_climate_emissions[mer_type] = lifetime_emissions_templates[mer_type]

                # Sum up lifetime damages using pandas operations
                lifetime_climate_damages = {}
                for key in yearly_damages_lists:
                    if yearly_damages_lists[key]:
                        # Convert list of Series to DataFrame and sum
                        damages_df = pd.concat(yearly_damages_lists[key], axis=1)
                        # total_damages = damages_df.sum(axis=1)
                        total_damages = damages_df.sum(axis=1, skipna=False)  # Use skipna=False to propagate NaN values

                        # Apply validation mask for measure packages
                        if menu_mp != 0:
                            total_damages = pd.Series(
                                np.where(valid_mask, total_damages, np.nan),
                                index=total_damages.index
                            )
                        lifetime_climate_damages[key] = total_damages
                    else:
                        lifetime_climate_damages[key] = lifetime_damages_templates[key]

                # Prepare lifetime columns
                lifetime_dict = {}
                for mer_type in MER_TYPES:
                    emissions_col = f'{scenario_prefix}{category}_lifetime_mt_co2e_{mer_type}'
                    lifetime_dict[emissions_col] = lifetime_climate_emissions[mer_type]
                    category_columns_to_mask.append(emissions_col)

                    for scc_assumption in SCC_ASSUMPTIONS:
                        damages_col = f'{scenario_prefix}{category}_lifetime_damages_climate_{mer_type}_{scc_assumption}'
                        lifetime_dict[damages_col] = lifetime_climate_damages[(mer_type, scc_assumption)]
                        category_columns_to_mask.append(damages_col)

                        # Calculate avoided damages if baseline data is provided
                        if menu_mp != 0 and df_baseline_damages is not None:
                            baseline_damages_col = f'baseline_{category}_lifetime_damages_climate_{mer_type}_{scc_assumption}'
                            avoided_damages_col = f'{scenario_prefix}{category}_avoided_damages_climate_{mer_type}_{scc_assumption}'

                            # Calculate avoided damages only for homes with retrofits
                            lifetime_dict[avoided_damages_col] = calculate_avoided_values(
                                baseline_values=df_baseline_damages[baseline_damages_col],
                                measure_values=lifetime_dict[damages_col],
                                retrofit_mask=valid_mask
                            )
                            category_columns_to_mask.extend([baseline_damages_col, avoided_damages_col])

                    # Calculate avoided emissions if baseline data is provided
                    if menu_mp != 0 and df_baseline_damages is not None:
                        baseline_emissions_col = f'baseline_{category}_lifetime_mt_co2e_{mer_type}'
                        avoided_emissions_col = f'{scenario_prefix}{category}_avoided_mt_co2e_{mer_type}'

                        # Calculate avoided emissions only for homes with retrofits
                        lifetime_dict[avoided_emissions_col] = calculate_avoided_values(
                            baseline_values=df_baseline_damages[baseline_emissions_col],
                            measure_values=lifetime_dict[emissions_col],
                            retrofit_mask=valid_mask
                        )
                        category_columns_to_mask.extend([baseline_emissions_col, avoided_emissions_col])

                # Store in global lifetime dictionary
                lifetime_columns_data.update(lifetime_dict)

                # # Append these columns to df_detailed for completeness
                # for col_name, values in lifetime_dict.items():
                #     df_detailed[col_name] = values

                # Create a temporary DataFrame from lifetime_dict and then concatenate
                lifetime_df = pd.DataFrame(lifetime_dict, index=df_copy.index)
                df_detailed = pd.concat([df_detailed, lifetime_df], axis=1)

                # Add all columns for this category to the masking dictionary
                all_columns_to_mask[category].extend(category_columns_to_mask)

            except Exception as e:
                # Convert any exception into a RuntimeError with additional context
>               raise RuntimeError(f"Error processing category '{category}': {e}")
E               RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': tuple index out of range     

..\public_impact\calculate_lifetime_climate_impacts_sensitivity.py:296: RuntimeError
_______________________________________________________ test_final_masking_implementation ________________________________________________________ 

sample_homes_df =   county_fips state  year  ... upgrade_water_heater_efficiency upgrade_clothes_dryer upgrade_cooking_range
0       010...05    IL  2023  ...                              HP                  None                  None

[5 rows x 110 columns]
dummy_define_scenario_settings = ('baseline_', 'MidCase', {'fuelOil': {'co2e': 0.07, 'nox': 0.0007, 'pm25': 0.0003, 'so2': 0.0008}, 'naturalGas': {'co2...rmer_mt_per_kWh_co2e': 0.03}, 2026: {'lrmer_mt_per_kWh_co2e': 0.02, 'srmer_mt_per_kWh_co2e': 0.03}, ...}, ...}, {}, {})
mock_precompute_hdd_factors = None, monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x000001CE990E5190>

    def test_final_masking_implementation(
            sample_homes_df,
            dummy_define_scenario_settings,
            mock_precompute_hdd_factors,
            monkeypatch):
        """
        Test Step 5: Final masking with apply_final_masking().

        This test verifies that calculate_lifetime_climate_impacts correctly:
        1. Applies final masking to ensure consistent NaN values
        2. Passes appropriate columns to apply_final_masking()
        3. Tracks lifetime columns for masking
        """
        # Track which columns are passed to apply_final_masking
        masking_columns_captured = {}

        def mock_apply_masking(df, all_columns_to_mask, verbose=True):
            """Mock to track calls to apply_final_masking."""
            nonlocal masking_columns_captured
            masking_columns_captured = all_columns_to_mask.copy()
            # Return the input DataFrame for testing simplicity
            return df

        # Apply monkeypatching
        monkeypatch.setattr(
            'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.apply_final_masking',
            mock_apply_masking
        )

        # Call the main function
        menu_mp = 8
        policy_scenario = 'AEO2023 Reference Case'

        df_main, _ = calculate_lifetime_climate_impacts(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )

        # Verify lifetime columns are tracked for masking
        category = 'heating'  # Test one category for simplicity

        assert category in masking_columns_captured, \
            f"Category '{category}' should be in masking columns"

        # Check if the lifetime columns are tracked
        for mer_type in MER_TYPES:
            lifetime_col = f'iraRef_mp{menu_mp}_{category}_lifetime_mt_co2e_{mer_type}'

            # Find if the column is in any of the tracked lists
            found = False
            for tracked_cols in masking_columns_captured.values():
                if lifetime_col in tracked_cols:
                    found = True
                    break

>           assert found, f"Lifetime column '{lifetime_col}' should be tracked for masking"
E           AssertionError: Lifetime column 'iraRef_mp8_heating_lifetime_mt_co2e_lrmer' should be tracked for masking
E           assert False

test_calculate_lifetime_climate_impacts_sensitivity_v2.py:686: AssertionError
-------------------------------------------------------------- Captured stdout call -------------------------------------------------------------- 

Verifying masking for all calculated columns:
___________________________________________ test_calculate_lifetime_climate_impacts_boundary_lifetime ____________________________________________ 

obj = <module 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity' from 'c:\\users\\14128\\research\\cmu-tare-model\\cmu_tare_model\\public_impact\\calculate_lifetime_climate_impacts_sensitivity.py'>
name = 'UPGRADE_COLUMNS', ann = 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity'

    def annotated_getattr(obj: object, name: str, ann: str) -> object:
        try:
>           obj = getattr(obj, name)
E           AttributeError: module 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity' has no attribute 'UPGRADE_COLUMNS'

..\..\..\..\anaconda3\Lib\site-packages\_pytest\monkeypatch.py:89: AttributeError

The above exception was the direct cause of the following exception:

sample_homes_df =   county_fips state  year  ... upgrade_water_heater_efficiency upgrade_clothes_dryer upgrade_cooking_range
0       010...05    IL  2023  ...                              HP                  None                  None

[5 rows x 110 columns]
dummy_define_scenario_settings = ('baseline_', 'MidCase', {'fuelOil': {'co2e': 0.07, 'nox': 0.0007, 'pm25': 0.0003, 'so2': 0.0008}, 'naturalGas': {'co2...rmer_mt_per_kWh_co2e': 0.03}, 2026: {'lrmer_mt_per_kWh_co2e': 0.02, 'srmer_mt_per_kWh_co2e': 0.03}, ...}, ...}, {}, {})
mock_precompute_hdd_factors = None, monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x000001CE992D2E10>

    def test_calculate_lifetime_climate_impacts_boundary_lifetime(
            sample_homes_df,
            dummy_define_scenario_settings,
            mock_precompute_hdd_factors,
            monkeypatch):
        """
        Test boundary condition for equipment lifetime by temporarily overriding
        EQUIPMENT_SPECS with a category that has lifetime=1.
        """
        # Create a temporary category with lifetime=1
        test_category = "test_cat"
        original_specs = EQUIPMENT_SPECS.copy()

        try:
            # Create a new DataFrame to avoid modifying the fixture
            sample_homes_df_modified = sample_homes_df.copy()

            # IMPORTANT: Add all required columns for the test_cat category
            # Include validation flags
            sample_homes_df_modified[f'include_{test_category}'] = True
            sample_homes_df_modified[f'valid_fuel_{test_category}'] = True
            sample_homes_df_modified[f'valid_tech_{test_category}'] = True

            # Add fuel type columns
            sample_homes_df_modified[f'base_{test_category}_fuel'] = ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil', 'Electricity']

            # Add all required consumption columns
            for fuel in ['electricity', 'naturalGas', 'fuelOil', 'propane']:
                col_name = f'base_{fuel}_{test_category}_consumption'
                sample_homes_df_modified[col_name] = [100, 200, 300, 400, 500]

            # Add baseline total consumption
            sample_homes_df_modified[f'baseline_{test_category}_consumption'] = [500, 600, 700, 800, 900]

            # Add consumption for all years needed
            for year in range(2024, 2027):  # Only need a few years since lifetime=1
                sample_homes_df_modified[f'mp8_{year}_{test_category}_consumption'] = [80, 90, 100, 110, 120]

            # Override EQUIPMENT_SPECS to use only test_cat with lifetime=1
            mock_test_specs = {test_category: 1}
            monkeypatch.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.EQUIPMENT_SPECS',
                mock_test_specs
            )

            # Add a temporary upgrade column for test_cat
            sample_homes_df_modified[f'upgrade_{test_category}_efficiency'] = ['HP', 'HP', 'HP', 'HP', 'HP']

            # Create a fake UPGRADE_COLUMNS for test_cat
            mock_upgrade_columns = {test_category: f'upgrade_{test_category}_efficiency'}
            monkeypatch.setattr(
                'cmu_tare_model.constants.UPGRADE_COLUMNS',
                mock_upgrade_columns
            )
>           monkeypatch.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.UPGRADE_COLUMNS',
                mock_upgrade_columns
            )

test_calculate_lifetime_climate_impacts_sensitivity_v2.py:1181:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  
..\..\..\..\anaconda3\Lib\site-packages\_pytest\monkeypatch.py:105: in derive_importpath
    annotated_getattr(target, attr, ann=module)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

obj = <module 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity' from 'c:\\users\\14128\\research\\cmu-tare-model\\cmu_tare_model\\public_impact\\calculate_lifetime_climate_impacts_sensitivity.py'>
name = 'UPGRADE_COLUMNS', ann = 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity'

    def annotated_getattr(obj: object, name: str, ann: str) -> object:
        try:
            obj = getattr(obj, name)
        except AttributeError as e:
>           raise AttributeError(
                "{!r} object at {} has no attribute {!r}".format(
                    type(obj).__name__, ann, name
                )
            ) from e
E           AttributeError: 'module' object at cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity has no attribute 'UPGRADE_COLUMNS'

..\..\..\..\anaconda3\Lib\site-packages\_pytest\monkeypatch.py:91: AttributeError

During handling of the above exception, another exception occurred:

obj = <module 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity' from 'c:\\users\\14128\\research\\cmu-tare-model\\cmu_tare_model\\public_impact\\calculate_lifetime_climate_impacts_sensitivity.py'>
name = 'UPGRADE_COLUMNS', ann = 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity'

    def annotated_getattr(obj: object, name: str, ann: str) -> object:
        try:
>           obj = getattr(obj, name)
E           AttributeError: module 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity' has no attribute 'UPGRADE_COLUMNS'

..\..\..\..\anaconda3\Lib\site-packages\_pytest\monkeypatch.py:89: AttributeError

The above exception was the direct cause of the following exception:

sample_homes_df =   county_fips state  year  ... upgrade_water_heater_efficiency upgrade_clothes_dryer upgrade_cooking_range
0       010...05    IL  2023  ...                              HP                  None                  None

[5 rows x 110 columns]
dummy_define_scenario_settings = ('baseline_', 'MidCase', {'fuelOil': {'co2e': 0.07, 'nox': 0.0007, 'pm25': 0.0003, 'so2': 0.0008}, 'naturalGas': {'co2...rmer_mt_per_kWh_co2e': 0.03}, 2026: {'lrmer_mt_per_kWh_co2e': 0.02, 'srmer_mt_per_kWh_co2e': 0.03}, ...}, ...}, {}, {})
mock_precompute_hdd_factors = None, monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x000001CE992D2E10>

    def test_calculate_lifetime_climate_impacts_boundary_lifetime(
            sample_homes_df,
            dummy_define_scenario_settings,
            mock_precompute_hdd_factors,
            monkeypatch):
        """
        Test boundary condition for equipment lifetime by temporarily overriding
        EQUIPMENT_SPECS with a category that has lifetime=1.
        """
        # Create a temporary category with lifetime=1
        test_category = "test_cat"
        original_specs = EQUIPMENT_SPECS.copy()

        try:
            # Create a new DataFrame to avoid modifying the fixture
            sample_homes_df_modified = sample_homes_df.copy()

            # IMPORTANT: Add all required columns for the test_cat category
            # Include validation flags
            sample_homes_df_modified[f'include_{test_category}'] = True
            sample_homes_df_modified[f'valid_fuel_{test_category}'] = True
            sample_homes_df_modified[f'valid_tech_{test_category}'] = True

            # Add fuel type columns
            sample_homes_df_modified[f'base_{test_category}_fuel'] = ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil', 'Electricity']

            # Add all required consumption columns
            for fuel in ['electricity', 'naturalGas', 'fuelOil', 'propane']:
                col_name = f'base_{fuel}_{test_category}_consumption'
                sample_homes_df_modified[col_name] = [100, 200, 300, 400, 500]

            # Add baseline total consumption
            sample_homes_df_modified[f'baseline_{test_category}_consumption'] = [500, 600, 700, 800, 900]

            # Add consumption for all years needed
            for year in range(2024, 2027):  # Only need a few years since lifetime=1
                sample_homes_df_modified[f'mp8_{year}_{test_category}_consumption'] = [80, 90, 100, 110, 120]

            # Override EQUIPMENT_SPECS to use only test_cat with lifetime=1
            mock_test_specs = {test_category: 1}
            monkeypatch.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.EQUIPMENT_SPECS',
                mock_test_specs
            )

            # Add a temporary upgrade column for test_cat
            sample_homes_df_modified[f'upgrade_{test_category}_efficiency'] = ['HP', 'HP', 'HP', 'HP', 'HP']

            # Create a fake UPGRADE_COLUMNS for test_cat
            mock_upgrade_columns = {test_category: f'upgrade_{test_category}_efficiency'}
            monkeypatch.setattr(
                'cmu_tare_model.constants.UPGRADE_COLUMNS',
                mock_upgrade_columns
            )
            monkeypatch.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.UPGRADE_COLUMNS',
                mock_upgrade_columns
            )

            # Call the function with a baseline scenario (menu_mp=0)
            df_main, _ = calculate_lifetime_climate_impacts(
                df=sample_homes_df_modified,
                menu_mp=0,
                policy_scenario="No Inflation Reduction Act",
                verbose=False
            )

            # Verify results contain the test category
            for mer_type in MER_TYPES:
                emissions_col = f"baseline_{test_category}_lifetime_mt_co2e_{mer_type}"
                assert emissions_col in df_main.columns, f"Missing column {emissions_col}"

        finally:
            # Restore original specs to avoid side effects
            monkeypatch.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.EQUIPMENT_SPECS',
                original_specs
            )
            # Restore original UPGRADE_COLUMNS if needed
            original_upgrade_columns = {
                'heating': 'upgrade_hvac_heating_efficiency',
                'waterHeating': 'upgrade_water_heater_efficiency',
                'clothesDrying': 'upgrade_clothes_dryer',
                'cooking': 'upgrade_cooking_range'
            }
            monkeypatch.setattr(
                'cmu_tare_model.constants.UPGRADE_COLUMNS',
                original_upgrade_columns
            )
>           monkeypatch.setattr(
                'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.UPGRADE_COLUMNS',
                original_upgrade_columns
            )

test_calculate_lifetime_climate_impacts_sensitivity_v2.py:1216:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
..\..\..\..\anaconda3\Lib\site-packages\_pytest\monkeypatch.py:105: in derive_importpath
    annotated_getattr(target, attr, ann=module)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  

obj = <module 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity' from 'c:\\users\\14128\\research\\cmu-tare-model\\cmu_tare_model\\public_impact\\calculate_lifetime_climate_impacts_sensitivity.py'>
name = 'UPGRADE_COLUMNS', ann = 'cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity'

    def annotated_getattr(obj: object, name: str, ann: str) -> object:
        try:
            obj = getattr(obj, name)
        except AttributeError as e:
>           raise AttributeError(
                "{!r} object at {} has no attribute {!r}".format(
                    type(obj).__name__, ann, name
                )
            ) from e
E           AttributeError: 'module' object at cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity has no attribute 'UPGRADE_COLUMNS'

..\..\..\..\anaconda3\Lib\site-packages\_pytest\monkeypatch.py:91: AttributeError
================================================================ warnings summary ================================================================ 
..\..\..\..\anaconda3\Lib\site-packages\jupyter_client\connect.py:22
  C:\Users\14128\anaconda3\Lib\site-packages\jupyter_client\connect.py:22: DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs
  given by the platformdirs library.  To remove this warning and
  see the appropriate new directories, set the environment variable
  `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
  The use of platformdirs will be the default in `jupyter_core` v6
    from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write

cmu_tare_model/tests/test_calculate_lifetime_climate_impacts_sensitivity_v2.py: 60 warnings
  c:\Users\14128\Research\cmu-tare-model\cmu_tare_model\tests\test_calculate_lifetime_climate_impacts_sensitivity_v2.py:138: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    df[f'mp8_{year}_{category}_consumption'] = [80, 90, 100, 110, 120]

cmu_tare_model/tests/test_calculate_lifetime_climate_impacts_sensitivity_v2.py: 30 warnings
  c:\Users\14128\Research\cmu-tare-model\cmu_tare_model\tests\test_calculate_lifetime_climate_impacts_sensitivity_v2.py:141: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    df['upgrade_hvac_heating_efficiency'] = ['ASHP', 'GSHP', None, 'ASHP', None]

cmu_tare_model/tests/test_calculate_lifetime_climate_impacts_sensitivity_v2.py: 30 warnings
  c:\Users\14128\Research\cmu-tare-model\cmu_tare_model\tests\test_calculate_lifetime_climate_impacts_sensitivity_v2.py:142: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    df['upgrade_water_heater_efficiency'] = ['HP', None, 'HP', None, 'HP']

cmu_tare_model/tests/test_calculate_lifetime_climate_impacts_sensitivity_v2.py: 30 warnings
  c:\Users\14128\Research\cmu-tare-model\cmu_tare_model\tests\test_calculate_lifetime_climate_impacts_sensitivity_v2.py:143: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    df['upgrade_clothes_dryer'] = [None, 'Electric', None, 'Electric', None]

cmu_tare_model/tests/test_calculate_lifetime_climate_impacts_sensitivity_v2.py: 30 warnings
  c:\Users\14128\Research\cmu-tare-model\cmu_tare_model\tests\test_calculate_lifetime_climate_impacts_sensitivity_v2.py:144: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    df['upgrade_cooking_range'] = ['Induction', None, 'Induction', None, None]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================================ short test summary info ============================================================= 
FAILED test_calculate_lifetime_climate_impacts_sensitivity_v2.py::test_valid_only_calculation_implementation - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': tuple index out of range
FAILED test_calculate_lifetime_climate_impacts_sensitivity_v2.py::test_final_masking_implementation - AssertionError: Lifetime column 'iraRef_mp8_heating_lifetime_mt_co2e_lrmer' should be tracked for masking
FAILED test_calculate_lifetime_climate_impacts_sensitivity_v2.py::test_calculate_lifetime_climate_impacts_boundary_lifetime - AttributeError: 'module' object at cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity has no attribute 'UPGRADE_COLUMNS'
=================================================== 3 failed, 28 passed, 181 warnings in 7.96s =================================================== 