import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.validation_framework import (
    apply_final_masking,
    create_retrofit_only_series,
    calculate_avoided_values,
    initialize_validation_tracking
)
from cmu_tare_model.utils.calculation_utils import (
    validate_common_parameters,
    apply_temporary_validation_and_mask
)

from cmu_tare_model.utils.hdd_consumption_utils import (
    get_hdd_adjusted_consumption
)

def calculate_lifetime_fuel_costs(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    df_baseline_costs: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate lifetime fuel costs for each equipment category.

    This function processes each equipment category over its lifetime, computing annual
    and lifetime fuel costs. Results are combined into two DataFrames:
    a main summary (df_main) and a detailed annual breakdown (df_detailed).
    
    This function follows the five-step validation framework:
    1. Mask Initialization: Identifies valid homes using inclusion flags and retrofit status
    2. Series Initialization: Creates result series with zeros for valid homes, NaN for others
    3. Valid-Only Calculation: Performs calculations only for valid homes
    4. Valid-Only Updates: Uses list-based collection of yearly values instead of incremental updates
    5. Final Masking: Applies consistent masking to all result columns

    Args:
        df: Input DataFrame containing equipment consumption data, region info, etc.
        menu_mp: Measure package identifier (0 for baseline, nonzero for different scenarios).
        policy_scenario: Determines fuel price scenario inputs 
            (e.g., 'No Inflation Reduction Act' or 'AEO2023 Reference Case').
        df_baseline_costs: Optional DataFrame with baseline costs for computing operational savings.
            Default is None.
        verbose: Whether to print detailed processing information. Default is True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_main: Main summary of lifetime fuel costs (rounded to 2 decimals).
            - df_detailed: Detailed annual and lifetime results (rounded to 2 decimals).

    Raises:
        RuntimeError: If processing fails at the category or year level.
        ValueError: If an invalid policy_scenario is provided.
        KeyError: If required columns are missing from the input DataFrame.
    """
    # Handle empty DataFrames gracefully
    if df.empty:
        if verbose:
            print("Warning: Empty DataFrame provided. Returning empty results.")
        # Return empty DataFrames with the same structure expected by callers
        return pd.DataFrame(), pd.DataFrame()
    
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

    # Initialize a dictionary to store lifetime fuel costs columns
    lifetime_columns_data = {}

    # Initialize dictionary to track columns for masking verification by category
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

    try:
        # Validate required columns
        required_columns = ['state', 'census_division']
        missing_columns = [col for col in required_columns if col not in df_copy.columns]
        if missing_columns:
            raise KeyError(f"Required columns missing from input dataframe: {', '.join(missing_columns)}")

        # Determine the scenario prefix and fuel price lookup based on menu_mp and policy_scenario
        scenario_prefix, _, _, _, _, lookup_fuel_prices = define_scenario_params(
            menu_mp=menu_mp,
            policy_scenario=policy_scenario
        )
    except ValueError as e:
        raise ValueError(f"Invalid policy scenario: {policy_scenario}. {str(e)}")
    except KeyError as e:
        raise KeyError(f"Missing required data: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error configuring scenario parameters: {str(e)}")

    # Loop over each equipment category and its lifetime
    for category, lifetime in EQUIPMENT_SPECS.items():
        try:
            if verbose:
                print(f"Calculating Fuel Costs from 2024 to {2024 + lifetime} for {category}")
            
            # ===== STEP 1: Initialize validation tracking =====
            df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                df_copy, category, menu_mp, verbose=verbose)
            
            # Check if all homes are invalid for this category
            if not valid_mask.any():
                if verbose:
                    print(f"Warning: All homes are invalid for category '{category}'. Results will be all NaN.")
                
                # Create NaN columns for lifetime fuel costs
                costs_col = f'{scenario_prefix}{category}_lifetime_fuel_cost'
                lifetime_fuel_costs = pd.Series(np.nan, index=df_copy.index)
                lifetime_dict = {costs_col: lifetime_fuel_costs}
                
                # Add savings column for measure packages
                if menu_mp != 0 and df_baseline_costs is not None:
                    baseline_costs_col = f'baseline_{category}_lifetime_fuel_cost'
                    if baseline_costs_col in df_baseline_costs.columns:
                        savings_cost_col = f'{scenario_prefix}{category}_lifetime_savings_fuel_cost'
                        lifetime_dict[savings_cost_col] = pd.Series(np.nan, index=df_copy.index)
                        lifetime_dict[baseline_costs_col] = df_baseline_costs[baseline_costs_col]
                        
                        # Track columns for masking
                        category_columns_to_mask.extend([costs_col, savings_cost_col, baseline_costs_col])
                    
                # Update lifetime columns data
                lifetime_columns_data.update(lifetime_dict)
                
                # Add columns to detailed DataFrame
                lifetime_df = pd.DataFrame(lifetime_dict, index=df_copy.index)
                df_detailed = pd.concat([df_detailed, lifetime_df], axis=1)
                
                # Add all columns for this category to the masking dictionary
                all_columns_to_mask[category].extend(category_columns_to_mask)
                
                # Skip further processing for this category
                continue
            
            # ===== STEP 2: Initialize result series with template =====
            # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
            fuel_costs_template = create_retrofit_only_series(df_copy, valid_mask)
            
            # Create a list to store yearly costs (instead of incrementally updating a single Series)
            yearly_costs_list = []

            # If baseline calculations are required, set up fuel type mapping
            if menu_mp == 0:
                # Validate fuel type column exists
                fuel_col = f'base_{category}_fuel'
                if fuel_col not in df_copy.columns:
                    raise KeyError(f"Required column '{fuel_col}' not found in dataframe")
                
                # Map each baseline fuel to its lower-case version
                df_copy[f'fuel_type_{category}'] = df_copy[fuel_col].map(FUEL_MAPPING)
                
                # Create a boolean mask indicating which rows use 'state' vs. 'census_division'
                # This differentiates between electricity/gas (state-based) and other fuels (census division-based)
                is_elec_or_gas = df_copy[f'fuel_type_{category}'].isin(['electricity', 'naturalGas'])
            else:
                is_elec_or_gas = None

            # ===== STEP 3 & 4: Valid-Only Calculation and Updates =====
            # Loop over each year in the equipment's lifetime
            for year in range(1, lifetime + 1):
                year_label = year + 2023
                
                try:
                    # Calculate the annual fuel costs for this category and year
                    annual_costs, annual_cost_value = calculate_annual_fuel_costs(
                        df=df_copy,
                        category=category,
                        year_label=year_label,
                        menu_mp=menu_mp,
                        lookup_fuel_prices=lookup_fuel_prices,
                        policy_scenario=policy_scenario,
                        scenario_prefix=scenario_prefix,
                        is_elec_or_gas=is_elec_or_gas,
                        valid_mask=valid_mask,  # Pass the valid mask for proper masking
                        verbose=verbose  # Pass verbose to show warnings
                    )
                    
                    # Skip year if no data was returned (empty dictionary)
                    if not annual_costs and annual_cost_value.sum() == 0:
                        if verbose:
                            print(f"  Skipping year {year_label} due to missing data")
                        continue
                    
                    # Apply validation mask to annual costs (for measure packages)
                    if menu_mp != 0:
                        annual_cost_values = annual_cost_value.copy()
                        annual_cost_values.loc[~valid_mask] = np.nan  # Changed from 0.0 to np.nan
                    else:
                        annual_cost_values = annual_cost_value

                    # Add to list (instead of incrementally updating)
                    yearly_costs_list.append(annual_cost_values)

                    # If baseline costs are provided, include baseline annual costs for reference only
                    if menu_mp != 0 and df_baseline_costs is not None:
                        baseline_col = f'baseline_{year_label}_{category}_fuel_cost'
                        if baseline_col in df_baseline_costs.columns:
                            annual_costs[baseline_col] = df_baseline_costs[baseline_col]
                            category_columns_to_mask.append(baseline_col)

                    # Add annual costs to detailed DataFrame
                    if annual_costs:
                        annual_df = pd.DataFrame(annual_costs, index=df_copy.index)
                        df_detailed = pd.concat([df_detailed, annual_df], axis=1)
                        category_columns_to_mask.extend(annual_costs.keys())

                except KeyError as e:
                    raise RuntimeError(f"Missing fuel price data for year {year_label}, category '{category}': {e}")
                except ValueError as e:
                    raise RuntimeError(f"Invalid data format for year {year_label}, category '{category}': {e}")
                except Exception as e:
                    raise RuntimeError(f"Error processing year {year_label} for category '{category}': {e}")

            # Calculate lifetime fuel costs using vectorized operations
            if yearly_costs_list:
                # Convert list to DataFrame and sum
                costs_df = pd.concat(yearly_costs_list, axis=1)
                # Use skipna=False to ensure proper NaN propagation
                lifetime_fuel_costs = costs_df.sum(axis=1, skipna=False)
                
                # Ensure explicit masking for invalid homes
                if menu_mp != 0:
                    lifetime_fuel_costs = pd.Series(
                        np.where(valid_mask, lifetime_fuel_costs, np.nan),
                        index=lifetime_fuel_costs.index
                    )
            else:
                # Use template if no costs were calculated
                lifetime_fuel_costs = fuel_costs_template

            # Prepare lifetime columns
            lifetime_dict = {}
            costs_col = f'{scenario_prefix}{category}_lifetime_fuel_cost'
            lifetime_dict[costs_col] = lifetime_fuel_costs
            category_columns_to_mask.append(costs_col)
                
            # Calculate avoided costs at lifetime level if baseline data is provided
            # This is consistent with how climate and health modules handle avoided calculations
            if menu_mp != 0 and df_baseline_costs is not None:
                baseline_costs_col = f'baseline_{category}_lifetime_fuel_cost'
                if baseline_costs_col in df_baseline_costs.columns:
                    savings_cost_col = f'{scenario_prefix}{category}_lifetime_savings_fuel_cost'

                    # Use calculate_avoided_values function for consistency with climate and health modules
                    lifetime_dict[savings_cost_col] = calculate_avoided_values(
                        baseline_values=df_baseline_costs[baseline_costs_col],
                        measure_values=lifetime_fuel_costs,
                        retrofit_mask=valid_mask
                    )
                    
                    # Add baseline column to results for reference
                    lifetime_dict[baseline_costs_col] = df_baseline_costs[baseline_costs_col]
                    
                    # Track columns for masking
                    category_columns_to_mask.extend([baseline_costs_col, savings_cost_col])
                elif verbose:
                    print(f"Warning: Baseline costs column '{baseline_costs_col}' not found. Skipping avoided cost calculation.")

            # Store in global lifetime dictionary and add to detailed DataFrame
            lifetime_columns_data.update(lifetime_dict)
            lifetime_df = pd.DataFrame(lifetime_dict, index=df_copy.index)
            df_detailed = pd.concat([df_detailed, lifetime_df], axis=1)

            # Add all columns for this category to the masking dictionary
            all_columns_to_mask[category].extend(category_columns_to_mask)

        except Exception as e:
            # Convert any exception into a RuntimeError with additional context
            raise RuntimeError(f"Error processing category '{category}': {e}")

    # Create a dataframe for lifetime results and merge with the main dataframe
    df_lifetime = pd.DataFrame(lifetime_columns_data, index=df_copy.index)
    
    # Before applying final masking, ensure all lifetime columns are tracked
    for category in EQUIPMENT_SPECS.keys():
        lifetime_col = f'{scenario_prefix}{category}_lifetime_fuel_cost'
        if lifetime_col in df_lifetime.columns:
            if lifetime_col not in all_columns_to_mask[category]:
                all_columns_to_mask[category].append(lifetime_col)

    # ===== STEP 5: Apply final masking =====
    # Use apply_temporary_validation_and_mask utility for df_main
    df_main = apply_temporary_validation_and_mask(df_copy, df_lifetime, all_columns_to_mask, verbose=verbose)

    # Use apply_final_masking for df_detailed
    df_detailed = apply_final_masking(df_detailed, all_columns_to_mask, verbose=verbose)
    
    # Round final results
    df_main = df_main.round(2)
    df_detailed = df_detailed.round(2)
    
    return df_main, df_detailed


# ====== Helper Function ======
def calculate_annual_fuel_costs(
    df: pd.DataFrame,
    category: str,
    year_label: int,
    menu_mp: int,
    lookup_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]],
    policy_scenario: str,
    scenario_prefix: str,
    is_elec_or_gas: Optional[pd.Series] = None,
    valid_mask: Optional[pd.Series] = None,
    verbose: bool = False
) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """
    Calculate annual fuel costs for a given category/year.

    This function looks up fuel prices for the specified region/state and year, 
    and calculates annual costs based on consumption. It uses a temporary column
    for storing per-row price lookups.

    Args:
        df: DataFrame containing consumption data and region info.
        category: Equipment category (e.g., 'heating', 'waterHeating').
        year_label: The calendar year (e.g., 2024).
        menu_mp: Measure package identifier (0 for baseline, nonzero for a measure scenario).
        lookup_fuel_prices: Nested dict with fuel prices for different locations and years.
        policy_scenario: The policy scenario to use for fuel price lookups.
        scenario_prefix: Prefix for output column naming.
        is_elec_or_gas: Boolean mask indicating which rows use state vs. census_division.
            Default is None. Required for baseline (menu_mp=0) calculations.
        valid_mask: Boolean Series indicating which homes have valid data.
            Default is None. If provided, will be used for masking calculations.
        verbose: Whether to print detailed processing information. Default is False.

    Returns:
        Tuple[Dict[str, pd.Series], pd.Series]:
            - Dict[str, pd.Series]: Annual columns of fuel costs, keyed by output column names.
            - pd.Series: Annual fuel costs (for aggregation).

    Raises:
        KeyError: If fuel prices for a specific region/year are missing.
        ValueError: If the required consumption column does not exist in the DataFrame
                   or if is_elec_or_gas mask is missing for baseline calculations.
    """
    # Results dictionaries (no rounding here)
    annual_costs = {}
    
    try:
        if menu_mp == 0:
            # For baseline, look up the appropriate price based on fuel type
            # Required fields check
            if is_elec_or_gas is None:
                raise ValueError("is_elec_or_gas mask is required for baseline calculations")
            
            if 'state' not in df.columns or 'census_division' not in df.columns:
                raise ValueError("Required columns 'state' and 'census_division' not found")
                
            if f'fuel_type_{category}' not in df.columns:
                raise ValueError(f"Required column 'fuel_type_{category}' not found")
            
            # Build a list/Series of per-row prices using a dictionary lookup based on state/census_division
            df['_temp_price'] = [
                lookup_fuel_prices
                    .get(
                        # Use 'state' if electric/natural gas, else use 'census_division'
                        state_val if use_state else cdiv_val,
                        {}
                    )
                    .get(fueltype_val, {})
                    .get(policy_scenario, {})
                    .get(year_label, 0)
                    
                for state_val, cdiv_val, fueltype_val, use_state in zip(
                    df['state'],
                    df['census_division'],
                    df[f'fuel_type_{category}'],
                    is_elec_or_gas
                )
            ]

            # Consumption now comes from the get_hdd_adjusted_consumption function

        else:
            # For measure packages, everything is mapped to electricity (via 'state')
            if 'state' not in df.columns:
                raise ValueError("Required column 'state' not found")
                
            df['_temp_price'] = [
                lookup_fuel_prices
                    .get(state_name, {})
                    .get('electricity', {})
                    .get(policy_scenario, {})
                    .get(year_label, 0)
                for state_name in df['state']
            ]

            # Consumption now comes from the get_hdd_adjusted_consumption function
                            
        # Get consumption data from the appropriate column (with null safety)
        consumption = get_hdd_adjusted_consumption(
            df=df,
            category=category,
            year_label=year_label,
            menu_mp=menu_mp
        ).fillna(0)

        # ===== STEP 3: Valid-Only Calculation =====
        # Apply valid mask if provided
        if valid_mask is not None and not valid_mask.all():
            # Make a copy to avoid modifying the original Series
            consumption = consumption.copy()
            # Set values to NaN for invalid homes (not zero)
            consumption.loc[~valid_mask] = np.nan  # Changed from 0.0 to np.nan

        # Calculate fuel costs
        fuel_costs = consumption * df['_temp_price']
        
        # Store the result
        cost_col = f'{scenario_prefix}{year_label}_{category}_fuel_cost'
        annual_costs[cost_col] = fuel_costs
        
        # Note: No baseline or avoided cost calculations here - consistent with climate and health modules
    
    except KeyError as e:
        if "consumption" in str(e):
            # More informative error for missing consumption columns
            if verbose:
                print(f"Warning: Missing consumption data for year {year_label}, category '{category}': {e}")
            return {}, pd.Series(0, index=df.index)
        else:
            # Re-raise other KeyErrors
            raise
    except Exception as e:
        # Re-raise to preserve the exception stack
        raise
    
    finally:
        # Drop the temporary column after processing (using try to ensure this happens even on error)
        try:
            if '_temp_price' in df.columns:
                df.drop(columns=['_temp_price'], inplace=True)
        except:
            # Ignore errors in cleanup
            pass
    
    return annual_costs, fuel_costs
