import pandas as pd
from typing import Optional, Tuple, Dict, List, Union, Any

from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING
from cmu_tare_model.utils.modeling_params import define_scenario_params

def calculate_lifetime_fuel_costs(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    df_baseline_costs: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate lifetime fuel costs for each equipment category.

    This function processes each equipment category over its lifetime, computing annual
    and lifetime fuel costs. Results are combined into two DataFrames:
    a main summary (df_main) and a detailed annual breakdown (df_detailed).

    Args:
        df (pd.DataFrame): Input DataFrame containing equipment consumption data, region info, etc.
        menu_mp (int): Measure package identifier (0 for baseline, nonzero for different scenarios).
        policy_scenario (str): Determines fuel price scenario inputs 
            (e.g., 'No Inflation Reduction Act' or 'AEO2023 Reference Case').
        df_baseline_costs (Optional[pd.DataFrame]): Baseline costs for computing operational savings.
            Default is None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_main: Main summary of lifetime fuel costs (rounded to 2 decimals).
            - df_detailed: Detailed annual and lifetime results (rounded to 2 decimals).

    Raises:
        RuntimeError: If processing fails at the category or year level.
        ValueError: If an invalid policy_scenario is provided.
        KeyError: If required columns are missing from the input DataFrame.
    """
    # Create a copy of the input df
    # Then initialize the detailed dataframe (df_copy will become df_main)
    df_copy = df.copy()
    df_detailed = pd.DataFrame(index=df_copy.index)
    
    # Initialize a dictionary to store lifetime fuel costs columns
    lifetime_columns_data = {}

    try:
        # Validate required columns
        required_columns = ['state', 'census_division']
        missing_columns = [col for col in required_columns if col not in df_copy.columns]
        if missing_columns:
            raise KeyError(f"Required columns missing from input dataframe: {', '.join(missing_columns)}")

        # Determine the scenario prefix and fuel price lookup based on menu_mp and policy_scenario
        # Note: We always use 'central' price assumption (removed sensitivity parameter)
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
        # Use try-except to wrap the entire category's processing
        try:
            print(f"Calculating Fuel Costs from 2024 to {2024 + lifetime} for {category}")
                    
            # Initialize lifetime fuel costs for this category
            lifetime_fuel_costs = pd.Series(0.0, index=df_copy.index)

            # If baseline calculations are required
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
                        is_elec_or_gas=is_elec_or_gas
                    )
                    
                    # Accumulate annual costs
                    lifetime_fuel_costs += annual_cost_value

                    # If there are results, attach them to the detailed DataFrame
                    if annual_costs:
                        df_detailed = pd.concat([df_detailed, pd.DataFrame(annual_costs, index=df_copy.index)], axis=1)
                    
                except KeyError as e:
                    raise RuntimeError(f"Missing fuel price data for year {year_label}, category '{category}': {e}")
                except ValueError as e:
                    raise RuntimeError(f"Invalid data format for year {year_label}, category '{category}': {e}")
                except Exception as e:
                    raise RuntimeError(f"Error processing year {year_label} for category '{category}': {e}")

            # Prepare lifetime columns
            lifetime_dict = {}
            costs_col = f'{scenario_prefix}{category}_lifetime_fuelCost'
            lifetime_dict[costs_col] = lifetime_fuel_costs
                
            # Calculate avoided costs if baseline data is provided
            if menu_mp != 0 and df_baseline_costs is not None:
                baseline_costs_col = f'baseline_{category}_lifetime_fuelCost'
                savings_cost_col = f'{scenario_prefix}{category}_lifetime_savings_fuelCost'

                # Check if baseline column exists before calculating operational savings
                if baseline_costs_col in df_baseline_costs.columns:
                    # Subtract measure package costs from baseline
                    lifetime_dict[savings_cost_col] = df_baseline_costs[baseline_costs_col] - lifetime_dict[costs_col]
                else:
                    # Log warning but continue processing
                    print(f"Warning: Baseline costs column '{baseline_costs_col}' not found. Skipping avoided cost calculation.")

            # Store in global lifetime dictionary
            lifetime_columns_data.update(lifetime_dict)
            # Append these columns to df_detailed for completeness
            df_detailed = pd.concat([df_detailed, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)

        except Exception as e:
            # Convert any exception into a RuntimeError with additional context
            raise RuntimeError(f"Error processing category '{category}': {e}")

    # Create a dataframe for lifetime results and merge with the main dataframe
    df_lifetime = pd.DataFrame(lifetime_columns_data, index=df_copy.index)
    df_main = df_copy.join(df_lifetime, how='left')
    
    # Round final results
    df_main = df_main.round(2)
    df_detailed = df_detailed.round(2)
    
    return df_main, df_detailed

def calculate_annual_fuel_costs(
    df: pd.DataFrame,
    category: str,
    year_label: int,
    menu_mp: int,
    lookup_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]],
    policy_scenario: str,
    scenario_prefix: str,
    is_elec_or_gas: Optional[pd.Series] = None
) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """Calculate annual fuel costs for a given category/year.

    This function looks up fuel prices for the specified region/state and year, 
    and calculates annual costs based on consumption. It uses a temporary column
    for storing per-row price lookups.

    Args:
        df (pd.DataFrame): DataFrame containing consumption data and region info.
        category (str): Equipment category (e.g., 'heating', 'waterHeating').
        year_label (int): The calendar year (e.g., 2024).
        menu_mp (int): Measure package identifier (0 for baseline, nonzero for a measure scenario).
        lookup_fuel_prices (Dict[str, Dict[str, Dict[str, Dict[int, float]]]]): 
            Nested dict with fuel prices for different locations and years.
        policy_scenario (str): The policy scenario to use for fuel price lookups.
        scenario_prefix (str): Prefix for output column naming.
        is_elec_or_gas (Optional[pd.Series]): Boolean mask indicating which rows use state vs. census_division.
            Default is None. Required for baseline (menu_mp=0) calculations.

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
            
            # Determine the consumption column name
            consumption_col = f'baseline_{year_label}_{category}_consumption'
            
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
            
            # Determine the consumption column name
            consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
        
        # Check if the column exists in the DataFrame
        if consumption_col not in df.columns:
            raise ValueError(f"Required consumption column '{consumption_col}' not found")
        
        # Get consumption data from the appropriate column (with null safety)
        consumption = df[consumption_col].fillna(0)
        
        # Calculate fuel costs
        fuel_costs = consumption * df['_temp_price']
        
        # Store the result
        cost_col = f'{scenario_prefix}{year_label}_{category}_fuelCost'
        annual_costs[cost_col] = fuel_costs
        
        # If this is a measure package, calculate savings relative to baseline
        if menu_mp != 0:
            baseline_col = f'baseline_{year_label}_{category}_fuelCost'
            if baseline_col in df.columns:
                savings_col = f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'
                annual_costs[savings_col] = df[baseline_col] - fuel_costs
    
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
