import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

from cmu_tare_model.constants import (
    ANCHOR_YEAR,
    EQUIPMENT_SPECS,
    FUEL_MAPPING,
    VERBOSE,
)
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

from cmu_tare_model.energy_consumption_and_metadata.projected_consumption import (
    build_projected_consumption,
)
from cmu_tare_model.utils.column_names import (
    create_annual_consumption_col,
    create_annual_fuel_consumption_col,
)

def calculate_lifetime_fuel_costs(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    df_baseline_costs: Optional[pd.DataFrame] = None,
    df_consumption: Optional[pd.DataFrame] = None,
    verbose: bool = VERBOSE
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
        policy_scenario: Key into the fuel price table. There is one scenario:
            '2025 Reference Case'.
        df_baseline_costs: Optional DataFrame with baseline costs for computing operational savings.
            Default is None.
        df_consumption: This scenario's table from build_projected_consumption
            (per-fuel, per-year consumption, every component counted). Built
            here from df if not given; pass it in to share one table with the
            emissions step.
        verbose: Whether to print detailed processing information. Default is VERBOSE constant.

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
    menu_mp, policy_scenario = validate_common_parameters(
        menu_mp, policy_scenario)

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
        scenario_prefix, _, _, _, lookup_fuel_prices = define_scenario_params(
            menu_mp=menu_mp,
            policy_scenario=policy_scenario
        )
    except ValueError as e:
        raise ValueError(f"Invalid policy scenario: {policy_scenario}. {str(e)}")
    except KeyError as e:
        raise KeyError(f"Missing required data: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error configuring scenario parameters: {str(e)}")

    # One consumption table for every year and category: the single source
    # of the energy use priced below.
    if df_consumption is None:
        df_consumption = build_projected_consumption(df_copy, menu_mp, verbose=verbose)
    if not df_consumption.index.equals(df_copy.index):
        raise ValueError(
            "df_consumption must have the same homes, in the same order, as df.")

    # Loop over each equipment category and its lifetime
    for category, lifetime in EQUIPMENT_SPECS.items():
        try:
            # The cost stream runs for `lifetime` years starting at ANCHOR_YEAR,
            # so the last year is ANCHOR_YEAR + lifetime - 1 (2025-2039 for a
            # 15-year lifetime). The old message added a year to the end of the
            # span and reported 16 years for a 15-year stream.
            if verbose:
                last_year = ANCHOR_YEAR + lifetime - 1
                print(
                    f"Calculating Fuel Costs from {ANCHOR_YEAR} to "
                    f"{last_year} for {category}")
            
            # ===== STEP 1: Initialize validation tracking =====
            # MEMORY OPTIMIZATION: copy=False since df_copy was already copied at the start
            _, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                df_copy, category, menu_mp, verbose=verbose, copy=False)
            
            # Check if all homes are invalid for this category
            if not valid_mask.any():
                if verbose:
                    raise ValueError(f"Warning: All homes are invalid for category '{category}'. Results will be all NaN.")
                
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

            # ===== STEP 3 & 4: Valid-Only Calculation and Updates =====
            # Loop over each year in the equipment's lifetime
            for year in range(1, lifetime + 1):
                # Year 1 of the stream is ANCHOR_YEAR itself, so a 15-year
                # lifetime runs 2025-2039. Derived from the shared constant so
                # the start year is set in exactly one place.
                year_label = year + (ANCHOR_YEAR - 1)


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
                        df_consumption=df_consumption,
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

                        # Carry the baseline consumption across too, so one
                        # measure-package file holds both the baseline
                        # consumption and the retrofit consumption. A reader
                        # needs both, each priced at its own fuel price; the
                        # measure-package run only computes the retrofit
                        # consumption.
                        baseline_consumption_col = create_annual_consumption_col(
                            'baseline_', year_label, category)
                        if baseline_consumption_col in df_baseline_costs.columns:
                            annual_costs[baseline_consumption_col] = (
                                df_baseline_costs[baseline_consumption_col])
                            category_columns_to_mask.append(baseline_consumption_col)
                        for fuel in FUEL_MAPPING.values():
                            baseline_fuel_col = create_annual_fuel_consumption_col(
                                'baseline_', year_label, category, fuel)
                            if baseline_fuel_col in df_baseline_costs.columns:
                                annual_costs[baseline_fuel_col] = (
                                    df_baseline_costs[baseline_fuel_col])
                                category_columns_to_mask.append(baseline_fuel_col)

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

                    # Average-annual framing of the same lifetime costs, for the
                    # manuscript operating-cost figure. Each lifetime cost is
                    # spread evenly over the equipment lifetime (years), so the
                    # average-annual value is just the lifetime cost / lifetime.
                    # These are additive reporting columns: the lifetime columns
                    # and every value derived from them are unchanged. Because the
                    # /lifetime factor cancels in the percent change, the
                    # avg-annual percent change equals the lifetime percent change
                    # exactly -- the map that reads it renders the same numbers,
                    # only the framing (and colorbar wording) changes.
                    baseline_annual = (
                        df_baseline_costs[baseline_costs_col] / lifetime
                    )
                    retrofit_annual = lifetime_fuel_costs / lifetime
                    baseline_annual_col = (
                        f'baseline_{category}_avg_annual_fuel_cost'
                    )
                    retrofit_annual_col = (
                        f'{scenario_prefix}{category}_avg_annual_fuel_cost'
                    )
                    annual_pct_change_col = (
                        f'{scenario_prefix}{category}'
                        f'_avg_annual_fuel_cost_pct_change'
                    )
                    lifetime_dict[baseline_annual_col] = baseline_annual
                    lifetime_dict[retrofit_annual_col] = retrofit_annual
                    # Guard a zero or negative baseline to NaN so those homes are
                    # excluded rather than producing infinite or misleading
                    # percent changes.
                    lifetime_dict[annual_pct_change_col] = (
                        (retrofit_annual - baseline_annual)
                        / baseline_annual * 100
                    ).where(baseline_annual > 0)
                    category_columns_to_mask.extend([
                        baseline_annual_col,
                        retrofit_annual_col,
                        annual_pct_change_col,
                    ])

                    # Cooling-only diagnostic flag (does NOT enter NPV or masking).
                    # For some homes the heat pump's cooling energy exceeds the
                    # baseline air conditioner's, so cooling operating savings go
                    # negative. This is a real ResStock result: most such homes
                    # have a baseline room AC that cools one room, while the heat
                    # pump cools the whole house, so cooling kWh legitimately
                    # rises. The negative savings stay in the NPV as a real
                    # operating cost; this boolean only marks the affected homes
                    # so the phenomenon can be reported. True means a valid
                    # cooling home whose lifetime cooling savings is below zero.
                    # Excluded homes have NaN savings, which compare False here,
                    # so they stay False. Left out of category_columns_to_mask on
                    # purpose so it is never masked to NaN and stays boolean.
                    if category == 'cooling':
                        negative_flag_col = (
                            f'{scenario_prefix}{category}_lifetime_savings_negative'
                        )
                        lifetime_dict[negative_flag_col] = (
                            lifetime_dict[savings_cost_col] < 0
                        )
                elif verbose:
                    raise ValueError(f"Warning: Baseline costs column '{baseline_costs_col}' not found. Skipping avoided cost calculation.")

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


# ====== Helper Functions ======
def _lookup_annual_fuel_price(
    lookup_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]],
    region: str,
    fuel_type: object,
    policy_scenario: str,
    year_label: int
) -> float:
    """Find one home's fuel price for one year, or fail loudly.

    The price table is nested region -> fuel -> policy scenario -> year. This
    walks those four levels and treats two kinds of miss differently, because
    they mean different things:

    - An unmapped FUEL returns NaN. Roughly 10,000 ResStock homes burn
      'Other Fuel' or have no heating fuel recorded, so FUEL_MAPPING leaves
      them blank. That is a real condition in the source data, not a mistake
      in the model, and every one of those homes is already excluded from
      results by include_heating. NaN is used rather than 0 because a zero
      price would silently pull down any average it reached, while a NaN
      cannot hide in one.
    - A missing REGION, POLICY SCENARIO, or YEAR raises. If the fuel resolved,
      the price table was meant to carry that combination, so a miss means the
      model asked for something the data does not contain -- most likely a
      year outside ANCHOR_YEAR through the end of the projection file.

    Args:
        lookup_fuel_prices: Nested price table, in USD per kWh.
        region: State abbreviation for electricity and natural gas, census
            division name for fuel oil and propane.
        fuel_type: Mapped fuel name ('electricity', 'naturalGas', 'fuelOil',
            'propane'), or a blank value for a fuel the model does not price.
        policy_scenario: Policy scenario key, e.g. '2025 Reference Case'.
        year_label: Calendar year to price.

    Returns:
        Price in USD per kWh, or NaN when the home's fuel is not one the model
        prices.

    Raises:
        KeyError: If the region, policy scenario, or year is missing for a fuel
            the model does price.
    """
    # Blank fuel: an excluded home, not a data problem. See docstring.
    if not isinstance(fuel_type, str):
        return np.nan

    region_prices = lookup_fuel_prices.get(region)
    if region_prices is None:
        raise KeyError(
            f"No fuel prices for region '{region}' "
            f"(fuel '{fuel_type}', scenario '{policy_scenario}', "
            f"year {year_label}).")

    fuel_prices = region_prices.get(fuel_type)
    if fuel_prices is None:
        raise KeyError(
            f"No '{fuel_type}' prices for region '{region}' "
            f"(scenario '{policy_scenario}', year {year_label}).")

    scenario_prices = fuel_prices.get(policy_scenario)
    if scenario_prices is None:
        raise KeyError(
            f"No prices under policy scenario '{policy_scenario}' for region "
            f"'{region}', fuel '{fuel_type}' (year {year_label}). "
            f"Available scenarios: {sorted(fuel_prices)}.")

    if year_label not in scenario_prices:
        available = sorted(scenario_prices)
        raise KeyError(
            f"No fuel price for year {year_label} -- region '{region}', "
            f"fuel '{fuel_type}', scenario '{policy_scenario}'. "
            f"The price data covers {available[0]}-{available[-1]}.")

    return scenario_prices[year_label]


def _annual_fuel_price_series(
    df: pd.DataFrame,
    lookup_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]],
    fuel_type: str,
    policy_scenario: str,
    year_label: int,
    uses_fuel: pd.Series,
) -> pd.Series:
    """Per-home price of one fuel in one year, for the homes that use it.

    Electricity and natural gas are priced by state; fuel oil and propane by
    census division. Each region is looked up once, then mapped onto homes.

    Args:
        df: DataFrame with 'state' and 'census_division'.
        lookup_fuel_prices: Nested price table, in USD per kWh.
        fuel_type: 'electricity', 'naturalGas', 'fuelOil', or 'propane'.
        policy_scenario: Policy scenario key, e.g. '2025 Reference Case'.
        year_label: Calendar year to price.
        uses_fuel: True for homes with nonzero use of this fuel; only their
            regions are looked up, so a fuel nobody in a region uses cannot
            raise a missing-price error.

    Returns:
        Price per home in USD per kWh; NaN for homes that do not use the fuel.

    Raises:
        KeyError: If a region with a home using the fuel has no price.
    """
    region_col = 'state' if fuel_type in ('electricity', 'naturalGas') else 'census_division'
    regions = df.loc[uses_fuel, region_col].unique()
    price_by_region = {
        region: _lookup_annual_fuel_price(
            lookup_fuel_prices, region, fuel_type, policy_scenario, year_label)
        for region in regions
    }
    return df[region_col].map(price_by_region).where(uses_fuel)


def calculate_annual_fuel_costs(
    df: pd.DataFrame,
    category: str,
    year_label: int,
    menu_mp: int,
    lookup_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]],
    policy_scenario: str,
    scenario_prefix: str,
    df_consumption: pd.DataFrame,
    valid_mask: Optional[pd.Series] = None,
    verbose: bool = VERBOSE
) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """
    Calculate annual fuel costs for a given category/year.

    Reads this scenario's per-fuel consumption from df_consumption (every
    component counted: primary energy, fans and pumps, heat-pump backup) and
    prices each fuel at its own price, so a home that uses several fuels pays
    each at the right rate.

    Args:
        df: DataFrame with region info ('state', 'census_division').
        category: Equipment category (e.g., 'heating', 'cooling').
        year_label: The calendar year (e.g., 2025).
        menu_mp: Measure package identifier (0 for baseline, nonzero for a measure scenario).
        lookup_fuel_prices: Nested dict with fuel prices for different locations and years.
        policy_scenario: The policy scenario to use for fuel price lookups.
        scenario_prefix: Prefix for output column naming.
        df_consumption: This scenario's table from build_projected_consumption,
            indexed like df.
        valid_mask: Boolean Series indicating which homes have valid data.
            Default is None. If provided, will be used for masking calculations.
        verbose: Whether to print detailed processing information. Default is VERBOSE.

    Returns:
        Tuple[Dict[str, pd.Series], pd.Series]:
            - Dict[str, pd.Series]: Annual cost and consumption columns, keyed by output column names.
            - pd.Series: Annual fuel costs (for aggregation).

    Raises:
        KeyError: If fuel prices for a specific region/year are missing.
        ValueError: If 'state' or 'census_division' is missing, or df_consumption
            has no consumption columns for this scenario, category and year.
    """
    # Results dictionaries (no rounding here)
    annual_costs = {}

    for col in ('state', 'census_division'):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found")

    # Price each fuel at its own price and add them up. A home that uses
    # several fuels (a gas furnace's electric blower, a dual-fuel heat pump's
    # gas backup) pays each at the right rate.
    fuel_costs = pd.Series(0.0, index=df.index)
    fuels_found = 0
    for fuel in FUEL_MAPPING.values():
        fuel_col = create_annual_fuel_consumption_col(
            scenario_prefix, year_label, category, fuel)
        if fuel_col not in df_consumption.columns:
            continue
        fuels_found += 1
        fuel_use = df_consumption[fuel_col].fillna(0.0)
        uses_fuel = fuel_use > 0
        price = _annual_fuel_price_series(
            df, lookup_fuel_prices, fuel, policy_scenario, year_label, uses_fuel)
        fuel_costs += (fuel_use * price).where(uses_fuel, 0.0)
        # Keep each fuel's consumption next to the cost it produced.
        annual_costs[fuel_col] = df_consumption[fuel_col]

    # ValueError, not KeyError: a missing table column is a setup mistake and
    # must not be mistaken for the missing-data case callers skip.
    total_col = create_annual_consumption_col(scenario_prefix, year_label, category)
    if fuels_found == 0 or total_col not in df_consumption.columns:
        raise ValueError(
            f"df_consumption has no '{scenario_prefix}' consumption columns for "
            f"{category} in {year_label}; build it with build_projected_consumption "
            f"for menu_mp={menu_mp}.")
    consumption = df_consumption[total_col].fillna(0.0)

    # Homes outside the calculation are NaN, not 0.
    if valid_mask is not None and not valid_mask.all():
        consumption = consumption.copy()
        consumption.loc[~valid_mask] = np.nan
        fuel_costs.loc[~valid_mask] = np.nan

    # Store the result
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuel_cost'
    annual_costs[cost_col] = fuel_costs

    # Keep the projected consumption that produced this cost, so a reader can
    # check the dollars against a fuel price. Stored in the same dictionary as
    # the cost column, so the caller applies the same masking to both.
    annual_costs[total_col] = consumption

    return annual_costs, fuel_costs
