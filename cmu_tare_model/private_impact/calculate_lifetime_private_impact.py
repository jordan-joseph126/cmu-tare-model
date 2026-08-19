# Updated
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from cmu_tare_model.constants import (
    ANCHOR_YEAR,
    EQUIPMENT_SPECS,
    PRIVATE_DISCOUNTING_METHOD_SUFFIXES,
    REBATE_ELIGIBLE_HEATING_MPS,
    REBATE_GUIDANCE_JUNE2026,
    REMDB_COST_SCENARIO_KEYS,
)
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.discounting import calculate_discount_factors
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    calculate_avoided_values,
    initialize_validation_tracking,
    replace_small_values_with_nan
)
from cmu_tare_model.utils.calculation_utils import (
    validate_common_parameters,
    apply_temporary_validation_and_mask
)
from cmu_tare_model.utils.column_names import (
    create_fuel_cost_col,
    create_cost_col,
    create_rebate_col,
    create_capital_col,
    create_npv_case_col,
    create_discounted_savings_col,
    create_cooling_credit_applied_col,
    create_enclosure_cost_col,
    create_weatherization_rebate_col,
)

"""
========================================================================================================================================================================
OVERVIEW: CALCULATE LIFETIME PRIVATE IMPACTS
========================================================================================================================================================================
This module calculates the private net present value (NPV) for various equipment categories,
considering different cost assumptions and potential IRA rebates.

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
# UPDATED APRIL 29, 2025 @ 10:30 AM - ADDED DATA VALIDATION CHECKS AND ERROR HANDLING
    def calculate_private_NPV:
        - Added comprehensive data validation checks and error handling for the function.
        - Applied a validation mask to ensure that only valid homes are considered in calculations.
        - Applied final verification masking to ensure data consistency before returning the result.
    def calculate_capital_costs:
        - Modified the function to accept a validation mask and applying masking to costs so that only valid homes have values.
# UPDATED JANUARY 23, 2026
    1.Add defensive column validation in calculate_capital_costs().
    2. Add _validate_required_columns() helper function and early column existence
    checking in calculate_capital_costs() to prevent KeyError when required
    installation cost columns are missing (e.g., mp10_heating_upgrade_installed_cost).

    The fix:
    - Builds a list of required columns based on category and policy scenario
    - Validates all required columns exist before accessing them
    - Raises a clear, actionable KeyError with the list of missing columns
    and guidance to ensure installation costs are calculated first

    This prevents cryptic KeyError messages and helps debug data pipeline
    issues where installation cost columns weren't created upstream.
# UPDATED JUNE 26, 2026
    1. Removed references to create_installation_premium_col() in calculate_capital_costs() and calculate_private_npv().
    2. Removed the v3 cost scenario from REMDB_COST_SCENARIO_KEYS in constants.py and all references in this file.
"""

# ========================================================================================================================================================================
# LIFETIME PRIVATE IMPACT: NPV OF CAPITAL COST INVESTMENT AND LIFETIME FUEL COSTS
# ========================================================================================================================================================================

def calculate_private_npv(
        df: pd.DataFrame,
        df_fuel_costs: pd.DataFrame,
        df_baseline_costs: pd.DataFrame,
        input_mp: str,
        menu_mp: int,
        policy_scenario: str,
        discount_rate_col_name: str,
        cost_scenario: str = 'v4MID',
        verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate private net present value (NPV) for the three heat-pump NPV cases.

    A single heat pump replaces the home's heating system and also serves the
    cooling load. All three cases count both the heating and cooling operating
    (energy-bill) savings; they differ only in which avoided-replacement credit
    reduces the net capital cost (see NPV_CASE_CATEGORIES):

      - heatingSavings_coolingLCC: credit avoided cooling replacement only
      - heatingLCC_coolingSavings: credit avoided heating replacement only
      - heatingLCC_coolingLCC:     credit both avoided replacements

    Cooling savings and the cooling replacement credit are zero for homes with
    no AC (include_cooling = False). For those homes heatingLCC_coolingLCC ==
    heatingLCC_coolingSavings, and heatingSavings_coolingLCC carries no credit.

    The NPV is the lifetime savings minus the incremental (net) capital cost of
    the heat pump over a like-for-like replacement. A single willingness-to-pay
    framing is modeled, so the NPV column name carries no WTP token, and the
    economic adoption decision adopts when this NPV >= 0.

    Args:
        df: Input DataFrame with installation costs and validation flags.
            IMPORTANT: Must contain the discount rate columns created by
            prepare_discount_rates().
        df_fuel_costs: DataFrame containing measure-package annual fuel costs.
        df_baseline_costs: DataFrame containing baseline annual fuel costs.
        input_mp: Upgrade label used for cost column selection (e.g., 'upgrade03').
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario for electricity grid projections.
            Accepted value: '2025 Reference Case'.
        discount_rate_col_name: Discount rate column name for private discounting.
        cost_scenario: Cost scenario identifier for column naming. Supported
            values: 'v4LOW', 'v4MID' (default), 'v4HIGH'.
        verbose: Whether to print detailed processing information. Default is True.

    Note:
        The cost stream always starts at ANCHOR_YEAR (2025), which is where the
        fuel-price and degree-day data begin, and that is also the year all
        future dollars are discounted back to. This is deliberately not a
        parameter: a caller passing a different start year would silently price
        the retrofit over years the projection data does not cover.

    Returns:
        DataFrame with, per measure package, one private NPV column and a net
        capital cost column for each of the three NPV cases, plus the shared
        gross capital cost column (see NPV_CASE_CATEGORIES).

    Raises:
        ValueError: If policy_scenario, menu_mp, or cost_scenario is invalid.
        KeyError: If a required cost or fuel-cost column is missing.
    """
    # ===== STEP 0: Validate input parameters =====
    menu_mp, policy_scenario = validate_common_parameters(
        menu_mp, policy_scenario)

    if cost_scenario not in REMDB_COST_SCENARIO_KEYS:
        raise ValueError(
            f"Invalid cost_scenario: '{cost_scenario}'. "
            f"Must be one of {REMDB_COST_SCENARIO_KEYS}")

    if verbose:
        print(
            f"\nCalculating Private NPV (three cases) | "
            f"input_mp={input_mp}, menu_mp={menu_mp}, "
            f"policy_scenario={policy_scenario}")

    # Create copies to avoid modifying original dataframes
    df_copy = df.copy()
    df_fuel_costs_copy = df_fuel_costs.copy()
    df_baseline_costs_copy = df_baseline_costs.copy()
    df_new_columns = pd.DataFrame(index=df_copy.index)

    # Copy inclusion flags and validation columns from df_copy to df_detailed
    validation_prefixes = ["include_", "valid_tech_", "valid_fuel_"]
    validation_cols = []
    for prefix in validation_prefixes:
        validation_cols.extend([col for col in df_copy.columns if col.startswith(prefix)])
        
    for col in validation_cols:
        if col not in df_fuel_costs_copy.columns:
            df_fuel_costs_copy[col] = df_copy[col]
        if col not in df_baseline_costs_copy.columns:
            df_baseline_costs_copy[col] = df_copy[col]

    # Initialize dictionary to track columns for masking verification
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

    # Determine the scenario prefix based on the policy scenario
    scenario_prefix, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)

    # Calculate the maximum lifetime across all equipment to determine how many years to pre-calculate
    max_lifetime = max(EQUIPMENT_SPECS.values())

    # Pre-calculate discount factors for each year to avoid redundant calculations
    # This maps from year_label to its discount factor. Series not scalar.
    discount_factors: Dict[int, pd.Series] = {}

    for year in range(1, max_lifetime + 1):
        # Year 1 of the stream is ANCHOR_YEAR itself, so a 15-year lifetime
        # runs 2025-2039 and the first year is undiscounted.
        year_label = year + (ANCHOR_YEAR - 1)

        # ===== Calculate private discount factors for fixed and variable methods =====
        discount_factors[year_label] = calculate_discount_factors(
            df=df_copy,
            base_year=ANCHOR_YEAR,
            target_year=year_label,
            discount_rate_col_name=discount_rate_col_name
        )

    method_suffix = PRIVATE_DISCOUNTING_METHOD_SUFFIXES[discount_rate_col_name]

    # ===== STEP 1: Heating validity drives all three cases =====
    # The retrofit is a single heat pump that replaces the heating system and
    # also serves cooling, so NPV is defined for homes with valid heating data
    # that are scheduled for this measure package.
    _, heating_valid_mask, _, _ = initialize_validation_tracking(
        df_copy, 'heating', menu_mp, verbose=verbose, copy=False)

    # Homes with no AC (include_cooling = False) get zero cooling savings and
    # zero cooling replacement credit, so for them Case 2 == Case 1 and
    # Case 3 == Case 1.
    if 'include_cooling' in df_copy.columns:
        include_cooling = df_copy['include_cooling'].fillna(False).astype(bool)
    else:
        include_cooling = pd.Series(False, index=df_copy.index)

    # ===== STEP 2-4: Discounted lifetime savings per category =====
    # Baseline cooling assumption: the home's existing AC (efficiency from the
    # ResStock source data) versus the ASHP in cooling mode (MP3 SEER1=15;
    # MP4 SEER1=24-29.3). Cooling savings are priced with the same electricity
    # $/kWh path as heating.
    heating_savings = _calculate_discounted_savings(
        df_measure_costs=df_fuel_costs_copy,
        df_baseline_costs=df_baseline_costs_copy,
        category='heating',
        lifetime=EQUIPMENT_SPECS['heating'],
        scenario_prefix=scenario_prefix,
        discount_factors=discount_factors,
        valid_mask=heating_valid_mask,
        menu_mp=menu_mp,
        verbose=verbose,
    )
    cooling_savings_raw = _calculate_discounted_savings(
        df_measure_costs=df_fuel_costs_copy,
        df_baseline_costs=df_baseline_costs_copy,
        category='cooling',
        lifetime=EQUIPMENT_SPECS['cooling'],
        scenario_prefix=scenario_prefix,
        discount_factors=discount_factors,
        valid_mask=heating_valid_mask,
        menu_mp=menu_mp,
        verbose=verbose,
    )

    # Zero cooling savings for no-AC homes; keep them where the home has AC.
    cooling_savings = cooling_savings_raw.where(include_cooling, other=0.0)
    heating_and_cooling_savings = heating_savings + cooling_savings

    # Save the discounted heating savings and the discounted cooling savings.
    # The NPV is heating savings plus cooling savings minus net capital cost.
    # All nine NPV cases use these same two savings figures and differ only in
    # net capital cost, so storing them once lets a reader reproduce the
    # subtraction for any case. Until now both figures were discarded after
    # the nine cases were built, leaving the savings side of the NPV
    # unverifiable from the exported columns.
    # Store the cooling figure AFTER the include_cooling adjustment above, so
    # the column holds what the NPV actually used: 0.0 for a home with
    # include_cooling = False, not the raw cooling savings.
    heating_savings_col = create_discounted_savings_col(
        scenario_prefix, 'heating', method_suffix)
    cooling_savings_col = create_discounted_savings_col(
        scenario_prefix, 'cooling', method_suffix)
    df_new_columns[heating_savings_col] = heating_savings
    df_new_columns[cooling_savings_col] = cooling_savings
    all_columns_to_mask['heating'].extend(
        [heating_savings_col, cooling_savings_col])

    # ===== Capital costs =====
    # Heating capital: heat-pump install (minus rebate) credited against the
    # heating system it replaces. The total (gross) capital is shared by all
    # three cases; only the net capital differs.
    total_capital, net_capital_heating = calculate_capital_costs(
        df_copy=df_copy,
        category='heating',
        input_mp=input_mp,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        cost_scenario=cost_scenario,
        valid_mask=heating_valid_mask,
        verbose=verbose,
    )

    # Two of the three cases credit the avoided cooling-system replacement, but
    # only for homes that actually have AC (include_cooling = True).
    cooling_replacement_col = create_cost_col(
        menu_mp=menu_mp, category='cooling',
        cost_type='replacement', cost_scenario=cost_scenario)
    
    cooling_replacement_cost = (
        df_copy[cooling_replacement_col].fillna(0).where(include_cooling, other=0.0))

    # Save the cooling replacement credit the NPV actually subtracted. It
    # differs from the raw mp{mp}_cooling_replacement_installed_cost_{scenario}
    # column in two ways: it is 0.0 for a home with include_cooling = False,
    # and it is 0.0 where the raw column is blank. Nationally 269 homes have
    # include_cooling = True but no recorded cooling replacement cost, so
    # their credit is 0.0 while the raw column reads blank.
    cooling_credit_col = create_cooling_credit_applied_col(
        menu_mp=menu_mp, cost_scenario=cost_scenario)
    df_new_columns[cooling_credit_col] = cooling_replacement_cost
    all_columns_to_mask['heating'].append(cooling_credit_col)
    
    net_capital_heating_and_cooling = net_capital_heating - cooling_replacement_cost
    
    # Cooling-only credit: heat-pump capital (net of rebate) credited against
    # the avoided AC replacement but NOT against the heating system it replaces.
    net_capital_cooling_only = total_capital - cooling_replacement_cost

    # ===== Subsidy handling =====
    # Compute the raw unsubsidized net capital first, then subtract the rebate
    # to obtain the subsidized values. This is more intuitive than adding the
    # rebate back to an already-subsidized number.
    if menu_mp in REBATE_ELIGIBLE_HEATING_MPS:
        rebate_col = create_rebate_col(menu_mp=menu_mp, category='heating', cost_scenario=cost_scenario)
        rebate_amount = df_copy[rebate_col].fillna(0.0).where(heating_valid_mask, other=0.0)
    else:
        rebate_amount = pd.Series(0.0, index=df_copy.index)

    net_capital_cooling_only_unsub = net_capital_cooling_only + rebate_amount
    net_capital_heating_unsub = net_capital_heating + rebate_amount
    net_capital_heating_and_cooling_unsub = net_capital_heating_and_cooling + rebate_amount

    net_capital_cooling_only_sub = net_capital_cooling_only_unsub - rebate_amount
    net_capital_heating_sub = net_capital_heating_unsub - rebate_amount
    net_capital_heating_and_cooling_sub = net_capital_heating_and_cooling_unsub - rebate_amount

    # June 2026 guidance: a third rebate policy scenario on the same NPV scopes. The
    # june2026 rebate column already encodes the efficiency (MP) and fuel gates,
    # so it is simply netted against the unsubsidized capital -- no second gate.
    # A missing column (rebate policy scenario not computed for this run) degrades
    # safely to the unsubsidized outcome.
    rebate_june2026_col = create_rebate_col(
        menu_mp=menu_mp, category='heating', cost_scenario=cost_scenario,
        guidance=REBATE_GUIDANCE_JUNE2026)
    if rebate_june2026_col in df_copy.columns:
        rebate_june2026_amount = (
            df_copy[rebate_june2026_col].fillna(0.0)
            .where(heating_valid_mask, other=0.0))
    else:
        rebate_june2026_amount = pd.Series(0.0, index=df_copy.index)

    net_capital_cooling_only_sub_june2026 = (
        net_capital_cooling_only_unsub - rebate_june2026_amount)
    net_capital_heating_sub_june2026 = (
        net_capital_heating_unsub - rebate_june2026_amount)
    net_capital_heating_and_cooling_sub_june2026 = (
        net_capital_heating_and_cooling_unsub - rebate_june2026_amount)

    # ===== Assemble the nine NPV cases =====
    # Every case counts both heating and cooling operating savings; the cases
    # differ only in which avoided-replacement credit reduces the net capital
    # cost, and each scope has three rebate-policy-scenario variants:
    # unsubsidized, subsidized under 2024 guidance (_sub), and subsidized under
    # June 2026 guidance (_sub_june2026).
    npv_case_inputs = {
        'heatingSavings_coolingLCC_unsub': (heating_and_cooling_savings, net_capital_cooling_only_unsub),
        'heatingSavings_coolingLCC_sub': (heating_and_cooling_savings, net_capital_cooling_only_sub),
        'heatingSavings_coolingLCC_sub_june2026': (heating_and_cooling_savings, net_capital_cooling_only_sub_june2026),
        'heatingLCC_coolingSavings_unsub': (heating_and_cooling_savings, net_capital_heating_unsub),
        'heatingLCC_coolingSavings_sub': (heating_and_cooling_savings, net_capital_heating_sub),
        'heatingLCC_coolingSavings_sub_june2026': (heating_and_cooling_savings, net_capital_heating_sub_june2026),
        'heatingLCC_coolingLCC_unsub': (heating_and_cooling_savings, net_capital_heating_and_cooling_unsub),
        'heatingLCC_coolingLCC_sub': (heating_and_cooling_savings, net_capital_heating_and_cooling_sub),
        'heatingLCC_coolingLCC_sub_june2026': (heating_and_cooling_savings, net_capital_heating_and_cooling_sub_june2026),
    }

    # The shared gross capital is stored once under the heating category.
    total_capital_col = create_capital_col(
        scenario_prefix=scenario_prefix, category='heating',
        net=False, cost_scenario=cost_scenario)
    df_new_columns[total_capital_col] = total_capital
    all_columns_to_mask['heating'].append(total_capital_col)

    for npv_case, (case_savings, case_net_capital) in npv_case_inputs.items():
        # Private NPV: lifetime energy-bill savings minus the incremental (net)
        # capital cost of the heat pump over a like-for-like baseline
        # replacement. This is the value the economic adoption decision uses
        # (NPV >= 0). A single willingness-to-pay framing is modeled, so the
        # column name carries no WTP token.
        npv_case_value = round(case_savings - case_net_capital, 2)

        npv_col = create_npv_case_col(
            scenario_prefix=scenario_prefix, npv_case=npv_case,
            method_suffix=method_suffix)
        net_capital_col = create_capital_col(
            scenario_prefix=scenario_prefix, category=npv_case,
            net=True, cost_scenario=cost_scenario)

        df_new_columns[npv_col] = npv_case_value
        df_new_columns[net_capital_col] = case_net_capital
        all_columns_to_mask['heating'].extend([npv_col, net_capital_col])

    # ===== STEP 5: Apply final verification masking for consistency =====
    df_result = apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        print(f"\nPrivate NPV calculation completed. Added {len(df_new_columns.columns)} new columns.")
    
    return df_result


def _calculate_discounted_savings(
    df_measure_costs: pd.DataFrame,
    df_baseline_costs: pd.DataFrame,
    category: str,
    lifetime: int,
    scenario_prefix: str,
    discount_factors: Dict[int, pd.Series],
    valid_mask: pd.Series,
    menu_mp: int,
    verbose: bool = False,
) -> pd.Series:
    """Compute discounted lifetime fuel-cost savings for one equipment category.

    Sums, over the equipment lifetime, the per-year avoided fuel cost
    (baseline minus measure) discounted to the base year. Values are masked to
    NaN outside valid_mask so excluded homes do not enter NPV aggregates.

    Args:
        df_measure_costs: DataFrame with measure-package annual fuel costs.
        df_baseline_costs: DataFrame with baseline annual fuel costs.
        category: Equipment category for fuel-cost column lookups
            ('heating' or 'cooling').
        lifetime: Equipment lifetime in years.
        scenario_prefix: Measure scenario prefix (e.g., 'ref2025_mp3_').
        discount_factors: Mapping from year label to a per-home discount factor.
        valid_mask: Homes with valid baseline data scheduled for the retrofit.
        menu_mp: Measure package identifier (0 = baseline; nonzero applies masking).
        verbose: Whether to raise on partial-year coverage. Default is False.

    Note:
        Year labels always start at ANCHOR_YEAR (2025), matching the year the
        fuel-cost columns were built for. Not a parameter, for the same reason
        as in calculate_private_npv.

    Returns:
        Series of discounted lifetime savings, NaN outside valid_mask.

    Raises:
        ValueError: If a required annual fuel-cost column is missing for any year.
    """
    # Initialize with zeros for valid homes, NaN for others.
    discounted_savings_template = create_retrofit_only_series(
        df_measure_costs, valid_mask)

    yearly_avoided_costs = []
    years_processed = 0

    # Sum the discounted avoided cost for each year of the equipment lifetime.
    for year in range(1, lifetime + 1):
        year_label = year + (ANCHOR_YEAR - 1)
        discount_factor = discount_factors[year_label]

        base_cost_col_name = create_fuel_cost_col('baseline_', year_label, category)
        measure_cost_col_name = create_fuel_cost_col(
            scenario_prefix, year_label, category)

        cols_exist = (
            base_cost_col_name in df_baseline_costs.columns
            and measure_cost_col_name in df_measure_costs.columns)
        if not cols_exist:
            raise ValueError(
                f"Fuel cost data missing for year {year_label}, "
                f"category '{category}'")

        avoided_costs = calculate_avoided_values(
            baseline_values=df_baseline_costs[base_cost_col_name],
            measure_values=df_measure_costs[measure_cost_col_name],
            retrofit_mask=(valid_mask if menu_mp != 0 else None),
        ) * discount_factor
        yearly_avoided_costs.append(avoided_costs)
        years_processed += 1

    if yearly_avoided_costs:
        avoided_costs_df = pd.concat(yearly_avoided_costs, axis=1)
        # skipna=False so a missing year propagates NaN rather than undercounting.
        total_discounted_savings = avoided_costs_df.sum(axis=1, skipna=False)
        if menu_mp != 0:
            total_discounted_savings = pd.Series(
                np.where(valid_mask, total_discounted_savings, np.nan),
                index=total_discounted_savings.index,
            )
    else:
        total_discounted_savings = discounted_savings_template

    # Replace tiny values with NaN to avoid numerical artifacts.
    total_discounted_savings = replace_small_values_with_nan(
        total_discounted_savings)

    if verbose and years_processed < lifetime:
        raise ValueError(
            f"Only processed {years_processed}/{lifetime} years for '{category}'")

    return total_discounted_savings


def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    context: str
) -> List[str]:
    """
    Validate that required columns exist in the DataFrame.

    Args:
        df: DataFrame to check.
        required_cols: List of column names that must exist.
        context: Description of the calculation context for error messages.

    Returns:
        List of missing column names (empty if all columns exist).
    """
    missing = [col for col in required_cols if col not in df.columns]
    return missing


def calculate_capital_costs(
    df_copy: pd.DataFrame,
    category: str,
    input_mp: str,
    menu_mp: int,
    policy_scenario: str,
    cost_scenario: str,
    valid_mask: pd.Series,
    verbose: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate total and net capital costs for an equipment category.

    This function computes the total capital cost and net capital cost (after accounting
    for replacement costs) based on the equipment category, measure package, and whether
    IRA rebates are applied. Net capital cost is the total capital cost minus
    the avoided heating-system replacement cost.

    Args:
        df_copy: DataFrame containing cost data.
        category: Equipment category (e.g., 'heating', 'waterHeating').
        input_mp: Measure package identifier (string) used for column naming.
        menu_mp: Measure package identifier (integer) used for column naming.
        policy_scenario: Policy scenario for the run. Single supported value:
            '2025 Reference Case'. IRA rebates are always applied for
            rebate-eligible measure packages.
        cost_scenario: Cost scenario identifier used for column naming (e.g., 'v4MID').
        valid_mask: Series indicating which rows have valid data for the category.
        verbose: Whether to print detailed processing information. Default is False.

    Returns:
        A tuple containing:
            - total_capital_cost: Series with total capital costs
            - net_capital_cost: Series with net capital costs (total - replacement)

    Raises:
        KeyError: If required installation cost columns are missing from the DataFrame.

    Notes:
        Single policy scenario ('2025 Reference Case'); IRA rebates are always
        applied for rebate-eligible measure packages.

        FOLLOW-UP FLAGGED 19 Aug 2026 -- the replacement cost subtracted here
        (create_cost_col(..., cost_type='replacement', ...), read a few lines
        below) is described everywhere as the avoided cost of replacing the
        home's EXISTING heating (or cooling) system like-for-like. For
        category='heating' that description does not match how the cost is
        actually priced. The REMDB v4 regression that produces this column is
        built from size_heating_system_primary_k_btu_h, which is the RETROFIT
        heat pump's ResStock-autosized capacity for this measure package, not
        a value read from the home's existing furnace or boiler. No baseline
        equipment capacity survives this pipeline at all: ResStock's own
        baseline run computes one, but it is dropped in
        process_euss_data.py's df_enduse_compare (the baseline-side
        equivalent, df_enduse_refactored, never carries any size_* column
        forward) and is not recoverable from any column in df_copy here.

        Concretely: on a 5-home spot check in the 19 Aug 2026 Allegheny
        export, one home's true baseline furnace capacity was 35.30 kBtu/h
        while the value actually priced for its "replacement" cost was 14.90
        kBtu/h (MP3) or 16.03 kBtu/h (MP4) -- the retrofit heat pump's
        capacity, roughly 2 to 2.5x smaller. Since the REMDB v4 regression is
        approximately linear in capacity, using a materially smaller capacity
        very likely understates net_capital_cost's replacement credit for
        many homes on the heating side, which would understate net_capital_cost
        and therefore make the NPV look worse than it should for adoption --
        but this was NOT measured across the full population this session, so
        do not treat "2 to 2.5x" or the direction of the NPV effect as a
        confirmed population-level result, only as what one small sample
        showed.

        UNCONFIRMED by this session, left for the follow-up: (1) whether
        category='cooling' is affected to a similar degree -- the same
        5-home sample showed the retrofit cooling capacity landing close to
        the baseline air conditioner's own capacity, unlike heating, but that
        was not checked at scale; (2) the actual dollar and NPV impact across
        the full sample; (3) whether MP3 and MP4 are affected differently,
        since their retrofit capacities differ from each other as well as
        from baseline.

        A fix is planned for a separate, value-critical session, because
        changing the replacement-cost capacity input would move net_capital_cost,
        the NPV, and the adoption rate for a large share of homes -- exactly
        the kind of change this session's scope excluded. Do not attempt the
        fix here. Full trace and the four flagged sites:
        docs/SESSION_CHANGELOG_2026-08-19.md.
    """
    if verbose:
        print(f"\nCalculating costs for {category}... ")

    # Build list of required columns based on category and policy scenario
    upgrade_cost_col_name = create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)
    replacement_cost_col_name = create_cost_col(menu_mp=menu_mp, category=category, cost_type='replacement', cost_scenario=cost_scenario)
    required_cols = [upgrade_cost_col_name, replacement_cost_col_name]

    if category == 'heating':
        if input_mp in ['upgrade09', 'upgrade10']:
            required_cols.append(create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario))
            # Weatherization rebate applies to MP9 and MP10.
            required_cols.append(create_weatherization_rebate_col(cost_scenario=cost_scenario))

        # Only high-efficiency MPs are eligible for heating rebates.
        if menu_mp in REBATE_ELIGIBLE_HEATING_MPS:
            required_cols.append(create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario))

    else:
        required_cols.append(create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario))

    # Validate required columns exist
    missing_cols = _validate_required_columns(df_copy, required_cols,
        f"{category} capital cost calculation for MP{menu_mp}")

    if missing_cols:
        raise KeyError(
            f"Missing required columns for {category} capital cost calculation "
            f"(MP{menu_mp}, {policy_scenario}): {missing_cols}. "
            f"Ensure installation costs are calculated before calling calculate_private_npv()."
        )

    # '2025 Reference Case' policy scenario has both a subsidized and unsubsidized sensitivity case.
    if category == 'heating':
        if input_mp in ('upgrade09', 'upgrade10'):
            # MP9/MP10 add a weatherization (enclosure) cost, net of its rebate.
            weatherization_cost = (
                df_copy[create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario)].fillna(0)
                - df_copy[create_weatherization_rebate_col(cost_scenario=cost_scenario)].fillna(0))
        else:
            weatherization_cost = 0.0

        # Heating installation cost is the upgrade cost plus any weatherization
        # cost. The former installation heating premium term has been removed.
        installation_cost = (
            df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)].fillna(0)
            + weatherization_cost)

        # Only high-efficiency MPs are eligible for heating rebates.
        if menu_mp in REBATE_ELIGIBLE_HEATING_MPS:
            rebate_amount = df_copy[create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario)].fillna(0)
        else:
            rebate_amount = 0.0

        total_capital_cost = installation_cost - rebate_amount
        # FOLLOW-UP FLAGGED 19 Aug 2026 (see the function docstring Notes for
        # the full trace): this replacement-cost column is priced off the
        # retrofit heat pump's capacity, not the existing heating system's,
        # so the credit subtracted here is not truly sized to what it claims
        # to replace. Not fixed this session.
        net_capital_cost = total_capital_cost - df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='replacement', cost_scenario=cost_scenario)].fillna(0)

    else:
        installation_cost = df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)].fillna(0)
        rebate_amount = df_copy[create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario)].fillna(0)
        total_capital_cost = installation_cost - rebate_amount
        net_capital_cost = total_capital_cost - df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='replacement', cost_scenario=cost_scenario)].fillna(0)

    # Apply masking to costs based on valid_mask. Valid homes keep their values, invalid homes get NaN
    total_capital_cost_masked = pd.Series(np.nan, index=df_copy.index)
    net_capital_cost_masked = pd.Series(np.nan, index=df_copy.index)

    total_capital_cost_masked.loc[valid_mask] = total_capital_cost.loc[valid_mask]
    net_capital_cost_masked.loc[valid_mask] = net_capital_cost.loc[valid_mask]

    return total_capital_cost_masked, net_capital_cost_masked
