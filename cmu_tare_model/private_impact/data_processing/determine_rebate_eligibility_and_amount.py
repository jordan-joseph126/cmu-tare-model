import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union, Callable

from cmu_tare_model.constants import (
    REBATE_MAPPING,
    REBATE_ELIGIBLE_HEATING_MPS,
    REBATE_GUIDANCE_IRA2024,
    REBATE_GUIDANCE_JUNE2026,
    REBATE_RULE_CONFIG,
    AMI_LOW_CUTOFF,
    AMI_MODERATE_CUTOFF,
    HEEHR_COVERAGE_LOW,
    HEEHR_COVERAGE_MOD,
    HEEHR_CAP_HEAT_PUMP,
    HOMES_MIN_SAVINGS_FRAC,
    HOMES_TIER2_SAVINGS_FRAC,
    HOMES_CAP_TIER1,
    HOMES_CAP_TIER2, 
    HOMES_COVERAGE_NON_LMI,
    ELECTRIC_RESISTANCE_BASELINE,
    NON_PARTICIPATING_REBATE_STATES,
    REBATE_NONE,
    REBATE_HEEHR,
    REBATE_HOMES,
    VERBOSE,
)
from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2025_2018
from cmu_tare_model.utils.column_names import (
    create_cost_col,
    create_rebate_col,
    create_enclosure_cost_col,
    create_weatherization_rebate_col
)
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_final_masking,
)

from cmu_tare_model.private_impact.data_processing.process_income_data_for_rebates import (
    df_county_medianIncome,
    df_state_medianIncome,
)

"""
================================================================================================================================================================================
FUNCTIONS: AMI AND INCOME GROUP DESIGNATION FOR REBATE ELIGIBILITY
================================================================================================================================================================================
- UPDATED APRIL 22, 2025 WITH IMPROVED DOCUMENTATION, MODULARITY, ERROR HANDLING
"""


def generate_household_medianIncome_2025(row: pd.Series) -> float:
    """
    Generate a household median income value in USD2025 using a probabilistic
    approach.

    Samples from a normal distribution based on income range bounds, then
    ensures the final value remains within the valid income range.

    Args:
        row: DataFrame row containing income_low, income_high, and income values

    Returns:
        float: Generated median income value in 2025 dollars
    """
    # The ResStock household-income bins are reported in USD2018, so inflate
    # them to the model reference year (USD2025) before sampling.
    low = row['income_low'] * cpi_ratio_2025_2018
    high = row['income_high'] * cpi_ratio_2025_2018
    mean = row['income'] * cpi_ratio_2025_2018

    # Calculate std assuming 10th and 90th percentiles
    std = (high - low) / (norm.ppf(0.90) - norm.ppf(0.10))

    # Sample from the normal distribution
    ami_2025 = np.random.normal(loc=mean, scale=std)

    # Ensure the generated income is within the bounds
    ami_2025 = max(low, min(high, ami_2025))
    return ami_2025


def fill_na_with_hierarchy(
        df: pd.DataFrame,
        df_county: pd.DataFrame,
        df_state: pd.DataFrame) -> pd.DataFrame:
    """
    Fills 'census_area_medianIncome' using a two-level lookup: county-level
    median income first, then state-level for any county that does not match.

    Connecticut is the main reason a county can miss the county-level join. The
    Census switched Connecticut from its legacy counties (FIPS 09001-09015) to
    nine planning regions (09110-09190), but ResStock still uses the legacy
    county codes. Those codes do not exist in the current ACS file, so
    Connecticut homes fall through to the state-level value. This is expected,
    not a data error.

    Args:
        df: The main DataFrame with area median income to fill
        df_county: DataFrame with median incomes at the county level
        df_state: DataFrame with median incomes at the state level

    Returns:
        DataFrame: Modified DataFrame with 'census_area_medianIncome' filled
    """
    # Fill using county-level median incomes first.
    df['census_area_medianIncome'] = df['county'].map(
        df_county.set_index('gis_joinID_county')['median_income_USD2025']
    )

    # Any county that did not match (notably Connecticut, see above) falls
    # back to the state-level median income.
    nan_mask = df['census_area_medianIncome'].isna()
    df.loc[nan_mask, 'census_area_medianIncome'] = df.loc[nan_mask, 'state'].map(
        df_state.set_index('state_abbrev')['median_income_USD2025']
    )

    return df


def calculate_percent_AMI(df_results_IRA: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Calculates the percentage of Area Median Income (AMI) and assigns income level designations.

    This function processes household income data, calculates the percentage relative to 
    Area Median Income, and creates two categorical classifications:
    1. Detailed income level categories (Low, Moderate, Middle-to-Upper Income)
    2. Binary Low-to-Moderate Income (LMI) or Middle-to-Upper Income (MUI) classification for policy analysis

    Args:
        df_results_IRA: Input DataFrame containing income information with columns:
                       - 'income': Income data (ranges or values)
                       - Other demographic/geographic columns for median income lookup
        random_seed: Random seed for reproducible income sampling. Ensures consistent
                    income classifications across different measure package runs (e.g.,
                    MP4 and MP8 produce identical rebate eligibility). Default: 42.

    Returns:
        DataFrame: Modified DataFrame with additional columns:
                  - 'household_income': Calculated household income (float)
                  - 'census_area_medianIncome': Area median income (float)
                  - 'percent_AMI': Percentage of AMI (float)
                  - 'income_level': Detailed income category (str)
                  - 'lmi_or_mui': Binary Low-to-Moderate Income (LMI) or Middle-to-Upper Income (MUI) (str)
        
    Raises:
        ValueError: If an unexpected income format is encountered during processing
    """
    # Create a mapping for special income ranges
    income_map = {
        '<10000': (9999.0, 9999.0),
        '200000+': (200000.0, 200000.0)
    }

    def split_income_range(income):
        """
        Processes income data which may be ranges, special values, or direct floats.
        
        Args:
            income: Income value (str, float, or special format)
            
        Returns:
            tuple: (low_income, high_income) for range calculation
            
        Raises:
            ValueError: If income format cannot be parsed
        """
        if isinstance(income, float):  # Handle float income directly
            return income, income
        if income in income_map:
            return income_map[income]
        try:
            # Parse income ranges like "50000-75000"
            low, high = map(float, income.split('-'))
            return low, high
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Unexpected income format: {income}. Expected format: 'low-high', '<10000', '200000+', or numeric value.") from e

    # Apply the income range split
    income_ranges = df_results_IRA['income'].apply(split_income_range)
    df_results_IRA['income_low'], df_results_IRA['income_high'] = zip(*income_ranges)
    df_results_IRA['income'] = (df_results_IRA['income_low'] + df_results_IRA['income_high']) / 2
    
    # Set random seed for reproducible income sampling across MP runs.
    # This ensures identical income classifications (and thus identical rebate
    # eligibility) for the same homes regardless of which measure package is
    # being processed — critical for MP4 vs MP8 result consistency.
    np.random.seed(random_seed)
    
    # Apply the generate_household_medianIncome_2025 function
    df_results_IRA['household_income'] = df_results_IRA.apply(generate_household_medianIncome_2025, axis=1)

    # Drop the intermediate columns
    df_results_IRA.drop(['income_low', 'income_high'], axis=1, inplace=True)

    # Fill 'census_area_medianIncome' with the hierarchical lookup:
    # match county-level median income first, then state-level.
    df_results_IRA = fill_na_with_hierarchy(
        df_results_IRA,
        df_county=df_county_medianIncome,
        df_state=df_state_medianIncome
    )

    # Ensure income and census_area_medianIncome columns are float
    df_results_IRA['household_income'] = df_results_IRA['household_income'].astype(float).round(2)
    df_results_IRA['census_area_medianIncome'] = df_results_IRA['census_area_medianIncome'].astype(float).round(2)

    # Calculate percent_AMI
    df_results_IRA['percent_AMI'] = ((df_results_IRA['household_income'] / df_results_IRA['census_area_medianIncome']) * 100).round(2)

    # Create detailed income level categories
    income_conditions = [
        df_results_IRA['percent_AMI'] <= 80.0,
        (df_results_IRA['percent_AMI'] > 80.0) & (df_results_IRA['percent_AMI'] <= 150.0)
    ]
    income_choices = ['Low-Income', 'Moderate-Income']

    df_results_IRA['income_level'] = np.select(
        income_conditions, 
        income_choices, 
        default='Middle-to-Upper-Income'
    )

    # Create binary LMI/MUI classification
    # Method 1: Using the income_level column we just created
    df_results_IRA['lmi_or_mui'] = df_results_IRA['income_level'].apply(
        lambda x: 'LMI' if x in ['Low-Income', 'Moderate-Income'] else 'MUI'
    )
    
    # Alternative Method 2: Direct threshold-based approach (more efficient for large datasets)
    # df_results_IRA['lmi_or_mui'] = np.where(
    #     df_results_IRA['percent_AMI'] <= 150.0, 'LMI', 'MUI'
    # )

    return df_results_IRA


"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS: CALCULATE REBATE AMOUNTS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def get_max_rebate_amount(
        row: pd.Series,
        category: str) -> Tuple[float, float]:
    """
    Determine the maximum rebate amounts based on the category and row data.
    
    Looks up rebate eligibility based on predefined mapping in REBATE_MAPPING.
    
    Args:
        row: DataFrame row containing upgrade information
        category: Equipment category (e.g., 'heating', 'waterHeating')
        
    Returns:
        Tuple containing:
            - max_rebate_amount: Maximum rebate amount for the equipment
            - max_weatherization_rebate_amount: Maximum rebate amount for weatherization
    """
    if category in REBATE_MAPPING:
        column, conditions, rebate_amount = REBATE_MAPPING[category]
        max_rebate_amount = rebate_amount if any(cond in str(row[column]) for cond in conditions) else 0.00
    else:
        max_rebate_amount = 0.00

    max_weatherization_rebate_amount = 1600.00
    return max_rebate_amount, max_weatherization_rebate_amount


def calculate_rebate(
        df_results_IRA: pd.DataFrame, 
        row: pd.Series,
        category: str, 
        menu_mp: int, 
        coverage_rate: float,
        cost_scenario: str) -> None:
    """
    Calculate and assign the rebate amounts for a specific row.
    
    Args:
        df_results_IRA: DataFrame to update with rebate amounts
        row: Row containing installation cost data
        category: Equipment category
        menu_mp: Measure package identifier
        coverage_rate: Rebate coverage rate (1.0 for low-income, 0.5 for moderate-income)
        cost_scenario: Cost methodology key ('v3' or 'v4LOW/MID/HIGH').
        
    Raises:
        ValueError: If an invalid category is provided
        KeyError: If required columns are missing
    """
    try:
        max_rebate_amount, max_weatherization_rebate_amount = get_max_rebate_amount(row, category)
        
        # Calculate equipment rebate
        install_cost_col_name = create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)
        rebate_col_name = create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario)
        
        if install_cost_col_name in row and not pd.isna(row[install_cost_col_name]):
            project_coverage = round(row[install_cost_col_name] * coverage_rate, 2)
            df_results_IRA.at[row.name, rebate_col_name] = min(project_coverage, max_rebate_amount)
        else:
            df_results_IRA.at[row.name, rebate_col_name] = 0.00
            if coverage_rate > 0 and max_rebate_amount > 0:
                raise ValueError(f"Warning: Installation cost data missing for row {row.name}, category {category}. Setting rebate to 0.")
        
        # Calculate weatherization rebate if applicable
        enclosure_cost_col_name = create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario)
        weatherization_rebate_col_name = create_weatherization_rebate_col(cost_scenario=cost_scenario)
        if enclosure_cost_col_name in df_results_IRA.columns and menu_mp in [9, 10]:
            if enclosure_cost_col_name in row and not pd.isna(row[enclosure_cost_col_name]):
                weatherization_project_coverage = round(row[enclosure_cost_col_name] * coverage_rate, 2)
                df_results_IRA.at[row.name, weatherization_rebate_col_name] = min(weatherization_project_coverage, max_weatherization_rebate_amount)
            else:
                df_results_IRA.at[row.name, weatherization_rebate_col_name] = 0.00
                if coverage_rate > 0 and menu_mp in [9, 10]:
                    raise ValueError(f"Warning: Enclosure cost data missing for row {row.name}. Setting weatherization rebate to 0.")
    
    except Exception as e:
        print(f"Error calculating rebate for row {row.name}, category {category}: {str(e)}")
        
        # Set default values to prevent calculations from breaking
        df_results_IRA.at[row.name, rebate_col_name] = 0.00
        weatherization_rebate_col_name = create_weatherization_rebate_col(cost_scenario=cost_scenario)
        if menu_mp in [9, 10] and weatherization_rebate_col_name in df_results_IRA.columns:
            df_results_IRA.at[row.name, weatherization_rebate_col_name] = 0.00


def _heehr_rebate_amount(
        df_copy: pd.DataFrame,
        menu_mp: int,
        cost_scenario: str,
        python_round: bool) -> pd.Series:
    """Vectorized HEEHR rebate amount per home, before eligibility masking.

    HEEHR pays a share of the heat-pump install cost, up to a fixed per-measure
    cap. Income sets the share (100% at or below 80% AMI, 50% between 80 and
    150%); the cap is the $8,000 heat-pump cap when the modeled upgrade is a
    heat pump, and $0 otherwise.

    The cap is looked up from REBATE_MAPPING (the same source the 2024 row-wise
    path used) so the tech-string check is preserved: only an 'ASHP'/'MSHP'
    upgrade earns the cap. For every modeled MP3/MP4 retrofit the upgrade is a
    heat pump, so this equals the flat HEEHR_CAP_HEAT_PUMP.

    Args:
        df_copy: DataFrame with percent_AMI, the heating upgrade cost column,
            and the upgrade-technology column.
        menu_mp: Measure package identifier.
        cost_scenario: Cost methodology key (e.g. 'v4MID').
        python_round: If True, round the capped amount with Python's builtin
            round() (the 2024 behavior); if False, with numpy's array round()
            (the June 2026 behavior). The two disagree by one cent on exact
            half-cent products, so each vintage keeps its original rounding to
            stay byte-identical. See the REBATE_RULE_CONFIG note.

    Returns:
        A float Series of HEEHR amounts aligned to df_copy's index.
    """
    heating_cost = df_copy[create_cost_col(
        menu_mp=menu_mp, category='heating', cost_type='upgrade',
        cost_scenario=cost_scenario)].fillna(0.0)

    pct_ami = df_copy['percent_AMI']
    coverage = np.where(
        pct_ami <= AMI_LOW_CUTOFF * 100, HEEHR_COVERAGE_LOW, HEEHR_COVERAGE_MOD)

    # Tech-gated cap from REBATE_MAPPING: only a heat-pump upgrade earns the cap.
    # This reproduces the 2024 get_max_rebate_amount behavior; the mapped amount
    # equals HEEHR_CAP_HEAT_PUMP for heat pumps.
    tech_column, tech_conditions, mapped_cap = REBATE_MAPPING['heating']
    is_heat_pump = df_copy[tech_column].astype(str).apply(
        lambda upgrade: any(cond in upgrade for cond in tech_conditions))
    cap = np.where(is_heat_pump, mapped_cap, 0.0)

    # Cap the covered project cost, then round with the vintage's rounding.
    capped = np.minimum(cap, coverage * heating_cost)
    capped = pd.Series(capped, index=df_copy.index, dtype=float)
    if python_round:
        amount = capped.apply(lambda value: round(value, 2))
    else:
        amount = capped.round(2)
    return amount.astype(float)


def _homes_rebate_amount(
        df_copy: pd.DataFrame,
        menu_mp: int,
        cost_scenario: str) -> Tuple[pd.Series, pd.Series]:
    """Vectorized HOMES rebate amount and savings-floor mask, before masking.

    HOMES is performance-based on the modeled whole-home percent savings:
    at least 20% savings earns the $2,000 tier, at least 35% earns the $4,000
    tier; the rebate covers 50% of the full electrification project cost
    (heating + cooling upgrade), up to the tier cap. Only the non-LMI amounts
    are implemented because HOMES is consulted only above 150% AMI, where the
    low-income doubling is unreachable by construction.

    Args:
        df_copy: DataFrame with the heating and cooling upgrade cost columns and
            mp{menu_mp}_modeled_savings_frac.
        menu_mp: Measure package identifier.
        cost_scenario: Cost methodology key (e.g. 'v4MID').

    Returns:
        Tuple (amount, qualifies_savings):
          amount: float Series of HOMES amounts aligned to df_copy's index.
          qualifies_savings: boolean Series, True where savings meet the 20%
            floor (homes below the floor earn nothing).
    """
    heating_cost = df_copy[create_cost_col(
        menu_mp=menu_mp, category='heating', cost_type='upgrade',
        cost_scenario=cost_scenario)].fillna(0.0)

    cooling_cost_col = create_cost_col(
        menu_mp=menu_mp, category='cooling', cost_type='upgrade',
        cost_scenario=cost_scenario)
    if cooling_cost_col in df_copy.columns:
        cooling_cost = df_copy[cooling_cost_col].fillna(0.0)
    else:
        cooling_cost = pd.Series(0.0, index=df_copy.index)
    total_project_cost = heating_cost + cooling_cost

    savings_frac = df_copy[f'mp{menu_mp}_modeled_savings_frac']
    homes_cap = np.where(
        savings_frac >= HOMES_TIER2_SAVINGS_FRAC,
        HOMES_CAP_TIER2, HOMES_CAP_TIER1)
    amount = np.minimum(
        homes_cap, HOMES_COVERAGE_NON_LMI * total_project_cost).round(2)

    qualifies_savings = savings_frac >= HOMES_MIN_SAVINGS_FRAC
    return pd.Series(amount, index=df_copy.index, dtype=float), qualifies_savings


def calculate_rebate_program(
    df_results_IRA: pd.DataFrame,
    category: str,
    menu_mp: int,
    cost_scenario: str,
    guidance: str,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    """Calculate heat-pump rebate amounts and program eligibility for one vintage.

    Single central rebate function for both guidance vintages. Each vintage
    models both federal programs, mutually exclusive and routed by income:

      - HEEHR (percent_AMI <= 150%): a fixed $8,000 heat-pump cap; income sets
        the share of project cost covered (100% at <=80% AMI, 50% at 80-150%).
      - HOMES (percent_AMI > 150%): savings-based tiers on the modeled whole-home
        percent savings (>=20% -> $2,000 cap, >=35% -> $4,000 cap), covering 50%
        of the full electrification project cost. Non-LMI amounts only.

    The per-vintage rule differences (fuel gates, whether HOMES is modeled, the
    output column names, and the eligibility label) come from
    REBATE_RULE_CONFIG[guidance] -- see constants.py for the field meanings.

    Gates applied before program routing (all vintages):
      - Efficiency (ENERGY STAR): only REBATE_ELIGIBLE_HEATING_MPS qualify.
      - State participation: homes in a never-participating state (e.g. South
        Dakota) get 0 / 'None'.
    HEEHR additionally applies a fuel gate under June 2026 (only existing
    electric-resistance heating qualifies, because a rebate may not fund removing
    a fossil system). HOMES is fuel-neutral (see the config note on the 2026
    homes_fuel_gate, kept only for byte-identity pending re-derivation).

    Args:
        df_results_IRA: DataFrame with percent_AMI, base_heating_fuel, state, the
            per-MP upgrade cost columns, and mp{menu_mp}_modeled_savings_frac.
        category: Equipment category. Only 'heating' carries the rebate;
            'cooling' is a no-op (the heat-pump rebate covers both end uses).
        menu_mp: Measure package identifier.
        cost_scenario: Cost methodology key (e.g. 'v4MID').
        guidance: Rebate vintage, one of REBATE_RULE_CONFIG's keys
            (REBATE_GUIDANCE_IRA2024 or REBATE_GUIDANCE_JUNE2026).
        verbose: Whether to print progress detail.

    Returns:
        The DataFrame with two new columns:
          - the rebate amount column from create_rebate_col (guidance-less for
            2024, june2026-tokened for June 2026); float64, NaN for excluded
            homes and 0.0 for valid-but-ineligible homes.
          - the program eligibility column named by the config
            ('HEEHR' | 'HOMES' | 'None'; NaN for excluded homes).

    Raises:
        ValueError: If category has no rebate mapping, or guidance is unknown.
    """
    # Cooling is covered by the heating heat-pump rebate; nothing to do here.
    if category == 'cooling':
        if verbose:
            print("Skipping rebate for 'cooling' "
                  "(covered by the heating heat-pump rebate).")
        return df_results_IRA

    if category not in REBATE_MAPPING:
        raise ValueError(
            f"Category '{category}' is not supported for rebate calculations. "
            f"Valid categories with rebates: {list(REBATE_MAPPING.keys())}."
        )

    if guidance not in REBATE_RULE_CONFIG:
        raise ValueError(
            f"Unknown rebate guidance '{guidance}'. "
            f"Must be one of {list(REBATE_RULE_CONFIG.keys())}.")
    config = REBATE_RULE_CONFIG[guidance]

    # Step 1 -- shared validation bookkeeping so excluded homes stay NaN-masked.
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = (
        initialize_validation_tracking(
            df_results_IRA, category, menu_mp, verbose=verbose)
    )

    # Step 2 -- create the amount and eligibility columns.
    # Amount: NaN for excluded homes, 0.0 for valid homes (the default outcome).
    rebate_col = create_rebate_col(
        menu_mp=menu_mp, category=category, cost_scenario=cost_scenario,
        guidance=config['column_guidance'])
    df_copy[rebate_col] = create_retrofit_only_series(df_copy, valid_mask)
    df_copy.loc[valid_mask, rebate_col] = 0.0
    category_columns_to_mask.append(rebate_col)

    # Eligibility label: NaN for excluded homes, 'None' for valid homes.
    eligibility_col = config['eligibility_col'].format(mp=menu_mp)
    df_copy[eligibility_col] = pd.Series(
        np.nan, index=df_copy.index, dtype=object)
    df_copy.loc[valid_mask, eligibility_col] = REBATE_NONE

    # Step 3 -- efficiency gate: only high-efficiency (ENERGY STAR) MPs qualify.
    # Applies to BOTH programs. Ineligible MPs stay at 0.0 / 'None'.
    if menu_mp not in REBATE_ELIGIBLE_HEATING_MPS:
        if verbose:
            print(f"  MP{menu_mp} is NOT rebate-eligible ({guidance}, standard "
                  f"efficiency). All amounts 0, eligibility 'None'.")
        df_copy = apply_final_masking(
            df_copy, all_columns_to_mask, verbose=verbose)
        return df_copy

    # Step 4 -- shared gate masks.
    # Fuel gate: only existing electric-resistance heating qualifies. Read
    # base_heating_fuel only when a fuel gate is actually active so the 2024
    # (fuel-neutral) path does not require the column. When no gate is active the
    # mask is all-True and never filters anyone out.
    homes_fuel_gate_active = config['homes_enabled'] and config['homes_fuel_gate']
    if config['heehr_fuel_gate'] or homes_fuel_gate_active:
        electric_mask = df_copy['base_heating_fuel'].isin(
            ELECTRIC_RESISTANCE_BASELINE)
    else:
        electric_mask = pd.Series(True, index=df_copy.index)
    # State-participation gate: never-participating states are ineligible.
    participating_mask = ~df_copy['state'].isin(NON_PARTICIPATING_REBATE_STATES)

    # percent_AMI is on a 0-150+ percent scale, so scale the ratio cutoff by 100.
    pct_ami = df_copy['percent_AMI']
    mod_cut = AMI_MODERATE_CUTOFF * 100

    # Step 5 -- HEEHR (percent_AMI <= 150%). Fixed cap; income sets the share.
    heehr_mask = valid_mask & participating_mask & (pct_ami <= mod_cut)
    if config['heehr_fuel_gate']:
        heehr_mask = heehr_mask & electric_mask
    heehr_amount = _heehr_rebate_amount(
        df_copy, menu_mp, cost_scenario,
        python_round=config['heehr_python_round'])
    df_copy.loc[heehr_mask, rebate_col] = heehr_amount.loc[heehr_mask]
    df_copy.loc[heehr_mask, eligibility_col] = REBATE_HEEHR

    # Step 6 -- HOMES (percent_AMI > 150%), savings-based, non-LMI amounts.
    if config['homes_enabled']:
        homes_mask = valid_mask & participating_mask & (pct_ami > mod_cut)
        if config['homes_fuel_gate']:
            homes_mask = homes_mask & electric_mask
        # Only read the HOMES inputs (modeled savings, cooling cost) when at
        # least one home routes to HOMES; a run with no home above 150% AMI need
        # not carry those columns.
        if homes_mask.any():
            homes_amount, homes_qualifies_savings = _homes_rebate_amount(
                df_copy, menu_mp, cost_scenario)
            # Homes below the 20% savings floor earn nothing (stay 0.0 / 'None').
            homes_qualifies = homes_mask & homes_qualifies_savings
            df_copy.loc[homes_qualifies, rebate_col] = (
                homes_amount.loc[homes_qualifies])
            df_copy.loc[homes_qualifies, eligibility_col] = REBATE_HOMES

    # Step 7 -- final verification masking keeps excluded homes NaN.
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)

    return df_copy


def calculate_rebateIRA(
    df_results_IRA: pd.DataFrame,
    category: str,
    menu_mp: int,
    cost_scenario: str,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """DEPRECATED thin wrapper for the 2024-guidance rebate.

    Prefer calling calculate_rebate_program(..., guidance=REBATE_GUIDANCE_IRA2024)
    directly. Retained so existing call sites keep working. Delegates to the central
    calculate_rebate_program with guidance=REBATE_GUIDANCE_IRA2024, which
    reproduces the original 2024 HEEHR amounts byte-for-byte (verified) and, once
    2024 HOMES is enabled, also credits the HOMES pathway above 150% AMI.

    Note: the original 2024 path created a separate weatherization rebate column
    for MP9/MP10. The consolidated function models only the heat-pump (heating)
    rebate; MP9/MP10 are inactive (VALID_MENU_MPS = [0, 3, 4]) and out of scope,
    so no weatherization column is produced. Reintroduce it if MP9/MP10 are
    activated.

    Args:
        df_results_IRA: DataFrame with income designations and cost data.
        category: Equipment category (only 'heating' carries a rebate).
        menu_mp: Measure package identifier.
        cost_scenario: Cost methodology key (e.g. 'v4MID').
        verbose: Flag to enable verbose logging.

    Returns:
        Updated DataFrame with calculated 2024-guidance rebate amounts.
    """
    return calculate_rebate_program(
        df_results_IRA=df_results_IRA,
        category=category,
        menu_mp=menu_mp,
        cost_scenario=cost_scenario,
        guidance=REBATE_GUIDANCE_IRA2024,
        verbose=verbose,
    )


def calculate_rebate_june2026(
    df_results_IRA: pd.DataFrame,
    category: str,
    menu_mp: int,
    cost_scenario: str,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    """DEPRECATED thin wrapper for the June 2026 DOE-guidance rebate.

    Prefer calling
    calculate_rebate_program(..., guidance=REBATE_GUIDANCE_JUNE2026) directly.
    Retained so existing call sites keep working; delegates to the central
    calculate_rebate_program.

    Implements the June 2026 rebate policy scenario as a parallel column to the existing
    (2024-guidance) HEEHR rebate. Two programs are modeled, routed by income and
    mutually exclusive (one program per retrofit):

      - HEEHR (percent_AMI <= 150%): a fixed $8,000 heat-pump cap; income sets the
        share of project cost covered (100% at <=80% AMI, 50% at 80-150% AMI).
      - HOMES (percent_AMI > 150%): savings-based tiers on the modeled whole-home
        percent savings (>=20% -> $2,000 cap; >=35% -> $4,000 cap), covering 50%
        of project cost. Every HOMES home is above 150% AMI, so the low-income
        doubling is unreachable and only the non-LMI amounts apply.

    June 2026 gates, applied before routing:
      - Efficiency (ENERGY STAR): only REBATE_ELIGIBLE_HEATING_MPS qualify, both
        programs. MP3 gets no rebate.
      - Fuel: rebates may not fund removing a fossil heating system. TARE models
        only full electrification, so only existing electric-resistance homes
        qualify; any fossil baseline gets 0 and 'None'.

    Placeholders (documented limitations, not enforced here): weatherization
    prerequisite, dual-fuel retention pathway, and state-level funding caps.

    Args:
        df_results_IRA: DataFrame with percent_AMI, base_heating_fuel, per-MP
            upgrade cost columns, and mp{menu_mp}_modeled_savings_frac.
        category: Equipment category. Only 'heating' carries the rebate; 'cooling'
            is a no-op (the heat-pump rebate covers both end uses).
        menu_mp: Measure package identifier.
        cost_scenario: Cost methodology key (e.g. 'v4MID').
        verbose: Whether to print progress detail.

    Returns:
        The DataFrame with two new columns:
          - mp{menu_mp}_heating_rebate_amount_june2026_{cost_scenario} (float64;
            NaN for excluded homes, 0.0 for valid-but-ineligible)
          - mp{menu_mp}_rebate_eligibility_june2026 ('HEEHR' | 'HOMES' | 'None';
            NaN for excluded homes)

    Raises:
        ValueError: If category has no rebate mapping.
    """
    return calculate_rebate_program(
        df_results_IRA=df_results_IRA,
        category=category,
        menu_mp=menu_mp,
        cost_scenario=cost_scenario,
        guidance=REBATE_GUIDANCE_JUNE2026,
        verbose=verbose,
    )


def summarize_june2026_rebate_totals(
    df_results_IRA: pd.DataFrame,
    menu_mp: int,
    cost_scenario: str,
    weight_col: str = 'weight',
) -> Tuple[pd.DataFrame, pd.Series]:
    """Total modeled June 2026 rebate dollars by program, nationally and by state.

    Each home's rebate is weighted by its ResStock household weight so the totals
    are national dollar estimates (each sampled home stands for many real homes).
    The model applies no aggregate or state-level funding cap, so these are
    uncapped program costs -- exactly the figure worth reporting.

    Args:
        df_results_IRA: Model output holding the June 2026 rebate amount and
            eligibility columns, a `state` column, and the household weight.
        menu_mp: Measure package identifier (only MP4 carries rebates).
        cost_scenario: Cost methodology key (e.g. 'v4MID').
        weight_col: Household weight column. Pass a column of ones for unweighted
            (sampled-home) sums.

    Returns:
        Tuple (by_state, national):
          by_state: DataFrame indexed by state with dollar columns 'HEEHR',
            'HOMES', and 'total'.
          national: Series with the 'HEEHR', 'HOMES', and 'total' dollar totals.

    Raises:
        KeyError: If a required column is missing.
    """
    amount_col = create_rebate_col(
        menu_mp=menu_mp, category='heating', cost_scenario=cost_scenario,
        guidance=REBATE_GUIDANCE_JUNE2026)
    eligibility_col = f'mp{menu_mp}_rebate_eligibility_june2026'

    required = [amount_col, eligibility_col, 'state', weight_col]
    missing = [c for c in required if c not in df_results_IRA.columns]
    if missing:
        raise KeyError(
            f"summarize_june2026_rebate_totals: missing columns {missing}. "
            f"Run calculate_rebate_june2026 (menu_mp={menu_mp}) first.")

    # Dollars per home = per-home rebate x how many real homes it represents.
    weights = df_results_IRA[weight_col].fillna(0.0)
    dollars = df_results_IRA[amount_col].fillna(0.0) * weights

    tidy = pd.DataFrame({
        'state': df_results_IRA['state'],
        'program': df_results_IRA[eligibility_col],
        'dollars': dollars,
    })
    # Keep only homes that actually received a rebate.
    tidy = tidy[tidy['program'].isin([REBATE_HEEHR, REBATE_HOMES])]

    by_state = (
        tidy.groupby(['state', 'program'])['dollars'].sum()
        .unstack(fill_value=0.0)
        .reindex(columns=[REBATE_HEEHR, REBATE_HOMES], fill_value=0.0)
    )
    by_state['total'] = by_state.sum(axis=1)
    by_state = by_state.sort_index()

    national = by_state.sum()
    national.name = 'national'

    return by_state, national


def summarize_rebate_funding(
    df_results_IRA: pd.DataFrame,
    menu_mp: int,
    cost_scenario: str,
    guidance: Optional[str] = None,
    weight_col: str = 'weight',
    adopter_col: Optional[str] = None,
    fuel_col: str = 'base_heating_fuel',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted rebate funding by program and by baseline fuel.

    Reports modeled rebate dollars two ways -- summed over ALL eligible homes and
    (optionally) over economic ADOPTERS only -- so the uncapped potential can be
    distinguished from the funding that reaches homes that actually retrofit. The
    model applies no aggregate/state funding cap, so 'total_eligible' is an
    uncapped potential, not a disbursement.

    The by-fuel table is a correctness check on eligibility. Under June 2026
    guidance (guidance='june2026') the HEEHR fuel gate forbids funding
    fossil-system removal, so fossil baselines earn HEEHR $0 (they can only earn
    the fuel-neutral HOMES pathway). Under 2024 guidance (guidance=None) fossil
    baselines DO receive HEEHR by design -- the HEEHR fuel restriction is a June
    2026 change. Both vintages now model HOMES (fuel-neutral), so the program
    split is read from an explicit eligibility label, not inferred from a
    positive amount.

    Args:
        df_results_IRA: Model output with the rebate amount column, household
            weight, baseline fuel, and the program eligibility column for the
            requested vintage.
        menu_mp: Measure package identifier (only MP4 carries rebates).
        cost_scenario: Cost methodology key (e.g. 'v4MID').
        guidance: None for the 2024 rebate column (guidance-less amount name), or
            'june2026' for the June 2026 column. Each vintage now carries an
            explicit HEEHR/HOMES eligibility label.
        weight_col: Household weight column.
        adopter_col: Optional 0/1 economic-adopter column. When given, an
            'adopters_only' column is added. Pass the column that matches this
            rebate policy scenario and NPV scope.
        fuel_col: Baseline heating-fuel column.

    Returns:
        Tuple (by_program, by_fuel):
          by_program: DataFrame indexed by program ('HEEHR', 'HOMES') with a
            dollar column 'total_eligible' and, if adopter_col is given,
            'adopters_only'.
          by_fuel: same dollar columns, indexed by baseline heating fuel.

    Raises:
        ValueError: If guidance is not None or a known vintage token.
        KeyError: If a required column is missing.
    """
    amount_col = create_rebate_col(
        menu_mp=menu_mp, category='heating', cost_scenario=cost_scenario,
        guidance=guidance)

    # Map the amount-column guidance token to the rule-config vintage key so the
    # correct eligibility label is read (2024's amount column is guidance-less,
    # but its label uses the 'ira2024' token).
    config_key = REBATE_GUIDANCE_IRA2024 if guidance is None else guidance
    if config_key not in REBATE_RULE_CONFIG:
        raise ValueError(
            f"summarize_rebate_funding: unknown guidance '{guidance}'. "
            f"Use None (2024) or one of {list(REBATE_RULE_CONFIG.keys())}.")
    eligibility_col = (
        REBATE_RULE_CONFIG[config_key]['eligibility_col'].format(mp=menu_mp))

    required = [amount_col, weight_col, fuel_col, eligibility_col]
    if adopter_col is not None:
        required.append(adopter_col)
    missing = [c for c in required if c not in df_results_IRA.columns]
    if missing:
        raise KeyError(f"summarize_rebate_funding: missing columns {missing}.")

    weights = df_results_IRA[weight_col].fillna(0.0)
    eligible_dollars = df_results_IRA[amount_col].fillna(0.0) * weights

    # Program label per home read from the explicit HEEHR/HOMES eligibility
    # column. Both vintages now model HOMES, so a positive-amount inference would
    # mislabel HOMES dollars as HEEHR.
    program = df_results_IRA[eligibility_col]

    frame = pd.DataFrame({
        'program': program,
        'fuel': df_results_IRA[fuel_col],
        'total_eligible': eligible_dollars,
    })
    if adopter_col is not None:
        is_adopter = df_results_IRA[adopter_col].fillna(0.0) == 1.0
        frame['adopters_only'] = eligible_dollars.where(is_adopter, 0.0)

    value_cols = [c for c in ('total_eligible', 'adopters_only')
                  if c in frame.columns]

    # By program: only homes that received a rebate.
    rebated = frame[frame['program'].isin([REBATE_HEEHR, REBATE_HOMES])]
    by_program = (rebated.groupby('program')[value_cols].sum()
                  .reindex([REBATE_HEEHR, REBATE_HOMES], fill_value=0.0))

    # By baseline fuel: over ALL homes so fossil fuels are visible -- they must
    # be $0 under June 2026 (the fossil-system-removal restriction).
    by_fuel = frame.groupby('fuel')[value_cols].sum().sort_index()

    return by_program, by_fuel
