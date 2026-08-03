"""Centralized column name builders for TARE model.

This module provides a single source of truth for all column naming conventions
across REMDB v3 and v4 scenarios, preventing naming mismatch bugs.

Key patterns:
- INPUT cost columns (from data pipeline):
    v3: No percentile suffix (e.g., 'mp3_heating_upgrade_installed_cost')
    v4: Includes percentile suffix (e.g., 'mp3_heating_upgrade_installed_cost_mid')
- OUTPUT columns (capital costs, NPV, adoption):
    Always include an REMDB suffix: _v3, _v4LOW, _v4MID, _v4HIGH
    e.g., 'iraRef_mp3_heating_total_capital_cost_v3'
    e.g., 'iraRef_mp3_heating_private_npv_lessWTP_fixed_low_v4MID'

Usage:
    from cmu_tare_model.utils.column_names import create_cost_col, create_capital_col
    
    # INPUT cost column (reads from data pipeline columns)
    col = create_cost_col(3, 'heating', 'upgrade', cost_scenario='v3')  # -> 'mp3_heating_upgrade_installed_cost_v3'
    col = create_cost_col(3, 'heating', 'upgrade', cost_scenario='v4MID')  # -> 'mp3_heating_upgrade_installed_cost_v4MID'
    
    # OUTPUT capital column (always has cost_scenario)
    col = create_capital_col('iraRef_mp3_', 'heating', net=False, cost_scenario='v3')
    # -> 'iraRef_mp3_heating_total_capital_cost_v3'
    col = create_capital_col('iraRef_mp3_', 'heating', net=True, cost_scenario='v4MID')
    # -> 'iraRef_mp3_heating_net_capital_cost_v4MID'
"""

from cmu_tare_model.constants import REMDB_COST_SCENARIO_KEYS
from typing import Optional

# =============================================================================
# PRIVATE IMPACT: COST COLUMNS
# =============================================================================

def create_fuel_cost_col(
    scenario_prefix: str,
    year_label: str,
    category: str) -> str:
    """Build fuel cost column name for a given scenario, year, and category

    Args:
        scenario_prefix: Scenario prefix (e.g., 'baseline_', 'iraRef_mp3_').
        year_label: Year label (e.g., 'year1', 'year2', ..., 'year30').
        category: Equipment category (e.g., 'heating').
    
    Returns:
        Column name string.
    """

    # Get column names for baseline and measure package fuel costs
    return f'{scenario_prefix}{year_label}_{category}_fuel_cost'


def create_cost_col(
    menu_mp: int,
    category: str,
    cost_type: str,
    cost_scenario: str) -> str:
    """Build installed cost column name.

    Args:
        menu_mp: Measure package number.
        category: Equipment category (e.g. 'heating').
        cost_type: 'upgrade' or 'replacement'.
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.

    Returns:
        Column name string.
    """
    return f'mp{menu_mp}_{category}_{cost_type}_installed_cost_{cost_scenario}'


def create_rebate_col(
    menu_mp: int,
    category: str,
    cost_scenario: str,
    guidance: Optional[str] = None) -> str:
    """Build rebate amount column name.

    Args:
        menu_mp: Measure package number.
        category: Equipment category (e.g. 'heating').
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
        guidance: Rebate-guidance token for the rebate-policy-scenario axis. When None,
            returns the original (guidance-less) name so the 2024-guidance
            columns stay byte-identical. When set (e.g. 'june2026'), the token
            is inserted so each guidance has its own rebate column.

    Returns:
        Column name string.

    Examples:
        >>> create_rebate_col(4, 'heating', 'v4MID')
        'mp4_heating_rebate_amount_v4MID'
        >>> create_rebate_col(4, 'heating', 'v4MID', guidance='june2026')
        'mp4_heating_rebate_amount_june2026_v4MID'
    """
    if guidance is None:
        return f'mp{menu_mp}_{category}_rebate_amount_{cost_scenario}'
    return f'mp{menu_mp}_{category}_rebate_amount_{guidance}_{cost_scenario}'


def create_capital_col(
    scenario_prefix: str,
    category: str,
    net: bool,
    cost_scenario: str) -> str:
    """Build capital cost column name.
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        net: If True, builds net capital cost column. If False, builds total.
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_capital_col('iraRef_mp3_', 'heating', net=False, cost_scenario='v3')
        'iraRef_mp3_heating_total_capital_cost_v3'
        >>> create_capital_col('iraRef_mp3_', 'heating', net=True, cost_scenario='v4MID')
        'iraRef_mp3_heating_net_capital_cost_v4MID'
    """

    kind = 'net' if net else 'total'
    return f'{scenario_prefix}{category}_{kind}_capital_cost_{cost_scenario}'



def create_npv_col(
    scenario_prefix: str,
    category: str,
    wtp: str,
    cost_scenario: str,
    method_suffix: str) -> str:
    """Build private NPV column name.

    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        wtp: 'lessWTP' or 'moreWTP'.
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
        method_suffix: Discount method suffix (e.g. '_fixed_low', '_fixed_high').

    Returns:
        Column name string.
    
    Examples:
        >>> create_npv_col('iraRef_mp3_', 'heating', 'lessWTP', cost_scenario='v3', method_suffix='_fixed_low')
        'iraRef_mp3_heating_private_npv_lessWTP_v3_fixed_low'
        >>> create_npv_col('iraRef_mp3_', 'heating', 'lessWTP', cost_scenario='v4MID', method_suffix='_fixed_low')
        'iraRef_mp3_heating_private_npv_lessWTP_v4MID_fixed_low'
    """

    return f'{scenario_prefix}{category}_private_npv_{wtp}_{cost_scenario}{method_suffix}'


# The private NPV is computed under three cases. All three count BOTH the
# heating and cooling operating (energy-bill) savings from the heat pump; the
# cases differ only in which capital costs enter the lifecycle calculation.
# In each case label, "LCC" means that end use's capital is included -- the
# avoided replacement of that end use's incumbent equipment is credited --
# while "Savings" means that end use contributes operating savings only.
#
# Each scope has three rebate-policy-scenario variants (the
# rebate-policy-scenario sensitivity axis): "_sub" is subsidized under 2024
# guidance (current HEEHR), "_unsub" is unsubsidized, and "_sub_june2026" is
# subsidized under June 2026 DOE guidance (adds the fuel gate and the HOMES
# pathway). The "_sub" and "_unsub" cases are unchanged; "_sub_june2026" is the
# new case added for the rebate-policy-scenario comparison.
#
#   heatingSavings_coolingLCC_sub          -> 2024 subsidized; cooling replacement only
#   heatingSavings_coolingLCC_unsub        -> unsubsidized; cooling replacement only
#   heatingSavings_coolingLCC_sub_june2026 -> June 2026 subsidized; cooling replacement only
#   heatingLCC_coolingSavings_sub          -> 2024 subsidized; heating replacement only
#   heatingLCC_coolingSavings_unsub        -> unsubsidized; heating replacement only
#   heatingLCC_coolingSavings_sub_june2026 -> June 2026 subsidized; heating replacement only
#   heatingLCC_coolingLCC_sub              -> 2024 subsidized; both replacements
#   heatingLCC_coolingLCC_unsub            -> unsubsidized; both replacements
#   heatingLCC_coolingLCC_sub_june2026     -> June 2026 subsidized; both replacements
#
# The case label is carried as the "category" segment of the NPV and
# economic-adopter column names. Defined here as the single source of truth so
# no module hardcodes the strings.
NPV_CASE_CATEGORIES = (
    "heatingSavings_coolingLCC_sub",
    "heatingSavings_coolingLCC_unsub",
    "heatingSavings_coolingLCC_sub_june2026",
    "heatingLCC_coolingSavings_sub",
    "heatingLCC_coolingSavings_unsub",
    "heatingLCC_coolingSavings_sub_june2026",
    "heatingLCC_coolingLCC_sub",
    "heatingLCC_coolingLCC_unsub",
    "heatingLCC_coolingLCC_sub_june2026",
)

# The study base case: the single NPV case reported as the headline result and
# used as the default economic-adopter definition (for example the constrained
# grid-impact set and the county adoption-rate map). Unsubsidized, with both the
# heating and cooling replacement costs credited in the NPV. Defined here as the
# one source of truth so the peak-load module default and every notebook cell
# import the same string -- change it here to move the base case everywhere.
# Must be one of NPV_CASE_CATEGORIES above.
BASE_CASE_NPV_CASE = "heatingLCC_coolingLCC_unsub"


def create_npv_case_col(
    scenario_prefix: str,
    npv_case: str,
    method_suffix: str) -> str:
    """Build a private NPV column name for one of the three NPV cases.

    The heat pump is the only heating technology, and the adoption decision uses
    a single willingness-to-pay framing and a single capital-cost scenario, so
    the NPV column name carries neither a WTP token nor a cost-scenario token.
    Validates npv_case so the three-case names stay consistent across the NPV,
    capital-cost, and economic-adopter modules. The npv_case value is used
    directly as the category segment.

    Args:
        scenario_prefix: Scenario prefix (e.g., 'ref2025_mp3_').
        npv_case: One of NPV_CASE_CATEGORIES.
        method_suffix: Discount method suffix (e.g., '_fixed_base').

    Returns:
        Column name string.

    Raises:
        ValueError: If npv_case is not one of NPV_CASE_CATEGORIES.

    Examples:
        >>> create_npv_case_col('ref2025_mp3_', 'heatingLCC_coolingLCC_sub',
        ...                     method_suffix='_fixed_base')
        'ref2025_mp3_heatingLCC_coolingLCC_sub_private_npv_fixed_base'
    """
    if npv_case not in NPV_CASE_CATEGORIES:
        raise ValueError(
            f"Invalid npv_case: '{npv_case}'. "
            f"Must be one of {NPV_CASE_CATEGORIES}")

    return f'{scenario_prefix}{npv_case}_private_npv{method_suffix}'


def create_enclosure_cost_col(
    menu_mp: int,
    cost_scenario: str) -> str:
    """Build enclosure/weatherization upgrade cost column name.
    
    Only applicable to MP9 and MP10.
    
    Args:
        menu_mp: Measure package number (typically 9 or 10).
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_enclosure_cost_col(9, 'v3')
        'mp9_enclosure_upgrade_installed_cost_v3'
        >>> create_enclosure_cost_col(9, 'v4MID')
        'mp9_enclosure_upgrade_installed_cost_v4MID'
    """   

    return f'mp{menu_mp}_enclosure_upgrade_installed_cost_{cost_scenario}'


def create_weatherization_rebate_col(cost_scenario: str) -> str:
    """Build weatherization rebate amount column name.
    
    Args:
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_weatherization_rebate_col('v3')
        'weatherization_rebate_amount_v3'
        >>> create_weatherization_rebate_col('v4MID')
        'weatherization_rebate_amount_v4MID'
    """

    return f'weatherization_rebate_amount_{cost_scenario}'


def create_installation_premium_col(
    menu_mp: int,
    category: str) -> str:
    """Build installation premium column name.
    
    Typically used for heating category to capture additional installation costs
    for heat pump systems.
    
    Args:
        menu_mp: Measure package number.
        category: Equipment category (e.g., 'heating').
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_installation_premium_col(3, 'heating')
        'mp3_heating_installation_premium'
    """

    return f'mp{menu_mp}_{category}_installation_premium'


def create_combined_heating_cooling_col(
    menu_mp: int,
    cost_type: str,
    cost_scenario: str) -> str:
    """Build combined heating and cooling column name.
    
    Used for MPs that upgrade both heating and cooling systems simultaneously.
    
    Args:
        menu_mp: Measure package number.
        cost_type: Type of cost column (e.g., 'replacement_installed_cost', 
                   'net_capital_cost').
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_combined_heating_cooling_col(3, 'replacement_installed_cost', 'v4MID')
        'mp3_heating_and_cooling_replacement_installed_cost_v4MID'
    """

    return f'mp{menu_mp}_heating_and_cooling_{cost_type}_{cost_scenario}'


# =============================================================================
# PUBLIC IMPACT: CLIMATE NPV COLUMNS
# =============================================================================

def create_climate_npv_col(
    scenario_prefix: str,
    category: str, 
    scc_assumption: str) -> str:
    """Build climate NPV column name.
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        scc_assumption: SCC assumption ('lower', 'central', 'upper').
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_climate_npv_col('iraRef_mp3_', 'heating', 'central')
        'iraRef_mp3_heating_climate_npv_central'
    """

    return f'{scenario_prefix}{category}_climate_npv_{scc_assumption}'


def create_lifetime_damages_col(
    scenario_prefix: str,
    category: str,
    impact_type: str,
    mer_type_or_rcm: str,
    scc_or_cr: str) -> str:
    """Build lifetime damages column name (climate or health).
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_', 'baseline_').
        category: Equipment category (e.g., 'heating').
        impact_type: 'climate' or 'health'.
        mer_type_or_rcm: For climate: 'lrmer' or 'srmer'. For health: RCM model.
        scc_or_cr: For climate: SCC assumption. For health: CR function.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_lifetime_damages_col('baseline_', 'heating', 'climate', 'lrmer', 'central')
        'baseline_heating_lifetime_damages_climate_lrmer_central'
        >>> create_lifetime_damages_col('iraRef_mp3_', 'heating', 'health', 'inmap', 'acs')
        'iraRef_mp3_heating_lifetime_damages_health_inmap_acs'
    """
    return f'{scenario_prefix}{category}_lifetime_damages_{impact_type}_{mer_type_or_rcm}_{scc_or_cr}'


def create_avoided_damages_col(
    scenario_prefix: str,
    category: str,
    impact_type: str,
    mer_type_or_rcm: str,
    scc_or_cr: str) -> str:
    """Build avoided damages column name (climate or health).
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        impact_type: 'climate' or 'health'.
        mer_type_or_rcm: For climate: 'lrmer' or 'srmer'. For health: RCM model.
        scc_or_cr: For climate: SCC assumption. For health: CR function.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_avoided_damages_col('iraRef_mp3_', 'heating', 'climate', 'lrmer', 'central')
        'iraRef_mp3_heating_avoided_damages_climate_lrmer_central'
        >>> create_avoided_damages_col('iraRef_mp3_', 'heating', 'health', 'inmap', 'acs')
        'iraRef_mp3_heating_avoided_damages_health_inmap_acs'
    """
    return f'{scenario_prefix}{category}_avoided_damages_{impact_type}_{mer_type_or_rcm}_{scc_or_cr}'


# =============================================================================
# ADOPTION POTENTIAL COLUMNS
# =============================================================================

def create_adoption_col(
    scenario_prefix: str,
    npv_case: str,
    method_suffix: str,
) -> str:
    """Build an economic-adopter column name for one NPV case.

    The economic-adopter columns flag homes where the heat-pump retrofit pays
    for itself (private NPV >= 0). One column exists per NPV case, and the
    adoption decision uses only private economics -- no climate or health terms
    appear in the name. A single willingness-to-pay framing and a single
    capital-cost scenario are modeled, so the name carries neither a WTP token
    nor a cost-scenario token. The name mirrors the private NPV column with
    ``private_npv`` swapped for ``econ_adopter``.

    Args:
        scenario_prefix: Scenario prefix ending in an underscore
            (e.g. 'ref2025_mp3_').
        npv_case: One of NPV_CASE_CATEGORIES, including both subsidized and
            unsubsidized variants (e.g. 'heatingLCC_coolingLCC_sub').
        method_suffix: Discount-method suffix (e.g. '_fixed_base').

    Returns:
        Column name string.

    Raises:
        ValueError: If npv_case is not one of NPV_CASE_CATEGORIES.

    Examples:
        >>> create_adoption_col('ref2025_mp3_', 'heatingLCC_coolingLCC_sub',
        ...                     method_suffix='_fixed_base')
        'ref2025_mp3_heatingLCC_coolingLCC_sub_econ_adopter_fixed_base'
    """
    if npv_case not in NPV_CASE_CATEGORIES:
        raise ValueError(
            f"Invalid npv_case: '{npv_case}'. "
            f"Must be one of {NPV_CASE_CATEGORIES}")

    return f'{scenario_prefix}{npv_case}_econ_adopter{method_suffix}'


def create_total_npv_col(
    scenario_prefix: str,
    category: str,
    cost_scenario: str,
    method_suffix: str,
    scc_assumption: str = None,
    climate_only: bool = True) -> str:
    """Build a climate-only total NPV column name (private + climate benefit).

    Combines the private NPV with the social cost of carbon benefit. Health
    damages are not part of the adoption model, so only the climate-only column
    is produced.

    Args:
        scenario_prefix: Scenario prefix (e.g., 'ref2025_mp3_').
        category: Equipment category (e.g., 'heating').
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
        method_suffix: Method suffix for discount variations (e.g. '_fixed_base').
        scc_assumption: SCC assumption ('lower', 'central', 'upper').
        climate_only: Retained for call-site compatibility; must be True.

    Returns:
        Column name string.

    Raises:
        ValueError: If climate_only is not True.

    Examples:
        >>> create_total_npv_col('ref2025_mp3_', 'heating', scc_assumption='central',
        ...                      climate_only=True, cost_scenario='v4MID',
        ...                      method_suffix='_fixed_base')
        'ref2025_mp3_heating_total_npv_climateOnly_central_v4MID_fixed_base'
    """
    if not climate_only:
        raise ValueError(
            "create_total_npv_col builds only climate-only columns; "
            "pass climate_only=True.")

    return f'{scenario_prefix}{category}_total_npv_climateOnly_{scc_assumption}_{cost_scenario}{method_suffix}'
