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
    cost_scenario: str) -> str:
    """Build rebate amount column name."""

    return f'mp{menu_mp}_{category}_rebate_amount_{cost_scenario}'


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
# PUBLIC IMPACT: CLIMATE & HEALTH NPV COLUMNS
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


def create_health_npv_col(
    scenario_prefix: str,
    category: str,
    rcm_model: str,
    cr_function: str) -> str:
    """Build health NPV column name.
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        rcm_model: RCM model ('ap2', 'easiur', 'inmap').
        cr_function: Concentration-response function ('acs', 'h6c').
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_health_npv_col('iraRef_mp3_', 'heating', 'inmap', 'acs')
        'iraRef_mp3_heating_health_npv_inmap_acs'
    """
    return f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'


def create_public_npv_col(
    scenario_prefix: str,
    category: str,
    scc_assumption: str,
    rcm_model: str,
    cr_function: str) -> str:
    """Build combined public NPV column name.
    
    Combines climate and health NPV with all sensitivity parameters.
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        scc_assumption: SCC assumption ('lower', 'central', 'upper').
        rcm_model: RCM model ('ap2', 'easiur', 'inmap').
        cr_function: Concentration-response function ('acs', 'h6c').
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_public_npv_col('iraRef_mp3_', 'heating', 'central', 'inmap', 'acs')
        'iraRef_mp3_heating_public_npv_central_inmap_acs'
    """

    return f'{scenario_prefix}{category}_public_npv_{scc_assumption}_{rcm_model}_{cr_function}'


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
    category: str,
    column_type: str,
    cost_scenario: str,
    method_suffix: str,
    scc_assumption: str = None,
    rcm_model: str = None,
    cr_function: str = None,) -> str:
    """Build adoption potential column name.
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        column_type: Type of column ('benefit', 'adoption', 'impact', 'health_sensitivity',
                     'total_npv_climateOnly', 'total_npv_healthOnly').
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
        method_suffix: Discount method suffix (e.g., '_fixed_low'), used for adoption only.
        scc_assumption: SCC assumption (required for benefit/adoption/impact).
        rcm_model: RCM model (required for benefit/adoption/impact).
        cr_function: CR function (required for benefit/adoption/impact).
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_adoption_col('iraRef_mp3_', 'heating', 'benefit', 'central', 'inmap', 'acs', cost_scenario='v3')
        'iraRef_mp3_heating_benefit_central_inmap_acs_v3'
        >>> create_adoption_col('iraRef_mp3_', 'heating', 'adoption', 'central', 'inmap', 'acs', '_fixed_low', cost_scenario='v3')
        'iraRef_mp3_heating_adoption_central_inmap_acs_fixed_low_v3'
        >>> create_adoption_col('iraRef_mp3_', 'heating', 'adoption', 'central', 'inmap', 'acs', '_fixed_low', cost_scenario='v4MID')
        'iraRef_mp3_heating_adoption_central_inmap_acs_fixed_low_v4MID'
        >>> create_adoption_col('iraRef_mp3_', 'heating', 'health_sensitivity', cost_scenario='v3')
        'iraRef_mp3_heating_health_sensitivity'
    """

    if column_type == 'health_sensitivity':
        return f'{scenario_prefix}{category}_health_sensitivity'
    elif column_type == 'benefit':
        return f'{scenario_prefix}{category}_benefit_{scc_assumption}_{rcm_model}_{cr_function}_{cost_scenario}'
    elif column_type == 'adoption':
        return f'{scenario_prefix}{category}_adoption_{scc_assumption}_{rcm_model}_{cr_function}_{cost_scenario}{method_suffix}'
    elif column_type == 'impact':
        return f'{scenario_prefix}{category}_impact_{scc_assumption}_{rcm_model}_{cr_function}_{cost_scenario}'
    else:
        raise ValueError(f"Invalid column_type '{column_type}'. Must be one of: "
                        "'benefit', 'adoption', 'impact', 'health_sensitivity', ")


def create_total_npv_col(
    scenario_prefix: str,
    category: str,
    cost_scenario: str,
    method_suffix: str,
    scc_assumption: str = None,
    rcm_model: str = None,
    cr_function: str = None,
    climate_only: bool = False,
    health_only: bool = False) -> str:
    """Build total NPV column name (private + public).
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        cost_scenario: 'v3' or 'v4LOW/MID/HIGH'.
        method_suffix: Method suffix for discount variations (e.g. '_fixed_low').
        scc_assumption: SCC assumption ('lower', 'central', 'upper').
        rcm_model: RCM model (required for default and health_only modes).
        cr_function: CR function (required for default and health_only modes).
        climate_only: If True, builds climate-only total NPV column.
        health_only: If True, builds health-only total NPV column.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_total_npv_col('iraRef_mp3_', 'heating', 'central', 'inmap', 'acs', cost_scenario='v3', method_suffix='_fixed_low')
        'iraRef_mp3_heating_total_npv_central_inmap_acs_fixed_low_v3'
        >>> create_total_npv_col('iraRef_mp3_', 'heating', 'central', 'inmap', 'acs', cost_scenario='v4MID', method_suffix='_fixed_low')
        'iraRef_mp3_heating_total_npv_central_inmap_acs_fixed_low_v4MID'
        >>> create_total_npv_col('iraRef_mp3_', 'heating', 'central', climate_only=True, cost_scenario='v3', method_suffix='_fixed_base')
        'iraRef_mp3_heating_total_npv_climateOnly_central_fixed_base_v3'
        >>> create_total_npv_col('iraRef_mp3_', 'heating', 'central', 'inmap', 'acs', health_only=True, cost_scenario='v3', method_suffix='_fixed_low')
        'iraRef_mp3_heating_total_npv_healthOnly_inmap_acs_fixed_low_v3'
    """

    if climate_only:
        return f'{scenario_prefix}{category}_total_npv_climateOnly_{scc_assumption}_{cost_scenario}{method_suffix}'
    
    if health_only:
        return f'{scenario_prefix}{category}_total_npv_healthOnly_{rcm_model}_{cr_function}_{cost_scenario}{method_suffix}'

    return f'{scenario_prefix}{category}_total_npv_{scc_assumption}_{rcm_model}_{cr_function}_{cost_scenario}{method_suffix}'
