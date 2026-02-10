"""Centralized column name builders for TARE model.

This module provides a single source of truth for all column naming conventions
across REMDB v3 and v4 scenarios, preventing naming mismatch bugs.

Key patterns:
- v3: No percentile suffix (e.g., 'mp3_heating_upgrade_installed_cost')
- v4: Includes percentile suffix (e.g., 'mp3_heating_upgrade_installed_cost_mid')

Usage:
    from cmu_tare_model.utils.column_names import create_cost_col, create_rebate_col
    
    # REMDB v3
    col = create_cost_col(3, 'heating', 'upgrade')
    # -> 'mp3_heating_upgrade_installed_cost'
    
    # REMDB v4
    col = create_cost_col(3, 'heating', 'upgrade', cost_scenario='remdb_v4_mid')
    # -> 'mp3_heating_upgrade_installed_cost_mid'
"""

from cmu_tare_model.constants import parse_cost_scenario


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
    cost_scenario: str = 'remdb_v3') -> str:
    """Build installed cost column name.

    Args:
        menu_mp: Measure package number.
        category: Equipment category (e.g. 'heating').
        cost_type: 'upgrade' or 'replacement'.
        cost_scenario: 'remdb_v3' or 'remdb_v4_low/mid/high'.

    Returns:
        Column name string.
    """
    _, percentile = parse_cost_scenario(cost_scenario)
    sfx = f'_{percentile}' if percentile else ''
    return f'mp{menu_mp}_{category}_{cost_type}_installed_cost{sfx}'


def create_rebate_col(
    menu_mp: int,
    category: str,
    cost_scenario: str = 'remdb_v3') -> str:
    """Build rebate amount column name."""
    _, percentile = parse_cost_scenario(cost_scenario)
    sfx = f'_{percentile}' if percentile else ''
    return f'mp{menu_mp}_{category}_rebate_amount{sfx}'


def create_capital_col(
    scenario_prefix: str,
    category: str,
    net: bool,
    cost_scenario: str = 'remdb_v3') -> str:
    """Build capital cost column name.
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        net: If True, builds net capital cost column. If False, builds total.
        cost_scenario: 'remdb_v3' or 'remdb_v4_low/mid/high'.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_capital_col('iraRef_mp3_', 'heating', net=False)
        'iraRef_mp3_heating_total_capital_cost'
        >>> create_capital_col('iraRef_mp3_', 'heating', net=True, cost_scenario='remdb_v4_mid')
        'iraRef_mp3_heating_net_capital_cost_mid'
    """
    _, percentile = parse_cost_scenario(cost_scenario)
    kind = 'net' if net else 'total'
    sfx = f'_{percentile}' if percentile else ''
    return f'{scenario_prefix}{category}_{kind}_capital_cost{sfx}'


def create_npv_col(
    scenario_prefix: str,
    category: str,
    wtp: str,
    method_suffix: str = '',
    percentile: str = None) -> str:
    """Build private NPV column name.

    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        wtp: 'lessWTP' or 'moreWTP'.
        method_suffix: v3 discount method suffix (e.g. '_fixed_low', '_fixed_high').
        percentile: v4 percentile (e.g. 'mid'). If provided, used as suffix.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_npv_col('iraRef_mp3_', 'heating', 'lessWTP', method_suffix='_fixed_low')
        'iraRef_mp3_heating_private_npv_lessWTP_fixed_low'
        >>> create_npv_col('iraRef_mp3_', 'heating', 'lessWTP', percentile='mid')
        'iraRef_mp3_heating_private_npv_lessWTP_mid'
    """
    if percentile:
        return f'{scenario_prefix}{category}_private_npv_{wtp}_{percentile}'
    return f'{scenario_prefix}{category}_private_npv_{wtp}{method_suffix}'


def create_enclosure_cost_col(
    menu_mp: int,
    cost_scenario: str = 'remdb_v3') -> str:
    """Build enclosure/weatherization upgrade cost column name.
    
    Only applicable to MP9 and MP10.
    
    Args:
        menu_mp: Measure package number (typically 9 or 10).
        cost_scenario: 'remdb_v3' or 'remdb_v4_low/mid/high'.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_enclosure_cost_col(9)
        'mp9_enclosure_upgradeCost'
        >>> create_enclosure_cost_col(9, 'remdb_v4_mid')
        'mp9_enclosure_upgrade_installed_cost_mid'
    """
    _, percentile = parse_cost_scenario(cost_scenario)
    if percentile:
        return f'mp{menu_mp}_enclosure_upgrade_installed_cost_{percentile}'
    return f'mp{menu_mp}_enclosure_upgradeCost'


def create_weatherization_rebate_col(cost_scenario: str = 'remdb_v3') -> str:
    """Build weatherization rebate amount column name.
    
    Args:
        cost_scenario: 'remdb_v3' or 'remdb_v4_low/mid/high'.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_weatherization_rebate_col()
        'weatherization_rebate_amount'
        >>> create_weatherization_rebate_col('remdb_v4_mid')
        'weatherization_rebate_amount_mid'
    """
    _, percentile = parse_cost_scenario(cost_scenario)
    sfx = f'_{percentile}' if percentile else ''
    return f'weatherization_rebate_amount{sfx}'


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
    cost_scenario: str = 'remdb_v3') -> str:
    """Build combined heating and cooling column name.
    
    Used for MPs that upgrade both heating and cooling systems simultaneously.
    
    Args:
        menu_mp: Measure package number.
        cost_type: Type of cost column (e.g., 'replacement_installed_cost', 
                   'net_capital_cost').
        cost_scenario: 'remdb_v3' or 'remdb_v4_low/mid/high'.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_combined_heating_cooling_col(3, 'replacement_installed_cost', 'remdb_v4_mid')
        'mp3_heating_and_cooling_replacement_installed_cost_mid'
    """
    _, percentile = parse_cost_scenario(cost_scenario)
    sfx = f'_{percentile}' if percentile else ''
    return f'mp{menu_mp}_heating_and_cooling_{cost_type}{sfx}'


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
    scc_assumption: str = None,
    rcm_model: str = None,
    cr_function: str = None,
    method_suffix: str = '') -> str:
    """Build adoption potential column name.
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        column_type: Type of column ('benefit', 'adoption', 'impact', 'health_sensitivity').
        scc_assumption: SCC assumption (required for benefit/adoption/impact).
        rcm_model: RCM model (required for benefit/adoption/impact).
        cr_function: CR function (required for benefit/adoption/impact).
        method_suffix: Discount method suffix (e.g., '_fixed_low'), used for adoption only.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_adoption_col('iraRef_mp3_', 'heating', 'benefit', 'central', 'inmap', 'acs')
        'iraRef_mp3_heating_benefit_central_inmap_acs'
        >>> create_adoption_col('iraRef_mp3_', 'heating', 'adoption', 'central', 'inmap', 'acs', '_fixed_low')
        'iraRef_mp3_heating_adoption_central_inmap_acs_fixed_low'
        >>> create_adoption_col('iraRef_mp3_', 'heating', 'health_sensitivity')
        'iraRef_mp3_heating_health_sensitivity'
    """
    if column_type == 'health_sensitivity':
        return f'{scenario_prefix}{category}_health_sensitivity'
    elif column_type == 'benefit':
        return f'{scenario_prefix}{category}_benefit_{scc_assumption}_{rcm_model}_{cr_function}'
    elif column_type == 'adoption':
        return f'{scenario_prefix}{category}_adoption_{scc_assumption}_{rcm_model}_{cr_function}{method_suffix}'
    elif column_type == 'impact':
        return f'{scenario_prefix}{category}_impact_{scc_assumption}_{rcm_model}_{cr_function}'
    else:
        raise ValueError(f"Invalid column_type '{column_type}'. Must be one of: "
                        "'benefit', 'adoption', 'impact', 'health_sensitivity'")


def create_total_npv_col(
    scenario_prefix: str,
    category: str,
    scc_assumption: str,
    rcm_model: str = None,
    cr_function: str = None,
    discount_rate: str = None,
    method_suffix: str = '',
    climate_only: bool = False) -> str:
    """Build total NPV column name (private + public).
    
    Args:
        scenario_prefix: Scenario prefix (e.g., 'iraRef_mp3_').
        category: Equipment category (e.g., 'heating').
        scc_assumption: SCC assumption ('lower', 'central', 'upper').
        rcm_model: RCM model (required unless climate_only=True).
        cr_function: CR function (required unless climate_only=True).
        discount_rate: Discount rate (e.g., 'fixed_low') for some variants.
        method_suffix: Method suffix for discount variations.
        climate_only: If True, builds climate-only total NPV column.
    
    Returns:
        Column name string.
    
    Examples:
        >>> create_total_npv_col('iraRef_mp3_', 'heating', 'central', 'inmap', 'acs', method_suffix='_fixed_low')
        'iraRef_mp3_heating_total_npv_central_inmap_acs_fixed_low'
        >>> create_total_npv_col('iraRef_mp3_', 'heating', 'central', climate_only=True, discount_rate='fixed_3pct')
        'iraRef_mp3_heating_total_npv_climateOnly_central_fixed_3pct'
    """
    if climate_only:
        return f'{scenario_prefix}{category}_total_npv_climateOnly_{scc_assumption}_{discount_rate}'
    return f'{scenario_prefix}{category}_total_npv_{scc_assumption}_{rcm_model}_{cr_function}{method_suffix}'
