import pandas as pd

from cmu_tare_model.constants import (
    UPGRADE_COLUMNS,
    VALID_HVAC_REPLACEMENT_SCENARIOS,
    VERBOSE,
    PRIVATE_DISCOUNTING_METHOD_SUFFIXES,
)
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.column_names import create_npv_col
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_new_columns_to_dataframe,
    apply_final_masking,
)

from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import (
    _validate_required_columns,
    fix_duplicate_columns,
)

__all__ = [
    "economic_adoption_decision",
]


def economic_adoption_decision(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    discount_rate_col_name: str,
    cost_scenario: str = 'v4MID',
    verbose: bool = VERBOSE,
    hvac_replacement_scenario: str = 'heating',
) -> pd.DataFrame:
    """
    Flags homes where the heat pump pays for its incremental cost from bill savings alone.

    A home is an "economic adopter" if its private incremental NPV (moreWTP framing) is
    >= 0: the extra upfront cost of a heat pump over a like-for-like baseline replacement
    is recovered from energy-bill savings over the equipment lifetime, with no help from
    monetized climate or health damages.  Break-even (NPV exactly 0) counts as adoption.

    Homes with invalid baseline data or not scheduled for this measure package receive
    NaN — not False — so they are excluded from both numerator and denominator when
    computing adoption rates.  This keeps row alignment with all upstream metrics.

    Args:
        df: DataFrame with per-building moreWTP private NPV columns.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario for electricity grid projections.
            Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
        discount_rate_col_name: Discount rate column name for private discounting.
        cost_scenario: Cost methodology key ('v3' or 'v4LOW/MID/HIGH'). Determines
            the REMDB suffix on column names (default 'v4MID').
        verbose: Enable detailed output (default: False).
        hvac_replacement_scenario: 'heating' (default, Case A) or 'heating_and_cooling'
            (Case B).  Controls which category segment appears in NPV column lookups
            and output names.

    Returns:
        DataFrame with one boolean economic-adopter column per equipment category
        appended and masked.  True = recovers incremental cost; False = valid home
        that does not; NaN = excluded home (invalid baseline or not in this package).

    Raises:
        ValueError: If policy_scenario or hvac_replacement_scenario is invalid.
        KeyError: If the required moreWTP private NPV column is missing.
    """
    # --- Input validation: fail fast with a clear message ---
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(
            f"Invalid policy_scenario: '{policy_scenario}'. Must be one of {valid_scenarios}"
        )

    if hvac_replacement_scenario not in VALID_HVAC_REPLACEMENT_SCENARIOS:
        raise ValueError(
            f"Invalid hvac_replacement_scenario: '{hvac_replacement_scenario}'. "
            f"Must be one of {VALID_HVAC_REPLACEMENT_SCENARIOS}"
        )

    # --- Setup: copy data, derive column-name parts from inputs ---
    df_copy = df.copy()
    df_copy = fix_duplicate_columns(df_copy)

    # scenario_prefix converts (menu_mp, policy_scenario) → the column-name prefix
    # (e.g., 'iraRef_mp3').  Never hardcode 'mp3' or 'iraRef' directly.
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    method_suffix = PRIVATE_DISCOUNTING_METHOD_SUFFIXES[discount_rate_col_name]
    output_category = hvac_replacement_scenario  # 'heating' or 'heating_and_cooling'

    # --- Verify the one required input column exists before doing any work ---
    # All equipment categories share the same single moreWTP NPV column
    # (the column name encodes the scenario and cost method, not the category).
    moreWTP_col = create_npv_col(
        scenario_prefix=scenario_prefix,
        category=output_category,
        wtp='moreWTP',
        cost_scenario=cost_scenario,
        method_suffix=method_suffix,
    )
    _validate_required_columns(
        df=df_copy,
        required_columns=[moreWTP_col],
        context_params={
            'Analysis': 'Economic Adopter',
            'Method': method_suffix,
            'Policy': policy_scenario,
        },
    )

    if verbose:
        print(f"\nEconomic Adopter Analysis: {policy_scenario} | {output_category}")

    # --- Per-category loop: write the boolean econ-adopter column ---
    # The loop exists so each equipment category gets its own masked column
    # aligned with the rest of the framework (heating, waterHeating, etc.).
    all_columns_to_mask = {cat: [] for cat in UPGRADE_COLUMNS}

    for category, upgrade_column in UPGRADE_COLUMNS.items():
        # valid_mask = homes with valid baseline data AND scheduled for this retrofit.
        # Homes outside valid_mask start as NaN and are never written to.
        _, valid_mask, _, category_columns_to_mask = initialize_validation_tracking(
            df_copy, category, menu_mp, verbose=verbose, copy=False
        )

        # Build the output column name from the input column name
        # (swap 'private_npv_moreWTP' → 'econ_adopter_moreWTP').
        economic_adopter_col_name = moreWTP_col.replace(
            'private_npv_moreWTP', 'econ_adopter_moreWTP'
        )
        category_columns_to_mask.append(economic_adopter_col_name)

        # Make sure the NPV column is numeric (coerce any stray strings to NaN).
        df_copy[moreWTP_col] = pd.to_numeric(df_copy[moreWTP_col], errors='coerce')

        # Initialize: valid homes start as 0.0 (not yet an adopter), excluded homes NaN.
        # The column is float64 (matching the framework), so we use 0.0 and 1.0
        # rather than bool True/False to avoid a pandas FutureWarning on assignment.
        # compute_adoption_rate(adopter_tiers=[True]) treats 1.0 as an adopter
        # because 1.0 == True in Python.
        df_new_columns = pd.DataFrame(index=df_copy.index)
        df_new_columns[economic_adopter_col_name] = create_retrofit_only_series(
            df_copy, valid_mask
        )
        df_new_columns.loc[valid_mask, economic_adopter_col_name] = 0.0  # non-adopter

        # THE RULE: a home is an economic adopter if moreWTP NPV >= 0.
        # Break-even (NPV == 0) counts — the heat pump covers its incremental cost exactly.
        economic_adopter_mask = (
            valid_mask
            & df_copy[moreWTP_col].notna()
            & (df_copy[moreWTP_col] >= 0)
        )
        df_new_columns.loc[economic_adopter_mask, economic_adopter_col_name] = 1.0  # adopter

        df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
            df_copy,
            df_new_columns,
            category,
            category_columns_to_mask,
            all_columns_to_mask,
        )

    # Final sweep: re-apply include_{category} mask to catch any accidental overwrites.
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)

    return df_copy
