import pandas as pd

from cmu_tare_model.constants import (
    VERBOSE,
    PRIVATE_DISCOUNTING_METHOD_SUFFIXES,
)
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.column_names import (
    create_npv_case_col,
    NPV_CASE_CATEGORIES,
)
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_new_columns_to_dataframe,
    apply_final_masking,
)

from cmu_tare_model.utils.calculation_utils import (
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
) -> pd.DataFrame:
    """
    Flag economic adopters for each of the three heat-pump NPV cases.

    A home is an "economic adopter" for a given NPV case if that case's private
    incremental NPV (moreWTP framing) is >= 0: the extra upfront cost of the heat
    pump over a like-for-like baseline replacement is recovered from energy-bill
    savings over the equipment lifetime, with no help from monetized climate or
    health damages. Break-even (NPV exactly 0) counts as adoption.

    One adopter column is produced per NPV case (see NPV_CASE_CATEGORIES):
    heating_only, heating_and_cooling_savings, and heating_and_cooling_full.

    Homes with invalid heating data or not scheduled for this measure package
    receive NaN -- not 0.0 -- so they are excluded from both numerator and
    denominator when computing adoption rates. This keeps row alignment with all
    upstream metrics.

    Args:
        df: DataFrame with the per-case moreWTP private NPV columns.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario for electricity grid projections.
            Accepted value: '2025 Reference Case'.
        discount_rate_col_name: Discount rate column name for private discounting.
        cost_scenario: Cost methodology key ('v3' or 'v4LOW/MID/HIGH'). Determines
            the REMDB suffix on column names (default 'v4MID').
        verbose: Enable detailed output (default: False).

    Returns:
        DataFrame with three economic-adopter columns appended and masked, one per
        NPV case. 1.0 = recovers incremental cost; 0.0 = valid home that does not;
        NaN = excluded home (invalid heating data or not in this package).

    Raises:
        ValueError: If policy_scenario is invalid.
        KeyError: If any required moreWTP private NPV column is missing.
    """
    # --- Input validation: fail fast with a clear message ---
    # Single-scenario design: only the 2025 Reference Case is modeled.
    valid_scenarios = ['2025 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(
            f"Invalid policy_scenario: '{policy_scenario}'. Must be one of {valid_scenarios}"
        )

    # --- Setup: copy data, derive column-name parts from inputs ---
    df_copy = df.copy()
    df_copy = fix_duplicate_columns(df_copy)

    # scenario_prefix converts (menu_mp, policy_scenario) -> the column-name prefix
    # (e.g., 'ref2025_mp3').  Never hardcode 'mp3' or 'ref2025' directly.
    scenario_prefix, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    method_suffix = PRIVATE_DISCOUNTING_METHOD_SUFFIXES[discount_rate_col_name]

    # Every NPV case is a heat-pump retrofit of the heating system, so the
    # economic-adopter columns are masked under the 'heating' category.
    _, valid_mask, _, _ = initialize_validation_tracking(
        df_copy, 'heating', menu_mp, verbose=verbose, copy=False
    )

    # Build the moreWTP NPV column name for each of the three NPV cases.
    moreWTP_cols = {
        npv_case: create_npv_case_col(
            scenario_prefix=scenario_prefix,
            npv_case=npv_case,
            wtp='moreWTP',
            cost_scenario=cost_scenario,
            method_suffix=method_suffix,
        )
        for npv_case in NPV_CASE_CATEGORIES
    }

    # Verify all three required NPV columns exist before doing any work.
    _validate_required_columns(
        df=df_copy,
        required_columns=list(moreWTP_cols.values()),
        context_params={
            'Analysis': 'Economic Adopter',
            'Method': method_suffix,
            'Policy': policy_scenario,
        },
    )

    if verbose:
        print(f"\nEconomic Adopter Analysis: {policy_scenario} | three NPV cases")

    # --- Generate all three case adopter columns in one block ---
    # Climate/health damages are deliberately absent from this decision: only
    # the private moreWTP NPV enters the >= 0 test.
    all_columns_to_mask = {'heating': []}
    df_new_columns = pd.DataFrame(index=df_copy.index)
    econ_adopter_cols = []

    for npv_case, moreWTP_col in moreWTP_cols.items():
        # Output name: swap 'private_npv_moreWTP' -> 'econ_adopter_moreWTP'.
        econ_adopter_col = moreWTP_col.replace(
            'private_npv_moreWTP', 'econ_adopter_moreWTP'
        )

        # Coerce stray strings to NaN so the comparison is numeric.
        df_copy[moreWTP_col] = pd.to_numeric(df_copy[moreWTP_col], errors='coerce')

        # Valid homes start at 0.0 (non-adopter); excluded homes stay NaN.
        # float64 (0.0/1.0) avoids a pandas FutureWarning and lets adoption-rate
        # helpers treat 1.0 as an adopter (1.0 == True).
        df_new_columns[econ_adopter_col] = create_retrofit_only_series(
            df_copy, valid_mask
        )
        df_new_columns.loc[valid_mask, econ_adopter_col] = 0.0

        # THE RULE: economic adopter if moreWTP NPV >= 0 (break-even counts).
        economic_adopter_mask = (
            valid_mask
            & df_copy[moreWTP_col].notna()
            & (df_copy[moreWTP_col] >= 0)
        )
        df_new_columns.loc[economic_adopter_mask, econ_adopter_col] = 1.0

        econ_adopter_cols.append(econ_adopter_col)

    # Attach the three columns and re-apply include_heating masking.
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy,
        df_new_columns,
        'heating',
        econ_adopter_cols,
        all_columns_to_mask,
    )

    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)

    return df_copy
