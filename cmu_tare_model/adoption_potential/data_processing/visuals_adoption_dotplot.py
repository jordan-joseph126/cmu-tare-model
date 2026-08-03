"""Horizontal dot-plot visualization for heat pump economic adoption potential.

``build_econ_plot_df`` supports two marker modes via ``shape_by``:

  'replacement_credit_scenario' (default) -- three markers per row, one per
    replacement-credit scope, in display order (heating, cooling, both):
      heatingLCC_coolingSavings (heating replacement credited only)
      heatingSavings_coolingLCC (cooling replacement credited only)
      heatingLCC_coolingLCC     (both heating + cooling replacements)
    Each marker plots the rate for the selected rebate vintage
    (rebate_vintage: 'unsub' = unsubsidized, 'sub' = December 2024,
    'sub_june2026' = June 2026); the delta annotation is that rate minus the
    unsubsidized rate for the scope (0 when rebate_vintage='unsub').

  'rebate_policy_scenario' -- three markers per row, one per rebate policy
    scenario (Unsubsidized, 2024 Rebate, 2026 Rebate w/o fuel switching), for a
    single fixed replacement-credit scope. Each marker plots
    that scenario's own adoption rate. Pair with REBATE_POLICY_SCENARIO_MARKERS
    and build_rebate_policy_scenario_legend_handles().

The labels are vintage-based (rebate-eligibility era), not program-mix based, so
they stay correct now that both the December 2024 and June 2026 vintages model
HEEHR + HOMES.

Disaggregated by incumbent heating fuel and income group.
Layout: one panel per measure package (MP rows).
"""

import os
from typing import Dict, List, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from cmu_tare_model.constants import FIGURE_DPI
from cmu_tare_model.utils.column_names import create_adoption_col
from cmu_tare_model.utils.modeling_params import define_scenario_params

# ===========================================================================
# APPEARANCE CONFIGURATION
# ===========================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': FIGURE_DPI,
})

# ===========================================================================
# COLOUR & MARKER CONFIGURATION
# ===========================================================================
FUEL_COLORS: Dict[str, str] = {
    'National': '#000000',      # black
    'Electricity': '#2ca02c',   # green
    'Natural Gas': '#1f77b4',   # blue
    'Fuel Oil': '#d62728',      # red
    'Propane': '#ff7f0e',       # orange/yellow
}

# Marker for the single economic-adopter tier
TIER_MARKERS: Dict[str, str] = {
    'Economic Adopter': 'o',
    'Total Adoption Potential': '^',
}

TIER_LABELS_SHORT: Dict[str, str] = {
    'Economic Adopter': 'Economic Adopter (NPV >= 0)',
    'Total Adoption Potential': 'Total Adoption Potential',
}

# Tiers extracted from the MultiIndex DataFrame for dotplot lookup
MI_TIER_NAMES: List[str] = [
    'Economic Adopter',
    'Total Adoption Potential',
]

# All tiers (plot order: left -> right)
ALL_TIER_NAMES: List[str] = [
    'Economic Adopter',
    'Total Adoption Potential',
]

# ===========================================================================
# REBATE-POLICY-SCENARIO MODE  (three markers: unsub / 2024 / June 2026)
# ===========================================================================
# Used when build_econ_plot_df(shape_by='rebate_policy_scenario'). Each token is
# the trailing segment of the NPV-case column name ('_unsub', '_sub',
# '_sub_june2026'); the label is what appears on the marker legend. This is the
# rebate policy scenario axis (which rebate policy applies), a different axis
# from the replacement_credit_scenario (which replacement cost the NPV credits).
REBATE_POLICY_SCENARIO_ORDER: List[str] = ['unsub', 'sub', 'sub_june2026']
REBATE_POLICY_SCENARIO_LABELS: Dict[str, str] = {
    'unsub': 'Unsubsidized',
    'sub': '2024 Rebate',
    'sub_june2026': '2026 Rebate w/o fuel switching',
}

# Marker shape per rebate policy scenario, keyed by the label that
# build_econ_plot_df writes into 'tier_label' so plot_adoption_panel can look it
# up via custom_tier_markers. Distinct from the two-shape TIER_MARKERS used by
# the default replacement_credit_scenario mode.
# June 2026 is the headline pick in this figure, so it gets the square shape;
# with fill_markers=False every marker is drawn as an empty outline, so the
# headline is set apart by its square shape alone (see plot_adoption_panel).
REBATE_POLICY_SCENARIO_MARKERS: Dict[str, str] = {
    'Unsubsidized': 'o',                        # circle
    '2024 Rebate': '^',                         # triangle
    '2026 Rebate w/o fuel switching': 's',      # square (headline)
}

# ===========================================================================
# REPLACEMENT-CREDIT-SCENARIO MODE  (three markers: heating / cooling / both)
# ===========================================================================
# Used when build_econ_plot_df(shape_by='replacement_credit_scenario') (the
# default): each row gets one marker per replacement-credit scope, each plotting
# the subsidized rate. These constants are the single source of truth for the
# case labels and marker shapes so the notebook plot cell does not define its own
# floating copies (which previously drifted out of sync -- the legend read
# "Heating Repl. Credit Only" while the marker key was "Heating Repl. Credit").
# The labels here match the case labels build_econ_plot_df writes into
# 'tier_label'.
# Ordered (scope token, marker label) pairs driving the three markers, in
# display order (heating, cooling, both). build_econ_plot_df loops this list so
# the plot, the case list, and the legend stay in one order.
REPLACEMENT_CREDIT_SCOPES: List[tuple] = [
    ('heatingLCC_coolingSavings', 'Heating Replacement Cost Offset'),
    ('heatingSavings_coolingLCC', 'Cooling Replacement Cost Offset'),
    ('heatingLCC_coolingLCC', 'Heating + Cooling Replacement Cost Offset'),
]
REPLACEMENT_CREDIT_CASES: List[str] = [
    label for _scope, label in REPLACEMENT_CREDIT_SCOPES
]
# The "both replacements" scope is the headline pick in this figure, so it gets
# the square shape; with fill_markers=False every marker is drawn as an empty
# outline, so the headline is set apart by its square shape alone (see
# plot_adoption_panel).
REPLACEMENT_CREDIT_MARKERS: Dict[str, str] = {
    'Heating Replacement Cost Offset': 'o',           # circle -- heatingLCC_coolingSavings
    'Cooling Replacement Cost Offset': '^',           # triangle -- heatingSavings_coolingLCC
    'Heating + Cooling Replacement Cost Offset': 's',  # square -- heatingLCC_coolingLCC (headline)
}

# ===========================================================================
# NATIONAL + PER-FUEL "OVERALL" GROUPING ORDER  (top -> bottom)
# ===========================================================================
# Shared y-axis order for the national/per-fuel dotplots (no LMI split). Both
# the replacement-credit and rebate-policy plot cells use this instead of a
# floating local copy.
NATIONAL_FUEL_GROUPING_ORDER: List[str] = [
    'National -- Overall',
    'Electricity -- Overall',
    'Natural Gas -- Overall',
    'Fuel Oil -- Overall',
    'Propane -- Overall',
]

# ===========================================================================
# Y-AXIS GROUPING ORDER  (top -> bottom)
# ===========================================================================
GROUPING_ORDER: List[str] = [
    'National -- Overall',
    'Electricity -- Overall',
    'Electricity -- LMI',
    'Natural Gas -- Overall',
    'Natural Gas -- LMI',
    'Fuel Oil -- Overall',
    'Fuel Oil -- LMI',
    'Propane -- Overall',
    'Propane -- LMI',
]


# ===================================================================
# build_econ_plot_df
# ===================================================================

def build_econ_plot_df(
    source_df: pd.DataFrame,
    mp: int,
    cost_scenario: str = 'v4MID',
    discount_rate: str = 'fixed_base',
    fuel_col: str = 'base_heating_fuel',
    income_col: str = 'lmi_or_mui',
    income_groups: Optional[List[str]] = None,
    scaling_factor: float = 242.0,
    shape_by: str = 'replacement_credit_scenario',
    fixed_replacement_credit_scenario: str = 'heatingLCC_coolingSavings',
    rebate_vintage: str = 'sub',
) -> pd.DataFrame:
    """Build a DataFrame for the economic adoption dotplot.

    Args:
        source_df: TARE output frame for one measure package.
        mp: Measure-package number.
        cost_scenario: Retained for caller compatibility; not embedded in
            column names post-refactor.
        discount_rate: Discount-rate key used to build the method suffix.
        fuel_col: Column holding the baseline heating fuel.
        income_col: Column holding the income group.
        income_groups: Income groups to break out (default ['LMI']).
        scaling_factor: Fallback homes-per-sample weight, used only when
            source_df has no 'weight' column. When a weight column is present the
            weighted-homes column is derived from the actual weight sum instead.
        shape_by: Which axis the marker shape encodes.
            'replacement_credit_scenario' (default) emits three rows per grouping
            -- one per replacement-credit scope (heating replacement only,
            cooling replacement only, both) -- each plotting its subsidized rate
            with the unsubsidized delta. 'rebate_policy_scenario' emits three
            rows per grouping -- one per rebate policy scenario (unsubsidized,
            2024, June 2026) -- for a single fixed replacement-credit scope.
        fixed_replacement_credit_scenario: The replacement-credit scope held fixed
            when shape_by='rebate_policy_scenario'. One of
            'heatingLCC_coolingSavings' or 'heatingLCC_coolingLCC'. Ignored in the
            default mode.
        rebate_vintage: Which rebate vintage each scope marker plots in the
            default replacement_credit_scenario mode. 'unsub' is the
            unsubsidized rate (no rebate); 'sub' is the December 2024 rate;
            'sub_june2026' is the June 2026 rate. With 'unsub' the marker value
            is the unsubsidized rate and the (unshown) subsidy delta is 0.
            Ignored when shape_by='rebate_policy_scenario' (that mode plots all
            three vintages).

    Returns:
        DataFrame formatted for ``plot_adoption_panel()``.

    Raises:
        ValueError: If shape_by, fixed_replacement_credit_scenario, or
            rebate_vintage is invalid.
    """
    valid_shape_by = ('replacement_credit_scenario', 'rebate_policy_scenario')
    if shape_by not in valid_shape_by:
        raise ValueError(
            f"shape_by={shape_by!r} is not valid. Choose one of {valid_shape_by}.")
    valid_vintage = ('unsub', 'sub', 'sub_june2026')
    if rebate_vintage not in valid_vintage:
        raise ValueError(
            f"rebate_vintage={rebate_vintage!r} is not valid. "
            f"Choose one of {valid_vintage}.")
    valid_fixed = ('heatingLCC_coolingSavings', 'heatingLCC_coolingLCC')
    if (shape_by == 'rebate_policy_scenario'
            and fixed_replacement_credit_scenario not in valid_fixed):
        raise ValueError(
            f"fixed_replacement_credit_scenario="
            f"{fixed_replacement_credit_scenario!r} is not valid. "
            f"Choose one of {valid_fixed}.")

    if income_groups is None:
        income_groups = ['LMI']

    scenario_prefix = define_scenario_params(mp)[0]
    method_suffix = f'_{discount_rate}'

    # replacement_credit_scenario mode: one marker per credit scope. The 'sub'
    # slot holds the plotted value -- the rate for the selected rebate vintage.
    # When rebate_vintage='unsub' this resolves to the unsubsidized column, so
    # the marker plots the unsubsidized rate and the delta below is 0. The
    # 'unsub' slot is always the delta reference. Keyed by scope token so
    # _append_group can loop them.
    scope_cols = {
        scope: {
            'sub': create_adoption_col(
                scenario_prefix, f'{scope}_{rebate_vintage}', method_suffix),
            'unsub': create_adoption_col(
                scenario_prefix, f'{scope}_unsub', method_suffix),
        }
        for scope, _label in REPLACEMENT_CREDIT_SCOPES
    }

    # rebate_policy_scenario mode: three markers, one per rebate policy scenario,
    # for the single fixed replacement-credit scope. Each marker plots that
    # scenario's own adoption rate (no subsidized/unsubsidized delta pairing).
    rps_cols = {
        token: create_adoption_col(
            scenario_prefix,
            f'{fixed_replacement_credit_scenario}_{token}',
            method_suffix)
        for token in REBATE_POLICY_SCENARIO_ORDER
    }

    def _rate(df_sub: pd.DataFrame, col: str) -> float:
        return df_sub[col].mean() * 100.0 if col in df_sub.columns else np.nan

    def _row(
        grouping: str,
        fuel: str,
        income_level: str,
        case_label: str,
        right_pct: float,
        left_pct: float,
        n: int,
        weighted_homes_millions: float,
    ) -> dict:
        return {
            'grouping': grouping,
            'fuel_type': fuel,
            'income_level': income_level,
            'tier_label': case_label,
            'case_b_pct': right_pct,
            'case_a_pct': left_pct,
            'delta_pct': right_pct - left_pct,
            'sample_n': n,
            'pct_of_sample': 100.0 * n / len(source_df) if len(source_df) else 0.0,
            'weighted_homes_millions': weighted_homes_millions,
        }

    rows: List[dict] = []

    def _weighted_homes_millions(sub_df: pd.DataFrame, n: int) -> float:
        """Weighted homes (millions) for a grouping.

        Uses the actual household weight sum when a 'weight' column is present --
        the same weight-derived approach the notebook uses for fuel_counts -- so
        the per-group homes annotation is consistent with the y-axis fuel totals.
        Falls back to the sample count times the scaling_factor default only when
        no weight column exists.
        """
        if 'weight' in sub_df.columns:
            return sub_df['weight'].sum() / 1_000_000
        return n * scaling_factor / 1_000_000

    def _append_group(
        grouping: str,
        fuel: str,
        income_level: str,
        sub_df: pd.DataFrame,
        n: int,
    ) -> None:
        """Append this grouping's rows for the active shape_by mode."""
        homes_m = _weighted_homes_millions(sub_df, n)
        if shape_by == 'replacement_credit_scenario':
            # One row per credit scope, in REPLACEMENT_CREDIT_SCOPES order. The
            # plotted value is the scope's subsidized rate; the delta is that
            # rate minus its unsubsidized rate.
            for scope, label in REPLACEMENT_CREDIT_SCOPES:
                rows.append(_row(
                    grouping, fuel, income_level, label,
                    _rate(sub_df, scope_cols[scope]['sub']),
                    _rate(sub_df, scope_cols[scope]['unsub']), n, homes_m))
        else:
            # One row per rebate policy scenario. The marker value is that
            # scenario's own adoption rate, so left and right pct are equal
            # (delta 0 -- there is no subsidized/unsubsidized pairing here).
            for token in REBATE_POLICY_SCENARIO_ORDER:
                rate = _rate(sub_df, rps_cols[token])
                rows.append(_row(
                    grouping, fuel, income_level,
                    REBATE_POLICY_SCENARIO_LABELS[token], rate, rate, n,
                    homes_m))

    sample_total = len(source_df)
    group_counts = source_df.groupby([fuel_col, income_col], observed=True).size()
    fuel_counts = source_df.groupby(fuel_col, observed=True).size()
    fuels_in_data = list(source_df[fuel_col].dropna().unique())

    for (fuel, income), n in group_counts.items():
        if income not in income_groups:
            continue
        sub = source_df[(source_df[fuel_col] == fuel) & (source_df[income_col] == income)]
        _append_group(f'{fuel} -- {income}', fuel, income, sub, int(n))

    for fuel in fuels_in_data:
        fuel_n = int(fuel_counts.get(fuel, 0))
        fuel_sub = source_df[source_df[fuel_col] == fuel]
        _append_group(f'{fuel} -- Overall', fuel, 'Overall', fuel_sub, fuel_n)

    _append_group(
        'National -- Overall', 'National', 'Overall', source_df, sample_total)

    return pd.DataFrame(rows)


# ===================================================================
# prepare_plot_data
# ===================================================================

def prepare_plot_data(
    mi_df_a: pd.DataFrame,
    source_df: pd.DataFrame,
    case_a_col: str,
    case_b_col: str,
    mi_df_b: Optional[pd.DataFrame] = None,
    fuel_col: str = 'base_heating_fuel',
    income_col: str = 'lmi_or_mui',
    income_groups: Optional[List[str]] = None,
    sample_total: Optional[int] = None,
    scaling_factor: float = 242.0,
) -> pd.DataFrame:
    """Flatten two MultiIndex adoption DataFrames into plot-ready long format.

    Returns one row per (grouping, tier) with right-case adoption %,
    left-case adoption %, and delta (right - left).

    For Option A: pass the same DataFrame as both mi_df_a and mi_df_b, with
    case_a_col = heatingLCC_coolingSavings adopter column (left/reference)
    and case_b_col = heatingLCC_coolingLCC adopter column (right/comparison).

    Parameters
    ----------
    mi_df_a : pd.DataFrame
        MultiIndex DataFrame for the left/reference case from
        ``create_multiIndex_adoption_df()``.
    source_df : pd.DataFrame
        Raw per-dwelling DataFrame used for population weights.
    case_a_col : str
        Adopter column name for the left/reference case
        (e.g., heatingLCC_coolingSavings adopter column).
    case_b_col : str
        Adopter column name for the right/comparison case
        (e.g., heatingLCC_coolingLCC adopter column).
    mi_df_b : pd.DataFrame, optional
        MultiIndex DataFrame for the right/comparison case. If None, uses
        mi_df_a for both (delta will be zero -- for single-case display).
    fuel_col, income_col : str
        Column names in *source_df*.
    income_groups : list of str, optional
        Income sub-groups to show (default ``['LMI']``).
    sample_total : int, optional
        Denominator for "% of sample" (default ``len(source_df)``).
    scaling_factor : float
        Sample-to-national multiplier (default 242).

    Returns
    -------
    pd.DataFrame
        Columns: grouping, fuel_type, income_level, tier_label,
        case_b_pct, case_a_pct, delta_pct, sample_n, pct_of_sample,
        weighted_homes_millions.
    """
    if income_groups is None:
        income_groups = ['LMI']
    if sample_total is None:
        sample_total = len(source_df)
    if mi_df_b is None:
        mi_df_b = mi_df_a

    group_counts = source_df.groupby([fuel_col, income_col], observed=True).size()
    total_homes = int(group_counts.sum())
    fuel_counts = source_df.groupby(fuel_col, observed=True).size()

    fuels_in_data = list(dict.fromkeys(f for f, _ in mi_df_a.index))

    rows: List[Dict] = []

    def _make_row(grouping, fuel, income_level, tier, case_b_val, case_a_val,
                  n):
        return {
            'grouping': grouping,
            'fuel_type': fuel,
            'income_level': income_level,
            'tier_label': tier,
            'case_b_pct': case_b_val,
            'case_a_pct': case_a_val,
            'delta_pct': case_b_val - case_a_val,
            'sample_n': n,
            'pct_of_sample': 100.0 * n / sample_total if sample_total else 0.0,
            'weighted_homes_millions': n * scaling_factor / 1_000_000,
        }

    # ------------------------------------------------------------------
    # Per fuel x selected income-group rows
    # ------------------------------------------------------------------
    for fuel, income in mi_df_a.index:
        if income not in income_groups:
            continue

        grouping = f'{fuel} \u2014 {income}'
        n = int(group_counts.get((fuel, income), 0))

        for tier in MI_TIER_NAMES:
            case_b_val = float(mi_df_b.loc[(fuel, income), (case_b_col, tier)])
            case_a_val = float(mi_df_a.loc[(fuel, income), (case_a_col, tier)])
            rows.append(_make_row(grouping, fuel, income, tier,
                                  case_b_val, case_a_val, n))

    # ------------------------------------------------------------------
    # Per-fuel "Overall" rows (population-weighted)
    # ------------------------------------------------------------------
    for fuel in fuels_in_data:
        fuel_n = int(fuel_counts.get(fuel, 0))
        income_levels = [inc for f, inc in mi_df_a.index if f == fuel]

        for tier in MI_TIER_NAMES:
            b_w = 0.0
            a_w = 0.0
            for income in income_levels:
                ni = int(group_counts.get((fuel, income), 0))
                w = ni / fuel_n if fuel_n else 0.0
                b_w += w * float(mi_df_b.loc[(fuel, income), (case_b_col, tier)])
                a_w += w * float(mi_df_a.loc[(fuel, income), (case_a_col, tier)])
            rows.append(_make_row(f'{fuel} \u2014 Overall', fuel, 'Overall',
                                  tier, b_w, a_w, fuel_n))

    # ------------------------------------------------------------------
    # National - Overall
    # ------------------------------------------------------------------
    for tier in MI_TIER_NAMES:
        b_w = 0.0
        a_w = 0.0
        for fuel, income in mi_df_a.index:
            ni = int(group_counts.get((fuel, income), 0))
            w = ni / total_homes if total_homes else 0.0
            b_w += w * float(mi_df_b.loc[(fuel, income), (case_b_col, tier)])
            a_w += w * float(mi_df_a.loc[(fuel, income), (case_a_col, tier)])
        rows.append(_make_row('National \u2014 Overall', 'National', 'Overall',
                              tier, b_w, a_w, total_homes))

    return pd.DataFrame(rows)


# ===================================================================
# Legend helper
# ===================================================================

def _build_legend_handles() -> List[mlines.Line2D]:
    """Create legend handles for tier shapes (IRA-Reference only)."""
    handles: List[mlines.Line2D] = []
    for tier in ALL_TIER_NAMES:
        marker = TIER_MARKERS[tier]
        handles.append(
            mlines.Line2D(
                [], [],
                marker=marker,
                color='none',
                markerfacecolor='gray',
                markeredgecolor='gray',
                markersize=8,
                linestyle='None',
                label=TIER_LABELS_SHORT[tier],
            )
        )
    return handles


def build_rebate_policy_scenario_legend_handles(
    filled_label: Optional[str] = None,
    fill_markers: bool = True,
) -> List[mlines.Line2D]:
    """Create legend handles for the three rebate policy scenario markers.

    Companion to _build_legend_handles for build_econ_plot_df's
    shape_by='rebate_policy_scenario' mode: one handle per rebate policy
    scenario (unsubsidized, 2024, June 2026) in plot order, using
    REBATE_POLICY_SCENARIO_MARKERS so the shapes on the plot match the legend.
    The notebook cell passes these handles to ax.legend.

    Args:
        filled_label: The one scenario label to draw filled (gray). Every other
            handle is drawn as an empty outline, matching plot_adoption_panel's
            filled_tier. When None, all handles are filled (old behavior).
        fill_markers: When False, every handle is drawn as an empty outline
            regardless of filled_label -- matching plot_adoption_panel's
            fill_markers=False all-outline mode, where the headline is set apart
            by its square shape alone. Default True keeps the old behavior.

    Returns:
        List of matplotlib Line2D legend handles, one per rebate policy scenario.
    """
    handles: List[mlines.Line2D] = []
    for token in REBATE_POLICY_SCENARIO_ORDER:
        label = REBATE_POLICY_SCENARIO_LABELS[token]
        if not fill_markers:
            face = 'none'
        else:
            face = 'gray' if filled_label is None or label == filled_label else 'none'
        handles.append(
            mlines.Line2D(
                [], [],
                marker=REBATE_POLICY_SCENARIO_MARKERS[label],
                color='none',
                markerfacecolor=face,
                markeredgecolor='gray',
                markersize=8,
                linestyle='None',
                label=label,
            )
        )
    return handles


def build_replacement_credit_legend_handles(
    filled_case: Optional[str] = None,
    fill_markers: bool = True,
) -> List[mlines.Line2D]:
    """Create legend handles for the three replacement-credit markers.

    Companion to build_econ_plot_df's default
    shape_by='replacement_credit_scenario' mode: one handle per
    replacement-credit scope (heating, cooling, both) in plot order, using
    REPLACEMENT_CREDIT_MARKERS so the shapes on the plot match the legend. Each
    legend label equals its marker key exactly, which is what the earlier inline
    notebook legend got wrong ("Heating Repl. Credit Only" vs the "Heating Repl.
    Credit" marker key).

    Args:
        filled_case: The one scope label to draw filled (gray). Every other
            handle is drawn as an empty outline, matching plot_adoption_panel's
            filled_tier. When None, all handles are filled (old behavior).
        fill_markers: When False, every handle is drawn as an empty outline
            regardless of filled_case -- matching plot_adoption_panel's
            fill_markers=False all-outline mode, where the headline is set apart
            by its square shape alone. Default True keeps the old behavior.

    Returns:
        List of matplotlib Line2D legend handles, one per replacement-credit
        scope.
    """
    handles: List[mlines.Line2D] = []
    for case in REPLACEMENT_CREDIT_CASES:
        if not fill_markers:
            face = 'none'
        else:
            face = 'gray' if filled_case is None or case == filled_case else 'none'
        handles.append(
            mlines.Line2D(
                [], [],
                marker=REPLACEMENT_CREDIT_MARKERS[case],
                color='none',
                markerfacecolor=face,
                markeredgecolor='gray',
                markersize=8,
                linestyle='None',
                label=case,
            )
        )
    return handles


# ===================================================================
# plot_adoption_panel
# ===================================================================

def plot_adoption_panel(
    plot_df: pd.DataFrame,
    ax: plt.Axes,
    grouping_order: Optional[List[str]] = None,
    title: str = '',
    marker_size: int = 60,
    ytick_fontsize: int = 8,
    ytick_linespacing: float = 1.4,
    title_fontsize: int = 11,
    connector_alpha: float = 0.35,
    connector_linewidth: float = 1.0,
    marker_linewidth: float = 1.5,
    grid_alpha: float = 0.3,
    separator_alpha: float = 0.5,
    annotation_fontsize: int = 7,
    annotation_y_offset_pts: float = 8.0,
    cluster_stagger_pts: float = 9.0,
    annotation_x_offset_pts: float = 14.0,
    xlim_margin: float = 12.0,
    show_delta_annotation: bool = False,
    show_homes_annotation: bool = True,
    fuel_counts_millions: Optional[Dict[str, float]] = None,
    ytick_label_style: str = 'detailed',
    custom_tier_markers: Optional[Dict[str, str]] = None,
    filled_tier: Optional[str] = None,
    fill_markers: bool = True,
    homes_unit: str = 'M',
) -> plt.Axes:
    """Draw a horizontal dot plot showing adoption rates and deltas between cases.

    Each y-axis row displays one marker per NPV case, colour-coded by fuel type.
    Above each marker a text annotation shows ``X% (+Y%)`` where X is the adoption
    rate and Y is the subsidy delta (subsidized vs unsubsidized) when enabled.
    By default the plot shows only the adoption rate, not the delta.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Output of ``prepare_plot_data()``.
    ax : plt.Axes
        Matplotlib Axes to draw on.
    grouping_order : list of str, optional
        Y-axis label order (top -> bottom).  Defaults to GROUPING_ORDER.
    title : str
        Panel title.
    annotation_fontsize : int
        Font size for the ``X (+Y)`` labels above markers. Default 7.
    annotation_y_offset_pts : float
        Vertical offset in points for annotations. Default 8.
    cluster_stagger_pts : float
        Extra vertical offset in points added per stagger level for clusters
        of 3 or more near-equal markers, so their labels stack in a ladder
        instead of overprinting. Default 9 (about one line height at
        ``annotation_fontsize=7``). Only affects 3+ marker clusters; single
        markers and 2-marker split clusters are unchanged.
    xlim_margin : float
        Extra horizontal margin (in data units) added to each side of the
        plot so split-cluster labels at x=0% and x=100% have room to
        render without clipping. Default 12 works for ``annotation_fontsize=7``.
        Bump to 14 (or higher) when using larger annotation fonts; pair
        with a wider ``panel_width`` if labels still touch (see README).
    ytick_label_style : str
        Controls y-axis tick label verbosity.  Two options:

        ``'detailed'`` (default)  --  multi-line labels including home counts
        and percentage breakdowns (e.g. ``"Electricity  --  LMI\n12.5/25.6 M
        Homes (48.8% Fuel)"``).  Preserves prior behavior; all existing
        callers that omit this argument are unaffected.

        ``'simple'``  --  bare ``fuel  --  grouping`` strings only (e.g.
        ``"Electricity  --  LMI"``), with no ``\n`` or counts.  Useful for
        manuscript figures where home counts appear in a separate table.

        Any other value raises ``ValueError``.
    filled_tier : str, optional
        The one case label (a ``tier_label`` value) to draw filled. Every other
        marker in the row is drawn as an empty outline in the fuel colour. Use
        this to highlight the headline case. When None (default) all markers are
        filled, which keeps the old behaviour for other callers.
    fill_markers : bool
        When False, every marker is drawn as an empty outline regardless of
        ``filled_tier`` -- the headline case is then set apart by its marker
        shape alone. Default True keeps the old behaviour (``filled_tier`` fills
        the headline; None fills all).

    Returns
    -------
    plt.Axes
    """
    if ytick_label_style not in ('detailed', 'simple'):
        raise ValueError(
            f"ytick_label_style={ytick_label_style!r} is not valid. "
            "Choose 'detailed' (default) or 'simple'."
        )

    if grouping_order is None:
        grouping_order = GROUPING_ORDER

    y_order = list(reversed(grouping_order))
    y_positions = {name: i for i, name in enumerate(y_order)}

    # --- Draw markers, connecting lines, and annotations ---
    for grouping in grouping_order:
        subset = plot_df[plot_df['grouping'] == grouping]
        if subset.empty:
            continue

        y = y_positions[grouping]
        fuel = subset['fuel_type'].iloc[0]
        color = FUEL_COLORS.get(fuel, '#333333')

        row_data = []
        for _, row in subset.iterrows():
            x_val = float(row['case_b_pct'])
            row_data.append({
                'tier': row['tier_label'],
                'x': x_val,
                'delta': float(row['delta_pct']),
                'homes_m': float(row['weighted_homes_millions']),
                'stagger': 0,
            })

        row_data.sort(key=lambda r: r['x'])
        for item in row_data:
            if item['x'] < 10:
                item['shift'] = 'edge_right'
            elif item['x'] > 90:
                item['shift'] = 'edge_left'
            else:
                item['shift'] = 'center'

        # Markers within this many percentage points of each other are
        # treated as a visual cluster and given split labels. Bumped from
        # 10 to 15 so near-equal pairs (e.g. 4% & 5%, 0% & 2%) are caught.
        close_threshold = 15.0
        clusters: List[List[dict]] = []
        current_cluster: List[dict] = []
        for item in row_data:
            if not current_cluster:
                current_cluster.append(item)
                continue
            if item['x'] - current_cluster[-1]['x'] < close_threshold:
                current_cluster.append(item)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]
        if current_cluster:
            clusters.append(current_cluster)

        for cluster in clusters:
            if len(cluster) == 2:
                # Any 2-marker cluster  --  push leftmost label LEFT, rightmost
                # label RIGHT. Position-independent: works for edge clusters
                # ([0%, 2%], [100%, 100%]), middle clusters ([33%, 67%]),
                # AND clusters that straddle the x=10 or x=90 boundary
                # (e.g., [7%, 17%] in the MP4 National row). The previous
                # edge_low/edge_high/middle classification produced a
                # fall-through bug for boundary-spanning clusters.
                cluster[0]['shift']  = 'cluster_left'
                cluster[-1]['shift'] = 'cluster_right'
            elif len(cluster) > 2:
                # 3+ markers sit too close for centered labels to separate.
                # Spread them left/right instead of stacking them vertically:
                # the leftmost label goes LEFT, the rightmost goes RIGHT, and any
                # middle markers stay centered on their own marker. The
                # xlim_margin gives the left/right labels room so they do not
                # clip at the plot edges.
                cluster[0]['shift'] = 'cluster_left'
                cluster[-1]['shift'] = 'cluster_right'
                for mid in cluster[1:-1]:
                    mid['shift'] = 'cluster_center'

        # De-duplicate labels for markers that land on the same value. Within
        # each cluster, group markers whose plotted value matches (within
        # annotation_equal_eps percentage points); the first marker in a value
        # group keeps its number and the rest are flagged so their duplicate
        # label is suppressed. The markers themselves are still drawn -- only the
        # redundant text is skipped. This catches, e.g., electric rows where the
        # 2024 and 2026 rebate rates are identical. When an entire cluster
        # collapses to one value, the single surviving label has nothing to
        # split from, so its cluster left/right shift is cleared and it falls
        # back to the default edge-aware centered placement. Near-but-distinct
        # values (e.g. 4% vs 5%) keep their left/right split.
        annotation_equal_eps = 0.5
        for cluster in clusters:
            value_groups: List[List[dict]] = []
            for item in cluster:
                if (value_groups
                        and abs(item['x'] - value_groups[-1][0]['x'])
                        <= annotation_equal_eps):
                    value_groups[-1].append(item)
                else:
                    value_groups.append([item])
            for group in value_groups:
                for duplicate in group[1:]:
                    duplicate['skip_annotation'] = True
            if len(value_groups) == 1:
                value_groups[0][0]['shift'] = 'center'

        all_x: List[float] = []
        for item in row_data:
            tier = item['tier']
            marker = (custom_tier_markers or TIER_MARKERS).get(tier, 'o')
            x_val = item['x']
            delta = item['delta']
            homes_m = item['homes_m']

            # Fill only the designated headline case (filled_tier); draw every
            # other marker as an empty outline. When filled_tier is None (other
            # callers) all markers stay filled, matching the old behavior. When
            # fill_markers is False every marker is an empty outline regardless,
            # so the headline is set apart by its marker shape alone.
            if not fill_markers:
                face = 'none'
            elif filled_tier is None or tier == filled_tier:
                face = color
            else:
                face = 'none'
            ax.scatter(
                x_val, y,
                marker=marker,
                s=marker_size,
                facecolors=face,
                edgecolors=color,
                linewidths=marker_linewidth,
                zorder=3,
            )
            all_x.append(x_val)

            # Skip the redundant label for a duplicate-value marker. The marker
            # and its x (above) are kept, so the connecting line still spans the
            # full range; only the overlapping number/homes text is suppressed.
            if item.get('skip_annotation'):
                continue

            # Choose horizontal alignment and x-offset.
            if item.get('shift') == 'cluster_left':
                # Leftmost marker of a close cluster  --  label goes LEFT.
                ha = 'right'
                x_text = -annotation_x_offset_pts
            elif item.get('shift') == 'cluster_right':
                # Rightmost marker of a close cluster  --  label goes RIGHT.
                ha = 'left'
                x_text = annotation_x_offset_pts
            elif item.get('shift') == 'cluster_center':
                # Middle marker of a 3+ cluster  --  keep the label centered on
                # its own marker even near an edge, so it does not collide with
                # the left/right labels on either side.
                ha = 'center'
                x_text = 0
            elif x_val < 10:
                ha = 'left'
                x_text = annotation_x_offset_pts
            elif x_val > 90:
                ha = 'right'
                x_text = -annotation_x_offset_pts
            else:
                ha = 'center'
                x_text = 0

            # Annotation above: always show adoption rate; show delta only when enabled.
            ira_val = x_val
            if show_delta_annotation and delta != 0:
                sign = '+' if delta >= 0 else ''
                ann_text = f'{ira_val:.0f}% ({sign}{delta:.0f}%)'
            else:
                ann_text = f'{ira_val:.0f}%'
            top_offset = annotation_y_offset_pts + item['stagger'] * cluster_stagger_pts
            ax.annotate(
                ann_text,
                xy=(x_val, y),
                xytext=(x_text, top_offset),
                textcoords='offset points',
                fontsize=annotation_fontsize,
                ha=ha,
                va='bottom',
                color=color,
                clip_on=True,
            )

            # Annotation below: "X.XM" absolute homes count; optionally show subsidy delta.
            # Optional  --  disable via ``show_homes_annotation=False`` for
            # tight manuscript figures. The homes count is also in the
            # y-axis label, so this is informationally redundant.
            if show_homes_annotation:
                ira_homes = ira_val / 100.0 * homes_m
                ann_homes = f'{ira_homes:.1f}{homes_unit}'
                if show_delta_annotation and delta != 0:
                    delta_homes = delta / 100.0 * homes_m
                    sign_h = '+' if delta_homes >= 0 else ''
                    ann_homes = (
                        f'{ira_homes:.1f}{homes_unit} '
                        f'({sign_h}{delta_homes:.1f}{homes_unit})'
                    )
                bottom_offset = -(annotation_y_offset_pts
                                  + item['stagger'] * cluster_stagger_pts)
                ax.annotate(
                    ann_homes,
                    xy=(x_val, y),
                    xytext=(x_text, bottom_offset),
                    textcoords='offset points',
                    fontsize=annotation_fontsize,
                    ha=ha,
                    va='top',
                    color=color,
                    clip_on=True,
                )

        # Connecting line
        if len(all_x) >= 2:
            ax.plot(
                [min(all_x), max(all_x)],
                [y, y],
                color=color,
                linewidth=connector_linewidth,
                alpha=connector_alpha,
                zorder=2,
            )

    # --- Y-axis tick labels ---
    y_ticks: List[int] = []
    y_labels: List[str] = []

    for grouping in y_order:
        y = y_positions[grouping]
        y_ticks.append(y)

        if ytick_label_style == 'simple':
            y_labels.append(grouping)
            continue

        sub = plot_df[plot_df['grouping'] == grouping]
        if not sub.empty:
            r = sub.iloc[0]
            fuel = r['fuel_type']
            income_level = r['income_level']
            homes_m = r['weighted_homes_millions']
            if fuel_counts_millions:
                national_total = sum(fuel_counts_millions.values())
                fuel_total_m = fuel_counts_millions.get(fuel, 0.0)
            else:
                national_total = 0.0
                fuel_total_m = 0.0

            if fuel == 'National':
                # Top-level geo row: use the actual grouping label for the title
                label = f'{grouping}\n{national_total:.1f} {homes_unit} Homes (100%)'
                # label = rf'$\mathbf{{National - Overall}}$\n{national_total:.1f} M Homes (100%)'
            elif income_level == 'Overall':
                # e.g. "Electricity  --  Overall\n25.6/80.0 M Homes\n(32.0% Total)"
                pct_total = (fuel_total_m / national_total * 100) if national_total else 0.0
                label = (
                    f'{fuel} \u2014 Overall\n'
                    # rf'$\mathbf{{{fuel} - Overall}}$\n'
                    f'{fuel_total_m:.1f}/{national_total:.1f} {homes_unit} Homes\n'
                    f'({pct_total:.1f}% Total)'
                )
                
            else:
                # LMI row  --  e.g. "Electricity \u2014 LMI\n12.5/25.6 M Homes\n(48.8% Fuel)"
                pct_fuel = (homes_m / fuel_total_m * 100) if fuel_total_m else 0.0
                label = (
                    f'{fuel} \u2014 LMI\n'
                    # rf'$\mathbf{{{fuel} - {income_level}}}$\n'
                    f'{homes_m:.1f}/{fuel_total_m:.1f} {homes_unit} Homes\n'
                    f'({pct_fuel:.1f}% Fuel)'
                )
        else:
            label = grouping
        y_labels.append(label)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        y_labels,
        fontsize=ytick_fontsize,
        linespacing=ytick_linespacing,
    )
    ax.set_ylim(-0.5, len(y_order) - 0.5)

    # --- X-axis: ticks every 20% ---
    # Margin is parameterized via ``xlim_margin`` so callers can scale it
    # with annotation_fontsize. Default 12 works for fontsize=7; use ~14
    # for fontsize=12. Ticks (0, 20, ..., 100) are unaffected.
    ax.set_xlim(-xlim_margin, 100 + xlim_margin)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('Share of households recovering electrification premium through discounted operational savings (%)', fontsize=ytick_fontsize)

    # --- Grid and separator ---
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=grid_alpha, linewidth=0.5)

    # Separator below the top row (works for any geo label, not just 'National')
    _top_grouping = grouping_order[0] if grouping_order else 'National -- Overall'
    national_y = y_positions.get(_top_grouping)
    if national_y is not None:
        ax.axhline(
            y=national_y - 0.5,
            color='gray',
            linewidth=0.8,
            linestyle='--',
            alpha=separator_alpha,
        )

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

    return ax


# ===================================================================
# plot_adoption_dotplot  (N×1 multi-panel figure)
# ===================================================================

def plot_adoption_dotplot(
    panels: List[tuple],
    fuel_counts_millions: Optional[Dict[str, float]] = None,
    grouping_order: Optional[List[str]] = None,
    panel_height: float = 7.0,
    panel_width: float = 12.0,
    save_figure: bool = False,
    output_path: Optional[str] = None,
) -> "plt.Figure":
    """Create an N×1 stacked dot-plot figure, one row per panel.

    Each panel renders one call to :func:`plot_adoption_panel`.  All rows
    share the x-axis so tick marks align.  A single legend is placed below
    the figure.

    Parameters
    ----------
    panels : list of (title, plot_df) tuples
        Each element is a *(title_str, plot_df)* pair where *plot_df* is
        the output of :func:`prepare_plot_data`.
    fuel_counts_millions : dict, optional
        ``{fuel_type: weighted_homes_millions}`` used to annotate y-axis
        labels.  Applied identically to every panel.  Include
        ``'National'`` key for the national total.
    grouping_order : list of str, optional
        Y-axis row order (top -> bottom).  Defaults to :data:`GROUPING_ORDER`.
    panel_height : float
        Height (inches) per row panel.  Default 7.
    panel_width : float
        Total figure width (inches).  Default 12.
    save_figure : bool
        If ``True`` and *output_path* is set, save the figure to disk.
    output_path : str, optional
        File path for the saved figure.

    Returns
    -------
    plt.Figure
    """
    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError("panels must contain at least one (title, plot_df) tuple")

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(panel_width, panel_height * n_panels),
        sharex=True,
    )

    # Ensure axes is always iterable
    if n_panels == 1:
        axes = [axes]

    for ax, (title, plot_df) in zip(axes, panels):
        plot_adoption_panel(
            plot_df, ax,
            grouping_order=grouping_order,
            title=title,
            fuel_counts_millions=fuel_counts_millions,
        )

    # Remove x-axis label from all but the bottom panel (sharex handles ticks)
    for ax in axes[:-1]:
        ax.set_xlabel('')

    # Shared legend below all panels
    legend_handles = _build_legend_handles()
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(legend_handles),
        fontsize=9,
        frameon=False,
    )

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

    if save_figure and output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=FIGURE_DPI)
        print(f"[OK] Dotplot saved: {output_path}")

    return fig
