"""Horizontal dot-plot visualization for heat pump adoption potential.

Plots adoption tiers (Already Upgraded, T1, T1+T2, T1+T2+T3) for the
IRA-Reference scenario with delta annotations relative to Pre-IRA,
disaggregated by incumbent fuel type and income group.

Layout: 2x2 grid — MP rows x Case A / Case B columns.
"""

import os
from typing import Dict, List, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

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
    'figure.dpi': 300,
})

# ===========================================================================
# COLOUR & MARKER CONFIGURATION
# ===========================================================================
FUEL_COLORS: Dict[str, str] = {
    'National': '#7f7f7f',      # gray
    'Electricity': '#2ca02c',   # green
    'Natural Gas': '#1f77b4',   # blue
    'Fuel Oil': '#d62728',      # red
    'Propane': '#ff7f0e',       # orange/yellow
}

# Marker for each tier (left-to-right on the dot plot)
TIER_MARKERS: Dict[str, str] = {
    'Tier 1: Feasible': 'o',                                     # circle
    'Total Adoption Potential': '^',                              # triangle
    'Total Adoption Potential (Additional Subsidy)': 'D',         # diamond
}

TIER_LABELS_SHORT: Dict[str, str] = {
    'Tier 1: Feasible': 'T1: Feasible',
    'Total Adoption Potential': 'T1+T2: Adoption Potential',
    'Total Adoption Potential (Additional Subsidy)': 'T1+T2+T3: With Subsidy',
}

# Tiers extracted from the MultiIndex DataFrame (excludes Already Upgraded)
MI_TIER_NAMES: List[str] = [
    'Tier 1: Feasible',
    'Total Adoption Potential',
    'Total Adoption Potential (Additional Subsidy)',
]

# All tiers (plot order: left -> right)
ALL_TIER_NAMES: List[str] = [
    'Tier 1: Feasible',
    'Total Adoption Potential',
    'Total Adoption Potential (Additional Subsidy)',
]

# ===========================================================================
# Y-AXIS GROUPING ORDER  (top -> bottom)
# ===========================================================================
GROUPING_ORDER: List[str] = [
    'National \u2014 Overall',
    'Electricity \u2014 Overall',
    'Electricity \u2014 LMI',
    'Natural Gas \u2014 Overall',
    'Natural Gas \u2014 LMI',
    'Fuel Oil \u2014 Overall',
    'Fuel Oil \u2014 LMI',
    'Propane \u2014 Overall',
    'Propane \u2014 LMI',
]


# ===================================================================
# prepare_plot_data
# ===================================================================

def prepare_plot_data(
    mi_df: pd.DataFrame,
    source_df: pd.DataFrame,
    preira_col: str,
    iraref_col: str,
    fuel_col: str = 'base_heating_fuel',
    income_col: str = 'lmi_or_mui',
    income_groups: Optional[List[str]] = None,
    sample_total: Optional[int] = None,
    scaling_factor: float = 242.0,
) -> pd.DataFrame:
    """Flatten MultiIndex adoption DataFrame into plot-ready long format.

    Returns one row per (grouping, tier) with IRA-Reference adoption %,
    Pre-IRA adoption %, and the delta.  Also computes "Already Upgraded"
    percentages directly from *source_df*.

    Parameters
    ----------
    mi_df : pd.DataFrame
        MultiIndex DataFrame from ``create_multiIndex_adoption_df()``.
    source_df : pd.DataFrame
        Raw per-dwelling DataFrame used for population weights and
        "Already Upgraded" calculation.
    preira_col, iraref_col : str
        Scenario column names for Pre-IRA and IRA-Reference.
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
        iraref_pct, preira_pct, delta_pct, sample_n, pct_of_sample,
        weighted_homes_millions.
    """
    if income_groups is None:
        income_groups = ['LMI']
    if sample_total is None:
        sample_total = len(source_df)

    group_counts = source_df.groupby([fuel_col, income_col], observed=True).size()
    total_homes = int(group_counts.sum())
    fuel_counts = source_df.groupby(fuel_col, observed=True).size()

    fuels_in_data = list(dict.fromkeys(f for f, _ in mi_df.index))

    rows: List[Dict] = []

    def _make_row(grouping, fuel, income_level, tier, iraref_val, preira_val,
                  n):
        return {
            'grouping': grouping,
            'fuel_type': fuel,
            'income_level': income_level,
            'tier_label': tier,
            'iraref_pct': iraref_val,
            'preira_pct': preira_val,
            'delta_pct': iraref_val - preira_val,
            'sample_n': n,
            'pct_of_sample': 100.0 * n / sample_total if sample_total else 0.0,
            'weighted_homes_millions': n * scaling_factor / 1_000_000,
        }

    # ------------------------------------------------------------------
    # Per fuel x selected income-group rows
    # ------------------------------------------------------------------
    for fuel, income in mi_df.index:
        if income not in income_groups:
            continue

        grouping = f'{fuel} \u2014 {income}'
        n = int(group_counts.get((fuel, income), 0))

        # MI tiers
        for tier in MI_TIER_NAMES:
            iraref_val = float(mi_df.loc[(fuel, income), (iraref_col, tier)])
            preira_val = float(mi_df.loc[(fuel, income), (preira_col, tier)])
            rows.append(_make_row(grouping, fuel, income, tier,
                                  iraref_val, preira_val, n))

    # ------------------------------------------------------------------
    # Per-fuel "Overall" rows (population-weighted)
    # ------------------------------------------------------------------
    for fuel in fuels_in_data:
        fuel_n = int(fuel_counts.get(fuel, 0))
        income_levels = [inc for f, inc in mi_df.index if f == fuel]

        # MI tiers - weighted
        for tier in MI_TIER_NAMES:
            ira_w = 0.0
            pre_w = 0.0
            for income in income_levels:
                ni = int(group_counts.get((fuel, income), 0))
                w = ni / fuel_n if fuel_n else 0.0
                ira_w += w * float(mi_df.loc[(fuel, income), (iraref_col, tier)])
                pre_w += w * float(mi_df.loc[(fuel, income), (preira_col, tier)])
            rows.append(_make_row(f'{fuel} \u2014 Overall', fuel, 'Overall',
                                  tier, ira_w, pre_w, fuel_n))

    # ------------------------------------------------------------------
    # National - Overall
    # ------------------------------------------------------------------
    for tier in MI_TIER_NAMES:
        ira_w = 0.0
        pre_w = 0.0
        for fuel, income in mi_df.index:
            ni = int(group_counts.get((fuel, income), 0))
            w = ni / total_homes if total_homes else 0.0
            ira_w += w * float(mi_df.loc[(fuel, income), (iraref_col, tier)])
            pre_w += w * float(mi_df.loc[(fuel, income), (preira_col, tier)])
        rows.append(_make_row('National \u2014 Overall', 'National', 'Overall',
                              tier, ira_w, pre_w, total_homes))

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
    fuel_counts_millions: Optional[Dict[str, float]] = None,
) -> plt.Axes:
    """Draw a horizontal dot plot showing IRA-Reference adoption with deltas.

    Each y-axis row displays 4 markers (Already Upgraded + 3 tiers),
    colour-coded by fuel type.  Above each marker a text annotation
    shows ``X (+Y)`` where X = IRA-Ref % and Y = change from Pre-IRA.
    For Already Upgraded, just the percentage is shown (no delta).

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

    Returns
    -------
    plt.Axes
    """
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

        all_x: List[float] = []

        for _, row in subset.iterrows():
            tier = row['tier_label']
            marker = TIER_MARKERS.get(tier, 'o')
            x_val = row['iraref_pct']

            # All markers are filled (IRA-Reference only)
            ax.scatter(
                x_val, y,
                marker=marker,
                s=marker_size,
                facecolors=color,
                edgecolors=color,
                linewidths=marker_linewidth,
                zorder=3,
            )
            all_x.append(float(x_val))

            # Annotation above: "X% (+Y%)"
            ira_val = row['iraref_pct']
            delta = row['delta_pct']
            homes_m = row['weighted_homes_millions']
            sign = '+' if delta >= 0 else ''
            ann_text = f'{ira_val:.0f}% ({sign}{delta:.0f}%)'

            ax.annotate(
                ann_text,
                xy=(x_val, y),
                xytext=(0, annotation_y_offset_pts),
                textcoords='offset points',
                fontsize=annotation_fontsize,
                ha='center',
                va='bottom',
                color=color,
                clip_on=True,
            )

            # Annotation below: "X.XM (+Y.YM)" absolute homes count
            ira_homes = ira_val / 100.0 * homes_m
            delta_homes = delta / 100.0 * homes_m
            sign_h = '+' if delta_homes >= 0 else ''
            ann_homes = f'{ira_homes:.1f}M ({sign_h}{delta_homes:.1f}M)'

            ax.annotate(
                ann_homes,
                xy=(x_val, y),
                xytext=(0, -annotation_y_offset_pts),
                textcoords='offset points',
                fontsize=annotation_fontsize,
                ha='center',
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
                # National — Overall: just show total homes
                label = f'National \u2014 Overall\n{national_total:.1f} M Homes (100%)'
            elif income_level == 'Overall':
                # e.g. "Electricity (25.6/80.0M, 32.0% Total)"
                pct_total = (fuel_total_m / national_total * 100) if national_total else 0.0
                label = f'{fuel} \u2014 Overall\n{fuel_total_m:.1f}/{national_total:.1f} M Homes ({pct_total:.1f}% Total)'
            else:
                # LMI row — e.g. "Electricity \u2014 LMI (12.5/25.6M, 48.8% Fuel)"
                pct_fuel = (homes_m / fuel_total_m * 100) if fuel_total_m else 0.0
                label = f'{fuel} \u2014 LMI\n{homes_m:.1f}/{fuel_total_m:.1f} M Homes ({pct_fuel:.1f}% Fuel)'
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
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('Adoption Potential (%)', fontsize=ytick_fontsize)

    # --- Grid and separator ---
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=grid_alpha, linewidth=0.5)

    national_y = y_positions.get('National \u2014 Overall')
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
