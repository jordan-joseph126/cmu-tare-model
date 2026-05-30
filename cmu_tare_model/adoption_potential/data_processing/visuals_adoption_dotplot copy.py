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
    'National': '#000000',      # black
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
    xlim_margin: float = 12.0,
    show_homes_annotation: bool = True,
    fuel_counts_millions: Optional[Dict[str, float]] = None,
    ytick_label_style: str = 'detailed',
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
    xlim_margin : float
        Extra horizontal margin (in data units) added to each side of the
        plot so split-cluster labels at x=0% and x=100% have room to
        render without clipping. Default 12 works for ``annotation_fontsize=7``.
        Bump to 14 (or higher) when using larger annotation fonts; pair
        with a wider ``panel_width`` if labels still touch (see README).
    ytick_label_style : str
        Controls y-axis tick label verbosity.  Two options:

        ``'detailed'`` (default) — multi-line labels including home counts
        and percentage breakdowns (e.g. ``"Electricity — LMI\n12.5/25.6 M
        Homes (48.8% Fuel)"``).  Preserves prior behavior; all existing
        callers that omit this argument are unaffected.

        ``'simple'`` — bare ``fuel — grouping`` strings only (e.g.
        ``"Electricity — LMI"``), with no ``\n`` or counts.  Useful for
        manuscript figures where home counts appear in a separate table.

        Any other value raises ``ValueError``.

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
            x_val = float(row['iraref_pct'])
            row_data.append({
                'tier': row['tier_label'],
                'x': x_val,
                'delta': float(row['delta_pct']),
                'homes_m': float(row['weighted_homes_millions']),
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
                # Any 2-marker cluster — push leftmost label LEFT, rightmost
                # label RIGHT. Position-independent: works for edge clusters
                # ([0%, 2%], [100%, 100%]), middle clusters ([33%, 67%]),
                # AND clusters that straddle the x=10 or x=90 boundary
                # (e.g., [7%, 17%] in the MP4 National row). The previous
                # edge_low/edge_high/middle classification produced a
                # fall-through bug for boundary-spanning clusters.
                cluster[0]['shift']  = 'cluster_left'
                cluster[-1]['shift'] = 'cluster_right'
            elif len(cluster) > 2:
                # 3+ markers: handle middle-only clusters; mixed/edge
                # 3-marker clusters fall through to default positioning
                # (rare; would need vertical-stagger to handle properly).
                if all(10 <= item['x'] <= 90 for item in cluster):
                    cluster[0]['shift']  = 'cluster_left'
                    cluster[-1]['shift'] = 'cluster_right'
                    for mid in cluster[1:-1]:
                        mid['shift'] = 'center'
                else:
                    for item in cluster:
                        item['shift'] = None

        all_x: List[float] = []
        for item in row_data:
            tier = item['tier']
            marker = TIER_MARKERS.get(tier, 'o')
            x_val = item['x']
            delta = item['delta']
            homes_m = item['homes_m']

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
            all_x.append(x_val)

            # Choose horizontal alignment and x-offset.
            if item.get('shift') == 'cluster_left':
                # Leftmost marker of any 2-marker cluster — label goes LEFT.
                ha = 'right'
                x_text = -14
            elif item.get('shift') == 'cluster_right':
                # Rightmost marker of any 2-marker cluster — label goes RIGHT.
                ha = 'left'
                x_text = 14
            elif x_val < 10:
                ha = 'left'
                x_text = 16
            elif x_val > 90:
                ha = 'right'
                x_text = -16
            else:
                ha = 'center'
                x_text = 0

            # Annotation above: "X% (+Y%)"
            ira_val = x_val
            sign = '+' if delta >= 0 else ''
            ann_text = f'{ira_val:.0f}% ({sign}{delta:.0f}%)'
            top_offset = annotation_y_offset_pts
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

            # Annotation below: "X.XM (+Y.YM)" absolute homes count.
            # Optional — disable via ``show_homes_annotation=False`` for
            # tight manuscript figures. The homes count is also in the
            # y-axis label, so this is informationally redundant.
            if show_homes_annotation:
                ira_homes = ira_val / 100.0 * homes_m
                delta_homes = delta / 100.0 * homes_m
                sign_h = '+' if delta_homes >= 0 else ''
                ann_homes = f'{ira_homes:.1f}M ({sign_h}{delta_homes:.1f}M)'
                bottom_offset = -annotation_y_offset_pts
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
    # Margin is parameterized via ``xlim_margin`` so callers can scale it
    # with annotation_fontsize. Default 12 works for fontsize=7; use ~14
    # for fontsize=12. Ticks (0, 20, ..., 100) are unaffected.
    ax.set_xlim(-xlim_margin, 100 + xlim_margin)
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
        Y-axis row order (top → bottom).  Defaults to :data:`GROUPING_ORDER`.
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
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"[OK] Dotplot saved: {output_path}")

    return fig
