"""Reusable functions for post-TARE peak load analysis.

Extracted from calculate_postTARE_ts_aws_peak_demand.ipynb (Phase 2 BSQ refactor).
Used by the notebook and by the national loop (Step 9).

Author: Jordan M. Joseph, PhD — Carnegie Mellon University
"""

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from cmu_tare_model.constants import BLDG_ID_COL, FIGURE_DPI
from cmu_tare_model.utils.column_names import BASE_CASE_NPV_CASE


def gisjoin_to_fips(gisjoin: str) -> str:
    """Convert a GISJOIN county identifier to a 5-digit FIPS code.

    GISJOIN format: G + 2-digit state FIPS + 0 + 3-digit county FIPS.
    Example: 'G4200030' → '42003'.

    Args:
        gisjoin: GISJOIN string from the EUSS ``in.county`` column.

    Returns:
        5-digit county FIPS code as a string.

    Raises:
        ValueError: If *gisjoin* is shorter than 7 characters.
    """
    if len(gisjoin) < 7:
        raise ValueError(
            f"GISJOIN string too short ({len(gisjoin)} chars): '{gisjoin}'. "
            f"Expected format 'G##0###' (≥7 chars)."
        )
    return gisjoin[1:3] + gisjoin[4:7]


def find_adoption_column(
    df: pd.DataFrame,
    mp: int,
    cost_scenario: str,
    discount_rate_key: str = "fixed_base",
    npv_case: str = BASE_CASE_NPV_CASE,
) -> str:
    """Locate the economic-adopter column in a TARE output DataFrame.

    Builds the expected column name using ``create_adoption_col`` for the
    2025 Reference Case with the given NPV case and discount rate, then
    checks whether that column exists in the DataFrame.  Falls back to
    listing candidates if the exact column is absent.

    Args:
        df: TARE output DataFrame (one row per building).
        mp: Measure-package number (e.g. 3 or 4).
        cost_scenario: Retained for caller compatibility; not used to build
            the column name (the cost-scenario token was removed from
            output column names in the July 2026 refactor).
        discount_rate_key: Discount rate variant key (e.g. ``'fixed_base'``).
        npv_case: One of the NPV cases in NPV_CASE_CATEGORIES. Defaults to
            ``BASE_CASE_NPV_CASE`` (``'heatingLCC_coolingLCC_unsub'``) from
            column_names.py -- the study base case: unsubsidized, with both the
            heating and cooling replacement costs credited in the NPV. Pass this
            argument by keyword at every call site so a positional slip cannot
            silently substitute another case.

    Returns:
        The matched column name string.

    Raises:
        KeyError: If no matching adoption column is found.
    """
    from cmu_tare_model.utils.column_names import create_adoption_col

    expected = create_adoption_col(
        scenario_prefix=f"ref2025_mp{mp}_",
        npv_case=npv_case,
        method_suffix=f"_{discount_rate_key}",
    )
    if expected in df.columns:
        return expected

    # Fallback: list any economic-adopter columns to aid debugging.
    adopter_candidates = [c for c in df.columns if "econ_adopter" in c.lower()]
    if adopter_candidates:
        raise KeyError(
            f"Expected adoption column '{expected}' not found.\n"
            f"  Candidates containing 'econ_adopter' ({len(adopter_candidates)}):\n"
            + "\n".join(f"    - {c}" for c in adopter_candidates)
        )
    raise KeyError(
        f"Expected adoption column '{expected}' not found, "
        f"and no columns containing 'econ_adopter' exist."
    )


def extract_adopter_ids(
    df_tare: pd.DataFrame,
    adoption_col: str,
    tier_1_value: str = "Tier 1: Feasible",
    tier_2_value: str = "Tier 2: Feasible vs. Alternative",
) -> dict[str, dict[str, list[int]]]:
    """Build per-county adopter ID dictionary from a TARE output DataFrame.

    For each county (identified via the ``county`` GISJOIN column or
    ``in.county`` column), extracts building IDs for Tier 1, Tier 2,
    constrained (T1 + T2), and all filtered buildings.

    Args:
        df_tare: TARE output DataFrame with ``bldg_id`` as index.
        adoption_col: Name of the adoption-tier column.
        tier_1_value: String label for Tier 1 in the adoption column.
        tier_2_value: String label for Tier 2 in the adoption column.

    Returns:
        Nested dict keyed by 5-digit FIPS string → sub-dict with keys
        ``'tier1'``, ``'tier2'``, ``'constrained'``, ``'all_filtered'``.
    """
    # Detect county column
    if "county" in df_tare.columns:
        county_col_name = "county"
    elif "in.county" in df_tare.columns:
        county_col_name = "in.county"
    else:
        raise KeyError(
            f"Neither 'county' nor 'in.county' found in TARE DataFrame.\n"
            f"  Available columns containing 'county': "
            f"{[c for c in df_tare.columns if 'county' in c.lower()]}"
        )

    df_work = df_tare[[county_col_name, adoption_col]].copy()
    df_work["county_fips"] = df_work[county_col_name].apply(gisjoin_to_fips)

    result: dict[str, dict[str, list[int]]] = {}
    for fips, grp in df_work.groupby("county_fips"):
        bldg_ids = grp.index.tolist()
        tier_vals = grp[adoption_col]
        tier1_ids = grp.index[tier_vals == tier_1_value].tolist()
        tier2_ids = grp.index[tier_vals == tier_2_value].tolist()
        result[str(fips)] = {
            "tier1": tier1_ids,
            "tier2": tier2_ids,
            "constrained": tier1_ids + tier2_ids,
            "all_filtered": bldg_ids,
        }
    return result


def compute_county_scenario_profile(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    adopter_bldg_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute hourly baseline and scenario demand profiles for one county.

    Args:
        df_baseline: Columns [bldg_id, hour, baseline_kwh]. 8,760 rows per building.
            Values are weight-applied (kWh × BSQ weight) from aggregate_timeseries.
        df_upgrade: Columns [bldg_id, hour, retrofit_kwh]. May be a subset of baseline.
            Values are weight-applied (kWh × BSQ weight) from aggregate_timeseries.
        adopter_bldg_ids: Buildings that adopt the retrofit.

    Returns:
        (df_profile, peak_dict) where df_profile has
        [hour, baseline_mw, scenario_mw, delta_mw].
    """
    adopter_set: set[int] = set(adopter_bldg_ids)
    all_baseline_bldgs: set[int] = set(df_baseline[BLDG_ID_COL].unique())
    upgrade_bldgs: set[int] = set(df_upgrade[BLDG_ID_COL].unique())

    adopters_missing_upgrade = adopter_set - upgrade_bldgs
    if adopters_missing_upgrade:
        print(
            f"  {len(adopters_missing_upgrade):,d} adopter bldg_ids "
            f"have no upgrade data — using baseline."
        )
    effective_adopters: set[int] = adopter_set & upgrade_bldgs

    # Left-join baseline ← upgrade
    df_merged: pd.DataFrame = df_baseline.merge(
        df_upgrade[[BLDG_ID_COL, "hour", "retrofit_kwh"]],
        on=[BLDG_ID_COL, "hour"],
        how="left",
    )

    # Vectorized adopter mask
    is_effective_adopter = df_merged[BLDG_ID_COL].isin(effective_adopters)
    retrofit_filled = df_merged["retrofit_kwh"].fillna(df_merged["baseline_kwh"])
    df_merged["scenario_kwh"] = np.where(
        is_effective_adopter, retrofit_filled, df_merged["baseline_kwh"]
    )

    # Aggregate across buildings → hourly county profile (MW)
    # BSQ values are already weight-applied, so just ÷ 1000 → MW
    df_profile: pd.DataFrame = (
        df_merged.groupby("hour", as_index=False)
        .agg(
            baseline_kwh=("baseline_kwh", "sum"),
            scenario_kwh=("scenario_kwh", "sum"),
        )
    )
    df_profile["baseline_mw"] = df_profile["baseline_kwh"] / 1000.0
    df_profile["scenario_mw"] = df_profile["scenario_kwh"] / 1000.0
    df_profile["delta_mw"] = df_profile["scenario_mw"] - df_profile["baseline_mw"]
    df_profile = df_profile[["hour", "baseline_mw", "scenario_mw", "delta_mw"]]

    if len(df_profile) != 8760:
        raise ValueError(
            f"Expected 8,760 hourly rows, got {len(df_profile):,d}. "
            f"Hour range: {df_profile['hour'].min()}..{df_profile['hour'].max()}"
        )

    peak_dict: dict[str, Any] = {
        "peak_hour_baseline": int(
            df_profile.loc[df_profile["baseline_mw"].idxmax(), "hour"]
        ),
        "peak_hour_scenario": int(
            df_profile.loc[df_profile["scenario_mw"].idxmax(), "hour"]
        ),
        "baseline_peak_mw": float(df_profile["baseline_mw"].max()),
        "scenario_peak_mw": float(df_profile["scenario_mw"].max()),
        "delta_mw": float(
            df_profile["scenario_mw"].max() - df_profile["baseline_mw"].max()
        ),
        "n_adopters": len(effective_adopters),
        "n_total_buildings": len(all_baseline_bldgs),
    }

    return df_profile, peak_dict


def plot_demand_panel(
    ax: Any,
    df_profile: pd.DataFrame,
    peak_result: dict[str, Any],
    mp: int,
    scenario_label: str,
    county_name: str = "Allegheny County, PA",
) -> None:
    """Plot baseline and scenario demand timeseries on a single axes panel.

    Args:
        ax: Matplotlib Axes to draw on.
        df_profile: DataFrame output of ``compute_county_scenario_profile``.
        peak_result: Dict output of ``compute_county_scenario_profile``.
        mp: Measure-package number.
        scenario_label: Human-readable scenario name.
        county_name: County name shown in the title.
    """
    ax.plot(df_profile["hour"], df_profile["baseline_mw"],
            color="tab:red", linewidth=0.8, alpha=0.5)
    ax.plot(df_profile["hour"], df_profile["scenario_mw"],
            color="tab:blue", linewidth=0.8, alpha=0.5)

    # Horizontal dashed lines mark each series' peak MW so panels sharing a
    # y-axis can be compared by eye. What solid vs. dashed and red vs. blue
    # mean is explained once in the shared figure legend built by the
    # notebook cell that calls this function, not repeated as per-panel text
    # (the old text-box annotations overlapped and were hard to read).
    ax.axhline(y=peak_result["baseline_peak_mw"], color="tab:red",
               linestyle="--", linewidth=2.0, alpha=0.85)
    ax.axhline(y=peak_result["scenario_peak_mw"], color="tab:blue",
               linestyle="--", linewidth=2.0, alpha=0.85)

    # Black open circle marks the exact (hour, MW) point where each series
    # peaks -- the dashed line shows the peak height across the whole panel,
    # this marker pins down exactly when it happens.
    ax.plot(peak_result["peak_hour_baseline"], peak_result["baseline_peak_mw"],
            marker="o", markerfacecolor="none", markeredgecolor="black",
            markersize=11, markeredgewidth=2.0, linestyle="none", zorder=5)
    ax.plot(peak_result["peak_hour_scenario"], peak_result["scenario_peak_mw"],
            marker="o", markerfacecolor="none", markeredgecolor="black",
            markersize=11, markeredgewidth=2.0, linestyle="none", zorder=5)

    ax.set_xlabel("Hour of Year", fontsize=17)
    ax.set_ylabel("Demand (MW)", fontsize=17)
    ax.tick_params(labelsize=15)


def plot_county_demand_grid(
    df_profiles_by_mp: Dict[int, Dict[str, pd.DataFrame]],
    peak_results_by_mp: Dict[int, Dict[str, Dict[str, Any]]],
    selected_mps: list,
    *,
    mp_labels: Optional[Dict[int, str]] = None,
    county_display_name: str = "Allegheny County, PA",
    save_figure: bool = False,
    output_dir: Optional[str] = None,
    figure_dpi: int = FIGURE_DPI,
) -> plt.Figure:
    """Draw the MP x scenario demand-profile grid with a shared legend.

    Rows are measure packages (in ``selected_mps`` order); columns are the two
    adoption scenarios, economic adopters left and 100% adoption right. All
    panels share a y-axis so peak MW is directly comparable across both
    scenarios and both measure packages.

    This consolidates two duplicate 2x2 notebook blocks that differed only in
    row/column axis ordering, font sizes, and a shared legend -- consolidated
    2 Sep 2026 during the notebook/codebase cleanup session onto the second,
    later block (row=MP, column=scenario, with the shared legend), which was
    the more complete of the two. The superseded first block (row=scenario,
    column=MP, no shared legend, smaller fonts) was not kept.

    Args:
        df_profiles_by_mp: ``{mp: {'100pct': df_profile, 'constrained':
            df_profile}}`` -- outputs of ``compute_county_scenario_profile``,
            one pair per measure package.
        peak_results_by_mp: ``{mp: {'100pct': peak_dict, 'constrained':
            peak_dict}}`` -- the matching peak dicts from the same calls.
        selected_mps: Measure-package numbers to render as rows, in order.
        mp_labels: Row label per MP (e.g. ``{3: 'MIN-efficiency ASHP
            Retrofit'}``). Defaults to the MP3/MP4 labels this notebook uses.
        county_display_name: County name shown in the legend box title.
        save_figure: If True and ``output_dir`` is set, save the figure.
        output_dir: Directory the figure is saved under (the file goes in
            ``output_dir/outputs/``); required when ``save_figure`` is True.
        figure_dpi: Resolution used when saving.

    Returns:
        The matplotlib Figure.

    Raises:
        ValueError: If save_figure is True but output_dir is None.
    """
    if save_figure and output_dir is None:
        raise ValueError("output_dir is required when save_figure=True.")

    if mp_labels is None:
        mp_labels = {
            3: "MIN-efficiency ASHP Retrofit",
            4: "HIGH-efficiency ASHP Retrofit",
        }

    scenarios = ["constrained", "100pct"]
    scenario_labels = ["ONLY Economic Adopters", "100% Adoption"]
    subplot_title_fontsize = 18
    tick_label_fontsize = 16

    # Month x-axis: hour is hour-of-year. Ticks land at the first hour of each
    # month (non-leap year, matching the 8,760-row profile). Hours are
    # cumulative, so this is built once from the days-per-month table rather
    # than hardcoding twelve hour offsets.
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _starts_days, _running = [], 0
    for _d in days_in_month:
        _starts_days.append(_running)
        _running += _d
    month_start_hours = [s * 24 for s in _starts_days]

    # Scope the white style to THIS figure only via sns.axes_style, independent
    # of any global sns.set_theme(style=...) the caller has set, so this
    # figure's styling stays fixed even if that global changes later.
    with sns.axes_style("white"):
        # sharey='all' -> all panels share a single y-scale, so peak MW is
        # directly comparable across both scenarios and both measure packages.
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey='all')
        fig.patch.set_facecolor("white")

        for row_idx, mp in enumerate(selected_mps):
            for col_idx, (scenario, scenario_label) in enumerate(
                    zip(scenarios, scenario_labels)):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("white")
                df_profile = df_profiles_by_mp[mp][scenario]
                peak_result = peak_results_by_mp[mp][scenario]
                plot_demand_panel(ax, df_profile, peak_result, mp, scenario_label)
                ax.set_title(
                    f"{mp_labels.get(mp, f'MP{mp}')} | {scenario_label}",
                    fontsize=subplot_title_fontsize,
                )

                # --- Override x-axis to months + enlarge tick labels ---
                h0 = df_profile["hour"].min()
                ax.set_xticks([h0 + m for m in month_start_hours])
                ax.set_xticklabels(month_labels)
                ax.set_xlim(h0, h0 + 8760)
                ax.set_xlabel("Month", fontsize=17)
                ax.tick_params(labelsize=tick_label_fontsize)

        # --- Shared legend, bottom center, drawn as a fancy box ---
        # Proxy handles only (no data) -- the real lines/markers are drawn per
        # panel by plot_demand_panel. The black peak-X marker is left out of
        # the legend on purpose (self-evident on the panels, and a fifth
        # entry would add a row and compress the figure) -- only solid vs.
        # dashed and red vs. blue are explained here.
        # Order is [solid_red, dashed_red, solid_blue, dashed_blue] so that
        # matplotlib's column-major legend fill (with ncol=2) lays them out as
        # two rows -- row 1 solid red/blue, row 2 dashed red/blue -- matching
        # "Solid Red | Solid Blue" then "Dashed Red | Dashed Blue".
        legend_handles = [
            Line2D([0], [0], color="tab:red", linewidth=2.5, linestyle="-"),
            Line2D([0], [0], color="tab:red", linewidth=2.5, linestyle="--"),
            Line2D([0], [0], color="tab:blue", linewidth=2.5, linestyle="-"),
            Line2D([0], [0], color="tab:blue", linewidth=2.5, linestyle="--"),
        ]
        legend_labels = [
            "Existing HVAC",
            "Peak Existing HVAC",
            "Post-Retrofit",
            "Peak Post-Retrofit",
        ]
        fig_legend = fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=2,
            fontsize=16,
            title=f"Residential Electricity Load (MW) for {county_display_name}",
            title_fontsize=17,
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="0.3",
            framealpha=0.95,
            borderpad=1.1,
            labelspacing=0.9,
            handlelength=2.5,
        )
        fig_legend.get_title().set_fontweight("bold")

        # Extra bottom margin so the legend box has room below the panels.
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        if save_figure:
            out_path = os.path.join(
                output_dir,
                "outputs",
                f"allegheny_demand_profiles_MP"
                f"{'_'.join(str(m) for m in selected_mps)}.png",
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=figure_dpi, bbox_inches="tight")
            print(f"[OK] Figure saved: {out_path}")
        plt.show()

    return fig


# =============================================================================
# The non-time-aligned peak-load summary path (compute_peak_load_summary,
# build_adopter_ids_for_scope, prompt_peak_load_scope, print_peak_load_summary,
# and the private _print_seasonal_block helper) was moved out of this module
# on 2 Sep 2026, during the notebook/codebase cleanup session -- the live
# notebook's grid-impact cells had already migrated to the time-aligned
# BuildStockQuery hourly-profile approach above (compute_county_scenario_profile
# + plot_demand_panel), and a full-repo grep found zero importers of the moved
# functions outside this file. They are kept, not deleted, at
# cmu_tare_model/grid_impact/archived_files/peak_load_functions_legacy.py.
# =============================================================================
