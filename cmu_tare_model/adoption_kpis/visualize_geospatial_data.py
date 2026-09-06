"""
Geospatial rendering for the adoption KPI choropleths (state and county maps).

County geometry vintage is matched to the ResStock 2022.1.1 data vintage, which
carries 2010-vintage geography (pre-2023 counties). The binding constraint is
Connecticut: it replaced its eight counties (FIPS 09001-09015) with nine
planning regions (09110-09190) starting with the 2022 Census cartographic
boundary vintage, while ResStock still uses the eight county codes. Any geometry
from 2022 on finds no polygon for those codes and drops all of Connecticut from
the county join. The layer therefore uses the 2021 cartographic boundary files
(cb_2021_us_county_500k, cb_2021_us_state_500k) -- the newest clean vintage.

The vintage is set once, in data_loading.py: COUNTY_GEOMETRY_PRODUCT /
COUNTY_GEOMETRY_VINTAGE / COUNTY_GEOMETRY_SCALE for counties and the matching
STATE_GEOMETRY_* constants for the state overlay. Change those to repoint the
maps; nothing here hardcodes a vintage.
"""

import os
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    FIGURE_DPI,
    MAP_TITLE_FONT_SIZE,
    MAP_TITLE_PAD,
    MAP_CBAR_LABEL_FONT_SIZE,
    MAP_CBAR_TICK_LABEL_SIZE,
    MAP_LEGEND_FONT_SIZE,
)
from cmu_tare_model.adoption_kpis.thermal_cop import assign_breakeven_category
from cmu_tare_model.adoption_kpis.visualize_tabular_data import (
    make_symmetric_norm,
    print_column_summary,
)
from cmu_tare_model.adoption_kpis.data_loading import SHAPEFILE_PATH

# ============================================================================
# Load State Boundaries and Merge with Analysis Data
# ============================================================================

def prepare_state_geodataframe(
    gdf_states: gpd.GeoDataFrame,
    df_analysis: pd.DataFrame,
    merge_col: str = 'state',
    exclude_territories: Optional[list] = None
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Merge analysis results with state geometries and prepare for choropleth mapping.

    Handles different Census shapefile column naming conventions
    (STUSPS, STUSAB, STATE_ABBR) by auto-detecting the state abbreviation column.
    Reprojects to US Albers Equal Area (ESRI:102003) and splits into
    CONUS and Alaska GeoDataFrames for inset plotting.

    Args:
        gdf_states: GeoDataFrame of US state boundaries (any CRS).
        df_analysis: DataFrame with a state abbreviation column and analysis columns.
        merge_col: Column name in df_analysis containing 2-letter state abbreviations.
        exclude_territories: State/territory codes to exclude (default: PR, VI, GU, AS, MP).

    Returns:
        Tuple of (gdf_all, gdf_conus, gdf_alaska) — all in ESRI:102003.
    """
    if exclude_territories is None:
        exclude_territories = ['PR', 'VI', 'GU', 'AS', 'MP']

    # Auto-detect state abbreviation column
    state_col = None
    for col in ['STUSPS', 'STUSAB', 'STATE_ABBR']:
        if col in gdf_states.columns:
            state_col = col
            break

    if state_col is None:
        raise ValueError(
            f"No state abbreviation column found. Available columns: {list(gdf_states.columns)}"
        )

    # Reproject to US Albers Equal Area for accurate area representation
    gdf_states = gdf_states.to_crs('ESRI:102003')

    # Merge analysis data with geometries
    gdf = gdf_states.merge(
        df_analysis,
        left_on=state_col,
        right_on=merge_col,
        how='left'
    )

    # Filter: exclude territories and rows with no analysis data
    # Use the first non-id numeric column to detect missing data
    numeric_cols = df_analysis.select_dtypes(include='number').columns.tolist()
    filter_col = numeric_cols[0] if numeric_cols else merge_col

    gdf_filtered = gdf[
        (~gdf[state_col].isin(exclude_territories)) &
        (gdf[filter_col].notna())
    ].copy()

    gdf_alaska = gdf_filtered[gdf_filtered[state_col] == 'AK'].copy()
    gdf_conus = gdf_filtered[
        ~gdf_filtered[state_col].isin(['AK', 'HI'])
    ].copy()

    return gdf_filtered, gdf_conus, gdf_alaska


# ============================================================================
# Generalized Choropleth Map Function + Generate Maps
# ============================================================================

def create_choropleth_map(
    gdf_conus: gpd.GeoDataFrame,
    gdf_alaska: gpd.GeoDataFrame,
    column: str,
    title: str,
    cbar_label: str,
    year: int = 2024,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    dpi: int = FIGURE_DPI,
    cmap: str = 'Blues',
    norm=None,
    show_plot: bool = True
) -> Optional[str]:
    """
    Create a choropleth map with CONUS main panel and Alaska inset.

    Pass a matplotlib Normalize (e.g. TwoSlopeNorm) via `norm` for diverging
    maps; otherwise linear normalization is derived from the data automatically.
    """
    plot_kw = dict(column=column, cmap=cmap, edgecolor='black', linewidth=0.5, legend=False)

    if norm is not None:
        plot_kw['norm'] = norm
    else:
        vals = pd.concat([gdf_conus[column], gdf_alaska[column]]).dropna()
        norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
        plot_kw['vmin'], plot_kw['vmax'] = vals.min(), vals.max()

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax_main = fig.add_axes([0.02, 0.02, 0.78, 0.90])
    ax_ak = fig.add_axes([0.02, 0.02, 0.22, 0.25])

    gdf_conus.plot(ax=ax_main, **plot_kw)
    ax_main.set_axis_off()
    ax_main.set_title(title, fontsize=MAP_TITLE_FONT_SIZE, fontweight='bold', pad=MAP_TITLE_PAD)

    if not gdf_alaska.empty:
        gdf_alaska.plot(ax=ax_ak, **plot_kw)
    ax_ak.set_axis_off()

    # Colorbar — matplotlib handles tick placement natively for any norm
    cax = fig.add_axes([0.82, 0.08, 0.03, 0.74])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=cbar_label).ax.yaxis.label.set_fontsize(MAP_CBAR_LABEL_FONT_SIZE)
    cax.tick_params(labelsize=MAP_CBAR_TICK_LABEL_SIZE)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path if output_path else None


# ============================================================================
# Combined Multi-MP Choropleth Figure
# ============================================================================

def plot_combined_choropleth(
    gdf_raw: gpd.GeoDataFrame,
    data_by_mp: dict,
    column: str,
    title_template: Union[str, Dict[int, str]],
    cbar_label: str,
    cmap,
    norm,
    selected_mps: list,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = FIGURE_DPI,
    output_path: Optional[str] = None,
    save_figure: bool = False,
    geo_level: str = 'state',
    cbar_ticks: Optional[list] = None,
) -> None:
    """Render one map panel per MP side-by-side in a single figure with a shared colorbar.

    Args:
        gdf_raw: Raw boundary GeoDataFrame (any CRS). Pass a state GeoDataFrame
            when ``geo_level='state'``, or a county GeoDataFrame (TIGER/Line with
            GEOID and STATEFP columns) when ``geo_level='county'``.
        data_by_mp: Dict mapping MP number → DataFrame with a ``state`` column
            (state level) or a ``county`` GISJOIN column (county level) plus the
            target ``column``.
        column: Column name to choropleth-shade.
        title_template: Per-panel title. Either a format string with an ``{mp}``
            placeholder (e.g. ``'Demand Change — MP{mp}'``) or a dict mapping
            MP integer keys to literal title strings (e.g. ``HEATING_MP_SUBTITLES``).
            Use the dict form when titles are equipment-based (no ``{mp}`` token needed).
        cbar_label: Colorbar axis label.
        cmap: Matplotlib colormap (name string or Colormap instance).
        norm: Matplotlib Normalize instance (e.g. ``TwoSlopeNorm``,
            ``BoundaryNorm``, ``Normalize``).
        selected_mps: Ordered list of MP keys to render as panels.
        figsize: Figure size ``(width, height)`` in inches. Defaults to
            ``(10 * n + 2, 8)`` where ``n`` is the number of MPs.
        dpi: Resolution used when saving.
        output_path: File path to save the figure. Only written when
            ``save_figure=True``.
        save_figure: If ``True`` and ``output_path`` is set, save to disk.
        geo_level: ``'state'`` (default) or ``'county'``. Controls which merge
            function is used and the polygon border linewidth (0.5 for state,
            0.1 for county — 3,000+ polygons require thin borders).
        cbar_ticks: Optional explicit tick values for the colorbar. When
            provided, overrides matplotlib's default tick placement. Useful
            with ``TwoSlopeNorm`` which can produce uneven automatic ticks.
    """
    if geo_level not in ('state', 'county'):
        raise ValueError(f"geo_level must be 'state' or 'county', got {geo_level!r}")

    n = len(selected_mps)
    if figsize is None:
        figsize = (10 * n + 2, 8)

    fig = plt.figure(figsize=figsize, facecolor='white')

    # Reserve bottom for the horizontal colorbar; maps occupy the top 84%
    map_bottom = 0.12
    map_height = 0.84

    panel_w = 0.96 / n
    gap = 0.01  # tight gap — visual spacing is dominated by map aspect ratio padding

    lw = 0.1 if geo_level == 'county' else 0.5

    # --- Pre-loop setup for county maps: CONUS base layer + state border overlay ---
    if geo_level == 'county':
        # All CONUS county geometries (reprojected, AK/HI/territories excluded)
        # → rendered as dark gray base so insufficient-data counties are visible
        # and clearly distinct from any color in the diverging colormap
        gdf_base = gdf_raw.to_crs('ESRI:102003')
        _excl = _TERRITORY_FIPS | {_ALASKA_FIPS, '15'}  # AK + HI + territories
        gdf_base_conus = gdf_base[~gdf_base['STATEFP'].isin(_excl)].copy()

        # State boundary lines — loaded once, reused across all MP panels
        _gdf_states = gpd.read_file(SHAPEFILE_PATH).to_crs('ESRI:102003')
        _sc = next(
            (c for c in ['STUSPS', 'STUSAB', 'STATE_ABBR'] if c in _gdf_states.columns),
            None,
        )
        gdf_sborder_conus = _gdf_states[
            ~_gdf_states[_sc].isin(['AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP'])
        ].copy()

    for i, mp in enumerate(selected_mps):
        x0 = 0.02 + i * (panel_w + gap)
        w = panel_w - gap

        ax = fig.add_axes([x0, map_bottom, w, map_height])

        if geo_level == 'county':
            _, gdf_conus, _ = prepare_county_geodataframe(
                gdf_raw, data_by_mp[mp]
            )

            # Layer 1 — All CONUS counties: dark gray fill (insufficient-data base)
            gdf_base_conus.plot(
                ax=ax, color="#707070", edgecolor='#505050', linewidth=0.1
            )
            # Layer 2 — Data counties: colored fill
            gdf_conus.plot(
                ax=ax, column=column, cmap=cmap, norm=norm,
                edgecolor='#505050', linewidth=0.1, legend=False,
            )
            # Layer 3 — State borders on top
            gdf_sborder_conus.boundary.plot(
                ax=ax, edgecolor='black', linewidth=0.5, zorder=3
            )

        else:
            _, gdf_conus, _ = prepare_state_geodataframe(
                gdf_raw, data_by_mp[mp], merge_col='state'
            )
            gdf_conus.plot(
                ax=ax, column=column, cmap=cmap, norm=norm,
                edgecolor='black', linewidth=lw, legend=False,
            )

        ax.set_axis_off()

        if isinstance(title_template, dict):
            title = title_template.get(mp, f"MP{mp}")
        else:
            title = title_template.format(mp=mp)

        ax.set_title(title, fontsize=MAP_TITLE_FONT_SIZE, fontweight='bold', pad=MAP_TITLE_PAD)

    # Horizontal colorbar — compact, centered at the bottom
    cax = fig.add_axes([0.25, 0.06, 0.50, 0.035])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', label=cbar_label)
    cbar.ax.xaxis.label.set_fontsize(MAP_CBAR_LABEL_FONT_SIZE)
    cbar.ax.xaxis.label.set_fontweight('bold')
    cbar.ax.xaxis.set_label_position('bottom')
    cax.tick_params(labelsize=MAP_CBAR_TICK_LABEL_SIZE, rotation=0)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)

    if save_figure and output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")
    plt.show()


def plot_national_county_choropleth(
    gdf_counties_raw: gpd.GeoDataFrame,
    data_by_mp: dict,
    column: str,
    title_template: Union[str, Dict[int, str]],
    cbar_label: str,
    cmap,
    norm,
    selected_mps: list,
    save_figure: bool,
    output_filename: str,
) -> None:
    """Render one national county choropleth with this notebook's shared settings.

    Thin wrapper around ``plot_combined_choropleth`` for the four national
    county maps the main notebook builds (adoption rate, operating-cost %
    change, demand GWh change, demand % change). All four calls always map
    counties and save under the project root, so ``geo_level`` is fixed to
    ``'county'`` here and ``output_path`` is built from ``PROJECT_ROOT`` plus a
    bare filename instead of each call site repeating
    ``os.path.join(PROJECT_ROOT, ...)``. ``title_template``, ``selected_mps``,
    and ``save_figure`` stay as parameters because they are notebook-session
    values (the discount rate and MP selection, the save-figures toggle), not
    constants this module can hardcode.

    Args:
        gdf_counties_raw: Raw county boundary GeoDataFrame, as passed to
            ``plot_combined_choropleth``.
        data_by_mp: Dict mapping MP number -> DataFrame with the target
            ``column`` and a county GISJOIN column.
        column: Column name to choropleth-shade.
        title_template: Per-panel title template or dict; see
            ``plot_combined_choropleth``.
        cbar_label: Colorbar axis label.
        cmap: Matplotlib colormap (name string or Colormap instance).
        norm: Matplotlib Normalize instance for this map's color scale.
        selected_mps: Ordered list of MP keys to render as panels.
        save_figure: If True, save the figure under ``output_filename``.
        output_filename: Bare filename (e.g. ``'county_econ_adoption_rate_
            combined.png'``); joined with ``PROJECT_ROOT`` to build the save path.

    Returns:
        None. Renders and, if requested, saves the figure -- same as
        ``plot_combined_choropleth``.
    """
    plot_combined_choropleth(
        gdf_counties_raw, data_by_mp,
        column=column,
        title_template=title_template,
        cbar_label=cbar_label,
        cmap=cmap,
        norm=norm,
        selected_mps=selected_mps,
        geo_level='county',
        save_figure=save_figure,
        output_path=os.path.join(PROJECT_ROOT, output_filename),
    )


def plot_national_county_change_map(
    gdf_counties_raw: gpd.GeoDataFrame,
    data_by_mp: dict,
    column: str,
    cbar_label: str,
    cmap,
    title_template: Union[str, Dict[int, str]],
    selected_mps: list,
    positive_direction: str,
    norm_label: str,
    norm_unit: str,
    save_figure: bool,
    output_filename: str,
) -> None:
    """Print the summary + symmetric-norm line, then render one change map.

    Consolidates the three near-identical blocks the main notebook's county
    choropleth cells build (operating-cost % change, electricity demand GWh
    change, electricity demand % change): compute a symmetric color norm
    across all selected MPs, print it, print the per-MP min/median/mean/max
    summary (``print_column_summary``), then render the map
    (``plot_national_county_choropleth``). Skips with a warning instead of
    raising when ``gdf_counties_raw`` is None (no county shapefile loaded),
    matching the notebook's existing fallback behavior.

    Args:
        gdf_counties_raw: Raw county boundary GeoDataFrame, or None if the
            shapefile failed to load.
        data_by_mp: Dict mapping MP number -> DataFrame with the target
            ``column`` and a county GISJOIN column.
        column: Column name to choropleth-shade and summarize.
        cbar_label: Colorbar axis label.
        cmap: Matplotlib colormap (name string or Colormap instance).
        title_template: Per-panel title template or dict; see
            ``plot_combined_choropleth``.
        selected_mps: Ordered list of MP keys to render as panels.
        positive_direction: Forwarded to ``print_column_summary`` -- which
            sign counts as "positive" in the printed share, e.g.
            ``'increase'`` or ``'HP saves money (< 0)'``.
        norm_label: Printed before " norm: [...]" (e.g. ``'Operating cost
            %'``, ``'Demand GWh'``).
        norm_unit: Printed after the norm's bracketed range (e.g. ``'%'``,
            ``' GWh'`` -- include a leading space when the unit is a word,
            not a percent sign).
        save_figure: If True, save the figure under ``output_filename``.
        output_filename: Bare filename; joined with ``PROJECT_ROOT`` to build
            the save path.

    Returns:
        None.
    """
    if gdf_counties_raw is None:
        print(f"[WARN] {norm_label} map skipped -- county shapefile not available")
        return

    all_values = pd.concat([data_by_mp[mp][column] for mp in selected_mps])
    norm = make_symmetric_norm(all_values)
    print(f"\n{norm_label} norm: [{norm.vmin:.1f}, 0, {norm.vmax:.1f}]{norm_unit}")

    print_column_summary(
        data_by_mp, column, norm_label, selected_mps, title_template,
        positive_direction=positive_direction,
    )
    plot_national_county_choropleth(
        gdf_counties_raw, data_by_mp,
        column=column,
        title_template=title_template,
        cbar_label=cbar_label,
        cmap=cmap, norm=norm,
        selected_mps=selected_mps,
        save_figure=save_figure,
        output_filename=output_filename,
    )



# ============================================================================
# Load County Boundaries and Merge with Analysis Data
# ============================================================================

# Territory STATEFP codes to exclude (US Census TIGER/Line).
_TERRITORY_FIPS = {'60', '66', '69', '72', '78'}  # AS, GU, MP, PR, VI
_ALASKA_FIPS = '02'


def prepare_county_geodataframe(
    gdf_counties: gpd.GeoDataFrame,
    df_analysis: pd.DataFrame,
    county_gisjoin_col: str = 'county',
    exclude_territories: Optional[list] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Merge county-level analysis results with US county geometries.

    Converts GISJOIN codes (e.g. 'G4200030') to 5-digit FIPS (e.g. '42003')
    and merges with the county geometry file on the GEOID column.
    Reprojects to US Albers Equal Area (ESRI:102003) and splits into
    CONUS and Alaska GeoDataFrames for inset plotting.

    Args:
        gdf_counties: GeoDataFrame of US county boundaries loaded from the
            source named by COUNTY_SHAPEFILE_PATH in data_loading.py (product
            and vintage set by the COUNTY_GEOMETRY_* constants there; any CRS).
            Must contain GEOID and STATEFP columns.
        df_analysis: DataFrame with a county GISJOIN column and analysis columns.
        county_gisjoin_col: Column in df_analysis with GISJOIN county codes
            (e.g. 'G4200030' → FIPS '42003'). Default: 'county'.
        exclude_territories: STATEFP codes to exclude. Defaults to AS/GU/MP/PR/VI.

    Returns:
        Tuple of (gdf_all, gdf_conus, gdf_alaska) — all in ESRI:102003.

    Note:
        The unmatched-polygon count printed here measures join coverage only:
        polygons with no matching data row (counties with no ResStock sample).
        It is not the total number of gray counties on the map. More counties
        may render gray because a KPI function masks a small-sample county to
        NaN downstream. Both are correct and measure different things -- this
        count answers "did the geometry vintage match the data's county codes",
        not "how many counties have a plottable value".
    """
    if exclude_territories is None:
        excl_fips = _TERRITORY_FIPS
    else:
        excl_fips = set(exclude_territories)

    if county_gisjoin_col not in df_analysis.columns:
        raise ValueError(
            f"Column '{county_gisjoin_col}' not found in df_analysis. "
            f"Available columns: {list(df_analysis.columns)}"
        )
    if 'GEOID' not in gdf_counties.columns:
        raise ValueError(
            f"'GEOID' column not found in county shapefile. "
            f"Available columns: {list(gdf_counties.columns)}"
        )

    # Convert GISJOIN (e.g. 'G4200030') → 5-digit FIPS (e.g. '42003')
    # GISJOIN format: G + 2-digit state FIPS + 0 + 3-digit county FIPS
    df_work = df_analysis.copy()
    df_work['GEOID'] = (
        df_work[county_gisjoin_col].str[1:3]
        + df_work[county_gisjoin_col].str[4:7]
    )

    # Reproject to US Albers Equal Area
    gdf_counties = gdf_counties.to_crs('ESRI:102003')

    # Merge analysis data with geometries
    gdf = gdf_counties.merge(df_work, on='GEOID', how='left')

    # Detect missing-data filter column
    numeric_cols = df_analysis.select_dtypes(include='number').columns.tolist()
    filter_col = numeric_cols[0] if numeric_cols else county_gisjoin_col

    # Join-coverage check, reported before the notna() drop below.
    # The merge above is a left join on the polygons, so a data row whose GEOID
    # matches no polygon is dropped silently and its county renders in the
    # missing-data color. Report both mismatch directions so a county geometry
    # vintage that does not match the ResStock county codes fails loudly. The
    # two directions are NOT symmetric:
    #   - Unmatched DATA ROWS is the alarm (should be zero). A post-2021 vintage
    #     drops all of Connecticut, whose ResStock codes (09001-09015) were
    #     replaced by planning regions (09110-09190) starting in 2022.
    #   - Unmatched POLYGONS is expected and benign: a few extreme low-population
    #     counties have no ResStock sample, so they have a polygon but no data
    #     row and correctly stay gray.
    # Territories are excluded from both counts because they never render.
    polygon_geoids = set(gdf_counties['GEOID'])
    data_geoids = set(df_work['GEOID'])

    unmatched_data = df_work[
        (~df_work['GEOID'].isin(polygon_geoids))
        & (~df_work['GEOID'].str[:2].isin(excl_fips))
    ]
    unmatched_poly = gdf_counties[
        (~gdf_counties['GEOID'].isin(data_geoids))
        & (~gdf_counties['STATEFP'].isin(excl_fips))
    ]
    unmatched_data_states = sorted(unmatched_data['GEOID'].str[:2].unique())
    unmatched_poly_states = sorted(unmatched_poly['STATEFP'].unique())

    if len(unmatched_data) > 0:
        print(
            f"[WARN] {len(unmatched_data)} county data row(s) matched no "
            f"polygon and were dropped (state FIPS {unmatched_data_states}). "
            f"The county geometry vintage (COUNTY_GEOMETRY_* in "
            f"data_loading.py) likely does not match the ResStock county "
            f"codes -- a post-2021 vintage drops all of Connecticut."
        )
    print(
        f"[OK] Join coverage: {len(unmatched_poly)} polygon(s) matched no data "
        f"row and stay gray (state FIPS {unmatched_poly_states}); expected for "
        f"counties with no ResStock sample."
    )

    gdf_filtered = gdf[
        (~gdf['STATEFP'].isin(excl_fips)) &
        (gdf[filter_col].notna())
    ].copy()

    gdf_alaska = gdf_filtered[gdf_filtered['STATEFP'] == _ALASKA_FIPS].copy()
    gdf_conus = gdf_filtered[
        ~gdf_filtered['STATEFP'].isin([_ALASKA_FIPS, '15'])  # exclude AK and HI
    ].copy()

    return gdf_filtered, gdf_conus, gdf_alaska


# ============================================================================
# Categorical Break-Even Map
# ============================================================================

BREAKEVEN_COLORS: list = ['#D32F2F', '#EF9A9A', '#90CAF9', '#1565C0']
"""Fill colors for 4 break-even favorability categories (0=Unfavorable … 3=Very Favorable)."""

BREAKEVEN_LABELS: list = ['HP does NOT beat 80% AFUE', 'HP beats 80% AFUE', 'HP beats 90% AFUE', 'HP beats 95% AFUE']
"""Display labels for the 4 break-even categories."""

def plot_categorical_breakeven_map(
    gdf_states_raw: gpd.GeoDataFrame,
    breakeven_results: dict,
    cop_results: dict,
    selected_mps: list,
    output_dir: Optional[str] = None,
    save_figure: bool = False,
    show_plot: bool = True,
) -> None:
    """Render a categorical break-even favorability map (one panel per MP).

    For each MP, merges COP results onto break-even results, calls
    :func:`assign_breakeven_category` to assign category codes 0–3, then
    plots a state-level choropleth using a ``ListedColormap`` + ``BoundaryNorm``
    with a discrete patch legend (no continuous colorbar).

    Args:
        gdf_states_raw: Raw state boundary GeoDataFrame (any CRS).
        breakeven_results: Dict mapping MP number → DataFrame returned by
            ``compute_breakeven_cop``.  Must contain ``state``,
            ``breakeven_cop_80``, ``breakeven_cop_90``, ``breakeven_cop_95``.
        cop_results: Dict mapping MP number → DataFrame returned by
            ``compute_thermal_cop``.  Must contain ``state`` and
            ``thermal_cop``.  Used to derive the ``hp_beats_breakeven_*``
            boolean columns internally — no pre-joining required.
        selected_mps: Ordered list of MP keys to render as panels.
        output_dir: Directory for saving figures.  Each MP produces one file
            named ``state_breakeven_categorical_mp{mp}_2024.png``.
        save_figure: If ``True`` and ``output_dir`` is provided, save to disk.
        show_plot: If ``True`` (default), display the figure interactively.
    """
    cmap_cat = ListedColormap(BREAKEVEN_COLORS)
    norm_cat = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_cat.N)

    n = len(selected_mps)
    fig = plt.figure(figsize=(14 * n, 8), facecolor='white')
    panel_w = 0.88 / n
    gap = 0.02

    for i, mp in enumerate(selected_mps):
        df = breakeven_results[mp].copy()

        # Join thermal COP from cop_results
        df['thermal_cop'] = (
            cop_results[mp].set_index('state')['thermal_cop']
            .reindex(df['state'].values).values
        )

        # Derive boolean beat-columns
        df['hp_beats_breakeven_80'] = df['thermal_cop'] > df['breakeven_cop_80']
        df['hp_beats_breakeven_90'] = df['thermal_cop'] > df['breakeven_cop_90']
        df['hp_beats_breakeven_95'] = df['thermal_cop'] > df['breakeven_cop_95']

        df['be_category'] = assign_breakeven_category(df)

        x0 = 0.02 + i * (panel_w + gap)
        w = panel_w - gap

        ax = fig.add_axes([x0, 0.06, w, 0.88])

        plot_kw = dict(
            column='be_category', cmap=cmap_cat, norm=norm_cat,
            edgecolor='black', linewidth=0.5, legend=False,
        )
        _, gdf_conus, _ = prepare_state_geodataframe(
            gdf_states_raw, df, merge_col='state'
        )
        gdf_conus.plot(ax=ax, **plot_kw)
        ax.set_axis_off()
        ax.set_title(
            f'Gas Furnace vs ASHP (MP{mp}, 2024)',
            fontsize=MAP_TITLE_FONT_SIZE, fontweight='bold', pad=MAP_TITLE_PAD,
        )

    # Shared discrete legend
    patches = [
        mpatches.Patch(color=BREAKEVEN_COLORS[i], label=BREAKEVEN_LABELS[i])
        for i in range(4)
    ]
    fig.legend(
        handles=patches, loc='lower center', ncol=4, fontsize=MAP_LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, 0.0),
    )

    if save_figure and output_dir:
        import os
        for mp in selected_mps:
            out_path = os.path.join(
                output_dir, f'state_breakeven_categorical_mp{mp}_2024.png'
            )
            fig.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"  Saved: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
