from typing import Optional, Tuple
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    dpi: int = 300,
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
    ax_main.set_title(title, fontsize=20, fontweight='bold', pad=12)

    if not gdf_alaska.empty:
        gdf_alaska.plot(ax=ax_ak, **plot_kw)
    ax_ak.set_axis_off()

    # Colorbar — matplotlib handles tick placement natively for any norm
    cax = fig.add_axes([0.82, 0.08, 0.03, 0.74])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=cbar_label).ax.yaxis.label.set_fontsize(14)
    cax.tick_params(labelsize=14)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path if output_path else None


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
    Merge county-level analysis results with TIGER county geometries.

    Converts GISJOIN codes (e.g. 'G4200030') to 5-digit FIPS (e.g. '42003')
    and merges with the TIGER/Line county shapefile on the GEOID column.
    Reprojects to US Albers Equal Area (ESRI:102003) and splits into
    CONUS and Alaska GeoDataFrames for inset plotting.

    Args:
        gdf_counties: GeoDataFrame of US county boundaries loaded from
            tl_2025_us_county.shp (any CRS). Must contain GEOID and STATEFP columns.
        df_analysis: DataFrame with a county GISJOIN column and analysis columns.
        county_gisjoin_col: Column in df_analysis with GISJOIN county codes
            (e.g. 'G4200030' → FIPS '42003'). Default: 'county'.
        exclude_territories: STATEFP codes to exclude. Defaults to AS/GU/MP/PR/VI.

    Returns:
        Tuple of (gdf_all, gdf_conus, gdf_alaska) — all in ESRI:102003.
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

    gdf_filtered = gdf[
        (~gdf['STATEFP'].isin(excl_fips)) &
        (gdf[filter_col].notna())
    ].copy()

    gdf_alaska = gdf_filtered[gdf_filtered['STATEFP'] == _ALASKA_FIPS].copy()
    gdf_conus = gdf_filtered[
        ~gdf_filtered['STATEFP'].isin([_ALASKA_FIPS, '15'])  # exclude AK and HI
    ].copy()

    return gdf_filtered, gdf_conus, gdf_alaska
