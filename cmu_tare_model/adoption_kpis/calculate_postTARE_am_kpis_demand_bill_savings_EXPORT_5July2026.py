# %% [markdown]
# # Post-TARE Adoption KPIs: Bill Savings & Demand Change
# **Author:** Jordan M. Joseph, PhD â€” Carnegie Mellon University
# 
# Computes bill savings ratio and electricity demand change metrics from TARE model outputs and EUSS data.

# %%
import os
import importlib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import PROJECT_ROOT
from cmu_tare_model.constants import VALID_MENU_MPS, PRINT_DEBUG
from cmu_tare_model.utils.load_exported_results_to_df import load_measure_package_data

from cmu_tare_model.adoption_kpis import (
    load_euss_baseline, load_euss_upgrade, mp_to_upgrade,
    compute_scenario_demand, aggregate_demand,
    compute_bill_savings_ratio, aggregate_bill_savings,
    compute_adoption_rate,
)
from cmu_tare_model.grid_impact.peak_load_functions import find_adoption_column

from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    prepare_state_geodataframe, plot_combined_choropleth,
)
from cmu_tare_model.adoption_kpis.data_loading import (
    SHAPEFILE_PATH, COUNTY_SHAPEFILE_PATH,
)

# Adoption potential: multi-index DataFrame and dotplot
from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_potential import (
    build_adoption_scenario_names,
    create_multiIndex_adoption_df,
    print_adoption_decision_percentages,
    subplot_grid_adoption_vBar
)

from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    prepare_plot_data,
    plot_adoption_panel,
    _build_legend_handles,
    GROUPING_ORDER,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)
SAVE_FIGURES = False  # Set to True to save figure files to disk

print("[OK] Imports loaded")


# %%
def pct_change(new: pd.Series, old: pd.Series) -> pd.Series:
    """Return per-element percent change: (new - old) / old * 100.

    NaN is propagated wherever old <= 0 (invalid baseline) or either
    input is NaN, so homes with zero or negative baseline cost are
    excluded rather than producing infinite or misleading values.

    Args:
        new: Post-retrofit values (e.g. retrofit lifetime fuel cost).
        old: Pre-retrofit baseline values (e.g. baseline lifetime fuel cost).

    Returns:
        Series of percent changes. NaN where old <= 0 or either input is NaN.
    """
    old_safe = old.where(old > 0, other=np.nan)
    return (new - old_safe) / old_safe * 100


def make_symmetric_norm(values: pd.Series, center: float = 0.0,
                        low_q: float = 0.02, high_q: float = 0.98) -> "Normalize":
    """Return a symmetric matplotlib Normalize centered at `center`.

    Clips to [low_q, high_q] percentiles before computing the symmetric
    deviation, so a single extreme outlier cannot compress the colormap.

    Args:
        values: Series of values (NaNs dropped internally).
        center: Value placed at the colormap midpoint.
        low_q: Lower clip percentile (default 0.02).
        high_q: Upper clip percentile (default 0.98).

    Returns:
        Normalize with vmin = center - dev, vmax = center + dev.
    """
    from matplotlib.colors import Normalize
    v = values.dropna()
    q_low = v.quantile(low_q)
    q_high = v.quantile(high_q)
    dev = max(abs(q_low - center), abs(q_high - center))
    return Normalize(vmin=center - dev, vmax=center + dev)


def print_column_summary(
    results: dict,
    column: str,
    label: str,
    selected_mps: list,
    mp_subtitles: dict,
    positive_direction: str = "increase",
) -> None:
    """Print per-MP min/median/mean/max summary for a county-level result column.

    Args:
        results: Dict mapping MP int -> county-level result DataFrame.
        column: Column name to summarise.
        label: Metric label for the section header.
        selected_mps: MP keys to iterate.
        mp_subtitles: Dict mapping MP int -> equipment subtitle string.
        positive_direction: Word used in the "% of counties X" phrase.
    """
    print(f"\n--- Summary: {column} ---")
    for mp in selected_mps:
        _v = results[mp][column].dropna()
        _pct = (_v > 0).mean() * 100 if positive_direction == "increase" else (_v < 0).mean() * 100
        _dir = positive_direction
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f} | med={_v.median():.1f} | "
              f"mean={_v.mean():.1f} | max={_v.max():.1f} | "
              f"{_pct:.1f}% of counties {_dir}")



# %%
SELECTABLE_MPS = [mp for mp in VALID_MENU_MPS if mp != 0]

try:
    _ = input_measure_package
    batch_mode = True
    selected_mps = [int(input_measure_package)]
    print(f"BATCH MODE: Running for MP{selected_mps[0]}")
except NameError:
    batch_mode = False
    print(f"Available measure packages: {SELECTABLE_MPS}")
    mp_input = input("Enter MP numbers (comma-separated, or 'all'): ").strip()
    if mp_input.lower() == 'all':
        selected_mps = SELECTABLE_MPS
    else:
        selected_mps = [int(x.strip()) for x in mp_input.split(',') if x.strip().isdigit()]
        selected_mps = [mp for mp in selected_mps if mp in SELECTABLE_MPS]
    if not selected_mps:
        selected_mps = [4]
        print("No valid MPs selected. Defaulting to MP4.")

print(f"Selected measure packages: {selected_mps}")

# %%
# Hardcoded for reproducibility — non-interactive session (no stdin).
# LOCATION_ID: str = "National"
# MODEL_RUN_DATE_TIME: str = "2026-04-10_00-05"

# Load TARE model outputs for selected measure packages.
# These contain per-building lifetime fuel costs used for bill savings ratio.

try:
    _ = DATAFRAMES_BY_MP
    print(f"DATAFRAMES_BY_MP already loaded: {list(DATAFRAMES_BY_MP.keys())}")
except NameError:
    try:
        _ = output_folder_path
        print(f"Using existing output_folder_path: {output_folder_path}")
    except NameError:
        output_folder_path = os.path.join(PROJECT_ROOT, "cmu_tare_model", "output_results")
        location_id = input("Enter location ID (e.g., 'National' or 'PA'): ").strip()
        model_run_date_time = input("Enter model run timestamp (YYYY-MM-DD_HH-MM): ").strip()

    DATAFRAMES_BY_MP = {}
    for mp in selected_mps:
        DATAFRAMES_BY_MP[mp] = load_measure_package_data(
            mp, output_folder_path, location_id, model_run_date_time
        )
    print(f"[OK] Loaded TARE data for MPs: {list(DATAFRAMES_BY_MP.keys())}")


# %%
df_baseline = load_euss_baseline()
print(f"Baseline: {len(df_baseline):,} occupied SF homes")

upgrade_data = {}
for mp in selected_mps:
    upgrade_name = mp_to_upgrade(mp)
    print(f"\nLoading MP{mp} ({upgrade_name})...")
    upgrade_data[mp] = load_euss_upgrade(upgrade_name)
    print(f"  MP{mp}: {len(upgrade_data[mp]):,} applicable homes")

print(f"\n[OK] EUSS data loaded")

# %%
# Load the EUSS MP data
df_euss_mp3 = upgrade_data[3]


# %%
print([col for col in df_euss_mp3.columns if 'cooling' in col.lower()])

# %% [markdown]
# ## Analysis of Economic Adoption Potential for ASHPs

# %%
# Map selected_mps to the naming convention used by the dotplot
HEATING_MEASURE_PACKAGES = selected_mps

HEATING_MP_SUBTITLES = {
    3: 'Single-stage, min-efficiency ASHP (15 SEER1, 9 HSPF1)',
    4: 'Variable-speed, high-efficiency ASHP (24-29.3 SEER1, 13-14 HSPF1)',
}

CASE_LABELS = {
    'heating': 'Heating Replacement Cost',
    'heating_and_cooling': 'Heating + Cooling Replacement Cost',
}

# %%
# County sample size diagnostic — gated behind PRINT_DEBUG flag.
# Set PRINT_DEBUG=True in constants.py to enable; False on clean runs.
if PRINT_DEBUG:
    print("=" * 65)
    print("DIAGNOSTIC: County sample size & coverage")
    print("=" * 65)

    for mp in selected_mps:
        print(f"\n===== County sample size distribution - {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")
        df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']

        county_counts = df_tare.groupby('county').size()

        print(f"Counties total:             {len(county_counts):,}")
        print(f"Min / Median / Mean / Max:  "
              f"{county_counts.min()} / {county_counts.median():.0f} / "
              f"{county_counts.mean():.1f} / {county_counts.max()}")

        print("Sample size by threshold:")
        for thresh in [5, 10, 15, 20, 30]:
            below = (county_counts < thresh).sum()
            above = (county_counts >= thresh).sum()
            print(f"  below {thresh:2d}: {below:4d}   |   at or above {thresh:2d}: {above:4d}")


# %%
print(f"\n{'='*60}")
print(f"OPERATING COST % CHANGE — All fuels, 100% adoption")
print(f"{'='*60}")

bill_savings_geo_level = 'county'
bill_savings_results = {}

for mp in selected_mps:
    print(f"\n===== {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']
    baseline_col = 'baseline_heating_lifetime_fuel_cost'
    retrofit_col = f'iraRef_mp{mp}_heating_lifetime_fuel_cost'
    # Per-home direct percent change: (retrofit − baseline) / baseline × 100
    pct = pct_change(df_tare[retrofit_col], df_tare[baseline_col])
    df_county = (
        pd.DataFrame({'county': df_tare['county'], 'operating_cost_pct_change': pct})
        .groupby('county')['operating_cost_pct_change']
        .median()
        .reset_index()
    )
    n_valid = pct.notna().sum()
    print(f"  Per-home valid records: {n_valid:,} | Counties: {len(df_county):,}")
    bill_savings_results[mp] = df_county

print(f"\n[OK] Operating cost % change complete ({bill_savings_geo_level}-level)")


# %%
print(f"\n{'='*60}")
print(f"DEMAND CHANGE — All fuels, 100% adoption")
print(f"{'='*60}")

demand_geo_level = 'county'
demand_results = {}

for mp in selected_mps:
    print(f"\n===== {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")
    df_demand = compute_scenario_demand(
        df_baseline, upgrade_data[mp], fuel_filter=None, verbose=True,
    )
    df_demand_county = aggregate_demand(
        df_demand, geo_level=demand_geo_level, verbose=True
    )
    # Re-derive demand % via shared pct_change helper for formula consistency
    # with the operating-cost % visual. Algebraically identical to aggregate_demand's
    # internal np.where, but makes the formula source of truth explicit.
    df_demand_county['pct_elec_demand_change'] = pct_change(
        df_demand_county['retrofit_elec_gwh'], df_demand_county['baseline_elec_gwh']
    )
    demand_results[mp] = df_demand_county

print(f"\n[OK] Demand change complete ({demand_geo_level}-level)")


# %%
from matplotlib.colors import Normalize

gdf_states_raw = None
gdf_counties_raw = None

try:
    gdf_states_raw = gpd.read_file(SHAPEFILE_PATH)
    print(f"[OK] State shapefile loaded: {len(gdf_states_raw)} features")
except Exception as e:
    print(f"[WARN] State shapefile not loaded: {e}")

try:
    gdf_counties_raw = gpd.read_file(COUNTY_SHAPEFILE_PATH)
    print(f"[OK] County shapefile loaded: {len(gdf_counties_raw)} features")
except Exception as e:
    print(f"[WARN] County shapefile not loaded: {e} — skipping county maps")

if gdf_counties_raw is not None:
    # ---- Compute shared norms BEFORE any per-MP map calls ----
    # Symmetric Normalize: keeps white at the meaningful center (0%)
    # and makes the colorbar evenly spaced on both sides.
    # Clip to 2nd/98th percentile to avoid extreme outliers compressing the scale.

    # Build shared symmetric norms (centered at 0) before map calls.
    # make_symmetric_norm clips to 2nd/98th percentile across all MPs so a single extreme county cannot compress the colormap.
    # Percent bill savings
    _all_pct_bill = pd.concat([
        bill_savings_results[mp]['operating_cost_pct_change'] for mp in selected_mps
    ])
    shared_pct_bill_norm = make_symmetric_norm(_all_pct_bill)
    print(f"Operating cost % norm: [{shared_pct_bill_norm.vmin:.1f}, 0 (center), {shared_pct_bill_norm.vmax:.1f}]%")

    # Electricity demand change (GWh)
    _all_demand_gwh = pd.concat([
        demand_results[mp]['elec_change_gwh'] for mp in selected_mps
    ])
    shared_demand_norm = make_symmetric_norm(_all_demand_gwh)
    print(f"Demand GWh norm: [{shared_demand_norm.vmin:.1f}, 0 (center), {shared_demand_norm.vmax:.1f}] GWh")

    # Electricity demand percent change (%)
    _all_pct_demand = pd.concat([
        demand_results[mp]['pct_elec_demand_change'] for mp in selected_mps
    ])
    shared_pct_demand_norm = make_symmetric_norm(_all_pct_demand)
    print(f"Demand % norm: [{shared_pct_demand_norm.vmin:.1f}, 0 (center), {shared_pct_demand_norm.vmax:.1f}]%")

    # ---- Operating cost percent change (county-level) ----
    # operating_cost_pct_change = county median of (retrofit − baseline) / baseline × 100
    print_column_summary(
        bill_savings_results, 'operating_cost_pct_change',
        'Operating cost % change', selected_mps, HEATING_MP_SUBTITLES,
        positive_direction='HP saves money (< 0)',
    )

    plot_combined_choropleth(
        gdf_counties_raw, bill_savings_results,
        column='operating_cost_pct_change',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Median Percent Change in Retrofit Lifetime Operating Costs\n(relative to baseline equipment)',
        cmap='RdBu_r', norm=shared_pct_bill_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_bill_pct_change_combined.png'),
    )

    # ---- Electricity demand change GWh (county-level) ----
    print_column_summary(
        demand_results, 'elec_change_gwh',
        'Elec demand change (GWh)', selected_mps, HEATING_MP_SUBTITLES,
        positive_direction='increase',
    )
        
    plot_combined_choropleth(
        gdf_counties_raw, demand_results,
        column='elec_change_gwh',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Post-retrofit change in annual electricity demand,\nrelative to baseline (GWh)',
        cmap='coolwarm', norm=shared_demand_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_elec_demand_gwh_combined.png'),
    )

    # ---- Electricity demand percent change (county-level) ----
    print_column_summary(
        demand_results, 'pct_elec_demand_change',
        'Elec demand % change', selected_mps, HEATING_MP_SUBTITLES,
        positive_direction='increase',
    )
        
    plot_combined_choropleth(
        gdf_counties_raw, demand_results,
        column='pct_elec_demand_change',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Post-retrofit change in annual electricity demand,\nrelative to baseline (%)',
        cmap='coolwarm', norm=shared_pct_demand_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_elec_demand_pct_combined.png'),
    )

    print("[OK] All county-level maps generated")
else:
    print("[WARN] County maps skipped — county shapefile not available")


# %%


# %% [markdown]
# ---
# # ADOPTION POTENTIAL
# ---

# %% [markdown]
# ## Does the heat pump pay for itself?
# 
# An **economic adopter** is a home where the heat pump's extra upfront cost is recovered
# from energy-bill savings alone — no climate or health benefit is needed to justify the
# investment.
# 
# **The rule:** a home is an economic adopter if its incremental private NPV >= 0.
# Break-even counts as adoption.
# 
# | Value | Meaning |
# |-------|---------|
# | `True`  | Heat pump covers its incremental cost (or better) from bill savings |
# | `False` | Valid home that cannot recover the incremental cost from savings alone |
# | `NaN`   | Excluded: invalid baseline fuel/tech or not in this measure package |
# 
# Climate and health damages are computed elsewhere and reported as outcomes,
# not as inputs to this decision.

# %%
# Note: first import loads bls_cpiu_2005-2023.xlsx for CPI inflation adjustment.
# This is expected and harmless — the table prints once on import, not on every call.

from cmu_tare_model.adoption_potential.determine_economic_adoption_potential import economic_adoption_decision
from cmu_tare_model.utils.modeling_params import define_scenario_params

_POLICY = '2025 Reference Case'
_DISCOUNT_COL = 'private_discount_rate_fixed_base'
_COST = 'v4MID'

# Generate the SIX economic-adopter columns (one per NPV case) for each
# measure package. After this cell, all SIX columns exist in the frame:
#   ref2025_mp{mp}_heatingSavings_coolingLCC_econ_adopter_fixed_base
#   ref2025_mp{mp}_heatingLCC_coolingSavings_econ_adopter_fixed_base
#   ref2025_mp{mp}_heatingLCC_coolingLCC_econ_adopter_fixed_base
# The dot-plot cell is pure-read -- no economic_adoption_decision calls inside it.
for mp in selected_mps:
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']
    df_econ = economic_adoption_decision(
        df_tare,
        menu_mp=mp,
        policy_scenario=_POLICY,
        discount_rate_col_name=_DISCOUNT_COL,
        cost_scenario=_COST,
        verbose=False,
    )
    # Copy only newly created columns back into the canonical frame.
    new_cols = [c for c in df_econ.columns if c not in df_tare.columns]
    for col in new_cols:
        DATAFRAMES_BY_MP[mp]['fixed_base'][col] = df_econ[col]
    if new_cols:
        print(f"[OK] MP{mp}: {new_cols}")

print("\n[DONE] Economic-adopter columns added for all selected MPs")

# %%
# Econ column probe — gated behind PRINT_DEBUG flag.
if PRINT_DEBUG:
    mp = selected_mps[0]
    econ_cols = [c for c in DATAFRAMES_BY_MP[mp]['fixed_base'].columns if 'econ_adopter' in c]
    print(econ_cols)

# %%
from cmu_tare_model.utils.modeling_params import define_scenario_params

_ADOPTION_COST_SCENARIO = 'v4MID'
_ADOPTION_GEO_LEVEL = 'county'
_POLICY = '2025 Reference Case'

# Use heatingLCC_coolingLCC (both avoided replacements credited) as the
# primary adoption metric for the choropleth. All three cases are in the
# frame after the adoption-decision cell above.
from cmu_tare_model.utils.column_names import create_adoption_col

print(f"\n{'='*60}")
print(f"Economic Adoption Rate -- 2025 Reference Case")
print(f"{'='*60}")

econ_adoption_rate_results = {}
for mp in selected_mps:
    print(f"\n===== {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")
    # Count homes where the heat pump pays for itself (econ_adopter == 1.0).
    # adopter_tiers=[True] selects 1.0 (adopter) vs 0.0 (non-adopter).
    # NaN rows (excluded homes) are automatically ignored by compute_adoption_rate.
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']
    prefix = define_scenario_params(mp, _POLICY)[0]
    adoption_col = create_adoption_col(
        scenario_prefix=prefix,
        npv_case='heatingLCC_coolingLCC_sub',
        method_suffix='_fixed_base',
    )
    print(f'  Adoption column: {adoption_col}')
    df_adopt = compute_adoption_rate(
        df_tare,
        adoption_col=adoption_col,
        adopter_tiers=[True],
        geo_level=_ADOPTION_GEO_LEVEL,
        df_euss=df_baseline,
        verbose=True,
    )
    econ_adoption_rate_results[mp] = df_adopt

print(f"\n[OK] Economic adoption rate complete ({_ADOPTION_GEO_LEVEL}-level)")


# %%
if gdf_counties_raw is not None:
    from matplotlib.colors import Normalize

    _adopt_cmap = 'Greens'
    _adopt_norm = Normalize(vmin=0, vmax=100)

    print("\n--- Summary: adoption_rate_pct ---")
    for mp in selected_mps:
        _v = econ_adoption_rate_results[mp]['adoption_rate_pct'].dropna()
        _pct_high = (_v >= 50).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f}% | med={_v.median():.1f}% | "
              f"mean={_v.mean():.1f}% | max={_v.max():.1f}% | "
              f"{_pct_high:.1f}% of counties ≥ 50% adoption potential")

    plot_combined_choropleth(
        gdf_counties_raw, econ_adoption_rate_results,
        column='adoption_rate_pct',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Share of households recovering incremental costs\nthrough discounted operational savings (%)',
        cmap=_adopt_cmap, norm=_adopt_norm,
        selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_econ_adoption_rate_combined.png'),
    )

    print("[OK] Economic adoption choropleth generated")
else:
    print("[WARN] Adoption choropleth skipped — county shapefile not available")


# %% [markdown]
# ## Adoption Potential Dotplot Maps

# %%
# =============================================================================
# ADOPTION POTENTIAL PARAMETERS (base-case defaults)
# =============================================================================
# These mirror the upstream TARE pipeline settings. The economic-adopter column
# name convention encodes these parameters: e.g.
# 'ref2025_mp3_heatingLCC_coolingLCC_econ_adopter_fixed_base'
#
# Only change these if running a sensitivity analysis. For the paper's
# primary results, these are the correct values.

cost_scenario = 'v4MID'    # REMDB v4 midpoint cost scenario (retained for API compat)
discount_rate = 'fixed_base'             # 7% fixed discount rate

print(f"  Cost: {cost_scenario}, Discount: {discount_rate}")
print(f"  Measure packages: {HEATING_MEASURE_PACKAGES}")

# %%
# -----------------------------------------------------------------
# Economic adoption potential dot plot
# Two markers per row: Heating Only (Case A) | Heating & Cooling (Case B)
# Annotation: same X% (+Y%) format as tier dotplot, Y = subsidized minus
# unsubsidized adoption rate for the same case.
# -----------------------------------------------------------------
import importlib
import matplotlib.lines as mlines
import cmu_tare_model.constants as constants
importlib.reload(constants)
import cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot as visuals_adoption_dotplot
importlib.reload(visuals_adoption_dotplot)
from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    plot_adoption_panel, GROUPING_ORDER, FUEL_COLORS,
)

_ECON_CASE_MARKERS = {
    'Heating Repl. Credit':           'o',   # circle  -- left:  heatingLCC_coolingSavings
    'Heating + Cooling Repl. Credit': 's',   # square  -- right: heatingLCC_coolingLCC
}
_ECON_CASES = [
    'Heating Repl. Credit',
    'Heating + Cooling Repl. Credit',
]

# Option A: two cases to compare. Left = heatingLCC_coolingSavings_sub;
# right = heatingLCC_coolingLCC_sub. Delta = subsidized minus unsubsidized
# adoption rate for the same case label.


def _build_econ_plot_df(
    source_df, mp, cost_scenario='v4MID', discount_rate='fixed_base',
    fuel_col='base_heating_fuel', income_col='lmi_or_mui',
    income_groups=None, scaling_factor=242.0,
):
    """Per-(fuel x income) mean economic adoption rate for the two Option-A NPV cases.

    Option A (both cases include heating + cooling operating savings):
      left  = heatingLCC_coolingSavings_sub   -- subsidized, heating replacement credited only
      right = heatingLCC_coolingLCC_sub       -- subsidized, both replacements credited
      delta = right - left                    -- subsidy-dependent gain in adoption rate

    The unsubsidized companion columns are computed upstream and are used only
    in the output delta calculation; the plot itself shows only adoption rates
    unless show_delta_annotation=True.

    Returns a DataFrame matching prepare_plot_data() output format so
    plot_adoption_panel() can render it without modification.
    Columns: grouping, fuel_type, income_level, tier_label,
             case_b_pct, case_a_pct, delta_pct,
             sample_n, pct_of_sample, weighted_homes_millions.
    """
    from cmu_tare_model.utils.column_names import create_adoption_col
    from cmu_tare_model.utils.modeling_params import define_scenario_params

    if income_groups is None:
        income_groups = ['LMI']

    # cost_scenario is not embedded in column names post-Session-A refactor
    # but is retained as a parameter for caller compatibility.
    scenario_prefix = define_scenario_params(mp)[0]
    method_suffix = f'_{discount_rate}'
    left_col_sub = create_adoption_col(
        scenario_prefix, 'heatingLCC_coolingSavings_sub', method_suffix)
    left_col_unsub = create_adoption_col(
        scenario_prefix, 'heatingLCC_coolingSavings_unsub', method_suffix)
    right_col_sub = create_adoption_col(
        scenario_prefix, 'heatingLCC_coolingLCC_sub', method_suffix)
    right_col_unsub = create_adoption_col(
        scenario_prefix, 'heatingLCC_coolingLCC_unsub', method_suffix)

    sample_total = len(source_df)
    group_counts = source_df.groupby([fuel_col, income_col], observed=True).size()
    fuel_counts = source_df.groupby(fuel_col, observed=True).size()
    total_homes = int(group_counts.sum())
    fuels_in_data = list(source_df[fuel_col].dropna().unique())

    def _rate(df_sub, col):
        # Mean adoption rate as a percentage; NaN if column not yet in frame.
        return df_sub[col].mean() * 100.0 if col in df_sub.columns else np.nan

    def _row(grouping, fuel, income_level, case_label, right_pct, left_pct, n):
        # Left-case rows pass right_pct == left_pct so delta == 0 (reference).
        # Right-case rows pass right_pct = heatingLCC_coolingLCC rate so
        # delta shows the gain from crediting the avoided AC replacement.
        return dict(
            grouping=grouping,
            fuel_type=fuel,
            income_level=income_level,
            tier_label=case_label,
            case_b_pct=right_pct,
            case_a_pct=left_pct,
            delta_pct=right_pct - left_pct,
            sample_n=n,
            pct_of_sample=100.0 * n / sample_total if sample_total else 0.0,
            weighted_homes_millions=n * scaling_factor / 1_000_000,
        )

    rows = []

    # -- Per fuel x selected income group (e.g. LMI only) --
    for (fuel, income), n in group_counts.items():
        if income not in income_groups:
            continue
        sub = source_df[
            (source_df[fuel_col] == fuel) & (source_df[income_col] == income)
        ]
        grouping = f'{fuel} -- {income}'
        l_sub = _rate(sub, left_col_sub)
        l_unsub = _rate(sub, left_col_unsub)
        r_sub = _rate(sub, right_col_sub)
        r_unsub = _rate(sub, right_col_unsub)
        rows.append(_row(grouping, fuel, income, 'Heating Repl. Credit',
                         l_sub, l_unsub, int(n)))
        rows.append(_row(grouping, fuel, income, 'Heating + Cooling Repl. Credit',
                         r_sub, r_unsub, int(n)))

    # -- Per fuel -- Overall (pooled across all income groups) --
    for fuel in fuels_in_data:
        fuel_n = int(fuel_counts.get(fuel, 0))
        fuel_sub = source_df[source_df[fuel_col] == fuel]
        grouping = f'{fuel} -- Overall'
        l_sub = _rate(fuel_sub, left_col_sub)
        l_unsub = _rate(fuel_sub, left_col_unsub)
        r_sub = _rate(fuel_sub, right_col_sub)
        r_unsub = _rate(fuel_sub, right_col_unsub)
        rows.append(_row(grouping, fuel, 'Overall', 'Heating Repl. Credit',
                         l_sub, l_unsub, fuel_n))
        rows.append(_row(grouping, fuel, 'Overall', 'Heating + Cooling Repl. Credit',
                         r_sub, r_unsub, fuel_n))

    # -- National -- Overall --
    l_sub = _rate(source_df, left_col_sub)
    l_unsub = _rate(source_df, left_col_unsub)
    r_sub = _rate(source_df, right_col_sub)
    r_unsub = _rate(source_df, right_col_unsub)
    rows.append(_row('National -- Overall', 'National', 'Overall',
                     'Heating Repl. Credit', l_sub, l_unsub, total_homes))
    rows.append(_row('National -- Overall', 'National', 'Overall',
                     'Heating + Cooling Repl. Credit', r_sub, r_unsub, total_homes))

    return pd.DataFrame(rows)


if not HEATING_MEASURE_PACKAGES:
    print("No active heating measure packages — skipping economic adoption dotplot.")
else:
    # Compute national fuel counts (same method as tier dotplot)
    _src = DATAFRAMES_BY_MP[HEATING_MEASURE_PACKAGES[0]][discount_rate]
    fuel_counts_millions = {
        str(fuel): int(n) * 242 / 1_000_000
        for fuel, n in _src.groupby('base_heating_fuel', observed=True).size().items()
    }

    n_mps = len(HEATING_MEASURE_PACKAGES)
    fig, axes = plt.subplots(
        n_mps, 1,
        # figsize=(16, 5.5 * n_mps),
        figsize=(12, 6 * n_mps),
        sharex=True,
        sharey=True,
    )
    if n_mps == 1:
        axes = [axes]

    for row_idx, mp in enumerate(HEATING_MEASURE_PACKAGES):
        ax = axes[row_idx]
        panel_title = (
            f'{HEATING_MP_SUBTITLES.get(mp, f"MP{mp}")}'
        )
        source_df = DATAFRAMES_BY_MP[mp][discount_rate]

        plot_df = _build_econ_plot_df(
            source_df, mp,
            cost_scenario=_COST,
            discount_rate=discount_rate,
        )

        # Print panel summary
        print(f"--- MP{mp} economic adoption summary ---")
        for case_label in _ECON_CASES:
            nat_row = plot_df[
                (plot_df['grouping'] == 'National -- Overall') &
                (plot_df['tier_label'] == case_label)
            ]
            if not nat_row.empty:
                rate = nat_row.iloc[0]['case_b_pct']
                delta = nat_row.iloc[0]['delta_pct']
                if delta == 0:
                    print(f"  {case_label}: {rate:.1f}%")
                else:
                    sign = '+' if delta >= 0 else ''
                    print(
                        f"  {case_label}: {rate:.1f}%"
                        f" ({sign}{delta:.1f}% vs unsubsidized)"
                    )
        print()

        _ECON_GROUPING_ORDER = [
            'National -- Overall',
            'Electricity -- Overall',
            'Natural Gas -- Overall',
            'Fuel Oil -- Overall',
            'Propane -- Overall',
        ]

        plot_adoption_panel(
            plot_df, ax,
            grouping_order=_ECON_GROUPING_ORDER,
            title=panel_title,
            title_fontsize=16,
            ytick_fontsize=14,
            annotation_fontsize=14,
            annotation_x_offset_pts=0,
            annotation_y_offset_pts=8,
            xlim_margin=20,
            fuel_counts_millions=fuel_counts_millions,
            custom_tier_markers=_ECON_CASE_MARKERS,
        )
        ax.tick_params(axis='both', labelsize=14)

        # Legend: circle = Heating Repl. Credit (left), square = Heating + Cooling Repl. Credit (right)
        legend_handles = [
            mlines.Line2D([], [], marker='o', color='none',
                          markerfacecolor='gray', markeredgecolor='gray',
                          markersize=8, linestyle='None',
                          label='Heating Repl. Credit Only'),
            mlines.Line2D([], [], marker='s', color='none',
                          markerfacecolor='gray', markeredgecolor='gray',
                          markersize=8, linestyle='None',
                          label='Heating + Cooling Repl. Credit'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=14, frameon=True)

        if row_idx < n_mps - 1:
            ax.set_xlabel('')

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.96])

    out_dir = os.path.join('.', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(
            os.path.join(out_dir, f'figure6_econ_adoption_dotplot_{location_id}.{ext}'),
            dpi=600, bbox_inches='tight',
        )
    print(f"Saved to {out_dir}/figure6_econ_adoption_dotplot_{location_id}.{{png,pdf}}")
    plt.show()


# %% [markdown]
# ---
# ## Private NPV Distribution: Break-Even Histogram
# ---
# 
# Histograms of per-building private NPV (more WTP) under two policy scenarios:
# - **Pre-IRA:** No rebates. Break-even at $0 — buildings left of zero need subsidy.
# - **IRA-Ref:** With IRA rebates applied. Shift right = more buildings break even.
# 
# Stacked by baseline heating fuel. Reference lines: U.S. and PA median/mean.
# `moreWTP` variant used (conservative willingness-to-pay assumption).
# 

# %%
# =============================================================================
# Verify WTP columns and print summary statistics
# =============================================================================
for mp in selected_mps:
    df_check = DATAFRAMES_BY_MP[mp]['fixed_base']
    col_pre = f'preIRA_mp{mp}_heating_private_npv_moreWTP_v4MID_fixed_base'
    col_ira = f'iraRef_mp{mp}_heating_private_npv_moreWTP_v4MID_fixed_base'
    assert col_pre in df_check.columns, f"Not found: {col_pre}"
    assert col_ira in df_check.columns, f"Not found: {col_ira}"

    df_pa = df_check[df_check['state'] == 'PA']
    mp_label = HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')
    print(f"\n--- {mp_label} ---")
    for label, s in [
        ('National pre-IRA', df_check[col_pre]),
        ('National IRA',     df_check[col_ira]),
        ('PA pre-IRA',       df_pa[col_pre]),
        ('PA IRA',           df_pa[col_ira]),
    ]:
        print(f"  {label:25s}: median={s.median():>10,.0f}  mean={s.mean():>10,.0f}")

print("\n[OK] WTP columns verified for all selected MPs")

# %%
# =============================================================================
# Private NPV break-even histogram (n_mps × 2 grid)
# =============================================================================
# Uses create_subplot_grid_histogram from utils — stacked by heating fuel.
# Reference lines (US/PA median & mean) added post-call per subplot.
# Font sizes corrected post-call to match dotplot reference style (18/16 pt).
# =============================================================================

from cmu_tare_model.utils.data_visualization_histograms import create_subplot_grid_histogram

# ── Font size constants (match dotplot reference style) ──────────────────────
_TITLE_FS  = 18
_LABEL_FS  = 16
_TICK_FS   = 16
_LEGEND_FS = 14

# ── Build per-MP column lists ─────────────────────────────────────────────────
_npv_dfs        = []
_npv_cols       = []
_npv_df_indices = []
_npv_titles     = []
_npv_positions  = []

for _i, _mp in enumerate(selected_mps):
    _df_mp   = DATAFRAMES_BY_MP[_mp]['fixed_base']
    _col_pre = f'preIRA_mp{_mp}_heating_private_npv_moreWTP_v4MID_fixed_base'
    _col_ira = f'iraRef_mp{_mp}_heating_private_npv_moreWTP_v4MID_fixed_base'
    _label   = HEATING_MP_SUBTITLES.get(_mp, f'MP{_mp}')

    _npv_dfs.append(_df_mp)
    _npv_cols       += [_col_pre, _col_ira]
    _npv_df_indices += [_i, _i]
    _npv_titles     += [
        f'{_label}\nPre-IRA NPV (no rebates)',
        f'{_label}\nNPV with IRA Rebates',
    ]
    _npv_positions  += [(_i, 0), (_i, 1)]

_n_mps = len(selected_mps)

# ── Call utility ──────────────────────────────────────────────────────────────
fig_npv = create_subplot_grid_histogram(
    dataframes=_npv_dfs,
    dataframe_indices=_npv_df_indices,
    subplot_positions=_npv_positions,
    x_cols=_npv_cols,
    x_labels=['Private NPV, more WTP ($)'] * (_n_mps * 2),
    y_labels=['Count'] * (_n_mps * 2),
    subplot_titles=_npv_titles,
    bin_number=50,
    lower_percentile=1,
    upper_percentile=99,
    figure_size=(18, 9 * _n_mps),
    color_code='base_heating_fuel',
    statistic='count',
    # suptitle='Private NPV Distribution by Policy Scenario\n(more WTP, v4MID, fixed_base discount rate)',
)

# ── Add reference lines (US / PA median & mean) per subplot ──────────────────
# Note: utility already draws axvline(0) internally; no need to repeat here.
_ax_list = [ax for ax in fig_npv.get_axes() if ax.get_visible()]

for _ax_idx, (_pos, _mp, _col) in enumerate(
        zip(_npv_positions,
            [_mp for _mp in selected_mps for _ in range(2)],
            _npv_cols)):
    _ax    = _ax_list[_ax_idx]
    _df_mp = DATAFRAMES_BY_MP[_mp]['fixed_base']
    _df_pa = _df_mp[_df_mp['state'] == 'PA']

    _nat_med  = _df_mp[_col].median()
    _nat_mean = _df_mp[_col].mean()
    _pa_med   = _df_pa[_col].median()
    _pa_mean  = _df_pa[_col].mean()

    _ax.axvline(_nat_med,  color='tab:blue', ls='-',  lw=2,
                label=f'US median:  ${_nat_med:,.0f}')
    _ax.axvline(_nat_mean, color='tab:blue', ls='--', lw=2,
                label=f'US mean:    ${_nat_mean:,.0f}')
    _ax.axvline(_pa_med,   color='tab:red',  ls='-',  lw=2,
                label=f'PA median:  ${_pa_med:,.0f}')
    _ax.axvline(_pa_mean,  color='tab:red',  ls='--', lw=2,
                label=f'PA mean:    ${_pa_mean:,.0f}')
    _ax.legend(fontsize=_LEGEND_FS, loc='upper right', frameon=True)

# ── Post-call font size correction ───────────────────────────────────────────
for _ax in fig_npv.get_axes():
    if not _ax.get_visible():
        continue
    _ax.title.set_fontsize(_TITLE_FS)
    _ax.xaxis.label.set_fontsize(_LABEL_FS)
    _ax.yaxis.label.set_fontsize(_LABEL_FS)
    _ax.tick_params(axis='both', labelsize=_TICK_FS)
    if _ax.get_legend():
        plt.setp(_ax.get_legend().get_texts(), fontsize=_LEGEND_FS)

if fig_npv._suptitle:
    fig_npv._suptitle.set_fontsize(_TITLE_FS)

fig_npv.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
_out_dir = os.path.join('.', 'figures')
os.makedirs(_out_dir, exist_ok=True)
for _ext in ('png', 'pdf'):
    fig_npv.savefig(
        os.path.join(_out_dir, f'figure_npv_histogram_{location_id}.{_ext}'),
        dpi=600, bbox_inches='tight',
    )
print(f"[OK] NPV histogram saved → figures/figure_npv_histogram_{location_id}.{{png,pdf}}")
plt.show()

# %% [markdown]
# ---
# ## Subsidy Required: Distribution Under Pre-IRA vs. IRA Policy
# ---
# 
# Histogram of the **subsidy required** to make the heat-pump private NPV ≥ 0,
# for homes that remain NPV-negative under each policy scenario.
# 
# - **Pre-IRA (col 0):** all homes with NPV < 0 before rebates — subsidy = |NPV_pre|
# - **IRA (col 1):** homes still NPV-negative after IRA rebates — remaining subsidy = |NPV_ira|
# 
# Stacked by baseline heating fuel. Median dashed line per panel.
# `moreWTP`, `v4MID`, `fixed_base` discount rate.
# 
# **Verification:** Each MP row — IRA panel has **fewer homes** and **smaller median** than pre-IRA.

# %%



