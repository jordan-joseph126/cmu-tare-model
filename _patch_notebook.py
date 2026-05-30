"""
Patch the notebook to:
1. Replace cell 18 (a9a531b6): old inmap_econ/heating-only code → new code
   that loops both HVAC scenarios and writes directly into inmap.
2. Replace cell 20 (01cabe46): inmap_econ → inmap.
3. Replace cell 21 (6c25bccd): update choropleth title/cbar labels.
"""
import json

NB_PATH = r"c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_kpis\calculate_postTARE_am_kpis_demand_bill_savings.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# ── helpers ─────────────────────────────────────────────────────────────────

def find_cell(nb, cell_id):
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == cell_id:
            return i, c
    raise KeyError(f"Cell id {cell_id!r} not found")

def set_source(cell, code_str):
    """Replace cell source with lines split by \\n (keep trailing \\n on each line)."""
    lines = code_str.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    # Clear outputs and reset execution count
    cell["outputs"] = []
    cell["execution_count"] = None

# ── Cell 18 (a9a531b6) ──────────────────────────────────────────────────────

NEW_CELL18 = """\
from cmu_tare_model.adoption_potential.determine_economic_adoption_potential import economic_adoption_decision
from cmu_tare_model.utils.modeling_params import define_scenario_params

_POLICY = 'AEO2023 Reference Case'
_DISCOUNT_COL = 'private_discount_rate_fixed_base'
_COST = 'v4MID'

# Generate economic-adopter columns for both HVAC replacement scenarios.
# 'heating'             = Case A: replace only the furnace/boiler with a heat pump.
# 'heating_and_cooling' = Case B: replace both the furnace AND the AC with a heat pump.
# Columns are written directly into the canonical 'inmap' frame so that all
# downstream cells share a single source of truth — no separate inmap_econ key.
for mp in selected_mps:
    df_inmap = DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']
    for hvac_scenario in ['heating', 'heating_and_cooling']:
        df_econ = economic_adoption_decision(
            df_inmap,
            menu_mp=mp,
            policy_scenario=_POLICY,
            discount_rate_col_name=_DISCOUNT_COL,
            cost_scenario=_COST,
            hvac_replacement_scenario=hvac_scenario,
            verbose=False,
        )
        # Copy only the newly created econ columns back into the canonical frame.
        new_cols = [c for c in df_econ.columns if c not in df_inmap.columns]
        for col in new_cols:
            DATAFRAMES_BY_MP[mp]['fixed_base']['inmap'][col] = df_econ[col]
        print(f"[OK] MP{mp} | {hvac_scenario}: {new_cols}")

print("\\n[DONE] Economic adopter columns added to inmap for all selected MPs")\
"""

idx18, cell18 = find_cell(nb, "a9a531b6")
old_preview = "".join(cell18["source"])[:80]
print(f"[CELL 18] old: {old_preview!r}")
set_source(cell18, NEW_CELL18)
print(f"[CELL 18] replaced OK")

# ── Cell 20 (01cabe46) ──────────────────────────────────────────────────────

NEW_CELL20 = """\
from cmu_tare_model.utils.modeling_params import define_scenario_params

_ADOPTION_COST_SCENARIO = 'v4MID'
_ADOPTION_GEO_LEVEL = 'county'
_POLICY = 'AEO2023 Reference Case'

# Count homes where the heat pump pays for itself (econ_adopter == 1.0),
# aggregated to county level.
# adopter_tiers=[True] selects the adopter tier: 1.0 (adopter) vs 0.0 (non-adopter).
# NaN rows (excluded homes) are automatically ignored by compute_adoption_rate.
econ_adoption_rate_results = {}
for mp in selected_mps:
    print(f"\\n{'='*60}")
    print(f"Economic Adoption Rate — MP{mp} (econ_adopter, IRA-Ref)")
    print(f"{'='*60}")
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']
    prefix = define_scenario_params(mp, _POLICY)[0]
    adoption_col = f'{prefix}heating_econ_adopter_moreWTP_{_ADOPTION_COST_SCENARIO}_fixed_base'
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

print(f"\\n[OK] Economic adoption rate complete ({_ADOPTION_GEO_LEVEL}-level)")\
"""

idx20, cell20 = find_cell(nb, "01cabe46")
old_preview20 = "".join(cell20["source"])[:80]
print(f"[CELL 20] old: {old_preview20!r}")
set_source(cell20, NEW_CELL20)
print(f"[CELL 20] replaced OK")

# ── Cell 21 (6c25bccd) ──────────────────────────────────────────────────────

NEW_CELL21 = """\
if gdf_counties_raw is not None:
    from matplotlib.colors import Normalize

    _adopt_cmap = 'Greens'
    _adopt_norm = Normalize(vmin=0, vmax=100)

    print("\\n--- Summary: adoption_rate_pct ---")
    for mp in selected_mps:
        _v = econ_adoption_rate_results[mp]['adoption_rate_pct'].dropna()
        _pct_high = (_v >= 50).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f}% | med={_v.median():.1f}% | "
              f"mean={_v.mean():.1f}% | max={_v.max():.1f}% | "
              f"{_pct_high:.1f}% of counties >= 50% adoption potential")

    plot_combined_choropleth(
        gdf_counties_raw, econ_adoption_rate_results,
        column='adoption_rate_pct',
        title_template='Economic Adoption Potential — MP{mp}\\n(Incremental Cost Recovered, IRA-Ref)',
        cbar_label='Economic Adopters — Incremental Cost Recovered (%)',
        cmap=_adopt_cmap, norm=_adopt_norm,
        selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_econ_adoption_rate_combined.png'),
    )

    print("[OK] Economic adoption choropleth generated")
else:
    print("[WARN] Adoption choropleth skipped — county shapefile not available")\
"""

idx21, cell21 = find_cell(nb, "6c25bccd")
old_preview21 = "".join(cell21["source"])[:80]
print(f"[CELL 21] old: {old_preview21!r}")
set_source(cell21, NEW_CELL21)
print(f"[CELL 21] replaced OK")

# ── Save ─────────────────────────────────────────────────────────────────────

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n[DONE] Notebook saved.")
