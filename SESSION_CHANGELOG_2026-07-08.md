# Session Changelog -- 2026-07-08 (Main Notebook Visual-Transfer Repair)

## Session Summary

This session repaired a failed content transfer into the main analysis notebook
`tare_model_main_v2_3`. Two source notebooks (`calculate_postTARE_am_kpis_demand_bill_savings`
and `calculate_postTARE_ts_aws_peak_demand`, both 5 July vintage) had been partially merged
into the main notebook; several *producer* cells did not come across, leaving a P0 NameError
and an empty demand-visuals cell. The repair added the missing producer logic WITHOUT
re-introducing retired column names (`iraRef`/`preIRA`/`moreWTP`/`v4MID` tokens) and without
moving any golden value.

All edits were made to the notebook's `.py` export
`cmu_tare_model/tare_model_main_v2_3_EXPORT_7July2026.py` (the `.ipynb` is not edited directly,
per CLAUDE.md; changes are hand-backported). No module files were modified this session.

Scope delivered: Tasks 1-5 of the session plan. Deferred items are listed at the end.

## Context / Problem

The main notebook's economic-adoption section and BSQ grid-impact scaffolding (AWS init,
baseline/upgrade timeseries, Step 7 profile plot) were already present and already used the
current `'2025 Reference Case'` / `ref2025_mp{mp}_` convention. The merge left three defects:

1. **P0 NameError:** the grid-impact block consumed `adopter_ids_by_mp[mp]` but nothing built
   it. It is produced only by the peak-demand source's Step 4 loop.
2. **Empty demand-visuals cell** under "Visuals - Retrofit Impact on Electricity Demand".
3. **Non-ASCII characters** (checkmarks, one em-dash) in the transferred grid-impact cells.

## Task 1 -- Audit (read-only; key findings)

- **Value-critical retrofit fuel-cost column (resolved by inspecting real CSV headers, not
  just code):**
  - Generator `private_impact/calculate_lifetime_fuel_costs.py` builds
    `{scenario_prefix}heating_lifetime_fuel_cost`; under `define_scenario_params(mp)[0]` the
    prefix is `ref2025_mp{mp}_`.
  - OLD exported run `2026-04-10_00-05` (the timestamp the peak-demand source hardcodes)
    carries `preIRA_mp{mp}_...` and `iraRef_mp{mp}_...`, and NO `ref2025_` column.
  - NEWEST run `2026-07-06_21-55` (folder `summary_mp3_fixed_base`, RCM token dropped)
    carries `ref2025_mp3_heating_lifetime_fuel_cost`, and NO `iraRef`/`preIRA`.
  - Both vintages carry `baseline_heating_lifetime_fuel_cost`.
  - Conclusion: the source's `iraRef_mp{mp}_heating_lifetime_fuel_cost` is a RETIRED prefix.
    On a current-convention run, Task 3 is a pure code-shape RENAME to
    `ref2025_mp{mp}_heating_lifetime_fuel_cost`, derived via the helper and asserted present
    -- NOT a silent value substitution.
- **Helper availability:** `pct_change`, `make_symmetric_norm`, `print_column_summary` are not
  importable anywhere under `cmu_tare_model` (inline defs only). `compute_scenario_demand`,
  `aggregate_demand`, `load_euss_upgrade`, `mp_to_upgrade`, `load_euss_baseline` ARE importable
  from `cmu_tare_model.adoption_kpis`.
- **`compute_bill_savings_ratio`/`aggregate_bill_savings` cannot reproduce
  `operating_cost_pct_change` on current data:** the module hardcodes
  `POLICY_SCENARIOS = ('iraRef','preIRA')` and builds `iraRef_mp{mp}_heating_lifetime_fuel_cost`
  (`adoption_kpis/bill_savings.py:37,120`), which is absent in `ref2025_` runs -> it would
  KeyError. (Also defaults `fuel_filter='Natural Gas'`, applies a `min_home_count` NaN
  threshold, and rounds.) The county-median statistic is algebraically identical to the manual
  `pct_change` path, but the module is itself unmigrated. The manual path was used.
- **`find_adoption_column` returns the ECONOMIC-adopter column** (float 0/1/NaN), NOT a tiered
  column. The source's `extract_adopter_ids` then matches tier STRING labels against it, so
  Tier 1/Tier 2/constrained come back EMPTY -- only `all_filtered` populates. As transferred,
  the grid "Constrained (Tier 1+2)" scenario would silently equal the baseline. Surfaced and
  resolved (see Decisions).
- **Confirmed absent from main:** `upgrade_data`, `compute_scenario_demand`, `aggregate_demand`,
  `SHAPEFILE_PATH`, `load_euss_upgrade`, `mp_to_upgrade`.

## Decisions made (with researcher)

1. **Constrained grid scenario = economic adopters (NPV>=0).** `constrained` = county building
   IDs where the econ-adopter column == 1.0; `all_filtered` = all county building IDs. The
   tier-string path is not used. Consistent with the notebook's economic-adoption framing and
   CLAUDE.md's deprecation of tiered adoption.
2. **Inline helpers** (`pct_change` / `make_symmetric_norm` / `print_column_summary`) kept inline
   in the transferred cell, each with a note that no module home exists yet.
3. **Current-convention run confirmed** (`ref2025_` columns present) -> Task 3 is a rename.

## Detailed Changelog (file: `cmu_tare_model/tare_model_main_v2_3_EXPORT_7July2026.py`)

Line numbers below are current at time of writing; the file was reformatted after editing (a
tool moved the demand/bill-savings cell ahead of the grid-impact section -- functionally fine,
it has no grid dependencies).

### Task 2 -- Fix P0 NameError: adopter-ID builder (new cell, ~line 771)
- Inserted a producer cell that builds `adopter_ids_by_mp` and `adoption_col_by_mp`.
- New import: `from cmu_tare_model.grid_impact.peak_load_functions import gisjoin_to_fips`.
- Per selected MP: derives the econ-adopter column via `find_adoption_column` iterated over
  `REMDB_COST_SCENARIO_KEYS` (no hardcoded `mp`/prefix; cost token ignored downstream). Groups
  buildings by 5-digit county FIPS (`gisjoin_to_fips`) and stores per county:
  - `all_filtered` = all county building IDs (100% adoption bound)
  - `constrained` = building IDs where econ-adopter column == 1.0 (NPV>=0). NaN (excluded
    homes) is not equal to 1.0, so it is correctly left out.
- Prints per-MP county counts and adopter totals.
- Placed after `DATAFRAMES_BY_MP` is loaded and the econ-adopter columns are generated, and
  before the first grid-impact consumer.

### Task 2 follow-on -- Step 7 label (line ~1050)
- Relabeled `scenario_labels = ["100% Adoption", "Constrained (Tier 1+2)"]` to
  `["100% Adoption", "Constrained (Economic Adopters, NPV>=0)"]` to match Decision 1.

### Task 3 -- Demand + bill-savings cell (filled empty cell, ~line 561)
Under "### Visuals - Retrofit Impact on Electricity Demand". Contents:
- New imports from `cmu_tare_model.adoption_kpis`: `load_euss_upgrade`, `mp_to_upgrade`,
  `compute_scenario_demand`, `aggregate_demand`.
- Three inline helpers (with a "no module home yet" note): `pct_change` (line ~578),
  `make_symmetric_norm` (line ~589), `print_column_summary` (line ~602).
- Loads `upgrade_data[mp]` per selected MP.
- `bill_savings_results[mp]`: county median of per-home `(retrofit - baseline)/baseline * 100`
  via the manual `pct_change` path. Retrofit column derived as
  `f"{define_scenario_params(mp, _POLICY)[0]}heating_lifetime_fuel_cost"` (resolves to
  `ref2025_mp{mp}_heating_lifetime_fuel_cost`) and asserted present with an informative
  KeyError; baseline column `baseline_heating_lifetime_fuel_cost`.
- `demand_results[mp]`: `compute_scenario_demand(df_baseline, upgrade_data[mp], fuel_filter=None)`
  then `aggregate_demand(..., geo_level='county')`. Uses aggregate_demand's own
  `elec_change_gwh` and `pct_elec_demand_change` columns.
- Shared symmetric norms (centered at 0, clipped to 2nd/98th percentile across MPs).
- Three county choropleths via `plot_combined_choropleth` (operating-cost %, demand GWh,
  demand %), reusing the already-loaded `gdf_counties_raw` (no shapefile re-read). Results are
  keyed by the GISJOIN `county` column; `plot_combined_choropleth` converts GISJOIN->FIPS
  internally at county level.

### Task 4 -- ASCII cleanup (7 lines in the grid-impact cells)
- Replaced 7 non-ASCII occurrences: checkmark `✓` -> `[OK]` (AWS-creds print, BSQ-initialized
  print, Allegheny bldg_ids print, Step 5 / Step 6 / Step 7 PASSED prints, Figure-saved print),
  and one em-dash `—` -> `--` on the Step 7 PASSED line.

### Task 5 -- Step 3 county-geo mapping + Step 7d fuel table (two new cells)
- **County geography lookup (~line 1079):** builds `county_geo_df` (fips_5digit / county_name /
  state_fips) and merge-ready `gdf_counties` (with geometry) from the already-loaded
  `gdf_counties_raw` (same TIGER `tl_2025_us_county` shapefile -- confirmed identical path via
  `adoption_kpis.data_loading.COUNTY_SHAPEFILE_PATH`; NOT re-read). Validates TIGER columns
  (GEOID/NAME/STATEFP) and that Allegheny FIPS (`TEST_FIPS`) is present. Guarded on
  `gdf_counties_raw is not None`.
- **Allegheny baseline heating-fuel distribution table (~line 1141):** prints counts and
  percentages of baseline heating fuel for the four combinations (MP3/MP4 x constrained/100%),
  reading `adopter_ids_by_mp[mp][TEST_FIPS]`. Adapted the source's nested
  `DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']` access to main's flat
  `DATAFRAMES_BY_MP[mp]['fixed_base']` (main keys by discount rate only, not by RCM). Removed a
  carried-over unused `_mp_label_fuel` dict. Guarded on `GRID_IMPACT_ANALYSIS`.

## Verification performed

- Static checks on the final file: 0 non-ASCII bytes; Python `ast.parse` succeeds (Jupyter
  line magics neutralized for the check); no newly added line exceeds 88 characters.
- NOT verified (requires a live run): golden-value targets. Executing the notebook needs a
  loaded current-convention model run, the county/state shapefiles, and (for the BSQ cells)
  AWS credentials. When run, compare against the CLAUDE.md golden table:
  - Operating-cost % county median ~ -38.5% (MP3) / -60.6% (MP4)
  - Total demand change and demand GWh / % symmetric norms within documented ranges
  Report any delta; do NOT overwrite golden rows (keep superseded rows).

## Open items / follow-ups

- **.ipynb backport:** the new/changed cells must be hand-backported into
  `cmu_tare_model/tare_model_main_v2_3.ipynb` (in progress by the researcher).
- **`bill_savings.py` is unmigrated:** still hardcodes `iraRef`/`preIRA`; would KeyError on
  current runs. Recommend a separate migration session.
- **Deferred (not started):** pre-IRA vs IRA-Ref NPV break-even histogram and subsidy-required
  histogram (built on the retired two-policy split; need a modeling decision first); Step 8
  validation, Step 9 national peak loop, Step 10 CSV export (source stubs raise
  NotImplementedError).

## Notes

- No notebook `.ipynb` files were edited directly this session; only the `.py` export.
- No module or test files were modified this session.
- ASCII-only was maintained across all added cells, comments, and this changelog.
