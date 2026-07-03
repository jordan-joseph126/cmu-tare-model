# TARE Model -- Retire the RCM / CR / Health Vestiges End-to-End (v2)

> **Version notes.** v2 supersedes v1 (which scoped only the results-dict collapse). This
> version is comprehensive: it audits for AND removes every lingering reference to RCM models,
> CR functions, and health-related content across the live codebase -- the nested results-dict
> key, the on-disk CSV directory scheme, the adoption column-name schema, the `constants.py`
> definitions, stale health narrative comments, and the health-bearing test scaffolding --
> while preserving all CLIMATE computation. Pairs with the completed health/public-NPV removal
> (see "What Was Done Before").

---

## Your Role

You are an expert research-code mentor finishing a multi-phase cleanup in a heat-pump
electrification project (Joseph et al. 2026, ResStock 2022.1.1 / EUSS). The health-damage and
combined public-NPV computation has already been removed from the live pipeline. What remains is
a scatter of *vestiges* of the retired health sensitivity: a redundant RCM nesting key in the
results dictionary and CSV layout, RCM/CR fields still baked into the adoption column-name
schema, the `RCM_MODELS` / `CR_FUNCTIONS` constants, health-flavored comments, and test helpers
that still synthesize health columns. You remove all of it without changing a single output
value -- teaching the researcher WHY at each step, never batching edits, always showing the diff
first.

## Project Context

The model computes heat-pump electrification economics across ~3,098 U.S. counties for measure
packages MP3/MP4/MP8/MP9/MP10 under a single policy scenario, `'2025 Reference Case'`. The
adoption decision is purely economic (`moreWTP >= 0`). Results per measure package are stored in
a nested dictionary keyed by discount rate and RCM model, exported to per-RCM CSV directories,
and reloaded for visualization and downstream KPIs (grid peak demand, bill savings).

RCM models (AP2, EASIUR, InMAP) are air-pollution **health-damage** models; CR functions
(ACS, H6C) are the **health** concentration-response curves. Neither ever affected climate
(SCC-based) or private (fuel-cost-based) results. Health is gone, `RCM_MODELS` is already
reduced to `['inmap']` and `CR_FUNCTIONS` to `['acs']`, and the live adoption columns are the
`econ_adopter_moreWTP` family (which carry NO rcm/cr). So every surviving RCM/CR reference is
either a redundant single-key indirection or a stale name from the prior architecture.

## Scope Constraint -- CRITICAL

**In scope:**
- Export notebook files: `model_scenarios/tare_scenarios_v2_3_EXPORT_28June2026.py`,
  `model_scenarios/tare_run_simulation_v2_3_EXPORT_28June2026.py`,
  `tare_model_main_v2_3_EXPORT_28June2026.py`,
  `adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_28June2026.py`,
  `grid_impact/calculate_postTARE_ts_aws_peak_demand_18May2026.py`.
- Export/load utilities: `utils/export_model_run_results.py`, `utils/load_exported_results_to_df.py`.
- Downstream KPI / viz modules: `adoption_potential/data_processing/visuals_adoption_potential.py`,
  `grid_impact/peak_load_functions.py`, `adoption_kpis/compute_adoption_rate.py`,
  `adoption_kpis/bill_savings.py`.
- Column-name builders: `utils/column_names.py` (the RCM/CR-bearing builders ONLY).
- `constants.py` (`RCM_MODELS`, `CR_FUNCTIONS`).
- Test scaffolding that synthesizes or asserts health/RCM/CR: `utils/create_sample_df.py`,
  `tests/conftest.py`, `tests/test_constants.py`, both `test_column_names.py` copies, and the
  health/tiered-adoption test modules.
- Stale health narrative comments in otherwise-live modules.

**Out of scope and never edited:** `utils/validation_framework.py`, any `.ipynb` (backport by
hand), the EIA fetch script, the preserved TARE/EUSS load and demand-computation cells. The
`DEPRECATED_health_impacts/*` modules and `determine_adoption_potential_sensitivity.py` are
header-only-deprecated -- do NOT rewrite their logic; touch only their `constants` imports per
Task 8. If a proposed change would touch out-of-scope code or move a golden value, STOP and
report.

## Load-Bearing Principle (Non-Negotiable)

**Removing RCM/CR/health is not removing climate.** Climate-damage computation stays in full:
`calculate_lifetime_climate_impacts`, `df_*_damages_climate`, `lookup_emissions_electricity_climate`,
`lookup_emissions_fossil_fuel`, `SCC_ASSUMPTIONS` (lower/central/upper), `MER_TYPES`
(lrmer/srmer), `create_climate_npv_col`, `create_lifetime_damages_col` /
`create_avoided_damages_col` (their generic `mer_type_or_rcm` / `scc_or_cr` parameters are used
by CLIMATE). Treat "remove climate", "touch SCC", and "touch MER" as forbidden actions. An
over-eager agent that strips a climate column because its builder has `rcm` in a parameter name
will silently change every downstream sensitivity.

**This is a shape/name change, not a value change.** Collapsing `{discount_rate: {rcm: df}}` to
`{discount_rate: df}` and removing rcm/cr from column names must leave every adoption rate, NPV,
climate-damage value, and demand figure byte-identical. The before/after capture (Tasks 2 and 9)
exists to prove that. A moved golden value means a bug, not a cleanup.

**Collapse the dimension; do not hardcode a single RCM.** The fix removes the `rcm` level from
dicts, CSV paths, and loops -- NOT leaving the structure and hardcoding `rcm_model='inmap'`.
Sites that index `[discount_rate][rcm_model]`, `[discount_rate][RCM_MODELS[0]]`, or
`[...][RCM_MODEL_KEY]` become `[discount_rate]`.

## What Was Done Before (28-29 Jun 2026)

- `define_scenario_params` arity fixed to a 5-tuple across all live consumers.
- Combined public NPV and all live `df_*_damages_health` / `_health_npv_` / `calculate_public_npv`
  code removed from the four export files; climate preserved.
- `CR_FUNCTIONS` removed from the export files (import in `tare_scenarios`; import + status-banner
  line in `tare_model_main`). `RCM_MODELS` deliberately left in place because it is the
  structural storage key -- which is THIS task.
- Constants now: `RCM_MODELS = ['inmap']`, `CR_FUNCTIONS = ['acs']` (AP2/EASIUR/H6C commented out).

## Attached / Relevant Files

(See the layered inventory below for line numbers. Line numbers predate this session's first
edit; re-grep to confirm before editing.)

## Known Reference Points

**Results dictionary (current -> target):**
```
{discount_rate: {rcm_model: DataFrame}}   ->   {discount_rate: DataFrame}
DATAFRAMES_MPX_RCM_DISCOUNT_RATE, DATAFRAMES_MP{N}_RCM_DISCOUNT_RATE_RESULTS, DATAFRAMES_BY_MP[mp]
```

**Summary CSV directory (current -> target):**
```
retrofit_mp{N}_results/summary_mp{N}_{rcm_model}_{discount_rate}/mp{N}_results_{loc}_{date}.csv
  -> retrofit_mp{N}_results/summary_mp{N}_{discount_rate}/mp{N}_results_{loc}_{date}.csv
```
Damages-climate and fuel-cost CSVs are NOT rcm-keyed and do not change.

**Adoption column-name schema (the second, subtler layer):**
`create_adoption_col` builds `..._{column_type}_{scc}_{rcm}_{cr}_{cost}{method}` (e.g.
`iraRef_mp3_heating_adoption_central_inmap_acs_v4MID_fixed_base`). The LIVE adoption columns are
now the `econ_adopter_moreWTP` family with NO rcm/cr (and the `ref2025_mp{mp}_` prefix). So
`find_adoption_column` (peak_load_functions) builds a column that no longer exists and survives
only on its fuzzy fallback; it also still uses the retired `iraRef` prefix. Fixing this layer
means building the current `econ_adopter_moreWTP` column name (no rcm/cr) instead.

**Operational consequence (raise up front):** changing the summary directory scheme makes
previously-exported summary CSVs unreadable by the new loader. After the refactor the model must
be re-run to regenerate outputs, OR the existing `summary_mp{N}_inmap_{discount}` directories
renamed to `summary_mp{N}_{discount}` as a one-time migration. Confirm the researcher's choice.

## Layered Reference Inventory (live, repo-wide)

### Layer 1 -- RCM nesting key (dict + CSV layout)
| File:line | Site | Action |
|---|---|---|
| `constants.py:91` | `RCM_MODELS = ['inmap']` | remove (Layer 3) |
| `tare_scenarios_EXPORT:674,682-687,717-720,803-825,855-884,831-833,892-894` | import; dict build; cost-merge / private-NPV / adoption loops; verbose prints | flatten to `{discount_rate: df}` |
| `export_model_run_results.py` (sig + `summary_mp{N}_{rcm_model}_{discount_rate}` dir + validation) | exporter contract | drop rcm from scheme |
| `load_exported_results_to_df.py:7,106-107,119,222,232,245` | import; validation; dir; dict build; load loop | drop rcm; build `{discount_rate: df}` |
| `tare_run_simulation_EXPORT:17,225,382,535,687,841,255,413,565,717,871,909-911` | import; per-MP export loops; `RCM_MODELS[0]` verification; CSV-count math | flatten; correct counts |
| `tare_model_main_EXPORT:38,95` + climate-viz index | import; banner; `[discount_rate][rcm_model]` w/ `rcm_model='inmap'` | remove import/banner; index `[discount_rate]` |
| `visuals_adoption_potential.py:614` (+ import) | `[discount_rate][RCM_MODELS[0]]` | index `[discount_rate]` |
| `adoption_potential/data_processing/visuals_adoption_potential copy.py` | STRAY DUPLICATE of the above | confirm unused; delete file or apply same change |
| `grid_impact/...18May2026.py:43,373,376,387-393` | import; `RCM_MODEL_KEY='inmap'`; key-check + `df_tare_nested[RCM_MODEL_KEY]` | remove key layer; index `[discount_rate]` |
| `adoption_kpis/...bill_savings_EXPORT:229,255,439` | `DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']` | index `['fixed_base']` |
| `adoption_kpis/bill_savings.py:84` | docstring example with `['fixed_base']['inmap']` | update docstring |

### Layer 2 -- adoption column-name schema (scc/rcm/cr embedded)
| File:line | Site | Action |
|---|---|---|
| `column_names.py:425 create_adoption_col` | builds `_{scc}_{rcm}_{cr}_`; `benefit`/`adoption`/`impact`/`health_sensitivity` types | retire rcm/cr-bearing branches OR retire the builder if only the deprecated tiered path uses it (audit) |
| `column_names.py:474 create_total_npv_col` (`health_only` branch + `rcm_model`/`cr_function` params) | health-only total NPV string | drop health_only path; keep `climate_only` path |
| `peak_load_functions.py:40-82 find_adoption_column` | builds `iraRef_mp{mp}_..._adoption_central_inmap_acs_...` via `create_adoption_col`; fuzzy fallback | rebuild to the current `econ_adopter_moreWTP` column (no rcm/cr; `ref2025_` prefix) |
| `compute_adoption_rate.py:~90` | docstring + "adopter_tiers" semantics referencing the rcm/cr column | update to the `econ_adopter_moreWTP` column |

> ADJACENT STALENESS (flag, do not silently expand scope): the Layer-2 sites also carry the
> retired `iraRef_` prefix and the deprecated tiered-adoption "tier" semantics. Removing rcm/cr
> here unavoidably touches those names. Surface this to the researcher: the minimal change points
> these consumers at the current `ref2025_..._econ_adopter_moreWTP_...` columns. A full
> tiered-adoption / `iraRef`->`ref2025` retirement is a separate task -- do not chase it beyond
> what rcm/cr removal forces.

### Layer 3 -- the constants and dead builders
| File:line | Site | Action |
|---|---|---|
| `constants.py:85,91` | `CR_FUNCTIONS`, `RCM_MODELS` definitions | delete (after Layers 1-2 remove all live uses) |
| `column_names.py:316,339` | commented-out `create_health_npv_col`, `create_public_npv_col` | delete the dead commented blocks |

### Layer 4 -- health narrative comments + test scaffolding
| File:line | Site | Action |
|---|---|---|
| `determine_economic_adoption_potential.py:44,123` | comments "no help from monetized climate or health damages" / "Climate/health damages are deliberately absent" | reword to reflect health removed; keep the (true, useful) point that CLIMATE damages do not enter the decision |
| `degree_day_consumption_utils.py:~202` | docstring "for climate/health modules" | reword to "climate modules" |
| `create_sample_df.py:21,47-48,60-61,114-147,170-213` | synthesizes `damages_health_{model}_{cr}`, `health_npv`, `public_npv`, `health_sensitivity`, `health_damages` group | remove health column generation + the `health_damages`/`npv health/public` groups; keep climate generation |
| `conftest.py:18,20,21,155,165` | references DEPRECATED health-lookup modules; `mock_elec_health`; stale 6-tuple `mock_scenario_params` (also retired `preIRA`/`iraRef` prefixes) | drop health module refs; make the mock return the 5-tuple with current prefixes |
| `tests/test_constants.py:16-17,98,103` | imports + asserts `set(RCM_MODELS)=={'inmap'}`, `set(CR_FUNCTIONS)=={'acs'}` | remove those imports + assertions |
| `tests/utils/test_column_names.py`, `utils/tests/test_column_names.py` | tests for `create_adoption_col` / health builders (TWO copies -- confirm which is live) | update/remove rcm/cr/health cases; resolve the duplicate |
| `tests/public_impact/test_calculate_lifetime_health_impacts_sensitivity.py`, `..._public_impact_sensitivity.py` | health/public test modules | remove or mark deprecated (they target removed code) |
| `tests/adoption_potential/test_determine_adoption_potential_sensitivity.py` | tiered-adoption tests; monkeypatch RCM/CR | header-deprecated; update monkeypatch only if it blocks collection |

### Out of scope / false positives (do NOT change)
`utils/inflation_adjustment.py`, `utils/inflation_adjustment_BACKUP_28June2026.py`,
`utils/hdd_consumption_utils.py` (already-deprecated), `private_impact/calculate_lifetime_fuel_costs.py`
(matched only on incidental text) -- confirm during the audit, then leave.

## Required First Action (the exhaustive audit -- no edits)

1. Re-grep the repo for the full marker set and classify EVERY hit as KEEP-climate, REMOVE
   (rcm/cr/health), or FALSE-POSITIVE:
   `RCM_MODELS`, `CR_FUNCTIONS`, `rcm_model`, `cr_function`, `RCM_MODEL_KEY`, `['inmap']`,
   `_inmap_`, `_acs`, `easiur`, `ap2`, `damages_health`, `_health`, `health_npv`, `public_npv`,
   `healthOnly`, `health_sensitivity`, `check_health`, `vsl`, `create_adoption_col`,
   `find_adoption_column`.
2. Open `export_model_run_results.py` and `load_exported_results_to_df.py` end to end; confirm
   the exact directory-build lines, the `rcm_model` validation, and that damages-climate /
   fuel-cost paths are NOT rcm-keyed.
3. Open `grid_impact/...18May2026.py` and `peak_load_functions.py` end to end; confirm how the
   RCM key and the adoption column name flow, and whether `find_adoption_column`'s exact match
   currently fails (relying on the fuzzy fallback) against live `econ_adopter_moreWTP` columns.
4. Determine which of the two `test_column_names.py` files is the live one.
5. Confirm with the researcher: (a) old-output handling -- re-run vs directory rename; (b) the
   Task 8 deprecated-constants choice.

Report the classified inventory before proposing any diff.

## Tasks

Order matters: capture the baseline, then change producer + storage contract together, then
consumers, then column-name schema, then the constant, then deprecated/test, then verify. The
pipeline must never be left in a half-shape across a stop gate.

### Task 1 -- Capture the golden baseline (P0, no edits)
From a current run (or existing outputs), capture per MP: `summarize_econ_adopters` /
county-adoption-rate output, and non-null counts + means of the `*_damages_climate` and NPV
columns. Save as the before-snapshot.

### Task 2 -- Flatten the producer dict in `tare_scenarios` (P0)
`{discount_rate: df_euss_am_mpX_home.copy()}`; remove the inner `for rcm in RCM_MODELS` from the
cost-merge, private-NPV, and adoption loops and the verbose prints; remove the import. One diff
at a time; confirm per-discount-rate DataFrames equal the old `['inmap']` entries.

### Task 3 -- Change the storage contract in export/load (P0)
Drop `rcm_model` from the summary directory name, function signatures, and validation; change
`load_measure_package_data` to build `{discount_rate: df}`; update both Google-style docstrings;
remove the loader's `RCM_MODELS` import. Leave damages-climate / fuel-cost paths untouched.

### Task 4 -- Update the exporter loops in `tare_run_simulation` (P0)
Replace each per-MP `for rcm_model in RCM_MODELS` export with a single per-discount-rate export
(no `rcm_model=`); change each `[...][RCM_MODELS[0]]` verification to `[discount_rate]`; remove
the import; correct the final-summary CSV-count math (CSVs per MP = discount-rate count; replace
the hardcoded `3 MPs` and the Unicode multiplication sign with ASCII `x`).

### Task 5 -- Update the dict consumers (P1)
`tare_model_main` (import, banner, climate-viz index), `visuals_adoption_potential.py:614` (+
the stray `copy.py`), `grid_impact/...18May2026.py` (`RCM_MODEL_KEY` + key-check + index),
`adoption_kpis/...bill_savings_EXPORT` (three `['fixed_base']['inmap']` indices) and
`bill_savings.py` docstring. Each becomes `[discount_rate]`. One diff per file.

### Task 6 -- Retire the adoption column-name rcm/cr layer (P1)
Point `find_adoption_column` (peak_load_functions) and `compute_adoption_rate` at the live
`ref2025_mp{mp}_..._econ_adopter_moreWTP_...` column (no rcm/cr). In `column_names.py`, retire
the rcm/cr-bearing branches of `create_adoption_col` and the `health_only` path of
`create_total_npv_col`, or retire `create_adoption_col` entirely if only the deprecated tiered
path consumes it (decide from the audit). Preserve `create_climate_npv_col` and the
`climate_only` total-NPV path. Surface the adjacent `iraRef`/tiered-adoption staleness; do not
expand into a full prefix migration.

### Task 7 -- Delete the constants + dead builders (P1)
Delete `RCM_MODELS` and `CR_FUNCTIONS` from `constants.py`; delete the commented-out
`create_health_npv_col` / `create_public_npv_col` blocks in `column_names.py`.

### Task 8 -- Neutralize deprecated-module + test references (P2)
Deprecated modules import the constants; per header-only deprecation, do NOT rewrite their logic.
Choose with the researcher: (a) keep the constants as single-element vestiges (skip Task 7
deletion), or (b) add a local fallback (`RCM_MODELS = ['inmap']`) inside each deprecated file.
Then fix the test scaffolding so the suite collects cleanly: `create_sample_df.py` (drop health
column generation), `conftest.py` (5-tuple mock, drop health-lookup refs), `test_constants.py`
(drop RCM/CR asserts), the `test_column_names.py` duplicate, and the health/tiered test modules.

### Task 9 -- Verify end to end (P0)
1. Re-run baseline + one MPX scenario through adoption; export + reload via the new flat scheme.
2. Re-capture the Task 1 metrics; require byte-identical (plumbing/name-only -- numbers must not
   move).
3. Re-grep the full marker set; confirm zero LIVE rcm/cr/health references (deprecated/test
   handled per Task 8).
4. Confirm new summary CSVs land in `summary_mp{N}_{discount_rate}` and reload into a 2-level
   `{discount_rate: df}` dict; confirm grid-impact and bill-savings KPIs run and match.
5. `pytest` collection succeeds with no ImportError.

## Reference Values (golden)

Must be IDENTICAL before vs after (binding check is the live Task 1 vs Task 9 capture):

| Quantity | MP3 | MP4 |
|---|---|---|
| Operating-cost % change, county median | -38.5% | -60.6% |
| Total electricity demand change (GWh) | +427,043.7 | +30,618.4 |
| Median demand % change | +22.5% | -8.1% |
| Mean economic adoption rate (heating only) | 20.8% | 20.5% |

## Code Standards (carry into every edit)

- Economic adopter = `moreWTP >= 0`. Never `lessWTP`, never `v3`, never strict `> 0`.
- Column names via `define_scenario_params(...)[0]` / `create_npv_col` / `create_npv_case_col` /
  `f'mp{mp}'`. Never hardcode `'mp3'` or any scenario prefix.
- Float64 for econ/adopter columns (0.0 / 1.0).
- ASCII only in code, comments, and markdown (`-->` not arrows, `--` not em dash, `x` not the
  multiplication sign, `[OK]` not check marks). Several RCM-count prints contain a Unicode
  multiplication sign -- replace with ASCII `x` when you rewrite those lines.
- Google-style docstrings + type hints on any edited function signature (especially the
  export/load contract and the adoption-column helpers).
- Comments explain WHY, not what.
- One edit per stop gate; show the diff; wait for approval. Never batch across files.
- Backport accepted changes into the `.ipynb` by hand -- never edit `.ipynb` JSON. Note the
  `.ipynb` backport set now includes `grid_impact/calculate_postTARE_ts_aws_peak_demand.ipynb`
  and `adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings.ipynb`.

## Known Anti-Patterns (do NOT suggest)

- Hardcode `rcm_model='inmap'` instead of removing the dict level -- collapse the dimension.
- Alter `SCC_ASSUMPTIONS`, `MER_TYPES`, or any climate-damage computation/column while removing
  rcm/cr/health.
- Edit `create_lifetime_damages_col` / `create_avoided_damages_col` -- their generic
  `mer_type_or_rcm` / `scc_or_cr` params serve CLIMATE.
- Change any output VALUE -- shape/name-only refactor; a moved golden value is a bug.
- Touch the damages-climate or fuel-cost CSV paths (not rcm-keyed).
- Expand into a full `iraRef`->`ref2025` prefix migration or tiered-adoption rewrite -- only do
  what rcm/cr removal forces, and flag the rest.
- Rewrite logic inside `DEPRECATED_health_impacts/*` or `determine_adoption_potential_sensitivity.py`
  (header-only; Task 8 touches their constants import only).
- Leave the pipeline half-shaped (producer flat, loader nested) across a stop gate.
- Edit `validation_framework.py`, any `.ipynb`, or the fetch script. Batch edits. Unicode symbols.

## Appendix A -- Why this is a shape/name change, not a value change

The RCM dimension was meaningful only for health damages (AP2/EASIUR/InMAP are alternative
air-pollution models, so health NPV differed by RCM; CR functions are the health dose-response
curves). Climate (SCC) and private (fuel cost) NPV never depended on RCM or CR. The dict copied
the same climate/private DataFrame into each RCM slot; with health gone and a single `'inmap'`
key, every slot is one identical copy, and no live column name actually carries rcm/cr anymore.
Removing the `rcm` level and the rcm/cr name fields therefore cannot change any stored number --
it removes a redundant indirection (`d[dr]['inmap']` -> `d[dr]`), a redundant CSV directory
segment, and dead name fields. Task 9's invariance check is the proof.

## Appendix B -- The four layers (mental model)

1. **Nesting key** -- `{discount_rate: {rcm: df}}` and `summary_mp{N}_{rcm}_{discount}` CSV dirs.
   Touches producer, exporter, loader, and every consumer that indexes a third level.
2. **Column-name schema** -- `create_adoption_col` and friends embed `_{scc}_{rcm}_{cr}_`; live
   adoption columns no longer do. Re-point consumers at `econ_adopter_moreWTP`.
3. **Constants** -- `RCM_MODELS`, `CR_FUNCTIONS`, and the dead commented health builders.
4. **Narrative + test scaffolding** -- health-flavored comments and the sample-data / conftest /
   test-constants helpers that synthesize or assert health/RCM/CR.

Do the layers in order (1 -> 4); each later layer depends on the earlier ones being gone.

## Session Summary Template

After Task 9, produce: (1) files edited + line ranges, grouped by layer; (2) dict-shape and
CSV-scheme before/after; (3) adoption-column before/after (old rcm/cr name -> live
`econ_adopter_moreWTP`); (4) every rcm/cr/health site changed vs every climate (SCC/MER) item
preserved; (5) before/after econ-adopter + climate-column verification (identical: yes/no);
(6) grid-impact + bill-savings KPI re-run result; (7) pytest collection status; (8) the Task 8
deprecated-constants decision and the old-output migration choice; (9) any references in files
NOT in this session still needing follow-up.
