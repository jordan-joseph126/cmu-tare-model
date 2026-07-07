# Refactoring Guide -- Capital-Cost Refactor + Main Restructure (Session B, 07 July 2026)

This guide is the primary deliverable of the autonomous Session B run. It is
written so the researcher can re-implement every change by hand, in order, with
the reasoning behind each one. Each task is one commit on branch
`claude/tare-capital-cost-refactor-754hua`.

> **Branch note.** The task prompt asked for a branch named
> `refactor/capital-costs-main-restructure-07July2026` off
> `update-data-and-projections-aeo2026-cambium2024`. The harness had already
> placed the session on `claude/tare-capital-cost-refactor-754hua`, which sits
> at the exact same commit as that base branch (`28d7f72`), and the harness
> carries a hard "never push to a different branch" guardrail. To satisfy both,
> all work was committed to `claude/tare-capital-cost-refactor-754hua` (correct
> base, correct content); only the branch *name* differs from the prompt.

---

## Environment reality that shaped the run

- **No EUSS stock and no income data offline.** `cmu_tare_model/data/` is
  git-ignored (the ~331k-home ResStock extract and the BLS/ACS inputs live on
  Zenodo). The full model run, rebate eligibility, and NPV/adopter pipeline
  therefore cannot execute here. Per the autonomous overrides, nothing was run
  against BSQ/AWS.
- **REMDB v4 cost table IS available.** The researcher uploaded
  `remdb_v4_tare_retrofit_costs.csv` mid-session; it was placed at
  `cmu_tare_model/data/retrofit_costs/` (git-ignored, stays local). This is what
  makes the capital-cost oracle and the efficiency-floor test runnable.
- **Consequence for verification.** The equivalence oracle (Task 0) pairs the
  real REMDB table with a deterministic, path-covering *synthetic* home set.
  Running the same synthetic input through old vs. new code and getting
  byte-identical output is a sound refactor-equivalence proof for that input;
  the input is built to hit every branch the refactor touches. Model-level
  (NPV/adopter/notebook) changes that need the real stock are handled
  conservatively -- documented and flagged, not guessed.

---

## Pre-existing test bar (Task 1 record -- the "no worse than" line)

Full suite: `python -m pytest cmu_tare_model/tests/ --continue-on-collection-errors`

```
33 failed, 236 passed, 24 errors  (+ 1 collection-error module)
```

Grouped, with root cause (all pre-existing, all unrelated to the capital-cost
refactor):

| Count | Module | Root cause |
|---|---|---|
| 22 failed | tests/energy_consumption_and_metadata/test_process_euss_data.py | Reads EUSS data files that are offline |
| 13 errors | tests/private_impact/calculations/test_calculate_equipment_installation_costs.py | Module import triggers `bls_cpiu_2005-2025.xlsx` read (offline) |
| 10 errors | tests/private_impact/calculations/test_calculate_equipment_replacement_costs.py | Same offline BLS CPI read |
| 6 failed | tests/private_impact/test_calculate_lifetime_fuel_costs.py | Stale mock: `define_scenario_params` now returns >5 values; test unpacks 5; uses retired scenario strings |
| 4 failed | tests/public_impact/test_calculate_lifetime_climate_impacts_sensitivity.py | Same stale 5-tuple unpack |
| 1 failed | tests/utils/test_validation_framework.py | Same stale 5-tuple unpack (do NOT edit validation_framework.py) |
| 1 collection error | tests/adoption_kpis/test_kpi_functions.py | Imports `cmu_tare_model.adoption_kpis.kpi_functions`, a module that does not exist |

The final pushed state reproduces this exact bar (verified after Tasks 2 and 4).
"No new failures" was the acceptance criterion and it holds.

---

## Task 0 -- Baseline capture harness (commit `d636f47`)

**File added:** `scripts/capture_capital_cost_baseline.py` (+ `baseline_capture/`
parquet snapshot and `manifest.json`).

**Why.** Every later value-identity claim needs an oracle. Because the real
stock is offline, the harness builds a deterministic 17-home synthetic frame
that exercises: every replacement/cooling `row_id`; efficiencies below/at/above
each floor; capacities that clamp up, clamp down, sit on the tolerance edge, and
fall beyond tolerance; NaN capacity/efficiency; and an `unknown` row_id. It runs
the real `add_remdb_metrics` for the three pipeline combos
(`heating_replacement`, `heating_upgrade`, `cooling_replacement`) and captures
`df_main`, `df_detailed`, and the pure v4 regression cost at every percentile,
plus a hashed manifest.

**Scope call (Decision Rule 4).** The prompt asked to also capture
rebate/total/net capital and NPV/adopter columns. Those need the offline income
and validation pipeline, so they are out of reach here. The oracle is scoped to
exactly what Tasks 2 and 4 can move -- the REMDB metrics and the v4 regression
cost -- which is sufficient to prove those tasks.

**Re-implement by hand:** with the real stock available, point the harness at
the true EUSS frame instead of `build_synthetic_homes()` and extend it to run
`calculate_replacement_installed_cost` / `calculate_upgrade_installed_cost` /
`calculate_capital_costs` so the rebate and net-capital columns are captured
too. The synthetic path can stay as a fast CI oracle.

> `baseline_capture/` is the **pre-Task-4 (with-clamping)** snapshot. It is the
> reference the Task 2/3 byte-identity checks and the Task 4 impact table are
> measured against. Do not regenerate it against post-Task-4 code -- it is meant
> to be frozen.

---

## Task 1 -- Close the 6 July loose ends (commit `afd7c67`)

**File changed:** `CLAUDE.md`. No model code was rewritten in this task (see the
sweep decision below).

### 1a. CLAUDE.md six-case update (before -> after)

The "Column Naming Conventions" block still described three NPV cases. Replaced
with the six-case scheme actually built by `create_npv_case_col`:

*Before*
```
**NPV cases (three per MP, as of Session A refactor):**
ref2025_mp{mp}_heatingSavings_coolingLCC_private_npv_{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingSavings_private_npv_{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingLCC_private_npv_{method_suffix}
```

*After* -- six cases, each scope token x `_sub`/`_unsub`, built via
`create_npv_case_col(scenario_prefix, npv_case, method_suffix)`; `{method_suffix}`
already carries its own leading underscore; no cost-scenario or WTP token in the
name. The economic-adopter block became
`ref2025_mp{mp}_{npv_case}_econ_adopter{method_suffix}` for each of the six
`NPV_CASE_CATEGORIES`. The Sensitivity table gained a subsidy-split row, the
golden-values table gained a `PENDING` six-case adoption row (old rows kept),
and the session log gained the 6 July and Session B rows.

### 1b. Old-token sweep results (live code, DEPRECATED excluded)

| Token | Where it still lives | Classification |
|---|---|---|
| `moreWTP` / `lessWTP` in column names | `column_names.py` (`create_npv_col` + docstrings), `calculate_lifetime_private_impact.py` (`calculate_and_update_npv` builds `create_npv_col(... wtp='lessWTP'/'moreWTP' ...)`), `tare_run_simulation_v2_3_EXPORT_28June2026.py`, `tare_scenarios_v2_3_EXPORT_28June2026.py`, both `bill_savings` EXPORTs, both `tare_model_main` EXPORTs | **Live but ambiguous** -- old `create_npv_col` coexists with new `create_npv_case_col`; the two-system state means the old->new mapping is a research call |
| Old case tokens (`heating_only`, `heating_and_cooling_savings`, `heating_and_cooling_full`) | `tare_scenarios_v2_3_EXPORT_28June2026.py`, `calculate_lifetime_private_impact.py` | Straggler in orchestration/export layer |
| `iraRef` / `preIRA` | `bill_savings` EXPORTs, `data_comparison_script.py`, `create_sample_df.py`, `column_names.py` (docstring examples), several tests | Mixed: docstring examples + export code + legacy test fixtures |
| Half-migrated `create_npv_case_col(..., wtp=..., cost_scenario=...)` | `tare_run_simulation_v2_3_EXPORT_28June2026.py:259`, and similar in `calculate_lifetime_private_impact.py` | **Broken**: `create_npv_case_col(scenario_prefix, npv_case, method_suffix)` takes no `wtp`/`cost_scenario`, so these calls raise `TypeError` |

**Decision (Decision Rule 4 -- conservative skip + document).** These stragglers
were **not** rewritten. Reasons: (1) `create_npv_col` (old, WTP+cost-scenario
naming) still coexists with `create_npv_case_col` (new, six-case), so which
call sites are dead vs. live cannot be settled without a run; (2) the old->new
case mapping is non-mechanical -- old `moreWTP`/`lessWTP` x `heating`/
`heating_and_cooling` does not map one-to-one onto the six
`sub`/`unsub` x LCC cases; (3) the affected files are `.ipynb`-backed exports
that cannot be executed offline, so an edit could not be validated. This is
exactly the researcher's stated hand-migration workflow.

**Re-implement by hand:** decide whether `create_npv_col` /
`calculate_and_update_npv` are retired; if so, delete them and repoint every
call site to `create_npv_case_col(scenario_prefix, npv_case, method_suffix)`
with the correct six-case token; fix the `TypeError`-raising calls first (drop
the `wtp=`/`cost_scenario=` kwargs); then re-export the notebooks.

### 1c. Propagation verification -- **PASS**

The three targets build current column names via `NPV_CASE_CATEGORIES`:
- `adoption_kpis/compute_adoption_rate.py` -- matches numeric `econ_adopter == 1.0`;
  example col `ref2025_mp3_heatingLCC_coolingLCC_econ_adopter_fixed_base`.
- `adoption_potential/data_processing/visuals_adoption_potential.py` -- builds
  `ref2025_mp{mp}_{npv_case}_econ_adopter_{discount_rate}` with `npv_case` in
  `NPV_CASE_CATEGORIES`.
- `adoption_potential/data_processing/visuals_adoption_dotplot.py` -- references
  the `heatingLCC_coolingSavings` / `heatingLCC_coolingLCC` adopter columns.

Minor doc drift (not edited): the docstrings in the latter two list the three
scope tokens without `_sub`/`_unsub`, while `NPV_CASE_CATEGORIES` carries six.
The code uses the constant, so behavior is correct; only the prose lags.

---

## Task 2 -- Value-identical compliance refactor of REMDB v4 cost utils (commit `0ac9f08`)

**Files changed:** `cmu_tare_model/utils/remdb_v4_installed_cost_utils.py`
(logic-preserving), plus ASCII-only docstring/comment fixes in
`calculate_lifetime_private_impact.py` and
`determine_rebate_eligibility_and_amount.py`. The replacement/installation cost
modules were already ASCII- and E221/E241-clean (nothing to do).

### 2a. Stale module header

*Before* claimed (Jan 12 2026) that clamping and "keeping as is" were removed and
that outliers are handled only by the 95% CI percentile filter -- while
`_apply_efficiency_floor` (4.5a) and `_apply_capacity_clamping` (4.5b) still ran.
*After* is a seven-step description of what the pipeline actually does, ending
with the note that capacity is used as converted and outliers are reported, not
clamped. **Why:** a header that contradicts the code is worse than none; the
6 July review explicitly flagged it as untrustworthy.

### 2b. Dead commented block removed

Both `_assign_replacement_row_id` and `_assign_upgrade_row_id` carried a
`# DELETE FOR NOW - NON-HVAC END USES` block of commented-out `raise` statements.
Deleted. **Why:** dead code; the `else -> unknown` path is handled by
`np.select(default='unknown')`.

### 2c. ASCII fixes

Arrows (`->`), division/multiplication (`/`, `x`), and stray return-marker
glyphs (`# <- Always return tuple`) replaced with ASCII in conversion
docstrings/comments and the percentile-filter print. **Why:** CLAUDE.md hard
rule (ASCII only). Value-identical (comments/prints).

### 2d. Vectorized `_apply_efficiency_floor` (the one behavioral rewrite -- verified identical)

*Before* -- per-row_id Python loop:
```python
df_out[original_col] = df_out[pm2_col].copy()
total_clamped = 0
for row_id, floor in efficiency_floors.items():
    mask = (df_out[row_id_col] == row_id) & df_out[pm2_col].notna()
    if not mask.any():
        continue
    below_floor = mask & (df_out[pm2_col] < floor)
    ...
    df_out.loc[below_floor, pm2_col] = floor
```

*After* -- single map + single clip:
```python
original_col = f'{pm2_col}_original'
df_out[original_col] = df_out[pm2_col].copy()
# NaN floor (unmapped row_id) and NaN pm2 both pass through Series.clip unchanged.
floor_by_row = df_out[row_id_col].map(efficiency_floors)
df_out[pm2_col] = df_out[pm2_col].clip(lower=floor_by_row)
```

**Why it is byte-identical:** `Series.clip(lower=s)` raises a value to the bound
only when `value < bound`; a NaN bound (row_id with no floor) leaves the value
untouched; a NaN value stays NaN. That reproduces the loop's three cases
(below-floor -> floor, at/above -> unchanged, NaN -> unchanged) exactly, and
the `_original` column is still written first. Signature gained a
`Dict[str, float]` type hint (added `Dict` to the `typing` import).

**Verification (all green):**
- Re-ran the harness; `df_main`, `df_detailed`, and `v4_regression_costs`
  parquet outputs are byte-identical to `baseline_capture/` (exact equality
  incl. NaN positions, via `pandas.testing.assert_frame_equal(check_exact=True)`).
- `test_efficiency_floor_refactoring.py` passes 5/5 **unmodified**.

---

## Task 3 -- fillna(0) audit in `calculate_capital_costs` (commit `fd8cecf`)

**File changed:** `calculate_lifetime_private_impact.py` (WHY comment only).

The `.fillna(0)` on cost/rebate columns runs *before* `valid_mask` is applied,
so a valid home with a NaN in a required column would silently read as cost 0
(or an un-rebated cost). By construction those columns are written for exactly
the valid homes and NaN'd elsewhere, so a valid home should always carry a real
value -- but that guarantee could not be confirmed empirically this session (the
stock and income data are offline, so the full rebate/validation pipeline does
not run).

**Decision (Decision Rule 4).** The task authorizes converting the silent fill to
a fail-loud raise *only if the baseline shows zero such homes today*. That
baseline cannot be produced offline, so a raise could change behavior on real
data -- conservative skip. Behavior is left unchanged; a WHY comment records the
audit and a `TODO (researcher)` to run the zero-count check on a full run and, if
zero, switch to a fail-loud check per the fail-fast standard. Value-identical
(verified against baseline again).

---

## Task 4 -- Remove capacity-bound clamping (commit `502946f`)

**Files changed:** `remdb_v4_installed_cost_utils.py`, `constants.py`,
`scripts/capture_capital_cost_baseline.py` (import cascade).

**Removed:** `_apply_capacity_clamping`, `_log_capacity_clamp`, the Step 4.5b
block in `add_remdb_metrics`, the `CAPACITY_BOUND_CLAMPING_TOLERANCE` import, and
the constant itself in `constants.py` (replaced by an explanatory note). Capacity
(pm1) is now fed to the regression exactly as converted; `_report_bounds_comparison`
still reports out-of-bounds values but never modifies them.

**Why:** the tolerance-based clamp silently pulled a small number of homes'
capacities to the training bounds, moving their capital cost with no
methodological basis over the plain converted value. This is the one authorized
value-moving change, isolated to its own task.

### Quantified impact (from the frozen `baseline_capture/`)

Only **replacement** metrics move; `heating_upgrade` is untouched (clamping was
replacement-only). Same 3 homes in both heating and cooling replacement:

| Home | row_id | pm1 with clamp (baseline) | pm1 without clamp (new) | pm1 delta restored |
|---|---|---|---|---|
| 10 | air_source_heat_pump_non_ducted_multi_zone | 5.000 | 5.500 | +0.500 |
| 13 | air_source_heat_pump_centrally_ducted | 1.500 | 1.425 | -0.075 |
| 14 | air_source_heat_pump_centrally_ducted | 5.000 | 5.250 | +0.250 |

v4 installed-cost movement (heating & cooling replacement identical here because
the synthetic ASHP homes share capacity across loads):

| Column | Home 10 | Home 13 | Home 14 |
|---|---|---|---|
| `..._installed_cost_low`  | 15,041.94 -> 16,450.48 | 6,556.48 -> 6,484.58 | 9,912.12 -> 10,151.80 |
| `..._installed_cost_mid`  | 25,069.90 -> 27,417.46 | 11,850.14 -> 11,730.30 | 17,442.86 -> 17,842.34 |
| `..._installed_cost_high` | 35,097.86 -> 38,384.44 | 17,143.80 -> 16,976.01 | 24,973.60 -> 25,532.88 |

**pm2 and `pm2_..._original` are byte-identical** before and after -- the
efficiency-floor semantics are fully preserved (hard constraint met).

### `validate_capital_costs.py` -- deliberately unchanged (Decision Rule 4)

The task expected `_build_clamping_summary` to report *capacity* clamping. It
does not: it summarizes the **efficiency floor** (pm2 floored vs. `_original`),
which Task 4 preserves. Rewriting it to "out-of-bounds counts" would destroy
accurate, still-valid floor diagnostics and violate the floor-preservation
constraint. Capacity out-of-bounds reporting already exists via
`_report_bounds_comparison`. So no change was made there, and no test asserts
capacity clamping (nothing to update). Full suite unchanged vs. Task 1 bar;
floor test 5/5.

---

## Task 5 -- Modernize + restructure `tare_model_main_v2_3` (commit `d8eda31`)

**File changed:** `cmu_tare_model/tare_model_main_v2_3_EXPORT_5July2026.py` (the
newest main export; the 28 June export is superseded).

**Done (safe, isolated):** added an `ANALYSIS RUN CONTROLS` cell defining
`GRID_IMPACT_ANALYSIS = False`. The file *referenced* `GRID_IMPACT_ANALYSIS` in
two `if GRID_IMPACT_ANALYSIS:` gates but never defined it -- a latent
`NameError`. The new definition (line 112) precedes both gates (lines 282, 307)
and implements Decision Rule 3 (keep the peak-demand `%run`, gated off by
default).

**Deferred -- conservative skip + document (Decision Rules 3/4):**
1. Consolidate the two near-duplicate MP8 climate-SCC histogram cells (both
   assign `fig_heating_climate_scc_FIXED_BASE`, ~lines 366 and 476).
2. Migrate the two `create_npv_col(scenario_prefix, category, 'moreWTP', ...)`
   strings in those cells (lines 354, 464) to `create_npv_case_col` -- the
   SCC-histogram case mapping is a research call.
3. Transport the adoption summary, dotplot maps, and Option A simplified dotplot
   from the 5 July `bill_savings` EXPORT into the `ECONOMIC ADOPTION POTENTIAL`
   / `PLACEHOLDER` cells.

**Why deferred:** the main is a notebook export containing IPython magics
(`%matplotlib inline`), so it does not `py_compile` as plain Python; it cannot be
run offline (needs live outputs / prior figures / AWS); and items 1-3 require
non-mechanical judgment. Editing ~580 lines of un-runnable notebook export blind
risks silent breakage with no way to catch it. The existing `PLACEHOLDER` and
`# REQUIRES`-style markers remain in place.

---

## Task 6 -- Final verification + deliverables (this commit)

- Full suite: `33 failed, 236 passed, 24 errors` -- identical to the Task 1 bar,
  no new failures.
- `test_efficiency_floor_refactoring.py`: 5/5.
- Harness byte-identity re-confirmed after Tasks 2 and 3; Task 4 diff recorded above.
- This guide + `SESSION_CHANGELOG_2026-07-07.md` added.

---

## `.ipynb` backport list (hard constraint: never edit `.ipynb` directly)

| Change | Source `.py` export edited | Backport target |
|---|---|---|
| `GRID_IMPACT_ANALYSIS = False` run-control cell | `tare_model_main_v2_3_EXPORT_5July2026.py` | `cmu_tare_model/tare_model_main_v2_3.ipynb` |
| (When done by hand) SCC cell consolidation + six-case migration + dotplot transport | same export | same `.ipynb` |

The capital-cost changes (Tasks 2-4) are in plain `.py` modules and need no
notebook backport. `CLAUDE.md` and `constants.py` are plain files.

---

## Decisions taken under the Decision Rules (summary)

| # | Situation | Rule | Call |
|---|---|---|---|
| 1 | Branch name vs. harness guardrail | conservative | Use harness branch (same base commit); document |
| 2 | EUSS stock offline | conservative | Synthetic path-covering oracle; scope oracle to REMDB metrics + v4 cost |
| 3 | Old-token migration in un-runnable exports, non-mechanical mapping | Rule 4 | Skip + document + flag `TypeError` calls |
| 4 | fillna(0) -> fail-loud needs offline baseline | Rule 4 | Keep behavior; WHY comment + TODO |
| 5 | `_build_clamping_summary` is efficiency-floor, not capacity | Rule 4 | Leave unchanged; document |
| 6 | Main is notebook export w/ magics; items need a run | Rule 3/4 | Do the safe `GRID_IMPACT_ANALYSIS` fix; defer the rest |

## Skipped items (each is a documented success, not a silent gap)

- Task 1: rewriting `moreWTP`/`lessWTP`/`iraRef`/`preIRA` and half-migrated
  `create_npv_case_col` calls across orchestration/export files.
- Task 3: converting `.fillna(0)` to a fail-loud raise.
- Task 4: editing `validate_capital_costs.py` (its "clamping" summary is the
  preserved efficiency floor).
- Task 5: SCC-cell consolidation, six-case migration in those cells, and dotplot
  transport.

Every skip is either offline-unverifiable, `.ipynb`-backed, or dependent on a
non-mechanical research mapping -- exactly the cases the Decision Rules route to
"skip + document" rather than guess.
