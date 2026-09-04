# CLAUDE.md — TARE Model / Joseph et al. 2026
# Heat-Pump Electrification Economics (ResStock 2022.1.1 / EUSS)
# Last updated: 3 September 2026 -- notebook cleanup session (missing `plot_county_demand_grid` import restored, Step 7 county-profile loop restored under its `GRID_IMPACT_ANALYSIS` guard, 22 unused imports removed); run `2026-09-02_19-04` verified BYTE-IDENTICAL to `2026-08-19_20-56` across all 17 output files, so no modeled value moved; the six `_sub` / `_sub_june2026` golden rows left at 17 Aug values by the 20 Aug session were measured and superseded (see the attribution caveat below the Golden Values table)
# Previously: 20 August 2026 -- fixed the replacement-cost capacity and efficiency inputs to read the OLD heating/cooling system's own size and efficiency instead of the new heat pump's; re-run `2026-08-19_20-56` (National, MP3 + MP4, fixed_base) confirmed against `2026-08-19_13-19`; five `_unsub` golden rows superseded, new CONFIRMED rows added (12 Aug anchor-year and exact-zero confirmation from 17 August 2026 below is unaffected)

> This file is read by Claude Code at the start of every session. It is the authoritative
> source of truth for project architecture, naming conventions, and permanent constraints.
> Session-specific prompts take precedence over this file when there is a conflict.

---

## Project at a Glance

**Research question:** Economics of heat-pump electrification across U.S. counties
**Data:** 331,531 baseline representative dwelling units | 3,098 counties (ResStock 2022.1.1 EUSS)
**Heat-pump models:** MP3 (standard ASHP, 15 SEER1, 9 HSPF1) | MP4 (high-efficiency ASHP, 24–29.3 SEER1, 13–14 HSPF1)
**Policy scenario:** Single — `'2025 Reference Case'` (see Canonical Values below)
**Adoption metric:** `NPV >= 0` — economic payback only; no climate/health damages in the adoption decision

---

## Terminology — representative dwelling units vs homes

**A ResStock row is a representative dwelling unit (rdu), NOT a home.** Every
row carries `weight = 242.131013` (uniform across this release), meaning it
stands for that many real U.S. dwellings. Multiply a row count by the weight,
or sum the `weight` column, to get actual homes.

- Say "331,531 representative dwelling units", not "331,531 homes". Those rows
  represent **80,273,937 actual homes**.
- Say "260,211 rdu have `include_heating = True`" — that is **63,005,153 real
  dwellings**.
- **Rule of thumb: any count below about 242 is a count of rdu, never homes.**
  One rdu is the smallest a non-zero count can be. A stated "14 homes" is
  almost certainly 14 rdu = ~3,390 homes.
- Weighted and unweighted **averages and shares are identical**, because the
  weight is the same for every row. Only totals and counts differ.

When reporting any count to the researcher or in documentation, either label it
`rdu` or convert it to actual homes. Give both where both are useful. Getting
this wrong understates real-world impact by a factor of 242.

**Reading the older rows in this file.** The Golden Values table and the
Session Log below were written before this rule and say "homes" where they mean
representative dwelling units -- for example "260,211 homes with
`include_heating = True`" is 260,211 rdu, representing 63,005,153 real
dwellings. Those rows are left as written so no golden value appears to have
been altered. Read every count in them as rdu unless it is explicitly weighted;
the dollar means, rates and percentages in them are unaffected, because the
weight is uniform.

Reference conversions: 331,531 rdu = 80,273,937 homes | 260,211 rdu =
63,005,153 homes | 250,576 rdu = 60,672,221 homes | 1 rdu = 242.131013 homes.

---

## Critical Rules — Read First

These apply to every session, every task, without exception.

### Files that must NEVER be edited

| File | Reason |
|---|---|
| `utils/validation_framework.py` | Core validation logic — never touch |
| Any `.ipynb` file | VSCode in-memory cache causes changes not to persist; backport manually |
| Any `*_EXPORT_*.py` file | Read-only snapshot of a notebook — see the rule below |
| `fetch_aeo_data_and_project_EXPORT_24June2026.py` | EIA API scenario string must match API identifier — do NOT rename |
| TARE/EUSS load cells (preserved region) | Upstream data source — do not modify |
| Demand computation cells (preserved region) | Preserve original computation — do not modify |
| `utils/validation_framework.py` | Repeated for emphasis — never, ever touch |

**Logged exception — 12 Aug 2026, `utils/validation_framework.py`.** The
researcher granted a one-off, explicitly scoped exception to change
`replace_small_values_with_nan` so that an exact `0.0` is kept as `0.0` while a
genuinely small NONZERO value is still filtered to NaN. Rationale: the old
`abs(x) > threshold` test swept exact zeros into NaN, which removed 1,279 real
homes from the MP4 NPV and adoption results — a true zero saving is an answer,
not missing data. Nothing else in the file changed, and the fix lives in the
shared function rather than a local copy so every caller behaves the same way.
Details and before/after numbers: `docs/SESSION_CHANGELOG_2026-08-12.md`.
**The never-edit rule still stands for every future session** — this exception
does not generalize; ask again.

### Never edit an `_EXPORT` file

Any file whose name contains `_EXPORT_` (for example
`tare_model_main_v2_3_EXPORT_1Aug2026.py`,
`calculate_postTARE_ts_aws_peak_demand_EXPORT_23July2026.py`) is a **read-only
snapshot** of a notebook, kept so the notebook's contents are easy to read and
diff in git. It is not the source of anything.

Editing one does nothing useful and is actively harmful: the notebook is the
real file, it does not read from the snapshot, so the edit changes no behaviour
while making the snapshot disagree with the notebook it is supposed to mirror.

**Instead:** make the change in the importable modules under `cmu_tare_model/`,
then hand the researcher a list of notebook cells to backport, with
copy-paste-ready replacement code for each cell. The researcher edits the
notebook and re-exports the snapshot. This is the same rule as the `.ipynb`
one above, from the other direction: do not edit the notebook, and do not edit
its snapshot either.

If an `_EXPORT` file has already been changed, revert those changes before
going further.

### One-edit-per-stop-gate rule

Before applying any edit, show the researcher the exact diff (old -> new) with
3-5 lines of context above and below the change, and wait for explicit approval.
Only call the Edit tool after approval is given. Do not batch edits across files
or functions.

### Audit before every edit

Read the actual current file state before proposing any change. Do not assume what a previous session did. Previous sessions sometimes ended mid-task with unknown final state.

### Committing is the researcher's job

Never run `git commit` (or `git add` staging for a commit, `git commit --amend`,
`git reset`, `git revert`, `git push`, or any history-changing git command). The
researcher makes ALL commits and writes ALL commit messages by hand. Make and save
the file changes only; leave them in the working tree for the researcher to stage
and commit. If a summary or a suggested commit message would help, write it in the
chat -- do not act on it.

---

## Canonical Values (hard-coded knowledge)

```python
SCENARIO_STRING = '2025 Reference Case'   # exact string — must byte-match CSV policy_scenario column
COLUMN_PREFIX   = 'ref2025_mp{mp}_'       # always derived — never hardcoded as 'ref2025_mp3_'
ANCHOR_YEAR     = 2025                    # fuel prices and degree-day factors base year
LIFETIME_YEARS  = 15                      # NPV calculation horizon
```

**Do NOT use these strings in model code** — they are retired:
- `'AEO2026 Counterfactual Baseline'` (was renamed in Session 1 — model code only; fetch script keeps it)
- `'AEO2023 Reference Case'`
- `'No Inflation Reduction Act'`
- `preIRA`, `iraRef` as column prefixes

---

## Data Sources (current state as of Session 1)

| Dataset | File | Notes |
|---|---|---|
| Fuel prices | `eia_fuel_price_data_2025_usd2025.csv` | Already USD2025/kWh — no CPI deflation needed |
| Fuel price factors | `aeo2026_fuel_price_factors_2025_2050.csv` | 40 rows; all 2025 values = 1.0 |
| Degree-day factors | `aeo2026_degree_day_factors_2025_2050.csv` | 20 rows; year columns MUST be cast to int on read |
| ResStock source | ResStock 2022.1.1 EUSS | Do not update to ResStock 2025.1 |
| County + state map geometry | `cb_2021_us_county_500k`, `cb_2021_us_state_500k` (under `data/shapefiles/`) | Census cartographic boundary files, 2021 vintage, 500k scale. Matched to ResStock's pre-2023 geography; Connecticut is the binding constraint (see CT note below). Vintage set once via `COUNTY_GEOMETRY_*` / `STATE_GEOMETRY_*` in `adoption_kpis/data_loading.py` -- never hardcode a shapefile name elsewhere. |
| Area median income (AMI) | `ACSDT5Y2024.B19013-Data.csv` (under `data/ami_calculations_data/`) | U.S. Census Bureau ACS 5-Year table B19013 (median household income), vintage 2024, from data.census.gov. One file holds county (`0500000US`) and state (`0400000US`) rows; inflated USD2024->2025. NOT NHGIS -- the NHGIS PUMA source was retired in Session 1e. |

**Data provisioning (geometry + AMI):** the shapefiles and the ACS B19013 CSV are
downloaded by hand once, committed to the repo, and read from a local path.
Nothing downloads at runtime. Repointing the map vintage means vendoring the new
`cb_*` files under `data/shapefiles/` and changing the `*_GEOMETRY_*` constants;
there is no fetch step.

**Degree-day read pattern (mandatory):**
```python
df = pd.read_csv(PATH)
df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]  # MUST cast to int
```
Skipping the int cast causes year lookups to silently return 1.0 (no projection applied).

**State key format:** Two-letter abbreviation (`'PA'`, `'TX'`), NOT full state name.
A wrong key returns silently as zero — no error, just wrong output.

**Connecticut income fallback (CLOSED limitation, not deferred).** The county AMI
source (ACS B19013, vintage 2024) uses post-2022 county geography, so its
Connecticut rows are the nine planning regions (FIPS 09110-09190). ResStock
2022.1.1 still carries the eight pre-2023 CT counties (09001-09015), so the
county AMI join misses every CT home and they fall back to a state-level
`census_area_medianIncome` in `fill_na_with_hierarchy`
(`private_impact/data_processing/determine_rebate_eligibility_and_amount.py`).
This is the same root cause as the county-map CT bug (vintage mismatch) but a
different consequence: it shifts `percent_AMI` and therefore rebate routing for
CT homes. It is accepted and documented -- **no PUMA tier will be added.** The
map layer was repointed to pre-2023 geometry (2021 cartographic boundary) so CT
renders; the income source is deliberately left on state-level fallback for CT.

---

## File Architecture

### Editable modules (Claude Code may edit these)

| File | Role |
|---|---|
| `cmu_tare_model/utils/degree_day_consumption_utils.py` | HDD + CDD-adjusted consumption; use this, not hdd_consumption_utils |
| `cmu_tare_model/private_impact/data_processing/create_lookup_fuel_prices.py` | Fuel price lookup |
| `cmu_tare_model/private_impact/calculate_lifetime_fuel_costs.py` | Lifetime fuel cost computation |
| `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py` | NPV computation |
| `cmu_tare_model/utils/modeling_params.py` | Scenario parameters |
| `cmu_tare_model/utils/calculation_utils.py` | Shared calculation helpers |
| `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | Data loading |
| `cmu_tare_model/constants.py` | EQUIPMENT_SPECS, VALID_CATEGORIES, REBATE_MAPPING |
| `determine_economic_adoption_potential.py` | Economic adoption framework (active) |
| `determine_adoption_potential_sensitivity.py` | Tiered adoption (DEPRECATED — header only, no logic changes) |
| `visualize_geospatial_data.py` | Choropleth / map rendering |
| `visuals_adoption_dotplot.py` | Economic dot plot |
| `calculate_postTARE_am_kpis_*_EXPORTED_*.py` | Main analysis notebook exports |

### Deprecated (do not import from; add header comment only)

| File | Status |
|---|---|
| `hdd_consumption_utils.py` | Superseded by `degree_day_consumption_utils.py` — does not handle cooling |
| `determine_adoption_potential_sensitivity.py` | Superseded by `determine_economic_adoption_potential.py` |

---

## Column Naming Conventions

**Always derive via helpers — never hardcode:**
```python
col_base = define_scenario_params(mp, policy)[0]   # → 'ref2025_mp3_'
mp_str   = f'mp{mp}'                               # '3' or '4' — never 'mp3' literal
```

**NPV cases (nine per MP: three scopes x three rebate policy scenarios, as of the 11 July 2026 session):**
```
ref2025_mp{mp}_heatingSavings_coolingLCC_sub_private_npv{method_suffix}
ref2025_mp{mp}_heatingSavings_coolingLCC_unsub_private_npv{method_suffix}
ref2025_mp{mp}_heatingSavings_coolingLCC_sub_june2026_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingSavings_sub_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingSavings_unsub_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingSavings_sub_june2026_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingLCC_sub_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_private_npv{method_suffix}
ref2025_mp{mp}_heatingLCC_coolingLCC_sub_june2026_private_npv{method_suffix}
```
- Build with `create_npv_case_col(scenario_prefix, npv_case, method_suffix)`;
  `npv_case` must be one of `NPV_CASE_CATEGORIES` (column_names.py). Note there is
  no cost-scenario token and no WTP token in these names.
- `LCC` = that end-use's avoided-replacement capital is credited in the NPV
- `Savings` = only operating savings credited for that end-use
- `_unsub` = unsubsidized; `_sub` = subsidized under 2024 guidance (current HEEHR);
  `_sub_june2026` = subsidized under June 2026 DOE guidance (HEEHR + HOMES + fuel gate).
  These three are the rebate-policy-scenario sensitivity axis (see Rebate Policy Scenarios below).
- All nine cases include BOTH heating and cooling operating savings
- `{method_suffix}` already carries its own leading underscore (e.g. `_fixed_base`)

**Economic adopter columns (nine per MP, as of the 11 July 2026 session):**
```
ref2025_mp{mp}_{npv_case}_econ_adopter{method_suffix}
```
one per `npv_case` in `NPV_CASE_CATEGORIES` (the same nine tokens listed above:
`_unsub`, `_sub`, `_sub_june2026` for each scope).

**Canonical variable suffixes:** `fixed_base` | `central`
Never use: `v3`, `v4MID`, `moreWTP`, `lessWTP`, `iraRef_mp{mp}_`, `preIRA_mp{mp}_`, `aeo2026_mp{mp}_`

> **Note:** The `CODEBASE_MASTER_REFERENCE.md` documents older column naming with `preIRA`/`iraRef`
> prefixes and four columns per MP. That predates the Session 1 scenario consolidation.
> The naming above is current. If you see old-style column names in existing code, flag them —
> they are the old architecture to be replaced.

---

## Sensitivity Dimensions

| Dimension | Active values |
|---|---|
| RCM models (health damage) | `ap2`, `easiur`, `inmap` |
| Private discount rates | `fixed_low` (3%) \| `fixed_base` (7%) \| `fixed_high` (10%) \| `variable` (Ramsey) |
| Policy scenario | Single: `'2025 Reference Case'` — no IRA/pre-IRA split |
| NPV scope | `heatingSavings_coolingLCC` \| `heatingLCC_coolingSavings` \| `heatingLCC_coolingLCC`, each x the three rebate policy scenarios (nine cases; see `NPV_CASE_CATEGORIES`) |
| Rebate policy scenario | `_unsub` (no rebate) \| `_sub` (2024 guidance: HEEHR + fuel-neutral HOMES) \| `_sub_june2026` (June 2026 guidance: HEEHR with fossil fuel gate + HOMES). Column axis under one scenario, like the discount-rate axis; see `REBATE_POLICY_SCENARIOS` |
---

## Rebate Policy Scenarios (2024 vs June 2026 DOE guidance)

Modeled as a sensitivity axis in one dataframe (no second `policy_scenario`), like
the discount-rate axis. Three rebate policy scenarios per scope: `_unsub`, `_sub`
(2024 guidance), `_sub_june2026` (June 2026 guidance). BOTH subsidized vintages
model HEEHR + HOMES -- "2024 = HEEHR only" is RETIRED (2024 HOMES was added
14 Jul 2026).

**One central rebate function (14 Jul 2026 consolidation).** All rebate math lives
in `calculate_rebate_program(df, category, menu_mp, cost_scenario, guidance,
verbose)` in `determine_rebate_eligibility_and_amount.py`, dispatched by a
per-vintage rule config `REBATE_RULE_CONFIG` in `constants.py`. The old names
`calculate_rebateIRA` (2024) and `calculate_rebate_june2026` are DEPRECATED thin
wrappers over it. 2024 writes the guidance-less amount column
`mp{mp}_heating_rebate_amount_{cost_scenario}` + `mp{mp}_rebate_eligibility_ira2024`;
June 2026 writes `mp{mp}_heating_rebate_amount_june2026_{cost_scenario}` +
`mp{mp}_rebate_eligibility_june2026`. Both labels are `'HEEHR'`/`'HOMES'`/`'None'`.

**State-participation gate (ALL rebate policy scenarios, incl. 2024 `_sub`):** homes in a state
that never participated in the federal rebate programs get 0 under every rebate policy scenario.
`NON_PARTICIPATING_REBATE_STATES = {'SD'}` (South Dakota). Applied for both vintages.

**Fuel gate -- HOMES is FUEL-NEUTRAL; the fuel gate is HEEHR-only (14 Jul 2026,
SUPERSEDES the 13 Jul "adjudicated KEEP" wording).** The June 2026 DOE guidance
forbids using a rebate to fund removing a fossil heating system, but that
restriction applies to HEEHR only -- HOMES (performance-based) may fund replacing
a fossil system. So:
- **HEEHR fuel gate:** June 2026 restricts HEEHR to existing electric-resistance
  heating (`base_heating_fuel == 'Electricity'`); any fossil baseline -> HEEHR
  $0, `'None'`. 2024 HEEHR has NO fuel gate (fuel switching allowed), so 2024
  HEEHR funds fossil baselines by design.
- **HOMES fuel-neutral:** 2024 HOMES credits homes above 150% AMI regardless of
  baseline fuel (implemented 14 Jul 2026 -- the one value move). The 2026 HOMES
  pathway is STILL electric-gated in the code THIS session
  (`homes_fuel_gate=True` for June 2026 in `REBATE_RULE_CONFIG`) purely to keep
  the `_sub_june2026` output byte-identical; making 2026 HOMES fuel-neutral is a
  DEFERRED value move (it moves the `_sub_june2026` golden -- see deferred list).
  Do NOT re-add an electric-only gate to 2024 HOMES, and do NOT assume 2026 HOMES
  is already fuel-neutral.

**Program rules (both vintages; both programs gated by `REBATE_ELIGIBLE_HEATING_MPS`,
which is MP3 + MP4 since the 12 Jul 2026 ENERGY STAR override):**
- **HEEHR (`percent_AMI <= 150%`):** $8,000 heat-pump cap; income sets the cost
  share (100% at <=80% AMI, 50% at 80-150%). Same cap both vintages; the only
  vintage difference is the June 2026 HEEHR fuel gate above.
- **HOMES (`percent_AMI > 150%`):** savings-based on whole-home modeled savings
  (`mp{mp}_modeled_savings_frac`): >=20% -> $2,000 cap, >=35% -> $4,000 cap; 50%
  of the full electrification project cost. Non-LMI amounts only.

Rounding note: the merged HEEHR path keeps a per-vintage rounding flag
(`heehr_python_round`) because the original 2024 path used Python `round()` and
the June 2026 path used numpy `.round()`; the two differ by one cent on exact
half-cent products, so preserving both avoids a sub-penny `_sub` move.

**Whole-home savings fraction:** numerator is TARE's degree-day-adjusted heating +
cooling energy delta; denominator is ResStock `baseline_total_site_consumption`
(propagated from `out.site_energy.total.energy_consumption.kwh`). The
adjusted-vs-raw mix is an accepted approximation. Now that 2024 HOMES is
fuel-neutral, this fraction is consumed for FOSSIL HOMES homes too (not only
electric-resistance homes) -- the approximation applies to those as well.

**Reporting / verification helpers** (in `determine_rebate_eligibility_and_amount.py`):
`summarize_june2026_rebate_totals` (weighted HEEHR/HOMES dollars, national + per
state) and `summarize_rebate_funding` (weighted funding by program and by baseline
fuel; `total_eligible` vs `adopters_only`). Both read the explicit
`mp{mp}_rebate_eligibility_*` label (14 Jul 2026: `summarize_rebate_funding` no
longer infers HEEHR from a positive amount, since 2024 now has HOMES too). The
enduring fuel-gate correctness check is on HEEHR, not "all non-electric fuels":
under June 2026, HEEHR must be $0 for every fossil baseline; HOMES MAY fund fossil
baselines (fuel-neutral) -- see `scripts/verify_june2026_rebate_fossil_gate.py`,
which pivots program x fuel for both vintages. `total_eligible` is uncapped
potential (no funding cap modeled), NOT a disbursement.

**Documented limitations (carry into the manuscript):**
1. Weatherization prerequisite not enforced (state criteria not finalized).
2. Dual-fuel systems not modeled. Under June 2026 HEEHR this means every
   fossil-baseline home loses HEEHR (full electrification removes the fossil
   system). HOMES is fuel-neutral, so fossil homes can still earn HOMES above
   150% AMI (2024 today; 2026 once its deferred fuel-neutral fix lands).
3. One program per home (HEEHR or HOMES, never both).
4. State-level funding caps not applied (allocations not finalized; Atlas
   Buildings Hub tracker).
5. HEEHR ENERGY STAR is statutory (both vintages); HOMES ENERGY STAR is optional
   under June 2026 (state discretion). The distinction is moot while only MP3/MP4
   exist -- both meet the ENERGY STAR spec since the 12 Jul 2026 override, so both
   qualify for both programs. Revisit if MP8-10 are activated.
6. HOMES LMI tier (doubled caps + 80% coverage) unreachable by construction
   (HOMES only consulted above 150% AMI). Consequence: an LMI fossil home at or
   below 150% AMI gets HEEHR $0 under June 2026 (fossil gate) and cannot reach
   fuel-neutral HOMES (routing sends <=150% AMI to HEEHR), so it gets $0 -- the
   HOMES LMI tier + a routing fallback are deferred (see deferred list).
---

## Masking and Validation Rules

**Heating:** `include_heating = valid_fuel_heating AND valid_tech_heating`
- `valid_fuel_heating`: fuel ∈ {Electricity, Natural Gas, Propane, Fuel Oil}
- `valid_tech_heating`: technology ∈ `ALLOWED_TECHNOLOGIES['heating']`
- Homes with `in.heating_fuel = 'None'` are automatically excluded

**Cooling:** `include_cooling = valid_fuel_cooling AND valid_tech_cooling`
- `valid_fuel_cooling`: hardcoded True (cooling is always electric) — this flag is a no-op
- `valid_tech_cooling`: technology ∈ {Central AC, Room AC} — this is the ONLY cooling filter
- Homes with no AC (`'None'`) or evaporative coolers are excluded here

**Cooling in NPV:**
For homes where `include_cooling = False`: cooling savings = 0, cooling capital = 0.
For these homes: `heatingLCC_coolingLCC` == `heatingLCC_coolingSavings` (cooling LCC credit = 0),
and both exceed `heatingSavings_coolingLCC` (heating LCC credit is the only differentiation).

**Negative cooling savings -- ACCEPTED as real (12 Jul 2026 session; national
share corrected 19 Aug 2026).**
For some homes the heat pump's cooling energy exceeds the baseline air conditioner's, so
`ref2025_mp{mp}_cooling_lifetime_savings_fuel_cost` is negative. This is a genuine ResStock
base-year result, not a projection artifact: baseline and retrofit cooling are scaled by the
SAME CDD factor and electricity price each year, so the sign is fixed by the raw base-year
kWh delta. It is overwhelmingly a service-level change, because the baseline room AC cools
one room while the whole-home heat pump cools the entire house. The share is measure-package
specific: MP3 90.68% of Room AC baselines go negative vs 10.84% of Central AC; MP4 61.97% vs
3.46%. Recomputed 19 Aug 2026 from the `2026-08-19_13-19` run
(`docs/SESSION_CHANGELOG_2026-08-19.md`), correcting the 12 Jul 2026 session's single,
MP-unsplit "about 54% / 2.5%" figure, which did not reproduce for either MP on this run.
Decision: keep the negative savings in the NPV
as a real operating cost (consistent with the dollars-only, no-WTP adoption threshold); do
NOT floor or exclude. A non-NPV boolean flag
`ref2025_mp{mp}_cooling_lifetime_savings_negative` marks the affected homes for reporting.
No golden value moves. Manuscript limitation: MP4 delivers whole-home cooling the room AC
never did, and the dollars-only NPV counts the added cooling cost with no offsetting
comfort credit.

**Existing-ASHP homes — RESOLVED: exclude.**
`'Electricity ASHP'` (and any variant) must NOT appear in `EQUIPMENT_SPECS` or
`ALLOWED_TECHNOLOGIES['heating']`. The study models fossil-fuel-to-ASHP transitions;
a home that already has an ASHP has no counterfactual fossil fuel system to replace.
If an existing-ASHP entry is found in `constants.py`, flag it and remove it.

---

## NPV Ordering Checks (enforce in verification)

Per home (for homes with AC, `include_cooling = True`):
- `heatingLCC_coolingLCC >= heatingLCC_coolingSavings` (adds avoided cooling replacement >= 0)
- `heatingLCC_coolingLCC >= heatingSavings_coolingLCC` (adds avoided heating replacement >= 0)
- No general ordering between `heatingLCC_coolingSavings` and `heatingSavings_coolingLCC`
  (depends on relative magnitude of heating vs cooling replacement costs)

Per home (no AC, `include_cooling = False`):
- `heatingLCC_coolingLCC` == `heatingLCC_coolingSavings` (cooling LCC credit = 0)
- Both exceed `heatingSavings_coolingLCC` (heating LCC credit is non-zero)

Per county (means):
- Adoption rate `heatingLCC_coolingLCC` >= `heatingLCC_coolingSavings`
- Adoption rate `heatingLCC_coolingLCC` >= `heatingSavings_coolingLCC`

---

## Golden Values

These were established under the pre-Session-1 data (old fuel prices, old degree-day factors).
They will change when Session 2 NPV runs with new data. Never silently overwrite —
add a new row marked "supersedes" and keep the old row.

| Quantity | MP3 | MP4 | Data vintage | Session |
|---|---|---|---|---|
| Operating-cost % change, county median | −38.5% | −60.6% | Pre-AEO2026 | Round 3 |
| Total electricity demand change (GWh) | +427,043.7 | +30,618.4 | Pre-AEO2026 | Round 3 |
| Median demand % change | +22.5% | −8.1% | Pre-AEO2026 | Round 3 |
| Mean economic adoption rate (heating only) | 20.8% | 20.5% | Pre-AEO2026 | Round 3 -- superseded by Session A (case retired) |
| Operating-cost % symmetric norm | ±81.4% | (shared) | Pre-AEO2026 | Round 3 |
| Demand GWh symmetric norm | ±1038.3 GWh | (shared) | Pre-AEO2026 | Round 3 |
| LMI eligibility share, single-family (NHGIS-2022 PUMA AMI; bins USD2022->23) | 71.6% | (shared) | Pre-USD2025 | superseded by Session 1e |
| LMI eligibility share, single-family (ACS-2024 county AMI; bins USD2018->25) | 62.4% | (shared) | USD2025 | Session 1e (28 Jun 2026) |
| Mean economic adoption rate (nine NPV cases, `_unsub`/`_sub`/`_sub_june2026`) | PENDING | PENDING | AEO2026/Cambium2024 | To be re-derived; no golden value exists yet. Do not backfill without a full model run. NOTE (12 Jul 2026): MP3 is now ENERGY STAR-respecified and rebate-eligible, so MP3 '_sub' adoption is no longer equal to '_unsub'. NOTE (14 Jul 2026): 2024 HOMES was ADDED, so every `_sub` row RISES (homes >150% AMI now earn fuel-neutral HOMES). Re-derive `_sub` on a full run. DERIVED 17 Aug 2026 from the full run -- kept as history; the nine CONFIRMED case rows are below. |
| Mean economic adoption rate, `_sub_june2026` cases | PENDING | PENDING | AEO2026/Cambium2024 | 11 Jul 2026 session; new rebate regime. Requires a full model run to derive. NOTE (12 Jul 2026): MP3 now passes the rebate MP gate, so MP3 june2026 adoption moves. NOTE (14 Jul 2026): `_sub_june2026` did NOT move this session (2026 HOMES stayed electric-gated for byte-identity); it will move when the deferred 2026-HOMES fuel-neutral fix lands. DERIVED 17 Aug 2026 from the full run -- kept as history; the three `_sub_june2026` CONFIRMED rows are below. |
| June 2026 rebate movement vs `_sub`: fossil MP4 lose HEEHR; electric MP4 >150% AMI gain HOMES | PENDING | PENDING | AEO2026/Cambium2024 | Directions to confirm in the movement cross-tab. NOTE (14 Jul 2026): the `_sub` baseline for this cross-tab shifted upward (2024 HOMES added), so recompute the movement against the NEW `_sub`. **HALF WRONG -- see the CORRECTED movement row below (17 Aug 2026).** The "fossil lose HEEHR" half holds. The "electric >150% AMI gain HOMES" half was only ever true against the OLD pre-14-Jul `_sub`, where 2024 had no HOMES at all. Against the CURRENT `_sub` those homes hold HOMES under both vintages, so they gain nothing, and the measured movement is losses only. Kept as history. |
| 2024 HOMES value move (14 Jul 2026): fossil + electric homes >150% AMI gain fuel-neutral HOMES under `_sub` | PENDING | PENDING | AEO2026/Cambium2024 | The one intended value move this session. On the full run, confirm `_sub` adoption rises vs the pre-14-Jul `_sub`, driven by >150% AMI homes (all fuels). No concrete golden yet -- requires a full model run. CONFIRMED 17 Aug 2026, with the pre-14-Jul side RECONSTRUCTED rather than measured -- kept as history; see the CONFIRMED row below. |
| MP3 ENERGY STAR override: heating-upgrade capital-cost increase (SEER 15->16, weighted) | +$796.83/home | n/a (MP4 unchanged) | AEO2026/Cambium2024 | 12 Jul 2026 session. Ducted +$942.59 (n=432,164); non-ducted +$254.24 (n=116,096). Additive: pm2_coef x 1 x mult(1.5) x cpi(1.0566). MP3 is now rebate-eligible, so the `_sub`/`_sub_june2026` adoption rows above also move for MP3 -- re-derive on a full run. |
| Mean lifetime heating fuel cost, baseline (National) | (shared) | (shared) | Years 2024-2038 | SUPERSEDED by 12 Aug 2026 -- old anchor year |
| PROVISIONAL -- Mean lifetime heating fuel cost, baseline (National) -- $20,362.56 | (shared) | (shared) | Years 2025-2039 | 12 Aug 2026 anchor-year fix. Supersedes $20,402.20 (-$39.64, -0.19%). CONFIRMED by the 17 Aug 2026 full run -- kept as history; cite the CONFIRMED row below. |
| Mean lifetime cooling fuel cost, baseline (National) | (shared) | (shared) | Years 2024-2038 | SUPERSEDED by 12 Aug 2026 -- old anchor year |
| PROVISIONAL -- Mean lifetime cooling fuel cost, baseline (National) -- $10,097.37 | (shared) | (shared) | Years 2025-2039 | 12 Aug 2026 anchor-year fix. Supersedes $9,988.10 (+$109.27, +1.09%). Cooling rises because the cooling degree-day factors climb over time, so moving the window one year later prices more cooling. CONFIRMED by the 17 Aug 2026 full run -- kept as history; cite the CONFIRMED row below. |
| Mean `heatingLCC_coolingLCC_unsub` NPV, `fixed_base` (National) | n/a (not re-run) | -$5,813.68 | Years 2024-2038 | SUPERSEDED by 12 Aug 2026 -- old anchor year |
| PROVISIONAL -- Mean `heatingLCC_coolingLCC_unsub` NPV, `fixed_base` (National) | PENDING (MP3 not re-run) | -$5,816.35 | Years 2025-2039 | 12 Aug 2026 anchor-year fix. Supersedes -$5,813.68 (-$2.67, -0.05%). 258,932 homes with a usable NPV, unchanged. NOT CONFIRMABLE BY A FULL RUN, by construction: the current code carries the exact-zero fix, so this intermediate step (anchor year moved, exact zeros still dropped) can no longer be produced. Kept as history; the confirmed value is the post-exact-zero row below. |
| Mean economic adoption rate, `heatingLCC_coolingLCC_unsub`, `fixed_base` (National) | n/a (not re-run) | 18.4493% | Years 2024-2038 | SUPERSEDED by 12 Aug 2026 -- old anchor year |
| PROVISIONAL -- Mean economic adoption rate, `heatingLCC_coolingLCC_unsub`, `fixed_base` (National) | PENDING (MP3 not re-run) | 18.4354% | Years 2025-2039 | 12 Aug 2026 anchor-year fix. Supersedes 18.4493% (-0.0138 pp). Denominator 260,211 homes (the non-null adopter count, equal to `include_heating = True`), unchanged. 237 homes crossed up, 273 crossed down, net -36 adopters (48,007 -> 47,971). Weighted and unweighted rates are identical because every home carries the same weight. SUPERSEDED the same day by the exact-zero fix below, and NOT CONFIRMABLE BY A FULL RUN for the same reason as the NPV row above. |
| PROVISIONAL -- Mean economic adoption rate, `heatingLCC_coolingLCC_unsub`, `fixed_base` (National), AFTER the exact-zero savings fix | PENDING (MP3 not re-run) | 18.4416% | Years 2025-2039 | 12 Aug 2026 exact-zero fix (Option B), applied on top of the anchor-year fix. Supersedes 18.4354% (+0.0061 pp). `replace_small_values_with_nan` no longer turns an exact `0.0` saving into NaN, so the 1,279 homes that previously had no NPV now get a real one: adopters 47,971 -> 47,987 (+16; 13 of the 1,265 heating-zero homes, 3 of the 14 cooling-zero homes). Denominator 260,211, unchanged. Homes with a usable NPV: 258,932 -> 260,211. CONFIRMED by the 17 Aug 2026 full run -- kept as history; cite the CONFIRMED row below. |
| PROVISIONAL -- Mean `heatingLCC_coolingLCC_unsub` NPV, `fixed_base` (National), AFTER the exact-zero savings fix | PENDING (MP3 not re-run) | -$5,838.23 | Years 2025-2039 | 12 Aug 2026 exact-zero fix. Supersedes -$5,816.35. The mean moves because the 1,279 newly-valued homes (mostly large negative NPVs) now enter the average, NOT because any already-valued home changed -- the denominator goes 258,932 -> 260,211. CONFIRMED by the 17 Aug 2026 full run -- kept as history; cite the CONFIRMED row below. |
| CONFIRMED -- Mean lifetime heating fuel cost, baseline (National) -- $20,362.56 | (shared) | (shared) | Years 2025-2039 | 17 Aug 2026 full run. Exactly $20,362.5614700378 over the 260,211 homes with `include_heating = True`. Matches the 12 Aug provisional value to the cent. |
| CONFIRMED -- Mean lifetime cooling fuel cost, baseline (National) -- $10,097.37 | (shared) | (shared) | Years 2025-2039 | 17 Aug 2026 full run. Exactly $10,097.3677096370 over the 250,576 homes with `include_cooling = True`. Matches the 12 Aug provisional value to the cent. |
| CONFIRMED -- Mean `heatingLCC_coolingLCC_unsub` NPV, `fixed_base` (National) | -$4,852.41 | -$5,838.23 | Years 2025-2039 | 17 Aug 2026 full run. MP4 is exactly -$5,838.2316748715, matching the 12 Aug provisional value to the cent. MP3 measured for the first time. Both MPs: 260,211 homes with a usable NPV. Independently reproduced to the same digits on the 19 Aug 2026 run (`2026-08-19_13-19`), which carries the Tepper-export changes from that session -- confirms those changes did not move this value. **SUPERSEDED by 20 Aug 2026** (old-system-size fix in the replacement-cost inputs) -- see the new CONFIRMED row below. |
| CONFIRMED -- Mean economic adoption rate, `heatingLCC_coolingLCC_unsub`, `fixed_base` (National) | 27.7140% | 18.4416% | Years 2025-2039 | 17 Aug 2026 full run. MP4 is 18.441572%, matching the 12 Aug provisional value; 47,987 adopters out of a 260,211 denominator. MP3 measured for the first time: 72,115 adopters out of the same 260,211. Weighted and unweighted rates are identical -- every home carries weight 242.13. Independently reproduced to the same digits on the 19 Aug 2026 run (`2026-08-19_13-19`), which carries the Tepper-export changes from that session -- confirms those changes did not move this value. **SUPERSEDED by 20 Aug 2026** (old-system-size fix in the replacement-cost inputs) -- see the new CONFIRMED row below. |
| CONFIRMED -- `baseline_heating_lifetime_mt_co2e_lrmer` (National) | (shared) | (shared) | Years 2025-2039 | 69.4238 tonnes over 260,204 homes. 17 Aug 2026 full run; matches the 12 Aug climate table. |
| CONFIRMED -- `baseline_heating_lifetime_damages_climate_lrmer_central` (National) | (shared) | (shared) | Years 2025-2039 | $18,377.40 over 260,204 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `baseline_heating_lifetime_mt_co2e_srmer` (National) | (shared) | (shared) | Years 2025-2039 | 80.6867 tonnes over 260,204 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `baseline_heating_lifetime_damages_climate_srmer_central` (National) | (shared) | (shared) | Years 2025-2039 | $21,350.01 over 260,204 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `baseline_cooling_lifetime_mt_co2e_lrmer` (National) | (shared) | (shared) | Years 2025-2039 | 16.0524 tonnes over 250,570 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `baseline_cooling_lifetime_damages_climate_lrmer_central` (National) | (shared) | (shared) | Years 2025-2039 | $4,174.75 over 250,570 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `baseline_cooling_lifetime_mt_co2e_srmer` (National) | (shared) | (shared) | Years 2025-2039 | 32.0207 tonnes over 250,570 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `baseline_cooling_lifetime_damages_climate_srmer_central` (National) | (shared) | (shared) | Years 2025-2039 | $8,415.79 over 250,570 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `ref2025_mp{mp}_heating_avoided_mt_co2e_lrmer` (National) | 51.1613 | 56.9415 | Years 2025-2039 | Tonnes, over 260,204 homes. 17 Aug 2026 full run; MP4 matches the 12 Aug climate table, MP3 measured for the first time. |
| CONFIRMED -- `ref2025_mp{mp}_heating_avoided_damages_climate_lrmer_central` (National) | $13,629.25 | $15,131.46 | Years 2025-2039 | 260,204 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `ref2025_mp{mp}_heating_avoided_mt_co2e_srmer` (National) | 41.6161 | 53.7673 | Years 2025-2039 | Tonnes, over 260,204 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `ref2025_mp{mp}_heating_avoided_damages_climate_srmer_central` (National) | $11,114.97 | $14,298.19 | Years 2025-2039 | 260,204 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `ref2025_mp{mp}_cooling_avoided_mt_co2e_lrmer` (National) | 1.1246 | 6.2063 | Years 2025-2039 | Tonnes, over 250,570 homes. 17 Aug 2026 full run. MP3 is far below MP4 because MP3 barely changes cooling energy. |
| CONFIRMED -- `ref2025_mp{mp}_cooling_avoided_damages_climate_lrmer_central` (National) | $291.22 | $1,613.45 | Years 2025-2039 | 250,570 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `ref2025_mp{mp}_cooling_avoided_mt_co2e_srmer` (National) | 2.0064 | 12.0076 | Years 2025-2039 | Tonnes, over 250,570 homes. 17 Aug 2026 full run. |
| CONFIRMED -- `ref2025_mp{mp}_cooling_avoided_damages_climate_srmer_central` (National) | $526.83 | $3,154.99 | Years 2025-2039 | 250,570 homes. 17 Aug 2026 full run. |
| CONFIRMED -- Adoption rate, `heatingSavings_coolingLCC_unsub`, `fixed_base` (National) | 15.9924% | 11.5314% | Years 2025-2039 | 17 Aug 2026 full run. 41,614 / 30,006 adopters out of 260,211. Mean NPV -$8,781.65 / -$9,555.69. **SUPERSEDED by 20 Aug 2026** (old-system-size fix) -- see the new CONFIRMED row below. |
| CONFIRMED -- Adoption rate, `heatingSavings_coolingLCC_sub`, `fixed_base` (National) | 44.8690% | 27.0062% | Years 2025-2039 | 17 Aug 2026 full run. 116,754 / 70,273 adopters. Mean NPV -$3,057.30 / -$3,438.31. Assumes uncapped rebates (see the note below the table). **SUPERSEDED by 3 Sep 2026** -- see the new CONFIRMED row below. |
| CONFIRMED -- Adoption rate, `heatingSavings_coolingLCC_sub_june2026`, `fixed_base` (National) | 24.1719% | 18.1849% | Years 2025-2039 | 17 Aug 2026 full run. 62,898 / 47,319 adopters. Mean NPV -$7,212.06 / -$7,856.72. **SUPERSEDED by 3 Sep 2026** -- see the new CONFIRMED row below. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingSavings_unsub`, `fixed_base` (National) | 12.3550% | 9.7421% | Years 2025-2039 | 17 Aug 2026 full run. 32,149 / 25,350 adopters. Mean NPV -$9,808.43 / -$10,709.45. **SUPERSEDED by 20 Aug 2026** (old-system-size fix) -- see the new CONFIRMED row below. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingSavings_sub`, `fixed_base` (National) | 32.0475% | 21.3142% | Years 2025-2039 | 17 Aug 2026 full run. 83,391 / 55,462 adopters. Mean NPV -$4,084.09 / -$4,592.08. **SUPERSEDED by 3 Sep 2026** -- see the new CONFIRMED row below. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingSavings_sub_june2026`, `fixed_base` (National) | 19.4677% | 15.6342% | Years 2025-2039 | 17 Aug 2026 full run. 50,657 / 40,682 adopters. Mean NPV -$8,238.85 / -$9,010.48. **SUPERSEDED by 3 Sep 2026** -- see the new CONFIRMED row below. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingLCC_unsub`, `fixed_base` (National) | 27.7140% | 18.4416% | Years 2025-2039 | 17 Aug 2026 full run. Same row as the confirmed MP4 anchor-year value above, with MP3 added. 72,115 / 47,987 adopters. Mean NPV -$4,852.41 / -$5,838.23. **SUPERSEDED by 20 Aug 2026** (old-system-size fix) -- see the new CONFIRMED row below. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingLCC_sub`, `fixed_base` (National) | 62.2111% | 46.3462% | Years 2025-2039 | 17 Aug 2026 full run. 161,880 / 120,598 adopters. Mean NPV +$871.94 / +$279.14 -- the only two cases with a positive mean NPV, and only because the rebate is uncapped. **SUPERSEDED by 3 Sep 2026** -- see the new CONFIRMED row below. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingLCC_sub_june2026`, `fixed_base` (National) | 34.8671% | 27.2537% | Years 2025-2039 | 17 Aug 2026 full run. 90,728 / 70,917 adopters. Mean NPV -$3,282.82 / -$4,139.26. **SUPERSEDED by 3 Sep 2026** -- see the new CONFIRMED row below. |
| CONFIRMED (CORRECTS the movement row above) -- December 2024 -> June 2026 rebate movement is LOSSES ONLY; zero homes gain a program | -27.3440 pp | -19.0926 pp | Years 2025-2039 | 17 Aug 2026 full run, `heatingLCC_coolingLCC`. Adopter crossings: 0 gained, 71,152 (MP3) / 49,681 (MP4) lost. Two loss channels, both fossil: (a) fossil at or below 150% AMI lose HEEHR to the June 2026 fuel gate -- 113,959 homes, identical for both MPs; (b) fossil above 150% AMI lose HOMES -- 60,747 (MP3) / 64,722 (MP4) -- because 2026 HOMES is STILL electric-gated (the deferred fuel-neutral fix). Electric homes are untouched: HEEHR -> HEEHR at or below 150% AMI, HOMES -> HOMES above it. |
| CONFIRMED -- June 2026 HEEHR fossil gate holds exactly | 0 fossil homes | 0 fossil homes | Years 2025-2039 | 17 Aug 2026 full run. Zero fossil-baseline homes carry the `'HEEHR'` label under June 2026, and the largest June 2026 rebate paid to any fossil home is $0.00 -- 2026 HOMES is still electric-gated, so fossil homes get nothing at all under June 2026. |
| CONFIRMED -- 2024 HOMES is fuel-neutral | 60,747 fossil + 16,249 electric | 64,722 fossil + 21,733 electric | Years 2025-2039 | 17 Aug 2026 full run. Every 2024 HOMES recipient is above 150% AMI (zero at or below), and fossil recipients outnumber electric roughly 3:1 -- the 14 Jul fuel-neutral change is live. Under June 2026 the same HOMES pathway pays only the electric homes. |
| RECONSTRUCTED -- 2024 HOMES value move: rise in `_sub` adoption vs the pre-14-Jul `_sub` | +5.6654 pp | +3.9510 pp | Years 2025-2039 | 17 Aug 2026, `heatingLCC_coolingLCC`. 56.5456% -> 62.2111% (MP3), 42.3952% -> 46.3462% (MP4); 14,742 / 10,281 homes newly adopt, every one above 150% AMI, majority fossil (MP4: 5,309 natural gas, 3,173 electricity, 903 propane, 896 fuel oil). No home loses adoption, as expected for an added rebate. Other scopes: MP3 +3.3730 / +2.3097 pp, MP4 +2.5080 / +1.9177 pp. **The pre-14-Jul side is RECONSTRUCTED, not measured** -- see the note below the table. |
| CONFIRMED -- Weighted rebate potential, uncapped (National) | $360,661.9M (2024) -> $98,891.8M (June 2026) | $385,424.6M (2024) -> $107,043.4M (June 2026) | Years 2025-2039 | 17 Aug 2026 full run, `v4MID`. 2024 splits HEEHR $296,554.3M / HOMES $64,107.6M (MP3) and $310,415.6M / $75,009.0M (MP4); June 2026 splits $87,199.8M / $11,692.0M (MP3) and $89,892.3M / $17,151.0M (MP4). This is `total_eligible` -- uncapped potential across every eligible home, NOT a disbursement. No state funding cap is modeled (limitation 4). |
| CONFIRMED -- South Dakota participation gate holds | $0.00 | $0.00 | Years 2025-2039 | 17 Aug 2026 full run. All 988 SD homes get $0 under both vintages and carry no program label, matching `NON_PARTICIPATING_REBATE_STATES = {'SD'}`. |
| CONFIRMED -- Old-system-size fix: mean replacement-cost change (National) | Heating -$62.79, cooling -$663.19 | Heating +$148.99, cooling -$502.02 | Years 2025-2039 | 20 Aug 2026 full run (`2026-08-19_20-56`), same 331,531 homes / same order as the `2026-08-19_13-19` run it is compared against. The replacement-cost credit is now priced off the OLD system's own size (`base_size_heating/cooling_system_primary_k_btu_h`, added in `df_enduse_refactored`) instead of the retrofit heat pump's size -- see `docs/SESSION_CHANGELOG_2026-08-20.md`. Direction is not uniform: by baseline heating fuel, mean heating replacement cost moved Electricity -$509.31 (MP3) / +$177.65 (MP4), Natural Gas +$109.12 / +$129.81, Propane +$155.87 / +$169.69, Fuel Oil +$120.71 / +$184.70. By baseline cooling type, mean cooling replacement cost moved Central AC -$180.04 (MP3) / -$199.35 (MP4), Room AC -$2,181.90 (MP3) / -$1,453.44 (MP4) -- Room AC drops sharply because a whole-home heat pump is much bigger than the room unit it replaces. The heating upgrade-cost column (the heat pump's own cost) is byte-identical to the pre-fix run, confirming the fix touched only the replacement-cost input. New identity check enabled by the fix: MP3 and MP4 now have byte-identical heating and cooling replacement-cost columns for every home (0 mismatches), because the "old system being replaced" does not depend on which heat pump replaces it -- this did NOT hold before the fix. |
| CONFIRMED -- Mean `heatingLCC_coolingLCC_unsub` NPV, `fixed_base` (National), AFTER the old-system-size fix | -$5,359.94 | -$6,049.17 | Years 2025-2039 | 20 Aug 2026 full run (`2026-08-19_20-56`). Supersedes -$4,852.41 (MP3, -$507.53) / -$5,838.23 (MP4, -$210.94). Same 260,211-home denominator as the 17 Aug run. NPV identity (savings - net_capital_cost = NPV) holds to the half-cent for all 260,211 homes on all nine NPV cases; CLAUDE.md's NPV ordering checks hold with 0 violations. |
| CONFIRMED -- Mean economic adoption rate, `heatingLCC_coolingLCC_unsub`, `fixed_base` (National), AFTER the old-system-size fix | 27.1760% | 18.0984% | Years 2025-2039 | 20 Aug 2026 full run. Supersedes 27.7140% (MP3, -0.5380 pp; 72,115 -> 70,715 adopters) / 18.4416% (MP4, -0.3432 pp; 47,987 -> 47,094 adopters). Denominator 260,211, unchanged. Adopter crossings by baseline heating fuel (MP3): Electricity 902 gained / 2,431 lost, Natural Gas 764 gained / 518 lost, Propane 18 gained / 114 lost, Fuel Oil 53 gained / 74 lost -- net loss driven by electric-baseboard homes losing replacement credit and by Room AC homes losing cooling replacement credit (see the crossings-by-cooling-type breakdown in the changelog). MP4 crossings: Electricity 891 gained / 1,474 lost, Natural Gas 168 gained / 334 lost, Propane 31 gained / 123 lost, Fuel Oil 48 gained / 100 lost. |
| CONFIRMED -- Adoption rate, `heatingSavings_coolingLCC_unsub`, `fixed_base` (National), AFTER the old-system-size fix | 15.5862% | 11.1317% | Years 2025-2039 | 20 Aug 2026 full run. Supersedes 15.9924% (MP3, -0.4062 pp; 41,614 -> 40,557 adopters) / 11.5314% (MP4, -0.3997 pp; 30,006 -> 28,966 adopters). Mean NPV -$9,226.39 / -$9,915.62 (was -$8,781.65 / -$9,555.69). This scope credits cooling replacement only, so it moved down the most -- cooling's replacement credit fell sharply (Room AC homes especially). |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingSavings_unsub`, `fixed_base` (National), AFTER the old-system-size fix | 12.9337% | 10.1552% | Years 2025-2039 | 20 Aug 2026 full run. Supersedes 12.3550% (MP3, +0.5788 pp; 32,149 -> 33,655 adopters) / 9.7421% (MP4, +0.4131 pp; 25,350 -> 26,425 adopters). Mean NPV -$9,871.23 / -$10,560.46 (was -$9,808.43 / -$10,709.45). This scope credits heating replacement only, so it moved up slightly on net -- most baseline fossil-fuel homes gained heating replacement credit. |
| CONFIRMED -- Adoption rate, `heatingSavings_coolingLCC_sub`, `fixed_base` (National), AFTER the old-system-size fix | 43.7649% | 26.0827% | Years 2025-2039 | 3 Sep 2026, measured from run `2026-09-02_19-04` (byte-identical to `2026-08-19_20-56`). Supersedes 44.8690% (MP3, -1.1041 pp; 116,754 -> 113,881 adopters) / 27.0062% (MP4, -0.9235 pp; 70,273 -> 67,870 adopters). Denominator 260,211. Mean NPV -$3,502.04 / -$3,798.25 (was -$3,057.30 / -$3,438.31). Assumes uncapped rebates. See the attribution caveat below the table. |
| CONFIRMED -- Adoption rate, `heatingSavings_coolingLCC_sub_june2026`, `fixed_base` (National), AFTER the old-system-size fix | 23.6969% | 17.6753% | Years 2025-2039 | 3 Sep 2026, run `2026-09-02_19-04`. Supersedes 24.1719% (MP3, -0.4750 pp; 62,898 -> 61,662 adopters) / 18.1849% (MP4, -0.5096 pp; 47,319 -> 45,993 adopters). Mean NPV -$7,656.80 / -$8,216.65 (was -$7,212.06 / -$7,856.72). |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingSavings_sub`, `fixed_base` (National), AFTER the old-system-size fix | 32.2611% | 21.6590% | Years 2025-2039 | 3 Sep 2026, run `2026-09-02_19-04`. Supersedes 32.0475% (MP3, +0.2136 pp; 83,391 -> 83,947 adopters) / 21.3142% (MP4, +0.3448 pp; 55,462 -> 56,359 adopters). Mean NPV -$4,146.88 / -$4,443.08 (was -$4,084.09 / -$4,592.08). This scope credits heating replacement only, so it rose, matching the `_unsub` direction. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingSavings_sub_june2026`, `fixed_base` (National), AFTER the old-system-size fix | 19.2152% | 15.8345% | Years 2025-2039 | 3 Sep 2026, run `2026-09-02_19-04`. Supersedes 19.4677% (MP3, -0.2525 pp; 50,657 -> 50,000 adopters) / 15.6342% (MP4, +0.2003 pp; 40,682 -> 41,203 adopters). Mean NPV -$8,301.64 / -$8,861.49 (was -$8,238.85 / -$9,010.48). The only case where the two MPs moved in opposite directions. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingLCC_sub`, `fixed_base` (National), AFTER the old-system-size fix | 61.2169% | 45.1168% | Years 2025-2039 | 3 Sep 2026, run `2026-09-02_19-04`. Supersedes 62.2111% (MP3, -0.9942 pp; 161,880 -> 159,293 adopters) / 46.3462% (MP4, -1.2294 pp; 120,598 -> 117,399 adopters). Mean NPV +$364.41 / +$68.20 (was +$871.94 / +$279.14) -- still the only two cases with a positive mean NPV, and only because the rebate is uncapped, but MP4 is now only $68 above zero. |
| CONFIRMED -- Adoption rate, `heatingLCC_coolingLCC_sub_june2026`, `fixed_base` (National), AFTER the old-system-size fix | 34.0908% | 26.3936% | Years 2025-2039 | 3 Sep 2026, run `2026-09-02_19-04`. Supersedes 34.8671% (MP3, -0.7763 pp; 90,728 -> 88,708 adopters) / 27.2537% (MP4, -0.8601 pp; 70,917 -> 68,679 adopters). Mean NPV -$3,790.35 / -$4,350.20 (was -$3,282.82 / -$4,139.26). |

> **The 12 Aug 2026 provisional rows were confirmed on 17 Aug 2026 by a full
> end-to-end model run. Cite the CONFIRMED rows, not the provisional ones.**
> The provisional rows came from re-running only
> `calculate_lifetime_fuel_costs`, `calculate_private_npv`, and
> `economic_adoption_decision` on the already-exported National result frames
> dated `2026-08-02_20-32`. The confirming run executed the whole pipeline --
> EUSS load, consumption processing, capital costs, rebates, fuel costs, NPV,
> adoption, and climate damages -- for both MPs, with the notebook fix from
> Section 8 of `docs/SESSION_CHANGELOG_2026-08-12.md` applied.
>
> **The run:** timestamp `2026-08-17_19-16`, National, MP3 and MP4, all 331,531
> homes, `fixed_base` discount rate. Outputs under
> `cmu_tare_model/output_results/`:
> `baseline_summary/summary_baseline/baseline_results_National_2026-08-17_19-16.csv`,
> `retrofit_mp{3,4}_results/summary_mp{3,4}_fixed_base/mp{3,4}_results_National_2026-08-17_19-16.csv`,
> plus the matching `supplemental_data_fuelCosts/`, `supplemental_data_damages/`
> and `tepper_export/` files. Every confirmed quantity agrees with its
> provisional value to the last digit the provisional row reported; the largest
> gap is last-place rounding against a two-decimal entry.
>
> **What the run does NOT cover, so no golden value exists for it yet:** the
> other three discount rates (`fixed_low`, `fixed_high`, `variable`) -- the run
> wrote `fixed_base` only. Everything else in the table has been derived: all
> nine NPV cases for both MPs, the rebate movement, and the rebate totals.
>
> **Two caveats on the rebate rows.** First, every `_sub` and `_sub_june2026`
> adoption rate assumes each eligible home receives its full modeled rebate. No
> state funding cap is modeled (limitation 4), so these rates are an upper
> bound, not a forecast -- which is why `heatingLCC_coolingLCC_sub` reaches a
> positive mean NPV. Second, the RECONSTRUCTED 2024 HOMES row is the one number
> in the table not taken from a run. The pre-14-Jul code no longer exists, so
> its `_sub` was rebuilt from this run: under the old rules HOMES did not exist,
> every home above 150% AMI got $0, so its subsidized NPV equalled its
> unsubsidized one, while homes at or below 150% AMI kept an unchanged HEEHR.
> The rebuild takes `_unsub` for today's 2024-HOMES recipients and `_sub` for
> everyone else. It is exact only if adding 2024 HOMES was the sole `_sub` value
> move on 14 Jul, which is what that session recorded. Treat it as a
> well-founded estimate, not a measurement.
>
> **Two intermediate rows can never be confirmed** -- the pre-exact-zero NPV
> (-$5,816.35) and adoption rate (18.4354%). The current code carries the
> exact-zero fix, so that halfway state (new anchor year, exact zeros still
> dropped) is no longer reproducible. Both stay in the table as history.
>
> **Observation, not a defect:** climate columns are non-null for 260,204 homes
> while heating fuel cost is non-null for 260,211 (cooling: 250,570 vs
> 250,576). This seven-home and six-home gap predates the anchor-year work -- it
> is present identically before and after, which is why the climate means
> reproduce exactly. Not diagnosed here.

> **20 Aug 2026 old-system-size fix.** The five rows above marked
> "AFTER the old-system-size fix" replace the `_unsub` rows for the same
> quantities from the 17 Aug 2026 run. **Only the `_unsub` cases were
> compared this session** -- the `_sub` and `_sub_june2026` cases were left
> alone because of a separate, documented issue where the household-income
> random draw shifts if a run covers a different set of homes; comparing
> them safely needs more care than this session's before/after diff gave it.
> The run compared is `2026-08-19_20-56` (National, MP3 and MP4, all 331,531
> homes, `fixed_base`) against `2026-08-19_13-19` (same scope, same homes,
> same order -- confirmed by matching `bldg_id` row-for-row). Nothing outside
> the replacement-cost columns and the values that depend on them (net
> capital cost, NPV, adoption) moved: baseline fuel costs and climate damages
> reproduced to the last cent/tonne. Full numbers, the fuel/cooling-type
> breakdowns, and the fix itself: `docs/SESSION_CHANGELOG_2026-08-20.md`.

> **3 Sep 2026 -- the six `_sub` / `_sub_june2026` rows were finally measured.**
> The 20 Aug session compared `_unsub` only, so those six rows sat at their
> 17 Aug values while the rest of the table moved. They have now been measured
> directly from run `2026-09-02_19-04`, which this session verified is
> BYTE-IDENTICAL to `2026-08-19_20-56` -- all 17 output files match on SHA-256
> (baseline, mp3, mp4 summaries; fuel costs; climate damages; and the eight
> Tepper household/county exports), same 331,531 rows, same column order, same
> `bldg_id` order. So the new rows describe the same run the `_unsub` rows
> already described; the table is now internally consistent for the first time
> since 20 Aug.
>
> **Attribution caveat -- read before citing these six as a size-fix effect.**
> The pp moves are stated against the 17 Aug values, but they are NOT cleanly
> attributable to the old-system-size fix alone. The 20 Aug session deliberately
> skipped `_sub` because the household-income random draw can shift between
> runs, which moves rebate routing independently of any code change. That same
> caveat applies to these deltas. What IS certain: the six new values describe
> the current run, and the six 17 Aug values did not. Directions are at least
> consistent with the `_unsub` pattern -- the two cooling-replacement-crediting
> scopes fell, the heating-replacement-crediting scope rose -- but treat the
> exact pp figures as a run-to-run difference, not a measured fix effect. An
> isolated re-run holding the income draw fixed would be needed to separate them.
>
> **Independently re-verified on 3 Sep 2026, unchanged:** every `_unsub`
> adoption rate and mean NPV, both baseline fuel-cost means (heating
> $20,362.5614700378 over 260,211; cooling $10,097.3677096370 over 250,576),
> and all sixteen climate means -- each reproduces to the last digit recorded
> above. The MP3-vs-MP4 replacement-cost identity holds with 0 mismatches, and
> the NPV ordering checks return 0 violations across all nine cases in both MPs.
>
> **One correction, applied 3 Sep 2026:** the 19-20 August Session Log row said
> the replacement-cost identity holds "on 260,211/250,576 homes". Heating's
> 260,211 is right, but the cooling replacement-cost column is non-null for
> 250,307, not 250,576 -- the latter is the `include_cooling` count, which is a
> different quantity (a home can pass the cooling filter and still carry no
> cooling replacement cost). The cooling figure has been CORRECTED in place in
> that row, with a marker naming the old value, since this was a transcription
> error rather than a superseded measurement. The 0-mismatch result is
> unaffected. Note that 250,576 remains correct wherever it describes
> `include_cooling` itself, such as the baseline cooling fuel-cost row above.

---

## Session Log (brief)

| Session | Date | Key outcomes |
|---|---|---|
| Round 1–2 | (dates TBD) | See SESSION_CHANGELOG.md |
| Round 3 | 10 Jun 2026 | moreWTP >= 0 adoption; econ adoption choropleth; subtitle convention locked; dict-title bug fixed |
| Session 1 | ~14 Jun 2026 | Scenario consolidated to `'2025 Reference Case'`; fuel prices + HDD/CDD rewired to AEO2026 CSVs; cooling re-enabled; baseline fuel costs verified on 331,531 homes |
| Session 2 | 16 Jun 2026 | NPV import fix; three NPV cases; tiered adoption deprecated; econ adopter columns for all three cases |
| Session 1c | 23 Jun 2026 | EIA fetch functions extracted to `eia_api_utils.py`; notebook has zero inline `def` statements |
| Session 1d | 23 Jun 2026 | PEP 8 cleanup: E221/E241 padding, E501 long lines, named API dicts, plain-language comments |
| Session 1e | 28 Jun 2026 | Income/rebate/capital to USD2025: ANCHOR_YEAR centralized; REMDB v4 costs inflated 2023->2025; income source swapped to ACS-2024 B19013 (PUMA dropped, county->state); rebate bins repointed USD2018->2025; BLS CPI read fixed; LMI share 71.6%->62.4% |
| Session A | Jul 2026 | NPV-case rename refactor: `heating_only`/`heating_and_cooling_*` retired; new tokens `heatingSavings_coolingLCC`, `heatingLCC_coolingSavings`, `heatingLCC_coolingLCC`; `moreWTP`/`v4MID` removed from column names; column-name builders updated; all downstream consumers migrated; tests updated |
| 6 July 2026 | 6 Jul 2026 | Six NPV/adopter cases: each of the three scope tokens split into `_sub`/`_unsub`; `create_npv_case_col` added; `peak_load_functions` defaults to `heatingLCC_coolingSavings_sub`; Option A dotplot plots subsidized adoption with unsubsidized deltas. Loose ends closed in Session B below. |
| Session B | 7 Jul 2026 | Capital-cost refactor + baseline oracle (`scripts/capture_capital_cost_baseline.py`). CLAUDE.md updated to six-case naming. Old-token sweep: `create_npv_col` (moreWTP/lessWTP) still coexists with `create_npv_case_col`; notebook exports carry `moreWTP`/`iraRef`/`preIRA` stragglers plus half-migrated `create_npv_case_col(..., wtp=..., cost_scenario=...)` calls that raise `TypeError`. Flagged for hand-migration. Propagation verified PASS for `compute_adoption_rate`, `visuals_adoption_potential`, `visuals_adoption_dotplot`. See `REFACTORING_GUIDE_07July2026.md`. |
| 11 July 2026 | 11 Jul 2026 | Rebate-regime axis: added `_sub_june2026` NPV/adopter cases (`NPV_CASE_CATEGORIES` 6->9); `calculate_rebate_june2026` (HEEHR + HOMES, fuel gate) writes `mp{mp}_heating_rebate_amount_june2026_*` + `mp{mp}_rebate_eligibility_june2026`; `create_rebate_col` gained a `guidance` token; whole-home `baseline_total_site_consumption` + `mp{mp}_modeled_savings_frac` added in `process_euss_data.py`. Existing `_sub`/`_unsub` left byte-identical EXCEPT South Dakota homes: `NON_PARTICIPATING_REBATE_STATES = {'SD'}` now zeroes rebates in ALL regimes (2024 and June 2026), since SD never participated. Added `summarize_june2026_rebate_totals` (weighted HEEHR/HOMES dollars, national + per state). 8 rebate tests pass; 0 new suite failures (11 pre-existing). Full-run verification (byte-identity numbers, movement cross-tab, golden values, visuals) and `.ipynb` backport deferred to the researcher's environment. |
| 12 July 2026 | 12 Jul 2026 | v3 dead-code removed from the scenarios export + nine-case/no-WTP doc-string refresh (zero output change). ENERGY STAR MP3 override: `process_euss_data.df_enduse_compare` rewrites MP3's `upgrade_hvac_heating_efficiency` SEER 15->16 / 9.0->9.5 HSPF (gated `menu_mp == 3`; cooling column rewritten in parallel, no-op today); only SEER1 (pm2) feeds the REMDB v4 upgrade cost, raising MP3 heating-upgrade capital cost +$796.83/home weighted. MP3 added to `REBATE_ELIGIBLE_HEATING_MPS`. Nothing outside MP3 moves. Task 3 (re-enable v4LOW/v4HIGH) SKIPPED -- NPV/adopter columns carry no cost-scenario token, so a multi-scenario loop would overwrite (last-wins); real NPV sensitivity needs a builder+consumer refactor, deferred. Full-run golden re-derivation + `.ipynb` backport deferred to the researcher. |
| 13 July 2026 | 13 Jul 2026 | Post-12-Jul audit fixes. Rebate fuel gate ADJUDICATED KEEP (electric-resistance-only; June 2026 guidance forbids funding fossil-system removal) -- NOT an inversion; two twin scenarios-tail verification cells consolidated into ONE spec-driven test (asserts fossil=$0 AND electric-resistance funded; false ".ipynb" docstring removed); zero value move. Tepper home_count WARN moved into the main notebook with a one-home tolerance read from `df_baseline['weight'].median()` (not hardcoded); in-function exact-match WARN removed; exported values unchanged. Fresh-run 'N'-path fixes: funding cell rebuilt on `DATAFRAMES_BY_MP[mp]['fixed_base']` via helpers, `importlib.reload` cell deleted, inventory cell guarded, `FIGURE_DPI=600` defined. Cleanup: choropleth deduped (vmax=100 kept), SIX->NINE adopter comment, `fuel_counts_millions` de-hardcoded (`* 242` -> `['weight'].sum()`, ~0.05% annotation nudge), dead `.columns` cells removed, `less/more WTP` string + mid-notebook `VERBOSE=True` removed, MP3 header ENERGY STAR parity with `.ipynb`. `.py` exports only; `.ipynb` backport + full-run verification deferred to the researcher. |
| 14 July 2026 | 14 Jul 2026 | Rebate consolidation (DRY) + 2024 HOMES. ONE central `calculate_rebate_program(guidance)` in `determine_rebate_eligibility_and_amount.py` dispatched by `REBATE_RULE_CONFIG` (constants.py); `calculate_rebateIRA`/`calculate_rebate_june2026` are DEPRECATED thin wrappers. Byte-identical for 2024 HEEHR, June 2026 HEEHR, June 2026 HOMES (verified on a 217-home grid incl. exact half-cent rounding -- kept per-vintage `heehr_python_round`). ONE value move: 2024 HOMES enabled, FUEL-NEUTRAL (SUPERSEDES the 13 Jul "adjudicated KEEP electric-only" wording -- fuel gate is HEEHR-only; HOMES may fund fossil). `summarize_rebate_funding` now reads the explicit `mp{mp}_rebate_eligibility_ira2024`/`_june2026` label instead of inferring HEEHR from a positive amount. 2026 HOMES stays electric-gated this session (byte-identity; fuel-neutral fix DEFERRED -- would move `_sub_june2026`). Migrated the scenarios export to the guidance loop. Dotplots relabeled (`Unsubsidized` / `December 2024 Rebate Eligibility` / `June 2026 Rebate Eligibility`); added `REPLACEMENT_CREDIT_*`, `NATIONAL_FUEL_GROUPING_ORDER`, `build_replacement_credit_legend_handles()`, and folded `scaling_factor=242` to weight-derived homes; fixed the "Heating Repl. Credit Only" legend mismatch. 13 rebate tests pass. `.py` exports + module only; `.ipynb` backport + full-run golden re-derivation deferred to the researcher. |
| 14 July 2026 (dotplot) | 14 Jul 2026 | Adoption dotplot revisions -- visualization only, no NPV/adopter/rebate math touched. (1) Added the third replacement-credit scope `heatingSavings_coolingLCC` (cooling replacement only) so the first plot cell now shows three markers per row instead of two; driven by the new ordered `REPLACEMENT_CREDIT_SCOPES` list (heating, cooling, both). (2) `build_econ_plot_df` gained a plain `rebate_vintage` argument (`'sub'` = December 2024, `'sub_june2026'` = June 2026) so the plotted vintage is not buried in a hardcoded column name; default `'sub'` keeps the two old scopes byte-identical (verified on MP3 and MP4 National). (3) VALUE MOVE: the first plot cell now plots the June 2026 subsidized rate instead of December 2024 (`rebate_vintage='sub_june2026'`); the cell prints the December 2024 -> June 2026 National rate per scope so the move is visible on every run. Real before/after needs a full run. (4) Filled-star emphasis: `plot_adoption_panel` gained `filled_tier` -- the one headline case is drawn filled with a star shape, every other marker is an empty outline. Cell 1 stars "Heating + Cooling Repl. Credit"; Cell 2 stars "June 2026 Rebate Eligibility". Legend builders gained a matching `filled_case`/`filled_label`. (5) Annotation spacing: 3+ marker clusters go back to a left/center/right split instead of the vertical ladder; both cells pass a nonzero `annotation_x_offset_pts` (26). Note: the label/marker constants live in `visuals_adoption_dotplot.py`, NOT `constants.py`. `.py` export + module only; `.ipynb` backport of the Cell 1/Cell 2 changes deferred to the researcher. |
| 23 July 2026 | 23 Jul 2026 | County/state map geometry vintage -- DISPLAY LAYER ONLY, no modeled value moved. Root cause: the maps loaded TIGER 2025 county polygons, but Connecticut switched its eight counties (FIPS 09001-09015) to nine planning regions (09110-09190) starting with the 2022 Census vintage, while ResStock 2022.1.1 still carries the eight county codes -- so every CT row found no polygon and the state rendered gray. Fix: (1) centralized the vintage in `adoption_kpis/data_loading.py` via `COUNTY_GEOMETRY_PRODUCT`/`_VINTAGE`/`_SCALE` (+ a `_county_shapefile_stem` helper) and mirrored `STATE_GEOMETRY_*` (+ `_state_shapefile_stem`); `COUNTY_SHAPEFILE_PATH` and `SHAPEFILE_PATH` now derive from these -- no shapefile name is hardcoded elsewhere. (2) Swapped counties AND the state overlay to the 2021 cartographic boundary files (`cb_2021_us_county_500k`, `cb_2021_us_state_500k`); 2021 is the newest vintage still carrying the eight CT counties (2022 already has planning regions -- one year earlier than the commonly cited 2023). State overlay moved to `cb` too so the generalized shorelines/interior borders align with the county fill. (3) Added a loud two-direction join-coverage check in `prepare_county_geodataframe` before the `notna()` drop: unmatched DATA ROWS (alarm, warns; 0 on cb_2021) vs unmatched POLYGONS (benign, 10 unsampled counties). Forcing tl_2025 reproduces 8 unmatched CT data rows + 19 unmatched polygons. Verified via `.dbf` parse: cb_2021 = 3,234 polygons, 8 CT counties, non-CT GEOID set byte-identical to tl_2025 (so non-CT joins unchanged); researcher confirmed 0/10 on the national frame. `.py` + geometry only; `.ipynb` backport deferred; `grid_impact/calculate_postTARE_ts_aws_peak_demand.ipynb` being archived (its inline `tl_2025` path is not live code). Cosmetic rename applied in the export: the `TIGER_*` county-column symbols became `CENSUS_*` (the product is no longer TIGER), with a paste-ready notebook cell handed to the researcher for the `.ipynb`. Also recorded: the CT income fallback is a CLOSED limitation (state-level AMI; no PUMA tier). |
| 12 August 2026 | 12 Aug 2026 | Anchor-year fix -- INTENDED VALUE MOVE. Every lifetime cost stream moved from 2024-2038 to 2025-2039. The code built year labels starting at 2024 (`year + 2023` in the fuel-cost loop, `base_year: int = 2024` in `calculate_private_npv`), and three data sources had been given an invented year 2024 to match, so the stream ran `[2025, 2025, 2026, ... 2038]` -- first year duplicated, 2039 never reached. All three workarounds deleted (fuel prices, degree days, climate emissions); `base_year` removed as a parameter from `calculate_private_npv`, `_calculate_discounted_savings` and `calculate_lifetime_climate_impacts`; there is now one `ANCHOR_YEAR` in the project and every source starts at 2025. Second change, under a logged one-off exception to the never-edit rule: `replace_small_values_with_nan` in `utils/validation_framework.py` keeps an exact `0.0` instead of filtering it to NaN, which restored 1,279 real homes to the MP4 NPV and adoption results. Full detail, before/after numbers and the notebook handoff: `docs/SESSION_CHANGELOG_2026-08-12.md`. All golden values from this session were provisional -- confirmed 17 Aug 2026 below. |
| 17 August 2026 | 17 Aug 2026 | Golden-value confirmation -- DOCUMENTATION ONLY, no code changed and no value moved. Audited the full end-to-end run `2026-08-17_19-16` (National, MP3 + MP4, all 331,531 homes, `fixed_base`) and recomputed every PROVISIONAL 12 Aug quantity directly from its output. All match: baseline heating fuel cost $20,362.56, baseline cooling fuel cost $10,097.37, MP4 `heatingLCC_coolingLCC_unsub` NPV -$5,838.23, MP4 adoption 18.4416% (47,987 of 260,211), and all sixteen climate means. Largest difference anywhere is last-place rounding against a two-decimal entry. Verified from the output itself that the run carried current code: year columns run 2025-2039 with no 2024 column, and usable NPVs = 260,211 = `include_heating` (the pre-Option-B count was 258,932). MP3 measured for the first time: NPV -$4,852.41, adoption 27.7140% (72,115 of 260,211). PROVISIONAL rows kept as history and annotated; CONFIRMED rows added. Two intermediate rows (-$5,816.35 and 18.4354%) are marked NOT CONFIRMABLE -- the current code carries the exact-zero fix, so that halfway state cannot be reproduced. Then derived the four PENDING rows from the same run: all nine NPV cases for both MPs (MP4 `heatingLCC_coolingLCC` runs 18.4416% unsub / 46.3462% sub / 27.2537% june2026), the rebate movement, the rebate totals, and the SD gate. **One correction:** the movement row's claim that electric homes above 150% AMI GAIN HOMES under June 2026 was only true against the OLD pre-14-Jul `_sub`; against the current `_sub` the movement is losses only -- 0 homes gain, 49,681 MP4 homes lose -- and fossil homes above 150% AMI lose HOMES too, because 2026 HOMES is still electric-gated. The 2024 HOMES rise (+3.9510 pp MP4) is RECONSTRUCTED, not measured -- the pre-14-Jul code is gone. Still uncovered: the other three discount rates (the run wrote `fixed_base` only). Detail: `docs/SESSION_CHANGELOG_2026-08-17.md`. |
| 19-20 August 2026 | 20 Aug 2026 | Old-system-size fix for replacement-cost pricing -- INTENDED VALUE MOVE (found and scoped 19 Aug, fixed and confirmed 20 Aug). The avoided-replacement-cost credit (what a household would have paid to replace their OLD heating/cooling system, credited against the heat pump's cost in the NPV) was being priced off the NEW heat pump's size and efficiency instead of the old system's, because no baseline capacity or efficiency column survived past `df_enduse_refactored`. Efficiency turned out to already be correct by construction (`hvac_heating/cooling_efficiency` matched `base_heating/cooling_efficiency` on all 331,531 homes in both MPs, confirmed before touching it) and was renamed only, with zero output change. Size was genuinely wrong: added `base_size_heating/cooling_system_primary_k_btu_h` to `df_enduse_refactored` (straight copy from the raw baseline file's `out.params.size_*`) and pointed `add_remdb_metrics`'s replacement case at them, leaving the upgrade case (the heat pump's own cost) on `size_*` unchanged -- confirmed byte-identical on the full population. Re-run `2026-08-19_20-56` (National, MP3+MP4, same 331,531 homes/order as `2026-08-19_13-19`) confirms the size fix moves numbers in both directions, not uniformly: mean heating replacement cost MP3 -$62.79 / MP4 +$148.99, mean cooling replacement cost MP3 -$663.19 / MP4 -$502.02, with Room AC baselines dropping sharply (heat pump capacity far exceeds a room unit) and fossil-fuel heating baselines mostly rising. Headline `heatingLCC_coolingLCC_unsub` adoption: MP3 27.7140% -> 27.1760% (-0.538 pp), MP4 18.4416% -> 18.0984% (-0.343 pp). New internal-consistency check enabled by the fix: MP3 and MP4 now produce byte-identical heating and cooling replacement-cost columns (0 mismatches on 260,211 heating / 250,307 cooling homes -- the cooling count was CORRECTED from 250,576 on 3 Sep 2026; 250,576 is the `include_cooling` count, not the non-null count of the cooling replacement-cost column. The 0-mismatch result is unaffected), which could not hold before since each MP priced replacement off its own differently-sized retrofit heat pump. NPV identity and CLAUDE.md's ordering checks verified with 0 violations. Compared `_unsub` cases only, per a documented income-random-seed caveat on `_sub`/`_sub_june2026`; other MPs, discount rates, the dead v3 cost path, and `validate_capital_costs.py`'s size-based groupings are unaffected/deferred. Five golden rows superseded, five new CONFIRMED rows added. Full detail: `docs/SESSION_CHANGELOG_2026-08-20.md`. |
| 2-3 September 2026 | 3 Sep 2026 | Notebook cleanup -- NO VALUE MOVED, plus six stale golden rows measured. Three functions introduced by the DRY-consolidation session were defined but never imported at their call sites; the first two were fixed earlier, and this session fixed the third, `plot_county_demand_grid`. Its call site also had no data to plot: the per-MP x per-scenario Step 7 loop that builds `df_profiles_by_mp` / `peak_results_allegheny_by_mp` via `compute_county_scenario_profile` had been dropped, and the bare call had lost its `if GRID_IMPACT_ANALYSIS:` guard. Both restored; the wrapper's signature was read first and needed no call-site change. Verified by a full Allegheny run: all four panels render, peak MW non-null, peak hours 4433 (baseline, summer) and 152/153 (post-retrofit, January) inside [1, 8760]. Standing lint check adopted: `ruff check --select F` on the exported `.py` (magics stripped) catches exactly this class of bug -- it flagged all three missing imports plus 22 unused ones, which were then removed across 6 declaration locations in cells 1, 9 and 17, leaving the file clean on ruff's F rules except one cosmetic F541. The commented-out raw-EUSS block was already gone (removed between the 1 Sep and 2 Sep exports). Verification: run `2026-09-02_19-04` is byte-identical to `2026-08-19_20-56` on all 17 output files (SHA-256), confirming the cleanup moved nothing. Every `_unsub`, baseline fuel-cost and climate golden value reproduces to the last digit; MP3-vs-MP4 replacement-cost identity 0 mismatches; NPV ordering checks 0 violations. Six `_sub` / `_sub_june2026` rows superseded -- they had sat at 17 Aug values since the 20 Aug session compared `_unsub` only. Their pp deltas are NOT cleanly attributable to the size fix (income random-draw caveat) -- see the note below the table. Also corrected: the 20 Aug row's cooling denominator, 250,307 not 250,576. Version made consistent at 3.0 (`setup.py` said 2.0, README said 2.1); README entry point and structure tree corrected; `setup.py` UTF-8 read fix so `python setup.py --version` runs on Windows. Full detail: `docs/SESSION_CHANGELOG_2026-09-02.md`. |

---

## Coding Standards

### Documentation and comments

- **Google-style docstrings** on all new functions -- include Args, Returns, and Raises sections.
- **Comments explain WHY, not what.** A comment that restates the code adds nothing. Explain the reason for a decision, the constraint it satisfies, or the non-obvious consequence.
- **Business logic and domain knowledge.** Any calculation or filter that depends on a research-specific decision (e.g. why a SEER threshold is set where it is, why an income group is excluded, what a specific AEO series ID represents) must have a comment explaining the rationale. Future readers will not have access to the methodology notes.
- **Multi-step processes.** For functions or cells with distinct phases, use labeled step comments so the structure is scannable without reading every line:
  ```python
  # Step 1 -- validate inputs
  # Step 2 -- fetch and tidy data
  # Step 3 -- compute factors and export
  ```
- **Assumptions.** When code assumes something about data shape, value ranges, or upstream processing, make it explicit in a comment:
  ```python
  # Assumes df_baseline has already been filtered to include_heating = True.
  ```
- **Plain language only.** Avoid technical jargon. Do not use terms like "invariant", "round-trips", or internal code-history references. Write as if the reader has never seen the git log.
- **No stale references.** If a comment names a function, file, or module, confirm it still exists before writing it.
- **Type hints** on all new function parameters and returns. For complex types, import from `typing`: `Optional`, `Union`, `Tuple`, `Dict`, `List`. Use `Optional[X]` rather than `X | None` for Python 3.9 compatibility.

### PEP 8 compliance

- **Line length: 88 characters maximum** (Black default). Wrap longer lines using implicit string concatenation, backslash continuation, or by extracting a named variable.
- **No alignment padding in assignments (E221).** Write `x = 1` not `x      = 1`. Extra spaces to align `=` signs across lines violate PEP 8 and create maintenance burden when a key is renamed.
- **No alignment padding in dicts (E241).** Write `"key": value` not `"key":    value`. Same rule as E221.
- **Validate inputs at the top of functions**, before any computation. Check types, ranges, and required columns with informative error messages that name both the expected and actual values.
- **Use specific exception types**: `ValueError` for invalid values, `TypeError` for wrong types, `KeyError` for missing keys. Avoid bare `Exception` or `RuntimeError`.
- **Fail fast, fail loud.** Let errors surface immediately where they originate; do not let a bad value propagate silently through several steps.
- **Graceful fallback where appropriate.** For data fetch operations that may fail for some states or regions, log a warning and continue with a national fallback rather than crashing the whole pipeline.
- **Float64** for all econ and adopter columns (0.0 / 1.0) — avoids pandas FutureWarning.
- **DEBUG = False** as default in `constants.py`; never ship with True.
- **ASCII characters only.** Do not use Unicode symbols in code, comments, or
  markdown cells. Use these ASCII equivalents instead:
  - Arrows: `-->` not `→`; `=>` not `⇒`
  - Em dash: `--` not `—`; en dash: `-` not `–`
  - Division: `/` not `÷`; multiplication: `x` not `×`
  - Check mark: `[OK]` not `✓`
  - Ellipsis: `...` not `…`
  - Box/rule separators: `-` repeated, not `─`

### Print statement conventions

Match the structure of the code to the structure of the output:

- **Independent output lines** → separate `print()` calls.
- **Single long status line** → implicit f-string concatenation (PEP 8 endorsed):
  ```python
  print(
      f"After tidy: {len(df)} rows | "
      f"fuels={sorted(df['fuel_type'].unique())} | "
      f"regions={df['region'].nunique()}"
  )
  ```
- **Formatted summary block** (PASS messages, multi-field summaries) → triple-quoted f-string with a backslash after the opening quotes to suppress the leading blank line:
  ```python
  print(f"""\
  [PASS] Fuel-price factors written
         Shape: {df.shape} | All {ANCHOR_YEAR} factors = 1.0""")
  ```

### API call parameter dicts

Never inline a multi-key parameter dict inside a function call. Define it as a named variable first, then unpack it. This keeps the call site to one readable line and makes the parameters independently inspectable.

```python
# Correct — parameters are named and scannable independently
aeo_params = {
    "facets[scenario][]": SCENARIO_ID,
    "frequency": "annual",
    "start": str(ANCHOR_YEAR),
}
rows = eia_get(f"aeo/{AEO_YEAR}/data", api_key=EIA_API_KEY, **aeo_params)
```

### Simplicity and naming

- Prefer named intermediate variables over complex inline expressions.
- Temporary DataFrames used only to derive the next step should have a descriptive name (`df_tidy`, `df_real`, `df_states`), not a generic name like `df`.

---

## Known Anti-Patterns

Do not suggest any of these:

```
❌ Import from hdd_consumption_utils — use degree_day_consumption_utils instead
❌ Use strict > 0 for adoption decision -- the threshold is NPV >= 0
❌ Use old WTP framing: moreWTP, lessWTP -- NPV >= 0 is the only threshold; no WTP token in column names
❌ Use old NPV case tokens: heating_only, heating_and_cooling_savings, heating_and_cooling_full -- retired in Session A
❌ Embed v4MID in NPV or adopter column names -- cost scenario is no longer a column-name token
❌ Let climate/health damages enter the adoption decision
❌ Hardcode 'mp3', 'ref2025_mp3_', 'aeo2026_mp3_', 'iraRef_mp3_', or any scenario prefix
❌ Use old scenario strings: 'AEO2023 Reference Case', 'No Inflation Reduction Act', preIRA, iraRef, aeo2026_mp{mp}_
❌ Rename anything in fetch_aeo_data_and_project_EXPORT_24June2026.py
❌ Add 'Electricity ASHP' or any ASHP variant to EQUIPMENT_SPECS / ALLOWED_TECHNOLOGIES['heating'] — existing-ASHP homes are excluded by design
❌ Read degree-day CSV without int-casting year columns — silent flat 1.0 results
❌ Use full state name as price lookup key ('Pennsylvania') — must be abbreviation ('PA') — fails silently as zero
❌ Apply cooling savings to homes with include_cooling = False
❌ Collapse three NPV cases into one combined value
❌ Derive operating-cost % from ratio formula — always use (new - old) / old * 100 on per-home cols
❌ Route adoption share through pct_change — it is a share (0–100%), not a percent change
❌ Delete the tiered adoption module — prepend deprecation header only
❌ Generate econ adopter columns inside a loop — generate all per MP in a single block
❌ Edit .ipynb JSON directly — backport accepted changes manually
❌ Edit any *_EXPORT_*.py file — it is a read-only snapshot of a notebook; change the modules and hand over copy-paste cells to backport
❌ Call a ResStock row count "homes" — rows are representative dwelling units; multiply by weight (242.131013) for actual homes. A count under ~242 is always rdu
❌ Edit validation_framework.py — never
❌ Silently overwrite a golden value — keep old row with 'superseded by Session N' note
❌ Skip the pre-edit audit — read actual file state before every change
❌ Batch edits across files — one diff at a time
❌ Alignment padding in assignments (E221): `x      = 1` — write `x = 1`
❌ Alignment padding in dicts (E241): `"key":    value` — write `"key": value`
❌ Lines over 88 characters — wrap using implicit concatenation or named variables
❌ Inline multi-key dicts in function calls — define as a named dict before the call
❌ Jargon in comments — use plain language; no internal code-history references
❌ Stale function or module references in comments — confirm they exist before naming them
```
