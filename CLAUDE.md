
# CLAUDE.md — TARE Model / Joseph et al. 2026
## Heat-Pump Electrification Economics (ResStock 2022.1.1 / EUSS)
#
```text
# Last updated: 3 September 2026 — notebook cleanup session
#   - Restored the missing `plot_county_demand_grid` import.
#   - Restored the Step 7 county-profile loop under its `GRID_IMPACT_ANALYSIS` guard.
#   - Removed 22 unused imports.
#   - Run `2026-09-02_19-04` verified BYTE-IDENTICAL to `2026-08-19_20-56` across all
#     17 output files — no modeled value moved.
#   - The six `_sub` / `_sub_june2026` reference rows that the 20 Aug session had left at
#     17 Aug values were measured and superseded (see the attribution caveat in
#     `docs/REFERENCE_VALUES.md`).
#
# Previously: 20 August 2026
#   - Fixed the replacement-cost capacity and efficiency inputs to read the OLD
#     heating/cooling system's own size and efficiency, instead of the new heat pump's.
#   - Re-run `2026-08-19_20-56` (National, MP3 + MP4, fixed_base) confirmed against
#     `2026-08-19_13-19`.
#   - Five `_unsub` reference rows superseded; new CONFIRMED rows added. (The 12 Aug
#     anchor-year fix and the 17 Aug confirmation in `docs/REFERENCE_VALUES.md`
#     are unaffected.)
```

> This file is read by Claude Code at the start of every session. It is the authoritative
> source of truth for project architecture, naming conventions, and permanent constraints.
> Session-specific prompts take precedence over this file when there is a conflict.

---

## Project at a Glance

- **Research question:** Economics of heat-pump electrification across U.S. counties
- **Data:** 331,531 baseline representative dwelling units | 3,098 counties (ResStock 2022.1.1 EUSS)
- **Heat-pump models:** MP3 (standard ASHP, 15 SEER1, 9 HSPF1) | MP4 (high-efficiency ASHP, 24–29.3 SEER1, 13–14 HSPF1)
- **Policy scenario:** Single — `'2025 Reference Case'` (see Hard-coded Values below)
- **Adoption metric:** `NPV >= 0` — economic payback only; no climate/health damages in the adoption decision

---

## Abbreviations Used Throughout This File

- **ASHP** — air-source heat pump (the technology this study evaluates)
- **MP3 / MP4** — the two heat-pump "measure packages" modeled (see above)
- **SEER / HSPF** — Seasonal Energy Efficiency Ratio / Heating Seasonal Performance Factor (efficiency ratings for cooling / heating equipment)
- **rdu** — representative dwelling unit (see Terminology, below — not the same as a home)
- **NPV** — net present value (the economic payback measure used for the adoption decision)
- **LCC** — lifecycle cost (the avoided-replacement capital cost credited in some NPV scopes)
- **AMI** — area median income (used to determine rebate eligibility)
- **HEEHR / HOMES** — the two federal rebate programs modeled (see Rebate Policy Scenarios, below)
- **CDD / HDD** — cooling degree-days / heating degree-days (used to project weather-driven energy use)
- **EUSS** — End Use Savings Shapes (the ResStock dataset release this project uses)

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

**Reading the older rows in this file.** The Reference Values table (`docs/REFERENCE_VALUES.md`) and the
Session Log (`docs/SESSION_LOG.md`) were written before this rule, so they say "homes" when they
mean representative dwelling units. For example, "260,211 homes with
`include_heating = True`" is really 260,211 rdu — 63,005,153 real dwellings.
Those rows are left exactly as written, so no reference value appears to have
been altered.

When reading them: treat every count as rdu unless it is explicitly weighted.
The dollar means, rates, and percentages in those rows are unaffected, because
the weight is the same for every row.

Reference conversions: 331,531 rdu = 80,273,937 homes | 260,211 rdu =
63,005,153 homes | 250,576 rdu = 60,672,221 homes | 1 rdu = 242.131013 homes.

---

## Critical Rules — Read First

These apply to every session, every task, without exception.

### Files that must NEVER be edited

| File | Reason |
|---|---|
| `utils/validation_framework.py` | Core validation logic — never touch unless given express permission from the researcher |
| Any `.ipynb` file | VSCode in-memory cache causes changes not to persist; backport manually |
| Any `*_EXPORT_*.py` file | Read-only snapshot of a notebook — change the importable module instead and hand the researcher cells to backport |

### One-edit-per-stop-gate rule

Before applying any edit:

1. Show the researcher the exact diff (old → new), with 3–5 lines of context
   above and below the change.
2. Wait for explicit approval.
3. Only then call the Edit tool.

Do not batch edits across files or functions — one edit, one approval, at a time.

### Audit before every edit

Read the actual current file state before proposing any change — do not
assume what a previous session left in place. Sessions have sometimes ended
mid-task, so the actual state can differ from what the log describes.

### Committing is the researcher's job

Never run `git commit`, `git add` (staged for a commit), `git commit --amend`,
`git reset`, `git revert`, `git push`, or any other history-changing git
command. The researcher makes every commit and writes every commit message by
hand.

Just make and save the file changes, and leave them in the working tree for
the researcher to stage and commit. A summary or a suggested commit message
can go in the chat — but don't act on it.

### Regression guarantee — dropped for the ResStock 2025.1 integration (19 Sep 2026)

The 2022.1.1 MP3/MP4 pipeline is no longer required to reproduce `2026-09-02_19-04`
byte-identically, on any task or at any phase boundary, including a final check at the end of
this integration. This branch exists to move onto ResStock 2025.1; the 2022.1.1 analysis is
preserved in a separate, already-archived branch and will not be run from here. The 2022.1.1
code path is still left alone by default -- this is not license to break it for no reason -- but
it is not verified, proven, or reconciled going forward. Work already done under the old
guarantee (through Phase 2 Task 6) is unaffected and does not need to be redone.

---

## Hard-coded Values

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
| ResStock source | ResStock 2022.1.1 EUSS | |
| County + state map geometry | `cb_2021_us_county_500k`, `cb_2021_us_state_500k` (under `data/shapefiles/`) | Census cartographic boundary files, 2021 vintage, 500k scale. Matched to ResStock's pre-2023 geography; Connecticut is the binding constraint (see CT note below). Vintage set once via `COUNTY_GEOMETRY_*` / `STATE_GEOMETRY_*` in `adoption_kpis/data_loading.py` -- never hardcode a shapefile name elsewhere. |
| Area median income (AMI) | `ACSDT5Y2024.B19013-Data.csv` (under `data/ami_calculations_data/`) | U.S. Census Bureau ACS 5-Year table B19013 (median household income), vintage 2024, from data.census.gov. One file holds county (`0500000US`) and state (`0400000US`) rows; inflated USD2024->2025. NOT NHGIS -- the NHGIS PUMA source was retired in Session 1e. |


**Degree-day read pattern (mandatory):**
```python
df = pd.read_csv(PATH)
df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]  # MUST cast to int
```
Skipping the int cast causes year lookups to silently return 1.0 (no projection applied).

**State key format:** Two-letter abbreviation (`'PA'`, `'TX'`), NOT full state name.
A wrong key returns silently as zero — no error, just wrong output.

**Connecticut income fallback (CLOSED limitation, not deferred).**

- The county AMI source (ACS B19013, vintage 2024) uses post-2022 county
  geography, so its Connecticut rows are the nine planning regions
  (FIPS 09110–09190).
- ResStock 2022.1.1 still carries the eight pre-2023 CT counties
  (09001–09015). So the county AMI join misses every CT home, and those homes
  fall back to a state-level `census_area_medianIncome` in
  `fill_na_with_hierarchy`
  (`private_impact/data_processing/determine_rebate_eligibility_and_amount.py`).
- This is accepted: no PUMA tier will be added, and the map and income
  sources are deliberately left decoupled for CT.

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

### Raw ResStock source columns (both releases)

A simple rename must never read as a new column. When a downstream frame (like
`df_enduse`) carries a field sourced from a column the map marks `renamed`, give it a
stable, release-invariant key -- never the raw physical name from either release, and
never a `RESSTOCK_COLUMN_MAP` logical_name. `df_enduse_refactored`'s existing keys
(`base_heating_fuel`, `square_footage`, ...) already follow this; extend the same
pattern for the new 2025.1-only pass-through columns Task 6 adds.

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
- Build these with `create_npv_case_col(scenario_prefix, npv_case, method_suffix)`.
  `npv_case` must be one of `NPV_CASE_CATEGORIES` (column_names.py). Note there is
  no cost-scenario token and no WTP token in these names.
- `LCC` = that end-use's avoided-replacement capital is credited in the NPV
- `Savings` = only operating savings credited for that end-use
- `_unsub` / `_sub` / `_sub_june2026` = the rebate-policy-scenario axis
  (unsubsidized / 2024 guidance / June 2026 guidance) — see Rebate Policy
  Scenarios, below, for what each means.
- All nine cases include BOTH heating and cooling operating savings
- `{method_suffix}` already carries its own leading underscore (e.g. `_fixed_base`)

**Economic adopter columns (nine per MP, as of the 11 July 2026 session):**
```
ref2025_mp{mp}_{npv_case}_econ_adopter{method_suffix}
```
one per `npv_case` in `NPV_CASE_CATEGORIES` (the same nine tokens listed above:
`_unsub`, `_sub`, `_sub_june2026` for each scope).

**Reference Model Scenario Variables:** `fixed_base` | `central`
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
| Rebate policy scenario | `_unsub` \| `_sub` \| `_sub_june2026` — see Rebate Policy Scenarios, below |

---

## Rebate Policy Scenarios (2024 vs June 2026 DOE guidance)

This is modeled as a sensitivity axis within one dataframe (there is no second
`policy_scenario` column) — the same way the discount-rate axis works. There
are three rebate policy scenarios per scope: `_unsub` (no rebate), `_sub`
(2024 guidance), and `_sub_june2026` (June 2026 guidance).

All rebate math is centralized in `calculate_rebate_program()`
(`determine_rebate_eligibility_and_amount.py`), dispatched via
`REBATE_RULE_CONFIG` in `constants.py`. `calculate_rebateIRA`/
`calculate_rebate_june2026` are deprecated wrappers — don't add rebate logic
outside the central function.

**State-participation gate (applies to every rebate policy scenario,
including 2024 `_sub`):** homes in a state that never participated in the
federal rebate programs get $0 under every scenario. Currently that's just
South Dakota: `NON_PARTICIPATING_REBATE_STATES = {'SD'}`. This gate applies to
both the 2024 and June 2026 vintages.

**Fuel gate — HOMES is fuel-neutral; the fuel gate applies to HEEHR only.**
(Will need to update this after reviewing and confirming new guidance post June 2026.)

The June 2026 DOE guidance forbids using a rebate to fund removing a fossil
heating system — but that restriction applies only to HEEHR. HOMES (which is
performance-based) may still fund replacing a fossil system.

- **HEEHR:** June 2026 restricts it to existing electric-resistance heating;
  any fossil baseline gets $0. 2024 HEEHR has no fuel gate (funds fossil
  baselines by design).
- **HOMES:** fuel-neutral since 14 Jul 2026 — credits homes above 150% AMI
  regardless of baseline fuel. (June 2026 HOMES is still electric-gated for
  byte-identity reasons; making it fuel-neutral is deferred.)

**Do not** re-add an electric-only gate to 2024 HOMES.

**Program rules** (apply to both vintages; both programs are gated by
`REBATE_ELIGIBLE_HEATING_MPS`, which has included MP3 + MP4 since the 12 Jul
2026 ENERGY STAR override):

- **HEEHR** (`percent_AMI <= 150%`): $8,000 heat-pump cap. Income sets the
  cost share — 100% at ≤80% AMI, 50% at 80–150% AMI. The cap is the same in
  both vintages; the only difference between vintages is the June 2026 HEEHR
  fuel gate described above.
- **HOMES** (`percent_AMI > 150%`): savings-based, using whole-home modeled
  savings (`mp{mp}_modeled_savings_frac`). ≥20% savings → $2,000 cap; ≥35%
  savings → $4,000 cap. Covers 50% of the full electrification project cost.
  These are non-LMI amounts only.

**Rounding note:** `heehr_python_round` preserves a legacy Python-vs-numpy
rounding difference between the two vintages (sub-cent only).

**Whole-home savings fraction:** the numerator is TARE's degree-day-adjusted
heating + cooling energy delta; the denominator is ResStock's
`baseline_total_site_consumption` (propagated from
`out.site_energy.total.energy_consumption.kwh`). Mixing an adjusted numerator
with a raw denominator is an accepted approximation. Now that 2024 HOMES is
fuel-neutral, this fraction is also used for fossil-fuel HOMES recipients, not
just electric-resistance homes — so the approximation applies to them too.

**Reporting / verification helpers** (both in
`determine_rebate_eligibility_and_amount.py`):

- `summarize_june2026_rebate_totals` — weighted HEEHR/HOMES dollars, national
  and per state.
- `summarize_rebate_funding` — weighted funding by program and by baseline
  fuel, split into `total_eligible` vs. `adopters_only`. As of 14 Jul 2026,
  this reads the explicit `mp{mp}_rebate_eligibility_*` label instead of
  inferring HEEHR from a positive dollar amount, since 2024 now has HOMES too.

The correctness check to run is on HEEHR specifically, not on "all
non-electric fuels": under June 2026, HEEHR must be $0 for every fossil-fuel
baseline, but HOMES may fund fossil baselines (it's fuel-neutral). See
`scripts/verify_june2026_rebate_fossil_gate.py`, which pivots program × fuel
for both vintages.

Note that `total_eligible` is uncapped potential — no state funding cap is
modeled — not an actual disbursement amount.

**Documented limitations (carry into the manuscript):**

1. Weatherization prerequisite is not enforced (state criteria not finalized).
2. Dual-fuel systems are not modeled, so fossil-baseline homes lose HEEHR
   under June 2026 (see Fuel gate, above) but can still earn fuel-neutral
   HOMES above 150% AMI.
3. Only one program per home — HEEHR or HOMES, never both.
4. State-level funding caps are not applied (allocations aren't finalized
   yet; see the Atlas Buildings Hub tracker).
5. HEEHR's ENERGY STAR requirement is statutory in both vintages; HOMES's is
   optional under June 2026 (state discretion). Moot for now since MP3/MP4
   both meet the spec (see Program rules, above). Revisit if MP8–10 are
   activated.
6. The HOMES LMI tier (doubled caps + 80% coverage) is unreachable by
   construction — HOMES only applies above 150% AMI. An LMI fossil home at
   or below 150% AMI gets HEEHR $0 (see Fuel gate, above) and can't reach
   HOMES either, so it gets $0 total. Fixing this is deferred (see the
   deferred list).
7. Uncertainty surrounding fuel switiching and project eligibility. Will need to update logic with new federal guidance. We use guidance as of June 2026.
8. Our analysis is focused marginal NPV (replacing existing fossil fuel systems with heat pumps) and does not include homes with existing heat pump systems.
9. The capital cost estimation is currently performed for homes outside of the NRL REMDB regression cost formula bounds. Plans to update this and be more clear about the sample size after each filtering step and why the homes were removed. 

---

## Masking and Validation Rules

**Heating:** `include_heating = valid_fuel_heating AND valid_tech_heating`
- `valid_fuel_heating`: fuel is one of {Electricity, Natural Gas, Propane, Fuel Oil}
- `valid_tech_heating`: technology is one of `ALLOWED_TECHNOLOGIES['heating']`
- Homes with `in.heating_fuel = 'None'` are automatically excluded

**Cooling:** `include_cooling = valid_fuel_cooling AND valid_tech_cooling`
- `valid_fuel_cooling`: hardcoded True (cooling is always electric) — this flag is a no-op
- `valid_tech_cooling`: technology is one of {Central AC, Room AC} — this is the ONLY cooling filter
- Homes with no AC (`'None'`) or evaporative coolers are excluded here

**Cooling in NPV:** for homes where `include_cooling = False`, cooling savings and capital are both 0 — see NPV Ordering Checks, below, for the resulting identities.

**Negative cooling savings — accepted as real** (12 Jul 2026 session;
national share corrected 19 Aug 2026).

For some homes, the heat pump's cooling energy use exceeds the baseline air
conditioner's, so `ref2025_mp{mp}_cooling_lifetime_savings_fuel_cost` comes
out negative.

- **Why this is real, not a bug:** baseline and retrofit cooling are scaled
  by the same CDD factor and electricity price each year, so the sign of the
  result is fixed by the raw base-year kWh difference. It's not a projection
  artifact.
- **Why it happens so often:** it's mostly a service-level change — the
  baseline room AC only cools one room, while the whole-home heat pump cools
  the entire house.
- **How common it is (measure-package specific):** for MP3, 90.68% of Room AC
  baselines go negative, vs. 10.84% of Central AC baselines. For MP4, it's
  61.97% vs. 3.46%. These figures were recomputed 19 Aug 2026 from the
  `2026-08-19_13-19` run (`docs/SESSION_CHANGELOG_2026-08-19.md`), correcting
  a single MP-unsplit "about 54% / 2.5%" figure from the 12 Jul 2026 session,
  which did not reproduce for either MP on this run.
- **Decision:** keep the negative savings in the NPV as a real operating
  cost, consistent with the dollars-only, no-WTP adoption threshold. Do not
  floor it at zero or exclude it.
- **Flagging:** a non-NPV boolean column,
  `ref2025_mp{mp}_cooling_lifetime_savings_negative`, marks the affected
  homes for reporting. This decision does not move any reference value.
- **Manuscript limitation to note:** MP4 delivers whole-home cooling that the
  room AC never did, and the dollars-only NPV counts the added cooling cost
  with no offsetting comfort credit.

**Existing-ASHP homes — resolved: exclude.**

`'Electricity ASHP'` (or any variant of it) must not appear in
`EQUIPMENT_SPECS` or `ALLOWED_TECHNOLOGIES['heating']`. The study models
transitions from fossil-fuel heating to an ASHP — a home that already has an
ASHP has no fossil-fuel system left to replace. If you find an existing-ASHP
entry in `constants.py`, flag it and remove it.

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

## Reference Values

Validated baseline numbers from past model runs, used to confirm whether a code change moved a modeled output. Full table and superseding history: `docs/REFERENCE_VALUES.md`.

**Open that file before:**
- citing any modeled quantity (an NPV, an adoption rate, a fuel cost, a climate-damage figure) in code, comments, or conversation with the researcher
- comparing a fresh run's output against a prior one
- deciding whether a code change is expected to move a modeled value

**When you do change a value:** never silently overwrite a row in that file. Add a new row marked "supersedes" and keep the old row, exactly as that file itself documents.

---

## Session Log (brief)

Full session-by-session history: `docs/SESSION_LOG.md`.

Open that file to see what changed in a specific past session, why a naming convention or column exists, or which dated `docs/SESSION_CHANGELOG_*.md` file has the full detail for a given date.

---

## Coding Standards

### Documentation and comments

- **Google-style docstrings** on every new function — include `Args`, `Returns`, and `Raises` sections.
- **Comments explain *why*, not *what*.** A comment that just restates the code adds nothing. Explain the reason for a decision, the constraint it satisfies, or a non-obvious consequence.
- **Comment on business logic and domain knowledge.** Any calculation or filter that depends on a research-specific decision — why a SEER threshold sits where it does, why an income group is excluded, what a specific AEO series ID represents — needs a comment explaining the rationale. Future readers won't have the methodology notes in front of them.
- **Label multi-step processes.** For functions or cells with distinct phases, use step comments so the structure is scannable without reading every line:
  ```python
  # Step 1 -- validate inputs
  # Step 2 -- fetch and tidy data
  # Step 3 -- compute factors and export
  ```
- **State assumptions explicitly.** When code assumes something about data shape, value ranges, or upstream processing, say so in a comment:
  ```python
  # Assumes df_baseline has already been filtered to include_heating = True.
  ```
- **Plain language only.** Avoid technical jargon — don't use terms like "invariant," "round-trips," or references to internal code history. Write as if the reader has never seen the git log.
- **No stale references.** If a comment names a function, file, or module, confirm it still exists before writing it.
- **Type hints** on every new function's parameters and return value. For complex types, import from `typing`: `Optional`, `Union`, `Tuple`, `Dict`, `List`. Use `Optional[X]` rather than `X | None`, for Python 3.9 compatibility.

### PEP 8 compliance

- **Line length: 88 characters maximum** (Black default). Wrap longer lines using implicit string concatenation, backslash continuation, or by extracting a named variable.
- **No alignment padding in assignments (E221).** Write `x = 1` not `x      = 1`. Extra spaces to align `=` signs violate PEP 8, and become a maintenance burden the moment a key is renamed.
- **No alignment padding in dicts (E241).** Write `"key": value` not `"key":    value`. Same rule as E221.
- **Validate inputs at the top of functions**, before any computation. Check types, ranges, and required columns with informative error messages that name both the expected and actual values.
- **Use specific exception types**: `ValueError` for invalid values, `TypeError` for wrong types, `KeyError` for missing keys. Avoid bare `Exception` or `RuntimeError`.
- **Fail fast, fail loud.** Let errors surface immediately where they originate; do not let a bad value propagate silently through several steps.
- **Graceful fallback where appropriate.** For data fetch operations that may fail for some states or regions, log a warning and continue with a national fallback rather than crashing the whole pipeline.
- **Float64** for all econ and adopter columns (0.0 / 1.0) — avoids pandas FutureWarning.
- **DEBUG = False** as default in `constants.py`; never ship with True.
- **ASCII characters only** (in the project's code, comments, and notebook markdown cells — not this file). Do not use Unicode symbols there. Use these ASCII equivalents instead:
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
❌ Edit validation_framework.py without the researcher's express permission — the default is still never touch it
❌ Silently overwrite a reference value — keep old row with 'superseded by Session N' note
❌ Skip the pre-edit audit — read actual file state before every change
❌ Batch edits across files — one diff at a time
❌ Alignment padding in assignments (E221): `x      = 1` — write `x = 1`
❌ Alignment padding in dicts (E241): `"key":    value` — write `"key": value`
❌ Lines over 88 characters — wrap using implicit concatenation or named variables
❌ Inline multi-key dicts in function calls — define as a named dict before the call
❌ Jargon in comments — use plain language; no internal code-history references
❌ Stale function or module references in comments — confirm they exist before naming them
```
