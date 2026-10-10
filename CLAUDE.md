
# CLAUDE.md — TARE Model / Joseph et al. 2026
## Heat-Pump Electrification Economics (ResStock 2022.1.1 / EUSS)
#
```text
# Last updated: 9 October 2026 — old boilers are priced on the REMDB boiler rows
#   - The credit for an old gas or propane boiler uses the non-condensing gas boiler
#     row, and for an old fuel oil boiler the oil boiler row, with a floor of AFUE
#     0.80. Both were on the gas furnace row. Value-moving on both releases: 20,973
#     rdu in 2022.1.1 and 2,528 in 2025.1; mean boiler credit $3,474 --> $7,638 and
#     $3,443 --> $7,766.
#   - `heatingLCC_coolingLCC` adoption with no rebate: MP3 18.81% --> 19.35%, MP4
#     18.99% --> 19.47%, MP5 3.27% --> 3.39%. The three old-AC-only cases, every
#     rebate, the bills and the study sample did not move. New rows are in
#     `docs/REFERENCE_VALUES.md`.
#   - Electric furnaces and electric boilers stay on the baseboard row, with a TODO
#     to ask NLR's REMDB researchers (Limitation 16).
#   - Limitation 9 and Limitations 15 to 18 say how each old system is priced and how
#     many rdu fall outside a cost row's range.
#   - Runs, grid impact off: 2022.1.1 `2026-10-09_00-46` and 2025.1 `2026-10-09_00-38`.
#     The default release stays 2022.1.1.
#   - Later on 9 October, with no code or value changed: Stage 2 checked the new files
#     for Chris (27 checks per package, all passed) and compared them with the 19 Aug
#     data. On the same 221,205 rdu, no-rebate adoption went from 30.18% to 19.35%
#     (MP3) and from 19.71% to 19.47% (MP4). Results: sections 7.3 and 7.4 of
#     `docs/DF_COST_FIXES_AND_DATA_SHARE_PLAN.md`.
#   - A change audit of 19 Aug to 9 Oct found that run `2026-08-19_20-56` was made on
#     commit 68f0964 plus the old-system size fix, first committed as 2f54ed2. From
#     there to 6a12e05 are 69 commits, 14 of them value-moving. `docs/SESSION_LOG.md`
#     gains the rows missing since 19 Sep. The audit and a detailed writeup are in
#     `~/tare_port/s11_outputs/`, off the repo.
#
# Previously: 7 October 2026 — dual-fuel package (ResStock 2025.1, MP5) runs end to end
#   - ResStock 2025.1 package 5, a heat pump with a gas backup furnace, runs nationally:
#     161,983 rdu (41,128,079 homes) in 2,973 counties, run `2026-10-07_19-14`. Its first
#     reference values are in their own table in `docs/REFERENCE_VALUES.md`.
#   - The release is chosen with the environment variable `TARE_RESSTOCK_RELEASE`, and
#     `scripts/run_tare_notebooks.py` runs the notebooks without the keyboard.
#   - The heat pump is priced at SEER1 16.0 (from SEER2 15.2), and the backup gas furnace
#     is priced and added to the capital cost (Limitation 14).
#   - A dual-fuel retrofit passes both June 2026 fuel gates, so its June 2026 rebate
#     equals its 2024 rebate. DOE Program Notice 26-3 is not yet reviewed.
#   - 2025.1 peak electric demand columns are named for winter and summer. The Tepper
#     household files for package 5 hold 182 and 332 columns (167 and 317 for 2022.1.1).
#   - No 2022.1.1 value moved: run `2026-10-07_19-22` is byte-identical in all 22 files
#     to run `2026-10-06_20-23`, made before the dual-fuel model code went in.
```

> This file is read by Claude Code at the start of every session. It is the authoritative
> source of truth for project architecture, naming conventions, and permanent constraints.
> Session-specific prompts take precedence over this file when there is a conflict.

---

## Project at a Glance

- **Research question:** Economics of heat-pump electrification across U.S. counties
- **Data:** ResStock 2022.1.1 EUSS. Study sample: 221,205 representative dwelling units (53,560,591 homes) in 3,079 counties (see Study Sample, below). The saved baseline table holds 331,531 rdu (occupied single-family homes) and each package table 331,526; results are filled only for the sample.
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
- Say "221,205 rdu are in the study sample" — that is **53,560,591 real
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

Reference conversions: 221,205 rdu = 53,560,591 homes (the study sample) |
331,531 rdu = 80,273,937 homes | 260,211 rdu = 63,005,153 homes | 250,576 rdu
= 60,672,221 homes | 1 rdu = 242.131013 homes.

---

## Study Sample

Every TARE result covers one set of homes: the study sample. In 2022.1.1 it
is **221,205 rdu = 53,560,591 homes in 3,079 counties**.

A home is in the sample when all three hold:

- its heating system is one the study can replace and cost
  (`include_heating = True`);
- it has a central or room AC of its own, not a shared cooling system
  (`include_cooling = True`);
- ResStock applied every measure package in the run to it.

Where it lives:

- `include_sample` is set once, in `df_enduse_refactored`
  (`process_euss_data.py`, STEP 4b). `get_valid_calculation_mask` requires it,
  so every calculation is limited to it.
- `build_tare_sample_ids` (`study_sample.py`) turns the flag into
  `TARE_SAMPLE_IDS`, for results that do not use the TARE table (county
  demand, peaks).
- `build_sample_funnel` (`study_sample.py`) builds the table below. Every run
  saves it under `baseline_summary/sample_funnel/`.

**Every computation filters on `include_sample`.** Never use
`include_heating` or `include_cooling` to mean the sample.

| Step | rdu | Removed |
|---|---|---|
| ResStock 2022.1.1 stock | 548,916 | |
| Package applies | 548,260 | 656 |
| Occupied | 482,050 | 66,210 |
| Single-family | 331,526 | 150,524 |
| Not Alaska or Hawaii | 331,526 | 0 |
| Heating fuel is electricity, natural gas, propane or fuel oil | 321,352 | 10,174 |
| No existing heat pump | 290,005 | 31,347 |
| Furnace, boiler or electric baseboard | 260,211 | 29,794 |
| Central or room AC | 221,301 | 38,910 |
| Cooling system not shared | 221,205 | 96 |

---

## Critical Rules — Read First

These apply to every session, every task, without exception.

### Files that must NEVER be edited

| File | Reason |
|---|---|
| `utils/validation_framework.py` | Core validation logic — never touch unless given express permission from the researcher |
| Any `*_EXPORT_*.py` file | Read-only snapshot of a notebook — change the importable module or the notebook itself instead (see Notebook edits, below) |

### Notebook edits — allowed, through the `notebook-cell-edit` skill (4 Oct 2026)

`.ipynb` files may be edited directly. The earlier ban was written because
GitHub Copilot had trouble reading and editing notebook cells; it does not
apply to Claude Code.

Every change to a notebook cell goes through the `notebook-cell-edit` skill
(`.claude/skills/notebook-cell-edit/`). Its script changes only the named
lines and stops unless it can show that nothing else in the file moves. Do
not retype a cell by hand, replace a whole cell with a notebook tool, or edit
the notebook JSON with a one-off script.

- The stop-gate rule below still applies: show the skill's dry-run diff, wait
  for approval, then apply.
- After every notebook edit, remind the researcher to close and reopen (or
  "Revert File") any open VS Code tab of that notebook. VS Code keeps an open
  notebook in memory, and saving a stale tab writes the old text back.
- Adding or deleting a whole cell is outside the skill: hand the researcher
  the cell to paste.

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

### Both releases must run (22 Sep 2026)

This branch runs ResStock 2022.1.1 (MP3/MP4, the Nature Comms analysis) and 2025.1
(MP5, dual fuel), chosen with `RESSTOCK_RELEASE_THIS_RUN`. Both must run end to end.
Byte-identity with the submitted run (`2026-08-19_20-56`) is NOT required: the
consumption fix (see Masking and Validation Rules) moves 2022.1.1 results on purpose.
Any other change that moves 2022.1.1 values must be flagged and made in its own commit.
The branch `joseph-2026-nature-comms-submission` is frozen as the as-submitted reference.

**Choosing the release.** `RESSTOCK_RELEASE_THIS_RUN` (`constants.py`) is read
from the environment variable `TARE_RESSTOCK_RELEASE` when `cmu_tare_model` is
first imported. Unset means `'2022.1.1'`; a value that is not a known release
stops the import. Editing `constants.py` does not switch the release, and
changing the variable after the import has no effect.

- With the runner (below): `--release 2025.1`.
- For notebooks in VS Code: close every VS Code window, then start it from Git
  Bash with `TARE_RESSTOCK_RELEASE=2025.1 code .`. Start it the usual way to
  go back to 2022.1.1.
- Never set the variable in Windows settings or a shell profile.

**Running the notebooks without the keyboard.**
`scripts/run_tare_notebooks.py` runs the main notebook from its first cell to
its last, answers the notebooks' questions from its options, and stops at the
first failing cell of any notebook. It then checks each package's results: the
NPV identity, the NPV orderings, the adopter flags, no blank value in a sample
home, and the rows of the Tepper files.

```bash
python scripts/run_tare_notebooks.py --release 2025.1 --skip-grid-impact
python scripts/run_tare_notebooks.py --release 2022.1.1 --state PA --skip-grid-impact --log run.log
```

- Leave out `--state` for the whole country. `--fips` (default 42003) names
  the grid-impact county.
- Every 2025.1 run needs `--skip-grid-impact`: the grid impact analysis works
  on ResStock 2022.1.1 only for now.
- Exit codes: 0, the run and every check passed; 1, a cell failed; 2, a
  notebook asked a question the runner does not know, or the same one twice;
  3, the run finished but a check failed.
- A national run took about 8 minutes and 11 GB of memory for 2025.1, and
  about 25 minutes and 19 GB for 2022.1.1 (7 Oct 2026). Shut down any leftover
  Jupyter kernel first.

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
| Social cost of carbon | `scc_climate_impact_sensitivity.xlsx` (under `data/projections/`) | Published in USD2020 (`scc_*_usd2020`). Inflated to USD2025 with `cpi_ratio_2025_2020` in `create_lookup_climate_impact_scc.py`, so climate damages and private costs share one dollar year. The workbook's `scc_*_usd2023` columns are not read. |


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
| Break-even COP: `compute_breakeven_cop`, `assign_breakeven_category` (`adoption_kpis/thermal_cop.py`), `plot_categorical_breakeven_map` (`adoption_kpis/visualize_geospatial_data.py`) | No longer part of the analysis. The note at `compute_breakeven_cop` says what to fix first if it is used again |

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

**Peak electric demand columns (the name depends on the ResStock release):**
```
{prefix}peak_electricity_heating_kw   {prefix}peak_electricity_cooling_kw   # 2022.1.1
{prefix}peak_electricity_winter_kw    {prefix}peak_electricity_summer_kw    # 2025.1
```
- `{prefix}` is `base_` or `mp{mp}_`. A retrofit column may end `_savings`.
- ResStock 2022.1.1 reports a home's peak during the hours its heating or its
  cooling runs. ResStock 2025.1 reports the largest daily peak in the winter
  or the summer months, whatever is running. They measure different things,
  so the two releases never share a name.
- Build them with `create_peak_electricity_col(prefix, end_use, release,
  savings)` (column_names.py), which takes `end_use` as `'heating'` or
  `'cooling'` for both releases. Never write a `..._heating_kw` or
  `..._cooling_kw` name for a 2025.1 run.
- The peak load columns (`{prefix}peak_load_heating_kbtu_hr`,
  `{prefix}peak_load_cooling_kbtu_hr`) have the same name in both releases.

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

**Program Notice 26-3.** DOE has released new guidance in Program Notice
26-3. The researcher has not yet reviewed how it differs from Program Notices
26-1 and 26-2 or how it affects this code. The rebate rules modeled here are
the ones documented in this file; no rule was changed because of 26-3.

**Dual fuel (ResStock 2025.1, package 5).** A dual-fuel retrofit keeps a gas
furnace as the heat pump's backup, so it removes no fossil heating system and
passes both June 2026 fuel gates (`dual_fuel_passes_fuel_gates` in
`REBATE_RULE_CONFIG`):

- **HEEHR:** at or below 150% AMI, for every baseline fuel.
- **HOMES:** above 150% AMI, whatever the baseline fuel.

Caps, cost shares, income routing, savings tiers and the South Dakota gate are
unchanged. So for this package the June 2026 rebate equals the 2024 rebate, to
within one cent per home (the rounding note below).

The backup furnace's cost is added to the retrofit's capital cost but is not
part of the cost a rebate covers: a fossil furnace is not a rebate measure.

To ask whether a package is dual fuel, call `is_dual_fuel_package(menu_mp)`
(`utils/calculation_utils.py`). Never test a package number alone: package
numbers repeat across ResStock releases.

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

**Whole-home savings fraction** (`mp{mp}_modeled_savings_frac`, the HOMES tier input):

Savings fraction = energy the retrofit saves ÷ energy the whole home used before the retrofit

- Energy saved: the home's annual heating + cooling energy before the retrofit
  minus after (`mp{mp}_hvac_energy_savings_kwh`), every fuel and every component
  counted. Only heating and cooling change, so the rest of the home cancels out.
- Whole-home energy before: `baseline_total_site_consumption`, all fuels and all
  end uses, because HOMES asks what share of the whole home's energy is saved.

Example home:

- Before: 20,000 kWh of gas heating, 3,000 kWh of AC, 12,000 kWh of everything
  else, so 35,000 kWh in total.
- After: 7,000 kWh of heat-pump heating, 2,500 kWh of cooling, the same 12,000 kWh
  for everything else.
- Savings = (20,000 + 3,000) − (7,000 + 2,500) = 13,500 kWh.
- Fraction = 13,500 ÷ 35,000 = 38.6%, which reaches the upper HOMES tier (35% or more).

A check column, `mp{mp}_whole_home_energy_savings_kwh` (ResStock's own whole-home
change), runs below the savings above mostly because ResStock's hot-water use also
changes when the heating and cooling equipment changes (an interaction ResStock
documents); TARE counts heating and cooling only. The
fraction is used for fossil and electric HOMES recipients alike.

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
2. The 2022.1.1 analysis (MP3/MP4) models no dual-fuel systems, so its
   fossil-baseline homes lose HEEHR under June 2026 (see Fuel gate, above) but
   can still earn fuel-neutral HOMES above 150% AMI. The 2025.1 dual-fuel
   package (MP5) is modeled. It keeps a gas furnace and removes no fossil
   heating system, so it passes both June 2026 fuel gates, and its June 2026
   rebate equals its 2024 rebate (see Dual fuel, above).
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
    The rule in the code, kept as it is on 8 Oct 2026: a size or an efficiency
    outside a REMDB row's range is priced by carrying the row's line beyond
    the range. Nothing is held at the bound. Study-sample rdu outside a range,
    2022.1.1 / 2025.1, on runs `2026-10-09_00-46` and `2026-10-09_00-38`:
    - Credit for an old furnace (gas furnace row, 30,000 to 156,250 Btu/h):
      16,859 / 18,210 below; 3,015 / 2,754 above.
    - Credit for an old boiler (both boiler rows start at 27,000 Btu/h):
      2,336 / 347 below. Propane boilers published at 90% AFUE, above the gas
      boiler row's 0.87: 192 / 1.
    - Credit for an old central AC (1.5 to 5 tons): 32,084 / 25,664 below;
      22,578 / 18,260 above.
    - Credit for an old room AC (0.4167 to 2.33 tons): 15,545 / 5,392 below;
      1,780 / 690 above.
    - Heat pump, ducted row (1.5 to 5 tons): below 24,787 (MP3), 21,120 (MP4)
      and 18,201 (MP5); above 31,510, 31,782 and 25,662. Non-ducted row
      (2022.1.1 only, 24,853 rdu): above 5 tons 15,560 (MP3) and 10,515 (MP4),
      and MP4's SEER1 of 29.3 is above the row's 23 on every one.
    - Backup furnace of the dual-fuel package (gas furnace row): 20,795 below;
      2,992 above.
10. For a dual-fuel heat pump (MP5), the split between heat-pump electricity
    and backup-furnace gas is fixed by ResStock's own base-year hours above and
    below the 35 F switchover. Degree-day factors scale each fuel by the same
    amount, so a projected year's total heating load moves but the split does
    not (`get_degree_day_adjusted_consumption_by_fuel`, Step 2 comment in
    `degree_day_consumption_utils.py`). The split is exact only in `ANCHOR_YEAR`.
11. Homes with no central or room AC are excluded from the study sample
    (38,910 rdu, 9.42M homes in 2022.1.1). So are homes with a shared cooling
    system (96 rdu, 23,245 homes), which has no replacement cost data. For a
    home with no AC the heat pump adds cooling the home never had, so any
    cooling cost is for a new service, not a change to an existing one; the
    same is partly true for room-AC homes, whose heat pump cools the whole
    house. Keeping no-AC homes in with cooling set to zero did not make sense.
    PLACEHOLDER: a future switch would bring no-AC homes back for a
    colleague's use case, at the points tagged 'TODO (no-AC homes)' in the
    code: energy as ResStock publishes it (0 kWh of cooling before the
    retrofit), cooling replacement cost and credit $0 on purpose, and cooling
    savings left negative, because the added cooling is a cost the home pays.
    See the cooling-list comment in `constants.py`.
12. Five sample rdu (1,211 homes) have blank climate emissions and damages.
    Their county (FIPS 46102) is not in the county-to-GEA-region crosswalk
    (`process_euss_data.py`, STEP 1b). Accepted: where data is missing the
    value is left blank, not guessed.
13. Some counties rest on very few sample rdu. With `MIN_HOME_COUNT = 1`
    (`constants.py`), 74 of the 3,079 study-sample counties have 1 rdu and 280
    have 3 or fewer, so one rdu can set a county's value. Check a county's rdu
    count before quoting its value.
14. Two costs of the dual-fuel package (2025.1, MP5) rest on choices of ours.
    (a) Its heat pump is rated in SEER2 (15.2), but the REMDB cost regression
    takes SEER1, so it is priced at SEER1 16.0 (SEER1 = SEER2 / 0.95,
    `SEER2_PER_SEER1` in `constants.py`), about $754 more per home than at
    15.2. That factor, and HSPF1 = HSPF2 / 0.85, are an assumption: a
    reasonable and standard one for the vast majority of residential
    installations, ducted split systems in particular, the kind of system the
    package installs. HSPF1 is kept for the record and prices nothing.
    (b) Its backup gas furnace is priced with the REMDB `furnaces_gas_furnace`
    row, at the backup's own size and rated AFUE (0.925 or 0.95), and added to
    the capital cost (national mean $4,112 per home). Only a gas backup can
    be priced; a propane or fuel-oil backup stops the run.
15. The credit for an old heating system is priced on the REMDB row that fits
    its equipment type first, then its fuel (9 Oct 2026). Where the table has
    no row for the fuel, the gas row of that type stands in:
    - a propane or fuel oil furnace on the gas furnace row
      (`furnaces_gas_furnace`): 11,594 and 9,888 rdu in 2022.1.1, 410 and
      1,270 in 2025.1;
    - a propane boiler on the non-condensing gas boiler row
      (`boiler_gas_non_condensing`), with the gas boilers: 655 rdu in 2022.1.1
      and 13 in 2025.1. A fuel oil boiler has a row of its own (`boiler_oil`).

    Every old boiler is priced as a non-condensing unit, at AFUE 0.80 or at
    its published value if that is higher. The condensing boiler row is not
    used. Until 9 Oct 2026 every fossil system was priced on the gas furnace
    row.
16. An old electric furnace or electric boiler is priced as electric
    baseboard (`electric_baseboard_default`), because REMDB has no row for
    either: 47,155 and 265 rdu in 2022.1.1, 7,355 and 48 in 2025.1. The
    efficiency terms of the rows that could price them do not line up. The
    baseboard row has none: its price is a straight line through zero in size
    alone. The gas furnace and gas boiler rows fit the equipment better, but
    they price AFUE as burner efficiency (about $4,570 and $818 per 1.00 of
    AFUE, 2025 dollars), and ResStock publishes these systems at 100% AFUE,
    above both rows' ranges (0.97 and 0.87). Moving them to the gas rows at a
    fixed AFUE of 0.80 was considered and not done (9 Oct 2026). Homes that
    heat with electricity are a large share of adopters, so this credit
    matters to the headline numbers.
    TODO: revisit after asking NLR's REMDB researchers for advice (the TODO in
    `_assign_replacement_row_id`, `remdb_v4_installed_cost_utils.py`).
17. The credit for an old room AC has no efficiency floor. A central AC credit
    is priced at SEER1 15 at least. A room AC credit is priced at its
    published value, which is below the row's lowest value (9.4) for 2,768 rdu
    in 2022.1.1 and 325 in 2025.1. The row takes CEER, and the value given to
    it is ResStock's EER. The effect is small: about $6 for one point.
18. No electric panel upgrade is assumed or priced. ResStock 2025.1 publishes
    panel columns (18,020 MP5 sample rdu are capacity constrained), and REMDB
    2024 has panel rows that the cost table leaves out. Two heat pump rows of
    the cost table are never used:
    `air_source_heat_pump_centrally_ducted_with_new_circuit` and
    `air_source_heat_pump_non_ducted_single_zone`. Every home without ducts is
    priced as a multi-zone system (24,853 rdu in 2022.1.1), and no home is
    priced with a new circuit.

---

## Masking and Validation Rules

**Heating:** `include_heating = valid_fuel_heating AND valid_tech_heating`
- `valid_fuel_heating`: fuel is one of {Electricity, Natural Gas, Propane, Fuel Oil}
- `valid_tech_heating`: technology is one of `ALLOWED_TECHNOLOGIES['heating']`
- Homes with `in.heating_fuel = 'None'` are automatically excluded

**Cooling:** `include_cooling = valid_fuel_cooling AND valid_tech_cooling`
- `valid_fuel_cooling`: hardcoded True (cooling is always electric) — this flag is a no-op
- `valid_tech_cooling`: technology is one of {Central AC, Room AC}, and the system is not shared between homes (`SHARED_COOLING_EFFICIENCY` in `constants.py`)
- Homes with no AC (`'None'`) or evaporative coolers are excluded here, and so are left out of the study sample (Limitation 11)

**Cooling in NPV:** every home in the study sample has an AC of its own, so cooling savings and the cooling replacement credit apply to every home in the NPV. Nothing is set to $0 by `include_cooling`. A blank energy value or a blank cooling replacement cost in a sample home stops the run; it is not counted as zero.

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
- **How common it is (measure-package specific):** for MP3, 92.94% of Room AC
  baselines go negative, vs. 13.32% of Central AC baselines. For MP4, it's
  66.17% vs. 2.29%. Measured 5 Oct 2026 on run `2026-10-05_00-54`, over the
  study sample (43,564 Room AC rdu, 177,641 Central AC rdu). The 19 Aug 2026
  figures (90.68% / 10.84% and 61.97% / 3.46%) were on the old population
  and counted primary cooling energy only.
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

**Consumption counts every component (22 Sep 2026).** Heating and cooling energy
include primary energy, fans and pumps, and heat-pump backup, on both the baseline and
retrofit side. The parts are read from the ResStock file itself
(`find_enduse_columns` in `calculation_utils.py`), not from a fixed list. Earlier versions
counted primary energy only, which dropped MP3/MP4's electric backup heat and every
system's fans. `build_projected_consumption` (`projected_consumption.py`) stores each
home's use by fuel and year (degree-day adjusted, 2025-2039) and is the single source
for fuel costs (each fuel at its own price), climate emissions, and the savings
fraction. The supplemental fuel-cost CSV carries its per-fuel columns.

**The home table is not rounded during the run (5 Oct 2026).** No step rounds
the whole table, so ResStock's values are saved as published (`weight` is
242.131013), and the HOMES savings tiers and the 80% / 150% income cut-offs
are tested on unrounded values. Rounded on purpose, to cents: the NPV (so a
leftover such as -0.0000000001 cannot fail `NPV >= 0`), installed costs, rebate
amounts, household income, and area median income. Do not add a whole-table
`.round()` to make output readable; round only when printing.

---

## NPV Ordering Checks (enforce in verification)

Per home (every home in the study sample has an AC of its own):
- `heatingLCC_coolingLCC >= heatingLCC_coolingSavings` (adds avoided cooling replacement >= 0)
- `heatingLCC_coolingLCC >= heatingSavings_coolingLCC` (adds avoided heating replacement >= 0)
- No general ordering between `heatingLCC_coolingSavings` and `heatingSavings_coolingLCC`
  (depends on relative magnitude of heating vs cooling replacement costs)

Homes with no AC are not in the sample (Limitation 11). If a future switch
brings them back, their cooling replacement credit is $0, so for them
`heatingLCC_coolingLCC` == `heatingLCC_coolingSavings`.

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
❌ Set cooling savings or the cooling replacement credit to $0 by include_cooling, or count a blank energy value or cost as zero — every sample home has its own AC, and a blank in a sample home stops the run
❌ Round the home table, or a value tested against a cut-off, in the middle of the run — round only when printing
❌ Collapse three NPV cases into one combined value
❌ Derive operating-cost % from ratio formula — always use (new - old) / old * 100 on per-home cols
❌ Route adoption share through pct_change — it is a share (0–100%), not a percent change
❌ Delete the tiered adoption module — prepend deprecation header only
❌ Generate econ adopter columns inside a loop — generate all per MP in a single block
❌ Change a notebook cell any way other than the notebook-cell-edit skill — no retyped cells, no whole-cell replacement, no one-off JSON edits
❌ Edit any *_EXPORT_*.py file — it is a read-only snapshot of a notebook; change the modules or the notebook itself
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
