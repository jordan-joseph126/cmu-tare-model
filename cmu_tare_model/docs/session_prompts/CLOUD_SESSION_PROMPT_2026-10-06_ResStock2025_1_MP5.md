# TARE Model -- ResStock 2025.1 Dual Fuel (mp=5) Overnight Cloud Session Prompt

CLAUDE.md is the source of truth for conventions, file rules, golden values, coding standards,
and all non-negotiables; it is auto-loaded, so this prompt only adds session-specific state and
the task list. Where this prompt and CLAUDE.md conflict, the override below decides.

## Autonomy override

> **Autonomy override (session-specific; per CLAUDE.md, session prompts take precedence where
> they conflict).** The researcher has explicitly authorized this session to run without
> per-diff approval. On the cloud branch only (`cloud/resstock2025-1-mp5`, or a `claude/...`
> branch the harness created from it), you may: edit any editable module, add modules, tests,
> scripts and docs, run anything, commit after each verified step with a descriptive message,
> and push that branch. This replaces CLAUDE.md's one-edit-per-stop-gate rule and its
> no-commit rule for this branch only. Still in force: never edit
> `utils/validation_framework.py`, any `.ipynb`, or any `*_EXPORT_*.py`; never push to, merge
> into, rebase, or reset `resstock2025-dual-fuel-codebase-update` or `main`; never force-push;
> never delete or rewrite branches; never overwrite or edit existing rows in
> `REFERENCE_VALUES.md`; never commit `data/resstock_2025_1/`, any parquet file, or anything
> under `output_results/`; all coding standards, naming rules, and anti-patterns in CLAUDE.md
> still apply. Do not stop to ask questions -- record the question and the default you took in
> `DECISIONS_TAKEN.md` and continue.

**This session runs unattended overnight. It must never wait for input.** No one will answer a
question, approve a diff, or respond to a prompt. Never call a tool that asks the researcher
something, never run a command that reads from the keyboard (`input()`, an interactive git
command, a pager), and never end a turn to wait. When a choice is needed, take the default named
in this prompt, or the refactoring guide's recommended default, write it to
`cmu_tare_model/docs/cloud_run/DECISIONS_TAKEN.md`, and keep going.

Also in force for every commit: no file over 50 MB, no parquet file, nothing under
`cmu_tare_model/data/euss_data/`, `data/resstock_2025_1/`, or any `output_results/` folder.

## Context

**Priority: by morning, the full pipeline runs end to end, nationally, for the ResStock 2025.1
Dual Fuel Heating System package (Upgrade 05, `mp=5`), from a new driver script, with the three
deliverables below committed and pushed.** The researcher will then port your commits to the
development branch `resstock2025-dual-fuel-codebase-update` by hand, one gated diff at a time,
using the implementation guide you write. So small, single-purpose commits with honest messages
matter more than speed.

The work is narrower than the plan documents suggest. An audit and a probe run on 6 Oct 2026
found that the pipeline already runs for `mp=5` from load through NPV and adoption, with correct
two-fuel energy, fuel costs and emissions. What remains is listed under "Gap list" below: mostly
capital cost (Phase 5), one rebate rule (Phase 7), the demand and county-table part of the KPIs
(Phase 10), and the driver itself.

Plan documents, all in this clone:

- `REFACTORING_GUIDE_17Sept2026_DualFuel_ResStock2025_1.md` (repo root) -- the guide. Its
  Section 1 lists the output families, Section 4 the phases, Section 5 the silent traps.
- `cmu_tare_model/docs/NEXT_STEPS_2026-09-19_DualFuel_ResStock2025_1.md` -- decisions D1-D10.
- `cmu_tare_model/docs/AUDIT_2026-09-18_ResStock2025_1_DualFuel.md` -- Phase 1 measurements.
- `cmu_tare_model/docs/plan/PHASE2_SESSION_PROMPT_ResStock2025_1_DualFuel.md` -- Phase 2 prompt.
- `cmu_tare_model/docs/resstock_2025_1_column_map.csv` -- the column map the loader reads.

No Phase 3 or Phase 4 prompt file exists. Those phases were carried out in later sessions; the
code is the record. Where a plan document disagrees with the code or with CLAUDE.md, the code
and CLAUDE.md are newer and win. The cases that matter are named below.

## Current state the tasks depend on

### Branches

- This clone is `cloud/resstock2025-1-mp5`. It is the dev branch at `b17d7af` plus cloud-only
  commits: `5ae9051` (small model inputs under `cmu_tare_model/data/`), `62aa53e` (lock file
  removed), and the commit that added this prompt, the Phase 2 prompt copy and
  `.claude/settings.json`. **None of those cloud-only commits is ever ported.**
- All line numbers below are for the module code at `b17d7af`, which this branch has unchanged.
- Never pushed, so absent here: `cmu_tare_model/data/euss_data/` (ResStock 2022.1.1, 14 GB),
  `data/resstock_2025_1/` (you download it, Task 1), `output_results/`, `.bsq_cache/`,
  `DATA_MANIFEST.md`, the `HANDOFF_*.md` notes, and `archived_files/`.
- **ResStock 2022.1.1 cannot run in this session** (no data). Leave its code path working by
  default: do not break it on purpose, do not try to verify it. NEXT_STEPS dropped the
  2022.1.1 regression guarantee.

### The cloud machine

- Ubuntu 24.04, 4 vCPU, 16 GB RAM, 30 GB disk. Python 3.x with pip and uv is installed.
- Reachable: `*.amazonaws.com` (OEDI S3 over https), PyPI, conda, GitHub for this repository.
  Not reachable: most other sites, including census.gov and ecfr.gov. Do not plan on web
  lookups; everything needed is in this prompt or the clone.
- A foreground command is cut off at about 10 minutes. Anything that may run longer goes to the
  background with a log file, and you poll the log.

### What the 6 Oct 2026 probe measured (dev code as it stands, before any fix)

The probe followed the notebooks' call order on ResStock 2025.1 with `mp=5`, for PA, MN, FL and
the whole country. **Treat these as PROVISIONAL sanity numbers, not reference values.** Your
driver, before you change any model code, should reproduce the PA row exactly.

| Scope | Study sample (rdu) | Homes | Core run time | Peak memory |
|---|---:|---:|---:|---:|
| PA | 6,469 | 1,642,503 | 11 s load through checks | 2.4 GB (column-limited read) |
| MN | 4,410 | 1,119,715 | | |
| FL | 2,177 | 552,748 | | |
| National | 161,983 | 41,128,079 | 134 s load through checks | 6.1 GB |

- Weight read from the frame: `253.90367272727272` per rdu, one value.
- National funnel for `mp=5` (rdu): stock 549,971 -> package applies 245,570 -> occupied
  221,752 -> single-family 190,630 -> not AK/HI 190,374 -> heating fuel 190,374 -> no existing
  heat pump 180,693 -> replaceable heating system 177,000 -> central or room AC 161,983 -> not
  shared cooling 161,983.
- National sample by baseline fuel (rdu): Natural Gas 151,359; Electricity 8,778; Fuel Oil
  1,423; Propane 423. By cooling: Central AC 145,871; Room AC 16,112. No sample rdu lacks a GEA
  region.
- National heating energy in 2025 (weighted GWh): baseline 882,119 (natural gas 829,286);
  after the retrofit 624,652, of which electricity 116,668 and natural gas backup 507,984.
  Propane and fuel-oil baselines switch their backup to natural gas, so those two retrofit
  columns are zero.
- With the code as it stands (ASHP cost only, SEER 15.2 unconverted, June 2026 HEEHR fuel gate
  applied to dual fuel), national means over the sample: heat-pump upgrade cost $14,957;
  heating replacement cost $3,742; cooling replacement cost $5,843; lifetime heating fuel
  savings $1,072; lifetime cooling fuel savings $1,353. National adoption (rdu adopters / 
  161,983): `heatingLCC_coolingLCC_unsub` 11.19%, `_sub` 60.23%, `_sub_june2026` 12.42%.
  **These will move when you fix the gaps below; that is expected.**
- Checks that already pass, per home, in all four scopes: NPV identity 0 violations at the
  half-cent; both NPV ordering checks 0 violations; every adopter flag equals `NPV >= 0`;
  the nine adopter columns are float64 and each has exactly the study-sample count of
  non-blank rows.
- Full national pass including CSV export (188 s), reload, county tables, maps, dot plot and
  the Tepper household files: 392 s, peak 6.5 GB. The national run fits this machine easily
  **only with the column-limited read** (next item).

### Memory: the loader must read fewer columns

`read_resstock_2025_1_parquet` (`process_euss_data.py:71-99`) reads all 770 columns. Measured:
7.5 GB after the read and 10.6 GB at peak for one national file; `load_and_filter_upgrade`
(`:132-269`) then copies the frame at each filter. The notebook's sequence (baseline frame held
while the package file loads) needs about 16.7 GB, which does not fit 16 GB.

Reading only the columns the pipeline uses fixes it: 141 of 770 columns, both national files
held at once in 2.3 GB, and every PA result identical to the full read. The column list is:

- `bldg_id`, `weight`, `in.hvac_has_ducts`;
- every physical name in `RESSTOCK_COLUMN_MAP['2025.1']` (`utils/resstock_schema.py`) that the
  file contains;
- every heating and cooling energy column `find_enduse_columns` (`calculation_utils.py:134`)
  finds in the file's schema, plus `get_resstock_savings_column` of each;
- every `...energy_savings..kwh` column whose name contains `.total.`, `.hot_water` or
  `.refrigerator.` (the savings check in `check_savings_against_resstock` reads them).

Read the schema with `pyarrow.parquet.read_schema` and pass `columns=` to
`pyarrow.parquet.read_table`. OEDI also publishes per-state files, a usable fallback that should
not be needed: `.../metadata_and_annual_results/by_state/full/parquet/state=PA/PA_upgrade5.parquet`
(34 MB; the baseline is `PA_upgrade0.parquet`).

### Phase status (file:line evidence)

| Phase | Status | Evidence |
|---|---|---|
| 2 Release-aware loader | DONE except two leftovers | Registry `resstock_schema.py:72-97`; constants `constants.py:109-120`; weight from the frame `process_euss_data.py:545`; MP3 override guarded by release `:1029`, `:1057`; gas backup and fan parts read from the file `:660-668`, `:1153-1167`; backup furnace size `:1009`; panel flags `:598-600`, `:1228-1234`. Leftovers: peak columns (G4) and the unused parser (G5). |
| 3 Masking | DONE | Applicability is the first filter `process_euss_data.py:233-239`; funnel `study_sample.py:68-134`; AK/HI excluded `constants.py:135`, `process_euss_data.py:253-257`; rebate-eligible packages by release `determine_rebate_eligibility_and_amount.py:461-492`. D4 was not done and is superseded (P2). |
| 4 Consumption | DONE | Per-fuel, every part: `degree_day_consumption_utils.py:434-501`; `projected_consumption.py:24-98`; savings fraction `process_euss_data.py:1301-1344`; checked against ResStock's own savings columns every run `:820-925`. |
| 5 Capital cost | NOT STARTED | G6, G7. |
| 6 Fuel costs | DONE | Each fuel at its own price `calculate_lifetime_fuel_costs.py:599-616`. |
| 7 Rebates | PARTIAL | `mp=5` is registered as eligible; the June 2026 HEEHR exception is missing (G8). |
| 8 NPV, adoption | Mechanics DONE | Needs only the furnace cost from G7 in net capital. |
| 9 Climate | DONE | Fossil use counted for any package `calculate_fossil_fuel_emissions.py:83-91`. |
| 10 KPIs, exports | PARTIAL | County adoption, maps, dot plot and Tepper household files run. Demand tables and the Tepper county file fail (G9, G10). |
| 11 Full run, docs | NOT STARTED | No 2025.1 row exists in `REFERENCE_VALUES.md`. Reference rows are out of scope here. |

### Gap list, in pipeline order

Each item: what is wrong, where, the phase or decision it belongs to, and a size (S / M / L).

**Driver and run setup**

- **G1 (M, new file).** No script runs the pipeline. The notebooks drive it with `%run` and
  keyboard prompts (`tare_baseline_v3_0.ipynb` cell 2, `tare_scenarios_v3_0.ipynb` cell 2), and
  notebooks cannot be edited. Add `scripts/run_resstock2025_1_mp5.py` (Task 2).
- **G2 (S, D1).** The release can only be chosen by editing `constants.py:109`
  (`RESSTOCK_RELEASE_THIS_RUN = '2022.1.1'`). `VALID_MENU_MPS` is derived from it at `:120`;
  functions copy it as a default argument when their module is imported
  (`process_euss_data.py:137`, `:481`, `:935`); `get_rebate_eligible_mps` reads it at
  `determine_rebate_eligibility_and_amount.py:485`; `_validate_inputs` in both cost modules
  checks `VALID_MENU_MPS`. So the release must be set before anything else is imported. See P3.
- **G3 (S, Phase 2).** Column-limited read, described above. Add it to the loader (an optional
  `columns` argument or a release-aware column list), not only to the driver, so it ports.

**Load**

- **G4 (S-M, Phase 2 Task 7, audit E.2).** The 2025.1 peak columns are stored under the 2022.1.1
  names: `base_peak_electricity_heating_kw` / `_cooling_kw` (`process_euss_data.py:700-705`) and
  the four `mp{mp}_peak_electricity_*` columns (`:1187-1198`). 2022.1.1 measures the peak while
  the equipment runs; 2025.1 measures the peak in the winter or summer months. They are not the
  same measurement, so NEXT_STEPS decided they must not share a name. Give the 2025.1 columns
  their own names built from ResStock's `winter` / `summer` (P7), leave the 2022.1.1 names and
  values alone, and carry the note into `cmu_tare_model/utils/export_tepper_csv.py` and
  `cmu_tare_model/docs/tare_tepper_exports_data_dictionary.md`.
- **G5 (S, Phase 2 Task 5).** `parse_dual_fuel_heating_efficiency` (`process_euss_data.py:295-353`)
  has no caller, so no `hp_seer2` or `backup_afue` value reaches the frame. G6 and G7 need both.
- Applicability is coerced with `.astype(bool)` (`:237`, `:1247`). The published files hold a
  true boolean, so this is right today, but a text column would turn `"False"` into True. Assert
  the dtype is boolean in the loader (S).

**Capital cost (Phase 5)**

- **G6 (S, D9).** `_convert_pm2` (`remdb_v4_installed_cost_utils.py:292-347`, the extract at
  `:320`) takes the first number in the efficiency string. For
  `Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, ...` it feeds 15.2 to a regression fitted on SEER1.
  The probe saw 15.2 on all 161,983 sample rdu. It raises no error because 15.2 is inside the
  regression's bounds. Add a named, documented SEER2 -> SEER1 conversion at this boundary (P1).
- **G7 (M-L, D6).** The new condensing gas furnace has no cost. `_assign_upgrade_row_id`
  (`:146-187`) knows only the two heat-pump rows; three `add_remdb_metrics` calls are made per
  package, never a fourth; `calculate_capital_costs`
  (`calculate_lifetime_private_impact.py:527-655`) adds one upgrade cost. Needed: a furnace
  upgrade branch using the REMDB row `furnaces_gas_furnace` (pm1 = heating capacity in BTU/hr
  from `size_heat_pump_backup_primary_k_btu_h`, pm2 = the backup AFUE as a fraction, 0.925 or
  0.95, from G5); a fourth cost call that runs only for a dual-fuel package; a new installed-cost
  column (P8); and that cost added to total and net capital for `mp=5` only. The backup size is
  positive on every sample rdu (mean 63.2 kBtu/h); 85% are inside REMDB's 30-156 kBtu/h range,
  the rest are priced as converted (CLAUDE.md Limitation 9). See P4 and P5.
- `cmu_tare_model/utils/validate_capital_costs.py` hard-codes `mp3_` and `mp4_` (`:1002-1003`, `:1279`). It is
  an MP3-versus-MP4 comparison and is not part of this session. Skip it.

**Rebates (Phase 7)**

- **G8 (S-M, D8).** `calculate_rebate_program`
  (`determine_rebate_eligibility_and_amount.py:604-620`) applies the June 2026 HEEHR fuel gate to
  the baseline fuel with no exception for a dual-fuel package. Nationally 153,205 fossil-baseline
  sample rdu get $0 under June 2026, while the 2024 rules pay them (mean HEEHR $7,476 for
  natural-gas baselines). D8: a dual-fuel retrofit passes the June 2026 HEEHR fuel gate
  whatever the baseline fuel. Put the exception in configuration (`REBATE_RULE_CONFIG` or a
  release-and-package set beside `get_rebate_eligible_mps`), not in an ad hoc `if`. See P5, P6.
- Three checks assume fossil baselines get nothing under June 2026 and will be wrong for `mp=5`
  once G8 lands: the dot plot's rule 2 (`visuals_adoption_dotplot.py:280`; pass
  `check_june2026_fossil_rule=False` through `build_df_kwargs`);
  `scripts/verify_june2026_rebate_fossil_gate.py` (`_MP = 4` at `:30`); and the assertion in
  `tare_scenarios_v3_0.ipynb` cell 29, which you cannot edit -- list it in the implementation
  guide as a notebook change for the researcher.

**KPIs and exports (Phase 10)**

- **G9 (M).** `adoption_kpis/data_loading.py` is 2022.1.1 only: `EUSS_DATA_DIR` (`:49-52`),
  `mp_to_upgrade` zero-pads (`:327`), `load_euss_baseline` (`:330-369`) and `load_euss_upgrade`
  (`:372-415`) read the 2022.1.1 CSV files. In this clone both raise `FileNotFoundError`. On the
  researcher's machine `load_euss_baseline()` would silently load the 2022.1.1 baseline during
  a 2025.1 run. The main notebook uses it for county weights. Make these loaders follow the
  release (reuse `load_and_filter_upgrade` and the column-limited read).
- **G10 (M).** The demand code names 2022.1.1 columns as literals (`data_loading.py:200-270`).
  `compute_scenario_demand` fails with `KeyError: 'out.electricity.total.energy_consumption.kwh'`
  at `adoption_kpis/demand.py:116` on 2025.1 frames. Route the names through `resstock_col` and
  `get_resstock_savings_column`. This blocks the county demand tables and the Tepper county file.
- The notebook's title table has no entry for package 5; the driver supplies its own titles.
  `visuals_adoption_dotplot.py:308` and `:523` default `scaling_factor` to `242.0`; it is used
  only when a frame has no `weight` column, but it is a 2022.1.1 number. Read the weight (S).
- Grid impact is deferred. `constants.py:483` pins the 2022.1.1 Athena table. The driver never
  calls the grid-impact code.

**Already correct -- do not "fix"**

- Two-fuel retrofit energy, per-fuel prices, post-retrofit gas emissions (Phases 4, 6, 9).
- `mp5_heating_consumption` (`process_euss_data.py:1139`) is the heat pump's own electricity
  only (national mean 2,301 kWh against 15,188 kWh for all heating energy). It is one part of
  the total by design. Never read it as the home's heating energy; use
  `mp5_heating_annual_consumption_kwh` or the projected per-fuel columns. The old readers
  `get_electricity_consumption_for_year` and `get_hdd_adjusted_consumption`
  (`degree_day_consumption_utils.py:281-366`) have no caller in model code.

### Known traps, as checked on 6 Oct 2026

| Trap | Status |
|---|---|
| Retrofit heating read as one fuel | Fixed in the live path (`degree_day_consumption_utils.py:434-501`). The old single-fuel readers remain, unused. |
| Retrofit fuel price fixed to electricity | Fixed (`calculate_lifetime_fuel_costs.py:599-616`). |
| No fossil emissions after a retrofit | Fixed (`calculate_fossil_fuel_emissions.py:83-91`). |
| SEER2 passed unconverted to the SEER1 regression | **Open** (G6). |
| `upgrade.heating_fuel` used on 2025.1 | Not used anywhere. Measured: `None` on every row, as is `upgrade.hvac_heating_type_and_fuel`. Never start using either. |
| A surviving `242.13` | None in model code. `242.0` survives as a default in the dot plot (above). Tests use `242.131013` as fixture data, which is fine. |
| `applicability` not coerced | Coerced at `process_euss_data.py:237` and `:1247`; add the dtype assertion. |

### Things that silently corrupt results (guide Section 5), with today's status

- Treating the retrofit as all-electric: fixed. Keep it fixed in everything you add (the
  furnace's gas is 81% of dual-fuel heating energy nationally).
- Ignoring `applicability`: fixed. Rows ResStock did not apply the package to are copies of the
  baseline and would show zero savings and zero cost.
- Loading 2025.1's weighted or intensity columns and weighting again: the column-limited read
  never loads them. Keep it that way.
- Mixing 2022.1.1 and 2025.1 package numbers in one frame: one release per run. 2025.1 Upgrades
  03 and 04 will later reuse `mp=3` and `mp=4`; every check on a package number also checks the
  release.
- Any "a count under 242 is rdu" reasoning: the 2025.1 weight is 253.90367272727272. Read it
  from the frame. In every count you report, label rdu and give homes as well.
- AK and HI: excluded on purpose (`EXCLUDED_STATES`). Their `in.county` values are name strings,
  not GISJOIN codes, and would produce a garbage county FIPS without raising.
- Connecticut: 2025.1 still uses the eight pre-2023 counties. No change.
- No-AC homes: excluded from the study sample by CLAUDE.md (Limitation 11). Do not bring them
  back and do not set cooling values to zero for them.

### Decisions D1-D10 as taken (NEXT_STEPS, 19 Sep 2026)

| # | Decision | State |
|---|---|---|
| D1 | `RESSTOCK_RELEASE_THIS_RUN` plus `RESSTOCK_RELEASE_AND_MP` | In code. |
| D2 | Upgrade 05 first, through every phase; 04 then 03 later | This session is 05 only. |
| D3 | `applicability` is the first filter; funnel table after each step | In code. |
| D4 | Remove the cooling-technology filter on both releases | **Superseded** by the study sample rule of 3-5 Oct 2026 (P2). |
| D5 | Retrofit fuels come from the file's own columns, not a per-package table | In code. |
| D6 | Furnace upgrade row added; the fourth cost call runs only for a dual-fuel retrofit | To do (G7). |
| D7 | No panel cost; carry the constraint flags | In code. |
| D8 | `mp=5` passes the June 2026 HEEHR fuel gate whatever the baseline fuel | To do (G8). |
| D9 | Apply the DOE Appendix M1 SEER2/HSPF2 conversion | To do (G6); factors provisional (P1). |
| D10 | Exclude AK and HI | In code. |

### PROVISIONAL decisions for this session

The researcher was not available to decide these. Each is the refactoring guide's recommended
default, or the smallest choice consistent with CLAUDE.md. **Record every one in
`DECISIONS_TAKEN.md` with how to reverse it**, and mark any result that depends on it.

- **P1 (D9 factors).** SEER1 = SEER2 / 0.95 and HSPF1 = HSPF2 / 0.85, the guide's placeholders.
  SEER2 15.2 becomes SEER1 16.0. The DOE Appendix M1 source was never checked (ecfr.gov is not
  reachable here either). Keep the two factors as named constants in one place. Only SEER feeds
  the cost regression; HSPF is converted for the record only. Note the audit's caution: the
  option string writes `SEER 15.2` while the measure documentation says SEER2 15.2.
- **P2 (D4).** Follow CLAUDE.md, not NEXT_STEPS: the cooling filter stays and the study sample is
  homes with a central or room AC of their own. For `mp=5` this leaves out 15,017 applicable rdu
  with no AC. Change no code for this.
- **P3 (release switch).** `constants.py` reads the release from the environment variable
  `TARE_RESSTOCK_RELEASE`, defaulting to `'2022.1.1'`, so it is still declared in one place and
  one run uses one release. The driver sets the variable before it imports anything from
  `cmu_tare_model`. Reject an unknown value with a `ValueError`.
- **P4 (furnace cost basis).** The REMDB `furnaces_gas_furnace` installed cost in full, sized on
  the backup size, with no discount for labor or ductwork shared with the heat pump, inflated to
  2025 dollars the same way as every other REMDB cost. This is the guide's D6 option (b).
- **P5 (rebate base).** The furnace cost is not part of the cost a rebate covers. HEEHR and HOMES
  keep reading the heat-pump upgrade cost column. A fossil furnace is not a rebate measure.
- **P6 (June 2026 HOMES).** Unchanged: still limited to electric baselines for byte-identity
  reasons CLAUDE.md records as deferred. D8 covers HEEHR only.
- **P7 (2025.1 peak names).** `base_peak_electricity_winter_kw`, `base_peak_electricity_summer_kw`,
  `mp{mp}_peak_electricity_winter_kw`, `mp{mp}_peak_electricity_summer_kw`, and the two savings
  columns with `_savings` appended.
- **P8 (furnace cost column).** Build it with `create_cost_col`, adding a cost type for the backup
  furnace, so the name follows the existing pattern
  (`mp{mp}_heating_<cost type>_installed_cost_{cost scenario}`). Never hard-code `mp5`.
- **P9 (adoption denominator).** The definition of done below says the denominator equals the
  study-sample count, `include_sample`. The guide's Phase 8 wording says `include_heating`; it
  predates the study sample rule, and CLAUDE.md says every computation filters on
  `include_sample`. Report both counts (national today: 161,983 and 177,000).
- **P10 (loop states).** PA, MN (cold) and FL (warm), because the probe measured all three.

## Tasks

### Task 1 -- Setup

1. **Install the environment inside this session.** Do this even if a prepared environment
   seems to exist. Run from the repo root:

   ```bash
   if [ -x /opt/tare-venv/bin/python ]; then VENV=/opt/tare-venv; else VENV="$HOME/tare-venv"; fi
   PINNED="pandas==2.1.4 numpy==1.26.4 pyarrow==23.0.1 scipy==1.16.0 matplotlib==3.10.0 \
   seaborn==0.13.2 geopandas==1.1.3 pyogrio==0.12.1 shapely==2.1.2 pyproj==3.7.2 \
   openpyxl==3.1.5 requests psutil pytest"
   LOOSE="pandas>=2.1,<2.3 numpy<2.3 pyarrow scipy matplotlib seaborn geopandas pyogrio \
   shapely pyproj openpyxl requests psutil pytest"
   if [ ! -x "$VENV/bin/python" ]; then
     uv venv --seed --python python3 "$VENV" || python3 -m venv "$VENV"
   fi
   uv pip install --python "$VENV/bin/python" $PINNED \
     || uv pip install --python "$VENV/bin/python" $LOOSE \
     || "$VENV/bin/python" -m pip install $LOOSE
   uv pip install --python "$VENV/bin/python" -e . || "$VENV/bin/python" -m pip install -e .
   "$VENV/bin/python" -c "import config, cmu_tare_model, pandas, pyarrow, openpyxl, geopandas; print(pandas.__version__)"
   ```

   The researcher's environment is Python 3.11.13 with the pinned versions. If the pinned set
   will not install on this machine's Python, the loose set is acceptable; write the versions
   you ended up with at the top of `BREAKAGE_LOG.md`. If the editable install fails, run
   everything with `PYTHONPATH=.` from the repo root instead (`config.py` is at the root).
   `environment-cmu-tare-model.yml` and `requirements.txt` are Windows-only snapshots; do not
   use them. `buildstock_query`, `boto3` and `toml` are needed only by the deferred grid-impact
   code; do not install them. Use `"$VENV/bin/python"` for every command below.

2. **Download the five ResStock 2025.1 files** into `data/resstock_2025_1/` (repo root; the
   whole `data/` folder is git-ignored). Run it in the background with a log and poll it:

   ```bash
   mkdir -p data/resstock_2025_1 && cd data/resstock_2025_1
   ROOT="https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1"
   nohup bash -c "
     curl -fsSL --retry 5 --retry-delay 10 -o upgrade0.parquet '$ROOT/metadata_and_annual_results/national/full/parquet/upgrade0.parquet'
     curl -fsSL --retry 5 --retry-delay 10 -o upgrade5.parquet '$ROOT/metadata_and_annual_results/national/full/parquet/upgrade5.parquet'
     curl -fsSL --retry 5 -o data_dictionary.tsv '$ROOT/data_dictionary.tsv'
     curl -fsSL --retry 5 -o enumeration_dictionary.tsv '$ROOT/enumeration_dictionary.tsv'
     curl -fsSL --retry 5 -o upgrades_lookup.json '$ROOT/upgrades_lookup.json'
     echo DOWNLOAD_FINISHED
   " > download.log 2>&1 &
   cd ../..
   ```

   | File (in `data/resstock_2025_1/`) | Bytes | SHA-256 |
   |---|---:|---|
   | `upgrade0.parquet` (baseline) | 465,060,991 | `83a288b30b5a83d0b5aded3102acae052c554c587cdb5f7293816325b9a48e06` |
   | `upgrade5.parquet` (dual fuel) | 691,778,814 | `faf42ad50847f5d7ef658fea67c3f428a193d26a97ec10298725a62855a87635` |
   | `data_dictionary.tsv` | 124,809 | `394b42c6be3dccb8b748993ed8ae7904a523d2dd1ead2b2b22df1b46bef2c568` |
   | `enumeration_dictionary.tsv` | 6,099,126 | `a60a0d216a309945f4451bca006d1490e8442fb1de7e7729842697a982677f7a` |
   | `upgrades_lookup.json` | 2,393 | `7939dd585612d4985db489733f99780e9f71c2ab562a79e9a4e13174cf936fc4` |

   All five URLs answered HTTP 200 with these sizes on 6 Oct 2026. When the log shows
   `DOWNLOAD_FINISHED`, verify every hash:

   ```bash
   cd data/resstock_2025_1 && sha256sum -c - <<'EOF'
   83a288b30b5a83d0b5aded3102acae052c554c587cdb5f7293816325b9a48e06  upgrade0.parquet
   faf42ad50847f5d7ef658fea67c3f428a193d26a97ec10298725a62855a87635  upgrade5.parquet
   394b42c6be3dccb8b748993ed8ae7904a523d2dd1ead2b2b22df1b46bef2c568  data_dictionary.tsv
   a60a0d216a309945f4451bca006d1490e8442fb1de7e7729842697a982677f7a  enumeration_dictionary.tsv
   7939dd585612d4985db489733f99780e9f71c2ab562a79e9a4e13174cf936fc4  upgrades_lookup.json
   EOF
   ```

   **A hash mismatch is the only thing that stops setup.** Download that file once more; if it
   still mismatches, log it as BLOCKED with both hashes, finish the three deliverables with what
   you have, push, and end. Only the two parquet files are opened by the pipeline; the other
   three are for looking up column names and values. `enumeration_dictionary.tsv` must be read
   with `encoding='latin-1'`. Each parquet file has 549,971 rows and one distinct `weight`.

3. **Record the test baseline.** `"$VENV/bin/python" -m pytest -q` from the repo root. A clean
   checkout of this branch gave 352 passed, 1 skipped (and 5 warnings that tests in
   `test_efficiency_floor_refactoring.py` return a value; leave those tests alone). Write your
   counts into `BREAKAGE_LOG.md` as the session baseline. If they differ from 352 / 1 before you
   change anything, record why (most likely a package version) and carry on.

4. Create `cmu_tare_model/docs/cloud_run/` with the three deliverable files started, and commit.

### Task 2 -- The driver and the iteration loop

Add `scripts/run_resstock2025_1_mp5.py`. It runs the whole `mp=5` pipeline with no notebook and
no keyboard input.

- Options: `--states` (one or more two-letter codes, or `National`) and `--out-dir` (default a
  folder outside the repo, for example `/tmp/tare_out`; never a tracked path). Add `--skip-export`
  if it speeds up the loop.
- It sets `TARE_RESSTOCK_RELEASE=2025.1` (P3) before importing `cmu_tare_model`, and uses the
  column-limited read (G3).
- It follows the notebooks' call order. Read the three run notebooks as text for the exact
  arguments (`cmu_tare_model/model_scenarios/tare_baseline_v3_0.ipynb`,
  `tare_scenarios_v3_0.ipynb`, `tare_run_simulation_v3_0.ipynb`) and
  `cmu_tare_model/tare_model_main_v3_0.ipynb` for the KPI and export steps:
  1. `load_and_filter_upgrade(menu_mp=0, ...)` and `(menu_mp=5, ...)`; keep the package's
     `bldg_id` index for the sample.
  2. `df_enduse_refactored(df_baseline, [package_ids], release=...)`, then
     `build_tare_sample_ids` and `build_sample_funnel`. (`df_enduse_refactored` changes its input
     frame in place.)
  3. Baseline `calculate_lifetime_climate_impacts(menu_mp=0)` and
     `calculate_lifetime_fuel_costs(menu_mp=0)`; each returns the home table and a detail table.
  4. `df_enduse_compare(df_mp, 'upgrade5', 5, df_baseline_home, release=...)`.
  5. Package climate impacts (with `df_baseline_damages`) and fuel costs (with
     `df_baseline_costs`).
  6. Capital cost: `load_remdb_v4_data`, then `add_remdb_metrics` followed by the matching
     installed-cost function for heating replacement, heating upgrade and cooling replacement --
     and the backup furnace once G7 lands. Copy the final cost columns onto the home table.
  7. `calculate_percent_AMI`, `prepare_discount_rates`, then `calculate_rebate_program` for each
     guidance in `REBATE_POLICY_SCENARIOS`.
  8. `calculate_private_npv`, then `economic_adoption_decision`
     (`discount_rate_col_name='private_discount_rate_fixed_base'`, `cost_scenario='v4MID'`).
  9. Checks (below), then `export_model_run_output` for the same result categories the
     simulation notebook writes, the sample funnel CSV, the county adoption and operating-cost
     tables, the county demand tables (after G9, G10), the maps and dot plot with the
     non-interactive `Agg` backend and figures saved under `--out-dir`, and the Tepper household
     and county files.
- It prints, for every stage, the wall time and the process's peak memory, and it runs the checks
  and exits non-zero if any fails: NPV identity (NPV = discounted heating savings + discounted
  cooling savings - net capital cost, to the half-cent, for all nine cases); CLAUDE.md's two
  per-home NPV ordering checks and the two county adoption-rate checks; each adopter flag equals
  `NPV >= 0`; each adopter column's non-blank count equals the `include_sample` count; the weight
  has one distinct value; no blank energy, cost or NPV value in a sample home.
- Follow CLAUDE.md's coding standards in the script and in every module you touch: Google-style
  docstrings, type hints, comments that say why, ASCII only, no `mp5` or `ref2025_mp5_` literals,
  88-character lines, no one-letter names.

**First run:** before changing any model code beyond P3 and G3, run the driver on PA and confirm
6,469 sample rdu, 1,642,503 homes, zero check failures. Commit the driver.

**Then loop:**

1. Run the driver on PA, MN and FL.
2. Take the first failure or the next gap in pipeline order: G4, G5, G6, G7, G8, G9, G10, then
   the remaining small items.
3. Fix it in the module where the fault is. Smallest change that is right for both releases.
4. Add or extend a test under `cmu_tare_model/tests/` (test data is made up in the test; no test
   may need the downloaded files).
5. Run the test suite. No regression against the Task 1 baseline.
6. Commit that one change with a message that says what it does and whether a modeled value
   moves. Push.
7. Append to `BREAKAGE_LOG.md`.

Run the national pipeline at each phase boundary: after capital cost (G6, G7), after rebates
(G8), after the KPIs (G9, G10), and at the end. Run it in the background with a log and poll.
Any command expected to take longer than about 10 minutes goes to the background the same way.

If the same failure survives three distinct fix attempts, write it in `BREAKAGE_LOG.md` as
**BLOCKED** with the traceback and what you tried, and move to the next stage that does not
depend on it.

Things each fix must show:

- **G6:** the value fed to the heat-pump regression is 16.0 for every sample home, the conversion
  is one named function, and a test covers it. State how much the mean upgrade cost moved.
- **G7:** a furnace cost exists for every sample home with no blank; total and net capital include
  it for `mp=5`; the NPV identity still holds; no MP3 or MP4 code path calls the furnace cost.
  Report the mean furnace cost and the change in adoption.
- **G8:** under June 2026, fossil-baseline dual-fuel homes at or below 150% AMI in a
  participating state get HEEHR; South Dakota still gets $0; the 2024 columns do not change.
  Give the program-by-fuel table before and after.
- **G9, G10:** county home counts in the adoption table and the demand table agree within one
  home's weight in every county, as the notebook's own export cell requires.

### Task 3 -- Definition of done

- The driver completes national `mp=5`, producing every output family in guide Section 1 except
  grid impact (deferred) and reference-value rows (out of scope; see the deliverables instead).
- The NPV identity holds to the half-cent (0 violations), and CLAUDE.md's NPV ordering checks
  show 0 violations.
- The adoption denominator equals the study-sample count (`include_sample`); see P9. Report the
  `include_heating` count beside it.
- There are no test regressions against the setup baseline.
- Key results are in `cmu_tare_model/docs/cloud_run/PROVISIONAL_RESULTS_2026-10-07.md`, with
  counts in rdu and homes, using the weight read from the frame. They are marked PROVISIONAL,
  not reference values. Include: the sample funnel; the sample by baseline fuel and cooling
  type; mean baseline and retrofit fuel costs and savings for heating and cooling; mean costs
  (heat pump, furnace, both replacements); rebate dollars and counts by program and baseline
  fuel for both guidance years; the nine NPV means and nine adoption rates; climate means; the
  share of homes with negative heating and negative cooling savings; run time and peak memory;
  and the same headline numbers from the first run before any fix, so the effect of each fix
  is visible.

### Task 4 -- Required deliverables, committed under `cmu_tare_model/docs/cloud_run/`

- `BREAKAGE_LOG.md`: the package versions installed, the test baseline, then every failure, its
  cause, the fix, and the commit SHA. Include BLOCKED items.
- `DECISIONS_TAKEN.md`: every judgment call where NEXT_STEPS gives no decision -- P1 to P10 and
  any you add -- the default taken, why, and how to reverse it.
- `IMPLEMENTATION_GUIDE_2026-10-07.md`: an ordered, step-by-step port plan for the dev branch,
  one task per gated diff. Each task gives the cloud commit SHA(s), the files and functions
  touched, the verification check, and the risks, and says whether a modeled value moves. It
  excludes commits `5ae9051` and `62aa53e`, the commit that added this prompt and
  `.claude/settings.json`, and any data file. List separately the notebook changes the
  researcher must make by hand (for example the cell 29 assertion).
- `PROVISIONAL_RESULTS_2026-10-07.md`, as described in Task 3.

Keep all four current as you go, not only at the end, so a session that is cut short still
leaves a usable record.

### Task 5 -- Stop conditions

Stop when the definition of done is met, or when every remaining item is BLOCKED on a researcher
decision. In either case, finish the deliverables, push, and end with a short summary: what is
done, what is blocked, the commit range, and the three numbers the researcher should look at
first. Do not start any deferred item to fill time.

## Start

Do Task 1, then Task 2's first run, then the loop. Do not wait for approval at any point.

## Deferred to a future session (do NOT start now)

- Upgrade 04, then Upgrade 03 (download and audit first), after `mp=5` completes Phase 11.
- Grid impact for dual fuel; panel-upgrade cost (D7b); utility-bill cross-check; AMY2012.
- Fixing the five `return`-instead-of-`assert` tests in `test_efficiency_floor_refactoring.py`.
- Making June 2026 HOMES fuel-neutral (P6), and confirming the Appendix M1 factors (P1).
- Promoting any PROVISIONAL cloud result to a CONFIRMED reference row. That happens only from a
  dev-branch run the researcher makes. Do not add rows to `REFERENCE_VALUES.md`.
- Editing CLAUDE.md, `SESSION_LOG.md` or any notebook. Put proposed text in the implementation
  guide instead.
- Any ResStock 2022.1.1 verification.
- `.claude/settings.json` on the cloud branch is cloud-only -- never port it.
