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
> `utils/validation_framework.py` or any `*_EXPORT_*.py`; never push to, merge
> into, rebase, or reset `resstock2025-dual-fuel-codebase-update` or `main`; never force-push;
> never delete or rewrite branches; never overwrite or edit existing rows in
> `REFERENCE_VALUES.md`; never commit `data/resstock_2025_1/`, any parquet file, or anything
> under `output_results/`; all coding standards, naming rules, and anti-patterns in CLAUDE.md
> still apply. Do not stop to ask questions -- record the question and the default you took in
> `DECISIONS_TAKEN.md` and continue.

**Amended by the researcher, 6 Oct 2026.** The first version of this override also barred
editing any `.ipynb`. The researcher lifted that for this branch: you may edit the notebooks
here, only through the notebook-cell-edit skill, as "Editing notebooks on this branch" below
describes. Everything else in the override stands.

**This session runs unattended overnight. It must never wait for input.** No one will answer a
question, approve a diff, or respond to a prompt. Never call a tool that asks the researcher
something, never run a command that reads from the keyboard (an unanswered `input()`, an
interactive git command, a pager), and never end a turn to wait. When a choice is needed, take
the default named in this prompt, or the refactoring guide's recommended default, write it to
`cmu_tare_model/docs/cloud_run/DECISIONS_TAKEN.md`, and keep going.

Also in force for every commit: no file over 50 MB, no parquet file, nothing under
`cmu_tare_model/data/euss_data/`, `data/resstock_2025_1/`, or any `output_results/` folder.

## Context

**Priority, in the researcher's words: "The most important part is getting the model running
for ResStock 2025 and the results output (specifically Tepper exports) produced for review."**
So, in this order:

1. The model's own notebooks run end to end, nationally, on ResStock 2025.1 for the Dual Fuel
   Heating System package (Upgrade 05, `mp=5`), started by a script with no keyboard input.
2. That run writes its results, and above all the Tepper household and county files, and a
   review summary of those files is committed.
3. The numbers are made right: furnace cost, efficiency rating, rebate rule, column names.

Reach 1 and 2 first, with the calculations as they are, and commit that. Then do 3 and produce
the files again. A session cut short after 1 and 2 has still delivered the main thing.

The researcher will port your commits to the development branch
`resstock2025-dual-fuel-codebase-update` by hand, one gated diff at a time, using the
implementation guide you write. So small, single-purpose commits with honest messages matter
more than speed.

The work is narrower than the plan documents suggest. An audit and a probe run on 6 Oct 2026
found that the pipeline already runs for `mp=5` from load through NPV and adoption, with correct
two-fuel energy, fuel costs and emissions. What remains is under "Gap list" below. The
researcher reviewed the open issues on 6 Oct 2026; those answers are under "Researcher
decisions" and are final.

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
  removed), and the commits that added or revised this prompt, the Phase 2 prompt copy and
  `.claude/settings.json` (their messages end "(cloud-only, do not port)"). **None of those
  cloud-only commits is ever ported.**
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
- No AWS credentials and no `buildstock_query`, so the grid-impact section cannot run here.
- A foreground command is cut off at about 10 minutes. Anything that may run longer goes to the
  background with a log file, and you poll the log.

### How the notebooks run

**Cell numbers in this prompt are the notebook-cell-edit skill's numbers: from 1, counting
markdown and code cells together.** The main notebook has 37 cells, the simulation notebook 26,
the scenarios notebook 30, the baseline notebook 13.

The researcher runs the model by opening `cmu_tare_model/tare_model_main_v3_0.ipynb`, running
every cell, and typing three answers. The chain, and the prompts in the order they appear:

1. Main notebook cell 3 asks "Would you like to begin a new simulation or visualize output
   results from a previous model run?" Answer **Y**. It then runs
   `cmu_tare_model/model_scenarios/tare_run_simulation_v3_0.ipynb` with `%run -i`.
2. The simulation notebook runs `tare_baseline_v3_0.ipynb`, whose cell 3 calls
   `get_menu_choice(menu_prompt, ...)`: "Would you like to filter for a specific state's data?"
   Answer **N** (the whole country). A "Y" is followed by a state prompt ("Which state would
   you like to analyze data for?") and a city menu (answer N for the whole state).
3. The simulation notebook then runs `tare_scenarios_v3_0.ipynb` once per measure package, in
   batch mode (it sets `input_measure_package` first, so no prompt), and exports CSV files to
   `cmu_tare_model/output_results/`. In the scenarios notebook: cell 3 loads the package,
   cell 16 makes the cost calls, cell 17 copies the cost columns onto the home table, cell 19
   does income and rebates, cell 21 copies the remaining cost columns into the NPV tables,
   cells 24 and 25 do NPV and adoption, cell 30 checks the June 2026 rebates.
4. Back in the main notebook: cell 9 loads the ResStock baseline for county weights, cell 19
   builds the county demand tables, cell 23 writes the Tepper files, and then, if
   `GRID_IMPACT_ANALYSIS` is on, cell 32 asks "Enter the 5-digit county FIPS code for the
   grid-impact case study". Answer **42003**.

### Editing notebooks on this branch

Allowed here by the amended override, and only this way:

- Read `.claude/skills/notebook-cell-edit/SKILL.md` first and use its script,
  `.claude/skills/notebook-cell-edit/scripts/edit_notebook_cells.py` (`list`, `show`, `check`,
  `apply`). It needs only the standard library.
- The skill's "stop for approval" step is replaced here: run `check`, read the diff it prints
  yourself, then `apply`. Then `git diff --stat -- '*.ipynb'` must show exactly the lines the
  changes file names, and nothing else.
- The script changes lines inside cells that exist. It cannot add a cell. Put new lines into
  an existing code cell (replace one line with a list of lines). Do not edit the notebook
  file any other way: no retyped cells, no whole-cell replacement tool, no one-off script on
  the notebook's JSON.
- Running a notebook must never change its file. The runner executes cells in memory and
  saves no outputs.
- One commit per notebook change, with nothing else in it. Copy the changes file you applied
  to `cmu_tare_model/docs/cloud_run/notebook_changes/` under a numbered name
  (`01_<what>.json`, `02_...`). The dev notebooks are identical to this branch's at `b17d7af`,
  so the researcher can replay those files, in order, with the same script.
- Keep notebook changes few and small. Fix a problem in a module whenever a module can fix
  it. New lines follow CLAUDE.md: ASCII only, plain-language comments, no `mp5` literals
  beyond the package number the existing blocks already spell out, counts labeled rdu.

### Who the dual-fuel package applies to

ResStock applies Upgrade 05 **only to homes with ducts and access to natural gas** (and not
to shared systems or to wood, other-fuel or no-fuel heating). Measured on the national file,
6 Oct 2026:

- Of the 245,570 rows ResStock applied the package to, 100% have `in.hvac_has_ducts = 'Yes'`
  and 100% have natural gas access (gas heating, water heater, clothes dryer, cooking range,
  pool heater or hot tub). No applied row lacks either.
- The study sample (161,983 rdu) is 100% applied rows, 100% with ducts, 100% with gas access.
  10,624 sample rdu are not heated by gas (electricity 8,778; fuel oil 1,423; propane 423);
  their gas access is a gas water heater (5,027), range (4,741), dryer (2,063), pool heater
  (330) or hot tub (363).
- The 3,951 sample rdu with a boiler or electric baseboard all have ducts and central AC; the
  ducts are their air conditioner's.

The rule that makes this so, and that every piece of new code must keep:

- `applicability` is the first filter in `load_and_filter_upgrade`
  (`process_euss_data.py:233-239`), and `include_sample` requires that ResStock applied every
  package in the run to the home (`:759-763`), together with replaceable heating and a
  central or room AC of the home's own.
- **Every calculation filters on `include_sample`** (through `get_valid_calculation_mask`).
  Never use `include_heating`, `include_cooling` or a ducts test of your own to mean the
  sample. Never re-derive applicability from ducts and gas columns; ResStock's flag is the
  source. The heat-pump cost row is always the ducted one for this package; do not add a
  non-ducted path for it.

### What the 6 Oct 2026 probe measured (dev code as it stands, before any fix)

The probe called the same functions in the same order as the notebooks, on ResStock 2025.1 with
`mp=5`, for PA, MN, FL and the whole country. **Treat these as PROVISIONAL sanity numbers, not
reference values.** Your Stage A runs (below) should reproduce them exactly, because Stage A
changes no calculation.

| Scope | Study sample (rdu) | Homes | Run time | Peak memory |
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
- With the code as it stands (heat pump cost only, SEER 15.2 unconverted, June 2026 fuel gates
  applied to dual fuel), national means over the sample: heat-pump upgrade cost $14,957;
  heating replacement cost $3,742; cooling replacement cost $5,843; lifetime heating fuel
  savings $1,072; lifetime cooling fuel savings $1,353. National adoption (rdu adopters /
  161,983): `heatingLCC_coolingLCC_unsub` 11.19%, `_sub` 60.23%, `_sub_june2026` 12.42%.
  **These will move in Stage B; that is expected.**
- Checks that already pass, per home, in all four scopes: NPV identity 0 violations at the
  half-cent; both NPV ordering checks 0 violations; every adopter flag equals `NPV >= 0`;
  the nine adopter columns are float64 and each has exactly the study-sample count of
  non-blank rows.
- The Tepper household export already runs for `mp=5`: 161,983 rows, 169 columns in the main
  file and 319 in the detailed copy. The county file does not, because the demand table fails
  (G10).
- A full national pass including CSV export (188 s), reload, county tables, maps, dot plot and
  the Tepper household files took 392 s at a 6.5 GB peak. The notebooks keep more copies of
  the tables than the probe did, so expect a higher peak; watch it (see the next item).

### Memory: the loader must read fewer columns

`read_resstock_2025_1_parquet` (`process_euss_data.py:71-99`) reads all 770 columns. Measured:
7.5 GB after the read and 10.6 GB at peak for one national file; `load_and_filter_upgrade`
(`:132-269`) then copies the frame at each filter. The notebooks' sequence (baseline frame held
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
`pyarrow.parquet.read_table`. **Make this the loader's own behavior**, so every caller gets it.
OEDI also publishes per-state files, a usable fallback that should not be needed:
`.../metadata_and_annual_results/by_state/full/parquet/state=PA/PA_upgrade5.parquet` (34 MB; the
baseline is `PA_upgrade0.parquet`).

### Phase status (file:line evidence)

| Phase | Status | Evidence |
|---|---|---|
| 2 Release-aware loader | DONE except two leftovers | Registry `resstock_schema.py:72-97`; constants `constants.py:109-120`; weight from the frame `process_euss_data.py:545`; MP3 override guarded by release `:1029`, `:1057`; gas backup and fan parts read from the file `:660-668`, `:1153-1167`; backup furnace size `:1009`; panel flags `:598-600`, `:1228-1234`. Leftovers: peak columns (G4) and the unused parser (G5). |
| 3 Masking | DONE | Applicability is the first filter `process_euss_data.py:233-239`; funnel `study_sample.py:68-134`; AK/HI excluded `constants.py:135`, `process_euss_data.py:253-257`; rebate-eligible packages by release `determine_rebate_eligibility_and_amount.py:461-492`. D4 was not done and is superseded (P2). |
| 4 Consumption | DONE | Per-fuel, every part: `degree_day_consumption_utils.py:434-501`; `projected_consumption.py:24-98`; savings fraction `process_euss_data.py:1301-1344`; checked against ResStock's own savings columns every run `:820-925`. |
| 5 Capital cost | NOT STARTED | G6, G7. |
| 6 Fuel costs | DONE | Each fuel at its own price `calculate_lifetime_fuel_costs.py:599-616`. |
| 7 Rebates | PARTIAL | `mp=5` is registered as eligible; the June 2026 rule for dual fuel is missing (G8). |
| 8 NPV, adoption | Mechanics DONE | Needs only the furnace cost from G7 in net capital. |
| 9 Climate | DONE | Fossil use counted for any package `calculate_fossil_fuel_emissions.py:83-91`. |
| 10 KPIs, exports | PARTIAL | County adoption, maps, dot plot and Tepper household files run. Demand tables and the Tepper county file fail (G9, G10). |
| 11 Full run, docs | NOT STARTED | No 2025.1 row exists in `REFERENCE_VALUES.md`. Reference rows are out of scope here. |

### Researcher decisions, 6 Oct 2026 (final -- do not reopen)

| # | The researcher's answer | What it settles |
|---|---|---|
| R1 | Price the gas furnace with the same coefficients. | The new backup furnace is costed with the REMDB `furnaces_gas_furnace` regression, the same row and coefficients (pm1, pm2, intercept, retrofit multiplier and adder) that already price a furnace replacement. No separate cost model, no discount. See G7. |
| R2 | Convert to the efficiency rating used in the REMDB cost estimation. | REMDB's heat-pump rows use pm2 = `SEER1`. The package's SEER2 15.2 is converted to SEER1 before the regression. See G6 and P1. |
| R3 | Fix the rebate logic for the dual-fuel application, so that both programs are accurately represented, as currently documented. | For a dual-fuel retrofit under June 2026: HEEHR passes the fuel gate whatever the baseline fuel (D8), and HOMES is fuel-neutral, as CLAUDE.md documents that program. See G8. |
| R4 | Leave a note that new guidance was released in Program Notice 26-3, and that the researcher has not yet reviewed how it differs from 26-1 and 26-2 or how it affects the codebase. | The rebate rules modeled here are the ones CLAUDE.md documents. Carry the note; change no rule because of it. See G8. |
| R5 | Use the module to generate the release-specific column names for electricity demand instead of hard-coding them, and search for other column-name problems. | Every raw ResStock column name comes from `utils/resstock_schema.py` (`resstock_col`). The search is done; results under "Column-name search". See G9, G10. |
| R6 | Create a script to run the notebooks, with the options Y for a new model run, N for National, and 42003 for the FIPS. | The run is driven by the notebooks themselves, not by a script that copies their steps. See G1 and Task 2. |
| R7 | Let the cloud session edit the notebooks on its own branch. | Notebook changes are made in the notebooks, with the skill. No in-memory substitutes. See "Editing notebooks on this branch". |
| R8 | Keep grid impact switched off for the cloud run. The most important part is getting the model running for ResStock 2025 and the results output, specifically the Tepper exports, produced for review. | Sets the order of work (Context) and the definition of done. |
| R9 | The dual-fuel retrofit only includes homes with ducts and access to natural gas. As always, include the applicability filter in the study sample and use `include_sample` in the calculations. | See "Who the dual-fuel package applies to". The adoption denominator is the `include_sample` count. |

### Gap list, in pipeline order

Each item: what is wrong, where, the phase or decision it belongs to, and a size (S / M / L).
"Stage" says when to do it (Task 2).

**Running the notebooks (Stage A)**

- **G1 (M, new file, R6).** No script runs the notebooks, and they stop at keyboard prompts.
  Add `scripts/run_tare_notebooks.py` (Task 2).
- **NB1 (S, notebook, R7).** `tare_run_simulation_v3_0.ipynb` has a run-and-export block only
  for packages 3, 4, 8, 9 and 10 (cells 10-11, 14-15, 17-18, 20-21, 23-24, each under
  `if <n> in VALID_MENU_MPS:`). There is none for 5. On 2025.1 it runs the baseline and no
  package, and the main notebook then fails to load `mp5` results. Add a block for package 5
  that mirrors the existing ones line for line (same steps, same exports, same
  `verify_cost_scenario_columns` call).
- **G2 (S, D1).** The release can only be chosen by editing `constants.py:109`
  (`RESSTOCK_RELEASE_THIS_RUN = '2022.1.1'`). `VALID_MENU_MPS` is derived from it at `:120`;
  functions copy it as a default argument when their module is imported
  (`process_euss_data.py:137`, `:481`, `:935`); `get_rebate_eligible_mps` reads it at
  `determine_rebate_eligibility_and_amount.py:485`; `_validate_inputs` in both cost modules
  checks `VALID_MENU_MPS`. So the release must be set before anything else is imported. See P3.
- **G3 (S, Phase 2).** Column-limited read, described above, as the loader's default.
- **G9 (M, R5).** `adoption_kpis/data_loading.py` is 2022.1.1 only: `EUSS_DATA_DIR` (`:49-52`),
  `mp_to_upgrade` zero-pads (`:327`), `load_euss_baseline` (`:330-369`) and `load_euss_upgrade`
  (`:372-415`) read the 2022.1.1 CSV files. In this clone both raise `FileNotFoundError`. On the
  researcher's machine `load_euss_baseline()` would silently load the 2022.1.1 baseline during
  a 2025.1 run. The main notebook calls all three by name (cells 9 and 19). Keep the names
  and signatures and make them follow the release: the same scope filters as
  `load_and_filter_upgrade` (applicability first, AK and HI out), and the column-limited read.
- **G10 (M, R5).** The demand code names 2022.1.1 columns as literals (`data_loading.py:200-270`).
  `compute_scenario_demand` fails with `KeyError: 'out.electricity.total.energy_consumption.kwh'`
  at `adoption_kpis/demand.py:116` on 2025.1 frames. Build every one of those names with
  `resstock_col(release, logical_name)` and `get_resstock_savings_column`; no raw ResStock name
  may be typed into this module. This blocks the county demand tables, and with them the
  Tepper county file. The main notebook reaches the Tepper cell (23) only after the demand
  cell (19), so G9 and G10 stand between you and any Tepper file.
- Grid impact stays off (R8): the runner switches it off; no notebook change. The main
  notebook's cell 33 names the 2022.1.1 Athena table, `constants.py:483` does too, and
  `constants.py:477` holds the 2022.1.1 column name in the query's form. All deferred.

**Load (Stage B)**

- **G4 (S-M, Phase 2 Task 7, audit E.2).** The 2025.1 peak columns are stored under the 2022.1.1
  names: `base_peak_electricity_heating_kw` / `_cooling_kw` (`process_euss_data.py:700-705`) and
  the four `mp{mp}_peak_electricity_*` columns (`:1187-1198`). 2022.1.1 measures the peak while
  the equipment runs; 2025.1 measures the peak in the winter or summer months. They are not the
  same measurement, so NEXT_STEPS decided they must not share a name. Give the 2025.1 columns
  their own names built from ResStock's `winter` / `summer` (P7), leave the 2022.1.1 names and
  values alone, and carry it into the Tepper export's column list
  (`cmu_tare_model/utils/export_tepper_csv.py:371-383`).
- **G5 (S, Phase 2 Task 5).** `parse_dual_fuel_heating_efficiency` (`process_euss_data.py:295-353`)
  has no caller, so no `hp_seer2` or `backup_afue` value reaches the frame. G6 and G7 need both.
- Applicability is coerced with `.astype(bool)` (`:237`, `:1247`). The published files hold a
  true boolean, so this is right today, but a text column would turn `"False"` into True. Assert
  the dtype is boolean in the loader (S).

**Capital cost (Phase 5, Stage B)**

- **G6 (S, D9, R2).** `_convert_pm2` (`remdb_v4_installed_cost_utils.py:292-347`, the extract at
  `:320`) takes the first number in the efficiency string. For
  `Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, ...` it feeds 15.2 to a regression whose pm2 metric is
  `SEER1` (`remdb_v4_tare_retrofit_costs.csv`, row `air_source_heat_pump_centrally_ducted`,
  bounds 13.8-24.0, mid coefficient 594.74 per SEER point, retrofit multiplier 1.5). The probe
  saw 15.2 on all 161,983 sample rdu. It raises no error because 15.2 is inside the bounds.
  Add one named, documented SEER2 -> SEER1 conversion at this boundary, applied only where the
  rating is SEER2 (P1). Expected effect: 15.2 -> 16.0 raises the heat-pump upgrade cost by
  about $754 per home in 2025 dollars (0.8 x 594.74 x 1.5 x the 2023-to-2025 CPI ratio).
- **G7 (M-L, D6, R1).** The new condensing gas furnace has no cost.
  `_assign_upgrade_row_id` (`:146-187`) knows only the two heat-pump rows, the scenarios
  notebook makes three cost calls (cell 16), and `calculate_capital_costs`
  (`calculate_lifetime_private_impact.py:527-655`) adds one upgrade cost. Follow D6:
  - a furnace upgrade branch that uses the REMDB row `furnaces_gas_furnace` with the same
    coefficients as the replacement cost (R1): pm1 = heating capacity in BTU/hr from
    `size_heat_pump_backup_primary_k_btu_h` (x 1000), pm2 = the backup AFUE, 0.925 or 0.95,
    from G5; inflated to 2025 dollars like every REMDB cost;
  - a fourth cost step in the scenarios notebook (cell 16) that runs only when the package is
    dual fuel, and the furnace cost column added to the columns cell 17 copies (notebook
    change NB2);
  - a new installed-cost column for it (P8), kept separate from the heat-pump cost;
  - that cost added to total and net capital for a dual-fuel package only;
  - the column added to the Tepper household export.
  **A trap in the converter:** when the REMDB metric is AFUE, `_convert_pm2` divides the number
  it is given by 100, and for a text value it takes the first number in the text. Handed the
  fraction 0.925 it would feed 0.00925; handed the option string it would feed 0.152. Either
  one runs without error and prices the furnace at a few hundred dollars. Check the value
  actually fed to the regression (it must be 0.925 or 0.95), the way G6 checks SEER.
  One helper should answer "is this package dual fuel?" from the release and package number;
  G7, G8, the notebook cells and the dot plot should all use it. Never test `menu_mp == 5`
  alone: package numbers repeat across releases.
- `cmu_tare_model/utils/validate_capital_costs.py` hard-codes `mp3_` and `mp4_` (`:1002-1003`,
  `:1279`). It is an MP3-versus-MP4 comparison and the main notebook already skips it on
  2025.1 (cell 11). Leave it.

**Furnace size, measured 6 Oct 2026 (national sample, 161,983 rdu).** ResStock sizes the backup
furnace again in the upgrade run; it does not copy the baseline size. In practice the two are
nearly the same where the home already had a furnace, and differ where it did not:

| Baseline heating system | rdu | Baseline size (kBtu/h, mean) | Backup furnace size (mean) | Median ratio | Within 5% |
|---|---:|---:|---:|---:|---:|
| Natural Gas Fuel Furnace | 148,997 | 63.8 | 63.7 | 1.000 | 98.2% |
| Electricity Electric Furnace | 7,355 | 50.5 | 50.2 | 0.994 | 97.3% |
| Natural Gas Fuel Boiler | 2,362 | 54.5 | 72.3 | 1.383 | 28.6% |
| Electricity Baseboard | 1,375 | 35.1 | 49.6 | 1.460 | 18.9% |
| Fuel Oil Fuel Furnace | 1,270 | 81.6 | 81.7 | 1.000 | 99.6% |
| Propane Fuel Furnace | 410 | 67.4 | 67.5 | 1.000 | 98.5% |
| Fuel Oil / Electric / Propane Boiler | 214 | 35.0-58.4 | 48.5-77.0 | 1.34-1.45 | 15-33% |

All sample homes: baseline 62.9, backup furnace 63.2, heat pump 40.7 kBtu/h (means);
correlation of baseline and backup size 0.994; exactly equal on 29.3% of rdu. So always price
the furnace from `size_heat_pump_backup_primary_k_btu_h`, never from the baseline size: for the
3,951 boiler and baseboard rdu the backup furnace is about 34-46% larger. The baseline file's
own backup-size column is zero for these homes. Preview with the same coefficients: the furnace
cost averages about $4,112 per home in 2025 dollars (median $4,069, range $3,646-$6,567; size
moves it little because the capacity coefficient is $0.00645 per BTU/hr). Your G7 result should
land close to that; if it does not, find out why before moving on.

**Rebates (Phase 7, Stage B)**

- **G8 (S-M, D8, R3).** `calculate_rebate_program`
  (`determine_rebate_eligibility_and_amount.py:604-620`) applies the June 2026 fuel gates to the
  baseline fuel with no exception for a dual-fuel package. Nationally 153,205 fossil-baseline
  sample rdu get $0 under June 2026, while the 2024 rules pay them (mean HEEHR $7,476 for
  natural-gas baselines). Represent both programs as documented, for a dual-fuel retrofit:
  - **HEEHR** (at or below 150% AMI): passes the June 2026 fuel gate whatever the baseline
    fuel (D8). Same $8,000 cap and the same 100% / 50% cost shares.
  - **HOMES** (above 150% AMI): fuel-neutral, as CLAUDE.md documents the program. Same savings
    tiers and caps.
  - Unchanged: the South Dakota gate, the income routing, one program per home, the 2024
    columns, and every 2022.1.1 package -- including the electric-only hold on June 2026
    HOMES for those packages, which CLAUDE.md lists as deferred.
  Put the rule in configuration (`REBATE_RULE_CONFIG` or a release-and-package set beside
  `get_rebate_eligible_mps`), not in an ad hoc `if`.
- **Program Notice 26-3 note (R4).** Carry this sentence, in substance, as a comment beside
  `REBATE_RULE_CONFIG` in `constants.py`, in `DECISIONS_TAKEN.md`, in the provisional results,
  and as proposed CLAUDE.md text in the implementation guide: "DOE has released new guidance
  in Program Notice 26-3. The researcher has not yet reviewed how it differs from Program
  Notices 26-1 and 26-2 or how it affects this code. The rebate rules modeled here are the
  ones documented in CLAUDE.md." Do not guess what 26-3 says and do not change a rule for it.
- Checks that assume fossil baselines get nothing under June 2026, and are wrong for dual fuel
  once G8 lands:
  - `tare_scenarios_v3_0.ipynb` cell 30 asserts no fossil baseline receives June 2026 dollars
    and stops the run. Make it right for a dual-fuel package (notebook change NB3): South
    Dakota still $0, electric baselines still funded, fossil baselines funded.
  - the dot plot's rule 2 (`visuals_adoption_dotplot.py:280`). The main notebook calls
    `plot_econ_adoption_dotplot_figure` without `check_june2026_fossil_rule`, so make the
    function decide for itself: apply the rule unless the package is dual fuel.
  - `scripts/verify_june2026_rebate_fossil_gate.py` (`_MP = 4` at `:30`).

**Exports (Stage B)**

- The main notebook's title table (cell 9, `HEATING_MP_SUBTITLES`) has no entry for package 5,
  so titles fall back to "MP5". Add one (notebook change NB4).
- `visuals_adoption_dotplot.py:308` and `:523` default `scaling_factor` to `242.0`; it is used
  only when a frame has no `weight` column, but it is a 2022.1.1 number. Read the weight (S).
- The Tepper household export takes its columns from fixed lists in
  `cmu_tare_model/utils/export_tepper_csv.py` (`build_household_column_list`, `:238`). After
  G4 and G7 it needs the 2025.1 peak names and the furnace cost. Its docstrings and
  `cmu_tare_model/docs/tare_tepper_exports_data_dictionary.md` describe 2022.1.1 only; write
  the 2025.1 differences in the review file (Task 4), not into the dictionary.

**Column-name search (R5), done 6 Oct 2026.** Every raw ResStock column name written as a
literal in the tracked model code and notebooks (55 distinct names) was checked against the
2025.1 file.

| Where | Names | State |
|---|---|---|
| `adoption_kpis/data_loading.py:200-269` | 12 names, 15 uses: heating by fuel, heat-pump backup, fans and pumps, cooling, hot water, electricity total, site energy total, heating load delivered | **Renamed in 2025.1. On the `mp=5` path. Fix (G10).** Every one already has a row in the column map. |
| `adoption_kpis/thermal_cop.py:149`, `:153` | electric and natural gas heating energy | Renamed. Not called by the main notebook. Fix the same way; it is two lines. |
| `constants.py:477` | `out.electricity.total.energy_consumption` (query form) | Grid impact only. Deferred. |
| `process_euss_data.py:1066-1151` | water heating, clothes drying, cooking, and the MP9 / MP10 enclosure blocks | Inactive: those categories are not in `EQUIPMENT_SPECS` and those packages are not in the run. Leave. |
| Everywhere else, including the four notebooks | 26 names: `in.state`, `in.county`, `in.city`, `in.heating_fuel`, `in.hvac_heating_type_and_fuel`, `in.hvac_has_ducts`, `in.vacancy_status`, `in.geometry_building_type_recs`, ... | Same name in 2025.1. No change needed. |

`get_resstock_savings_column` and `find_enduse_columns` (`calculation_utils.py`) already handle
both releases' spellings. The remaining name problem is on TARE's side: the peak columns (G4).
When you add a name the column map lacks, add a row to
`cmu_tare_model/docs/resstock_2025_1_column_map.csv`; do not type the name into a module.

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
| AFUE divided by 100 twice, or read from the wrong number in the string | **New risk in G7**; check the value fed to the regression. |
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
| D3 | `applicability` is the first filter; funnel table after each step | In code (R9). |
| D4 | Remove the cooling-technology filter on both releases | **Superseded** by the study sample rule of 3-5 Oct 2026 (P2). |
| D5 | Retrofit fuels come from the file's own columns, not a per-package table | In code. |
| D6 | Furnace upgrade row; the fourth cost call runs only for a dual-fuel retrofit | To do (G7, R1). |
| D7 | No panel cost; carry the constraint flags | In code. |
| D8 | `mp=5` passes the June 2026 HEEHR fuel gate whatever the baseline fuel | To do (G8, R3). |
| D9 | Apply the DOE Appendix M1 SEER2/HSPF2 conversion | To do (G6, R2); factor value provisional (P1). |
| D10 | Exclude AK and HI | In code. |

### PROVISIONAL decisions for this session

The researcher has not ruled on these. Each is the refactoring guide's recommended default, or
the smallest choice consistent with CLAUDE.md and with R1-R9. **Record every one in
`DECISIONS_TAKEN.md` with how to reverse it**, and mark any result that depends on it.

- **P1 (conversion factors).** SEER1 = SEER2 / 0.95 and HSPF1 = HSPF2 / 0.85. SEER2 15.2
  becomes SEER1 16.0. This agrees with what the repo already records: the ENERGY STAR floor is
  written as 16.0 SEER1 at `process_euss_data.py:1016`, the guide (Section 2.2) describes this
  package's SEER2 15.2 as the ENERGY STAR minimum, and `constants.py:431-435` equates SEER2
  14.3 with about SEER1 15. The DOE Appendix M1 text itself was never checked (ecfr.gov is not
  reachable here). Keep the two factors as named constants in one place. Only SEER feeds the
  cost regression; HSPF is converted for the record only. The option string writes `SEER 15.2`
  while the measure documentation says SEER2 15.2; treat it as SEER2.
- **P2 (D4).** Follow CLAUDE.md, not NEXT_STEPS: the cooling filter stays and the study sample is
  homes with a central or room AC of their own. For `mp=5` this leaves out 15,017 applicable rdu
  with no AC. Change no code for this.
- **P3 (release switch).** `constants.py` reads the release from the environment variable
  `TARE_RESSTOCK_RELEASE`, defaulting to `'2022.1.1'`, so it is still declared in one place and
  one run uses one release. The runner sets the variable before anything imports
  `cmu_tare_model`. Reject an unknown value with a `ValueError`.
- **P5 (rebate base).** The furnace cost is not part of the cost a rebate covers. HEEHR and HOMES
  keep reading the heat-pump upgrade cost column. A fossil furnace is not a rebate measure.
- **P7 (2025.1 peak names).** `base_peak_electricity_winter_kw`, `base_peak_electricity_summer_kw`,
  `mp{mp}_peak_electricity_winter_kw`, `mp{mp}_peak_electricity_summer_kw`, and the two savings
  columns with `_savings` appended.
- **P8 (furnace cost column).** Build it with `create_cost_col`, adding a cost type for the backup
  furnace, so the name follows the existing pattern
  (`mp{mp}_heating_<cost type>_installed_cost_{cost scenario}`). Never hard-code `mp5`.
- **P10 (loop states).** PA, MN (cold) and FL (warm), because the probe measured all three.
- **P13 (export files stay off the branch).** The Tepper files and every other run output stay
  on this machine: the override bars committing anything under `output_results/`, the national
  household files are far over 50 MB, and the repository is public. What is committed is a
  review summary of them (Task 4) and the commands that reproduce them on the researcher's
  machine.

## Tasks

### Task 1 -- Setup

1. **Install the environment inside this session.** Do this even if a prepared environment
   seems to exist. Run from the repo root:

   ```bash
   if [ -x /opt/tare-venv/bin/python ]; then VENV=/opt/tare-venv; else VENV="$HOME/tare-venv"; fi
   PINNED="pandas==2.1.4 numpy==1.26.4 pyarrow==23.0.1 scipy==1.16.0 matplotlib==3.10.0 \
   seaborn==0.13.2 geopandas==1.1.3 pyogrio==0.12.1 shapely==2.1.2 pyproj==3.7.2 \
   openpyxl==3.1.5 ipython nbformat matplotlib-inline requests psutil pytest"
   LOOSE="pandas>=2.1,<2.3 numpy<2.3 pyarrow scipy matplotlib seaborn geopandas pyogrio \
   shapely pyproj openpyxl ipython nbformat matplotlib-inline requests psutil pytest"
   if [ ! -x "$VENV/bin/python" ]; then
     uv venv --seed --python python3 "$VENV" || python3 -m venv "$VENV"
   fi
   uv pip install --python "$VENV/bin/python" $PINNED \
     || uv pip install --python "$VENV/bin/python" $LOOSE \
     || "$VENV/bin/python" -m pip install $LOOSE
   uv pip install --python "$VENV/bin/python" -e . || "$VENV/bin/python" -m pip install -e .
   "$VENV/bin/python" -c "import config, cmu_tare_model, pandas, pyarrow, openpyxl, geopandas, IPython; print(pandas.__version__)"
   ```

   The researcher's environment is Python 3.11.13 with the pinned versions (IPython 9.1.0). If
   the pinned set will not install on this machine's Python, the loose set is acceptable; write
   the versions you ended up with at the top of `BREAKAGE_LOG.md`. If the editable install
   fails, run everything with `PYTHONPATH=.` from the repo root instead (`config.py` is at the
   root). `environment-cmu-tare-model.yml` and `requirements.txt` are Windows-only snapshots; do
   not use them. `buildstock_query`, `boto3` and `toml` are needed only by the deferred
   grid-impact code; do not install them. Use `"$VENV/bin/python"` for every command below.

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
   still mismatches, log it as BLOCKED with both hashes, finish the deliverables with what you
   have, push, and end. Only the two parquet files are opened by the pipeline; the other three
   are for looking up column names and values. `enumeration_dictionary.tsv` must be read with
   `encoding='latin-1'`. Each parquet file has 549,971 rows and one distinct `weight`.

3. **Record the test baseline.** `"$VENV/bin/python" -m pytest -q` from the repo root. A clean
   checkout of this branch gave 352 passed, 1 skipped (and 5 warnings that tests in
   `test_efficiency_floor_refactoring.py` return a value; leave those tests alone). Write your
   counts into `BREAKAGE_LOG.md` as the session baseline. If they differ from 352 / 1 before you
   change anything, record why (most likely a package version) and carry on.

4. Create `cmu_tare_model/docs/cloud_run/` with the deliverable files started, and commit.

### Task 2 -- The notebook runner, then two stages

**The runner.** Add `scripts/run_tare_notebooks.py` (R6). It runs
`cmu_tare_model/tare_model_main_v3_0.ipynb` from the first cell to the last, and through it the
simulation, baseline and scenario notebooks, with no keyboard input and without writing to any
notebook file.

- **Default answers, as the researcher set them:** `Y` to "begin a new simulation", `N` to
  "filter for a specific state's data" (so the run is National), and `42003` to the county
  FIPS question. Running the script with no options must give exactly that run.
- **Options:** `--release` (default: whatever `constants.py` says; this session passes
  `2025.1`, P3); `--state PA` for a one-state run (then it answers Y to the state menu, the
  state code to the state question, and N to the city menu); `--fips` (default `42003`);
  `--skip-grid-impact`; `--log` for a log file.
- **Prompts:** it answers each `input()` by matching the prompt's text to a known question. A
  prompt it does not recognize stops the run with an error that prints the prompt. It never
  waits. An answer that is never asked for (the FIPS, when grid impact is off) is not an error.
- **Grid impact (R8):** `--skip-grid-impact` sets `GRID_IMPACT_ANALYSIS` to False in
  `cmu_tare_model.constants` before any notebook imports it, so the main notebook's guarded
  cells do nothing. Every run in this session passes it. On the researcher's machine the
  default leaves grid impact on and answers 42003.
- **How it runs a notebook:** the simplest thing that works. One IPython shell, `input`
  replaced by the answering function, the non-interactive `Agg` drawing backend, and IPython's
  own `%run -i` on the main notebook, which then starts the other notebooks itself. Confirm
  two things before trusting it: a failing cell in any of the four notebooks stops the whole
  run with a non-zero exit and a message naming the notebook and the error (a run that carries
  on past an error is worse than one that stops), and `%matplotlib inline` in the main
  notebook does not fail outside Jupyter. If IPython's `%run` cannot give you both, run the
  cells yourself from the notebook file instead.
- **Checks after the run.** From the tables the notebooks leave in memory, check and print, and
  exit non-zero if any fails: NPV identity (NPV = discounted heating savings + discounted
  cooling savings - net capital cost, to the half-cent, for all nine cases); CLAUDE.md's two
  per-home NPV ordering checks and the two county adoption-rate checks; each adopter flag
  equals `NPV >= 0`; each adopter column's non-blank count equals the `include_sample` count;
  the weight has one distinct value; no blank energy, cost or NPV value in a sample home; the
  Tepper household file has one row per sample home.
- It prints the wall time of the run and the process's peak memory.
- The notebooks write to `cmu_tare_model/output_results/` and save figures under the repo
  root. Both are git-ignored. Never add them to a commit, and delete old runs when disk gets
  short.
- Follow CLAUDE.md's coding standards in the script and in every module you touch: Google-style
  docstrings, type hints, comments that say why, ASCII only, no `mp5` or `ref2025_mp5_` literals,
  88-character lines, no one-letter names. The script must work for both releases; nothing in
  it may be specific to package 5.

**Stage A -- get it running and get the files out (R8). No calculation changes.**

1. G2 and P3 (release switch), G3 (column-limited read), G1 (the runner). Commit each.
2. Run `--release 2025.1 --state PA --skip-grid-impact`. It stops at the first thing that is
   broken. Fix that, test, commit, run again. You will meet, in this order: NB1 (no package 5
   block), then G9 and G10 in the main notebook. Fix what else stops the run the same way, in
   a module where a module can fix it.
3. When PA runs to the last cell: confirm 6,469 sample rdu and 1,642,503 homes, and run MN
   and FL.
4. Run National (`--release 2025.1 --skip-grid-impact`) in the background with a log. Confirm
   the probe's numbers above: 161,983 sample rdu, the funnel, the three adoption rates. Stage
   A changes no calculation, so they must match; a difference is a bug to find now.
5. Write the Tepper review summary for this run (Task 4), marked "Stage A: before the cost and
   rebate fixes", and the provisional results. Commit and push. **The main deliverable now
   exists.**

**Stage B -- make the numbers right, then produce the files again.**

Work through, one commit each, with a test for each: G5, G6, G7 with NB2, G8 with NB3, G4, the
Tepper export columns, NB4, then the small items. For each:

1. Fix it in the module where the fault is. Smallest change that is right for both releases.
2. Add or extend a test under `cmu_tare_model/tests/` (test data is made up in the test; no test
   may need the downloaded files).
3. Run the test suite. No regression against the Task 1 baseline.
4. Run the runner on PA, MN and FL.
5. Commit that one change with a message that says what it does and whether a modeled value
   moves. Push. Append to `BREAKAGE_LOG.md`.

Run National after capital cost (G6, G7), after rebates (G8), and at the end; each in the
background with a log. After the last one, rewrite the Tepper review summary and the
provisional results for the final run, keeping the Stage A numbers beside the final ones.

For any run: a command expected to take longer than about 10 minutes goes to the background
with a log, and you poll it. If the national run comes close to the machine's 16 GB, record
the peak and where it happens; cut copies in module code if that is where they are, and
otherwise report it.

If the same failure survives three distinct fix attempts, write it in `BREAKAGE_LOG.md` as
**BLOCKED** with the traceback and what you tried, and move to the next item that does not
depend on it.

Things each fix must show:

- **G9, G10:** `adoption_kpis/data_loading.py` contains no typed-in ResStock column name; the
  KPI loaders apply the applicability filter and leave out AK and HI; county home counts in the
  adoption table and the demand table agree within one home's weight in every county, as the
  main notebook's own export cell requires.
- **G6:** the value fed to the heat-pump regression is 16.0 for every sample home, the conversion
  is one named function, and a test covers it. State how much the mean upgrade cost moved
  (expect about $754).
- **G7:** a furnace cost exists for every sample home with no blank, from the same
  `furnaces_gas_furnace` coefficients as the replacement cost and the backup furnace's own
  size; the AFUE fed to the regression is 0.925 or 0.95; total and net capital include the
  furnace for a dual-fuel package; the NPV identity still holds; no MP3 or MP4 code path
  produces a furnace cost. Report the mean furnace cost (expect about $4,112) and the change
  in adoption.
- **G8:** under June 2026, fossil-baseline dual-fuel homes in a participating state get HEEHR
  at or below 150% AMI and HOMES above it when their savings reach 20%; South Dakota still gets
  $0; the 2024 columns do not change. Give the program-by-fuel table before and after.
- **Every item:** new code filters on `include_sample` (R9).

### Task 3 -- Definition of done

- The runner completes the national `mp=5` notebook run (`--release 2025.1
  --skip-grid-impact`, otherwise the default answers), producing every output family in guide
  Section 1 except grid impact (deferred) and reference-value rows (out of scope).
- **The Tepper files exist for the final run**: the household main file, the detailed
  household copy and the county file, for the National scope and for Allegheny County
  (`G4200030`), as the main notebook's cell 23 writes them. The review summary of them is
  committed.
- The NPV identity holds to the half-cent (0 violations), and CLAUDE.md's NPV ordering checks
  show 0 violations.
- The adoption denominator equals the study-sample count (`include_sample`, R9). Report the
  `include_heating` count beside it for reference (national today: 161,983 and 177,000).
- There are no test regressions against the setup baseline.
- Key results are in `cmu_tare_model/docs/cloud_run/PROVISIONAL_RESULTS_2026-10-07.md`, with
  counts in rdu and homes, using the weight read from the frame. They are marked PROVISIONAL,
  not reference values. Include: the sample funnel; the sample by baseline fuel and cooling
  type; mean baseline and retrofit fuel costs and savings for heating and cooling; mean costs
  (heat pump, furnace, both replacements); rebate dollars and counts by program and baseline
  fuel for both guidance years; the nine NPV means and nine adoption rates; climate means; the
  share of homes with negative heating and negative cooling savings; run time and peak memory;
  the Program Notice 26-3 note; and the Stage A numbers beside the final ones, so the effect
  of the fixes is visible.

### Task 4 -- Required deliverables, committed under `cmu_tare_model/docs/cloud_run/`

- `TEPPER_EXPORT_REVIEW_2026-10-07.md` -- what the researcher reads first. For each Tepper file
  the final run wrote (National and Allegheny; household main, household detailed, county):
  its name; rows in rdu and homes; number of columns; every column name in order; for each
  numeric column the count of blanks and the minimum, median, mean and maximum; for each text
  column its distinct values with counts (or the count of distinct values when there are
  many). Then: which columns are new or mean something different from the 2022.1.1 export
  (natural gas backup energy, furnace cost, winter and summer peaks, panel flags, the weight
  of 253.9), written as dictionary entries the researcher can paste into the data dictionary;
  the checks run on the files and their results; and the same table of headline numbers for
  the Stage A files. End it with **the exact Git Bash commands that reproduce the full files
  on the researcher's Windows machine** from this branch: where to put the two parquet files
  (the researcher already has them, under the main folder's `data/resstock_2025_1/`), the
  runner command, and where the files land.
- `BREAKAGE_LOG.md`: the package versions installed, the test baseline, then every failure, its
  cause, the fix, and the commit SHA. Include BLOCKED items.
- `DECISIONS_TAKEN.md`: every judgment call where neither NEXT_STEPS nor R1-R9 gives a decision
  -- the P items above and any you add -- the default taken, why, and how to reverse it. Also
  the Program Notice 26-3 note.
- `IMPLEMENTATION_GUIDE_2026-10-07.md`: an ordered, step-by-step port plan for the dev branch,
  one task per gated diff. Each task gives the cloud commit SHA(s), the files and functions
  touched, the verification check, and the risks, and says whether a modeled value moves. It
  excludes commits `5ae9051` and `62aa53e`, every commit whose message ends "(cloud-only, do
  not port)", and any data file. **A separate section covers the notebook changes**: for each
  one, the notebook, the cell, what changed and why, its commit, and the changes file under
  `notebook_changes/` to replay with the notebook-cell-edit skill, in order. Remind the
  researcher to close and reopen any VS Code tab of an edited notebook.
- `notebook_changes/` with the numbered changes files.
- `PROVISIONAL_RESULTS_2026-10-07.md`, as described in Task 3.

Keep all of them current as you go, not only at the end, so a session that is cut short still
leaves a usable record.

### Task 5 -- Stop conditions

Stop when the definition of done is met, or when every remaining item is BLOCKED on a researcher
decision. In either case, finish the deliverables, push, and end with a short summary: what is
done, what is blocked, the commit range, where the Tepper review is, and the three numbers the
researcher should look at first. Do not start any deferred item to fill time.

## Start

Do Task 1, then the runner, then Stage A, then Stage B. Do not wait for approval at any point.

## Deferred to a future session (do NOT start now)

- Upgrade 04, then Upgrade 03 (download and audit first), after `mp=5` completes Phase 11.
- Grid impact for dual fuel, including the 2025.1 query table and column names; panel-upgrade
  cost (D7b); utility-bill cross-check; AMY2012.
- Reviewing Program Notice 26-3 against 26-1 and 26-2 and changing any rebate rule for it. The
  researcher does that.
- Fixing the five `return`-instead-of-`assert` tests in `test_efficiency_floor_refactoring.py`.
- Making June 2026 HOMES fuel-neutral for the 2022.1.1 packages, and confirming the Appendix M1
  factors against the DOE text (P1).
- A column-limited read for the ResStock 2022.1.1 CSV files. It cannot be tested here.
- Promoting any PROVISIONAL cloud result to a CONFIRMED reference row. That happens only from a
  dev-branch run the researcher makes. Do not add rows to `REFERENCE_VALUES.md`.
- Editing CLAUDE.md, `SESSION_LOG.md` or the Tepper data dictionary. Put proposed text in the
  deliverables instead.
- Any ResStock 2022.1.1 verification.
- `.claude/settings.json` on the cloud branch is cloud-only -- never port it.
