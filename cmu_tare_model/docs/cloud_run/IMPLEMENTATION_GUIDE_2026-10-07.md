# Implementation guide -- porting the cloud mp=5 work to `resstock2025-dual-fuel-codebase-update`

Status: complete for this session. Fifteen patches; the final national run with all of
them passed every check (PROVISIONAL_RESULTS_2026-10-07.md). Suggested port order is the
patch order; 01-05 alone give the Stage A run.

One task per gated diff, in order. Each task names the patch that holds it, the files and
functions touched, the check to run, the risks, and whether a modeled value moves.

## Read this first: there are no cloud commits

The cloud environment refused the session's first `git commit` (and a following
`git status`) as a destructive git action, so the session made no commits and pushed
nothing. Every change sits in the cloud working tree, and each port task below is also a
standalone patch file:

```
cmu_tare_model/docs/cloud_run/patches/NN_<task>.patch
```

Patches are plain unified diffs against the branch as it was at the start of the session
(the dev branch's `b17d7af` for every file they touch), made one task at a time, so
`git apply` takes them in number order. Where this guide would name a cloud commit SHA it
names the patch. To port one task on the dev branch:

```bash
git apply --check cmu_tare_model/docs/cloud_run/patches/NN_<task>.patch
git apply cmu_tare_model/docs/cloud_run/patches/NN_<task>.patch
git diff            # review, then commit with your own message
```

Patches that change a notebook (`04`, `09`, `11`, ...) are also listed under "Notebook
changes" with the changes file that replays them through the notebook-cell-edit skill;
use either the patch or the changes file, not both.

Do not port: `5ae9051`, `62aa53e`, any commit whose message ends "(cloud-only, do not
port)", `.claude/settings.json`, any data file, or `cmu_tare_model/docs/cloud_run/`
itself (session records).

After any notebook change, close and reopen (or "Revert File") every open VS Code tab of
that notebook before running or saving it. VS Code keeps an open notebook in memory, and
saving a stale tab writes the old text back over the change.

## Module and notebook tasks, in order

### 01 -- Release switch read from an environment variable (G2, P3)

- Patch: `01_release_switch_env_var.patch`
- Files: `cmu_tare_model/constants.py` (`RESSTOCK_RELEASE_AND_MP` moved above the
  release; `RESSTOCK_RELEASE_THIS_RUN` read from `TARE_RESSTOCK_RELEASE`, default
  `'2022.1.1'`, unknown value raises `ValueError`); `tests/test_constants.py` (three tests,
  each in a fresh process).
- Check: `python -m pytest cmu_tare_model/tests/test_constants.py`; with the variable
  unset, `python -c "from cmu_tare_model.constants import VALID_MENU_MPS;
  print(VALID_MENU_MPS)"` prints `[0, 3, 4]`.
- Risk: a shell with `TARE_RESSTOCK_RELEASE` left set runs the other release. The runner
  prints the release first; Jupyter reads the variable from the environment VS Code was
  started in.
- Modeled values: none move.

### 02 -- Column-limited 2025.1 read (G3)

- Patch: `02_column_limited_2025_1_read.patch`
- Files: `energy_consumption_and_metadata/process_euss_data.py` (new
  `select_resstock_2025_1_columns`; `read_resstock_2025_1_parquet` reads the schema with
  `pyarrow.parquet.read_schema` and passes `columns=` to `read_table`);
  `tests/energy_consumption_and_metadata/test_process_euss_data.py` (four tests).
- Check: 141 of 770 columns for both national files; PA `df_enduse_refactored` and
  `df_enduse_compare` frames are identical (`DataFrame.equals`) between the limited and
  a full read (checked in the session: 6,469 sample rdu either way).
- Risk: code that later reads a ResStock column not in the column map, not a heating or
  cooling part, and not a total/hot-water/refrigerator savings column will get a
  `KeyError`. Add the column to the map CSV (the right fix anyway).
- Modeled values: none move. National peak memory falls from about 16.7 GB (estimated,
  full read) to 10.7 GB (measured, whole run).

### 03 -- Notebook runner (G1, R6)

- Patch: `03_notebook_runner.patch` (new file `scripts/run_tare_notebooks.py`).
- What it does: one IPython terminal shell runs `tare_model_main_v3_0.ipynb` with
  IPython's own `%run -i`; the notebooks' own `%run` calls start the others. `input()` is
  answered by matching the question text (Y / N / 42003 by default; `--state XX` answers
  the state questions); an unknown or repeated question stops the run. IPython's notebook
  runner is replaced, for this shell only, by one that runs the same code cells but stops
  at the first failing cell of any notebook and raises an error naming the notebook and
  cell (a `BaseException`, so the main notebook's `try/except Exception` "try again"
  loop cannot swallow it). Checks after the run, wall time, peak memory and where it
  happened. Notebook files are only read.
- Options: `--release`, `--state`, `--fips`, `--skip-grid-impact`, `--log`,
  `--skip-checks`.
- Check: `python scripts/run_tare_notebooks.py --release 2025.1 --state PA
  --skip-grid-impact` exits 0 with "All 20 checks passed".
- Risks: uses `TerminalInteractiveShell` because the base shell cannot run
  `%matplotlib inline`; it never starts the keyboard loop. Figures show as text
  (`<Figure ...>`) and are saved by the notebooks as usual. On Windows the peak memory
  comes from psutil's `peak_wset`.
- Modeled values: none move.

### 04 -- NB1: package 5 block in the simulation notebook

- Patch: `04_NB1_sim_notebook_mp5_block.patch`; notebook change `01` (below).
- Without it a 2025.1 run computes the baseline only and the main notebook fails to load
  `mp5` results.
- Modeled values: none move.

### 05 -- KPI loaders and column names follow the release (G9, G10, R5)

- Patch: `05_G9_G10_kpi_loaders_and_names.patch`
- Files: `adoption_kpis/data_loading.py` (every ResStock name from `resstock_col`; new
  `STATE_COL`, `HEATING_FUEL_COL`; `HEATING_ELEC_SAVINGS_COLS` adds the 2025.1 backup-fan
  part; `mp_to_upgrade` does not zero-pad on 2025.1; `load_euss_baseline` and
  `load_euss_upgrade` keep their names and signatures and load through
  `load_and_filter_upgrade` -- applicability first, AK and HI out, column-limited read);
  `adoption_kpis/demand.py` (reads the release's names, keeps its output labels);
  `adoption_kpis/thermal_cop.py` (two names); `tests/adoption_kpis/test_data_loading.py`
  (eight tests, including a demand run on 2025.1-named frames and a test that no
  ResStock name is typed into `data_loading.py`).
- Check: the main notebook's cell 23 prints "home_count agrees in all N counties"; PA,
  MN, FL and National all pass.
- Risks: `load_euss_baseline(filename=...)` with a non-default file now raises (the file
  follows the release). On 2022.1.1 the rows are the same as before (applicability and
  AK/HI remove nothing there) but this was not run here (no 2022.1.1 data in the cloud).
- Modeled values: none move on 2022.1.1. On 2025.1 the demand "Net heating" row now
  includes the backup furnace's fan electricity (it used to fall into "Other end uses").

### 06 -- Parse the dual-fuel option string (G5)

- Patch: `06_G5_dual_fuel_spec_columns.patch`
- Files: `constants.py` (`DUAL_FUEL_PACKAGES_BY_RELEASE`); new
  `utils/measure_packages.py` (`is_dual_fuel_package(menu_mp, release=None)`, the one
  helper every dual-fuel decision uses); `process_euss_data.py` (new
  `add_dual_fuel_spec_columns`, called in `df_enduse_compare` for a dual-fuel package:
  `upgrade_hp_seer2`, `upgrade_hp_hspf2`, `upgrade_backup_fuel`, `upgrade_backup_afue` as a
  fraction, `upgrade_switchover_f`); tests.
- Check: national: SEER2 15.2 and HSPF2 7.8 on every sample home; AFUE 0.925 on 93,501
  rdu and 0.95 on 68,482.
- Modeled values: none move (the columns are read by 07 and 08).

### 07 -- SEER2 to SEER1 before the heat-pump cost regression (G6, R2, P1)

- Patch: `07_G6_seer2_to_seer1.patch`
- Files: `constants.py` (`SEER2_PER_SEER1 = 0.95`, `HSPF2_PER_HSPF1 = 0.85`, PROVISIONAL);
  new `utils/efficiency_ratings.py` (`seer2_to_seer1`, `hspf2_to_hspf1`);
  `process_euss_data.add_dual_fuel_spec_columns` adds `upgrade_hp_seer1` and
  `upgrade_hp_hspf1`; `remdb_v4_installed_cost_utils.add_remdb_metrics` (Step 4a: homes
  with `upgrade_hp_seer1` are priced at it, after checking their REMDB row's pm2 metric is
  SEER1); tests.
- Check: `heating_upgrade_pm2_euss` is 16.0 on every sample home (it was 15.2).
- Modeled value moves: heat-pump upgrade cost up about $754 per home (0.8 SEER x $594.74 x
  1.5 x the 2023-to-2025 CPI ratio), and with it net capital, NPV, adoption and
  cost-share rebates. MP3/MP4 carry no SEER2 column, so they do not move. Measured
  nationally (patches 01-09 run): $14,957.16 to $15,711.23 (+$754.07); PA +$754.07.

### 08 -- Backup furnace cost (G7, R1, P5, P8)

- Patch: `08_G7_backup_furnace_cost.patch`
- Files: `utils/column_names.py` (`COST_TYPE_BACKUP_FURNACE = 'backupFurnace'`, so the
  column is `mp{mp}_heating_backupFurnace_installed_cost_{scenario}`);
  `remdb_v4_installed_cost_utils.py` (new `add_backup_furnace_metrics`: REMDB row
  `furnaces_gas_furnace`, pm1 = `size_heat_pump_backup_primary_k_btu_h` x 1000, pm2 = the
  parsed AFUE fraction fed directly, checked to lie in (0.5, 1.0]);
  `calculate_equipment_installation_costs.py` (new
  `calculate_backup_furnace_installed_cost`, raises for a package that is not dual fuel;
  `_calculate_v4_upgrade` takes a column prefix); `calculate_lifetime_private_impact.
  calculate_capital_costs` (adds the furnace cost to the heating installation cost for a
  dual-fuel package; a blank furnace cost in the calculation raises); tests.
- Check: every sample home has a furnace cost, no blank; AFUE fed = 0.925 or 0.95; mean
  close to the $4,112 preview; NPV identity still 0 violations; MP3/MP4 raise if asked
  for a furnace cost and ignore a furnace column.
- Risks: the rebate base is unchanged (P5: the furnace is not a rebate measure).
- Modeled value moves: total and net capital up by the furnace cost (mean about $4,112),
  NPV down by the same, adoption down. Measured nationally with 07 (patches 01-09 run):
  furnace mean $4,112.21 (median $4,069, range $3,646-$6,567); mean total capital
  $10,082 to $14,864; adoption heatingLCC_coolingLCC_unsub 11.19% to 3.27%, _sub 60.23%
  to 29.04%, _sub_june2026 12.42% to 4.71%.

### 09 -- NB2: the furnace cost step in the scenarios notebook

- Patch: `09_NB2_scenarios_notebook_backup_furnace_cost.patch`; notebook change `02`.
- Needs 08 first (imports `add_backup_furnace_metrics`,
  `calculate_backup_furnace_installed_cost`, `COST_TYPE_BACKUP_FURNACE`,
  `is_dual_fuel_package`).

### 10 -- June 2026 rebates for a dual-fuel retrofit (G8, R3, R4, D8)

- Patch: `10_G8_dual_fuel_rebate_fuel_gates.patch`
- Files: `constants.py` (`dual_fuel_passes_fuel_gates: True` in both
  `REBATE_RULE_CONFIG` vintages, documented; the Program Notice 26-3 note beside the
  config); `determine_rebate_eligibility_and_amount.calculate_rebate_program` (both fuel
  gates are off for a dual-fuel package where the config says so; everything else
  unchanged); `visuals_adoption_dotplot.py` (`check_june2026_fossil_rule` defaults to
  None: rule 2 applies unless the package is dual fuel);
  `scripts/verify_june2026_rebate_fossil_gate.py` (loops over every loaded package;
  reversed HEEHR check for dual fuel); tests.
- Check: June 2026: fossil-baseline dual-fuel homes in a participating state get HEEHR at
  or below 150% AMI and HOMES above it at 20%+ savings; South Dakota $0; the 2024 columns
  identical with and without the rule (test).
- Risks: 2022.1.1 packages are untouched (none is dual fuel), so June 2026 HOMES stays
  electric-gated for them, as CLAUDE.md lists as deferred.
- Modeled value moves: `_sub_june2026` rebate, NPV and adoption for mp=5. Measured
  nationally (patches 01-11 run): June 2026 recipients 5,732 to 121,432 rdu; June 2026
  adoption equals the `_sub` adoption in every scope (heatingLCC_coolingLCC 4.71% to
  29.04%). For this package the R3 rule is the 2024 rule, so the two vintages agree
  except for the per-vintage half-cent rounding ($424 nationally). Nothing else moved.

### 11 -- NB3: the June 2026 check in the scenarios notebook

- Patch: `11_NB3_scenarios_notebook_june2026_check.patch`; notebook change `03`.
- Needs 06 (for `is_dual_fuel_package`). Without it the scenarios notebook's cell 30
  stops a 2025.1 run once 10 is in, because it asserts fossil baselines get $0.

### 12 -- 2025.1 peak demand columns get their own names (G4, P7)

- Patch: `12_G4_peak_columns_winter_summer.patch`
- Files: `utils/column_names.py` (new `PEAK_ELECTRICITY_PERIODS` and
  `create_peak_electricity_col(prefix, end_use, release, savings=False)`);
  `process_euss_data.py` (`df_enduse_refactored` and `df_enduse_compare` name the four
  electric-peak columns and the two savings columns through it); `export_tepper_csv.py`
  (the Tepper peak list follows the release); tests.
- Names: 2022.1.1 unchanged (`base_peak_electricity_heating_kw`, ...); 2025.1
  `base_peak_electricity_winter_kw`, `base_peak_electricity_summer_kw`,
  `mp{mp}_peak_electricity_winter_kw`, `mp{mp}_peak_electricity_summer_kw` and the
  `_savings` pair. The thermal-load peaks (`..._peak_load_..._kbtu_hr`) keep their names.
- Risk: `grid_impact/build_parcel_frame.py` still lists the 2022.1.1 peak names as
  required columns; grid impact for 2025.1 is deferred, so it was left alone. Fix it with
  the 2025.1 grid-impact work.
- Modeled values: none move (names only).

### 13 -- Tepper household export: dual-fuel and panel columns

- Patch: `13_tepper_export_dual_fuel_and_panel_columns.patch`
- Files: `utils/export_tepper_csv.py` (`build_household_column_list`: for a dual-fuel
  package the parsed ratings `upgrade_hp_seer2`, `upgrade_hp_seer1`, `upgrade_hp_hspf2`,
  `upgrade_hp_hspf1`, `upgrade_backup_fuel`, `upgrade_backup_afue`,
  `upgrade_switchover_f`, the furnace size `size_heat_pump_backup_primary_k_btu_h` and the
  furnace cost (beside the heat-pump cost); for 2025.1 the four panel columns); tests.
- Check: main file 169 + 13 = 182 columns after bldg_id for mp=5 (8 ratings and size, 1
  furnace cost, 4 panel); MP3/MP4 lists unchanged.
- Modeled values: none move.

### 14 -- NB4: a title for package 5 in the main notebook

- Patch: `14_NB4_main_notebook_mp5_title.patch`; notebook change `04`.
- Without it map and dot-plot titles read "MP5".

### 15 -- Small items: applicability must be true/false; no 2022.1.1 weight as a default

- Patch: `15_small_items_applicability_check_and_weight.patch`
- Files: `process_euss_data.py` (new `require_true_false`, used for the applicability
  filter in `load_and_filter_upgrade` and the `mp{mp}_resstock_applicable` column;
  refuses text, blanks and numbers, which `.astype(bool)` would have turned into True);
  `visuals_adoption_dotplot.py` (`scaling_factor` defaults to None: `prepare_plot_data`
  reads the frame's one weight, `build_econ_plot_df` has no fallback weight); tests.
- Risk: a 2022.1.1 CSV whose applicability column came back as text would now stop at
  load instead of loading. It reads as booleans in the files used to date; this could not
  be run here.
- Modeled values: none move.

## Notebook changes, in order

Replay each file with the notebook-cell-edit skill, from the repo root:

```bash
python .claude/skills/notebook-cell-edit/scripts/edit_notebook_cells.py check cmu_tare_model/docs/cloud_run/notebook_changes/NN_<name>.json
python .claude/skills/notebook-cell-edit/scripts/edit_notebook_cells.py apply cmu_tare_model/docs/cloud_run/notebook_changes/NN_<name>.json
git diff --stat -- '*.ipynb'
```

The changes files name cells and lines as the notebooks stand at `b17d7af`, and later
files assume the earlier ones are applied. Close and reopen any VS Code tab of the
notebook after each one.

| # | Notebook | Cell(s) | What and why | Patch |
|---|---|---|---|---|
| 01 | `model_scenarios/tare_run_simulation_v3_0.ipynb` | 15 (last line replaced by itself plus 113 lines) | NB1: run-and-export block for package 5, a line-for-line copy of the MP4 block (same steps, exports and `verify_cost_scenario_columns` call), guarded by `if 5 in VALID_MENU_MPS:`. It sits in the MP4 export cell only because the skill cannot add a cell; it can be moved into two cells by hand. | `04` |
| 02 | `model_scenarios/tare_scenarios_v3_0.ipynb` | 13 (imports), 16 (Step 5 after the cooling replacement cost), 17 (cost columns copied) | NB2: price the backup furnace for a dual-fuel package and copy its cost column onto the home table. For other packages the step is skipped and the column list entry finds nothing. | `09` |
| 03 | `model_scenarios/tare_scenarios_v3_0.ipynb` | 30 | NB3: the June 2026 check. For a dual-fuel package fossil baselines must be funded; for every other package they must get $0 (unchanged); new check 3: South Dakota gets $0 for every package. | `11` |
| 04 | `tare_model_main_v3_0.ipynb` | 9 | NB4: `HEATING_MP_SUBTITLES` gets `5: 'Dual-fuel heat pump with gas backup furnace'`, with a note that 3 and 4 there are the 2022.1.1 packages. | `14` |

Not changed in any notebook (left for the researcher): the main notebook's first markdown
cell still describes the 2022.1.1 MP3/MP4 analysis, cell 33 names the 2022.1.1 Athena
table (grid impact, deferred), and the scenarios notebook's cell 30 docstring still
describes the 2022.1.1 rule first.

## Proposed CLAUDE.md text (not applied; CLAUDE.md was not edited)

Under "Rebate Policy Scenarios", after the fuel-gate bullets:

> **Program Notice 26-3.** DOE has released new guidance in Program Notice 26-3. The
> researcher has not yet reviewed how it differs from Program Notices 26-1 and 26-2 or
> how it affects this code. The rebate rules modeled here are the ones documented in
> this file.
>
> **Dual fuel (2025.1 Upgrade 05).** A dual-fuel retrofit keeps a gas furnace as the heat
> pump's backup, so it removes no fossil heating system and passes both June 2026 fuel
> gates (`dual_fuel_passes_fuel_gates` in `REBATE_RULE_CONFIG`): HEEHR at or below 150%
> AMI for every baseline fuel (D8), HOMES fuel-neutral above it (R3). Caps, cost shares,
> income routing, savings tiers and the South Dakota gate are unchanged. The furnace cost
> is not part of the rebate base. Ask `is_dual_fuel_package(menu_mp)`
> (`utils/measure_packages.py`); never test a package number alone.

Under "Hard-coded Values" or "Both releases must run":

> The release is set by the environment variable `TARE_RESSTOCK_RELEASE`
> (`'2022.1.1'` when unset) before anything imports `cmu_tare_model`. Run the notebooks
> with no keyboard input with `python scripts/run_tare_notebooks.py --release 2025.1
> --skip-grid-impact` (add `--state PA` for one state); it answers Y / N / 42003, stops at
> the first failing cell of any notebook, and checks the NPV identity, orderings and
> adopter counts afterwards.

Under "Capital cost" / Documented limitations:

> 14. The dual-fuel package's heat pump is rated in SEER2 (15.2); it is priced at SEER1
>     16.0 (SEER1 = SEER2 / 0.95; `SEER2_PER_SEER1` in `constants.py`, PROVISIONAL until
>     checked against DOE Appendix M1). Its backup gas furnace is priced with the REMDB
>     `furnaces_gas_furnace` row at the backup's own size and rated AFUE (0.925 or 0.95)
>     and added to capital cost.

Under "Column Naming Conventions":

> 2025.1's electric-peak columns are `..._peak_electricity_winter_kw` /
> `..._summer_kw` (ResStock's winter and summer maximum daily peak), never the 2022.1.1
> `..._heating_kw` / `..._cooling_kw` (peak while heating or cooling runs). Build them
> with `create_peak_electricity_col`.
