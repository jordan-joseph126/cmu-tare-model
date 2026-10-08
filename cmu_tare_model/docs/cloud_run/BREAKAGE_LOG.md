# Breakage log -- ResStock 2025.1 dual fuel (mp=5) cloud session, 6-7 Oct 2026

Branch: `cloud/resstock2025-1-mp5`. Every failure met on the way to a full notebook
run, its cause, the fix and the patch that holds the fix. Newest entries at the
bottom.

**No commits were made.** The session's first `git commit` was refused by the
cloud environment's permission check ("Git Destructive"), and a following
`git status` was refused too. The session therefore stopped using git and left
every change in the working tree, as CLAUDE.md's "Committing is the
researcher's job" rule describes. Each change is also saved as its own patch
under `cmu_tare_model/docs/cloud_run/patches/` (numbered, one purpose each), so
it can be reviewed, applied and committed one at a time. Where this log would
give a commit SHA it gives the patch name.

## Environment

- Python 3.11.17 (`/usr/bin/python3.11`), virtual environment at `~/tare-venv`, made
  with `uv venv --seed --python /usr/bin/python3.11`. The machine's default `python3` is
  3.13, which pandas 2.1.4 does not support; 3.11 is the researcher's minor version
  (3.11.13), so the pinned set installed as written (see `DECISIONS_TAKEN.md`, D-S1).
- Pinned set installed without fallback: pandas 2.1.4, numpy 1.26.4, pyarrow 23.0.1,
  scipy 1.16.0, matplotlib 3.10.0, seaborn 0.13.2, geopandas 1.1.3, pyogrio 0.12.1,
  shapely 2.1.2, pyproj 3.7.2, openpyxl 3.1.5. Unpinned: IPython 9.17.1 (researcher:
  9.1.0), nbformat 5.11.1, psutil 7.2.2, pytest 9.1.1, pyparsing 3.3.3.
- `uv pip install -e .` worked. It rewrites the tracked `cmu_tare_model.egg-info/`
  files; those changes were restored with `git checkout` and are never committed.
- Data: the five ResStock 2025.1 files downloaded from OEDI into
  `data/resstock_2025_1/`; all five SHA-256 hashes match the session prompt.

## Test baseline (before any change)

`python -m pytest -q` from the repo root: **352 passed, 1 skipped, 19 warnings**.
Pass and skip counts match the prompt (352 / 1). The prompt saw 5 warnings; the 14
extra are `PyparsingDeprecationWarning`s raised inside matplotlib 3.10.0 by the newer
pyparsing 3.3.3 (an unpinned dependency here). The 5 known
`PytestReturnNotNoneWarning`s from `test_efficiency_floor_refactoring.py` are
unchanged and were left alone (deferred).

## Failures and fixes

| # | Where it stopped | Cause | Fix | Commit |
|---|---|---|---|---|
| 1 | `tare_model_main_v3_0.ipynb`, cell 2 (first runner try) | `%matplotlib inline` raises `NotImplementedError: Implement enable_gui in a subclass` in IPython's base `InteractiveShell` | Runner uses IPython's `TerminalInteractiveShell` (never starts its keyboard loop). Confirmed `%matplotlib inline` then works and `plt.show()` closes figures | `03_notebook_runner` |
| 2 | `tare_model_main_v3_0.ipynb`, cell 6 | NB1: the simulation notebook has no block for package 5, so no `mp5` results were written (`FileNotFoundError: .../retrofit_mp5_results/summary_mp5_fixed_base`) | Added the package 5 run and export block to the simulation notebook, cell 15, mirroring MP4 line for line | `04_NB1_sim_notebook_mp5_block`; notebook change `01` |
| 3 | `tare_model_main_v3_0.ipynb`, cell 9 | G9: `load_euss_baseline()` reads the 2022.1.1 CSV (`FileNotFoundError: .../euss_data/.../baseline_metadata_and_annual_results.csv`) | G9 + G10 together: the KPI loaders read this run's release through `load_and_filter_upgrade` (same scope filters as the run, column-limited read), and every ResStock column name in `adoption_kpis/data_loading.py` comes from the column map | `05_G9_G10_kpi_loaders_and_names` |

## Stage A results

- PA, MN, FL: exit 0, all 20 checks each. PA 6,469 rdu (1,642,503 homes); MN 4,410 rdu
  (1,119,715 homes); FL 2,177 rdu (552,748 homes) -- as the probe measured.
- National (`2026-10-06_18-19`): exit 0, all 20 checks; 11.4 min, peak 10.68 GB in the
  main notebook's Tepper cell (23). Sample, funnel and the three adoption rates match the
  probe exactly (161,983 rdu; 11.19% / 60.23% / 12.42%).

## Stage B log

Stage B changed the model in this order, one patch each, tests added each time (the
suite count is after the change):

| Patch | Item | Tests | Suite |
|---|---|---|---|
| `06_G5_dual_fuel_spec_columns` | G5 | `is_dual_fuel_package`, option-string parse, blank/other string stops | 365 passed, 1 skipped |
| `07_G6_seer2_to_seer1` | G6 | conversion; SEER2 home priced at 16.0; SEER1 package unchanged; non-SEER1 row stops | 370 passed |
| `08_G7_backup_furnace_cost` | G7 | furnace metrics; AFUE read wrongly stops (0.00925 and 92.5); propane backup stops; REMDB formula; MP3/MP4 refuse; capital cost adds the furnace only for dual fuel; blank stops | 380 passed |
| `09_NB2_...` | NB2 | notebook change 02 | -- |
| `10_G8_dual_fuel_rebate_fuel_gates` | G8 | dual fuel funds fossil under June 2026 with SD $0; 2024 unchanged; heat-pump-only package keeps the gates; dot plot rule 2 skipped for dual fuel | 384 passed |
| `11_NB3_...` | NB3 | notebook change 03 | -- |
| `12_G4_peak_columns_winter_summer` | G4 | 2022.1.1 names unchanged; 2025.1 winter/summer names; Tepper list keeps 2022.1.1 names | 387 passed |
| `13_tepper_export_dual_fuel_and_panel_columns` | Tepper columns | furnace cost and ratings only for a dual-fuel package, beside the heat-pump cost | 388 passed |
| `14_NB4_main_notebook_mp5_title` | NB4 | notebook change 04 | -- |
| `15_small_items_applicability_check_and_weight` | small items | text / blank / number applicability refused; no fallback weight; `prepare_plot_data` reads the weight | 394 passed (tests folder); **399 passed, 1 skipped, 19 warnings** for the whole suite from the repo root |

No failures needed fixing in Stage B: every change passed its tests first time except
two of the session's own new tests, corrected before use (a docstring that quoted
`'out.*'` tripped the "no typed-in ResStock name" test; `build_econ_plot_df` stops on a
missing weight with a `KeyError` from the reconciliation table before its own
`ValueError`, so the test accepts either).

PA, MN and FL with patches 01-15: exit 0, all 20 checks each; furnace cost means $4,153
(PA), $4,285 (MN), $3,902 (FL); heat-pump upgrade cost mean in PA $14,444 against
$13,690 in Stage A (+$754, the expected SEER1 effect).

One command was refused by the cloud's safety check: an `rm -rf` on a path built from a
shell variable, while rebuilding scratch trees. It was rewritten with literal paths (the
directories did not yet exist); nothing outside the scratch folder was involved.

## Stage B national runs

The cloud container restarted once during the first Stage B national run (patches 01-09);
that run was lost after the baseline stage and simply started again. Files on disk,
including the virtual environment, the data and all patches, survived.

| Run | Patches | Exit | Checks | Wall time | Peak (OS) |
|---|---|---:|---|---:|---:|
| After capital cost | 01-09 | 0 | 20 of 20 | 11.6 min | 10.71 GB |
| After rebates | 01-11 | 0 | 20 of 20 | 10.6 min | 10.70 GB |
| Final | 01-15 | 0 | 20 of 20 | 10.6 min | 10.88 GB |

Each ran from a copy of the original tree with the patches applied in order (no failure
applying any patch); the 01-15 copy is byte-identical to the working tree.

## BLOCKED

- **Commits and pushes.** Refused by the cloud environment's permission check (see the
  top of this log). Not retried. Everything is in the working tree and in
  `patches/`; the researcher commits.
- Nothing in the model is blocked.

## Final clean-up

A last scan of every added line found four over 88 characters (one in the runner, three
in new tests); they were wrapped, the three patches holding them were rebuilt, and the
whole chain was re-verified (original tree + patches 01-15 = working tree, byte for
byte). Suite after: 399 passed, 1 skipped, 19 warnings. PA with the final working tree:
exit 0, 20 of 20 checks. No added line is non-ASCII.
