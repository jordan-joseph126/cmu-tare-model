# Decisions taken -- ResStock 2025.1 dual fuel (mp=5) cloud session, 6-7 Oct 2026

The session ran unattended. Where neither NEXT_STEPS (D1-D10) nor the researcher's
6 Oct answers (R1-R9) settled a choice, the default below was taken and the run
continued. Each entry says what was chosen, why, how to reverse it, and which
results depend on it. None of these is a reference value.

## Program Notice 26-3 (R4)

DOE has released new guidance in Program Notice 26-3. The researcher has not yet
reviewed how it differs from Program Notices 26-1 and 26-2 or how it affects this
code. The rebate rules modeled here are the ones documented in CLAUDE.md. No rule
was changed because of 26-3, and nothing here guesses what it says.

## Provisional decisions named in the session prompt

### P1 -- SEER2 and HSPF2 conversion factors

- **Chosen:** SEER1 = SEER2 / 0.95 and HSPF1 = HSPF2 / 0.85, as two named constants
  in one place. SEER2 15.2 becomes SEER1 16.0. Only SEER feeds the REMDB heat-pump
  regression (its pm2 metric is SEER1); HSPF is converted for the record only. The
  option string writes `SEER 15.2`; it is read as SEER2, as the measure documentation
  says.
- **Why:** agrees with what the repo already records (the ENERGY STAR floor written
  as 16.0 SEER1 in `process_euss_data.py`, the guide's description of SEER2 15.2 as
  the ENERGY STAR minimum, and `constants.py` equating SEER2 14.3 with about SEER1
  15). The DOE Appendix M1 text itself was not checked (ecfr.gov is not reachable
  from the cloud machine).
- **Reverse:** change the two constants. Setting the SEER factor to 1.0 restores the
  unconverted 15.2.
- **Depends on it:** the heat-pump upgrade cost and everything downstream (net
  capital, NPV, adoption, rebate amounts capped by cost share).

### P2 -- the cooling filter stays (D4 superseded)

- **Chosen:** follow CLAUDE.md, not NEXT_STEPS: the study sample is homes with a
  central or room AC of their own. For mp=5 this leaves out 15,017 applicable rdu
  with no AC. No code was changed for this.
- **Reverse:** see CLAUDE.md Limitation 11 and the 'TODO (no-AC homes)' points.

### P3 -- release switch through an environment variable

- **Chosen:** `constants.py` reads the release from `TARE_RESSTOCK_RELEASE`,
  defaulting to `'2022.1.1'`; an unknown value raises `ValueError`. The runner sets
  the variable before anything imports `cmu_tare_model`.
- **Why:** functions copy `RESSTOCK_RELEASE_THIS_RUN` as a default argument when
  their module is imported, so the release must be fixed before the first import.
  An environment variable does that without editing a file, and still declares the
  release in one place per run.
- **Reverse:** delete the two lines that read the variable and set the constant by
  hand as before. With the variable unset, behavior is exactly as before.

### P5 -- the furnace cost is not part of the rebate base

- **Chosen:** HEEHR and HOMES keep reading the heat-pump upgrade cost column. The
  backup furnace is a fossil appliance, not a rebate measure.
- **Reverse:** add the furnace cost to the project cost the rebate functions read.

### P7 -- names for the 2025.1 peak columns

- **Chosen:** `base_peak_electricity_winter_kw`, `base_peak_electricity_summer_kw`,
  `mp{mp}_peak_electricity_winter_kw`, `mp{mp}_peak_electricity_summer_kw`, and the
  two savings columns with `_savings` appended. The 2022.1.1 names and values are
  unchanged.

### P8 -- name of the furnace cost column

- **Chosen:** built with `create_cost_col`, with a new cost type for the backup
  furnace, so the name follows `mp{mp}_heating_<cost type>_installed_cost_{scenario}`.

### P10 -- loop states

- **Chosen:** PA, MN (cold) and FL (warm), the three the 6 Oct probe measured.

### P13 -- run outputs stay off the branch

- **Chosen:** the Tepper files and every other run output stay on the cloud machine.
  What is committed is a review summary of them and the commands that reproduce them
  on the researcher's machine.

## Decisions added during the session

### D-S1 -- Python 3.11 for the virtual environment

- **Chosen:** the venv was made with `/usr/bin/python3.11` (3.11.17), not the
  machine's default `python3` (3.13.16).
- **Why:** pandas 2.1.4 (pinned) has no build for Python 3.13, and the researcher runs
  3.11.13. With 3.11 the pinned set installed exactly, so the run uses the
  researcher's package versions.
- **Reverse:** not needed on the researcher's machine.

### D-S2 -- No commits; one patch file per task instead

- **What happened:** the session's first `git commit` was refused by the cloud
  environment's permission check ("Git Destructive"), and a following `git status` was
  refused too. The session did not try to commit, push, or reach git another way.
- **Chosen:** every change stays in the working tree, as CLAUDE.md's "Committing is the
  researcher's job" rule describes, and each port task is also saved as its own patch,
  `patches/NN_<task>.patch`, made with plain `diff -u` against a snapshot of the branch
  taken before the first edit, refreshed after each task. The deliverables and patches
  were sent to the researcher as a file bundle so they survive the cloud machine.
- **Reverse / next step:** apply the patches in order and commit each with its own
  message (IMPLEMENTATION_GUIDE). If commits are wanted from a cloud session, the
  environment's permission settings must allow `git commit` and `git push` on this branch.

### D-S3 -- How the runner runs a notebook

- **Chosen:** IPython's own `%run -i` on the main notebook, in IPython's terminal shell,
  with the shell's notebook runner (`safe_execfile_ipy`) replaced for this shell by one
  that runs the same code cells in the same namespace and stops at the first failing
  cell of any notebook with an error naming the notebook and the cell.
- **Why:** with IPython's own runner a failing cell inside the simulation notebook comes
  back to the main notebook's question loop (`try/except Exception` around cell 3's
  `input()`), which prints "Please try again" and asks again -- the run would loop, not
  stop. The replacement raises a `BaseException` that loop cannot catch. The base
  `InteractiveShell` cannot run `%matplotlib inline` (`NotImplementedError: Implement
  enable_gui in a subclass`); the terminal shell can, and its keyboard loop is never
  started.
- **Reverse:** none needed; the notebooks run unchanged in Jupyter.

### D-S4 -- National runs from a frozen copy of the tree

- **Chosen:** each national run ran from a copy of the working tree (data folders linked,
  not copied) so code edits made while it ran could not leak into it. The runner prints
  which copy of `cmu_tare_model` it imported.
- **Reverse:** not applicable on the researcher's machine.

### D-S5 -- `load_euss_baseline(filename=...)` keeps its argument but only the default

- **Chosen:** the argument stays (signature unchanged), but a non-default file name
  raises `ValueError`: the file is now chosen by the release, through
  `load_and_filter_upgrade`. No caller passes a file name.
- **Reverse:** drop the argument in a later clean-up.

### D-S6 -- One dual-fuel helper, in its own module

- **Chosen:** `DUAL_FUEL_PACKAGES_BY_RELEASE` in `constants.py` and
  `is_dual_fuel_package(menu_mp, release=None)` in a new `utils/measure_packages.py`
  (imports only `constants.py`, so every module and notebook can use it without an import
  cycle). `release=None` reads `RESSTOCK_RELEASE_THIS_RUN` when called, not at import.
- **Reverse:** move the function; callers import it by name.

### D-S7 -- Furnace cost type token `backupFurnace`

- **Chosen:** `COST_TYPE_BACKUP_FURNACE = 'backupFurnace'`, so P8's column is
  `mp5_heating_backupFurnace_installed_cost_v4MID`, camel case like the other
  multi-word tokens (`naturalGas`, `heatingLCC`).
- **Reverse:** change the constant; every reader builds the name from it.

### D-S8 -- Only a gas backup furnace is priced

- **Chosen:** `BACKUP_FURNACE_ROW_BY_FUEL = {'Natural Gas': 'furnaces_gas_furnace'}`.
  A propane or oil backup (none is published) stops the run instead of being priced as
  gas, because choosing a proxy row for it is a pricing decision. The replacement cost
  already uses the gas furnace row as a proxy for propane and oil furnaces.
- **Reverse:** add rows to the mapping once a proxy is decided.

### D-S9 -- The dual-fuel rule is `dual_fuel_passes_fuel_gates` in `REBATE_RULE_CONFIG`

- **Chosen:** a per-vintage config flag (True in both vintages; it changes nothing in 2024,
  which has no fuel gate) read together with `is_dual_fuel_package`, rather than a set of
  exempt packages beside `get_rebate_eligible_mps`. The gates' reason (no fossil system
  removed) belongs to the rule; which packages are dual fuel belongs to the package list.
- **Reverse:** set the flag False in the June 2026 vintage to go back to Stage A's rule.

### D-S10 -- The dot plot decides rule 2 for itself

- **Chosen:** `check_june2026_fossil_rule` defaults to None in
  `adoption_reconciliation_table` and `build_econ_plot_df`: None applies the rule unless
  the package is dual fuel. An explicit True or False still wins.

### D-S11 -- What the Tepper household export gains for 2025.1 mp=5

- **Chosen:** for a dual-fuel package, the parsed ratings (SEER2 and the SEER1 it is
  priced at, HSPF2 and HSPF1, backup fuel, AFUE, switchover temperature), the backup
  furnace's size and its installed cost; for 2025.1, the panel service rating and the
  three panel-constraint flags; the peak columns under their winter/summer names. Nothing
  removed. The 2022.1.1 lists are unchanged.
- **Why:** the furnace cost and the inputs it is priced from are needed to check the
  capital cost; the panel flags are 2025.1's new reporting-only fields named in the
  session prompt's review list.
- **Reverse:** drop the groups in `build_household_column_list`.

### D-S12 -- The package 5 title

- **Chosen:** `'Dual-fuel heat pump with gas backup furnace'`, in the style of the MP3/MP4
  titles (plain words, no ratings).

### D-S13 -- `require_true_false` accepts booleans stored as objects

- **Chosen:** a column of Python or numpy booleans with object dtype passes; text, blanks
  and numbers stop the load. The 2022.1.1 CSV read could give object dtype if a column
  mixes types; only a real True/False is safe to convert.

### D-S14 -- Three national Stage B runs from patch-built trees

- **Chosen:** instead of re-running between edits, the national runs "after capital cost",
  "after rebates" and "at the end" were made from three copies of the original tree with
  patches 01-09, 01-11 and 01-15 applied. That also shows the patches reproduce the
  working tree: the 01-15 copy is byte-identical to it.

## Observation for the researcher (no decision taken)

### O-1 -- Under R3, June 2026 equals 2024 for the dual-fuel package

With both June 2026 fuel gates off for a dual-fuel retrofit (HEEHR at or below 150% AMI
whatever the fuel, D8; HOMES fuel-neutral, R3), the June 2026 rules are the 2024 rules:
same programs, caps, cost shares, income routing, savings tiers and state gate. The
final national run confirms it: the `_sub` and `_sub_june2026` rebates pay the same
121,432 rdu, the national totals differ by $424 (each vintage keeps its own half-cent
rounding, `heehr_python_round`), and the nine adoption rates match pairwise (for example
29.04% for heatingLCC_coolingLCC). If June 2026 was meant to differ for dual fuel (for
example a HOMES rule that still keys on fuel, or Program Notice 26-3), that is the place
to change it: `dual_fuel_passes_fuel_gates` in `REBATE_RULE_CONFIG`.
