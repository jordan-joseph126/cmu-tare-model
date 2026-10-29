# TARE Sample Switches -- Plan for Tamar

Written from `main` (commit `d08004a`, 27 Oct 2026). Each session checks your branch first and tells you anything that differs.

## What and why

The Boulder and Larimer grid impact study needs homes that TARE leaves out of its study sample:
- homes with **no air conditioner (AC)**;
- homes that **already have a heat pump**.

You'll add one switch for each in `constants.py`. Both start as `False`. With both off, no result may change; that is how you show the change is safe. The work stays on your branch, `TM_dev_load_data_MERGE_Oct2026`.

## Before you start (once)

- Save the files from Jordan: this plan and the three prompts in `cmu_tare_model/docs/plans/`, and `compare_runs.py` in `scripts/`.
- Have the Python environment (`ENVIRONMENT_SETUP.md`), the input data (README, section 1.4) and AWS access for grid impact (`cmu_tare_model/docs/BSQ_AWS_SETUP.md`).
- Optional: put your own preferences for Claude, such as "explain each change in plain language", in a file named `CLAUDE.local.md` at the top of the repo, and add the line `CLAUDE.local.md` to `.gitignore`. Don't put them in `CLAUDE.md`: Jordan shares and edits that file.

## How to run a session

1. In the VS Code terminal: `git switch TM_dev_load_data_MERGE_Oct2026`, then `git pull`.
2. Open a new Claude Code tab (Command Palette, then "Claude Code: Open in New Tab").
3. Click the model name under the prompt box: choose Opus 5.5 and set Effort to High.
4. Set the mode to **Plan**, also under the prompt box.
5. Paste the whole prompt file and press Enter.
6. Answer its questions. Read the plan it opens; you can add comments to it.
7. Approve with **"Yes, manually approve edits"**. Don't pick "use auto mode": it makes changes without showing you.
8. Accept or reject each change. Ask "why?" whenever you like.
9. Run the commit commands it gives you. Claude never commits for you.

If a session stops partway, open a new tab and paste: "Read the plan and the Session N prompt in `cmu_tare_model/docs/plans/`, check git status, and continue where the last session stopped."

## The three sessions

1. **Branch sync and baseline.** Brings `main` and any missing work into your branch, adds `compare_runs.py`, and records a Colorado baseline run.
2. **Homes with no AC.** Adds `INCLUDE_HOMES_WITHOUT_COOLING`.
3. **Homes with a heat pump.** Adds `INCLUDE_EXISTING_HEAT_PUMPS`, and settles how these homes are credited and which rebates they get.

Each switch is checked the same way. Switch off: a Colorado run matches the baseline (`compare_runs.py` reports `identical`). Switch on: the runner's checks pass.

When Jordan's `main` changes, he'll tell you; merge it in as in Session 1.

## Log

Each session fills in its row.

| Session | Date | Commit(s) | Run stamps | Notes and decisions |
|---|---|---|---|---|
| 1 | 28 Oct 2026 | `2f55d9c` plan files; `2424041` merge of main `d08004a`; `6f055b0` CO price override; `74d45ad` notebook print fix and price override comments; `15bffb0`, `aa28449` CO CSV untracked and ignored; `1a06910` compare_runs.py and test | `2026-10-28_14-22` baseline: 2022.1.1, CO, grid impact on, 50 of 50 checks passed (log `baseline.log`) | Sessions 2 and 3 must use the same settings. CO electricity 12.3 cents/kWh (`CASE_STUDY_ELECTRICITY_PRICE_CENTS_PER_KWH`; EIA price 15.85). Kept on purpose: `'Electricity ASHP'` in the heating list (Session 3's switch-off run must allow for it), wider housing types (mobile homes, 2-4 unit multi-family), `VERBOSE = True`, `TEST_FIPS`, the old v3_0 and v2_3 notebooks. Grid impact uses `CUSTOM_WEIGHTING = True` (78 of 169 matched buildings in the sample, 22,277 parcels), so `--fips` is not used. Sample: 3,406 rdu (824,698 homes) in 62 counties. Tests: 421 passed, 1 skipped. compare_runs.py also counts a file found in only one run as a difference. County shapefile missing locally, so the adoption map is skipped. |
| 2 | | | | |
| 3 | | | | |
