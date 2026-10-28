# TARE Session 1: branch sync and baseline

Read `cmu_tare_model/docs/plans/sample_switches_plan.md` first; CLAUDE.md has the project rules. This prompt was written from `main` at `d08004a` (27 Oct 2026): check my branch and tell me anything that differs. I'm new to this codebase and to git merges, so explain each step in plain language. Give me the git commands; I run every commit, merge and push.

**Goal.** My branch, `TM_dev_load_data_MERGE_Oct2026`, has everything from `main` and all my own work, plus a baseline run that Sessions 2 and 3 compare against.

**First, ask me** which branches hold my work, and whether I have changes I haven't committed. On 27 Oct these branches had commits mine lacks: `TM_dev_load_data`, `case_study_TM_dev`, `price-updates-TM-dev` and `tamar`. Never change `TM_dev_load_data`: it's my backup.

**Steps to plan:**
1. Commit the plan files in `cmu_tare_model/docs/plans/`.
2. Merge `origin/main` into my branch.
3. Bring in the work I choose. Compare file contents, not only commits: some work was copied by hand. From branches built on the old v2_3 notebook, copy what I keep by hand.
4. Check that `tare_model_main_v3_1.ipynb` has my REBATE_CASE work from the v3_0 notebook. Change notebook cells only with the `notebook-cell-edit` skill.
5. Add `scripts/compare_runs.py` (from Jordan) with a test, in its own commit. Fix three things: its default output folder (a path on Jordan's computer); compare the county Tepper file county by county, since tied counties can swap rows; treat numbers within about 1e-11 as equal.
6. Run the tests and tell me the count.
7. Run the baseline and put its run stamp in the plan's Log:
   `python scripts/run_tare_notebooks.py --release 2022.1.1 --state CO --fips 08013 --log baseline.log`

**Ask me about these, each with a recommendation:**
- My `'Electricity ASHP'` line in `constants.py`. It changes no result, but it changes the saved sample funnel and the `include_heating` column of the Tepper files, so Session 3's switch-off run couldn't match the baseline. Recommend undoing it before the baseline.
- My wider housing types (mobile homes, 2-4 unit multi-family). Are they intended?
- `TEST_FIPS = "08031"` is labeled Boulder, but 08031 is Denver; Boulder is 08013.
- `VERBOSE = True`: keep it?
- The old notebooks next to v3_1 (`v3_0`, `v2_3` and the `_v3_0_EXPORT_` copies): remove them once v3_1 is confirmed?

Make a plan first.
