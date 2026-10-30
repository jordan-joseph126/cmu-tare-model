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


## Plan: heating systems with no cost data (not built)

Written 30 Oct 2026 in Session 3. A plan only; nothing here is in the code.

**Who is left out.** Nationally (2022.1.1), 28,741 rdu (6.96M homes) have a wall
or floor furnace and 1,053 rdu (255,000 homes) have shared heating. In the
Boulder and Larimer case study, 6 of the 169 matched buildings have a gas wall
furnace. With `INCLUDE_ALL_HOMES_IN_GRID_IMPACT` on, their load already counts in
the grid totals; they just never adopt.

**How a heating system gets a cost row today.** `_assign_replacement_row_id`
(`remdb_v4_installed_cost_utils.py`) takes the first line that fits: oil
boiler, gas or propane boiler, then by fuel alone (propane, fuel oil and natural
gas to `furnaces_gas_furnace`, electricity to `electric_baseboard_default`).
The REMDB table has no wall furnace row and no shared heating row. Because the
fuel lines do not look at the equipment type, a wall furnace let into the sample
would be priced as a ducted gas furnace without any warning. Only the heating list
in `constants.py` keeps these homes out.

**Plan for wall and floor furnaces.**
1. Add a switch, `INCLUDE_WALL_FLOOR_FURNACES = False`, like the other two:
   2022.1.1 only, and results unchanged with it off.
2. Give wall and floor furnaces their own line in `_assign_replacement_row_id`,
   before the fuel lines, so they never fall through to the gas furnace row.
3. Credit for the old wall furnace: choose one.
   - (a) $0, like the old AC in a no-AC home. Needs no new cost data. It gives
     the lowest NPV, because a wall furnace does cost something to replace.
     Recommended first.
   - (b) Priced on the gas furnace row (electric ones on baseboard). This
     overstates the credit, since a wall furnace costs much less than a
     ducted furnace, so it gives the highest NPV. Use it only as the other end
     of a range.
   - (c) A real wall furnace cost from another source, or advice from NLR's
     REMDB researchers (the same question as Limitation 16).
4. The new heat pump: most of these homes have no ducts, so they would be priced
   on the non-ducted multi-zone row. Check `hvac_has_ducts` for them first.
5. Tests for a wall furnace home with the switch on and off; a Colorado run
   with the switch off must match the baseline (`compare_runs.py`).

**Plan for shared heating.** Keep it out of the sample. One system heats
several homes, so a per-home replacement cost and a per-home adoption decision
don't make sense. The grid switch already counts these homes' load.

**Safety fix to consider separately.** Make the fuel lines in
`_assign_replacement_row_id` stop with an error for any heating type that has no
row of its own, rather than pricing it on its fuel alone. With today's heating list
this changes no result.


## Log

Each session fills in its row.

| Session | Date | Commit(s) | Run stamps | Notes and decisions |
|---|---|---|---|---|
| 1 | 28 Oct 2026 | `2f55d9c` plan files; `2424041` merge of main `d08004a`; `6f055b0` CO price override; `74d45ad` notebook print fix and price override comments; `15bffb0`, `aa28449` CO CSV untracked and ignored; `1a06910` compare_runs.py and test | `2026-10-28_14-22` baseline: 2022.1.1, CO, grid impact on, 50 of 50 checks passed (log `baseline.log`) | Sessions 2 and 3 must use the same settings. CO electricity 12.3 cents/kWh (`CASE_STUDY_ELECTRICITY_PRICE_CENTS_PER_KWH`; EIA price 15.85). Kept on purpose: `'Electricity ASHP'` in the heating list (Session 3's switch-off run must allow for it), wider housing types (mobile homes, 2-4 unit multi-family), `VERBOSE = True`, `TEST_FIPS`, the old v3_0 and v2_3 notebooks. Grid impact uses `CUSTOM_WEIGHTING = True` (78 of 169 matched buildings in the sample, 22,277 parcels), so `--fips` is not used. Sample: 3,406 rdu (824,698 homes) in 62 counties. Tests: 421 passed, 1 skipped. compare_runs.py also counts a file found in only one run as a difference. County shapefile missing locally, so the adoption map is skipped. |
| 2 | 29 Oct 2026 | '2b46aea'  Implemented INCLUDE_HOMES_WITHOUT_COOLING swtich to include homes without AC in analysis, updated the relevant cost calculations (negative operational savings for homes that gain AC) | `2026-10-29_14-58` switch off: identical to the baseline (compare_runs.py, log `switch_off.log`); `2026-10-29_15-16` switch on: 50 of 50 checks passed (log `switch_on.log`) | Added `INCLUDE_HOMES_WITHOUT_COOLING` (default False). On: homes with no AC join the sample with a $0 old-AC cost and credit, and their cooling savings stay negative; a blank AC cost for a home that has an AC still stops the run. The funnel's AC step is named `central_room_or_no_ac` when on. The switch works on 2022.1.1 only; on 2025.1 the run stops with an error. Switch on: 5,725 rdu (1,386,200 homes) in 63 counties; 153 of 169 matched buildings in the sample (35,486 parcels). Tests: 426 passed, 1 skipped. |
| 3 | 29–30 Oct 2026 | `5158b7b` INCLUDE_ALL_HOMES_IN_GRID_IMPACT switch (committed with both switches True); next: switches back to False; CLAUDE.md and plan docs; ELECTRIC_BASELINE rename | `2026-10-29_22-17` switch off: identical to the baseline (compare_runs.py, rechecked 30 Oct; log `all_homes_off.log`); `2026-10-29_22-38` both switches on: 50 of 50 checks passed (log `all_homes_on_2.log`; `all_homes_on.log` is a broken log, ignore it) | Changed from the prompt: instead of `INCLUDE_EXISTING_HEAT_PUMPS` (heat-pump homes join the sample), added `INCLUDE_ALL_HOMES_IN_GRID_IMPACT` (default False). Why: the grid study needs every building's load, not an NPV for heat-pump homes. On: every home counts in the grid impact load totals; homes outside the sample (existing heat pumps, wall or floor furnaces) keep their own load and never adopt, in the economic or the 100% case. Sample, costs, NPV and rebates unchanged, so the prompt's questions on crediting a heat pump and on rebates don't arise. Read in the grid impact cell of `tare_model_main_v3_1.ipynb`; works with CUSTOM_WEIGHTING on or off. 2022.1.1 only; a 2025.1 run with it on stops with an error. Both on: 169 of 169 matched buildings kept (153 in the sample, 16 outside: 10 existing heat pump, 6 gas wall furnace), 37,663 parcels. Heat-pump homes stay out of the sample through the cooling list (`'Heat Pump'` not in it); `'Electricity ASHP'` stays in the heating list. Tests: 431 passed, 1 skipped. Plan for heating systems with no cost data: below. |
