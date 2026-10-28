# TARE Session 3: switch for homes that already have a heat pump

Read `cmu_tare_model/docs/plans/sample_switches_plan.md` first; CLAUDE.md has the project rules. This prompt was written from `main` at `d08004a` (27 Oct 2026): check my branch and tell me anything that differs. Explain each change in plain language. I make every commit.

**Goal.** Add `INCLUDE_EXISTING_HEAT_PUMPS = False` to `constants.py`, following the pattern of `INCLUDE_HOMES_WITHOUT_COOLING`. When True, homes that already have a heat pump join the study sample. When False, no result may change.

CLAUDE.md says to exclude these homes ("Existing-ASHP homes -- resolved: exclude", and its line in Known Anti-Patterns). On my branch, change both to "only when the switch is on". That is allowed here.

**Watch out for:**
- Two lists keep these homes out: the heating list leaves out `'Electricity ASHP'` and the cooling list leaves out `'Heat Pump'`. Also update `identify_valid_homes` and the sample funnel (`study_sample.py`).
- Double credit: `_assign_replacement_row_id` would price one heat pump as both the old heater and the old AC.
- ResStock 2022.1.1 has only ducted heat pumps. ResStock 2025.1 uses other labels, which the cost code has no rows for.
- Never edit `validation_framework.py`.

**Also:** rename `ELECTRIC_RESISTANCE_BASELINE` to `ELECTRIC_BASELINE`, and fix its comment, in its own commit. It tests the fuel `Electricity`, so it covers heat pumps too, not only resistance heat.

**Ask me, each with a recommendation:**
1. How to credit an existing heat pump. Recommend: once, as the heating credit, with a $0 cooling credit.
2. Which rebates a heat-pump home gets: 2024 and June 2026 HEEHR and HOMES, and my local rebate ($9,000 to every sample home). Show what the code does today. Jordan's reading: the rules limit only fuel switching, so these homes stay eligible. Don't change the June 2026 HOMES fuel test unless I decide to. Check whether these homes reach the 20% savings floor HOMES requires.
3. Whether a ResStock 2025.1 run may use the switch. Recommend stopping it with a clear message.
4. Heating systems with no cost data (wall or floor furnaces, shared heating). Walk me through how a heating system gets a cost row, and add a plan for them to the plan file. Don't build it.

**Done when:**
- the tests pass, including new ones for a heat-pump home;
- with both switches off, a Colorado run matches my baseline (`scripts/compare_runs.py`);
- with both on, the runner's checks pass, and you tell me how many of my 169 buildings are kept;
- CLAUDE.md (the rule above and Limitation 8) and the plan's Log are updated.

Separate commits for the switch, the rename and any rebate change; I make them. Make a plan first.
