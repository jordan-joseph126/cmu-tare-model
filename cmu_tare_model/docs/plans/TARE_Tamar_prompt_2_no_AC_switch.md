# TARE Session 2: switch for homes with no AC

Read `cmu_tare_model/docs/plans/sample_switches_plan.md` first; CLAUDE.md has the project rules. This prompt was written from `main` at `d08004a` (27 Oct 2026): check my branch and tell me anything that differs. Explain each change in plain language. I make every commit.

**Goal.** Add `INCLUDE_HOMES_WITHOUT_COOLING = False` to `constants.py`. When True, homes with no AC join the study sample. When False, no result may change.

**With the switch on:**
- these homes get $0 for the old-AC replacement cost and credit;
- their cooling savings stay negative, because the heat pump adds cooling they never had;
- a blank AC cost for a home that has an AC still stops the run.

**Watch out for:**
- In ResStock 2022.1.1, the model reads the label `None` as a blank, so no-AC homes have a blank cooling type. In 2025.1 it is the text `'None'`.
- The places to change are tagged `TODO (no-AC homes)`. Also update `identify_valid_homes` and the sample funnel (`study_sample.py`), which stops the run if it disagrees with the sample.
- Once the switch is on, `include_cooling` no longer means "has an AC".
- Never edit `validation_framework.py`.

**Ask me** whether a ResStock 2025.1 run may use the switch. Recommend stopping it with a clear message.

**Done when:**
- the tests pass, including new ones for a no-AC home;
- a Colorado run with the switch off matches my baseline (`scripts/compare_runs.py`);
- a run with it on passes the runner's checks, and you tell me how many of my 169 buildings are kept;
- CLAUDE.md Limitation 11 and the plan's Log are updated.

One commit, which I make. Make a plan first.
