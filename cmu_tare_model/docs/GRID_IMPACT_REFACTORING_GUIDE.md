# Grid Impact Refactoring Guide

**Status: Task 2 of 8 complete. Tasks 3-8 not started.**

This guide exists to hand off the remaining work. It states the objective and gives
step-by-step instructions with pseudocode for each task still open. Full session-by-session
history (what was tried, what broke, what was decided and why) lives in
`cmu_tare_model/docs/SESSION_CHANGELOG_2026-09-17.md` -- this guide does not repeat it.

The code lives in `cmu_tare_model/grid_impact/peak_load_functions.py` (the module with all
the reusable functions) and the grid-impact section of
`cmu_tare_model/tare_model_main_v3_0.ipynb` (the notebook that calls them). A plain-`.py`
mirror of the current notebook, for reading without opening Jupyter, is at
`cmu_tare_model/tare_model_main_v3_0_EXPORT_18Sep2026.py` -- it is read-only; edit the
`.ipynb` and the module, never the export.

---

## Task 2 (done) -- what you're building on

`compute_county_scenario_profile()` and `build_weight_dict_from_mapping()` already support
two parallel weighting modes, selected by a `custom_weighting` flag:

```python
compute_county_scenario_profile(
    df_baseline, df_upgrade, adopter_bldg_ids,
    custom_weighting=False,   # default: today's ResStock-weighted behavior, untouched
    weight_dict=None,         # or: custom_weighting=True, weight_dict={bldg_id: weight}
)
```

Every task below must keep both modes working -- test against `custom_weighting=False` first
(it must stay byte-identical to today), then `custom_weighting=True`. Full detail on how this
works: `SESSION_CHANGELOG_2026-09-17.md`, "Task 2."

---

## Task 3: Add heating- and cooling-season peaks

**Objective.** `compute_county_scenario_profile()`'s returned `peak_dict` currently only has
one peak (the max over all 8,760 hours). Add two more: the max restricted to
Dec/Jan/Feb (heating season) and the max restricted to Jun/Jul/Aug (cooling season) -- for
both `baseline_mw` and `scenario_mw`, and for both weighting modes.

**Do not edit** `plot_demand_panel` or `plot_county_demand_grid` -- they keep showing only
the absolute peak. Its value changing once weighting is added is expected.

**Step 1 -- factor out the month-hour-range logic.** `plot_county_demand_grid` already
builds an hour-of-year lookup for month boundaries (`days_in_month` / `month_start_hours`).
Pull the same math into a small standalone helper in `peak_load_functions.py`, e.g.:

```python
def get_season_hour_ranges() -> dict:
    """Return the hour-of-year range for heating and cooling season.

    Non-leap year, 1-indexed hours matching an 8,760-row profile.
    """
    # Reuse plot_county_demand_grid's days_in_month / month_start_hours logic
    # here instead of duplicating it -- read that function first.
    ...
    return {
        "heating": set(range(dec_start, dec_end)) | set(range(jan_start, mar_start)),
        "cooling": set(range(jun_start, sep_start)),
    }
```

**Step 2 -- extend `peak_dict`.** Inside `compute_county_scenario_profile`, after
`df_profile` is built (right before the existing 8,760-row check), add:

```python
season_hours = get_season_hour_ranges()
heating_mask = df_profile["hour"].isin(season_hours["heating"])
cooling_mask = df_profile["hour"].isin(season_hours["cooling"])

peak_dict["baseline_heating_season_peak_mw"] = df_profile.loc[heating_mask, "baseline_mw"].max()
peak_dict["scenario_heating_season_peak_mw"] = df_profile.loc[heating_mask, "scenario_mw"].max()
peak_dict["baseline_cooling_season_peak_mw"] = df_profile.loc[cooling_mask, "baseline_mw"].max()
peak_dict["scenario_cooling_season_peak_mw"] = df_profile.loc[cooling_mask, "scenario_mw"].max()
```

Keep this working identically for both `custom_weighting` branches -- it operates on
`df_profile`, which is already produced the same way regardless of weighting mode, so no
branching should be needed here.

**Verify:** `custom_weighting=False` absolute peak (`baseline_peak_mw`, `scenario_peak_mw`)
is numerically unchanged from before. All four new season-peak values are plausible
(heating peak > cooling peak for baseline in most counties; a heat-pump retrofit's
`scenario` cooling peak may exceed baseline -- expected, see the "negative cooling savings"
note in the main `CLAUDE.md`).

---

## Task 4: Season-consistency test

**Objective.** Add a test confirming, for every (mp, weighting_mode, scenario) combination:

```
baseline_peak_mw == max(baseline_heating_season_peak_mw, baseline_cooling_season_peak_mw)
scenario_peak_mw == max(scenario_heating_season_peak_mw, scenario_cooling_season_peak_mw)
```

**Where.** Add to `cmu_tare_model/tests/adoption_kpis/test_peak_load_functions.py` --
follow the existing `TestComputeCountyScenarioProfile` class's pattern (synthetic
`_make_hourly_df` fixtures, no AWS connection needed).

**Pseudocode:**

```python
def test_absolute_peak_equals_max_of_seasonal_peaks():
    df_profile, peak_dict = compute_county_scenario_profile(baseline_df, upgrade_df, adopter_ids)
    assert peak_dict["baseline_peak_mw"] == max(
        peak_dict["baseline_heating_season_peak_mw"],
        peak_dict["baseline_cooling_season_peak_mw"],
    )
    assert peak_dict["scenario_peak_mw"] == max(
        peak_dict["scenario_heating_season_peak_mw"],
        peak_dict["scenario_cooling_season_peak_mw"],
    )
```

Then run this against real MP3/MP4 profiles (both weighting modes) via the notebook, not
just synthetic data -- this invariant is expected to hold *exactly*, not approximately,
because every peak in the existing figure falls in January or July, never a shoulder month.

**If it doesn't hold for some real profile:** that's a genuine finding (the true peak fell
in a shoulder month) -- report it, don't silently adjust the test or the code to force a
match.

---

## Task 5: Annual MWh aggregation

**Objective.** Report each scenario's total annual energy (MWh), alongside the peak.

**No legacy code exists to port** (checked; the archived pre-2 Sep module and old exports
have no MWh-aggregation logic). This is new code:

```python
peak_dict["annual_baseline_mwh"] = df_profile["baseline_mw"].sum()
peak_dict["annual_scenario_mwh"] = df_profile["scenario_mw"].sum()
```

(Each row is one hour of MW, so summing MW over 8,760 hourly rows gives MWh directly --
no unit conversion needed.)

**One thing to flag with Jordan before implementing the "internal consistency" check the
original task description asks for:** comparing this annual sum against the peak-MW profile
from the *same* `df_profile` will always match by construction (they're computed from the
same data), so it isn't an independent check the way it may have been intended. Confirm
what real check is wanted -- e.g. a comparison against ResStock's own annual consumption
columns instead -- before building it.

---

## Task 6: Loop across NPV/subsidy scenarios

**Objective.** The adopter-ID cell currently hardcodes one NPV case
(`BASE_CASE_NPV_CASE`). Loop it across all nine cases in `NPV_CASE_CATEGORIES`
(`column_names.py`), nested inside the existing `mp` loop.

**Critical rule: every `find_adoption_column` call must use keyword arguments.** A past
positional call silently routed a figure to the wrong NPV case -- there is no `*` in the
function signature enforcing this in Python, so it's on you to keep doing it by hand.

**Pseudocode** (adapting the existing adopter-ID cell):

```python
from cmu_tare_model.utils.column_names import NPV_CASE_CATEGORIES

adopter_ids_by_mp = {}
adoption_col_by_mp = {}

for mp in selected_mps:
    df_tare = DATAFRAMES_BY_MP[mp][discount_rate]
    adopter_ids_by_mp[mp] = {}
    adoption_col_by_mp[mp] = {}

    for npv_case in NPV_CASE_CATEGORIES:
        adoption_col = find_adoption_column(
            df=df_tare,
            mp=mp,
            cost_scenario=cost_scenario,
            discount_rate_key=discount_rate,
            npv_case=npv_case,          # <-- always keyworded
        )
        adoption_col_by_mp[mp][npv_case] = adoption_col

        county_fips = df_tare["county"].apply(gisjoin_to_fips)
        is_adopter = df_tare[adoption_col] == 1.0

        adopter_ids_by_mp[mp][npv_case] = {}
        for fips, idx in df_tare.groupby(county_fips).groups.items():
            adopter_mask = is_adopter.loc[idx].to_numpy()
            adopter_ids_by_mp[mp][npv_case][str(fips)] = {
                "all_filtered": list(idx),
                "constrained": list(idx[adopter_mask]),
            }
```

Note the shape changes: `adopter_ids_by_mp[mp]` currently maps straight to
`{fips: {...}}`; after this change it maps `{npv_case: {fips: {...}}}`. **Every downstream
cell that reads `adopter_ids_by_mp[mp][some_fips]` needs an `npv_case` key inserted** --
search the notebook for `adopter_ids_by_mp[` to find every call site (the weighting-mode
cell, the profile-computation cell, and `resolve`-style lookups all read from this dict).
Decide whether downstream cells loop over all nine cases too, or default to
`BASE_CASE_NPV_CASE` unless told otherwise -- confirm which with Jordan before changing
every call site, since it affects how much of the rest of the notebook needs to change.

Keep whichever `custom_weighting` mode is active applied consistently across every case in
the loop -- don't let it vary per NPV case.

**Verify:** printed adopter counts per (mp, npv_case) include today's unsubsidized numbers
unchanged, plus the eight new cases.

---

## Task 7: Documentation (ongoing)

As you write the code for Tasks 3-6 above, comment it at the density already used in this
module: one or two lines per non-trivial step, stating what it does and why, placed right
above that step -- not a paragraph up top before the code. The existing functions in
`peak_load_functions.py` (`prepare_bsq_timeseries`, `check_upgrade_building_coverage`, etc.)
are the reference for this density. Full Google-style docstrings (Args/Returns/Raises) on
every new function, same as the rest of the module.

---

## Reminders that apply to every task above

- Keep `custom_weighting=False` behavior byte-identical to today; test that branch first.
- Never edit `plot_demand_panel` or `plot_county_demand_grid`.
- `compute_county_scenario_profile` hard-requires exactly 8,760 rows per building -- any
  change to how buildings are queried or filtered must preserve that.
- ASCII only in code and comments (no unicode arrows/dashes); 88-char line length; Google
  docstrings; see the project's `CLAUDE.md` for the full standard.
