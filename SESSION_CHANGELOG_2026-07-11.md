# Session Changelog -- 2026-07-11 (June 2026 Rebate Guidance: HEEHR + HOMES Regime Axis)

## Session Summary

Added the DOE June 2026 rebate guidance to the model as a **rebate-regime sensitivity
axis** inside one dataframe (like the discount-rate axis), rather than a second
`policy_scenario`. Three regimes per NPV scope now coexist:

- `_unsub` -- unsubsidized (existing, unchanged)
- `_sub` -- subsidized under 2024 guidance / current HEEHR (existing, unchanged)
- `_sub_june2026` -- subsidized under June 2026 guidance (NEW: HEEHR + a simplified
  HOMES pathway + a fuel-eligibility gate)

`NPV_CASE_CATEGORIES` grew from six to nine cases. The existing `_sub`/`_unsub`
computation and `calculate_capital_costs` were left untouched, so the current results
stay byte-identical; the new regime is applied additively by netting its own rebate
column against the already-computed unsubsidized capital.

`policy_scenario` remains `'2025 Reference Case'` with prefix `ref2025_mp{mp}_` -- no
new scenario string, and no changes to `define_scenario_params`,
`validate_common_parameters`, or the fuel-price lookups.

## Decisions (from the researcher)

1. **HOMES savings fraction = whole-home site energy.** Propagate ResStock
   `out.site_energy.total.energy_consumption.kwh` as the denominator; numerator is the
   heating + cooling energy delta.
2. **Rebate regime as a sensitivity axis** (a constant in `constants.py`, threaded like
   the discount methods), encoded by **adding a third token** and keeping `_sub`/`_unsub`.
3. **`REBATE_ELIGIBLE_HEATING_MPS` = `[4, 8, 9, 10]` kept as-is** (the session prompt
   stated `{MP4}`, but the actual constant is `[4, 8, 9, 10]`; MP4 in, MP3 out for both
   programs).

## Changes by file

- **`constants.py`** -- added `REBATE_GUIDANCE_IRA2024`, `REBATE_GUIDANCE_JUNE2026`,
  `REBATE_POLICY_SCENARIOS` (renamed from `REBATE_GUIDANCE_REGIMES` on 12 Jul 2026),
  the HEEHR/HOMES rule constants (AMI cutoffs, coverage,
  caps, savings tiers), `ELECTRIC_RESISTANCE_BASELINE`, and `REBATE_NONE/HEEHR/HOMES`.
- **`utils/column_names.py`** -- `create_rebate_col` gained an optional `guidance` token
  (default `None` keeps existing names byte-identical); `NPV_CASE_CATEGORIES` expanded to
  nine (three `_sub_june2026` tokens).
- **`energy_consumption_and_metadata/process_euss_data.py`** -- propagated
  `baseline_total_site_consumption`; added per-MP `mp{mp}_modeled_savings_frac`
  (heating + cooling delta over whole-home baseline site energy; masking respected).
- **`private_impact/data_processing/determine_rebate_eligibility_and_amount.py`** --
  new `calculate_rebate_june2026` (vectorized): efficiency gate (MP), fuel gate
  (electric-resistance only), state-participation gate, HEEHR (<=150% AMI) vs HOMES
  (>150% AMI, savings-based), writing `mp{mp}_heating_rebate_amount_june2026_{cost_scenario}`
  and `mp{mp}_rebate_eligibility_june2026`. Preserves `random_seed=42`, NaN-masking,
  float64. Also added `summarize_june2026_rebate_totals` (weight-scaled HEEHR/HOMES
  dollar totals, national + per state -- there is no aggregate/state funding cap, so
  these are uncapped program costs).
- **State-participation exclusion (ALL regimes):** `NON_PARTICIPATING_REBATE_STATES =
  {'SD'}` in `constants.py`. South Dakota never participated, so its homes now get 0
  under both the 2024 path (`calculate_rebateIRA`) and June 2026
  (`calculate_rebate_june2026`). This makes the 2024 `_sub` columns change for SD homes
  (a correction, not a regression).
- **`private_impact/calculate_lifetime_private_impact.py`** -- reads the june2026 rebate
  column and builds three `*_sub_june2026` net-capital variants; `npv_case_inputs`
  expanded 6 -> 9. Existing lines untouched.
- **`model_scenarios/tare_scenarios_v2_3_EXPORT_28June2026.py`** -- rebate loop now calls
  `calculate_rebate_june2026` alongside `calculate_rebateIRA` (mirror `.py`; backport to
  the `.ipynb` manually).
- **`utils/export_tepper_csv.py`**, **`adoption_potential/determine_economic_adoption_potential.py`**
  -- both already iterate `NPV_CASE_CATEGORIES`, so they auto-emit nine cases; only
  stale "six" names/docstrings were updated to "nine".
- **`tests/private_impact/test_rebate_june2026.py`** -- new: guidance-token naming,
  HEEHR/HOMES amounts across fuel x income tiers, MP3-never-eligible, cooling no-op,
  excluded-home NaN masking.

## Verification (this environment)

- `test_rebate_june2026.py` (5 tests) and the existing NPV tests (now covering nine
  cases) pass.
- Full suite: **11 failed, 281 passed** with the changes vs **11 failed, 276 passed**
  on a stashed baseline -- i.e. +5 new passing tests, **0 new failures**. The 11
  failures are pre-existing (a test-isolation issue where `EQUIPMENT_SPECS = {heating,
  cooling}` conflicts with fixtures expecting the four-category set) and are unrelated to
  this work. `test_kpi_functions.py` fails collection because `geopandas` is not
  installed (pre-existing environment gap).

## Deferred to the researcher's environment (full 331k-home run required)

- **Byte-identity check:** confirm the six existing NPV/adopter columns and the CLAUDE.md
  golden values reproduce exactly. This run is also the numeric before-snapshot.
- **Movement cross-tab:** `base_heating_fuel` x income tier x `rebate_eligibility` for MP3
  and MP4, confirming directions (a) fossil MP4 lose rebate -> adoption falls; (b) electric
  MP4 <=150% AMI unchanged; (c) electric MP4 >150% AMI gain HOMES -> adoption rises; (d) MP3
  unchanged.
- **Golden values:** derive and record the `_sub_june2026` mean adoption rates (rows added
  to CLAUDE.md as PENDING).
- **Visuals + national table:** redo affected adoption visuals and add a national
  adopters-by-baseline-fuel table for both MPs, broken out by `rebate_eligibility`.
- **`.ipynb` backport:** hand-backport the `calculate_rebate_june2026` call into the main
  notebook cell (never edit `.ipynb` JSON directly, per CLAUDE.md).
