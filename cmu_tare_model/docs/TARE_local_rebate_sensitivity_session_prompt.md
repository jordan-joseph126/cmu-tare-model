# TARE -- Local-Rebate Sensitivity (`sub_june2026_plusLocal`) Session Prompt

CLAUDE.md is the source of truth for conventions, file rules, golden values, coding standards,
and all non-negotiables; it is auto-loaded, so this prompt only adds session-specific state and
the task list. Audit before editing and use one diff per stop gate, per CLAUDE.md.

## Context

**Priority: this change only ADDS. The existing nine NPV cases must still produce exactly the same
values as before; a local rebate only ever adds columns, never moves an existing value.** Work on a
new feature branch off `main` (do not commit to `main`).

The task is to add Colorado/Longmont/Boulder local heat-pump rebates as a new sensitivity that
stacks on top of the June 2026 federal rebate. This introduces a fourth rebate option,
`sub_june2026_plusLocal`, on the existing June 2026 base only -- taking the case count from
**9 to 12** (three NPV cost offsets x four rebate options). A local rebate lowers the capital cost
and never changes the energy savings, so a home's `plusLocal` NPV equals its `sub_june2026` NPV plus
the local adder, and no savings or federal-rebate math is recomputed.

The three NPV cost offsets, each of which gets its own `plusLocal` version (hence three new cases,
not one):

| Case name | What cost it credits back |
|---|---|
| `heatingSavings_coolingLCC` | the air conditioner the home did not have to replace |
| `heatingLCC_coolingSavings` | the heating system the home did not have to replace |
| `heatingLCC_coolingLCC` | both of the above |

The local rebate conditional logic is specified in full in the "Local rebate logic" section below.
Implement exactly that -- do not re-derive the rates, tiers, or rules.

Out of scope this session: running the actual CO/Boulder model run; recomputing the PA-specific
applicable-home denominators; Xcel-territory ASHP rebate amounts (a known data gap -- leave a
documented placeholder, do not invent values); any figures/visuals beyond the adoption-rate KPI.

## Current state the tasks depend on

- **Producer/consumer mismatch (the load-bearing trap).** `calculate_private_npv` in
  `calculate_lifetime_private_impact.py` **hardcodes** its nine-entry `npv_case_inputs` dict
  (around lines 316-325). The export (`export_tepper_csv.py`, lines 172-182) and the economic-
  adoption module instead **loop over `NPV_CASE_CATEGORIES`**. Extending `NPV_CASE_CATEGORIES`
  alone makes those consumers request three columns the producer never wrote -> `KeyError`. **Both
  the constant and the hardcoded dict must be extended together, or the change is broken.**
- **The plusLocal net-capital equation, per case:**
  `net_capital_<case>_sub_june2026_plusLocal = net_capital_<case>_unsub - rebate_june2026 - rebate_local`.
  Mirror the existing `rebate_june2026_col` block (lines 293-311): read the local column with the
  same `if col in df_copy.columns` guard so a run without local dollars (e.g. the PA run) falls back
  safely to the `sub_june2026` outcome.
- **Double counting must be prevented (correctness, not style).** Some Colorado rebates are not local
  money at all -- they are federal money that Colorado hands out, and the model already counts those
  dollars in the `sub_june2026` case. The local module must include ONLY the CO Heat Pump tax credit
  (~$333 net to customer), Efficiency Works (PRPA/LPC), and EnergySmart YES (Boulder County). It must
  EXCLUDE CO Home Efficiency Rebates (HER) and CO HEAR -- those are Colorado's pass-through of the
  federal IRA HOMES/HEEHR money. Including them pays the same dollars twice.
- **Local rebate depends on geography and system type, NOT income.** It does not fit the
  `REBATE_RULE_CONFIG[guidance]` HEEHR/HOMES shape. Keep it in a SEPARATE function/module so it
  composes rather than contorting the income-based config path.
- **Local applicability inputs, all per-home columns on the loaded household frame:**
  `hvac_has_ducts` (ducted -> per-unit rate; non-ducted -> per-ton), `base_heating_fuel` +
  `heating_type` (electric baseboard replacement -> higher per-ton rate),
  `size_heating_system_primary_k_btu_h` (tons = size / 12), `percent_AMI` (EnergySmart under-100%-AMI
  check), `county` (EnergySmart Boulder-County check), and a per-home utility-territory value
  (Efficiency Works applies only in Longmont/PRPA/LPC territory; Xcel territory earns $0 Efficiency
  Works for now). EnergySmart is capped so combined local rebates do not exceed 50% of project cost
  (~$2,000 max).
  **The measure package number (3 or 4) is NOT a column** -- it is passed into the function as an
  argument, the same way the existing rebate functions receive `menu_mp`. It selects the Efficiency
  Works ducted tier (MP3 -> Tier 1, MP4 -> Tier 2).
- **Hand-checked test homes, for the module's tests:**
  Home 1 (MP4, ducted, electric furnace, 2 ton, 90% AMI, Longmont) local adder = **$4,333**;
  Home 2 (MP3, non-ducted, electric baseboard, 1.5 ton, 70% AMI, Longmont, $6k project) the
  EnergySmart cap applies and trims $2,000 -> $1,167, adder = **$3,000**; Home 3 (MP4, ducted, gas,
  2.5 ton, 200% AMI, Xcel) earns no Efficiency Works and no EnergySmart, adder = **$333**.
- **Optional overall cap -- OPEN DECISION, list it in the audit, do not assume.** If federal + local
  could exceed installed cost on a small system, cap with
  `total_rebate = MIN(project_cost, federal + local)`, then reduce the local amount to fit under that
  limit. Confirm with the researcher whether to enforce this before wiring it into net capital.
- **Unconfirmed assumption to flag in the audit.** We assume a local rebate does not reduce or
  otherwise affect the federal rebate cap. The whole "just add the local adder" approach depends on
  this. Report it as an open item; do not build anything that relies on the opposite.
- **Naming via helpers only.** Use `create_rebate_col(..., guidance=...)`, `create_npv_case_col`,
  `create_capital_col`, `create_adoption_col`. Never hardcode `mp3`/`mp4` or a scenario prefix. Add a
  new local guidance token in `constants.py` rather than a literal string. Any full column name shown
  in this prompt is written out for readability only -- build it with the helpers.

## Local rebate logic (self-contained -- implement exactly this)

Rates and levers (put in `constants.py`):
- Efficiency Works (EW): `Ducted_T1` $1000/unit (MP3), `Ducted_T2` $2000/unit (MP4),
  `NonDucted_Std` $500/ton, `NonDucted_Baseboard` $1000/ton.
- `CO_HEATPUMP_DISCOUNT` = $333 flat (all CO homes). `ENERGYSMART_MAX` = $2000.
  `COMBINED_CAP_FRAC` = 0.50.

Per home, `local_adder = ew_amount + co_discount + energysmart`, where (vectorized over the frame):

```
tons         = size_heating_system_primary_k_btu_h / 12
is_baseboard = (lower(base_heating_fuel) == "electricity")
               AND ("baseboard" in lower(heating_type))
tier         = Ducted_T2 if (hvac_has_ducts and menu_mp == 4)
               Ducted_T1 if (hvac_has_ducts and menu_mp != 4)
               NonDucted_Baseboard if (not hvac_has_ducts and is_baseboard)
               NonDucted_Std       if (not hvac_has_ducts and not is_baseboard)
ew_amount    = (rate if basis == "unit" else rate * tons)
               only if utility_territory == "Longmont-EffWorks", else 0
co_discount  = 333
es_raw       = 2000 if (county == "Boulder" and percent_AMI < 100) else 0
room_left    = max(0, 0.50 * project_cost - ew_amount - co_discount)  # combined 50%-of-cost cap
energysmart  = min(es_raw, room_left)
```

Then per NPV case: `NPV_plusLocal = NPV_sub_june2026 + local_adder`;
`adopter_plusLocal = 1 if NPV_plusLocal >= 0 else 0`. EXCLUDE CO HER and CO HEAR entirely -- they are
federal IRA pass-through already counted in `sub_june2026`. Match case-insensitively for the fuel,
heating-type, territory, and county strings; confirm the actual ResStock spellings during the audit
rather than trusting the literals above.

## Tasks (execute in order; stop at each gate)

**Task 1 -- Audit (no edits).** Locate and report, without changing anything:
(a) where `NPV_CASE_CATEGORIES` is defined and every site that loops over it (export, adoption, any
KPI/figure loop, any test);
(b) confirm `calculate_private_npv` hardcodes `npv_case_inputs` and list every other producer that
would need the three new keys;
(c) the economic-adoption module -- how it builds adopter flags from NPV cases, and whether it loops
over the constant or hardcodes;
(d) all call sites of `calculate_rebate_june2026` / `calculate_rebate_program`;
(e) every test or assertion that counts cases (e.g. "9", "27 nine-case columns", "36") and the data
dictionary section 5 -- these become 12 / 36;
(f) whether a per-home utility-territory column already exists in the loaded frame or must be created
from `county`/`city`, and confirm the other local-applicability columns are present with their actual
category spellings;
(g) list the two open decisions for the researcher: the optional overall cap, and whether federal
dollars count toward the EnergySmart 50% cap.
Report findings and stop.

**Task 2 -- Extend the case constant and guidance token.** In `constants.py`: add the CO local-rebate
rates and levers (CO discount, Efficiency Works ducted/non-ducted tiers, EnergySmart max +
combined-cap fraction) and a `REBATE_GUIDANCE_LOCAL` token; append the three
`<case>_sub_june2026_plusLocal` names to `NPV_CASE_CATEGORIES` **after** the existing nine (do not
interleave). Verify: the list has 12 entries, and the first nine are unchanged in value and order.
Stop.

**Task 3 -- Build the local-rebate module.** Add a new `calculate_rebate_local` function (kept
separate from the income-based `calculate_rebate_program`) that writes a
`create_rebate_col(..., guidance=REBATE_GUIDANCE_LOCAL)` amount column. Implement the "Local rebate
logic" section above exactly: tons = size / 12; ducted per-unit vs non-ducted per-ton;
electric-baseboard-replacement uplift; MP-selected Efficiency Works tier; Efficiency Works only in
Longmont territory; CO discount always on for CO homes; EnergySmart for Boulder County homes under
100% AMI with the 50%-of-project combined cap; and the exclusion of HER/HEAR. Use the same
validation/masking bookkeeping as the existing rebate functions (excluded homes NaN,
valid-but-ineligible 0.0). Verify against the three test homes ($4,333 / $3,000 / $333) and confirm
HER/HEAR contribute nothing. Stop.

**Task 4 -- Wire the local rebate into private NPV.** In `calculate_lifetime_private_impact.py`,
mirror the `rebate_june2026_col` block to read the local column under the same
`if col in df_copy.columns` guard; compute the three
`net_capital_<case>_sub_june2026_plusLocal = net_capital_<case>_unsub - rebate_june2026 - rebate_local`;
add the three matching entries to `npv_case_inputs`. Apply the optional overall cap only if the
researcher approved it in Task 1. Verify: the nine existing NPV and net-capital columns produce
exactly the same values as before the change (no local dollars -> unchanged); the three new columns
appear; and the equation `NPV_plusLocal == NPV_sub_june2026 + local_adder` holds on a spot-checked
home. Stop.

**Task 5 -- Confirm adoption picks up the new cases.** In the economic-adoption module, confirm the
three new adopter flags are produced automatically from `NPV_CASE_CATEGORIES`; if the module
hardcodes cases anywhere, extend it the same way. Verify: 12 `econ_adopter` flags exist and each new
flag equals `1 if NPV_plusLocal >= 0 else 0`. Stop.

**Task 6 -- Export + data dictionary.** The nine-NPV / nine-net-capital / nine-adopter export groups
follow `NPV_CASE_CATEGORIES` and become twelve on their own; only the local rebate *amount* column
must be added by hand, to the rebate group in `build_household_column_list`. Update
`tepper_export_data_dictionary.md` from nine to twelve cases (and the "27 -> 36" nine-case count in
section 5). Verify: the household column list length increased by exactly the expected count, the
round-trip check still passes, and no `valid_*`/`include_*` bookkeeping column leaked in. Stop.

**Task 7 -- Tests.** Add unit tests: the three test homes; the exclusion of HER/HEAR (both = 0);
Efficiency Works only in Longmont (Xcel -> $0); the EnergySmart 50%-of-cost cap applying; a check
that the nine existing cases produce the same numbers as before when no local rebates apply; and the
`NPV_plusLocal = NPV_sub_june2026 + local_adder` equation. Run the full suite. Report pass/fail
counts and stop.

## Start

Do Task 1 only. Produce the audit -- including the utility-territory data-availability finding, the
actual ResStock category spellings, and the two open decisions for the researcher -- and stop at the
first gate for approval.

## Deferred to a future session (do NOT start now)

- Run the model for CO/Boulder (set the geographic filter to CO/Boulder) and recompute the
  applicable-home denominators; the current run is PA-only.
- Source and add Xcel-territory ASHP rebate amounts (currently a data gap); the module leaves a
  documented placeholder until then.
- Carry the three new cases into any downstream figures/grid-impact views beyond the adoption KPI;
  update the relevant CLAUDE.md section before approving.
