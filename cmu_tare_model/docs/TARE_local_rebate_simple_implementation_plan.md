# Local Rebate Scenario, Version 1 (Flat Rebate) -- Implementation Plan

Written 27 Oct 2026. This plan replaces the approach in
`TARE_local_rebate_implementation_plan.md`, which stacked Colorado, Longmont and Boulder
rebates on the June 2026 federal rebate. That rebate content is out of date and is not used
here. Only that plan's workflow (the order of files, and the checks) is kept.

**What we are building, in one sentence:** a new, stand-alone rebate policy scenario,
`_sub_local`, that gives every home in the study sample the same flat local rebate, with no
federal money in it, so we can see how many homes become economic adopters under that rebate
alone.

Version 1 is deliberately as simple as possible. Its purpose is to put the pipeline in place:
one new rebate column, three new NPV cases, three new adopter flags. Eligibility rules
(location, income, system type, cost caps) can be added later inside one function, without
touching the rest of the model.

---

## 1. Overview

The model has three rebate policy scenarios today:

- `_unsub`: no rebate
- `_sub`: 2024 federal guidance
- `_sub_june2026`: June 2026 federal guidance

This plan adds a fourth, `_sub_local`. It starts from the unsubsidized case and subtracts only
the local rebate:

    net_capital_<scope>_sub_local = net_capital_<scope>_unsub - local_rebate
    NPV_sub_local                 = NPV_unsub + local_rebate
    adopter_sub_local             = 1 if NPV_sub_local >= 0 else 0

A rebate lowers the capital cost and never changes the energy savings, so no savings are
computed again.

**No link to the federal programs.** The local scenario reads no federal rebate column and
applies none of the federal rules: no HEEHR or HOMES income routing, no fuel gate, no South
Dakota gate, no ENERGY STAR gate (`get_rebate_eligible_mps`). It is not added to
`REBATE_POLICY_SCENARIOS` or `REBATE_RULE_CONFIG`, which describe the federal programs only.

Each of the three NPV scopes gets a local version, so the case count goes from **9 to 12**:

| New case | What cost it credits back |
|---|---|
| `heatingSavings_coolingLCC_sub_local` | the air conditioner the home did not have to replace |
| `heatingLCC_coolingSavings_sub_local` | the heating system the home did not have to replace |
| `heatingLCC_coolingLCC_sub_local` | both of the above |

The nine existing cases do not change.

---

## 2. Rebate logic (version 1)

Every home in the study sample (`include_sample = True`) receives the same flat amount,
`LOCAL_REBATE_AMOUNT`. That applies to every measure package (MP3 and MP4 on 2022.1.1, MP5 on
2025.1). Homes outside the sample are left blank (NaN), the same as every other result.

    function local_rebate(home):
        if not home.include_sample:
            return NaN
        return LOCAL_REBATE_AMOUNT

Like the federal rebates, the local rebate is recorded once, on the heating side. The cooling
call does nothing, because the heat-pump rebate covers both end uses.

### 2a. Settings (`constants.py`)

    REBATE_GUIDANCE_LOCAL = "local"   # column-name token
    LOCAL_REBATE_AMOUNT = ...         # dollars per home; set by the researcher

### 2b. Where later complexity goes

Every future rule is a mask or an amount inside `calculate_rebate_local`, plus a setting in
`constants.py`. Nothing downstream changes. Candidates noted so far:

- which measure packages qualify (version 1: all of them);
- location (state or county);
- income (`percent_AMI`);
- a cap at a share of the heat-pump project cost
  (`create_cost_col(menu_mp, 'heating', 'upgrade', cost_scenario)`, the cost the federal
  HEEHR calculation uses);
- an eligibility label column (for example `mp{mp}_rebate_eligibility_local`), worth adding
  once some sample homes can be ineligible.

---

## 3. Home columns used

| What | Column | Used for |
|---|---|---|
| Study sample flag | `include_sample` (through `initialize_validation_tracking`) | which homes get the rebate |
| Unsubsidized net capital | built inside `calculate_private_npv` | the base the rebate is subtracted from |

Names are built with the helpers (`create_rebate_col`, `create_npv_case_col`,
`create_capital_col`, `create_adoption_col`), never typed out.

---

## 4. Output columns

- `mp{mp}_heating_rebate_amount_local_{cost_scenario}`, from
  `create_rebate_col(..., guidance=REBATE_GUIDANCE_LOCAL)`: `LOCAL_REBATE_AMOUNT` for every
  sample home, NaN outside the sample. Float64.
- For each of the three scopes, the NPV, net capital and economic adopter columns, named by
  the helpers from the new case names.

---

## 5. How it plugs into the model

The order matters, because each step writes a column the next one reads.

1. **`constants.py`**: `REBATE_GUIDANCE_LOCAL` and `LOCAL_REBATE_AMOUNT`.
2. **`utils/column_names.py`**: add the three `<scope>_sub_local` names to
   `NPV_CASE_CATEGORIES`, after the existing nine, and update the comment block above it.
   `create_npv_case_col` rejects any name that is not in this tuple.
3. **`determine_rebate_eligibility_and_amount.py`**: a new function, `calculate_rebate_local`,
   separate from `calculate_rebate_program`. Same signature style (`df`, `category`,
   `menu_mp`, `cost_scenario`, `verbose`). It returns at once for `category == 'cooling'`, and
   uses the same validation bookkeeping (`initialize_validation_tracking`,
   `apply_final_masking`), so homes outside the sample stay NaN.
4. **`calculate_lifetime_private_impact.py`** (`calculate_private_npv`): read the local column
   with the same "if the column exists" check as the June 2026 block, compute the three
   `net_capital_*_sub_local = net_capital_*_unsub - local_amount`, and **add three entries to
   the hardcoded `npv_case_inputs` dict**. Steps 2 and 4 must be done together: if the tuple
   gains three names and the dict does not, the adoption module and the runner look for
   columns that were never written and stop with a `KeyError`.
5. **`model_scenarios/tare_scenarios_v3_1.ipynb`**, rebate cell: call `calculate_rebate_local`
   after the federal `REBATE_POLICY_SCENARIOS` loop, not inside it (that loop passes each
   token to `calculate_rebate_program`, which would reject `'local'`). This is a notebook
   edit, so it goes through the notebook-cell-edit skill.
6. **Picks up the new cases with no change:** `determine_economic_adoption_potential.py`
   (adopter flags), the runner's NPV-identity and adopter checks, and the NPV tests, all of
   which loop over `NPV_CASE_CATEGORIES`.
7. **Optional, listed by hand:** `REBATE_POLICY_SCENARIO_ORDER`, `_LABELS` and `_MARKERS` and
   `valid_vintage` in `visuals_adoption_dotplot.py`, and `rebate_scenarios` in
   `scripts/run_tare_notebooks.py`. Add `'sub_local'` only if the scenario should appear in
   the dot plot and the runner's per-scenario summaries. The runner could also check that
   `NPV_sub_local == NPV_unsub + LOCAL_REBATE_AMOUNT` on every sample home.
8. **Not changed:** the Tepper export ships only `heatingLCC_coolingLCC_unsub`, so the new
   cases do not reach it. Add the local amount column to its `rebate` list only if its readers
   need it.

---

## 6. Worked examples (acceptance test for the module)

Illustrative setting, chosen only for the test: `LOCAL_REBATE_AMOUNT = 2000`.

| Field | Home A | Home B | Home C | Home D |
|---|---|---|---|---|
| In study sample | yes | yes | yes | no |
| measure package | MP3 | MP4 | MP5 (2025.1) | MP3 |
| NPV_unsub | -1,500 | 400 | -3,000 | NaN |
| **local rebate** | **2,000** | **2,000** | **2,000** | **NaN** |
| **NPV_sub_local** | 500 | 2,400 | -1,000 | NaN |
| **outcome** | **becomes an adopter** | adopter (already) | non-adopter | outside the sample |

---

## 7. Open decisions (Tamar)

1. **The amount:** the value of `LOCAL_REBATE_AMOUNT` for version 1.

Decided 27 Oct 2026: no location or income rules in version 1; every measure package
qualifies (may change later).

---

## 8. Build order checklist

Each step ends with a check and a stop, per the repo's audit-first, one-diff-per-stop-gate rule.

- [ ] Task 1 -- Audit only: confirm `npv_case_inputs` is still the only hardcoded producer of
      NPV cases, and list every hand-written scenario list. No edits.
- [ ] Task 2 -- `constants.py`: `REBATE_GUIDANCE_LOCAL`, `LOCAL_REBATE_AMOUNT`.
- [ ] Task 3 -- `column_names.py`: three new cases appended. Check: 12 entries, first nine
      unchanged in value and order.
- [ ] Task 4 -- `calculate_rebate_local`; check against Homes A to D, and that the cooling call
      changes nothing.
- [ ] Task 5 -- wire into `calculate_private_npv`; check that the nine existing cases are
      unchanged and that `NPV_sub_local == NPV_unsub + LOCAL_REBATE_AMOUNT` on every sample
      home.
- [ ] Task 6 -- notebook call (notebook-cell-edit skill); adoption gives 12 adopter flags.
- [ ] Task 7 -- optional dot plot and runner lists.
- [ ] Task 8 -- tests: Homes A to D, cooling skipped, NaN outside the sample, the nine cases
      unchanged, and the NPV equation. Run the full suite, then both releases with the runner
      (`--skip-grid-impact` for 2025.1).
