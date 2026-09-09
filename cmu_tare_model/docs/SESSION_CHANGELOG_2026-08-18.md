# Session Changelog -- 2026-08-18

## Tepper MBA export rebuild: making the NPV reconcilable from the shipped columns

> Branch `joseph-2026-nature-comms-submission`. **Additive only.** Three
> quantities that the model computed and discarded are now persisted, and the
> Tepper export was rebuilt around them. No existing column moved, no golden
> value moved, and nothing was committed. The notebook changes were backported
> by the researcher and re-exported as
> `tare_model_main_v2_3_EXPORT_18Aug2026.py`.

---

## 1. Why this session existed

A Carnegie Mellon MBA capstone team receives a household CSV from this model.
They work in Excel only, never touch the codebase, and do their own fuel-price
lookups, because varying local prices is one of their planned analyses.

The governing requirement was that a reader can reconcile the shipped NPV from
the shipped columns with no hidden steps:

```
discounted heating savings + discounted cooling savings - net capital cost = NPV
```

They could not. Three quantities were computed and thrown away:

- **Per-year projected consumption** -- a local inside
  `calculate_annual_fuel_costs`, discarded the moment it was multiplied by a
  fuel price. Without it there is no way to check the dollars against a price.
- **The two discounted lifetime savings series** -- locals inside
  `calculate_private_npv`, collapsed straight into the nine NPV cases. Without
  them the savings half of the NPV is unverifiable.
- **The applied cooling replacement credit** -- never distinguished from the raw
  replacement cost, so the capital half could not be checked either.

---

## 2. Terminology fixed partway through

**A ResStock row is a representative dwelling unit (rdu), not a home.** Every
row carries `weight = 242.131013`, uniform across this release. Counts in this
document are labeled `rdu` or converted to actual homes.

The researcher corrected this after I wrote "in Allegheny County that is only 14
homes" about existing heat pumps. It is 14 rdu, which represent about **3,390
actual homes**. Rule of thumb recorded in CLAUDE.md: **any count below about 242
is a count of rdu, never homes.**

Weighted and unweighted averages and shares are identical here, because the
weight is uniform. Only totals and counts differ.

---

## 3. What changed, task by task

### Task 1 -- Audit, and two routing decisions

The supplemental fuel-cost file (`fuel_costs_ref2025`) already carries both the
baseline and retrofit annual cost columns in one file, indexed by `bldg_id`,
aligned exactly with the summary frame -- `index.equals()` true in both
directions, 331,531 rdu, same order.

Measured cost of the two routes:

| Route | Cost per run | Cost across four discount rates |
|---|---|---|
| `df_detailed` (supplemental) | +307 MB | +307 MB (does not multiply) |
| `df_main` (summary) | +307 MB | +1.04 GB, plus ~318 MB RAM through every downstream module |

**Decision: `df_detailed`.** The widening stays in the one artifact that exists
to hold per-year data, and nothing downstream reads it.

**Second decision -- consumption rounding.** `df_detailed.round(2)` rounds the
whole frame, but the model computes each annual cost from *unrounded*
consumption. Shipping consumption at 2 decimals means a reader's
`consumption x price` lands one cent away for about 4.7% of dwelling-year cells.
Six decimals would have been exact for +3 MB. **The researcher chose blanket 2
decimals**, so the documentation states the reconciliation is exact to within
one cent per dwelling-year cell rather than claiming exactness.

### Task 2 -- Persist annual projected consumption

Added `create_annual_consumption_col` to `column_names.py`, mirroring how
`create_fuel_cost_col` places its year token. Persisted the consumption series
inside `calculate_annual_fuel_costs` by adding it to the same dictionary that
already carries the cost column, so it inherits the caller's masking. Baseline
consumption is carried into the measure-package frame the same way baseline
costs already travel.

Result: 30 new columns in the baseline supplemental file, 60 in each
measure-package file (baseline heating, retrofit heating, baseline cooling,
retrofit cooling, 15 years each).

**Verified.** 39 baseline and 80 MP3 pre-existing columns compared against run
`2026-08-17_19-16`: zero mismatches, worst absolute difference `0.00e+00`.
`df_main` width unchanged at 71 and 189, so nothing leaked into the summary
frames. An independent rebuild of the price path from the vendored CSVs
reproduced the stored costs exactly for 2025, and for 2039 showed only the
predicted one-cent cells with none worse.

### Task 3 -- Persist discounted savings and the applied credit

Added `create_discounted_savings_col` and `create_cooling_credit_applied_col`.
In `calculate_private_npv`, wrote `heating_savings`, the **post-gate**
`cooling_savings` (the series after the `include_cooling` adjustment, which is
what the NPV used), and the applied cooling credit.

**Verified.** Ran the committed module and the edited module side by side in one
process, with no CSV round trip in between: **189 shared columns compared with
no tolerance, zero differences**, exactly 3 columns added. The reconciliation
identity holds for **all nine NPV cases**, max difference 0.0000. The 269 rdu
that have air conditioning but no recorded replacement cost get `0.0`, not
blank.

### Task 4 -- Rebuild the export column list

Rewrote `build_household_column_list` and turned the export into a validated
two-frame merge. Final count is **154 columns**: 94 from the summary frame, 60
from the supplemental frame. The two frames share no column names, so the join
overwrites nothing.

Removed: eight of the nine NPV cases with their net capital and adopter columns,
the December 2024 rebate amount, `private_discount_rate_variable`, and
`heating_total_capital_cost`. That last one was confirmed numerically to net out
the December 2024 rebate -- up to $8,000, matching `upgrade - rebate` to
2.9e-11 -- which has no place in an unsubsidized export. **The researcher
additionally dropped all 12 emissions and damages columns**, taking the count
from 166 to 154.

Two wiring decisions, both chosen by the researcher:

- The supplemental frame is passed in as a **required** argument. An optional
  one would let a forgotten call silently ship a 94-column file.
- The annual frame may be a **superset** of the household frame. That is what
  lets a county export pass the county subset plus the full national annual
  frame, with no second filter to keep in step.

**Verified end to end for MP3 and MP4.** Column order and membership exact, row
count preserved, all 94 summary and 60 annual columns matching their sources,
and **the NPV reconciles from the shipped columns alone across 3,163 rdu with
zero off by more than a cent.**

### Task 5 -- Export scope filter

Originally specified as a hardcoded Allegheny block; the researcher redirected
it mid-task to a general location filter. That proved cleaner: both the
household frame and the county tables carry `state` and `county`, and
`county == 'G4200030'` selects exactly the same rows as `county_fips == 42003`,
so one scope specification filters either kind of frame and no second code is
needed.

`filter_to_export_scope(df, scope_column, scope_value)` handles both. The
notebook drives it from a `TEPPER_EXPORT_SCOPES` list; adding a state or another
county is one line.

**Verified.** National 331,531 rdu; Allegheny 1,610 rdu with 254 dropped as not
applicable leaving **1,356**; PA as an example scope 15,651 to 12,266; county
tables filter 3,098 to 67 to 1. A scope the run does not cover is skipped with a
message; a mistyped GISJOIN code raises.

### Task 6 -- Data dictionary

Consolidated two stale documents into one at
`docs/tare_tepper_exports_data_dictionary.md` and deleted the duplicate at
`utils/tepper_export_data_dictionary.md`. The PDF is superseded and still on
disk.

Corrected all six documented contradictions: scope is National plus Allegheny
rather than PA-only; the column count rebuilt from the code rather than the old
100-versus-112 mismatch; the 12-column peak block documented for the first time;
national denominators 78.49% heating and 75.58% cooling; the 2025-2039 range
stated throughout; the validating run updated.

Documented the fuel-price rule precisely rather than by restatement. Traced it
to `groupby(['census_division','fuel_type']).mean()` and confirmed fuel oil and
propane inherit a PADD price per state, with a U.S. national fallback for 18
states (fuel oil) and 7 states (propane), then an **unweighted arithmetic mean**
across the states in each census division.

Included a full worked example for `bldg_id 491` -- Allegheny, gas furnace to
heat pump -- from base-year consumption through degree-day projection, per-fuel
pricing, discounting and net capital to the shipped NPV of **-$1,980.58**,
matching to the cent.

---

## 4. The worked example, in brief

One representative dwelling unit, standing for about 242 real homes.

| Quantity | Value |
|---|---|
| Baseline heating | Natural Gas Fuel Furnace, 92.5% AFUE |
| Baseline cooling | Central AC, SEER 13 |
| Retrofit | ASHP, SEER 16, 9.5 HSPF |
| Baseline heating consumption, 2025 | 12,795.78 kWh |
| Retrofit heating consumption, 2025 | 2,941.26 kWh |
| Discounted heating saving, 2025-2039 | $291.33 |
| Discounted cooling saving, 2025-2039 | $626.50 |
| Heat pump installed cost | $12,084.50 |
| less avoided heating replacement | $3,734.48 |
| less applied cooling credit | $5,451.61 |
| Net capital cost | $2,898.41 |
| **NPV** | **-$1,980.58** |

It makes the fuel-switching point concrete: heating energy falls 77%, but
because electricity costs roughly four times what natural gas does per kWh, the
first-year heating saving is only $64. Cooling ends up contributing more than
double the heating savings over the lifetime.

---

## 5. The sample-size cascade, and one finding worth carrying forward

The researcher asked for a filter breakdown for a README. Added
`cmu_tare_model/utils/tare_sample_size.py`, which recomputes the cascade from
the raw ResStock files for any county, state, or the nation. Read-only.

| Step | Allegheny rdu | Allegheny homes | National rdu | National homes | % of stock |
|---|---|---|---|---|---|
| Sampled units, no filters | 2,434 | 589,347 | 548,916 | 132,909,587 | 100.00% |
| 1. Occupied only | 2,197 | 531,962 | 482,597 | 116,851,700 | 87.92% |
| 2. Single-family only | 1,610 | 389,831 | 331,531 | 80,273,937 | 60.40% |
| 3. Heating fuel in scope | 1,604 | 388,378 | 321,357 | 77,810,496 | 58.54% |
| 4. Heating technology in scope | **1,356** | **328,330** | **260,211** | **63,005,153** | **47.40%** |

**The ResStock `applicability` flag is `True` for all 1,610 Allegheny rdu**, for
both MP3 and MP4. Every exclusion past step 2 is TARE's own scope, not
ResStock's.

**Allegheny is not representative of what step 4 removes:**

| Removed at step 4 | National rdu | Share | Allegheny rdu | Share |
|---|---|---|---|---|
| Electricity ASHP | 31,347 | 51% | 14 | 6% |
| Wall/Floor Furnace, all fuels | 28,741 | 47% | 224 | 88% |
| Shared Heating, all fuels | 1,058 | 2% | 10 | 4% |

Nationally the largest excluded group is dwellings that **already have a heat
pump** -- a deliberate design exclusion, since there is no fossil system for the
retrofit to replace. In Allegheny it is wall and floor furnaces that dominate.
Anything that leads with Allegheny will wrongly suggest the exclusions are
mostly a cost-database gap.

**Comparison with ResStock 2025 dual-fuel eligibility** (researcher supplied
Table 3): its Dual Fuel package reaches 44.65% of stock against TARE's 47.40%.
Its extra conditions are ducts and a natural gas **hookup** -- note the hookup
is a separate column from heating fuel, and its fuel list is the same four fuels
TARE uses, so it is **not** a natural-gas-only heating-fuel filter. 85% of the
dwellings TARE evaluates are ducted, so adding a duct requirement would take
TARE to 40.15%.

---

## 6. Open question for a later session

Whether the 254 Allegheny rdu should be dropped at all. They represent
**61,501 real dwellings, about 16% of the county's single-family stock.**

The researcher's position is that they should be kept where the ResStock
applicability data is valid, reasoning that a dwelling with a wall furnace, or
with none, can still install a heat pump for the first time when it becomes
affordable. The counter-argument is that the model produced no NPV, no savings
and no adoption flag for them, so every result column is blank, and Excel treats
a blank as zero in arithmetic.

**The real question is narrower than it first appeared: should wall and floor
furnaces be a modeled heating technology?** That is a model-scope change with a
value impact on every national result, not an export-only fix. It would move the
golden values.

No regression exists. `include_heating` gives 254 in both the 17 August and 18
August runs. Past Allegheny data had 1,610 rows because the export never dropped
rows; the drop is the new pre-filter added this session.

Documented for readers in sections 3.1 to 3.4 of the data dictionary, where the
rule is explicitly marked as under review.

---

## 7. One process correction

I edited `tare_model_main_v2_3_EXPORT_1Aug2026.py`. Those files are read-only
snapshots of notebooks; editing one changes no behaviour while making the
snapshot disagree with the notebook it mirrors. The researcher caught it. Both
edits were reversed by hand rather than by `git checkout`, so nothing else in
the working tree was disturbed, and the file verified clean.

The rule is now in CLAUDE.md as a table row, a full section, and an
anti-pattern line: change the importable modules, then hand over copy-paste
notebook cells to backport.

---

## 8. Files changed this session

| File | Change |
|---|---|
| `utils/column_names.py` | +77 lines, three new builders |
| `private_impact/calculate_lifetime_fuel_costs.py` | +25, persists 60 annual consumption columns |
| `private_impact/calculate_lifetime_private_impact.py` | +32, persists 2 savings columns and the applied credit |
| `utils/export_tepper_csv.py` | +362, 154-column list, two-frame merge, scope filter, source-data copies |
| `utils/export_model_run_results.py` | +15, `df_annual_consumption` passthrough |
| `utils/tare_sample_size.py` | new, filter cascade from raw ResStock files |
| `docs/tare_tepper_exports_data_dictionary.md` | rewritten |
| `utils/tepper_export_data_dictionary.md` | deleted, duplicate |
| `CLAUDE.md` | `_EXPORT` rule, rdu terminology section, two anti-patterns |
| `tare_model_main_v2_3.ipynb` | three cells backported by the researcher |
| `tare_model_main_v2_3_EXPORT_18Aug2026.py` | new snapshot, reviewed |

269 tests pass. No `_EXPORT_` snapshot carries changes from this session.

---

## 9. Outstanding

1. **Full pipeline run.** Verification rebuilt the new columns on top of run
   `2026-08-17_19-16`, which proves the arithmetic but is not a real end-to-end
   run.
2. **After that run:** update the run timestamp in section 1 of the data
   dictionary, and the golden-value rows in CLAUDE.md. Golden values were not
   touched this session, since guessing at them would violate the
   never-silently-overwrite rule.
3. **The confirmed golden values should reproduce exactly** -- MP3 and MP4
   `heatingLCC_coolingLCC_unsub` NPV of -$4,852.41 and -$5,838.23, adoption of
   27.7140% and 18.4416%. If they move, something is wrong.
4. Optional: delete the superseded PDF; tighten the notebook verification cell
   so it stops reporting on stale Colorado exports from earlier runs.
