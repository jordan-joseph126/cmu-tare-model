# Session Changelog -- 2026-08-20

## Fixing the size and efficiency used to price the replacement cost

> Branch `joseph-2026-nature-comms-submission`. Follows the 19 August
> reconciliation session, which found and scoped this defect but did not fix
> it. This session makes the fix, and the researcher re-ran the full national
> pipeline (`2026-08-19_20-56`) to confirm it. Two code changes, one real
> value move. Nothing was committed.

---

## 1. What was wrong

The model prices two different things for each home's heating and cooling
system: the cost of the new heat pump ("upgrade" cost) and the cost the
household avoids by not having to replace their OLD system with another
like-for-like one ("replacement" cost, credited against the heat pump's cost
in the net capital cost and the NPV). Both costs come from the same REMDB v4
cost formula, which needs two facts about the equipment: its size and its
efficiency.

The 19 August session found that the replacement cost was being priced using
the NEW heat pump's size and efficiency, not the OLD system's -- even though
it is supposed to represent the cost of replacing the old system. This
happened because no column carrying the old system's own size or efficiency
survived the data pipeline past the point where costs are calculated. See
`SESSION_CHANGELOG_2026-08-19.md`, section 5, for how that session found and
measured the gap.

## 2. What changed, task by task

### Task 1 -- Confirm the map still holds (no edits)

Re-read all six files named in the 19 August trace and confirmed every line
number, column name, and claim was still accurate: the raw baseline EUSS file
does carry `out.params.size_heating_system_primary_k_btu_h` and
`out.params.size_cooling_system_primary_k_btu_h` for the home's existing
system; `hvac_heating_efficiency`/`hvac_cooling_efficiency` are copied from
the upgrade file's `in.hvac_*_efficiency`, which describes the home BEFORE
the upgrade; and `base_heating_efficiency`/`base_cooling_efficiency` already
exist, copied from the baseline file's own `in.hvac_*_efficiency`.

### Task 2 -- The efficiency fix (moves no numbers)

Two changes:

1. Added `base_size_heating_system_primary_k_btu_h` and
   `base_size_cooling_system_primary_k_btu_h` to `df_enduse_refactored`
   (`process_euss_data.py`), copied straight from the raw baseline file --
   the same simple pattern already used for `base_heating_efficiency`.
2. In `add_remdb_metrics` (`remdb_v4_installed_cost_utils.py`), switched the
   replacement case's efficiency source from `hvac_heating_efficiency` /
   `hvac_cooling_efficiency` to `base_heating_efficiency` /
   `base_cooling_efficiency`. The upgrade case is untouched.

**Verified this moved nothing.** Before touching the code, compared
`hvac_heating_efficiency` against `base_heating_efficiency` (and the cooling
pair) directly in the most recent National export
(`tepper_household_mp{3,4}_National_2026-08-19_13-19.csv`, 331,531 rows each):
**zero mismatches**, including on every one of the 260,211 / 250,576 homes
that actually get priced. The rename was safe by construction. Also confirmed
the new `base_size_*` columns are present and above zero for every included
home by joining the raw baseline file's size columns onto the export's
`include_heating`/`include_cooling` flags (0 of 260,211 / 250,576 at or below
zero). `pytest cmu_tare_model/tests/` -- 272 passed (one pre-existing,
unrelated `geopandas` import gap).

### Task 3 -- The size fix (moves numbers)

In `add_remdb_metrics`, split `capacity_col` by `metric_type` the same way
`efficiency_col` already was: the upgrade case keeps `size_heating_...` /
`size_cooling_...` (the heat pump's own size); the replacement case now reads
`base_size_heating_...` / `base_size_cooling_...` (the old system's own
size). Updated the function's docstring and the stale forward-looking
comments left by the 19 August session to describe the finished fix instead
of flagging it.

**Added a test** (`cmu_tare_model/tests/utils/test_remdb_v4_installed_cost_utils.py`,
new file -- no test previously existed for this function): one synthetic
home with a gas furnace (60 kBtu/h) and a much smaller heat pump (20 kBtu/h),
and a central AC (36 kBtu/h) versus the heat pump's cooling capacity
(24 kBtu/h). Confirms the heating replacement case reads the furnace's size,
the heating upgrade case reads the heat pump's size, and the cooling
replacement case reads the AC's size. All three pass.

**Checked the direction on real data before the re-run**, using only the raw
EUSS size columns (no pipeline run needed for this part): across all
331,531 homes, the old system is bigger than the new heat pump for 67-74% of
heating homes and smaller for 1-14% of cooling homes (see the CLAUDE.md
20 Aug row for the fuel/type breakdown) -- confirming the change moves
numbers in both directions, not just one.

## 3. The re-run and the comparison

The researcher restarted the kernel and re-ran the National pipeline for MP3
and MP4, `fixed_base` -- timestamp `2026-08-19_20-56`. Confirmed before
comparing: same 331,531 `bldg_id` values, same order, same
`include_heating`/`include_cooling` counts (260,211 / 250,576) as the
`2026-08-19_13-19` run being compared against. Per the documented income
random-seed caveat (the household-income draw shifts if a run covers a
different set of homes, which only affects the subsidized NPV cases), this
session's comparison uses the `_unsub` cases only.

**Nothing outside capital costs, NPV, and adoption moved.** Baseline lifetime
heating and cooling fuel costs, and all sixteen climate-damage means,
reproduced to the last cent/tonne against the 17 Aug and 19 Aug runs.

**The heating upgrade-cost column (the heat pump's own cost) is byte-identical**
to the pre-fix run for every included home -- confirms the fix touched only
the replacement-cost input, as intended.

**Replacement costs moved, in both directions:**

| | MP3 heating | MP4 heating | MP3 cooling | MP4 cooling |
|---|---|---|---|---|
| Mean replacement cost, old run | $3,929.24 | $3,717.46 | $5,710.46 | $5,549.29 |
| Mean replacement cost, new run | $3,866.45 | $3,866.45 | $5,047.27 | $5,047.27 |
| Change | -$62.79 | +$148.99 | -$663.19 | -$502.02 |

Note MP3 and MP4 now land on the exact same new-run mean for heating, and the
exact same new-run mean for cooling. That is not a coincidence or a bug: a
home's OLD system does not depend on which heat pump replaces it, so once the
replacement cost is priced off the old system alone, it should be identical
across MPs. Checked directly, per-home: **0 mismatches** between MP3's and
MP4's heating replacement cost, and 0 mismatches for cooling, across every
included home. This could not have held before the fix, since the old code
priced replacement off each MP's own differently-sized retrofit heat pump.

**By baseline heating fuel** (mean change in heating replacement cost):

| Fuel | n | MP3 | MP4 |
|---|---|---|---|
| Electricity (baseboard) | 73,699 | -$509.31 | +$177.65 |
| Natural Gas | 155,980 | +$109.12 | +$129.81 |
| Propane | 13,941 | +$155.87 | +$169.69 |
| Fuel Oil | 16,591 | +$120.71 | +$184.70 |

**By baseline cooling type** (mean change in cooling replacement cost):

| Type | n | MP3 | MP4 |
|---|---|---|---|
| Central AC | 189,896-190,165 | -$180.04 | -$199.35 |
| Room AC | 60,411 | -$2,181.90 | -$1,453.44 |

Room AC drops the most by far: a whole-home heat pump's cooling capacity is
much bigger than the single room unit it is credited as replacing, so the old
code (pricing off the heat pump's capacity) substantially overstated the
avoided-replacement credit for these homes. This lines up with CLAUDE.md's
existing note that Room AC-to-heat-pump is largely a service-level change.

**NPV identity and ordering checks hold on the new run.** For every home and
all nine NPV cases, `savings - net_capital_cost` matches the reported NPV to
within half a cent (rounding only). CLAUDE.md's ordering checks
(`heatingLCC_coolingLCC >= heatingLCC_coolingSavings` and
`>= heatingSavings_coolingLCC`) hold with 0 violations across 260,211 homes,
both MPs.

**Mean NPV and adoption, `_unsub` cases, `fixed_base`, National:**

| Case | MP3 old -> new | MP4 old -> new |
|---|---|---|
| `heatingSavings_coolingLCC` NPV | -$8,781.65 -> -$9,226.39 | -$9,555.69 -> -$9,915.62 |
| `heatingSavings_coolingLCC` adoption | 15.9924% -> 15.5862% (41,614 -> 40,557) | 11.5314% -> 11.1317% (30,006 -> 28,966) |
| `heatingLCC_coolingSavings` NPV | -$9,808.43 -> -$9,871.23 | -$10,709.45 -> -$10,560.46 |
| `heatingLCC_coolingSavings` adoption | 12.3550% -> 12.9337% (32,149 -> 33,655) | 9.7421% -> 10.1552% (25,350 -> 26,425) |
| `heatingLCC_coolingLCC` NPV | -$4,852.41 -> -$5,359.94 | -$5,838.23 -> -$6,049.17 |
| `heatingLCC_coolingLCC` adoption | 27.7140% -> 27.1760% (72,115 -> 70,715) | 18.4416% -> 18.0984% (47,987 -> 47,094) |

The pattern makes sense given the cost changes above: `heatingSavings_coolingLCC`
(cooling replacement credit only) drops the most, because cooling's
replacement credit fell sharply, especially for Room AC homes.
`heatingLCC_coolingSavings` (heating replacement credit only) rises slightly
on net, because most fossil-fuel heating homes gained replacement credit.
`heatingLCC_coolingLCC` (both credits) falls overall because cooling's drop
outweighs heating's gain.

**Adopter crossings for `heatingLCC_coolingLCC_unsub`, by baseline heating
fuel:**

| Fuel | MP3 gained / lost | MP4 gained / lost |
|---|---|---|
| Electricity | 902 / 2,431 | 891 / 1,474 |
| Natural Gas | 764 / 518 | 168 / 334 |
| Propane | 18 / 114 | 31 / 123 |
| Fuel Oil | 53 / 74 | 48 / 100 |

Electric-baseboard homes lose the most adopters net -- consistent with their
replacement credit falling the most (MP3: -$509.31 mean). Natural Gas gains
more than it loses for MP3, consistent with its replacement credit rising.

**By baseline cooling type** (same case, same crossing direction):

| Type | MP3 gained / lost | MP4 gained / lost |
|---|---|---|
| Central AC | 1,466 / 2,603 | 937 / 1,110 |
| Room AC | 50 / 438 | 43 / 812 |

Room AC homes lose adopters heavily relative to their population size (438
of 43,564 for MP3, 812 of 43,564 for MP4), matching the sharp drop in their
cooling replacement credit.

## 4. Comments and documentation updated

Updated the content of the five comments/docstrings flagged by the 19 August
session to describe the finished fix (not just flag it), in
`process_euss_data.py` (both FOLLOW-UP blocks in `df_enduse_compare`),
`calculate_lifetime_private_impact.py` (the `calculate_capital_costs`
docstring Notes, its inline comment, and a new comment added in
`calculate_private_npv` at the point where the cooling replacement credit is
actually subtracted), and `calculate_equipment_replacement_costs.py` (the
`calculate_replacement_installed_cost` description block, which also notes
the dead v3 path still has the old bug). Added a one-line note next to
`REMDB_COST_SCENARIO_KEYS` in `constants.py` pointing at the same v3 note.

**A separate conciseness pass was also done this session, but only on
`remdb_v4_installed_cost_utils.py`** (the module docstring, the two row_id
helper comments, the `_apply_efficiency_floor` docstring, and the
capacity/efficiency comments written in Tasks 2-3 above, which were the most
verbose). **Flagged for a future session:** the same conciseness pass has
not been applied to the other five files touched this session
(`process_euss_data.py`, `calculate_lifetime_private_impact.py`,
`calculate_equipment_replacement_costs.py`, `constants.py`) -- their comments
are correct as of this session but were not tightened for length.

## 5. What this session did not touch (left for later, as scoped)

1. **The dead v3 cost path** (`calculate_equipment_replacement_costs.py`,
   `_calculate_replacement_cost_per_row`) still has the old bug -- it reads
   `size_heating_system_primary_k_btu_h`, not the baseline size. Harmless
   today because `REMDB_COST_SCENARIO_KEYS` only runs `'v4MID'`, so v3 never
   executes. Apply the same fix there if v3 is ever turned back on.
2. **`validate_capital_costs.py`'s size-based groupings** still use the heat
   pump's size, the same problem in a report-only script. Not fixed.
3. **The household-income random-seed issue** that limited this session's
   comparison to `_unsub` cases is still open -- needs its own investigation
   and a CLAUDE.md note, not done here.
4. **Carried over from 19 August, still open:** wall/floor-furnace handling,
   the CBSA/MSA column, envelope details, and 6-decimal consumption.
5. **The comment-conciseness pass** on the four files listed in section 4,
   above.
6. **Other MPs and discount rates.** This session's re-run covered MP3 and
   MP4, `fixed_base`, National only -- the same scope as every prior full run
   in this project. `fixed_low`, `fixed_high`, and `variable` were not
   re-run.
