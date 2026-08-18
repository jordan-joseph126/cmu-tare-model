# Session Changelog -- 2026-08-12

## Anchor Year Fix: cost streams move from 2024-2038 to 2025-2039

> Branch `joseph-2026-nature-comms-submission`. This is an **intended value
> change**: fuel costs, NPV, and adoption all move, and the corrected numbers
> become the reference for later sessions. `.py` and test files only -- the
> `.ipynb` files were not touched (never-edit-`.ipynb` rule); the one notebook
> change needed is handed off below. Nothing was committed.

---

## 1. What was wrong

The model is meant to run 2025-2039. The code built its year labels starting
from 2024 in two places that agreed with each other, so nothing ever raised an
error:

- `year_label = year + 2023` in `calculate_lifetime_fuel_costs.py`
- `base_year: int = 2024` in `calculate_private_npv` and
  `_calculate_discounted_savings`, which then computed
  `year_label = year + (base_year - 1)`

The Task 1 audit found that a **previous session had already accommodated this**
by inventing a year 2024 in all three data sources, each one a copy of 2025:

| Source | The 2024 workaround | Removed in |
|---|---|---|
| Fuel prices (`create_lookup_fuel_prices.py`) | `FIRST_CALC_YEAR = 2024` plus a block writing factor 1.0 for 2024 | Section 2 |
| Degree days (`degree_day_consumption_utils.py`) | `.get(year_label, 1.0)` -- 2024 was simply absent and silently returned "no adjustment" | Section 2 |
| Climate emissions (`create_lookup_emissions_electricity_climate.py`) | `CLIMATE_FIRST_MODEL_YEAR = 2024` plus a block copying the 2025 rows back to 2024 | Section 7 |

All three are gone. There is now exactly one anchor year in the project,
`ANCHOR_YEAR` in `constants.py`, and no data source carries an invented 2024.

So no home was ever priced at zero. The real effect was that the stream ran
`[2025, 2025, 2026, ..., 2038]` -- one duplicated first year, and year 2039
never reached. The corrected stream is `[2025, 2026, ..., 2039]`.

**Core principle adopted this session:** 2025 is the anchor year. Every
projection factor is relative to 2025 and the 2025 factor is exactly 1.0. The
data starts in 2025. Nothing invents, synthesizes, or defaults a value for a
year outside the data; asking for one is a programming error and fails loudly.

---

## 2. Files edited in the working tree

| File | Change | Value move? |
|---|---|---|
| `cmu_tare_model/constants.py` | `ANCHOR_YEAR` comment broadened; added `PROJECTION_END_YEAR = 2050` | No |
| `cmu_tare_model/private_impact/data_processing/create_lookup_fuel_prices.py` | Deleted `FIRST_CALC_YEAR` and the 2024 block; imports the shared `ANCHOR_YEAR`; added load-time checks | Yes (removes year 2024) |
| `cmu_tare_model/private_impact/calculate_lifetime_fuel_costs.py` | `year_label = year + (ANCHOR_YEAR - 1)`; price lookup raises on a missing year and returns NaN on an unmapped fuel; fixed the year-span message | Yes |
| `cmu_tare_model/utils/degree_day_consumption_utils.py` | Missing year now raises instead of defaulting to 1.0; range guard follows `ANCHOR_YEAR`; added load-time checks; removed the `try/except` that swallowed a failed file load | Yes (removes year 2024) |
| `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py` | `base_year` **removed as a parameter** from `calculate_private_npv` and `_calculate_discounted_savings`; both now derive the start year from `ANCHOR_YEAR` | Yes (the year fix); the parameter removal itself moves nothing |
| `cmu_tare_model/tests/private_impact/test_calculate_lifetime_private_impact.py` | Dropped the three `base_year=BASE_YEAR` arguments; added the `mp3_heating_rebate_amount_v4MID` fixture column | No |
| `cmu_tare_model/utils/validation_framework.py` | `replace_small_values_with_nan` keeps an exact `0.0` (Section 6a). **Logged exception to the never-edit rule** | Yes (Option B) |
| `cmu_tare_model/public_impact/calculate_lifetime_climate_impacts_sensitivity.py` | `base_year` removed; year loop and span message derive from `ANCHOR_YEAR` (Section 7) | Yes (climate only) |
| `cmu_tare_model/public_impact/data_processing/create_lookup_emissions_electricity_climate.py` | Deleted the invented 2024 rows and both local year constants (Section 7) | Yes (climate only) |
| `cmu_tare_model/tests/utils/test_validation_framework.py` | Updated to the exact-zero contract; new pinning test; fixture patches every copy of the constants | No |
| `cmu_tare_model/tests/public_impact/test_calculate_lifetime_climate_impacts_sensitivity.py` | 5-tuple mocks, current scenario string and prefix, fixture patches every copy of the constants | No |
| `cmu_tare_model/tests/conftest.py` | `BASE_YEAR` and `create_sample_homes_df` follow `ANCHOR_YEAR` | No |
| `cmu_tare_model/tests/utils/test_degree_day_consumption_utils.py` | Year arguments follow `ANCHOR_YEAR` | No |
| `cmu_tare_model/tests/utils/test_discounting.py` | Base and target years follow `ANCHOR_YEAR` | No |
| `cmu_tare_model/tests/public_impact/calculations/test_calculate_fossil_fuel_emissions.py` | Year arguments follow `ANCHOR_YEAR` | No |

The three deletions and the year-loop fix were made together, so the repository
was never left in a half-fixed state where the loop asks for a year the data no
longer carries.

---

## 3. One anchor year, one place

`ANCHOR_YEAR = 2025` now lives only in `constants.py`. The duplicate in
`create_lookup_fuel_prices.py` was deleted and that module imports the shared
one. `PROJECTION_END_YEAR = 2050` was added alongside it so the load-time checks
have an end of range to test against without scattering the literal.

`CLIMATE_ANCHOR_YEAR = 2025` in
`create_lookup_emissions_electricity_climate.py` was the same idea under a
different name -- the first year the Cambium 2024 data covers -- and was used
for exactly one thing: selecting the 2025 rows to copy back to 2024. It and
`CLIMATE_FIRST_MODEL_YEAR` were both deleted along with the copy block in the
climate step (Section 7), and that module now imports the shared `ANCHOR_YEAR`.

**Result: `ANCHOR_YEAR` in `constants.py` is the only anchor year in the
project.** No module defines its own, and no data source carries a synthesized
2024.

---

## 4. How a missing value now behaves

The old price lookup ended in `.get(year_label, 0)`, so any miss at any level
became a price of zero. It now separates two causes:

- **Missing year, region, or policy scenario** -> raises, naming the region,
  fuel, policy scenario, and year, and reporting the range the data covers.
  If the fuel resolved, the price table was meant to carry that combination,
  so a miss means the model asked for something the data does not contain.
- **Unmapped fuel** -> returns NaN, not zero. About 10,000 ResStock homes burn
  `'Other Fuel'` or record no heating fuel, so `FUEL_MAPPING` leaves them
  blank. That is a real condition in the source data, and crashing on it would
  make the model unrunnable. **Verified before making this change: zero homes
  with `include_heating = True` have an unmapped heating fuel** -- all 10,174
  are already excluded from results. NaN is used rather than zero because a
  zero price would silently pull down any average it reached.

The degree-day lookups now raise on a missing year for the same reason: a
factor of 1.0 reads as a real answer meaning "no degree-day adjustment," which
would quietly leave that year's heating or cooling energy unscaled. An
unfamiliar census division may still fall back to the `National` row -- that is
a deliberate allowance for a region the file does not list, not a silent
default for missing data.

---

## 5. Load-time checks added

Both projection files are now checked when their module is imported, so the
model refuses to start on bad data rather than running and producing subtly
wrong numbers. Each check asserts:

1. The year keys are exactly `ANCHOR_YEAR` through `PROJECTION_END_YEAR`, with
   no gaps and no year below `ANCHOR_YEAR`.
2. Every `ANCHOR_YEAR` factor is exactly 1.0, for every region and every fuel
   (fuel prices) or every census division and both heating and cooling
   (degree days).

**What the checks found on the current CSVs: both pass, with nothing to
report.** `aeo2026_fuel_price_factors_2025_2050.csv` carries 40 rows covering
2025-2050 with all 40 anchor factors at exactly 1.0.
`aeo2026_degree_day_factors_2025_2050.csv` carries 20 rows (10 census divisions
including `National`, heating and cooling) covering 2025-2050, all anchor
factors exactly 1.0. Neither file needed a change.

`degree_day_consumption_utils.py` previously wrapped its file load in a
`try/except` that printed a warning and left both lookups empty on failure,
which would have produced a whole model run with no degree-day adjustment at
all. That was removed; a failed load now stops the run.

---

## 6. Checks run, and results

### Checked and reported, not fixed

- **The fuel price CSV carries exactly one policy scenario.** The builder
  hardcodes `'2025 Reference Case'` and ignores the CSV's own column, so two
  scenarios in that file would silently overwrite each other. There is no risk
  today: the anchor price file
  (`eia_fuel_price_data_2025_usd2025.csv`) has no `policy_scenario` column at
  all, and the factors file has one value
  (`'AEO2026 Counterfactual Baseline'`) across 40 rows with 40 unique
  region/fuel pairs and no duplicates.

### KNOWN DEFECT (recorded, not fixed)

`cmu_tare_model/grid_impact/calculate_postTARE_ts_aws_peak_demand_EXPORT_23July2026.py:826`
reads

```python
_df_tare = DATAFRAMES_BY_MP[_mp]['fixed_base']['inmap']
```

`DATAFRAMES_BY_MP[mp]['fixed_base']` is a DataFrame. The trailing `['inmap']`
is left over from the retired structure that nested results under a health
damage model, and it would fail on any run that reaches this line. Every other
line in that file, and everywhere in the main notebook, correctly treats
`['fixed_base']` as the DataFrame itself. Not fixed this session -- out of
scope, and the file is a notebook export rather than live module code.

### Behaviour checks (all pass)

- Every year 2025-2039 returns a real, positive price for electricity, natural
  gas, fuel oil, and propane across state and census-division regions.
- Asking for 2024 or 2051 raises, naming region, fuel, scenario, year, and the
  range the data covers.
- Asking for a retired scenario string raises and lists the scenarios available.
- An unmapped fuel returns NaN.
- Heating and cooling degree-day factors are exactly 1.0 at 2025 and raise on
  2024.
- The corrected baseline stream builds 15 year columns, 2025-2039, with no
  column for any year outside it. 99.50% of included homes have a positive cost
  in every one of the 15 years (the remaining 0.50% have zero baseline heating
  consumption).
- The 71,320 excluded homes carry NaN in the year columns, never zero.

### Test suite

Full suite before the change: **14 failed, 254 passed**. After: **14 failed,
254 passed** -- the same 14, all pre-existing and unrelated (they assert
equipment categories such as `waterHeating` and `cooking` that were removed
from `EQUIPMENT_SPECS` long ago). The pre-existing baseline was measured by
extracting an unmodified copy of the repository at `HEAD` and running the same
command there.

`cmu_tare_model/tests/adoption_kpis/` was excluded from both runs: it cannot be
collected because `geopandas` is not installed in this environment. That is an
environment gap, not a code defect, and it predates this session.

---

## 6a. Adoption numbers, reconciled

The adoption rate is the mean of the economic-adopter column, so its
**denominator is 260,211** -- the number of homes with a non-null adopter flag,
which equals the `include_heating = True` count. It is not 331,531 (every home
in the file) and not 258,932 (homes with a usable NPV).

The 510 figure is the gross count of homes moving in **either** direction:

| | Homes |
|---|---:|
| Crossed up (non-adopter -> adopter) | 237 |
| Crossed down (adopter -> non-adopter) | 273 |
| Gross total moving | 510 |
| **Net change in adopters** | **-36** |

-36 / 260,211 = -0.0138 pp, which matches the rate moving from 18.4493% to
18.4354%. Total adopters go from 48,007 to 47,971.

### The 1,279 homes with valid heating but no usable NPV

These are **not** the same homes as the 1,314 with zero baseline heating
consumption. The two sets overlap in 1,260 homes but neither contains the
other. This is pre-existing behaviour, unchanged by the anchor-year fix -- the
count is 1,279 in both the old and the new run.

A home ends up with a NaN NPV when either its heating or its cooling
discounted lifetime saving comes out **exactly zero**. `_calculate_discounted_savings`
passes its result through `replace_small_values_with_nan`, which converts
anything below 1e-10 to NaN, and NPV is savings minus capital, so a NaN on
either side makes the whole NPV NaN. Two causes account for all 1,279 with no
remainder:

| Cause | Homes |
|---|---:|
| Heating saving exactly zero (baseline and heat-pump heating energy identical -- almost always both zero) | 1,265 |
| Has AC, and baseline and heat-pump **cooling** energy are identical, so the cooling saving is exactly zero | 14 |
| **Total** | **1,279** |

The 54 homes that have zero baseline heating consumption yet still get a usable
NPV are the mirror image: their heat pump does draw some heating electricity,
so the saving is negative rather than zero, and a negative number survives the
small-value filter.

**Worth knowing, not fixed here:** all 1,279 of these homes are counted as
non-adopters (adopter flag 0.0) rather than being excluded. The rule in
`economic_adoption_decision` is `valid & (npv >= 0)`, and a NaN compared with
`>=` is False, so a home with a NaN NPV silently becomes a 0. The function's
docstring says homes without usable data get NaN, which is true for invalid
heating data but not for this case. Flagged for a future decision -- out of
scope this session. Two ways of fixing it were computed below (not applied).

### Two candidate fixes, computed but NOT applied

**A correction first:** an earlier draft of this section called the effect
"roughly 0.5 percentage points." That was a units error -- 1,279 / 260,211 =
0.49% is the SHARE OF HOMES in that group, not a move in the adoption rate.
The actual rate move for option (a) is +0.09 pp, below.

**(a) Exclude the 1,279 from the denominator.** None of them are currently
counted as adopters, so the numerator (47,971) is unchanged and only the
denominator shrinks:

| | Current | Option (a) |
|---|---:|---:|
| Denominator | 260,211 | 258,932 |
| Adopters | 47,971 | 47,971 |
| Rate | 18.4354% | 18.5265% |

Change: **+0.0911 pp**.

**(b) Stop converting an exact-zero saving to NaN; keep the tiny-artifact
filter for genuinely nonzero values only.** The NaN comes from
`replace_small_values_with_nan` in `_calculate_discounted_savings`, which
sends anything with `abs(value) <= 1e-10` to NaN -- that range includes an
honest 0.0, not just floating-point noise. Patched (for this measurement
only) to leave an exact 0.0 as 0.0 and only NaN a nonzero value that rounds to
near-zero. Every one of the 1,279 then gets a real NPV, and nothing outside
that group changes:

| Group | n | Became adopter | Stayed non-adopter | NPV range |
|---|---:|---:|---:|---|
| Heating-zero | 1,265 | 13 | 1,252 | -$43,422 to +$7,129 |
| Cooling-zero | 14 | 3 | 11 | -$20,145 to +$6,153 |
| **Total** | **1,279** | **16** | **1,263** | |

| | Current | Option (b) |
|---|---:|---:|
| Denominator | 260,211 | 260,211 |
| Adopters | 47,971 | 47,987 |
| Rate | 18.4354% | 18.4416% |

Change: **+0.0061 pp**. Zero homes outside the 1,279 changed under this patch,
confirming it touches exactly the target group.

**Your expectation was partly right.** Most of the 1,265 heating-zero homes do
stay non-adopters (1,252 of 1,265, 99%) -- that part holds. But the 14
cooling-zero homes are not simply "the ones currently misclassified": only 3
of the 14 flip to adopter under option (b); the other 11 have a real NPV that
is still negative. So the cooling-zero group is not uniformly mislabeled
either -- it splits the same way the heating-zero group does, just at much
smaller n.

Options (a) and (b) disagree with each other by about 0.085 pp (18.5265% vs
18.4416%) and both disagree with leaving the code as-is. This is a modeling
decision, not a mechanical fix -- deferred to the researcher.

### DECISION: Option B implemented

The researcher chose Option B. Implemented in
`utils/validation_framework.py`, in `replace_small_values_with_nan` only:

```python
# before -- an exact 0.0 fails "abs(x) > threshold" and becomes NaN
return series_or_dict.where(abs(series_or_dict) > threshold, np.nan)

# after -- an exact 0.0 is kept; only a tiny NONZERO value is an artifact
keep = (series_or_dict == 0) | (abs(series_or_dict) > threshold)
return series_or_dict.where(keep, np.nan)
```

The same change is applied to the DataFrame branch; the dict branch recurses
and needed no change. Values that were already NaN stay NaN.

**This required an explicit exception to the never-edit rule on
`utils/validation_framework.py`,** granted by the researcher on 12 Aug 2026 and
recorded in CLAUDE.md next to that rule. The exception is one-off and does not
generalize. The fix was deliberately placed in the shared function rather than
duplicated as a local helper in `calculate_lifetime_private_impact.py`, so
every caller behaves consistently. Scope of that decision: the only live caller
is `_calculate_discounted_savings`; the other caller sits in
`DEPRECATED_health_impacts/` and is dead code.

**Confirmed identical to the measured Option B numbers**, i.e. moving the fix
from the measurement's local patch into the shared function changed nothing:

| | Measured (local patch) | Implemented (shared function) |
|---|---:|---:|
| Homes still NaN NPV | 0 | 0 |
| Homes with a usable NPV | 260,211 | 260,211 |
| Adopters | 47,987 | 47,987 |
| Adoption rate | 18.4416% | 18.4416% |
| Mean lifetime heating fuel cost | $20,362.56 | $20,362.56 |
| Mean lifetime cooling fuel cost | $10,097.37 | $10,097.37 |

One knock-on worth recording: the **mean NPV moves from -$5,816.35 to
-$5,838.23**. No already-valued home changed. The mean moves purely because the
1,279 newly-valued homes -- mostly large negative NPVs -- now enter the average,
so the denominator goes from 258,932 to 260,211. Both new figures are in the
CLAUDE.md golden table as their own labeled rows.

Two tests in `tests/utils/test_validation_framework.py` encoded the old
behaviour (asserting `0.0` becomes NaN) and were updated to the new contract. A
new test, `test_replace_small_values_with_nan_keeps_exact_zero`, pins it: exact
`0.0` and `-0.0` survive, `1e-11` is still filtered.

---

## 6b. `base_year` removed as a parameter

Fixing the notebook line would have fixed today only; the parameter would still
let any future caller silently run the model over years the projection data
does not cover, which is the exact failure this session removed. So
`base_year` is gone from both `calculate_private_npv` and
`_calculate_discounted_savings`; both derive the start year from `ANCHOR_YEAR`.
`calculate_discount_factors` keeps its `base_year` argument -- it is a general
arithmetic helper with no notion of the projection data -- and is now called
with `ANCHOR_YEAR`.

**Confirmed value-neutral.** All four measured quantities are byte-identical
before and after the removal. Note these are the numbers **as they stood at
this point in the session** -- the anchor-year fix only, before Option B was
implemented in Section 6a. Both sides of this comparison are on the same
footing, which is what makes it a valid value-neutrality check; the adoption
rate and mean NPV later move to 18.4416% and -$5,838.23 once Option B lands.

| Quantity | Before removal | After removal |
|---|---:|---:|
| Mean lifetime heating fuel cost | $20,362.56 | $20,362.56 |
| Mean lifetime cooling fuel cost | $10,097.37 | $10,097.37 |
| Mean MP4 `heatingLCC_coolingLCC_unsub` NPV | -$5,816.35 | -$5,816.35 |
| Adoption rate | 18.4354% | 18.4354% |
| Homes with a usable NPV | 258,932 | 258,932 |

No caller needs a different base year. The only call sites are the notebook
cell in Section 8 and three tests; **no `tare_model_main_v2_3_EXPORT_*.py` file
carries the argument** (checked all five).

Side effect worth noting: passing `base_year` now raises `TypeError`. The
notebook line in Section 8 will therefore fail loudly instead of quietly
reverting the run to the old window.

---

## 6c. Environment gap: `cmu_tare_model/tests/adoption_kpis/`

Two separate problems, and geopandas is only the first.

**1. geopandas is not installed.** The environment is the Anaconda base
install at `C:\Users\jorda\AppData\Local\anaconda3`, Python 3.12.12. Missing:
`geopandas`, `shapely`, `pyproj`, and a file-IO backend (`fiona`/`pyogrio`).
`rtree` 1.4.1 is already present. A dependency resolution (dry run, nothing
installed) comes back clean:

```
Would install geopandas-1.1.4 pyogrio-0.13.0 pyproj-3.7.2 shapely-2.1.2
```

All four ship prebuilt Windows wheels for Python 3.12, so `pip install
geopandas` needs no compiler and no GDAL setup, and pip reported no conflicts
with the installed numpy 2.3.5 / pandas 2.3.3. The alternative, `conda install
-c conda-forge geopandas`, is the traditionally safer route on Windows but
would be solving into the Anaconda base environment, which is slower and more
disruptive. Recommendation: pip. **Nothing was installed.**

`geopandas` is imported only by `adoption_kpis/visualize_geospatial_data.py`,
which the package `__init__.py` re-exports -- so a missing map library blocks
every test in the folder, including ones that never touch a map.

**2. `test_kpi_functions.py` is stale and geopandas will not fix it.**
Verified by standing in a placeholder `geopandas` module and re-running (no
package installed, per instruction): the import chain then resolves, and the
test file fails with a different error. It imports 23 names from
`cmu_tare_model.adoption_kpis.kpi_functions`, **a module that no longer exists
anywhere in the repository.** It was split into `data_loading.py`,
`spark_gap.py`, `thermal_cop.py`, `demand.py`, `bill_savings.py`, and
`compute_adoption_rate.py`, and the test was never updated.

20 of the 23 names exist in the split modules and only need the import block
repointed to the right file:

| Name | Now lives in |
|---|---|
| `BTU_PER_CF_NATURAL_GAS`, `BTU_PER_KWH`, `DWELLING_UNIT_WEIGHT`, `HEATING_FUEL_COLS`, `load_euss_baseline`, `load_euss_upgrade`, `mp_to_upgrade` | `data_loading.py` |
| `HEATING_LOAD_COL`, `HP_BACKUP_ELEC_COL`, `HP_FANS_PUMPS_COL`, `KBTU_PER_KWH`, `COP_BENCHMARK_RANGES`, `compute_breakeven_cop`, `compute_thermal_cop`, `iecc_to_cz_group` | `thermal_cop.py` |
| `KWH_PER_MMBTU`, `NG_CONVERSION_FACTOR`, `STATE_NAMES` | `spark_gap.py` |
| `aggregate_demand_by_state`, `compute_scenario_demand` | `demand.py` |

(`HEATING_LOAD_COL`, `HP_BACKUP_ELEC_COL`, `HP_FANS_PUMPS_COL`, and
`KBTU_PER_KWH` are also re-exported from `data_loading.py`; either import
source works.)

3 do not exist anywhere under any name and were renamed or dropped:
`calculate_price_ratios`, `compute_spark_gap_metrics`,
`compute_thermal_cop_by_state`. The researcher will decide what these should
be before the next session.

**3. The sibling file is healthy.** `test_peak_load_functions.py` runs
**32 tests, all passing**, with only the placeholder import in place. Installing
geopandas unblocks those 32 immediately. `test_kpi_functions.py` needs its
import block rewritten and a decision on the three missing names before it can
run at all.

---

## 6d. KNOWN STALE TESTS (recorded for a future cleanup)

**Correction to an earlier draft of this section:** it claimed all 14
pre-existing failures were "one problem," an `EQUIPMENT_SPECS` mismatch. That
was checked properly this round by reading each failure's actual error
message, and it is wrong for 13 of the 14. There are three distinct causes:

| File | Failing tests | Actual cause |
|---|---:|---|
| `tests/private_impact/test_calculate_lifetime_fuel_costs.py` | 6 | Test fixtures still pass the retired scenario string `'AEO2023 Reference Case'`; `define_scenario_params` now returns 5 values and the retired-scenario branch raises `ValueError: too many values to unpack (expected 5)` |
| `tests/public_impact/test_calculate_lifetime_climate_impacts_sensitivity.py` | 4 | Same retired-scenario cause as above |
| `tests/private_impact/test_calculate_lifetime_private_impact.py` | 3 | Test fixtures for MP3 are missing the `mp3_heating_rebate_amount_v4MID` column that `calculate_capital_costs` now requires (MP3 became rebate-eligible in the 12 Jul 2026 session); raises `KeyError` naming the missing column |
| `tests/utils/test_validation_framework.py` | 1 | Genuinely an `EQUIPMENT_SPECS` mismatch, but **only when this test runs after `test_calculate_lifetime_climate_impacts_sensitivity.py` in the same session** -- it passes on its own (40/40) and passes when paired with the fuel-costs file. Something in the climate test file's `mock_constants` fixture leaks state across test modules; not diagnosed further here since it is not this session's territory |

Confirmed for all three NPV tests (the ones this session's edits touched
directly): the failure is byte-identical, including the exact `KeyError`
message, in an unmodified copy of the repository extracted at `HEAD`. This
session's `base_year` removal did not change what they fail on.

### FIXED -- 12 Aug 2026

All 14 were repaired rather than left red. No test was deleted; every one was
still asserting something real, only against stale fixtures.

**The 10 "retired scenario string" failures had a second cause underneath.**
The visible error was `too many values to unpack (expected 5)`, wrapped in a
message naming `'AEO2023 Reference Case'` -- but the scenario string was not
what broke them. `define_scenario_params` returns 5 values; the test mocks
still returned the old 6-tuple. Both were fixed:

- mock return values trimmed to the current 5-tuple
  `(scenario_prefix, cambium_scenario, fossil_lookup, electricity_lookup, fuel_prices)`
- `policy_scenario='AEO2023 Reference Case'` -> `'2025 Reference Case'`
- price-fixture scenario keys rebuilt on the single current scenario
- retired column prefix `iraRef_mp8_` -> `ref2025_mp8_`

**The 3 NPV failures** needed `mp3_heating_rebate_amount_v4MID` in the
`npv_cases_df` fixture -- MP3 became rebate-eligible in the 12 Jul 2026 ENERGY
STAR session, so the capital-cost step now requires it. Added at `0.0` so the
subsidized and unsubsidized cases carry identical capital and the fixture's
exact NPV arithmetic (`8550 + 2850 - 8000` and friends) still reads clearly.
Note this leaves the subsidized rebate arithmetic uncovered by these
particular tests; it is covered by the dedicated rebate tests.

**The 1 order-dependent failure was not a leaking fixture.**
`validation_framework.py` (and `calculate_lifetime_fuel_costs.py`) do
`from cmu_tare_model.constants import EQUIPMENT_SPECS, ...`, which copies those
names into the importing module's own namespace the first time it loads.
Patching `cmu_tare_model.constants.EQUIPMENT_SPECS` never touches those copies,
so whether a test saw the patched value came down to which test file imported
the module first -- passing alone, failing after the climate tests. Fixed by
patching every copy in the two `mock_constants` fixtures, not by reordering
anything.

That also closed a second, separately-flagged order bug: the `FOLLOW-UP` note
at the top of `test_calculate_lifetime_fuel_costs.py` described a `KeyError:
'waterHeating'` when the `private_impact` folder ran on its own. Same root
cause, fixed by the same change; the note has been replaced with a short record
of the fix. Verified: `pytest cmu_tare_model/tests/private_impact/` now passes
62/62 on its own, and `test_validation_framework.py` passes both alone and
immediately after the climate tests.

**Also observed, not changed:** `validate_common_parameters` in
`utils/calculation_utils.py` still whitelists the two retired scenario strings
(`'No Inflation Reduction Act'`, `'AEO2023 Reference Case'`) alongside
`'2025 Reference Case'`. Nothing passes them any more, but the whitelist is
what let the stale tests look like a scenario problem. Worth pruning in a
future cleanup.

---

## 7. Climate module -- the third 2024 workaround (DONE, own step)

Done as a separate step after the fuel-cost and NPV work was approved, so its
value move could be measured on its own.

### What changed

| File | Change |
|---|---|
| `public_impact/calculate_lifetime_climate_impacts_sensitivity.py` | `base_year` **removed as a parameter**; `year_label = year + (ANCHOR_YEAR - 1)`; year-span message corrected (it reported 16 years for a 15-year stream, same bug as the fuel-cost message) |
| `public_impact/data_processing/create_lookup_emissions_electricity_climate.py` | Deleted `CLIMATE_ANCHOR_YEAR` and `CLIMATE_FIRST_MODEL_YEAR` and the block that copied the 2025 rows back to 2024; imports the shared `ANCHOR_YEAR` and now checks the Cambium data actually starts there |

`CLIMATE_ANCHOR_YEAR` was the same idea as `ANCHOR_YEAR` under a different
name, and had no users outside its own file. Both local constants are gone;
there is now one anchor year in the project. The climate emissions lookup
covers 2025-2050 (26 years) with no 2024 entry.

The notebook needs **no edit** for this: cell 10 of `tare_scenarios_v2_3.ipynb`
never passed `base_year` to `calculate_lifetime_climate_impacts`.

### Before / after -- mean climate damages, National, all 331,531 homes

Baseline lifetime values (per home, USD except the tonne rows):

| Column | Before | After | Change |
|---|---:|---:|---:|
| `baseline_heating_lifetime_mt_co2e_lrmer` | 70.76 | 69.42 | -1.89% |
| `baseline_heating_lifetime_damages_climate_lrmer_central` | 18,475.70 | 18,377.40 | -0.53% |
| `baseline_heating_lifetime_mt_co2e_srmer` | 82.33 | 80.69 | -2.00% |
| `baseline_heating_lifetime_damages_climate_srmer_central` | 21,490.06 | 21,350.01 | -0.65% |
| `baseline_cooling_lifetime_mt_co2e_lrmer` | 17.27 | 16.05 | -7.07% |
| `baseline_cooling_lifetime_damages_climate_lrmer_central` | 4,431.04 | 4,174.75 | -5.78% |
| `baseline_cooling_lifetime_mt_co2e_srmer` | 33.28 | 32.02 | -3.77% |
| `baseline_cooling_lifetime_damages_climate_srmer_central` | 8,629.15 | 8,415.79 | -2.47% |

MP4 avoided values (the quantity the manuscript reports):

| Column | Before | After | Change |
|---|---:|---:|---:|
| `ref2025_mp4_heating_avoided_mt_co2e_lrmer` | 57.26 | 56.94 | -0.55% |
| `ref2025_mp4_heating_avoided_damages_climate_lrmer_central` | 15,009.87 | 15,131.46 | +0.81% |
| `ref2025_mp4_heating_avoided_mt_co2e_srmer` | 53.93 | 53.77 | -0.30% |
| `ref2025_mp4_heating_avoided_damages_climate_srmer_central` | 14,149.64 | 14,298.19 | +1.05% |
| `ref2025_mp4_cooling_avoided_mt_co2e_lrmer` | 6.69 | 6.21 | -7.18% |
| `ref2025_mp4_cooling_avoided_damages_climate_lrmer_central` | 1,714.37 | 1,613.45 | -5.89% |
| `ref2025_mp4_cooling_avoided_mt_co2e_srmer` | 12.49 | 12.01 | -3.88% |
| `ref2025_mp4_cooling_avoided_damages_climate_srmer_central` | 3,238.56 | 3,154.99 | -2.58% |

Emissions fall almost everywhere because the grid keeps decarbonizing, so
shifting the window one year later prices a cleaner grid. Cooling moves most
(-4% to -7% on tonnes) because it is entirely electric, so it feels the full
grid change, while heating is mostly fossil combustion whose emission factor
does not change year to year. Avoided heating DAMAGES rise slightly (+0.8% to
+1.1%) even though avoided heating tonnes fall, because the social cost of
carbon climbs over time and the later window prices each tonne higher -- the
per-tonne price rises faster than the tonnage falls.

### Nothing feeding the adoption decision moved -- confirmed

Adoption depends only on the private NPV, which is built from fuel costs and
capital costs. Climate damages never enter it, by design (CLAUDE.md).

**A first version of this check was invalid and was redone.** It fed the
exported MP4 CSV straight into the climate module. Every NPV, adopter and
fuel-cost column in that file was produced by the OLD code, so both sides of
the comparison carried stale values (adopter count 48,007) and the check never
exercised the anchor-year fix or Option B at all. It showed only that the
climate module does not disturb columns it passes through -- true, but not the
claim being made.

Redone on a frame rebuilt with the current code: baseline and MP4 fuel costs,
private NPV, and adopter flags all regenerated (anchor-year fix + Option B),
giving **47,987 adopters**, 260,211 usable NPVs, mean NPV -$5,838.23 -- the
numbers the model produces today. The climate module was then run on that
frame, and all 42 columns matching `private_npv`, `econ_adopter`,
`capital_cost`, `fuel_cost` or `rebate_amount` were compared:

| | Before climate run | After climate run |
|---|---:|---:|
| Adopter count | 47,987 | 47,987 |
| Usable NPVs | 260,211 | 260,211 |
| Mean NPV | -5838.2316748715 | -5838.2316748715 |

- Largest absolute difference across all 42 columns: **2.9e-11**. Ten columns
  are not bitwise identical, and the cause is known and benign: the climate
  module ends with `df_main.round(2)` (line 314), which re-rounds every column
  in the merged frame including the pass-through ones, shifting the last bits
  of a float. `a.round(2).equals(b.round(2))` is **True** -- at the 2-decimal
  precision the model actually reports and exports, the columns are identical.
  No column differs by as much as half a cent.
- The climate run writes **zero** new non-climate columns.
- Confirmed the climate module genuinely recomputed rather than passing
  everything through: `ref2025_mp4_heating_avoided_damages_climate_lrmer_central`
  moved from 15,009.87 (stale export) to 15,131.46 (recomputed) in the same run.

---

## 8. Notebook change the researcher must apply

`cmu_tare_model/model_scenarios/tare_scenarios_v2_3.ipynb`, the cell that calls
`calculate_private_npv` (cell 26), passes the old anchor year explicitly:

```python
            cost_scenario=cost_scenario_key,
            base_year=2024,
            verbose=VERBOSE,
```

`base_year` is no longer a parameter at all (Section 6b), so this line now
raises `TypeError: calculate_private_npv() got an unexpected keyword argument
'base_year'`. **The scenarios notebook will not run until the line is
deleted.** That is deliberate -- a loud failure is better than silently
reverting the whole run to the old 2024-2038 window, which is what would have
happened if the parameter had been left in place with a corrected default.

Delete the single line `base_year=2024,` so the call reads:

```python
            cost_scenario=cost_scenario_key,
            verbose=VERBOSE,
```

This is the only notebook edit required. No `tare_model_main_v2_3_EXPORT_*.py`
file carries the argument (all five checked), and no other `.py` call site
passes it.

**Confirmed applied.** The researcher made this edit by hand. Cell 26 of
`tare_scenarios_v2_3.ipynb` no longer contains `base_year` anywhere, and its
full keyword set (`df`, `df_fuel_costs`, `df_baseline_costs`, `menu_mp`,
`input_mp`, `policy_scenario`, `discount_rate_col_name`, `cost_scenario`,
`verbose`) was re-run against the live function signature on the National
export frames: it runs cleanly and returns the expected 175-column result.

---

## 8a. `tare_model_main_v2_3.ipynb` diff -- outputs only

The researcher reviewed the notebook diff directly: all ~4,770 changed lines
are cleared execution outputs and `execution_count` resets. No source cell
changed. Recorded here so the diff does not need re-checking in a later
session.
