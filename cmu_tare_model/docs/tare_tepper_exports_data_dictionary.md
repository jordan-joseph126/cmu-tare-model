# TARE Model -- Tepper CSV Exports, Data Dictionary

Written for a reader working in Excel who has never seen this codebase. It
describes the household and county CSVs produced by
`cmu_tare_model/utils/export_tepper_csv.py`, what every column means, and how
to rebuild the final answer from the columns in the file.

**This file supersedes** `cmu_tare_model/utils/tepper_export_data_dictionary.md`
and `cmu_tare_model/docs/tare_tepper_exports_data_dictionary.pdf`, both of which
described a Pennsylvania-only run with 100 columns and nine NPV cases. Neither
matches what the export produces now.

---

## The one thing to know first

Every number in the household file traces back to the same chain, and you can
walk it yourself:

```
projected consumption (kWh)  x  fuel price ($/kWh)   =  annual fuel cost ($)
baseline annual cost  -  retrofit annual cost         =  annual saving ($)
annual saving  x  discount factor, summed over 2025-2039
                                                      =  discounted lifetime saving ($)

discounted heating saving + discounted cooling saving - net capital cost = NPV
NPV >= 0                                              =  the home adopts
```

The last line is the model's entire adoption rule. There is no carbon price, no
health damage, and no comfort value in it -- only dollars. Section 10 works the
whole chain through for one real home.

---

## 1. Scope and provenance

### Representative dwelling units versus actual homes

**Every row in these files is a representative dwelling unit, not a house.**
ResStock is a sample. Each sampled row stands for many real dwellings, and the
`weight` column says how many: **242.131013**, the same for every row in this
release.

To get a count of actual homes, multiply by the weight, or sum the `weight`
column. To get an average or a share, you can ignore the weight entirely,
because it is the same for every row.

A worked instance of this, used later in section 3.2: 14 representative
dwelling units already have a heat pump. That is not 14 houses -- it is about
**3,390 actual homes**.

**Rule of thumb:** a count below about 242 is a count of representative
dwelling units, never of homes. One representative dwelling unit is the
smallest a non-zero count can be.

Throughout this document, counts are labeled either "representative dwelling
units" (abbreviated **rdu**) or "actual homes". Where both are useful, both are
given.

| | |
|---|---|
| Source data | ResStock 2022.1.1 (EUSS) |
| Representative dwelling units in the national run | 331,531 rdu |
| Actual homes those represent | 80,273,937 |
| Weight per representative dwelling unit | 242.131013, uniform |
| Counties | 3,098 |
| Measure packages | MP3 (standard heat pump, 15 SEER1 / 9 HSPF1, respecified to 16 SEER1 / 9.5 HSPF1 for ENERGY STAR) and MP4 (high-efficiency, 24-29.3 SEER1 / 13-14 HSPF1) |
| Policy scenario | `2025 Reference Case` (a single scenario -- there is no pre-IRA comparison in this export) |
| Private discount rate | 7% (`fixed_base`) |
| Cost year | All dollars are **USD2025**, in **real** terms |
| Cost stream | 15 years, **2025 through 2039** inclusive |
| Cost scenario | REMDB v4 mid (`v4MID`) |

Every value is copied straight from a model-run DataFrame. The export
recomputes nothing, rounds nothing, fills nothing, and drops no rows except
where section 3 says so.

**Which run these numbers come from.** The structural facts in this document
(the 154 columns, the groups, the naming) are read from the code. The
population figures (331,531 rdu and the applicability shares in section 5),
the negative-cooling-savings shares in group 11 of section 4, and the worked
example in section 10 are all read from the full end-to-end pipeline run
`2026-08-19_13-19` (National and Allegheny scopes, MP3 and MP4).

---

## 2. Files produced

Written to `{output_folder_path}/tepper_export/`.

| File | Grain | Rows |
|---|---|---|
| `tepper_household_mp{mp}_{scope}_{date}.csv` | one row per representative dwelling unit | see section 3 |
| `tepper_county_mp{mp}_{scope}_{date}.csv` | one row per county | 3,098 national, 1 for Allegheny |
| `source_data/` (three CSVs) | the model's fuel-price inputs | see section 9 |

One household file and one county file per measure package **per scope**. A
scope is a filter applied at export time, not at model run time, so a single
national run can emit a national file plus any number of state or county files.
The current configuration emits two scopes: the full run, and Allegheny County,
Pennsylvania.

---

## 3. Row counts, and why the two household files differ

| File | Rows (rdu) | Actual homes | Units the model did not evaluate |
|---|---|---|---|
| National household | 331,531 | 80,273,937 | kept, with blank result columns |
| Allegheny household | 1,356 | 328,330 | **removed** (254 rdu of 1,610 dropped) |

The national file is the complete record and keeps every representative
dwelling unit, including the ones the model could not evaluate. Their result
columns are blank.

The Allegheny file has those rows removed. The reason is an Excel hazard:
**Excel treats a blank cell as zero in arithmetic.** Averaging an NPV column
over rows the model never evaluated would quietly pull the average toward zero,
with no warning and no error. Because the Allegheny file is the one meant to be
opened and worked in directly, it is pre-filtered so that every row in it is a
home the model actually priced.

If you filter the national file yourself, apply the same rule: keep only rows
where `include_heating` is `True`.

### 3.1 How the sample narrows, step by step

The model starts from the full ResStock sample and applies four filters in
order. The same cascade runs for any geography; Allegheny County and the
national run are shown side by side.

| Step | Allegheny rdu | Allegheny homes | National rdu | National homes | % of stock |
|---|---|---|---|---|---|
| Sampled units, no filters | 2,434 | 589,347 | 548,916 | 132,909,587 | 100.00% |
| 1. Occupied dwellings only | 2,197 | 531,962 | 482,597 | 116,851,700 | 87.92% |
| 2. Single-family only | 1,610 | 389,831 | 331,531 | 80,273,937 | 60.40% |
| 3. Heating fuel the study prices | 1,604 | 388,378 | 321,357 | 77,810,496 | 58.54% |
| 4. Heating technology in the cost database | **1,356** | **328,330** | **260,211** | **63,005,153** | **47.40%** |

Steps 3 and 4 together are what the model records as `include_heating`, so the
last row is what the model actually evaluates: **63.0 million real dwellings
nationally, 328,330 in Allegheny County.**

**The sample is occupied, single-family homes heated by electricity, natural
gas, fuel oil, or propane, using a heating technology the cost database
covers.** That last condition is what makes the cost estimates real figures
rather than extrapolations, and it is the reason wall and floor furnaces,
shared building heating systems, and a handful of "Other Fuel" homes are
dropped.

Cooling is never a filter. A dwelling with no air conditioning still gets a
heating result. Of the 1,610 single-family Allegheny rdu:

| Cooling system | rdu | Actual homes | In scope |
|---|---|---|---|
| Central AC | 824 | 199,516 | yes |
| Room AC | 504 | 122,034 | yes |
| No cooling recorded | 268 | 64,891 | no |
| Heat Pump | 14 | 3,390 | no |
| **Cooling in scope** | **1,328** | **321,550** | |

These counts are reproducible: `cmu_tare_model/utils/tare_sample_size.py`
recomputes the whole cascade from the raw ResStock files for any county, state,
or the nation.

### 3.2 Exactly which dwelling units were dropped, and why

A row is dropped when `include_heating` is `False`. That flag is the AND of two
checks: the baseline heating **fuel** must be one the study models, and the
baseline heating **technology** must be one it models. Of the 254 Allegheny rdu
dropped, 248 fail only the technology check -- their fuel is fine.

| Heating system | rdu | Actual homes | Why it is out of scope |
|---|---|---|---|
| Natural Gas Wall/Floor Furnace | 222 | 53,753 | wall and floor furnaces are not among the modeled heating technologies |
| Electricity ASHP | 14 | 3,390 | already a heat pump, so there is no fossil system for this retrofit to replace |
| Natural Gas Shared Heating | 10 | 2,421 | heating is shared across a building, so a per-dwelling retrofit cost cannot be assigned |
| Propane or Fuel Oil Wall/Floor Furnace | 2 | 484 | same wall and floor furnace exclusion |
| No heating data recorded | 6 | 1,453 | nothing to compare a retrofit against |
| **Total dropped** | **254** | **61,501** | |

The modeled heating technologies are furnaces and boilers only: electric
baseboard, electric boiler, electric furnace, and fuel boilers and fuel furnaces
running on natural gas, propane, or fuel oil.

**Those 254 rdu represent 61,501 real dwellings, about 16% of Allegheny
County's single-family stock.** That is the figure to quote, not 254.

Two things are worth saying plainly, because the summary count is easy to
misread:

- **These are mostly not dwellings without heating.** Only 6 rdu (1,453 homes)
  have no heating system. 224 rdu (54,237 homes) have a wall or floor furnace,
  which is a real heating system that this version of the model does not cost
  out.
- **Most of them do have air conditioning.** 181 of the 254 rdu have
  `include_cooling = True`. Being dropped from this file says nothing about
  whether the dwelling has cooling.

One exclusion is study design rather than a gap in the cost database: a
dwelling that already has an air-source heat pump is removed because there is
no fossil heating system for this retrofit to replace. In Allegheny County that
is 14 rdu, about 3,390 homes. **Nationally it is the single largest excluded
group** -- 31,347 rdu, roughly 7.6 million homes, about half of everything
removed at step 4. Allegheny is unusual in being dominated by wall and floor
furnaces.

National step 4 removals, for contrast with the Allegheny table above:

| Removed nationally | rdu | Actual homes | Share of step 4 |
|---|---|---|---|
| Electricity ASHP | 31,347 | 7,590,081 | 51% |
| Wall/Floor Furnace, all fuels | 28,741 | 6,959,088 | 47% |
| Shared Heating, all fuels | 1,058 | 256,175 | 2% |
| **Total** | **61,146** | **14,805,343** | |

### 3.3 How this compares with ResStock 2025 dual-fuel eligibility

The filters are similar in spirit to the ones ResStock 2025 applies for its
Dual Fuel Heating System package, which reaches 44.65% of stock against TARE's
47.40%. The two arrive at a similar share by different routes:

| Requirement | TARE | ResStock 2025 dual fuel |
|---|---|---|
| Occupied dwellings only | yes | not specified |
| Single-family only | yes | no, all dwelling types |
| Heating fuel: electricity, natural gas, fuel oil, propane | yes | yes |
| Excludes shared building heating | yes | yes |
| Excludes wall and floor furnaces | yes | no |
| Excludes homes that already have a heat pump | yes | no |
| Requires ducts | **no** | **yes** |
| Requires a natural gas hookup | **no** | **yes** |

Two points a reader should not misread. ResStock 2025's natural-gas condition
is a **hookup** requirement, not a restriction on the existing heating fuel --
its fuel list is the same four fuels TARE uses. The hookup is what makes gas
available as the dual-fuel backup. And the duct requirement is substantial:
85% of the homes TARE evaluates are ducted, so adding that condition would take
TARE from 47.40% to 40.15% of stock, just below ResStock's figure.

The two studies are built on different ResStock vintages (2022.1.1 here, 2025
there), so this is a comparison of scope, not a like-for-like overlap.

### 3.4 This rule is under review

Whether these homes should be dropped is an open question as of 18 August 2026.
The case for keeping them: a home with a wall furnace, or with no system at all,
can still install a heat pump, and in the real world households do install
central HVAC for the first time when it becomes affordable. The case for
dropping them: the model produced no NPV, no savings, and no adoption flag for
them, so every result column is blank, and blanks in Excel behave as zeros.

If the modeled technology list is widened in a later release, these homes gain
real numbers and the drop becomes unnecessary. Until then, treat the 1,356-row
count as a property of **this** export rather than a fixed feature of Allegheny
County, and use the national file if you need the complete 1,610.

---

## 4. The household CSV: 154 columns

`bldg_id` is the row index and is written as the first column, so the file has
155 columns on disk.

The columns are ordered left to right as the derivation runs: who the home is,
what it consumes, what that costs, what the equipment costs, and finally the
NPV and the adoption flag.

| # | Group | Columns |
|---|---|---|
| 1 | Identifiers | 6 |
| 2 | Geography | 11 |
| 3 | Building | 6 |
| 4 | Household income | 7 |
| 5 | Existing HVAC | 15 |
| 6 | Retrofit HVAC | 2 |
| 7 | Applicability flags | 2 |
| 8 | Peak demand | 12 |
| 9 | Base-year consumption | 12 |
| 10 | Annual projected consumption | 60 |
| 11 | Lifetime fuel costs | 7 |
| 12 | Installed costs and applied credit | 4 |
| 13 | Rebate inputs | 3 |
| 14 | Model parameters | 2 |
| 15 | Discounted lifetime savings | 2 |
| 16 | Net capital cost | 1 |
| 17 | NPV | 1 |
| 18 | Economic adopter flag | 1 |
| | **Total** | **154** |

Below, `{mp}` is `3` or `4`. Everything with a `ref2025_mp{mp}_` prefix is a
model result for that measure package under the 2025 Reference Case.

### Group 1 -- Identifiers (6)

`weight`, `state`, `county`, `county_fips`, `puma`, `county_and_puma`

`county` is a Census GISJOIN string such as `G4200030`. **Do not convert it to
a number** -- the leading `G` and the trailing zeros are meaningful.
`county_fips` is the numeric equivalent (Allegheny is `42003`).

`weight` is how many real U.S. dwellings this row represents:
**242.131013**, identical for every row in this release. Multiply by it, or sum
it, to convert a count of rows into a count of actual homes. Because it is the
same everywhere, weighting changes totals but never changes an average or a
share -- a weighted mean and an unweighted mean are the same number here.

### Group 2 -- Geography (11)

`census_region`, `census_division`, `census_division_recs`,
`building_america_climate_zone`, `reeds_balancing_area`, `city`, `urbanicity`,
`weather_file_city`, `Longitude`, `Latitude`, `gea_region`

`census_division` is the join key for fuel oil and propane prices and for both
degree-day tables. See section 8.

### Group 3 -- Building (6)

`square_footage`, `building_type`, `occupancy`, `tenure`, `vacancy_status`,
`vintage`

### Group 4 -- Household income (7)

`income`, `federal_poverty_level`, `household_income`,
`census_area_medianIncome`, `income_level`, `percent_AMI`, `lmi_or_mui`

`percent_AMI` is the home's income as a percentage of the area median income
and is what routes a home between the two rebate programs in section 7.
`lmi_or_mui` labels each home Low-to-Moderate Income (LMI) or
Middle-to-Upper Income (MUI); the stored values are the two-letter codes
`LMI` and `MUI`.

*Known limitation:* Connecticut homes fall back to a state-level median income
rather than a county one. ResStock 2022.1.1 still uses Connecticut's eight
pre-2023 counties, while the income source uses the nine planning regions that
replaced them, so the county-level join finds nothing. This shifts
`percent_AMI` and therefore rebate routing for Connecticut homes only.

### Group 5 -- Existing HVAC (15)

`base_heating_fuel`, `heating_type`, `base_heating_efficiency`,
`base_cooling_fuel`, `cooling_type`, `base_cooling_efficiency`,
`fuel_type_heating`, `fuel_type_cooling`, `hvac_has_ducts`,
`hvac_heating_type_and_fuel`, `hvac_heating_efficiency`,
`size_heating_system_primary_k_btu_h`, `hvac_cooling_type`,
`hvac_cooling_efficiency`, `size_cooling_system_primary_k_btu_h`

`base_heating_fuel` is one of Electricity, Natural Gas, Propane, Fuel Oil. It
decides which fuel price the baseline heating consumption is costed at.

**Two columns in this group hold a retrofit quantity, despite sitting in the
"Existing HVAC" group.** `size_heating_system_primary_k_btu_h` and
`size_cooling_system_primary_k_btu_h` are the **retrofit heat pump's**
ResStock-autosized capacity for this measure package, not a value read from
the home's existing furnace, boiler, or air conditioner. The two columns are
equal for every home in this export, because one heat pump serves both loads.
ResStock's own baseline run computes its own capacity for the existing
equipment, but that value is not carried into this export under any column
name -- see `docs/SESSION_CHANGELOG_2026-08-19.md` for what depends on that
and why it matters for the installed-cost columns in group 12. These two
columns are grouped here because they describe HVAC equipment on the same row
as the other existing-system columns, not because they hold a baseline value.

### Group 6 -- Retrofit HVAC (2)

`upgrade_hvac_heating_efficiency`, `upgrade_hvac_cooling_efficiency`

### Group 7 -- Applicability flags (2)

`include_heating`, `include_cooling`

These two flags explain every blank cell in the file. See section 5.

### Group 8 -- Peak demand (12)

`base_peak_electricity_cooling_kw`, `base_peak_electricity_heating_kw`,
`base_peak_load_cooling_kbtu_hr`, `base_peak_load_heating_kbtu_hr`,
`mp{mp}_peak_electricity_cooling_kw`, `mp{mp}_peak_electricity_heating_kw`,
`mp{mp}_peak_electricity_cooling_kw_savings`,
`mp{mp}_peak_electricity_heating_kw_savings`,
`mp{mp}_peak_load_cooling_kbtu_hr`, `mp{mp}_peak_load_heating_kbtu_hr`,
`mp{mp}_peak_load_cooling_kbtu_hr_savings`,
`mp{mp}_peak_load_heating_kbtu_hr_savings`

Each home's own annual maximum, passed through from ResStock. **These peaks are
not aligned in time across homes**, so summing them across a county does not
give that county's peak -- it gives the sum of individual maxima, which is
always higher than the real coincident peak. Use them per home, or for a rough
upper bound only.

Two different quantities are kept, and they must not be combined: the `_kw`
pair is electric demand, the `_kbtu_hr` pair is thermal load.

The `_savings` columns are ResStock's baseline minus upgrade. **A negative
value means the heat pump raises the peak**, which is common on the heating
side because the baseline furnace burned fuel while the heat pump draws
electricity.

### Group 9 -- Base-year consumption (12)

All in kWh of site energy, for the year 2025.

`base_electricity_heating_consumption`, `base_electricity_cooling_consumption`,
`base_fuelOil_heating_consumption`, `base_naturalGas_heating_consumption`,
`base_propane_heating_consumption`, `baseline_heating_consumption`,
`baseline_cooling_consumption`, `mp{mp}_heating_consumption`,
`mp{mp}_cooling_consumption`, `base_total_electricity_consumption`,
`mp{mp}_total_electricity_consumption`, `baseline_total_site_consumption`

`baseline_heating_consumption` is the **sum across all four baseline heating
fuels** for that home, expressed in kWh. A home heats with one fuel, so in
practice one of the four `base_*_heating_consumption` columns is non-zero and
the sum equals it.

`base_total_electricity_consumption` and
`mp{mp}_total_electricity_consumption` are whole-home **electricity**, not all
fuels. `baseline_total_site_consumption` is whole-home **all-fuel** site energy
and is the denominator of `mp{mp}_modeled_savings_frac` in group 13.

### Group 10 -- Annual projected consumption (60)

kWh per year, for each of the 15 years 2025 through 2039, for four streams:

| Stream | Column pattern | Example |
|---|---|---|
| Baseline heating | `baseline_{year}_heating_consumption` | `baseline_2025_heating_consumption` |
| Retrofit heating | `ref2025_mp{mp}_{year}_heating_consumption` | `ref2025_mp3_2039_heating_consumption` |
| Baseline cooling | `baseline_{year}_cooling_consumption` | `baseline_2031_cooling_consumption` |
| Retrofit cooling | `ref2025_mp{mp}_{year}_cooling_consumption` | `ref2025_mp4_2028_cooling_consumption` |

4 streams x 15 years = 60 columns. They appear grouped by stream, so each
stream is one contiguous block of 15 columns.

Each year is the 2025 value scaled by that year's degree-day factor for the
home's census division: heating uses the `hdd` rows, cooling the `cdd` rows, of
`aeo2026_degree_day_factors_2025_2050.csv`. The 2025 factor is exactly 1.0 for
every division, which is what makes 2025 the anchor year. Heating factors fall
over time and cooling factors rise, reflecting the projected climate.

These columns are what let you apply your own fuel prices. Multiply a year's
consumption by a $/kWh price and you have that year's cost.

### Group 11 -- Lifetime fuel costs (7)

USD2025, summed over 2025-2039, **not discounted**.

`baseline_heating_lifetime_fuel_cost`,
`ref2025_mp{mp}_heating_lifetime_fuel_cost`,
`ref2025_mp{mp}_heating_lifetime_savings_fuel_cost`,
`baseline_cooling_lifetime_fuel_cost`,
`ref2025_mp{mp}_cooling_lifetime_fuel_cost`,
`ref2025_mp{mp}_cooling_lifetime_savings_fuel_cost`,
`ref2025_mp{mp}_cooling_lifetime_savings_negative`

A `savings` column is baseline minus retrofit, so positive means the retrofit
is cheaper. These are undiscounted sums; the discounted figures that feed the
NPV are in group 15.

`ref2025_mp{mp}_cooling_lifetime_savings_negative` is `True` where cooling
savings came out negative -- the heat pump uses **more** cooling energy than
the existing air conditioner. **This is a real result, not an error.** It is
overwhelmingly a change in service: a room unit cools one room while the heat
pump cools the whole house. The share of affected homes is measure-package
specific and, in Allegheny County, well above the national share:

| Scope | MP | Room AC | Central AC |
|---|---|---|---|
| National | MP3 | 90.68% | 10.84% |
| National | MP4 | 61.97% | 3.46% |
| Allegheny | MP3 | 95.3% | 16.5% |
| Allegheny | MP4 | 74.5% | 4.5% |

The model counts the extra cost and gives no credit for the extra comfort,
because the adoption
rule is dollars only.

### Group 12 -- Installed costs and applied credit (4)

USD2025, one-time, undiscounted, **before any rebate**.

| Column | Meaning |
|---|---|
| `mp{mp}_heating_upgrade_installed_cost_v4MID` | installed cost of the heat pump |
| `mp{mp}_heating_replacement_installed_cost_v4MID` | what replacing the existing heating system like for like would have cost |
| `mp{mp}_cooling_replacement_installed_cost_v4MID` | what replacing the existing air conditioner would have cost |
| `mp{mp}_cooling_replacement_credit_applied_v4MID` | the cooling credit the NPV actually subtracted |

The heat pump provides heating **and** cooling, so its single installed cost is
recorded once, on the heating side. There is deliberately no separate cooling
upgrade cost; splitting one piece of equipment in two would double-count it.

The two `replacement` columns are counterfactuals -- money the household does
not spend because it bought a heat pump instead. They are credits, not costs.

The last column exists because the applied credit is not always the same as the
raw cooling replacement cost. It is `0.00` for a dwelling with no air
conditioner, and `0.00` for one that has an air conditioner but no recorded
replacement cost (269 rdu nationally, about 65,100 homes). Use the **applied**
column when checking the arithmetic.

### Group 13 -- Rebate inputs (3)

`mp{mp}_heating_rebate_amount_june2026_v4MID`,
`mp{mp}_rebate_eligibility_june2026`, `mp{mp}_modeled_savings_frac`

**These columns are information only. They are not subtracted anywhere in this
file.** The NPV shipped here is unsubsidized. They are provided so you can model
a rebate yourself. See section 7 for the rules and the caveats.

### Group 14 -- Model parameters (2)

`public_discount_rate`, `private_discount_rate_fixed_base`

Both are fractions, so 0.07 means 7%. Only the private rate is used by anything
in this file.

### Group 15 -- Discounted lifetime savings (2)

`ref2025_mp{mp}_heating_discounted_lifetime_savings_fixed_base`,
`ref2025_mp{mp}_cooling_discounted_lifetime_savings_fixed_base`

USD2025. Each is the sum over 2025-2039 of that year's saving divided by
`(1 + 0.07) ^ (year - 2025)`. Year 2025 is not discounted.

The cooling column is `0.00` for homes with no air conditioner, not blank --
it is the value the NPV actually used.

### Group 16-18 -- The result (3)

| Column | Meaning |
|---|---|
| `ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | heat pump cost minus both avoided replacements |
| `ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_private_npv_fixed_base` | the NPV, in USD2025 |
| `ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base` | 1.0 adopts, 0.0 does not, blank not applicable |

Reading the name: `heatingLCC_coolingLCC` means both avoided replacements are
credited; `unsub` means no rebate is applied.

The model computes nine NPV variants in total (three credit scopes, each with
no rebate, a 2024-guidance rebate, and a June 2026-guidance rebate). **This
export ships one of the nine** -- the unsubsidized, both-credits case -- because
the intended use is to model the unsubsidized economics and layer your own
rebate assumptions on top. The other eight still exist in the full model output.

---

## 4.1 What is deliberately not in the file, and why

The model produces far more than 154 columns per home. This export selects a
subset. Nothing below was lost or forgotten -- each was left out for a stated
reason, and all of it still exists in the full model output.

| Left out | How many | Why |
|---|---|---|
| Eight of the nine NPV cases, with their net capital cost and adopter flags | 24 | The model prices three credit scopes, each with no rebate, a December 2024-guidance rebate, and a June 2026-guidance rebate. This export ships the **unsubsidized** case that credits both avoided replacements, because the intended use is to model unsubsidized economics and apply your own rebate assumptions on top. Shipping all nine invites averaging across cases that are alternatives, not additive. |
| The December 2024-guidance rebate amount | 1 | This export is unsubsidized. The June 2026 amount is kept as reference (section 7); carrying two competing rebate vintages beside an unsubsidized NPV is an invitation to subtract the wrong one. |
| `ref2025_mp{mp}_heating_total_capital_cost_v4MID` | 1 | Despite the name, this column has the December 2024 rebate already netted out of it -- for some homes by as much as $8,000. In an unsubsidized file that is a trap. The gross installed cost is shipped instead, as `mp{mp}_heating_upgrade_installed_cost_v4MID`. |
| The other private discount rates | 3 | The model can run at 3%, 7%, 10%, and a variable rate. Only the 7% (`fixed_base`) run exists for this release, so the other columns would be blank or misleading. |
| Emissions and climate damages | 12 shipped previously, 48 in the model | Removed at the researcher's direction. They play no part in the adoption decision, which is based on the private NPV alone, so carrying them beside the NPV suggests a link that the model does not make. |
| Bookkeeping columns | 18 | REMDB cost-table row lookups (`*_pm1_euss`, `*_pm2_euss`, `*_pm2_euss_original`), internal `row_id_*` fields, and the intermediate validation flags. They describe how the model found a number, not the number itself. |

Two of the validation flags are an exception and **are** shipped:
`include_heating` and `include_cooling`. They are the explanation for every
blank cell in the file, so they travel with it (see sections 3.1 and 5).

---

## 5. Blank cells mean "not applicable", never zero

The export never fills or coerces a blank. A blank means the model did not
evaluate that quantity for that home.

| Flag | True when | Share of the national run |
|---|---|---|
| `include_heating` | baseline heating fuel and technology are both in scope | 260,211 of 331,531 rdu = **78.49%** |
| `include_cooling` | the dwelling has central or room air conditioning | 250,576 of 331,531 rdu = **75.58%** |

Because the weight is the same for every row, those shares are identical
whether you count rows or actual homes.

Every heating-side column is blank where `include_heating` is `False`. Every
cooling-side column is blank where `include_cooling` is `False`. A home is out
of scope for heating if it has no heating system, or heats with a fuel or
technology the study does not model -- including homes that **already have a
heat pump**, which are excluded because there is no fossil system for the
retrofit to replace.

**These two shares are the correct denominators for any per-home average.**
Using 331,531 will understate every average.

The Allegheny file's pre-filter drops on `include_heating` alone (section 3),
so its row count -- 1,356 -- is the correct heating-side denominator, but
**not** the cooling-side one. 209 of the 1,356 exported rdu have
`include_cooling = False` (no central or room air conditioning recorded), so
every cooling-side column is blank for them. The correct cooling denominator
for the Allegheny file is **1,147**, not 1,356.

---

## 6. Rounding, and the one place it shows

The model rounds each annual fuel cost to 2 decimals **before** summing and
discounting. If you rebuild the lifetime totals, round each year to cents first,
or you will drift a few cents from the shipped figure.

**One known small mismatch.** The consumption columns in group 10 are stored
rounded to 2 decimals, but the model computed each annual cost from the
unrounded consumption. So when you do `consumption x price` yourself and round
to cents, you will match the model's annual cost for about 95% of home-year
cells and land **one cent** away for the rest. The error is one cent, never
more, and it does not accumulate into anything material over 15 years. The
shipped lifetime totals, discounted savings, and NPV are all computed from the
unrounded values and are exact.

---

## 7. The rebate columns, and what they do not tell you

`mp{mp}_rebate_eligibility_june2026` is `'HEEHR'`, `'HOMES'`, or
`'Not Eligible'`. Under the June 2026 Department of Energy guidance:

- **HEEHR** applies at or below 150% of area median income. It caps the heat
  pump rebate at $8,000 and covers 100% of the cost at or below 80% AMI, 50%
  between 80% and 150%. Under this guidance HEEHR is restricted to homes whose
  existing heating is **electric resistance** -- any fossil-fuel baseline gets
  nothing from HEEHR.
- **HOMES** applies above 150% of area median income and is based on
  `mp{mp}_modeled_savings_frac`, the projected whole-home energy saving:
  20% or more caps at $2,000, 35% or more caps at $4,000, covering half the
  project cost.

**Treat these as provisional.** Four things are not modeled:

1. **No state funding cap.** The amounts are uncapped potential, not money that
   will actually be paid. Real programs run out of funds.
2. **South Dakota is zeroed** in every scenario, because it never took part in
   the federal rebate programs.
3. **Weatherization prerequisites are not enforced** and dual-fuel systems are
   not modeled.
4. **One program per home** -- never both.

`mp{mp}_modeled_savings_frac` divides the model's heating and cooling energy
change by `baseline_total_site_consumption` (group 9), so you can check it.
The numerator is degree-day-adjusted while the denominator is the raw ResStock
total; this mix is an accepted approximation.

---

## 8. Fuel prices: how the model looks them up

Prices are **real USD2025 per kWh**, not nominal. There is no inflation in this
model. A 2039 cost is in the same dollars as a 2025 cost, so you can compare
them directly and you must **not** deflate them again.

A year's price is the 2025 anchor price multiplied by that year's projection
factor:

```
price(year) = anchor price  x  factor(census division, fuel, year)
```

### The join keys

| Fuel | Anchor price key | Source |
|---|---|---|
| Electricity | **state** (two-letter code) | EIA state annual 2025 |
| Natural gas | **state** (two-letter code) | EIA state annual 2025 |
| Fuel oil | **census division** | see below |
| Propane | **census division** | see below |

The projection factor always keys on **census division and fuel**, for all four
fuels.

Everything the retrofit consumes is electricity, so a retrofit cost is always
the home's state electricity price. A baseline cost uses the price for the
home's `base_heating_fuel`.

### The fuel oil and propane rule, exactly

EIA does not publish state-level prices for fuel oil and propane. It publishes
them by PADD (Petroleum Administration for Defense District). The model builds
the census-division price in two steps:

1. **Each state inherits its PADD's price.** Where EIA publishes no PADD price
   covering that state, the state falls back to the U.S. national price. This
   affects 18 states for fuel oil and 7 for propane.
2. **Each census division takes an unweighted arithmetic mean** of the state
   prices in it -- a plain average of the states, with no weighting by
   population, households, or fuel use.

The consequence worth knowing: a census division that spans more than one PADD
ends up with a flat average across them. The South Atlantic division, for
example, contains both PADD 1B and PADD 1C states, so no state in it is priced
at its own PADD price. This is a deliberate simplification, not a bug.

### Never difference the two consumption columns

For most homes the baseline and the retrofit run on **different fuels** -- a gas
furnace replaced by an electric heat pump. Subtracting
`ref2025_mp{mp}_2030_heating_consumption` from
`baseline_2030_heating_consumption` gives a kWh number that has no single price
and therefore no meaning.

**Always cost each side at its own fuel price first, then subtract the
dollars.** The worked example in section 10 shows a home where the baseline
kWh falls by 78% but the dollar saving is only $64 in the first year, because
the gas it stops buying is a quarter the price of the electricity it starts
buying.

---

## 9. The `source_data/` folder

Three CSVs, copied **unchanged** from the model's own inputs, so what you hold
is exactly what produced the numbers.

| File | What it is |
|---|---|
| `eia_fuel_price_data_2025_usd2025.csv` | 2025 anchor prices, already converted to USD2025 per kWh |
| `aeo2026_fuel_price_factors_2025_2050.csv` | price projection factors by census division and fuel |
| `aeo2026_degree_day_factors_2025_2050.csv` | heating (`hdd`) and cooling (`cdd`) factors by census division |

Two things to know about them:

- **The `National` rows are a fallback**, used by the model only when it meets a
  region it does not recognise. Nothing in the ResStock data triggers them,
  because every home carries a real two-letter state and a real census
  division. They are kept so the files match the model's actual inputs, but a
  lookup keyed on a real state or division will never hit them. If one of your
  lookups returns a `National` value, the lookup key is wrong.
- **The projection tables run to 2050, but this export stops at 2039.** The
  2040-2050 columns are not used by anything in the household file. Projecting
  past 2039 goes beyond the 15-year equipment lifetime the model assumes.

All three tables use `1.0` for every 2025 factor. That is what makes 2025 the
anchor year: 2025 energy use and 2025 prices are the unscaled ResStock and EIA
values.

---

## 10. Worked example: one dwelling unit, start to finish

`bldg_id 491`, MP3, from the Allegheny household file. This is one
representative dwelling unit, standing for about 242 real homes, so every
dollar figure below is per dwelling, not per row-of-242.

| | |
|---|---|
| State / county | PA / `G4200030` (Allegheny) |
| Census division | Middle Atlantic |
| Baseline heating | Natural Gas Fuel Furnace, 92.5% AFUE |
| Baseline cooling | Central AC, SEER 13 |
| Retrofit | ASHP, SEER 16, 9.5 HSPF |
| Floor area | 1,690 sq ft |
| `weight` | 242.131013 (real homes represented) |
| `include_heating` / `include_cooling` | True / True |
| `private_discount_rate_fixed_base` | 0.07 |

### Step 1 -- base-year consumption (kWh, 2025)

| Column | Value |
|---|---|
| `baseline_heating_consumption` | 12,795.78 |
| `baseline_cooling_consumption` | 913.80 |
| `mp3_heating_consumption` | 2,941.26 |
| `mp3_cooling_consumption` | 630.69 |

The heat pump uses 77% less heating energy. That is the efficiency gain, in
energy. It is not the dollar saving, because the fuel changes.

### Step 2 -- project each year with the degree-day factor

2025 factors are 1.0, so 2025 consumption equals the base year exactly. By 2039
the Middle Atlantic heating factor has fallen and the cooling factor has risen:

| Year | Baseline heating kWh | Retrofit heating kWh | Baseline cooling kWh | Retrofit cooling kWh |
|---|---|---|---|---|
| 2025 | 12,795.78 | 2,941.26 | 913.80 | 630.69 |
| 2032 | 12,175.41 | 2,798.66 | 1,082.01 | 746.78 |
| 2039 | 11,822.60 | 2,717.56 | 1,150.00 | 793.71 |

### Step 3 -- price each side at its own fuel

Baseline heating is natural gas; everything else is electricity.

| Year | Base heat $/kWh | Base heat cost | Retrofit heat $/kWh | Retrofit heat cost | Heating saving |
|---|---|---|---|---|---|
| 2025 | 0.049391 | $631.99 | 0.192999 | $567.66 | $64.33 |
| 2032 | 0.047931 | $583.58 | 0.198084 | $554.37 | $29.21 |
| 2039 | 0.048939 | $578.59 | 0.203834 | $553.93 | $24.66 |

This is the point of section 8. Electricity costs roughly four times as much
per kWh as gas here, so a 77% cut in energy becomes a saving of only about $64
in the first year.

Cooling is electricity on both sides, so the saving is larger relative to the
energy change: $54.64 in 2025, rising to $72.62 by 2039.

### Step 4 -- discount and sum

Discount factor is `1 / 1.07 ^ (year - 2025)`, so 1.000000 in 2025, 0.622750 in
2032, 0.387817 in 2039.

| | Sum over 2025-2039 |
|---|---|
| Discounted heating saving | **$291.33** |
| Discounted cooling saving | **$626.50** |

These match `ref2025_mp3_heating_discounted_lifetime_savings_fixed_base` and
`ref2025_mp3_cooling_discounted_lifetime_savings_fixed_base` exactly.

Note the cooling saving is more than double the heating saving, even though
heating is by far the larger end use. Fuel switching, not efficiency, is what
governs the heating side.

### Step 5 -- net capital cost

| | |
|---|---|
| `mp3_heating_upgrade_installed_cost_v4MID` | $12,084.50 |
| less `mp3_heating_replacement_installed_cost_v4MID` | $3,734.48 |
| less `mp3_cooling_replacement_credit_applied_v4MID` | $5,451.61 |
| = `ref2025_mp3_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | **$2,898.41** |

The heat pump costs $12,084.50, but this household was going to have to replace
a furnace and an air conditioner anyway. Crediting both, the extra cost of
choosing a heat pump is $2,898.41.

### Step 6 -- the NPV

```
  $291.33   discounted heating saving
+ $626.50   discounted cooling saving
- $2,898.41 net capital cost
= -$1,980.58
```

`ref2025_mp3_heatingLCC_coolingLCC_unsub_private_npv_fixed_base` = **-$1,980.58**,
matching to the cent, and
`ref2025_mp3_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base` = **0.0**,
because the NPV is below zero.

The story for this home: the heat pump is far more efficient and its extra
capital cost is modest, but cheap natural gas means the energy savings do not
repay the $2,898 premium within 15 years at a 7% discount rate.

---

## 11. The county CSV

Eleven columns, one row per county, assembled from three separate model tables
and joined on `county`.

| Column | Unit |
|---|---|
| `county` | Census GISJOIN string -- do not cast to a number |
| `state` | two-letter abbreviation |
| `home_count` | homes represented, that is the sum of `weight` |
| `adoption_rate_pct` | percent 0-100, the share of **applicable** homes with NPV >= 0 |
| `operating_cost_pct_change` | percent, the county median of each home's `(retrofit - baseline) / baseline x 100` |
| `baseline_elec_gwh`, `retrofit_elec_gwh`, `elec_change_gwh`, `site_energy_change_gwh` | GWh |
| `pct_elec_demand_change`, `pct_site_energy_change` | percent |

A county with too few sampled homes has blank metrics rather than zeros.

**`site_energy_change_gwh` and `pct_site_energy_change` are aliases**, not
independent all-fuel numbers. Both sides are read from whole-home electricity,
and because the retrofit fully electrifies heating and cooling the two measures
converge by construction. For an electricity reading use `elec_change_gwh` and
`pct_elec_demand_change`; do not treat the site-energy pair as a separate
result.

---

## 12. What will change in the next release

This export is built on ResStock 2022.1.1. The next version of the model moves
to ResStock 2025 and adds dual-fuel (hybrid) systems, where a heat pump and a
fossil backup share the heating load.

Expect these to change: **column names**, the retrofit consumption columns
(which will split by fuel rather than being electricity-only), the fuel-price
path (which will become fuel-aware on the retrofit side), and the county
geography. Analysis built on this file will need rework, not just a refresh.
Anything you build should keep the column names in one place rather than
scattered through formulas.

---

## 13. Validation performed

Checked on both MP3 and MP4:

- Round-trip comparison of the written CSV against its two source frames --
  every value, the `bldg_id` index, and every blank in the same place.
- Exported row count equals input row count for the full-scope file.
- The 154 column names all resolve for both measure packages, in the declared
  order, with no duplicates.
- Column-by-column check that no value in any pre-existing column moved when
  the new columns were added: 189 NPV-side columns compared with no tolerance,
  zero differences.
- The reconciliation in section 10 holds across every applicable dwelling unit:
  discounted heating saving + discounted cooling saving - net capital cost
  equals the shipped NPV, with none off by more than a cent.
- Scope filtering verified against the national run: 1,610 Allegheny rdu,
  254 removed as not applicable, 1,356 exported; the county table filters to
  exactly one row.
- The filter cascade in section 3.1 recomputed from the raw ResStock files by
  `cmu_tare_model/utils/tare_sample_size.py`, matching at every step.
- A scope the run does not contain is skipped with a message rather than
  writing an empty file; a mistyped county code raises an error.
