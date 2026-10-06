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
kWh of each fuel  x  that fuel's price ($/kWh), added up
                                                      =  annual fuel cost ($)
baseline annual cost  -  retrofit annual cost         =  annual saving ($)
annual saving  x  discount factor, summed over 2025-2039
                                                      =  discounted lifetime saving ($)

discounted heating saving + discounted cooling saving - net capital cost = NPV
NPV >= 0                                              =  the home adopts
```

The first line is done fuel by fuel: a gas furnace burns gas and its fan uses
electricity, and each is priced at its own price (section 8).

The last line is the model's entire adoption rule. There is no carbon price, no
health damage, and no comfort value in it -- only dollars. Section 10 works the
whole chain through for one real home.

---

## 1. Scope and provenance

### Representative dwelling units versus actual homes

**Every row in these files is a representative dwelling unit, not a house.**
ResStock is a sample. Each sampled row stands for many real dwellings, and the
`weight` column says how many: **242.131013**, the same for every row in this
release. (The file holds the full value, 242.13101272727272.)

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
| Representative dwelling units in the household files (the study sample) | 221,205 rdu |
| Actual homes those represent | 53,560,591 |
| Weight per representative dwelling unit | 242.131013, uniform |
| Counties | 3,079 |
| Measure packages | MP3 (standard heat pump, 15 SEER1 / 9 HSPF1, respecified to 16 SEER1 / 9.5 HSPF1 for ENERGY STAR) and MP4 (high-efficiency, 24-29.3 SEER1 / 13-14 HSPF1) |
| Policy scenario | `2025 Reference Case` (a single scenario -- there is no pre-IRA comparison in this export) |
| Private discount rate | 7% (`fixed_base`) |
| Cost year | All dollars are **USD2025**, in **real** terms |
| Cost stream | 15 years, **2025 through 2039** inclusive |
| Cost scenario | REMDB v4 mid (`v4MID`) |

Every value is copied straight from a model-run DataFrame. The export
recomputes nothing, rounds nothing, and fills nothing. It keeps only the rows
in the study sample (section 3).

**Which run these numbers come from.** The structural facts in this document
(the 167 columns, the groups, the naming) are read from the code. The counts,
the shares, and the worked example in section 10 are all read from the
pipeline run `2026-10-05_21-33` (a national run, MP3 and MP4). The Allegheny
figures are that run's Allegheny County rows.

---

## 2. Files produced

Written to `{output_folder_path}/tepper_export/`.

| File | Grain | Rows |
|---|---|---|
| `tepper_household_mp{mp}_{scope}_{date}.csv` | one row per representative dwelling unit in the study sample | see section 3 |
| `tepper_household_detailed_mp{mp}_{scope}_{date}.csv` | the same rows and columns, plus each year's consumption split by fuel (group 10b) | see section 3 |
| `tepper_county_mp{mp}_{scope}_{date}.csv` | one row per county | 3,079 national, 1 for Allegheny |
| `source_data/` (three CSVs) | the model's fuel-price inputs | see section 9 |

Two household files and one county file per measure package **per scope**. A
scope is a filter applied at export time, not at model run time, so a single
national run can emit a national file plus any number of state or county files.
The two scopes produced so far are the full run and Allegheny County,
Pennsylvania.

---

## 3. Row counts: every row is in the study sample

| File | Rows (rdu) | Actual homes |
|---|---|---|
| National household, main and detailed | 221,205 | 53,560,591 |
| Allegheny household, main and detailed | 1,146 | 277,482 |

Both household files hold only the **study sample**: the homes the model
evaluates. The model's own results table has more rows (331,526 rdu
nationally, 1,610 in Allegheny County), but every result is blank for a row
outside the sample.

Those rows are left out of these files because of an Excel hazard:
**Excel treats a blank cell as zero in arithmetic.** Averaging an NPV column
over rows the model never evaluated would quietly pull the average toward zero,
with no warning and no error. Every row in these files is a home the model
actually priced.

The filter is the `include_sample` column, which is `True` on every row here.
If you ever work from the model's full results table instead, apply the same
rule: keep only rows where `include_sample` is `True`. Do not use
`include_heating` or `include_cooling` for this; neither one alone is the
sample.

### 3.1 How the sample narrows, step by step

A home is in the study sample when all three of these hold:

- its heating system is one the study can replace and cost: a furnace, boiler
  or electric baseboard that runs on electricity, natural gas, propane or fuel
  oil, and is not already a heat pump (`include_heating`);
- it has a central or room air conditioner of its own, not a cooling system
  shared with other homes (`include_cooling`);
- ResStock applied every measure package in the run to it.

| Step | National rdu | National homes | Removed (rdu) | % of stock |
|---|---|---|---|---|
| ResStock 2022.1.1 stock | 548,916 | 132,909,587 | | 100.00% |
| Package applies | 548,260 | 132,750,749 | 656 | 99.88% |
| Occupied | 482,050 | 116,719,255 | 66,210 | 87.82% |
| Single-family | 331,526 | 80,272,726 | 150,524 | 60.40% |
| Not Alaska or Hawaii | 331,526 | 80,272,726 | 0 | 60.40% |
| Heating fuel is electricity, natural gas, propane or fuel oil | 321,352 | 77,809,285 | 10,174 | 58.54% |
| No existing heat pump | 290,005 | 70,219,204 | 31,347 | 52.83% |
| Furnace, boiler or electric baseboard | 260,211 | 63,005,153 | 29,794 | 47.40% |
| Central or room AC | 221,301 | 53,583,835 | 38,910 | 40.32% |
| Cooling system not shared | **221,205** | **53,560,591** | 96 | **40.30%** |

The last row is what the model evaluates: **53.6 million real dwellings
nationally.**

The same steps for Allegheny County, starting from its single-family rdu:

| Step | Allegheny rdu | Allegheny homes | Removed (rdu) |
|---|---|---|---|
| Single-family | 1,610 | 389,831 | |
| Heating fuel is electricity, natural gas, propane or fuel oil | 1,604 | 388,378 | 6 |
| No existing heat pump | 1,590 | 384,988 | 14 |
| Furnace, boiler or electric baseboard | 1,356 | 328,330 | 234 |
| Central or room AC | 1,147 | 277,724 | 209 |
| Cooling system not shared | **1,146** | **277,482** | 1 |

**Cooling is a filter.** A home with no air conditioning is not in the sample.
For that home the heat pump would add cooling it never had, which is a new
service, not a change to an existing one. A home whose cooling system is
shared with other homes is also left out, because there is no replacement cost
data for a shared system. Of the 1,146 Allegheny rdu in the sample, 782 have
central AC and 364 have room AC.

The national table is built by `build_sample_funnel`
(`cmu_tare_model/energy_consumption_and_metadata/study_sample.py`) and saved
with each run under `output_results/baseline_summary/sample_funnel/`.

### 3.2 Which dwelling units are left out, and why

In Allegheny County, 464 of the 1,610 single-family rdu are outside the
sample: 254 because of their heating system and 210 because of their cooling.

| Heating system | rdu | Actual homes | Why it is out of scope |
|---|---|---|---|
| Natural Gas Wall/Floor Furnace | 222 | 53,753 | wall and floor furnaces are not among the modeled heating technologies |
| Electricity ASHP | 14 | 3,390 | already a heat pump, so there is no fossil system for this retrofit to replace |
| Natural Gas Shared Heating | 10 | 2,421 | heating is shared across a building, so a per-dwelling retrofit cost cannot be assigned |
| Propane or Fuel Oil Wall/Floor Furnace | 2 | 484 | same wall and floor furnace exclusion |
| Heating fuel not priced, or none recorded | 6 | 1,453 | the study prices electricity, natural gas, propane and fuel oil only |
| **Total, heating** | **254** | **61,501** | |

| Cooling, among the 1,356 rdu whose heating is in scope | rdu | Actual homes | Why it is out of scope |
|---|---|---|---|
| No central or room air conditioner | 209 | 50,605 | the heat pump would add a service the home never had |
| Shared cooling system | 1 | 242 | no replacement cost data for a shared system |
| **Total, cooling** | **210** | **50,848** | |

The modeled heating technologies are furnaces and boilers only: electric
baseboard, electric boiler, electric furnace, and fuel boilers and fuel furnaces
running on natural gas, propane, or fuel oil.

**Those 464 rdu represent 112,349 real dwellings, about 29% of Allegheny
County's single-family stock.** That is the figure to quote, not 464.

Two things are worth saying plainly, because the summary count is easy to
misread:

- **The heating group is mostly not dwellings without heating.** Only 6 rdu
  (1,453 homes) have no priced heating fuel. 224 rdu (54,237 homes) have a
  wall or floor furnace, which is a real heating system that this version of
  the model does not cost out.
- **The cooling group has heating the study could replace.** Those 210 rdu are
  left out only because they have no air conditioner of their own.

Two exclusions are study design rather than gaps in the cost database. A
dwelling that already has an air-source heat pump is removed because there is
no fossil heating system for this retrofit to replace: 14 rdu (about 3,390
homes) in Allegheny County and 31,347 rdu (about 7.6 million homes)
nationally. A dwelling with no air conditioner of its own is removed for the
reason given in section 3.1: 209 rdu in Allegheny County and 38,910 rdu (about
9.4 million homes) nationally, the largest single group removed.

National removals after the single-family step, for contrast with the
Allegheny tables above:

| Removed nationally | rdu | Actual homes |
|---|---|---|
| Heating fuel not priced, or none recorded | 10,174 | 2,463,441 |
| Electricity ASHP (existing heat pump) | 31,347 | 7,590,081 |
| Wall/Floor Furnace, all fuels | 28,741 | 6,959,087 |
| Shared Heating, all fuels | 1,053 | 254,964 |
| No central or room air conditioner | 38,910 | 9,421,318 |
| Shared cooling system | 96 | 23,245 |
| **Total** | **110,321** | **26,712,135** |

### 3.3 How this compares with ResStock 2025 dual-fuel eligibility

The filters are similar in spirit to the ones ResStock 2025 applies for its
Dual Fuel Heating System package, which reaches 44.65% of stock against TARE's
40.30%. The two arrive at a similar share by different routes:

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

TARE also requires a central or room air conditioner of the home's own, which
this table does not compare.

Two points a reader should not misread. ResStock 2025's natural-gas condition
is a **hookup** requirement, not a restriction on the existing heating fuel --
its fuel list is the same four fuels TARE uses. The hookup is what makes gas
available as the dual-fuel backup. And the duct requirement is substantial:
88.76% of the homes TARE evaluates are ducted, so adding that condition would
take TARE from 40.30% to 35.77% of stock, further below ResStock's figure.

The two studies are built on different ResStock vintages (2022.1.1 here, 2025
there), so this is a comparison of scope, not a like-for-like overlap.

### 3.4 What is decided, and what is still open

**Decided: homes with no air conditioner of their own are left out.** The
reason is in section 3.1, and it is listed among the study's limitations.

**Still open: wall and floor furnaces.** Whether these homes should be
evaluated was an open question as of 18 August 2026 and has not been settled.
The case for keeping them: a home with a wall furnace can still install a heat
pump, and in the real world households do install central HVAC for the first
time when it becomes affordable. The case for leaving them out: the model
produces no NPV, no savings, and no adoption flag for them.

If the modeled technology list is widened in a later release, these homes gain
real numbers. Until then, treat the 1,146-row count as a property of **this**
release rather than a fixed feature of Allegheny County. The complete 1,610
rows are in the model's saved results table, not in these files.

---

## 4. The household CSV: 167 columns

`bldg_id` is the row index and is written as the first column, so the file has
168 columns on disk. The detailed copy has the same 167 columns plus 150
per-fuel columns (group 10b): 317 in all, 318 on disk.

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
| 7 | Sample flags | 3 |
| 8 | Peak demand | 12 |
| 9 | Base-year consumption: primary system and whole home | 12 |
| 9b | Base-year consumption: fans, pumps and heat-pump backup | 12 |
| 10 | Annual projected consumption | 60 |
| 11 | Lifetime fuel costs | 7 |
| 12 | Installed costs and applied credit | 4 |
| 13 | Rebate inputs | 3 |
| 14 | Model parameters | 2 |
| 15 | Discounted lifetime savings | 2 |
| 16 | Net capital cost | 1 |
| 17 | NPV | 1 |
| 18 | Economic adopter flag | 1 |
| | **Total, main file** | **167** |
| 10b | Annual consumption by fuel, detailed copy only | 150 |
| | **Total, detailed copy** | **317** |

Below, `{mp}` is `3` or `4`. Everything with a `ref2025_mp{mp}_` prefix is a
model result for that measure package under the 2025 Reference Case.

### Group 1 -- Identifiers (6)

`weight`, `state`, `county`, `county_fips`, `puma`, `county_and_puma`

`county` is a Census GISJOIN string such as `G4200030`. **Do not convert it to
a number** -- the leading `G` and the trailing zeros are meaningful.
`county_fips` is the numeric equivalent (Allegheny is `42003`).

`weight` is how many real U.S. dwellings this row represents: **242.131013**
(written in the file as 242.13101272727272), identical for every row in this
release. Multiply by it, or sum it, to convert a count of rows into a count of
actual homes. Because it is the same everywhere, weighting changes totals but
never changes an average or a share -- a weighted mean and an unweighted mean
are the same number here.

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

### Group 7 -- Sample flags (3)

`include_sample`, `include_heating`, `include_cooling`

`include_sample` is the study-sample flag (section 3.1). The other two are the
heating and cooling checks behind it. **All three are `True` on every row of
these files**, because the files hold only study-sample rows. They are shipped
so the file states the filter it was built with. See section 5.

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

### Group 9 -- Base-year consumption: primary system and whole home (12)

All in kWh of site energy, for the year 2025.

`base_electricity_heating_consumption`, `base_electricity_cooling_consumption`,
`base_fuelOil_heating_consumption`, `base_naturalGas_heating_consumption`,
`base_propane_heating_consumption`, `baseline_heating_consumption`,
`baseline_cooling_consumption`, `mp{mp}_heating_consumption`,
`mp{mp}_cooling_consumption`, `base_total_electricity_consumption`,
`mp{mp}_total_electricity_consumption`, `baseline_total_site_consumption`

**The first nine columns are primary-system energy only**: the energy used by
the furnace, boiler, baseboard, air conditioner or heat pump itself. They do
not include fans, pumps, or a heat pump's backup heat. Those parts are in
group 9b, and the per-year columns in group 10 count everything.

`baseline_heating_consumption` is the **sum across all four baseline heating
fuels** of the primary-system columns for that home, expressed in kWh. A home
heats with one fuel, so in practice one of the four
`base_*_heating_consumption` columns is non-zero and the sum equals it.

**A zero in `baseline_heating_consumption` is a real zero, not a missing
value.** It is exactly 0 for 1,020 rdu (about 246,974 homes), in California
(486), Florida (368), Arizona (137), Nevada (25) and Texas (4): homes whose
heating system used no energy in the weather year ResStock simulates. They are
in the sample and have an NPV.

`base_total_electricity_consumption` and
`mp{mp}_total_electricity_consumption` are whole-home **electricity**, not all
fuels. `baseline_total_site_consumption` is whole-home **all-fuel** site energy
and is the denominator of `mp{mp}_modeled_savings_frac` in group 13.

### Group 9b -- Base-year consumption: fans, pumps and heat-pump backup (12)

kWh of site energy for the year 2025, one column per fuel and part. The last
column of the table counts study-sample rdu with a value above zero.

| Column | What it is | Above zero in |
|---|---|---|
| `base_electricity_heating_fansPumps_consumption` | electricity for the existing heating system's fan or pumps | 202,058 rdu |
| `base_electricity_cooling_fansPumps_consumption` | electricity for the existing air conditioner's fan | 177,638 rdu |
| `mp{mp}_electricity_heating_fansPumps_consumption` | electricity for the heat pump's fan, heating | 219,093 rdu (MP3), 218,901 (MP4) |
| `mp{mp}_electricity_heating_hpBackup_consumption` | the heat pump's electric backup heat | 194,258 rdu (MP3), 179,810 (MP4) |
| `mp{mp}_electricity_cooling_fansPumps_consumption` | electricity for the heat pump's fan, cooling | 221,192 rdu (MP3), 221,191 (MP4) |
| `base_electricity_heating_hpBackup_consumption`, `base_naturalGas_heating_hpBackup_consumption`, `base_propane_heating_hpBackup_consumption`, `base_fuelOil_heating_hpBackup_consumption` | backup heat of an existing heat pump | none: 0 on every row, because homes with an existing heat pump are outside the sample |
| `mp{mp}_naturalGas_heating_hpBackup_consumption`, `mp{mp}_propane_heating_hpBackup_consumption`, `mp{mp}_fuelOil_heating_hpBackup_consumption` | fossil backup heat of the new heat pump | none: 0 on every row, because MP3 and MP4 use electric backup only |

The seven columns that are 0 on every row are kept so the column set does not
change when a release reports them. The ResStock 2025 dual-fuel package, for
example, burns gas as backup heat.

Add these to the primary-system columns of group 9 and you have a stream's
full 2025 energy use, which is exactly the 2025 column of group 10:

```
baseline heating, 2025 = the four base_*_heating_consumption columns
                         + base_electricity_heating_fansPumps_consumption
                         + the four base_*_heating_hpBackup_consumption columns
retrofit heating, 2025 = mp{mp}_heating_consumption
                         + mp{mp}_electricity_heating_fansPumps_consumption
                         + the four mp{mp}_*_heating_hpBackup_consumption columns
baseline cooling, 2025 = base_electricity_cooling_consumption
                         + base_electricity_cooling_fansPumps_consumption
retrofit cooling, 2025 = mp{mp}_cooling_consumption
                         + mp{mp}_electricity_cooling_fansPumps_consumption
```

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

**Each total counts every fuel and every part** (primary system, fans and
pumps, heat-pump backup), added together in kWh. The 2025 column equals the
sum of that stream's group 9 and group 9b columns.

**A baseline heating total cannot always be priced with one price.** For a
gas, propane or fuel oil home it is mostly fuel plus a little electricity for
the fan or pumps. To apply your own prices, split the total by fuel first.
There are two ways:

- use the detailed copy, which has each year's total already split by fuel
  (group 10b); or
- split it yourself. Every fuel in a stream is scaled by the same degree-day
  factor, so a fuel's share of the total is the same in every year as in 2025:
  fuel kWh in a year = that fuel's 2025 kWh (groups 9 and 9b) x (that year's
  total / the 2025 total). Where the 2025 total is 0, every year is 0.

Retrofit heating and both cooling streams are all electricity for MP3 and MP4,
so those three totals can be priced directly at the electricity price.

### Group 10b -- Annual consumption by fuel (150), detailed copy only

In the detailed copy, each stream's 15 totals are followed by the same totals
split by fuel:

| Stream | Fuels | Columns | Column pattern |
|---|---|---|---|
| Baseline heating | electricity, naturalGas, fuelOil, propane | 60 | `baseline_{year}_heating_{fuel}_consumption` |
| Retrofit heating | electricity, naturalGas, fuelOil, propane | 60 | `ref2025_mp{mp}_{year}_heating_{fuel}_consumption` |
| Baseline cooling | electricity | 15 | `baseline_{year}_cooling_electricity_consumption` |
| Retrofit cooling | electricity | 15 | `ref2025_mp{mp}_{year}_cooling_electricity_consumption` |

Each value is that fuel's full use for the end use in that year, every part
included. A stream's fuel columns add up to its total column. A fuel the home
does not use is 0, not blank; the retrofit heating columns for natural gas,
fuel oil and propane are 0 on every row for MP3 and MP4.

To cost a year, multiply each fuel column by that fuel's price for the year
and add the results (section 8).

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
specific and, in Allegheny County, above the national share:

| Scope | MP | Room AC | Central AC |
|---|---|---|---|
| National | MP3 | 92.94% | 13.32% |
| National | MP4 | 66.17% | 2.29% |
| Allegheny | MP3 | 97.25% | 20.72% |
| Allegheny | MP4 | 75.82% | 3.32% |

The sample has 43,564 room AC rdu and 177,641 central AC rdu nationally, and
364 and 782 in Allegheny County.

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

The last column is the cooling credit the NPV subtracted. In these files it
equals `mp{mp}_cooling_replacement_installed_cost_v4MID` on every row, because
every study-sample home has an air conditioner of its own with a replacement
cost.

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

Either column can be negative. The heat pump then costs more to run for that
end use than the existing system did. Group 11 explains this for cooling, and
the worked example in section 10 shows it for heating.

### Group 16-18 -- The result (3)

| Column | Meaning |
|---|---|
| `ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | heat pump cost minus both avoided replacements |
| `ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_private_npv_fixed_base` | the NPV, in USD2025 |
| `ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base` | 1.0 adopts, 0.0 does not |

Reading the name: `heatingLCC_coolingLCC` means both avoided replacements are
credited; `unsub` means no rebate is applied.

The model computes nine NPV variants in total (three credit scopes, each with
no rebate, a 2024-guidance rebate, and a June 2026-guidance rebate). **This
export ships one of the nine** -- the unsubsidized, both-credits case -- because
the intended use is to model the unsubsidized economics and layer your own
rebate assumptions on top. The other eight still exist in the full model output.

---

## 4.1 What is deliberately not in the file, and why

The model's results table has 215 columns per home, and this export takes 107
of them (the other 60 come from the fuel-cost table). Nothing below was lost
or forgotten -- each was left out for a stated reason, and all of it still
exists in the full model output.

| Left out | How many | Why |
|---|---|---|
| Eight of the nine NPV cases, with their net capital cost and adopter flags | 24 | The model prices three credit scopes, each with no rebate, a December 2024-guidance rebate, and a June 2026-guidance rebate. This export ships the **unsubsidized** case that credits both avoided replacements, because the intended use is to model unsubsidized economics and apply your own rebate assumptions on top. Shipping all nine invites averaging across cases that are alternatives, not additive. |
| The December 2024-guidance rebate amount and its program label | 2 | This export is unsubsidized. The June 2026 amount is kept as reference (section 7); carrying two competing rebate vintages beside an unsubsidized NPV is an invitation to subtract the wrong one. |
| `ref2025_mp{mp}_heating_total_capital_cost_v4MID` | 1 | Despite the name, this column has the December 2024 rebate already netted out of it -- for some homes by as much as $8,000. In an unsubsidized file that is a trap. The gross installed cost is shipped instead, as `mp{mp}_heating_upgrade_installed_cost_v4MID`. |
| `private_discount_rate_variable` | 1 | The model can run at 3%, 7%, 10%, and a variable rate. Only the 7% (`fixed_base`) run exists for this release, so this column would be misleading. |
| Emissions and climate damages | 48 | Removed at the researcher's direction. They play no part in the adoption decision, which is based on the private NPV alone, so carrying them beside the NPV suggests a link that the model does not make. |
| Average annual fuel costs and their percent change | 6 | Summaries of the lifetime fuel costs in group 11, which are shipped. |
| Intermediate energy totals and checks | 6 | `mp{mp}_total_site_consumption`, the two `..._heating_annual_consumption_kwh` columns, `mp{mp}_hvac_energy_savings_kwh`, `mp{mp}_whole_home_energy_savings_kwh`, and `mp{mp}_modeled_savings_frac_whole_home`. The heating and cooling ones can be rebuilt from the 2025 columns of group 10. The whole-home ones are a cross-check against ResStock's own whole-home change and feed no result. |
| Three equipment sizes and one climate zone label | 4 | `base_size_heating_system_primary_k_btu_h`, `base_size_cooling_system_primary_k_btu_h`, `size_heat_pump_backup_primary_k_btu_h`, `climate_zone_iecc`. Not selected for this export; see the note on equipment sizes in group 5. |
| Bookkeeping columns | 16 | REMDB cost-table row lookups (`row_id_*`, `*_pm1_euss`, `*_pm2_euss`, `*_pm2_euss_original`) and the intermediate validation flags (`include_all`, `valid_fuel_*`, `valid_tech_*`). They describe how the model found a number, not the number itself. |

Three of the validation flags **are** shipped: `include_sample`,
`include_heating` and `include_cooling` (see sections 3.1 and 5).

---

## 5. Blanks, zeros, and the three flags

The export never fills a blank, and it never turns a zero into a blank.

**Blanks.** These files hold only study-sample rows, so no result is blank
here. One shipped column has blanks: `gea_region`, for 5 rdu (about 1,211
homes) in one county (FIPS 46102) that the model's county-to-region table does
not cover. That column is used only for emissions and climate damages, which
are not in this file, so nothing else on those rows is affected.

**Zeros.** A zero is a real value. `baseline_heating_consumption` is 0 for
1,020 rdu (group 9). `mp{mp}_heating_consumption` is 0 for 103 rdu under MP3
and 1,211 under MP4. `baseline_cooling_consumption` is 0 for 7 rdu. All of
these homes are in the sample and have an NPV.

**The three flags.** In the model's full results table of 331,526 rdu:

| Flag | True when | Share of the full results table |
|---|---|---|
| `include_sample` | every condition in section 3.1 holds | 221,205 of 331,526 rdu = **66.72%** |
| `include_heating` | baseline heating fuel and technology are both in scope | 260,211 of 331,526 rdu = 78.49% |
| `include_cooling` | the dwelling has a central or room air conditioner of its own | 250,307 of 331,526 rdu = 75.50% |

Because the weight is the same for every row, those shares are identical
whether you count rows or actual homes.

In these files all three are `True` on every row. `include_sample` is the one
that defines the sample. The other two are not enough on their own: a home can
pass the heating check and have no air conditioner.

**Denominators.** Every row is a home the model evaluated, so the row count of
the file is the correct denominator for any per-home average: 221,205
nationally and 1,146 in Allegheny County. (Of Allegheny County's 1,610
single-family rdu, 1,356 pass the heating check and 1,327 the cooling check;
1,146 pass both.)

---

## 6. Rounding

**Values are saved as the model computed them, not rounded.** Energy, yearly
costs, lifetime costs and discounted savings carry full precision, so a value
such as 12795.77598727 kWh is normal. `weight` is written as
242.13101272727272; this document quotes it as 242.131013.

Five kinds of value are rounded to cents on purpose: the NPV, installed costs,
rebate amounts, household income, and area median income.

Two consequences:

- If you rebuild the NPV from its parts (section 10), you land within half a
  cent of the shipped NPV, because the NPV is rounded to cents and the
  discounted savings are not.
- Do not round as you go. The model does not round a year's cost before
  summing or discounting. Round only the final figure you report.

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

**As modeled here, both programs reach only homes with electric heating.** In
these files no natural gas, propane or fuel oil home has a label other than
`'Not Eligible'`.

**Treat these as provisional.** Four things are not modeled:

1. **No state funding cap.** The amounts are uncapped potential, not money that
   will actually be paid. Real programs run out of funds.
2. **South Dakota is zeroed** in every scenario, because it never took part in
   the federal rebate programs.
3. **Weatherization prerequisites are not enforced** and dual-fuel systems are
   not modeled.
4. **One program per home** -- never both.

`mp{mp}_modeled_savings_frac` is the home's heating and cooling energy saving
divided by `baseline_total_site_consumption` (group 9), so you can check it.
The saving is read from the 2025 columns of group 10: (baseline heating +
baseline cooling) - (retrofit heating + retrofit cooling). Every fuel and
every part is counted, and both sides are ResStock's own base-year values.

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
the home's state electricity price. A baseline cost prices each fuel at its own
price: the heating fuel named in `base_heating_fuel`, plus electricity for the
fan or pumps of a gas, propane or fuel oil system.

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
dollars.** The worked example in section 10 shows a home whose heating energy
falls by 49% while its heating cost almost doubles, from $687 to $1,286 in the
first year, because the gas it stops buying is about a quarter the price of
the electricity it starts buying.

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
dollar figure below is per dwelling, not per row-of-242. Values are shown to
two decimals here; the file holds full precision.

| | |
|---|---|
| State / county | PA / `G4200030` (Allegheny) |
| Census division | Middle Atlantic |
| Baseline heating | Natural Gas Fuel Furnace, 92.5% AFUE |
| Baseline cooling | Central AC, SEER 13 |
| Retrofit | ASHP, SEER 16, 9.5 HSPF |
| Floor area | 1,690 sq ft |
| `weight` | 242.13101272727272 (real homes represented) |
| `include_sample` | True |
| `private_discount_rate_fixed_base` | 0.07 |

### Step 1 -- base-year consumption (kWh, 2025)

| Column | Value |
|---|---|
| `base_naturalGas_heating_consumption` (the furnace's gas) | 12,795.78 |
| `base_electricity_heating_fansPumps_consumption` (the furnace's fan) | 286.92 |
| **Baseline heating, every part** | **13,082.69** |
| `mp3_heating_consumption` (the heat pump itself) | 2,941.26 |
| `mp3_electricity_heating_hpBackup_consumption` (electric backup heat) | 3,467.91 |
| `mp3_electricity_heating_fansPumps_consumption` (the heat pump's fan) | 256.44 |
| **Retrofit heating, every part** | **6,665.61** |
| `base_electricity_cooling_consumption` | 913.80 |
| `base_electricity_cooling_fansPumps_consumption` | 77.96 |
| **Baseline cooling, every part** | **991.75** |
| `mp3_cooling_consumption` | 630.69 |
| `mp3_electricity_cooling_fansPumps_consumption` | 142.73 |
| **Retrofit cooling, every part** | **773.41** |

The four bold totals are the 2025 columns of group 10.

The heat pump uses 49% less heating energy than the furnace and its fan. Just
over half of the heat pump's heating energy is electric backup heat. That is
the change in energy. It is not the change in dollars, because the fuel
changes.

### Step 2 -- project each year with the degree-day factor

2025 factors are 1.0, so 2025 consumption equals the base year exactly. By 2039
the Middle Atlantic heating factor has fallen and the cooling factor has risen:

| Year | Baseline heating kWh | of which gas | of which electricity | Retrofit heating kWh | Baseline cooling kWh | Retrofit cooling kWh |
|---|---|---|---|---|---|---|
| 2025 | 13,082.69 | 12,795.78 | 286.92 | 6,665.61 | 991.75 | 773.41 |
| 2032 | 12,448.41 | 12,175.40 | 273.01 | 6,342.44 | 1,174.31 | 915.78 |
| 2039 | 12,087.69 | 11,822.60 | 265.10 | 6,158.66 | 1,248.11 | 973.33 |

The gas and electricity columns are in the detailed copy (group 10b). From the
main file, split the baseline heating total with the 2025 shares, as group 10
describes.

### Step 3 -- price each fuel at its own price

Baseline heating is natural gas for the furnace and electricity for its fan.
Everything else is electricity.

| Year | Gas $/kWh | Electricity $/kWh | Baseline heating cost | Retrofit heating cost | Heating saving |
|---|---|---|---|---|---|
| 2025 | 0.049390 | 0.193000 | $687.36 | $1,286.46 | -$599.10 |
| 2032 | 0.047931 | 0.198085 | $637.66 | $1,256.34 | -$618.68 |
| 2039 | 0.048939 | 0.203835 | $632.62 | $1,255.35 | -$622.73 |

For 2025 the baseline heating cost is 12,795.78 kWh of gas x $0.049390 =
$631.99, plus 286.92 kWh of fan electricity x $0.193000 = $55.37.

This is the point of section 8. Electricity costs roughly four times as much
per kWh as gas here, so a 49% cut in heating energy becomes a heating bill
that is about $600 a year higher.

Cooling is electricity on both sides, and the heat pump uses less of it: the
cooling saving is $42.14 in 2025, rising to $56.01 by 2039.

### Step 4 -- discount and sum

Discount factor is `1 / 1.07 ^ (year - 2025)`, so 1.000000 in 2025, 0.622750 in
2032, 0.387817 in 2039.

| | Sum over 2025-2039 |
|---|---|
| Discounted heating saving | **-$5,989.72** |
| Discounted cooling saving | **$483.16** |

These match `ref2025_mp3_heating_discounted_lifetime_savings_fixed_base` and
`ref2025_mp3_cooling_discounted_lifetime_savings_fixed_base`.

The heating saving is negative: this home pays more to heat with the heat pump
than with gas in every year. Fuel switching, not efficiency, is what governs
the heating side.

### Step 5 -- net capital cost

| | |
|---|---|
| `mp3_heating_upgrade_installed_cost_v4MID` | $12,084.50 |
| less `mp3_heating_replacement_installed_cost_v4MID` | $3,873.50 |
| less `mp3_cooling_replacement_credit_applied_v4MID` | $5,401.75 |
| = `ref2025_mp3_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | **$2,809.25** |

The heat pump costs $12,084.50, but this household was going to have to replace
a furnace and an air conditioner anyway. Crediting both, the extra cost of
choosing a heat pump is $2,809.25.

`mp3_rebate_eligibility_june2026` is `'Not Eligible'` for this home and its
rebate amount is 0. Neither enters this NPV, which is unsubsidized.

### Step 6 -- the NPV

```
  -$5,989.7165   discounted heating saving
+    $483.1618   discounted cooling saving
-  $2,809.2500   net capital cost
=  -$8,315.8046, which the model rounds to -$8,315.80
```

`ref2025_mp3_heatingLCC_coolingLCC_unsub_private_npv_fixed_base` = **-$8,315.80**,
and
`ref2025_mp3_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base` = **0.0**,
because the NPV is below zero.

The arithmetic above uses four decimals on purpose. With the savings rounded
to cents first, the sum comes to -$8,315.81, one cent off. Section 6 explains
why.

The story for this home: just over half of the heat pump's heating energy is
electric backup heat, and electricity costs about four times as much per kWh
as gas here, so heating costs about $600 more a year. The cooling saving and
the two avoided replacements do not make up for it.

---

## 11. The county CSV

Eleven columns, one row per county, assembled from three separate model tables
and joined on `county`.

| Column | Unit |
|---|---|
| `county` | Census GISJOIN string -- do not cast to a number |
| `state` | two-letter abbreviation |
| `home_count` | study-sample homes in the county, that is the sum of `weight` |
| `adoption_rate_pct` | percent 0-100, the share of study-sample homes with NPV >= 0 |
| `operating_cost_pct_change` | percent, the county median of each home's `(retrofit - baseline) / baseline x 100` |
| `baseline_elec_gwh`, `retrofit_elec_gwh`, `elec_change_gwh`, `site_energy_change_gwh` | GWh |
| `pct_elec_demand_change`, `pct_site_energy_change` | percent |

Some counties rest on very few sampled dwelling units: 74 of the 3,079 counties
have 1 rdu and 280 have 3 or fewer, so one rdu can set a county's value. Check
`home_count` before quoting a county (242 homes is 1 rdu).

**`site_energy_change_gwh` and `pct_site_energy_change` are all-fuel numbers**,
separate from the electricity pair. They come from ResStock's whole-home site
energy, which counts natural gas, fuel oil, and propane in kWh alongside
electricity. Electricity usually rises when a fossil system is replaced, while
site energy falls because the fuel is no longer burned, so the two pairs often
have opposite signs. `pct_site_energy_change` is taken against baseline site
energy, not baseline electricity. For an electricity reading use
`elec_change_gwh` and `pct_elec_demand_change`.

---

## 12. What will change in the next release

This export is built on ResStock 2022.1.1. The next version of the model moves
to ResStock 2025 and adds dual-fuel (hybrid) systems, where a heat pump and a
fossil backup share the heating load.

Expect these to change: **column names**, the retrofit heating columns (a
dual-fuel heat pump burns gas as backup, so the gas columns that are 0 today
will hold numbers), and the county geography. Analysis built on this file will
need rework, not just a refresh. Anything you build should keep the column
names in one place rather than scattered through formulas.

---

## 13. Validation performed

Checked on run `2026-10-05_21-33`, for both MP3 and MP4, by building the files
with `export_tepper_household` from that run's saved results:

- Row counts: 221,205 in each national file and 1,146 in each Allegheny file.
- Column counts: 167 in the main file and 317 in the detailed copy, in the
  declared order, with no duplicate names.
- `include_sample`, `include_heating` and `include_cooling` are `True` on
  every row of every file.
- The Allegheny files were read back and compared with the two source tables:
  every value and every blank is in the same place.
- The only blanks in a national file are the 5 in `gea_region`.
- Each stream's per-fuel columns add up to its total in every year, and the
  base-year columns of groups 9 and 9b add up to the 2025 totals.
- The reconciliation in section 10 holds on every row: discounted heating
  saving + discounted cooling saving - net capital cost equals the shipped
  NPV, with none off by more than half a cent. The net capital cost equals the
  heat pump cost minus the two credits, and the adopter flag is 1.0 exactly
  where the NPV is zero or above.
- The discounted savings were rebuilt from the per-year fuel costs and match
  the shipped columns.
- The county file was built with the main notebook's Tepper export cell, run
  on the same saved results: 3,079 rows nationally and 1 for Allegheny County,
  11 columns, no blank cell, and `home_count` adds up to 53,560,591 homes
  nationally. The adoption and demand tables agree on every county's
  `home_count`.
