# Tepper CSV Exports -- Data Dictionary

One-time additive exports for the Tepper MBA team's household and county
cash-flow analysis. Produced by `cmu_tare_model/utils/export_tepper_csv.py`.

**Scope of this run:** Pennsylvania only (`location_id = PA`), 15,651 homes,
measure packages MP3 and MP4, private discount rate `fixed_base`. Every value is
copied verbatim from a loaded model-run DataFrame -- nothing here is recomputed,
re-derived, or rounded by the export. The applicable-home denominators quoted
below (78.4% heating / 82.7% cooling) are specific to this PA run and will
change for any other geography.

All dollar columns are in **USD2025**.

---

## 1. Files produced

Written to `{output_folder_path}/tepper_export/`, distinct from the frozen
`retrofit_mp{mp}_results/` and `baseline_summary/` trees:

| File | Grain | Rows (PA run) |
|---|---|---|
| `tepper_household_mp{mp}_{location_id}_{date}.csv` | one row per home | 15,651 |
| `tepper_county_mp{mp}_{location_id}_{date}.csv` | one row per county | 67 |

One of each per measure package (MP3, MP4).

---

## 2. Household CSV

**Source frame:** `DATAFRAMES_BY_MP[mp]['fixed_base']` -- the final loaded
household frame, indexed by `bldg_id`. The `bldg_id` index is written as the
first CSV column and survives a read/write round trip.

**Selection:** an explicit, ordered list of 100 included columns (plus the
`bldg_id` index) built in `build_household_column_list()`. The list is named
column-by-column, not inferred from a theme heuristic. Every
measure-package-specific name is derived from the scenario-prefix helper and
the shared NPV-case list, so no scenario string is hardcoded.

### Column groups and their source

All groups come from the single household frame above. Group sizes below sum to
100.

| # | Group | Count | Notes |
|---|---|---|---|
| 1 | Identifiers | 6 (+ index) | `bldg_id` (index), `weight`, `state`, `county`, `county_fips`, `puma`, `county_and_puma` |
| 2 | Geography | 11 | census region/division, climate zone, ReEDS balancing area, city, urbanicity, weather-file city, Longitude, Latitude, GEA region |
| 3 | Building | 6 | square footage, building type, occupancy, tenure, vacancy status, vintage |
| 4 | Household income | 7 | `income`, `federal_poverty_level`, `household_income`, `census_area_medianIncome`, `income_level`, `percent_AMI`, `lmi_or_mui` |
| 5 | Existing HVAC | 15 | baseline heating/cooling fuel, type, efficiency; ducts; primary system sizes |
| 6 | Retrofit HVAC | 2 | `upgrade_hvac_heating_efficiency`, `upgrade_hvac_cooling_efficiency` |
| 7 | Consumption | 9 | baseline electricity/fuel-oil/natural-gas/propane heating, baseline heating/cooling, retrofit `mp{mp}` heating/cooling |
| 8 | Fuel costs | 6 | baseline + retrofit lifetime heating/cooling fuel cost and heating/cooling savings |
| 9 | Installed costs | 4 | heating replacement, heating upgrade, cooling replacement, heating total capital |
| 10 | Rebate | 4 | 2024 `mp{mp}_heating_rebate_amount_v4MID`; June 2026 `mp{mp}_heating_rebate_amount_june2026_v4MID`; `mp{mp}_rebate_eligibility_june2026`; `mp{mp}_modeled_savings_frac` |
| 11 | Emissions and damages | 12 | LRMER mid-case tonnage and central climate damages only (see note) |
| 12 | Model parameters | 3 | `public_discount_rate`, `private_discount_rate_fixed_base`, `private_discount_rate_variable` |
| 13 | Nine NPV | 9 | `ref2025_mp{mp}_{case}_private_npv_fixed_base` for each of the nine cases |
| 14 | Nine net capital | 9 | `ref2025_mp{mp}_{case}_net_capital_cost_v4MID` for each of the nine cases |
| 15 | Nine adopter | 9 | `ref2025_mp{mp}_{case}_econ_adopter_fixed_base` for each of the nine cases |

**Why these columns.** The export keeps the columns that describe a home and its
retrofit economics -- now including both rebate policy scenarios and all nine NPV cases
(with matching net-capital and adopter columns) -- and drops:

- **18 bookkeeping columns** -- eight REMDB lookup fields
  (`*_pm1_euss` / `*_pm2_euss` / `*_pm2_euss_original`), three `row_id_*`
  fields, and seven validation flags (`include_all`, `valid_fuel_heating`,
  `valid_tech_heating`, `include_heating`, `valid_fuel_cooling`,
  `valid_tech_cooling`, `include_cooling`).
- **36 emissions/damages columns** not in the central estimate -- the SRMER
  series and the lower/upper damage bounds (see note below).

The two validation flags `include_heating` and `include_cooling` are dropped
from the CSV but are the *reason* for the missing-value pattern in section 2.1;
they are documented here rather than exported.

### The nine NPV cases

Each measure package carries all nine cases -- three cost scopes, each split into
three rebate policy scenarios: unsubsidized (`_unsub`), subsidized under 2024 guidance
(`_sub`, current HEEHR), and subsidized under June 2026 guidance
(`_sub_june2026`, HEEHR + HOMES + fuel gate):

- `heatingSavings_coolingLCC` -- credits avoided cooling replacement capital;
  heating contributes operating savings only.
- `heatingLCC_coolingSavings` -- credits avoided heating replacement capital;
  cooling contributes operating savings only.
- `heatingLCC_coolingLCC` -- credits both avoided replacements.

All nine count both heating and cooling operating savings. The nine cases are
never collapsed and no rebate-policy-scenario arm is dropped; each NPV case has a matching
net-capital-cost column and an economic-adopter flag.

### Emissions/damages -- central estimate only

To keep the damages block focused, the export ships only the LRMER mid-case
tonnage (`_mt_co2e_lrmer`) and the central climate damages
(`_damages_climate_lrmer_central`), for baseline and retrofit, heating and
cooling (12 columns). Lower/upper bounds and the SRMER series are available in
the full model-run frame if a wider sensitivity is needed later.

### 2.1 Units, missing values, and denominator

| Column family | Unit |
|---|---|
| Fuel costs, installed costs, total/net capital, rebate, NPV, climate damages | USD2025 |
| Consumption (`*_consumption`) | kWh (site energy) |
| Emissions tonnage (`*_mt_co2e_lrmer`) | metric tons CO2-equivalent, lifetime |
| System sizes (`size_*_k_btu_h`) | kBtu/h |
| `weight` | ResStock sampling weight (homes represented; ~242, uniform) |
| Longitude / Latitude | decimal degrees |
| Discount-rate columns | fraction (e.g. 0.07 = 7%) |
| Economic-adopter columns | 1.0 = adopter, 0.0 = non-adopter, NaN = excluded |

**Missing values mean "not applicable / failed validation," never zero.** The
export never fills, coerces, or drops NaN, and the exported row count equals the
input row count (15,651). In this PA run:

- Every **heating-side** column (consumption, fuel cost, NPV, net capital,
  adopter) is **78.4%** non-null -- the share of homes with
  `include_heating = True`.
- Every **cooling-side** column is **82.7%** non-null -- the share with
  `include_cooling = True`.

These denominators are the applicable-home base for any per-home average and are
**run-specific**: a home is heating-applicable only if its baseline heating fuel
and technology are in scope, and cooling-applicable only if it has central or
room AC. Recompute the denominators for any non-PA run.

---

## 3. County CSV

The county results are three separate per-MP tables produced by the analysis
notebook, not one frame. `export_tepper_county()` assembles them; it recomputes
nothing.

| Source table | Columns taken |
|---|---|
| `econ_adoption_rate_results[mp]` | `county`, `state`, `home_count`, `adoption_rate_pct` |
| `bill_savings_results[mp]` | `county`, `operating_cost_pct_change` |
| `demand_results[mp]` | `county`, `baseline_elec_gwh`, `retrofit_elec_gwh`, `elec_change_gwh`, `site_energy_change_gwh`, `pct_elec_demand_change`, `pct_site_energy_change` |

### Merge strategy

- Join on **`county` alone** (a GISJOIN-style string, e.g. `G4200030`), with
  `how='outer'` and `validate='one_to_one'`. The validate flag raises if any
  table has a duplicate county key.
- `state` and `home_count` appear in both the adoption and demand tables. They
  are taken from the adoption table and **dropped from the demand table before
  the merge**, so pandas never emits `_x` / `_y` suffixes.
- The two `home_count` values are compared first; any county where they
  disagree is reported as a warning (it does not block the export).
- `county` stays a string -- it is never cast to int or zero-stripped.

Output is exactly eleven columns in this order: `county`, `state`,
`home_count`, `adoption_rate_pct`, `operating_cost_pct_change`,
`baseline_elec_gwh`, `retrofit_elec_gwh`, `elec_change_gwh`,
`site_energy_change_gwh`, `pct_elec_demand_change`, `pct_site_energy_change`.
The Grid Impact / Peak Load analysis is intentionally excluded. In the PA run
there are 67 counties.

### 3.1 Units

| Column | Unit |
|---|---|
| `county` | GISJOIN string (do not cast to int) |
| `state` | two-letter abbreviation |
| `home_count` | homes represented (sum of ResStock weights) |
| `adoption_rate_pct` | percent, 0-100 (share of applicable homes with NPV >= 0) |
| `operating_cost_pct_change` | percent; county median of per-home `(retrofit - baseline) / baseline * 100` |
| `baseline_elec_gwh`, `retrofit_elec_gwh`, `elec_change_gwh`, `site_energy_change_gwh` | GWh |
| `pct_elec_demand_change`, `pct_site_energy_change` | percent |

A county with too few sample homes has NaN metrics (small-sample masking in the
KPI functions); NaN is preserved, not zeroed.

**On `site_energy_change_gwh` / `pct_site_energy_change`:** these are **aliases**
of the electricity metrics, not independent all-fuel numbers. Both the baseline
and retrofit sides are read from the whole-home electricity total
(`out.electricity.total.energy_consumption.kwh`), and because the retrofit fully
electrifies heating and cooling, the all-fuel site-energy change and the
electricity change converge -- so `site_energy_change_gwh == elec_change_gwh` and
`pct_site_energy_change == pct_elec_demand_change` by construction (see
`adoption_kpis/demand.py`). For an electricity read, use `elec_change_gwh` /
`pct_elec_demand_change`.

---

## 4. How the heat-pump upgrade cost and rebate are recorded

The retrofit installs a single air-source heat pump that provides **both
heating and cooling**. Its installed cost and its IRA rebate therefore cover the
whole system and are recorded **once, on the heating side**:

- `mp{mp}_heating_upgrade_installed_cost_v4MID` -- the installed cost of the
  heat pump (heating and cooling together).
- `mp{mp}_heating_rebate_amount_v4MID` -- the rebate for that heat pump.
- `mp{mp}_heating_rebate_amount_june2026_v4MID` -- the June 2026-guidance rebate
  for that heat pump (HEEHR or HOMES).
- `mp{mp}_rebate_eligibility_june2026` -- which June 2026 program applied
  (`'HEEHR'`, `'HOMES'`, or `'None'`).
- `mp{mp}_modeled_savings_frac` -- whole-home modeled savings fraction; drives the
  June 2026 HOMES tiers.

There is deliberately no separate "cooling upgrade" cost or "cooling rebate";
splitting them would double-count the one piece of equipment. The cooling cost
column instead describes the **counterfactual**:

- `mp{mp}_cooling_replacement_installed_cost_v4MID` -- the cost of replacing the
  home's existing air conditioner, credited as avoided capital in the
  `coolingLCC` NPV cases.

So the absence of a cooling upgrade cost and a cooling rebate is by design, not
a data gap.

---

## 5. Validation performed

Both exports were checked against the PA run (`2026-07-07_23-31`) for MP3 and
MP4.

**Household:**
- Round-trip `pandas.testing.assert_frame_equal` against the selected source
  slice -- values, `bldg_id` index, and NaN placement all match.
- Row count out equals row count in (15,651).
- All 27 nine-case columns (9 NPV + 9 net capital + 9 adopter) present.
- No `valid_*`, `include_*`, `row_id_*`, or `*_pm*_euss` column present.

**County:**
- `validate='one_to_one'` raises on a duplicate county key (tested).
- Output row count equals the number of unique counties and equals each input
  table's row count (67).
- No `_x` / `_y` suffix survives; `county` dtype stays object (string).
- `home_count` agreement between the adoption and demand tables is reported; a
  seeded mismatch triggers the warning (tested).
- The eleven required columns are present and no household-level column leaks
  in.

**Dispatch wiring:** adding the `tepper_household` and `tepper_county` branches
to `export_model_run_output()` left every existing branch byte-identical -- a
`summary` export produces the same bytes before and after the change.
