# TARE Column Derivation / Rename Map

Companion to `tare_column_inventory.csv` / `.txt`. For each column in the loaded
analysis frames, this file records **where the name came from**: either the raw
ResStock (EUSS 2022.1.1) field it was renamed from, or the calculation that
produced it.

- **Section A (passthrough renames)** is extracted directly from
  `energy_consumption_and_metadata/process_euss_data.py`
  (`df_enduse_refactored` / `df_enduse_compare` and the STEP 1 metadata block).
  These rows are authoritative: `new_name <- raw ResStock field`.
- **Section B (derived columns)** documents the computed column families by
  naming pattern. These derivations come from project conventions (CLAUDE.md)
  and the session changelogs, not from a line-by-line parse, so treat them as
  descriptive rather than authoritative.

Conventions used below:
- `{mp}` = measure-package number (3, 4). `ref2025_` = the `'2025 Reference
  Case'` scenario prefix (always derived via `define_scenario_params`).
- `{category}` in {heating, cooling, waterHeating, clothesDrying, cooking}.
- `df_baseline` = raw baseline (MP0) EUSS frame; `df_mp` = raw post-retrofit
  (upgrade) EUSS frame. A `base_` prefix marks a baseline value; equipment
  columns without it are the post-retrofit value.

To regenerate Section A after a rename change, re-run the AST extraction over
`process_euss_data.py` (the `df_enduse[...] = df_*[...]` and STEP 1 dict
assignments whose right-hand side is an `in.` / `out.` field).

---

## Section A -- Passthrough renames from ResStock (source-extracted)

### A.1 Identifiers and geography

| Analysis column | Renamed from (ResStock) | Transform |
|---|---|---|
| `weight` | `weight` | already named upstream (sample weight) |
| `square_footage` | `in.sqft` | |
| `state` | `in.state` | |
| `census_region` | `in.census_region` | |
| `census_division` | `in.census_division` | |
| `census_division_recs` | `in.census_division_recs` | |
| `building_america_climate_zone` | `in.building_america_climate_zone` | |
| `reeds_balancing_area` | `in.reeds_balancing_area` | |
| `county` | `in.county` | |
| `county_fips` | `in.county` | `.apply(lambda x: x[1:3] + x[4:7])` (5-digit FIPS) |
| `puma` | `in.puma` | |
| `county_and_puma` | `in.county_and_puma` | |
| `city` | `in.city` | `.apply(extract_city_name)` |
| `urbanicity` | `in.puma_metro_status` | `.apply(map_metro_status)` |
| `weather_file_city` | `in.weather_file_city` | |
| `Longitude` | `in.weather_file_longitude` | |
| `Latitude` | `in.weather_file_latitude` | |
| `gea_region` | (computed) | `county_fips.map(COUNTY_TO_GEA)` -- Cambium GEA crosswalk |

### A.2 Building and household

| Analysis column | Renamed from (ResStock) | Transform |
|---|---|---|
| `building_type` | `in.geometry_building_type_recs` | |
| `income` | `in.income` | |
| `federal_poverty_level` | `in.federal_poverty_level` | |
| `occupancy` | `in.occupants` | |
| `tenure` | `in.tenure` | |
| `vacancy_status` | `in.vacancy_status` | |
| `vintage` | `in.vintage` | |

### A.3 Baseline equipment and fuels

| Analysis column | Renamed from (ResStock) | Transform |
|---|---|---|
| `base_heating_fuel` | `in.heating_fuel` | |
| `heating_type` | `in.hvac_heating_type_and_fuel` | |
| `base_heating_efficiency` | `in.hvac_heating_efficiency` | |
| `cooling_type` | `in.hvac_cooling_type` | |
| `base_cooling_efficiency` | `in.hvac_cooling_efficiency` | |
| `base_cooling_fuel` | (computed) | constant `'Electricity'` (cooling is always electric) |
| `base_waterHeating_fuel` | `in.water_heater_fuel` | |
| `waterHeating_type` | `in.water_heater_efficiency` | |
| `base_clothesDrying_fuel` | `in.clothes_dryer` | |
| `base_cooking_fuel` | `in.cooking_range` | |

### A.4 Post-retrofit equipment (from the upgrade frame `df_mp`)

| Analysis column | Renamed from (ResStock) | Transform |
|---|---|---|
| `hvac_heating_type_and_fuel` | `in.hvac_heating_type_and_fuel` | |
| `hvac_heating_efficiency` | `in.hvac_heating_efficiency` | |
| `size_heating_system_primary_k_btu_h` | `out.params.size_heating_system_primary_k_btu_h` | |
| `hvac_cooling_type` | `in.hvac_cooling_type` | |
| `hvac_cooling_efficiency` | `in.hvac_cooling_efficiency` | |
| `size_cooling_system_primary_k_btu_h` | `out.params.size_cooling_system_primary_k_btu_h` | |
| `hvac_has_ducts` | `in.hvac_has_ducts` | |
| `water_heater_efficiency` | `in.water_heater_efficiency` | |
| `water_heater_fuel` | `in.water_heater_fuel` | |
| `water_heater_in_unit` | `in.water_heater_in_unit` | |
| `size_water_heater_gal` | `out.params.size_water_heater_gal` | |
| `clothes_dryer_in_unit` | `in.clothes_dryer` | |
| `cooking_range_in_unit` | `in.cooking_range` | |

### A.5 Envelope and geometry (from the upgrade frame `df_mp`)

| Analysis column | Renamed from (ResStock) | Transform |
|---|---|---|
| `base_insulation_atticFloor` | `in.insulation_ceiling` | |
| `floor_area_attic_ft2` | `out.params.floor_area_attic_ft_2` | |
| `base_ducts` | `in.ducts` | |
| `duct_unconditioned_area_ft2` | `out.params.duct_unconditioned_surface_area_ft_2` | |
| `base_insulation_wall` | `in.insulation_wall` | |
| `wall_area_above_grade_ft2` | `out.params.wall_area_above_grade_exterior_ft_2` | |
| `base_foundation_type` | `in.geometry_foundation_type` | |
| `base_insulation_foundation_wall` | `in.insulation_foundation_wall` | |
| `base_insulation_rim_joist` | `in.insulation_rim_joist` | |
| `floor_area_foundation_ft2` | `out.params.floor_area_foundation_ft_2` | |
| `rim_joist_area_above_grade_ft2` | `out.params.rim_joist_area_above_grade_exterior_ft_2` | |
| `base_insulation_roof` | `in.insulation_roof` | |
| `roof_area_ft2` | `out.params.roof_area_ft_2` | |

### A.6 Baseline end-use consumption (kWh, from `df_baseline` `out.*`)

| Analysis column | Renamed from (ResStock) |
|---|---|
| `baseline_total_site_consumption` | `out.site_energy.total.energy_consumption.kwh` |
| `base_electricity_heating_consumption` | `out.electricity.heating.energy_consumption.kwh` |
| `base_fuelOil_heating_consumption` | `out.fuel_oil.heating.energy_consumption.kwh` |
| `base_naturalGas_heating_consumption` | `out.natural_gas.heating.energy_consumption.kwh` |
| `base_propane_heating_consumption` | `out.propane.heating.energy_consumption.kwh` |
| `base_electricity_cooling_consumption` | `out.electricity.cooling.energy_consumption.kwh` |
| `base_electricity_waterHeating_consumption` | `out.electricity.hot_water.energy_consumption.kwh` |
| `base_fuelOil_waterHeating_consumption` | `out.fuel_oil.hot_water.energy_consumption.kwh` |
| `base_naturalGas_waterHeating_consumption` | `out.natural_gas.hot_water.energy_consumption.kwh` |
| `base_propane_waterHeating_consumption` | `out.propane.hot_water.energy_consumption.kwh` |
| `base_electricity_clothesDrying_consumption` | `out.electricity.clothes_dryer.energy_consumption.kwh` |
| `base_naturalGas_clothesDrying_consumption` | `out.natural_gas.clothes_dryer.energy_consumption.kwh` |
| `base_propane_clothesDrying_consumption` | `out.propane.clothes_dryer.energy_consumption.kwh` |
| `base_electricity_cooking_consumption` | `out.electricity.range_oven.energy_consumption.kwh` |
| `base_naturalGas_cooking_consumption` | `out.natural_gas.range_oven.energy_consumption.kwh` |
| `base_propane_cooking_consumption` | `out.propane.range_oven.energy_consumption.kwh` |

---

## Section B -- Derived columns (by naming pattern)

These are not ResStock renames; they are computed downstream. Derivations are
described from project conventions and the session changelogs.

### B.1 Lifetime fuel cost (private_impact/calculate_lifetime_fuel_costs.py)

| Column pattern | Derivation |
|---|---|
| `baseline_{category}_lifetime_fuel_cost` | 15-year fuel cost of the baseline system for that end use (price x consumption x price/degree-day factors, summed over `LIFETIME_YEARS`) |
| `ref2025_mp{mp}_{category}_lifetime_fuel_cost` | same, for the post-retrofit system |
| `ref2025_mp{mp}_{category}_lifetime_savings_fuel_cost` | baseline lifetime cost - retrofit lifetime cost (can be negative; see the cooling-negative note) |
| `ref2025_mp{mp}_cooling_lifetime_savings_negative` | boolean flag: cooling lifetime savings < 0 (reporting only; not used in NPV) |

### B.2 Average-annual fuel cost (added 2026-07-20/21; additive, map-neutral)

| Column pattern | Derivation |
|---|---|
| `baseline_{category}_avg_annual_fuel_cost` | `baseline_{category}_lifetime_fuel_cost / lifetime_years` |
| `ref2025_mp{mp}_{category}_avg_annual_fuel_cost` | retrofit lifetime cost / `lifetime_years` |
| `ref2025_mp{mp}_{category}_avg_annual_fuel_cost_pct_change` | `(retrofit_annual - baseline_annual) / baseline_annual * 100`; NaN where `baseline_annual <= 0`. Equals the lifetime percent change (the `/lifetime` cancels), so the choropleth is numerically identical. |

### B.3 Capital cost (private_impact / REMDB v4)

| Column pattern | Derivation |
|---|---|
| `...capital_cost...`, `...installed_cost...`, `...replacement...` | REMDB v4 installed / replacement costs, inflated to USD2025 via `ANCHOR_YEAR`. MP3 carries the ENERGY STAR SEER 15->16 override (+~$796.83/home weighted). |

### B.4 NPV -- nine cases per MP (three scopes x three rebate policy scenarios)

Column name: `ref2025_mp{mp}_{scope}_{rebate}_private_npv{method_suffix}`
built via `create_npv_case_col`.

| Token | Values | Meaning |
|---|---|---|
| `{scope}` | `heatingSavings_coolingLCC`, `heatingLCC_coolingSavings`, `heatingLCC_coolingLCC` | which end use gets avoided-replacement capital credited (`LCC`) vs operating savings only (`Savings`). All nine cases include both heating and cooling operating savings. |
| `{rebate}` | `unsub`, `sub`, `sub_june2026` | no rebate / 2024 DOE guidance / June 2026 DOE guidance (HEEHR fuel gate) |
| `{method_suffix}` | `_fixed_base` (7%), `_central`, ... | private discount-rate method (carries its own leading underscore) |

Derivation: lifetime heating + cooling operating savings, plus the credited
avoided-replacement capital for the `LCC` end use(s), minus the incremental
install cost, minus the rebate amount for the `sub`/`sub_june2026` cases,
discounted over `LIFETIME_YEARS`.

### B.5 Economic adopter -- nine per MP

Column name: `ref2025_mp{mp}_{npv_case}_econ_adopter{method_suffix}`.

| Value | Meaning |
|---|---|
| `1.0` | that case's private NPV `>= 0` (heat pump pays for itself) |
| `0.0` | valid home, NPV `< 0` |
| `NaN` | excluded (invalid baseline fuel/tech, or not in this MP) |

### B.6 Rebate (determine_rebate_eligibility_and_amount.py)

| Column pattern | Derivation |
|---|---|
| `mp{mp}_heating_rebate_amount_{cost_scenario}` | 2024 guidance rebate dollars (HEEHR or HOMES) |
| `mp{mp}_heating_rebate_amount_june2026_{cost_scenario}` | June 2026 guidance rebate dollars (HEEHR with fossil fuel gate, or HOMES) |
| `mp{mp}_rebate_eligibility_ira2024` | label: `HEEHR` / `HOMES` / `None` (2024) |
| `mp{mp}_rebate_eligibility_june2026` | label: `HEEHR` / `HOMES` / `None` (June 2026) |
| `mp{mp}_modeled_savings_frac` | whole-home degree-day-adjusted heating + cooling energy delta / `baseline_total_site_consumption` (drives HOMES tier) |

### B.7 Emissions and damages (RCM: ap2, easiur, inmap)

| Column pattern | Derivation |
|---|---|
| `...mt_co2e...`, `...lrmer...`, `...srmer...` | avoided CO2-equivalent tons; Cambium long/short-run marginal emission rates |
| `...damages_climate...`, `...damages_health...` | monetized climate / health damages by RCM model. Reported as outcomes only -- NOT inputs to the adoption decision. |

---

*Section A extracted from `process_euss_data.py`. Section B follows CLAUDE.md
conventions and the 2026-07-06 through 2026-07-21 session changelogs.*
