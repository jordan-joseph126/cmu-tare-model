# Tepper export review -- ResStock 2025.1, Dual Fuel Heating System (mp=5)

**Final run `2026-10-06_19-09`** (all fifteen patches: SEER2 converted to SEER1, backup
furnace priced, the June 2026 rebate rule for dual fuel, winter/summer peak names, the
new export columns). Stage A numbers (run `2026-10-06_18-19`, before the cost and rebate
fixes) are kept beside the final ones. Everything here is **PROVISIONAL**, not a
reference value.

Every count is given in representative dwelling units (rdu) and in homes. The 2025.1
weight, read from the files, is **253.90367272727272 homes per rdu** (2022.1.1:
242.131013), the same on every row.

## Three numbers to look at first

1. **Adoption in the exported case (heatingLCC_coolingLCC_unsub) is 3.27%** of the study
   sample (5,294 rdu = 1,344,166 homes), down from 11.19% in Stage A. The backup furnace
   (mean $4,112) and the SEER1 heat-pump price (+$754) together add about $4,866 to every
   home's capital cost; operating savings did not change.
2. **Mean backup furnace cost $4,112** (median $4,069, range $3,646-$6,567), priced with
   the REMDB gas furnace row at the backup's own size and AFUE 0.925 or 0.95 -- the
   session prompt's preview was $4,112.
3. **For a dual-fuel retrofit the June 2026 rebates now equal the 2024 rebates.** With
   R3, HEEHR has no fuel gate and HOMES is fuel-neutral -- exactly the 2024 rules -- so
   `_sub_june2026` and `_sub` give the same rebates (national totals differ by $424, a
   rounding rule kept per vintage) and the same adoption (29.04% for
   heatingLCC_coolingLCC). Under Stage A's rule June 2026 funded only electric baselines
   (5,732 rdu); now 121,432 rdu are funded. The Tepper files carry the June 2026 rebate.

## The files

Written by the main notebook's Tepper cell (cell 23), unchanged, to
`cmu_tare_model/output_results/tepper_export/` on the cloud machine. Not committed: the
repository is public, the national household files are about 400 MB and 700 MB, and
the session rules bar `output_results/` from commits. The commands at the end reproduce
them on the researcher's machine.

| File (final run `2026-10-06_19-09`) | Rows | Columns (after bldg_id) | Size | Stage A columns |
|---|---|---:|---:|---:|
| `tepper_household_mp5_National_...csv` | 161,983 rdu = 41,128,079 homes | 182 | 407.0 MB | 169 |
| `tepper_household_detailed_mp5_National_...csv` | 161,983 rdu = 41,128,079 homes | 332 | 704.8 MB | 319 |
| `tepper_county_mp5_National_...csv` | 2,973 counties | 11 | 0.3 MB | 11 |
| `tepper_household_mp5_Allegheny_...csv` | 1,091 rdu = 277,009 homes | 182 | 2.8 MB | 169 |
| `tepper_household_detailed_mp5_Allegheny_...csv` | 1,091 rdu = 277,009 homes | 332 | 4.8 MB | 319 |
| `tepper_county_mp5_Allegheny_...csv` | 1 county (G4200030) | 11 | 0.0 MB | 11 |

Against the 2022.1.1 export (167 / 317 columns after bldg_id): +2 for the 2025.1
backup-fan part, +8 dual-fuel ratings and backup size, +1 furnace cost, +4 panel columns.
The county file has 2,973 counties, against 3,079 for 2022.1.1: the dual-fuel package
applies only to homes with ducts and natural gas access.

## Checks run on the files

| Check | Final | Stage A |
|---|---|---|
| Household files: one row per study-sample rdu, bldg_id unique, all `include_sample` True, main and detailed copies hold the same homes | [OK] 161,983 / 1,091 rows | [OK] |
| County `home_count` = sum of household weights per county | [OK] all 2,973 counties, largest gap 6e-11 homes | [OK] |
| County `adoption_rate_pct` = household adopter share (heatingLCC_coolingLCC_unsub) | [OK] largest gap 0.005 points (file rounds to 2 decimals) | [OK] |
| NPV identity in the exported case: heating + cooling discounted savings - net capital = NPV | [OK] largest gap $0.005 (the NPV is rounded to cents) | [OK] |
| Adopter flag = NPV >= 0 in the exported case | [OK] every row | [OK] |
| Adoption and demand tables count the same homes (the notebook's own cell 23 check) | [OK] all 2,973 counties | [OK] |
| Runner checks on the run that wrote the files: NPV identity (9 cases), CLAUDE.md's per-home and county orderings (3 rebate scenarios), adopter = NPV >= 0, adopter non-blank count = 161,983 sample rdu, float64 adopters, no blank energy/cost/NPV in a sample home (38 columns, the furnace cost among them) | [OK] 20 of 20 | [OK] 20 of 20 |
| Blank values in household files | none | none |
| Blank values in the county file | `operating_cost_pct_change` in 1 county (G3500210, NM, 1 rdu): its one sample home uses no heating energy before the retrofit, so a percent change has no base. 1,159 sample rdu nationally (0.7%) have zero baseline heating energy. Same rule as 2022.1.1. | same |
| Peak columns | only the winter/summer names; no 2022.1.1 peak name in the files | 2022.1.1 names holding 2025.1 values |
| `county_fips` | written as a number, so one-digit-state FIPS lose the leading zero (Alabama `1117`), as in the 2022.1.1 export; `county` (GISJOIN) is the text key | same |

## Columns that are new or mean something different from the 2022.1.1 export

Written as entries that can be pasted into the data dictionary
(`cmu_tare_model/docs/tare_tepper_exports_data_dictionary.md`, not edited by this
session). `{mp}` is 5.

| Column | Entry |
|---|---|
| `weight` | Homes each row stands for: **253.90367272727272** in ResStock 2025.1 (242.131013 in 2022.1.1), the same on every row. Multiply a row count by it, or sum it, for homes; averages and shares are unaffected. A count below about 254 is a count of rdu. |
| `upgrade_hvac_heating_efficiency` | The dual-fuel system as ResStock publishes it: `Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 92.5% AFUE NG, 35F switchover` (93,501 rdu) or the same with `95.0% AFUE` (68,482 rdu). The string writes "SEER", but the heat pump is rated on DOE's 2023 test (SEER2 15.2, HSPF2 7.8). |
| `upgrade_hp_seer2`, `upgrade_hp_hspf2` | **New.** The heat pump's ratings parsed from the string: 15.2 and 7.8 on every row. |
| `upgrade_hp_seer1`, `upgrade_hp_hspf1` | **New.** The same ratings on the older test: SEER1 = SEER2 / 0.95 = 16.0; HSPF1 = HSPF2 / 0.85 = 9.18. The heat-pump upgrade cost is priced at SEER1 16.0 (REMDB's heat-pump regression takes SEER1). HSPF1 is for the record only. Factors PROVISIONAL (not checked against DOE Appendix M1). |
| `upgrade_backup_fuel`, `upgrade_backup_afue`, `upgrade_switchover_f` | **New.** The backup furnace's fuel (`Natural Gas` on every row), AFUE as a fraction (0.925 or 0.95), and the outdoor temperature below which it heats instead of the heat pump (35 F). |
| `size_heat_pump_backup_primary_k_btu_h` | **New in the export.** The backup furnace's size from the retrofit run, kBtu/h (mean 63.2). ResStock sizes it again in the retrofit run; for boiler and baseboard homes it is about a third larger than the old system. The furnace cost is priced from it. |
| `mp{mp}_heating_backupFurnace_installed_cost_v4MID` | **New.** Installed cost of the new condensing gas furnace, 2025 dollars, REMDB v4 `furnaces_gas_furnace` row (same coefficients as the furnace replacement cost), at the backup's size and AFUE. Mean $4,112, median $4,069, range $3,646-$6,567; no blanks. Added to the heat-pump upgrade cost in total and net capital cost. Not part of the cost a rebate covers. |
| `mp{mp}_heating_upgrade_installed_cost_v4MID` | Same meaning (the heat pump alone). Priced at SEER1 16.0: mean $15,711 (Stage A, at 15.2 unconverted: $14,957). |
| `ref2025_mp{mp}_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | Same formula, now including the backup furnace for this package: heat pump + furnace - heating replacement - cooling replacement. |
| `mp{mp}_heating_rebate_amount_june2026_v4MID`, `mp{mp}_rebate_eligibility_june2026` | Same columns, but for a dual-fuel retrofit both June 2026 fuel gates are off: HEEHR at or below 150% AMI whatever the baseline fuel (D8), HOMES fuel-neutral above it (R3). The result equals the 2024 rule for this package. South Dakota still $0. 121,432 rdu receive a rebate (Stage A: 5,732, electric baselines only). |
| `mp{mp}_naturalGas_heating_hpBackup_consumption` | Natural gas burned by the backup furnace after the retrofit, kWh, base year 2025. Non-zero for 149,331 rdu; mean 12,351 kWh. About 81% of retrofit heating energy nationally (507,984 of 624,652 weighted GWh). Always 0 in 2022.1.1. Priced at the natural gas price and counted in retrofit emissions. |
| `mp{mp}_electricity_heating_hpBackupFans_consumption` | **New in 2025.1.** Fan electricity while the backup furnace heats, kWh, base year. Non-zero for 149,190 rdu; mean 330 kWh. |
| `base_electricity_heating_hpBackupFans_consumption` | **New in 2025.1.** The same part before the retrofit: 0 on every row (homes with a heat pump already are outside the sample). |
| `mp{mp}_electricity_heating_hpBackup_consumption` | Electric backup heat: 0 on every row for this package (its backup is gas). |
| `mp{mp}_heating_consumption` | The heat pump's own electricity only (mean 2,301 kWh), not the home's heating energy after the retrofit; that is this plus the fan, backup-fan and backup-gas parts (per-year totals in `ref2025_mp{mp}_YYYY_heating_consumption`). |
| `ref2025_mp{mp}_YYYY_heating_naturalGas_consumption` (detailed copy) | Retrofit natural gas by year, 2025-2039, degree-day adjusted -- most of the retrofit total. The gas/electric split is fixed at ResStock's base-year split (CLAUDE.md Limitation 10). Retrofit propane and fuel oil are 0: those homes switch to a gas backup. |
| `base_peak_electricity_winter_kw`, `base_peak_electricity_summer_kw`, `mp{mp}_peak_electricity_winter_kw`, `mp{mp}_peak_electricity_summer_kw`, `mp{mp}_peak_electricity_winter_kw_savings`, `mp{mp}_peak_electricity_summer_kw_savings` | **Renamed for 2025.1** (were `..._heating_kw` / `..._cooling_kw` in Stage A and in 2022.1.1). ResStock 2025.1's largest daily electric peak in the winter or summer months, whatever is running (`out.qoi.electricity.maximum_daily_peak_winter..kw` / `_summer`), and its baseline-minus-retrofit change. Not comparable with 2022.1.1's peak while heating or cooling runs. |
| `panel_service_rating_amps` | **New, 2025.1 only.** The home's main electric panel rating, amps. |
| `mp{mp}_panel_constraint_overall`, `mp{mp}_panel_constraint_capacity`, `mp{mp}_panel_constraint_breaker_space` | **New, 2025.1 only.** Whether the retrofit runs into the panel's capacity or breaker space under the 2023 NEC existing-dwelling load calculation. Overall: No Constraint 138,809 rdu; Capacity Constrained Only 16,877; Space Constrained Only 5,154; Capacity and Space Constrained 1,143. Reporting only: no panel cost is modeled. |
| `hvac_has_ducts` | `Yes` on every row: ResStock applies the package only to ducted homes with gas access. |
| `upgrade_hvac_cooling_efficiency` | `Ducted Heat Pump` on every row (2022.1.1 wrote the heat pump's rating). |

## Headline numbers

| | Final | Stage A |
|---|---:|---:|
| Study sample | 161,983 rdu = 41,128,079 homes | same |
| Heat-pump rating fed to the cost regression | SEER1 16.0 | 15.2 (SEER2, unconverted) |
| Mean heat-pump upgrade cost | $15,711 | $14,957 |
| Mean backup furnace cost | $4,112 | not priced |
| Mean heating / cooling replacement cost | $3,742 / $5,843 | same |
| Mean lifetime fuel savings, heating / cooling | $1,072 / $1,353 | same |
| Mean NPV, heatingLCC_coolingLCC_unsub (exported) | -$8,703 | -$3,837 |
| Adoption, heatingLCC_coolingLCC_unsub (exported) | **3.27%** (5,294 rdu = 1,344,166 homes) | 11.19% (18,126 rdu = 4,602,258 homes) |
| Adoption, heatingLCC_coolingLCC_sub / _sub_june2026 | 29.04% / 29.04% | 60.23% / 12.42% |
| June 2026 rebate recipients | 121,432 rdu (fossil baselines included) | 5,732 rdu (electric only) |
| Allegheny County (G4200030): adoption, exported case | 0.37% of 1,091 rdu | 4.49% |
| Allegheny: mean heat pump / furnace cost; mean NPV | $14,415 / $4,237; -$10,790 | $13,661 / --; -$5,800 |

The county demand columns (electricity and site energy change) are the same in both runs:
they are ResStock's energy, which no fix touched.

## Every column of every file (final run 2026-10-06_19-09)

For each numeric column: blanks, minimum, median, mean and maximum. For each text column: its values with counts, or the number of distinct values when there are more than 12. Booleans show their counts. (The Stage A files have the same layout without the 13 new columns, and the peak columns under their 2022.1.1 names.)

#### `tepper_household_mp5_National_2026-10-06_19-09.csv`

- Rows: 161,983 rdu = 41,128,079 homes; columns: 183 (bldg_id first); size 407.0 MB

| # | Column | Type | Blanks | Min | Median | Mean | Max / values |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `bldg_id` | num | 0 | 11 | 2.742e+05 | 2.747e+05 | 5.5e+05 |
| 2 | `weight` | num | 0 | 253.9 | 253.9 | 253.9 | 253.9 |
| 3 | `state` | text | 0 | | | | 49 distinct |
| 4 | `county` | text | 0 | | | | 2,973 distinct |
| 5 | `county_fips` | num | 0 | 1,001 | 2.705e+04 | 2.794e+04 | 5.604e+04 |
| 6 | `puma` | text | 0 | | | | 2,334 distinct |
| 7 | `county_and_puma` | text | 0 | | | | 4,356 distinct |
| 8 | `census_region` | text | 0 | | | | Midwest: 54,160; South: 47,838; West: 38,135; Northeast: 21,850 |
| 9 | `census_division` | text | 0 | | | | East North Central: 37,901; Pacific: 23,741; South Atlantic: 20,003; West South Central: 19,692; Middle Atlantic: 17,889; West North Central: 16,259; Mountain: 14,394; East South Central: 8,143; New England: 3,961 |
| 10 | `census_division_recs` | text | 0 | | | | East North Central: 37,901; Pacific: 23,741; South Atlantic: 20,003; West South Central: 19,692; Middle Atlantic: 17,889; West North Central: 16,259; East South Central: 8,143; Mountain North: 8,113; Mountain South: 6,281; New England: 3,961 |
| 11 | `building_america_climate_zone` | text | 0 | | | | Cold: 67,370; Mixed-Humid: 46,084; Hot-Dry: 20,932; Hot-Humid: 17,203; Marine: 7,877; Mixed-Dry: 1,394; Very Cold: 1,123 |
| 12 | `reeds_balancing_area` | num | 0 | 1 | 80 | 73.27 | 134 |
| 13 | `city` | text | 0 | | | | 1,014 distinct |
| 14 | `urbanicity` | text | 0 | | | | Suburban: 118,570; Rural: 27,348; Urban: 16,065 |
| 15 | `weather_file_city` | text | 0 | | | | 1,016 distinct |
| 16 | `Longitude` | num | 0 | -124.2 | -88.7 | -93.03 | -68.03 |
| 17 | `Latitude` | num | 0 | 24.73 | 39.17 | 38.36 | 48.86 |
| 18 | `gea_region` | text | 0 | | | | 18 distinct |
| 19 | `square_footage` | num | 0 | 273 | 1,698 | 2,029 | 7,414 |
| 20 | `building_type` | text | 0 | | | | Single-Family Detached: 147,294; Single-Family Attached: 14,689 |
| 21 | `occupancy` | text | 0 | | | | 2: 58,992; 1: 32,238; 3: 27,406; 4: 24,349; 5: 11,581; 6: 4,595; 7: 1,689; 8: 649; 9: 246; 10+: 238 |
| 22 | `tenure` | text | 0 | | | | Owner: 133,189; Renter: 28,794 |
| 23 | `vacancy_status` | text | 0 | | | | Occupied: 161,983 |
| 24 | `vintage` | text | 0 | | | | 2000s: 24,134; 1950s: 22,992; 1990s: 21,546; <1940: 19,942; 1970s: 19,924; 1960s: 19,078; 1980s: 16,749; 1940s: 9,851; 2010s: 7,767 |
| 25 | `income` | num | 0 | 9,999 | 9e+04 | 9.284e+04 | 2e+05 |
| 26 | `federal_poverty_level` | text | 0 | | | | 400%+: 84,699; 200-300%: 23,221; 300-400%: 22,765; 0-100%: 11,796; 150-200%: 10,258; 100-150%: 9,244 |
| 27 | `household_income` | num | 0 | 1.282e+04 | 1.026e+05 | 1.19e+05 | 2.564e+05 |
| 28 | `census_area_medianIncome` | num | 0 | 2.277e+04 | 8.291e+04 | 8.729e+04 | 1.865e+05 |
| 29 | `income_level` | text | 0 | | | | Middle-to-Upper-Income: 62,772; Moderate-Income: 51,180; Low-Income: 48,031 |
| 30 | `percent_AMI` | num | 0 | 6.872 | 122.3 | 138 | 768.1 |
| 31 | `lmi_or_mui` | text | 0 | | | | LMI: 99,211; MUI: 62,772 |
| 32 | `base_heating_fuel` | text | 0 | | | | Natural Gas: 151,359; Electricity: 8,778; Fuel Oil: 1,423; Propane: 423 |
| 33 | `heating_type` | text | 0 | | | | Natural Gas Fuel Furnace: 148,997; Electricity Electric Furnace: 7,355; Natural Gas Fuel Boiler: 2,362; Electricity Baseboard: 1,375; Fuel Oil Fuel Furnace: 1,270; Propane Fuel Furnace: 410; Fuel Oil Fuel Boiler: 153; Electricity Electric Boiler: 48; Propane Fuel Boiler: 13 |
| 34 | `base_heating_efficiency` | text | 0 | | | | Fuel Furnace, 80% AFUE: 82,749; Fuel Furnace, 92.5% AFUE: 57,018; Fuel Furnace, 76% AFUE: 9,851; Electric Furnace, 100% AFUE: 7,355; Fuel Boiler, 80% AFUE: 2,126; Electric Baseboard, 100% Efficiency: 1,375; Fuel Furnace, 60% AFUE: 1,059; Fuel Boiler, 76% AFUE: 352; Fuel Boiler, 90% AFUE: 50; Electric Boiler, 100% AFUE: 48 |
| 35 | `base_cooling_fuel` | text | 0 | | | | Electricity: 161,983 |
| 36 | `cooling_type` | text | 0 | | | | Central AC: 145,871; Room AC: 16,112 |
| 37 | `base_cooling_efficiency` | text | 0 | | | | AC, SEER 13: 85,792; AC, SEER 15: 39,540; AC, SEER 10: 17,878; Room AC, EER 10.7: 7,741; Room AC, EER 12.0: 6,043; AC, SEER 8: 2,661; Room AC, EER 9.8: 2,003; Room AC, EER 8.5: 325 |
| 38 | `fuel_type_heating` | text | 0 | | | | naturalGas: 151,359; electricity: 8,778; fuelOil: 1,423; propane: 423 |
| 39 | `fuel_type_cooling` | text | 0 | | | | electricity: 161,983 |
| 40 | `hvac_has_ducts` | text | 0 | | | | Yes: 161,983 |
| 41 | `hvac_heating_type_and_fuel` | text | 0 | | | | Natural Gas Fuel Furnace: 148,997; Electricity Electric Furnace: 7,355; Natural Gas Fuel Boiler: 2,362; Electricity Baseboard: 1,375; Fuel Oil Fuel Furnace: 1,270; Propane Fuel Furnace: 410; Fuel Oil Fuel Boiler: 153; Electricity Electric Boiler: 48; Propane Fuel Boiler: 13 |
| 42 | `hvac_heating_efficiency` | text | 0 | | | | Fuel Furnace, 80% AFUE: 82,749; Fuel Furnace, 92.5% AFUE: 57,018; Fuel Furnace, 76% AFUE: 9,851; Electric Furnace, 100% AFUE: 7,355; Fuel Boiler, 80% AFUE: 2,126; Electric Baseboard, 100% Efficiency: 1,375; Fuel Furnace, 60% AFUE: 1,059; Fuel Boiler, 76% AFUE: 352; Fuel Boiler, 90% AFUE: 50; Electric Boiler, 100% AFUE: 48 |
| 43 | `size_heating_system_primary_k_btu_h` | num | 0 | 4.37 | 35.79 | 40.68 | 611.2 |
| 44 | `hvac_cooling_type` | text | 0 | | | | Central AC: 145,871; Room AC: 16,112 |
| 45 | `hvac_cooling_efficiency` | text | 0 | | | | AC, SEER 13: 85,792; AC, SEER 15: 39,540; AC, SEER 10: 17,878; Room AC, EER 10.7: 7,741; Room AC, EER 12.0: 6,043; AC, SEER 8: 2,661; Room AC, EER 9.8: 2,003; Room AC, EER 8.5: 325 |
| 46 | `size_cooling_system_primary_k_btu_h` | num | 0 | 4.37 | 35.79 | 40.68 | 611.2 |
| 47 | `upgrade_hvac_heating_efficiency` | text | 0 | | | | Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 92.5% AFUE NG, 35F switchover: 93,501; Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 95.0% AFUE NG, 35F switchover: 68,482 |
| 48 | `upgrade_hvac_cooling_efficiency` | text | 0 | | | | Ducted Heat Pump: 161,983 |
| 49 | `upgrade_hp_seer2` | num | 0 | 15.2 | 15.2 | 15.2 | 15.2 |
| 50 | `upgrade_hp_seer1` | num | 0 | 16 | 16 | 16 | 16 |
| 51 | `upgrade_hp_hspf2` | num | 0 | 7.8 | 7.8 | 7.8 | 7.8 |
| 52 | `upgrade_hp_hspf1` | num | 0 | 9.176 | 9.176 | 9.176 | 9.176 |
| 53 | `upgrade_backup_fuel` | text | 0 | | | | Natural Gas: 161,983 |
| 54 | `upgrade_backup_afue` | num | 0 | 0.925 | 0.925 | 0.9356 | 0.95 |
| 55 | `upgrade_switchover_f` | num | 0 | 35 | 35 | 35 | 35 |
| 56 | `size_heat_pump_backup_primary_k_btu_h` | num | 0 | 1.96 | 56.59 | 63.24 | 413.7 |
| 57 | `include_sample` | bool | 0 | | | | {True: 161983} |
| 58 | `include_heating` | bool | 0 | | | | {True: 161983} |
| 59 | `include_cooling` | bool | 0 | | | | {True: 161983} |
| 60 | `base_peak_electricity_summer_kw` | num | 0 | 0.612 | 8.068 | 8.835 | 78.21 |
| 61 | `base_peak_electricity_winter_kw` | num | 0 | 0.6365 | 6.606 | 7.715 | 77.85 |
| 62 | `base_peak_load_cooling_kbtu_hr` | num | 0 | 0 | 26.65 | 29.8 | 359.5 |
| 63 | `base_peak_load_heating_kbtu_hr` | num | 0 | 0 | 55.57 | 61.55 | 410.2 |
| 64 | `mp5_peak_electricity_summer_kw` | num | 0 | 0.9379 | 7.987 | 8.727 | 78.41 |
| 65 | `mp5_peak_electricity_winter_kw` | num | 0 | 0.7904 | 7.883 | 8.605 | 80.66 |
| 66 | `mp5_peak_electricity_summer_kw_savings` | num | 0 | -18.15 | 0.1178 | 0.1085 | 36.57 |
| 67 | `mp5_peak_electricity_winter_kw_savings` | num | 0 | -25.62 | -1.158 | -0.8902 | 57.98 |
| 68 | `mp5_peak_load_cooling_kbtu_hr` | num | 0 | 0 | 30.44 | 33.62 | 357.5 |
| 69 | `mp5_peak_load_heating_kbtu_hr` | num | 0 | 0 | 56.23 | 62 | 400.3 |
| 70 | `mp5_peak_load_cooling_kbtu_hr_savings` | num | 0 | -206.2 | -1.609 | -3.823 | 166.8 |
| 71 | `mp5_peak_load_heating_kbtu_hr_savings` | num | 0 | -143.5 | 0.231 | -0.4499 | 131.5 |
| 72 | `panel_service_rating_amps` | num | 0 | 60 | 150 | 154.3 | 400 |
| 73 | `mp5_panel_constraint_overall` | text | 0 | | | | No Constraint: 138,809; Capacity Constrained Only: 16,877; Space Constrained Only: 5,154; Capacity and Space Constrained: 1,143 |
| 74 | `mp5_panel_constraint_capacity` | bool | 0 | | | | {False: 143963, True: 18020} |
| 75 | `mp5_panel_constraint_breaker_space` | bool | 0 | | | | {False: 155686, True: 6297} |
| 76 | `base_electricity_heating_consumption` | num | 0 | 0 | 0 | 460.8 | 1.028e+05 |
| 77 | `base_electricity_cooling_consumption` | num | 0 | 0 | 3,032 | 3,791 | 5.26e+04 |
| 78 | `base_fuelOil_heating_consumption` | num | 0 | 0 | 0 | 305 | 1.49e+05 |
| 79 | `base_naturalGas_heating_consumption` | num | 0 | 0 | 1.642e+04 | 2.016e+04 | 2.46e+05 |
| 80 | `base_propane_heating_consumption` | num | 0 | 0 | 0 | 61.79 | 1.199e+05 |
| 81 | `baseline_heating_consumption` | num | 0 | 0 | 1.718e+04 | 2.099e+04 | 2.46e+05 |
| 82 | `baseline_cooling_consumption` | num | 0 | 0 | 3,032 | 3,791 | 5.26e+04 |
| 83 | `mp5_heating_consumption` | num | 0 | 0 | 1,935 | 2,301 | 2.666e+04 |
| 84 | `mp5_cooling_consumption` | num | 0 | 0 | 2,636 | 3,217 | 2.754e+04 |
| 85 | `base_total_electricity_consumption` | num | 0 | 1,657 | 1.109e+04 | 1.237e+04 | 1.151e+05 |
| 86 | `mp5_total_electricity_consumption` | num | 0 | 1,914 | 1.3e+04 | 1.379e+04 | 5.811e+04 |
| 87 | `baseline_total_site_consumption` | num | 0 | 4,177 | 3.376e+04 | 3.758e+04 | 2.726e+05 |
| 88 | `base_electricity_heating_fansPumps_consumption` | num | 0 | 0 | 343.8 | 457 | 6,300 |
| 89 | `base_electricity_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 90 | `base_electricity_heating_hpBackupFans_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 91 | `base_naturalGas_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 92 | `base_propane_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 93 | `base_fuelOil_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 94 | `base_electricity_cooling_fansPumps_consumption` | num | 0 | 0 | 430.5 | 618.4 | 8,038 |
| 95 | `mp5_electricity_heating_fansPumps_consumption` | num | 0 | 0 | 170.9 | 206 | 2,521 |
| 96 | `mp5_electricity_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 97 | `mp5_electricity_heating_hpBackupFans_consumption` | num | 0 | 0 | 274.3 | 329.6 | 4,466 |
| 98 | `mp5_naturalGas_heating_hpBackup_consumption` | num | 0 | 0 | 8,864 | 1.235e+04 | 1.981e+05 |
| 99 | `mp5_propane_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 100 | `mp5_fuelOil_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 101 | `mp5_electricity_cooling_fansPumps_consumption` | num | 0 | 0 | 590.8 | 700.9 | 6,002 |
| 102 | `baseline_2025_heating_consumption` | num | 0 | 0 | 1.753e+04 | 2.145e+04 | 2.521e+05 |
| 103 | `baseline_2026_heating_consumption` | num | 0 | 0 | 1.713e+04 | 2.101e+04 | 2.519e+05 |
| 104 | `baseline_2027_heating_consumption` | num | 0 | 0 | 1.707e+04 | 2.094e+04 | 2.515e+05 |
| 105 | `baseline_2028_heating_consumption` | num | 0 | 0 | 1.7e+04 | 2.086e+04 | 2.51e+05 |
| 106 | `baseline_2029_heating_consumption` | num | 0 | 0 | 1.693e+04 | 2.079e+04 | 2.505e+05 |
| 107 | `baseline_2030_heating_consumption` | num | 0 | 0 | 1.686e+04 | 2.071e+04 | 2.5e+05 |
| 108 | `baseline_2031_heating_consumption` | num | 0 | 0 | 1.679e+04 | 2.064e+04 | 2.495e+05 |
| 109 | `baseline_2032_heating_consumption` | num | 0 | 0 | 1.672e+04 | 2.056e+04 | 2.49e+05 |
| 110 | `baseline_2033_heating_consumption` | num | 0 | 0 | 1.665e+04 | 2.049e+04 | 2.485e+05 |
| 111 | `baseline_2034_heating_consumption` | num | 0 | 0 | 1.658e+04 | 2.041e+04 | 2.48e+05 |
| 112 | `baseline_2035_heating_consumption` | num | 0 | 0 | 1.651e+04 | 2.034e+04 | 2.475e+05 |
| 113 | `baseline_2036_heating_consumption` | num | 0 | 0 | 1.644e+04 | 2.026e+04 | 2.47e+05 |
| 114 | `baseline_2037_heating_consumption` | num | 0 | 0 | 1.637e+04 | 2.018e+04 | 2.465e+05 |
| 115 | `baseline_2038_heating_consumption` | num | 0 | 0 | 1.629e+04 | 2.011e+04 | 2.46e+05 |
| 116 | `baseline_2039_heating_consumption` | num | 0 | 0 | 1.623e+04 | 2.004e+04 | 2.455e+05 |
| 117 | `ref2025_mp5_2025_heating_consumption` | num | 0 | 0 | 1.159e+04 | 1.519e+04 | 2.097e+05 |
| 118 | `ref2025_mp5_2026_heating_consumption` | num | 0 | 0 | 1.13e+04 | 1.488e+04 | 2.096e+05 |
| 119 | `ref2025_mp5_2027_heating_consumption` | num | 0 | 0 | 1.126e+04 | 1.483e+04 | 2.092e+05 |
| 120 | `ref2025_mp5_2028_heating_consumption` | num | 0 | 0 | 1.121e+04 | 1.478e+04 | 2.088e+05 |
| 121 | `ref2025_mp5_2029_heating_consumption` | num | 0 | 0 | 1.116e+04 | 1.473e+04 | 2.084e+05 |
| 122 | `ref2025_mp5_2030_heating_consumption` | num | 0 | 0 | 1.112e+04 | 1.468e+04 | 2.08e+05 |
| 123 | `ref2025_mp5_2031_heating_consumption` | num | 0 | 0 | 1.107e+04 | 1.463e+04 | 2.076e+05 |
| 124 | `ref2025_mp5_2032_heating_consumption` | num | 0 | 0 | 1.102e+04 | 1.457e+04 | 2.072e+05 |
| 125 | `ref2025_mp5_2033_heating_consumption` | num | 0 | 0 | 1.098e+04 | 1.452e+04 | 2.067e+05 |
| 126 | `ref2025_mp5_2034_heating_consumption` | num | 0 | 0 | 1.093e+04 | 1.447e+04 | 2.063e+05 |
| 127 | `ref2025_mp5_2035_heating_consumption` | num | 0 | 0 | 1.088e+04 | 1.442e+04 | 2.059e+05 |
| 128 | `ref2025_mp5_2036_heating_consumption` | num | 0 | 0 | 1.083e+04 | 1.437e+04 | 2.055e+05 |
| 129 | `ref2025_mp5_2037_heating_consumption` | num | 0 | 0 | 1.078e+04 | 1.431e+04 | 2.051e+05 |
| 130 | `ref2025_mp5_2038_heating_consumption` | num | 0 | 0 | 1.074e+04 | 1.426e+04 | 2.046e+05 |
| 131 | `ref2025_mp5_2039_heating_consumption` | num | 0 | 0 | 1.069e+04 | 1.421e+04 | 2.042e+05 |
| 132 | `baseline_2025_cooling_consumption` | num | 0 | 0 | 3,465 | 4,410 | 5.773e+04 |
| 133 | `baseline_2026_cooling_consumption` | num | 0 | 0 | 3,636 | 4,603 | 6.127e+04 |
| 134 | `baseline_2027_cooling_consumption` | num | 0 | 0 | 3,660 | 4,631 | 6.162e+04 |
| 135 | `baseline_2028_cooling_consumption` | num | 0 | 0 | 3,684 | 4,660 | 6.2e+04 |
| 136 | `baseline_2029_cooling_consumption` | num | 0 | 0 | 3,706 | 4,688 | 6.234e+04 |
| 137 | `baseline_2030_cooling_consumption` | num | 0 | 0 | 3,730 | 4,716 | 6.268e+04 |
| 138 | `baseline_2031_cooling_consumption` | num | 0 | 0 | 3,754 | 4,745 | 6.307e+04 |
| 139 | `baseline_2032_cooling_consumption` | num | 0 | 0 | 3,779 | 4,775 | 6.341e+04 |
| 140 | `baseline_2033_cooling_consumption` | num | 0 | 0 | 3,801 | 4,802 | 6.375e+04 |
| 141 | `baseline_2034_cooling_consumption` | num | 0 | 0 | 3,825 | 4,831 | 6.409e+04 |
| 142 | `baseline_2035_cooling_consumption` | num | 0 | 0 | 3,849 | 4,860 | 6.444e+04 |
| 143 | `baseline_2036_cooling_consumption` | num | 0 | 0 | 3,875 | 4,889 | 6.482e+04 |
| 144 | `baseline_2037_cooling_consumption` | num | 0 | 0 | 3,898 | 4,918 | 6.516e+04 |
| 145 | `baseline_2038_cooling_consumption` | num | 0 | 0 | 3,922 | 4,947 | 6.55e+04 |
| 146 | `baseline_2039_cooling_consumption` | num | 0 | 0 | 3,947 | 4,975 | 6.585e+04 |
| 147 | `ref2025_mp5_2025_cooling_consumption` | num | 0 | 0 | 3,229 | 3,918 | 3.307e+04 |
| 148 | `ref2025_mp5_2026_cooling_consumption` | num | 0 | 0 | 3,390 | 4,092 | 3.461e+04 |
| 149 | `ref2025_mp5_2027_cooling_consumption` | num | 0 | 0 | 3,413 | 4,117 | 3.483e+04 |
| 150 | `ref2025_mp5_2028_cooling_consumption` | num | 0 | 0 | 3,435 | 4,143 | 3.506e+04 |
| 151 | `ref2025_mp5_2029_cooling_consumption` | num | 0 | 0 | 3,457 | 4,168 | 3.527e+04 |
| 152 | `ref2025_mp5_2030_cooling_consumption` | num | 0 | 0 | 3,479 | 4,194 | 3.55e+04 |
| 153 | `ref2025_mp5_2031_cooling_consumption` | num | 0 | 0 | 3,502 | 4,220 | 3.573e+04 |
| 154 | `ref2025_mp5_2032_cooling_consumption` | num | 0 | 0 | 3,525 | 4,246 | 3.597e+04 |
| 155 | `ref2025_mp5_2033_cooling_consumption` | num | 0 | 0 | 3,547 | 4,271 | 3.62e+04 |
| 156 | `ref2025_mp5_2034_cooling_consumption` | num | 0 | 0 | 3,569 | 4,297 | 3.643e+04 |
| 157 | `ref2025_mp5_2035_cooling_consumption` | num | 0 | 0 | 3,592 | 4,323 | 3.665e+04 |
| 158 | `ref2025_mp5_2036_cooling_consumption` | num | 0 | 0 | 3,615 | 4,349 | 3.688e+04 |
| 159 | `ref2025_mp5_2037_cooling_consumption` | num | 0 | 0 | 3,637 | 4,375 | 3.712e+04 |
| 160 | `ref2025_mp5_2038_cooling_consumption` | num | 0 | 0 | 3,661 | 4,401 | 3.735e+04 |
| 161 | `ref2025_mp5_2039_cooling_consumption` | num | 0 | 0 | 3,683 | 4,426 | 3.758e+04 |
| 162 | `baseline_heating_lifetime_fuel_cost` | num | 0 | 0 | 1.35e+04 | 1.651e+04 | 3.579e+05 |
| 163 | `ref2025_mp5_heating_lifetime_fuel_cost` | num | 0 | 0 | 1.265e+04 | 1.544e+04 | 1.72e+05 |
| 164 | `ref2025_mp5_heating_lifetime_savings_fuel_cost` | num | 0 | -3.88e+04 | 97.39 | 1,072 | 2.43e+05 |
| 165 | `baseline_cooling_lifetime_fuel_cost` | num | 0 | 0 | 1.004e+04 | 1.291e+04 | 1.649e+05 |
| 166 | `ref2025_mp5_cooling_lifetime_fuel_cost` | num | 0 | 0 | 9,404 | 1.156e+04 | 1.319e+05 |
| 167 | `ref2025_mp5_cooling_lifetime_savings_fuel_cost` | num | 0 | -8.913e+04 | 1,352 | 1,353 | 7.431e+04 |
| 168 | `ref2025_mp5_cooling_lifetime_savings_negative` | bool | 0 | | | | {False: 126808, True: 35175} |
| 169 | `mp5_heating_replacement_installed_cost_v4MID` | num | 0 | 270.8 | 3,656 | 3,742 | 2.326e+04 |
| 170 | `mp5_heating_upgrade_installed_cost_v4MID` | num | 0 | 1.06e+04 | 1.502e+04 | 1.571e+04 | 9.598e+04 |
| 171 | `mp5_heating_backupFurnace_installed_cost_v4MID` | num | 0 | 3,646 | 4,069 | 4,112 | 6,567 |
| 172 | `mp5_cooling_replacement_installed_cost_v4MID` | num | 0 | 478 | 6,075 | 5,843 | 3.13e+04 |
| 173 | `mp5_cooling_replacement_credit_applied_v4MID` | num | 0 | 478 | 6,075 | 5,843 | 3.13e+04 |
| 174 | `mp5_heating_rebate_amount_june2026_v4MID` | num | 0 | 0 | 6,957 | 4,959 | 8,000 |
| 175 | `mp5_rebate_eligibility_june2026` | text | 0 | | | | HEEHR: 98,895; Not Eligible: 40,551; HOMES: 22,537 |
| 176 | `mp5_modeled_savings_frac` | num | 0 | -0.9813 | 0.1676 | 0.1692 | 0.7347 |
| 177 | `public_discount_rate` | num | 0 | 0.02 | 0.02 | 0.02 | 0.02 |
| 178 | `private_discount_rate_fixed_base` | num | 0 | 0.07 | 0.07 | 0.07 | 0.07 |
| 179 | `ref2025_mp5_heating_discounted_lifetime_savings_fixed_base` | num | 0 | -2.523e+04 | 46.52 | 667.8 | 1.587e+05 |
| 180 | `ref2025_mp5_cooling_discounted_lifetime_savings_fixed_base` | num | 0 | -5.704e+04 | 868.4 | 867.4 | 4.756e+04 |
| 181 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | num | 0 | -3,632 | 9,549 | 1.024e+04 | 6.47e+04 |
| 182 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_private_npv_fixed_base` | num | 0 | -7.816e+04 | -7,941 | -8,703 | 1.577e+05 |
| 183 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base` | num | 0 | 0 | 0 | 0.03268 | 1 |

#### `tepper_household_detailed_mp5_National_2026-10-06_19-09.csv`

- Rows: 161,983 rdu = 41,128,079 homes; columns: 333 (bldg_id first); size 704.8 MB

| # | Column | Type | Blanks | Min | Median | Mean | Max / values |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `bldg_id` | num | 0 | 11 | 2.742e+05 | 2.747e+05 | 5.5e+05 |
| 2 | `weight` | num | 0 | 253.9 | 253.9 | 253.9 | 253.9 |
| 3 | `state` | text | 0 | | | | 49 distinct |
| 4 | `county` | text | 0 | | | | 2,973 distinct |
| 5 | `county_fips` | num | 0 | 1,001 | 2.705e+04 | 2.794e+04 | 5.604e+04 |
| 6 | `puma` | text | 0 | | | | 2,334 distinct |
| 7 | `county_and_puma` | text | 0 | | | | 4,356 distinct |
| 8 | `census_region` | text | 0 | | | | Midwest: 54,160; South: 47,838; West: 38,135; Northeast: 21,850 |
| 9 | `census_division` | text | 0 | | | | East North Central: 37,901; Pacific: 23,741; South Atlantic: 20,003; West South Central: 19,692; Middle Atlantic: 17,889; West North Central: 16,259; Mountain: 14,394; East South Central: 8,143; New England: 3,961 |
| 10 | `census_division_recs` | text | 0 | | | | East North Central: 37,901; Pacific: 23,741; South Atlantic: 20,003; West South Central: 19,692; Middle Atlantic: 17,889; West North Central: 16,259; East South Central: 8,143; Mountain North: 8,113; Mountain South: 6,281; New England: 3,961 |
| 11 | `building_america_climate_zone` | text | 0 | | | | Cold: 67,370; Mixed-Humid: 46,084; Hot-Dry: 20,932; Hot-Humid: 17,203; Marine: 7,877; Mixed-Dry: 1,394; Very Cold: 1,123 |
| 12 | `reeds_balancing_area` | num | 0 | 1 | 80 | 73.27 | 134 |
| 13 | `city` | text | 0 | | | | 1,014 distinct |
| 14 | `urbanicity` | text | 0 | | | | Suburban: 118,570; Rural: 27,348; Urban: 16,065 |
| 15 | `weather_file_city` | text | 0 | | | | 1,016 distinct |
| 16 | `Longitude` | num | 0 | -124.2 | -88.7 | -93.03 | -68.03 |
| 17 | `Latitude` | num | 0 | 24.73 | 39.17 | 38.36 | 48.86 |
| 18 | `gea_region` | text | 0 | | | | 18 distinct |
| 19 | `square_footage` | num | 0 | 273 | 1,698 | 2,029 | 7,414 |
| 20 | `building_type` | text | 0 | | | | Single-Family Detached: 147,294; Single-Family Attached: 14,689 |
| 21 | `occupancy` | text | 0 | | | | 2: 58,992; 1: 32,238; 3: 27,406; 4: 24,349; 5: 11,581; 6: 4,595; 7: 1,689; 8: 649; 9: 246; 10+: 238 |
| 22 | `tenure` | text | 0 | | | | Owner: 133,189; Renter: 28,794 |
| 23 | `vacancy_status` | text | 0 | | | | Occupied: 161,983 |
| 24 | `vintage` | text | 0 | | | | 2000s: 24,134; 1950s: 22,992; 1990s: 21,546; <1940: 19,942; 1970s: 19,924; 1960s: 19,078; 1980s: 16,749; 1940s: 9,851; 2010s: 7,767 |
| 25 | `income` | num | 0 | 9,999 | 9e+04 | 9.284e+04 | 2e+05 |
| 26 | `federal_poverty_level` | text | 0 | | | | 400%+: 84,699; 200-300%: 23,221; 300-400%: 22,765; 0-100%: 11,796; 150-200%: 10,258; 100-150%: 9,244 |
| 27 | `household_income` | num | 0 | 1.282e+04 | 1.026e+05 | 1.19e+05 | 2.564e+05 |
| 28 | `census_area_medianIncome` | num | 0 | 2.277e+04 | 8.291e+04 | 8.729e+04 | 1.865e+05 |
| 29 | `income_level` | text | 0 | | | | Middle-to-Upper-Income: 62,772; Moderate-Income: 51,180; Low-Income: 48,031 |
| 30 | `percent_AMI` | num | 0 | 6.872 | 122.3 | 138 | 768.1 |
| 31 | `lmi_or_mui` | text | 0 | | | | LMI: 99,211; MUI: 62,772 |
| 32 | `base_heating_fuel` | text | 0 | | | | Natural Gas: 151,359; Electricity: 8,778; Fuel Oil: 1,423; Propane: 423 |
| 33 | `heating_type` | text | 0 | | | | Natural Gas Fuel Furnace: 148,997; Electricity Electric Furnace: 7,355; Natural Gas Fuel Boiler: 2,362; Electricity Baseboard: 1,375; Fuel Oil Fuel Furnace: 1,270; Propane Fuel Furnace: 410; Fuel Oil Fuel Boiler: 153; Electricity Electric Boiler: 48; Propane Fuel Boiler: 13 |
| 34 | `base_heating_efficiency` | text | 0 | | | | Fuel Furnace, 80% AFUE: 82,749; Fuel Furnace, 92.5% AFUE: 57,018; Fuel Furnace, 76% AFUE: 9,851; Electric Furnace, 100% AFUE: 7,355; Fuel Boiler, 80% AFUE: 2,126; Electric Baseboard, 100% Efficiency: 1,375; Fuel Furnace, 60% AFUE: 1,059; Fuel Boiler, 76% AFUE: 352; Fuel Boiler, 90% AFUE: 50; Electric Boiler, 100% AFUE: 48 |
| 35 | `base_cooling_fuel` | text | 0 | | | | Electricity: 161,983 |
| 36 | `cooling_type` | text | 0 | | | | Central AC: 145,871; Room AC: 16,112 |
| 37 | `base_cooling_efficiency` | text | 0 | | | | AC, SEER 13: 85,792; AC, SEER 15: 39,540; AC, SEER 10: 17,878; Room AC, EER 10.7: 7,741; Room AC, EER 12.0: 6,043; AC, SEER 8: 2,661; Room AC, EER 9.8: 2,003; Room AC, EER 8.5: 325 |
| 38 | `fuel_type_heating` | text | 0 | | | | naturalGas: 151,359; electricity: 8,778; fuelOil: 1,423; propane: 423 |
| 39 | `fuel_type_cooling` | text | 0 | | | | electricity: 161,983 |
| 40 | `hvac_has_ducts` | text | 0 | | | | Yes: 161,983 |
| 41 | `hvac_heating_type_and_fuel` | text | 0 | | | | Natural Gas Fuel Furnace: 148,997; Electricity Electric Furnace: 7,355; Natural Gas Fuel Boiler: 2,362; Electricity Baseboard: 1,375; Fuel Oil Fuel Furnace: 1,270; Propane Fuel Furnace: 410; Fuel Oil Fuel Boiler: 153; Electricity Electric Boiler: 48; Propane Fuel Boiler: 13 |
| 42 | `hvac_heating_efficiency` | text | 0 | | | | Fuel Furnace, 80% AFUE: 82,749; Fuel Furnace, 92.5% AFUE: 57,018; Fuel Furnace, 76% AFUE: 9,851; Electric Furnace, 100% AFUE: 7,355; Fuel Boiler, 80% AFUE: 2,126; Electric Baseboard, 100% Efficiency: 1,375; Fuel Furnace, 60% AFUE: 1,059; Fuel Boiler, 76% AFUE: 352; Fuel Boiler, 90% AFUE: 50; Electric Boiler, 100% AFUE: 48 |
| 43 | `size_heating_system_primary_k_btu_h` | num | 0 | 4.37 | 35.79 | 40.68 | 611.2 |
| 44 | `hvac_cooling_type` | text | 0 | | | | Central AC: 145,871; Room AC: 16,112 |
| 45 | `hvac_cooling_efficiency` | text | 0 | | | | AC, SEER 13: 85,792; AC, SEER 15: 39,540; AC, SEER 10: 17,878; Room AC, EER 10.7: 7,741; Room AC, EER 12.0: 6,043; AC, SEER 8: 2,661; Room AC, EER 9.8: 2,003; Room AC, EER 8.5: 325 |
| 46 | `size_cooling_system_primary_k_btu_h` | num | 0 | 4.37 | 35.79 | 40.68 | 611.2 |
| 47 | `upgrade_hvac_heating_efficiency` | text | 0 | | | | Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 92.5% AFUE NG, 35F switchover: 93,501; Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 95.0% AFUE NG, 35F switchover: 68,482 |
| 48 | `upgrade_hvac_cooling_efficiency` | text | 0 | | | | Ducted Heat Pump: 161,983 |
| 49 | `upgrade_hp_seer2` | num | 0 | 15.2 | 15.2 | 15.2 | 15.2 |
| 50 | `upgrade_hp_seer1` | num | 0 | 16 | 16 | 16 | 16 |
| 51 | `upgrade_hp_hspf2` | num | 0 | 7.8 | 7.8 | 7.8 | 7.8 |
| 52 | `upgrade_hp_hspf1` | num | 0 | 9.176 | 9.176 | 9.176 | 9.176 |
| 53 | `upgrade_backup_fuel` | text | 0 | | | | Natural Gas: 161,983 |
| 54 | `upgrade_backup_afue` | num | 0 | 0.925 | 0.925 | 0.9356 | 0.95 |
| 55 | `upgrade_switchover_f` | num | 0 | 35 | 35 | 35 | 35 |
| 56 | `size_heat_pump_backup_primary_k_btu_h` | num | 0 | 1.96 | 56.59 | 63.24 | 413.7 |
| 57 | `include_sample` | bool | 0 | | | | {True: 161983} |
| 58 | `include_heating` | bool | 0 | | | | {True: 161983} |
| 59 | `include_cooling` | bool | 0 | | | | {True: 161983} |
| 60 | `base_peak_electricity_summer_kw` | num | 0 | 0.612 | 8.068 | 8.835 | 78.21 |
| 61 | `base_peak_electricity_winter_kw` | num | 0 | 0.6365 | 6.606 | 7.715 | 77.85 |
| 62 | `base_peak_load_cooling_kbtu_hr` | num | 0 | 0 | 26.65 | 29.8 | 359.5 |
| 63 | `base_peak_load_heating_kbtu_hr` | num | 0 | 0 | 55.57 | 61.55 | 410.2 |
| 64 | `mp5_peak_electricity_summer_kw` | num | 0 | 0.9379 | 7.987 | 8.727 | 78.41 |
| 65 | `mp5_peak_electricity_winter_kw` | num | 0 | 0.7904 | 7.883 | 8.605 | 80.66 |
| 66 | `mp5_peak_electricity_summer_kw_savings` | num | 0 | -18.15 | 0.1178 | 0.1085 | 36.57 |
| 67 | `mp5_peak_electricity_winter_kw_savings` | num | 0 | -25.62 | -1.158 | -0.8902 | 57.98 |
| 68 | `mp5_peak_load_cooling_kbtu_hr` | num | 0 | 0 | 30.44 | 33.62 | 357.5 |
| 69 | `mp5_peak_load_heating_kbtu_hr` | num | 0 | 0 | 56.23 | 62 | 400.3 |
| 70 | `mp5_peak_load_cooling_kbtu_hr_savings` | num | 0 | -206.2 | -1.609 | -3.823 | 166.8 |
| 71 | `mp5_peak_load_heating_kbtu_hr_savings` | num | 0 | -143.5 | 0.231 | -0.4499 | 131.5 |
| 72 | `panel_service_rating_amps` | num | 0 | 60 | 150 | 154.3 | 400 |
| 73 | `mp5_panel_constraint_overall` | text | 0 | | | | No Constraint: 138,809; Capacity Constrained Only: 16,877; Space Constrained Only: 5,154; Capacity and Space Constrained: 1,143 |
| 74 | `mp5_panel_constraint_capacity` | bool | 0 | | | | {False: 143963, True: 18020} |
| 75 | `mp5_panel_constraint_breaker_space` | bool | 0 | | | | {False: 155686, True: 6297} |
| 76 | `base_electricity_heating_consumption` | num | 0 | 0 | 0 | 460.8 | 1.028e+05 |
| 77 | `base_electricity_cooling_consumption` | num | 0 | 0 | 3,032 | 3,791 | 5.26e+04 |
| 78 | `base_fuelOil_heating_consumption` | num | 0 | 0 | 0 | 305 | 1.49e+05 |
| 79 | `base_naturalGas_heating_consumption` | num | 0 | 0 | 1.642e+04 | 2.016e+04 | 2.46e+05 |
| 80 | `base_propane_heating_consumption` | num | 0 | 0 | 0 | 61.79 | 1.199e+05 |
| 81 | `baseline_heating_consumption` | num | 0 | 0 | 1.718e+04 | 2.099e+04 | 2.46e+05 |
| 82 | `baseline_cooling_consumption` | num | 0 | 0 | 3,032 | 3,791 | 5.26e+04 |
| 83 | `mp5_heating_consumption` | num | 0 | 0 | 1,935 | 2,301 | 2.666e+04 |
| 84 | `mp5_cooling_consumption` | num | 0 | 0 | 2,636 | 3,217 | 2.754e+04 |
| 85 | `base_total_electricity_consumption` | num | 0 | 1,657 | 1.109e+04 | 1.237e+04 | 1.151e+05 |
| 86 | `mp5_total_electricity_consumption` | num | 0 | 1,914 | 1.3e+04 | 1.379e+04 | 5.811e+04 |
| 87 | `baseline_total_site_consumption` | num | 0 | 4,177 | 3.376e+04 | 3.758e+04 | 2.726e+05 |
| 88 | `base_electricity_heating_fansPumps_consumption` | num | 0 | 0 | 343.8 | 457 | 6,300 |
| 89 | `base_electricity_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 90 | `base_electricity_heating_hpBackupFans_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 91 | `base_naturalGas_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 92 | `base_propane_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 93 | `base_fuelOil_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 94 | `base_electricity_cooling_fansPumps_consumption` | num | 0 | 0 | 430.5 | 618.4 | 8,038 |
| 95 | `mp5_electricity_heating_fansPumps_consumption` | num | 0 | 0 | 170.9 | 206 | 2,521 |
| 96 | `mp5_electricity_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 97 | `mp5_electricity_heating_hpBackupFans_consumption` | num | 0 | 0 | 274.3 | 329.6 | 4,466 |
| 98 | `mp5_naturalGas_heating_hpBackup_consumption` | num | 0 | 0 | 8,864 | 1.235e+04 | 1.981e+05 |
| 99 | `mp5_propane_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 100 | `mp5_fuelOil_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 101 | `mp5_electricity_cooling_fansPumps_consumption` | num | 0 | 0 | 590.8 | 700.9 | 6,002 |
| 102 | `baseline_2025_heating_consumption` | num | 0 | 0 | 1.753e+04 | 2.145e+04 | 2.521e+05 |
| 103 | `baseline_2026_heating_consumption` | num | 0 | 0 | 1.713e+04 | 2.101e+04 | 2.519e+05 |
| 104 | `baseline_2027_heating_consumption` | num | 0 | 0 | 1.707e+04 | 2.094e+04 | 2.515e+05 |
| 105 | `baseline_2028_heating_consumption` | num | 0 | 0 | 1.7e+04 | 2.086e+04 | 2.51e+05 |
| 106 | `baseline_2029_heating_consumption` | num | 0 | 0 | 1.693e+04 | 2.079e+04 | 2.505e+05 |
| 107 | `baseline_2030_heating_consumption` | num | 0 | 0 | 1.686e+04 | 2.071e+04 | 2.5e+05 |
| 108 | `baseline_2031_heating_consumption` | num | 0 | 0 | 1.679e+04 | 2.064e+04 | 2.495e+05 |
| 109 | `baseline_2032_heating_consumption` | num | 0 | 0 | 1.672e+04 | 2.056e+04 | 2.49e+05 |
| 110 | `baseline_2033_heating_consumption` | num | 0 | 0 | 1.665e+04 | 2.049e+04 | 2.485e+05 |
| 111 | `baseline_2034_heating_consumption` | num | 0 | 0 | 1.658e+04 | 2.041e+04 | 2.48e+05 |
| 112 | `baseline_2035_heating_consumption` | num | 0 | 0 | 1.651e+04 | 2.034e+04 | 2.475e+05 |
| 113 | `baseline_2036_heating_consumption` | num | 0 | 0 | 1.644e+04 | 2.026e+04 | 2.47e+05 |
| 114 | `baseline_2037_heating_consumption` | num | 0 | 0 | 1.637e+04 | 2.018e+04 | 2.465e+05 |
| 115 | `baseline_2038_heating_consumption` | num | 0 | 0 | 1.629e+04 | 2.011e+04 | 2.46e+05 |
| 116 | `baseline_2039_heating_consumption` | num | 0 | 0 | 1.623e+04 | 2.004e+04 | 2.455e+05 |
| 117 | `baseline_2025_heating_electricity_consumption` | num | 0 | 0 | 398.6 | 917.8 | 1.053e+05 |
| 118 | `baseline_2026_heating_electricity_consumption` | num | 0 | 0 | 390 | 899.2 | 1.052e+05 |
| 119 | `baseline_2027_heating_electricity_consumption` | num | 0 | 0 | 388.6 | 896 | 1.05e+05 |
| 120 | `baseline_2028_heating_electricity_consumption` | num | 0 | 0 | 387.2 | 892.7 | 1.048e+05 |
| 121 | `baseline_2029_heating_electricity_consumption` | num | 0 | 0 | 385.7 | 889.3 | 1.046e+05 |
| 122 | `baseline_2030_heating_electricity_consumption` | num | 0 | 0 | 384.1 | 886.1 | 1.044e+05 |
| 123 | `baseline_2031_heating_electricity_consumption` | num | 0 | 0 | 382.6 | 882.7 | 1.042e+05 |
| 124 | `baseline_2032_heating_electricity_consumption` | num | 0 | 0 | 381.1 | 879.3 | 1.04e+05 |
| 125 | `baseline_2033_heating_electricity_consumption` | num | 0 | 0 | 379.6 | 876 | 1.038e+05 |
| 126 | `baseline_2034_heating_electricity_consumption` | num | 0 | 0 | 378.2 | 872.6 | 1.036e+05 |
| 127 | `baseline_2035_heating_electricity_consumption` | num | 0 | 0 | 376.7 | 869.2 | 1.034e+05 |
| 128 | `baseline_2036_heating_electricity_consumption` | num | 0 | 0 | 375.4 | 865.9 | 1.032e+05 |
| 129 | `baseline_2037_heating_electricity_consumption` | num | 0 | 0 | 373.7 | 862.5 | 1.03e+05 |
| 130 | `baseline_2038_heating_electricity_consumption` | num | 0 | 0 | 372.3 | 859.2 | 1.028e+05 |
| 131 | `baseline_2039_heating_electricity_consumption` | num | 0 | 0 | 370.7 | 855.9 | 1.025e+05 |
| 132 | `baseline_2025_heating_naturalGas_consumption` | num | 0 | 0 | 1.642e+04 | 2.016e+04 | 2.46e+05 |
| 133 | `baseline_2026_heating_naturalGas_consumption` | num | 0 | 0 | 1.604e+04 | 1.975e+04 | 2.459e+05 |
| 134 | `baseline_2027_heating_naturalGas_consumption` | num | 0 | 0 | 1.598e+04 | 1.968e+04 | 2.454e+05 |
| 135 | `baseline_2028_heating_naturalGas_consumption` | num | 0 | 0 | 1.591e+04 | 1.961e+04 | 2.449e+05 |
| 136 | `baseline_2029_heating_naturalGas_consumption` | num | 0 | 0 | 1.585e+04 | 1.954e+04 | 2.445e+05 |
| 137 | `baseline_2030_heating_naturalGas_consumption` | num | 0 | 0 | 1.578e+04 | 1.947e+04 | 2.44e+05 |
| 138 | `baseline_2031_heating_naturalGas_consumption` | num | 0 | 0 | 1.572e+04 | 1.94e+04 | 2.435e+05 |
| 139 | `baseline_2032_heating_naturalGas_consumption` | num | 0 | 0 | 1.565e+04 | 1.933e+04 | 2.43e+05 |
| 140 | `baseline_2033_heating_naturalGas_consumption` | num | 0 | 0 | 1.559e+04 | 1.926e+04 | 2.425e+05 |
| 141 | `baseline_2034_heating_naturalGas_consumption` | num | 0 | 0 | 1.552e+04 | 1.919e+04 | 2.421e+05 |
| 142 | `baseline_2035_heating_naturalGas_consumption` | num | 0 | 0 | 1.545e+04 | 1.912e+04 | 2.416e+05 |
| 143 | `baseline_2036_heating_naturalGas_consumption` | num | 0 | 0 | 1.539e+04 | 1.905e+04 | 2.411e+05 |
| 144 | `baseline_2037_heating_naturalGas_consumption` | num | 0 | 0 | 1.533e+04 | 1.898e+04 | 2.406e+05 |
| 145 | `baseline_2038_heating_naturalGas_consumption` | num | 0 | 0 | 1.526e+04 | 1.891e+04 | 2.401e+05 |
| 146 | `baseline_2039_heating_naturalGas_consumption` | num | 0 | 0 | 1.519e+04 | 1.884e+04 | 2.396e+05 |
| 147 | `baseline_2025_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 305 | 1.49e+05 |
| 148 | `baseline_2026_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 297 | 1.453e+05 |
| 149 | `baseline_2027_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 295.8 | 1.447e+05 |
| 150 | `baseline_2028_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 294.6 | 1.442e+05 |
| 151 | `baseline_2029_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 293.4 | 1.436e+05 |
| 152 | `baseline_2030_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 292.1 | 1.43e+05 |
| 153 | `baseline_2031_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 290.9 | 1.424e+05 |
| 154 | `baseline_2032_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 289.7 | 1.418e+05 |
| 155 | `baseline_2033_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 288.5 | 1.412e+05 |
| 156 | `baseline_2034_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 287.3 | 1.406e+05 |
| 157 | `baseline_2035_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 286 | 1.4e+05 |
| 158 | `baseline_2036_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 284.8 | 1.394e+05 |
| 159 | `baseline_2037_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 283.6 | 1.388e+05 |
| 160 | `baseline_2038_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 282.4 | 1.383e+05 |
| 161 | `baseline_2039_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 281.2 | 1.377e+05 |
| 162 | `baseline_2025_heating_propane_consumption` | num | 0 | 0 | 0 | 61.79 | 1.199e+05 |
| 163 | `baseline_2026_heating_propane_consumption` | num | 0 | 0 | 0 | 61.05 | 1.198e+05 |
| 164 | `baseline_2027_heating_propane_consumption` | num | 0 | 0 | 0 | 60.87 | 1.196e+05 |
| 165 | `baseline_2028_heating_propane_consumption` | num | 0 | 0 | 0 | 60.68 | 1.194e+05 |
| 166 | `baseline_2029_heating_propane_consumption` | num | 0 | 0 | 0 | 60.49 | 1.191e+05 |
| 167 | `baseline_2030_heating_propane_consumption` | num | 0 | 0 | 0 | 60.31 | 1.189e+05 |
| 168 | `baseline_2031_heating_propane_consumption` | num | 0 | 0 | 0 | 60.12 | 1.187e+05 |
| 169 | `baseline_2032_heating_propane_consumption` | num | 0 | 0 | 0 | 59.93 | 1.184e+05 |
| 170 | `baseline_2033_heating_propane_consumption` | num | 0 | 0 | 0 | 59.74 | 1.182e+05 |
| 171 | `baseline_2034_heating_propane_consumption` | num | 0 | 0 | 0 | 59.55 | 1.18e+05 |
| 172 | `baseline_2035_heating_propane_consumption` | num | 0 | 0 | 0 | 59.36 | 1.177e+05 |
| 173 | `baseline_2036_heating_propane_consumption` | num | 0 | 0 | 0 | 59.17 | 1.175e+05 |
| 174 | `baseline_2037_heating_propane_consumption` | num | 0 | 0 | 0 | 58.98 | 1.172e+05 |
| 175 | `baseline_2038_heating_propane_consumption` | num | 0 | 0 | 0 | 58.79 | 1.17e+05 |
| 176 | `baseline_2039_heating_propane_consumption` | num | 0 | 0 | 0 | 58.6 | 1.168e+05 |
| 177 | `ref2025_mp5_2025_heating_consumption` | num | 0 | 0 | 1.159e+04 | 1.519e+04 | 2.097e+05 |
| 178 | `ref2025_mp5_2026_heating_consumption` | num | 0 | 0 | 1.13e+04 | 1.488e+04 | 2.096e+05 |
| 179 | `ref2025_mp5_2027_heating_consumption` | num | 0 | 0 | 1.126e+04 | 1.483e+04 | 2.092e+05 |
| 180 | `ref2025_mp5_2028_heating_consumption` | num | 0 | 0 | 1.121e+04 | 1.478e+04 | 2.088e+05 |
| 181 | `ref2025_mp5_2029_heating_consumption` | num | 0 | 0 | 1.116e+04 | 1.473e+04 | 2.084e+05 |
| 182 | `ref2025_mp5_2030_heating_consumption` | num | 0 | 0 | 1.112e+04 | 1.468e+04 | 2.08e+05 |
| 183 | `ref2025_mp5_2031_heating_consumption` | num | 0 | 0 | 1.107e+04 | 1.463e+04 | 2.076e+05 |
| 184 | `ref2025_mp5_2032_heating_consumption` | num | 0 | 0 | 1.102e+04 | 1.457e+04 | 2.072e+05 |
| 185 | `ref2025_mp5_2033_heating_consumption` | num | 0 | 0 | 1.098e+04 | 1.452e+04 | 2.067e+05 |
| 186 | `ref2025_mp5_2034_heating_consumption` | num | 0 | 0 | 1.093e+04 | 1.447e+04 | 2.063e+05 |
| 187 | `ref2025_mp5_2035_heating_consumption` | num | 0 | 0 | 1.088e+04 | 1.442e+04 | 2.059e+05 |
| 188 | `ref2025_mp5_2036_heating_consumption` | num | 0 | 0 | 1.083e+04 | 1.437e+04 | 2.055e+05 |
| 189 | `ref2025_mp5_2037_heating_consumption` | num | 0 | 0 | 1.078e+04 | 1.431e+04 | 2.051e+05 |
| 190 | `ref2025_mp5_2038_heating_consumption` | num | 0 | 0 | 1.074e+04 | 1.426e+04 | 2.046e+05 |
| 191 | `ref2025_mp5_2039_heating_consumption` | num | 0 | 0 | 1.069e+04 | 1.421e+04 | 2.042e+05 |
| 192 | `ref2025_mp5_2025_heating_electricity_consumption` | num | 0 | 0 | 2,444 | 2,837 | 3.03e+04 |
| 193 | `ref2025_mp5_2026_heating_electricity_consumption` | num | 0 | 0 | 2,391 | 2,776 | 2.956e+04 |
| 194 | `ref2025_mp5_2027_heating_electricity_consumption` | num | 0 | 0 | 2,383 | 2,766 | 2.943e+04 |
| 195 | `ref2025_mp5_2028_heating_electricity_consumption` | num | 0 | 0 | 2,373 | 2,755 | 2.932e+04 |
| 196 | `ref2025_mp5_2029_heating_electricity_consumption` | num | 0 | 0 | 2,364 | 2,745 | 2.92e+04 |
| 197 | `ref2025_mp5_2030_heating_electricity_consumption` | num | 0 | 0 | 2,356 | 2,734 | 2.908e+04 |
| 198 | `ref2025_mp5_2031_heating_electricity_consumption` | num | 0 | 0 | 2,346 | 2,724 | 2.896e+04 |
| 199 | `ref2025_mp5_2032_heating_electricity_consumption` | num | 0 | 0 | 2,337 | 2,713 | 2.883e+04 |
| 200 | `ref2025_mp5_2033_heating_electricity_consumption` | num | 0 | 0 | 2,328 | 2,703 | 2.872e+04 |
| 201 | `ref2025_mp5_2034_heating_electricity_consumption` | num | 0 | 0 | 2,319 | 2,692 | 2.86e+04 |
| 202 | `ref2025_mp5_2035_heating_electricity_consumption` | num | 0 | 0 | 2,309 | 2,682 | 2.848e+04 |
| 203 | `ref2025_mp5_2036_heating_electricity_consumption` | num | 0 | 0 | 2,300 | 2,671 | 2.836e+04 |
| 204 | `ref2025_mp5_2037_heating_electricity_consumption` | num | 0 | 0 | 2,291 | 2,661 | 2.823e+04 |
| 205 | `ref2025_mp5_2038_heating_electricity_consumption` | num | 0 | 0 | 2,282 | 2,650 | 2.812e+04 |
| 206 | `ref2025_mp5_2039_heating_electricity_consumption` | num | 0 | 0 | 2,272 | 2,640 | 2.8e+04 |
| 207 | `ref2025_mp5_2025_heating_naturalGas_consumption` | num | 0 | 0 | 8,864 | 1.235e+04 | 1.981e+05 |
| 208 | `ref2025_mp5_2026_heating_naturalGas_consumption` | num | 0 | 0 | 8,634 | 1.211e+04 | 1.98e+05 |
| 209 | `ref2025_mp5_2027_heating_naturalGas_consumption` | num | 0 | 0 | 8,600 | 1.207e+04 | 1.976e+05 |
| 210 | `ref2025_mp5_2028_heating_naturalGas_consumption` | num | 0 | 0 | 8,564 | 1.203e+04 | 1.972e+05 |
| 211 | `ref2025_mp5_2029_heating_naturalGas_consumption` | num | 0 | 0 | 8,526 | 1.199e+04 | 1.968e+05 |
| 212 | `ref2025_mp5_2030_heating_naturalGas_consumption` | num | 0 | 0 | 8,490 | 1.194e+04 | 1.965e+05 |
| 213 | `ref2025_mp5_2031_heating_naturalGas_consumption` | num | 0 | 0 | 8,451 | 1.19e+04 | 1.961e+05 |
| 214 | `ref2025_mp5_2032_heating_naturalGas_consumption` | num | 0 | 0 | 8,413 | 1.186e+04 | 1.957e+05 |
| 215 | `ref2025_mp5_2033_heating_naturalGas_consumption` | num | 0 | 0 | 8,376 | 1.182e+04 | 1.953e+05 |
| 216 | `ref2025_mp5_2034_heating_naturalGas_consumption` | num | 0 | 0 | 8,338 | 1.178e+04 | 1.949e+05 |
| 217 | `ref2025_mp5_2035_heating_naturalGas_consumption` | num | 0 | 0 | 8,303 | 1.174e+04 | 1.945e+05 |
| 218 | `ref2025_mp5_2036_heating_naturalGas_consumption` | num | 0 | 0 | 8,267 | 1.17e+04 | 1.941e+05 |
| 219 | `ref2025_mp5_2037_heating_naturalGas_consumption` | num | 0 | 0 | 8,231 | 1.165e+04 | 1.937e+05 |
| 220 | `ref2025_mp5_2038_heating_naturalGas_consumption` | num | 0 | 0 | 8,195 | 1.161e+04 | 1.933e+05 |
| 221 | `ref2025_mp5_2039_heating_naturalGas_consumption` | num | 0 | 0 | 8,158 | 1.157e+04 | 1.929e+05 |
| 222 | `ref2025_mp5_2025_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 223 | `ref2025_mp5_2026_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 224 | `ref2025_mp5_2027_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 225 | `ref2025_mp5_2028_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 226 | `ref2025_mp5_2029_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 227 | `ref2025_mp5_2030_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 228 | `ref2025_mp5_2031_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 229 | `ref2025_mp5_2032_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 230 | `ref2025_mp5_2033_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 231 | `ref2025_mp5_2034_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 232 | `ref2025_mp5_2035_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 233 | `ref2025_mp5_2036_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 234 | `ref2025_mp5_2037_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 235 | `ref2025_mp5_2038_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 236 | `ref2025_mp5_2039_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 237 | `ref2025_mp5_2025_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 238 | `ref2025_mp5_2026_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 239 | `ref2025_mp5_2027_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 240 | `ref2025_mp5_2028_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 241 | `ref2025_mp5_2029_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 242 | `ref2025_mp5_2030_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 243 | `ref2025_mp5_2031_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 244 | `ref2025_mp5_2032_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 245 | `ref2025_mp5_2033_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 246 | `ref2025_mp5_2034_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 247 | `ref2025_mp5_2035_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 248 | `ref2025_mp5_2036_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 249 | `ref2025_mp5_2037_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 250 | `ref2025_mp5_2038_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 251 | `ref2025_mp5_2039_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 252 | `baseline_2025_cooling_consumption` | num | 0 | 0 | 3,465 | 4,410 | 5.773e+04 |
| 253 | `baseline_2026_cooling_consumption` | num | 0 | 0 | 3,636 | 4,603 | 6.127e+04 |
| 254 | `baseline_2027_cooling_consumption` | num | 0 | 0 | 3,660 | 4,631 | 6.162e+04 |
| 255 | `baseline_2028_cooling_consumption` | num | 0 | 0 | 3,684 | 4,660 | 6.2e+04 |
| 256 | `baseline_2029_cooling_consumption` | num | 0 | 0 | 3,706 | 4,688 | 6.234e+04 |
| 257 | `baseline_2030_cooling_consumption` | num | 0 | 0 | 3,730 | 4,716 | 6.268e+04 |
| 258 | `baseline_2031_cooling_consumption` | num | 0 | 0 | 3,754 | 4,745 | 6.307e+04 |
| 259 | `baseline_2032_cooling_consumption` | num | 0 | 0 | 3,779 | 4,775 | 6.341e+04 |
| 260 | `baseline_2033_cooling_consumption` | num | 0 | 0 | 3,801 | 4,802 | 6.375e+04 |
| 261 | `baseline_2034_cooling_consumption` | num | 0 | 0 | 3,825 | 4,831 | 6.409e+04 |
| 262 | `baseline_2035_cooling_consumption` | num | 0 | 0 | 3,849 | 4,860 | 6.444e+04 |
| 263 | `baseline_2036_cooling_consumption` | num | 0 | 0 | 3,875 | 4,889 | 6.482e+04 |
| 264 | `baseline_2037_cooling_consumption` | num | 0 | 0 | 3,898 | 4,918 | 6.516e+04 |
| 265 | `baseline_2038_cooling_consumption` | num | 0 | 0 | 3,922 | 4,947 | 6.55e+04 |
| 266 | `baseline_2039_cooling_consumption` | num | 0 | 0 | 3,947 | 4,975 | 6.585e+04 |
| 267 | `baseline_2025_cooling_electricity_consumption` | num | 0 | 0 | 3,465 | 4,410 | 5.773e+04 |
| 268 | `baseline_2026_cooling_electricity_consumption` | num | 0 | 0 | 3,636 | 4,603 | 6.127e+04 |
| 269 | `baseline_2027_cooling_electricity_consumption` | num | 0 | 0 | 3,660 | 4,631 | 6.162e+04 |
| 270 | `baseline_2028_cooling_electricity_consumption` | num | 0 | 0 | 3,684 | 4,660 | 6.2e+04 |
| 271 | `baseline_2029_cooling_electricity_consumption` | num | 0 | 0 | 3,706 | 4,688 | 6.234e+04 |
| 272 | `baseline_2030_cooling_electricity_consumption` | num | 0 | 0 | 3,730 | 4,716 | 6.268e+04 |
| 273 | `baseline_2031_cooling_electricity_consumption` | num | 0 | 0 | 3,754 | 4,745 | 6.307e+04 |
| 274 | `baseline_2032_cooling_electricity_consumption` | num | 0 | 0 | 3,779 | 4,775 | 6.341e+04 |
| 275 | `baseline_2033_cooling_electricity_consumption` | num | 0 | 0 | 3,801 | 4,802 | 6.375e+04 |
| 276 | `baseline_2034_cooling_electricity_consumption` | num | 0 | 0 | 3,825 | 4,831 | 6.409e+04 |
| 277 | `baseline_2035_cooling_electricity_consumption` | num | 0 | 0 | 3,849 | 4,860 | 6.444e+04 |
| 278 | `baseline_2036_cooling_electricity_consumption` | num | 0 | 0 | 3,875 | 4,889 | 6.482e+04 |
| 279 | `baseline_2037_cooling_electricity_consumption` | num | 0 | 0 | 3,898 | 4,918 | 6.516e+04 |
| 280 | `baseline_2038_cooling_electricity_consumption` | num | 0 | 0 | 3,922 | 4,947 | 6.55e+04 |
| 281 | `baseline_2039_cooling_electricity_consumption` | num | 0 | 0 | 3,947 | 4,975 | 6.585e+04 |
| 282 | `ref2025_mp5_2025_cooling_consumption` | num | 0 | 0 | 3,229 | 3,918 | 3.307e+04 |
| 283 | `ref2025_mp5_2026_cooling_consumption` | num | 0 | 0 | 3,390 | 4,092 | 3.461e+04 |
| 284 | `ref2025_mp5_2027_cooling_consumption` | num | 0 | 0 | 3,413 | 4,117 | 3.483e+04 |
| 285 | `ref2025_mp5_2028_cooling_consumption` | num | 0 | 0 | 3,435 | 4,143 | 3.506e+04 |
| 286 | `ref2025_mp5_2029_cooling_consumption` | num | 0 | 0 | 3,457 | 4,168 | 3.527e+04 |
| 287 | `ref2025_mp5_2030_cooling_consumption` | num | 0 | 0 | 3,479 | 4,194 | 3.55e+04 |
| 288 | `ref2025_mp5_2031_cooling_consumption` | num | 0 | 0 | 3,502 | 4,220 | 3.573e+04 |
| 289 | `ref2025_mp5_2032_cooling_consumption` | num | 0 | 0 | 3,525 | 4,246 | 3.597e+04 |
| 290 | `ref2025_mp5_2033_cooling_consumption` | num | 0 | 0 | 3,547 | 4,271 | 3.62e+04 |
| 291 | `ref2025_mp5_2034_cooling_consumption` | num | 0 | 0 | 3,569 | 4,297 | 3.643e+04 |
| 292 | `ref2025_mp5_2035_cooling_consumption` | num | 0 | 0 | 3,592 | 4,323 | 3.665e+04 |
| 293 | `ref2025_mp5_2036_cooling_consumption` | num | 0 | 0 | 3,615 | 4,349 | 3.688e+04 |
| 294 | `ref2025_mp5_2037_cooling_consumption` | num | 0 | 0 | 3,637 | 4,375 | 3.712e+04 |
| 295 | `ref2025_mp5_2038_cooling_consumption` | num | 0 | 0 | 3,661 | 4,401 | 3.735e+04 |
| 296 | `ref2025_mp5_2039_cooling_consumption` | num | 0 | 0 | 3,683 | 4,426 | 3.758e+04 |
| 297 | `ref2025_mp5_2025_cooling_electricity_consumption` | num | 0 | 0 | 3,229 | 3,918 | 3.307e+04 |
| 298 | `ref2025_mp5_2026_cooling_electricity_consumption` | num | 0 | 0 | 3,390 | 4,092 | 3.461e+04 |
| 299 | `ref2025_mp5_2027_cooling_electricity_consumption` | num | 0 | 0 | 3,413 | 4,117 | 3.483e+04 |
| 300 | `ref2025_mp5_2028_cooling_electricity_consumption` | num | 0 | 0 | 3,435 | 4,143 | 3.506e+04 |
| 301 | `ref2025_mp5_2029_cooling_electricity_consumption` | num | 0 | 0 | 3,457 | 4,168 | 3.527e+04 |
| 302 | `ref2025_mp5_2030_cooling_electricity_consumption` | num | 0 | 0 | 3,479 | 4,194 | 3.55e+04 |
| 303 | `ref2025_mp5_2031_cooling_electricity_consumption` | num | 0 | 0 | 3,502 | 4,220 | 3.573e+04 |
| 304 | `ref2025_mp5_2032_cooling_electricity_consumption` | num | 0 | 0 | 3,525 | 4,246 | 3.597e+04 |
| 305 | `ref2025_mp5_2033_cooling_electricity_consumption` | num | 0 | 0 | 3,547 | 4,271 | 3.62e+04 |
| 306 | `ref2025_mp5_2034_cooling_electricity_consumption` | num | 0 | 0 | 3,569 | 4,297 | 3.643e+04 |
| 307 | `ref2025_mp5_2035_cooling_electricity_consumption` | num | 0 | 0 | 3,592 | 4,323 | 3.665e+04 |
| 308 | `ref2025_mp5_2036_cooling_electricity_consumption` | num | 0 | 0 | 3,615 | 4,349 | 3.688e+04 |
| 309 | `ref2025_mp5_2037_cooling_electricity_consumption` | num | 0 | 0 | 3,637 | 4,375 | 3.712e+04 |
| 310 | `ref2025_mp5_2038_cooling_electricity_consumption` | num | 0 | 0 | 3,661 | 4,401 | 3.735e+04 |
| 311 | `ref2025_mp5_2039_cooling_electricity_consumption` | num | 0 | 0 | 3,683 | 4,426 | 3.758e+04 |
| 312 | `baseline_heating_lifetime_fuel_cost` | num | 0 | 0 | 1.35e+04 | 1.651e+04 | 3.579e+05 |
| 313 | `ref2025_mp5_heating_lifetime_fuel_cost` | num | 0 | 0 | 1.265e+04 | 1.544e+04 | 1.72e+05 |
| 314 | `ref2025_mp5_heating_lifetime_savings_fuel_cost` | num | 0 | -3.88e+04 | 97.39 | 1,072 | 2.43e+05 |
| 315 | `baseline_cooling_lifetime_fuel_cost` | num | 0 | 0 | 1.004e+04 | 1.291e+04 | 1.649e+05 |
| 316 | `ref2025_mp5_cooling_lifetime_fuel_cost` | num | 0 | 0 | 9,404 | 1.156e+04 | 1.319e+05 |
| 317 | `ref2025_mp5_cooling_lifetime_savings_fuel_cost` | num | 0 | -8.913e+04 | 1,352 | 1,353 | 7.431e+04 |
| 318 | `ref2025_mp5_cooling_lifetime_savings_negative` | bool | 0 | | | | {False: 126808, True: 35175} |
| 319 | `mp5_heating_replacement_installed_cost_v4MID` | num | 0 | 270.8 | 3,656 | 3,742 | 2.326e+04 |
| 320 | `mp5_heating_upgrade_installed_cost_v4MID` | num | 0 | 1.06e+04 | 1.502e+04 | 1.571e+04 | 9.598e+04 |
| 321 | `mp5_heating_backupFurnace_installed_cost_v4MID` | num | 0 | 3,646 | 4,069 | 4,112 | 6,567 |
| 322 | `mp5_cooling_replacement_installed_cost_v4MID` | num | 0 | 478 | 6,075 | 5,843 | 3.13e+04 |
| 323 | `mp5_cooling_replacement_credit_applied_v4MID` | num | 0 | 478 | 6,075 | 5,843 | 3.13e+04 |
| 324 | `mp5_heating_rebate_amount_june2026_v4MID` | num | 0 | 0 | 6,957 | 4,959 | 8,000 |
| 325 | `mp5_rebate_eligibility_june2026` | text | 0 | | | | HEEHR: 98,895; Not Eligible: 40,551; HOMES: 22,537 |
| 326 | `mp5_modeled_savings_frac` | num | 0 | -0.9813 | 0.1676 | 0.1692 | 0.7347 |
| 327 | `public_discount_rate` | num | 0 | 0.02 | 0.02 | 0.02 | 0.02 |
| 328 | `private_discount_rate_fixed_base` | num | 0 | 0.07 | 0.07 | 0.07 | 0.07 |
| 329 | `ref2025_mp5_heating_discounted_lifetime_savings_fixed_base` | num | 0 | -2.523e+04 | 46.52 | 667.8 | 1.587e+05 |
| 330 | `ref2025_mp5_cooling_discounted_lifetime_savings_fixed_base` | num | 0 | -5.704e+04 | 868.4 | 867.4 | 4.756e+04 |
| 331 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | num | 0 | -3,632 | 9,549 | 1.024e+04 | 6.47e+04 |
| 332 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_private_npv_fixed_base` | num | 0 | -7.816e+04 | -7,941 | -8,703 | 1.577e+05 |
| 333 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base` | num | 0 | 0 | 0 | 0.03268 | 1 |

#### `tepper_county_mp5_National_2026-10-06_19-09.csv`

- Rows: 2,973 counties; columns: 11; size 0.3 MB
- Sum of `home_count`: 41,128,079 homes (161,983 rdu)

| # | Column | Type | Blanks | Min | Median | Mean | Max / values |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `county` | text | 0 | | | | 2,973 distinct |
| 2 | `state` | text | 0 | | | | 49 distinct |
| 3 | `home_count` | num | 0 | 253.9 | 2,793 | 1.383e+04 | 1.042e+06 |
| 4 | `adoption_rate_pct` | num | 0 | 0 | 0 | 5.376 | 100 |
| 5 | `operating_cost_pct_change` | num | 1 | -75.71 | -7.214 | -9.48 | 49.21 |
| 6 | `baseline_elec_gwh` | num | 0 | 0.63 | 37.5 | 171.1 | 1.164e+04 |
| 7 | `retrofit_elec_gwh` | num | 0 | 1.12 | 40.16 | 190.8 | 1.142e+04 |
| 8 | `elec_change_gwh` | num | 0 | -855.9 | 2.93 | 19.74 | 2,464 |
| 9 | `site_energy_change_gwh` | num | 0 | -7,759 | -19.11 | -93.44 | 1.47 |
| 10 | `pct_elec_demand_change` | num | 0 | -64.97 | 11.96 | 10.28 | 138.5 |
| 11 | `pct_site_energy_change` | num | 0 | -44.08 | -17.81 | -17.29 | 29.61 |

#### `tepper_household_mp5_Allegheny_2026-10-06_19-09.csv`

- Rows: 1,091 rdu = 277,009 homes; columns: 183 (bldg_id first); size 2.8 MB

| # | Column | Type | Blanks | Min | Median | Mean | Max / values |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `bldg_id` | num | 0 | 264 | 2.875e+05 | 2.806e+05 | 5.491e+05 |
| 2 | `weight` | num | 0 | 253.9 | 253.9 | 253.9 | 253.9 |
| 3 | `state` | text | 0 | | | | PA: 1,091 |
| 4 | `county` | text | 0 | | | | G4200030: 1,091 |
| 5 | `county_fips` | num | 0 | 4.2e+04 | 4.2e+04 | 4.2e+04 | 4.2e+04 |
| 6 | `puma` | text | 0 | | | | G42001804: 180; G42001802: 155; G42001803: 121; G42001806: 119; G42001805: 110; G42001807: 109; G42001801: 103; G42001702: 100; G42001701: 94 |
| 7 | `county_and_puma` | text | 0 | | | | G4200030, G42001804: 180; G4200030, G42001802: 155; G4200030, G42001803: 121; G4200030, G42001806: 119; G4200030, G42001805: 110; G4200030, G42001807: 109; G4200030, G42001801: 103; G4200030, G42001702: 100; G4200030, G42001701: 94 |
| 8 | `census_region` | text | 0 | | | | Northeast: 1,091 |
| 9 | `census_division` | text | 0 | | | | Middle Atlantic: 1,091 |
| 10 | `census_division_recs` | text | 0 | | | | Middle Atlantic: 1,091 |
| 11 | `building_america_climate_zone` | text | 0 | | | | Cold: 1,091 |
| 12 | `reeds_balancing_area` | num | 0 | 115 | 115 | 115 | 115 |
| 13 | `city` | text | 0 | | | | In another census Place: 553; Not in a census Place: 348; Pittsburgh: 190 |
| 14 | `urbanicity` | text | 0 | | | | Suburban: 997; Urban: 94 |
| 15 | `weather_file_city` | text | 0 | | | | Allegheny Co: 1,091 |
| 16 | `Longitude` | num | 0 | -79.92 | -79.92 | -79.92 | -79.92 |
| 17 | `Latitude` | num | 0 | 40.36 | 40.36 | 40.36 | 40.36 |
| 18 | `gea_region` | text | 0 | | | | PJM_East: 1,091 |
| 19 | `square_footage` | num | 0 | 273 | 1,698 | 1,859 | 5,587 |
| 20 | `building_type` | text | 0 | | | | Single-Family Detached: 968; Single-Family Attached: 123 |
| 21 | `occupancy` | num | 0 | 1 | 2 | 2.578 | 9 |
| 22 | `tenure` | text | 0 | | | | Owner: 924; Renter: 167 |
| 23 | `vacancy_status` | text | 0 | | | | Occupied: 1,091 |
| 24 | `vintage` | text | 0 | | | | 1950s: 278; <1940: 270; 1960s: 154; 1940s: 103; 1970s: 88; 1980s: 60; 1990s: 56; 2000s: 51; 2010s: 31 |
| 25 | `income` | num | 0 | 9,999 | 7.5e+04 | 8.876e+04 | 2e+05 |
| 26 | `federal_poverty_level` | text | 0 | | | | 400%+: 545; 200-300%: 166; 300-400%: 159; 0-100%: 80; 150-200%: 73; 100-150%: 68 |
| 27 | `household_income` | num | 0 | 1.282e+04 | 9.856e+04 | 1.138e+05 | 2.564e+05 |
| 28 | `census_area_medianIncome` | num | 0 | 8.061e+04 | 8.061e+04 | 8.061e+04 | 8.061e+04 |
| 29 | `income_level` | text | 0 | | | | Middle-to-Upper-Income: 426; Low-Income: 352; Moderate-Income: 313 |
| 30 | `percent_AMI` | num | 0 | 15.9 | 122.3 | 141.1 | 318.1 |
| 31 | `lmi_or_mui` | text | 0 | | | | LMI: 665; MUI: 426 |
| 32 | `base_heating_fuel` | text | 0 | | | | Natural Gas: 1,088; Electricity: 3 |
| 33 | `heating_type` | text | 0 | | | | Natural Gas Fuel Furnace: 1,056; Natural Gas Fuel Boiler: 32; Electricity Baseboard: 2; Electricity Electric Furnace: 1 |
| 34 | `base_heating_efficiency` | text | 0 | | | | Fuel Furnace, 80% AFUE: 566; Fuel Furnace, 92.5% AFUE: 409; Fuel Furnace, 76% AFUE: 72; Fuel Boiler, 80% AFUE: 23; Fuel Boiler, 76% AFUE: 9; Fuel Furnace, 60% AFUE: 9; Electric Baseboard, 100% Efficiency: 2; Electric Furnace, 100% AFUE: 1 |
| 35 | `base_cooling_fuel` | text | 0 | | | | Electricity: 1,091 |
| 36 | `cooling_type` | text | 0 | | | | Central AC: 835; Room AC: 256 |
| 37 | `base_cooling_efficiency` | text | 0 | | | | AC, SEER 13: 485; AC, SEER 15: 243; Room AC, EER 10.7: 131; Room AC, EER 12.0: 91; AC, SEER 10: 89; Room AC, EER 9.8: 27; AC, SEER 8: 18; Room AC, EER 8.5: 7 |
| 38 | `fuel_type_heating` | text | 0 | | | | naturalGas: 1,088; electricity: 3 |
| 39 | `fuel_type_cooling` | text | 0 | | | | electricity: 1,091 |
| 40 | `hvac_has_ducts` | text | 0 | | | | Yes: 1,091 |
| 41 | `hvac_heating_type_and_fuel` | text | 0 | | | | Natural Gas Fuel Furnace: 1,056; Natural Gas Fuel Boiler: 32; Electricity Baseboard: 2; Electricity Electric Furnace: 1 |
| 42 | `hvac_heating_efficiency` | text | 0 | | | | Fuel Furnace, 80% AFUE: 566; Fuel Furnace, 92.5% AFUE: 409; Fuel Furnace, 76% AFUE: 72; Fuel Boiler, 80% AFUE: 23; Fuel Boiler, 76% AFUE: 9; Fuel Furnace, 60% AFUE: 9; Electric Baseboard, 100% Efficiency: 2; Electric Furnace, 100% AFUE: 1 |
| 43 | `size_heating_system_primary_k_btu_h` | num | 0 | 6.43 | 28.38 | 31.47 | 113.2 |
| 44 | `hvac_cooling_type` | text | 0 | | | | Central AC: 835; Room AC: 256 |
| 45 | `hvac_cooling_efficiency` | text | 0 | | | | AC, SEER 13: 485; AC, SEER 15: 243; Room AC, EER 10.7: 131; Room AC, EER 12.0: 91; AC, SEER 10: 89; Room AC, EER 9.8: 27; AC, SEER 8: 18; Room AC, EER 8.5: 7 |
| 46 | `size_cooling_system_primary_k_btu_h` | num | 0 | 6.43 | 28.38 | 31.47 | 113.2 |
| 47 | `upgrade_hvac_heating_efficiency` | text | 0 | | | | Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 95.0% AFUE NG, 35F switchover: 1,091 |
| 48 | `upgrade_hvac_cooling_efficiency` | text | 0 | | | | Ducted Heat Pump: 1,091 |
| 49 | `upgrade_hp_seer2` | num | 0 | 15.2 | 15.2 | 15.2 | 15.2 |
| 50 | `upgrade_hp_seer1` | num | 0 | 16 | 16 | 16 | 16 |
| 51 | `upgrade_hp_hspf2` | num | 0 | 7.8 | 7.8 | 7.8 | 7.8 |
| 52 | `upgrade_hp_hspf1` | num | 0 | 9.176 | 9.176 | 9.176 | 9.176 |
| 53 | `upgrade_backup_fuel` | text | 0 | | | | Natural Gas: 1,091 |
| 54 | `upgrade_backup_afue` | num | 0 | 0.95 | 0.95 | 0.95 | 0.95 |
| 55 | `upgrade_switchover_f` | num | 0 | 35 | 35 | 35 | 35 |
| 56 | `size_heat_pump_backup_primary_k_btu_h` | num | 0 | 8.36 | 66.65 | 71.82 | 236.5 |
| 57 | `include_sample` | bool | 0 | | | | {True: 1091} |
| 58 | `include_heating` | bool | 0 | | | | {True: 1091} |
| 59 | `include_cooling` | bool | 0 | | | | {True: 1091} |
| 60 | `base_peak_electricity_summer_kw` | num | 0 | 0.9624 | 6.369 | 7.148 | 49.11 |
| 61 | `base_peak_electricity_winter_kw` | num | 0 | 1.1 | 5.699 | 6.555 | 48.44 |
| 62 | `base_peak_load_cooling_kbtu_hr` | num | 0 | 1.346 | 18.9 | 20.87 | 76.76 |
| 63 | `base_peak_load_heating_kbtu_hr` | num | 0 | 0 | 67.25 | 72.09 | 239.2 |
| 64 | `mp5_peak_electricity_summer_kw` | num | 0 | 1.569 | 6.66 | 7.491 | 49.69 |
| 65 | `mp5_peak_electricity_winter_kw` | num | 0 | 1.533 | 7.21 | 7.932 | 49.17 |
| 66 | `mp5_peak_electricity_summer_kw_savings` | num | 0 | -4.76 | -0.1176 | -0.3433 | 1.902 |
| 67 | `mp5_peak_electricity_winter_kw_savings` | num | 0 | -6.678 | -1.268 | -1.377 | 14.64 |
| 68 | `mp5_peak_load_cooling_kbtu_hr` | num | 0 | 6.174 | 24.61 | 26.5 | 82.64 |
| 69 | `mp5_peak_load_heating_kbtu_hr` | num | 0 | 0 | 67.24 | 71.29 | 239.1 |
| 70 | `mp5_peak_load_cooling_kbtu_hr_savings` | num | 0 | -70.84 | -1.986 | -5.628 | 31.39 |
| 71 | `mp5_peak_load_heating_kbtu_hr_savings` | num | 0 | -33.67 | 0.843 | 0.8023 | 106.5 |
| 72 | `panel_service_rating_amps` | num | 0 | 60 | 125 | 146.2 | 400 |
| 73 | `mp5_panel_constraint_overall` | text | 0 | | | | No Constraint: 934; Space Constrained Only: 86; Capacity Constrained Only: 61; Capacity and Space Constrained: 10 |
| 74 | `mp5_panel_constraint_capacity` | bool | 0 | | | | {False: 1020, True: 71} |
| 75 | `mp5_panel_constraint_breaker_space` | bool | 0 | | | | {False: 995, True: 96} |
| 76 | `base_electricity_heating_consumption` | num | 0 | 0 | 0 | 56.17 | 2.664e+04 |
| 77 | `base_electricity_cooling_consumption` | num | 0 | 110.5 | 2,109 | 2,215 | 8,238 |
| 78 | `base_fuelOil_heating_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 79 | `base_naturalGas_heating_consumption` | num | 0 | 0 | 2.69e+04 | 2.972e+04 | 1.139e+05 |
| 80 | `base_propane_heating_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 81 | `baseline_heating_consumption` | num | 0 | 0 | 2.69e+04 | 2.977e+04 | 1.139e+05 |
| 82 | `baseline_cooling_consumption` | num | 0 | 110.5 | 2,109 | 2,215 | 8,238 |
| 83 | `mp5_heating_consumption` | num | 0 | 202.8 | 2,483 | 2,748 | 9,859 |
| 84 | `mp5_cooling_consumption` | num | 0 | 307.1 | 2,052 | 2,149 | 6,474 |
| 85 | `base_total_electricity_consumption` | num | 0 | 2,838 | 9,470 | 1.01e+04 | 4.037e+04 |
| 86 | `mp5_total_electricity_consumption` | num | 0 | 3,344 | 1.236e+04 | 1.3e+04 | 3.161e+04 |
| 87 | `baseline_total_site_consumption` | num | 0 | 1.045e+04 | 4.193e+04 | 4.5e+04 | 1.311e+05 |
| 88 | `base_electricity_heating_fansPumps_consumption` | num | 0 | 0 | 642.4 | 706.1 | 2,941 |
| 89 | `base_electricity_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 90 | `base_electricity_heating_hpBackupFans_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 91 | `base_naturalGas_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 92 | `base_propane_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 93 | `base_fuelOil_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 94 | `base_electricity_cooling_fansPumps_consumption` | num | 0 | 0 | 179.7 | 217.4 | 1,541 |
| 95 | `mp5_electricity_heating_fansPumps_consumption` | num | 0 | 0 | 228 | 250 | 961.6 |
| 96 | `mp5_electricity_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 97 | `mp5_electricity_heating_hpBackupFans_consumption` | num | 0 | 0 | 402.4 | 440.8 | 1,595 |
| 98 | `mp5_naturalGas_heating_hpBackup_consumption` | num | 0 | 0 | 1.734e+04 | 1.898e+04 | 7.034e+04 |
| 99 | `mp5_propane_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 100 | `mp5_fuelOil_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 101 | `mp5_electricity_cooling_fansPumps_consumption` | num | 0 | 69.16 | 477.4 | 505.7 | 1,666 |
| 102 | `baseline_2025_heating_consumption` | num | 0 | 0 | 2.756e+04 | 3.048e+04 | 1.168e+05 |
| 103 | `baseline_2026_heating_consumption` | num | 0 | 0 | 2.688e+04 | 2.973e+04 | 1.139e+05 |
| 104 | `baseline_2027_heating_consumption` | num | 0 | 0 | 2.677e+04 | 2.96e+04 | 1.135e+05 |
| 105 | `baseline_2028_heating_consumption` | num | 0 | 0 | 2.666e+04 | 2.949e+04 | 1.13e+05 |
| 106 | `baseline_2029_heating_consumption` | num | 0 | 0 | 2.655e+04 | 2.937e+04 | 1.125e+05 |
| 107 | `baseline_2030_heating_consumption` | num | 0 | 0 | 2.644e+04 | 2.924e+04 | 1.121e+05 |
| 108 | `baseline_2031_heating_consumption` | num | 0 | 0 | 2.633e+04 | 2.912e+04 | 1.116e+05 |
| 109 | `baseline_2032_heating_consumption` | num | 0 | 0 | 2.622e+04 | 2.9e+04 | 1.111e+05 |
| 110 | `baseline_2033_heating_consumption` | num | 0 | 0 | 2.611e+04 | 2.888e+04 | 1.107e+05 |
| 111 | `baseline_2034_heating_consumption` | num | 0 | 0 | 2.601e+04 | 2.876e+04 | 1.102e+05 |
| 112 | `baseline_2035_heating_consumption` | num | 0 | 0 | 2.589e+04 | 2.864e+04 | 1.098e+05 |
| 113 | `baseline_2036_heating_consumption` | num | 0 | 0 | 2.579e+04 | 2.852e+04 | 1.093e+05 |
| 114 | `baseline_2037_heating_consumption` | num | 0 | 0 | 2.567e+04 | 2.84e+04 | 1.088e+05 |
| 115 | `baseline_2038_heating_consumption` | num | 0 | 0 | 2.557e+04 | 2.828e+04 | 1.084e+05 |
| 116 | `baseline_2039_heating_consumption` | num | 0 | 0 | 2.546e+04 | 2.816e+04 | 1.079e+05 |
| 117 | `ref2025_mp5_2025_heating_consumption` | num | 0 | 202.8 | 2.054e+04 | 2.242e+04 | 8.129e+04 |
| 118 | `ref2025_mp5_2026_heating_consumption` | num | 0 | 197.8 | 2.003e+04 | 2.187e+04 | 7.929e+04 |
| 119 | `ref2025_mp5_2027_heating_consumption` | num | 0 | 197 | 1.995e+04 | 2.178e+04 | 7.896e+04 |
| 120 | `ref2025_mp5_2028_heating_consumption` | num | 0 | 196.2 | 1.987e+04 | 2.169e+04 | 7.864e+04 |
| 121 | `ref2025_mp5_2029_heating_consumption` | num | 0 | 195.4 | 1.979e+04 | 2.16e+04 | 7.832e+04 |
| 122 | `ref2025_mp5_2030_heating_consumption` | num | 0 | 194.6 | 1.97e+04 | 2.151e+04 | 7.799e+04 |
| 123 | `ref2025_mp5_2031_heating_consumption` | num | 0 | 193.8 | 1.962e+04 | 2.142e+04 | 7.768e+04 |
| 124 | `ref2025_mp5_2032_heating_consumption` | num | 0 | 193 | 1.954e+04 | 2.133e+04 | 7.735e+04 |
| 125 | `ref2025_mp5_2033_heating_consumption` | num | 0 | 192.2 | 1.946e+04 | 2.125e+04 | 7.703e+04 |
| 126 | `ref2025_mp5_2034_heating_consumption` | num | 0 | 191.4 | 1.938e+04 | 2.116e+04 | 7.671e+04 |
| 127 | `ref2025_mp5_2035_heating_consumption` | num | 0 | 190.6 | 1.93e+04 | 2.107e+04 | 7.638e+04 |
| 128 | `ref2025_mp5_2036_heating_consumption` | num | 0 | 189.8 | 1.922e+04 | 2.098e+04 | 7.607e+04 |
| 129 | `ref2025_mp5_2037_heating_consumption` | num | 0 | 189 | 1.913e+04 | 2.089e+04 | 7.574e+04 |
| 130 | `ref2025_mp5_2038_heating_consumption` | num | 0 | 188.2 | 1.905e+04 | 2.08e+04 | 7.542e+04 |
| 131 | `ref2025_mp5_2039_heating_consumption` | num | 0 | 187.4 | 1.897e+04 | 2.071e+04 | 7.51e+04 |
| 132 | `baseline_2025_cooling_consumption` | num | 0 | 110.5 | 2,303 | 2,433 | 8,957 |
| 133 | `baseline_2026_cooling_consumption` | num | 0 | 123.8 | 2,579 | 2,725 | 1.003e+04 |
| 134 | `baseline_2027_cooling_consumption` | num | 0 | 124.9 | 2,603 | 2,750 | 1.013e+04 |
| 135 | `baseline_2028_cooling_consumption` | num | 0 | 126.1 | 2,627 | 2,776 | 1.022e+04 |
| 136 | `baseline_2029_cooling_consumption` | num | 0 | 127.4 | 2,654 | 2,804 | 1.033e+04 |
| 137 | `baseline_2030_cooling_consumption` | num | 0 | 128.5 | 2,678 | 2,830 | 1.042e+04 |
| 138 | `baseline_2031_cooling_consumption` | num | 0 | 129.7 | 2,702 | 2,855 | 1.051e+04 |
| 139 | `baseline_2032_cooling_consumption` | num | 0 | 130.8 | 2,727 | 2,881 | 1.061e+04 |
| 140 | `baseline_2033_cooling_consumption` | num | 0 | 132 | 2,751 | 2,906 | 1.07e+04 |
| 141 | `baseline_2034_cooling_consumption` | num | 0 | 133.1 | 2,775 | 2,931 | 1.079e+04 |
| 142 | `baseline_2035_cooling_consumption` | num | 0 | 134.4 | 2,802 | 2,960 | 1.09e+04 |
| 143 | `baseline_2036_cooling_consumption` | num | 0 | 135.6 | 2,826 | 2,985 | 1.099e+04 |
| 144 | `baseline_2037_cooling_consumption` | num | 0 | 136.7 | 2,850 | 3,011 | 1.109e+04 |
| 145 | `baseline_2038_cooling_consumption` | num | 0 | 137.9 | 2,874 | 3,036 | 1.118e+04 |
| 146 | `baseline_2039_cooling_consumption` | num | 0 | 139 | 2,898 | 3,062 | 1.127e+04 |
| 147 | `ref2025_mp5_2025_cooling_consumption` | num | 0 | 376.3 | 2,529 | 2,655 | 8,125 |
| 148 | `ref2025_mp5_2026_cooling_consumption` | num | 0 | 421.5 | 2,832 | 2,974 | 9,101 |
| 149 | `ref2025_mp5_2027_cooling_consumption` | num | 0 | 425.4 | 2,859 | 3,002 | 9,185 |
| 150 | `ref2025_mp5_2028_cooling_consumption` | num | 0 | 429.4 | 2,885 | 3,029 | 9,270 |
| 151 | `ref2025_mp5_2029_cooling_consumption` | num | 0 | 433.8 | 2,915 | 3,060 | 9,366 |
| 152 | `ref2025_mp5_2030_cooling_consumption` | num | 0 | 437.7 | 2,941 | 3,088 | 9,451 |
| 153 | `ref2025_mp5_2031_cooling_consumption` | num | 0 | 441.6 | 2,968 | 3,116 | 9,536 |
| 154 | `ref2025_mp5_2032_cooling_consumption` | num | 0 | 445.6 | 2,994 | 3,144 | 9,620 |
| 155 | `ref2025_mp5_2033_cooling_consumption` | num | 0 | 449.5 | 3,020 | 3,171 | 9,705 |
| 156 | `ref2025_mp5_2034_cooling_consumption` | num | 0 | 453.4 | 3,047 | 3,199 | 9,790 |
| 157 | `ref2025_mp5_2035_cooling_consumption` | num | 0 | 457.9 | 3,077 | 3,230 | 9,886 |
| 158 | `ref2025_mp5_2036_cooling_consumption` | num | 0 | 461.8 | 3,103 | 3,258 | 9,970 |
| 159 | `ref2025_mp5_2037_cooling_consumption` | num | 0 | 465.7 | 3,129 | 3,286 | 1.006e+04 |
| 160 | `ref2025_mp5_2038_cooling_consumption` | num | 0 | 469.6 | 3,156 | 3,313 | 1.014e+04 |
| 161 | `ref2025_mp5_2039_cooling_consumption` | num | 0 | 473.6 | 3,182 | 3,341 | 1.022e+04 |
| 162 | `baseline_heating_lifetime_fuel_cost` | num | 0 | 0 | 2.017e+04 | 2.235e+04 | 8.569e+04 |
| 163 | `ref2025_mp5_heating_lifetime_fuel_cost` | num | 0 | 570 | 2.058e+04 | 2.257e+04 | 7.859e+04 |
| 164 | `ref2025_mp5_heating_lifetime_savings_fuel_cost` | num | 0 | -1.431e+04 | -336.3 | -222 | 4.209e+04 |
| 165 | `baseline_cooling_lifetime_fuel_cost` | num | 0 | 383.9 | 8,002 | 8,454 | 3.113e+04 |
| 166 | `ref2025_mp5_cooling_lifetime_fuel_cost` | num | 0 | 1,308 | 8,787 | 9,226 | 2.823e+04 |
| 167 | `ref2025_mp5_cooling_lifetime_savings_fuel_cost` | num | 0 | -1.983e+04 | 657.9 | -771.8 | 1.17e+04 |
| 168 | `ref2025_mp5_cooling_lifetime_savings_negative` | bool | 0 | | | | {False: 654, True: 437} |
| 169 | `mp5_heating_replacement_installed_cost_v4MID` | num | 0 | 3,119 | 3,712 | 3,766 | 6,274 |
| 170 | `mp5_heating_upgrade_installed_cost_v4MID` | num | 0 | 1.089e+04 | 1.398e+04 | 1.442e+04 | 2.591e+04 |
| 171 | `mp5_heating_backupFurnace_installed_cost_v4MID` | num | 0 | 3,804 | 4,201 | 4,237 | 5,359 |
| 172 | `mp5_cooling_replacement_installed_cost_v4MID` | num | 0 | 492.1 | 5,665 | 4,724 | 9,559 |
| 173 | `mp5_cooling_replacement_credit_applied_v4MID` | num | 0 | 492.1 | 5,665 | 4,724 | 9,559 |
| 174 | `mp5_heating_rebate_amount_june2026_v4MID` | num | 0 | 0 | 6,644 | 4,812 | 8,000 |
| 175 | `mp5_rebate_eligibility_june2026` | text | 0 | | | | HEEHR: 665; Not Eligible: 311; HOMES: 115 |
| 176 | `mp5_modeled_savings_frac` | num | 0 | -0.1023 | 0.1652 | 0.1632 | 0.4207 |
| 177 | `public_discount_rate` | num | 0 | 0.02 | 0.02 | 0.02 | 0.02 |
| 178 | `private_discount_rate_fixed_base` | num | 0 | 0.07 | 0.07 | 0.07 | 0.07 |
| 179 | `ref2025_mp5_heating_discounted_lifetime_savings_fixed_base` | num | 0 | -9,299 | -212.3 | -137 | 2.735e+04 |
| 180 | `ref2025_mp5_cooling_discounted_lifetime_savings_fixed_base` | num | 0 | -1.263e+04 | 419 | -491.5 | 7,448 |
| 181 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | num | 0 | 6,201 | 9,227 | 1.016e+04 | 2.446e+04 |
| 182 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_private_npv_fixed_base` | num | 0 | -3.586e+04 | -8,902 | -1.079e+04 | 2.172e+04 |
| 183 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base` | num | 0 | 0 | 0 | 0.003666 | 1 |

#### `tepper_household_detailed_mp5_Allegheny_2026-10-06_19-09.csv`

- Rows: 1,091 rdu = 277,009 homes; columns: 333 (bldg_id first); size 4.8 MB

| # | Column | Type | Blanks | Min | Median | Mean | Max / values |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `bldg_id` | num | 0 | 264 | 2.875e+05 | 2.806e+05 | 5.491e+05 |
| 2 | `weight` | num | 0 | 253.9 | 253.9 | 253.9 | 253.9 |
| 3 | `state` | text | 0 | | | | PA: 1,091 |
| 4 | `county` | text | 0 | | | | G4200030: 1,091 |
| 5 | `county_fips` | num | 0 | 4.2e+04 | 4.2e+04 | 4.2e+04 | 4.2e+04 |
| 6 | `puma` | text | 0 | | | | G42001804: 180; G42001802: 155; G42001803: 121; G42001806: 119; G42001805: 110; G42001807: 109; G42001801: 103; G42001702: 100; G42001701: 94 |
| 7 | `county_and_puma` | text | 0 | | | | G4200030, G42001804: 180; G4200030, G42001802: 155; G4200030, G42001803: 121; G4200030, G42001806: 119; G4200030, G42001805: 110; G4200030, G42001807: 109; G4200030, G42001801: 103; G4200030, G42001702: 100; G4200030, G42001701: 94 |
| 8 | `census_region` | text | 0 | | | | Northeast: 1,091 |
| 9 | `census_division` | text | 0 | | | | Middle Atlantic: 1,091 |
| 10 | `census_division_recs` | text | 0 | | | | Middle Atlantic: 1,091 |
| 11 | `building_america_climate_zone` | text | 0 | | | | Cold: 1,091 |
| 12 | `reeds_balancing_area` | num | 0 | 115 | 115 | 115 | 115 |
| 13 | `city` | text | 0 | | | | In another census Place: 553; Not in a census Place: 348; Pittsburgh: 190 |
| 14 | `urbanicity` | text | 0 | | | | Suburban: 997; Urban: 94 |
| 15 | `weather_file_city` | text | 0 | | | | Allegheny Co: 1,091 |
| 16 | `Longitude` | num | 0 | -79.92 | -79.92 | -79.92 | -79.92 |
| 17 | `Latitude` | num | 0 | 40.36 | 40.36 | 40.36 | 40.36 |
| 18 | `gea_region` | text | 0 | | | | PJM_East: 1,091 |
| 19 | `square_footage` | num | 0 | 273 | 1,698 | 1,859 | 5,587 |
| 20 | `building_type` | text | 0 | | | | Single-Family Detached: 968; Single-Family Attached: 123 |
| 21 | `occupancy` | num | 0 | 1 | 2 | 2.578 | 9 |
| 22 | `tenure` | text | 0 | | | | Owner: 924; Renter: 167 |
| 23 | `vacancy_status` | text | 0 | | | | Occupied: 1,091 |
| 24 | `vintage` | text | 0 | | | | 1950s: 278; <1940: 270; 1960s: 154; 1940s: 103; 1970s: 88; 1980s: 60; 1990s: 56; 2000s: 51; 2010s: 31 |
| 25 | `income` | num | 0 | 9,999 | 7.5e+04 | 8.876e+04 | 2e+05 |
| 26 | `federal_poverty_level` | text | 0 | | | | 400%+: 545; 200-300%: 166; 300-400%: 159; 0-100%: 80; 150-200%: 73; 100-150%: 68 |
| 27 | `household_income` | num | 0 | 1.282e+04 | 9.856e+04 | 1.138e+05 | 2.564e+05 |
| 28 | `census_area_medianIncome` | num | 0 | 8.061e+04 | 8.061e+04 | 8.061e+04 | 8.061e+04 |
| 29 | `income_level` | text | 0 | | | | Middle-to-Upper-Income: 426; Low-Income: 352; Moderate-Income: 313 |
| 30 | `percent_AMI` | num | 0 | 15.9 | 122.3 | 141.1 | 318.1 |
| 31 | `lmi_or_mui` | text | 0 | | | | LMI: 665; MUI: 426 |
| 32 | `base_heating_fuel` | text | 0 | | | | Natural Gas: 1,088; Electricity: 3 |
| 33 | `heating_type` | text | 0 | | | | Natural Gas Fuel Furnace: 1,056; Natural Gas Fuel Boiler: 32; Electricity Baseboard: 2; Electricity Electric Furnace: 1 |
| 34 | `base_heating_efficiency` | text | 0 | | | | Fuel Furnace, 80% AFUE: 566; Fuel Furnace, 92.5% AFUE: 409; Fuel Furnace, 76% AFUE: 72; Fuel Boiler, 80% AFUE: 23; Fuel Boiler, 76% AFUE: 9; Fuel Furnace, 60% AFUE: 9; Electric Baseboard, 100% Efficiency: 2; Electric Furnace, 100% AFUE: 1 |
| 35 | `base_cooling_fuel` | text | 0 | | | | Electricity: 1,091 |
| 36 | `cooling_type` | text | 0 | | | | Central AC: 835; Room AC: 256 |
| 37 | `base_cooling_efficiency` | text | 0 | | | | AC, SEER 13: 485; AC, SEER 15: 243; Room AC, EER 10.7: 131; Room AC, EER 12.0: 91; AC, SEER 10: 89; Room AC, EER 9.8: 27; AC, SEER 8: 18; Room AC, EER 8.5: 7 |
| 38 | `fuel_type_heating` | text | 0 | | | | naturalGas: 1,088; electricity: 3 |
| 39 | `fuel_type_cooling` | text | 0 | | | | electricity: 1,091 |
| 40 | `hvac_has_ducts` | text | 0 | | | | Yes: 1,091 |
| 41 | `hvac_heating_type_and_fuel` | text | 0 | | | | Natural Gas Fuel Furnace: 1,056; Natural Gas Fuel Boiler: 32; Electricity Baseboard: 2; Electricity Electric Furnace: 1 |
| 42 | `hvac_heating_efficiency` | text | 0 | | | | Fuel Furnace, 80% AFUE: 566; Fuel Furnace, 92.5% AFUE: 409; Fuel Furnace, 76% AFUE: 72; Fuel Boiler, 80% AFUE: 23; Fuel Boiler, 76% AFUE: 9; Fuel Furnace, 60% AFUE: 9; Electric Baseboard, 100% Efficiency: 2; Electric Furnace, 100% AFUE: 1 |
| 43 | `size_heating_system_primary_k_btu_h` | num | 0 | 6.43 | 28.38 | 31.47 | 113.2 |
| 44 | `hvac_cooling_type` | text | 0 | | | | Central AC: 835; Room AC: 256 |
| 45 | `hvac_cooling_efficiency` | text | 0 | | | | AC, SEER 13: 485; AC, SEER 15: 243; Room AC, EER 10.7: 131; Room AC, EER 12.0: 91; AC, SEER 10: 89; Room AC, EER 9.8: 27; AC, SEER 8: 18; Room AC, EER 8.5: 7 |
| 46 | `size_cooling_system_primary_k_btu_h` | num | 0 | 6.43 | 28.38 | 31.47 | 113.2 |
| 47 | `upgrade_hvac_heating_efficiency` | text | 0 | | | | Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 95.0% AFUE NG, 35F switchover: 1,091 |
| 48 | `upgrade_hvac_cooling_efficiency` | text | 0 | | | | Ducted Heat Pump: 1,091 |
| 49 | `upgrade_hp_seer2` | num | 0 | 15.2 | 15.2 | 15.2 | 15.2 |
| 50 | `upgrade_hp_seer1` | num | 0 | 16 | 16 | 16 | 16 |
| 51 | `upgrade_hp_hspf2` | num | 0 | 7.8 | 7.8 | 7.8 | 7.8 |
| 52 | `upgrade_hp_hspf1` | num | 0 | 9.176 | 9.176 | 9.176 | 9.176 |
| 53 | `upgrade_backup_fuel` | text | 0 | | | | Natural Gas: 1,091 |
| 54 | `upgrade_backup_afue` | num | 0 | 0.95 | 0.95 | 0.95 | 0.95 |
| 55 | `upgrade_switchover_f` | num | 0 | 35 | 35 | 35 | 35 |
| 56 | `size_heat_pump_backup_primary_k_btu_h` | num | 0 | 8.36 | 66.65 | 71.82 | 236.5 |
| 57 | `include_sample` | bool | 0 | | | | {True: 1091} |
| 58 | `include_heating` | bool | 0 | | | | {True: 1091} |
| 59 | `include_cooling` | bool | 0 | | | | {True: 1091} |
| 60 | `base_peak_electricity_summer_kw` | num | 0 | 0.9624 | 6.369 | 7.148 | 49.11 |
| 61 | `base_peak_electricity_winter_kw` | num | 0 | 1.1 | 5.699 | 6.555 | 48.44 |
| 62 | `base_peak_load_cooling_kbtu_hr` | num | 0 | 1.346 | 18.9 | 20.87 | 76.76 |
| 63 | `base_peak_load_heating_kbtu_hr` | num | 0 | 0 | 67.25 | 72.09 | 239.2 |
| 64 | `mp5_peak_electricity_summer_kw` | num | 0 | 1.569 | 6.66 | 7.491 | 49.69 |
| 65 | `mp5_peak_electricity_winter_kw` | num | 0 | 1.533 | 7.21 | 7.932 | 49.17 |
| 66 | `mp5_peak_electricity_summer_kw_savings` | num | 0 | -4.76 | -0.1176 | -0.3433 | 1.902 |
| 67 | `mp5_peak_electricity_winter_kw_savings` | num | 0 | -6.678 | -1.268 | -1.377 | 14.64 |
| 68 | `mp5_peak_load_cooling_kbtu_hr` | num | 0 | 6.174 | 24.61 | 26.5 | 82.64 |
| 69 | `mp5_peak_load_heating_kbtu_hr` | num | 0 | 0 | 67.24 | 71.29 | 239.1 |
| 70 | `mp5_peak_load_cooling_kbtu_hr_savings` | num | 0 | -70.84 | -1.986 | -5.628 | 31.39 |
| 71 | `mp5_peak_load_heating_kbtu_hr_savings` | num | 0 | -33.67 | 0.843 | 0.8023 | 106.5 |
| 72 | `panel_service_rating_amps` | num | 0 | 60 | 125 | 146.2 | 400 |
| 73 | `mp5_panel_constraint_overall` | text | 0 | | | | No Constraint: 934; Space Constrained Only: 86; Capacity Constrained Only: 61; Capacity and Space Constrained: 10 |
| 74 | `mp5_panel_constraint_capacity` | bool | 0 | | | | {False: 1020, True: 71} |
| 75 | `mp5_panel_constraint_breaker_space` | bool | 0 | | | | {False: 995, True: 96} |
| 76 | `base_electricity_heating_consumption` | num | 0 | 0 | 0 | 56.17 | 2.664e+04 |
| 77 | `base_electricity_cooling_consumption` | num | 0 | 110.5 | 2,109 | 2,215 | 8,238 |
| 78 | `base_fuelOil_heating_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 79 | `base_naturalGas_heating_consumption` | num | 0 | 0 | 2.69e+04 | 2.972e+04 | 1.139e+05 |
| 80 | `base_propane_heating_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 81 | `baseline_heating_consumption` | num | 0 | 0 | 2.69e+04 | 2.977e+04 | 1.139e+05 |
| 82 | `baseline_cooling_consumption` | num | 0 | 110.5 | 2,109 | 2,215 | 8,238 |
| 83 | `mp5_heating_consumption` | num | 0 | 202.8 | 2,483 | 2,748 | 9,859 |
| 84 | `mp5_cooling_consumption` | num | 0 | 307.1 | 2,052 | 2,149 | 6,474 |
| 85 | `base_total_electricity_consumption` | num | 0 | 2,838 | 9,470 | 1.01e+04 | 4.037e+04 |
| 86 | `mp5_total_electricity_consumption` | num | 0 | 3,344 | 1.236e+04 | 1.3e+04 | 3.161e+04 |
| 87 | `baseline_total_site_consumption` | num | 0 | 1.045e+04 | 4.193e+04 | 4.5e+04 | 1.311e+05 |
| 88 | `base_electricity_heating_fansPumps_consumption` | num | 0 | 0 | 642.4 | 706.1 | 2,941 |
| 89 | `base_electricity_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 90 | `base_electricity_heating_hpBackupFans_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 91 | `base_naturalGas_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 92 | `base_propane_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 93 | `base_fuelOil_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 94 | `base_electricity_cooling_fansPumps_consumption` | num | 0 | 0 | 179.7 | 217.4 | 1,541 |
| 95 | `mp5_electricity_heating_fansPumps_consumption` | num | 0 | 0 | 228 | 250 | 961.6 |
| 96 | `mp5_electricity_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 97 | `mp5_electricity_heating_hpBackupFans_consumption` | num | 0 | 0 | 402.4 | 440.8 | 1,595 |
| 98 | `mp5_naturalGas_heating_hpBackup_consumption` | num | 0 | 0 | 1.734e+04 | 1.898e+04 | 7.034e+04 |
| 99 | `mp5_propane_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 100 | `mp5_fuelOil_heating_hpBackup_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 101 | `mp5_electricity_cooling_fansPumps_consumption` | num | 0 | 69.16 | 477.4 | 505.7 | 1,666 |
| 102 | `baseline_2025_heating_consumption` | num | 0 | 0 | 2.756e+04 | 3.048e+04 | 1.168e+05 |
| 103 | `baseline_2026_heating_consumption` | num | 0 | 0 | 2.688e+04 | 2.973e+04 | 1.139e+05 |
| 104 | `baseline_2027_heating_consumption` | num | 0 | 0 | 2.677e+04 | 2.96e+04 | 1.135e+05 |
| 105 | `baseline_2028_heating_consumption` | num | 0 | 0 | 2.666e+04 | 2.949e+04 | 1.13e+05 |
| 106 | `baseline_2029_heating_consumption` | num | 0 | 0 | 2.655e+04 | 2.937e+04 | 1.125e+05 |
| 107 | `baseline_2030_heating_consumption` | num | 0 | 0 | 2.644e+04 | 2.924e+04 | 1.121e+05 |
| 108 | `baseline_2031_heating_consumption` | num | 0 | 0 | 2.633e+04 | 2.912e+04 | 1.116e+05 |
| 109 | `baseline_2032_heating_consumption` | num | 0 | 0 | 2.622e+04 | 2.9e+04 | 1.111e+05 |
| 110 | `baseline_2033_heating_consumption` | num | 0 | 0 | 2.611e+04 | 2.888e+04 | 1.107e+05 |
| 111 | `baseline_2034_heating_consumption` | num | 0 | 0 | 2.601e+04 | 2.876e+04 | 1.102e+05 |
| 112 | `baseline_2035_heating_consumption` | num | 0 | 0 | 2.589e+04 | 2.864e+04 | 1.098e+05 |
| 113 | `baseline_2036_heating_consumption` | num | 0 | 0 | 2.579e+04 | 2.852e+04 | 1.093e+05 |
| 114 | `baseline_2037_heating_consumption` | num | 0 | 0 | 2.567e+04 | 2.84e+04 | 1.088e+05 |
| 115 | `baseline_2038_heating_consumption` | num | 0 | 0 | 2.557e+04 | 2.828e+04 | 1.084e+05 |
| 116 | `baseline_2039_heating_consumption` | num | 0 | 0 | 2.546e+04 | 2.816e+04 | 1.079e+05 |
| 117 | `baseline_2025_heating_electricity_consumption` | num | 0 | 0 | 645 | 762.2 | 2.664e+04 |
| 118 | `baseline_2026_heating_electricity_consumption` | num | 0 | 0 | 629.2 | 743.5 | 2.598e+04 |
| 119 | `baseline_2027_heating_electricity_consumption` | num | 0 | 0 | 626.5 | 740.4 | 2.588e+04 |
| 120 | `baseline_2028_heating_electricity_consumption` | num | 0 | 0 | 624 | 737.4 | 2.577e+04 |
| 121 | `baseline_2029_heating_electricity_consumption` | num | 0 | 0 | 621.5 | 734.5 | 2.567e+04 |
| 122 | `baseline_2030_heating_electricity_consumption` | num | 0 | 0 | 618.9 | 731.4 | 2.556e+04 |
| 123 | `baseline_2031_heating_electricity_consumption` | num | 0 | 0 | 616.4 | 728.4 | 2.546e+04 |
| 124 | `baseline_2032_heating_electricity_consumption` | num | 0 | 0 | 613.8 | 725.3 | 2.535e+04 |
| 125 | `baseline_2033_heating_electricity_consumption` | num | 0 | 0 | 611.3 | 722.3 | 2.525e+04 |
| 126 | `baseline_2034_heating_electricity_consumption` | num | 0 | 0 | 608.8 | 719.4 | 2.514e+04 |
| 127 | `baseline_2035_heating_electricity_consumption` | num | 0 | 0 | 606.1 | 716.3 | 2.503e+04 |
| 128 | `baseline_2036_heating_electricity_consumption` | num | 0 | 0 | 603.6 | 713.3 | 2.493e+04 |
| 129 | `baseline_2037_heating_electricity_consumption` | num | 0 | 0 | 601 | 710.2 | 2.482e+04 |
| 130 | `baseline_2038_heating_electricity_consumption` | num | 0 | 0 | 598.5 | 707.2 | 2.472e+04 |
| 131 | `baseline_2039_heating_electricity_consumption` | num | 0 | 0 | 596 | 704.3 | 2.461e+04 |
| 132 | `baseline_2025_heating_naturalGas_consumption` | num | 0 | 0 | 2.69e+04 | 2.972e+04 | 1.139e+05 |
| 133 | `baseline_2026_heating_naturalGas_consumption` | num | 0 | 0 | 2.623e+04 | 2.898e+04 | 1.111e+05 |
| 134 | `baseline_2027_heating_naturalGas_consumption` | num | 0 | 0 | 2.613e+04 | 2.886e+04 | 1.106e+05 |
| 135 | `baseline_2028_heating_naturalGas_consumption` | num | 0 | 0 | 2.602e+04 | 2.875e+04 | 1.102e+05 |
| 136 | `baseline_2029_heating_naturalGas_consumption` | num | 0 | 0 | 2.592e+04 | 2.863e+04 | 1.097e+05 |
| 137 | `baseline_2030_heating_naturalGas_consumption` | num | 0 | 0 | 2.581e+04 | 2.851e+04 | 1.092e+05 |
| 138 | `baseline_2031_heating_naturalGas_consumption` | num | 0 | 0 | 2.57e+04 | 2.84e+04 | 1.088e+05 |
| 139 | `baseline_2032_heating_naturalGas_consumption` | num | 0 | 0 | 2.559e+04 | 2.828e+04 | 1.083e+05 |
| 140 | `baseline_2033_heating_naturalGas_consumption` | num | 0 | 0 | 2.549e+04 | 2.816e+04 | 1.079e+05 |
| 141 | `baseline_2034_heating_naturalGas_consumption` | num | 0 | 0 | 2.538e+04 | 2.804e+04 | 1.075e+05 |
| 142 | `baseline_2035_heating_naturalGas_consumption` | num | 0 | 0 | 2.527e+04 | 2.792e+04 | 1.07e+05 |
| 143 | `baseline_2036_heating_naturalGas_consumption` | num | 0 | 0 | 2.517e+04 | 2.781e+04 | 1.066e+05 |
| 144 | `baseline_2037_heating_naturalGas_consumption` | num | 0 | 0 | 2.506e+04 | 2.769e+04 | 1.061e+05 |
| 145 | `baseline_2038_heating_naturalGas_consumption` | num | 0 | 0 | 2.496e+04 | 2.757e+04 | 1.056e+05 |
| 146 | `baseline_2039_heating_naturalGas_consumption` | num | 0 | 0 | 2.485e+04 | 2.746e+04 | 1.052e+05 |
| 147 | `baseline_2025_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 148 | `baseline_2026_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 149 | `baseline_2027_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 150 | `baseline_2028_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 151 | `baseline_2029_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 152 | `baseline_2030_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 153 | `baseline_2031_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 154 | `baseline_2032_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 155 | `baseline_2033_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 156 | `baseline_2034_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 157 | `baseline_2035_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 158 | `baseline_2036_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 159 | `baseline_2037_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 160 | `baseline_2038_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 161 | `baseline_2039_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 162 | `baseline_2025_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 163 | `baseline_2026_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 164 | `baseline_2027_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 165 | `baseline_2028_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 166 | `baseline_2029_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 167 | `baseline_2030_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 168 | `baseline_2031_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 169 | `baseline_2032_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 170 | `baseline_2033_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 171 | `baseline_2034_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 172 | `baseline_2035_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 173 | `baseline_2036_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 174 | `baseline_2037_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 175 | `baseline_2038_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 176 | `baseline_2039_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 177 | `ref2025_mp5_2025_heating_consumption` | num | 0 | 202.8 | 2.054e+04 | 2.242e+04 | 8.129e+04 |
| 178 | `ref2025_mp5_2026_heating_consumption` | num | 0 | 197.8 | 2.003e+04 | 2.187e+04 | 7.929e+04 |
| 179 | `ref2025_mp5_2027_heating_consumption` | num | 0 | 197 | 1.995e+04 | 2.178e+04 | 7.896e+04 |
| 180 | `ref2025_mp5_2028_heating_consumption` | num | 0 | 196.2 | 1.987e+04 | 2.169e+04 | 7.864e+04 |
| 181 | `ref2025_mp5_2029_heating_consumption` | num | 0 | 195.4 | 1.979e+04 | 2.16e+04 | 7.832e+04 |
| 182 | `ref2025_mp5_2030_heating_consumption` | num | 0 | 194.6 | 1.97e+04 | 2.151e+04 | 7.799e+04 |
| 183 | `ref2025_mp5_2031_heating_consumption` | num | 0 | 193.8 | 1.962e+04 | 2.142e+04 | 7.768e+04 |
| 184 | `ref2025_mp5_2032_heating_consumption` | num | 0 | 193 | 1.954e+04 | 2.133e+04 | 7.735e+04 |
| 185 | `ref2025_mp5_2033_heating_consumption` | num | 0 | 192.2 | 1.946e+04 | 2.125e+04 | 7.703e+04 |
| 186 | `ref2025_mp5_2034_heating_consumption` | num | 0 | 191.4 | 1.938e+04 | 2.116e+04 | 7.671e+04 |
| 187 | `ref2025_mp5_2035_heating_consumption` | num | 0 | 190.6 | 1.93e+04 | 2.107e+04 | 7.638e+04 |
| 188 | `ref2025_mp5_2036_heating_consumption` | num | 0 | 189.8 | 1.922e+04 | 2.098e+04 | 7.607e+04 |
| 189 | `ref2025_mp5_2037_heating_consumption` | num | 0 | 189 | 1.913e+04 | 2.089e+04 | 7.574e+04 |
| 190 | `ref2025_mp5_2038_heating_consumption` | num | 0 | 188.2 | 1.905e+04 | 2.08e+04 | 7.542e+04 |
| 191 | `ref2025_mp5_2039_heating_consumption` | num | 0 | 187.4 | 1.897e+04 | 2.071e+04 | 7.51e+04 |
| 192 | `ref2025_mp5_2025_heating_electricity_consumption` | num | 0 | 202.8 | 3,131 | 3,438 | 1.207e+04 |
| 193 | `ref2025_mp5_2026_heating_electricity_consumption` | num | 0 | 197.8 | 3,054 | 3,354 | 1.177e+04 |
| 194 | `ref2025_mp5_2027_heating_electricity_consumption` | num | 0 | 197 | 3,041 | 3,340 | 1.172e+04 |
| 195 | `ref2025_mp5_2028_heating_electricity_consumption` | num | 0 | 196.2 | 3,029 | 3,326 | 1.167e+04 |
| 196 | `ref2025_mp5_2029_heating_electricity_consumption` | num | 0 | 195.4 | 3,017 | 3,313 | 1.163e+04 |
| 197 | `ref2025_mp5_2030_heating_electricity_consumption` | num | 0 | 194.6 | 3,004 | 3,299 | 1.158e+04 |
| 198 | `ref2025_mp5_2031_heating_electricity_consumption` | num | 0 | 193.8 | 2,992 | 3,286 | 1.153e+04 |
| 199 | `ref2025_mp5_2032_heating_electricity_consumption` | num | 0 | 193 | 2,979 | 3,272 | 1.148e+04 |
| 200 | `ref2025_mp5_2033_heating_electricity_consumption` | num | 0 | 192.2 | 2,967 | 3,258 | 1.144e+04 |
| 201 | `ref2025_mp5_2034_heating_electricity_consumption` | num | 0 | 191.4 | 2,955 | 3,245 | 1.139e+04 |
| 202 | `ref2025_mp5_2035_heating_electricity_consumption` | num | 0 | 190.6 | 2,942 | 3,231 | 1.134e+04 |
| 203 | `ref2025_mp5_2036_heating_electricity_consumption` | num | 0 | 189.8 | 2,930 | 3,218 | 1.129e+04 |
| 204 | `ref2025_mp5_2037_heating_electricity_consumption` | num | 0 | 189 | 2,917 | 3,204 | 1.124e+04 |
| 205 | `ref2025_mp5_2038_heating_electricity_consumption` | num | 0 | 188.2 | 2,905 | 3,190 | 1.12e+04 |
| 206 | `ref2025_mp5_2039_heating_electricity_consumption` | num | 0 | 187.4 | 2,893 | 3,177 | 1.115e+04 |
| 207 | `ref2025_mp5_2025_heating_naturalGas_consumption` | num | 0 | 0 | 1.734e+04 | 1.898e+04 | 7.034e+04 |
| 208 | `ref2025_mp5_2026_heating_naturalGas_consumption` | num | 0 | 0 | 1.691e+04 | 1.851e+04 | 6.861e+04 |
| 209 | `ref2025_mp5_2027_heating_naturalGas_consumption` | num | 0 | 0 | 1.684e+04 | 1.844e+04 | 6.832e+04 |
| 210 | `ref2025_mp5_2028_heating_naturalGas_consumption` | num | 0 | 0 | 1.678e+04 | 1.836e+04 | 6.805e+04 |
| 211 | `ref2025_mp5_2029_heating_naturalGas_consumption` | num | 0 | 0 | 1.671e+04 | 1.829e+04 | 6.778e+04 |
| 212 | `ref2025_mp5_2030_heating_naturalGas_consumption` | num | 0 | 0 | 1.664e+04 | 1.821e+04 | 6.749e+04 |
| 213 | `ref2025_mp5_2031_heating_naturalGas_consumption` | num | 0 | 0 | 1.657e+04 | 1.814e+04 | 6.722e+04 |
| 214 | `ref2025_mp5_2032_heating_naturalGas_consumption` | num | 0 | 0 | 1.65e+04 | 1.806e+04 | 6.693e+04 |
| 215 | `ref2025_mp5_2033_heating_naturalGas_consumption` | num | 0 | 0 | 1.643e+04 | 1.799e+04 | 6.666e+04 |
| 216 | `ref2025_mp5_2034_heating_naturalGas_consumption` | num | 0 | 0 | 1.637e+04 | 1.791e+04 | 6.638e+04 |
| 217 | `ref2025_mp5_2035_heating_naturalGas_consumption` | num | 0 | 0 | 1.629e+04 | 1.784e+04 | 6.61e+04 |
| 218 | `ref2025_mp5_2036_heating_naturalGas_consumption` | num | 0 | 0 | 1.623e+04 | 1.776e+04 | 6.582e+04 |
| 219 | `ref2025_mp5_2037_heating_naturalGas_consumption` | num | 0 | 0 | 1.616e+04 | 1.769e+04 | 6.554e+04 |
| 220 | `ref2025_mp5_2038_heating_naturalGas_consumption` | num | 0 | 0 | 1.609e+04 | 1.761e+04 | 6.526e+04 |
| 221 | `ref2025_mp5_2039_heating_naturalGas_consumption` | num | 0 | 0 | 1.602e+04 | 1.754e+04 | 6.499e+04 |
| 222 | `ref2025_mp5_2025_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 223 | `ref2025_mp5_2026_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 224 | `ref2025_mp5_2027_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 225 | `ref2025_mp5_2028_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 226 | `ref2025_mp5_2029_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 227 | `ref2025_mp5_2030_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 228 | `ref2025_mp5_2031_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 229 | `ref2025_mp5_2032_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 230 | `ref2025_mp5_2033_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 231 | `ref2025_mp5_2034_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 232 | `ref2025_mp5_2035_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 233 | `ref2025_mp5_2036_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 234 | `ref2025_mp5_2037_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 235 | `ref2025_mp5_2038_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 236 | `ref2025_mp5_2039_heating_fuelOil_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 237 | `ref2025_mp5_2025_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 238 | `ref2025_mp5_2026_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 239 | `ref2025_mp5_2027_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 240 | `ref2025_mp5_2028_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 241 | `ref2025_mp5_2029_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 242 | `ref2025_mp5_2030_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 243 | `ref2025_mp5_2031_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 244 | `ref2025_mp5_2032_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 245 | `ref2025_mp5_2033_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 246 | `ref2025_mp5_2034_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 247 | `ref2025_mp5_2035_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 248 | `ref2025_mp5_2036_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 249 | `ref2025_mp5_2037_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 250 | `ref2025_mp5_2038_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 251 | `ref2025_mp5_2039_heating_propane_consumption` | num | 0 | 0 | 0 | 0 | 0 |
| 252 | `baseline_2025_cooling_consumption` | num | 0 | 110.5 | 2,303 | 2,433 | 8,957 |
| 253 | `baseline_2026_cooling_consumption` | num | 0 | 123.8 | 2,579 | 2,725 | 1.003e+04 |
| 254 | `baseline_2027_cooling_consumption` | num | 0 | 124.9 | 2,603 | 2,750 | 1.013e+04 |
| 255 | `baseline_2028_cooling_consumption` | num | 0 | 126.1 | 2,627 | 2,776 | 1.022e+04 |
| 256 | `baseline_2029_cooling_consumption` | num | 0 | 127.4 | 2,654 | 2,804 | 1.033e+04 |
| 257 | `baseline_2030_cooling_consumption` | num | 0 | 128.5 | 2,678 | 2,830 | 1.042e+04 |
| 258 | `baseline_2031_cooling_consumption` | num | 0 | 129.7 | 2,702 | 2,855 | 1.051e+04 |
| 259 | `baseline_2032_cooling_consumption` | num | 0 | 130.8 | 2,727 | 2,881 | 1.061e+04 |
| 260 | `baseline_2033_cooling_consumption` | num | 0 | 132 | 2,751 | 2,906 | 1.07e+04 |
| 261 | `baseline_2034_cooling_consumption` | num | 0 | 133.1 | 2,775 | 2,931 | 1.079e+04 |
| 262 | `baseline_2035_cooling_consumption` | num | 0 | 134.4 | 2,802 | 2,960 | 1.09e+04 |
| 263 | `baseline_2036_cooling_consumption` | num | 0 | 135.6 | 2,826 | 2,985 | 1.099e+04 |
| 264 | `baseline_2037_cooling_consumption` | num | 0 | 136.7 | 2,850 | 3,011 | 1.109e+04 |
| 265 | `baseline_2038_cooling_consumption` | num | 0 | 137.9 | 2,874 | 3,036 | 1.118e+04 |
| 266 | `baseline_2039_cooling_consumption` | num | 0 | 139 | 2,898 | 3,062 | 1.127e+04 |
| 267 | `baseline_2025_cooling_electricity_consumption` | num | 0 | 110.5 | 2,303 | 2,433 | 8,957 |
| 268 | `baseline_2026_cooling_electricity_consumption` | num | 0 | 123.8 | 2,579 | 2,725 | 1.003e+04 |
| 269 | `baseline_2027_cooling_electricity_consumption` | num | 0 | 124.9 | 2,603 | 2,750 | 1.013e+04 |
| 270 | `baseline_2028_cooling_electricity_consumption` | num | 0 | 126.1 | 2,627 | 2,776 | 1.022e+04 |
| 271 | `baseline_2029_cooling_electricity_consumption` | num | 0 | 127.4 | 2,654 | 2,804 | 1.033e+04 |
| 272 | `baseline_2030_cooling_electricity_consumption` | num | 0 | 128.5 | 2,678 | 2,830 | 1.042e+04 |
| 273 | `baseline_2031_cooling_electricity_consumption` | num | 0 | 129.7 | 2,702 | 2,855 | 1.051e+04 |
| 274 | `baseline_2032_cooling_electricity_consumption` | num | 0 | 130.8 | 2,727 | 2,881 | 1.061e+04 |
| 275 | `baseline_2033_cooling_electricity_consumption` | num | 0 | 132 | 2,751 | 2,906 | 1.07e+04 |
| 276 | `baseline_2034_cooling_electricity_consumption` | num | 0 | 133.1 | 2,775 | 2,931 | 1.079e+04 |
| 277 | `baseline_2035_cooling_electricity_consumption` | num | 0 | 134.4 | 2,802 | 2,960 | 1.09e+04 |
| 278 | `baseline_2036_cooling_electricity_consumption` | num | 0 | 135.6 | 2,826 | 2,985 | 1.099e+04 |
| 279 | `baseline_2037_cooling_electricity_consumption` | num | 0 | 136.7 | 2,850 | 3,011 | 1.109e+04 |
| 280 | `baseline_2038_cooling_electricity_consumption` | num | 0 | 137.9 | 2,874 | 3,036 | 1.118e+04 |
| 281 | `baseline_2039_cooling_electricity_consumption` | num | 0 | 139 | 2,898 | 3,062 | 1.127e+04 |
| 282 | `ref2025_mp5_2025_cooling_consumption` | num | 0 | 376.3 | 2,529 | 2,655 | 8,125 |
| 283 | `ref2025_mp5_2026_cooling_consumption` | num | 0 | 421.5 | 2,832 | 2,974 | 9,101 |
| 284 | `ref2025_mp5_2027_cooling_consumption` | num | 0 | 425.4 | 2,859 | 3,002 | 9,185 |
| 285 | `ref2025_mp5_2028_cooling_consumption` | num | 0 | 429.4 | 2,885 | 3,029 | 9,270 |
| 286 | `ref2025_mp5_2029_cooling_consumption` | num | 0 | 433.8 | 2,915 | 3,060 | 9,366 |
| 287 | `ref2025_mp5_2030_cooling_consumption` | num | 0 | 437.7 | 2,941 | 3,088 | 9,451 |
| 288 | `ref2025_mp5_2031_cooling_consumption` | num | 0 | 441.6 | 2,968 | 3,116 | 9,536 |
| 289 | `ref2025_mp5_2032_cooling_consumption` | num | 0 | 445.6 | 2,994 | 3,144 | 9,620 |
| 290 | `ref2025_mp5_2033_cooling_consumption` | num | 0 | 449.5 | 3,020 | 3,171 | 9,705 |
| 291 | `ref2025_mp5_2034_cooling_consumption` | num | 0 | 453.4 | 3,047 | 3,199 | 9,790 |
| 292 | `ref2025_mp5_2035_cooling_consumption` | num | 0 | 457.9 | 3,077 | 3,230 | 9,886 |
| 293 | `ref2025_mp5_2036_cooling_consumption` | num | 0 | 461.8 | 3,103 | 3,258 | 9,970 |
| 294 | `ref2025_mp5_2037_cooling_consumption` | num | 0 | 465.7 | 3,129 | 3,286 | 1.006e+04 |
| 295 | `ref2025_mp5_2038_cooling_consumption` | num | 0 | 469.6 | 3,156 | 3,313 | 1.014e+04 |
| 296 | `ref2025_mp5_2039_cooling_consumption` | num | 0 | 473.6 | 3,182 | 3,341 | 1.022e+04 |
| 297 | `ref2025_mp5_2025_cooling_electricity_consumption` | num | 0 | 376.3 | 2,529 | 2,655 | 8,125 |
| 298 | `ref2025_mp5_2026_cooling_electricity_consumption` | num | 0 | 421.5 | 2,832 | 2,974 | 9,101 |
| 299 | `ref2025_mp5_2027_cooling_electricity_consumption` | num | 0 | 425.4 | 2,859 | 3,002 | 9,185 |
| 300 | `ref2025_mp5_2028_cooling_electricity_consumption` | num | 0 | 429.4 | 2,885 | 3,029 | 9,270 |
| 301 | `ref2025_mp5_2029_cooling_electricity_consumption` | num | 0 | 433.8 | 2,915 | 3,060 | 9,366 |
| 302 | `ref2025_mp5_2030_cooling_electricity_consumption` | num | 0 | 437.7 | 2,941 | 3,088 | 9,451 |
| 303 | `ref2025_mp5_2031_cooling_electricity_consumption` | num | 0 | 441.6 | 2,968 | 3,116 | 9,536 |
| 304 | `ref2025_mp5_2032_cooling_electricity_consumption` | num | 0 | 445.6 | 2,994 | 3,144 | 9,620 |
| 305 | `ref2025_mp5_2033_cooling_electricity_consumption` | num | 0 | 449.5 | 3,020 | 3,171 | 9,705 |
| 306 | `ref2025_mp5_2034_cooling_electricity_consumption` | num | 0 | 453.4 | 3,047 | 3,199 | 9,790 |
| 307 | `ref2025_mp5_2035_cooling_electricity_consumption` | num | 0 | 457.9 | 3,077 | 3,230 | 9,886 |
| 308 | `ref2025_mp5_2036_cooling_electricity_consumption` | num | 0 | 461.8 | 3,103 | 3,258 | 9,970 |
| 309 | `ref2025_mp5_2037_cooling_electricity_consumption` | num | 0 | 465.7 | 3,129 | 3,286 | 1.006e+04 |
| 310 | `ref2025_mp5_2038_cooling_electricity_consumption` | num | 0 | 469.6 | 3,156 | 3,313 | 1.014e+04 |
| 311 | `ref2025_mp5_2039_cooling_electricity_consumption` | num | 0 | 473.6 | 3,182 | 3,341 | 1.022e+04 |
| 312 | `baseline_heating_lifetime_fuel_cost` | num | 0 | 0 | 2.017e+04 | 2.235e+04 | 8.569e+04 |
| 313 | `ref2025_mp5_heating_lifetime_fuel_cost` | num | 0 | 570 | 2.058e+04 | 2.257e+04 | 7.859e+04 |
| 314 | `ref2025_mp5_heating_lifetime_savings_fuel_cost` | num | 0 | -1.431e+04 | -336.3 | -222 | 4.209e+04 |
| 315 | `baseline_cooling_lifetime_fuel_cost` | num | 0 | 383.9 | 8,002 | 8,454 | 3.113e+04 |
| 316 | `ref2025_mp5_cooling_lifetime_fuel_cost` | num | 0 | 1,308 | 8,787 | 9,226 | 2.823e+04 |
| 317 | `ref2025_mp5_cooling_lifetime_savings_fuel_cost` | num | 0 | -1.983e+04 | 657.9 | -771.8 | 1.17e+04 |
| 318 | `ref2025_mp5_cooling_lifetime_savings_negative` | bool | 0 | | | | {False: 654, True: 437} |
| 319 | `mp5_heating_replacement_installed_cost_v4MID` | num | 0 | 3,119 | 3,712 | 3,766 | 6,274 |
| 320 | `mp5_heating_upgrade_installed_cost_v4MID` | num | 0 | 1.089e+04 | 1.398e+04 | 1.442e+04 | 2.591e+04 |
| 321 | `mp5_heating_backupFurnace_installed_cost_v4MID` | num | 0 | 3,804 | 4,201 | 4,237 | 5,359 |
| 322 | `mp5_cooling_replacement_installed_cost_v4MID` | num | 0 | 492.1 | 5,665 | 4,724 | 9,559 |
| 323 | `mp5_cooling_replacement_credit_applied_v4MID` | num | 0 | 492.1 | 5,665 | 4,724 | 9,559 |
| 324 | `mp5_heating_rebate_amount_june2026_v4MID` | num | 0 | 0 | 6,644 | 4,812 | 8,000 |
| 325 | `mp5_rebate_eligibility_june2026` | text | 0 | | | | HEEHR: 665; Not Eligible: 311; HOMES: 115 |
| 326 | `mp5_modeled_savings_frac` | num | 0 | -0.1023 | 0.1652 | 0.1632 | 0.4207 |
| 327 | `public_discount_rate` | num | 0 | 0.02 | 0.02 | 0.02 | 0.02 |
| 328 | `private_discount_rate_fixed_base` | num | 0 | 0.07 | 0.07 | 0.07 | 0.07 |
| 329 | `ref2025_mp5_heating_discounted_lifetime_savings_fixed_base` | num | 0 | -9,299 | -212.3 | -137 | 2.735e+04 |
| 330 | `ref2025_mp5_cooling_discounted_lifetime_savings_fixed_base` | num | 0 | -1.263e+04 | 419 | -491.5 | 7,448 |
| 331 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_net_capital_cost_v4MID` | num | 0 | 6,201 | 9,227 | 1.016e+04 | 2.446e+04 |
| 332 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_private_npv_fixed_base` | num | 0 | -3.586e+04 | -8,902 | -1.079e+04 | 2.172e+04 |
| 333 | `ref2025_mp5_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base` | num | 0 | 0 | 0 | 0.003666 | 1 |

#### `tepper_county_mp5_Allegheny_2026-10-06_19-09.csv`

- Rows: 1 counties; columns: 11; size 0.0 MB
- Sum of `home_count`: 277,009 homes (1,091 rdu)

| # | Column | Type | Blanks | Min | Median | Mean | Max / values |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `county` | text | 0 | | | | G4200030: 1 |
| 2 | `state` | text | 0 | | | | PA: 1 |
| 3 | `home_count` | num | 0 | 2.77e+05 | 2.77e+05 | 2.77e+05 | 2.77e+05 |
| 4 | `adoption_rate_pct` | num | 0 | 0.37 | 0.37 | 0.37 | 0.37 |
| 5 | `operating_cost_pct_change` | num | 0 | 1.76 | 1.76 | 1.76 | 1.76 |
| 6 | `baseline_elec_gwh` | num | 0 | 2,799 | 2,799 | 2,799 | 2,799 |
| 7 | `retrofit_elec_gwh` | num | 0 | 3,602 | 3,602 | 3,602 | 3,602 |
| 8 | `elec_change_gwh` | num | 0 | 802.9 | 802.9 | 802.9 | 802.9 |
| 9 | `site_energy_change_gwh` | num | 0 | -2,172 | -2,172 | -2,172 | -2,172 |
| 10 | `pct_elec_demand_change` | num | 0 | 28.69 | 28.69 | 28.69 | 28.69 |
| 11 | `pct_site_energy_change` | num | 0 | -17.43 | -17.43 | -17.43 | -17.43 |


## Reproducing the files on the researcher's Windows machine (Git Bash)

The session could not commit or push (the cloud environment refused `git commit`), so the
changes reach the branch only once they are applied and committed by hand. Each change is
a numbered patch under `cmu_tare_model/docs/cloud_run/patches/`, sent to the researcher
with this review in `cloud_run_bundle.tar.gz`. All fifteen patches reproduce the final files; patches `01` to `05` alone reproduce the
Stage A files.

```bash
# 1. The main folder: the repository root that holds config.py
cd "/c/path/to/cmu-tare-model"

# 2. The cloud branch, as it is on GitHub
git fetch origin cloud/resstock2025-1-mp5
git checkout cloud/resstock2025-1-mp5
git pull origin cloud/resstock2025-1-mp5

# 3. Unpack the bundle here (it holds cmu_tare_model/docs/cloud_run/), then apply the
#    patches in number order; the loop stops at the first that does not apply cleanly
tar -xzf ~/Downloads/cloud_run_bundle.tar.gz
for p in cmu_tare_model/docs/cloud_run/patches/*.patch; do
  git apply --check "$p" && git apply "$p" && echo "applied $p" || { echo "STOPPED at $p"; break; }
done

# 4. The two ResStock 2025.1 files the loader reads (already on this machine)
ls -l data/resstock_2025_1/upgrade0.parquet data/resstock_2025_1/upgrade5.parquet

# 5. The project environment
source /c/Users/jorda/AppData/Local/anaconda3/etc/profile.d/conda.sh
conda activate cmu-tare-model

# 6. The run: Y to a new run, N for National, grid impact off (no AWS needed).
#    About 12 minutes and 11 GB of memory on the cloud machine.
python scripts/run_tare_notebooks.py --release 2025.1 --skip-grid-impact --log run_2025_1_national.log

# 7. The files land here, named tepper_*_mp5_{National|Allegheny}_{run time}.csv
ls -l cmu_tare_model/output_results/tepper_export/
```

The runner prints `[OK] All N checks passed.` and exits 0 when the run and every check
passed; any other exit code names the notebook, the cell and the error. Without
`--skip-grid-impact` it leaves grid impact on and answers 42003 to the FIPS question (AWS
credentials needed). `--state PA` runs one state instead of the whole country.
