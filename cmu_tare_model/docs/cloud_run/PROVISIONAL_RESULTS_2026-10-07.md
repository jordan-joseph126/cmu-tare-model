# PROVISIONAL results -- ResStock 2025.1, Dual Fuel Heating System (Upgrade 05, mp=5)

**PROVISIONAL. These are cloud-session sanity numbers, not reference values.** No row is
added to `REFERENCE_VALUES.md`; that happens only from a dev-branch run the researcher
makes. Counts are in representative dwelling units (rdu) and homes, with homes = rdu x
the weight read from the frame (253.90367272727272). Dollars are 2025 dollars.

**Program Notice 26-3.** DOE has released new guidance in Program Notice 26-3. The
researcher has not yet reviewed how it differs from Program Notices 26-1 and 26-2 or how
it affects this code. The rebate rules modeled here are the ones documented in CLAUDE.md.

**Results that depend on a PROVISIONAL decision** (see `DECISIONS_TAKEN.md`): every
cost, NPV and adoption number after Stage A depends on P1 (SEER1 = SEER2 / 0.95) and P5
(furnace not in the rebate base); the June 2026 numbers depend on R3 as implemented in
D-S9.

## Runs

All national, `python scripts/run_tare_notebooks.py --release 2025.1 --skip-grid-impact`
(Y / N, grid impact off), 4 vCPU / 16 GB cloud machine. Every run passed all 20 checks
after the run: NPV identity 0 violations in all nine cases; CLAUDE.md's per-home and
county orderings 0 violations; adopter = NPV >= 0; adopter non-blank count = 161,983 =
the `include_sample` count (the adoption denominator); no blank energy, cost or NPV value
in a sample home.

| Run | Patches | Stamp | Wall time | Peak memory (OS record) | Where the peak happens |
|---|---|---|---:|---:|---|
| Stage A | 01-05 | `2026-10-06_18-19` | 11.4 min | 10.68 GB | main notebook cell 23 (Tepper export) |
| After capital cost (G5-G7) | 01-09 | `2026-10-06_18-47` | 11.6 min | 10.71 GB | main notebook cell 23 |
| After rebates (G8) | 01-11 | `2026-10-06_18-58` | 10.6 min | 10.70 GB | main notebook cell 23 |
| Final | 01-15 | `2026-10-06_19-09` | 10.6 min | 10.88 GB | main notebook cell 23 |

The peak is the Tepper cell, which holds the household frame, the per-year consumption
frame and the joined export together. It leaves about 5 GB free on a 16 GB machine.

`include_heating` rdu (applied, occupied, single-family, outside AK/HI): 177,000
(44,940,950 homes), beside the study sample of 161,983 (41,128,079 homes).

## Stage A beside each fix

| | Stage A | After capital cost | After rebates | Final |
|---|---:|---:|---:|---:|
| Run | `2026-10-06_18-19` | `2026-10-06_18-47` | `2026-10-06_18-58` | `2026-10-06_19-09` |
| Study sample (rdu) | 161,983 | 161,983 | 161,983 | 161,983 |
| Heat-pump pm2 fed to the regression | 15.2 (161,983 rdu) | 16.0 (161,983 rdu) | 16.0 (161,983 rdu) | 16.0 (161,983 rdu) |
| Mean heat-pump upgrade cost | $14,957 | $15,711 | $15,711 | $15,711 |
| Mean backup furnace cost | not priced | $4,112 | $4,112 | $4,112 |
| Mean heating replacement cost | $3,742 | $3,742 | $3,742 | $3,742 |
| Mean cooling replacement cost | $5,843 | $5,843 | $5,843 | $5,843 |
| Mean total capital (2024 rebate netted) | $10,082 | $14,864 | $14,864 | $14,864 |
| Mean lifetime heating fuel savings | $1,072 | $1,072 | $1,072 | $1,072 |
| Mean lifetime cooling fuel savings | $1,353 | $1,353 | $1,353 | $1,353 |
| Rebate total, 2024 guidance (weighted) | $200,525,078,611 to 121,432 rdu | $203,964,161,778 to 121,432 rdu | $203,964,161,778 to 121,432 rdu | $203,964,161,778 to 121,432 rdu |
| Rebate total, June 2026 guidance (weighted) | $10,269,636,836 to 5,732 rdu | $10,419,949,958 to 5,732 rdu | $203,964,161,354 to 121,432 rdu | $203,964,161,354 to 121,432 rdu |
| Adoption heatingSavings_coolingLCC_unsub | 3.96% | 1.78% | 1.78% | 1.78% |
| Adoption heatingSavings_coolingLCC_sub | 38.78% | 6.30% | 6.30% | 6.30% |
| Adoption heatingSavings_coolingLCC_sub_june2026 | 5.69% | 2.91% | 6.30% | 6.30% |
| Adoption heatingLCC_coolingSavings_unsub | 2.54% | 1.36% | 1.36% | 1.36% |
| Adoption heatingLCC_coolingSavings_sub | 16.30% | 3.10% | 3.10% | 3.10% |
| Adoption heatingLCC_coolingSavings_sub_june2026 | 3.73% | 2.10% | 3.10% | 3.10% |
| Adoption heatingLCC_coolingLCC_unsub | 11.19% | 3.27% | 3.27% | 3.27% |
| Adoption heatingLCC_coolingLCC_sub | 60.23% | 29.04% | 29.04% | 29.04% |
| Adoption heatingLCC_coolingLCC_sub_june2026 | 12.42% | 4.71% | 29.04% | 29.04% |
| Mean NPV heatingLCC_coolingLCC_unsub | $-3,837 | $-8,703 | $-8,703 | $-8,703 |
| Mean NPV heatingLCC_coolingLCC_sub | $1,039 | $-3,744 | $-3,744 | $-3,744 |
| Mean NPV heatingLCC_coolingLCC_sub_june2026 | $-3,587 | $-8,450 | $-3,744 | $-3,744 |

How to read it:

- **G6 (SEER2 to SEER1):** the heat-pump cost regression now gets SEER1 16.0 for every
  sample home (15.2 before). Mean heat-pump cost +$754.07 ($14,957.16 to $15,711.23), the
  expected 0.8 x $594.74 x 1.5 x the 2023-to-2025 CPI ratio.
- **G7 (backup furnace):** mean furnace cost $4,112.21 (preview $4,112), no blank, AFUE
  fed 0.925 (93,501 rdu) or 0.95 (68,482 rdu). With G6, mean total capital +$4,782
  ($10,082 to $14,864; the 2024 rebate rises a little with the higher heat-pump cost,
  since moderate-income HEEHR covers 50% of it). Adoption falls in every case, for
  example heatingLCC_coolingLCC_unsub 11.19% to 3.27%.
- **G8 (June 2026 rule for dual fuel):** only the `_sub_june2026` cases move, up to
  exactly the `_sub` values. For this package the R3 rule (HEEHR with no fuel gate, HOMES
  fuel-neutral) is the 2024 rule, so the two vintages pay the same homes the same
  amounts; the $424 national difference is the per-vintage rounding of half-cent HEEHR
  shares (`heehr_python_round`). The researcher may want to confirm this is the intended
  reading of R3.
- **G4, export columns, NB4, small items:** no modeled value moves (final = after
  rebates).

## G8: June 2026 rebates by program and baseline fuel, before and after

Before (Stage A rule, fuel gates applied to the dual-fuel package):

| Program | Baseline fuel | rdu | Homes | Total $ (weighted) |
|---|---|---:|---:|---:|
| HEEHR | Electricity | 5,163 | 1,310,905 | $9,955,811,897 |
| HOMES | Electricity | 569 | 144,471 | $313,824,939 |
| All | | 5,732 | 1,455,376 | $10,269,636,836 |

After (final): see the June 2026 table in the final section below -- 121,432 rdu
(30,832,031 homes), $203,964,161,354, of which natural-gas baselines receive HEEHR
$179.06 billion (92,618 rdu) and HOMES $12.11 billion (21,568 rdu). South Dakota $0 in
both. The 2024 columns did not change with G8 (after capital cost = after rebates).

## Final run in detail (`2026-10-06_19-09`)

Weight read from the frame: `253.90367272727272` per rdu (one distinct value).

Study sample: **161,983 rdu = 41,128,079 homes**. `include_heating` rdu (in the applied, occupied, single-family, non-AK/HI frame): 177,000.

**Sample funnel** (rdu; homes = rdu x weight)

| Step | rdu | Homes | Removed rdu |
|---|---:|---:|---:|
| load | 549,971 | 139,639,657 |  |
| applicability | 245,570 | 62,351,125 | 304,401 |
| occupancy | 221,752 | 56,303,647 | 23,818 |
| housing_type | 190,630 | 48,401,657 | 31,122 |
| exclude_AK_HI | 190,374 | 48,336,658 | 256 |
| heating_fuel | 190,374 | 48,336,658 | 0 |
| no_existing_heat_pump | 180,693 | 45,878,616 | 9,681 |
| replaceable_heating_system | 177,000 | 44,940,950 | 3,693 |
| central_or_room_ac | 161,983 | 41,128,079 | 15,017 |
| no_shared_cooling | 161,983 | 41,128,079 | 0 |

**Sample by baseline heating fuel and by cooling type**

| Group | rdu | Homes | Share |
|---|---:|---:|---:|
| Fuel: Natural Gas | 151,359 | 38,430,606 | 93.44% |
| Fuel: Electricity | 8,778 | 2,228,766 | 5.42% |
| Fuel: Fuel Oil | 1,423 | 361,305 | 0.88% |
| Fuel: Propane | 423 | 107,401 | 0.26% |
| Cooling: Central AC | 145,871 | 37,037,183 | 90.05% |
| Cooling: Room AC | 16,112 | 4,090,896 | 9.95% |

**Heating energy in 2025, study sample (weighted GWh, every part)**

| Fuel | Baseline | After retrofit |
|---|---:|---:|
| electricity | 37,748 | 116,668 |
| naturalGas | 829,286 | 507,984 |
| propane | 2,541 | 0 |
| fuelOil | 12,544 | 0 |
| **All fuels** | 882,119 | 624,652 |

**Mean fuel costs and savings per home (2025 dollars; lifetime = 15 years, undiscounted sum)**

| Quantity | Heating | Cooling |
|---|---:|---:|
| Baseline average annual fuel cost | $1,101 | $861 |
| Retrofit average annual fuel cost | $1,029 | $770 |
| Baseline lifetime fuel cost | $16,508 | $12,909 |
| Retrofit lifetime fuel cost | $15,436 | $11,556 |
| Lifetime fuel savings | $1,072 | $1,353 |
| Discounted lifetime savings (7%) | $668 | $867 |
| Share of homes with negative lifetime savings | 47.52% | 21.72% |

Negative cooling savings by cooling type: Central AC 13.42%, Room AC 96.80%

**Mean installed costs per home (2025 dollars)**

| Cost | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Heat-pump upgrade cost | $15,711 | $15,024 | $10,603 | $95,979 |
| Backup furnace cost | $4,112 | $4,069 | $3,646 | $6,567 |
| Heating replacement cost (avoided) | $3,742 | $3,656 | $271 | $23,259 |
| Cooling replacement cost (avoided) | $5,843 | $6,075 | $478 | $31,305 |
| Total capital cost (2024 rebate netted) | $14,864 | $13,720 | $6,309 | $93,428 |

Efficiency fed to the heat-pump cost regression (`heating_upgrade_pm2_euss`): {16.0: 161983}

AFUE fed to the furnace cost regression: {0.925: 93501, 0.95: 68482}

**Rebates by program and baseline fuel (weighted dollars; rdu receiving)**

*December 2024 guidance*

| Program | Baseline fuel | rdu | Homes | Mean per receiving home | Total $ (weighted) |
|---|---|---:|---:|---:|---:|
| HEEHR | Electricity | 5,163 | 1,310,905 | $7,709 | $10,106,125,065 |
| HEEHR | Fuel Oil | 858 | 217,849 | $7,588 | $1,653,008,486 |
| HEEHR | Natural Gas | 92,618 | 23,516,050 | $7,614 | $179,056,035,023 |
| HEEHR | Propane | 256 | 64,999 | $7,622 | $495,449,770 |
| HOMES | Electricity | 569 | 144,471 | $2,172 | $313,824,939 |
| HOMES | Fuel Oil | 332 | 84,296 | $2,229 | $187,888,718 |
| HOMES | Natural Gas | 21,568 | 5,476,194 | $2,211 | $12,110,189,574 |
| HOMES | Propane | 68 | 17,265 | $2,412 | $41,640,202 |
| **All** | | 121,432 | 30,832,031 | $6,615 | $203,964,161,778 |

South Dakota rebate dollars: $0.

*June 2026 guidance*

| Program | Baseline fuel | rdu | Homes | Mean per receiving home | Total $ (weighted) |
|---|---|---:|---:|---:|---:|
| HEEHR | Electricity | 5,163 | 1,310,905 | $7,709 | $10,106,125,019 |
| HEEHR | Fuel Oil | 858 | 217,849 | $7,588 | $1,653,008,494 |
| HEEHR | Natural Gas | 92,618 | 23,516,050 | $7,614 | $179,056,034,630 |
| HEEHR | Propane | 256 | 64,999 | $7,622 | $495,449,778 |
| HOMES | Electricity | 569 | 144,471 | $2,172 | $313,824,939 |
| HOMES | Fuel Oil | 332 | 84,296 | $2,229 | $187,888,718 |
| HOMES | Natural Gas | 21,568 | 5,476,194 | $2,211 | $12,110,189,574 |
| HOMES | Propane | 68 | 17,265 | $2,412 | $41,640,202 |
| **All** | | 121,432 | 30,832,031 | $6,615 | $203,964,161,354 |

South Dakota rebate dollars: $0.

**NPV and economic adoption, nine cases (7% discount rate; adoption = adopter rdu / study-sample rdu)**

| NPV case | Mean NPV | Median NPV | Adopters (rdu) | Adopters (homes) | Adoption rate |
|---|---:|---:|---:|---:|---:|
| heatingSavings_coolingLCC_unsub | $-12,445 | $-11,597 | 2,878 | 730,735 | 1.78% |
| heatingSavings_coolingLCC_sub | $-7,486 | $-6,400 | 10,207 | 2,591,595 | 6.30% |
| heatingSavings_coolingLCC_sub_june2026 | $-7,486 | $-6,400 | 10,207 | 2,591,595 | 6.30% |
| heatingLCC_coolingSavings_unsub | $-14,546 | $-14,125 | 2,206 | 560,112 | 1.36% |
| heatingLCC_coolingSavings_sub | $-9,587 | $-8,715 | 5,023 | 1,275,358 | 3.10% |
| heatingLCC_coolingSavings_sub_june2026 | $-9,587 | $-8,715 | 5,023 | 1,275,358 | 3.10% |
| heatingLCC_coolingLCC_unsub | $-8,703 | $-7,941 | 5,294 | 1,344,166 | 3.27% |
| heatingLCC_coolingLCC_sub | $-3,744 | $-2,598 | 47,045 | 11,944,898 | 29.04% |
| heatingLCC_coolingLCC_sub_june2026 | $-3,744 | $-2,598 | 47,045 | 11,944,898 | 29.04% |

**Climate (lifetime, central SCC, long-run marginal emissions; 2025 dollars)**

| Quantity | Heating | Cooling |
|---|---:|---:|
| Avoided CO2e (t) | 19.64 | 2.21 |
| Avoided climate damages ($) | 5,559.80 | 604.22 |
| Baseline lifetime damages ($) | 20,055.90 | 5,297.25 |
| Retrofit lifetime damages ($) | 14,496.10 | 4,693.04 |

Sample rdu with blank climate values (no GEA region): 0.


## Stage A run in detail (`2026-10-06_18-19`, before the cost and rebate fixes)

Weight read from the frame: `253.90367272727272` per rdu (one distinct value).

Study sample: **161,983 rdu = 41,128,079 homes**. `include_heating` rdu (in the applied, occupied, single-family, non-AK/HI frame): 177,000.

**Sample funnel** (rdu; homes = rdu x weight)

| Step | rdu | Homes | Removed rdu |
|---|---:|---:|---:|
| load | 549,971 | 139,639,657 |  |
| applicability | 245,570 | 62,351,125 | 304,401 |
| occupancy | 221,752 | 56,303,647 | 23,818 |
| housing_type | 190,630 | 48,401,657 | 31,122 |
| exclude_AK_HI | 190,374 | 48,336,658 | 256 |
| heating_fuel | 190,374 | 48,336,658 | 0 |
| no_existing_heat_pump | 180,693 | 45,878,616 | 9,681 |
| replaceable_heating_system | 177,000 | 44,940,950 | 3,693 |
| central_or_room_ac | 161,983 | 41,128,079 | 15,017 |
| no_shared_cooling | 161,983 | 41,128,079 | 0 |

**Sample by baseline heating fuel and by cooling type**

| Group | rdu | Homes | Share |
|---|---:|---:|---:|
| Fuel: Natural Gas | 151,359 | 38,430,606 | 93.44% |
| Fuel: Electricity | 8,778 | 2,228,766 | 5.42% |
| Fuel: Fuel Oil | 1,423 | 361,305 | 0.88% |
| Fuel: Propane | 423 | 107,401 | 0.26% |
| Cooling: Central AC | 145,871 | 37,037,183 | 90.05% |
| Cooling: Room AC | 16,112 | 4,090,896 | 9.95% |

**Heating energy in 2025, study sample (weighted GWh, every part)**

| Fuel | Baseline | After retrofit |
|---|---:|---:|
| electricity | 37,748 | 116,668 |
| naturalGas | 829,286 | 507,984 |
| propane | 2,541 | 0 |
| fuelOil | 12,544 | 0 |
| **All fuels** | 882,119 | 624,652 |

**Mean fuel costs and savings per home (2025 dollars; lifetime = 15 years, undiscounted sum)**

| Quantity | Heating | Cooling |
|---|---:|---:|
| Baseline average annual fuel cost | $1,101 | $861 |
| Retrofit average annual fuel cost | $1,029 | $770 |
| Baseline lifetime fuel cost | $16,508 | $12,909 |
| Retrofit lifetime fuel cost | $15,436 | $11,556 |
| Lifetime fuel savings | $1,072 | $1,353 |
| Discounted lifetime savings (7%) | $668 | $867 |
| Share of homes with negative lifetime savings | 47.52% | 21.72% |

Negative cooling savings by cooling type: Central AC 13.42%, Room AC 96.80%

**Mean installed costs per home (2025 dollars)**

| Cost | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Heat-pump upgrade cost | $14,957 | $14,270 | $9,849 | $95,225 |
| Backup furnace cost | not in this run | | | |
| Heating replacement cost (avoided) | $3,742 | $3,656 | $271 | $23,259 |
| Cooling replacement cost (avoided) | $5,843 | $6,075 | $478 | $31,305 |
| Total capital cost (2024 rebate netted) | $10,082 | $8,801 | $1,873 | $87,225 |

Efficiency fed to the heat-pump cost regression (`heating_upgrade_pm2_euss`): {15.2: 161983}

**Rebates by program and baseline fuel (weighted dollars; rdu receiving)**

*December 2024 guidance*

| Program | Baseline fuel | rdu | Homes | Mean per receiving home | Total $ (weighted) |
|---|---|---:|---:|---:|---:|
| HEEHR | Electricity | 5,163 | 1,310,905 | $7,595 | $9,955,811,912 |
| HEEHR | Fuel Oil | 858 | 217,849 | $7,435 | $1,619,689,098 |
| HEEHR | Natural Gas | 92,618 | 23,516,050 | $7,476 | $175,809,824,466 |
| HEEHR | Propane | 256 | 64,999 | $7,480 | $486,209,700 |
| HOMES | Electricity | 569 | 144,471 | $2,172 | $313,824,939 |
| HOMES | Fuel Oil | 332 | 84,296 | $2,229 | $187,888,718 |
| HOMES | Natural Gas | 21,568 | 5,476,194 | $2,211 | $12,110,189,574 |
| HOMES | Propane | 68 | 17,265 | $2,412 | $41,640,202 |
| **All** | | 121,432 | 30,832,031 | $6,504 | $200,525,078,611 |

South Dakota rebate dollars: $0.

*June 2026 guidance*

| Program | Baseline fuel | rdu | Homes | Mean per receiving home | Total $ (weighted) |
|---|---|---:|---:|---:|---:|
| HEEHR | Electricity | 5,163 | 1,310,905 | $7,595 | $9,955,811,897 |
| HOMES | Electricity | 569 | 144,471 | $2,172 | $313,824,939 |
| **All** | | 5,732 | 1,455,376 | $7,056 | $10,269,636,836 |

South Dakota rebate dollars: $0.

**NPV and economic adoption, nine cases (7% discount rate; adoption = adopter rdu / study-sample rdu)**

| NPV case | Mean NPV | Median NPV | Adopters (rdu) | Adopters (homes) | Adoption rate |
|---|---:|---:|---:|---:|---:|
| heatingSavings_coolingLCC_unsub | $-7,579 | $-6,782 | 6,414 | 1,628,538 | 3.96% |
| heatingSavings_coolingLCC_sub | $-2,703 | $-1,599 | 62,816 | 15,949,213 | 38.78% |
| heatingSavings_coolingLCC_sub_june2026 | $-7,329 | $-6,755 | 9,210 | 2,338,453 | 5.69% |
| heatingLCC_coolingSavings_unsub | $-9,680 | $-9,305 | 4,110 | 1,043,544 | 2.54% |
| heatingLCC_coolingSavings_sub | $-4,804 | $-3,937 | 26,397 | 6,702,295 | 16.30% |
| heatingLCC_coolingSavings_sub_june2026 | $-9,430 | $-9,257 | 6,042 | 1,534,086 | 3.73% |
| heatingLCC_coolingLCC_unsub | $-3,837 | $-3,134 | 18,126 | 4,602,258 | 11.19% |
| heatingLCC_coolingLCC_sub | $1,039 | $2,188 | 97,564 | 24,771,858 | 60.23% |
| heatingLCC_coolingLCC_sub_june2026 | $-3,587 | $-3,095 | 20,115 | 5,107,272 | 12.42% |

**Climate (lifetime, central SCC, long-run marginal emissions; 2025 dollars)**

| Quantity | Heating | Cooling |
|---|---:|---:|
| Avoided CO2e (t) | 19.64 | 2.21 |
| Avoided climate damages ($) | 5,559.80 | 604.22 |
| Baseline lifetime damages ($) | 20,055.90 | 5,297.25 |
| Retrofit lifetime damages ($) | 14,496.10 | 4,693.04 |

Sample rdu with blank climate values (no GEA region): 0.

