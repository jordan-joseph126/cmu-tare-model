# Session Changelog -- 2026-08-17

## Golden-value confirmation: the 12 Aug anchor-year numbers verified against a full run

> Branch `joseph-2026-nature-comms-submission`. **Documentation only.** No code
> was changed, no model was re-run by this session, and no value moved. The work
> was to audit an existing full run and decide whether the provisional golden
> values from `SESSION_CHANGELOG_2026-08-12.md` hold. They do. Nothing was
> committed.

---

## 1. Why this session existed

The 12 Aug session made two intended value changes -- the anchor year moved from
2024 to 2025, and `replace_small_values_with_nan` stopped filtering an exact
`0.0` to NaN. It measured the effect of both, but not by running the model. It
re-ran three functions (`calculate_lifetime_fuel_costs`, `calculate_private_npv`,
`economic_adoption_decision`) on top of already-exported result frames dated
`2026-08-02_20-32`. That is real code on real data for all 331,531 homes, enough
to size the change, but the pipeline upstream of those functions was never
re-executed and the notebook still carried a `base_year=2024` argument that would
have raised a `TypeError`. Every number the session produced was therefore
written into CLAUDE.md marked PROVISIONAL, with an instruction not to cite it.

A full end-to-end run has since been completed. This session checks the
provisional numbers against it.

---

## 2. The run that was audited

Timestamp `2026-08-17_19-16`, National, MP3 and MP4, all 331,531 homes,
`fixed_base` discount rate. A PA run at `2026-08-17_19-15` immediately precedes
it in the same session. Files, all under `cmu_tare_model/output_results/`:

| File | Size |
|---|---|
| `baseline_summary/summary_baseline/baseline_results_National_2026-08-17_19-16.csv` | 196 MB |
| `supplemental_data_fuelCosts/fuel_costs_baseline/mp0_fuel_costs_baseline_National_2026-08-17_19-16.csv` | 76 MB |
| `supplemental_data_damages/damages_climate_baseline/mp0_damages_climate_baseline_National_2026-08-17_19-16.csv` | 459 MB |
| `retrofit_mp3_results/summary_mp3_fixed_base/mp3_results_National_2026-08-17_19-16.csv` | 499 MB |
| `retrofit_mp4_results/summary_mp4_fixed_base/mp4_results_National_2026-08-17_19-16.csv` | 498 MB |
| `supplemental_data_fuelCosts/fuel_costs_ref2025/mp{3,4}_fuel_costs_ref2025_National_2026-08-17_19-16.csv` | 154 / 153 MB |
| `supplemental_data_damages/damages_climate_ref2025/mp{3,4}_damages_climate_ref2025_National_2026-08-17_19-16.csv` | 478 / 469 MB |
| `tepper_export/tepper_household_mp{3,4}_National_2026-08-17_19-16.csv` | 341 / 339 MB |
| `tepper_export/tepper_county_mp{3,4}_National_2026-08-17_19-16.csv` | 231 / 233 KB |

### The run used current code -- checked from the output, not only the source

A source check alone would not prove which code the run actually executed, so
each item below is confirmed by something visible in the output files.

| Requirement | Evidence in the output |
|---|---|
| Anchor year 2025 | The baseline fuel-cost file's year columns run `baseline_2025_heating_fuel_cost` through `baseline_2039_heating_fuel_cost` -- 15 years. Zero columns matching `2024_` in either MP's fuel-cost file. Climate damage files span `_2025_` through `_2039_`. |
| Exact-zero fix (Option B) in place | Homes with a usable MP4 NPV = 260,211, equal to the `include_heating = True` count. Under the old behavior this was 258,932; the 1,279-home gap is closed. |
| `base_year` removed as a parameter | Source side: gone from `calculate_lifetime_fuel_costs`, `calculate_lifetime_climate_impacts`, and `calculate_private_npv`. The Section 8 notebook handoff was applied -- zero occurrences of `base_year` in `tare_scenarios_v2_3.ipynb` or `tare_model_main_v2_3.ipynb`. A run with the old notebook line would have raised a `TypeError` and produced no output at all, so the existence of these files is itself the confirmation. |
| Modules predate the run | Every module involved was last edited at or before 16:52; the run started at 19:15. |

---

## 3. Comparison -- provisional versus full run

Means are taken over non-null values, matching how the 12 Aug numbers were
computed.

### Private side (MP4, `heatingLCC_coolingLCC_unsub`, `fixed_base`, National)

| Quantity | Golden table (PROVISIONAL) | Full run | Difference |
|---|---:|---:|---:|
| Mean lifetime heating fuel cost, baseline | $20,362.56 | $20,362.5614700378 | +$0.0015 (rounding) |
| Mean lifetime cooling fuel cost, baseline | $10,097.37 | $10,097.3677096370 | -$0.0023 (rounding) |
| Mean NPV, after the exact-zero fix | -$5,838.23 | -$5,838.2316748715 | -$0.0017 (rounding) |
| Adoption rate, after the exact-zero fix | 18.4416% | 18.441572% | -0.00003 pp (rounding) |
| Adopters | 47,987 | 47,987 | 0 |
| Homes with a usable NPV | 260,211 | 260,211 | 0 |
| Denominator (non-null adopter flag) | 260,211 | 260,211 | 0 |

Weighted and unweighted adoption rates are identical, as the golden row says --
every home carries weight 242.13.

Heating fuel cost is averaged over the 260,211 homes with `include_heating =
True`; cooling over the 250,576 with `include_cooling = True`.

### Climate means (National)

All sixteen columns from the 12 Aug "After" tables reproduce exactly at the
precision those tables reported.

Baseline (means over 260,204 homes for heating, 250,570 for cooling):

| Column | 12 Aug "After" | Full run |
|---|---:|---:|
| `baseline_heating_lifetime_mt_co2e_lrmer` | 69.42 | 69.423800 |
| `baseline_heating_lifetime_damages_climate_lrmer_central` | 18,377.40 | 18,377.402328 |
| `baseline_heating_lifetime_mt_co2e_srmer` | 80.69 | 80.686702 |
| `baseline_heating_lifetime_damages_climate_srmer_central` | 21,350.01 | 21,350.014773 |
| `baseline_cooling_lifetime_mt_co2e_lrmer` | 16.05 | 16.052380 |
| `baseline_cooling_lifetime_damages_climate_lrmer_central` | 4,174.75 | 4,174.754416 |
| `baseline_cooling_lifetime_mt_co2e_srmer` | 32.02 | 32.020744 |
| `baseline_cooling_lifetime_damages_climate_srmer_central` | 8,415.79 | 8,415.792095 |

MP4 avoided:

| Column | 12 Aug "After" | Full run |
|---|---:|---:|
| `ref2025_mp4_heating_avoided_mt_co2e_lrmer` | 56.94 | 56.941503 |
| `ref2025_mp4_heating_avoided_damages_climate_lrmer_central` | 15,131.46 | 15,131.462675 |
| `ref2025_mp4_heating_avoided_mt_co2e_srmer` | 53.77 | 53.767255 |
| `ref2025_mp4_heating_avoided_damages_climate_srmer_central` | 14,298.19 | 14,298.185511 |
| `ref2025_mp4_cooling_avoided_mt_co2e_lrmer` | 6.21 | 6.206286 |
| `ref2025_mp4_cooling_avoided_damages_climate_lrmer_central` | 1,613.45 | 1,613.452970 |
| `ref2025_mp4_cooling_avoided_mt_co2e_srmer` | 12.01 | 12.007562 |
| `ref2025_mp4_cooling_avoided_damages_climate_srmer_central` | 3,154.99 | 3,154.994623 |

**No mismatch above floating-point noise anywhere.** Every difference is
last-place rounding against a value the golden table records to two decimals.

---

## 4. MP3, measured for the first time

The golden table carried `PENDING (MP3 not re-run)` on every anchor-year row.
The full run covers MP3, so those cells now have values.

| Quantity (MP3, `heatingLCC_coolingLCC_unsub`, `fixed_base`) | Value |
|---|---:|
| Mean NPV | -$4,852.41 |
| Adopters / denominator | 72,115 / 260,211 |
| Adoption rate | 27.7140% |
| `ref2025_mp3_heating_avoided_mt_co2e_lrmer` | 51.1613 |
| `ref2025_mp3_heating_avoided_damages_climate_lrmer_central` | $13,629.25 |
| `ref2025_mp3_heating_avoided_mt_co2e_srmer` | 41.6161 |
| `ref2025_mp3_heating_avoided_damages_climate_srmer_central` | $11,114.97 |
| `ref2025_mp3_cooling_avoided_mt_co2e_lrmer` | 1.1246 |
| `ref2025_mp3_cooling_avoided_damages_climate_lrmer_central` | $291.22 |
| `ref2025_mp3_cooling_avoided_mt_co2e_srmer` | 2.0064 |
| `ref2025_mp3_cooling_avoided_damages_climate_srmer_central` | $526.83 |

MP3's avoided cooling emissions are a small fraction of MP4's because the
standard heat pump barely changes cooling energy relative to the baseline air
conditioner, while the high-efficiency unit cuts it substantially.

---

## 5. Two rows that can never be confirmed

The pre-exact-zero NPV (-$5,816.35) and adoption rate (18.4354%) describe a
halfway state: the anchor year already moved, but exact zeros were still being
turned into NaN. The current code carries the exact-zero fix, so no run can
reproduce that state. Both rows stay in the golden table as history and are now
labeled NOT CONFIRMABLE rather than NOT CONFIRMED, so a later session does not
keep trying to verify them.

---

## 6. The PENDING rows, derived from the same run

The golden table carried four rows marked `PENDING | PENDING`, all of them
waiting on a full run that had not happened. It has now, and the run's output
carries everything they asked for. All four are derived below. The rows
themselves stay in the table as history, each annotated with a pointer.

### 6a. Adoption rate, all nine NPV cases, both MPs

`fixed_base`, National, denominator 260,211 for every case (the non-null adopter
count, equal to `include_heating = True`). Weight is uniform at 242.13, so
weighted and unweighted rates are identical.

| NPV case | MP3 rate | MP3 adopters | MP3 mean NPV | MP4 rate | MP4 adopters | MP4 mean NPV |
|---|---:|---:|---:|---:|---:|---:|
| `heatingSavings_coolingLCC_unsub` | 15.9924% | 41,614 | -$8,781.65 | 11.5314% | 30,006 | -$9,555.69 |
| `heatingSavings_coolingLCC_sub` | 44.8690% | 116,754 | -$3,057.30 | 27.0062% | 70,273 | -$3,438.31 |
| `heatingSavings_coolingLCC_sub_june2026` | 24.1719% | 62,898 | -$7,212.06 | 18.1849% | 47,319 | -$7,856.72 |
| `heatingLCC_coolingSavings_unsub` | 12.3550% | 32,149 | -$9,808.43 | 9.7421% | 25,350 | -$10,709.45 |
| `heatingLCC_coolingSavings_sub` | 32.0475% | 83,391 | -$4,084.09 | 21.3142% | 55,462 | -$4,592.08 |
| `heatingLCC_coolingSavings_sub_june2026` | 19.4677% | 50,657 | -$8,238.85 | 15.6342% | 40,682 | -$9,010.48 |
| `heatingLCC_coolingLCC_unsub` | 27.7140% | 72,115 | -$4,852.41 | 18.4416% | 47,987 | -$5,838.23 |
| `heatingLCC_coolingLCC_sub` | 62.2111% | 161,880 | +$871.94 | 46.3462% | 120,598 | +$279.14 |
| `heatingLCC_coolingLCC_sub_june2026` | 34.8671% | 90,728 | -$3,282.82 | 27.2537% | 70,917 | -$4,139.26 |

The NPV ordering checks in CLAUDE.md hold throughout: within each rebate
vintage, `heatingLCC_coolingLCC` has the highest adoption rate and the least
negative mean NPV, since it is the only case crediting both avoided
replacements.

Rebate deltas in percentage points:

| Scope | `_sub` - `_unsub` | `_sub_june2026` - `_unsub` | `_sub_june2026` - `_sub` |
|---|---:|---:|---:|
| MP3 `heatingSavings_coolingLCC` | +28.8766 | +8.1795 | -20.6970 |
| MP3 `heatingLCC_coolingSavings` | +19.6925 | +7.1127 | -12.5798 |
| MP3 `heatingLCC_coolingLCC` | +34.4970 | +7.1530 | -27.3440 |
| MP4 `heatingSavings_coolingLCC` | +15.4747 | +6.6534 | -8.8213 |
| MP4 `heatingLCC_coolingSavings` | +11.5721 | +5.8921 | -5.6800 |
| MP4 `heatingLCC_coolingLCC` | +27.9047 | +8.8121 | -19.0926 |

MP3 responds to rebates more strongly than MP4 across the board. That is what
the mechanism predicts: MP3's capital cost is lower, so a rebate of a given size
closes a larger share of its NPV gap and tips more homes over the threshold.

**These `_sub` rates are an upper bound, not a forecast.** No state funding cap
is modeled (limitation 4), so every eligible home is assumed to receive its full
modeled rebate. That is why `heatingLCC_coolingLCC_sub` is the only case with a
positive mean NPV. The uncapped assumption should travel with any of these
numbers into the manuscript.

### 6b. June 2026 versus December 2024 movement -- the old row was half wrong

The golden row read: *"fossil MP4 lose HEEHR; electric MP4 >150% AMI gain
HOMES"*. The first half holds. The second half does not, and the row is
corrected rather than quietly reconciled.

Measured movement, identical in structure for both MPs:

| Baseline fuel | AMI band | December 2024 | June 2026 | MP3 homes | MP4 homes |
|---|---|---|---|---:|---:|
| Electricity | <=80% | HEEHR | HEEHR | 23,526 | 23,526 |
| Electricity | 80-150% | HEEHR | HEEHR | 23,034 | 23,034 |
| Electricity | >150% | HOMES | HOMES | 16,249 | 21,733 |
| Fossil | <=80% | HEEHR | None | 55,431 | 55,431 |
| Fossil | 80-150% | HEEHR | None | 58,528 | 58,528 |
| Fossil | >150% | HOMES | None | 60,747 | 64,722 |

Nothing appears in a "gain" cell. Electric homes above 150% AMI hold HOMES under
both vintages, so they gain nothing. The "gain" in the original row was written
on 11 Jul, when 2024 was HEEHR-only and those homes really did go from $0 to
HOMES. Adding fuel-neutral 2024 HOMES on 14 Jul moved the baseline underneath
the claim, which is exactly what that session's note warned about -- the claim
was simply never re-derived.

There are also two loss channels, not one. Fossil homes at or below 150% AMI
lose HEEHR to the June 2026 fuel gate, as documented. But fossil homes ABOVE
150% AMI lose HOMES as well, because 2026 HOMES is still electric-gated -- the
deferred fuel-neutral fix. Under June 2026 a fossil home receives nothing at all,
at any income. That is a consequence of a deliberate byte-identity choice, not a
bug, but it is larger than the golden row implied.

Adopter crossings on `heatingLCC_coolingLCC`, `_sub` to `_sub_june2026`:

| | Gained | Lost | Net |
|---|---:|---:|---:|
| MP3 | 0 | 71,152 | -71,152 |
| MP4 | 0 | 49,681 | -49,681 |

**When the deferred 2026-HOMES fuel-neutral fix lands, the fossil >150% AMI row
flips from "None" back to "HOMES" and roughly 60-65k homes per MP regain a
rebate.** The `_sub_june2026` rates in 6a will rise accordingly, so they are
correct for today's code and will move with that fix.

### 6c. Gate checks -- all three hold

- **June 2026 HEEHR fossil gate.** Zero fossil-baseline homes carry the `HEEHR`
  label under June 2026, for either MP, and the largest June 2026 rebate paid to
  any fossil home is $0.00.
- **2024 HOMES is fuel-neutral.** Recipients are 60,747 fossil + 16,249 electric
  (MP3) and 64,722 fossil + 21,733 electric (MP4). Every recipient is above 150%
  AMI; zero are at or below, consistent with HOMES only being consulted above the
  threshold (limitation 6). Fossil recipients outnumber electric about 3:1, so
  the 14 Jul change is unmistakably live.
- **South Dakota participation gate.** All 988 SD homes receive $0 under both
  vintages and carry no program label, matching
  `NON_PARTICIPATING_REBATE_STATES = {'SD'}`.

### 6d. Weighted rebate potential (uncapped)

`v4MID`, weighted to households, millions of USD2025:

| | MP3 December 2024 | MP3 June 2026 | MP4 December 2024 | MP4 June 2026 |
|---|---:|---:|---:|---:|
| HEEHR | $296,554.3M | $87,199.8M | $310,415.6M | $89,892.3M |
| HOMES | $64,107.6M | $11,692.0M | $75,009.0M | $17,151.0M |
| Total | $360,661.9M | $98,891.8M | $385,424.6M | $107,043.4M |

June 2026 cuts the modeled potential to roughly 27-28% of the 2024 figure. This
is `total_eligible` -- uncapped potential summed over every eligible home, NOT a
disbursement and NOT a budget estimate.

### 6e. The 2024 HOMES value move -- reconstructed, not measured

The golden row asks to confirm that `_sub` adoption rises against the pre-14-Jul
`_sub`. That older `_sub` cannot be produced: the HEEHR-only 2024 code path is
gone, and no run of the current code will recreate it.

It can be rebuilt exactly, though, from the rules rather than from code. Under
the old 2024 rules HOMES did not exist, so every home above 150% AMI received $0
and its subsidized NPV equalled its unsubsidized NPV; homes at or below 150% AMI
were routed to HEEHR, which 14 Jul did not touch. So the old `_sub` adopter flag
is `_unsub` for today's 2024-HOMES recipients and `_sub` for everyone else.

| Scope | MP3 old -> new | MP3 rise | MP4 old -> new | MP4 rise |
|---|---|---:|---|---:|
| `heatingSavings_coolingLCC` | 41.4959% -> 44.8690% | +3.3730 pp | 24.4982% -> 27.0062% | +2.5080 pp |
| `heatingLCC_coolingSavings` | 29.7378% -> 32.0475% | +2.3097 pp | 19.3966% -> 21.3142% | +1.9177 pp |
| `heatingLCC_coolingLCC` | 56.5456% -> 62.2111% | +5.6654 pp | 42.3952% -> 46.3462% | +3.9510 pp |

The rise is confirmed in direction and size. On `heatingLCC_coolingLCC`, 14,742
(MP3) and 10,281 (MP4) homes newly adopt; every one is above 150% AMI, and the
majority are fossil -- MP4 splits 5,309 natural gas, 3,173 electricity, 903
propane, 896 fuel oil, which is the fuel-neutrality the 14 Jul session intended.
No home loses adoption, as expected when a rebate is added and nothing else
changes.

**This is the one number in the golden table not taken from a run.** It is exact
only if adding 2024 HOMES was the sole `_sub` value move on 14 Jul, which is what
that session recorded. Labeled RECONSTRUCTED in the table.

---

## 7. What the run does not cover

**The other three discount rates.** The run wrote `fixed_base` only -- there is a
`summary_mp{3,4}_fixed_base` directory and no equivalent for `fixed_low`,
`fixed_high` or `variable`. The lone `_variable` column in the output
(`private_discount_rate_variable`) is the input rate, not an NPV. No golden value
exists for those three rates, and confirming them needs another run.

---

## 8. One observation, not a defect

Climate columns are non-null for 260,204 homes while heating fuel cost is
non-null for 260,211; on the cooling side it is 250,570 against 250,576. Seven
homes and six homes respectively have a fuel cost but no climate value.

This is not something the anchor-year work introduced. It is present identically
on both sides of the change, which is precisely why the climate means reproduce
to the digit. Recording it here so it is not rediscovered as a new problem; it
was not diagnosed.

---

## 9. Files changed this session

| File | Change |
|---|---|
| `CLAUDE.md` | Provisional golden rows annotated (confirmed, or not confirmable) and kept as history; CONFIRMED rows added for the four private-side quantities, the sixteen climate means, and MP3; the four PENDING rows annotated and kept, with fourteen further CONFIRMED rows added for the nine NPV cases, the corrected rebate movement, the three gate checks, the rebate totals, and the reconstructed 2024 HOMES move; the "PROVISIONAL, do not cite" note below the table rewritten to record the confirming run, its file paths, the uncapped-rebate caveat, the reconstruction caveat, and what the run does not cover; header date updated; session log rows added for 12 Aug and 17 Aug |
| `cmu_tare_model/docs/SESSION_CHANGELOG_2026-08-17.md` | This file |

No `.py`, `.ipynb`, or data file was touched. Nothing was committed -- the
changes sit in the working tree for the researcher.
