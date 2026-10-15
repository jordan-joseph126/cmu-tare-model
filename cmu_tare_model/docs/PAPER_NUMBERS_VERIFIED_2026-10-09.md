# Paper numbers verified -- the draft against the current run (ResStock 2022.1.1, MP3 and MP4)

Written 10 Oct 2026, session 12. Read-only session: no model run, no notebook run, no
commit, and no change to any file in the repo other than this one. Every check script is
outside the repo, in `~/tare_port/s12_paper_verify/scripts/` (list at the end).

**Updated 11 Oct 2026.** Jordan loaded run `2026-10-09_00-46` again in the main notebook
with grid impact on. Its peaks fill the one item that was still open on 10 Oct, the
Allegheny economic-adopter peak after boiler pricing (rows P-3 and P-7, section 5.5). The
notebook's other outputs, and the export files it wrote again, were checked against this
file: nothing differs (section 2.13).

**What this file is.** Every modeled number the Nature Communications draft prints, worked
out again from the saved run files, next to its value in the current run. It checks the
paper number crosswalk of 9 Oct 2026 row by row and adds the claims the crosswalk does not
map. It is meant to be the source of numbers for the manuscript revision.

**The two runs.**

| | Draft | Current |
|---|---|---|
| Run | `2026-08-19_20-56` (its 9 result files are byte-identical to `2026-09-02_19-04`, SHA-256) | `2026-10-09_00-46` (its 22 files are byte-identical to a later run found on disk, `2026-10-09_21-04`) |
| Rows in each results table | 331,531 | 331,526 |
| rdu the results cover | 260,211 (`include_heating` True) = 63,005,153 homes | 221,205 (`include_sample` True) = 53,560,591 homes in 3,079 counties |
| Demand and peaks covered | 331,526 rdu (demand); 1,610 Allegheny rdu (peaks) | the study sample; 1,146 Allegheny rdu |

**How to read the tables.**

- **rdu and homes.** A count is labeled rdu or homes. One rdu is 242.13101272727272 homes,
  the value the ResStock file and the current results tables hold (CLAUDE.md quotes it as
  242.131013; the difference is under 0.1 home over the whole sample). The draft run saved
  the weight as 242.13.
- **Packages.** MP3 is the minimum-efficiency, single-stage heat pump. MP4 is the
  high-efficiency, variable-speed heat pump. MP5 and ResStock 2025.1 are not used anywhere.
- **Adoption** is NPV >= 0 at the 7% discount rate (`fixed_base`). Unless a row says
  otherwise it is the base case, `heatingLCC_coolingLCC` (credit for both old systems).
  The three rebate cases are no rebate (`_unsub`), December 2024 rules (`_sub`) and June
  2026 rules (`_sub_june2026`). In every run and case the adopter flag equals NPV >= 0 on
  every rdu, and no flag is blank on an rdu the results cover.
- **Sources.** [D] draft results tables, `mp{3,4}_results_National_2026-08-19_20-56.csv`.
  [C] current results tables, `mp{3,4}_results_National_2026-10-09_00-46.csv`. [RS] the
  ResStock 2022.1.1 annual files (baseline, upgrade 03, upgrade 04), whole-home
  electricity. [TS] the hourly tables in the grid-impact query cache. [F] the sample
  funnel of the current run. [XH] export household file, [XD] export detailed household
  file, [XC] export county file, all national, run `2026-10-09_00-46`; [XA] the Allegheny
  export files. Column names were built with the repo's helpers (`create_adoption_col`,
  `create_npv_case_col`, `create_rebate_col`, `create_cost_col`).
- **Suggested text** follows the session's rounding rule: whole percent for a rate, one
  decimal for millions, three significant figures for GWh, percentage points for a change
  between two rates.
- **Export value.** The exports ship one case only (base case, no rebate) plus the June
  2026 rebate amount and label. Every export number below equals the results table on
  every rdu (largest NPV gap $0.00). A December 2024 case and a peak are not in the exports.
- **Driver.** The step that moved the number, from results tables of each step that are on
  disk (section 2.0). "Energy fix" is the 23 Sep consumption fix. "Sample rule" is which
  homes a result counts. For adoption it is the move from 260,211 rdu to the 221,205-rdu
  study sample (homes with no central or room AC out, then homes with a shared cooling
  system out). For demand and peaks it also covers the earlier move from every occupied
  single-family rdu to the rdu with a result. "Boiler pricing" is the 9 Oct change.
  "No rounding" is the 5 Oct change. "Income draw" is the 23 Sep change to how household
  income is drawn.
- **Status.** "matches crosswalk" means the crosswalk's current value equals the data. A
  row that differs says how. Unmapped rows (U1 to U16) say whether the draft's claim holds.

**Counts for this file:** 106 numbered rows checked, plus the 30 Fig 4 markers
(section 2.11) and the 7 writeup rows (section 4); 28 discrepancy lines
in section 1 (10 in the draft's own text, 8 in the crosswalk, 10 in wording the crosswalk
does not map).

---

## 1. Discrepancies

One line each. Section 1.1 is the draft's own text. Section 1.2 is the crosswalk. Section
1.3 is draft wording the crosswalk does not map and the current data does not support.

### 1.1 The draft's text against the draft run

1. **DM-7, p10 par 2, "a 23% median county-level increase".** The draft run gives
   +22.4926% (3,098 counties), which rounds to 22%. Likely cause: rounded twice (22.49 to
   22.5 in the notes, then to 23). Current: +18.1811%.
2. **DS-4, p17 par 3, "an 11-29% decrease for homes using fuel oil and propane".** The
   four draft values are 11.0904 (fuel oil, MP3), 15.0563 (propane, MP3), 23.4886 (fuel
   oil, MP4) and 29.6679 (propane, MP4), so the range rounds to 11-30. Likely cause: 29.67
   was cut off, not rounded. Current: 19.5083 to 35.5947.
3. **DS-3 and DS-4, p17 par 3, "43% drop" and "11-29% decrease".** Both are differences
   between two rates, so they are percentage points, not percent. The 43 is MP3 only
   (42.7273 points); MP4 is 26.0848 points and is not given.
4. **Every home count in the text and in Fig 4.** A printed count is a rate times the
   homes of a larger group, not adopters times the weight. National: rate x 80,273,601
   homes (all 331,531 rows of the draft table), while the rate is a share of 260,211 rdu.
   Example: "22 million" is 27.1760% x 80.27M = 21.82M; the 70,715 adopter rdu stand for
   17.12M homes. By fuel: rate x every row of that fuel, with or without a result.
5. **DM-1, p10 par 1, "63M modeled single-family homes" (twice).** 63.0M is the 260,211
   rdu with a result. The demand numbers in the same sentences cover 331,526 rdu = 80.27M
   homes.
6. **DM-7, p10 par 2, "(Fig 3B)".** The sentence is about the minimum-efficiency system.
   The Fig 3 caption (p11) says panel (A) is minimum-efficiency and panel (B) is
   high-efficiency.
7. **F2-8, p8 par 1, "Counties dominated by natural gas ... with only 4% of homes
   adopting ... and 12%".** 4.2980% and 11.5412% are the national rates of homes heated
   with natural gas (the Fig 4 markers), not rates of counties where gas dominates. In the
   draft run, counties where natural gas is the most common fuel in the rdu with a result
   have a pooled rate of 23.2482% (MP3) and 15.1708% (MP4) (section 3, U3).
8. **DS-6, p18 par 1, "$42 billion in HEEHR and $8 billion in HOMES rebates would flow to
   homes".** The figures are the high-efficiency package only, June 2026 rules, adopters
   only, in the heating-credit-only case (`heatingLCC_coolingSavings_sub_june2026`):
   $41.5770B and $8.3012B on the draft run. The sentence before it calls them "uncapped
   modeled potential", which in the code is the total over every eligible home ($89.8923B
   and $17.1510B for the same package and rules). They are not the paper's base case
   (section 5.2).
9. **Fig 4 y-axis labels, p13.** "80.1 M Homes (100%)" is 330,634 rows x 242.13 =
   80,056,410 homes: the sum over fuel labels, which includes "Other Fuel" (9,277 rows)
   and leaves out 897 rows with no fuel label. The national markers use 80,273,601 homes.
   The four fuels drawn add up to 77,810,170 homes (321,357 rows). Each fuel label counts
   rows with no result too: "Electricity 25.6/80.1 M" is 105,792 rows, of which 73,699
   have a result.
10. **MT-2, p20 par 1, "reduced to ~330,000 dwelling units or ~80 million homes".** True
    for the occupied single-family step (331,526 rdu). The draft's adoption results cover
    260,211 of them.

### 1.2 The crosswalk (9 Oct 2026 version) against the data

1. **DM-4, crosswalk section 4.4, share of residential sales.** Crosswalk: about 18.2%,
   on an implied 1,525,156 GWh. With the draft's own 1,514,993 GWh: 18.3616% (MP3) and
   0.0844% (MP4). Both versions round to 18%. Cause: the implied denominator.
2. **Crosswalk section 1, "Fig 4 fuel-group sizes ... 321,352 rdu = 77.81M".** The draft
   table holds 321,357 rows for the four fuels, 77,810,170 homes at its weight of 242.13.
   The 5 extra rdu are the shared-heating rdu a later filter removed (331,531 rows then,
   331,526 now). Same to one decimal.
3. **Crosswalk section 1, "rate x 80.27M, which is 331,526 rdu x 242.131013".** The draft
   multiplied by its own table total, 331,531 rows x 242.13 = 80,273,601 homes. 331,526 x
   242.13101273 is 80,272,726. Same to two decimals.
4. **Crosswalk section 3, row 3, "MP3 256 --> 45; MP4 125 --> 68".** The two sides count
   different homes. 256 and 125 are among the 1,356 Allegheny rdu with a result in the
   draft run; 45 and 68 are among today's 1,146. On the same 1,146 rdu the draft run has
   251 (MP3) and 118 (MP4).
5. **Crosswalk section 4.6, economic-adopter peak.** The crosswalk carries 715.26 MW
   (+0.74 MW) for MP4, measured with 67 adopters before boiler pricing. The grid impact
   run of 11 Oct 2026 on the current run gives 715.16 MW (+0.65 MW) with 68 adopters, hour
   4433 (715.1649 MW and +0.6504 MW from the cached hourly tables). MP3 does not move:
   713.86 MW (-0.65 MW), the same 45 rdu (section 5.5).
6. **DM-8, crosswalk section 4.4, "County-level increases can exceed 600% ... still
   true".** True for one county only, in the draft and now: G4600210 (Campbell County,
   South Dakota), 2 rdu, +622.3300%. The next highest county is +534.6303% (section 3, U5).
7. **DM-10, crosswalk section 4.4, "-12.5%".** The exact value is -12.4931%, which is
   -12% as a whole percent. Rounding the crosswalk's -12.5 again would give 13.
8. **Crosswalk section 4.3, note on county figures.** The draft figures "on the same
   footing" (37.93%, 39.20%, 31.94% and 23.27%, 25.71%, 10.94%) are over 3,090 counties,
   the counties with an rdu that has a result, not 3,098. The crosswalk gives no count.

### 1.3 Draft wording the crosswalk does not map, and the current data does not support

1. **U1, p2, "shift peak demand from summer to winter".** At 100% adoption the Allegheny
   peak moves to hour 152 (7 January) for both packages. With economic adopters it stays
   at hour 4433 (4 July) for both packages.
2. **U3, p8, "the most favorable counties are those where households currently buy
   delivered heating fuels".** Current median county adoption with no rebate, by the
   county's most common fuel, MP3: electricity 42.8571%, propane 18.7500%, natural gas
   14.2857%, fuel oil 9.0537%. MP4: electricity 33.3333%, fuel oil 22.4747%, propane
   21.4286%, natural gas 16.5391%.
3. **U4, p10, "70% of the 3,100 modeled counties".** Now 64.4040% of 3,079 counties.
4. **U5, p10, "County-level increases can exceed 600%".** One county of 2 rdu.
5. **U9, p18, "the minimum-efficiency, single-stage heat pump ... is the most likely
   rebate-supported choice".** June 2026 adoption of electric-heated rdu: MP3 71.2689%,
   MP4 72.4605%. MP3 is higher in the South and West, MP4 in the Midwest and Northeast.
6. **U10, p20 and p21, heating degree days only.** The code scales cooling energy by
   cooling degree-day factors, in the draft run's code and now.
7. **U12, p24, "brought the ResStock equipment within the NLR regression bounds".** The
   floors bring central AC, furnace and boiler efficiency up to the bottom of each range.
   Size is not changed: 73,683 of 221,205 rdu have an old cooling system outside its
   row's size or efficiency range, 22,378 an old heating system, and 72,375 (MP3) or
   77,755 (MP4) a heat pump.
8. **U13, p23, "REMDB v4.0.0 ... USD2023".** The repo's table is a cut of REMDB 2024
   (`REMDB_2024_12_23.xlsx`), in 2023 dollars. "4.0.0" is not written anywhere in the repo.
9. **U14, p25, "all filtered buildings".** The draft's 100% scenario covered 331,526 rdu
   (every occupied single-family rdu the package applies to), not the 260,211 rdu with a
   result. Now it covers the 221,205-rdu sample.
10. **U15, p9 and p11, "a county identifier mismatch, insufficient data, or too few
    modeled homes".** No county is grey for an identifier mismatch or a minimum count.
    Every grey county is one with no home counted: 10 in the draft, 29 now.

---

## 2. Verified crosswalk

### 2.0 Which run made the draft, what is current, and the steps between (crosswalk sections 1 and 2)

| ID | Crosswalk statement | From the data | Status |
|---|---|---|---|
| X1-1 | The draft's adoption numbers come from run `2026-08-19_20-56`; 27% / 18% round from 27.1760% / 18.0984% | 27.176023% (70,715 of 260,211 rdu) and 18.098389% (47,094) [D] | matches crosswalk |
| X1-2 | 61% / 45% and 34% / 26% round from 61.2169% / 45.1168% and 34.0908% / 26.3936% | 61.216859% (159,293), 45.116847% (117,399), 34.090796% (88,708), 26.393581% (68,679) [D] | matches crosswalk |
| X1-3 | Run `2026-09-02_19-04` is byte-identical to `2026-08-19_20-56` | All 9 result files (baseline, MP3, MP4, three fuel-cost and three climate files) have equal SHA-256 | matches crosswalk |
| X1-4 | Allegheny peaks in the draft: 862.51 MW, 6,629.87 MW, 5,364.10 MW, hours 4433 and 152 | 862.5113 MW @ 4433; 6,629.8728 MW @ 152; 5,364.0998 MW @ 152, on 1,610 rdu [TS] | matches crosswalk |
| X1-5 | Demand +427,000 / +30,600 GWh reproduces on 331,526 rdu, baseline 1,136,778.6 GWh | +427,043.7197 and +30,618.3300 GWh; baseline 1,136,778.6108 GWh; 331,526 rdu in 3,098 counties [RS] | matches crosswalk |
| X1-6 | Home counts equal rate x 80.27M; the label says 80.1 M; the fuel groups sum to 77.8M | 80,273,601 homes (331,531 rows x 242.13); 80,056,410 (330,634 rows with a fuel label); 77,810,170 (321,357 rows of the four fuels) [D] | matches crosswalk to the printed digits; two row counts differ (section 1.2, items 2 and 3) |
| X2-1 | Current run `2026-10-09_00-46`; study sample 221,205 rdu = 53.56M homes in 3,079 counties | 221,205 rdu; 53,560,590.67 homes; 3,079 counties [C][F] | matches crosswalk |
| X2-2 | Consumption fix alone: 16.6592% (-10.52 points) and 17.0273% (-1.07) on the same 260,211 rdu | 16.659173% (-10.5168) and 17.027336% (-1.0711), run `2026-09-23_15-34` | matches crosswalk |
| X2-3 | Income draw moves the rebate cases by at most 0.03 points | Run `2026-09-22_20-45` against the draft run: -0.0050 and +0.0161 (MP3), +0.0054 and +0.0158 (MP4); no-rebate case 0 | matches crosswalk |
| X2-4 | Sample narrowed to 221,205 rdu: 18.8088% (+2.15) and 18.9914% (+1.96) | 18.808797% (+2.1496) and 18.991433% (+1.9641) | matches crosswalk |
| X2-5 | No rounding: at most 231 rdu of adopters move in any case | Largest move is MP3, December 2024: 101,611 --> 101,380 (-231 rdu) | matches crosswalk |
| X2-6 | Boiler pricing: 20,973 rdu (5.08M homes), mean credit $3,474 --> $7,638; +0.54 and +0.48 points; no home stops adopting | 20,973 rdu = 5,078,214 homes; $3,473.72 --> $7,638.08; 19.347212% (+0.5384) and 19.467462% (+0.4760); new adopters 1,191 / 1,792 / 1,191 (MP3) and 1,053 / 3,374 / 1,053 (MP4), all with a fossil boiler; 0 lost | matches crosswalk |
| X2-7 | The draft's data limited to today's 221,205 rdu: 30.18% and 19.71% | 30.1824% (66,765 rdu) and 19.7134% (43,607 rdu) [D] | matches crosswalk |
| X2-8 | Demand on 260,211 rdu: +353,404.4 GWh (+39.59%) and +34,992.8 GWh (+3.92%), baseline 892,740.7 | +353,404.3626 (+39.5865%) and +34,992.7771 (+3.9197%); baseline 892,740.7135 GWh [RS] | matches crosswalk |
| X2-9 | About 33,700 GWh of the old MP4 increase came from the homes with no AC | 34,992.7771 - 1,298.8171 = 33,693.96 GWh (260,211 --> 221,301 rdu) | matches crosswalk |

**Adoption, base case, step by step.** Each cell: rate (change from the step above, in
percentage points); adopter rdu. Step S4a and S4b are run `2026-09-25_22-55` limited to
the 221,301 and 221,205 rdu: run `2026-09-30_22-35` (homes with no AC out) is not on disk.

| Step | rdu counted | MP3, No rebate | MP3, Dec 2024 | MP3, June 2026 | MP4, No rebate | MP4, Dec 2024 | MP4, June 2026 |
|---|---|---|---|---|---|---|---|
| S0 draft | 260,211 | 27.1760%; 70,715 | 61.2169%; 159,293 | 34.0908%; 88,708 | 18.0984%; 47,094 | 45.1168%; 117,399 | 26.3936%; 68,679 |
| S1 income draw | 260,211 | 27.1760% (+0.0000); 70,715 | 61.2119% (-0.0050); 159,280 | 34.1069% (+0.0161); 88,750 | 18.0984% (+0.0000); 47,094 | 45.1222% (+0.0054); 117,413 | 26.4093% (+0.0158); 68,720 |
| S2 energy fix | 260,211 | 16.6592% (-10.5168); 43,349 | 41.0282% (-20.1836); 106,760 | 23.9452% (-10.1618); 62,308 | 17.0273% (-1.0711); 44,307 | 42.9144% (-2.2078); 111,668 | 26.0189% (-0.3905); 67,704 |
| S3 sample flag | 260,211 | 16.6592% (+0.0000); 43,349 | 41.0282% (+0.0000); 106,760 | 23.9452% (+0.0000); 62,308 | 17.0273% (+0.0000); 44,307 | 42.9144% (+0.0000); 111,668 | 26.0189% (+0.0000); 67,704 |
| S4a no-AC homes out (derived) | 221,301 | 18.8020% (+2.1428); 41,609 | 45.9216% (+4.8934); 101,625 | 26.3763% (+2.4311); 58,371 | 18.9859% (+1.9586); 42,016 | 47.8782% (+4.9638); 105,955 | 28.4436% (+2.4247); 62,946 |
| S4b shared cooling out (derived) | 221,205 | 18.8088% (+0.0068); 41,606 | 45.9352% (+0.0136); 101,611 | 26.3819% (+0.0056); 58,358 | 18.9914% (+0.0055); 42,010 | 47.8927% (+0.0144); 105,941 | 28.4505% (+0.0069); 62,934 |
| S5 other changes, 25 Sep to 4 Oct | 221,205 | 18.8088% (+0.0000); 41,606 | 45.9352% (+0.0000); 101,611 | 26.3819% (+0.0000); 58,358 | 18.9914% (+0.0000); 42,010 | 47.8927% (+0.0000); 105,941 | 28.4505% (+0.0000); 62,934 |
| S6 no rounding | 221,205 | 18.8088% (+0.0000); 41,606 | 45.8308% (-0.1044); 101,380 | 26.3394% (-0.0425); 58,264 | 18.9914% (+0.0000); 42,010 | 47.8407% (-0.0520); 105,826 | 28.4189% (-0.0316); 62,864 |
| S7 damages in USD2025 | 221,205 | 18.8088% (+0.0000); 41,606 | 45.8308% (+0.0000); 101,380 | 26.3394% (+0.0000); 58,264 | 18.9914% (+0.0000); 42,010 | 47.8407% (+0.0000); 105,826 | 28.4189% (+0.0000); 62,864 |
| S8 boiler pricing | 221,205 | 19.3472% (+0.5384); 42,797 | 46.6409% (+0.8101); 103,172 | 26.8778% (+0.5384); 59,455 | 19.4675% (+0.4760); 43,063 | 49.3660% (+1.5253); 109,200 | 28.8949% (+0.4760); 63,917 |

**Adoption by baseline heating fuel, step by step.** Under June 2026 rules each fossil
fuel's rate equals its no-rebate rate, and electricity's equals its December 2024 rate, in
every step.

**MP3, No rebate** (change from the step above, in percentage points)

| Step | Electricity | Natural Gas | Propane | Fuel Oil |
|---|---|---|---|---|
| S0 draft | 43.7428% | 11.5412% | 73.3950% | 61.7383% |
| S1 income draw | 43.7428% (+0.00) | 11.5412% (+0.00) | 73.3950% (+0.00) | 61.7383% (+0.00) |
| S2 energy fix | 41.1254% (-2.62) | 3.8883% (-7.65) | 36.2815% (-37.11) | 11.5545% (-50.18) |
| S3 sample flag | 41.1254% (+0.00) | 3.8883% (+0.00) | 36.2815% (+0.00) | 11.5545% (+0.00) |
| S4a no-AC homes out (derived) | 45.1312% (+4.01) | 4.6124% (+0.72) | 40.1306% (+3.85) | 13.2190% (+1.66) |
| S4b shared cooling out (derived) | 45.1514% (+0.02) | 4.6145% (+0.00) | 40.1257% (-0.00) | 13.2218% (+0.00) |
| S5 other changes, 25 Sep to 4 Oct | 45.1514% (+0.00) | 4.6145% (+0.00) | 40.1257% (+0.00) | 13.2218% (+0.00) |
| S6 no rounding | 45.1514% (+0.00) | 4.6145% (+0.00) | 40.1257% (+0.00) | 13.2218% (+0.00) |
| S7 damages in USD2025 | 45.1514% (+0.00) | 4.6145% (+0.00) | 40.1257% (+0.00) | 13.2218% (+0.00) |
| S8 boiler pricing | 45.1514% (+0.00) | 5.3327% (+0.72) | 40.6401% (+0.51) | 14.5555% (+1.33) |

**MP3, Dec 2024** (change from the step above, in percentage points)

| Step | Electricity | Natural Gas | Propane | Fuel Oil |
|---|---|---|---|---|
| S0 draft | 68.1570% | 54.2685% | 88.4513% | 72.8286% |
| S1 income draw | 68.2140% (+0.06) | 54.2396% (-0.03) | 88.4155% (-0.04) | 72.7985% (-0.03) |
| S2 energy fix | 66.8503% (-1.36) | 28.0933% (-26.15) | 63.1303% (-25.29) | 29.3593% (-43.44) |
| S3 sample flag | 66.8503% (+0.00) | 28.0933% (+0.00) | 63.1303% (+0.00) | 29.3593% (+0.00) |
| S4a no-AC homes out (derived) | 71.3981% (+4.55) | 32.8540% (+4.76) | 68.1551% (+5.02) | 32.8384% (+3.48) |
| S4b shared cooling out (derived) | 71.4163% (+0.02) | 32.8688% (+0.01) | 68.1525% (-0.00) | 32.8383% (-0.00) |
| S5 other changes, 25 Sep to 4 Oct | 71.4163% (+0.00) | 32.8688% (+0.00) | 68.1525% (+0.00) | 32.8383% (+0.00) |
| S6 no rounding | 71.2689% (-0.15) | 32.7842% (-0.08) | 68.0137% (-0.14) | 32.7734% (-0.06) |
| S7 damages in USD2025 | 71.2689% (+0.00) | 32.7842% (+0.00) | 68.0137% (+0.00) | 32.7734% (+0.00) |
| S8 boiler pricing | 71.2689% (+0.00) | 33.9571% (+1.17) | 68.6097% (+0.60) | 34.0639% (+1.29) |

**MP4, No rebate** (change from the step above, in percentage points)

| Step | Electricity | Natural Gas | Propane | Fuel Oil |
|---|---|---|---|---|
| S0 draft | 34.2243% | 4.2980% | 51.3163% | 48.2973% |
| S1 income draw | 34.2243% (+0.00) | 4.2980% (+0.00) | 51.3163% (+0.00) | 48.2973% (+0.00) |
| S2 energy fix | 37.0819% (+2.86) | 4.9058% (+0.61) | 36.9270% (-14.39) | 25.1823% (-23.11) |
| S3 sample flag | 37.0819% (+0.00) | 4.9058% (+0.00) | 36.9270% (+0.00) | 25.1823% (+0.00) |
| S4a no-AC homes out (derived) | 39.7483% (+2.67) | 5.8167% (+0.91) | 40.5714% (+3.64) | 29.1192% (+3.94) |
| S4b shared cooling out (derived) | 39.7642% (+0.02) | 5.8193% (+0.00) | 40.5666% (-0.00) | 29.1111% (-0.01) |
| S5 other changes, 25 Sep to 4 Oct | 39.7642% (+0.00) | 5.8193% (+0.00) | 40.5666% (+0.00) | 29.1111% (+0.00) |
| S6 no rounding | 39.7642% (+0.00) | 5.8193% (+0.00) | 40.5666% (+0.00) | 29.1111% (+0.00) |
| S7 damages in USD2025 | 39.7642% (+0.00) | 5.8193% (+0.00) | 40.5666% (+0.00) | 29.1111% (+0.00) |
| S8 boiler pricing | 39.7642% (+0.00) | 6.3197% (+0.50) | 41.4564% (+0.89) | 31.1802% (+2.07) |

**MP4, Dec 2024** (change from the step above, in percentage points)

| Step | Electricity | Natural Gas | Propane | Fuel Oil |
|---|---|---|---|---|
| S0 draft | 63.5124% | 30.3827% | 80.9841% | 71.7859% |
| S1 income draw | 63.5680% (+0.06) | 30.3661% (-0.02) | 80.9411% (-0.04) | 71.8160% (+0.03) |
| S2 energy fix | 68.8286% (+5.26) | 27.1810% (-3.19) | 70.2676% (-10.67) | 52.7334% (-19.08) |
| S3 sample flag | 68.8286% (+0.00) | 27.1810% (+0.00) | 70.2676% (+0.00) | 52.7334% (+0.00) |
| S4a no-AC homes out (derived) | 72.5468% (+3.72) | 32.0912% (+4.91) | 76.1306% (+5.86) | 58.9448% (+6.21) |
| S4b shared cooling out (derived) | 72.5702% (+0.02) | 32.1041% (+0.01) | 76.1287% (-0.00) | 58.9431% (-0.00) |
| S5 other changes, 25 Sep to 4 Oct | 72.5702% (+0.00) | 32.1041% (+0.00) | 76.1287% (+0.00) | 58.9431% (+0.00) |
| S6 no rounding | 72.4605% (-0.11) | 32.0767% (-0.03) | 76.0715% (-0.06) | 58.9287% (-0.01) |
| S7 damages in USD2025 | 72.4605% (+0.00) | 32.0767% (+0.00) | 76.0715% (+0.00) | 58.9287% (+0.00) |
| S8 boiler pricing | 72.4605% (+0.00) | 34.1322% (+2.06) | 77.0512% (+0.98) | 62.9299% (+4.00) |

**County adoption figures, step by step.** Plain county figures: each county counts once.

**No rebate**

| Package | Step | Counties | Median | Mean | Share at 50% or more |
|---|---|---|---|---|---|
| MP3 | Draft as printed: adopters over every rdu of the county | 3,098 | 27.4510% | 28.5846% | 9.8128% |
| MP3 | S0 draft (adopters over rdu counted) | 3,090 | 37.9310% | 39.2010% | 31.9417% |
| MP3 | S1 income draw (adopters over rdu counted) | 3,090 | 37.9310% | 39.2010% | 31.9417% |
| MP3 | S2 energy fix (adopters over rdu counted) | 3,090 | 20.8333% | 24.4199% | 14.0453% |
| MP3 | S3 sample flag (adopters over rdu counted) | 3,090 | 20.8333% | 24.4199% | 14.0453% |
| MP3 | S4a no-AC homes out (derived) (adopters over rdu counted) | 3,079 | 23.0769% | 26.3542% | 16.6937% |
| MP3 | S4b shared cooling out (derived) (adopters over rdu counted) | 3,079 | 23.0769% | 26.3609% | 16.6937% |
| MP3 | S5 other changes, 25 Sep to 4 Oct (adopters over rdu counted) | 3,079 | 23.0769% | 26.3609% | 16.6937% |
| MP3 | S6 no rounding (adopters over rdu counted) | 3,079 | 23.0769% | 26.3609% | 16.6937% |
| MP3 | S7 damages in USD2025 (adopters over rdu counted) | 3,079 | 23.0769% | 26.3609% | 16.6937% |
| MP3 | S8 boiler pricing (adopters over rdu counted) | 3,079 | 23.5741% | 26.8143% | 17.3433% |
| MP4 | Draft as printed: adopters over every rdu of the county | 3,098 | 16.6667% | 18.8041% | 3.1633% |
| MP4 | S0 draft (adopters over rdu counted) | 3,090 | 23.2717% | 25.7104% | 10.9385% |
| MP4 | S1 income draw (adopters over rdu counted) | 3,090 | 23.2717% | 25.7104% | 10.9385% |
| MP4 | S2 energy fix (adopters over rdu counted) | 3,090 | 20.0000% | 22.3908% | 7.7023% |
| MP4 | S3 sample flag (adopters over rdu counted) | 3,090 | 20.0000% | 22.3908% | 7.7023% |
| MP4 | S4a no-AC homes out (derived) (adopters over rdu counted) | 3,079 | 21.4286% | 23.9631% | 9.7759% |
| MP4 | S4b shared cooling out (derived) (adopters over rdu counted) | 3,079 | 21.4286% | 23.9688% | 9.7759% |
| MP4 | S5 other changes, 25 Sep to 4 Oct (adopters over rdu counted) | 3,079 | 21.4286% | 23.9688% | 9.7759% |
| MP4 | S6 no rounding (adopters over rdu counted) | 3,079 | 21.4286% | 23.9688% | 9.7759% |
| MP4 | S7 damages in USD2025 (adopters over rdu counted) | 3,079 | 21.4286% | 23.9688% | 9.7759% |
| MP4 | S8 boiler pricing (adopters over rdu counted) | 3,079 | 21.8638% | 24.3321% | 10.2306% |

**Dec 2024**

| Package | Step | Counties | Median | Mean | Share at 50% or more |
|---|---|---|---|---|---|
| MP3 | Draft as printed: adopters over every rdu of the county | 3,098 | 50.0000% | 49.6139% | 53.8089% |
| MP3 | S0 draft (adopters over rdu counted) | 3,090 | 68.7500% | 67.2939% | 87.7994% |
| MP3 | S1 income draw (adopters over rdu counted) | 3,090 | 68.8650% | 67.3250% | 87.9612% |
| MP3 | S2 energy fix (adopters over rdu counted) | 3,090 | 50.0000% | 46.3059% | 50.8091% |
| MP3 | S3 sample flag (adopters over rdu counted) | 3,090 | 50.0000% | 46.3059% | 50.8091% |
| MP3 | S4a no-AC homes out (derived) (adopters over rdu counted) | 3,079 | 54.5455% | 49.9160% | 56.2520% |
| MP3 | S4b shared cooling out (derived) (adopters over rdu counted) | 3,079 | 54.5455% | 49.9216% | 56.2520% |
| MP3 | S5 other changes, 25 Sep to 4 Oct (adopters over rdu counted) | 3,079 | 54.5455% | 49.9216% | 56.2520% |
| MP3 | S6 no rounding (adopters over rdu counted) | 3,079 | 54.5455% | 49.8247% | 56.1221% |
| MP3 | S7 damages in USD2025 (adopters over rdu counted) | 3,079 | 54.5455% | 49.8247% | 56.1221% |
| MP3 | S8 boiler pricing (adopters over rdu counted) | 3,079 | 55.5556% | 50.4666% | 56.5119% |
| MP4 | Draft as printed: adopters over every rdu of the county | 3,098 | 40.0000% | 39.5805% | 28.1472% |
| MP4 | S0 draft (adopters over rdu counted) | 3,090 | 54.7778% | 54.0226% | 63.2039% |
| MP4 | S1 income draw (adopters over rdu counted) | 3,090 | 54.7394% | 53.9674% | 63.1715% |
| MP4 | S2 energy fix (adopters over rdu counted) | 3,090 | 50.0000% | 48.3524% | 52.8479% |
| MP4 | S3 sample flag (adopters over rdu counted) | 3,090 | 50.0000% | 48.3524% | 52.8479% |
| MP4 | S4a no-AC homes out (derived) (adopters over rdu counted) | 3,079 | 54.5455% | 51.4584% | 58.3306% |
| MP4 | S4b shared cooling out (derived) (adopters over rdu counted) | 3,079 | 54.5455% | 51.4641% | 58.3306% |
| MP4 | S5 other changes, 25 Sep to 4 Oct (adopters over rdu counted) | 3,079 | 54.5455% | 51.4641% | 58.3306% |
| MP4 | S6 no rounding (adopters over rdu counted) | 3,079 | 54.2857% | 51.4045% | 58.2657% |
| MP4 | S7 damages in USD2025 (adopters over rdu counted) | 3,079 | 54.2857% | 51.4045% | 58.2657% |
| MP4 | S8 boiler pricing (adopters over rdu counted) | 3,079 | 56.6667% | 52.7149% | 60.1819% |

**June 2026**

| Package | Step | Counties | Median | Mean | Share at 50% or more |
|---|---|---|---|---|---|
| MP3 | Draft as printed: adopters over every rdu of the county | 3,098 | 32.5581% | 32.8505% | 14.9774% |
| MP3 | S0 draft (adopters over rdu counted) | 3,090 | 44.9490% | 45.1374% | 44.2395% |
| MP3 | S1 income draw (adopters over rdu counted) | 3,090 | 45.0000% | 45.1283% | 44.2071% |
| MP3 | S2 energy fix (adopters over rdu counted) | 3,090 | 27.7778% | 31.0314% | 27.0550% |
| MP3 | S3 sample flag (adopters over rdu counted) | 3,090 | 27.7778% | 31.0314% | 27.0550% |
| MP3 | S4a no-AC homes out (derived) (adopters over rdu counted) | 3,079 | 30.7692% | 32.9149% | 29.9773% |
| MP3 | S4b shared cooling out (derived) (adopters over rdu counted) | 3,079 | 30.7692% | 32.9121% | 29.9773% |
| MP3 | S5 other changes, 25 Sep to 4 Oct (adopters over rdu counted) | 3,079 | 30.7692% | 32.9121% | 29.9773% |
| MP3 | S6 no rounding (adopters over rdu counted) | 3,079 | 30.7692% | 32.8666% | 29.9448% |
| MP3 | S7 damages in USD2025 (adopters over rdu counted) | 3,079 | 30.7692% | 32.8666% | 29.9448% |
| MP3 | S8 boiler pricing (adopters over rdu counted) | 3,079 | 31.5789% | 33.3200% | 30.7567% |
| MP4 | Draft as printed: adopters over every rdu of the county | 3,098 | 25.0000% | 26.2545% | 6.5203% |
| MP4 | S0 draft (adopters over rdu counted) | 3,090 | 34.6154% | 36.1103% | 26.4725% |
| MP4 | S1 income draw (adopters over rdu counted) | 3,090 | 34.6154% | 36.0968% | 26.3430% |
| MP4 | S2 energy fix (adopters over rdu counted) | 3,090 | 32.2781% | 33.0959% | 25.3722% |
| MP4 | S3 sample flag (adopters over rdu counted) | 3,090 | 32.2781% | 33.0959% | 25.3722% |
| MP4 | S4a no-AC homes out (derived) (adopters over rdu counted) | 3,079 | 33.3333% | 34.6725% | 28.6132% |
| MP4 | S4b shared cooling out (derived) (adopters over rdu counted) | 3,079 | 33.3333% | 34.6720% | 28.5807% |
| MP4 | S5 other changes, 25 Sep to 4 Oct (adopters over rdu counted) | 3,079 | 33.3333% | 34.6720% | 28.5807% |
| MP4 | S6 no rounding (adopters over rdu counted) | 3,079 | 33.3333% | 34.6222% | 28.4183% |
| MP4 | S7 damages in USD2025 (adopters over rdu counted) | 3,079 | 33.3333% | 34.6222% | 28.4183% |
| MP4 | S8 boiler pricing (adopters over rdu counted) | 3,079 | 33.3333% | 34.9855% | 29.0354% |

**Adopters under December 2024 rules who do not adopt under June 2026 rules.**

| Package | Step | rdu counted | Adopters lost (rdu) | Adopters lost (homes) | Points of the sample | rdu that gain adoption | Lost with an electric baseline |
|---|---|---|---|---|---|---|---|
| MP3 | S0 draft | 260,211 | 70,585 | 17,090,818 | 27.1261 | 0 | 0 |
| MP3 | S1 income draw | 260,211 | 70,530 | 17,077,500 | 27.1049 | 0 | 0 |
| MP3 | S2 energy fix | 260,211 | 44,452 | 10,763,208 | 17.0831 | 0 | 0 |
| MP3 | S3 sample flag | 260,211 | 44,452 | 10,763,208 | 17.0831 | 0 | 0 |
| MP3 | S4a no-AC homes out (derived) | 221,301 | 43,254 | 10,473,135 | 19.5453 | 0 | 0 |
| MP3 | S4b shared cooling out (derived) | 221,205 | 43,253 | 10,472,893 | 19.5534 | 0 | 0 |
| MP3 | S5 other changes, 25 Sep to 4 Oct | 221,205 | 43,253 | 10,472,893 | 19.5534 | 0 | 0 |
| MP3 | S6 no rounding | 221,205 | 43,116 | 10,439,721 | 19.4914 | 0 | 0 |
| MP3 | S7 damages in USD2025 | 221,205 | 43,116 | 10,439,721 | 19.4914 | 0 | 0 |
| MP3 | S8 boiler pricing | 221,205 | 43,717 | 10,585,241 | 19.7631 | 0 | 0 |
| MP4 | S0 draft | 260,211 | 48,720 | 11,796,623 | 18.7233 | 0 | 0 |
| MP4 | S1 income draw | 260,211 | 48,693 | 11,790,085 | 18.7129 | 0 | 0 |
| MP4 | S2 energy fix | 260,211 | 43,964 | 10,645,048 | 16.8955 | 0 | 0 |
| MP4 | S3 sample flag | 260,211 | 43,964 | 10,645,048 | 16.8955 | 0 | 0 |
| MP4 | S4a no-AC homes out (derived) | 221,301 | 43,009 | 10,413,813 | 19.4346 | 0 | 0 |
| MP4 | S4b shared cooling out (derived) | 221,205 | 43,007 | 10,413,328 | 19.4421 | 0 | 0 |
| MP4 | S5 other changes, 25 Sep to 4 Oct | 221,205 | 43,007 | 10,413,328 | 19.4421 | 0 | 0 |
| MP4 | S6 no rounding | 221,205 | 42,962 | 10,402,433 | 19.4218 | 0 | 0 |
| MP4 | S7 damages in USD2025 | 221,205 | 42,962 | 10,402,433 | 19.4218 | 0 | 0 |
| MP4 | S8 boiler pricing | 221,205 | 45,283 | 10,964,419 | 20.4711 | 0 | 0 |

**Demand at 100% adoption, for each set of homes.** EIA residential sales: 1,514,993 GWh
(from the draft's deleted calc notes, as quoted in the version reconciliation).

| Homes counted | Package | rdu | Homes | Counties | Baseline (GWh) | Change (GWh) | Change, % of baseline | Change, % of 1,514,993 GWh | Counties with an increase | Median county change | Mean county change | County range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Draft: occupied single-family rdu the package applies to | MP3 | 331,526 | 80,272,726 | 3,098 | 1,136,778.6108 | +427,043.7197 | +37.5661% | 28.1878% | 69.5287% (2,154) | +22.4926% | +50.5522% | -41.58% to +622.33% |
| rdu with a result (run 2026-09-25_22-55) | MP3 | 260,211 | 63,005,153 | 3,090 | 892,740.7135 | +353,404.3626 | +39.5865% | 23.3271% | 66.4078% (2,052) | +20.3004% | +50.0363% | -41.58% to +981.37% |
| less homes with no central or room AC (derived) | MP3 | 221,301 | 53,583,835 | 3,079 | 795,714.9800 | +278,295.4884 | +34.9743% | 18.3694% | 64.4040% (1,983) | +18.1811% | +49.2125% | -45.01% to +622.33% |
| Current study sample | MP3 | 221,205 | 53,560,591 | 3,079 | 795,464.1896 | +278,176.5931 | +34.9703% | 18.3616% | 64.4040% (1,983) | +18.1811% | +49.2144% | -45.01% to +622.33% |
| Draft: occupied single-family rdu the package applies to | MP4 | 331,526 | 80,272,726 | 3,098 | 1,136,778.6108 | +30,618.3300 | +2.6934% | 2.0210% | 44.0284% (1,364) | -8.0495% | +12.2565% | -56.87% to +474.96% |
| rdu with a result (run 2026-09-25_22-55) | MP4 | 260,211 | 63,005,153 | 3,090 | 892,740.7135 | +34,992.7771 | +3.9197% | 2.3098% | 43.1715% (1,334) | -10.4474% | +11.7516% | -60.62% to +836.41% |
| less homes with no central or room AC (derived) | MP4 | 221,301 | 53,583,835 | 3,079 | 795,714.9800 | +1,298.8171 | +0.1632% | 0.0857% | 42.0266% (1,294) | -12.4931% | +11.2224% | -65.95% to +474.96% |
| Current study sample | MP4 | 221,205 | 53,560,591 | 3,079 | 795,464.1896 | +1,278.6002 | +0.1607% | 0.0844% | 42.0266% (1,294) | -12.4931% | +11.2234% | -65.95% to +474.96% |

### 2.1 Abstract (p2)

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| A1 | p2, abstract | "economically viable without subsidies for 18-27% (15-22 million) of modeled single-family homes" | MP4 18.0984% (47,094 of 260,211 rdu), MP3 27.1760% (70,715 rdu) [D]. Printed counts are rate x 80,273,601 homes: 14.53M and 21.82M. Adopter rdu x weight: 11.40M and 17.12M | MP3 19.3472% (42,797 of 221,205 rdu = 10,362,481 homes); MP4 19.4675% (43,063 rdu = 10,426,888 homes) [C] | "about 19% (10.4 million)"; in whole millions, "10 million" | [XH] 42,797 and 43,063 rdu; equal | matches crosswalk | MP3 -7.83 points: energy fix -10.52, sample rule +2.15, boiler pricing +0.54. MP4 +1.37: energy fix -1.07, sample rule +1.96, boiler pricing +0.48. Counts also fall because they are now adopters x weight |
| A2 | p2, abstract | "could rise to 36-49 million with the subsidies in the Inflation Reduction Act" | MP4 45.1168% (117,399 rdu), MP3 61.2169% (159,293 rdu) [D]. Printed: 36.22M and 49.14M. Adopter rdu x weight: 28.43M and 38.57M | MP3 46.6409% (103,172 rdu = 24,981,141 homes); MP4 49.3660% (109,200 rdu = 26,440,707 homes) [C] | "25.0-26.4 million"; in whole millions, "25-26 million" | not in exports (December 2024 case) | matches crosswalk | MP3 -14.58 points: energy fix -20.18, sample rule +4.91, boiler pricing +0.81, no rounding -0.10, income draw -0.005. MP4 +4.25: energy fix -2.21, sample rule +4.98, boiler pricing +1.53, no rounding -0.05, income draw +0.005 |
| A3 | p2, abstract | "or 21-27 million under new guidance eliminating fuel-switching project eligibility" | MP4 26.3936% (68,679 rdu), MP3 34.0908% (88,708 rdu) [D]. Printed: 21.19M and 27.37M. Adopter rdu x weight: 16.63M and 21.48M | MP3 26.8778% (59,455 rdu = 14,395,899 homes); MP4 28.8949% (63,917 rdu = 15,476,288 homes) [C] | "14.4-15.5 million"; in whole millions, "14-15 million" | Rebuilt from [XH] as no-rebate NPV plus the June 2026 rebate amount >= 0: 59,455 and 63,917 rdu; equal on every rdu | matches crosswalk | MP3 -7.21 points: energy fix -10.16, sample rule +2.44, boiler pricing +0.54, no rounding -0.04, income draw +0.02. MP4 +2.50: energy fix -0.39, sample rule +2.43, boiler pricing +0.48, no rounding -0.03, income draw +0.02 |
| A4 | p2, abstract | "increase annual residential electricity demand by up to 3-38% (depending on equipment efficiency)" | MP4 +2.6934% (+30,618.33 GWh), MP3 +37.5661% (+427,043.72 GWh), baseline 1,136,778.61 GWh, 331,526 rdu [RS] | MP4 +0.1607% (+1,278.60 GWh); MP3 +34.9703% (+278,176.59 GWh); baseline 795,464.19 GWh, 221,205 rdu [RS] | "0.2-35%" | [XC] sum of `elec_change_gwh`: 1,278.84 and 278,176.75 (each county rounded to 2 decimals). [XH] total electricity columns: 1,278.6002 and 278,176.5931 | matches crosswalk | Sample rule only. Demand is read from ResStock: 331,526 --> 260,211 --> 221,301 --> 221,205 rdu |

### 2.2 Results, Fig 1 section (p5 par 3 and p6 par 1)

The shares on p6 ("40% of counties (46% of homes)", "49% of counties (41% of homes)") and
the tract shares on p4 come from the heating-fuel toolkit and the American Community
Survey, not from a TARE run. They are not checked here (section 7). The hypothesis in p6
par 1 is tested in section 3, U3.

### 2.3 Results, county adoption with no rebate (p8 par 1)

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| F2-1 | p8 par 1 | "minimum-efficiency heat pump is 27% (22 million single-family homes)" | 27.1760%; printed 21.82M; adopter homes 17.12M [D] | 19.3472%; 10,362,481 homes [C] | "19% (10.4 million)" | [XH] equal | matches crosswalk | as A1 |
| F2-2 | p8 par 1 | "only 18% of single-family homes (15 million)" | 18.0984%; printed 14.53M; adopter homes 11.40M [D] | 19.4675%; 10,426,888 homes [C] | "19% (10.4 million)". The two packages are now 0.12 points apart, MP4 higher | [XH] equal | matches crosswalk | as A1 |
| F2-3 | p8 par 1 | "Across nearly 3,100 analyzed counties" | 3,098 counties in the draft's county table; 3,090 hold an rdu with a result [D] | 3,079 counties [C][F] | "3,079" ("nearly 3,100" still rounds right) | [XC] 3,079 rows | matches crosswalk | sample rule |
| F2-4 | p8 par 1 | "median county-level adoption potential for the minimum-efficiency heat pump is 27% (mean 29%), and roughly 10% of counties are at or above 50%" | As printed, adopters over every rdu of the county, 3,098 counties: median 27.4510%, mean 28.5846%, 9.8128% at 50% or more. Over rdu with a result, 3,090 counties: 37.9310%, 39.2010%, 31.9417% [D] | Adopters over study-sample rdu, 3,079 counties: median 23.5741%, mean 26.8143%, 17.3433% [C] | "median 24% (mean 27%), about 17% of counties at or above 50%". Say that a county rate is a share of modeled homes | [XC] 23.57, 26.8143, 17.3433 | matches crosswalk | Median: county denominator +10.48 points, energy fix -17.10, sample rule +2.24, boiler pricing +0.50 |
| F2-5 | p8 par 1 | "high-efficiency heat pump ... 17% (mean 19%), with about 3% of counties at or above 50%" | As printed: 16.6667%, 18.8041%, 3.1633%. Over rdu with a result: 23.2717%, 25.7104%, 10.9385% [D] | 21.8638%, 24.3321%, 10.2306% [C] | "median 22% (mean 24%), about 10% of counties at or above 50%" | [XC] 21.86, 24.3320, 10.2306 | matches crosswalk | Median: county denominator +6.61 points, energy fix -3.27, sample rule +1.43, boiler pricing +0.44 |
| F2-6 | p8 par 1 | "propane- and fuel-oil-heated homes have the highest unsubsidized adoption potential of any fuel group (73% and 62% for the minimum-efficiency system, respectively)" | Propane 73.3950% (10,232 of 13,941 rdu); fuel oil 61.7383% (10,243 of 16,591) [D] | Propane 40.6401% (4,978 of 12,249); fuel oil 14.5555% (2,019 of 13,871). Electricity is now the highest group, 45.1514% [C] | "41% and 15%"; the order is now electricity 45%, propane 41%, fuel oil 15%, natural gas 5% | [XH] equal | matches crosswalk | Propane: energy fix -37.11 points, sample rule +3.85, boiler pricing +0.51. Fuel oil: energy fix -50.18, sample rule +1.66, boiler pricing +1.33 |
| F2-7 | p8 par 1 | "followed by 44% for electrically heated homes and 12% for homes using natural gas (Fig 4A)" | Electricity 43.7428% (32,238 of 73,699 rdu); natural gas 11.5412% (18,002 of 155,980) [D] | Electricity 45.1514% (28,798 of 63,781); natural gas 5.3327% (7,002 of 131,304) [C] | "45% ... and 5%" | [XH] equal | matches crosswalk | Electricity: energy fix -2.62 points, sample rule +4.03. Natural gas: energy fix -7.65, sample rule +0.72, boiler pricing +0.72 |
| F2-8 | p8 par 1 | "Counties dominated by natural gas ... with only 4% of homes adopting a high-efficiency heat pump (Fig 4B) and 12% adopting a less efficient single-stage system (Fig 4A)" | MP4 4.2980% (6,704 of 155,980 rdu); MP3 11.5412%: rates of gas-heated homes nationally [D] | MP4 6.3197% (8,298 of 131,304); MP3 5.3327% [C] | "6% ... and 5%" (MP4 is now above MP3). For counties, see U3 | [XH] equal | matches crosswalk; the draft's wording mixes homes and counties (section 1.1, item 7) | MP4 natural gas: energy fix +0.61 points, sample rule +0.91, boiler pricing +0.50 |

### 2.4 Results, demand at 100% adoption (p10)

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| DM-1 | p10 par 1 | "63M modeled single-family homes" (twice) | 63,005,153 homes is the 260,211 rdu with a result; the demand numbers cover 331,526 rdu = 80,272,726 homes [RS] | 221,205 rdu = 53,560,591 homes [F] | "53.6M" | [XH] 221,205 rows; [XC] `home_count` sums to 53,560,590.67 | matches crosswalk (the draft's mismatch is confirmed) | sample rule |
| DM-2 | p10 par 1 | "by 427,000 GWh for the min-efficiency system (about a 38% increase relative to the modeled baseline)" | +427,043.7197 GWh; +37.5661%; baseline 1,136,778.6108 GWh [RS] | +278,176.5931 GWh; +34.9703%; baseline 795,464.1896 GWh [RS] | "278,000 GWh (about a 35% increase)" | [XC] 278,176.75; [XH] 278,176.5931 | matches crosswalk | Sample rule: -73,639 GWh (to 260,211 rdu), -75,109 (homes with no AC out), -119 (shared cooling out) |
| DM-3 | p10 par 1 | "by only 30,600 GWh (about a 3% increase relative to the modeled baseline)" | +30,618.3300 GWh; +2.6934% [RS] | +1,278.6002 GWh; +0.1607% [RS] | "1,280 GWh (about a 0.2% increase)" | [XC] 1,278.84; [XH] 1,278.6002 | matches crosswalk | Sample rule: +4,374 GWh (to 260,211 rdu), -33,694 (homes with no AC out), -20 (shared cooling out) |
| DM-4 | p10 par 1 | "(427,000 GWh) represents about 28% of total current U.S. annual residential electricity sales" | 28.1878% of 1,514,993 GWh | 18.3616% | "about 18%" | from DM-2 | discrepancy with the crosswalk's 18.2% (section 1.2, item 1); same rounded text | sample rule |
| DM-5 | p10 par 1 | "this drops to only about 2% if the high-efficiency variable-speed system is installed instead [22]" | 2.0210% | 0.0844% | "less than 0.1%" | from DM-3 | matches crosswalk | sample rule |
| DM-6 | p10 par 2 | "70% of the 3,100 modeled counties ... would have a net increase in electricity demand" | 69.5287% (2,154 of 3,098 counties) [RS] | 64.4040% (1,983 of 3,079 counties) [RS] | "64% of the 3,079 modeled counties" | [XC] 64.4040% of 3,079 | matches crosswalk | Sample rule: 66.41% on 260,211 rdu, 64.40% once homes with no AC are out |
| DM-7 | p10 par 2 | "with a 23% median county-level increase (Fig 3B)" | +22.4926%, which rounds to 22% [RS] | +18.1811% [RS] | "an 18% median county-level increase (Fig 3A)" | [XC] 18.18 | matches crosswalk for the current value; the draft's 23 and its panel letter do not check (section 1.1, items 1 and 6) | Sample rule: 22.49 --> 20.30 --> 18.18 |
| DM-8 | p10 par 2 | "County-level increases can exceed 600%" | Largest: +622.3300% (MP3), +474.9585% (MP4), the same county [RS] | The same: +622.3300% and +474.9585% [RS] | See U5 before keeping this sentence | [XC] 622.33 and 474.96 | matches crosswalk on the value; one county of 2 rdu (section 1.2, item 6) | none: the county's 2 rdu are in every set |
| DM-9 | p10 par 2 | "cuts the share of counties with increasing demand from 70% to 44%" | 69.5287% --> 44.0284% (1,364 of 3,098) [RS] | 64.4040% --> 42.0266% (1,294 of 3,079) [RS] | "from 64% to 42%" | [XC] 42.0266% | matches crosswalk | sample rule |
| DM-10 | p10 par 2 | "turns the median county-level change into a net decrease of 8%" | -8.0495% [RS] | -12.4931% [RS] | "a net decrease of 12%" | [XC] -12.49 | matches crosswalk (-12.5%); see section 1.2, item 7, on rounding | Sample rule: -8.05 --> -10.45 --> -12.49 |
| DM-11 | not in reference draft (deleted 24 Sep) | "a 93% reduction relative to the minimum-efficiency case" | 92.8302% | 99.5404% | optional: "more than 99%" | from DM-2 and DM-3 | not in reference draft; matches crosswalk (99.5%) | sample rule |
| DM-12 | not in reference draft (deleted 24 Sep) | "the mean county-level increase ... is 51%" | +50.5522% (3,098 counties) [RS] | +49.2144% (3,079 counties); MP4 +11.2234% [RS] | optional: "49%"; moot unless the sentence comes back | [XC] mean of `pct_elec_demand_change`: 49.2144 and 11.2234 | not in reference draft; the crosswalk had no value | sample rule |
| DM-13 | not in reference draft | Crosswalk's note on duplicated sentences in this section | The accepted text has no duplicate. One thing is still open: no paragraph break before "In the Supplementary Materials, we include a map" (p10 par 2) | not a number | restore the paragraph break | not in exports | not in reference draft | none |

### 2.5 Results, rebate policy (p12 par 1)

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| RB-1 | p12 par 1 | "from 27% of single-family homes without any subsidy (22 million) to 61% (49 million)" | 27.1760% --> 61.2169%; printed 21.82M --> 49.14M [D] | 19.3472% (10,362,481 homes) --> 46.6409% (24,981,141 homes) [C] | "from 19% (10.4 million) to 47% (25.0 million)" | no rebate: [XH] equal; December 2024: not in exports | matches crosswalk | as A1 and A2 |
| RB-2 | p12 par 1 | "The June 2026 rules reduce this to 34% (27 million)" | 34.0908%; printed 27.37M [D] | 26.8778% (14,395,899 homes) [C] | "27% (14.4 million)" | rebuilt from [XH]; equal | matches crosswalk | as A3 |
| RB-3 | p12 par 1 | "eliminating 22 million homes from the rebate program's original benefit" | 70,585 rdu adopt under December 2024 and not under June 2026 = 27.1261 points of 260,211. Printed: 27.1261% x 80.27M = 21.78M. Adopter rdu x weight: 17.09M [D] | 43,717 rdu = 10,585,241 homes (19.7631 points). MP4: 45,283 rdu = 10,964,419 homes [C] | "10.6 million" | not in exports (needs the December 2024 case) | matches crosswalk | In rdu: energy fix -26,078, sample rule -1,199, boiler pricing +601, no rounding -137, income draw -55 |
| RB-4 | p12 par 1 | "unsubsidized adoption potential of 18%, 45% under December 2024 rebate eligibility rules, and 26% under June 2026 eligibility rules" | 18.0984%, 45.1168%, 26.3936% [D] | 19.4675%, 49.3660%, 28.8949% [C] | "19%, 49% ... and 29%" | 19.4675% [XH]; 49.3660% not in exports; 28.8949% rebuilt from [XH], equal | matches crosswalk | as A1 to A3 |
| RB-5 | p12 par 1 | "This reduction falls entirely on homes that heat with fossil fuels" | All 70,585 (MP3) and 48,720 (MP4) lost adopter rdu have a fossil baseline; 0 electric; 0 rdu gain adoption [D] | All 43,717 (MP3: natural gas 37,585, propane 3,426, fuel oil 2,706) and 45,283 (MP4: 36,519, 4,360, 4,404); 0 electric; 0 rdu gain adoption [C] | keep "entirely", as modeled (crosswalk section 6, item 1) | not in exports | matches crosswalk | none |
| RB-6 | p12 par 1 | "natural-gas adoption potential falls from 54% under December 2024 to its unsubsidized level of 12%" | 54.2685% --> 11.5412% (MP3) [D] | 33.9571% --> 5.3327% (MP3); MP4 34.1322% --> 6.3197% [C] | "from 34% ... to 5%" | 5.3327% [XH]; December 2024 not in exports | matches crosswalk | December 2024 rate: energy fix -26.15 points, sample rule +4.77, boiler pricing +1.17 |
| RB-7 | p12 par 1 | "with fuel oil (73% to 62%) and propane (88% to 73%) reverting the same way" | Fuel oil 72.8286% --> 61.7383%; propane 88.4513% --> 73.3950% (MP3) [D] | Fuel oil 34.0639% --> 14.5555%; propane 68.6097% --> 40.6401% (MP3) [C] | "fuel oil (34% to 15%) and propane (69% to 41%)" | no-rebate values [XH]; December 2024 not in exports | matches crosswalk | December 2024 rates: fuel oil, energy fix -43.44 points, sample rule +3.48, boiler pricing +1.29; propane, -25.29, +5.02, +0.60 |
| RB-8 | p12 par 1 | "holds at 68% for the minimum-efficiency system, and 64% for the high-efficiency system under both rule sets -- well above the 44% and 34% they would reach without any subsidy" | 68.1570% and 63.5124%; 43.7428% and 34.2243% [D] | 71.2689% and 72.4605%; 45.1514% and 39.7642% [C] | "71% ... and 72% ... well above the 45% and 40%" | 45.1514% and 39.7642% [XH]; the rebate values rebuilt from [XH] | matches crosswalk | With rebate: MP3 sample rule +4.57 points, energy fix -1.36; MP4 energy fix +5.26, sample rule +3.74 |

### 2.6 Results, Allegheny County peaks (p14 par 1)

All values [TS]: the cached hourly ResStock tables, through the repo's
`prepare_bsq_timeseries` and `compute_county_scenario_profile`. Hour 1 is the hour that
starts 1 Jan 2018 at 00:00; hour 4433 is 4 July, 16:00; hour 152 is 7 January, 07:00.
Peaks are not in the exports.

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| P-1 | p14 par 1 | "with a peak of 863 MW" | 862.5113 MW @ hour 4433; 1,610 rdu | 714.5146 MW @ hour 4433; 1,146 rdu | "715 MW" | not in exports | matches crosswalk | Sample rule (the set of homes): 862.51 --> 762.75 MW @ 4362 on 1,356 rdu --> 714.51 |
| P-2 | p14 par 1 | "a peak that occurs in early July (hour 4433 of the year)" | hour 4433 = 4 July, 16:00 | hour 4433 | no change | not in exports | matches crosswalk | none |
| P-3 | p14 par 1 | "Under all modeled scenarios, the adoption of heat pumps in Allegheny County leads to a net increase in peak demand and shifts the summer cooling peak to a winter heating peak" | True in all four: 100% MP3 6,629.8728 MW @ 152; 100% MP4 5,364.0998 MW @ 152; economic adopters MP3 (256 rdu) 1,855.5931 MW @ 153; MP4 (125 rdu) 1,097.1036 MW @ 153 | True at 100% only. Economic adopters, grid impact run of 11 Oct 2026 on the current run: MP3 45 rdu, 713.8625 MW @ 4433 (-0.6521 MW); MP4 68 rdu, 715.1649 MW @ 4433 (+0.6504 MW) | "Under 100% adoption ..."; with economic adopters the peak stays in July | not in exports | matches crosswalk for MP3; the MP4 value replaces the crosswalk's 715.26 MW (section 1.2, item 5) | Energy fix (who adopts: 256 --> 46 and 125 --> 73 on the 1,356 rdu), then sample rule; boiler pricing adds one MP4 adopter |
| P-4 | p14 par 1 | "the aggregate household peak reaches 6,630 MW (hour 152), nearly 8 times the baseline" | 6,629.8728 MW @ 152; 7.6867 times | 4,901.6039 MW @ 152; 6.8600 times | "4,900 MW (hour 152), nearly 7 times the baseline" | not in exports | matches crosswalk | Sample rule: 6,629.87 --> 5,795.71 (1,356 rdu) --> 4,901.60 |
| P-5 | p14 par 1 | "(Fig 5D) results in a peak electric load that is roughly 20% lower than the minimum-efficiency bound" | 5,364.0998 MW; 19.0920% lower | 4,060.1916 MW; 17.1661% lower | "roughly 17% lower" | not in exports | matches crosswalk | Sample rule: 5,364.10 --> 4,791.72 (1,356 rdu) --> 4,060.19 |
| P-6 | p14 par 1 | "natural-gas-heated homes, which make up 92% of the county's representative homes" | 91.9876% (1,481 of the 1,610 county rows; one row has no fuel label); 92.1091% (1,249 of the 1,356 rdu with a result) [D] | 91.6230% (1,050 of 1,146 rdu) [C] | "92%" | [XA] 1,050 of 1,146 | matches crosswalk | sample rule |
| P-7 | p14 par 1 | "(Fig 5A and Fig 5C), resulting in a more modest increase in peak load" | +993.0817 MW (MP3, 256 adopters) and +234.5923 MW (MP4, 125 adopters) | Grid impact run of 11 Oct 2026 on the current run: -0.6521 MW (MP3, 45 adopters), +0.6504 MW (MP4, 68 adopters). Before boiler pricing MP4 was +0.7433 MW with 67 adopters | The peak barely moves: MP3 slightly lower, MP4 under 1 MW higher | not in exports | matches crosswalk for MP3; the MP4 value replaces the crosswalk's +0.74 MW (section 1.2, item 5) | energy fix and sample rule; boiler pricing adds one MP4 adopter |

**Every peak worked out here.** The rows for the 1,356-rdu set reproduce the reference
values of 25 Sep 2026, which checks the method. The rows for the 1,146-rdu set at 100%
adoption, and with the economic adopters of run `2026-10-09_00-46`, equal the peaks the
main notebook printed on 11 Oct 2026, to the two decimals it prints (section 2.13).

| Homes | Package | Scenario | Adopters (rdu) | Old systems: peak MW @ hour | After retrofit: peak MW @ hour (2018) | Change (MW) | Ratio to the old peak | Peak moves to winter |
|---|---|---|---|---|---|---|---|---|
| 1,610 rdu (every county row, draft) | MP3 | 100% adoption | 1,610 | 862.5113 @ 4433 | 6,629.8728 @ 152 (07 Jan, 07:00) | +5,767.3615 | 7.6867 | yes |
| 1,610 rdu (every county row, draft) | MP3 | economic adopters of run 2026-08-19_20-56 | 256 | 862.5113 @ 4433 | 1,855.5931 @ 153 (07 Jan, 08:00) | +993.0817 | 2.1514 | yes |
| 1,610 rdu (every county row, draft) | MP4 | 100% adoption | 1,610 | 862.5113 @ 4433 | 5,364.0998 @ 152 (07 Jan, 07:00) | +4,501.5884 | 6.2192 | yes |
| 1,610 rdu (every county row, draft) | MP4 | economic adopters of run 2026-08-19_20-56 | 125 | 862.5113 @ 4433 | 1,097.1036 @ 153 (07 Jan, 08:00) | +234.5923 | 1.2720 | yes |
| 1,356 rdu (rdu with a result) | MP3 | 100% adoption | 1,356 | 762.7468 @ 4362 | 5,795.7135 @ 152 (07 Jan, 07:00) | +5,032.9667 | 7.5985 | yes |
| 1,356 rdu (rdu with a result) | MP3 | economic adopters of run 2026-08-19_20-56 | 256 | 762.7468 @ 4362 | 1,770.8804 @ 153 (07 Jan, 08:00) | +1,008.1335 | 2.3217 | yes |
| 1,356 rdu (rdu with a result) | MP3 | economic adopters of run 2026-09-25_22-55 | 46 | 762.7468 @ 4362 | 767.9255 @ 140 (06 Jan, 19:00) | +5.1787 | 1.0068 | yes |
| 1,356 rdu (rdu with a result) | MP4 | 100% adoption | 1,356 | 762.7468 @ 4362 | 4,791.7241 @ 152 (07 Jan, 07:00) | +4,028.9772 | 6.2822 | yes |
| 1,356 rdu (rdu with a result) | MP4 | economic adopters of run 2026-08-19_20-56 | 125 | 762.7468 @ 4362 | 1,012.3909 @ 153 (07 Jan, 08:00) | +249.6441 | 1.3273 | yes |
| 1,356 rdu (rdu with a result) | MP4 | economic adopters of run 2026-09-25_22-55 | 73 | 762.7468 @ 4362 | 763.1514 @ 4433 (04 Jul, 16:00) | +0.4046 | 1.0005 | no |
| 1,146 rdu (study sample) | MP3 | 100% adoption | 1,146 | 714.5146 @ 4433 | 4,901.6039 @ 152 (07 Jan, 07:00) | +4,187.0893 | 6.8600 | yes |
| 1,146 rdu (study sample) | MP3 | economic adopters of run 2026-08-19_20-56 | 251 | 714.5146 @ 4433 | 1,642.3250 @ 153 (07 Jan, 08:00) | +927.8104 | 2.2985 | yes |
| 1,146 rdu (study sample) | MP3 | economic adopters of run 2026-10-05_21-33 | 45 | 714.5146 @ 4433 | 713.8625 @ 4433 (04 Jul, 16:00) | -0.6521 | 0.9991 | no |
| 1,146 rdu (study sample) | MP3 | economic adopters of run 2026-10-09_00-46 | 45 | 714.5146 @ 4433 | 713.8625 @ 4433 (04 Jul, 16:00) | -0.6521 | 0.9991 | no |
| 1,146 rdu (study sample) | MP4 | 100% adoption | 1,146 | 714.5146 @ 4433 | 4,060.1916 @ 152 (07 Jan, 07:00) | +3,345.6770 | 5.6824 | yes |
| 1,146 rdu (study sample) | MP4 | economic adopters of run 2026-08-19_20-56 | 118 | 714.5146 @ 4433 | 919.0693 @ 153 (07 Jan, 08:00) | +204.5547 | 1.2863 | yes |
| 1,146 rdu (study sample) | MP4 | economic adopters of run 2026-10-05_21-33 | 67 | 714.5146 @ 4433 | 715.2579 @ 4433 (04 Jul, 16:00) | +0.7433 | 1.0010 | no |
| 1,146 rdu (study sample) | MP4 | economic adopters of run 2026-10-09_00-46 | 68 | 714.5146 @ 4433 | 715.1649 @ 4433 (04 Jul, 16:00) | +0.6504 | 1.0009 | no |

### 2.7 Discussion (p17 par 3 and p18 par 1)

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| DS-1 | p17 par 3 | "is 27% (22 million single-family homes), while only 18% of single-family homes (15 million)" | 27.1760% and 18.0984%; printed 21.82M and 14.53M [D] | 19.3472% (10,362,481 homes) and 19.4675% (10,426,888 homes) [C] | "19% (10.4 million) ... and 19% (10.4 million)"; nearly identical | [XH] equal | matches crosswalk | as A1 |
| DS-2 | p17 par 3 | "decreases overall adoption potential by 15-22 million single-family homes" | MP4 18.7233 points x 80.27M = 15.03M (48,720 rdu = 11.80M homes); MP3 21.78M (70,585 rdu = 17.09M homes) [D] | MP3 43,717 rdu = 10,585,241 homes; MP4 45,283 rdu = 10,964,419 homes [C] | "10.6-11.0 million"; "about 11 million" | not in exports | matches crosswalk | as RB-3 |
| DS-3 | p17 par 3 | "which includes a 43% drop for homes with natural gas systems" | 42.7273 points (MP3: 54.2685 - 11.5412); MP4 26.0848 points [D] | 28.6244 points (MP3: 33.9571 - 5.3327); 27.8126 points (MP4: 34.1322 - 6.3197) [C] | "a 28-29 percentage point drop" | not in exports | matches crosswalk (28.6 and 27.8); percent should read percentage points (section 1.1, item 3) | energy fix, mostly: the December 2024 rate of gas homes fell more than the no-rebate rate |
| DS-4 | p17 par 3 | "and an 11-29% decrease for homes using fuel oil and propane" | Fuel oil 11.0904 (MP3), 23.4886 (MP4); propane 15.0563 (MP3), 29.6679 (MP4) points: rounds to 11-30 [D] | Fuel oil 19.5083 (MP3), 31.7497 (MP4); propane 27.9696 (MP3), 35.5947 (MP4) points [C] | "20-36 percentage points" | not in exports | matches crosswalk for the current values; the draft's 29 does not round from 29.67 (section 1.1, item 2) | energy fix (the no-rebate rates fell more than the December 2024 rates), then boiler pricing |
| DS-5 | p17 par 3 | "increasing up to 8-fold" | 7.6867 times [TS] | 6.8600 times [TS] | "up to 7-fold" | not in exports | matches crosswalk | sample rule |
| DS-6 | p18 par 1 | "$42 billion in HEEHR and $8 billion in HOMES rebates would flow to homes" | $41.5770B and $8.3012B: MP4, June 2026 rules, adopters only, heating-credit-only case [D] | Same metric: $38.2919B and $7.6552B [C] | See section 5.2 before reusing the sentence | not in exports as a total; the June 2026 amount is in [XH] but the heating-credit-only adopter flag is not | the crosswalk had no source; now traced (section 5.2) | energy fix and sample rule |

### 2.8 Methods (p19 to p25)

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| MT-1 | p19 par 2 | "all costs are expressed in inflation-adjusted 2025 dollars" | Private costs: yes. Climate damages were in 2023 dollars (records; the paper prints no damages) | Climate damages are in 2025 dollars from run `2026-10-05_21-33` on (records; not measured again here) | no change | not in exports | matches crosswalk (record only) | none |
| MT-2 | p20 par 1 | "reduced to ~330,000 dwelling units or ~80 million homes" | 331,526 rdu = 80,272,726 homes [RS]; results on 260,211 of them | 221,205 rdu = 53,560,591 homes [F] | "~221,000 dwelling units or ~53.6 million homes" | [XH] 221,205 rows | matches crosswalk | sample rule |
| MT-3 | p23, second Table XX | "15 SEER1, 9 HSPF1" (MP3), priced at "16 SEER1, 9.5 HSPF1"; MP4 "SEER 24" ducted and "SEER 29.3" non-ducted | not measured on the draft run | Every MP3 heat pump is priced at SEER1 16.0; MP4 at 24.0 (196,352 ducted rdu) or 29.3 (24,853 non-ducted rdu) (`heating_upgrade_pm2_euss`) [C] | no change | not in exports | holds | none |
| MT-4 | new | Crosswalk section 4.8: sentence on the filters, and sentence on the consumption fix | not a number | The filter counts behind the sentence are in section 2.10 | as the crosswalk suggests | not in exports | not a number | none |

### 2.9 Captions (p9, p11, p13, p15)

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| CP-1 | p9, Fig 2 caption | "The national sample size is 3,098 counties." | 3,098 counties in the county table (adopters over every rdu) [D] | 3,079 counties [C] | "3,079 counties" | [XC] 3,079 rows | matches crosswalk | sample rule |
| CP-2 | p11, Fig 3 caption | "The national sample size is 3,098 counties." | 3,098 counties in the demand table [RS] | 3,079 counties [RS] | "3,079 counties" | [XC] 3,079 rows | matches crosswalk | sample rule |
| CP-3 | p9 and p11 captions | "Gray counties indicate a county identifier mismatch, insufficient data, or too few modeled homes" (p9); "a county identifier mismatch or insufficient data" (p11) | see U15 | see U15 | see U15 | not in exports | not in the crosswalk | sample rule |
| CP-4 | p13, Fig 4 | every marker and y-axis label | section 2.11 | section 2.11 | section 2.11 | no-rebate markers: [XH] | matches crosswalk | section 2.11 |
| CP-5 | p15, Fig 5 caption | no number | - | - | - | - | nothing to check | - |

### 2.10 Table XX, the sample table (p21)

Every number of the draft's table equals the sample funnel file to the printed digits. The
current funnel [F] is byte-identical in runs `2026-10-04_23-02`, `2026-10-05_00-54`,
`2026-10-05_21-33`, `2026-10-09_00-46` and `2026-10-09_21-04` (SHA-256 below).

| ID | Row | Draft table (p21) | Current exact [F] | Status |
|---|---|---|---|---|
| T-1 | ResStock 2022.1.1 stock | 548,916 rdu; 132.91M homes | 548,916 rdu; 132,909,587 homes | holds |
| T-2 | Upgrade applicable | 548,260; 132.75M; removed 656 rdu, 0.16M | 548,260; 132,750,749 homes; removed 656 rdu, 158,838 homes | holds |
| T-3 | Occupied | 482,050; 116.72M; removed 66,210 rdu, 16.03M | 482,050; 116,719,255; removed 66,210 rdu, 16,031,494 homes | holds |
| T-4 | Single-family (attached, detached) | 331,526; 80.27M; removed 150,524 rdu, 36.45M | 331,526; 80,272,726; removed 150,524 rdu, 36,446,529 homes | holds |
| T-5 | Heating fuel: electricity, natural gas, propane, fuel oil | 321,352; 77.81M; removed 10,174 rdu, 2.46M | 321,352; 77,809,285; removed 10,174 rdu, 2,463,441 homes | holds |
| T-6 | No existing heat pump | 290,005; 70.22M; removed 31,347 rdu, 7.59M | 290,005; 70,219,204; removed 31,347 rdu, 7,590,081 homes | holds |
| T-7 | Heating system with cost data | 260,211; 63.01M; removed 29,794 rdu, 7.21M | 260,211; 63,005,153; removed 29,794 rdu, 7,214,051 homes | holds |
| T-8 | "Of the final sample: with central or room AC" | 221,301; 53.58M | Now a filter step, "Central or room AC": 221,301 rdu; 53,583,835 homes; removed 38,910 rdu, 9,421,318 homes | matches crosswalk: make it a step |
| T-9 | "Of the final sample: no AC (cooling counted as zero)" | 38,910; 9.42M | Not in the sample. Delete the row | matches crosswalk |
| T-10 | (new) Cooling system not shared | not in the draft | 221,205 rdu; 53,560,591 homes; removed 96 rdu, 23,245 homes (0.02M) | matches crosswalk |

The funnel file also has a step "Not Alaska or Hawaii" between T-4 and T-5 that removes 0
rdu (331,526 before and after).

| Run | Bytes | SHA-256 of the sample funnel |
|---|---|---|
| `2026-09-25_22-55` | 1,569 | `3be2e28d8157aa9412a2000f981b76084b1f90817548c65d143816f421b9b6b2` |
| `2026-10-04_23-02` | 1,623 | `7fe25cc1b9aaa135044fa64f51c7207d9e314f2c44abe4f96c552a1c8eedc56a` |
| `2026-10-05_00-54` | 1,623 | `7fe25cc1b9aaa135044fa64f51c7207d9e314f2c44abe4f96c552a1c8eedc56a` |
| `2026-10-05_21-33` | 1,623 | `7fe25cc1b9aaa135044fa64f51c7207d9e314f2c44abe4f96c552a1c8eedc56a` |
| `2026-10-09_00-46` | 1,623 | `7fe25cc1b9aaa135044fa64f51c7207d9e314f2c44abe4f96c552a1c8eedc56a` |
| `2026-10-09_21-04` | 1,623 | `7fe25cc1b9aaa135044fa64f51c7207d9e314f2c44abe4f96c552a1c8eedc56a` |

### 2.11 Fig 4 (p13): every marker and label

**The 30 markers** (15 per package: national and four fuels, three rebate cases). Every
draft rate and every draft home count below rounds to the text printed in the figure.
A draft marker's homes were rate x the homes of every row of the group in the table. A
current marker's homes are adopter rdu x weight.

| Package | Group | Rebate case | Draft rate (adopters of rdu with a result) | Draft homes as printed (rate x homes of every row of the group) | Draft adopter rdu x weight | Current rate (adopters of sample rdu) | Current homes (adopter rdu x weight) | Suggested marker label |
|---|---|---|---|---|---|---|---|---|
| MP3 | National | No rebate | 27.1760% (70,715 of 260,211) | 21.815M | 17.122M | 19.3472% (42,797 of 221,205) | 10.362M | 19% / 10.4M |
| MP3 | National | Dec 2024 | 61.2169% (159,293 of 260,211) | 49.141M | 38.570M | 46.6409% (103,172 of 221,205) | 24.981M | 47% / 25.0M |
| MP3 | National | June 2026 | 34.0908% (88,708 of 260,211) | 27.366M | 21.479M | 26.8778% (59,455 of 221,205) | 14.396M | 27% / 14.4M |
| MP3 | Electricity | No rebate | 43.7428% (32,238 of 73,699) | 11.205M | 7.806M | 45.1514% (28,798 of 63,781) | 6.973M | 45% / 7.0M |
| MP3 | Electricity | Dec 2024 | 68.1570% (50,231 of 73,699) | 17.459M | 12.162M | 71.2689% (45,456 of 63,781) | 11.006M | 71% / 11.0M |
| MP3 | Electricity | June 2026 | 68.1570% (50,231 of 73,699) | 17.459M | 12.162M | 71.2689% (45,456 of 63,781) | 11.006M | 71% / 11.0M |
| MP3 | Natural Gas | No rebate | 11.5412% (18,002 of 155,980) | 5.056M | 4.359M | 5.3327% (7,002 of 131,304) | 1.695M | 5% / 1.7M |
| MP3 | Natural Gas | Dec 2024 | 54.2685% (84,648 of 155,980) | 23.775M | 20.496M | 33.9571% (44,587 of 131,304) | 10.796M | 34% / 10.8M |
| MP3 | Natural Gas | June 2026 | 11.5412% (18,002 of 155,980) | 5.056M | 4.359M | 5.3327% (7,002 of 131,304) | 1.695M | 5% / 1.7M |
| MP3 | Propane | No rebate | 73.3950% (10,232 of 13,941) | 3.105M | 2.477M | 40.6401% (4,978 of 12,249) | 1.205M | 41% / 1.2M |
| MP3 | Propane | Dec 2024 | 88.4513% (12,331 of 13,941) | 3.742M | 2.986M | 68.6097% (8,404 of 12,249) | 2.035M | 69% / 2.0M |
| MP3 | Propane | June 2026 | 73.3950% (10,232 of 13,941) | 3.105M | 2.477M | 40.6401% (4,978 of 12,249) | 1.205M | 41% / 1.2M |
| MP3 | Fuel Oil | No rebate | 61.7383% (10,243 of 16,591) | 2.564M | 2.480M | 14.5555% (2,019 of 13,871) | 0.489M | 15% / 0.5M |
| MP3 | Fuel Oil | Dec 2024 | 72.8286% (12,083 of 16,591) | 3.025M | 2.926M | 34.0639% (4,725 of 13,871) | 1.144M | 34% / 1.1M |
| MP3 | Fuel Oil | June 2026 | 61.7383% (10,243 of 16,591) | 2.564M | 2.480M | 14.5555% (2,019 of 13,871) | 0.489M | 15% / 0.5M |
| MP4 | National | No rebate | 18.0984% (47,094 of 260,211) | 14.528M | 11.403M | 19.4675% (43,063 of 221,205) | 10.427M | 19% / 10.4M |
| MP4 | National | Dec 2024 | 45.1168% (117,399 of 260,211) | 36.217M | 28.426M | 49.3660% (109,200 of 221,205) | 26.441M | 49% / 26.4M |
| MP4 | National | June 2026 | 26.3936% (68,679 of 260,211) | 21.187M | 16.629M | 28.8949% (63,917 of 221,205) | 15.476M | 29% / 15.5M |
| MP4 | Electricity | No rebate | 34.2243% (25,223 of 73,699) | 8.767M | 6.107M | 39.7642% (25,362 of 63,781) | 6.141M | 40% / 6.1M |
| MP4 | Electricity | Dec 2024 | 63.5124% (46,808 of 73,699) | 16.269M | 11.334M | 72.4605% (46,216 of 63,781) | 11.190M | 72% / 11.2M |
| MP4 | Electricity | June 2026 | 63.5124% (46,808 of 73,699) | 16.269M | 11.334M | 72.4605% (46,216 of 63,781) | 11.190M | 72% / 11.2M |
| MP4 | Natural Gas | No rebate | 4.2980% (6,704 of 155,980) | 1.883M | 1.623M | 6.3197% (8,298 of 131,304) | 2.009M | 6% / 2.0M |
| MP4 | Natural Gas | Dec 2024 | 30.3827% (47,391 of 155,980) | 13.311M | 11.475M | 34.1322% (44,817 of 131,304) | 10.852M | 34% / 10.9M |
| MP4 | Natural Gas | June 2026 | 4.2980% (6,704 of 155,980) | 1.883M | 1.623M | 6.3197% (8,298 of 131,304) | 2.009M | 6% / 2.0M |
| MP4 | Propane | No rebate | 51.3163% (7,154 of 13,941) | 2.171M | 1.732M | 41.4564% (5,078 of 12,249) | 1.230M | 41% / 1.2M |
| MP4 | Propane | Dec 2024 | 80.9841% (11,290 of 13,941) | 3.426M | 2.734M | 77.0512% (9,438 of 12,249) | 2.285M | 77% / 2.3M |
| MP4 | Propane | June 2026 | 51.3163% (7,154 of 13,941) | 2.171M | 1.732M | 41.4564% (5,078 of 12,249) | 1.230M | 41% / 1.2M |
| MP4 | Fuel Oil | No rebate | 48.2973% (8,013 of 16,591) | 2.006M | 1.940M | 31.1802% (4,325 of 13,871) | 1.047M | 31% / 1.0M |
| MP4 | Fuel Oil | Dec 2024 | 71.7859% (11,910 of 16,591) | 2.981M | 2.884M | 62.9299% (8,729 of 13,871) | 2.114M | 63% / 2.1M |
| MP4 | Fuel Oil | June 2026 | 48.2973% (8,013 of 16,591) | 2.006M | 1.940M | 31.1802% (4,325 of 13,871) | 1.047M | 31% / 1.0M |

**Y-axis labels.** The draft printed "National -- Overall, 80.1 M Homes (100%)",
"Electricity 25.6/80.1 M Homes (32.0% Total)", "Natural Gas 43.8/80.1 M Homes (54.7%
Total)", "Fuel Oil 4.2/80.1 M Homes (5.2% Total)" and "Propane 4.2/80.1 M Homes (5.3%
Total)". Where 80.1 comes from: the draft-era notebook passed the dot plot a dict of
homes by `base_heating_fuel` (`groupby('base_heating_fuel')['weight'].sum()`), and
`plot_adoption_panel` printed `sum(fuel_counts_millions.values())` as the national total
(`visuals_adoption_dotplot.py` at commit `2f54ed2`, and
`tare_model_main_v2_3_EXPORT_26Aug2026.py` in the same commit). That sum is 330,634 rows x
242.13 = 80,056,410 homes: it counts the 9,277 "Other Fuel" rows and skips the 897 rows
with no fuel label.

| Group | Draft: rows of the 331,531-row table | Draft: homes (rows x 242.13) | Draft: share of the label total (80,056,410 homes) | Current: sample rdu | Current: homes | Current: share of the sample | Suggested y-axis label |
|---|---|---|---|---|---|---|---|
| National | 331,531 | 80.2736M | 100.271% | 221,205 | 53.5606M | 100.0000% | 53.6 M Homes (100%) |
| Electricity | 105,792 | 25.6154M | 31.997% | 63,781 | 15.4434M | 28.8334% | 15.4/53.6 M Homes (28.8% Total) |
| Natural Gas | 180,939 | 43.8108M | 54.725% | 131,304 | 31.7928M | 59.3585% | 31.8/53.6 M Homes (59.4% Total) |
| Propane | 17,474 | 4.2310M | 5.285% | 12,249 | 2.9659M | 5.5374% | 3.0/53.6 M Homes (5.5% Total) |
| Fuel Oil | 17,152 | 4.1530M | 5.188% | 13,871 | 3.3586M | 6.2707% | 3.4/53.6 M Homes (6.3% Total) |

### 2.12 Crosswalk section 3 (narrative changes) and section 6, item 2

| ID | Crosswalk statement | From the data | Status |
|---|---|---|---|
| N-1 | MP3 19.35% against MP4 19.47% with no rebate; MP4 higher in all three cases (46.64 against 49.37; 26.88 against 28.89) | 19.3472 / 19.4675; 46.6409 / 49.3660; 26.8778 / 28.8949 [C] | matches crosswalk |
| N-2 | Grid impacts still differ: +35% against +0.2% demand; a 6.9 times against 5.7 times Allegheny peak | +34.9703% / +0.1607% [RS]; 6.8600 / 5.6824 times [TS] | matches crosswalk |
| N-3 | MP3 by fuel: electricity 45%, propane 41%, fuel oil 15%, natural gas 5%. MP4: propane 41%, electricity 40%, fuel oil 31%, natural gas 6% | 45.1514, 40.6401, 14.5555, 5.3327; 41.4564, 39.7642, 31.1802, 6.3197 [C] | matches crosswalk |
| N-4 | On the same 221,205 rdu, mean discounted heating savings for MP3 fell by $1,913 (electricity), $7,544 (natural gas), $9,617 (propane), $15,200 (fuel oil) | -$1,913.23, -$7,544.11, -$9,617.49, -$15,199.53 [D][C], `ref2025_mp3_heating_discounted_lifetime_savings_fixed_base` | matches crosswalk |
| N-5 | For gas homes the minimum-efficiency system is now below the high-efficiency one (5% against 6%); the draft has 12% against 4% | 5.3327 / 6.3197; draft 11.5412 / 4.2980 | matches crosswalk |
| N-6 | Economic adopters no longer move the Allegheny peak: MP3 45 adopters, 713.86 MW @ 4433 (-0.65); MP4 67 adopters, 715.26 MW @ 4433 (+0.74) | 713.8625 MW (-0.6521) and 715.2579 MW (+0.7433), adopters of run `2026-10-05_21-33` [TS]. Current run, grid impact run of 11 Oct 2026: 713.8625 MW (-0.6521) and, with 68 adopters, 715.1649 MW (+0.6504) | matches crosswalk; the MP4 value is now 715.16 MW (section 1.2, item 5) |
| N-7 | Fewer adopters: "MP3 256 --> 45; MP4 125 --> 68" | 256 and 125 of 1,356 rdu (draft); 45 and 68 of 1,146 rdu (current); 251 and 118 for the draft run on the same 1,146 rdu | the counts are right; the denominators differ (section 1.2, item 4) |
| N-8 | The June 2026 loss is 10.6M (MP3) and 11.0M (MP4) homes, against 22M and 15M in the draft | 10,585,241 and 10,964,419 homes; draft printed 21.78M and 15.03M | matches crosswalk |
| N-9 | Section 6, item 2: homes with an electric furnace are 25,305 of the 42,797 rdu that adopt MP3 with no rebate, and 20,632 of the 43,063 that adopt MP4 | 25,305 of 42,797 (59.13%) and 20,632 of 43,063 (47.91%) [C]; [XH] equal. 47,155 sample rdu have an electric furnace and 265 an electric boiler | matches crosswalk; the MP4 figure is now verified |
| N-10 | Allegheny adopters after boiler pricing: 45 (MP3) and 68 (MP4) | 45 and 68 of 1,146 rdu [C]; [XA] equal | matches crosswalk |

**Base case adopters with no rebate, by old heating system.**

| Run | Package | Old heating system | rdu | Adopters (rdu) | Adoption | Share of all adopters |
|---|---|---|---|---|---|---|
| Draft | MP3 | Electricity Baseboard | 23,383 | 4,409 | 18.8556% | 6.2349% |
| Draft | MP3 | Electricity Electric Boiler | 382 | 67 | 17.5393% | 0.0947% |
| Draft | MP3 | Electricity Electric Furnace | 49,934 | 27,762 | 55.5974% | 39.2590% |
| Draft | MP3 | Fuel Oil Fuel Boiler | 5,214 | 937 | 17.9708% | 1.3250% |
| Draft | MP3 | Fuel Oil Fuel Furnace | 11,377 | 9,306 | 81.7966% | 13.1599% |
| Draft | MP3 | Natural Gas Fuel Boiler | 22,747 | 730 | 3.2092% | 1.0323% |
| Draft | MP3 | Natural Gas Fuel Furnace | 133,233 | 17,272 | 12.9638% | 24.4248% |
| Draft | MP3 | Propane Fuel Boiler | 889 | 178 | 20.0225% | 0.2517% |
| Draft | MP3 | Propane Fuel Furnace | 13,052 | 10,054 | 77.0303% | 14.2176% |
| Draft | MP4 | Electricity Baseboard | 23,383 | 5,741 | 24.5520% | 12.1905% |
| Draft | MP4 | Electricity Electric Boiler | 382 | 110 | 28.7958% | 0.2336% |
| Draft | MP4 | Electricity Electric Furnace | 49,934 | 19,372 | 38.7952% | 41.1348% |
| Draft | MP4 | Fuel Oil Fuel Boiler | 5,214 | 739 | 14.1734% | 1.5692% |
| Draft | MP4 | Fuel Oil Fuel Furnace | 11,377 | 7,274 | 63.9360% | 15.4457% |
| Draft | MP4 | Natural Gas Fuel Boiler | 22,747 | 314 | 1.3804% | 0.6668% |
| Draft | MP4 | Natural Gas Fuel Furnace | 133,233 | 6,390 | 4.7961% | 13.5686% |
| Draft | MP4 | Propane Fuel Boiler | 889 | 153 | 17.2103% | 0.3249% |
| Draft | MP4 | Propane Fuel Furnace | 13,052 | 7,001 | 53.6393% | 14.8660% |
| Current | MP3 | Electricity Baseboard | 16,361 | 3,438 | 21.0134% | 8.0333% |
| Current | MP3 | Electricity Electric Boiler | 265 | 55 | 20.7547% | 0.1285% |
| Current | MP3 | Electricity Electric Furnace | 47,155 | 25,305 | 53.6635% | 59.1280% |
| Current | MP3 | Fuel Oil Fuel Boiler | 3,983 | 323 | 8.1095% | 0.7547% |
| Current | MP3 | Fuel Oil Fuel Furnace | 9,888 | 1,696 | 17.1521% | 3.9629% |
| Current | MP3 | Natural Gas Fuel Boiler | 16,335 | 1,099 | 6.7279% | 2.5679% |
| Current | MP3 | Natural Gas Fuel Furnace | 114,969 | 5,903 | 5.1344% | 13.7930% |
| Current | MP3 | Propane Fuel Boiler | 655 | 144 | 21.9847% | 0.3365% |
| Current | MP3 | Propane Fuel Furnace | 11,594 | 4,834 | 41.6940% | 11.2952% |
| Current | MP4 | Electricity Baseboard | 16,361 | 4,648 | 28.4090% | 10.7935% |
| Current | MP4 | Electricity Electric Boiler | 265 | 82 | 30.9434% | 0.1904% |
| Current | MP4 | Electricity Electric Furnace | 47,155 | 20,632 | 43.7536% | 47.9112% |
| Current | MP4 | Fuel Oil Fuel Boiler | 3,983 | 646 | 16.2189% | 1.5001% |
| Current | MP4 | Fuel Oil Fuel Furnace | 9,888 | 3,679 | 37.2067% | 8.5433% |
| Current | MP4 | Natural Gas Fuel Boiler | 16,335 | 924 | 5.6566% | 2.1457% |
| Current | MP4 | Natural Gas Fuel Furnace | 114,969 | 7,374 | 6.4139% | 17.1237% |
| Current | MP4 | Propane Fuel Boiler | 655 | 219 | 33.4351% | 0.5086% |
| Current | MP4 | Propane Fuel Furnace | 11,594 | 4,859 | 41.9096% | 11.2835% |

**Allegheny County, current run, all nine cases** (1,146 sample rdu = 277,482 homes).

| Package | Case | Adopters (of 1,146 rdu) | Rate |
|---|---|---|---|
| MP3 | heatingSavings_coolingLCC_sub | 55 | 4.7993% |
| MP3 | heatingSavings_coolingLCC_unsub | 36 | 3.1414% |
| MP3 | heatingSavings_coolingLCC_sub_june2026 | 43 | 3.7522% |
| MP3 | heatingLCC_coolingSavings_sub | 51 | 4.4503% |
| MP3 | heatingLCC_coolingSavings_unsub | 28 | 2.4433% |
| MP3 | heatingLCC_coolingSavings_sub_june2026 | 40 | 3.4904% |
| MP3 | heatingLCC_coolingLCC_sub | 118 | 10.2967% |
| MP3 | heatingLCC_coolingLCC_unsub | 45 | 3.9267% |
| MP3 | heatingLCC_coolingLCC_sub_june2026 | 50 | 4.3630% |
| MP4 | heatingSavings_coolingLCC_sub | 84 | 7.3298% |
| MP4 | heatingSavings_coolingLCC_unsub | 49 | 4.2757% |
| MP4 | heatingSavings_coolingLCC_sub_june2026 | 67 | 5.8464% |
| MP4 | heatingLCC_coolingSavings_sub | 82 | 7.1553% |
| MP4 | heatingLCC_coolingSavings_unsub | 49 | 4.2757% |
| MP4 | heatingLCC_coolingSavings_sub_june2026 | 65 | 5.6719% |
| MP4 | heatingLCC_coolingLCC_sub | 164 | 14.3106% |
| MP4 | heatingLCC_coolingLCC_unsub | 68 | 5.9337% |
| MP4 | heatingLCC_coolingLCC_sub_june2026 | 77 | 6.7190% |

### 2.13 The notebook run of 11 Oct 2026 (run `2026-10-09_00-46` loaded again, grid impact on)

Jordan loaded the run in `tare_model_main_v3_0.ipynb` (answer "N", load a previous run),
with grid impact on. No model code ran: the run's 10 model files were read, the analysis
cells ran, and cell 23 wrote the 12 export files again under the same stamp. Checked on
11 Oct 2026 from the outputs saved in the notebook (24 code cells with output, no error)
and from the PDF export, `tare_model_main_v3_0_EXPORT_11Oct2026_run2026-10-09_00-46.pdf`.

**Peaks, Allegheny County, 1,146 sample rdu.** The notebook prints two decimals.

| Scenario | Notebook, 11 Oct 2026 | This file, from the cached hourly tables | Equal |
|---|---|---|---|
| Old systems | 714.51 MW @ hour 4433 | 714.5146 MW @ 4433 | yes |
| MP3, 100% adoption (1,146 rdu) | 4901.60 MW @ hour 152; +4187.09 MW | 4,901.6039 MW @ 152; +4,187.0893 MW | yes |
| MP3, economic adopters (45 rdu) | 713.86 MW @ hour 4433; -0.65 MW | 713.8625 MW @ 4433; -0.6521 MW | yes |
| MP4, 100% adoption (1,146 rdu) | 4060.19 MW @ hour 152; +3345.68 MW | 4,060.1916 MW @ 152; +3,345.6770 MW | yes |
| MP4, economic adopters (68 rdu) | 715.16 MW @ hour 4433; +0.65 MW | 715.1649 MW @ 4433; +0.6504 MW | yes |

The query cache file was not changed by the run (same size and date as on 8 Oct 2026), so
the notebook read the hourly tables this file used.

**Everything else the notebook printed.**

- 193 numbers were compared with the values behind this file, each at the precision the
  notebook prints: the study sample, county adoption, the three cost offsets and the
  three rebate cases, the demand breakdown, the three county summaries, heat-pump heating
  energy, adopters, the Allegheny adopters by fuel, and the peaks. All 193 are equal.
- Every result line of cells 6 to 28 is word for word in the log of the original run
  (`~/tare_port/logs/s8_national_2022.log`, 9 Oct 2026, grid impact off). The lines that
  differ are timestamps, file paths (capital letters only), the question about loading a
  run, and the grid impact cells.
- All 234 output lines are in the PDF's text (44 pages). Five are broken by a line wrap
  or a page number.
- The rebate dot plot it saved
  (`figures/econ_adoption_dotplot_rebate_policy_scenario_heatingLCC_coolingLCC.png`)
  shows the 30 marker labels and the y-axis labels of section 2.11.
- Allegheny adopters by fuel (cell 32): MP3, 45 = 39 electricity, 4 fuel oil, 0 natural
  gas, 2 propane. MP4, 68 = 52 electricity, 10 fuel oil, 2 natural gas, 4 propane.

**The files it wrote again.** The 12 export files (written 11 Oct, 12:24 to 12:27) have
the SHA-256 values recorded on 10 Oct (section 6) and equal the copies of run
`2026-10-09_21-04`. The run's other 10 files and the three `source_data` files were not
written. The notebook's source is the same as the last commit in all 37 cells; only
outputs were added.

**Not checked.** The county maps and the two peak figures hold no printed number. They
were looked at and agree with the numbers above, no more than that. Cell 11 also wrote
the cost validation workbook again, outside the repo
(`Equipment_Installed_TARE_v4MID_current_2026-09-16.xlsx`); no earlier checksum of it
exists, so whether it changed is not known.

---

## 3. Unmapped rows U1 to U16

Measured facts only. Nothing here says why.

| ID | Page, paragraph | Draft text | Draft exact (source) | Current exact (source) | Suggested text | Export value | Status | Driver |
|---|---|---|---|---|---|---|---|---|
| U1 | p2, abstract | "and shift peak demand from summer to winter" | Allegheny, 1,610 rdu: the peak moves from hour 4433 (4 July) to hour 152 or 153 (7 January) in all four scenarios [TS] | 100% adoption: moves to hour 152 for MP3 and MP4. Economic adopters: stays at hour 4433 for both (grid impact run of 11 Oct 2026 on the current run, 45 and 68 adopters) [TS] | "and, if every home adopts, shift peak demand from summer to winter" | not in exports | holds at 100% adoption only | energy fix (who adopts), then sample rule |
| U2 | p3 par 2, last sentence before the page break (the session prompt lists p4) | "will remove access to energy savings measures for millions of these delivered fuel homes" | Fuel oil and propane rdu eligible under December 2024 and not under June 2026: 29,808 rdu = 7.22M homes (MP3), 30,076 rdu = 7.28M (MP4). Adopters under December 2024 who do not adopt under June 2026: 3,939 rdu = 0.95M (MP3), 8,033 rdu = 1.95M (MP4) [D] | Eligible then not: 25,039 rdu = 6,062,718 homes (MP3), 25,808 rdu = 6,248,917 (MP4). Adopters lost: 6,132 rdu = 1,484,747 homes (MP3), 8,764 rdu = 2,122,036 (MP4). No fuel oil or propane rdu is eligible under June 2026 [C] | "millions" holds for eligibility (6.1 to 6.2 million); for adopters it is 1.5 to 2.1 million | not in exports (December 2024 label and flag) | holds | sample rule (eligibility); energy fix and boiler pricing (adopters) |
| U3 | p8 par 1 | "the most favorable counties are those where households currently buy delivered heating fuels"; "Electric-resistance dominated counties ... follow at intermediate levels"; "Counties dominated by natural gas are the least favorable" | MP3, median county rate by the county's most common fuel among rdu with a result: fuel oil 66.2162%, propane 58.3333%, electricity 44.5299%, natural gas 30.9278%. MP4: 52.5180%, 40.9091%, 25.9259%, 20.0000% [D] | MP3: electricity 42.8571%, propane 18.7500%, natural gas 14.2857%, fuel oil 9.0537%. MP4: electricity 33.3333%, fuel oil 22.4747%, propane 21.4286%, natural gas 16.5391% [C] | Counties where electricity is the most common fuel are now the highest for both packages; for MP3 the fuel oil counties are the lowest | [XC] holds each county's rate; the county's fuel must come from [XH] | does not hold on current data | energy fix |
| U4 | p10 par 2 | "70% of the 3,100 modeled counties" | 3,098 counties in the demand table [RS] | 3,079 [RS] | "3,079" | [XC] 3,079 rows | holds to the nearest hundred | sample rule |
| U5 | p10 par 2 | "County-level increases can exceed 600% where a large fraction of homes currently heat with fossil fuels, especially in colder regions where natural gas dominates" | Above +600% (MP3): 1 county, G4600210 (Campbell County, South Dakota), 2 rdu, both natural gas, zone 6A. Above +400% (MP4): the same county only [RS] | The same single county in both cases. MP3: 3 counties above +500% (2 with 3 rdu or fewer), 15 above +400% (9 with 3 rdu or fewer) [RS] | "can exceed 400%" rests on 15 counties (MP3); "exceed 600%" rests on one county of 2 rdu | [XC] 1 county above 600 (MP3) and 1 above 400 (MP4) | holds for one county of 2 rdu | none |
| U6 | p14 par 1 | "a typical single-stage system needs to supply backup electric resistance heating (strip heat) more often to meet the heating load" | not in the draft run: it did not count backup heat | Share of heat-pump heating energy that is electric backup, base year: MP3 42.9345%, MP4 34.2161% (national); MP3 52.6013%, MP4 41.4241% (Allegheny). rdu with backup above zero: 194,258 (MP3) and 179,810 (MP4) of 221,205; Allegheny 1,146 and 1,139 of 1,146 [C] | holds as a comparison: MP3 uses more backup energy and more rdu use it | [XH] carries the same columns | holds | none |
| U7 | p17 par 3 | "We analyze nearly 3,100 counties" | 3,098 | 3,079 [F] | "3,079" | [XC] 3,079 rows | location only | sample rule |
| U8 | p18 par 2; p24 par 2 | "The remaining subsidized market for existing electric resistance systems is concentrated in the South"; "households with existing electric resistance systems (those eligible for the rebates) are concentrated in the South" | not measured | South's share of electric-heated sample rdu: 71.0462% (45,314 of 63,781). Of June 2026 rebate-eligible rdu: 70.0301% (MP3), 71.4797% (MP4); every eligible rdu is electric-heated. Of electric-heated June 2026 adopters: 72.9761% (MP3), 69.4565% (MP4) [C] | "about 70% are in the South" | [XH] has the region, the fuel and the June 2026 label | holds | none |
| U9 | p18 par 2 | "the minimum-efficiency, single-stage heat pump ... is the most likely rebate-supported choice" | not measured | June 2026 adoption of electric-heated rdu: MP3 71.2689%, MP4 72.4605%. South: 73.2047% against 70.8390%. Of 63,781 rdu: 41,230 adopt both, 4,226 MP3 only, 4,986 MP4 only, 13,339 neither. MP3 has the higher NPV on 32,422 rdu, MP4 on 31,359 [C] | no measured basis nationally; MP3 is ahead in the South and West only | rebuilt June 2026 flags from [XH] | does not hold nationally | none |
| U10 | p20 par 1; p21 par 1 | "multiplying the 2018 energy consumption ... by projection factors derived from ... Heating Degree Day (HDD) projections"; "After projecting future space heating demand using HDD projections" | Commit `2f54ed2`: `apply_degree_day_adjustment` multiplies heating by the HDD factor and cooling by the CDD factor (`utils/degree_day_consumption_utils.py`) | Commit `6a12e05`: the same, also in `get_degree_day_adjusted_consumption_by_fuel`. The factor file has 10 HDD and 10 CDD rows; 2039 CDD factors run from 1.048 to 1.357 | add cooling degree days: "HDD and CDD projections" | not in exports | the text leaves out the cooling factors | none |
| U11 | p24 par 1 | "legacy efficiencies like 8-10 SEER central ACs and 60-76% AFUE furnaces" | not measured | Central AC SEER (177,641 rdu): min 8, 5th percentile 10, median 13, 95th 15, max 15; SEER 8 or 10 on 38,863 rdu (21.88%). Fossil furnace AFUE (136,451 rdu): min 60%, 5th 76%, median 80%, 95th 92.5%, max 92.5%; 60% or 76% on 12,789 rdu (9.37%) [C] | holds as examples of the lowest values | [XH] `base_cooling_efficiency`, `base_heating_efficiency` | holds | none |
| U12 | p24 par 1 | "Applying these changes brought the ResStock equipment within the NLR regression bounds" | not measured | After the floors, efficiency is below a row's range only for room AC (2,768 rdu below 9.4). It is above for 192 propane boilers (0.90 against 0.87) and every MP4 non-ducted heat pump (24,853 rdu, 29.3 against 23). Size is outside the range for the rdu in the table below [C] | "brought the efficiency of central ACs, furnaces and boilers within the bounds"; sizes are priced on the row's line beyond its range | not in exports | holds for efficiency, not for size; equals CLAUDE.md Limitations 9 and 17 | none |
| U13 | p23 par 1 | "REMDB v4.0.0, which represents 2023 technology and practice in USD2023" | not measured | `data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv`: 32 rows, `output_units` "2023$" or "2023$/sqft". Beside it: `REMDB_2024_12_23.xlsx`. The Stage 1 plan records that the table equals REMDB 2024 in all 32 rows. Costs are inflated with `cpi_ratio_2025_2023` | "REMDB 2024, in 2023 dollars" unless NLR's version number is confirmed | not in exports | the dollar year holds; "v4.0.0" is not in the repo | none |
| U14 | p25 par 1 | "A 100% adoption bounding scenario was modeled by assigning all filtered buildings to their post-retrofit consumption ... all single-family heating systems" | Demand: 331,526 rdu (every occupied single-family rdu the package applies to). Peaks: 1,610 Allegheny rdu, the same rule [RS][TS] | 221,205 rdu; 1,146 in Allegheny [C] | "all homes in the study sample" | [XH] 221,205 rows | the draft's set was wider than its filtered sample | sample rule |
| U15 | p9, Fig 2 caption (and p11) | "Gray counties indicate a county identifier mismatch, insufficient data, or too few modeled homes" | 3,108 counties on the map (lower 48 states and DC, `cb_2021_us_county_500k`). 10 grey: 1 with no ResStock rdu at all, 9 with no occupied single-family rdu. 0 identifier mismatches. `MIN_HOME_COUNT` was 1, so none is grey for too few homes [D] | 29 grey: the same 10, and 19 with occupied single-family rdu but none in the study sample (35 rdu). 0 identifier mismatches [C] | "Gray counties have no modeled home in the study sample" | [XC] 3,079 rows | the listed reasons do not match | sample rule |
| U16 | p20 par 1 | "~550,000 dwelling unit sample (1 unit for approximately every 240 homes that exist in the U.S.)" | 548,916 rdu; 242.13101272727272 homes per rdu [RS] | the same | no change | not in exports | location only; holds | none |

**U2, by fuel.**

| Run | Package | Baseline fuel | rdu | Eligible, Dec 2024 (rdu) | Eligible, June 2026 (rdu) | Eligible under Dec 2024, not June 2026 | Adopt under Dec 2024, not June 2026 |
|---|---|---|---|---|---|---|---|
| Draft | MP3 | Fuel Oil | 16,591 | 16,520 | 0 | 16,520 = 3,999,988 homes | 1,840 = 445,519 homes |
| Draft | MP3 | Propane | 13,941 | 13,288 | 0 | 13,288 = 3,217,423 homes | 2,099 = 508,231 homes |
| Draft | MP3 | Fuel Oil + Propane | 30,532 | 29,808 | 0 | 29,808 = 7,217,411 homes | 3,939 = 953,750 homes |
| Draft | MP3 | Natural Gas | 155,980 | 144,898 | 0 | 144,898 = 35,084,153 homes | 66,646 = 16,136,996 homes |
| Draft | MP4 | Fuel Oil | 16,591 | 16,553 | 0 | 16,553 = 4,007,978 homes | 3,897 = 943,581 homes |
| Draft | MP4 | Propane | 13,941 | 13,523 | 0 | 13,523 = 3,274,324 homes | 4,136 = 1,001,450 homes |
| Draft | MP4 | Fuel Oil + Propane | 30,532 | 30,076 | 0 | 30,076 = 7,282,302 homes | 8,033 = 1,945,030 homes |
| Draft | MP4 | Natural Gas | 155,980 | 148,605 | 0 | 148,605 = 35,981,729 homes | 40,687 = 9,851,543 homes |
| Current | MP3 | Fuel Oil | 13,871 | 13,667 | 0 | 13,667 = 3,309,205 homes | 2,706 = 655,207 homes |
| Current | MP3 | Propane | 12,249 | 11,372 | 0 | 11,372 = 2,753,514 homes | 3,426 = 829,541 homes |
| Current | MP3 | Fuel Oil + Propane | 26,120 | 25,039 | 0 | 25,039 = 6,062,718 homes | 6,132 = 1,484,747 homes |
| Current | MP3 | Natural Gas | 131,304 | 120,078 | 0 | 120,078 = 29,074,608 homes | 37,585 = 9,100,494 homes |
| Current | MP4 | Fuel Oil | 13,871 | 13,831 | 0 | 13,831 = 3,348,914 homes | 4,404 = 1,066,345 homes |
| Current | MP4 | Propane | 12,249 | 11,977 | 0 | 11,977 = 2,900,003 homes | 4,360 = 1,055,691 homes |
| Current | MP4 | Fuel Oil + Propane | 26,120 | 25,808 | 0 | 25,808 = 6,248,917 homes | 8,764 = 2,122,036 homes |
| Current | MP4 | Natural Gas | 131,304 | 127,037 | 0 | 127,037 = 30,759,597 homes | 36,519 = 8,842,382 homes |

**U3, county adoption with no rebate by the county's fuel.** The county's fuel is set from
the rdu the results cover (sample-based, not the American Community Survey). Two rules are
shown: the most common fuel (a tie is its own group), and a fuel held by more than half of
the rdu.

| Run | Package | How the county's fuel is set | County's fuel | Counties | rdu in them | Median county adoption | Mean county adoption | Adoption of all rdu in them |
|---|---|---|---|---|---|---|---|---|
| Draft | MP3 | most common fuel | Electricity | 1022 | 53,236 | 44.5299% | 44.4380% | 33.1430% |
| Draft | MP3 | most common fuel | Fuel Oil | 97 | 12,111 | 66.2162% | 62.4175% | 55.2886% |
| Draft | MP3 | most common fuel | Natural Gas | 1656 | 190,320 | 30.9278% | 32.0587% | 23.2482% |
| Draft | MP3 | most common fuel | Propane | 95 | 1,660 | 58.3333% | 59.8932% | 51.1446% |
| Draft | MP3 | most common fuel | tie | 220 | 2,884 | 50.0000% | 49.4632% | 44.3828% |
| Draft | MP3 | majority (more than half) | Electricity | 814 | 45,888 | 42.8571% | 42.9020% | 30.4786% |
| Draft | MP3 | majority (more than half) | Fuel Oil | 74 | 9,143 | 68.4698% | 65.2608% | 57.8038% |
| Draft | MP3 | majority (more than half) | Natural Gas | 1379 | 177,391 | 27.8557% | 29.5805% | 21.8985% |
| Draft | MP3 | majority (more than half) | Propane | 41 | 287 | 66.6667% | 65.2897% | 56.0976% |
| Draft | MP3 | majority (more than half) | no majority | 782 | 27,502 | 50.0000% | 48.4798% | 45.2222% |
| Draft | MP4 | most common fuel | Electricity | 1022 | 53,236 | 25.9259% | 28.0086% | 21.0365% |
| Draft | MP4 | most common fuel | Fuel Oil | 97 | 12,111 | 52.5180% | 52.7229% | 45.6775% |
| Draft | MP4 | most common fuel | Natural Gas | 1656 | 190,320 | 20.0000% | 20.6836% | 15.1708% |
| Draft | MP4 | most common fuel | Propane | 95 | 1,660 | 40.9091% | 43.6681% | 38.8554% |
| Draft | MP4 | most common fuel | tie | 220 | 2,884 | 33.3333% | 33.2080% | 29.2996% |
| Draft | MP4 | majority (more than half) | Electricity | 814 | 45,888 | 25.0000% | 26.7123% | 19.0900% |
| Draft | MP4 | majority (more than half) | Fuel Oil | 74 | 9,143 | 57.3982% | 54.5758% | 47.7196% |
| Draft | MP4 | majority (more than half) | Natural Gas | 1379 | 177,391 | 17.5644% | 18.5692% | 13.9618% |
| Draft | MP4 | majority (more than half) | Propane | 41 | 287 | 42.8571% | 46.9316% | 39.7213% |
| Draft | MP4 | majority (more than half) | no majority | 782 | 27,502 | 33.3333% | 33.4164% | 33.0521% |
| Current | MP3 | most common fuel | Electricity | 1004 | 48,925 | 42.8571% | 42.4291% | 32.3066% |
| Current | MP3 | most common fuel | Fuel Oil | 102 | 9,499 | 9.0537% | 15.1030% | 10.8538% |
| Current | MP3 | most common fuel | Natural Gas | 1618 | 158,789 | 14.2857% | 16.8604% | 15.6201% |
| Current | MP3 | most common fuel | Propane | 115 | 1,308 | 18.7500% | 30.1678% | 23.7003% |
| Current | MP3 | most common fuel | tie | 240 | 2,684 | 33.3333% | 31.9691% | 31.5574% |
| Current | MP3 | majority (more than half) | Electricity | 808 | 42,470 | 42.2282% | 41.9977% | 30.4780% |
| Current | MP3 | majority (more than half) | Fuel Oil | 76 | 7,760 | 7.4015% | 11.5543% | 8.5180% |
| Current | MP3 | majority (more than half) | Natural Gas | 1365 | 147,922 | 12.1212% | 14.6750% | 14.6510% |
| Current | MP3 | majority (more than half) | Propane | 68 | 512 | 22.8758% | 34.0603% | 26.3672% |
| Current | MP3 | majority (more than half) | no majority | 762 | 22,541 | 33.3333% | 33.3354% | 32.7625% |
| Current | MP4 | most common fuel | Electricity | 1004 | 48,925 | 33.3333% | 33.1525% | 26.8963% |
| Current | MP4 | most common fuel | Fuel Oil | 102 | 9,499 | 22.4747% | 23.2522% | 25.1921% |
| Current | MP4 | most common fuel | Natural Gas | 1618 | 158,789 | 16.5391% | 17.8021% | 16.6624% |
| Current | MP4 | most common fuel | Propane | 115 | 1,308 | 21.4286% | 29.9024% | 22.4771% |
| Current | MP4 | most common fuel | tie | 240 | 2,684 | 30.0000% | 29.2460% | 28.2787% |
| Current | MP4 | majority (more than half) | Electricity | 808 | 42,470 | 31.0728% | 32.3075% | 25.4085% |
| Current | MP4 | majority (more than half) | Fuel Oil | 76 | 7,760 | 17.7866% | 20.2501% | 23.9691% |
| Current | MP4 | majority (more than half) | Natural Gas | 1365 | 147,922 | 14.2857% | 16.0895% | 15.6042% |
| Current | MP4 | majority (more than half) | Propane | 68 | 512 | 17.9144% | 33.1169% | 23.4375% |
| Current | MP4 | majority (more than half) | no majority | 762 | 22,541 | 30.7692% | 30.2637% | 31.9862% |

**U5, counties above +600% (MP3) and +400% (MP4).**

| Homes counted | Package | County | rdu | Baseline (GWh) | Change (GWh) | Change | Fossil share of rdu | Natural gas share of rdu | Climate zone |
|---|---|---|---|---|---|---|---|---|---|
| Draft (331,526 rdu) | MP3 | G4600210 (SD) | 2 | 4.7865 | +29.7878 | +622.3300% | 100% | 100% | 6A (Cold) |
| Current (221,205 rdu) | MP3 | G4600210 (SD) | 2 | 4.7865 | +29.7878 | +622.3300% | 100% | 100% | 6A (Cold) |
| Draft (331,526 rdu) | MP4 | G4600210 (SD) | 2 | 4.7865 | +22.7339 | +474.9585% | 100% | 100% | 6A (Cold) |
| Current (221,205 rdu) | MP4 | G4600210 (SD) | 2 | 4.7865 | +22.7339 | +474.9585% | 100% | 100% | 6A (Cold) |

**U6, electric backup in heat-pump heating energy** (base-year values; in the model's
first year, 2025, the degree-day factor is 1).

| Package | Place | Sample rdu | Heat pump heating energy, all parts (kWh, unweighted sum) | Of which electric backup (kWh) | Backup share | Mean backup per rdu (kWh) | Mean heating energy per rdu (kWh) | rdu with backup above zero |
|---|---|---|---|---|---|---|---|---|
| MP3 | National | 221,205 | 1,911,224,337 | 820,574,232 | 42.9345% | 3,709.56 | 8,640.06 | 194,258 (87.8181%) |
| MP3 | Allegheny | 1,146 | 16,601,418 | 8,732,569 | 52.6013% | 7,620.04 | 14,486.40 | 1,146 (100.0000%) |
| MP4 | National | 221,205 | 1,146,374,917 | 392,244,966 | 34.2161% | 1,773.22 | 5,182.41 | 179,810 (81.2866%) |
| MP4 | Allegheny | 1,146 | 10,183,860 | 4,218,574 | 41.4241% | 3,681.13 | 8,886.44 | 1,139 (99.3892%) |

**U8 and U9, by census region** (current run).

| Package | Census region | Sample rdu | Electric-heated rdu (share of all electric-heated) | June 2026 rebate-eligible rdu (share of all eligible) | Eligible rdu not electric-heated | Electric-heated June 2026 adopters (share of all such adopters) | June 2026 adoption of electric-heated rdu | Electric-heated share of the region's rdu |
|---|---|---|---|---|---|---|---|---|
| MP3 | South | 91,078 | 45,314 (71.0462%) | 36,478 (70.0301%) | 0 | 33,172 (72.9761%) | 73.2047% | 49.7530% |
| MP3 | Midwest | 55,043 | 6,989 (10.9578%) | 6,255 (12.0083%) | 0 | 4,437 (9.7611%) | 63.4855% | 12.6973% |
| MP3 | West | 37,858 | 8,556 (13.4147%) | 6,703 (12.8684%) | 0 | 5,641 (12.4098%) | 65.9303% | 22.6002% |
| MP3 | Northeast | 37,226 | 2,922 (4.5813%) | 2,653 (5.0932%) | 0 | 2,206 (4.8530%) | 75.4962% | 7.8494% |
| MP3 | All | 221,205 | 63,781 (100.0000%) | 52,089 (100.0000%) | 0 | 45,456 (100.0000%) | 71.2689% | 28.8334% |
| MP4 | South | 91,078 | 45,314 (71.0462%) | 43,519 (71.4797%) | 0 | 32,100 (69.4565%) | 70.8390% | 49.7530% |
| MP4 | Midwest | 55,043 | 6,989 (10.9578%) | 6,788 (11.1493%) | 0 | 5,848 (12.6536%) | 83.6743% | 12.6973% |
| MP4 | West | 37,858 | 8,556 (13.4147%) | 7,690 (12.6308%) | 0 | 5,544 (11.9958%) | 64.7966% | 22.6002% |
| MP4 | Northeast | 37,226 | 2,922 (4.5813%) | 2,886 (4.7402%) | 0 | 2,724 (5.8941%) | 93.2238% | 7.8494% |
| MP4 | All | 221,205 | 63,781 (100.0000%) | 60,883 (100.0000%) | 0 | 46,216 (100.0000%) | 72.4605% | 28.8334% |

**U9, electric-heated rdu under June 2026 rules, MP3 against MP4.**

| Census region | Electric-heated rdu | MP3 adoption | MP4 adoption | Adopt both | MP3 only | MP4 only | Neither | rdu where MP3's NPV is higher | rdu where MP4's NPV is higher | Mean NPV, MP3 | Mean NPV, MP4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| South | 45,314 | 73.2047% | 70.8390% | 29,763 | 3,409 | 2,337 | 9,805 | 26,257 | 19,057 | $4,772.99 | $5,061.85 |
| Midwest | 6,989 | 63.4855% | 83.6743% | 4,345 | 92 | 1,503 | 1,049 | 1,492 | 5,497 | -$1,485.20 | $12,336.72 |
| West | 8,556 | 65.9303% | 64.7966% | 4,952 | 689 | 592 | 2,323 | 4,229 | 4,327 | $3,410.25 | $4,975.68 |
| Northeast | 2,922 | 75.4962% | 93.2238% | 2,170 | 36 | 554 | 162 | 444 | 2,478 | $10,371.42 | $20,769.96 |
| All | 63,781 | 71.2689% | 72.4605% | 41,230 | 4,226 | 4,986 | 13,339 | 32,422 | 31,359 | $4,160.90 | $6,567.09 |

**U12, rdu outside each cost row's range** (current run, study sample). The counts equal
CLAUDE.md Limitations 9 and 17. Two are not listed there: 1 electric baseboard credit
above 416,000 Btu/h, and non-ducted heat pumps below 1.5 tons (518 rdu for MP3, 729 for
MP4).

| Cost | Package | Cost table row | Sample rdu | Size and its range | Size below | Size above | Efficiency and its range | Efficiency below, as priced | Efficiency above, as priced | Efficiency below before the floor |
|---|---|---|---|---|---|---|---|---|---|---|
| credit for the old heating system | MP3 and MP4 | boiler_gas_non_condensing | 16,990 | Heating Capacity (BTU/hr): 27000 to 416000 | 2,162 | 0 | AFUE: 0.8 to 0.87 | 0 | 192 | 2,085 |
| credit for the old heating system | MP3 and MP4 | boiler_oil | 3,983 | Heating Capacity (BTU/hr): 27000 to 416000 | 174 | 0 | AFUE: 0.8 to 0.95 | 0 | 0 | 99 |
| credit for the old heating system | MP3 and MP4 | electric_baseboard_default | 63,781 | Capacity (BTU/Hr): 1194.2 to 416000 | 0 | 1 | none | 0 | 0 |  |
| credit for the old heating system | MP3 and MP4 | furnaces_gas_furnace | 136,451 | Heating Capacity (BTU/Hr): 30000 to 156250 | 16,859 | 3,015 | AFUE: 0.8 to 0.97 | 0 | 0 | 12,789 |
| credit for the old cooling system | MP3 and MP4 | air_conditioner_centrally_ducted | 177,641 | Cooling Capacity (Tons): 1.5 to 5 | 32,084 | 22,578 | SEER1: 13 to 24 | 0 | 0 | 38,863 |
| credit for the old cooling system | MP3 and MP4 | air_conditioner_room_ac_window_or_through_wall | 43,564 | Capacity (Tons): 0.416667 to 2.33333 | 15,545 | 1,780 | CEER: 9.4 to 15 | 2,768 | 0 | 2,768 |
| heat pump | MP3 | air_source_heat_pump_centrally_ducted | 196,352 | Cooling Capacity (Tons): 1.5 to 5 | 24,787 | 31,510 | SEER1: 13.8 to 24 | 0 | 0 |  |
| heat pump | MP3 | air_source_heat_pump_non_ducted_multi_zone | 24,853 | Cooling Capacity (Tons): 1.5 to 5 | 518 | 15,560 | SEER1: 16 to 23 | 0 | 0 |  |
| heat pump | MP4 | air_source_heat_pump_centrally_ducted | 196,352 | Cooling Capacity (Tons): 1.5 to 5 | 21,120 | 31,782 | SEER1: 13.8 to 24 | 0 | 0 |  |
| heat pump | MP4 | air_source_heat_pump_non_ducted_multi_zone | 24,853 | Cooling Capacity (Tons): 1.5 to 5 | 729 | 10,515 | SEER1: 16 to 23 | 0 | 24,853 |  |

**U15, counties on the map with no value.**

| Map | Counties with a value | Grey counties (of 3,108) | No ResStock rdu at all | rdu, but none occupied single-family | Occupied single-family rdu, none counted | Counties with a value that are missing from the map |
|---|---|---|---|---|---|---|
| draft adoption and demand maps (every rdu in the table) | 3,098 | 10 | 1 | 9 | 0 | 0 |
| draft, rdu with a result (not what the draft mapped) | 3,090 | 18 | 1 | 9 | 8 | 0 |
| current maps (study sample) | 3,079 | 29 | 1 | 9 | 19 | 0 |

---

## 4. Rows of the co-author writeup, section 2.5, that the main text does not print

Recomputed from the two runs. The writeup took its August rebate totals from run
`2026-08-17_19-16`; that run has no MP3 or MP4 results table on disk. Run
`2026-08-19_20-56` gives the same totals to the $0.1 million the records hold ($360,661.9M
and $98,891.8M for MP3; $385,424.6M and $107,043.4M for MP4).

| Quantity (MP3 / MP4) | Draft run 2026-08-19_20-56, 260,211 rdu | Current run 2026-10-09_00-46, 221,205 rdu | Export value | Status |
|---|---|---|---|---|
| Mean NPV, headline case (both credits, no rebate) | -$5,359.94 / -$6,049.17 | -$9,867.74 / -$6,431.45 | -$9,867.74 / -$6,431.45; equal | not in main text |
| Mean lifetime heating bill savings, not discounted | $8,133.78 / $11,947.14 | -$1,988.85 / $7,169.25 | -$1,988.85 / $7,169.25; equal | not in main text |
| Share of homes whose heating bills rise | 22.7358% / 2.1886% | 44.7743% / 19.2405% | 44.7743% / 19.2405%; equal | not in main text |
| Heating cost change per year, median of county medians | -50.0825% / -66.8325% | -21.0016% / -55.3387% | county file, median of operating_cost_pct_change: see the note below | not in main text |
| Rebate potential, Dec 2024 rules, no funding cap (every eligible home) | $360,661,901,945 / $385,424,579,340 | $293,646,491,629 / $329,035,335,238 | not in exports (Dec 2024 amount not shipped) | not in main text |
| Rebate potential, June 2026 rules, no funding cap (every eligible home) | $98,891,807,865 / $107,043,353,680 | $82,422,182,433 / $93,739,472,406 | household file: weight x June 2026 amount; see the note below | not in main text |
| Heating emissions avoided per home over 15 years, long-run rate (t CO2e) | 51.1613 / 56.9415 | 36.2282 / 49.9072 | not in exports | not in main text |

Notes.

- The draft column is over 260,211 rdu and the current column over 221,205 rdu, so each
  row mixes the fixes with the change of sample.
- Heating cost change per year: 258,897 (draft) and 220,185 (current) rdu have a value; it
  is blank where the old heating cost is zero. Over 3,090 and 3,079 counties. The export
  county file's `operating_cost_pct_change` has a median of -21.0016% (MP3) and -55.3387%
  (MP4), equal to the current column.
- Heating emissions avoided: 260,204 and 221,200 rdu have a value (5 sample rdu have no
  grid region; CLAUDE.md Limitation 12).
- Rebate potential is weight x rebate amount over every eligible home, as
  `summarize_rebate_funding` reports `total_eligible`. No funding cap is modeled. The June
  2026 amount is in the export household file; its eligible rdu count there is 52,089
  (MP3) and 60,883 (MP4), equal to the results tables.

---

## 5. Crosswalk section 5, item by item

### 5.1 Per-marker homes by fuel (item 1)

Resolved. Section 2.11 lists all 30 markers with adopter rdu and homes. Current, in
millions of homes (no rebate / December 2024 / June 2026): MP3 electricity 6.973 / 11.006
/ 11.006; natural gas 1.695 / 10.796 / 1.695; propane 1.205 / 2.035 / 1.205; fuel oil
0.489 / 1.144 / 0.489. MP4 electricity 6.141 / 11.190 / 11.190; natural gas 2.009 / 10.852
/ 2.009; propane 1.230 / 2.285 / 1.230; fuel oil 1.047 / 2.114 / 1.047.

### 5.2 "$42 billion in HEEHR and $8 billion in HOMES" (item 4; p18 par 1)

Resolved. The metric is `summarize_rebate_funding(df, 4, 'v4MID', guidance='june2026',
adopter_col='ref2025_mp4_heatingLCC_coolingSavings_sub_june2026_econ_adopter_fixed_base')`,
column `adopters_only`. That call is in the main notebook's exports of 21, 23 and 30 July
2026 (`tare_model_main_v2_3_EXPORT_*.py` at commit `2f54ed2`). In words: the
high-efficiency package, June 2026 rules, dollars reaching economic adopters only, where
adopters are those of the heating-credit-only case.

| | HEEHR | HOMES |
|---|---|---|
| Draft text | $42 billion | $8 billion |
| Draft run `2026-08-19_20-56`, that metric | $41.5770B | $8.3012B |
| Current run `2026-10-09_00-46`, that metric | $38.2919B | $7.6552B |
| Current run, base case adopters (`heatingLCC_coolingLCC_sub_june2026`), MP4 | $63.3184B | $11.7443B |
| Current run, base case adopters, MP3 | $62.6330B | $5.5254B |
| Current run, every eligible home, June 2026, MP4 | $77.5133B | $16.2262B |
| Current run, every eligible home, June 2026, MP3 | $75.2294B | $7.1927B |

Three things to settle before the sentence is reused: it is one package only; it is not
the paper's base case; and it is adopters only, while the sentence before it describes an
uncapped potential. The table below gives every candidate.

| Run | Package | Rules | Homes summed | HEEHR | HOMES | Total |
|---|---|---|---|---|---|---|
| Draft run | MP3 | Dec 2024 | adopters only: heatingLCC_coolingLCC_sub | $214.8842B | $34.5537B | $249.4379B |
| Draft run | MP3 | Dec 2024 | adopters only: heatingLCC_coolingSavings_sub | $119.1222B | $17.1760B | $136.2982B |
| Draft run | MP3 | Dec 2024 | adopters only: heatingSavings_coolingLCC_sub | $164.2733B | $21.2624B | $185.5357B |
| Draft run | MP3 | Dec 2024 | every eligible home (total_eligible) | $296.5543B | $64.1076B | $360.6619B |
| Draft run | MP3 | June 2026 | adopters only: heatingLCC_coolingLCC_sub_june2026 | $68.3837B | $8.7394B | $77.1231B |
| Draft run | MP3 | June 2026 | adopters only: heatingLCC_coolingSavings_sub_june2026 | $47.4293B | $6.8862B | $54.3155B |
| Draft run | MP3 | June 2026 | adopters only: heatingSavings_coolingLCC_sub_june2026 | $58.7969B | $7.2804B | $66.0773B |
| Draft run | MP3 | June 2026 | every eligible home (total_eligible) | $87.1998B | $11.6920B | $98.8918B |
| Draft run | MP4 | Dec 2024 | adopters only: heatingLCC_coolingLCC_sub | $169.9459B | $27.2144B | $197.1604B |
| Draft run | MP4 | Dec 2024 | adopters only: heatingLCC_coolingSavings_sub | $77.6354B | $15.2857B | $92.9211B |
| Draft run | MP4 | Dec 2024 | adopters only: heatingSavings_coolingLCC_sub | $95.2947B | $17.5365B | $112.8312B |
| Draft run | MP4 | Dec 2024 | every eligible home (total_eligible) | $310.4156B | $75.0090B | $385.4246B |
| Draft run | MP4 | June 2026 | adopters only: heatingLCC_coolingLCC_sub_june2026 | $64.7752B | $11.7777B | $76.5529B |
| Draft run | MP4 | June 2026 | adopters only: heatingLCC_coolingSavings_sub_june2026 | $41.5770B | $8.3012B | $49.8782B |
| Draft run | MP4 | June 2026 | adopters only: heatingSavings_coolingLCC_sub_june2026 | $45.5456B | $8.3874B | $53.9330B |
| Draft run | MP4 | June 2026 | every eligible home (total_eligible) | $89.8923B | $17.1510B | $107.0434B |
| Current run | MP3 | Dec 2024 | adopters only: heatingLCC_coolingLCC_sub | $150.5238B | $13.0649B | $163.5887B |
| Current run | MP3 | Dec 2024 | adopters only: heatingLCC_coolingSavings_sub | $68.5157B | $5.6407B | $74.1564B |
| Current run | MP3 | Dec 2024 | adopters only: heatingSavings_coolingLCC_sub | $110.1501B | $7.8726B | $118.0227B |
| Current run | MP3 | Dec 2024 | every eligible home (total_eligible) | $250.4852B | $43.1613B | $293.6465B |
| Current run | MP3 | June 2026 | adopters only: heatingLCC_coolingLCC_sub_june2026 | $62.6330B | $5.5254B | $68.1584B |
| Current run | MP3 | June 2026 | adopters only: heatingLCC_coolingSavings_sub_june2026 | $40.6146B | $3.7772B | $44.3918B |
| Current run | MP3 | June 2026 | adopters only: heatingSavings_coolingLCC_sub_june2026 | $54.6914B | $4.3666B | $59.0580B |
| Current run | MP3 | June 2026 | every eligible home (total_eligible) | $75.2294B | $7.1927B | $82.4222B |
| Current run | MP4 | Dec 2024 | adopters only: heatingLCC_coolingLCC_sub | $156.8544B | $25.3095B | $182.1639B |
| Current run | MP4 | Dec 2024 | adopters only: heatingLCC_coolingSavings_sub | $69.0115B | $12.0421B | $81.0537B |
| Current run | MP4 | Dec 2024 | adopters only: heatingSavings_coolingLCC_sub | $91.9156B | $15.4451B | $107.3607B |
| Current run | MP4 | Dec 2024 | every eligible home (total_eligible) | $262.0716B | $66.9638B | $329.0353B |
| Current run | MP4 | June 2026 | adopters only: heatingLCC_coolingLCC_sub_june2026 | $63.3184B | $11.7443B | $75.0627B |
| Current run | MP4 | June 2026 | adopters only: heatingLCC_coolingSavings_sub_june2026 | $38.2919B | $7.6552B | $45.9471B |
| Current run | MP4 | June 2026 | adopters only: heatingSavings_coolingLCC_sub_june2026 | $46.9558B | $8.4668B | $55.4226B |
| Current run | MP4 | June 2026 | every eligible home (total_eligible) | $77.5133B | $16.2262B | $93.7395B |

### 5.3 Mean county demand increase, MP3 (item 2)

Measured: +50.5522% in the draft (3,098 counties) and +49.2144% now (3,079 counties). The
sentence is not in the reference draft, so this is moot unless it is reinstated.

### 5.4 The home-count wrinkle (crosswalk section 1)

- 80.27M: the draft multiplied a rate by 80,273,601 homes, which is 331,531 rows x 242.13.
  The funnel's 331,526 rdu x 242.13101273 is 80,272,726.
- "80.1 M": 330,634 rows with a fuel label x 242.13 = 80,056,410 homes. The code path is
  in section 2.11.
- 77.8M: the four fuels drawn hold 321,357 rows = 77,810,170 homes in the draft table. The
  funnel step is 321,352 rdu = 77,809,285 homes.

### 5.5 Allegheny economic-adopter peak after boiler pricing (item 9)

Resolved on 11 Oct 2026. Jordan loaded run `2026-10-09_00-46` in the main notebook with
grid impact on (section 2.13). On 10 Oct this item read [NEEDED].

| Package | Economic adopters (rdu) | Peak after the retrofit | Change from 714.51 MW |
|---|---|---|---|
| MP3 | 45 of 1,146 | 713.86 MW @ hour 4433 | -0.65 MW |
| MP4 | 68 of 1,146 | 715.16 MW @ hour 4433 | +0.65 MW |

- The notebook prints two decimals. From the cached hourly tables, with the repo's own
  function, the same peaks are 713.8625 MW (-0.6521 MW) and 715.1649 MW (+0.6504 MW). The
  same method reproduces every recorded peak (section 2.6).
- MP3 is the same 45 rdu as before boiler pricing, so its value did not move.
- MP4 gained one adopter. Its value replaces 715.26 MW (+0.74 MW), measured with 67
  adopters on run `2026-10-05_21-33`.
- The peak stays at hour 4433 (4 July) for both packages.

### 5.6 Items already answered, checked here

- Item 1, fuel-group sizes: 63,781 / 131,304 / 12,249 / 13,871 rdu (28.8334%, 59.3585%,
  5.5374%, 6.2707%). Matches.
- Item 3, Allegheny natural gas share: 91.6230% (1,050 of 1,146). Matches.
- Item 5, EIA denominator: 1,514,993 GWh reproduces the draft's 28% and 2% (28.1878% and
  2.0210%). The figure itself was not checked against EIA.
- Item 6, county figures: confirmed, the draft and the current figures are different
  measures (section 2.3, F2-4 and F2-5).
- Item 8, SI values: out of scope here.

---

## 6. Zenodo files

**Staging folder.** The session prompt left the path as a placeholder. A search of the
repo, its parent and the home folder for "zenodo" found one staging folder:
`~/Desktop/CMU/Scott-Trane Externship/Nature Communications/September/Zenodo/`. It holds
`ncomms_joseph_2026_supplementary_data/` (60 files, staged 2 Sep 2026) and a zip of it.

**What is staged is the draft run.** All 9 result files under `study_results/` are run
`2026-08-19_20-56`. Nothing from the current run is staged.

- Every staged file is byte-identical to the file it was copied from (the repo's
  `cmu_tare_model/data/` for inputs, `cmu_tare_model/output_results/` for results).
- The zip holds the same 60 files; each entry's size and CRC-32 equal the staged file.
- No staged file holds MP5 or ResStock 2025.1 rows, and none describes them. The export
  README and dictionary workbook is not staged.
- For Jordan to decide, not decided here: the folder also holds a temporary Office lock
  file (`~$REMDB_2024_12_23.xlsx`, 165 bytes); the `tl_2025` county and state shapefiles,
  which the code does not read (it reads `cb_2021`); ResStock upgrade files 07 to 10
  (8.5 GB), which MP3 and MP4 do not use; and no export file at all.

| Staged file | Bytes | SHA-256 | Compared with | Result | In the zip | MP5 or ResStock 2025.1 content |
|---|---|---|---|---|---|---|
| `data/.gitkeep` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `cmu_tare_model/data/.gitkeep` | byte-identical | same size and CRC-32 | no |
| `data/__init__.py` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `cmu_tare_model/data/__init__.py` | byte-identical | same size and CRC-32 | no |
| `data/ami_calculations_data/ACSDT5Y2024.B19013/ACSDT5Y2024.B19013-Column-Metadata.csv` | 298 | `bd97576ed693e3aaa69c63686894dca7085c3dcf084f3a4696a43f2c0922510b` | `cmu_tare_model/data/ami_calculations_data/ACSDT5Y2024.B19013/ACSDT5Y2024.B19013-Column-Metadata.csv` | byte-identical | same size and CRC-32 | no |
| `data/ami_calculations_data/ACSDT5Y2024.B19013/ACSDT5Y2024.B19013-Data.csv` | 197,205 | `9b385c47135208bc00e226dd87de02ac80f521ccbca3a588c2abaf462ad771a9` | `cmu_tare_model/data/ami_calculations_data/ACSDT5Y2024.B19013/ACSDT5Y2024.B19013-Data.csv` | byte-identical | same size and CRC-32 | no |
| `data/ami_calculations_data/ACSDT5Y2024.B19013/ACSDT5Y2024.B19013-Table-Notes.txt` | 11,500 | `448d1a1e10dc2a0cd7a2383213e7a1f1c245992c4c58aded8b6a7e8c5fb8e446` | `cmu_tare_model/data/ami_calculations_data/ACSDT5Y2024.B19013/ACSDT5Y2024.B19013-Table-Notes.txt` | byte-identical | same size and CRC-32 | no |
| `data/euss_data/.gitkeep` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `cmu_tare_model/data/euss_data/.gitkeep` | byte-identical | same size and CRC-32 | no |
| `data/euss_data/resstock_amy2018_release_1.1/national/csv/baseline_metadata_and_annual_results.csv` | 1,663,147,758 | `91d39a98ebfcb9a99173ee61348e22565ca9bdaa59dd3c612dd82522ab3d0d8e` | `cmu_tare_model/data/euss_data/resstock_amy2018_release_1.1/national/csv/baseline_metadata_and_annual_results.csv` | byte-identical | same size and CRC-32 | no |
| `data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade03_metadata_and_annual_results.csv` | 2,072,669,419 | `8112832892b0b5786d2d39306b2fa7128b1fb5a2a1d2322f3e67248faa469b4a` | `cmu_tare_model/data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade03_metadata_and_annual_results.csv` | byte-identical | same size and CRC-32 | no |
| `data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade04_metadata_and_annual_results.csv` | 2,077,357,175 | `bb35217854f1e3ba189ae16ded95f464222ed63188a4b505eb878e88e6d5ebcd` | `cmu_tare_model/data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade04_metadata_and_annual_results.csv` | byte-identical | same size and CRC-32 | no |
| `data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade07_metadata_and_annual_results.csv` | 2,025,677,545 | `8bdcbd434b325e9239b741ae8642af82c6800b804f151702cdd1e6876fd19705` | `cmu_tare_model/data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade07_metadata_and_annual_results.csv` | byte-identical | same size and CRC-32 | no |
| `data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade08_metadata_and_annual_results.csv` | 2,140,984,327 | `65cd5bf9ef9aec770dbd63cf7c147eb8181f6cabbc169c928a52a3e2a61d53b4` | `cmu_tare_model/data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade08_metadata_and_annual_results.csv` | byte-identical | same size and CRC-32 | no |
| `data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade09_metadata_and_annual_results.csv` | 2,153,555,844 | `ea9e568decf49b1c4a5fed057c1bb766b3e2ed4b6b6a894cb38230348453be73` | `cmu_tare_model/data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade09_metadata_and_annual_results.csv` | byte-identical | same size and CRC-32 | no |
| `data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade10_metadata_and_annual_results.csv` | 2,160,131,545 | `e957b708b013c7d4a7f49ccb4c71a6f176166898ac9914982500aa25958d3079` | `cmu_tare_model/data/euss_data/resstock_amy2018_release_1.1/national/csv/upgrade10_metadata_and_annual_results.csv` | byte-identical | same size and CRC-32 | no |
| `data/fuel_prices/eia_fuel_price_data_2025_usd2025.csv` | 20,909 | `de1f38d9171df9aa5a275ed20c6ba668b8647f816d0a25183a5fef996434171c` | `cmu_tare_model/data/fuel_prices/eia_fuel_price_data_2025_usd2025.csv` | byte-identical | same size and CRC-32 | no |
| `data/inflation_data/bls_cpiu_2005-2025.xlsx` | 10,171 | `2556aa2fa25376bf64f550b5d92cec0eafb2ce3bb78563896df7538c2780dbb3` | `cmu_tare_model/data/inflation_data/bls_cpiu_2005-2025.xlsx` | byte-identical | same size and CRC-32 | no |
| `data/projections/aeo2026_degree_day_factors_2025_2050.csv` | 9,986 | `7282ec45c9b9be9d485496ad9e8e64564d71468a97b603711ad6aae004cc827d` | `cmu_tare_model/data/projections/aeo2026_degree_day_factors_2025_2050.csv` | byte-identical | same size and CRC-32 | no |
| `data/projections/aeo2026_fuel_price_factors_2025_2050.csv` | 21,397 | `2ee88e986586f271e31c5995b6cc40899d17ff99b383b99586e3bc71bcb27b97` | `cmu_tare_model/data/projections/aeo2026_fuel_price_factors_2025_2050.csv` | byte-identical | same size and CRC-32 | no |
| `data/projections/cambium24_allScenarios_annual_gea.csv` | 660,398 | `e5d83c6b31e78d343a6921953da79287928b804073c973ce92c1be909ab947c6` | `cmu_tare_model/data/projections/cambium24_allScenarios_annual_gea.csv` | byte-identical | same size and CRC-32 | no: the search hit is a number (2025.1...), not a mention of ResStock 2025.1 |
| `data/projections/county_to_gea_mapping_cambium23.csv` | 128,982 | `fdb2fd42650e24345e717d102e7f0e035099e8f012e8814d05b96a4b398cff87` | `cmu_tare_model/data/projections/county_to_gea_mapping_cambium23.csv` | byte-identical | same size and CRC-32 | no |
| `data/projections/scc_climate_impact_sensitivity.xlsx` | 141,571 | `bd6a196c594b1df7f017cab28f9dc624e65ff7b4d86212bb11b09ad5f3efcaa0` | `cmu_tare_model/data/projections/scc_climate_impact_sensitivity.xlsx` | byte-identical | same size and CRC-32 | no |
| `data/retrofit_costs/REMDB_2024_12_23.xlsx` | 58,410 | `69aed9119d1a844754b5b51b28621941b11a8fb4c8cabe8a47c9ac066e85a253` | `cmu_tare_model/data/retrofit_costs/REMDB_2024_12_23.xlsx` | byte-identical | same size and CRC-32 | no |
| `data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv` | 8,305 | `55e14a683659638a158be7311ceb0e1383d86ff41441eeaff057e1142d8d3c74` | `cmu_tare_model/data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv` | byte-identical | same size and CRC-32 | no |
| `data/retrofit_costs/~$REMDB_2024_12_23.xlsx` | 165 | `92c1e64723137486ce4ded9ef471b58543bf2dcf70011cdf77b21db94997aa6e` | `cmu_tare_model/data/retrofit_costs/~$REMDB_2024_12_23.xlsx` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.cpg` | 5 | `3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824` | `cmu_tare_model/data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.cpg` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.dbf` | 1,180,828 | `263406c27466b8dd41ed22c1d498fbad59bea10415cc47a4f77f270c6cef0909` | `cmu_tare_model/data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.dbf` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.prj` | 165 | `0b9041e921d9ebb43247d314608fe9e38a0b008ee793067fc1806199ea1fb9dd` | `cmu_tare_model/data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.prj` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.shp` | 16,837,620 | `d78457ade3064b7628623602da8b71e908b4fa97a3d2edbaaa780ad3c770a23f` | `cmu_tare_model/data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.shp` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.shp.ea.iso.xml` | 26,550 | `36c48969f84a9be6f3bca6d54e5bca7b699c23a3bf6530790f766ca04984070b` | `cmu_tare_model/data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.shp.ea.iso.xml` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.shp.iso.xml` | 35,074 | `62b05547e097151a1d80e9dae7cc7b3edc2bf38f37daa525e11d2b4e6f17a1f9` | `cmu_tare_model/data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.shp.iso.xml` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.shx` | 25,972 | `7f60297a50eccab3e210f6eb7123a42db387837b29a8a17c81fb529e0bd3f140` | `cmu_tare_model/data/shapefiles/cb_2021_us_county_500k/cb_2021_us_county_500k.shx` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.cpg` | 5 | `3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824` | `cmu_tare_model/data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.cpg` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.dbf` | 9,058 | `d20ab2d8b762816430781318a4f312b66aa351eb167c668c817c54a9c780ddc1` | `cmu_tare_model/data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.dbf` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.prj` | 165 | `0b9041e921d9ebb43247d314608fe9e38a0b008ee793067fc1806199ea1fb9dd` | `cmu_tare_model/data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.prj` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.shp` | 4,697,740 | `4041681e91991640e1e0554cb8014592922bd3aeb39a050a806c38c427ef9532` | `cmu_tare_model/data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.shp` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.shp.ea.iso.xml` | 15,908 | `577948fee1e14acc0783269973a345571212a5e9371b3517b3ce25512261550e` | `cmu_tare_model/data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.shp.ea.iso.xml` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.shp.iso.xml` | 32,423 | `5f074fee265fb7e5644fda6acd773fe014e8b3aee03e17f708a2b2fe0e56a513` | `cmu_tare_model/data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.shp.iso.xml` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.shx` | 548 | `9d45a7e54d02331a5c58686cef16e0ffa53bc7fa6ec4cdb1162e1ce69ed7bec8` | `cmu_tare_model/data/shapefiles/cb_2021_us_state_500k/cb_2021_us_state_500k.shx` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_county/tl_2025_us_county.cpg` | 5 | `3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824` | `cmu_tare_model/data/shapefiles/tl_2025_us_county/tl_2025_us_county.cpg` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_county/tl_2025_us_county.dbf` | 993,755 | `966a258724c861daa722dfa64038df494fac089781c695a88615d9786e61a5cb` | `cmu_tare_model/data/shapefiles/tl_2025_us_county/tl_2025_us_county.dbf` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_county/tl_2025_us_county.prj` | 165 | `0b9041e921d9ebb43247d314608fe9e38a0b008ee793067fc1806199ea1fb9dd` | `cmu_tare_model/data/shapefiles/tl_2025_us_county/tl_2025_us_county.prj` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_county/tl_2025_us_county.shp` | 132,270,192 | `d2add252ccd38f1c46efaba677efe0d20cdd9bf1f3ff7af7ce84fe06a3205df7` | `cmu_tare_model/data/shapefiles/tl_2025_us_county/tl_2025_us_county.shp` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_county/tl_2025_us_county.shp.ea.iso.xml` | 42,065 | `b5f0d05b1df7a4db80b897b1bb5a56819e1ca10f43e3e3c965d64cef5cee34c7` | `cmu_tare_model/data/shapefiles/tl_2025_us_county/tl_2025_us_county.shp.ea.iso.xml` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_county/tl_2025_us_county.shp.iso.xml` | 50,716 | `8c5b6359d59a06e0fdb8c9564a925e993aeea5b0575505365555442a316444c8` | `cmu_tare_model/data/shapefiles/tl_2025_us_county/tl_2025_us_county.shp.iso.xml` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_county/tl_2025_us_county.shx` | 25,980 | `b570cb996fa7620e36ebeae8964f0b606fd8d7df700b33f6950c7e268ea281fe` | `cmu_tare_model/data/shapefiles/tl_2025_us_county/tl_2025_us_county.shx` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_state/tl_2025_us_state.cpg` | 5 | `3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824` | `cmu_tare_model/data/shapefiles/tl_2025_us_state/tl_2025_us_state.cpg` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_state/tl_2025_us_state.dbf` | 11,098 | `b7716edb108a571e5892fc11dbbb027a46e343113ab9b872dae72177476d807d` | `cmu_tare_model/data/shapefiles/tl_2025_us_state/tl_2025_us_state.dbf` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_state/tl_2025_us_state.prj` | 165 | `0b9041e921d9ebb43247d314608fe9e38a0b008ee793067fc1806199ea1fb9dd` | `cmu_tare_model/data/shapefiles/tl_2025_us_state/tl_2025_us_state.prj` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_state/tl_2025_us_state.shp` | 15,534,960 | `181a2728c6c0ecc94e02ed45d6a6f28871ec8d52191893d5b1fea48ac7d99be7` | `cmu_tare_model/data/shapefiles/tl_2025_us_state/tl_2025_us_state.shp` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_state/tl_2025_us_state.shp.ea.iso.xml` | 32,964 | `266aab70b82d1a0964e621951ea5c185d5863c842697d6936294671e34ea4d85` | `cmu_tare_model/data/shapefiles/tl_2025_us_state/tl_2025_us_state.shp.ea.iso.xml` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_state/tl_2025_us_state.shp.iso.xml` | 49,139 | `b199416773957109d33a91c9e7c7d64ee9ceada2ce11363fe3f911aaf683d669` | `cmu_tare_model/data/shapefiles/tl_2025_us_state/tl_2025_us_state.shp.iso.xml` | byte-identical | same size and CRC-32 | no |
| `data/shapefiles/tl_2025_us_state/tl_2025_us_state.shx` | 548 | `734438dabaee5dd5069cfd18ea17a16453548a48007f504c070982f4820bee42` | `cmu_tare_model/data/shapefiles/tl_2025_us_state/tl_2025_us_state.shx` | byte-identical | same size and CRC-32 | no |
| `study_results/baseline_summary/summary_baseline/baseline_results_National_2026-08-19_20-56.csv` | 200,286,877 | `0bf650633d2b29cd0c7879c928ab66b3e56fbbd6c413e55426c55992a336b7ba` | `cmu_tare_model/output_results/baseline_summary/summary_baseline/baseline_results_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `study_results/retrofit_mp3_results/summary_mp3_fixed_base/mp3_results_National_2026-08-19_20-56.csv` | 515,647,726 | `57936aec274f300aa03ec75e4ce17863dde8da10644de1edbc4dfad3dc80e0c9` | `cmu_tare_model/output_results/retrofit_mp3_results/summary_mp3_fixed_base/mp3_results_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `study_results/retrofit_mp4_results/summary_mp4_fixed_base/mp4_results_National_2026-08-19_20-56.csv` | 514,348,002 | `7814df813d07ad14a1b974a3c6232078c13ad32021c5b7c40cfdcb67c3c98fbf` | `cmu_tare_model/output_results/retrofit_mp4_results/summary_mp4_fixed_base/mp4_results_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `study_results/supplemental_data_damages/damages_climate_baseline/mp0_damages_climate_baseline_National_2026-08-19_20-56.csv` | 458,695,389 | `67a2a9e10c6a1fcf54e3a3728d8214a1e4da7e59200ee8c7d5ff7493ad5f0e94` | `cmu_tare_model/output_results/supplemental_data_damages/damages_climate_baseline/mp0_damages_climate_baseline_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `study_results/supplemental_data_damages/damages_climate_ref2025/mp3_damages_climate_ref2025_National_2026-08-19_20-56.csv` | 477,800,364 | `790016ff57af03e1a3ecbcac9529d057c97daf8e41a8aa0bb2353576c34cf577` | `cmu_tare_model/output_results/supplemental_data_damages/damages_climate_ref2025/mp3_damages_climate_ref2025_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `study_results/supplemental_data_damages/damages_climate_ref2025/mp4_damages_climate_ref2025_National_2026-08-19_20-56.csv` | 468,731,743 | `fd57e967b6005fe45353dae34be174fb75d32d21f953cdd48955717d3cd9c616` | `cmu_tare_model/output_results/supplemental_data_damages/damages_climate_ref2025/mp4_damages_climate_ref2025_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `study_results/supplemental_data_fuelCosts/fuel_costs_baseline/mp0_fuel_costs_baseline_National_2026-08-19_20-56.csv` | 140,576,464 | `ef915a4aabd27aa9882649360ed0fa103ad5b4f6e6d11cf2101eb06cf4e29cc5` | `cmu_tare_model/output_results/supplemental_data_fuelCosts/fuel_costs_baseline/mp0_fuel_costs_baseline_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `study_results/supplemental_data_fuelCosts/fuel_costs_ref2025/mp3_fuel_costs_ref2025_National_2026-08-19_20-56.csv` | 280,604,282 | `b6a154531e8e5f54313ff29167c2a424bc7f76d34e17a6c94a12b4c242748f37` | `cmu_tare_model/output_results/supplemental_data_fuelCosts/fuel_costs_ref2025/mp3_fuel_costs_ref2025_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `study_results/supplemental_data_fuelCosts/fuel_costs_ref2025/mp4_fuel_costs_ref2025_National_2026-08-19_20-56.csv` | 278,073,959 | `a95aeaa6ef83775ce708513eec4c6a0e9f8e0214a5c1013acd64123016072bbc` | `cmu_tare_model/output_results/supplemental_data_fuelCosts/fuel_costs_ref2025/mp4_fuel_costs_ref2025_National_2026-08-19_20-56.csv` | byte-identical | same size and CRC-32 | no |
| `ncomms_joseph_2026_supplementary_data.zip` | 3,953,499,041 | `2bebac72f46129426b7643caeb95a7a315728835683c59b3e31cff66136a3454` | `the staged folder, entry by entry` | 60 entries |  | no |

**What a results deposit for the paper would hold, from the current run.** These are the
22 files run `2026-10-09_00-46` wrote, with their SHA-256. The same list with stamp
`2026-10-09_21-04` is byte-identical. The main notebook wrote the 12 export files again
on 11 Oct 2026 (cell 23); all 22 files still have the SHA-256 values below. Which of them
go to Zenodo is Jordan's call. The
exports are the `tepper_export/` files; their README and dictionary workbook describes
both releases and would need a 2022.1.1-only version.

| Output of run 2026-10-09_00-46 | Bytes | SHA-256 | Same file of run 2026-10-09_21-04 |
|---|---|---|---|
| `baseline_summary/sample_funnel/sample_funnel_National_2026-10-09_00-46.csv` | 1,623 | `7fe25cc1b9aaa135044fa64f51c7207d9e314f2c44abe4f96c552a1c8eedc56a` | byte-identical |
| `baseline_summary/summary_baseline/baseline_results_National_2026-10-09_00-46.csv` | 268,904,740 | `c6d3d74f743975a441195b3a613b9eceb790815614fa6bbe70d9d30434c7ed6b` | byte-identical |
| `retrofit_mp3_results/summary_mp3_fixed_base/mp3_results_National_2026-10-09_00-46.csv` | 738,542,669 | `5e6ad790315e0315c26bbcb3259d591aa30f6183a9d3d374ef5b8c0e4283938f` | byte-identical |
| `retrofit_mp4_results/summary_mp4_fixed_base/mp4_results_National_2026-10-09_00-46.csv` | 735,417,975 | `ae0fb339b18b9ad4e29e081c470a01f22ad81ca515001bb475e0fac54dfaf7c5` | byte-identical |
| `supplemental_data_damages/damages_climate_baseline/mp0_damages_climate_baseline_National_2026-10-09_00-46.csv` | 1,087,378,043 | `03d2f017886d0e2c5157af95955282a426ad53cba85f9508fba162ab1645e559` | byte-identical |
| `supplemental_data_damages/damages_climate_ref2025/mp3_damages_climate_ref2025_National_2026-10-09_00-46.csv` | 1,157,956,055 | `c5c7319c87f4c7642886adb33e1da4ae35ede01b9a946fc7e077f21b6e69b528` | byte-identical |
| `supplemental_data_damages/damages_climate_ref2025/mp4_damages_climate_ref2025_National_2026-10-09_00-46.csv` | 1,158,127,955 | `465cf6167f84f3ca047da990be9f279dc7bd8a89c615271bf95a459ab0c6fa9a` | byte-identical |
| `supplemental_data_fuelCosts/fuel_costs_baseline/mp0_fuel_costs_baseline_National_2026-10-09_00-46.csv` | 474,833,898 | `90310c7b421bfc74bf7f812ea92e6e95a68df5b87d667ab1323c4dbee2d616f1` | byte-identical |
| `supplemental_data_fuelCosts/fuel_costs_ref2025/mp3_fuel_costs_ref2025_National_2026-10-09_00-46.csv` | 936,425,947 | `2dcca3a86031ac5d7e7523b2a64a4578ac5cb19fd3d6dc5f4ac4b0937e7cfffc` | byte-identical |
| `supplemental_data_fuelCosts/fuel_costs_ref2025/mp4_fuel_costs_ref2025_National_2026-10-09_00-46.csv` | 936,195,148 | `6426ca1e3e208e1b0fe430815ad48f453d66dd74f84cbad97b2554c8f5ffbad0` | byte-identical |
| `tepper_export/tepper_county_mp3_Allegheny_2026-10-09_00-46.csv` | 290 | `5c90e49b03092d5100dc4e2fc066ff34fbcf9bd8ec9e02e0f9d878a74c039f7c` | byte-identical |
| `tepper_export/tepper_county_mp3_National_2026-10-09_00-46.csv` | 290,732 | `84500d6bcc7721ecbf236a29c21b64469b71515655e6a34f221d72efcdf3e877` | byte-identical |
| `tepper_export/tepper_county_mp4_Allegheny_2026-10-09_00-46.csv` | 289 | `e87921156fac008c2d23c328e827e661390d916a0a5eee30fda8e0f478d597f9` | byte-identical |
| `tepper_export/tepper_county_mp4_National_2026-10-09_00-46.csv` | 292,844 | `c15bb7d9a74456eb3159eadd9d7339355ecff274828a759dadce2c08b7bb36d4` | byte-identical |
| `tepper_export/tepper_household_detailed_mp3_Allegheny_2026-10-09_00-46.csv` | 4,551,165 | `7d464e97287c777a819f1d3121ce513ab00705c1cfecf53b536264700dc11012` | byte-identical |
| `tepper_export/tepper_household_detailed_mp3_National_2026-10-09_00-46.csv` | 863,123,018 | `a86e089648ef4640f4686a2a1a8ce093e9e64f06c09cdb6299c0819e4c402c0c` | byte-identical |
| `tepper_export/tepper_household_detailed_mp4_Allegheny_2026-10-09_00-46.csv` | 4,543,748 | `dcf1e2b7828bc330020a6f5f0af335d74cd95ddf53e4b3c0dfda49f8dfd0fb35` | byte-identical |
| `tepper_export/tepper_household_detailed_mp4_National_2026-10-09_00-46.csv` | 861,681,342 | `1f3869b218ab556d794d92dff7e4025a2f092ae00634667124066088a75c46b2` | byte-identical |
| `tepper_export/tepper_household_mp3_Allegheny_2026-10-09_00-46.csv` | 2,659,806 | `79e6e88903a67f97daee61fffa8a02df76fe59a135f77379f808ce6977295019` | byte-identical |
| `tepper_export/tepper_household_mp3_National_2026-10-09_00-46.csv` | 511,306,217 | `a0805b181d7058245a51798d5b2311e6765ee9c5b4b9bb16c209132af6d90a28` | byte-identical |
| `tepper_export/tepper_household_mp4_Allegheny_2026-10-09_00-46.csv` | 2,653,360 | `06a8ef8d71c0dd352a2377451e2f7edb771ab1992dea52a23e0f883c0d9ea5c8` | byte-identical |
| `tepper_export/tepper_household_mp4_National_2026-10-09_00-46.csv` | 509,935,690 | `4c7a3e57a5e26c10eebe9ae567eb9499527bca0c5b61392757596111ad2d12e7` | byte-identical |

---

## 7. What could not be checked, and why

1. **Allegheny economic-adopter peak after boiler pricing.** Closed on 11 Oct 2026. On
   10 Oct no grid impact run existed after the change. Section 5.5 now holds the value
   from the notebook run of 11 Oct.
2. **County adoption tables, county demand tables, peak records and rebate funding
   summaries of each run.** The model does not save them, and the notebooks hold no
   output. They were worked out again from the results tables, the ResStock annual files
   and the query cache, with the repo's formulas. Where a reference value exists, it is
   reproduced.
3. **Step runs not on disk.** `2026-09-30_19-06` and `2026-09-30_22-35` (homes with no AC
   out, 221,301 rdu) and `2026-10-03_03-32` are gone. The 221,301-rdu step was derived
   from run `2026-09-25_22-55` limited to those rdu. The records say every home that
   stayed was identical in the real run. Run `2026-08-17_19-16` has no MP3 or MP4 results
   table.
4. **December 2024 cases, by-fuel rebate rows and peaks against the exports.** The
   exports ship only the base case with no rebate. June 2026 adoption can be rebuilt from
   them, and it reproduces the results tables exactly (0 rdu differ).
5. **Fig 1 and census numbers** (p4 par 2; p6 par 1): 55.2%, 38.2%, 3.5%, 2.7%, the 4%
   and 5% changes, 40% / 46% and 49% / 41%. They come from the heating-fuel toolkit, not
   from a TARE run.
6. **EIA residential sales, 1,514,993 GWh.** Taken from the draft's deleted calc notes as
   quoted in the version reconciliation. Not checked against EIA.
7. **"the combined appropriation of roughly $9 billion"** (p18 par 1). Not a model number.
8. **MT-1, the dollar year of climate damages.** From the records only.
9. **Maps and figures as drawn.** Only their numbers were checked. No figure was redrawn.
10. **Supplementary Information values.** Out of scope.
11. **MP5 and ResStock 2025.1.** Not used. The query cache also holds two hourly tables
    for a 1,091-rdu set and upgrade 5, from the dual-fuel work; they were listed in the
    index of the cache and not used.

**Blockers met, and what was done.**

- The inputs were not in `~/tare_port/s12_paper_verify/inputs/` (the folder did not
  exist). They were read where they were found:

| Input | Where it was found | Bytes | SHA-256 |
|---|---|---|---|
| Crosswalk, 9 Oct 2026 version (used) | `~/tare_port/s11_task6_proposed/TARE_natcomms_results_paper_number_crosswalk.md` | 25,683 | `b046522d2cb4a3639850b83383573718e3462cd100e1cd7c951d5981c36355c5` |
| Crosswalk, 7 Oct 2026 version (not used) | `~/Downloads/TARE_natcomms_results_paper_number_crosswalk.md` | 19,439 | `8bef0ed3b82b034c6998e6c3baddbbb047898268490fd0291943b4943965969e` |
| Reference draft | `~/Desktop/CMU/Scott-Trane Externship/Nature Communications/September/NatureComm_Joseph_2026_MANUSCRIPT_5Oct2026.pdf` | 879,368 | `1ed2d2fdcb7c6610c0c5e7e176d5e565a557cb353758b72a293ff82b56098386` |
| Version reconciliation | `~/Downloads/manuscript_version_reconciliation_2026-10-10.md` | 19,574 | `429b27c1f2dde197e00d6671f6b395e065571a95c7a220327788461b637bbb29` |

- Two crosswalk files exist. The one in `~/Downloads/` is the 7 Oct 2026 version (current
  run `2026-10-05_21-33`, no section 6). The 9 Oct 2026 version, which the session prompt
  describes, is in `~/tare_port/s11_task6_proposed/`. That one was used.
- The repo's working copy of `constants.py` defaults to ResStock 2025.1 with grid impact
  off (uncommitted). Every script sets `TARE_RESSTOCK_RELEASE=2022.1.1` itself. The file
  was not touched.
- The grid-impact query cache (8.65 GB) does not fit in the memory that was free, and one
  of its columns holds 10 to 14 million Decimal objects per table. It was read one query
  at a time (`s12v_08_read_ts_cache.py`); all 23 cached queries are accounted for.
- One run of the peak script stopped on a divide-by-zero for the 1,356-rdu set, right
  after the cache had been read again. The cause was not found. It did not happen in any
  other run, and those runs gave identical results, equal to the reference values. The
  script now checks every saved table against the index of the cache before using it.
- A later 2022.1.1 run, `2026-10-09_21-04`, is on disk. Its 22 files are byte-identical to
  `2026-10-09_00-46`. No record of it was found in the session log.

---

## Check scripts

All in `~/tare_port/s12_paper_verify/scripts/`. Run with the `cmu-tare-model`
environment's python. Results are in `~/tare_port/s12_paper_verify/work/results/`.

| Script | What it does |
|---|---|
| `s12v_common.py` | Paths, column names built with the repo's helpers, and the reader that caches the needed columns of a results table |
| `s12v_00_headers.py` | Lists the columns of every results table and export file |
| `s12v_01_build_cache.py` | Reads the needed columns of each run's MP3 and MP4 results tables once |
| `s12v_02_adoption.py` | Adoption by run: national (nine cases), by fuel, by county, Allegheny, adopters lost, by old heating system |
| `s12v_03_demand.py` | Demand at 100% adoption from the ResStock annual files, national and by county, for each set of homes |
| `s12v_04_drivers_fig4.py` | Step-by-step drivers; the Allegheny denominators; the 30 Fig 4 markers and the y-axis labels |
| `s12v_05_bills_rebates_unmapped.py` | Writeup rows (Task 4), rebate dollars (the $42B and $8B search), and U2, U3, U6, U8, U9, U11, U12 |
| `s12v_06_exports.py` | The export files against the results tables; the county file's demand sum; June 2026 adoption rebuilt from the exports |
| `s12v_07_zenodo.py` | SHA-256 of every staged Zenodo file and of its source; the zip; run `2026-10-09_21-04` against `2026-10-09_00-46` |
| `s12v_08_read_ts_cache.py` | Reads the grid-impact query cache one query at a time and saves each hourly table compactly |
| `s12v_09_map_counties.py` | Counties on the map with no value, and why (U15) |
| `s12v_10_peaks.py` | Allegheny peaks for the 1,610, 1,356 and 1,146 rdu sets (Fig 5 rows and U1) |
| `s12v_11_same_homes_and_hashes.py` | Draft against current on the same 221,205 rdu; the boiler credit; run `2026-09-02_19-04` against the draft run |
| `s12v_12_build_handoff.py` and `s12v_handoff_template.md` | Build this file: the wording is in the template, and the long tables are generated from the result files. The build stops if a key number in the wording differs from the result files |
| `s12v_13_check_typed_numbers.py` | Lists every number typed in the template that equals no value in the result files, to be checked by eye (all are sums, differences, funnel values or rounded text) |
| `s12v_14_recheck_run_files.py` | 11 Oct: SHA-256 of the run's 22 files after the notebook wrote the exports again, against the 10 Oct record and run `2026-10-09_21-04` |
| `s12v_15_dump_notebook_outputs.py` | 11 Oct: writes the text outputs saved in the main notebook to one file, cell by cell |
| `s12v_16_check_notebook_outputs.py` | 11 Oct: the notebook's outputs against the original run's log, the values behind this file, and the PDF export |
