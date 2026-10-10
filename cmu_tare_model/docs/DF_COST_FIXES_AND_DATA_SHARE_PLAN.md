# Dual-fuel cost fixes and data share -- implementation plan and log

Written 8 Oct 2026 in the Claude project chat, from the continuation prompt of
8 Oct 2026 (`00_tare_web_continuation_prompt.md`), Session 7 of
`cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md`, CLAUDE.md,
`docs/REFERENCE_VALUES.md` and the files sent to Chris on 19 Aug 2026.

- CLAUDE.md and `LOCAL_KICKOFF_PROMPT.md` win wherever this plan differs from
  them. This plan relaxes neither.
- This plan covers two stages of work. **Stage 1** is the cleanup and cost
  fixes. **Stage 2** is the data validation and the data share with Chris.
  They are called stages so that they are not confused with the four parts of
  Session 7 in `LOCAL_PORT_SESSION_PLAN.md`.
- Each stage has its own session prompt (`DF_STAGE1_SESSION_PROMPT.md`,
  `DF_STAGE2_SESSION_PROMPT.md`). Both prompts read this file and take their
  decisions and expected values from it.
- **Small changes to a model that already works.** Nearly all of the code
  architecture is in place: both releases run end to end, the cost, rebate, NPV
  and export paths exist, and every REMDB row Stage 1 needs is already in the cost
  table. Stage 1 changes a few assumptions (which row prices an old system, and at
  what efficiency), adds a guard and moves a default. It should not need large
  blocks of new code, new modules or a new pricing path. Each change extends what
  is there (a mapping, a floor, a constant, a test) with the smallest diff that does
  the job, and leaves the code around it alone. A change that seems to need more is
  a stop: the session explains why before writing it. Stage 2 writes analysis
  scripts outside the repo and reuses the model's own helpers rather than
  re-deriving them.
- **Every line of code follows CLAUDE.md and the researcher's saved preferences**
  (memory): plain-language comments that say why, docstrings, type hints, 88
  characters, ASCII, one gated diff per stop, no commits by the session.
- Any edit to this file is a gated diff. The researcher makes every commit of it.
- Every count is labeled rdu or homes. One rdu is 242.131013 homes in ResStock
  2022.1.1 and 253.90367 homes in ResStock 2025.1.

Contents: 1 Results of record -- 2 Decisions taken -- 3 Decisions open -- 4 Stage 1
-- 5 Stage 2 -- 6 Changes since 19 Aug 2026 -- 7 Log.

---

## 1. Results of record, before Stage 1

Runs: `2026-10-07_19-22` (ResStock 2022.1.1, MP3 and MP4: 221,205 rdu = 53,560,591
homes) and `2026-10-07_19-14` (ResStock 2025.1, MP5: 161,983 rdu = 41,128,079
homes). National, 7% discount rate, USD2025. They agree with
`REFERENCE_VALUES.md`. A home adopts when its 15-year NPV is zero or more.

**These numbers are expected to change in Stage 1.** The column "Moves with" names
the open question (section 3) whose decision would move each one.

### 1.1 Adoption, percent of the study sample (no rebate / 2024 guidance / June 2026 guidance)

| Credit scope | MP3 | MP4 | MP5 | Moves with |
|---|---|---|---|---|
| Old AC credit only (`heatingSavings_coolingLCC`) | 9.19 / 32.11 / 18.94 | 10.70 / 29.01 / 18.99 | 1.78 / 6.30 / 6.30 | Q4, Q5, Q6 (and Q1 for MP5) |
| Old heating credit only (`heatingLCC_coolingSavings`) | 6.76 / 19.70 / 13.74 | 8.55 / 20.83 / 14.95 | 1.36 / 3.10 / 3.10 | Q2, Q3, Q4, Q6 (and Q1) |
| Both credits (`heatingLCC_coolingLCC`) | 18.81 / 45.83 / 26.34 | 18.99 / 47.84 / 28.42 | 3.27 / 29.04 / 29.04 | Q2 to Q6 (and Q1) |

MP5's two rebate scenarios are equal because a dual-fuel retrofit passes both
June 2026 fuel gates.

### 1.2 MP5, mean dollars per home (both credits)

| Part | Mean | Moves with |
|---|---|---|
| Heat pump | 15,711.23 | Q4, Q6 |
| Backup gas furnace | 4,112.21 | Q4 |
| Credit for the old heating system | 3,742.25 | Q2, Q3, Q4 |
| Credit for the old AC | 5,842.98 | Q4, Q5 |
| Net capital cost, no rebate | 10,238.21 (median 9,548.51) | all of the above |
| Bill savings, 15 years, discounted | 1,535.17 | nothing in Stage 1 |
| NPV, no rebate | -8,703.04 (median -7,941.37) | all of the above |

### 1.3 Why MP5 is low without a rebate

- **Fuel mix.** 93.44% of the MP5 sample heats with natural gas, against 59.36%
  of MP3's. No-rebate adoption of gas homes: 0.82% (MP5), 4.61% (MP3). MP5's
  rates by fuel applied to MP3's fuel mix give 16.27%.
- **The rebate decides adoption.** 73.65% of MP5 rdu have a no-rebate NPV between
  -12,000 and -4,000. The typical rebate is about 7,000 (cap 8,000). Without it,
  adoption falls 25.8 points for MP5 and 27.0 for MP3.
- **The furnace.** The backup furnace costs 370 more on average than the credit for
  the old heating system. A free furnace gives 8.48% no-rebate adoption.
- **Electric-heated homes** are 5.42% of the MP5 sample (8,778 rdu = 2,228,766
  homes) and 61.52% of its no-rebate adopters (3,257 of 5,294 rdu).
- **The releases hold different homes.** MP3 and MP5 are compared as groups
  (fuel, system type, size), never home by home. MP3 and MP4 share the same
  221,205 rdu.

### 1.4 What Stage 1 does not move

Bill savings, fuel mix, sample counts (unless Q1 changes the MP5 sample), the
default-release switch, the grid impact guard.

---

## 2. Decisions taken

| ID | Date | Decision |
|---|---|---|
| D-1 | 8 Oct 2026 | The work runs in two stages: Stage 1, cleanup and cost fixes; Stage 2, data validation and the data share with Chris. |
| D-2 | 8 Oct 2026 | One central plan and log (this file, in `cmu_tare_model/docs/`) and one session prompt per stage. The prompts take their decisions and expected values from this file. |
| D-3 | 8 Oct 2026 | **Stage 2 waits for Stage 1.** Chris gets one stable set of files, made after the boiler fix (Session 7 item 13) and every other cost item decided in section 3, not the current MP5 files. |
| D-4 | 8 Oct 2026 | Stage 2's spreadsheet deliverable is two sheets only: an updated **README** that explains every change since the 19 Aug 2026 data, and an updated **Data Dictionary** in the same five-column layout as the 19 Aug sheet (`column_name`, `group`, `description`, `units`, `source`). |
| D-5 | 8 Oct 2026 | Stage 2 covers both Tepper scopes Chris has: national and Allegheny County. The 19 Aug 2026 national and Allegheny files come from run `2026-08-19_20-56`. |
| D-6 | 8 Oct 2026 | The email to Chris is drafted in the Claude project chat, from Stage 2's results, as text the researcher edits. |
| D-7 | 8 Oct 2026, earlier | Carried over from Session 7 (decision (S7)): the boiler fix comes first; the default release changes to 2025.1 only after it; the EUSS-to-ResStock rename is one future change, not piecemeal; a 2025.1 run must not enter the grid impact cells. |
| D-8 | 8 Oct 2026 | Pricing rule for the credit for an old heating system: match the equipment type first, then the fuel (section 2.1). |
| D-9 | 9 Oct 2026 | Electric furnaces and electric boilers stay on `electric_baseboard_default`. This takes back the Q3 and Q3b decisions of 8 Oct. Reasons (researcher): leaving them prevents further value moves for now, and the case for the gas rows does not hold up once the efficiency terms are compared. The gas rows price AFUE, ResStock publishes these systems at 100% AFUE, and the baseboard row has no efficiency term. Recorded as CLAUDE.md Limitation 16, with a TODO in the code to revisit after asking NLR's REMDB researchers. |
| D-10 | 8 Oct 2026 | No grid impact guard in this stage (step 2 of section 4 dropped). Grid impact is off in every run of the stage, on both releases: the deliverable does not include the grid impact analysis. |
| D-11 | 8 Oct 2026 | The default release stays 2022.1.1 (step 7 of section 4 dropped). The plan to make 2025.1 the default was a typo: a colleague's analysis uses 2022.1.1. The default is changed by hand only for a 2025.1 run. This replaces the second part of D-7 and item 10 of Session 7. The 17 tests that fail while the default reads 2025.1 have a known cause; fixing them is deferred. |
| D-12 | 8 Oct 2026 | No change may alter the study sample: 221,205 rdu for 2022.1.1 and 161,983 rdu for 2025.1. |

### 2.1 The pricing rule for the credit for an old heating system (D-8)

Decided 8 Oct 2026, from Q2, Q3 and Q3b. **Match the equipment type first, then the
fuel.** Pick the REMDB rows for the old system's type (furnace, boiler, baseboard).
Among them, take the row for its fuel if the table has one; if not, take the gas row
of that type. Every credit is priced at the old system's own size.

| Old system | REMDB row | Efficiency priced at | Type matches | Fuel matches |
|---|---|---|---|---|
| Gas furnace | `furnaces_gas_furnace` | published AFUE, floor 0.80 | yes | yes |
| Propane or fuel oil furnace | `furnaces_gas_furnace` | published AFUE, floor 0.80 | yes | no: no propane or oil furnace row |
| Electric furnace | `furnaces_gas_furnace` | 0.80, fixed | yes | no: no electric furnace row |
| Gas boiler | `boiler_gas_non_condensing` | published AFUE, floor 0.80 | yes | yes |
| Propane boiler | `boiler_gas_non_condensing` | published AFUE, floor 0.80 | yes | no: no propane boiler row |
| Fuel oil boiler | `boiler_oil` | published AFUE, floor 0.80 | yes | yes |
| Electric boiler | `boiler_gas_non_condensing` | 0.80, fixed | yes | no: no electric boiler row |
| Electric baseboard | `electric_baseboard_default` | (row has no efficiency term) | yes | yes |

Changes from today: every boiler row (Q2), and the electric furnace and electric
boiler rows (Q3, Q3b), which today are all priced on `electric_baseboard_default`.
An electric unit is fixed at 0.80 because on a gas row AFUE prices burner
technology, not the unit's own efficiency (about 100% for electric resistance).

**9 Oct 2026 (D-9): the two electric rows of this table were not carried out.**
An electric furnace and an electric boiler stay on `electric_baseboard_default`,
as before. The boiler rows (Q2) are in the code. What the code does:

| Old system | REMDB row | Efficiency priced at |
|---|---|---|
| Gas, propane or fuel oil furnace | `furnaces_gas_furnace` | published AFUE, floor 0.80 |
| Gas or propane boiler | `boiler_gas_non_condensing` | published AFUE, floor 0.80 |
| Fuel oil boiler | `boiler_oil` | published AFUE, floor 0.80 |
| Electric baseboard, electric furnace, electric boiler | `electric_baseboard_default` | (row has no efficiency term) |

---

## 3. Decisions open

Each question gives the options, what each changes, whether it moves a modeled
value and on which release, and a recommendation. The researcher writes the
decision on the "Decision" line. **A Stage 1 item whose question still reads OPEN
is not started: the session asks first.**

Questions 2 and 3 come first, because they change the paper's ResStock 2022.1.1
results.

### Q2 -- Boiler pricing details (Session 7 item 13; the fix itself is decided)

Boilers now on the gas furnace row: 20,973 rdu in 2022.1.1 (gas 16,335, fuel oil
3,983, propane 655) and 2,528 rdu in 2025.1 (gas 2,362, fuel oil 153, propane 13).

| Sub-question | Options | Recommendation |
|---|---|---|
| (a) Row for a propane boiler | gas non-condensing / gas condensing by AFUE; or `boiler_oil` | **Gas boiler rows, chosen by AFUE as for gas.** A propane boiler burns a gaseous fuel in the same kind of appliance as a gas boiler; an oil boiler has a different burner, fuel supply and venting. |
| (b) Split of a gas (and propane) boiler between the two gas rows | condensing row covers AFUE 0.91 to 0.97, non-condensing 0.80 to 0.87; 6.5% of 2022.1.1 boilers and 2.0% of 2025.1 boilers are published at 0.90 or more | **AFUE 0.90 or more: condensing row; below 0.90: non-condensing row.** 0.90 is the usual dividing line for a condensing boiler, and it falls in the gap between the two rows. |
| (c) Lowest efficiency a replacement is priced at | the furnace floor is 0.80 | **Each row's own lower bound** (0.80 for non-condensing and oil, 0.91 for condensing). Values between the floor and a row's upper bound are priced as published, as for furnaces. |

- **Moves values:** yes, on both releases, the paper's 2022.1.1 results among
  them. The credit for the old heating system changes for boiler homes, and with
  it net capital cost, NPV and adoption in the two scopes that take the heating
  credit. The old-AC-only scope does not move.
- **Direction to expect:** the boiler rows carry larger fixed adders than the
  furnace row ($4,040 to $4,924 against $2,218, 2023 dollars, in
  `remdb_v4_tare_retrofit_costs.csv`) and steeper size coefficients, so boiler
  credits are expected to rise and boiler-home adoption to rise. Stage 1 measures
  it before any diff.
- **Decision (researcher, 8 Oct 2026):** every old gas and propane boiler is
  priced on `boiler_gas_non_condensing`, because the systems being replaced are
  older and less efficient; the condensing row is not used for a replacement
  credit. An old fuel oil boiler is priced on `boiler_oil`. The efficiency floor
  is AFUE 0.80, as for furnaces. An old boiler published above the row's top
  value (0.87) is priced as published, carrying the row's line beyond its range,
  as every other cost is today (Q4); the difference is about $23 (2023 dollars)
  per 0.03 of AFUE on that row. This replaces the recommendations of (a) to (c).

### Q3 -- Electric furnace priced on the baseboard row (Session 7 item 14)

47,155 rdu in 2022.1.1 and 7,355 in 2025.1. The baseboard row is a straight line
through zero, so the credit runs from 1,773 to 9,309 dollars between the 5th and
95th percentile (2022.1.1), where a gas furnace's credit runs from 3,228 to 4,279.

| Option | What changes | Moves values |
|---|---|---|
| (a) Keep the baseboard row | nothing | no |
| (b) Price a ducted electric furnace on the gas furnace row, AFUE held at the row floor (0.80) | the credit for electric furnaces follows forced-air furnace prices (cabinet, blower, ducts) and is far less sensitive to size | yes, both releases |
| (c) Keep (a) now; flag in CLAUDE.md as a limitation | nothing | no |

- **Recommendation: (b), decided after Stage 1 measures it.** An electric furnace
  is a forced-air appliance with ducts, like a gas furnace, not a set of room
  heaters. But homes that heat with electricity are most of the no-rebate
  adopters, so the measured change in adoption is shown before the decision.
  Electric baseboard and electric boilers stay on the baseboard row.
- **Decision (researcher, 8 Oct 2026):** an electric furnace is priced as a
  forced-air furnace, not as baseboard: it is the same appliance with a different
  fuel. Neither `remdb_v4_tare_retrofit_costs.csv` nor the full REMDB 2024 table
  (Q12) has an electric furnace row, so it is priced on `furnaces_gas_furnace` with
  AFUE held at the row's floor, 0.80: the row's AFUE term prices gas-combustion
  efficiency, which an electric furnace does not have. Stage 1 measures the effect
  (Task 3) before the diff. Electric baseboard stays on the baseboard row. Electric
  boilers: see Q3b.
- **No adjustment to the gas row's price** (researcher, 8 Oct 2026). Every cost
  stays a REMDB value, as decision (S7) sets ("the closest fit available"). The
  difference is written into CLAUDE.md's limitations instead (Stage 1, Task 8):
  the gas furnace row prices a burner, heat exchanger, flue and gas line that an
  electric furnace does not have, so the credit for an old electric furnace is
  likely somewhat high, and so is the adoption of the homes that have one. Pricing
  at the row's floor (AFUE 0.80, its least expensive point) takes out the part of
  the price that rises with combustion efficiency, but not the venting and gas
  line in its fixed parts. A panel upgrade does not apply: the credit prices
  replacing the existing furnace, on a panel that already carries it.
- **Changed 9 Oct 2026 (researcher), D-9:** leave electric furnaces on the
  baseboard row. An electric furnace is published at 100% AFUE, so on the gas
  furnace row it would be priced at 1.00 unless it were held at 0.80 by a step of
  its own (1.00 against 0.80 is $865 in 2023 dollars); the baseboard row has no
  efficiency term at all. With the efficiency terms that far apart, the case for
  the gas row does not hold up well enough to move values. The discrepancy is
  documented (CLAUDE.md Limitation 16) and a TODO in the code says to revisit it
  after asking NLR's REMDB researchers. No code change, no value moved.

### Q3b -- Electric boiler (follows from Q3)

Today an electric boiler is priced on the baseboard row, like an electric furnace.
By the reasoning of Q3 (same appliance, different fuel), it would be priced on
`boiler_gas_non_condensing` at the floor, 0.80. REMDB 2024 has no electric boiler
row (Q12). Count to be measured in Stage 1's Task 3.

- **Recommendation:** treat it as Q3 does: price it as a boiler.
- **Decision (researcher, 8 Oct 2026):** an electric boiler is priced on
  `boiler_gas_non_condensing` at AFUE 0.80, with the same limitation as Q3 written
  into CLAUDE.md (the gas row prices venting and a gas line an electric boiler does
  not have). Electric baseboard is now the only system on the baseboard row.
- **Changed 9 Oct 2026 (researcher), D-9:** not carried out, for the reasons given
  under Q3. Electric boilers stay on the baseboard row (265 rdu in 2022.1.1, 48
  in 2025.1). On the gas boiler row, 1.00 against 0.80 is $155 in 2023 dollars.

### Q12 -- Which REMDB edition the costs come from

NREL's OEDI submission 8336 (https://data.openei.org/submissions/8336) holds
`REMDB 2024.xlsx` (file dated 2024.12.23; submission last updated 7 Mar 2025;
data dated 29 Sep 2023) with two raw measure workbooks and a guidance document.
The model's `remdb_v4_tare_retrofit_costs.csv` is a cut of a REMDB table in 2023
dollars. Not yet known: whether that is the same edition as `REMDB 2024.xlsx`,
and whether the 2024 file has electric furnace or electric boiler rows.

| Option | Moves values |
|---|---|
| (a) Same edition: take any missing rows (Q3, Q3b) from it; coefficients unchanged | only through Q3 and Q3b |
| (b) Newer edition with different coefficients: adopt it as its own decision and commit | yes, every cost, both releases |
| (c) Newer edition: keep the current table for Stage 1; record the newer one for later | no |

- **Recommendation:** compare first, then (a) if the coefficients match.
- **Checked 8 Oct 2026** (`REMDB_2024.12.23_1.xlsx`, sheet `Machine Read`, 133
  rows): all 32 rows of `remdb_v4_tare_retrofit_costs.csv` are the rows of the same
  position in REMDB 2024, with the same names and the same values in all 16 numeric
  fields (three coefficients and two bounds for each metric, three intercepts, the
  retrofit multiplier and adder, the lifetime). The current table is a cut of REMDB
  2024. Its only heating row not in the cut is `Ground Source Heat Pump`. It has no
  electric furnace and no electric boiler row. It does have two `Electric Panel`
  rows (100A, 200A), not in the cut (see Q6).
- **Decision (8 Oct 2026):** (a). Same edition; nothing to adopt; no value moves.

### Q1 -- Do homes that heat with electricity stay in the dual-fuel sample? (Session 7 item 9)

| Option | What changes | Moves values |
|---|---|---|
| (a) Keep them; report MP5 results by baseline fuel, with gas homes as the headline | nothing in the model | no |
| (b) Limit the MP5 sample to fossil-heated homes | MP5 sample loses 8,778 rdu; MP5 adoption falls (they are 61.52% of no-rebate adopters) | yes, 2025.1 only |

- **Recommendation: (a).** ResStock applies the package to these homes, and
  keeping one sample rule for every package keeps the funnel comparable. Report
  them separately, since a gas backup in an electric-heated home adds a fossil
  appliance. Revisit for the paper.
- **Decision (researcher, 8 Oct 2026):** (a). Keep them. The study sample set on
  5 to 7 Oct 2026 is not changed. Results are reported by baseline fuel. Session 7
  item 9 is closed with no code change.

### Q4 -- Sizes outside a cost row's range (Session 7 item 16)

Priced today by carrying the row's line beyond the range (CLAUDE.md Limitation 9).
Counts are in item 16 of Session 7 (for example 19,994 rdu of 2022.1.1 below the
gas furnace row's 30,000 Btu/h; 32,084 rdu below the central AC row's 1.5 tons).

| Option | Moves values |
|---|---|
| (a) Keep, and write the counts into CLAUDE.md Limitation 9 | no |
| (b) Hold at the bound | yes, both releases, tens of thousands of rdu |
| (c) Keep, and add a flag column | no, but the results header changes |

- **Recommendation: (a)** for now. Holding at the bound would itself be an
  assumption about how cost behaves outside the data, and it moves many values at
  once. Revisit for the paper.
- **Decision (researcher, 8 Oct 2026):** (a). No change to the code's logic. The
  existing assumption (a size outside a row's range is priced by carrying the
  row's line beyond it) and the counts of item 16 are written into CLAUDE.md
  Limitation 9 in Stage 1, Task 8.

### Q5 -- Efficiency floor for a room AC credit (Session 7 item 17)

2,768 rdu (2022.1.1) and 325 (2025.1) are priced below the row's lowest value, 9.4;
about 6 dollars per point.

- **Recommendation: no change now; document in CLAUDE.md** (the floor, and that the
  row takes CEER where ResStock gives EER). The effect is small. Boiler floors are
  set with Q2(c).
- **Decision (8 Oct 2026, inferred from the Q4 answer; to be confirmed by the
  researcher):** no change to the code; document in CLAUDE.md that a room AC credit
  has no efficiency floor, that 2,768 rdu (2022.1.1) and 325 (2025.1) are priced
  below the row's lowest value, and that the row takes CEER where ResStock gives
  EER.
- **Confirmed (researcher, 8 Oct 2026):** do not change; document only. Recorded
  as CLAUDE.md Limitation 17.

### Q6 -- The two unused heat pump rows (Session 7 item 18)

`air_source_heat_pump_centrally_ducted_with_new_circuit` and
`air_source_heat_pump_non_ducted_single_zone`.

- **Recommendation: later.** Choosing either needs a rule (which homes need a new
  circuit; which non-ducted homes are single-zone) that the data do not give
  directly, and the 2025.1 panel columns describe panel limits, not circuits. No
  change in Stage 1. For later: REMDB 2024 has `Electric Panel` rows (100A, 200A)
  that the cut left out; with the 2025.1 panel columns (18,020 MP5 sample rdu
  capacity-constrained) they would let a panel upgrade be priced.
- **Decision (researcher, 8 Oct 2026):** no change. No panel upgrade is assumed
  and none is priced; the two unused heat pump rows stay unused. Both are written
  into CLAUDE.md's limitations in Stage 1, Task 8.

### Q7, Q8 -- `_euss` result columns and `process_euss_data.py` (Session 7 items 6, 7)

- **Recommendation: later, with the EUSS-to-ResStock rename (D-7).** Neither is in
  a Tepper file, so Chris's files do not depend on them.
- **Decision (researcher, 8 Oct 2026):** DEFERRED. Not part of Stage 1.

### Q9 -- The AWS table for 2025.1 timeseries (Session 7 item 2)

- **Recommendation: later.** Stage 1 adds only the grid impact guard (Session 7
  item 1).
- **Decision (researcher, 8 Oct 2026):** DEFERRED. The guard is not added either
  (D-10): grid impact is switched off for every run of this stage.

### Q10 -- Which packages Chris receives

| Option | Files |
|---|---|
| (a) MP5 only | national and Allegheny, MP5 |
| (b) MP3 and MP5 | national and Allegheny, MP3 (refreshed) and MP5 |
| (c) MP3, MP4 and MP5 | everything the two releases produce |

- **Recommendation: (c).** His 19 Aug MP3 and MP4 files are out of date in every
  value column (section 6), so leaving him with them invites mixing old and new
  numbers. MP3 is also the comparison point for MP5.
- **Decision (researcher, 8 Oct 2026):** Chris receives the updated outputs of
  the re-run (Stage 1, Task 9): every package it produces, so (c), MP3, MP4 and
  MP5, national and Allegheny.

### Q11 -- Shape of the Stage 2 comparison

| Option | What it shows |
|---|---|
| (a) 19 Aug MP3 against new MP5 | the request as written; mixes model changes with the change of package and release |
| (b) 19 Aug MP3, then new MP3, then new MP5 | the first step is the effect of the model changes on the same homes (matched by `bldg_id`, same release); the second is the change of package and release, compared by group only |

- **Recommendation: (b).** It lets the README say how much of the change Chris sees
  comes from fixes and how much from dual fuel.
- **Decision (researcher, 8 Oct 2026):** (b). Old MP3, then new MP3, then new
  MP5.

---

## 4. Stage 1 -- cleanup and cost fixes

Prompt: `DF_STAGE1_SESSION_PROMPT.md`. Scratch files in `~/tare_port` take the
prefix `s8_`. Runs follow section 3.1 of `LOCAL_PORT_SESSION_PLAN.md` (runner,
logs, `--skip-grid-impact` for every 2025.1 run, test command of decision (T),
commit messages of decision (C)).

| Step | Work | Session 7 item | Moves values | Commit |
|---:|---|---|---|---|
| 1 | Audit, no edits | -- | no | -- |
| 2 | Grid impact guard: the grid impact cells stop with a clear message on a release the analysis does not support | 1 | no | own |
| 3 | Measure every open cost option on the saved national results, no edits | 13, 14, 16, 17, 18 | no | -- |
| 4 | Boiler fix, as decided in Q2 | 13 | **yes, both releases** | own, `VALUE-MOVING` |
| 5 | Electric furnace and electric boiler pricing (Q3, Q3b) | 14 | **yes, both releases** | own, `VALUE-MOVING` |
| 6 | Closed: Q1 keeps the sample; Q4, Q5, Q6 change no code (documented in step 8) | 9, 16, 17, 18 | no | -- |
| 7 | Default release to 2025.1, with the tests that assume 2022.1.1 made to name their release | 10 | no | own |
| 8 | Records: CLAUDE.md limitations (item 15; Q3 and Q3b gas-row pricing of electric furnaces and boilers; Q4 out-of-range sizes in Limitation 9; Q5 room AC floor and CEER; Q6 no panel upgrade, unused heat pump rows) and "Last updated"; new REFERENCE_VALUES rows; SESSION_LOG entry; Tepper data dictionary; this plan's sections 6 and 7 | 15, 16, 17, 18 | no | own |
| 9 | Final national runs on both releases, with the Tepper exports (national and Allegheny) for Stage 2 | -- | no | -- |

**As carried out, 8 to 9 Oct 2026.** The table above is the plan as written on
8 Oct. What was done:

| Step | Outcome |
|---:|---|
| 1 | Done. Tree clean after the three plan documents were committed; 399 passed, 1 skipped. |
| 2 | Dropped (D-10). No guard; grid impact off in every run. |
| 3 | Done for the boiler decision (Q2). Electric furnaces were measured both ways and then left as they are (D-9). |
| 4 | Done. Three files: the row choice, the two boiler floors, 18 tests. `VALUE-MOVING` on both releases. |
| 5 | Dropped (D-9). A TODO in the code and CLAUDE.md Limitation 16 instead. |
| 6 | Nothing to change, as planned. |
| 7 | Dropped (D-11). The default release stays 2022.1.1. |
| 8 | Done: `REFERENCE_VALUES.md` (17 rows), CLAUDE.md (Limitation 9, Limitations 15 to 18, "Last updated"), `SESSION_LOG.md`, the Tepper data dictionary, and this file. |
| 9 | The national runs of step 4 serve as the final runs (section 7.1): they were made on the code as it is to be committed, and step 8 changed no code. The Tepper checks of sections 13 and 14.4 of the data dictionary pass on them. |

**Size of the work.** Each code step is a small edit to existing code: step 2 a short
function and its call in the notebook; step 4 the replacement row mapping and the floor
table for boilers; step 5 the same mapping and a fixed AFUE for electric furnaces and
boilers; step 7 one constant and the tests that assumed it. No step adds a module, a new
pricing function, or a refactor of code it does not need to touch.

**Checks.** After each commit that touches code: a Pennsylvania run on each release,
identical to the run before it apart from what the step changes on purpose. For a
value-moving step: only the columns that step names, and the columns computed
from them, differ, and only on the rows it names. A national run on each release
closes each value-moving step. Test count starts at 399 passed, 1 skipped, and
changes only by the tests a step adds.

**Stage 1 is done when:** Q5 is confirmed; every taken item is
committed by the researcher; the final national runs' timestamps are in section 7;
REFERENCE_VALUES holds new rows for them; the Tepper exports exist for every
package of Q10.

**Status, 9 Oct 2026:** Q5 is confirmed; the final runs' timestamps are in
section 7.1; REFERENCE_VALUES holds their rows; the Tepper exports exist for MP3,
MP4 and MP5, national and Allegheny. Still open: the researcher's commits. The
code and the records are in the working tree, not yet committed.

**Update, later on 9 Oct 2026:** the researcher committed the code as `5a91996` and the
records as `0624802`, both at 01:58, after the final runs. Stage 1 is done. Stage 2 is
recorded in commit `6a12e05`.

---

## 5. Stage 2 -- data validation and the data share

Prompt: `DF_STAGE2_SESSION_PROMPT.md`. Starts only when Stage 1 is done (D-3).
Scratch files take the prefix `s9_`.

**Inputs.**

- 19 Aug 2026: run `2026-08-19_20-56` (ResStock 2022.1.1, MP3 and MP4), its Tepper
  household and county files, national and Allegheny, the `source_data/` CSVs sent
  with them, and the workbook `TARE_Export_Tepper_Cashflow_Model_JMJ_DRAFT.xlsx`
  (sheets README, Data Dictionary, Fuel Price Data, Projection Factors, TARE MP3
  Data). **Corrected 9 Oct 2026:** the 19 Aug Tepper files are not in
  `cmu_tare_model/output_results/tepper_export/`. That folder holds the files of
  the runs made from 6 Oct 2026 on, told apart by run stamp. The files as sent to
  Chris (two national and two Allegheny household files, two national county
  files; no Allegheny county file was sent), the two source CSVs sent with them
  and the workbook are in `~/tare_port/s9_inputs/`, outside the repo, copied there
  by the researcher on 9 Oct 2026.
- Stage 1's final runs (section 7) and their Tepper files, which are in
  `tepper_export/`.

**What the 19 Aug files hold** (read from the Allegheny files on 8 Oct 2026): 155
columns; 1,356 rdu in Allegheny (328,330 homes), which is every home that passed
the heating check, including 209 rdu with no AC, whose cooling replacement cost is
blank; no `include_sample` column. MP3 no-rebate adoption in that file is 18.88%
of 1,356 rdu; MP4 9.22%. National, the run's no-rebate adoption was 27.1760% (MP3)
and 18.0984% (MP4) of 260,211 rdu (`REFERENCE_VALUES.md`).

**Work.**

1. Audit: locate every input; row and column counts; header differences between
   the 19 Aug files and the new ones.
2. Comparison, national and Allegheny, in the shape of Q11: adoption of the
   shipped case (`heatingLCC_coolingLCC_unsub`) for all homes and by baseline
   fuel; and the parts of the adoption decision: credit for the old heating
   system, credit for the old AC, heat pump installed cost, backup furnace
   installed cost (MP5), net capital cost, discounted heating and cooling
   savings, NPV. Means and medians, counts in rdu and homes. The 19 Aug files are
   shown twice: as shipped, and limited to homes with a central or room AC, so
   that the populations compare.
3. Validation of the new files, as in sections 13 and 14.4 of the Tepper data
   dictionary, and the three `source_data/` CSVs compared with the 19 Aug copies.
4. The README and Data Dictionary sheets (D-4).
5. A facts file for the email (D-6).

**Stage 2 is done when:** the comparison tables and the validation report are
saved; the two sheets are in a workbook the researcher has reviewed; the facts
file is in the Claude project chat.

---

## 6. Changes since the 19 Aug 2026 data (source for the README)

Run `2026-08-19_20-56` is the run behind Chris's files and the paper submission.
Every change below is recorded in `SESSION_LOG.md`, `REFERENCE_VALUES.md` or
CLAUDE.md.

| Date | Change | Moved values | Release |
|---|---|---|---|
| 2 to 3 Sep 2026 | Notebook cleanup; run `2026-09-02_19-04` byte-identical to `2026-08-19_20-56` | no | 2022.1.1 |
| 22 to 23 Sep 2026 | **Consumption fix** (commit 5ff4d4f): heating and cooling energy count fans, pumps and heat-pump backup heat on both sides, and each fuel is priced at its own price. The 19 Aug run left out MP3 and MP4's electric backup heat and every system's fans. Also: household income drawn per home, seeded by `bldg_id`; 5 rdu with shared heating dropped. | **yes**: MP3 no-rebate adoption 27.18% to 16.66%, MP4 18.10% to 17.03% (260,211 rdu) | 2022.1.1 |
| 25 Sep 2026 | **Study sample flag** `include_sample`: every result covers one set of homes | cooling results of out-of-sample homes only | 2022.1.1 |
| 2 Oct 2026 | **County and state adoption rates** count study-sample homes only (commit e4b670d). In the 19 Aug county files `adoption_rate_pct` is adopters over every rdu of the county, a blank flag counted as a non-adopter, and `home_count` is every rdu times the weight. Row added 9 Oct 2026: Stage 2 found this difference had no row of its own | yes: county and state rates; national rates do not move | 2022.1.1 |
| 3 Oct 2026 | **A true zero stays a zero** (commit cf43350). In the 19 Aug files `baseline_heating_consumption` is blank where the home used no energy (1,020 rdu of today's sample), and `baseline_cooling_consumption` on 7 rdu; both are 0 now. Row added 9 Oct 2026, for the same reason | no result moved | 2022.1.1 |
| 5 Oct 2026 | **Study sample narrowed** to homes with a central or room AC of their own (260,211 to 221,205 rdu); retrofit energy and the home table no longer rounded during the run; climate damages in 2025 dollars | yes: MP3 no-rebate adoption 18.81%, MP4 18.99% (221,205 rdu) | 2022.1.1 |
| 6 Oct 2026 | **Tepper files**: study-sample rows only (1,146 rdu in Allegheny, 221,205 national), ship `include_sample`, gain the base-year fan, pump and backup columns (167 columns), and a detailed copy splits each year by fuel (317 columns) | no | 2022.1.1 |
| 7 Oct 2026 | **ResStock 2025.1 MP5, dual fuel**: heat pump priced at SEER1 16.0; backup gas furnace priced and added to the capital cost; June 2026 rebate equals 2024 for MP5; winter and summer peak names; four electric panel columns; MP5 Tepper files hold 182 and 332 columns (1,091 rdu in Allegheny, 161,983 national) | new package | 2025.1 |
| 9 Oct 2026 | **Boiler pricing** (Stage 1): the credit for an old natural gas or propane boiler is priced on the REMDB non-condensing gas boiler row, and for an old fuel oil boiler on the oil boiler row, with a floor of AFUE 0.80. Until then they were priced as gas furnaces. 20,973 rdu (5,078,214 homes) in 2022.1.1 and 2,528 rdu (641,868 homes) in 2025.1; their mean credit goes from $3,473.72 to $7,638.08 and from $3,442.91 to $7,766.02. Nothing else decided in section 3 changed the code: electric furnaces and electric boilers stay on the baseboard row (D-9). | yes: no-rebate adoption with both credits, MP3 18.81% to 19.35%, MP4 18.99% to 19.47%, MP5 3.27% to 3.39% (section 7.2). In the Tepper files only homes with a fossil boiler differ, in the heating credit, the net capital cost, the NPV and the adopter flag | both |

**The commit and the run behind each row** (added 9 Oct 2026, from the change audit in
`~/tare_port/s11_outputs/`). A session's date and its commit's date can differ by a day
or two: a run was usually made just before its commit. Run `2026-08-19_20-56` itself
was made on commit `68f0964` plus the old-system size fix, before `2f54ed2` existed.

| Row above | Commits | Run that shows the effect |
|---|---|---|
| Before the 19 Aug data: the credit is priced at the old system's own size | `2f54ed2`, committed 28 Aug | `2026-08-19_20-56` against `2026-08-19_13-19` |
| 22 to 23 Sep, consumption fix and income per home | `5ff4d4f`, committed 24 Sep; `3ea1a40` | `2026-09-23_15-34` against `2026-08-19_20-56` |
| 25 Sep, study sample flag | `ef849cc`, committed 26 Sep | `2026-09-25_22-55` against `2026-09-23_15-34` |
| 2 Oct, county and state rates | `e4b670d` | no run of its own; the figures are in the commit message |
| 3 Oct, zeros; 5 Oct, no rounding during the run | `cf43350`, `c35eacf` | `2026-10-05_00-54` against `2026-10-04_23-02` |
| 5 Oct, study sample narrowed | `55848a1` (2 Oct, homes with no AC), `6d82f3b` (3 Oct, shared cooling) | `2026-09-30_22-35` against `2026-09-30_19-06`; `2026-10-05_00-54` |
| 5 Oct, climate damages in 2025 dollars | `43a8ef9` | `2026-10-05_21-33` against `2026-10-05_00-54` |
| 6 Oct, Tepper files | `ed56dc3`, committed 5 Oct | checked on `2026-10-05_21-33` |
| 7 Oct, ResStock 2025.1 MP5 | `6e5c549` to `3944048` | `2026-10-07_19-14`; `2026-10-07_19-22` is byte-identical to `2026-10-06_20-23` |
| 9 Oct, boiler pricing | `5a91996` | `2026-10-09_00-46` and `2026-10-09_00-38` against `2026-10-07_19-22` and `2026-10-07_19-14` |

---

## 7. Log

Newest entry last. Each entry: date, who, what was decided or measured, run
timestamps, and where the detail is.

| Date | Entry |
|---|---|
| 8 Oct 2026 | Plan written in the Claude project chat. D-1 to D-7 taken. Q1 to Q11 open, with recommendations. Results of record in section 1. |
| 8 Oct 2026 | Researcher: the Tepper CSVs, including those of `2026-08-19_20-56`, are in `cmu_tare_model/output_results/tepper_export/`. Section 5 and the Stage 2 prompt updated; only the workbook goes in `~/tare_port/s9_inputs/`. |
| 8 Oct 2026 | Q2 decided: gas and propane boilers on `boiler_gas_non_condensing`, fuel oil boilers on `boiler_oil`, floor AFUE 0.80, values above 0.87 carried beyond the row as published. |
| 8 Oct 2026 | Q3 decided: an electric furnace is priced as a furnace (an electric furnace row of the full REMDB v4 table if one exists, otherwise the gas furnace row at AFUE 0.80). Q3b (electric boiler) added, open. |
| 8 Oct 2026 | Researcher found an updated REMDB release (OEDI submission 8336, `REMDB 2024.xlsx`). Q12 added, open. The file could not be fetched from this chat (the site is blocked here); the researcher is to attach it for a row-by-row comparison. |
| 8 Oct 2026 | REMDB 2024 attached and compared: the current table is an exact cut of it (32 rows, all values equal). Q12 decided (a). No electric furnace or electric boiler row exists, so Q3 uses the gas furnace row at AFUE 0.80. Panel rows noted under Q6 for later. |
| 8 Oct 2026 | Q3b decided: electric boilers on `boiler_gas_non_condensing` at AFUE 0.80. Q3 and Q3b: no price adjustment to the gas rows; the difference (venting, gas line) is documented in CLAUDE.md's limitations. |
| 8 Oct 2026 | Q1 (keep the sample), Q4 (no change, document), Q6 (no panel upgrade, no change, document), Q10 (MP3, MP4 and MP5 from the re-run) and Q11 (three-step comparison) decided. Q5 recorded as no change and document, inferred from the Q4 answer, to be confirmed. Stage 1 value-moving work is now the boiler fix (Q2) and the electric furnace and boiler pricing (Q3, Q3b) only. Q7 to Q9 stay open and outside Stage 1. |
| 8 Oct 2026 | Stage 1 session, audit (step 1). Researcher: Q5 confirmed (document only); Q7, Q8 and Q9 deferred; no change may alter the study sample (D-12); no grid impact guard, grid impact off in every run (D-10); the default release stays 2022.1.1, since making 2025.1 the default was a typo (D-11). Found in the audit: 17 tests in five files assume 2022.1.1 when the default reads 2025.1; cause known, fix deferred. |
| 9 Oct 2026 | Researcher: electric furnaces and electric boilers stay on the baseboard row (D-9). An electric furnace is published at 100% AFUE; on the gas furnace row that is $865 (2023 dollars) above the price at 0.80, and the baseboard row has no efficiency term. Document the discrepancy and leave a TODO to ask NLR's REMDB researchers. |
| 9 Oct 2026 | Step 3, measured on the saved national runs before any edit (`~/tare_port/s8_task3_boiler_credit.py`): the boiler decision touches 20,973 rdu (2022.1.1) and 2,528 rdu (2025.1); the credit rises on every one; no rebate changes, because a rebate is worked out from the heat pump's installed cost only, so the NPV identity is exact for the rebated cases too. Corrected the same day after the researcher's question: with size held fixed, a 90% AFUE oil boiler is credited $110 less (2025 dollars) than an 80% one, not $570; the $570 was the difference in how much the two credits rise. The model's oil boiler row gives the three prices of NLR's website for 221,500 Btu/h at AFUE 0.875 exactly ($8,309.32, $9,295.97, $11,092.38). |
| 9 Oct 2026 | Step 4, boiler fix made and run. Pennsylvania and national runs on both releases (section 7.1), grid impact off. Every comparison passed: only fossil boiler rdu differ, in 20 columns; new NPV = old NPV + change in credit to the cent; study sample unchanged; the national credits and the adoption of all nine cases equal step 3's figures. Tests 417 passed, 1 skipped (18 new). Results in section 7.2. |
| 9 Oct 2026 | Steps 8 and 9, records and Tepper checks, made with the researcher away and every edit auto-approved for the session at the researcher's request. The national runs of step 4 are used as the final runs; the researcher may ask for a re-run after committing. The Tepper checks of sections 13 and 14.4 of the data dictionary pass on them (25 checks per package). Two things seen in those checks, both the same in the runs before the fix and neither a fault in the files: `occupancy` reads as text in the national package 5 file and as numbers in the Allegheny one; and a number in a Tepper file can differ from the same number in the results file in its last digits (relative difference up to 7.5e-13). Also noted: the county Tepper file is sorted by adoption rate, and counties with equal rates can change places between runs, so those files are compared county by county. Nothing is committed yet. |
| 9 Oct 2026 | Stage 2 session, with every gate auto-approved until the commit point at the researcher's request. No model code changed, no model was run and no value moved. The files as sent to Chris on 19 Aug 2026 are in `~/tare_port/s9_inputs/`, not in `tepper_export/` (section 5 corrected). Comparison, validation, the two sheets and the facts file are done; results in section 7.3, files in `~/tare_port/s9_outputs/`, scripts `~/tare_port/s9_*.py`. Validation: the checks of sections 13 and 14.4 of the Tepper data dictionary, with two added for the Allegheny county file, pass on all 18 new files (27 per package); the three `source_data/` CSVs equal the model's inputs, and the two Chris has are unchanged byte for byte. Two differences between the old and the new files have no row of their own in section 6: what the county file's `home_count` and `adoption_rate_pct` are measured over (the nearest row is 25 Sep 2026), and a zero that was a blank in `baseline_heating_consumption` (1,020 rdu of today's sample) and `baseline_cooling_consumption` (7 rdu), which the hand-over note of 2 Oct 2026 records. Neither contradicts the plan, and section 6 is left as it is. Still open: the researcher's review of the workbook, and the email (D-6). |
| 9 Oct 2026 | Change audit and writeup session, with each gate approved by the researcher. No model code changed, no model was run and no value moved. Found: no commit is exactly the code of run `2026-08-19_20-56`. It was made on `68f0964` plus the old-system size fix, first committed as `2f54ed2` on 28 Aug. From there to `6a12e05` are 69 commits, 14 of them value-moving. Stage 1 is committed as `5a91996` (code) and `0624802` (records), so the status line of section 4 has an update. Section 6 gains two rows (county rates, and a zero that was a blank) and a table of the commit and the run behind each row. Section 7.4 holds the nine cases of Stage 2's follow-up, which were only in `~/tare_port/s9_outputs/`. Q5 was already recorded as confirmed (8 Oct). `SESSION_LOG.md` gains seven rows. Researcher: `GRID_IMPACT_ANALYSIS = False` stays uncommitted for now, because the dual-fuel analysis for Chris does not need grid impact; a separate ResStock 2022.1.1 run with it on is planned for the paper. Documents: the change audit and the detailed writeup, in `~/tare_port/s11_outputs/`. |

### 7.1 Runs

| Run | Release | Scope | Purpose | Result |
|---|---|---|---|---|
| `2026-08-19_20-56` | 2022.1.1 | National | Data sent to Chris on 19 Aug 2026 | Reference for Stage 2 |
| `2026-10-07_19-22` | 2022.1.1 | National | Results of record, MP3 and MP4 | Section 1 |
| `2026-10-07_19-14` | 2025.1 | National | Results of record, MP5 | Section 1 |
| `2026-10-08_01-51` | 2022.1.1 | Pennsylvania | Reference for Stage 1's Pennsylvania comparisons | -- |
| `2026-10-08_01-49` | 2025.1 | Pennsylvania | Reference for Stage 1's Pennsylvania comparisons | -- |
| `2026-10-09_00-29` | 2022.1.1 | Pennsylvania | After the boiler fix | Exit 0, 40 runner checks passed; differs from `2026-10-08_01-51` only on fossil boiler rdu (1,816 in the sample) |
| `2026-10-09_00-28` | 2025.1 | Pennsylvania | After the boiler fix | Exit 0, 20 runner checks passed; differs from `2026-10-08_01-49` only on fossil boiler rdu (203 in the sample) |
| `2026-10-09_00-46` | 2022.1.1 | National | After the boiler fix; **final run of Stage 1 for MP3 and MP4**, with the Tepper files, national and Allegheny | Exit 0, 40 runner checks passed; section 7.2 |
| `2026-10-09_00-38` | 2025.1 | National | After the boiler fix; **final run of Stage 1 for MP5**, with the Tepper files, national and Allegheny | Exit 0, 20 runner checks passed; section 7.2 |

Every run of 9 Oct was made with grid impact off, as the reference runs were. The
final runs were made on the working tree, before the researcher's commit. The only
change to a code file since they started is the researcher's
`GRID_IMPACT_ANALYSIS = False` in `constants.py`, which changes no result: both
runs already had grid impact off through the runner.

**Update, later on 9 Oct 2026:** the Stage 1 code is commit `5a91996`, made after both
final runs. The `GRID_IMPACT_ANALYSIS = False` line is still uncommitted, on purpose;
the committed value is True.

### 7.2 Results after Stage 1

Runs `2026-10-09_00-46` (2022.1.1, MP3 and MP4: 221,205 rdu) and `2026-10-09_00-38`
(2025.1, MP5: 161,983 rdu). National, 7% discount rate, USD2025. They agree with
the rows added to `REFERENCE_VALUES.md` on 9 Oct 2026. Section 1 holds the same
tables before Stage 1.

Adoption, percent of the study sample (no rebate / 2024 guidance / June 2026 guidance):

| Credit scope | MP3 | MP4 | MP5 | Moved in Stage 1 |
|---|---|---|---|---|
| Old AC credit only (`heatingSavings_coolingLCC`) | 9.19 / 32.11 / 18.94 | 10.70 / 29.01 / 18.99 | 1.78 / 6.30 / 6.30 | no |
| Old heating credit only (`heatingLCC_coolingSavings`) | 6.84 / 20.46 / 13.82 | 8.71 / 22.01 / 15.12 | 1.38 / 3.28 / 3.28 | yes |
| Both credits (`heatingLCC_coolingLCC`) | 19.35 / 46.64 / 26.88 | 19.47 / 49.37 / 28.89 | 3.39 / 29.48 / 29.48 | yes |

MP5, mean dollars per home (both credits):

| Part | Before Stage 1 | After Stage 1 |
|---|---|---|
| Heat pump | 15,711.23 | 15,711.23 |
| Backup gas furnace | 4,112.21 | 4,112.21 |
| Credit for the old heating system | 3,742.25 | 3,809.72 |
| Credit for the old AC | 5,842.98 | 5,842.98 |
| Net capital cost, no rebate | 10,238.21 (median 9,548.51) | 10,170.74 (median 9,508.95) |
| Bill savings, 15 years, discounted | 1,535.17 | 1,535.17 |
| NPV, no rebate | -8,703.04 (median -7,941.37) | -8,635.57 (median -7,906.24) |

Credit for the old heating system, 2022.1.1 (the same for MP3 and MP4): mean
3,887.87 before and 4,282.71 after; for the 20,973 rdu with a fossil boiler,
3,473.72 before and 7,638.08 after.

What section 1.3 says about MP5, after Stage 1:

- No-rebate adoption of natural gas homes: 0.93% (MP5), 5.33% (MP3). It was
  0.82% and 4.61%.
- The backup furnace costs 302 more on average than the credit for the old
  heating system. It was 370.
- Homes that heat with electricity are still 5.42% of the MP5 sample (8,778 rdu)
  and are 59.30% of its no-rebate adopters (3,257 of 5,492 rdu). They were
  61.52% (3,257 of 5,294).

### 7.3 Results of Stage 2

The files as sent to Chris on 19 Aug 2026 (run `2026-08-19_20-56`) against the
files of the final runs of section 7.1, in the shape of Q11. The case is the one
the files ship (`heatingLCC_coolingLCC_unsub`), 7% discount rate, USD2025. The
detail is in `~/tare_port/s9_outputs/s9_comparison_summary.md` and its tables.

No-rebate adoption. "Today's study sample" is the 19 Aug data limited to the rdu
of the new file, the homes with a result and a central or room AC of their own:

| File and view | National | Allegheny |
|---|---|---|
| 19 Aug MP3, rdu with a result | 27.18% (70,715 of 260,211 rdu) | 18.88% (256 of 1,356 rdu) |
| 19 Aug MP3, today's study sample | 30.18% (66,765 of 221,205) | 21.90% (251 of 1,146) |
| New MP3 | 19.35% (42,797 of 221,205) | 3.93% (45 of 1,146) |
| New MP5 (other homes; compared by group only) | 3.39% (5,492 of 161,983) | 0.64% (7 of 1,091) |
| 19 Aug MP4, today's study sample | 19.71% (43,607 of 221,205) | 10.30% (118 of 1,146) |
| New MP4 | 19.47% (43,063 of 221,205) | 5.93% (68 of 1,146) |

The 19 Aug national file as shipped holds 331,531 rdu, 71,320 of them with no
result; its 70,715 adopters are 21.33% of all its rows.

Old MP3 to new MP3, national, the same 221,205 rdu matched by `bldg_id`:

- Adopters: 66,765 in the 19 Aug data; 41,606 with today's savings and the old
  heating credit (the consumption fix of 22 to 23 Sep); 42,797 with the boiler
  pricing of 9 Oct.
- 25,614 rdu (6,201,944 homes) stop adopting and 1,646 rdu (398,548 homes)
  start. Those that stop lose 20,902 of discounted heating savings on average,
  and for 25,496 of them it is the part that moved most. For 935 of those that
  start it is the heating credit, and for 707 the cooling savings.
- Mean per home: discounted heating savings 5,138.59 to -1,376.79; discounted
  cooling savings 555.24 to 500.02; credit for the old heating system 3,887.87
  to 4,282.71; net capital cost 9,385.80 to 8,990.96; NPV -3,691.96 to
  -9,867.74.
- The heat pump's installed cost (mean 18,580.45) and the credit for the old AC
  (mean 5,306.78) are the same on every rdu. The heating credit differs on the
  20,973 rdu with a fossil boiler and on no other.
- Allegheny: 206 rdu stop and none start; 189 of the 206 heat with natural gas,
  and no natural gas home adopts MP3 in the new file (0 of 1,050 rdu).

The national county files, matched by `county`:

- 3,098 counties on 19 Aug and 3,079 now, for MP3 and MP4 (2,973 for MP5). The
  19 that left hold no rdu of today's study sample.
- On 19 Aug `adoption_rate_pct` is adopters over every rdu of the county, a
  blank flag counted as a non-adopter, in all 3,098 counties, and `home_count`
  is every rdu times 242.13 (80,273,601 homes in all). Both are now measured
  over the study sample (53,560,591 homes).
- County rate, MP3, median over counties: 27.45% as shipped, 37.93% over rdu
  with a flag (3,090 counties have one), 23.57% in the new file. Allegheny:
  15.90%, 18.88% and 3.93%.

Files written, all in `~/tare_port/s9_outputs/`:

| File | Holds |
|---|---|
| `s9_comparison_summary.md`, `s9_t2_*.csv` | the comparison and its tables |
| `s9_validation_report.md` | the checks on the 18 new files and the three source CSVs |
| `TARE_Export_README_and_Dictionary.xlsx` | the README (73 rows) and the Data Dictionary (136 rows), laid out as the two sheets of the 19 Aug workbook. Every column of every new household file has one dictionary row |
| `s9_email_facts.md` | the facts for the email (D-6) |

### 7.4 Stage 2 follow-up: all nine cases

Added to this file on 9 Oct 2026. Measured the same day by
`~/tare_port/s9_task2b_rebate_cases.py`, which reads each run's results table; no model
was run. Detail: section 6 of `~/tare_port/s9_outputs/s9_comparison_summary.md` and the
tables `s9_t2_rebate_cases_*.csv` and `s9_t2_cost_offsets_*.csv`. "19 Aug MP3" is run
`2026-08-19_20-56` limited to the rdu of today's study sample. 7% discount rate,
USD2025.

Adoption, national, percent of the study sample (no rebate / 2024 guidance / June 2026
guidance):

| Credit scope | 19 Aug MP3 | New MP3 | New MP4 | New MP5 |
|---|---|---|---|---|
| Both credits | 30.18 / 67.48 / 37.37 | 19.35 / 46.64 / 26.88 | 19.47 / 49.37 / 28.89 | 3.39 / 29.48 / 29.48 |
| Old heating credit only | 13.43 / 33.42 / 19.87 | 6.84 / 20.46 / 13.82 | 8.71 / 22.01 / 15.12 | 1.38 / 3.28 / 3.28 |
| Old AC credit only | 17.13 / 48.92 / 26.04 | 9.19 / 32.11 / 18.94 | 10.70 / 29.01 / 18.99 | 1.78 / 6.30 / 6.30 |

Adoption, Allegheny County (1,146 rdu; 1,091 rdu for MP5):

| Credit scope | 19 Aug MP3 | New MP3 | New MP4 | New MP5 |
|---|---|---|---|---|
| Both credits | 21.90 / 62.13 / 22.25 | 3.93 / 10.30 / 4.36 | 5.93 / 14.31 / 6.72 | 0.64 / 18.70 / 18.70 |
| Old heating credit only | 6.37 / 28.97 / 6.72 | 2.44 / 4.45 / 3.49 | 4.28 / 7.16 / 5.67 | 0.27 / 0.73 / 0.73 |
| Old AC credit only | 8.29 / 43.19 / 8.55 | 3.14 / 4.80 / 3.75 | 4.28 / 7.33 / 5.85 | 0.27 / 1.19 / 1.19 |

Mean dollars per home in the new files:

| Part | MP3, national | MP5, national | MP3, Allegheny | MP5, Allegheny |
|---|---|---|---|---|
| Credit for the old heating system | 4,283 | 3,810 | 4,371 | 3,895 |
| Credit for the old AC | 5,307 | 5,843 | 4,268 | 4,724 |
| Rebate, 2024 guidance, over every home (share of homes with one) | 5,483 (89.15%) | 4,959 (74.97%) | 5,603 (96.51%) | 4,812 (71.49%) |
| Rebate, June 2026 guidance, over every home (share of homes with one) | 1,539 (23.55%) | 4,959 (74.97%) | 309 (5.24%) | 4,812 (71.49%) |

- With no rebate, the credit for the old AC moves MP3 adoption more than the credit for
  the old heating system: 19.35% with both, 9.19% with the AC credit alone, 6.84% with
  the heating credit alone.
- Under the June 2026 guidance, as modeled, no home that heats with natural gas, propane
  or fuel oil gets a rebate for MP3 or MP4, so its adoption is the no-rebate rate. For
  MP3 and MP4 the June 2026 HOMES rebate is still limited to homes that heat with
  electricity (CLAUDE.md, "Fuel gate": the fuel-neutral change is deferred). MP5 passes
  both fuel gates. Read the June 2026 comparison of MP5 with MP3 and MP4 with that in
  mind.
- The script stops unless the adopters of every case listed in `REFERENCE_VALUES.md` are
  the ones in the run (37 counts), and unless the NPV equals savings less net capital
  cost on every rdu in all nine cases. All passed.
