# TARE Peak-Load Analysis Guide

A setup-to-interpretation guide for running the post-TARE county peak-load
analysis on your own AWS account. Written for someone comfortable with Python who
has not used AWS before. Plain language, ASCII only.

Companion documents: `PEAK_LOAD_QA_Tamar.md` (direct answers to six specific
questions) and, for AWS specifics, `reference_material/aws_resstock_setup_guide.md`.

---

## 1. What this computes, and why it matters for Section 3.6

The analysis answers one question for a single county: if homes replace their
existing heating with a heat pump, how much does the county's peak electricity
demand (the single highest hour of the year) change? It produces, per measure
package and per adoption scenario, an hourly demand profile of 8,760 hours and a
peak-load change in megawatts (MW).

Two adoption scenarios are computed:
- "100 percent adoption" (`all_filtered`): every eligible home in the county
  adopts the heat pump. This is the upper bound.
- "constrained" (`constrained`): only homes where the heat pump pays for itself
  adopt, i.e. economic adopters with net present value (NPV) at or above zero.
  This is the economically realistic case.

Peak-load change is defined as:

    peak_load_change = max(scenario_profile) - max(baseline_profile)

taken over the 8,760 hours. Important: the two maxima can fall in DIFFERENT hours,
so this is a difference of two annual peaks, not an hour-by-hour difference.

Scope, stated plainly: this pipeline computes peak load for ONE county at a time.
The only validated case is Allegheny County, PA (FIPS 42003). A national,
all-county version (looping over ~3,098 counties and exporting a table) is NOT
implemented and is NOT currently planned -- see Section 10. The dataset is ResStock
2022.1.1 (EUSS, AMY2018); ResStock 2025.1 (cold-climate and dual-fuel heat pumps)
is future work.

For Section 3.6, the deliverable today is the Allegheny demand-profile figure and
its baseline-vs-scenario peak numbers, for MP3 (standard ASHP) and MP4
(high-efficiency ASHP), under both adoption scenarios.

---

## 2. Prerequisites and environment

Python side (same environment the rest of TARE runs in):
- Python 3.12, with `pandas`, `numpy`, `geopandas`, `matplotlib`, and `boto3`.
- `buildstock-query` (BSQ), installed from NREL's GitHub (Section 6).
- The TARE model outputs already exported to `cmu_tare_model/output_results/`
  (the per-measure-package result CSVs the notebook loads). The peak-load section
  reads those to decide who the economic adopters are.

AWS side (Sections 3 to 5):
- An AWS account with an IAM user, access keys, and region `us-west-2`.
- An Athena workgroup with a query-results S3 bucket.
- A Glue database that points at the public ResStock data in the OEDI data lake.

You do NOT need to download any ResStock data by hand. The analysis reads it
through Athena.

---

## 3. AWS account, IAM user, access keys, region

1. Create or use an AWS account. A standard individual account is enough; the free
   tier is fine to start. (CMU may offer an institutional/billed account -- ask CMU
   research computing. It is a billing convenience, not a technical requirement.)
2. Create an IAM user for programmatic access. Attach these managed policies:
   `AmazonAthenaFullAccess` and `AmazonS3ReadOnlyAccess`. (Athena also needs write
   access to its own query-results bucket; see Section 4.)
3. Create an access key for that user. You will get an Access Key ID and a Secret
   Access Key.
4. Configure them locally: run `aws configure`, paste the two keys, and set the
   default region to `us-west-2`. The ResStock EUSS data, the Athena workgroup, and
   the Glue catalog all live in `us-west-2`; a different default region is the most
   common silent misconfiguration.

Verify: `aws sts get-caller-identity` should print your account number and user
ARN. `aws configure get region` should print `us-west-2`.

---

## 4. Athena workgroup and query-results bucket

The notebook initializes BSQ with `workgroup="resstock-euss"`. Athena will error if
that workgroup does not exist in your account, so you must create it once:

1. In the Athena console (region us-west-2), create a workgroup named exactly
   `resstock-euss`.
2. Give it a query-results location: an S3 bucket you own, e.g.
   `s3://<your-bucket>/athena-results/`. Athena writes every query's output there.
   Without this, Athena raises "No output location provided."
3. Confirm your IAM user can read and write that bucket.

Verify: `aws athena get-work-group --work-group resstock-euss` should return the
workgroup with an `OutputLocation` set.

---

## 5. Glue database and crawler pointed at the OEDI prefix

The notebook initializes BSQ with `db_name="euss-oedi"`,
`db_schema="resstock_oedi"`, and `table_name="resstock_amy2018_release_1_1"`. Those
are Glue Data Catalog objects that must resolve in your account and point at the
public ResStock data.

The public data lives here (open, no sign-in needed):

    s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2018_release_1.1/

Confirm you can see it anonymously:

    aws s3 ls --no-sign-request s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2018_release_1.1/

You should see subfolders such as `metadata/` and
`timeseries_individual_buildings/`. (An S3 GUI client like Cyberduck is optional
and is NOT the supported route; you do not browse this bucket through your own
console because it is not your bucket. See PEAK_LOAD_QA_Tamar.md Q4.)

To register it in Glue, either run a Glue crawler over that prefix to create the
database and tables, or use the table definitions NREL publishes for the OEDI
release. The BSQ names above must match whatever you create: database `euss-oedi`,
schema `resstock_oedi`, table stem `resstock_amy2018_release_1_1`. See
`reference_material/aws_resstock_setup_guide.md` for the detailed crawler steps.

Verify: in the Athena console, with workgroup `resstock-euss`, a simple
`SELECT ... LIMIT 10` against the metadata table should return rows.

---

## 6. Installing and verifying BuildStockQuery

Install from NREL's GitHub:

    pip install git+https://github.com/NREL/buildstock-query.git

Verify the import and that it can see your AWS setup by running the first part of
the grid-impact cell (Section 7, Step 1). A successful BSQ initialization prints:

    [OK] BuildStockQuery initialized: BuildStockQuery

If it raises before that line, the problem is credentials or region (Section 3),
not BSQ itself.

---

## 7. Running the peak-load section step by step

Open the main model-run notebook `tare_model_main_v2_3.ipynb`. Near the top it asks
whether to start a new model run. Answer `N` to load existing results, then enter
the `location_id` and `model_run_date_time` that match the result files in
`cmu_tare_model/output_results/`. Then set `GRID_IMPACT_ANALYSIS = True` and run the
grid-impact section. The section runs in these stages (function names in
parentheses are in `cmu_tare_model/grid_impact/peak_load_functions.py` unless noted):

Stage A -- Build adopter IDs by measure package and county.
For each MP, the notebook finds the economic-adopter column
(`find_adoption_column`), converts each home's county code to a 5-digit FIPS
(`gisjoin_to_fips`), and splits each county into `all_filtered` (everyone) and
`constrained` (economic adopters, where the adopter column equals 1.0). Expected
output, per MP:

    [OK] MP3: adoption column ref2025_mp3_heatingLCC_coolingSavings_sub_econ_adopter_fixed_base
         Counties: N | all_filtered: N | constrained (NPV>=0): N

Note: the adopter set is the NPV-based economic-adoption definition used throughout
TARE. It is NOT a Tier 1 to Tier 4 split; that older tiered scheme is retired.

Stage B -- BSQ init and Allegheny baseline timeseries (Step 5).
The notebook verifies AWS credentials, initializes BSQ, takes the union of
Allegheny building IDs across the selected MPs, and queries the baseline
(pre-retrofit) hourly electricity via `my_run.agg.aggregate_timeseries`. The query
uses `ELEC_TOTAL_COL` (total electricity, all end uses), `upgrade_id="0"`,
`timestamp_grouping_func="hour"`, and `split_enduses=False`.
`split_enduses=False` is REQUIRED; `True` triggers a validation error inside BSQ.
Expected output (values are from a real run):

    [OK] BuildStockQuery initialized: BuildStockQuery
    [OK] Allegheny County baseline bldg_ids (union across MPs [3, 4]): 1,610
    ========== df_ts_baseline_allegheny summary ==========
      Rows       : 14,103,600
      Buildings  : 1,610
      Hours/bldg : 8,760 - 8,760
      kWh range  : 31.961 to 10401.222
      BSQ weight : 242.131013
      Query time : 344.90 s
    [OK] Step 5 PASSED

The baseline query takes several minutes (about 345 s in the reference run).
`14,103,600` rows = 1,610 buildings x 8,760 hours.

Stage C -- Upgrade timeseries per MP (Step 6).
Same query with `upgrade_id=str(mp)`. It checks that no building appears in the
upgrade set but not the baseline (that raises a ValueError), and notes any
buildings with no upgrade data (they fall back to baseline). Expected tail:

    [OK] Step 6 PASSED

Stage D -- Scenario profiles and the figure (Step 7).
For each MP and each scenario, `compute_county_scenario_profile` swaps adopters to
their retrofit electricity, aggregates across buildings to an hourly county profile
in MW, and returns the profile plus a peak dictionary. The notebook prints the peak
results and draws a 2x2 figure (`plot_demand_panel`). Expected per MP/scenario:

    [100pct] adopters: N / N
      baseline peak : X.XX MW @ hour H
      scenario peak : Y.YY MW @ hour H
      delta         : +Z.ZZ MW

Stage E -- County geography lookup and the heating-fuel table.
Builds a FIPS-to-name lookup from the county shapefile (`county_geo_df`,
`gdf_counties`) and prints the baseline heating-fuel breakdown for the four
MP-by-scenario combinations. The table ends with:

    [OK] Baseline heating fuel distribution table complete
    [OK] Verification: all pct columns sum to 100.00%

---

## 8. What each output object contains

- `adopter_ids_by_mp[mp][fips]`: dict with `all_filtered` (all county buildings)
  and `constrained` (economic adopters) building-ID lists.
- `adoption_col_by_mp[mp]`: the economic-adopter column name used for that MP.
- `df_ts_baseline_allegheny`: one row per (building, hour); `baseline_kwh` is
  weight-applied electricity; `units_count` is the BSQ sampling weight.
- `df_ts_upgrade_allegheny_by_mp[mp]`: same shape, `retrofit_kwh` per building-hour.
- `df_profiles_by_mp[mp][scenario]`: 8,760-row hourly county profile with
  `baseline_mw`, `scenario_mw`, `delta_mw`.
- `peak_results_allegheny_by_mp[mp][scenario]`: dict with `baseline_peak_mw`,
  `scenario_peak_mw`, `delta_mw`, `peak_hour_baseline`, `peak_hour_scenario`,
  `n_adopters`, `n_total_buildings`.
- `county_geo_df` / `gdf_counties`: FIPS-to-county-name lookup and a merge-ready
  county GeoDataFrame.

### Sampling weight -- do not double-apply (important)

Every EUSS 2022.1.1 building carries a uniform sampling weight of 242.131013. BSQ
applies this weight INSIDE the SQL it generates (`SUM(enduse x weight)`), so the
kWh values returned by `aggregate_timeseries` are ALREADY weight-applied. The code
converts to MW by dividing by 1000 and does NOT multiply by the weight again. The
natural assumption is the opposite, so state it plainly to anyone reviewing the
code: multiplying by 242 a second time would inflate every number by about 242x.
An older hardcode of 240.0 was removed (it was a 0.9 percent error).

### Fuel prices (context, not used by the peak numbers)

Fuel prices live in `create_lookup_fuel_prices.py` (function `_build_lookup`,
reading `eia_fuel_price_data_2025_usd2025.csv`, column `price_usd2025_per_kwh`,
anchor year 2025). Two facts a CSV reader needs, because the lookup key is not the
same as where the value came from:
- Electricity and natural gas are keyed by two-letter STATE, with state-specific
  values (e.g. Colorado electricity is the `CO` row).
- Fuel oil and propane are averaged up to CENSUS DIVISION and keyed by the division
  name. The CSV DOES also contain a per-state row for oil and propane (so you will
  see a `CO` propane row), but that row's value is a PADD-region or national
  fallback broadcast, and the code averages those state rows to a division figure
  and keys on the division. So for oil and propane the per-state row is the value
  source, not the lookup target.
- Silent-zero warning (both groups): a wrong location key (for example the full
  name `Colorado`, or a state key where a division key is expected) returns ZERO
  with no error. You would not be warned.

These prices feed the NPV / adoption decision, not the peak MW numbers directly
(the peak kWh come straight from ResStock). See PEAK_LOAD_QA_Tamar.md Q1.

---

## 9. How to read the demand figure and the peak delta

Each panel plots two hourly curves over the year: baseline (red) and scenario
(blue), each in MW, with a dashed vertical line at that curve's annual peak hour
and a labeled peak value. The rows are the two scenarios (100 percent and
constrained); the columns are MP3 and MP4.

Read the peak delta as: scenario annual peak minus baseline annual peak. A positive
delta means electrification raises the county peak (heat pumps add winter load); a
negative delta means it lowers it. Because the baseline and scenario peaks can occur
in different hours, do not read the delta as the height difference at a single hour.

### Weight sanity check (do this once to trust the numbers)

From the reference run, the baseline per-building-hour kWh range was 31.961 to
10401.222. These are weight-applied, so divide by 242.131013 to get per-home values:
about 0.13 to 43.0 kWh per home per hour. That is plausible residential load. If the
weight had been double-applied, the maximum would be about 2.5 million (three orders
of magnitude too high). Seeing per-home values in the tens-of-kWh range is your
confirmation the weight was applied exactly once.

---

## 10. Known limitations and what is not implemented

- Single county only. The only validated case is Allegheny County, PA (FIPS 42003).
  The national all-county loop (former "Step 9") and its CSV export (former
  "Step 10") are DESCOPED -- not currently required or feasible -- so they are not in
  the live pipeline. The empirical reason is visible in the numbers: one county
  already returns 14,103,600 timeseries rows and a ~345 s query, so a naive
  3,098-county loop is not practical as written.
- No internal cross-check yet. The Allegheny peak delta has NOT been validated
  against an independent number. A planned check (former "Step 8", PRESERVED, not
  descoped) will sum the EUSS metadata annual peak columns (the winter and summer
  peak-kW fields; confirm the exact column names against the metadata schema when
  implementing) across the county's buildings and compare to the profile-derived
  peak, investigating divergence above 20 percent. It is a metadata read, not a
  timeseries scan, so it is unaffected by the Step 9/10 descoping. Until it is done,
  do NOT describe the reported peak as validated.
- No external benchmark yet (e.g. a published statewide peak-change comparison).
- Dual-fuel and cold-climate heat pumps are not modeled (ResStock 2022.1.1).
- `extract_adopter_ids` in `peak_load_functions.py` is legacy and not used by the
  live path; ignore it.

### AWS cost

Running the single-county analysis that exists today costs a negligible amount in
Athena charges (Athena bills per terabyte scanned; a single county scans a small
slice), and there is no large national scan to plan for because that path is
descoped.

Data scanned for the Step 5 baseline query: <PLACEHOLDER -- to be filled from the
Athena console query history for this query; do not substitute the pyathena
result-set read, and do not estimate.>

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NoCredentialsError` / "AWS credentials not found" | No local credentials configured | Run `aws configure`; set keys and region us-west-2 (Section 3) |
| BSQ init or a query raises `OperationalError`, ranked most-likely first | (1) workgroup `resstock-euss` does not exist; (2) workgroup has no query-results S3 location; (3) Glue db `euss-oedi` / table not registered; (4) region not us-west-2; (5) missing IAM Athena/S3 permissions | Read the innermost message in the traceback (it names the cause), then Sections 4, 5, 3 in that order. See PEAK_LOAD_QA_Tamar.md Q5 for the exact strings to look for |
| Running for one state returns national figures | You ran the archived standalone notebook (which hardcodes `LOCATION_ID = "National"`), OR only `*_National_*` result files exist, OR you entered `National` at the prompt | Use the live main notebook (no hardcode); check filenames in `output_results/`; enter the two-letter state at the `location_id` prompt. See PEAK_LOAD_QA_Tamar.md Q6 |
| Cyberduck / console will not browse the OEDI bucket | It is NREL's public bucket, not yours; signed listing is denied | Use `aws s3 ls --no-sign-request ...` or the OEDI viewer, or just let Athena read it (Section 5) |
| BSQ raises a Pydantic `ValidationError` on the timeseries query | `split_enduses=True` in the TSQuery | Use `split_enduses=False` (already set in the notebook) |
| `KeyError: 'units_count'` in the Step 5 summary | Your BSQ version returns a different count-column name | Report it; the weight-diagnostic line reads `units_count` and may need the column name adjusted |
| `KeyError` naming the adoption column, listing `econ_adopter` candidates | The loaded results do not contain the expected economic-adopter column for that MP/discount rate | Confirm you loaded the right `location_id`/date results and that the model run produced the `..._econ_adopter_fixed_base` columns |
| `ModuleNotFoundError: geopandas` | geopandas not installed in the active environment | Install geopandas into the TARE environment (Section 2) |
| Connecticut renders gray on a county map | County geometry vintage mismatch (2022+ vintages drop CT's eight counties) | The live geometry is already `cb_2021_us_county_500k`, which retains them; do not advance the vintage past 2021 |

---

## Appendix -- key files and constants

- `cmu_tare_model/grid_impact/peak_load_functions.py`: `gisjoin_to_fips`,
  `find_adoption_column`, `compute_county_scenario_profile`, `plot_demand_panel`.
- `cmu_tare_model/adoption_kpis/data_loading.py`: `load_euss_baseline`,
  `load_euss_upgrade`, `mp_to_upgrade`, `COUNTY_SHAPEFILE_PATH`, `SHAPEFILE_PATH`,
  and the geometry vintage constants (`COUNTY_GEOMETRY_PRODUCT` / `_VINTAGE` /
  `_SCALE`, and the mirrored `STATE_GEOMETRY_*`).
- `cmu_tare_model/adoption_kpis/demand.py`: `compute_scenario_demand`,
  `aggregate_demand` (alias `aggregate_demand_by_state`).
- `cmu_tare_model/utils/load_exported_results_to_df.py`: `load_model_run_output`,
  `load_measure_package_data`.
- `cmu_tare_model/constants.py`: `BLDG_ID_COL`, `TIMESTAMP_COL`, `ELEC_TOTAL_COL`
  (BSQ form, no `.kwh`), `BSQ_ELEC_COL`, `METADATA_TABLE`, `COUNTY_COL`,
  `STATE_COL`, `WEIGHT_COL`, `MIN_HOME_COUNT`, `TEST_FIPS` (`42003`),
  `TEST_GISJOIN` (`G4200030`).
- BSQ init: `workgroup="resstock-euss"`, `db_name="euss-oedi"`,
  `table_name="resstock_amy2018_release_1_1"`, `db_schema="resstock_oedi"`,
  `buildstock_type="resstock"`, `skip_reports=True`.
