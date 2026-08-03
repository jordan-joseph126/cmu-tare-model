# PEAK_LOAD_QA_Tamar.md

Direct answers to the six questions about standing up the TARE peak-load analysis
on a fresh AWS account. Each answer has three parts: the answer, where it lives in
the code, and what to check next. Written to be read on its own; a companion
setup-to-interpretation guide (TARE_Peak_Load_Analysis_Guide.md) is being written
separately and is referenced by section name at the end.

ASCII only. No AWS calls were made to produce this; nothing here bills an account.

---

## Q1. Is using a 2024 Colorado electricity price instead of a 2025 price a problem?

Answer: It is a basis mismatch, not a fatal error. TARE anchors every fuel price
to 2025 dollars; the price table is already in USD2025 per kWh, so dropping in a
raw 2024 Colorado price mixes a 2024-dollar number into a 2025-dollar model. The
gap is about one year of consumer inflation (roughly 2 to 3 percent). A 2024
electricity price is LOW in 2025 real terms, and the effect is not uniform across
baseline fuels: a lower electricity price narrows the spark gap (the
electricity-versus-fossil operating-cost difference), so for fossil-baseline homes
the heat pump looks better than it should and economic adoption is biased UP; for
electric-resistance-baseline homes both the baseline and the heat pump scale with
the same electricity price, so the sign does not flip but the absolute dollar
savings shrink. Two structural points: (a) how a price is keyed in the lookup depends
on the fuel, and the key is not the same as where the value came from. Electricity
and natural gas are keyed by two-letter STATE, with state-specific values
(Colorado electricity is the `CO` row, method `state_annual_2025`). Fuel oil and
propane are keyed by CENSUS DIVISION: the CSV does carry a per-state row for them,
but that row's value is a PADD-region broadcast (Colorado propane = PADD 4 Rocky
Mountain; Colorado fuel oil = U.S. national fallback), and the code averages those
per-state rows up to a census-division figure and keys the lookup by division -- so
for oil and propane the state row is the value source, not the lookup key. On an
electricity row the census division only selects the multi-year projection factor,
NOT the 2025 anchor price.
(b) This price feeds the private NPV / bill-savings side of TARE, not the peak-load
kWh, which come straight from ResStock. So it is price-invariant for the baseline
peak and the 100 percent adoption peak -- neither depends on who adopts -- but it
DOES move the CONSTRAINED scenario peak, because the constrained adopter set is an
NPV output and the NPV depends on the price.

Where this lives in the code:
`cmu_tare_model/private_impact/data_processing/create_lookup_fuel_prices.py`,
function `_build_lookup` (electricity / naturalGas keyed by `row['state']`;
fuelOil / propane keyed by `row['census_division']` after a groupby-mean over the
states in each division; `ANCHOR_YEAR = 2025`). The CSV
`eia_fuel_price_data_2025_usd2025.csv` has columns including `state`,
`census_division`, `fuel_type`, and `price_usd2025_per_kwh`.

What to check next: to change Colorado's electricity price, edit the row where
`state == 'CO'` and `fuel_type == 'electricity'` (the `price_usd2025_per_kwh`
column) -- NOT a Mountain-division row; the division only affects the year-over-year
projection factor. Inflate the 2024 price to USD2025 (CPI 2024->2025) before
entering it. IMPORTANT -- the lookup fails silently: a wrong location key (for
example the full name `Colorado`, or a census-division key used for electricity)
returns ZERO with no error, so the model would run with a zero electricity price
and you would not be warned. Confirm the key matches the state abbreviation for
electricity and natural gas. If you only need the peak MW figure, the price does
not matter.

---

## Q2. What is the MP0 like-for-like replacement baseline? Is the baseline already more efficient than the actual 2018 stock, and is the "baseline" in TARE the actual stock or MP0?

Answer: There are two different things both loosely called "baseline," and they
are not the same object. (1) ResStock MP0 (upgrade_id = 0) is the as-simulated
2018 housing stock -- the real equipment each home actually has, at its real
efficiency. It is NOT made more efficient; it is the starting condition. (2) The
TARE economic counterfactual is a modeling construct on the cost side: when a
home's existing system reaches end of life, TARE assumes the owner would otherwise
buy a new like-for-like system at today's minimum efficiency, and it credits that
avoided replacement cost against the heat pump. That "like-for-like new minimum-
efficiency replacement" is more efficient than the aging 2018 unit, but it exists
only in the NPV capital-cost accounting, not in the energy simulation. The
peak-load analysis uses the ResStock MP0 baseline (the actual stock): its baseline
timeseries query is literally `upgrade_id = "0"`. So for grid/peak purposes,
"baseline" = actual 2018 stock energy use, full stop. The like-for-like
counterfactual never enters the peak-load kWh.

Where this lives in the code: peak-load baseline query uses `upgrade_id="0"` in the
main notebook grid-impact cell "STEP 5: Baseline timeseries" (as of
tare_model_main_v2_3_EXPORT_23July2026.py:1175). The like-for-like replacement
credit is on the private-impact side
(`cmu_tare_model/private_impact/` capital-cost / replacement-credit logic), not in
`grid_impact/`.

What to check next: when you read a peak number, treat the baseline as the actual
existing stock. If you are reconciling against the NPV tables and see a different
"baseline," that is the cost-side counterfactual, not the energy baseline -- do not
expect them to match.

---

## Q3. What kind of AWS account is needed? Is there a CMU option?

Answer: A standard individual AWS account is enough. The analysis uses Athena (SQL
over the public ResStock data) and S3 reads. The ResStock End-Use Load Profiles
live in NREL's open OEDI data lake, which is public -- you are not charged to read
NREL's data; you are charged by Athena for the volume your queries scan (about USD
5 per terabyte scanned). The single-county Allegheny query scans a small slice, so
the cost to run what exists today is negligible (cents, and likely inside the free
tier). The free tier is fine to start. For a CMU option: CMU does have
institutional cloud arrangements -- ask CMU's research/cloud computing group
whether you can run under a CMU-billed AWS account rather than a personal card.
It is a billing/convenience choice, not a technical requirement; the code runs the
same either way.

Where this lives in the code: BSQ initialization in the main notebook grid-impact
cell (`BuildStockQuery(workgroup="resstock-euss", db_name="euss-oedi", ...)`, as of
tare_model_main_v2_3_EXPORT_23July2026.py:1141-1148). The AWS account is what backs
the Athena workgroup and S3 access that this call assumes.

What to check next: decide personal vs CMU-billed now, because the Athena workgroup
and query-results bucket (Q5) are configured per account -- you do not want to set
them up twice. Since national scaling is descoped, there is no large scan cost to
plan for; account setup is the only AWS cost consideration.

---

## Q4. How do you actually reach the OEDI data lake? Cyberduck fails; logging into AWS works but the bucket will not browse.

Answer: You do not browse the OEDI bucket through your own account's console,
because it is not your bucket -- it is NREL's public bucket, and your account has no
listing permission on it. That is why the console "will not browse" it and why
Cyberduck (which tries to sign requests as you) fails. The bucket is public for
anonymous access, so you reach it one of three ways: (1) the AWS CLI with the
anonymous flag, e.g.
`aws s3 ls --no-sign-request s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2018_release_1.1/`;
(2) the OEDI S3 data-lake viewer in a web browser; or (3) -- the route this
analysis actually uses -- you do not download files at all; you point Athena (via a
Glue database) at that public prefix and query it as SQL tables. An S3 GUI client
like Cyberduck is optional and is not the supported path here.

Where this lives in the code: the analysis never issues raw S3 reads; it goes
through Athena. See the BSQ init (`db_name="euss-oedi"`, `db_schema="resstock_oedi"`,
`table_name="resstock_amy2018_release_1_1"`) in the grid-impact cell (as of
tare_model_main_v2_3_EXPORT_23July2026.py:1141-1148). Those names are the Glue
catalog objects that must point at the OEDI prefix.

What to check next: verify anonymous access with the `--no-sign-request` command
above (you should see subfolders like `timeseries_individual_buildings/` and
`metadata/`). Once that works, stop trying to browse the bucket in the console and
move to the Glue-plus-Athena setup in Q5 -- that is how the code reaches the data.

---

## Q5. An OperationalError fires when running the grid-impacts cell.

Answer: `OperationalError` is the generic database error that Athena's Python
driver raises through BuildStockQuery; the useful information is the message
wrapped inside it. Given a brand-new AWS account, the causes ranked by likelihood
are:

1. The Athena workgroup named `resstock-euss` does not exist in your account. BSQ
   passes `workgroup="resstock-euss"`, and Athena errors if that workgroup is
   missing. Fresh accounts only have `primary`. (Most likely.)
2. The workgroup exists but has no query-results S3 output location set. Athena
   refuses to run with "No output location provided." (Very common on new setups.)
3. The Glue catalog objects are not registered in your account: database
   `euss-oedi` / schema `resstock_oedi` / table `resstock_amy2018_release_1_1` are
   not present because no Glue crawler has been pointed at the OEDI prefix. Athena
   then cannot resolve the table.
4. Region mismatch. Your credentials' default region is not `us-west-2`, so the
   workgroup, Glue catalog, and data are not where BSQ looks.
5. IAM permissions. The user is missing `AmazonAthenaFullAccess` or S3 read, so the
   call is denied and surfaces as an OperationalError.

To discriminate, the exact traceback detail needed is the innermost message under
the OperationalError -- specifically the Athena/botocore string, which will be one
of: "WorkGroup resstock-euss is not found" (cause 1), "No output location
provided" / "query results location" (cause 2), "Table ... does not exist" or
"Database euss-oedi not found" (cause 3), an endpoint/region error (cause 4), or
"AccessDenied" / "not authorized" (cause 5). Also send: the output of
`aws configure get region`, and `aws athena get-work-group --work-group resstock-euss`
(whether it exists and has an OutputLocation).

Where this lives in the code: `BuildStockQuery(workgroup="resstock-euss", ...)` in
the grid-impact cell (as of tare_model_main_v2_3_EXPORT_23July2026.py:1141-1148),
and the enclosing AWS-credential guard just above it (export:1122-1139) which only
checks that credentials exist -- it does NOT check the workgroup, the results
bucket, or the Glue catalog, which is why those three failures fall through to an
OperationalError at query time rather than a clear message.

What to check next: paste the innermost error message. In the meantime, confirm in
order: region is `us-west-2`; a workgroup `resstock-euss` exists with a query-
results S3 location; and the Glue database/table resolve in the Athena console.

---

## Q6. Running the main script for one state returns national figures.

Answer: The live main notebook does not hardcode the location. It prompts you for
`location_id` and `model_run_date_time` and composes both into the results
filename it loads (`..._{location_id}_{date}.csv`). So "one state returns national
figures" is almost always a file-selection issue, not a filter bug: the code loaded
a National results file even though you meant a state. Note explicitly: the LIVE
main notebook has NO `LOCATION_ID = "National"` hardcode. The ARCHIVED standalone
peak-demand notebook DID hardcode `LOCATION_ID = "National"`, and running that file
reproduces this exact symptom. Check three things, in order:

1. Which notebook file did you actually run? If it was the archived
   `grid_impact/calculate_postTARE_ts_aws_peak_demand` (or its exported `.py`),
   that file hardcodes `LOCATION_ID = "National"` and will always load national
   results regardless of intent. Switch to the live main notebook.
2. What files exist in `output_results/`, and which location token is in their
   names? The loader builds `..._{location_id}_{date}.csv`. If only
   `*_National_*` files are on disk, entering `PA` finds nothing (or you fall back
   to National). You need state-scoped result files to get state figures.
3. What did you type at the location prompt? Entering `National` (or accepting a
   default) loads national results by definition. Enter the two-letter state code
   (e.g. `PA`) and the matching timestamp.

Where this lives in the code: the interactive prompt in the main notebook
"start_new_model_run == 'N'" cell (`location_id = input(...)`,
`model_run_date_time = input(...)`, as of
tare_model_main_v2_3_EXPORT_23July2026.py:155-156), consumed by
`cmu_tare_model/utils/load_exported_results_to_df.py:load_model_run_output` /
`load_measure_package_data`, which assemble the `..._{location_id}_{date}.csv`
filename. The hardcode is only in the archived export
(calculate_postTARE_ts_aws_peak_demand_EXPORT_23July2026.py:126-127).

What to check next: run check 1 first (which file), since it is the one that
reproduces the symptom deterministically. If you are on the live notebook, do check
2 (list `output_results/`) then check 3 (what you entered).

---

## Guide sections these answers imply (input to Task 4)

The setup-to-interpretation guide should contain, at minimum:
- "Fuel prices are USD2025 and keyed by state (electricity/gas) or census division
  (oil/propane), with a silent-zero on a wrong key; price is invariant for the
  baseline and 100 percent peaks but moves the constrained peak" (from Q1).
- "Two baselines: ResStock MP0 actual stock vs the NPV like-for-like counterfactual;
  peak load uses MP0" (from Q2).
- "AWS account: individual or CMU-billed; single-county cost is negligible; national
  scaling is descoped so there is no scan-cost planning" (from Q3).
- "Reaching OEDI: anonymous S3 / OEDI viewer / Athena+Glue; Cyberduck and console
  browsing are not the route" (from Q4).
- "Athena workgroup + query-results bucket + Glue database setup," and a
  troubleshooting row for OperationalError with the five ranked causes (from Q5).
- "location_id / model_run_date_time prompt and the results-filename convention,"
  and a troubleshooting row for the national-figures symptom with the three ordered
  checks (from Q6).
- A known-limitations note that the Allegheny peak delta has no internal cross-check
  and no external benchmark yet; Step 8 (metadata-peak validation) is the planned
  internal check (not from a specific question, but required context so the reader
  does not treat the number as validated).
