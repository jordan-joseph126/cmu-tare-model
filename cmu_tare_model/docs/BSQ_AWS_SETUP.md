# BuildStockQuery + AWS Setup

**For:** anyone who needs to run the grid-impact section of the TARE model on
their own AWS account for the first time.
**Last updated:** 4 September 2026
**Companion script:** `cmu_tare_model/grid_impact/diagnose_bsq_aws.py`

Part 1 is the setup: follow it top to bottom once, and the grid-impact section
will run. Part 2 answers questions that come up afterwards. You do not need to
read Part 2 to finish the setup.

This document assumes you have an AWS account and nothing else. You do not need
to have used Athena, Glue, or BuildStockQuery before.

Nothing here affects any modeled value. The grid-impact section reads hourly
electricity profiles from a public dataset; it does not change NPV, adoption,
rebates, or any number in the manuscript.

---
---

# Part 1 -- Setup Guide

---

## 1. Prerequisites

### An AWS account

A standard individual account is enough, and the free tier is a fine place to
start. Reading NREL's public ResStock data is free; Athena charges for the volume
your queries scan, about USD 5 per terabyte. The single-county analysis scans a
small slice, so running what exists today costs cents.

CMU has institutional cloud arrangements, so you can ask CMU's research computing
group about running under a CMU-billed account instead of a personal card. That is
a billing choice, not a technical one -- the code runs the same either way. Decide
before you start, because the workgroup and results bucket in Section 2 are
configured per account and you do not want to set them up twice.

### Credentials that can make these calls

| Service | Calls used |
|---|---|
| STS | `sts:GetCallerIdentity` |
| Athena | `athena:GetWorkGroup`, `athena:StartQueryExecution`, `athena:GetQueryExecution` |
| Glue | `glue:GetDatabase`, `glue:GetTables`, `glue:GetTable` |
| S3 | `s3:GetObject`, `s3:PutObject` on the results bucket you create in Section 2a |

The `AmazonAthenaFullAccess` managed policy covers the Athena and Glue calls if
you would rather attach a policy than write one.

### The AWS CLI, configured for us-west-2

Run `aws configure`, enter your access key and secret key, and set the default
region to `us-west-2`.

BuildStockQuery uses `us-west-2` by default and the model does not override it,
so every Glue and Athena call the model makes goes to `us-west-2`. Create all
three AWS resources in Section 2 there, and set your CLI to the same region so
the console shows you what the model sees.

### The project environment, plus BuildStockQuery

    conda activate cmu-tare-model
    pip install git+https://github.com/NREL/buildstock-query.git

BuildStockQuery is installed by hand, as above: it is carried in neither
`environment-cmu-tare-model.yml` nor `requirements.txt`. The version this project
has been run against is 0.2.0.

---

## 2. Create your AWS resources

Three resources, created once, all in region `us-west-2`.

### 2a. An S3 bucket for query results

Athena writes a result file for every query it runs. Create a bucket in
`us-west-2` to hold them. Bucket names are globally unique, so include your own
account id. Replace `{YOUR-ACCOUNT-ID}` with your twelve-digit AWS account
number everywhere it appears below:

    resstock-euss-query-results-{YOUR-ACCOUNT-ID}

Your account number is the `Account` field printed by:

    aws sts get-caller-identity

Default settings are correct -- no public access and no bucket policy are needed.
It only has to be yours.

### 2b. An Athena workgroup named `resstock-euss`

In the Athena console, create a workgroup named exactly `resstock-euss`. The
model passes this name, so the spelling matters.

Set its **query result location** to the bucket from Section 2a:

    s3://resstock-euss-query-results-{YOUR-ACCOUNT-ID}/

The workgroup is the only place this location is set. BuildStockQuery reads it
from the workgroup and takes no argument for it, so this one setting governs
where all query output goes.

Confirm it:

    aws athena get-work-group --work-group resstock-euss --region us-west-2

The output contains an `OutputLocation` naming your bucket.

### 2c. A Glue database named `euss-oedi`

The ResStock data itself is public and already hosted -- you register it in your
own Glue catalog rather than copying it. It lives here:

    s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2018_release_1.1/

Confirm you can read it with no credentials at all:

    aws s3 ls --no-sign-request s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2018_release_1.1/

In the Glue console, create a database named exactly `euss-oedi`, then register
the tables from that prefix. A Glue crawler pointed at the prefix is the usual
way to do it.

**The end state to aim for:** the `euss-oedi` database contains these two tables,
spelled exactly like this.

    resstock_amy2018_release_1_1_metadata
    resstock_amy2018_release_1_1_by_state

If the crawler names them something else, rename them to match. Section 3
explains where the two names come from.

---

## 3. Initialize BuildStockQuery

This is the call the model makes. Use it verbatim.

```python
from buildstock_query import BuildStockQuery

my_run = BuildStockQuery(
    workgroup="resstock-euss",
    db_name="euss-oedi",
    table_name="resstock_amy2018_release_1_1",
    db_schema="resstock_oedi",
    buildstock_type="resstock",
    skip_reports=True,
)
```

| Argument | What it does |
|---|---|
| `workgroup` | Names the Athena workgroup, which sets where results are written and which bill the queries land on. |
| `db_name` | Names the Glue database holding the table definitions. |
| `table_name` | A **stem**. BuildStockQuery appends a suffix to it to reach each table -- see below. |
| `db_schema` | Selects the naming and column convention the tables follow. `resstock_oedi` is the convention the OEDI-hosted release uses. |
| `buildstock_type` | `resstock` or `comstock`. This project is ResStock. |
| `skip_reports` | Skips the summary report printed while the object is being built. |

**How the stem becomes table names.** Under `db_schema="resstock_oedi"`,
BuildStockQuery appends `_metadata` for the baseline table and `_by_state` for
the timeseries table:

    table_name = "resstock_amy2018_release_1_1"

    becomes    resstock_amy2018_release_1_1_metadata     (baseline)
               resstock_amy2018_release_1_1_by_state     (timeseries)

Those are the two tables Section 2c set up.

---

## 4. Run the diagnostic

`cmu_tare_model/grid_impact/diagnose_bsq_aws.py` confirms the whole setup in the
order the model needs it. Run it from the repository root:

    conda activate cmu-tare-model
    python -m cmu_tare_model.grid_impact.diagnose_bsq_aws

It needs only `boto3` and `buildstock_query`: it imports no other part of the
model package, so it works whether or not the model itself is usable yet.

| Stage | What it confirms |
|---|---|
| 1 | `buildstock_query` imports, and reports its version |
| 2 | AWS credentials resolve; reports your account, ARN, and region |
| 3 | The workgroup exists; prints its `OutputLocation` verbatim |
| 4 | The results bucket is readable and writable (writes a zero-byte test object, then deletes it) |
| 5 | The Glue database exists and the stem resolves to tables that are there |
| 6 | `BuildStockQuery` builds with the configuration from Section 3 |
| 7 | A real `SELECT 1` runs, then the first metadata query the model makes |

Pass `--no-write-probe` to skip the test write in stage 4. The default is
preferred: writing is what Athena actually does with the bucket.

A finished run ends with:

    [OK] All 7 stages passed. The grid-impact section should run.

Exit code 0 means every stage passed. Part 2 covers what to do about a stage
that does not.

---

## 5. Where to go next

The grid-impact code is the `if GRID_IMPACT_ANALYSIS:` block in
`cmu_tare_model/tare_model_main_v3_0.ipynb`. Set `GRID_IMPACT_ANALYSIS = True`
and run the notebook.

The exported snapshot `tare_model_main_v3_0_EXPORT_3Sep2026.py` mirrors that
notebook for reading and diffing. It is read-only; the notebook is what runs.

### Sampling weight -- do not apply it twice

Every EUSS 2022.1.1 building carries a uniform sampling weight of 242.131013.
BuildStockQuery applies that weight inside the SQL it generates
(`SUM(enduse x weight)`), so the kWh values `aggregate_timeseries` returns are
already weight-applied. The grid-impact code converts them to MW by dividing by
1000 and does not multiply by the weight again. The natural assumption is the
opposite, so check this before trusting any peak number you compute yourself.

**Sanity check.** Baseline per-building-hour kWh from the reference run ranged
from 31.961 to 10401.222. Divide by 242.131013 to get per-home values: about
0.13 to 43.0 kWh per home per hour, which is plausible residential load. Had the
weight been applied twice, the maximum would be roughly 2.5 million -- three
orders of magnitude too high. Per-home values in the tens of kWh confirm the
weight was applied exactly once.

**Dataset scope.** This project is pinned to ResStock 2022.1.1 (EUSS). Keep the
table names, database, and schema above as written -- CLAUDE.md pins the release,
and ResStock 2025.1 differs in table names, column names, and upgrade numbering.

---
---

# Part 2 -- Troubleshooting and FAQ

You do not need this section to complete the setup. It answers the questions that
come up once something has gone wrong.

If the diagnostic reported a failing stage, start here:

| Failing stage | Question to read |
|---|---|
| 3 or 4 | "I got a 403 Forbidden from S3 -- is this a credentials problem?" |
| 5 or 6 | "I got `NoSuchTableError` -- what's wrong?" |
| 7 | Either of the above, depending on the message |

---

## "Why can't I change the type of the `upgrade` column in Glue?"

Because AWS does not allow it, and you do not need to.

If you try to retype the `upgrade` partition column from `string` to `int`, AWS
refuses:

    InvalidInputException: Change of partitionColumn is not allowed on indexKey : upgrade

This refusal is correct and it is not blocking anything. AWS does not allow
retyping a partition key on a table that already has registered partitions,
because the partition values are stored as the folder names they were crawled
from. That is by design.

The column does not need to be an `int`. BuildStockQuery casts it in the SQL it
generates. Under `db_schema="resstock_oedi"` the baseline and upgrade tables are
the same physical metadata table, and BuildStockQuery splits them into two SQL
views by casting `upgrade` to a string and comparing it to `"0"`. That cast
happens on every query, whatever the Glue column type is.

You can see this in the diagnostic: stage 6 prints the two handles as `baseline`
and `upgrade`, which are views rather than tables.

So if you hit `InvalidInputException` here, stop and move on -- nothing is broken.
There is no need to delete and recreate the table to work around it.

---

## "I got `NoSuchTableError` -- what's wrong?"

Almost always: `table_name` was given a real table name instead of a stem, or
`db_schema` was left out. Often both at once.

### `table_name` is a stem

`table_name` is a prefix that BuildStockQuery appends a suffix to. It is not the
name of a table in your Glue catalog. Which suffix it appends depends entirely on
`db_schema`:

| | `resstock_oedi` (this project) | `resstock_default` (the fallback when `db_schema` is omitted) |
|---|---|---|
| Baseline table suffix | `_metadata` | `_baseline` |
| Timeseries table suffix | `_by_state` | `_timeseries` |
| Building id column | `bldg_id` | `building_id` |
| Weight column | `weight` | `build_existing_model.sample_weight` |
| Timestamp column | `timestamp` | `time` |

Note that the two schemas differ in column names as well as table suffixes, so an
omitted `db_schema` can also produce errors about missing columns.

### How the double suffix happens

You look in the Glue console, see a real table named
`resstock_amy2018_release_1_1_by_state`, and pass that as `table_name`.
BuildStockQuery appends a suffix to it and looks for a table suffixed twice:

    resstock_amy2018_release_1_1_by_state_baseline    <-- does not exist

which surfaces as:

    NoSuchTableError: resstock_amy2018_release_1_1_by_state_baseline

Two mistakes combine to produce that exact string: a real table name passed as
the stem, and a missing `db_schema` -- which is why the suffix is `_baseline`
rather than `_metadata`.

**The fix:** pass the stem, and always pass `db_schema="resstock_oedi"`. Use the
call in Part 1, Section 3 verbatim.

Diagnostic stage 5 catches this before the model runs: it prints the table names
your stem resolves to alongside the tables that actually exist in the database.

---

## "I got a 403 Forbidden from S3 -- is this a credentials problem?"

No. It is almost always the query result location.

### There are two S3 locations, and they are unrelated

| | **Location A: the source data** | **Location B: your query results** |
|---|---|---|
| What it is | The public ResStock dataset | Scratch files Athena writes for every query |
| Where | `s3://oedi-data-lake/nrel-pds-building-stock/...` | The bucket you created in Part 1, Section 2a |
| Who owns it | NREL / OEDI, public | You |
| Do you need permission? | No, it is open to everyone | Yes, and you have it because it is yours |
| Set by | The Glue database and table definitions | The Athena workgroup, and nothing else |

Every Athena query reads from Location A and writes a result file to Location B.
Both have to work, and they fail in different ways.

### Reading the error

    QueryException: AmazonS3Exception: Forbidden (403) ...
    Bucket: resstock-core, Key: ...

Read the bucket name. If it is not a bucket in your account, Location B is wrong:
your workgroup is configured to write results somewhere you have no access to.

`resstock-core` in particular is NREL's own internal bucket -- it appears inside
the BuildStockQuery package itself. No external user can write to it, and it is
not part of the public dataset.

### What is not the problem

- **Your credentials.** They are working; that is how you got as far as running a
  query.
- **The public ResStock data.** That is Location A, a different bucket entirely,
  open to everyone.
- **Your IAM policies.** No policy you can attach to your own identity grants
  write access to a bucket in an account you do not control.

### The fix

1. Run `aws athena get-work-group --work-group resstock-euss --region us-west-2`
2. Look at `OutputLocation`.
3. If it names a bucket you do not own, change it: Athena console --> Workgroups
   --> `resstock-euss` --> Edit --> Query result configuration. Point it at your
   own bucket from Part 1, Section 2a.

Do not ask anyone to share credentials with you, and do not try to reuse another
account's workgroup. Neither can work, and neither is necessary -- your own
results bucket costs nothing to create.

---

## "Why won't the OEDI bucket open in my console, or in Cyberduck?"

Because it is not your bucket. It is NREL's, and your account has no listing
permission on it -- which is why the console will not browse it, and why Cyberduck
fails when it tries to sign requests as you.

The bucket is open to anonymous access, so you reach it either with the AWS CLI's
anonymous flag:

    aws s3 ls --no-sign-request s3://oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2018_release_1.1/

or through the OEDI data-lake viewer in a browser. You should see subfolders such
as `metadata/` and `timeseries_individual_buildings/`.

The analysis itself does neither. It never issues a raw S3 read and downloads no
files: Glue points Athena at that public prefix, and the code queries it as SQL
tables. Once the command above works, stop trying to browse the bucket and
continue with the Glue setup in Part 1, Section 2c.

---

## "Why didn't `skip_reports=True` skip this query?"

Because it only suppresses the report printed while the object is being built,
not every report query.

The first time you call `aggregate_timeseries`, this happens:

    aggregate_timeseries
      --> _validate_upgrade          (checks your upgrade id is real)
        --> get_available_upgrades
          --> report.get_success_report()   <-- runs a real Athena query

So the first aggregate call reaches Athena and S3 even with
`skip_reports=True`. Two consequences:

1. A preflight check that only validates credentials proves nothing. It has to
   run a real query end to end, which is what the diagnostic's stage 7 does.
2. A traceback naming `get_success_report` looks like a reporting problem but is
   not one. It is your first real query, and the cause is Location A or
   Location B.

Constructing `BuildStockQuery(...)` is not free either: it reflects the table
schemas through Athena, so a `NoSuchTableError` surfaces at construction time,
before you have run anything yourself.
