"""AWS and BuildStockQuery checks for the grid-impact section.

This module does two jobs.

1. `check_athena_output_location` is the preflight the grid-impact notebook cell
   calls before it constructs BuildStockQuery. Athena writes a result file for
   every query it runs, and where those files go is set by the Athena workgroup
   -- BuildStockQuery takes no argument for it. A workgroup pointing at a bucket
   the caller cannot write to therefore fails every query, however valid the
   caller's credentials are. Without this check that failure surfaces late and
   confusingly, inside a report query, naming a function rather than the bucket.

2. Run as a script, it is a seven-stage diagnostic covering the whole setup:

       python -m cmu_tare_model.grid_impact.diagnose_bsq_aws

   Stages 3 and 4 call the same `check_athena_output_location` used by the
   notebook, so the bucket-write check has exactly one implementation.

The module imports only `boto3` and `buildstock_query`. It deliberately imports
nothing else from `cmu_tare_model`, so it can be run before the rest of the
model package is usable.

Background and remediation steps: cmu_tare_model/docs/BSQ_AWS_SETUP.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reference configuration -- the arguments the grid-impact block passes to
# BuildStockQuery. Kept here so the diagnostic can run standalone, without
# importing the notebook or the model package.
# ---------------------------------------------------------------------------
WORKGROUP = "resstock-euss"
DB_NAME = "euss-oedi"
TABLE_NAME_STEM = "resstock_amy2018_release_1_1"
DB_SCHEMA = "resstock_oedi"
BUILDSTOCK_TYPE = "resstock"

# BuildStockQuery defaults region_name to 'us-west-2' and the model does not
# override it, so every Glue and Athena call the model makes lands there no
# matter which region the AWS CLI is configured for.
BSQ_DEFAULT_REGION = "us-west-2"

SETUP_DOC = "cmu_tare_model/docs/BSQ_AWS_SETUP.md"


# ---------------------------------------------------------------------------
# The preflight check -- one function, one error message
# ---------------------------------------------------------------------------
def _split_s3_uri(uri: str) -> Tuple[str, str]:
    """Split an s3:// URI into its bucket and key prefix.

    Args:
        uri: An S3 URI, for example 's3://my-bucket/query-results/'.

    Returns:
        A (bucket, prefix) tuple. The prefix has no leading slash and may be ''.

    Raises:
        ValueError: If the string does not start with 's3://' or names no
            bucket.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri!r}")
    remainder = uri[len("s3://"):]
    bucket, _, prefix = remainder.partition("/")
    if not bucket:
        raise ValueError(f"S3 URI names no bucket: {uri!r}")
    return bucket, prefix.lstrip("/")


def _preflight_error(
    cause: str,
    workgroup: str,
    output_location: Optional[str],
    account_id: Optional[str],
) -> RuntimeError:
    """Build the single error message used for every preflight failure.

    Every way this check can fail -- a missing workgroup, a workgroup with no
    result location, a bucket that does not exist, and a bucket the caller
    cannot write -- has the same fix: point the workgroup at a bucket you own.
    So they share one message, with the underlying cause quoted at the top.

    Args:
        cause: The specific underlying problem, one line.
        workgroup: The Athena workgroup name that was checked.
        output_location: The workgroup's result location, if it was resolved.
        account_id: The caller's AWS account id, if known.

    Returns:
        A RuntimeError ready to raise.
    """
    return RuntimeError(
        f"Athena query results cannot be written, so no query can succeed.\n"
        f"  Cause           : {cause}\n"
        f"  Workgroup       : {workgroup}\n"
        f"  Result location : {output_location or '(not resolved)'}\n"
        f"  Your AWS account: {account_id or '(unknown)'}\n"
        f"Athena writes a result file for every query, to the location set on "
        f"the workgroup. Point the workgroup at a bucket you own: Athena "
        f"console -> Workgroups -> {workgroup} -> Edit -> Query result "
        f"configuration.\n"
        f"This is not a credentials problem, and it is unrelated to the public "
        f"ResStock source data. See {SETUP_DOC}."
    )


def check_athena_output_location(
    session: Any,
    workgroup: str = WORKGROUP,
    region_name: str = BSQ_DEFAULT_REGION,
    account_id: Optional[str] = None,
    write_probe: bool = True,
    verbose: bool = True,
) -> str:
    """Resolve the workgroup's query result location and confirm it is writable.

    Call this after the AWS credential check and before BuildStockQuery is
    constructed, so a misdirected result location is reported by name here
    rather than surfacing later inside a report query.

    Args:
        session: A boto3 Session, so the caller's profile and credentials are
            reused rather than resolved a second time.
        workgroup: The Athena workgroup name passed to BuildStockQuery.
        region_name: The region the workgroup lives in. Defaults to
            'us-west-2', matching BuildStockQuery's own default.
        account_id: The caller's AWS account id, used only to make the error
            message clearer.
        write_probe: If True, write a zero-byte object to the result location
            and delete it again. Reading the bucket does not prove Athena can
            write to it, so this defaults to True.
        verbose: If True, print the resolved location on success.

    Returns:
        The workgroup's query result location, an s3:// URI.

    Raises:
        RuntimeError: If the workgroup cannot be read, defines no result
            location, or that location is not writable by the caller.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    # Step 1 -- read the result location off the workgroup.
    athena = session.client("athena", region_name=region_name)
    try:
        response = athena.get_work_group(WorkGroup=workgroup)
    except (ClientError, BotoCoreError) as exc:
        raise _preflight_error(
            f"workgroup could not be read in region {region_name} ({exc})",
            workgroup,
            None,
            account_id,
        ) from exc

    configuration = response["WorkGroup"].get("Configuration", {})
    output_location = configuration.get("ResultConfiguration", {}).get(
        "OutputLocation"
    )
    if not output_location:
        raise _preflight_error(
            "workgroup defines no query result location",
            workgroup,
            None,
            account_id,
        )
    output_location = str(output_location)

    # Step 2 -- confirm the caller can reach, and optionally write, that bucket.
    try:
        bucket, prefix = _split_s3_uri(output_location)
    except ValueError as exc:
        raise _preflight_error(
            f"result location is unusable ({exc})",
            workgroup,
            output_location,
            account_id,
        ) from exc

    s3 = session.client("s3", region_name=region_name)
    try:
        s3.head_bucket(Bucket=bucket)
    except (ClientError, BotoCoreError) as exc:
        raise _preflight_error(
            f"bucket {bucket!r} is missing or not readable by you ({exc})",
            workgroup,
            output_location,
            account_id,
        ) from exc

    if write_probe:
        # A zero-byte marker is the smallest object that proves PutObject.
        probe_key = f"{prefix}bsq_preflight_probe_{uuid.uuid4().hex}.tmp"
        try:
            s3.put_object(Bucket=bucket, Key=probe_key, Body=b"")
        except (ClientError, BotoCoreError) as exc:
            raise _preflight_error(
                f"bucket {bucket!r} is not writable by you ({exc})",
                workgroup,
                output_location,
                account_id,
            ) from exc
        try:
            s3.delete_object(Bucket=bucket, Key=probe_key)
        except ClientError:
            # Writing succeeded, which is what was being tested. A marker we
            # could not clean up is not worth failing the run over.
            print(
                f"[WARN] Could not delete the probe object "
                f"s3://{bucket}/{probe_key} -- delete it by hand."
            )

    if verbose:
        print(f"""
          [OK] Athena query result location writable
            Workgroup : {workgroup}
            Location  : {output_location}
          """)
    return output_location


# ---------------------------------------------------------------------------
# The seven-stage diagnostic
# ---------------------------------------------------------------------------
def _ok(stage: str, *lines: str) -> None:
    """Print a passing stage result.

    Args:
        stage: Short stage name shown on the header line.
        *lines: Detail lines, printed indented under the header.
    """
    print(f"[OK]   {stage}")
    for line in lines:
        print(f"       {line}")
    print()


def _fail(stage: str, problem: str) -> None:
    """Print a failing stage result.

    Args:
        stage: Short stage name shown on the header line.
        problem: What went wrong. May span several lines.
    """
    print(f"[FAIL] {stage}")
    for line in str(problem).splitlines():
        print(f"       {line}")
    print()


def stage_1_import_bsq() -> bool:
    """Check that buildstock_query is installed and report where it came from.

    Returns:
        True if the package imported.
    """
    stage = "Stage 1 -- buildstock_query import"
    try:
        import buildstock_query
    except ImportError as exc:
        _fail(
            stage,
            f"Could not import buildstock_query: {exc}\n"
            f"Install it: conda activate cmu-tare-model, then\n"
            f"pip install git+https://github.com/NREL/buildstock-query.git",
        )
        return False

    try:
        from importlib.metadata import version

        pkg_version = version("buildstock_query")
    except ImportError:
        pkg_version = "unknown"

    _ok(
        stage,
        f"Version  : {pkg_version}",
        f"Location : {buildstock_query.__file__}",
        f"Python   : {sys.version.split()[0]} ({sys.executable})",
    )
    return True


def stage_2_caller_identity(session: Any) -> Optional[str]:
    """Check AWS credentials resolve, and report who they belong to.

    Also reports the region mismatch that causes the most confusion here: the
    AWS CLI region and the region BuildStockQuery queries are two different
    settings, and the model never passes the former to the latter.

    Args:
        session: A boto3 Session.

    Returns:
        The caller's AWS account id, or None if credentials did not resolve.
    """
    stage = "Stage 2 -- AWS caller identity"
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

    cli_region = session.region_name
    try:
        identity = session.client(
            "sts", region_name=BSQ_DEFAULT_REGION
        ).get_caller_identity()
    except NoCredentialsError:
        _fail(
            stage,
            "No AWS credentials found.\n"
            "Run `aws configure`, or set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY.\n"
            "If you use SSO: aws sso login --profile <your-profile>",
        )
        return None
    except (ClientError, BotoCoreError) as exc:
        _fail(
            stage,
            f"AWS rejected the credentials: {exc}\n"
            f"They may be expired. Refresh them, or set AWS_PROFILE.",
        )
        return None

    detail = [
        f"Account     : {identity['Account']}",
        f"ARN         : {identity['Arn']}",
        f"AWS_PROFILE : {os.environ.get('AWS_PROFILE', '(not set)')}",
        f"CLI region  : {cli_region or '(not set)'}",
        f"BSQ region  : {BSQ_DEFAULT_REGION} (BuildStockQuery default)",
    ]
    if cli_region and cli_region != BSQ_DEFAULT_REGION:
        detail.append(
            "NOTE: your CLI region differs from the region BuildStockQuery "
            f"queries. The workgroup and Glue database must exist in "
            f"{BSQ_DEFAULT_REGION}."
        )
    _ok(stage, *detail)
    return str(identity["Account"])


def stage_3_workgroup(session: Any, account_id: Optional[str]) -> Optional[str]:
    """Report the workgroup's query result location, without writing to it.

    Args:
        session: A boto3 Session.
        account_id: The caller's AWS account id.

    Returns:
        The result location, or None if it could not be resolved.
    """
    stage = "Stage 3 -- Athena workgroup"
    try:
        location = check_athena_output_location(
            session=session,
            account_id=account_id,
            write_probe=False,
            verbose=False,
        )
    except RuntimeError as exc:
        _fail(stage, str(exc))
        return None

    _ok(stage, f"Workgroup      : {WORKGROUP}", f"OutputLocation : {location}")
    return location


def stage_4_result_bucket(
    session: Any, account_id: Optional[str], write_probe: bool
) -> bool:
    """Confirm the result location is writable, by writing to it.

    Args:
        session: A boto3 Session.
        account_id: The caller's AWS account id.
        write_probe: If False, skip the test write.

    Returns:
        True if the location is usable.
    """
    stage = "Stage 4 -- Athena result bucket"
    try:
        location = check_athena_output_location(
            session=session,
            account_id=account_id,
            write_probe=write_probe,
            verbose=False,
        )
    except RuntimeError as exc:
        _fail(stage, str(exc))
        return False

    bucket, prefix = _split_s3_uri(location)
    note = (
        "Write probe: wrote and deleted a zero-byte test object."
        if write_probe
        else "Write probe skipped (--no-write-probe); read access alone does "
        "not prove Athena can write here."
    )
    _ok(stage, f"Bucket : {bucket}", f"Prefix : {prefix or '(bucket root)'}", note)
    return True


def _read_table_suffixes() -> Dict[str, str]:
    """Read the table-name suffixes the configured db_schema defines.

    Read from the installed buildstock_query package rather than assumed, so
    this stays correct if the package changes.

    Returns:
        A dict with 'baseline', 'timeseries' and 'upgrades' suffix strings.

    Raises:
        FileNotFoundError: If the schema file for DB_SCHEMA is not present.
        KeyError: If it has no table_suffix section.
    """
    import buildstock_query
    import toml

    schema_path = os.path.join(
        os.path.dirname(buildstock_query.__file__), "db_schema", f"{DB_SCHEMA}.toml"
    )
    return dict(toml.load(schema_path)["table_suffix"])


def stage_5_glue_tables(session: Any) -> bool:
    """Check the Glue database exists and the stem resolves to real tables.

    `table_name` is a stem, not a table name: BuildStockQuery appends a suffix
    to it. Listing the real tables makes a stem/suffix mismatch visible instead
    of surfacing later as a bare NoSuchTableError.

    Args:
        session: A boto3 Session.

    Returns:
        True if the database exists and both required tables were found.
    """
    stage = "Stage 5 -- Glue database and tables"
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        suffixes = _read_table_suffixes()
    except (FileNotFoundError, KeyError) as exc:
        _fail(stage, f"Could not read the {DB_SCHEMA!r} schema: {exc}")
        return False

    expected = [
        f"{TABLE_NAME_STEM}{suffixes['baseline']}",
        f"{TABLE_NAME_STEM}{suffixes['timeseries']}",
    ]

    glue = session.client("glue", region_name=BSQ_DEFAULT_REGION)
    table_names: List[str] = []
    try:
        glue.get_database(Name=DB_NAME)
        for page in glue.get_paginator("get_tables").paginate(DatabaseName=DB_NAME):
            table_names.extend(t["Name"] for t in page.get("TableList", []))
    except (ClientError, BotoCoreError) as exc:
        _fail(
            stage,
            f"Glue database {DB_NAME!r} not readable in "
            f"{BSQ_DEFAULT_REGION}: {exc}\n"
            f"See {SETUP_DOC}, Part 1 Section 2c.",
        )
        return False

    missing = [name for name in expected if name not in table_names]
    if missing:
        found = "\n".join(f"  - {n}" for n in sorted(table_names)) or "  (none)"
        _fail(
            stage,
            f"db_schema={DB_SCHEMA!r} turns the stem {TABLE_NAME_STEM!r} into "
            f"tables that do not exist: {', '.join(missing)}\n"
            f"table_name is a STEM -- BuildStockQuery appends "
            f"'{suffixes['baseline']}' and '{suffixes['timeseries']}' to it. "
            f"Passing a real table name gets it suffixed twice.\n"
            f"Tables actually in {DB_NAME!r}:\n{found}\n"
            f"See {SETUP_DOC}, Part 2, 'I got NoSuchTableError'.",
        )
        return False

    _ok(
        stage,
        f"Database         : {DB_NAME}",
        f"db_schema        : {DB_SCHEMA}",
        f"Stem             : {TABLE_NAME_STEM}",
        f"Baseline table   : {expected[0]} (suffix '{suffixes['baseline']}')",
        f"Timeseries table : {expected[1]} (suffix '{suffixes['timeseries']}')",
    )
    return True


def _table_label(table: Any) -> str:
    """Return a short, printable name for a BuildStockQuery table handle.

    Under 'resstock_oedi' the baseline and upgrade handles are SQL views over
    one metadata table, not plain tables. Putting one in an f-string compiles
    its entire SELECT statement, which is tens of kilobytes, so read the name
    off the object instead.

    Args:
        table: A SQLAlchemy Table or Alias, or None.

    Returns:
        The handle's name, or '(none)' if there is no table.
    """
    if table is None:
        return "(none)"
    return str(getattr(table, "name", "(unnamed)"))


def stage_6_bsq_init() -> Optional[Any]:
    """Construct BuildStockQuery exactly as the grid-impact block does.

    Construction is not free: BuildStockQuery reflects the table schemas through
    the Athena engine, so this stage already touches Glue and Athena.

    Returns:
        The initialized BuildStockQuery object, or None if construction failed.
    """
    stage = "Stage 6 -- BuildStockQuery initialization"
    from buildstock_query import BuildStockQuery

    try:
        my_run = BuildStockQuery(
            workgroup=WORKGROUP,
            db_name=DB_NAME,
            table_name=TABLE_NAME_STEM,
            db_schema=DB_SCHEMA,
            buildstock_type=BUILDSTOCK_TYPE,
            skip_reports=True,
        )
    except Exception as exc:
        # BuildStockQuery raises driver-level errors of many types here
        # (SQLAlchemy, pyathena, botocore), so a broad catch is warranted --
        # the message is what the user needs.
        _fail(
            stage,
            f"{type(exc).__name__}: {exc}\n"
            f"A NoSuchTableError here almost always means db_schema was "
            f"omitted, or table_name was given as a full table name rather "
            f"than a stem. Compare against the tables listed in stage 5.",
        )
        return None

    _ok(
        stage,
        f"Baseline handle   : {_table_label(my_run.bs_table)}",
        f"Timeseries handle : {_table_label(my_run.ts_table)}",
        f"Upgrade handle    : {_table_label(my_run.up_table)}",
        "Baseline and upgrade are SQL views over the one metadata table, "
        "split on the 'upgrade' column. That split happens in the generated "
        "SQL, which is why the Glue partition column never needs retyping.",
    )
    return my_run


def stage_7_smoke_query(my_run: Any) -> bool:
    """Run two small real queries to prove the whole path works.

    The first is a trivial 'SELECT 1', which scans no data but still forces
    Athena to write a result file. The second is `get_available_upgrades()`,
    the exact call the model makes first: `aggregate_timeseries` calls
    `_validate_upgrade`, which calls it. `skip_reports=True` does not
    suppress it.

    Args:
        my_run: An initialized BuildStockQuery object from stage 6.

    Returns:
        True if both queries returned successfully.
    """
    stage = "Stage 7 -- end-to-end query"
    try:
        status, reason = my_run.execute_raw("SELECT 1")
    except Exception as exc:
        _fail(stage, f"Athena rejected 'SELECT 1': {type(exc).__name__}: {exc}")
        return False

    if str(status).upper() != "SUCCEEDED":
        _fail(stage, f"'SELECT 1' finished with status {status}: {reason}")
        return False

    try:
        upgrades = my_run.get_available_upgrades()
    except Exception as exc:
        _fail(
            stage,
            f"The metadata query failed: {type(exc).__name__}: {exc}\n"
            f"This is the query the model runs first, before any timeseries "
            f"data is read. A missing-column error means db_schema is wrong "
            f"for these tables.",
        )
        return False

    _ok(
        stage,
        "'SELECT 1' : SUCCEEDED (proves the query result path is writable)",
        f"Available upgrades: {sorted(upgrades)}",
        "Upgrade '0' is the baseline; the model queries upgrades '3' and '4'.",
    )
    return True


def run_diagnostic(write_probe: bool = True) -> int:
    """Run every diagnostic stage in order and report an overall verdict.

    Call this from a notebook or another script. `main` is the command-line
    wrapper around it; this function takes no command-line arguments, so it is
    safe to call from Jupyter, where `sys.argv` holds the kernel's own
    arguments and would confuse an argument parser.

    Args:
        write_probe: If False, skip the test write in stage 4. Faster, but read
            access alone does not prove Athena can write results.

    Returns:
        0 if every stage passed, 1 otherwise.
    """
    print("=" * 72)
    print("BuildStockQuery / AWS diagnostic")
    print("=" * 72)
    print()

    # Importing buildstock_query configures root logging, which makes botocore
    # announce its credential lookup before every stage.
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("buildstock_query").setLevel(logging.ERROR)

    if not stage_1_import_bsq():
        return 1

    import boto3

    session = boto3.session.Session()

    account_id = stage_2_caller_identity(session)
    if account_id is None:
        return 1
    if stage_3_workgroup(session, account_id) is None:
        return 1
    if not stage_4_result_bucket(session, account_id, write_probe=write_probe):
        return 1
    if not stage_5_glue_tables(session):
        return 1

    my_run = stage_6_bsq_init()
    if my_run is None:
        return 1
    if not stage_7_smoke_query(my_run):
        return 1

    print("=" * 72)
    print("[OK] All 7 stages passed. The grid-impact section should run.")
    print("=" * 72)
    return 0


def main() -> int:
    """Parse command-line arguments and run the diagnostic.

    Returns:
        0 if every stage passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Preflight check for BuildStockQuery / AWS access, to run before "
            "the TARE grid-impact section."
        )
    )
    parser.add_argument(
        "--no-write-probe",
        action="store_true",
        help=(
            "Skip writing a zero-byte test object to the Athena result "
            "bucket. Faster, but read access alone does not prove Athena can "
            "write results there."
        ),
    )
    args = parser.parse_args()
    return run_diagnostic(write_probe=not args.no_write_probe)


if __name__ == "__main__":
    sys.exit(main())
