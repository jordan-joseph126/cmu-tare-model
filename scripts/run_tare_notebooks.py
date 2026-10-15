"""Runs the TARE model notebooks end to end with no keyboard input.

The researcher runs the model by opening tare_model_main_v3_0.ipynb, running
every cell, and typing three answers. This script does the same thing: it runs
the main notebook from the first cell to the last, and the main notebook in
turn runs the simulation, baseline and scenario notebooks with its own
``%run -i`` calls. The questions the notebooks ask are answered from the
command-line options, so the run never waits for the keyboard.

Default answers (the researcher's own):
    Y      to "begin a new simulation"
    N      to "filter for a specific state's data" (the whole country)
    42003  to the grid-impact county FIPS question

Examples (from the repo root, in Git Bash or any shell):

    python scripts/run_tare_notebooks.py --release 2025.1 --skip-grid-impact \\
        --log run_national.log
    python scripts/run_tare_notebooks.py --release 2025.1 --state PA \\
        --skip-grid-impact

How a notebook is run:
    One IPython shell runs the main notebook with IPython's own ``%run -i``.
    IPython's notebook runner is replaced, for this shell only, by one that
    runs the same code cells in the same namespace but stops at the first
    failing cell of ANY notebook and names it. Without that, a failure inside
    the simulation notebook would be caught by the main notebook's
    "try again" loop around its first question. The notebook files are only
    read, never written, and no cell output is saved.

Exit codes:
    0  the run and every check passed
    1  a notebook cell failed
    2  a notebook asked a question this script does not recognize, or asked
       the same question twice (an answer was rejected)
    3  the run finished but a check after it failed
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_NOTEBOOK = REPO_ROOT / "cmu_tare_model" / "tare_model_main_v3_0.ipynb"

# The question the main notebook asks first, and the other questions the
# notebooks can ask, each found by a piece of its text. The text is matched
# in lower case, so a change of capitals in a notebook does not matter.
QUESTION_NEW_RUN = "begin a new simulation"
QUESTION_STATE_MENU = "filter for a specific state's data"
QUESTION_WHICH_STATE = "which state would you like to analyze"
QUESTION_CITY_MENU = "filter a subset of city-level data"
QUESTION_FIPS = "county fips code for the grid-impact"

# A NPV is rounded to cents, so its identity can be off by at most half a
# cent, plus a little floating-point room.
NPV_IDENTITY_TOLERANCE = 0.005 + 1e-6

# How often the memory sampler looks at the process, in seconds.
MEMORY_SAMPLE_SECONDS = 0.5


class RunnerStop(BaseException):
    """Stops the whole run.

    A BaseException, not an Exception, on purpose: the main notebook wraps its
    first question in ``try/except Exception`` and asks again on any error, so
    an ordinary exception raised inside the run would loop instead of stop.
    """

    exit_code = 1


class NotebookCellFailed(RunnerStop):
    """A cell in one of the notebooks raised an error."""

    exit_code = 1

    def __init__(self, notebook: str, cell_number: int, error: BaseException):
        self.notebook = notebook
        self.cell_number = cell_number
        self.error = error
        super().__init__(
            f"{notebook}, cell {cell_number}: "
            f"{type(error).__name__}: {error}")


class UnexpectedQuestion(RunnerStop):
    """A notebook asked something this script cannot answer."""

    exit_code = 2


# ---------------------------------------------------------------------------
# Output to the screen and, optionally, a log file
# ---------------------------------------------------------------------------

class _Tee:
    """Writes everything to the terminal and to a log file."""

    def __init__(self, stream: Any, log_file: Any):
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _say(message: str) -> None:
    """Prints one runner line, marked so it stands out from notebook output."""
    print(f"[runner] {message}", flush=True)


# ---------------------------------------------------------------------------
# Memory: peak for the process, and which notebook cell it happened in
# ---------------------------------------------------------------------------

class _MemoryWatch:
    """Samples the process's memory in the background.

    Records the largest resident memory seen and the notebook cell that was
    running at the time, so a run that comes close to the machine's memory
    shows where.
    """

    def __init__(self) -> None:
        import psutil
        self._process = psutil.Process()
        self.current_label = "start"
        self.peak_bytes = 0
        self.peak_label = "start"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.check()
            self._stop.wait(MEMORY_SAMPLE_SECONDS)

    def check(self) -> int:
        """Reads the memory now and keeps it if it is the largest so far."""
        rss = self._process.memory_info().rss
        if rss > self.peak_bytes:
            self.peak_bytes = rss
            self.peak_label = self.current_label
        return rss

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.check()

    def operating_system_peak_bytes(self) -> Optional[int]:
        """The operating system's own record of the peak, where it keeps one.

        Linux and macOS keep it in the resource module (kilobytes on Linux,
        bytes on macOS); Windows keeps it as the peak working set.
        """
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak if sys.platform == "darwin" else peak * 1024
        except ImportError:
            memory = self._process.memory_info()
            return getattr(memory, "peak_wset", None)


# ---------------------------------------------------------------------------
# Answering the notebooks' questions
# ---------------------------------------------------------------------------

def build_answers(
    state: Optional[str], fips: str
) -> List[Tuple[str, str]]:
    """Pairs each known question with the answer for this run.

    Args:
        state: Two-letter state code for a one-state run, or None for the
            whole country.
        fips: Five-digit county FIPS code for the grid-impact case study.

    Returns:
        (question text, answer) pairs. The state and city questions are
        included only for a one-state run, so on a national run they are
        unexpected and stop it.
    """
    answers = [
        (QUESTION_NEW_RUN, "Y"),
        (QUESTION_STATE_MENU, "N" if state is None else "Y"),
        (QUESTION_FIPS, fips),
    ]
    if state is not None:
        answers += [
            (QUESTION_WHICH_STATE, state),
            (QUESTION_CITY_MENU, "N"),
        ]
    return answers


def make_input_answerer(
    answers: List[Tuple[str, str]],
    asked: Dict[str, int],
) -> Callable[[str], str]:
    """Builds the function that stands in for ``input()`` during the run.

    Args:
        answers: (question text, answer) pairs from build_answers.
        asked: Filled in with how many times each question was asked, so the
            runner can report which answers were used.

    Returns:
        A function with the same call shape as ``input()``.
    """

    def answer(prompt: object = "") -> str:
        prompt_text = str(prompt)
        lowered = prompt_text.lower()
        for question, reply in answers:
            if question in lowered:
                asked[question] = asked.get(question, 0) + 1
                # A second time means the notebook rejected the answer (or
                # is looping after an error); stop instead of looping.
                if asked[question] > 1:
                    raise UnexpectedQuestion(
                        f"The question below was asked a second time, so the "
                        f"answer '{reply}' was not accepted:\n{prompt_text}")
                print(prompt_text)
                _say(f"answered: {reply}")
                return reply
        raise UnexpectedQuestion(
            f"A notebook asked a question this runner does not answer:\n"
            f"{prompt_text}")

    return answer


# ---------------------------------------------------------------------------
# Running the notebooks
# ---------------------------------------------------------------------------

def make_notebook_runner(shell: Any, memory: _MemoryWatch) -> Callable[..., None]:
    """Builds the replacement for IPython's notebook runner.

    IPython's ``%run`` hands a .ipynb file to ``shell.safe_execfile_ipy``,
    which runs its code cells one by one. This version does the same, in the
    same namespace and with the same settings, but stops at the first failing
    cell and raises NotebookCellFailed naming the notebook and the cell.

    Args:
        shell: The IPython shell running the notebooks.
        memory: Memory watch to tell which cell is running.

    Returns:
        A function with the signature of ``safe_execfile_ipy``.
    """
    import nbformat
    from IPython.utils.syspathcontext import prepended_to_syspath

    def run_notebook(
        fname: Any, shell_futures: bool = False, raise_exceptions: bool = False
    ) -> None:
        path = Path(fname).expanduser().resolve()
        notebook = nbformat.read(str(path), as_version=4)
        _say(f"start {path.name}")
        started = time.perf_counter()
        # The notebook's own folder goes on the import path while it runs,
        # as IPython's own runner does.
        with prepended_to_syspath(str(path.parent)):
            # Cells are numbered from 1, counting markdown and code cells, as
            # the notebook-cell-edit skill and the researcher count them.
            for cell_number, cell in enumerate(notebook.cells, start=1):
                if cell.cell_type != "code":
                    continue
                label = f"{path.name} cell {cell_number}"
                outer_label = memory.current_label
                memory.current_label = label
                cell_started = time.perf_counter()
                result = shell.run_cell(
                    cell.source, silent=True, shell_futures=shell_futures)
                memory.current_label = outer_label
                error = result.error_before_exec or result.error_in_exec
                if error is not None:
                    if isinstance(error, RunnerStop):
                        raise error
                    raise NotebookCellFailed(path.name, cell_number, error)
                rss_gb = memory.check() / 1e9
                _say(
                    f"{label} done in "
                    f"{time.perf_counter() - cell_started:,.1f} s "
                    f"(memory {rss_gb:.1f} GB)")
        _say(f"end {path.name} ({time.perf_counter() - started:,.1f} s)")

    return run_notebook


def _quiet_stop_handler(
    shell: Any, etype: type, value: BaseException, tb: Any, tb_offset: Any = None
) -> None:
    """Shows nothing for a stop that is passing up through an outer notebook.

    The inner notebook's error has already been printed in full where it
    happened; printing it again at every level only buries it.
    """
    return None


# ---------------------------------------------------------------------------
# Checks after the run
# ---------------------------------------------------------------------------

class CheckReport:
    """Collects check results and prints each one as it is made."""

    def __init__(self) -> None:
        self.failures: List[str] = []
        self.n_checks = 0

    def record(self, passed: bool, label: str, detail: str = "") -> None:
        self.n_checks += 1
        status = "[OK]  " if passed else "[FAIL]"
        print(f"  {status} {label}" + (f" -- {detail}" if detail else ""))
        if not passed:
            self.failures.append(f"{label}: {detail}")


def _energy_and_cost_columns(df: Any, menu_mp: int) -> List[str]:
    """Columns that must have a value for every study-sample home.

    Heating and cooling energy before and after the retrofit (every fuel and
    part), and every installed cost.

    Args:
        df: The household frame for one measure package.
        menu_mp: Its measure package number.

    Returns:
        The column names, in the frame's order.
    """
    import re
    energy_pattern = re.compile(
        rf"^(base|baseline|mp{menu_mp})_.*(heating|cooling).*consumption")
    cost_pattern = re.compile(rf"^mp{menu_mp}_.*_installed_cost_")
    return [
        column for column in df.columns
        if energy_pattern.match(column) or cost_pattern.match(column)]


def run_checks(
    user_ns: Dict[str, Any], output_folder: Path
) -> CheckReport:
    """Checks the tables the notebooks left in memory and the Tepper files.

    Args:
        user_ns: The notebooks' shared namespace after the run.
        output_folder: The run's output_results folder.

    Returns:
        The report; its failures list is empty when every check passed.
    """
    import numpy as np
    import pandas as pd
    from cmu_tare_model.constants import (
        PRIVATE_DISCOUNTING_METHOD_SUFFIXES,
        REMDB_COST_SCENARIO_KEYS,
    )
    from cmu_tare_model.utils.column_names import (
        NPV_CASE_CATEGORIES,
        create_adoption_col,
        create_capital_col,
        create_discounted_savings_col,
        create_npv_case_col,
    )
    from cmu_tare_model.utils.modeling_params import define_scenario_params

    report = CheckReport()
    print("\n" + "=" * 78)
    print("CHECKS AFTER THE RUN")
    print("=" * 78)

    dataframes_by_mp = user_ns.get("DATAFRAMES_BY_MP")
    if not dataframes_by_mp:
        report.record(False, "result tables", "DATAFRAMES_BY_MP is missing or empty")
        return report

    location_id = user_ns.get("location_id")
    run_stamp = user_ns.get("model_run_date_time")
    method_suffixes = {
        suffix.lstrip("_"): suffix
        for suffix in PRIVATE_DISCOUNTING_METHOD_SUFFIXES.values()}
    rebate_scenarios = ("unsub", "sub", "sub_june2026")

    for menu_mp, frames in dataframes_by_mp.items():
        scenario_prefix = define_scenario_params(menu_mp, "2025 Reference Case")[0]
        for discount_key, df in frames.items():
            method_suffix = method_suffixes[discount_key]
            print(f"\nMP{menu_mp}, discount rate {discount_key}:")
            in_sample = df["include_sample"].astype(bool)
            n_sample = int(in_sample.sum())
            weights = df["weight"].unique()
            report.record(
                len(weights) == 1, "one distinct weight",
                f"{len(weights)} distinct: {weights[:3]}")
            weight = float(weights[0])
            print(f"  study sample: {n_sample:,} rdu = "
                  f"{n_sample * weight:,.0f} homes (weight {weight!r})")

            # NPV identity, adopter flags, and adopter counts, per case
            heating_savings = df[create_discounted_savings_col(
                scenario_prefix, "heating", method_suffix)]
            cooling_savings = df[create_discounted_savings_col(
                scenario_prefix, "cooling", method_suffix)]
            npv_columns = []
            for cost_scenario in REMDB_COST_SCENARIO_KEYS:
                for npv_case in NPV_CASE_CATEGORIES:
                    npv_col = create_npv_case_col(
                        scenario_prefix, npv_case, method_suffix)
                    net_capital_col = create_capital_col(
                        scenario_prefix, npv_case, net=True,
                        cost_scenario=cost_scenario)
                    adopter_col = create_adoption_col(
                        scenario_prefix, npv_case, method_suffix)
                    npv_columns.append(npv_col)
                    npv = df[npv_col]
                    gap = (heating_savings + cooling_savings
                           - df[net_capital_col] - npv).abs()
                    within_tolerance = gap[in_sample] <= NPV_IDENTITY_TOLERANCE
                    n_identity = int((~within_tolerance).sum())
                    adopter = df[adopter_col]
                    expected = (npv >= 0).astype("float64").where(npv.notna())
                    n_flag = int((~((adopter == expected)
                                    | (adopter.isna() & expected.isna()))).sum())
                    n_nonblank = int(adopter.notna().sum())
                    report.record(
                        n_identity == 0 and n_flag == 0 and n_nonblank == n_sample
                        and adopter.dtype == "float64",
                        f"{npv_case}",
                        f"identity violations {n_identity}, flag != NPV>=0 "
                        f"{n_flag}, non-blank adopters {n_nonblank:,} "
                        f"(sample {n_sample:,}), dtype {adopter.dtype}, "
                        f"adoption {100 * adopter[in_sample].mean():.2f}%")

            # CLAUDE.md's two per-home ordering checks and the two county
            # adoption-rate checks, for each rebate policy scenario
            for scenario in rebate_scenarios:
                both = f"heatingLCC_coolingLCC_{scenario}"
                for other in (f"heatingLCC_coolingSavings_{scenario}",
                              f"heatingSavings_coolingLCC_{scenario}"):
                    npv_both = df.loc[in_sample, create_npv_case_col(
                        scenario_prefix, both, method_suffix)]
                    npv_other = df.loc[in_sample, create_npv_case_col(
                        scenario_prefix, other, method_suffix)]
                    n_order = int((~(npv_both >= npv_other)).sum())
                    adopt_both = df.loc[in_sample, create_adoption_col(
                        scenario_prefix, both, method_suffix)]
                    adopt_other = df.loc[in_sample, create_adoption_col(
                        scenario_prefix, other, method_suffix)]
                    county = df.loc[in_sample, "county"]
                    rate_both = adopt_both.groupby(county).mean()
                    rate_other = adopt_other.groupby(county).mean()
                    n_county = int((~(rate_both >= rate_other)).sum())
                    report.record(
                        n_order == 0 and n_county == 0,
                        f"{both} >= {other}",
                        f"per-home violations {n_order}, county violations "
                        f"{n_county} of {len(rate_both):,} counties")

            # No blank energy, cost or NPV value in a sample home
            required = _energy_and_cost_columns(df, menu_mp) + npv_columns
            blank_counts = df.loc[in_sample, required].isna().sum()
            blank_counts = blank_counts[blank_counts > 0]
            report.record(
                blank_counts.empty,
                f"no blank energy, cost or NPV value ({len(required)} columns)",
                "" if blank_counts.empty else str(blank_counts.to_dict()))

            # The Tepper household files have one row per sample home
            if discount_key != "fixed_base":
                continue
            for file_label in ("tepper_household", "tepper_household_detailed"):
                path = (output_folder / "tepper_export"
                        / f"{file_label}_mp{menu_mp}_{location_id}_{run_stamp}.csv")
                if not path.exists():
                    report.record(False, f"{path.name} exists", "not written")
                    continue
                ids = pd.read_csv(path, usecols=["bldg_id"])["bldg_id"]
                sample_ids = set(df.index[in_sample])
                report.record(
                    len(ids) == n_sample and ids.is_unique
                    and set(ids) == sample_ids,
                    f"{path.name} has one row per sample home",
                    f"{len(ids):,} rows, {n_sample:,} sample rdu")
            county_path = (output_folder / "tepper_export"
                           / f"tepper_county_mp{menu_mp}_{location_id}_{run_stamp}.csv")
            report.record(
                county_path.exists(), f"{county_path.name} exists",
                "" if county_path.exists() else "not written")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Reads the command-line options.

    Args:
        argv: Options to read; None reads sys.argv.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Run the TARE model notebooks with no keyboard input.")
    parser.add_argument(
        "--release", choices=["2022.1.1", "2025.1"], default=None,
        help="ResStock release for this run. Default: what constants.py "
             "says (2022.1.1 unless TARE_RESSTOCK_RELEASE is set).")
    parser.add_argument(
        "--state", default=None,
        help="Two-letter state code for a one-state run (answers Y to the "
             "state menu and N to the city menu). Default: the whole country.")
    parser.add_argument(
        "--fips", default="42003",
        help="County FIPS for the grid-impact case study. Default: 42003.")
    parser.add_argument(
        "--skip-grid-impact", action="store_true",
        help="Switch GRID_IMPACT_ANALYSIS off for this run (no AWS needed).")
    parser.add_argument(
        "--log", default=None,
        help="Also write everything printed to this file.")
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Do not run the checks after the run.")
    options = parser.parse_args(argv)

    if options.state is not None:
        options.state = options.state.strip().upper()
        if len(options.state) != 2 or not options.state.isalpha():
            parser.error(f"--state must be a two-letter code, got {options.state!r}")
    if len(options.fips) != 5 or not options.fips.isdigit():
        parser.error(f"--fips must be five digits, got {options.fips!r}")
    return options


def main(argv: Optional[List[str]] = None) -> int:
    """Runs the main notebook and the checks after it.

    Args:
        argv: Command-line options; None reads sys.argv.

    Returns:
        The exit code (see the module docstring).
    """
    options = parse_arguments(argv)

    log_file = None
    if options.log:
        log_file = open(options.log, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)

    # Step 1 -- settings that must be in place before any TARE import: the
    # release (constants.py reads it once, at import) and a drawing backend
    # that needs no screen.
    if options.release is not None:
        os.environ["TARE_RESSTOCK_RELEASE"] = options.release
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import cmu_tare_model.constants as constants
    if options.skip_grid_impact:
        # The main notebook imports this flag from constants.py, so setting it
        # here, before the notebook runs, switches its guarded cells off.
        constants.GRID_IMPACT_ANALYSIS = False

    _say(f"model code from {Path(constants.__file__).resolve().parent}")
    _say(f"release {constants.RESSTOCK_RELEASE_THIS_RUN}, packages "
         f"{constants.VALID_MENU_MPS}, scope "
         f"{options.state or 'National'}, grid impact "
         f"{'on, FIPS ' + options.fips if constants.GRID_IMPACT_ANALYSIS else 'off'}")

    # Step 2 -- one IPython shell, with input() answered from the options and
    # IPython's notebook runner replaced by one that stops at a failing cell.
    # The terminal shell, not IPython's base shell: the main notebook's
    # '%matplotlib inline' needs a shell that can switch drawing backends, and
    # the base shell raises NotImplementedError there. The shell never reads
    # the keyboard: its interactive loop is not started. 'nocolor' keeps
    # tracebacks readable in a log file.
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    shell = TerminalInteractiveShell.instance(colors="nocolor")
    memory = _MemoryWatch()
    shell.safe_execfile_ipy = make_notebook_runner(shell, memory)
    shell.set_custom_exc((RunnerStop,), _quiet_stop_handler)

    asked: Dict[str, int] = {}
    answer = make_input_answerer(build_answers(options.state, options.fips), asked)
    original_input = builtins.input
    builtins.input = answer
    shell.user_ns["input"] = answer

    # Step 3 -- run the main notebook, which runs the others
    exit_code = 0
    started = time.perf_counter()
    memory.start()
    try:
        # Forward slashes, no quotes, like the notebooks' own %run calls: on Windows
        # IPython fails on a quoted path holding '-m', as in 'cmu-tare-model'.
        shell.run_line_magic("run", f"-i {MAIN_NOTEBOOK.as_posix()}")
    except RunnerStop as stop:
        exit_code = stop.exit_code
        print("\n" + "=" * 78)
        print(f"RUN STOPPED: {stop}")
        print("=" * 78)
    finally:
        builtins.input = original_input
        memory.stop()
    wall_seconds = time.perf_counter() - started

    # Step 4 -- checks on what the run left in memory and on disk
    if exit_code == 0 and not options.skip_checks:
        report = run_checks(
            shell.user_ns, REPO_ROOT / "cmu_tare_model" / "output_results")
        if report.failures:
            exit_code = 3
            print(f"\n[FAIL] {len(report.failures)} of {report.n_checks} checks "
                  "failed:")
            for failure in report.failures:
                print(f"  - {failure}")
        else:
            print(f"\n[OK] All {report.n_checks} checks passed.")

    # Step 5 -- time and memory, whatever happened
    unanswered = [q for q, _ in build_answers(options.state, options.fips)
                  if q not in asked]
    os_peak = memory.operating_system_peak_bytes()
    print("\n" + "=" * 78)
    _say(f"wall time {wall_seconds / 60:,.1f} min ({wall_seconds:,.0f} s)")
    _say(f"peak memory {memory.peak_bytes / 1e9:.2f} GB (sampled), during "
         f"{memory.peak_label}")
    if os_peak is not None:
        _say(f"peak memory {os_peak / 1e9:.2f} GB (operating system record)")
    _say(f"questions answered: {asked}; not asked: {unanswered}")
    _say(f"exit code {exit_code}")
    if log_file is not None:
        sys.stdout.flush()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
