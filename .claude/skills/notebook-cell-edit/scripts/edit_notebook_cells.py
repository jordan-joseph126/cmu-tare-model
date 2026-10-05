"""Change named lines inside Jupyter notebook cells, and nothing else.

Commands:
    list  NOTEBOOK [--grep TEXT]   One row per cell, or only cells holding TEXT.
    show  NOTEBOOK CELL [--json]   Print one cell with line numbers.
    check CHANGES.json             Dry run: show what would change. Writes nothing.
    apply CHANGES.json             Make the changes, after the same checks.

Cells are numbered from 1, counting markdown and code cells together. Lines are
numbered from 1 inside a cell. Both always refer to the notebook as it is on
disk before the run, so several changes in one cell never shift each other.

A changes file looks like this ("new" is the new line, null to delete the
line, or a list of lines to put in its place):

    {"changes": [
        {"notebook": "cmu_tare_model/model_scenarios/tare_scenarios_v3_0.ipynb",
         "cell": 16, "line": 14, "old": "VERBOSE = True", "new": null}
    ]}

Standard library only; runs on Python 3.9 or later.
"""

import argparse
import copy
import datetime
import difflib
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple


def read_notebook(notebook_path: str) -> Tuple[bytes, Dict[str, Any]]:
    """Reads a notebook file as raw bytes and as parsed JSON.

    Args:
        notebook_path: Path to the .ipynb file.

    Returns:
        The file's bytes and the parsed notebook.

    Raises:
        ValueError: If the path is not an .ipynb file, is a checkpoint copy,
            or holds no list of cells.
        FileNotFoundError: If the file does not exist.
    """
    if not notebook_path.endswith(".ipynb"):
        raise ValueError(f"{notebook_path}: not an .ipynb file")
    if ".ipynb_checkpoints" in notebook_path.replace("\\", "/"):
        raise ValueError(
            f"{notebook_path}: this is a checkpoint copy; edit the notebook itself")
    with open(notebook_path, "rb") as handle:
        raw_bytes = handle.read()
    notebook = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{notebook_path}: no list of cells found")
    return raw_bytes, notebook


def serialize_notebook(notebook: Dict[str, Any], layout: Dict[str, Any]) -> bytes:
    """Turns a parsed notebook back into file bytes using one layout.

    Args:
        notebook: The parsed notebook.
        layout: Keys 'indent', 'ensure_ascii', 'final_newline', 'line_ending'.

    Returns:
        The bytes to write to disk.
    """
    text = json.dumps(
        notebook, indent=layout["indent"], ensure_ascii=layout["ensure_ascii"])
    if layout["final_newline"]:
        text += "\n"
    if layout["line_ending"] != "\n":
        text = text.replace("\n", layout["line_ending"])
    return text.encode("utf-8")


def find_write_layout(raw_bytes: bytes, notebook: Dict[str, Any]) -> Dict[str, Any]:
    """Finds the layout that writes the unchanged notebook back byte for byte.

    This is the check that makes a line-level edit safe: if the unchanged
    notebook comes back identical, then after the edit the only bytes that
    differ are the lines that were changed on purpose.

    Args:
        raw_bytes: The notebook file as it is on disk.
        notebook: The same file, parsed.

    Returns:
        The first layout that reproduces raw_bytes exactly.

    Raises:
        ValueError: If no layout reproduces the file.
    """
    line_ending = "\r\n" if b"\r\n" in raw_bytes else "\n"
    for indent in (1, 2, 4):
        for ensure_ascii in (False, True):
            for final_newline in (True, False):
                layout = {
                    "indent": indent,
                    "ensure_ascii": ensure_ascii,
                    "final_newline": final_newline,
                    "line_ending": line_ending,
                }
                if serialize_notebook(notebook, layout) == raw_bytes:
                    return layout
    raise ValueError(
        "the file cannot be written back byte for byte, so an edit would "
        "change more than the named lines")


def get_cell(
    notebook: Dict[str, Any],
    cell_position: int,
    label: str,
) -> Dict[str, Any]:
    """Returns one cell by its position, counted from 1.

    Args:
        notebook: The parsed notebook.
        cell_position: Position among all cells, markdown and code together.
        label: Notebook name for the error message.

    Returns:
        The cell.

    Raises:
        ValueError: If there is no cell at that position.
    """
    n_cells = len(notebook["cells"])
    if not 1 <= cell_position <= n_cells:
        raise ValueError(
            f"{label}: no cell {cell_position}; the notebook has {n_cells} cells")
    return notebook["cells"][cell_position - 1]


def get_cell_text(cell: Dict[str, Any]) -> str:
    """Returns a cell's source as one string, whichever way it is stored."""
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def split_into_source_entries(cell_text: str) -> List[str]:
    """Splits cell text the way Jupyter stores it: one entry per line.

    Every entry keeps its line break except the last. Splitting on the line
    break only (not str.splitlines) keeps other break-like characters inside
    a line where they were.

    Args:
        cell_text: The cell's full text.

    Returns:
        The list to store as the cell's 'source'.
    """
    parts = cell_text.split("\n")
    entries = [part + "\n" for part in parts[:-1]]
    if parts[-1] != "":
        entries.append(parts[-1])
    return entries


def compiles_as_python(cell_text: str) -> bool:
    """Says whether a code cell's text is valid Python.

    Tried first as written. If that fails, tried again with IPython-only
    lines (starting with % or !) set aside, since those are not Python.

    Args:
        cell_text: The cell's full text.

    Returns:
        True if either try compiles.
    """
    plain_lines = []
    for line in cell_text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            plain_lines.append(line[:len(line) - len(stripped)] + "pass")
        else:
            plain_lines.append(line)
    for candidate in (cell_text, "\n".join(plain_lines)):
        try:
            compile(candidate, "<cell>", "exec")
            return True
        except (SyntaxError, ValueError):
            continue
    return False


def load_changes(changes_path: str) -> List[Dict[str, Any]]:
    """Reads and validates a changes file.

    Args:
        changes_path: Path to the JSON file described in the module docstring.

    Returns:
        The list of change entries, each with 'new' as None or a list of lines.

    Raises:
        ValueError: If the file or any entry is not laid out as expected, or
            two entries name the same line.
    """
    with open(changes_path, encoding="utf-8") as handle:
        document = json.load(handle)
    entries = document.get("changes") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{changes_path}: expected a non-empty 'changes' list")

    seen_lines = set()
    changes = []
    for entry_number, entry in enumerate(entries, start=1):
        where = f"{changes_path}, change {entry_number}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected an object, got {type(entry).__name__}")
        missing = [key for key in ("notebook", "cell", "line", "old", "new")
                   if key not in entry]
        if missing:
            raise ValueError(f"{where}: missing {missing}")
        for key in ("cell", "line"):
            is_whole_number = isinstance(entry[key], int) and not isinstance(
                entry[key], bool)
            if not is_whole_number or entry[key] < 1:
                raise ValueError(
                    f"{where}: '{key}' must be a whole number from 1, "
                    f"got {entry[key]!r}")
        if not isinstance(entry["notebook"], str) or not isinstance(entry["old"], str):
            raise ValueError(f"{where}: 'notebook' and 'old' must be text")

        new_value = entry["new"]
        if new_value is None:
            new_lines: Optional[List[str]] = None
        elif isinstance(new_value, str):
            new_lines = [new_value]
        elif isinstance(new_value, list) and all(
                isinstance(line, str) for line in new_value):
            new_lines = list(new_value)
        else:
            raise ValueError(
                f"{where}: 'new' must be a line of text, a list of lines, or null")
        for line in [entry["old"]] + (new_lines or []):
            if "\n" in line or "\r" in line:
                raise ValueError(
                    f"{where}: a line cannot hold a line break; "
                    "give several lines as a list in 'new'")

        key = (os.path.normcase(os.path.abspath(entry["notebook"])),
               entry["cell"], entry["line"])
        if key in seen_lines:
            raise ValueError(
                f"{where}: cell {entry['cell']} line {entry['line']} is named twice")
        seen_lines.add(key)
        changes.append({
            "notebook": entry["notebook"],
            "cell": entry["cell"],
            "line": entry["line"],
            "old": entry["old"],
            "new": new_lines,
        })
    return changes


def plan_notebook(
    notebook_path: str,
    notebook_changes: List[Dict[str, Any]],
    skip_python_check: bool = False,
) -> Dict[str, Any]:
    """Works out one notebook's new contents without writing anything.

    Args:
        notebook_path: Path to the .ipynb file.
        notebook_changes: The change entries for this notebook.
        skip_python_check: If True, a code cell that stops being valid Python
            does not stop the run.

    Returns:
        Keys 'path', 'layout', 'new_bytes', and 'cells' (one record per changed
        cell: position, type, old lines, new lines).

    Raises:
        ValueError: If the file cannot be written back byte for byte, a named
            line does not read as expected, a code cell stops being valid
            Python, or the planned file differs anywhere it should not.
    """
    raw_bytes, notebook = read_notebook(notebook_path)
    try:
        layout = find_write_layout(raw_bytes, notebook)
    except ValueError as error:
        raise ValueError(f"{notebook_path}: {error}") from error

    # Step 1 -- group the changes by cell
    changes_by_cell: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for change in notebook_changes:
        changes_by_cell.setdefault(change["cell"], {})[change["line"]] = change

    # Step 2 -- build each changed cell's new text from its text today
    new_notebook = copy.deepcopy(notebook)
    cell_records = []
    for cell_position in sorted(changes_by_cell):
        line_changes = changes_by_cell[cell_position]
        label = f"{notebook_path} cell {cell_position}"
        cell = get_cell(new_notebook, cell_position, notebook_path)
        old_lines = get_cell_text(cell).split("\n")
        if max(line_changes) > len(old_lines):
            raise ValueError(
                f"{label}: no line {max(line_changes)}; "
                f"the cell has {len(old_lines)} lines")

        new_lines: List[str] = []
        for line_number, line_text in enumerate(old_lines, start=1):
            if line_number not in line_changes:
                new_lines.append(line_text)
                continue
            change = line_changes[line_number]
            # The line must read exactly as expected, so a notebook that has
            # moved on since it was last looked at is never edited blind.
            if line_text != change["old"]:
                raise ValueError(
                    f"{label} line {line_number} does not read as expected.\n"
                    f"  expected: {change['old']!r}\n"
                    f"  found:    {line_text!r}")
            if change["new"] is not None:
                new_lines.extend(change["new"])

        new_text = "\n".join(new_lines)
        if (cell.get("cell_type") == "code" and not skip_python_check
                and compiles_as_python("\n".join(old_lines))
                and not compiles_as_python(new_text)):
            raise ValueError(
                f"{label}: the cell is valid Python today and would not be "
                "after the change")
        stored_as_text = isinstance(cell.get("source", ""), str)
        cell["source"] = (
            new_text if stored_as_text else split_into_source_entries(new_text))
        cell_records.append({
            "position": cell_position,
            "cell_type": cell.get("cell_type"),
            "old_lines": old_lines,
            "new_lines": new_lines,
        })

    # Step 3 -- confirm the planned file differs only in those cells' text
    new_bytes = serialize_notebook(new_notebook, layout)
    reread = json.loads(new_bytes.decode("utf-8"))
    without_cells = {key: value for key, value in notebook.items() if key != "cells"}
    reread_without_cells = {
        key: value for key, value in reread.items() if key != "cells"}
    if without_cells != reread_without_cells or len(reread["cells"]) != len(
            notebook["cells"]):
        raise ValueError(f"{notebook_path}: something outside the cells would change")
    new_text_by_cell = {
        record["position"]: "\n".join(record["new_lines"]) for record in cell_records}
    for cell_position, (old_cell, new_cell) in enumerate(
            zip(notebook["cells"], reread["cells"]), start=1):
        if cell_position not in new_text_by_cell:
            if old_cell != new_cell:
                raise ValueError(
                    f"{notebook_path}: cell {cell_position} would change "
                    "although no change names it")
            continue
        old_rest = {key: value for key, value in old_cell.items() if key != "source"}
        new_rest = {key: value for key, value in new_cell.items() if key != "source"}
        if (old_rest != new_rest
                or get_cell_text(new_cell) != new_text_by_cell[cell_position]):
            raise ValueError(
                f"{notebook_path}: cell {cell_position} would not come out "
                "as planned")

    return {
        "path": notebook_path,
        "layout": layout,
        "new_bytes": new_bytes,
        "cells": cell_records,
    }


def print_plan(plan: Dict[str, Any]) -> None:
    """Prints one notebook's planned changes as a diff per changed cell."""
    layout = plan["layout"]
    line_ending_name = "CRLF" if layout["line_ending"] == "\r\n" else "LF"
    print("=" * 78)
    print(plan["path"])
    print(
        f"  writes back byte for byte: yes (indent {layout['indent']}, "
        f"{line_ending_name} line endings)")
    for record in plan["cells"]:
        n_old = len(record["old_lines"])
        n_new = len(record["new_lines"])
        print(
            f"\n  cell {record['position']} ({record['cell_type']}): "
            f"{n_old} --> {n_new} lines")
        diff_lines = list(difflib.unified_diff(
            record["old_lines"], record["new_lines"], lineterm="", n=1))[2:]
        for diff_line in diff_lines:
            print(f"    {diff_line}")
        non_ascii_lines = [
            line for line in record["new_lines"]
            if line not in record["old_lines"] and not line.isascii()]
        if non_ascii_lines:
            print(
                f"    NOTE: {len(non_ascii_lines)} new line(s) hold non-ASCII "
                "characters")


def run_list(notebook_path: str, grep_text: Optional[str]) -> None:
    """Prints one row per cell, or only the cells and lines holding grep_text."""
    _, notebook = read_notebook(notebook_path)
    print(f"{notebook_path}: {len(notebook['cells'])} cells")
    for cell_position, cell in enumerate(notebook["cells"], start=1):
        lines = get_cell_text(cell).split("\n")
        first_line = lines[0][:70] if lines else ""
        header = (
            f"cell {cell_position:>3}  {cell.get('cell_type', '?'):<8} "
            f"{len(lines):>4} lines  | {first_line}")
        if grep_text is None:
            print(header)
            continue
        matches = [
            (line_number, line) for line_number, line in enumerate(lines, start=1)
            if grep_text in line]
        if matches:
            print(header)
            for line_number, line in matches:
                print(f"      line {line_number:>4}: {line}")


def run_show(notebook_path: str, cell_position: int, as_json: bool) -> None:
    """Prints one cell with line numbers.

    With as_json, each line is printed as a JSON string, ready to copy into
    the 'old' field of a changes file (quotes and backslashes already escaped,
    and any trailing spaces visible).
    """
    _, notebook = read_notebook(notebook_path)
    cell = get_cell(notebook, cell_position, notebook_path)
    lines = get_cell_text(cell).split("\n")
    print(
        f"{notebook_path} cell {cell_position} "
        f"({cell.get('cell_type', '?')}, {len(lines)} lines)")
    for line_number, line in enumerate(lines, start=1):
        shown = json.dumps(line, ensure_ascii=False) if as_json else line
        print(f"{line_number:>4}  {shown}")


def run_changes(
    changes_path: str,
    write: bool,
    backup_dir: Optional[str],
    skip_python_check: bool,
) -> None:
    """Plans every change, prints the diffs, and writes only if asked.

    Every notebook is planned and checked before any is written, so a problem
    with one leaves all of them untouched.

    Args:
        changes_path: Path to the changes file.
        write: False for a dry run.
        backup_dir: Where to copy the notebooks before writing. A dated folder
            in the system temp folder if None.
        skip_python_check: Passed to plan_notebook.

    Raises:
        ValueError: From load_changes or plan_notebook; nothing is written.
    """
    changes = load_changes(changes_path)
    changes_by_notebook: Dict[str, List[Dict[str, Any]]] = {}
    for change in changes:
        changes_by_notebook.setdefault(change["notebook"], []).append(change)

    plans = [
        plan_notebook(notebook_path, notebook_changes, skip_python_check)
        for notebook_path, notebook_changes in changes_by_notebook.items()]
    for plan in plans:
        print_plan(plan)
    n_cells = sum(len(plan["cells"]) for plan in plans)
    print("=" * 78)
    summary = (
        f"{len(changes)} line change(s) in {n_cells} cell(s) "
        f"of {len(plans)} notebook(s)")

    if not write:
        print(f"DRY RUN: {summary}. Nothing written.")
        return

    if backup_dir is None:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_dir = os.path.join(
            tempfile.gettempdir(), "notebook_cell_edit_backups", stamp)
    os.makedirs(backup_dir, exist_ok=True)
    for plan_number, plan in enumerate(plans, start=1):
        backup_name = f"{plan_number}_{os.path.basename(plan['path'])}"
        shutil.copy2(plan["path"], os.path.join(backup_dir, backup_name))
    for plan in plans:
        with open(plan["path"], "wb") as handle:
            handle.write(plan["new_bytes"])
    print(f"WRITTEN: {summary}.")
    print(f"Copies of the notebooks as they were: {backup_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Change named lines inside Jupyter notebook cells.")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="one row per cell")
    list_parser.add_argument("notebook")
    list_parser.add_argument(
        "--grep", default=None, help="show only cells and lines holding this text")

    show_parser = commands.add_parser("show", help="print one cell with line numbers")
    show_parser.add_argument("notebook")
    show_parser.add_argument("cell", type=int)
    show_parser.add_argument(
        "--json", action="store_true",
        help="print each line as a JSON string, ready for a changes file")

    for name, help_text in (("check", "dry run; writes nothing"),
                            ("apply", "make the changes")):
        change_parser = commands.add_parser(name, help=help_text)
        change_parser.add_argument("changes_file")
        change_parser.add_argument(
            "--skip-python-check", action="store_true",
            help="do not stop when a code cell stops being valid Python")
        if name == "apply":
            change_parser.add_argument(
                "--backup-dir", default=None,
                help="where to copy the notebooks before writing")
    return parser


def main() -> int:
    """Runs one command. Returns 0 on success and 1 on a stop."""
    # Notebook text can hold characters the Windows console cannot print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "list":
            run_list(arguments.notebook, arguments.grep)
        elif arguments.command == "show":
            run_show(arguments.notebook, arguments.cell, arguments.json)
        else:
            run_changes(
                arguments.changes_file,
                write=(arguments.command == "apply"),
                backup_dir=getattr(arguments, "backup_dir", None),
                skip_python_check=arguments.skip_python_check,
            )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        print(f"STOPPED, nothing written: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
