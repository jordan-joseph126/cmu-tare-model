"""Release-aware ResStock column names.

RESSTOCK_COLUMN_MAP[release][logical_name] gives the physical column name to
read for that release. It is built from the Phase 1 column map
(cmu_tare_model/docs/resstock_2025_1_column_map.csv), so a rename is fixed in
one CSV row instead of in every place that reads the column.
"""
import os
from typing import Dict

import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import RESSTOCK_RELEASE_AND_MP

_COLUMN_MAP_PATH = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "docs", "resstock_2025_1_column_map.csv")

# CSV column holding each release's physical name.
_RELEASE_NAME_COLS = {
    '2022.1.1': 'name_2022_1_1',
    '2025.1': 'name_2025_1',
}


def _build_column_map(path: str) -> Dict[str, Dict[str, str]]:
    """Builds {release: {logical_name: physical_name}} from the column map CSV.

    A release gets every row that has a name for that release. So 2022.1.1
    includes the 'missing' rows (columns that exist only in 2022.1.1, all on
    inactive paths today), and 2025.1 includes the 'new' rows.

    Args:
        path: Path to the column map CSV.

    Returns:
        Nested dict keyed by release, then logical name.

    Raises:
        ValueError: If the CSV has a duplicate logical name, a duplicate
            physical name within a release, or a release not listed in
            RESSTOCK_RELEASE_AND_MP.
    """
    df_map = pd.read_csv(path)

    # Step 1 -- validate: a repeated key would silently overwrite a mapping
    duplicated_logical = df_map.loc[
        df_map['logical_name'].duplicated(), 'logical_name'].tolist()
    if duplicated_logical:
        raise ValueError(
            f"Duplicate logical_name in column map: {duplicated_logical}")
    unknown_releases = set(_RELEASE_NAME_COLS) - set(RESSTOCK_RELEASE_AND_MP)
    if unknown_releases:
        raise ValueError(
            f"Releases {sorted(unknown_releases)} are not in "
            f"RESSTOCK_RELEASE_AND_MP {sorted(RESSTOCK_RELEASE_AND_MP)}")

    # Step 2 -- one {logical: physical} dict per release
    column_map = {}
    for release, name_col in _RELEASE_NAME_COLS.items():
        df_release = df_map.dropna(subset=[name_col])
        duplicated_physical = df_release.loc[
            df_release[name_col].duplicated(), name_col].tolist()
        if duplicated_physical:
            raise ValueError(
                f"Duplicate {name_col} in column map: {duplicated_physical}")
        column_map[release] = dict(
            zip(df_release['logical_name'], df_release[name_col]))
    return column_map


RESSTOCK_COLUMN_MAP = _build_column_map(_COLUMN_MAP_PATH)


def resstock_col(release: str, logical_name: str) -> str:
    """Returns the physical column name for a logical name in one release.

    Args:
        release: ResStock release key, e.g. '2022.1.1' or '2025.1'.
        logical_name: Logical name from the column map, e.g. 'heating_fuel'.

    Returns:
        The column name to read from that release's file.

    Raises:
        KeyError: If the release is unknown, or the column does not exist in
            that release (for example a 2022.1.1-only column requested for
            2025.1).
    """
    if release not in RESSTOCK_COLUMN_MAP:
        raise KeyError(
            f"Unknown ResStock release '{release}'; expected one of "
            f"{sorted(RESSTOCK_COLUMN_MAP)}")
    if logical_name not in RESSTOCK_COLUMN_MAP[release]:
        raise KeyError(
            f"'{logical_name}' has no column in ResStock {release}")
    return RESSTOCK_COLUMN_MAP[release][logical_name]
