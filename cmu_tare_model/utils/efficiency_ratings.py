"""Converts heat-pump ratings between the 2023 test (SEER2, HSPF2) and the older one.

The REMDB v4 heat-pump cost regression was fitted on SEER1 ratings. A package
rated in SEER2 (the 2025.1 dual-fuel package) must be converted before it is
priced, or its cost comes out too low. The factors live in constants.py
(SEER2_PER_SEER1, HSPF2_PER_HSPF1) and are used only here.
"""
from typing import Union

import pandas as pd

from cmu_tare_model.constants import HSPF2_PER_HSPF1, SEER2_PER_SEER1

Rating = Union[float, pd.Series]


def seer2_to_seer1(seer2: Rating) -> Rating:
    """Converts a SEER2 rating to the SEER1 the REMDB regression takes.

    Args:
        seer2: One SEER2 rating, or a Series of them.

    Returns:
        The SEER1 rating(s): SEER2 / SEER2_PER_SEER1. SEER2 15.2 gives 16.0.
    """
    return seer2 / SEER2_PER_SEER1


def hspf2_to_hspf1(hspf2: Rating) -> Rating:
    """Converts an HSPF2 rating to HSPF1, for the record only.

    No cost in this model depends on HSPF; the converted value is carried so
    the package can be compared with the 2022.1.1 packages, which are rated
    in HSPF1.

    Args:
        hspf2: One HSPF2 rating, or a Series of them.

    Returns:
        The HSPF1 rating(s): HSPF2 / HSPF2_PER_HSPF1.
    """
    return hspf2 / HSPF2_PER_HSPF1
