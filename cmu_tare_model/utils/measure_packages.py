"""What kind of retrofit a measure package is, for the release being run.

Package numbers repeat across ResStock releases (2025.1 Upgrades 03 and 04 will
later load as mp=3 and mp=4, which are different packages from 2022.1.1's MP3
and MP4), so every question about a package is answered from the release and
the package number together, in one place.
"""
from typing import Optional

from cmu_tare_model.constants import (
    DUAL_FUEL_PACKAGES_BY_RELEASE,
    RESSTOCK_RELEASE_AND_MP,
    RESSTOCK_RELEASE_THIS_RUN,
)


def is_dual_fuel_package(menu_mp: int, release: Optional[str] = None) -> bool:
    """Says whether a measure package installs a dual-fuel heating system.

    A dual-fuel package keeps a fossil furnace as the heat pump's backup. Its
    retrofit therefore has a furnace cost on top of the heat pump's, burns gas
    after the retrofit, and passes the June 2026 rebate fuel gates whatever the
    home's heating fuel was before (NEXT_STEPS D8; researcher's decision R3).

    Args:
        menu_mp: Measure package number (0 is the baseline, never dual fuel).
        release: ResStock release. None uses RESSTOCK_RELEASE_THIS_RUN.

    Returns:
        True only if the package is listed as dual fuel for that release in
        DUAL_FUEL_PACKAGES_BY_RELEASE (constants.py).

    Raises:
        ValueError: If the release is not a known release.
    """
    if release is None:
        release = RESSTOCK_RELEASE_THIS_RUN
    if release not in RESSTOCK_RELEASE_AND_MP:
        raise ValueError(
            f"Unknown ResStock release '{release}'; expected one of "
            f"{sorted(RESSTOCK_RELEASE_AND_MP)}")
    return int(menu_mp) in DUAL_FUEL_PACKAGES_BY_RELEASE.get(release, [])
