from typing import NamedTuple


class PhysicalUnits(NamedTuple):
    """A collections of physical constants for units conversions."""

    BOHR_TO_ANGSTROM: float = 0.529177


units = PhysicalUnits()
