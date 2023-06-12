from typing import NamedTuple

PhysicalUnits = NamedTuple


class HartreeAtomicUnits(PhysicalUnits):
    """A collections of physical constants for Hartree units conversions."""

    KB: float = 3.166811563e-6  # Hartree/K

    TO_ANGSTROM: float = 0.529177
    TO_FEMTO_SECOND: float = 2.418884326
    TO_PICO_SECOND: float = 2.418884326e-03
    TO_NANO_SECOND: float = 2.418884326e-06
    TO_GIGA_PASCAL: float = 2.942102648e04
    TO_KILO_BAR: float = 2.942102648e5
    TO_BAR: float = 2.942102648e08
    TO_ELECTRON_VOLT: float = 27.2114

    FROM_ANGSTROM: float = 1.889726
    FROM_FEMTO_SECOND: float = 41.3413733
    FROM_PICO_SECOND: float = 41.3413733e-03
    FROM_NANO_SECOND: float = 41.3413733e-06
    FROM_GIGA_PASCAL: float = 3.398929675e-05
    FROM_KILO_BAR: float = 3.398929675e-06
    FROM_BAR: float = 3.398929675e-09
    FROM_ELECTRON_VOLT: float = 0.0367493


units = HartreeAtomicUnits()
