from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field


_optional_attr = field(default=None, repr=False)
if TYPE_CHECKING:
    Float = Any
else:
    Float = float


@dataclass
class PhysicalUnits:
    """Physical constants for units conversions."""

    TO_ANGSTROM: float
    TO_PICO_SECOND: float
    TO_BAR: float
    TO_ELECTRON_VOLT: float
    BOLTZMANN_CONSTANT: float
    TO_NANO_METER: Float = _optional_attr
    TO_FEMTO_SECOND: Float = _optional_attr
    TO_NANO_SECOND: Float = _optional_attr
    TO_GIGA_PASCAL: Float = _optional_attr
    TO_PASCAL: Float = _optional_attr
    TO_ATMOSPHERE: Float = _optional_attr
    TO_KILO_BAR: Float = _optional_attr
    FROM_ANGSTROM: Float = _optional_attr
    FROM_NANO_METER: Float = _optional_attr
    FROM_FEMTO_SECOND: Float = _optional_attr
    FROM_PICO_SECOND: Float = _optional_attr
    FROM_NANO_SECOND: Float = _optional_attr
    FROM_GIGA_PASCAL: Float = _optional_attr
    FROM_KILO_BAR: Float = _optional_attr
    FROM_BAR: Float = _optional_attr
    FROM_ELECTRON_VOLT: Float = _optional_attr

    def __post_init__(self) -> None:
        self.TO_NANO_METER = self.TO_ANGSTROM * 0.1
        self.TO_FEMTO_SECOND = self.TO_PICO_SECOND * 1000
        self.TO_NANO_SECOND = self.TO_PICO_SECOND * 0.001
        self.TO_GIGA_PASCAL = self.TO_BAR * 0.0001
        self.TO_PASCAL = self.TO_BAR * 100000
        self.TO_ATMOSPHERE = self.TO_BAR * 0.986923
        self.TO_KILO_BAR = self.TO_BAR * 0.001
        self.FROM_ANGSTROM = 1 / self.TO_ANGSTROM
        self.FROM_NANO_METER = 1 / self.TO_NANO_METER
        self.FROM_FEMTO_SECOND = 1 / self.TO_FEMTO_SECOND
        self.FROM_PICO_SECOND = 1 / self.TO_PICO_SECOND
        self.FROM_NANO_SECOND = 1 / self.TO_NANO_SECOND
        self.FROM_GIGA_PASCAL = 1 / self.TO_GIGA_PASCAL
        self.FROM_KILO_BAR = 1 / self.TO_KILO_BAR
        self.FROM_BAR = 1 / self.TO_BAR
        self.FROM_ELECTRON_VOLT = 1 / self.TO_ELECTRON_VOLT


hartree_units = PhysicalUnits(
    TO_ANGSTROM=5.29177249e-01,
    TO_PICO_SECOND=2.418884326e-03,
    TO_BAR=2.942102648e08,
    TO_ELECTRON_VOLT=2.7211407953e01,
    BOLTZMANN_CONSTANT=3.166811563e-6,
)


units = hartree_units
