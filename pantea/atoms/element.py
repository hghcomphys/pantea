from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Protocol, Sequence, Tuple

import jax
import jax.numpy as jnp

from pantea.logger import logger
from pantea.types import Array, Element
from pantea.units import units

# fmt: off
_KNOWN_ELEMENTS_LIST: Tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md",
)

_KNOWN_ELEMENTS_DICT_MASS: Dict[str, float] = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012,
    'B': 10.811, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.066,
    'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
    'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.723, 'Ge': 72.631,
    'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 84.798,
    'Rb': 84.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.95, 'Tc': 98.907, 'Ru': 101.07,
    'Rh': 102.906, 'Pd': 106.42, 'Ag': 107.868, 'Cd': 112.414,
    'In': 114.818, 'Sn': 118.711, 'Sb': 121.760, 'Te': 126.7,
    'I': 126.904, 'Xe': 131.294, 'Cs': 132.905, 'Ba': 137.328,
    'La': 138.905, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.243,
    'Pm': 144.913, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
    'Tb': 158.925, 'Dy': 162.500, 'Ho': 164.930, 'Er': 167.259,
    'Tm': 168.934, 'Yb': 173.055, 'Lu': 174.967, 'Hf': 178.49,
    'Ta': 180.948, 'W': 183.84, 'Re': 186.207, 'Os': 190.23,
    'Ir': 192.217, 'Pt': 195.085, 'Au': 196.967, 'Hg': 200.592,
    'Tl': 204.383, 'Pb': 207.2, 'Bi': 208.980, 'Po': 208.982,
    'At': 209.987, 'Rn': 222.081, 'Fr': 223.020, 'Ra': 226.025,
    'Ac': 227.028, 'Th': 232.038, 'Pa': 231.036, 'U': 238.029,
    'Np': 237, 'Pu': 244, 'An': 243, 'Cm': 247, 'Bk': 247,
    'Ct': 251, 'Es': 252, 'Fm': 257, 'Mf': 258, 'No': 259,
    'Lr': 262, 'Rf': 261, 'Db': 262, 'Sg': 266, 'Bh': 264,
    'Hs': 269, 'Mt': 268, 'Ds': 271, 'Rg': 272, 'Cn': 285,
    'Nh': 284, 'Fl': 289, 'Mc': 288, 'Lv': 292, 'Ts': 294,
    'Og': 294,
}
# fmt: on


_KNOWN_ELEMENTS_DICT: Mapping[Element, int] = {
    element: atomic_number
    for atomic_number, element in enumerate(_KNOWN_ELEMENTS_LIST, start=1)
}


class StructureInterface(Protocol):
    atom_types: Array
    element_map: ElementMap


@dataclass  # (frozen=True)
class ElementMap:
    """
    Mapping element name to atom type and more.

    It assigns an atomic type to each element which allows
    efficient array processing (e.g. applying conditions) using
    the arrays of integer (i.e. atom types) instead of strings.
    """

    unique_elements: Tuple[Element, ...]
    element_to_atomic_number: Dict[Element, int]
    element_to_atom_type: Dict[Element, int]
    atom_type_to_element: Dict[int, Element]

    @classmethod
    def from_list(cls, elements: Sequence[Element]) -> ElementMap:
        """
        Create dictionary to map elements, atom types, and atomic numbers.
        The atom types are sorted based on elements' atomic number.
        """
        logger.debug("Creating element map from list of elements")
        unique_elements: Tuple[Element, ...] = tuple(sorted(set(elements)))
        element_to_atomic_number = {
            elem: _KNOWN_ELEMENTS_DICT[elem] for elem in unique_elements
        }
        element_to_atom_type = {
            elem: atom_type
            for atom_type, elem in enumerate(
                sorted(
                    element_to_atomic_number,
                    key=element_to_atomic_number.get,  # type: ignore
                ),
                start=1,
            )
        }
        atom_type_to_element = {
            atom_type: elem for elem, atom_type in element_to_atom_type.items()
        }
        return cls(
            unique_elements,
            element_to_atomic_number,
            element_to_atom_type,
            atom_type_to_element,
        )

    def get_atom_type_from_element(self, name: Element) -> int:
        """Map element name to atom type."""
        return self.element_to_atom_type[name]

    def get_element_from_atom_type(self, value: int) -> Element:
        """Map atom type to element name."""
        return self.atom_type_to_element[value]

    @classmethod
    def get_element_from_atomic_number(cls, value: int) -> Element:
        return _KNOWN_ELEMENTS_LIST[value - 1]

    @classmethod
    def get_atomic_number_from_element(cls, name: Element) -> int:
        return _KNOWN_ELEMENTS_DICT[name]

    @classmethod
    def get_atomic_mass_from_element(cls, name: Element) -> float:
        return _KNOWN_ELEMENTS_DICT_MASS[name] * units.FROM_ATOMIC_MASS

    @classmethod
    def get_masses_from_structure(cls, structure: StructureInterface) -> Array:
        """Get array of atomic masses."""
        to_element = structure.element_map.atom_type_to_element
        atom_types_host = jax.device_get(structure.atom_types)
        elements = (to_element[at] for at in atom_types_host)
        return jnp.array(
            tuple(ElementMap.get_atomic_mass_from_element(name) for name in elements)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(element_to_atom_type={self.element_to_atom_type})"
