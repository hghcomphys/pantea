from typing import Dict, List, Mapping, Set, Tuple, Union

from jaxip.logger import logger
from jaxip.types import Element
from jaxip.units import units

# fmt: off
_KNOWN_ELEMENTS_LIST: Tuple = (
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

_KNOWN_ELEMENTS_DICT_MASS = {
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


class ElementMap:
    """
    Mapping element to atomic number utility class.

    It assigns an atomic type to each element which allows
    efficient array processing (e.g. applying conditions) using
    the arrays of integer (i.e. atom types) instead of strings.
    """

    def __init__(self, elements: List[Element]) -> None:
        """Initialize element map."""
        self.unique_elements: Set[Element] = set(elements)
        self._element_to_atomic_number: Dict[Element, int] = dict()
        self._element_to_atom_type: Dict[Element, int] = dict()
        self._atom_type_to_element: Dict[int, Element] = dict()
        self._create_mapping_dicts()
        logger.debug(f"Initialized {self.__class__.__name__}")

    def __call__(self, item: Union[Element, int]) -> Union[int, Element]:
        """Map an element to the atom type and vice versa."""
        result: Union[int, Element]
        if isinstance(item, int):
            return self._atom_type_to_element[item]
        elif isinstance(item, Element):
            return self._element_to_atom_type[item]
        else:
            logger.error(
                f"Unknown item type '{type(item)}'", exception=TypeError
            )
            return  # type: ignore

    def _create_mapping_dicts(self) -> None:
        """
        Create dictionary to map elements, atom types, and atomic numbers.
        The atom types are sorted based on elements' atomic number.
        """
        self._element_to_atomic_number = {
            elem: _KNOWN_ELEMENTS_DICT[elem] for elem in self.unique_elements
        }
        self._element_to_atom_type = {
            elem: atom_type
            for atom_type, elem in enumerate(
                sorted(
                    self._element_to_atomic_number,
                    key=self._element_to_atomic_number.get,  # type: ignore
                ),
                start=1,
            )
        }
        self._atom_type_to_element = {
            atom_type: elem
            for elem, atom_type in self._element_to_atom_type.items()
        }

    @classmethod
    def get_atomic_number(cls, element: Element) -> int:
        """
        Return atomic number of the input element.

        :param element: element name
        :return: atomic number
        """
        return _KNOWN_ELEMENTS_DICT[element]

    @classmethod
    def get_element(cls, atomic_number: int) -> Element:
        """
        Return element name of the given atomic number.

        :param atomic_number: atomic number
        :return: element name
        """
        return _KNOWN_ELEMENTS_LIST[atomic_number - 1]

    @property
    def atom_type_to_element(self) -> Dict[int, Element]:
        """
        Return a dictionary mapping of atom type to element.

        This property is defined due to a serialization issue
        of the ElementMap class during parallelization.

        :return: a dictionary of mapping an atom type (integer)
        to the corresponding element (string)
        """
        return self._atom_type_to_element

    @property
    def element_to_atom_type(self) -> Dict[Element, int]:
        """
        Return a mapping dictionary of element to atom type.

        This property is defined due to a serialization issue of
        the ElementMap class during parallelization.

        :return: a dictionary of mapping an element (string) to
        the corresponding atom type (integer)
        """
        return self._element_to_atom_type

    @classmethod
    def atomic_number_to_element(cls, atomic_number: int) -> Element:
        return _KNOWN_ELEMENTS_LIST[atomic_number - 1]

    @classmethod
    def element_to_atomic_number(cls, element: Element) -> int:
        return _KNOWN_ELEMENTS_DICT[element]

    @classmethod
    def element_to_atomic_mass(cls, element: Element) -> float:
        return _KNOWN_ELEMENTS_DICT_MASS[element] * units.FROM_ATOMIC_MASS
