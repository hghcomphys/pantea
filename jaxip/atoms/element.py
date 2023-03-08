from typing import Dict, List, Mapping, Set, Tuple, Union

from jaxip.logger import logger
from jaxip.types import Element

_KNOWN_ELEMENTS_LIST: Tuple = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
)

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

    # FIXME: issue with hashing of static values (jit compilation)

    def __init__(self, elements: List[Element]) -> None:
        """Initialize element map."""
        self.unique_elements: Set[Element] = set(elements)
        self._elem_to_atomic_num: Dict[Element, int] = dict()
        self._elem_to_atom_type: Dict[Element, int] = dict()
        self._atom_type_to_elem: Dict[int, Element] = dict()
        self._create_mapping_dicts()

        logger.debug(f"Initializing {self.__class__.__name__}()")

    def _create_mapping_dicts(self) -> None:
        """
        Create dictionary to map elements, atom types, and atomic numbers.
        The given atom types are given and sorted based on elements' atomic number.
        """
        self._elem_to_atomic_num = {
            elem: _KNOWN_ELEMENTS_DICT[elem] for elem in self.unique_elements
        }
        self._elem_to_atom_type = {
            elem: atom_type
            for atom_type, elem in enumerate(
                sorted(
                    self._elem_to_atomic_num,
                    key=self._elem_to_atomic_num.get,  # type: ignore
                ),
                start=1,
            )
        }
        self._atom_type_to_elem = {
            atom_type: elem for elem, atom_type in self._elem_to_atom_type.items()
        }

    def __getitem__(self, item: Union[Element, int]) -> Union[int, Element]:
        """Map an element to the atom type and vice versa."""
        result: Union[int, Element]
        if isinstance(item, int):
            result = self._atom_type_to_elem[item]
        elif isinstance(item, Element):
            result = self._elem_to_atom_type[item]
        else:
            logger.error(f"Unknown item type '{type(item)}'", exception=TypeError)
        return result  # type: ignore

    def __call__(self, item: Union[Element, int]) -> Union[int, Element]:
        return self[item]

    @staticmethod
    def get_atomic_number(element: Element) -> int:
        """
        Return atomic number of the input element.

        :param element: element name
        :return: atomic number
        """
        return _KNOWN_ELEMENTS_DICT[element]

    @staticmethod
    def get_element(atomic_number: int) -> Element:
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
        This property is defined due to a serialization issue of the ElementMap class during parallelization.

        :return: a dictionary of mapping an atom type (integer) to the corresponding element (string)
        """
        return self._atom_type_to_elem

    @property
    def element_to_atype(self) -> Dict[Element, int]:
        """
        Return a mapping dictionary of element to atom type.
        This property is defined due to a serialization issue of the ElementMap class during parallelization.

        :return: a dictionary of mapping an element (string) to the corresponding atom type (integer)
        """
        return self._elem_to_atom_type

    @staticmethod
    def atomic_number_to_element(atomic_number: int) -> Element:
        return _KNOWN_ELEMENTS_LIST[atomic_number - 1]

    @staticmethod
    def element_to_atomic_number(element: Element) -> int:
        return _KNOWN_ELEMENTS_DICT[element]
