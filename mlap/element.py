from typing import Union, List


_KNOWN_ELEMENTS_LIST = [
  "H" , "He", "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne", "Na", "Mg", "Al",
  "Si", "P" , "S" , "Cl", "Ar", "K" , "Ca", "Sc", "Ti", "V" , "Cr", "Mn", "Fe",
  "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y" ,
  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
  "I" , "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
  "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W" , "Re", "Os", "Ir", "Pt",
  "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
  "U" , "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
]

_KNOWN_ELEMENTS_DICT = {element:atomic_number \
  for atomic_number, element in enumerate(_KNOWN_ELEMENTS_LIST, start=1) 
}


class ElementMap:
  """
  This class maps elements to atomic number end vice versa. 
  Also assign an atomic type to each element. 
  """
  
  def __init__(self, elements: List[str] = None) -> None:
    self.clear_maps()
    if elements is not None:
      self.unique_elements = set(elements)
      self.create_maps()

  def insert(self, element: str) -> None:
    self.unique_elements.add( element )

  def create_maps(self) -> None:
    """
    Create dictionary to map element, atom type, and atomic numbers.
    The given atom types are set based on elements' atomic number.
    """
    self._elem_to_atomic_num = {elem:_KNOWN_ELEMENTS_DICT[elem] \
      for elem in self.unique_elements}
    self._elem_to_atom_type = {elem:atom_type \
      for atom_type, elem in enumerate(sorted(self._elem_to_atomic_num, key=self._elem_to_atomic_num.get), start=1)}
    self._atom_type_to_elem = {atom_type:elem \
      for elem, atom_type in self._elem_to_atom_type.items()}

  def clear_maps(self) -> None:
    self.unique_elements = set()
    self._elem_to_atomic_num = None
    self._elem_to_atom_type = None
    self._atom_type_to_elem = None

  def __getitem__(self, item: Union[str, int]) -> Union[int, str]:
    """
    Map an element to the atom type and vice versa.
    """
    if isinstance(item, int):
      return self._atom_type_to_elem[item]
    elif isinstance(item, str):
      return self._elem_to_atom_type[item]
    else:
      #TODO: log and raise error
      raise TypeError(f"Unknown item type '{type(item)}'")

  def __str__(self):
    pass



