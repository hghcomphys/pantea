from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterator, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms as AseAtoms
from jax import tree_util

from pantea.atoms.box import Box, _shift_inside_box
from pantea.atoms.element import ElementMap
from pantea.logger import logger
from pantea.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from pantea.types import Array, Dtype, Element, default_dtype
from pantea.units import units


@dataclass
class Structure(BaseJaxPytreeDataClass):
    """
    A structure in the context of simulation box consists of
    arrays that store atomic attributes for a collection of atoms.

    The attributes of atoms within a structure can be described as follows:

    * `positions`: position of atoms, an array of (natoms, 3)
    * `forces`: force components, an array of (natoms, 3)
    * `energies`: associated atom potential energies, an array of (natoms,)
    * `charges`: charge of atoms, an array of (natoms,)
    * `total_energy`: total potential energy, scalar value
    * `total_charge`: total charge, scalar value

    The structure serves as a fundamental data unit for the atoms in the simulation box.
    Multiple structures can be gathered into a list to train a potential, or alternatively,
    the total energy and force components can be computed for a specific structure.

    Each structure has three additional instances:

    * `Box`: applying periodic boundary condition (PBC) along x, y, and z directions
    * `ElementMap`: determines how to extract assigned atom types from the element and vice versa
    * `Neighbor`: computes the list of neighboring atoms inside a specified cutoff radius

    .. note::
        The structure can be viewed as a separate domain for implementing MPI in large-scale
        molecular dynamics (MD) simulations, as demonstrated in the `miniMD`_ code.

    .. _miniMD: https://github.com/Mantevo/miniMD
    """

    positions: Array
    forces: Array
    energies: Array
    charges: Array
    total_energy: Array
    total_charge: Array
    atom_types: Array
    element_map: ElementMap
    box: Optional[Box] = None

    def __post_init__(self) -> None:
        self._assert_jit_dynamic_attributes(
            expected=(
                "positions",
                "forces",
                "energies",
                "charges",
                "total_energy",
                "total_charge",
                "atom_types",
            )
        )
        self._assert_jit_static_attributes(
            expected=(
                "element_map",
                "box",
            )
        )
        if self.box is not None:
            self.positions = _shift_inside_box(self.positions, self.lattice)

    @classmethod
    def from_ase(
        cls,
        atoms: AseAtoms,
        dtype: Optional[Dtype] = None,
    ) -> Structure:
        """
        Create an instance of the structure from `ASE`_ atoms.

        :param atoms: input `ASE`_ atoms instance
        :param dtype: data type for arrays, defaults to None
        :return: initialized structure

        .. _ASE: https://wiki.fysik.dtu.dk/ase/index.html
        """
        logging.debug(f"Initializing {cls.__name__} from ASE atoms")

        if dtype is None:
            dtype = default_dtype.FLOATX

        kwargs = dict()
        data = {
            "elements": [
                ElementMap.get_element_from_atomic_number(n)
                for n in atoms.get_atomic_numbers()
            ],
            "lattice": np.array(atoms.get_cell() * units.FROM_ANGSTROM, dtype=dtype),
            "positions": atoms.get_positions() * units.FROM_ANGSTROM,
        }
        for attr, ase_attr in zip(
            ("energies", "charges"),
            ("potential_energies", "charges"),
        ):
            try:
                data[attr] = getattr(atoms, f"get_{ase_attr}")()
            except RuntimeError:
                continue
        for attr in ("energies", "charges"):
            if attr in data:
                data[f"total_{attr}"] = sum(data[attr])

        input_data = defaultdict(list, data)
        try:
            element_map: ElementMap = ElementMap.from_list(input_data["elements"])
            kwargs.update(
                cls._init_arrays(input_data, element_map=element_map, dtype=dtype),
            )
            kwargs["element_map"] = element_map
            kwargs["box"] = cls._init_box(input_data["lattice"], dtype=dtype)
        except KeyError:
            logger.error(
                "Can not find at least one of the expected keyword in the input data.",
                exception=KeyError,
            )
        return cls(**kwargs)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        dtype: Optional[Dtype] = None,
    ) -> Structure:
        """
        Instantiate a structure object using an input data dictionary that contains
        distinct lists of positions, forces, elements, lattice, etc.

        :param data: input data
        :param dtype: data type for arrays, defaults to None
        :return: the initialized Structure
        """
        logging.debug(f"Initializing {cls.__name__} from an input dictionary")

        if dtype is None:
            dtype = default_dtype.FLOATX

        input_data: DefaultDict[str, List] = defaultdict(list, data)
        kwargs: Dict[str, Any] = dict()
        try:
            element_map = ElementMap.from_list(input_data["elements"])
            kwargs.update(
                cls._init_arrays(input_data, element_map=element_map, dtype=dtype),
            )
            kwargs["element_map"] = element_map
            kwargs["box"] = cls._init_box(input_data["lattice"], dtype=dtype)
        except KeyError:
            logger.error(
                "Cannot find at least one of the expected keyword in the input data dictionary.",
                exception=KeyError,
            )
        return cls(**kwargs)

    @classmethod
    def _init_arrays(
        cls,
        data: Dict[str, Any],
        element_map: ElementMap,
        dtype: Dtype,
    ) -> Dict[str, Array]:
        """Initialize atom attribute arrays from the input data dictionary."""
        logger.debug(f"{cls.__name__}: allocating arrays as follows:")
        arrays: Dict[str, Array] = dict()
        for atom_attr in Structure._get_atom_attributes():
            try:
                array: Array
                if atom_attr == "atom_types":
                    array = jnp.array(
                        [
                            element_map.get_atom_type_from_element(name)
                            for name in data["elements"]
                        ],
                        dtype=default_dtype.INDEX,
                    )
                else:
                    array = jnp.array(data[atom_attr], dtype=dtype)
                arrays[atom_attr] = jnp.squeeze(array)
                logger.debug(
                    f"{atom_attr:12} -> Array(shape={array.shape}, dtype='{array.dtype}')"
                )
            except KeyError:
                logger.error(
                    f"Cannot find atom attribute {atom_attr} in the input data",
                    exception=KeyError,
                )
        return arrays

    @classmethod
    def _init_box(
        cls,
        lattice: List[float],
        dtype: Dtype,
    ) -> Optional[Box]:
        if len(lattice) > 0:
            return Box.from_list(lattice, dtype=dtype)
        else:
            logger.debug("No lattice info were found")
            return None

    @classmethod
    def _get_atom_attributes(cls) -> Tuple[str, ...]:
        return cls._get_jit_dynamic_attributes()

    def __hash__(self) -> int:
        """Use parent class's hash method because of JIT."""
        return super().__hash__()

    @property
    def natoms(self) -> int:
        """Return number of atoms in the structure"""
        return self.positions.shape[0]

    @property
    def dtype(self) -> Dtype:
        """Return data type of the arrays in the structure (e.g., float64)."""
        return self.positions.dtype

    @property
    def lattice(self) -> Optional[Array]:
        """Cell 3x3 matrix."""
        if self.box is not None:
            return self.box.lattice

    def get_unique_elements(self) -> Tuple[Element, ...]:
        return self.element_map.unique_elements

    def get_elements(self) -> Tuple[Element, ...]:
        """Get array of elements."""
        to_element = self.element_map.atom_type_to_element
        atom_types_host = jax.device_get(self.atom_types)
        return tuple(str(to_element[at]) for at in atom_types_host)

    def select(self, element: Element) -> Array:
        """
        Retrieve the indices of all atoms that correspond to the given element.

        :param element: element name (e.g. `H` for hydrogen)
        :return: atom indices
        """
        return jnp.nonzero(
            self.atom_types == self.element_map.element_to_atom_type[element]
        )[0]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        The atomic attributes are represented as a dictionary of NumPy arrays.
        This format can be employed, for instance, when saving the structure data into a file.

        :return: dictionary of atom attributes.
        :rtype: Dict[str, np.ndarray]
        """
        data = dict()
        for atom_attr in self._get_atom_attributes():
            array: Array = getattr(self, atom_attr)
            data[atom_attr] = np.asarray(array)
        data["lattice"] = self.box.lattice if self.box else []
        data["elements"] = [
            self.element_map.get_element_from_atom_type(n) for n in data["atom_types"]
        ]
        return data

    def to_ase(self) -> AseAtoms:
        """
        Represent the structure as ASE atoms.

        The returned object can be utilized with the `ASE`_ package
        for visualization or modification of the structure.

        :return: `ASE`_ representation of the structure

        .. _ASE: https://wiki.fysik.dtu.dk/ase/index.html
        """
        logger.debug(f"Converting {self.__class__.__name__} to ASE atoms")
        to_element = self.element_map.atom_type_to_element
        cell = (
            units.TO_ANGSTROM * np.asarray(self.box.lattice)
            if self.box is not None
            else None
        )
        return AseAtoms(
            symbols=[to_element[int(at)] for at in self.atom_types],
            positions=[units.TO_ANGSTROM * np.asarray(pos) for pos in self.positions],
            cell=cell,
            pbc=True if self.box else False,
            charges=[np.asarray(ch) for ch in self.charges],
        )

    def _get_energy_offset(self, atom_energy: Dict[Element, float]) -> Array:
        energy_offset: Array = jnp.empty_like(self.energies)
        for element in self.get_unique_elements():
            energy_offset = energy_offset.at[self.select(element)].set(
                atom_energy[element]
            )
        return energy_offset

    def remove_energy_offset(self, atom_energy: Dict[Element, float]) -> None:
        """
        Remove the input reference energies from individual atoms and the total energy.

        :param atom_energy: atom reference energy
        :type atom_energy: Dict[Element, float]]
        """
        energy_offset: Array = self._get_energy_offset(atom_energy)
        self.energies -= energy_offset
        self.total_energy -= energy_offset.sum()

    def add_energy_offset(self, atom_energy: Dict[Element, float]) -> None:
        """
        Add the input reference energies to individual atoms and the total energy.

        :param atom_energy: atom reference energy
        :type atom_energy: Dict[Element, float]]
        """
        energy_offset = self._get_energy_offset(atom_energy)
        self.energies += energy_offset
        self.total_energy += energy_offset.sum()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(natoms={self.natoms}, "
            f"elements={self.get_unique_elements()}, "
            f"dtype={self.dtype})"
        )

    # ---

    def _get_inputs_per_element(self) -> Iterator[Tuple[Element, Inputs]]:
        for element in self.get_unique_elements():
            atom_indices: Array = self.select(element)
            yield element, Inputs(
                self.positions[atom_indices],
                self.positions,
                self.atom_types,
                self.lattice,
                tree_util.tree_map(
                    lambda x: jnp.array(x),
                    self.element_map.element_to_atom_type,
                ),
            )

    def get_inputs_per_element(self) -> Dict[Element, Inputs]:
        """Get required info per element for training and evaluating a potential."""
        return {element: input for element, input in self._get_inputs_per_element()}

    def _get_positions_per_element(self) -> Iterator[Tuple[Element, Array]]:
        for element in self.get_unique_elements():
            atom_indices = self.select(element)
            yield element, self.positions[atom_indices]

    def get_positions_per_element(self) -> Dict[Element, Array]:
        """Get position of atoms per element."""
        return {
            element: position for element, position in self._get_positions_per_element()
        }

    def _get_forces_per_element(self) -> Iterator[Tuple[Element, Array]]:
        for element in self.get_unique_elements():
            atom_indices = self.select(element)
            yield element, self.forces[atom_indices]

    def get_forces_per_element(self) -> Dict[Element, Array]:
        """Get force components per element."""
        return {element: force for element, force in self._get_forces_per_element()}


class Inputs(NamedTuple):
    """Represent array data of Structure for computing energy and forces."""

    atom_positions: Array
    positions: Array
    atom_types: Array
    lattice: Array
    emap: Dict[Element, Array]


register_jax_pytree_node(Structure)
