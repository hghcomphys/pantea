from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms as AseAtoms
from jax import tree_util

from jaxip.atoms._structure import (
    _calculate_center_of_mass,
    _calculate_distances,
)
from jaxip.atoms.box import Box
from jaxip.atoms.element import ElementMap
from jaxip.atoms.neighbor import Neighbor
from jaxip.logger import logger
from jaxip.pytree import BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.types import Array, Dtype, Element, _dtype
from jaxip.units import units


@dataclass
class Structure(BaseJaxPytreeDataClass):
    """
    A structure in the context of simulation box consists of
    arrays that store atomic attributes for a collection of atoms.

    Atomic attributes:

    * `positions`: per-atom position x, y, and z
    * `forces`: per-atom force components x, y, and z
    * `energies`:  per-atom energy
    * `total_energy`: total energy of atoms in simulation box
    * `charges`:  per-atom electric charge
    * `total_charge`: total charge of atoms in simulation box
    * `atom_types`: per-atom type (an unique integer) corresponding to each element

    The structure serves as a fundamental data unit for the atoms in the simulation box.
    Multiple structures can be gathered into a list to train a potential, or alternatively,
    the total energy and force components can be computed for a specific structure.

    Each structure has three additional instances:

    * `Box`: applying periodic boundary condition (PBC) along x, y, and z directions
    * `ElementMap`: determines how to extract the atom type (integer) from the element (string)
    * `Neighbor`: computes the list of neighboring atoms inside a cutoff radius

    .. note::
        `Structure` can be considered as an individual domain
        for MPI implementation of large-scale MD simulation (see `miniMD`_).

    .. _miniMD: https://github.com/Mantevo/miniMD
    """

    positions: Array
    forces: Array
    energies: Array
    total_energy: Array
    charges: Array
    total_charge: Array
    atom_types: Array
    element_map: ElementMap
    box: Optional[Box] = None
    neighbor: Optional[Neighbor] = None

    def __post_init__(self) -> None:
        logger.debug(f"Initializing {self.__class__.__name__}()")
        self._assert_jit_dynamic_attributes(
            expected=(
                "positions",
                "forces",
                "energies",
                "total_energy",
                "charges",
                "total_charge",
                "atom_types",
            )
        )
        self._assert_jit_static_attributes(
            expected=(
                "element_map",
                "box",
                "neighbor",
            )
        )
        if self.box is not None:
            self.positions = self.box.shift_inside_box(self.positions)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        r_cutoff: Optional[float] = None,
        dtype: Optional[Dtype] = None,
    ) -> Structure:
        """
        Create an instance of structure from an input data dictionary that includes
        separate lists of positions, forces, elements, lattice, etc.

        :param data: input data
        :param r_cutoff: neighbor atoms cutoff radius, defaults to None
        :param dtype: data type for arrays, defaults to None
        :return: an initialized Structure
        """
        if dtype is None:
            dtype = _dtype.FLOATX

        input_data: DefaultDict[str, List] = defaultdict(list, data)
        inputs: Dict[str, Any] = dict()
        try:
            element_map: ElementMap = ElementMap(input_data["elements"])
            inputs.update(
                cls._init_arrays(
                    input_data,
                    element_map=element_map,
                    dtype=dtype,
                ),
            )
            inputs["element_map"] = element_map
            inputs["box"] = cls._init_box(input_data["lattice"], dtype=dtype)
            inputs["neighbor"] = cls._init_neighbor(r_cutoff)
        except KeyError:
            logger.error(
                "Cannot find at least one of the expected keyword in the input data.",
                exception=KeyError,
            )
        return cls(**inputs)

    @classmethod
    def from_ase(
        cls,
        atoms: AseAtoms,
        r_cutoff: Optional[float] = None,
        dtype: Optional[Dtype] = None,
    ) -> Structure:
        """
        Create an instance of structure from `ASE`_ atoms.

        :param atoms: input ASE atoms instance
        :param r_cutoff: neighbor atoms cutoff radius, defaults to None
        :param dtype: data type for arrays, defaults to None
        :return: an initialized instance

        .. _ASE: https://wiki.fysik.dtu.dk/ase/index.html
        """
        if dtype is None:
            dtype = _dtype.FLOATX

        inputs: Dict[str, Any] = dict()
        data = {
            "elements": [
                ElementMap.atomic_number_to_element(n)
                for n in atoms.get_atomic_numbers()
            ],
            "lattice": np.asarray(atoms.get_cell() * units.FROM_ANGSTROM),
            "positions": atoms.get_positions() * units.FROM_ANGSTROM,
        }
        for attr, ase_attr in zip(
            ("charges", "energies"),
            ("charges", "potential_energies"),
        ):
            try:
                data[attr] = getattr(atoms, f"get_{ase_attr}")()
            except RuntimeError:
                continue
        for attr in ("energies", "charges"):
            if attr in data:
                data[f"total_{attr}"] = sum(data[attr])

        input_data: DefaultDict[str, List] = defaultdict(list, data)
        try:
            element_map: ElementMap = ElementMap(input_data["elements"])
            inputs.update(
                cls._init_arrays(
                    input_data, element_map=element_map, dtype=dtype
                ),
            )
            inputs["element_map"] = element_map
            inputs["box"] = cls._init_box(input_data["lattice"], dtype=dtype)
            inputs["neighbor"] = cls._init_neighbor(r_cutoff)
        except KeyError:
            logger.error(
                "Can not find at least one of the expected keyword in the input data.",
                exception=KeyError,
            )
        return cls(**inputs)

    @classmethod
    def _init_arrays(
        cls,
        data: Dict[str, Any],
        element_map: ElementMap,
        dtype: Dtype,
    ) -> Dict[str, Array]:
        """Initialize array atomic attributes from the input data dictionary."""
        logger.debug(f"{cls.__name__} is allocating arrays as follows:")
        arrays: Dict[str, Array] = dict()
        for atom_attr in Structure._get_atom_attributes():
            try:
                array: Array
                if atom_attr == "atom_types":
                    array = jnp.array(
                        [element_map(atom) for atom in data["elements"]],
                        dtype=_dtype.INDEX,
                    )
                else:
                    array = jnp.array(data[atom_attr], dtype=dtype)
                arrays[atom_attr] = array
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
        lattice: List[List[float]],
        dtype: Dtype,
    ) -> Optional[Box]:
        """Initialize simulation box from input lattice matrix."""
        if len(lattice) > 0:
            return Box(lattice, dtype=dtype)
        else:
            logger.debug("No lattice info were found")

    @classmethod
    def _init_neighbor(
        cls, r_cutoff: Optional[float] = None
    ) -> Optional[Neighbor]:
        """Initialize neighbor input cutoff radius."""
        if r_cutoff is not None:
            return Neighbor(r_cutoff=r_cutoff)
        else:
            logger.debug("No neighbor info were found")

    def set_cutoff_radius(self, r_cutoff: float) -> None:
        """
        Set cutoff radius of neighbor atoms in the structure.

        This is a method for working with potentials with different cutoff radius.
        It's important to note that the neighbor object in Structure behaves
        as a buffer and not part of atomic structure data.
        It is used for calculating descriptors, potential, etc.
        The neighbor list of atoms must be updated before using it.

        :param r_cutoff: a new values for the cutoff radius
        :type r_cutoff: float
        """
        if self.neighbor is not None:
            self.neighbor.set_cutoff_radius(r_cutoff)
        else:
            self.neighbor = Neighbor(r_cutoff)

    def update_neighbor(self) -> None:
        """Update the neighbor list of atoms in the structure."""
        if self.neighbor is not None:
            self.neighbor.update(self)
        else:
            logger.error(
                "Cannot update neighbor list, set the cutoff radius first",
                exception=ValueError,
            )

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
        """Return data type of the arrays in the structure (e.g., float32)."""
        return self.positions.dtype

    @property
    def r_cutoff(self) -> Optional[float]:
        """Return cutoff radius for neighboring atoms."""
        if self.neighbor is not None:
            return self.neighbor.r_cutoff

    @property
    def lattice(self) -> Optional[Array]:
        """Cell 3x3 matrix."""
        if self.box is not None:
            return self.box.lattice

    def get_unique_elements(self) -> Tuple[Element, ...]:
        return tuple(sorted(set(self.get_elements())))

    def get_elements(self) -> Tuple[Element, ...]:
        """Get array of elements."""
        to_element = self.element_map.atom_type_to_element
        atom_types_host = jax.device_get(self.atom_types)
        return tuple(str(to_element[int(at)]) for at in atom_types_host)

    def get_masses(self) -> Array:
        """Get array of atomic masses."""
        to_element = self.element_map.atom_type_to_element
        elements = (to_element[int(at)] for at in self.atom_types)
        return jnp.array(
            tuple(
                ElementMap.element_to_atomic_mass(element)
                for element in elements
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(natoms={self.natoms}, "
            f"elements={self.get_unique_elements()}, "
            f"dtype={self.dtype})"
        )

    def select(self, element: Element) -> Array:
        """
        Return all atom indices of the element.

        :param element: element name (e.g. `H` for hydrogen)
        :return: atom indices
        """
        return jnp.nonzero(
            self.atom_types == self.element_map.element_to_atom_type[element]
        )[0]

    # @jax.jit
    def calculate_distances(
        self,
        atom_indices: Array,
        neighbor_indices: Optional[Array] = None,
        return_position_differences: bool = False,
    ) -> Tuple[Array, ...]:
        """
        Calculate distances between specific atoms (given by atom indices)
        and the neighboring atoms in the structure.
        This method optionally also returns the corresponding position differences.

        If no neighbor index is given, all atoms in the structure
        will be considered as neighbor atoms.

        :param atom_indices: array of atom indices
        :type atom_indices: Array
        :param neighbor_indices: indices of neighbor atoms, defaults to None
        :type neighbor_indices: Optional[Array], optional
        :type neighbor_indices: bool, optional
        :param return_position_differences: whether returning position differences, defaults to False
        :type return_position_differences: bool, optional
        :return:  distances between atoms
        :rtype: Tuple[Array, ...]
        """
        atom_positions = self.positions[jnp.asarray([atom_indices])].reshape(
            -1, 3
        )
        if neighbor_indices is not None:
            neighbor_positions = self.positions[
                jnp.atleast_1d(neighbor_indices)
            ]
        else:
            neighbor_positions = self.positions

        distances, position_differences = _calculate_distances(
            atom_positions,
            neighbor_positions,
            self.lattice,
        )
        if return_position_differences:
            return jnp.squeeze(distances), jnp.squeeze(position_differences)
        return (jnp.squeeze(distances),)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Represent atomic attributes in form of dictionary of numpy arrays.
        To be used, for example, for dumping structure data into a file.

        :return: dictionary of atom attributes.
        :rtype: Dict[str, np.ndarray]
        """
        data = dict()
        for atom_attr in self._get_atom_attributes():
            array: Array = getattr(self, atom_attr)
            data[atom_attr] = np.asarray(array)
        return data

    def to_ase(self) -> AseAtoms:
        """
        An ASE (atoms) representation of the structure.
        The returned atoms object can be used to visualize or
        modify the structure using ASE package.

        .. warning::
            This works only for orthogonal cells.

        :return: ASE representation of the structure
        :rtype: AseAtoms
        """
        logger.info("Creating an ASE representation of the structure")
        to_element = self.element_map.atom_type_to_element
        return AseAtoms(
            symbols=[to_element[int(at)] for at in self.atom_types],
            positions=[
                units.TO_ANGSTROM * np.asarray(pos) for pos in self.positions
            ],
            cell=units.TO_ANGSTROM * np.asarray(self.box.lattice)
            if self.box is not None
            else None,
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
        Remove input atom reference energies from each atom and also the total energy.

        :param atom_energy: atom reference energy
        """
        energy_offset: Array = self._get_energy_offset(atom_energy)
        self.energies -= energy_offset
        self.total_energy -= energy_offset.sum()

    def add_energy_offset(self, atom_energy: Dict[Element, float]) -> None:
        """
        Add input atom reference energies to each the atom and the total energy.

        :param atom_energy: atom reference energy
        """
        energy_offset = self._get_energy_offset(atom_energy)
        self.energies += energy_offset
        self.total_energy += energy_offset.sum()

    def get_center_of_mass_position(self) -> Array:
        """Get center of mass position."""
        return _calculate_center_of_mass(self.positions, self.get_masses())

    def get_per_element_inputs(self) -> Dict[Element, Inputs]:
        """Get required info per element for training and evaluating a potential."""

        def extract_inputs() -> Iterator[Tuple[Element, Inputs]]:
            for element in self.get_unique_elements():
                atom_indices: Array = self.select(element)
                yield element, Inputs(
                    self.positions[atom_indices],
                    self.positions,
                    self.atom_types,
                    self.lattice,
                    tree_util.tree_map(
                        lambda x: jnp.asarray(x),
                        self.element_map.element_to_atom_type,
                    ),
                )

        return {element: input for element, input in extract_inputs()}

    def get_per_element_positions(self) -> Dict[Element, Array]:
        """Get position of atoms per element."""

        def extract_positions() -> Iterator[Tuple[Element, Array]]:
            for element in self.get_unique_elements():
                atom_indices = self.select(element)
                yield element, self.positions[atom_indices]

        return {element: position for element, position in extract_positions()}

    def get_per_element_forces(self) -> Dict[Element, Array]:
        """Get force components per element."""

        def extract_forces() -> Iterator[Tuple[Element, Array]]:
            for element in self.get_unique_elements():
                atom_indices = self.select(element)
                yield element, self.forces[atom_indices]

        return {element: force for element, force in extract_forces()}


class Inputs(NamedTuple):
    """Represent array data of Structure for computing energy and forces."""

    atom_positions: Array
    positions: Array
    atom_types: Array
    lattice: Array
    emap: Dict[Element, Array]


register_jax_pytree_node(Structure)
