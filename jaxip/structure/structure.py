from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Generator, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms as AseAtoms
from jax import tree_util

from jaxip.base import _BaseJaxPytreeDataClass, register_jax_pytree_node
from jaxip.logger import logger
from jaxip.structure._structure import _calculate_distance
from jaxip.structure.box import Box
from jaxip.structure.element import ElementMap
from jaxip.structure.neighbor import Neighbor
from jaxip.types import Array, Dtype, Element
from jaxip.types import dtype as _dtype
from jaxip.units import units


# FIXME: move to the trainer module
class Inputs(NamedTuple):
    """
    Represents array data types of Structure.
    """

    atom_position: Array
    position: Array
    atype: Array
    lattice: Array
    emap: Dict[Element, Array]


@dataclass
class Structure(_BaseJaxPytreeDataClass):
    """
    A structure contains arrays of atomic attributes
    for a collection of atoms in the simulation box.

    Atomic attributes:

    * `position`: per-atom position x, y, and z
    * `force`: per-atom force components x, y, and z
    * `energy`:  per-atom energy
    * `total_energy`: total energy of atoms in simulation box
    * `charge`:  per-atom electric charge
    * `total_charge`: total charge of atoms in simulation box
    * `atom_type`: per-atom type (unique integer) corresponding to each element

    An instance structure can be seen as unit of data.
    list of structures will be used to train a potential,
    or new energy and forces components can be computed for a given structure.

    Each structure has three additional instances:

    * `Box`: applying periodic boundary conditions (PBC)
    * `ElementMap`: determines how to extract atom types (integer) from the element (string)
    * `Neighbor`: computes the list of neighboring atoms inside a given cutoff radius

    .. note::
        `Structure` can be considered as one domain
        in the domain decomposition method for the MPI implementation (see `miniMD`_).

    .. _miniMD: https://github.com/Mantevo/miniMD
    """

    # Array type attributes must be define first (with type hint)
    position: Array
    force: Array
    energy: Array
    total_energy: Array
    charge: Array
    total_charge: Array
    atom_type: Array
    box: Box
    element_map: ElementMap
    neighbor: Neighbor
    requires_neighbor_update: bool = True

    # --------------------------------------------------------------------------------------

    @classmethod
    def create_from_dict(
        cls,
        data: Dict[str, Any],
        r_cutoff: Optional[float] = None,
        dtype: Optional[Dtype] = None,
    ) -> Structure:
        """
        Create a new instance of structure from a dictionary of data including
        positions, forces, box, neighbor, etc.

        :param data: input data
        :param r_cutoff: neighbor atom cutoff radius, defaults to None
        :param dtype: data type of arrays, defaults to None
        :return: an initialized instance
        """
        if dtype is None:
            dtype = _dtype.FLOATX

        kwargs: Dict[str, Any] = dict()
        data_: DefaultDict[str, List] = defaultdict(list, data)
        try:
            element_map: ElementMap = ElementMap(data_["element"])
            kwargs.update(
                cls._init_arrays(data_, element_map=element_map, dtype=dtype),
            )
            kwargs["box"] = cls._init_box(data_["lattice"], dtype=dtype)
            kwargs["element_map"] = element_map
            kwargs["neighbor"] = Neighbor(r_cutoff=r_cutoff)

        except KeyError:
            logger.error(
                f"Cannot find at least one of the expected keyword in the input data.",
                exception=KeyError,
            )
        return cls(**kwargs)

    @classmethod
    def create_from_ase(
        cls,
        atoms: AseAtoms,
        r_cutoff: Optional[float] = None,
        dtype: Optional[Dtype] = None,
    ) -> Structure:
        """
        Create a new instance of structure from the ASE atoms.

        :param atoms: input ASE atoms instance
        :param r_cutoff: neighbor atom cutoff radius, defaults to None
        :param dtype: data type of arrays, defaults to None
        :return: an initialized instance
        """
        # TODO: add test

        if dtype is None:
            dtype = _dtype.FLOATX

        kwargs: Dict[str, Any] = dict()

        # Extract atom info from the ASE atoms instance
        data = {
            "element": [
                ElementMap.atomic_number_to_element(n)
                for n in atoms.get_atomic_numbers()
            ],
            "lattice": np.asarray(atoms.get_cell()),
            "position": atoms.get_positions(),
        }
        for key, attr in zip(
            ("charge", "energy"),
            ("charges", "potential_energies"),
        ):
            try:
                data[key] = getattr(atoms, f"get_{attr}")()
            except RuntimeError:
                continue
        for attr in ("energy", "charge"):
            if attr in data:
                data[f"total_{attr}"] = sum(data[attr])

        # Use extracted info to initialize structure # FIXME: avoid DRY
        data_: DefaultDict[str, List] = defaultdict(list, data)
        try:
            element_map: ElementMap = ElementMap(data_["element"])
            kwargs.update(
                cls._init_arrays(data_, element_map=element_map, dtype=dtype),
            )
            kwargs["box"] = cls._init_box(data_["lattice"], dtype=dtype)
            kwargs["element_map"] = element_map
            kwargs["neighbor"] = Neighbor(r_cutoff=r_cutoff)

        except KeyError:
            logger.error(
                f"Cannot find at least one of the expected keyword in the input data.",
                exception=KeyError,
            )
        return cls(**kwargs)

    @staticmethod
    def _init_arrays(
        data: Dict, element_map: ElementMap, dtype: Dtype
    ) -> Dict[str, Array]:
        """Initialize a dictionary of arrays of atomic attributes from the input data."""
        logger.debug("Allocating arrays for the structure:")
        arrays: Dict[str, Array] = dict()

        for atom_attr in Structure._get_atom_attributes():
            try:
                array: Array
                if atom_attr == "atom_type":
                    array = jnp.asarray(
                        [element_map(atom) for atom in data["element"]],
                        dtype=_dtype.INDEX,
                    )
                else:
                    array = jnp.asarray(data[atom_attr], dtype=dtype)
                arrays[atom_attr] = array
                logger.debug(
                    f"{atom_attr:12} -> Array(shape='{array.shape}', dtype='{array.dtype}')"
                )
            except KeyError:
                logger.error(
                    f"Cannot find atom attribute {atom_attr} in the input data",
                    exception=KeyError,  # type:ignore
                )
        return arrays

    @staticmethod
    def _init_box(lattice: List[List[float]], dtype: Dtype) -> Box:
        """Initialize a simulation box from the input lattice matrix."""
        box: Box
        if len(lattice) > 0:
            box = Box(jnp.asarray(lattice, dtype=dtype))
        else:
            logger.debug("No lattice info were found in structure")
            box = Box()
        return box

    @classmethod
    def _get_atom_attributes(cls) -> Tuple[str, ...]:
        """Get atom attributes which are basically array values."""
        return cls._get_jit_dynamic_attributes()

    def __post_init__(self) -> None:
        """Post initializations."""
        self.position = self.box.shift_inside_box(self.position)
        logger.debug(f"Initializing {self.__class__.__name__}()")

    def __hash__(self) -> int:
        """Enforce to use the parent class's hash method (JIT)."""
        return super().__hash__()

    # --------------------------------------------------------------------------------------

    def set_cutoff_radius(self, r_cutoff: float) -> None:
        """
        Set cutoff radius of neighbor atoms in the structure.
        This method is useful when having a potential with different cutoff radius.

        Updating of the neighbor list for a new cutoff radius is skipped if it is the same the existing one.
        It's important to note that the Neighbor object in Structure is considered as a buffer and not
        part of the atomic structure data, and it is used for calculating descriptors, potential, etc.
        It is the task of calling classes to prepare the buffer neighbor before using it.

        :param r_cutoff: a new values for the cutoff radius
        :type r_cutoff: float
        """
        if self.r_cutoff == r_cutoff:
            logger.info(
                f"Skipping updating the neighbor list (cutoff radius): "
                f"{self.r_cutoff} vs. {r_cutoff} (new)"
            )
            return

        self.neighbor.set_cutoff_radius(r_cutoff)
        self.requires_neighbor_update = True

    def update_neighbor(self) -> None:
        """
        Update the neighbor list of the structure.
        This task can be computationally expensive.
        """
        self.neighbor.update(self)

    @property
    def r_cutoff(self) -> Optional[float]:
        """Return cutoff radius of neighboring atoms."""
        return self.neighbor.r_cutoff

    # --------------------------------------------------------------------------------------

    @property
    def natoms(self) -> int:
        """Number of atoms in the structure"""
        return self.position.shape[0]

    @property
    def dtype(self) -> Dtype:
        """Datatype of the arrays in the structure (e.g. float32)."""
        return self.position.dtype

    @property
    def lattice(self) -> Optional[Array]:
        """Cell 3x3 matrix."""
        return self.box.lattice

    @property
    def elements(self) -> tuple[Element, ...]:
        # FIXME: optimize
        return tuple(sorted({str(self.element_map(int(at))) for at in self.atom_type}))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(natoms={self.natoms}, elements={self.elements}, dtype={self.dtype})"

    # TODO: jit
    def select(self, element: Element) -> Array:
        """
        Return all atom indices of the element.

        :param element: element name (e.g. `H` for hydrogen)
        :return: atom indices
        """
        return jnp.nonzero(self.atom_type == self.element_map[element])[0]

    @jax.jit
    def calculate_distance(
        self,
        atom_index: Array,
        neighbor_index: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """
        Calculate distances between specific atoms (given by atom indices)
        and the neighboring atoms in the structure.
        This method also returns the corresponding position difference.

        If no neighbor index is given, all atoms in the structure
        will be considered as neighbor atoms.

        :param atom_index: array of atom indices
        :type atom_index: Array
        :param neighbor_index: indices of neighbor atoms, defaults to None
        :type neighbor_index: Optional[Array], optional
        :return:  distances, position differences
        :rtype: Tuple[Array, Array]
        """
        atom_position = self.position[jnp.asarray([atom_index])].reshape(-1, 3)
        if neighbor_index is not None:
            neighbor_position = self.position[jnp.atleast_1d(neighbor_index)]
        else:
            neighbor_position = self.position

        dis, dx = _calculate_distance(
            atom_position, neighbor_position, self.box.lattice
        )
        return jnp.squeeze(dis), jnp.squeeze(dx)

    # --------------------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Return arrays of atomic attributes in form of dictionary of numpy arrays.
        To be used, for example, for dumping structure data into a file.

        :return: dictionary of atom attributes.
        :rtype: Dict[str, np.ndarray]
        """
        data = dict()
        for atom_attr in self._get_atom_attributes():
            array: Array = getattr(self, atom_attr)
            data[atom_attr] = np.asarray(array)
        return data

    def to_ase_atoms(self) -> AseAtoms:
        """
        An ASE representation of the structure.
        The returned atoms object can be used to visualize or modify the structure using the ASE package.

        .. warning::
            This works only for orthogonal cells.

        :return: ASE representation of the structure
        :rtype: AseAtoms
        """
        logger.info("Creating a representation of the structure in form of ASE atoms")
        return AseAtoms(
            symbols=[self.element_map(int(at)) for at in self.atom_type],
            positions=[
                units.BOHR_TO_ANGSTROM * np.asarray(pos) for pos in self.position
            ],
            cell=units.BOHR_TO_ANGSTROM * np.asarray(self.box.lattice)
            if self.box
            else None,
            pbc=True if self.box else False,
            charges=[np.asarray(ch) for ch in self.charge],
        )

    # --------------------------------------------------------------------------------------

    def _get_energy_offset(self, atom_energy: Dict[Element, float]) -> Array:
        """Return a array of energy offset."""

        energy_offset: Array = jnp.empty_like(self.energy)
        # TODO: optimize item assignment
        for element in self.elements:
            energy_offset = energy_offset.at[self.select(element)].set(
                atom_energy[element]
            )
        return energy_offset

    def remove_energy_offset(self, atom_energy: Dict[Element, float]) -> None:
        """
        Remove the input atom reference energies from the per-atom and total energy.

        :param atom_energy: atom reference energy
        """
        energy_offset = self._get_energy_offset(atom_energy)
        self.energy -= energy_offset
        self.total_energy -= energy_offset.sum()

    def add_energy_offset(self, atom_energy: Dict[Element, float]) -> None:
        """
        Add the input atom reference energies from the per-atom and total energy.

        :param atom_energy: atom reference energy
        """
        energy_offset = self._get_energy_offset(atom_energy)
        self.energy += energy_offset
        self.total_energy += energy_offset.sum()

    # --------------------------------------------------------------------------------------

    def get_inputs(self) -> Dict[Element, Inputs]:
        """A tuple of required info which are used for training and evaluating the potential."""

        def extract_input() -> Generator[Tuple[Element, Inputs], None, None]:
            for element in self.elements:
                atom_index: Array = self.select(element)
                yield element, Inputs(
                    self.position[atom_index],
                    self.position,
                    self.atom_type,
                    self.box.lattice,  # type:ignore
                    tree_util.tree_map(
                        lambda x: jnp.asarray(x), self.element_map.element_to_atype
                    ),
                )

        return {element: input for element, input in extract_input()}

    def get_positions(self) -> Dict[Element, Array]:
        """Get position of atoms per element."""

        def extract_position() -> Generator[Tuple[Element, Array], None, None]:
            for element in self.elements:
                atom_index: Array = self.select(element)
                yield element, self.position[atom_index]

        return {element: position for element, position in extract_position()}

    def get_forces(self) -> Dict[Element, Array]:
        """Get force components per element."""

        def extract_force() -> Generator[Tuple[Element, Array], None, None]:
            for element in self.elements:
                atom_index: Array = self.select(element)
                yield element, self.force[atom_index]

        return {element: force for element, force in extract_force()}


register_jax_pytree_node(Structure)
