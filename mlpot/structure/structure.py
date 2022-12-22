from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms as AseAtoms

from mlpot.base import _Base
from mlpot.logger import logger
from mlpot.structure._structure import _calculate_distance
from mlpot.structure.box import Box
from mlpot.structure.element import ElementMap
from mlpot.structure.neighbor import Neighbor
from mlpot.types import Array, Dtype
from mlpot.types import dtype as _dtype


class Input(NamedTuple):
    """
    Represents an array data types of Structure.
    """

    position_aid: Array
    position: Array
    atype: Array
    lattice: Array
    emap: Dict[str, Array]


@dataclass
class Structure(_Base):
    """
    A structure contains arrays of atomic attributes
    for a collection of atoms in a simulation box.

    Atomic attributes:

    * `position`: per-atom position x, y, and z
    * `force`: per-atom force components x, y, and z
    * `energy`:  per-atom energy
    * `lattice`: vectors of super cell 3x3 matrix
    * `total_energy`: total energy of atoms in simulation box
    * `charge`:  per-atom electric charge
    * `total_charge`: total charge of atoms in simulation box

    An instance structure can be seen as unit of data.
    list of structures will be used to train a potential,
    or new energy and forces components can be computed for a given structure.

    Each structure has two instances of the `ElementMap` and `Neighbor`.
    The element map determines how to extract atom types (integer)
    from the element (string such as 'H' for hydrogen atoms).
    The neighbor computes the list of neighboring atoms inside a given cutoff radius.

    .. note::
        `Structure` can be considered as one domain
        in the domain decomposition method for the MPI implementation (see `miniMD`_).

    .. _miniMD: https://github.com/Mantevo/miniMD
    """

    position: Array
    force: Array
    energy: Array
    lattice: Array
    total_energy: Array
    charge: Array
    total_charge: Array
    atom_type: Array
    element_map: ElementMap
    box: Box
    neighbor: Neighbor

    atom_attributes: ClassVar[Tuple[str, ...]] = (
        "position",
        "force",
        "energy",
        "lattice",
        "total_energy",
        "charge",
        "total_charge",
    )

    @classmethod
    def create_from_dict(
        cls,
        data: Dict[str, Any],
        r_cutoff: Optional[float] = None,
        dtype: Dtype = _dtype.FLOATX,
    ) -> Structure:
        """
        Initialize structure from a dictionary of arrays, box, neighbor, etc.
        """
        kwargs: Dict[str, Any] = dict()
        try:
            element_map = ElementMap(data["element"])
            kwargs["element_map"] = element_map
            kwargs["atom_type"] = jnp.asarray(
                [element_map(atom) for atom in data["element"]],
                dtype=_dtype.INDEX,
            )
            kwargs.update(
                cls._init_arrays(data, dtype=dtype),
            )
            kwargs["box"] = cls._init_box(data["lattice"], dtype=dtype)
            kwargs["neighbor"] = Neighbor(r_cutoff)

        except KeyError:
            logger.error(
                f"Cannot find at least one of the expected atomic attributes in the input data:"
                f"{''.join(cls.atom_attributes)}",
                exception=KeyError,  # type: ignore
            )
        return cls(**kwargs)

    @staticmethod
    def _init_arrays(data: Dict, dtype: Dtype) -> Dict[str, Array]:
        """
        Create arrays from the input dictionary of structure data.
        It convert element (string) to atom type (integer) because of computational efficiency.
        """
        logger.debug("Allocating arrays for the structure:")
        arrays: Dict[str, Array] = dict()

        for atom_attr in Structure.atom_attributes:
            try:
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
    def _init_box(lattice: List, dtype: Dtype) -> Box:
        """
        Create a simulation box object using the provided lattice matrix.

        :param lattice: 3x3 lattice matrix
        :type lattice: Array
        """
        box: Box
        if len(lattice) > 0:
            box = Box(jnp.asarray(lattice, dtype=dtype))
        else:
            logger.debug("No lattice info were found in structure")
            box = Box()
        return box

    def __post_init__(self) -> None:

        self.requires_neighbor_update: bool = True

        if self.box:
            logger.debug("Shift all the atoms inside the simulation box")
            self.position = self.box.shift_inside_box(self.position)

        super().__init__()

    def set_cutoff_radius(self, r_cutoff: float) -> None:
        """
        Set cutoff radius of the structure.
        This method is useful when having a potential with different cutoff radius.

        It ignores updating the neighbor list if the new cutoff radius is the same the existing one.
        It's important to note that the Neighbor object in Structure is considered as a buffer and not
        part of the atomic structure data. But, it uses for calculating descriptors, potential, etc.
        It is the task of those classes to prepare the buffer neighbor for their own usage.

        :param r_cutoff: new cutoff radius
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
        This can be computationally expensive.
        """
        self.neighbor.update(self)

    @property
    def r_cutoff(self) -> Optional[float]:
        return self.neighbor.r_cutoff

    @property
    def natoms(self) -> int:
        return self.position.shape[0]

    @property
    def dtype(self) -> Dtype:
        return self.position.dtype

    @property
    def elements(self) -> tuple[str]:
        # FIXME: optimize
        return tuple({self.element_map(int(at)) for at in self.atom_type})

    def __repr__(self) -> str:
        return f"Structure(natoms={self.natoms}, elements={self.elements}, dtype={self.dtype})"

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Return arrays of atomic attribute to a dictionary.
        To be used for dumping structure into a file.
        """
        data = dict()
        for atom_attr in self.atom_attributes:
            array: Array = getattr(self, atom_attr)
            data[atom_attr] = np.asarray(array)
        return data

    def to_ase_atoms(self) -> AseAtoms:
        """
        This method returns an ASE representation (atoms) of the structure.
        The returned object can be used to visualize or modify the structure using the ASE package.
        """
        logger.info("Creating a representation of the structure in form of ASE atoms")
        BOHR_TO_ANGSTROM = 0.529177  # TODO: Unit class or input length_conversion
        return AseAtoms(
            symbols=[self.element_map(int(at)) for at in self.atom_type],
            positions=[BOHR_TO_ANGSTROM * np.asarray(pos) for pos in self.position],
            cell=[BOHR_TO_ANGSTROM * float(l) for l in self.box.length]
            if self.box
            else None,  # FIXME: works only for orthogonal cells
        )

    # TODO: jit
    def select(self, element: str) -> Array:
        """
        Return all atom ids with atom type same as the input element.

        :param element: element
        :type element: str
        :return: atom indices
        :rtype: Array
        """
        return jnp.nonzero(self.atom_type == self.element_map[element])[0]

    @partial(jax.jit, static_argnums=(0,))  # FIXME
    def calculate_distance(
        self,
        aid: Array,
        neighbor_index: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return distances between a specific atom and given neighboring atoms in the structure,
        and corresponding position difference.
        If no neighbor index is given, all atoms in the structure will be considered.
        """
        dis, dx = _calculate_distance(
            jnp.atleast_2d(self.position[jnp.asarray(aid)]),
            self.position[jnp.atleast_1d(neighbor_index)]
            if neighbor_index is not None
            else self.position,
            self.box.lattice,
        )

        return jnp.squeeze(dis), jnp.squeeze(dx)

    def _get_energy_offset(self, atom_energy: Dict[str, float]) -> Array:
        """
        Return a array of energy offset.

        :param atom_energy: atom reference energy
        :type atom_energy: Dict[str, float]
        :return: energy offset
        :rtype: Array
        """
        energy_offset: Array = jnp.empty_like(self.energy)
        for element in self.elements:  # TODO: optimize item assignment
            energy_offset = energy_offset.at[self.select(element)].set(
                atom_energy[element]
            )
        return energy_offset

    def remove_energy_offset(self, atom_energy: Dict[str, float]) -> None:
        """
        Remove the given atom reference energies from the per-atom and total energy.

        :param atom_energy: atom reference energy
        :type atom_energy: Dict[str, float]
        """
        energy_offset = self._get_energy_offset(atom_energy)
        self.energy -= energy_offset
        self.total_energy -= energy_offset.sum()

    def add_energy_offset(self, atom_energy: Dict[str, float]) -> None:
        """
        Add the given atom reference energies from the per-atom and total energy.

        :param atom_energy: atom reference energy
        :type atom_energy: Dict[str, float]
        """
        energy_offset = self._get_energy_offset(atom_energy)
        self.energy += energy_offset
        self.total_energy += energy_offset.sum()

    def get_inputs(self) -> Dict[str, Input]:
        """
        A tuple of required info that are used for training and evaluating the potential.
        """

        def extract_input():
            for element in self.elements:
                aid: Array = self.select(element)
                yield element, Input(
                    self.position[aid],
                    self.position,
                    self.atom_type,
                    self.box.lattice,
                    self.element_map.element_to_atype,
                )

        return {element: input for element, input in extract_input()}

    def get_positions(self) -> Dict[str, Array]:
        def extract_position():
            for element in self.elements:
                aid: Array = self.select(element)
                yield element, self.position[aid]

        return {element: position for element, position in extract_position()}

    def get_forces(self) -> Dict[str, Array]:
        def extract_force():
            for element in self.elements:
                aid: Array = self.select(element)
                yield element, self.force[aid]

        return {element: force for element, force in extract_force()}

    def compare(
        self,
        other: Structure,
        errors: Union[str, List] = "RMSEpa",
        return_difference: bool = False,
    ) -> Dict[str, Array]:
        """
        Compare force and total energy values between two structures and return desired errors metrics.

        :param other: other structure
        :type other: Structure
        :param error: a list of error metrics including 'RMSE', 'RMSEpa', 'MSE', and 'MSEpa'. Defaults to ['RMSEpa']
        :type errors: list, optional
        :type return_difference: bool, optional
        :return: whether return energy and force array differences or not, defaults to False
        :return: a dictionary of error metrics.
        :rtype: Dict
        """
        # TODO: add charge, total_charge
        result = dict()
        frc_diff = self.force - other.force
        eng_diff = self.total_energy - other.total_energy
        errors = [errors] if isinstance(errors, str) else errors
        logger.info(f"Comparing two structures, error metrics: {', '.join(errors)}")
        errors = [x.lower() for x in errors]

        # TODO: use metric classes
        if "rmse" in errors:
            result["force_RMSE"] = jnp.sqrt(jnp.mean(frc_diff**2))
            result["energy_RMSE"] = jnp.sqrt(jnp.mean(eng_diff**2))
        if "rmsepa" in errors:
            result["force_RMSEpa"] = jnp.sqrt(jnp.mean(frc_diff**2))
            result["energy_RMSEpa"] = jnp.sqrt(jnp.mean(eng_diff**2)) / self.natoms
        if "mse" in errors:
            result["force_MSE"] = jnp.mean(frc_diff**2)
            result["energy_MSE"] = jnp.mean(eng_diff**2)
        if "msepa" in errors:
            result["force_MSEpa"] = jnp.mean(frc_diff**2)
            result["energy_MSEpa"] = jnp.mean(eng_diff**2) / self.natoms
        if return_difference:
            result["frc_diff"] = frc_diff
            result["eng_diff"] = eng_diff

        return result
