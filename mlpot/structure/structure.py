from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Union, Optional
from ase import Atoms as AseAtoms
from functools import partial
from mlpot.logger import logger
from mlpot.base import _Base
from mlpot.types import dtype as _dtype, Array
from mlpot.utils.attribute import set_as_attribute
from mlpot.structure._structure import _calculate_distance
from mlpot.structure.element import ElementMap
from mlpot.structure.neighbor import Neighbor
from mlpot.structure.box import Box


class Structure(_Base):
    """
    A structure tensors of atomic information including positions, forces, per-atom and total energy,
    cell matrix, and  more) for a collection of atoms in the simulation box.
    An instance of the Structure class is a defined unit of the atomic data that
    is used to calculate the (atomic) descriptors.
    For computational reasons tensors of atomic data are used instead of defining
    individual atoms as a unit of the atomic data.

    The most computationally expensive section of a structure is utilized for calculating the neighbor list.
    This is done by giving an instance of Structure to the Neighbor class which is responsible
    for updating the neighbor lists.

    For the MPI implementation, this class can be considered as one domain
    in domain decomposition method (see miniMD code).
    An C++ implementation might be required for MD simulation but not necessarily developing ML potential.
    """

    _atomic_attributes: Tuple[str] = (
        "position",  # per-atom position x, y, and z
        "force",  # per-atom force components x, y, and z
        "energy",  # per-atom energy
        "lattice",  # vectors of super cell 3x3 matrix
        "total_energy",  # total energy of atoms in simulation box
        # 'charge',       # per-atom electric charge
        # 'total_charger' # total charge of atoms in simulation box
    )

    def __init__(
        self,
        data: Dict = None,
        r_cutoff: float = None,
        dtype: jnp.dtype = jnp.float32,  # FIXME
        **kwargs,
    ) -> None:
        """
        Initialize structure including tensors, simulation box, neighbor list, etc.
        """
        self.dtype = dtype if dtype else _dtype.FLOATX
        self.requires_neighbor_update = True

        self.element_map: ElementMap = kwargs.get("element_map", None)
        self.tensors: Dict[str, jnp.ndarray] = kwargs.get("tensors", None)
        self.box: Box = kwargs.get("box", None)
        self.neighbor: Neighbor = kwargs.get("neighbor", None)

        if data:
            try:
                self.element_map = ElementMap(data["element"])
                self._init_tensors(data)
                self._init_box(data["lattice"])
            except KeyError:
                logger.error(
                    f"Cannot find at least one of the expected atomic attributes in the input data:"
                    f"{''.join(self._atomic_attributes)}",
                    exception=KeyError,
                )
        self._init_neighbor(r_cutoff)

        if self.tensors:
            set_as_attribute(self, self.tensors)

        if self.box:
            logger.debug("Shift all atoms into the PBC simulation box")
            self.position = self.box.shift_inside_box(self.position)

        super().__init__()

    def _init_neighbor(self, r_cutoff: float) -> None:
        """
        Initialize a neighbor list instance for a given cutoff radius.

        It ignores updating the neighbor list if the new cutoff radius is the same the existing one.
        It's important to note that the Neighbor object in Structure is considered as a buffer and not
        part of the atomic structure data. But, it uses for calculating descriptors, potential, etc.
        It is the task of those classes to prepare the buffer neighbor for their own usage.
        """
        if not self.neighbor:
            self.neighbor = Neighbor(r_cutoff)
            self.requires_neighbor_update = True
            return

        if self.r_cutoff and self.r_cutoff == r_cutoff:
            logger.info(
                f"Skipping updating the neighbor list (cutoff radius): "
                f"{self.r_cutoff} vs {r_cutoff} (new)"
            )
            return

        self.neighbor.set_cutoff_radius(r_cutoff)
        self.requires_neighbor_update = True

    def set_cutoff_radius(self, r_cutoff: float) -> None:
        """
        Set cutoff radius of the structure.
        This method is useful when having a potential with different cutoff radius.

        :param r_cutoff: new cutoff radius
        :type r_cutoff: float
        """
        self._init_neighbor(r_cutoff)

    def _init_box(self, lattice: List) -> None:
        """
        Create a simulation box object using the provided lattice tensor.

        :param lattice: 3x3 lattice matrix
        :type lattice: Array
        """
        if len(lattice) > 0:
            self.box = Box(lattice)
        else:
            logger.debug("No lattice info were found in structure")
            self.box = Box()

    def _prepare_atype_tensor(self, elements: List[str]) -> Array:
        """
        Set atom types using the element map
        """
        return jnp.asarray(
            [self.element_map(elem) for elem in elements],
            dtype=int,  # FIXME _dtype.INDEX,
        )

    def _init_tensors(self, data: Dict) -> None:
        """
        Create tensors (allocate memory) from the input dictionary of structure data.
        It convert element (string) to atom type (integer) because of computational efficiency.
        """
        logger.debug("Allocating tensors for the structure:")
        self.tensors = {}
        try:
            # Tensors for atomic attributes
            for atomic_attr in self._atomic_attributes:
                self.tensors[atomic_attr] = jnp.asarray(
                    data[atomic_attr],
                    dtype=self.dtype,
                )
            # A tensor for atomic type
            self.tensors["atype"] = self._prepare_atype_tensor(data["element"])

        except KeyError:
            logger.error(
                f"Cannot find expected atomic attribute {atomic_attr} in the input data dictionary",
                exception=KeyError,
            )

        # Logging
        for attr, tensor in self.tensors.items():
            logger.debug(
                f"{attr:12} -> Tensor(shape='{tensor.shape}', dtype='{tensor.dtype}')"
            )

    def update_neighbor(self) -> None:
        """
        Update the neighbor list of the structure.
        This can be computationally expensive.
        """
        self.neighbor.update(self)

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

    # TODO: jit
    def select(self, element: str) -> Array:
        """
        Return all atom ids with atom type same as the input element.

        :param element: element
        :type element: str
        :return: atom indices
        :rtype: Array
        """
        return jnp.nonzero(self.atype == self.element_map[element])[0]

    def get_inputs_per_element(self) -> Tuple:
        """
        A tuple of required info that are used for training and evaluating the potential.
        """

        def extract_inputs():
            for element in self.elements:
                aid = self.select(element)
                yield (
                    self.position[aid],
                    self.position,
                    self.atype,
                    self.box.lattice,
                    self.element_map.element_to_atype,
                )

        return tuple(inputs for inputs in extract_inputs())

    def get_position_per_element(self) -> Tuple[jnp.ndarray]:
        def extract_position():
            for element in self.elements:
                aid = self.select(element)
                yield self.position[aid]

        return tuple(position for position in extract_position())

    def get_force_per_element(self) -> Tuple[jnp.ndarray]:
        def extract_force():
            for element in self.elements:
                aid = self.select(element)
                yield self.force[aid]

        return tuple(force for force in extract_force())

    @property
    def n_atoms(self) -> int:
        return self.position.shape[0]

    @property
    def elements(self) -> List[str]:
        # FIXME: optimize
        return sorted(list({self.element_map(int(at)) for at in self.atype}))

    @property
    def r_cutoff(self) -> float:
        return self.neighbor.r_cutoff

    def __repr__(self) -> str:
        return f"Structure(n_atoms={self.n_atoms}, elements={self.elements}, dtype={self.dtype})"

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Cast the tensors to structure data.
        To be used for dumping structure into a file.
        """
        data = {}
        for name, tensor in self.tensors.items():
            data[name] = np.asarray(tensor)
        return data

    def to_ase_atoms(self) -> AseAtoms:
        """
        This method returns an ASE representation (atoms) of the structure.
        The returned object can be used to visualize or modify the structure using the ASE package.
        """
        logger.info("Creating a representation of the structure in form of ASE atoms")
        BOHR_TO_ANGSTROM = 0.529177  # TODO: Unit class or input length_conversion
        return AseAtoms(
            symbols=[self.element_map(int(at)) for at in self.atype],
            positions=[BOHR_TO_ANGSTROM * np.asarray(pos) for pos in self.position],
            cell=[BOHR_TO_ANGSTROM * float(l) for l in self.box.length]
            if self.box
            else None,  # FIXME: works only for orthogonal cells
        )

    def compare(
        self,
        other: Structure,
        errors: Union[str, List] = "RMSEpa",
        return_difference: bool = False,
    ) -> Dict[str, float]:
        """
        Compare force and total energy values between two structures and return desired errors metrics.

        :param other: other structure
        :type other: Structure
        :param error: a list of error metrics including 'RMSE', 'RMSEpa', 'MSE', and 'MSEpa'. Defaults to ['RMSEpa']
        :type errors: list, optional
        :type return_difference: bool, optional
        :return: whether return energy and force tensor differences or not, defaults to False
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
            result["energy_RMSEpa"] = jnp.sqrt(jnp.mean(eng_diff**2)) / self.n_atoms
        if "mse" in errors:
            result["force_MSE"] = jnp.mean(frc_diff**2)
            result["energy_MSE"] = jnp.mean(eng_diff**2)
        if "msepa" in errors:
            result["force_MSEpa"] = jnp.mean(frc_diff**2)
            result["energy_MSEpa"] = jnp.mean(eng_diff**2) / self.n_atoms
        if return_difference:
            result["frc_diff"] = frc_diff
            result["eng_diff"] = eng_diff

        return result

    def _get_energy_offset(self, atom_energy: Dict[str, float]) -> Array:
        """
        Return a tensor of energy offset.

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
