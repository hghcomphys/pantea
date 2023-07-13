from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TextIO

from jaxip.atoms.structure import Structure
from jaxip.datasets.dataset import DatasetInterface
from jaxip.logger import logger
from jaxip.types import Dtype, _dtype
from jaxip.utils.tokenize import tokenize


class RunnerDataset(DatasetInterface):
    """
    Dataset for `RuNNer`_ input data format.

    The input structure file contains atomic attribute  and a simulation box.
    Each snapshot contains per-atom and collective properties as follows:

    * `per-atom` properties include the element name, positions, energy, charge, force components, etc.
    * `collective` properties such as lattice, total energy, and total charge.

    .. _RuNNer: https://www.uni-goettingen.de/de/560580.html
    """

    def __init__(
        self,
        filename: Path,
        persist: bool = False,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """
        Initialize `RuNNer`_ structure dataset from input file.

        :param filename: input file name
        :type filename: Path
        :param persist: Persist structure data in the memory, defaults to False
        :type persist: bool, optional
        :param dtype: floating point precision for the structure data, defaults to None
        :type dtype: Optional[Dtype], optional

        .. _RuNNer: https://www.uni-goettingen.de/de/560580.html
        """
        self.filename: Path = Path(filename)
        self.persist: bool = persist
        self._cache: Dict[int, Structure] = dict()
        self._current_index: int = 0
        if dtype is None:
            self.dtype = _dtype.FLOATX

    def __len__(self) -> int:
        """Return number of structures."""
        num_structures: int = 0
        with open(str(self.filename), "r") as file:
            while self._ignore_next(file):
                num_structures += 1
        return num_structures

    def __getitem__(self, index: int) -> Structure:
        """
        Return i-th structure.

        This is a lazy call which means that only required section
        of data is loaded into the memory.
        """
        return self._read_structure_from_cache(index)

    def read_structures(self) -> Iterator[Structure]:
        """
        Read structures sequentially.

        :return: Structure
        :rtype: Iterator[Structure]
        """
        logger.debug("Read structures sequentially")
        index: int = 0
        with open(str(self.filename), "r") as file:
            while True:
                data = self._read_next(file)
                if not data:
                    break
                structure = self._to_structure(data)
                if self.persist:
                    if index in self._cache:
                        yield self._cache[index]
                    else:
                        self._cache[index] = structure
                yield structure
                index += 1

    def _read_next(self, file: TextIO) -> Dict[str, List]:
        data = defaultdict(list)
        while True:
            line: str = file.readline()
            if not line:
                break
            keyword, tokens = tokenize(line)
            if keyword == "atom":
                data["positions"].append([float(t) for t in tokens[:3]])
                data["elements"].append(tokens[3])
                data["charges"].append(float(tokens[4]))
                data["energies"].append(float(tokens[5]))
                data["forces"].append([float(t) for t in tokens[6:9]])
            elif keyword == "lattice":
                data["lattice"].append([float(t) for t in tokens[:3]])
            elif keyword == "energy":
                data["total_energy"].append(float(tokens[0]))
            elif keyword == "charge":
                data["total_charge"].append(float(tokens[0]))
            elif keyword == "comment":
                data["comment"].append(" ".join(line.split()[1:]))
            elif keyword == "end":
                break
        return data

    def _ignore_next(self, file: TextIO) -> bool:
        while True:
            line = file.readline()
            if not line:
                return False
            keyword, tokens = tokenize(line)
            if keyword == "end":
                break
        return True

    def _read_structure(self, index: int) -> Structure:
        logger.debug(f"loading structure({index=})")
        with open(str(self.filename), "r") as file:
            for _ in range(index):
                self._ignore_next(file)
            data = self._read_next(file)
            if not data:
                raise IndexError(
                    f"The given index {index} is out of bound (len={len(self)})"
                )
        return Structure.from_dict(data, dtype=self.dtype)

    def _to_structure(self, data: Dict[str, List]) -> Structure:
        return Structure.from_dict(data, dtype=self.dtype)

    def _read_structure_from_cache(self, index: int) -> Structure:
        if not self.persist:
            return self._read_structure(index)
        if index not in self._cache:
            structure = self._read_structure(index)
            self._cache[index] = structure
            return structure
        else:
            return self._cache[index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(filename='{str(self.filename)}', dtype={self.dtype.dtype})"
        )
