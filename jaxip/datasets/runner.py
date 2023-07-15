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

    The dataset used for the input data format of `RuNNer`_ consists of atomic attributes
    and simulation box information. Within each snapshot, there are two types of
    properties: `per-atom` properties and `collective` properties.

    The per-atom properties encompass various attributes like the element name,
    positions, energy, charge, force components, and more.

    On the other hand, the collective properties include attributes
    such as lattice parameters, total energy, and total charge.

    .. _RuNNer: https://www.uni-goettingen.de/de/560580.html
    """

    def __init__(
        self,
        filename: Path,
        persist: bool = False,
        dtype: Optional[Dtype] = None,
    ) -> None:
        """
        Create a `RuNNer`_ structure dataset by initializing it from an input file.

        :param filename: input file name
        :type filename: Path
        :param persist: Persist any loaded structure data in the memory, defaults to False
        :type persist: bool, optional
        :param dtype: floating point precision for the structure data, defaults to None
        :type dtype: Optional[Dtype], optional

        .. _RuNNer: https://www.uni-goettingen.de/de/560580.html
        """
        self.filename: Path = Path(filename)
        self.persist: bool = persist
        self._cache: Dict[int, Structure] = dict()
        self.dtype = dtype if dtype is not None else _dtype.FLOATX

    def __len__(self) -> int:
        """Return number of available structures."""
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
        return self._read_from_cache(index)

    def read_sequential(self) -> Iterator[Structure]:
        """
        Read structures sequentially.

        It must be noted that reading data in a consecutive manner is
        faster compared to random indexing read.

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

    def preload(self) -> None:
        """
        Preload (cache) all structures into memory.

        This ensures that any structure can be rapidly loaded from memory in subsequent operations.
        """
        self.persist = True
        for _ in self.read_sequential():
            pass

    @classmethod
    def _read_next(cls, file: TextIO) -> Dict[str, List]:
        """Read next structure data between `begin` and `end` keywords."""
        data = defaultdict(list)
        read_block: bool = False
        while True:
            line = file.readline()
            if not line:
                break
            keyword, _ = tokenize(line)
            if keyword == "begin":
                read_block = True
                break
        while read_block:
            line = file.readline()
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
                read_block = False
        return data

    @classmethod
    def _ignore_next(cls, file: TextIO) -> bool:
        while True:
            line = file.readline()
            if not line:
                return False
            keyword, _ = tokenize(line)
            if keyword == "end":
                break
        return True

    def _read(self, index: int) -> Structure:
        logger.debug(f"load structure({index=})")
        with open(str(self.filename), "r") as file:
            for _ in range(index):
                self._ignore_next(file)
            data = self._read_next(file)
            if not data:
                raise IndexError(
                    f"The given index {index} is out of bound (len={len(self)})"
                )
        return self._to_structure(data)

    def _to_structure(self, data: Dict[str, List]) -> Structure:
        return Structure.from_dict(data, dtype=self.dtype)

    def _read_from_cache(self, index: int) -> Structure:
        """Read the desired structure from cache, if possible."""
        if not self.persist:
            return self._read(index)
        if index not in self._cache:
            structure = self._read(index)
            self._cache[index] = structure
            return structure
        else:
            return self._cache[index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(filename='{str(self.filename)}'"
            f", persist={self.persist}, dtype={self.dtype.dtype})"
        )
