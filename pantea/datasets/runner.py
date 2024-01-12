from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, TextIO

from pantea.atoms.structure import Structure
from pantea.types import Dtype, default_dtype
from pantea.utils.tokenize import tokenize


class RunnerDataSource:
    """

    The class is intended for the input data format of `RuNNer`_ consists of atomic attributes
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
        dtype: Optional[Dtype] = None,
    ) -> None:
        """
        Create a `RuNNer`_ structure data by initializing it from an input file.

        :param filename: input file name
        :type filename: Path
        :param dtype: precision for the structure data, defaults to None
        :type dtype: Optional[Dtype], optional

        .. _RuNNer: https://www.uni-goettingen.de/de/560580.html
        """
        self.filename = Path(filename)
        self.dtype = dtype if dtype is not None else default_dtype.FLOATX

    def __len__(self) -> int:
        """Return number of available structures."""
        num_structures: int = 0
        with open(str(self.filename), "r") as file:
            while self._ignore_next_structure(file):
                num_structures += 1
        return num_structures

    def __getitem__(self, index: int) -> Structure:
        """
        Return i-th structure.

        This is a lazy call which means that only required section
        of data is loaded into the memory.
        """
        with open(str(self.filename), "r") as file:
            for _ in range(index):
                self._ignore_next_structure(file)
            data = self._read_next_structure(file)
            if not data:
                raise IndexError(
                    f"The given index {index} is out of bound (len={len(self)})"
                )
        return self._to_structure(data)

    def read_structures(self) -> Iterator[Structure]:
        """
        Read structures consecutively.

        It must be noted that reading data in a consecutive manner from file is
        faster compared to indexing read. This can be used for performant preloading
        of structures into the memory, if needed.

        :return: Structure
        :rtype: Iterator[Structure]
        """
        with open(str(self.filename), "r") as file:
            while True:
                data = self._read_next_structure(file)
                if not data:
                    break
                yield self._to_structure(data)

    @classmethod
    def _read_next_structure(cls, file: TextIO) -> Dict[str, List]:
        """Read next structure."""
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
    def _ignore_next_structure(cls, file: TextIO) -> bool:
        while True:
            line = file.readline()
            if not line:
                return False
            keyword, _ = tokenize(line)
            if keyword == "end":
                break
        return True

    def _to_structure(self, data: Dict[str, List]) -> Structure:
        return Structure.from_dict(data, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filename='{str(self.filename)}', dtype={self.dtype.dtype})"
