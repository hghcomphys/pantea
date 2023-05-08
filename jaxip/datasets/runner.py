from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Protocol, TextIO

from jaxip.atoms.structure import Structure
from jaxip.datasets.transformer import ToStructure, Transformer
from jaxip.logger import logger
from jaxip.utils.tokenize import tokenize


class Dataset(Protocol):
    """
    A data container for atom data structure.

    Features:

    * it must access data item in a lazy mode.
    * it should be able to cache data via a `persist` input flag.
    """

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Structure:
        ...

class RunnerDataset(Dataset):
    """
    Dataset for `RuNNer`_ data file format.

    The input structure file contains atom info and simulation box.
    Each snapshot contains two per-atom and collective properties as follows:

    * `per-atom` properties include the element name, positions, energy, charge, force components, etc.
    * `collective` properties such as lattice, total energy, and total charge.

    .. _RuNNer: https://www.uni-goettingen.de/de/560580.html
    """

    def __init__(
        self,
        filename: Path,
        persist: bool = False,
        transform: Optional[Transformer] = None,
    ) -> None:
        """
        Initialize the `RuNNer`_ structure dataset.

        :param filename: Path
        :type filename: path to the RuNNer structure file
        :param persist: Persist structure data in the memory, defaults to False
        :type persist: bool, optional
        :param transform: applied transformation on raw data, default is ToStructure.

        .. _RuNNer: https://www.uni-goettingen.de/de/560580.html
        """
        self.filename: Path = Path(filename)
        self.persist: bool = persist
        self.transform: Transformer = ToStructure() if transform is None else transform
        self._cached_structures: Dict[int, Structure] = dict()
        self._current_index: int = 0

    def __len__(self) -> int:
        """Return number of structures."""
        num_structures: int = 0
        with open(str(self.filename), "r") as file:
            while self._ignore_next_structure(file):
                num_structures += 1
        return num_structures

    def __getitem__(self, index: int) -> Structure:
        """
        Return i-th structure data.

        This is a lazy call which means that only required section
        of data is loaded from the file into the memory.
        """
        return self._read_from_cache(index)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(filename='{str(self.filename)}', transform={self.transform})"
        )

    # ----------------------------------------------

    def _read_next_structure(self, file: TextIO) -> Dict[str, List]:
        """
        Read the next structure from the input file.

        :param file: input structure file handler
        :type file: TextIO
        :return: a sample of thr dataset
        :rtype: Dict
        """
        data: DefaultDict = defaultdict(list)
        while True:
            line: str = file.readline()
            if not line:
                break
            keyword, tokens = tokenize(line)
            if keyword == "atom":
                data["position"].append([float(t) for t in tokens[:3]])
                data["element"].append(tokens[3])
                data["charge"].append(float(tokens[4]))
                data["energy"].append(float(tokens[5]))
                data["force"].append([float(t) for t in tokens[6:9]])
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

    def _ignore_next_structure(self, file: TextIO) -> bool:
        """
        Ignore the next structure.

        :param file: input structure file handler
        :type file: TextIO
        :return: whether the ignoring of next structure was successful or not
        :rtype: bool
        """
        while True:
            line = file.readline()
            if not line:
                return False
            keyword, tokens = tokenize(line)
            if keyword == "end":
                break
        return True

    def _read_structure(self, index: int) -> Structure:
        """
        Read the i-th structure from the input file.

        :param index: index for structures
        :type index: int
        :return: Structure
        :rtype: Any
        """
        logger.debug(f"Reading structure[{index}]")
        with open(str(self.filename), "r") as file:
            for _ in range(index):
                self._ignore_next_structure(file)
            sample = self._read_next_structure(file)
            if not sample:
                raise IndexError(f"index {index} is out of bound with size {len(self)}")
        return self.transform(sample)

    def _read_from_cache(self, index: int) -> Structure:
        """
        Read from the cached structures if the `persist` input flag is enabled.

        :param index: index for structures
        :type index: int
        :return: Structure
        :rtype: Any
        """
        if not self.persist:
            return self._read_structure(index)

        sample: Structure
        if index not in self._cached_structures:
            sample = self._read_structure(index)
            self._cached_structures[index] = sample
        else:
            sample = self._cached_structures[index]

        return sample

    # ----------------------------------------------

    def __next__(self) -> Structure:
        """
        Iterate directly over the dataset.

        .. warning::
            Due to its slow performance, lack of shuffling, and no parallel loading,
            it is recommended to be only used for testing.

        :raises StopIteration: stop iteration
        :return: Return next structure
        :rtype: Transformed structure
        """
        try:
            sample: Structure = self.__getitem__(self._current_index)
            self._current_index += 1
            return sample
        except IndexError:
            self._current_index = 0
            raise StopIteration

    def __iter__(self) -> RunnerDataset:
        return self
