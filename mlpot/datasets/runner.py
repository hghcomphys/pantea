from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, TextIO, Union

from mlpot.datasets.base import StructureDataset
from mlpot.datasets.transformer import ToStructure, Transformer
from mlpot.logger import logger
from mlpot.utils.tokenize import tokenize


class RunnerStructureDataset(StructureDataset):
    """
    Structure dataset for RuNNer data file format.
    The input structure file contains atoms info inside a simulation box.

    Each snapshot includes per-atom and collective properties:
    - per-atom properties include the element name, atom coordinates, energy, charge, and force components.
    - collective properties are the lattice matrix, total energy, and total charge.

    See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    # TODO: add logging

    def __init__(
        self,
        filename: Path,
        transform: Transformer = ToStructure(),
        persist: bool = False,
        # TODO: download: bool = False,
    ) -> None:
        """
        Initialize RuNNer structure dataset.

        :param filename: Path
        :type filename: RuNNer structure file name
        :param transform: transform to be applied on a structure, defaults to ToStructure()
        :type transform: Transformer, optional
        :param persist: Persist structure data in the memory, defaults to False
        :type persist: bool, optional
        """

        self.filename: Path = Path(filename)
        self.transform: Transformer = transform  # transform after loading each sample
        self.persist: bool = persist  # enabling caching
        self._cached_structures: Dict = dict()  # a dictionary of cached structures
        self._current_index: int = 0  # used only for direct iteration
        super().__init__()

    def __len__(self) -> int:
        """
        This method opens the structure file and return the number of structures.
        """
        n_structures: int = 0
        with open(str(self.filename), "r") as file:
            while self.ignore(file):
                n_structures += 1
        return n_structures

    def __getitem__(self, index: Union[int, List[int]]) -> Any:
        """
        Return i-th structure.
        This is a lazy load. Data is read from the file only if this method is called.
        Multiple-indexing with limitation is possible.
        """
        # TODO: assert range of index
        if isinstance(index, list):
            return [self._read_cache(idx) for idx in index]

        return self._read_cache(index)

    def __next__(self):
        """
        This method is used for iterating directly over the dataset instance.
        Due to its slow performance, lack of shuffling, and no parallel loading,
        it's better to be only used for testing and debugging.

        :raises StopIteration: stop iteration
        :return: Return next structure
        :rtype: Transformed structure
        """
        if self._current_index < len(self):
            sample = self[self._current_index]
            self._current_index += 1
            return sample

        self._current_index = 0
        raise StopIteration

    def __iter__(self) -> RunnerStructureDataset:
        return self

    def ignore(self, file: TextIO) -> bool:
        """
        This method ignores the next structure.
        It reduces the spent time while reading a range of structures.

        Args:
            file (TextIO): Input structure file handler

        Returns:
            bool: whether ignoring the next structure was successful or not
        """
        # Read next structure
        while True:
            # Read one line from file
            line = file.readline()
            if not line:
                return False

            keyword, tokens = tokenize(line)
            # TODO: check begin keyword
            if keyword == "end":
                break

        return True

    def read(self, file: TextIO) -> Dict:
        """
        This method reads the next structure from the input file.

        :param file: input structure file handler
        :type file: TextIO
        :return: sample of dataset
        :rtype: Dict
        """

        sample = defaultdict(list)
        # Read next structure
        while True:
            # Read one line from file handler
            line = file.readline()
            if not line:
                return False

            # Read keyword and values
            keyword, tokens = tokenize(line)
            # TODO: check begin keyword
            if keyword == "atom":
                sample["position"].append([float(t) for t in tokens[:3]])
                sample["element"].append(tokens[3])
                sample["charge"].append(float(tokens[4]))
                sample["energy"].append(float(tokens[5]))
                sample["force"].append([float(t) for t in tokens[6:9]])
            elif keyword == "lattice":
                sample["lattice"].append([float(t) for t in tokens[:3]])
            elif keyword == "energy":
                sample["total_energy"].append(float(tokens[0]))
            elif keyword == "charge":
                sample["total_charge"].append(float(tokens[0]))
            elif keyword == "comment":
                sample["comment"].append(" ".join(line.split()[1:]))
            elif keyword == "end":
                break

        return sample

    def _read_cache(self, index: int) -> Any:
        """
        This method reads cached structure if persist flag is True.
        """
        if not self.persist:
            return self._read_and_transform(index)

        if index not in self._cached_structures:
            sample = self._read_and_transform(index)
            self._cached_structures[index] = sample
        else:
            # logger.debug(f"Using cached structure {index}")
            sample = self._cached_structures[index]

        return sample

    def _read_and_transform(self, index: int):
        """
        This method reads the i-th structure and then applying the transformation.
        """
        logger.debug(f"Reading structure[{index}]")
        with open(str(self.filename), "r") as file:
            for _ in range(index):
                self.ignore(file)
            sample = self.read(file)

            if self.transform:
                sample = self.transform(sample)

        return sample
