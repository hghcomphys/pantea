from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol

from tqdm import tqdm

from pantea.atoms.element import ElementMap
from pantea.datasets.dataset import Dataset
from pantea.logger import logger
from pantea.potentials.nnp.gradient_descent import GradientDescent
from pantea.potentials.nnp.kalman_filter import KalmanFilter
from pantea.potentials.nnp.potential import NeuralNetworkPotential


class UpdaterInterface(Protocol):
    """Interface for potential weight updaters."""

    def fit(self, dataset: Dataset) -> Dict[str, Any]: ...


@dataclass
class NeuralNetworkPotentialTrainer:
    """Train both scaler and model parameters of a neural network potential."""

    potential: NeuralNetworkPotential
    updater: UpdaterInterface

    @classmethod
    def from_runner(
        cls,
        potential: NeuralNetworkPotential,
    ) -> NeuralNetworkPotentialTrainer:
        return cls(
            potential=potential,
            updater=cls._build_updater(potential),
        )

    @classmethod
    def _build_updater(
        cls,
        potential: NeuralNetworkPotential,
    ) -> UpdaterInterface:
        """Initialize selected updater from the potential settings."""
        logger.info("Initializing updater")
        updater: UpdaterInterface
        updater_type: str = potential.settings.updater_type
        if updater_type == "kalman_filter":
            updater = KalmanFilter(potential)
        elif updater_type == "gradient_descent":
            updater = GradientDescent(potential)
            raise NotImplementedError("Updater type: Gradient Descent")
        else:
            logger.error(
                f"Unknown updater type: {updater_type}",
                exception=TypeError,
            )
        logger.info(f"Updater type: {updater_type}")
        return updater

    def fit_scaler(self, dataset: Dataset) -> None:
        """Fit scaler parameters for all the elements using the input structure data."""
        print("Fitting scaler for ACSF descriptor...")
        try:
            dataset_size: int = len(dataset)
            for index in tqdm(range(dataset_size)):
                structure = dataset[index]
                elements = structure.get_unique_elements()
                for element in elements:
                    x = self.potential.atomic_potentials[element].descriptor(structure)
                    self.potential.atomic_potentials[element].scaler.fit(x)
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        else:
            print("Done.")

    def fit_model(self, dataset: Dataset) -> Dict[str, Any]:
        """Fit model for all elements using the input structure loader."""
        # kwargs["validation_split"] = kwargs.get(
        #     "validation_split", self.settings.test_fraction
        # )
        # kwargs["epochs"] = kwargs.get("epochs", self.settings.epochs)
        history = defaultdict(list)
        print("Training potential...")
        try:
            history = self.updater.fit(dataset)  # type: ignore
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        else:
            print("Done.")
        return history

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        """Fit both scaler and the model."""
        self.fit_scaler(dataset)
        history = self.fit_model(dataset)
        return history

    def save_scaler(self) -> None:
        """This method saves scaler parameters for all the elements."""
        # Save scaler parameters for each element separately
        for element in self.potential.settings.elements:
            atomic_number: int = ElementMap.get_atomic_number_from_element(element)
            scaler_file = Path(
                self.potential.directory,
                self.potential.settings.scaler_save_naming_format.format(atomic_number),
            )
            logger.info(
                f"Saving scaler parameters for element ({element}): {scaler_file.name}"
            )
            self.potential.atomic_potentials[element].scaler.save(scaler_file)

    def save_model(self) -> None:
        """Save model weights separately for all the elements."""
        for element in self.potential.elements:
            atomic_number = ElementMap.get_atomic_number_from_element(element)
            model_file = Path(
                self.potential.directory,
                self.potential.settings.model_save_naming_format.format(atomic_number),
            )
            logger.info(
                f"Saving model weights for element ({element}): {model_file.name}"
            )
            self.potential.atomic_potentials[element].model.save(
                model_file, self.potential.models_params[element]
            )

    def save(self) -> None:
        """Save scaler and model into corresponding files for each element."""
        self.save_scaler()
        self.save_model()
