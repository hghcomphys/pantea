from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Protocol

from tqdm import tqdm

from pantea.atoms.element import ElementMap
from pantea.datasets.dataset import Dataset
from pantea.logger import logger
from pantea.potentials.nnp.gradient_descent import GradientDescent
from pantea.potentials.nnp.kalman_filter import KalmanFilter
from pantea.potentials.nnp.potential import NeuralNetworkPotential
from pantea.potentials.nnp.settings import NeuralNetworkPotentialSettings


class UpdaterInterface(Protocol):
    """Interface for potential weight updaters."""

    def fit(
        self,
        dataset: Dataset,
        trainer_params: TrainingParams,
    ) -> Dict[str, Any]: ...


@dataclass
class TrainingParams:
    force_weight: float
    energy_fraction: float
    force_fraction: float
    epochs: int

    @classmethod
    def from_runner(cls, settings: NeuralNetworkPotentialSettings) -> TrainingParams:
        return cls(
            force_weight=settings.force_weight,
            energy_fraction=settings.short_energy_fraction,
            force_fraction=settings.short_force_fraction,
            epochs=settings.epochs,
        )


@dataclass
class NeuralNetworkPotentialTrainer:
    """Train both scaler and model parameters of a neural network potential."""

    params: TrainingParams
    updater: UpdaterInterface
    potential: NeuralNetworkPotential = field(repr=False)

    @classmethod
    def from_runner(
        cls,
        potential: NeuralNetworkPotential,
        filename: str = "input.nn",
    ) -> NeuralNetworkPotentialTrainer:
        potfile = potential.directory / filename
        settings = NeuralNetworkPotentialSettings.from_file(potfile)
        return cls(
            params=TrainingParams.from_runner(settings),
            updater=cls._build_updater(potential, settings),
            potential=potential,
        )

    def fit_scaler(self, dataset: Dataset) -> None:
        """Fit scaler parameters for all the elements."""
        print("Fitting scaler for ACSF descriptor...")
        try:
            dataset_size: int = len(dataset)
            for index in tqdm(range(dataset_size)):
                structure = dataset[index]
                elements = structure.get_unique_elements()
                for element in elements:
                    x = self.potential.atomic_potentials[element].descriptor(structure)
                    scaler = self.potential.atomic_potentials[element].scaler
                    params = self.potential.scalers_params[element]
                    if params is None:
                        params = scaler.fit(x)
                    else:
                        params = scaler.partial_fit(params, x)
                    self.potential.scalers_params[element] = params
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        else:
            print("Done.")

    def fit_model(self, dataset: Dataset) -> Dict[str, Any]:
        """Fit model parameters for all the elements."""
        # kwargs["validation_split"] = kwargs.get(
        #     "validation_split", self.settings.test_fraction
        # )
        # kwargs["epochs"] = kwargs.get("epochs", self.settings.epochs)
        history = defaultdict(list)
        print("Training potential...")
        try:
            history = self.updater.fit(dataset, training_params=self.params)  # type: ignore
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
        for element in self.potential.elements:
            atomic_number: int = ElementMap.get_atomic_number_from_element(element)
            scaler_file = Path(
                self.potential.directory,
                self.potential.scaler_save_format.format(atomic_number),
            )
            logger.info(
                f"Saving scaler parameters for element ({element}): "
                f"{scaler_file.name}"
            )
            scaler = self.potential.atomic_potentials[element].scaler
            params = self.potential.scalers_params[element]
            if params is not None:
                scaler.save(params, scaler_file)
            else:
                logger.warning("No scaler parameters were found. Skipped saving.")

    def save_model(self) -> None:
        """Save model weights separately for all the elements."""
        for element in self.potential.elements:
            atomic_number = ElementMap.get_atomic_number_from_element(element)
            model_file = Path(
                self.potential.directory,
                self.potential.model_save_format.format(atomic_number),
            )
            logger.info(
                f"Saving model weights for element ({element}): {model_file.name}"
            )
            model = self.potential.atomic_potentials[element].model
            model.save(model_file, self.potential.models_params[element])

    def save(self) -> None:
        """Save scaler and model into corresponding files for each element."""
        self.save_scaler()
        self.save_model()

    @classmethod
    def _build_updater(
        cls,
        potential: NeuralNetworkPotential,
        settings: NeuralNetworkPotentialSettings,
    ) -> UpdaterInterface:
        """Initialize selected updater from the potential settings."""
        logger.info("Initializing updater")
        updater: UpdaterInterface
        updater_type: str = settings.updater_type
        if updater_type == "kalman_filter":
            updater = KalmanFilter.from_runner(potential)
        elif updater_type == "gradient_descent":
            updater = GradientDescent(potential)
            # raise NotImplementedError("Updater type: Gradient Descent")
        else:
            logger.error(
                f"Unknown updater type: {updater_type}",
                exception=TypeError,
            )
        logger.info(f"Updater type: {updater_type}")
        return updater
