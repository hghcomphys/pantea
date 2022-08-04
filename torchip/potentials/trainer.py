from ..logger import logger
from ..potentials.base import Potential
from ..datasets.base import StructureDataset
from ..utils.gradient import gradient
from .metrics import MSE
from collections import defaultdict
from typing import Dict
from torch.utils.data import DataLoader as TorchDataLoader
from torch import nn
from math import sqrt
import numpy as np
import torch


class BasePotentialTrainer:
  """
  A trainer class for fitting a generic potential.
  This class must be independent of the type of the potential.

  A derived trainer class, which is specific to a potential, can benefit from the best algorithms to 
  train the model(s) in the potential using energy and force components. 
  """
  pass


class NeuralNetworkPotentialTrainer(BasePotentialTrainer):
  """
  This derived trainer class that trains the neural network potential using energy and force components.

  See https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
  """
  def __init__(self, potential: Potential, **kwargs) -> None:
    """
    Initialize trainer.
    """
    self.potential = potential
    self.learning_rate = kwargs.get('learning_rate', 0.001)
    self.optimizer_func = kwargs.get('optimizer_func', torch.optim.Adam)
    self.optimizer_func_kwargs = kwargs.get('optimizer_func_kwargs', {'lr': 0.001})
    self.criterion = kwargs.get('criterion', nn.MSELoss())
    self.save_best_model = kwargs.get('save_best_model', True)
    self.error_metric=kwargs.get('error_metric', MSE())

    # The implementation can be either as a single or multiple optimizers.
    self.optimizer = self.optimizer_func(
      [{'params': self.potential.model[element].parameters()} for element in self.potential.elements], 
      **self.optimizer_func_kwargs,
    )

  def fit(self, dataset: StructureDataset, **kwargs) -> Dict:
    """
    Fit models.
    """
    # TODO: more arguments to have better control on training
    epochs = kwargs.get("epochs", 1)
    validation_split = kwargs.get("validation_split", None)
    validation_dataset = kwargs.get("validation_dataset", None)

    # Prepare structure dataset and loader for training elemental models
    #dataset_ = dataset.copy() # because of having new r_cutoff specific to the potential, no structure data will be copied
    #dataset_.transform = ToStructure(r_cutoff=self.potential.r_cutoff)     

    # TODO: further optimization using the existing parameters in TorchDataloader 
    # workers, pinned memory, etc.      
    params = {
        "batch_size": 1, 
        #"shuffle": True, 
        #"num_workers": 4,
        #"prefetch_factor": 3,
        #"pin_memory": True,
        #"persistent_workers": True,
        "collate_fn": lambda batch: batch,
    }     

    if validation_dataset:
      # Setting loaders
      train_loader = TorchDataLoader(dataset,            shuffle=True,  **params)
      valid_loader = TorchDataLoader(validation_dataset, shuffle=False, **params)
      # Logging
      logger.debug(f"Using separate training and validation datasets")
      logger.print(f"Number of structures (training)  : {len(dataset)}")
      logger.print(f"Number of structures (validation): {len(validation_dataset)}")
      logger.print()

    elif validation_split:
      nsamples = len(dataset)
      split = int(np.floor(validation_split * nsamples))
      train_dataset, valid_dataset = torch.utils.data.random_split(dataset, lengths=[nsamples-split, split])
      # Setting loaders
      train_loader = TorchDataLoader(train_dataset, shuffle=True,  **params)
      valid_loader = TorchDataLoader(valid_dataset, shuffle=False, **params)
      # Logging
      logger.debug(f"Splitting dataset into training and validation subsets")
      logger.print(f"Number of structures (training)  : {nsamples - split} of {nsamples}")
      logger.print(f"Number of structures (validation): {split} ({validation_split:0.2%} split)")
      logger.print()
    
    else:
      train_loader = TorchDataLoader(dataset, shuffle=True, **params)
      valid_loader = None

    logger.debug("Fitting energy models")
    history = defaultdict(list)
    for epoch in range(epochs+1):
      logger.print(f"[Epoch {epoch}/{epochs}]")

      # ======================================================
      # Set potential models in training status
      self.potential.train()

      nbatch = 0
      train_eng_loss = 0.0
      train_frc_loss = 0.0
      train_eng_error = 0.0
      train_frc_error = 0.0
      # Loop over training structures
      for batch in train_loader:

        # Reset optimizer
        self.optimizer.zero_grad(set_to_none=True)
        
        # TODO: what if batch size > 1
        # TODO: spawn process
        structure = batch[0]
        structure.set_cutoff_radius(self.potential.r_cutoff)
        
        # Calculate energy and force
        energy = self.potential(structure) # total energy
        force = -gradient(energy, structure.position)

        # Energy and force losses
        eng_loss = self.criterion(energy, structure.total_energy); 
        frc_loss = self.criterion(force, structure.force); 
        loss = eng_loss + frc_loss

        # Error metrics
        eng_error = self.error_metric(energy, structure.total_energy, structure.natoms)
        frc_error = self.error_metric(force, structure.force)
        
        # Update weights
        if epoch > 0:
          loss.backward(retain_graph=True)
          self.optimizer.step()

        # Accumulate energy and force loss values for each structure
        train_eng_loss += eng_loss.data.item()
        train_frc_loss += frc_loss.data.item()
        train_loss = train_eng_loss + train_frc_loss

        # Accumulate error metrics for each structure
        train_eng_error += eng_error.data.item()
        train_frc_error += frc_error.data.item()
        
        # Increment number of batches
        nbatch += 1

        logger.print("Training     " \
                    f"loss: {sqrt(train_loss / nbatch):<12.8E}, " \
                    f"energy [{self.error_metric}]: {train_eng_error/nbatch:<12.8E}, " \
                    f"force [{self.error_metric}]: {train_frc_error/nbatch:<12.8E}", end="\r")

      # Get mean training losses
      train_eng_loss /= nbatch
      train_frc_loss /= nbatch
      train_loss = train_eng_loss + train_frc_loss
      train_eng_error /= nbatch
      train_frc_error /= nbatch

      history['train_energy_loss'].append(train_eng_loss)
      history['train_force_loss'].append(train_frc_loss)
      history['train_loss'].append(train_loss)
      history[f'train_energy_{self.error_metric}'].append(train_eng_error)
      history[f'train_force_{self.error_metric}'].append(train_frc_error)

      # ======================================================
      # FIXME: DRY training & validation 

      if valid_loader:
        
        # Set potential models in evaluation status
        self.potential.eval()

        nbatch = 0
        valid_eng_loss = 0.0
        valid_frc_loss = 0.0
        valid_eng_error = 0.0
        valid_frc_error = 0.0
        # Loop over validation structures
        for batch in valid_loader:
          
          # TODO: what if batch size > 1
          # TODO: spawn process
          structure = batch[0]
          structure.set_cutoff_radius(self.potential.r_cutoff)

          # Calculate energy and force components
          energy = self.potential(structure) # total energy
          force = -gradient(energy, structure.position)

          # Energy and force losses
          eng_loss = self.criterion(energy, structure.total_energy); 
          frc_loss = self.criterion(force, structure.force); 
          loss = eng_loss + frc_loss

          # Error metrics
          eng_error = self.error_metric(energy, structure.total_energy, structure.natoms)
          frc_error = self.error_metric(force, structure.force)

          # Accumulate energy and force loss values for each structure
          valid_eng_loss += eng_loss.data.item()
          valid_frc_loss += frc_loss.data.item()
          valid_loss = valid_eng_loss + valid_frc_loss

          # Accumulate error metrics for each structure
          valid_eng_error += eng_error.data.item()
          valid_frc_error += frc_error.data.item()

          # Increment number of batches
          nbatch += 1

        # Get mean validation losses
        valid_eng_loss /= nbatch
        valid_frc_loss /= nbatch
        valid_loss = valid_eng_loss + valid_frc_loss
        valid_eng_error /= nbatch
        valid_frc_error /= nbatch

        history['valid_energy_loss'].append(valid_eng_loss)
        history['valid_force_loss'].append(valid_frc_loss)
        history['valid_loss'].append(valid_loss)
        history[f'valid_energy_{self.error_metric}'].append(valid_eng_error)
        history[f'valid_force_{self.error_metric}'].append(valid_frc_error)

        logger.print()
        logger.print("Validation   " \
                    f"loss: {sqrt(valid_loss):<12.8E}, " \
                    f"energy [{self.error_metric}]: {valid_eng_error:<12.8E}, " \
                    f"force [{self.error_metric}]: {valid_frc_error:<12.8E}")
      
      logger.print()

    #TODO: save the best model
    if self.save_best_model:
      self.potential.save_model()

    return history
