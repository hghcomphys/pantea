from ...logger import logger
from ...potentials.base import Potential
from ...dataset.base import StructureDataset
from ...structure import ToStructure
from ...utils.gradient import gradient
from collections import defaultdict
from typing import Dict
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import numpy as np
import torch


class NeuralNetworkPotentialTrainer:
  """
  This derived trainer class trains the neural network potential using energy and force components (gradients).

  TODO: 
  A base trainer class for fitting a generic potential.
  This class must be independent of the type of the potential.
  A derived trainer class specific to each potential then utilizing the best algorithms to train the models inside the potential
  using energy and force components. 

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

    self.optimizer = {
      element: self.optimizer_func(self.potential.model[element].parameters(), **self.optimizer_func_kwargs) \
      for element in self.potential.elements
    }

  def fit(self, dataset: StructureDataset, **kwargs) -> Dict:
    """
    Fit models.
    """
    # TODO: train and test dataset
    # TODO: more arguments to have better control on training
    epochs = kwargs.get("epochs", 1)
    validation_split = kwargs.get("validation_split", 0.2)
    history = defaultdict(list)

    # Prepare structure dataset and loader (for training model)
    # TODO: a better approach instead of the cloning?
    dataset = dataset.clone() # because of the new transformer, no structure data will be copied
    dataset.transform = ToStructure(r_cutoff=self.potential.r_cutoff)            

    # Train and Validation split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split]) # FIXME: simpler sampler!

    # TODO: further optimization using the existing parameters in TorchDataloader         
    train_loader = TorchDataLoader(
        dataset, 
        batch_size=1, 
        #shuffle=True, 
        #num_workers=2,
        #prefetch_factor=3,
        #pin_memory=True,
        #persistent_workers=True,
        collate_fn=lambda batch: batch,
        sampler=train_sampler,
      )
    valid_loader = TorchDataLoader(
        dataset, 
        batch_size=1, 
        #shuffle=False, 
        #num_workers=2,
        #prefetch_factor=3,
        #pin_memory=True,
        #persistent_workers=True,
        collate_fn=lambda batch: batch,
        sampler=valid_sampler,
      )

    logger.info("Fitting energy models")
    for epoch in range(epochs):
      print(f"Epoch {epoch+1}/{epochs}")  # TODO: print result for epoch 0

      [self.potential.model[element].train() for element in self.potential.elements]

      nbatch = 0
      train_eng_loss = 0.0
      train_frc_loss = 0.0
      # Loop over training structures
      for batch in train_loader:
        
        # TODO: what if batch size > 1
        # TODO: spawn process
        structure = batch[0] 

        # Initialize energy and optimizer
        energy = None
        [self.optimizer[element].zero_grad() for element in self.potential.elements]
        
        # Loop over elements
        for element in self.potential.elements:
          aids = structure.select(element).detach()
          x = self.potential.descriptor[element](structure, aid=aids)
          x = self.potential.scaler[element](x)
          x = self.potential.model[element](x.float())
          x = torch.sum(x, dim=0)
          # FIXME: float type neural network
          energy = x if energy is None else energy + x

        # Calculate force components
        force = -gradient(energy, structure.position)

        # Energy and force losses
        eng_loss = self.criterion(energy.float(), structure.total_energy.float()); 
        frc_loss = self.criterion(force.float(), structure.force.float()); 
        loss = eng_loss + frc_loss
        
        # Update weights
        loss.backward(retain_graph=True)
        [self.optimizer[element].step() for element in self.potential.elements]

        # Accumulate energy and force loss values for each structure
        train_eng_loss += eng_loss.data.item()
        train_frc_loss += frc_loss.data.item()
        train_loss = train_eng_loss + train_frc_loss
        nbatch += 1

        print(f"Training   Loss: {train_loss / nbatch:<12.8E} " \
          f"(Energy: {train_eng_loss / nbatch:<12.8E}, Force: {train_frc_loss / nbatch:<12.8E})", end="\r")

      # Get mean training losses
      train_eng_loss /= nbatch
      train_frc_loss /= nbatch
      train_loss = train_eng_loss + train_frc_loss
      history['train_energy_loss'].append(train_eng_loss)
      history['train_force_loss'].append(train_frc_loss)
      history['train_loss'].append(train_loss)

      # ======================================================

      [self.potential.model[element].eval() for element in self.potential.elements]

      nbatch = 0
      valid_eng_loss = 0.0
      valid_frc_loss = 0.0
      # Loop over validation structures
      for batch in valid_loader:
        
        # TODO: what if batch size > 1
        # TODO: spawn process
        structure = batch[0] 

        # Initialize energy and optimizer
        energy = None
        [self.optimizer[element].zero_grad() for element in self.potential.elements]
        
        # Loop over elements
        for element in self.potential.elements:
          aids = structure.select(element).detach()
          x = self.potential.descriptor[element](structure, aid=aids)
          x = self.potential.scaler[element](x)
          x = self.potential.model[element](x.float())
          x = torch.sum(x, dim=0)
          # FIXME: float type neural network
          energy = x if energy is None else energy + x

        # Calculate force components
        force = -gradient(energy, structure.position)

        # Energy and force losses
        eng_loss = self.criterion(energy.float(), structure.total_energy.float()); 
        frc_loss = self.criterion(force.float(), structure.force.float()); 
        loss = eng_loss + frc_loss

        # Accumulate energy and force loss values for each structure
        valid_eng_loss += eng_loss.data.item()
        valid_frc_loss += frc_loss.data.item()
        valid_loss = valid_eng_loss + valid_frc_loss
        nbatch += 1

      # Get mean validation losses
      valid_eng_loss /= nbatch
      valid_frc_loss /= nbatch
      valid_loss = valid_eng_loss + valid_frc_loss
      history['valid_energy_loss'].append(valid_eng_loss)
      history['valid_force_loss'].append(valid_frc_loss)
      history['valid_loss'].append(valid_loss)

      print()
      print(f"Validation Loss: {valid_loss:<12.8E} "\
        f"(Energy: {valid_eng_loss/nbatch:<12.8E}, Force: {valid_frc_loss:<12.8E})")
      print()

    if self.save_best_model:
      self.potential.save_model()

    return history
