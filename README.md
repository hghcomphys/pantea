<!-- # TorchIP - A Framework for Machine Learning Interatomic Potential Development -->
# TorchIP - PyTorch Interatomic Potential

<!-- 
╭━━━━╮╱╱╱╱╱╱╭╮╱╭━━┳━━━╮ \
┃╭╮╭╮┃╱╱╱╱╱╱┃┃╱╰┫┣┫╭━╮┃ \
╰╯┃┃┣┻━┳━┳━━┫╰━╮┃┃┃╰━╯┃ \
╱╱┃┃┃╭╮┃╭┫╭━┫╭╮┃┃┃┃╭━━╯ \
╱╱┃┃┃╰╯┃┃┃╰━┫┃┃┣┫┣┫┃    \
╱╱╰╯╰━━┻╯╰━━┻╯╰┻━━┻╯    
 -->

<!-- <img src="./docs/images/logo.png" alt="NNP" width="300"/> -->

<!-- ## What is it? -->
TorchIP is a __generic__ and __GPU-accelerated__ software _framework_ written to facilitate the development of emerging machine learning interatomic potentials. 
It is intended to help researchers to quickly construct ML-based potentials and allowing them to perform large-scale molecular dynamics simulations of complex materials in computational physics and chemistry.

The core implementation of TorchIP relies on [PyTorch](https://github.com/pytorch/pytorch) which provides two high-level features including _optimized tensor computation_ and _automatic differentiation_.

<!--  -->
TorchIP is NOT a molecular dynamics (MD) simulation package but a framework to develop ML-based potentials used for the large-scale MD simulators (e.g. [LAMMPS](https://github.com/lammps/lammps)).

<!--  -->
This repository is currently under heavy development and the focus is on the implementation of _high-dimensional neural network potential (HDNNP)_ proposed by Behler _et al._ ([2007](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)).

## Installation
### Dependencies
- [PyTorch](https://github.com/pytorch/pytorch)
- [ASE](https://wiki.fysik.dtu.dk/ase/#)

```bash
$ pip install torchip
```

## Usage
### Defining atomic descriptors
```python
from torchip.datasets import RunnerStructureDataset
from torchip.descriptors import AtomicSymmetryFunction  
from torchip.descriptors import CutoffFunction, G2, G3

# Read atomic structure data
structures = RunnerStructureDataset('input.data')
structure = structures[0]

# Define descriptor and adding radial and angular terms
descriptor = AtomicSymmetryFunction(element='H')
cfn = CutoffFunction(r_cutoff=12.0, cutoff_type='tanh')
descriptor.register( G2(cfn, eta=0.5, r_shift=0.0), 'H' )
descriptor.register( G3(cfn, eta=0.001, zeta=2.0, lambda0=1.0, r_shift=12.0), 'H', 'O' )

# Calculate descriptor values
results = descriptor(structure)
```

### Training a potential
```python
from torchip.datasets import RunnerStructureDataset
from torchip.potentials import NeuralNetworkPotential
from torchip.utils import gradient

# Atomic data
structures = RunnerStructureDataset("input.data")

# Potential
nnp = NeuralNetworkPotential("input.nn")

# Descriptor
nnp.fit_scaler(structures)
#nnp.load_scaler()

# Train 
nnp.fit_model(structures)
#nnp.load_model()

# Predict energy and force components
structure = structures[0]
energy = nnp(structure)
force = -gradient(energy, structure.position)
```



