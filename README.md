# MLPOT: Machine-Learning Interatomic Potential

<!-- <img src="./docs/images/logo.png" alt="NNP" width="300"/> -->

<!-- ## What is it? -->

MLPOT is an _framework_ written in Python to facilitate the development of emerging machine learning interatomic potentials. It is intended to help researchers to construct their ML-based potentials and allowing them to perform large-scale molecular dynamics simulations of complex materials in computational physics and chemistry.

### Why MLPOT?

- Having generic and flexible design in mind which allows introducing any atomic descriptors and potentials
- Utilize automatic differentiation instead of explicitly defining gradients of descriptors
- Optimized on CPU, just-in-time compilation and data/task level parallelization
- Support GPU-computing, this speeds up model trainings orders of magnitude

<!--  -->

MLPOT is not a molecular dynamics simulation package (unlike [LAMMPS](https://github.com/lammps/lammps)) but it's in fact a framework to develop ML-based potentials used for the large-scale simulations.

This repository is currently under heavy development and the main focus is on the implementation of _high-dimensional neural network potential (HDNNP)_ proposed by Behler _et al._ ([2007](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)).

## Installation

### Requirements

The following packages have to be installed beforehand:

- [JAX](https://github.com/google/jax) (0.3.23)
- [PyTorch](https://github.com/pytorch/pytorch) (1.12.0)
- [ASE](https://wiki.fysik.dtu.dk/ase/#) (3.22.1)

You can install `MLPOT` using:

```bash
$ pip install mlpot
```

## Examples

### Defining atomic descriptors

The below example shows how to define a vector of Atomic-centered Symmetry Functions
([ASF](https://aip.scitation.org/doi/10.1063/1.3553717)) for an element.
The defined descriptor can be calculated on a given structure and the evaluated vector of descriptor values are eventually used for constructing ML potentials.

```python
from mlpot.datasets import RunnerStructureDataset
from mlpot.descriptors import AtomicSymmetryFunction
from mlpot.descriptors import CutoffFunction, G2, G3

# Read atomic structure data
structures = RunnerStructureDataset('input.data')
structure = structures[0]

# Define descriptor and adding radial and angular terms
descriptor = AtomicSymmetryFunction(element='H')
cfn = CutoffFunction(r_cutoff=12.0, cutoff_type='tanh')
descriptor.add( G2(cfn, eta=0.5, r_shift=0.0), 'H' )
descriptor.add( G3(cfn, eta=0.001, zeta=2.0, lambda0=1.0, r_shift=12.0), 'H', 'O' )

# Calculate descriptor values
values = descriptor(structure)
```

<!--
### Training a potential

This example shows hwo to quickly create a high-dimensional neural network potential ([HDNNP](https://pubs.acs.org/doi/10.1021/acs.chemrev.0c00868)) and training on the input structures. The energy and force components can be evaluated for (new) structures from the trained potential.

```python
from mlpot.datasets import RunnerStructureDataset
from mlpot.potentials import NeuralNetworkPotential
from mlpot.utils import gradient

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
``` -->
