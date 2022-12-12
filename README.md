# MLPOT

<!-- Machine-Learning Interatomic Potential -->

<!-- <img src="./docs/images/logo.png" alt="NNP" width="300"/> -->

<!-- ## What is it? -->

MLPOT is a Python package to facilitate development of the emerging machine-learning (ML) interatomic potentials in computational physics and chemistry. Such potentials are essential for performing large-scale molecular dynamics (MD) simulations of complex materials at the atomic scale and with ab initio accuracy.

### Why MLPOT?

- Offers a generic and flexible design which allows introducing any atomic descriptors and potentials
- Utilizes automatic differentiation that makes definition of new descriptors quite easy
- Pythonic design with an optimized implementation using just-in-time compilations and task/data parallelization
- Supports GPU-computing that can speeds up model trainings orders of magnitude

<!--  -->

MLPOT is not a simulation package but a framework to develop ML-based potentials used for the MD simulations.

This repository is under heavy development and the current focus is on the implementation of _high-dimensional neural network potential (HDNNP)_ proposed by Behler _et al._ ([2007](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)).

## Install

### Requirements

The following packages have to be installed beforehand:

- [JAX](https://github.com/google/jax) (0.3.23)
- [ASE](https://wiki.fysik.dtu.dk/ase/#) (3.22.1)
<!-- - [PyTorch](https://github.com/pytorch/pytorch) (1.12.0) -->

We can install `MLPOT` via pip:

```bash
$ pip install mlpot
```

## Examples

### Defining an atomic descriptor

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
