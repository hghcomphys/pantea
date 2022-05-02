<!-- # TorchIP - A Framework for Machine Learning Interatomic Potential Development -->
<!-- # TorchIP - Torch Interatomic Potential -->

<!-- 
╭━━━━╮╱╱╱╱╱╱╭╮╱╭━━┳━━━╮ \
┃╭╮╭╮┃╱╱╱╱╱╱┃┃╱╰┫┣┫╭━╮┃ \
╰╯┃┃┣┻━┳━┳━━┫╰━╮┃┃┃╰━╯┃ \
╱╱┃┃┃╭╮┃╭┫╭━┫╭╮┃┃┃┃╭━━╯ \
╱╱┃┃┃╰╯┃┃┃╰━┫┃┃┣┫┣┫┃    \
╱╱╰╯╰━━┻╯╰━━┻╯╰┻━━┻╯    
 -->
<img src="./docs/images/logo.png" alt="NNP" width="300"/>
<!-- <p align = "left">
A Framework for Machine Learning Interatomic Potential Development -->
</p>

--------------------------------------------------------------------------------

<!-- ## What is it? -->
TorchIP is a generic and GPU-accelerated framework written in Python/C++ to facilitate the development of emerging machine learning (ML) interatomic potentials. 
TorchIP is intended to help researchers to construct potentials that are employed to perform large-scale molecular dynamics simulations of complex materials in computational physics and chemistry.

The core implementation of TorchIP is based on the [PyTorch](https://github.com/pytorch/pytorch) and its C++ API which provides two high-level features including _optimized tensor computation_ and _automatic differentiation_.

<!--  -->
TorchIP is NOT a molecular dynamics (MD) simulation package BUT a framework to develop ML-based potentials used for the large-scale MD simulators, such as [LAMMPS](https://github.com/lammps/lammps).

<!--  -->
TorchIP is currently under heavy development and the focus is on the implementation of _high-dimensional neural network potential_ proposed by Behler _et al._ ([2007](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401)).

## Installation
```bash
$ pip install torchip
```

## Usage
### High-level API
```python
# Atomic data
loader = RunnerStructureLoader("input.data")

# potential
nnp = NeuralNetworkPotential("input.nn")

# Scaler
nnp.fit_scaler(loader)
nnp.read_scaler("scaler.data")

# Model
nnp.fit_model(loader)
nnp.read_model()

# Train
nnp.fit(loader)

# Inference
energy = nnp.compute(loader)
force = nnp.compute_force(loader)
```

### Low-level API
```python
# Atomic data
loader = RunnerStructureLoader("input.data")
structures = read_structures(loader, between=(1, 5))

# Descriptor
asf = AtomicSymmetryFunction(element="H")
cfn = CutoffFunction(r_cutoff=12.0, cutoff_type="tanh")
asf.register( G2(cutoff_function=cfn, eta=0.5, r_shift=0.0), "H" )
asf.register( G3(cutoff_function=cfn, eta=0.0010, zeta=2.0, lambda0=1.0, r_shift=12.0), "H", "O" )
val = asf(structures[0], aid=0)

# Scaler
scaler = AtomicSymmetryFunctionScaler(scaler_type="scale_center")
scaler.fit(val)
val_scaled = scaler(val)

# Potential
potential = NeuralNetworkModel(**kwargs)

# Train
trainer = NeuralNetworkTrainer(model=potential, **kwargs)
trainer.train(val_scaled)

# Inference
energy = potential(val_scaled)
force = -gradient(energy, structures[0].position)
```
