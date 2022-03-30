# TorchIP - Torch Interatomic Potential

<!-- A Machine Learning Framework for Interatomic Potential Development -->

## What is it?
_TorchIP_ is a generic and GPU-accelerated framework written in Python/C++ to facilitate the development of emerging machine learning (ML) interatomic potentials. A reconstructed potential is employed to perform large-scale molecular dynamics simulations of complex materials in computational physics and chemistry. The core implementation of TorchIP is based on [PyTorch](https://github.com/pytorch/pytorch) package  and its C++ API which provides two high-level features including tensor computation with strong GPU acceleration and automatic differentiation.


<!--  -->
TorchIP is NOT a molecular dynamics (MD) simulation package but a framework to train ML-based force fields employed for the MD simulations.

<!--  -->
TorchIP is under substantial development and the current focus is on the implementation of high-dimensional neural network potential (NNP) proposed by Behler _et al._ ([2007](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401))


## How to use it? 
### High-level API
```python
# Atomic data
loader = RunnerStructureLoader("input.data")

# potential
nnp = NeuralNetworkPotential("input.nn")

# Scaler
nnp.fit_scaler(loader)
nnp.read_scaler("scaling.data")

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
asf.add( G2(cutoff_function=cfn, eta=0.5, r_shift=0.0), "H" )
asf.add( G3(cutoff_function=cfn, eta=0.0010, zeta=2.0, lambda0=1.0, r_shift=12.0), "H", "O" )
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