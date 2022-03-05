# MLP-framework

## What is it?
_Machine Learning Potential framework_ is a generic and GPU-accelerated library written in Python/C++ to facilitate the development of emerging machine learning interatomic potentials. Such potentials are employed to perform large-scale molecular dynamics simulations of complex materials in computational physics and chemistry. 
 
<!--  -->
_MLP framework_ is NOT a molecular dynamics (MD) simulation package but a framework to construct ML-based force fields employed for the MD simulations.

<!--  -->
The current focus is on the implementation of high-dimensional neural network potential (NNP) proposed by Behler _et al._ ([2007](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401))


## Designed API
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

<!-- 
#### Implementation TODOs
- [ ] define a customized exception class that handles internal error messages and also python exceptions
- [ ] improve CFG design e.g. config file, defaults values, on-the-fly settings.
- [ ] optimize memory allocation of neighbor list atoms and avoiding redundant tensor creation (use torch.resizes)
- [ ] optimize neighbor list update for large number of atoms (not used for training but MD simulations)
- [ ] utilize torch multi-thread or distributed torch
- [ ] optimize code performance regarding python dynamic types (torch script, cython)
- [ ] parallelize descriptor calculations using vectorization or thread pool

#### Optimization approach
- torch.jit.trace
- torch.jit.script
- torch.jit.Future
-->

