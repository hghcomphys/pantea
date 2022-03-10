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

### Optimization
- Serial vs. multi-thread/process vs. GPU
- TorchScript JIT: torch.jit.trace, torch.jit.script, torch.jit.Future (GIL limitation?)
- C++ Extension https://pytorch.org/tutorials/advanced/cpp_extension.html
- C++ Frontend API https://pytorch.org/tutorials/advanced/cpp_frontend.html

### How big is the code?
```bash
$ pygount --format=summary ./
```

<!-- 
#### TODOs
- [ ] define a customized exception class that handles internal error messages and also python exceptions
- [ ] improve CFG design e.g. config file, defaults values, on-the-fly settings.
- [ ] optimize memory allocation of neighbor list atoms and avoiding redundant tensor creation (use torch.resizes)
- [ ] optimize neighbor list update for large number of atoms (not used for training but MD simulations)
- [ ] utilize torch multi-thread or distributed torch
- [ ] optimize code performance regarding python dynamic types (torch script, cython)
- [ ] parallelize descriptor calculations using vectorization or thread pool
 -->

### Theoretical background
Force field molecular dynamics simulation is a powerful tool for exploring the equilibrium and transport properties of many-body systems on the atomic scaler. A force field regenerates interatomic interactions in a functional form. which boosts the computational efficiency and thus lets us reach time and length scales far beyond what is possible with direct _ab initio_ molecular dynamics, where energy and forces are calculated on the fly. However, the power of such simulations would be enormously enhanced if the potentials used to simulate materials were not limited by the given functional forms but accurately represented the Born-Oppenheimer potential energy surface. Conventional force fields are typically constructed on the basis of physical assumptions and are therefore limited to specific functional forms which are less suitable for reactive large-scale MD simulations of materials with strong interfacial chemical characteristics.

To overcome this limitation, __data-intensive__ approach is beginning to emerge as a separate approach. There are currently two different approaches that successfully apply machine learning methods to interatomic potential development 1) Gaussian approximated potential (GAP) [P. Rowe et al., Phys. Rev. B 97, 054303 (2018)] and 2) neural network potential (NNP) [J. Behler, J. Chem. Phys. 145, 170901 (2016)]. Both frameworks rely purely on atomic energy and forces data obtained from first-principles calculations (e.g. DFT) and attempt to realize an accurate representation of the potential energy surface. This is achieved at a computational __efficiency__ which is orders of magnitude larger than that of comparable calculations which directly invoke electronic structure methods while preserving quantum mechanical __accuracy__.

#### High-dimensional neural network potential
High-dimensional NNPs are constructed using the method proposed by Behler and Parrinello. The total energy of the system is determined as a sum of environment-dependent atomic energies. For each atom in the system, the positions of all neighboring atoms inside a cutoff radius are described by a set of atom-centered many-body symmetry functions. These sets of function values are then used as input features for the atomic neural networks that give the atomic energy contributions and
corresponding derivatives as force components applied on each atom.

<img src="./docs/images/nnp.png" alt="NNP" width="300"/>
<p align = "left">
Figure 1 - High-dimensional
neural network potential.
</p>