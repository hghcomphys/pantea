# MLP Framework

## What is it?
_Machine Learning Potential Framework_ is a generic and GPU-accelerated package written in Python/C++ to facilitate the development of emerging machine learning interatomic potentials which are used for performing large-scale molecular dynamics simulations of complex materials in computational physics and chemistry. 
 
<!--  -->
_MLP framework_ is NOT a molecular dynamics (MD) simulation package but a framework to construct ML-based force fields employed for the MD simulations.

<!--  -->
The current focus is on the implementation of high-dimensional neural network potential (NNP) proposed by Behler _et al._ ([2007](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401))


### TODOs
- [ ] handle internal err message from a defined MLP exceptions
- [ ] improve CFG design e.g. config file, defaults values, on-the-fly settings.
- [ ] optimize memory allocation of neighbor list atoms and avoiding redundant tensor creation (use torch.resizes)
- [ ] optimize neighbor list update for large number of atoms (not used for training but MD simulations)
- [ ] Complete ASF including reading input.nn 
- [ ] validating ASF values and gradients
- [ ] vectorize descriptor calculation for array of atom ids
- [ ] torch multi-thread
- [x] ASF scalers
- [x] add angular ASF
- [x] optimize loader to read thousands of structures from input file
- [x] quickly ignore unwanted structure  
- [x] add dtype_index to CFG
- [x] remove intermediate _data from loader and structure classes



