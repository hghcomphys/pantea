=======
History
=======


0.10
-------------------
* added an optimized implementation (100x faster) of ACSF gradient calculation per element

0.9
-------------------
* required adjustments on the base codes and example notebooks to work with recent version of JAX
* added `autodiff` gradient method to LJ force calculations

0.8
-------------------
* Changed package name to `Pantea` 

0.7
-------------------
* Refactored structure including design, documentation, and performance (JIT kernels)
* Refactored RuNNer dataset

0.6
-------------------
* Implemented Molecular dynamics (MD) simulator
* Added Monte-Carlo (MC) simulator

0.5
-------------------
* Implemented Kalman filter trainer 

0.4
-------------------
* Applied extensive refactoring
* Replaced `PyTorch` main dependency with `JAX`
* First release on PyPI.

0.3
-------------------
* `JAX` optimizations applied to the `ACSF` descriptor

0.2
-------------------
* Some optimizations using `torch.jit.script`

0.1
-------------------
* Primary implementation and validation