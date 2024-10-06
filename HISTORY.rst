=======
History
=======

0.11
-------------------
* refactored Scaler, Potential, and Trainer
* optimized Kalman Filter JIT kernels
* separated scaler and model parameters from potential
* bug fixes


0.10
-------------------
* added an optimized implementation (100x faster) of the ACSF gradient calculation

0.9
-------------------
* required adjustments on the base codes and example notebooks to work with the recent version of JAX
* added `autodiff` gradient method to the LJ force calculation kernel

0.8
-------------------
* Changed package name to `Pantea` 

0.7
-------------------
* Refactored structure including design, documentation, and performance (JIT kernels)
* Refactored RuNNer dataset

0.6
-------------------
* Implemented a Molecular dynamics (MD) simulator
* Monte-Carlo (MC) simulator

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