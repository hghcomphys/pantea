# Notes
Technical notes regarding implementation, validation, references, etc.

<!--
TODO: 
- [ ] define a customized exception class that handles internal error messages and also python exceptions
- [ ] improve CFG design e.g. config file, defaults values, on-the-fly settings.
- [ ] optimize memory allocation of neighbor list atoms and avoiding redundant tensor creation (use torch.resizes)
- [ ] optimize neighbor list update for large number of atoms (not used for training but MD simulations)
- [ ] utilize torch multi-thread or distributed torch
- [ ] optimize code performance regarding python dynamic types (torch script, cython)
- [ ] parallelize descriptor calculations using vectorization or thread pool 
-->

## Utils
### How big is the code?
```bash
$ pygount --format=summary ./
```

### Class dependency graph
```bash
$ pydeps  torchip/potentials/nnp/nnp.py -o pydepsgraph.svg
```

## Optimizations
### TorchScript JIT 
- https://pytorch.org/docs/stable/jit.html 
- torch.jit.Future

Tried torch script, future, pool, and mp (examples are available) but still cpu process cannot exceed 100% probably due to `GIL`.
Torch script works only for functions and `nn.Module`. A generic python class that contains tensor cannot be convert to torch script (e.g. the `self` is not know within the methods with the `torch.jit.script` decorator). 

### C++ Extension 
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- Example files https://github.com/pytorch/extension-cpp
- Multi-threading https://jedyang.com/post/multithreading-in-python-pytorch-using-c++-extension/

A C++ kernel is faster (40%) specially when reducing number of kernel calls.
GIL issue is still there when trying a parallel version either from python call or even within the C++ code using `#include <thread>`.

C++ extension can be used to define routines (and not classes) in C++. It limit us to load C++ kernels only in form of function from python. Still don't know how to use complete C++ class containing torch.tensors in python, so the current approach is routine-based and adjusting python classes to work with those C++ functions. 

### C++ Frontend API
- https://pytorch.org/tutorials/advanced/cpp_frontend.html

### GPU
- torch.cuda.stream

More tests require regarding the GPU-computing. 

### Dask + Torch
Dask client could be an option for parallel pytorch if the GIL would be still a limitation.
It's just similar to numpy and can be parallelized using Dask. 


### Algorithm
Defining groups of symmetry functions and caching the repeated terms seems to be a more efficient approach (not tested yet).


### Documentation
Using `Sphinx` as it is suitable for both Python and C++.

Following [this](https://betterprogramming.pub/auto-documenting-a-python-project-using-sphinx-8878f9ddc6e9) tutorial for python project.

How to install
```bash
$ pip install -U Sphinx
$ pip install sphinx-rtd-theme
```

Go to `docs/` and then 
``` bash
$ sphinx-quickstart
```

Create `.rst` and `Makefile`
```bash
$ sphinx-apidoc -f -o ./ ../torchip
```
_Note_: The `-F` flag must be added for the first time in order to create default `index.rst` and `conf.py` files.

Make html
```bash
$ make html
```


## References
### Tutorials
- https://www.pytorchlightning.ai/

### Similar codes
- https://github.com/aiqm/torchani
- https://github.com/torchmd/torchmd
- https://github.com/torchmd/mdgrad
- https://github.com/deepmodeling/deepmd-kit
- https://github.com/SINGROUP/dscribe
- https://github.com/Teoroo-CMC/PiNN

### YouTube
- https://youtu.be/1xfwB6cNeMA

### Logo
- https://fsymbols.com/text-art/

