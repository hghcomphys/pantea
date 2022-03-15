# Technical notes

<!-- #### TODOs
- [ ] define a customized exception class that handles internal error messages and also python exceptions
- [ ] improve CFG design e.g. config file, defaults values, on-the-fly settings.
- [ ] optimize memory allocation of neighbor list atoms and avoiding redundant tensor creation (use torch.resizes)
- [ ] optimize neighbor list update for large number of atoms (not used for training but MD simulations)
- [ ] utilize torch multi-thread or distributed torch
- [ ] optimize code performance regarding python dynamic types (torch script, cython)
- [ ] parallelize descriptor calculations using vectorization or thread pool -->


### How big it the code?
```bash
$ pygount --format=summary ./
```

## Optimizations

### TorchScript JIT 
- https://pytorch.org/docs/stable/jit.html 
- torch.jit.Future

Tried torch script, future, pool, and mp (examples are available) but still cpu process cannot exceed 100% probably due to `GIL`.

### C++ Extension 
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- Example files https://github.com/pytorch/extension-cpp
- Multi-threading https://jedyang.com/post/multithreading-in-python-pytorch-using-c++-extension/

CPP kernel is faster (40%) specially when reducing number of kernel calls.
GIL issue is still there when trying parallel version either from python call or even within the cpp code using `#include <thread>`.

### C++ Frontend API
- https://pytorch.org/tutorials/advanced/cpp_frontend.html

### GPU
- torch.cuda.stream

### Dask
Dask client could be an option for parallel pytorch if the GIL would be still a limitation.





