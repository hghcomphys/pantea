#include <torch/extension.h>


torch::Tensor g2_kernel(torch::Tensor rij,
                        std::vector<double> params) {
  auto tmp = rij - params[1];
  return torch::exp( -params[0] * tmp*tmp ); 
  // self.cutoff_function(rij)
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("g2_kernel", &g2_kernel, "CPP G2 kernel"); 
}
