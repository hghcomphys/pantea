#include <torch/extension.h>


torch::Tensor g3_kernel(torch::Tensor rij,
                        torch::Tensor rik,
                        torch::Tensor rjk,
                        torch::Tensor cost,
                        std::vector<double> params) {
  return params[4] * torch::pow(1.0D + params[2] * cost, params[1]) * torch::exp( -params[0] * (rij*rij + rik*rik + rjk*rjk) ); // TODO: r_shift
  // return res * self.cutoff_function(rij) * self.cutoff_function(rik) * self.cutoff_function(rjk)
}


torch::Tensor g9_kernel(torch::Tensor rij,
                        torch::Tensor rik,
                        torch::Tensor rjk,
                        torch::Tensor cost,
                        std::vector<double> params) {
  return params[4] * torch::pow(1.0D + params[2] * cost, params[1]) * torch::exp( -params[0] * (rij*rij + rik*rik) ); // TODO: r_shift
  // return res * self.cutoff_function(rij) * self.cutoff_function(rik) * self.cutoff_function(rjk)
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("g3_kernel", &g3_kernel, "CPP G3 kernel"); 
  m.def("g9_kernel", &g9_kernel, "CPP G9 kernel"); 
}
