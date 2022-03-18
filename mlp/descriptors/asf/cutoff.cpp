#include <torch/extension.h>


// TODO: add other type functions
const double _TANH_PRE = 2.2637537952253504D; // math.pow((math.e + 1 / math.e) / (math.e - 1 / math.e), 3)

torch::Tensor _tanhu(torch::Tensor r, double inv_r_cutoff) {
  return torch::pow(torch::tanh(1.0D - r * inv_r_cutoff), 3);
}

torch::Tensor _tanh(torch::Tensor r, double inv_r_cutoff) {
  return _TANH_PRE * torch::pow(torch::tanh(1.0D - r * inv_r_cutoff), 3);
}

// torch::Tensor _call(torch::Tensor r, double r_cutoff, double inv_r_cutoff) {
//   return torch::where( r < r_cutoff, _tanhu(r, inv_r_cutoff), torch::zeros_like(r));
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_tanhu", &_tanhu, "CPP _tanhu kernel");
  m.def("_tanh", &_tanh, "CPP _tanh kernel");
  // m.def("_call", &_call, "CPP _call kernel");
}
