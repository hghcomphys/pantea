#include <torch/extension.h>


// TODO: add other type functions
const double _TANH_PRE = 2.2637537952253504D; // math.pow((math.e + 1 / math.e) / (math.e - 1 / math.e), 3)
const double PI = 3.14159265359D;


torch::Tensor _hard(torch::Tensor r) {
  return torch::ones_like(r);
}

torch::Tensor _tanhu(torch::Tensor r, double inv_r_cutoff) {
  return torch::pow(torch::tanh(1.0D - r * inv_r_cutoff), 3);
}

torch::Tensor _tanh(torch::Tensor r, double inv_r_cutoff) {
  return _TANH_PRE * torch::pow(torch::tanh(1.0D - r * inv_r_cutoff), 3);
}

torch::Tensor _cos(torch::Tensor r, double inv_r_cutoff) {
  return 0.5D * (torch::cos(PI * r * inv_r_cutoff) + 1.0D);
}

torch::Tensor _exp(torch::Tensor r, double inv_r_cutoff) {
  auto tmp = r * inv_r_cutoff;
  return torch::exp(1.0D - 1.0D / (1.0D - tmp*tmp));
}

torch::Tensor _poly1(torch::Tensor r) {
  return (2.0D*r - 3.0D) * r*r + 1.0D;
}

torch::Tensor _poly2(torch::Tensor r) {
  return ((15.0D - 6.0D*r) * r - 10.0D) * r*r*r + 1.0D;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_hard" , &_hard , "CPP hard kernel"       );
  m.def("_tanhu", &_tanhu, "CPP tanhu kernel"      );
  m.def("_tanh" , &_tanh , "CPP tanh kernel"       );
  m.def("_cos"  , &_cos  , "CPP cosine kernel"     );
  m.def("_exp"  , &_exp  , "CPP exponential kernel");
  m.def("_poly1", &_poly1, "CPP polynomial kernel" );
  m.def("_poly2", &_poly2, "CPP polynomial kernel" );
}
