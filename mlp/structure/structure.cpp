#include <torch/extension.h>

using namespace torch::indexing;


torch::Tensor _apply_pbc(torch::Tensor dx, torch::Tensor l) {
  // TODO: in-place dx?
  auto dx_ = torch::where(dx  >  0.5E0*l, dx  - l, dx );
       dx_ = torch::where(dx_ < -0.5E0*l, dx_ + l, dx_);
  return dx_;
}

torch::Tensor apply_pbc(torch::Tensor dx, torch::Tensor box) {
  // TODO: in-place dx?
  for (int dim=0; dim<3; dim++)
    dx.index_put_( {"...", dim}, apply_pbc(dx, box.index({dim})) );
  return dx;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_apply_pbc", &_apply_pbc, "CPP _apply_pbc kernel");
  m.def("apply_pbc", &apply_pbc, "CPP apply_pbc kernel");
}
