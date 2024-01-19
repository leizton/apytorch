#pragma once

#include <ATen/native/SparseTensorUtils.h>

namespace at::native {

sparse::SparseTensor& mul_out_sparse_scalar(sparse::SparseTensor& r, const sparse::SparseTensor& t, const Scalar& value);
sparse::SparseTensor& mul_out_sparse_zerodim(sparse::SparseTensor& r, const sparse::SparseTensor& t, const Tensor& value);
sparse::SparseTensor& _mul_dense_sparse_out(const Tensor& d, const Tensor& s, Tensor& res);
sparse::SparseTensor& _mul_sparse_sparse_zero_dim_out(const Tensor& zero_dim, const Tensor& other, Tensor& res);
sparse::SparseTensor& _mul_sparse_sparse_out(const Tensor& x, const Tensor& y, Tensor& res);

} // namespace at::native
