#ifndef CAFFE2_UTILS_MATH_ELEMENTWISE_H_
#define CAFFE2_UTILS_MATH_ELEMENTWISE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/types.h"

namespace caffe2 {
namespace math {

template <typename T, class Context>
void Exp(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Log(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Log1p(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sin(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Asin(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Cos(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Acos(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Tan(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Atan(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sinh(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Cosh(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void SinCos(int N, const T* X, T* S, T* C, Context* context);
template <typename T, class Context>
void Tanh(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Abs(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sqr(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sqrt(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Rsqrt(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Cube(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Cbrt(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Neg(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Sign(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Not(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Powx(int N, const T* A, const T b, T* Y, Context* context);
template <typename T, class Context>
void Inv(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void Erf(int N, const T* X, T* Y, Context* context);
template <typename T, class Context>
void CdfNorm(int N, const T* X, T* Y, Context* context);

template <typename T, class Context>
void Set(std::int64_t N, T alpha, T* X, Context* context);

template <typename TAlpha, typename TData, class Context>
void
Scale(std::int64_t N, TAlpha alpha, const TData* X, TData* Y, Context* context);

// Different from the Scale function above, if alpha is passed in as a pointer,
// we will assume that it lives on the Context device, for example on GPU.
template <typename TAlpha, typename TData, class Context>
void Scale(
    std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    TData* Y,
    Context* context);

template <typename T, class Context>
void Add(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
void Sub(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
void Mul(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
void Div(int N, const T* A, const T* B, T* C, Context* context);

template <typename T, class Context>
void Min(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
void Max(int N, const T* A, const T* B, T* C, Context* context);

template <typename T, class Context>
void And(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
void Or(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
void Xor(int N, const T* A, const T* B, T* C, Context* context);

template <typename T, class Context>
void
BitwiseAnd(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
void
BitwiseOr(int N, const T* A, const T* B, T* C, Context* context);
template <typename T, class Context>
void
BitwiseXor(int N, const T* A, const T* B, T* C, Context* context);

template <typename T, class Context>
void EQ(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
void NE(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
void LT(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
void LE(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
void GT(int N, const T* A, const T* B, bool* C, Context* context);
template <typename T, class Context>
void GE(int N, const T* A, const T* B, bool* C, Context* context);

template <typename TAlpha, typename TData, class Context>
void
Axpy(std::int64_t N, TAlpha alpha, const TData* X, TData* Y, Context* context);

// Different from the Axpy function above, if alpha is passed in
// as a pointer, we will assume that it lives on the Context device,
// for example on GPU.
template <typename TAlpha, typename TData, class Context>
void Axpy(
    std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    TData* Y,
    Context* context);

template <typename TAlpha, typename TData, class Context>
void Axpby(
    std::int64_t N,
    TAlpha alpha,
    const TData* X,
    TAlpha beta,
    TData* Y,
    Context* context);

template <typename TAlpha, typename TData, class Context>
void Axpby(
    std::int64_t N,
    const TAlpha* alpha,
    const TData* X,
    const TAlpha* beta,
    TData* Y,
    Context* context);

} // namespace math
} // namespace caffe2

#endif // CAFFE2_UTILS_MATH_ELEMENTWISE_H_
