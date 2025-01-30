#ifndef __VECTOR_SPACE_H__
#define __VECTOR_SPACE_H__

#include <utility>

#include <scfd/utils/device_tag.h>

/************************************************************
 * Backend independent kernels implementing vector operations
 * to run on heterogenous devices.
 ************************************************************/

namespace kernels
{

template <class IdxND, class VectorType>
struct shur_prod
{
    VectorType x, y, z;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        z(idx) = x(idx) * y(idx);
    }
};

template <class IdxND, class Scalar, class VectorType>
struct assign_scalar
{
    Scalar      s;
    VectorType  x;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
       x(idx) = s;
    }
};

template <class IdxND, class Scalar, class VectorType>
struct add_mul_scalar
{
    Scalar s, mul_x;
    VectorType    x;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
       x(idx) = mul_x * x(idx) + s;
    }
};


template <class IdxND, class Scalar, class VectorType>
struct scale
{
    Scalar     s;
    VectorType x;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        x(idx) = s * x(idx);
    }
};

template <class IdxND, class Scalar, class VectorType>
struct assign
{
    VectorType x, y;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        y(idx) = x(idx);
    }
};

template <class IdxND, class Scalar, class VectorType>
struct assign_lin_comb
{
    Scalar    mul_x;
    VectorType x, y;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        y(idx) = mul_x * x(idx);
    }
};

template <class IdxND, class Scalar, class VectorType>
struct add_lin_comb
{
    Scalar    mul_x;
    Scalar    mul_y;
    VectorType x, y;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        y(idx) = mul_x * x(idx) + mul_y * y(idx);
    }
};

}// namespace kernels

#endif
