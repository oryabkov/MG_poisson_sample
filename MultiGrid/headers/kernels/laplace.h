#ifndef __LAPLACE_H__
#define __LAPLACE_H__

namespace kernels
{

template <class IdxND, class Scalar, class VectorType, class GridStep, class BoundaryCond>
struct laplace_op
{

    VectorType in, out;
    IdxND        range;
    GridStep      step;
    BoundaryCond  cond;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        Scalar laplace{0};

        #pragma unroll
        for (int j = 0u; j < IdxND::dim; ++j)
        {
            auto N     = range[j];

            auto ej    = IdxND::make_unit(j);
            auto hj    = step[j];

            auto curr  = in(idx);

            auto prev  = idx[j] ==     0u ? cond. left[j] * in(idx) : in(idx - ej);
            auto next  = idx[j] == N - 1u ? cond.right[j] * in(idx) : in(idx + ej);

            laplace   += (2 * curr - next - prev) / (hj * hj);
        }

        out(idx) = laplace; // assign -delta u;
    }
};

}// namespace kernels

#endif
