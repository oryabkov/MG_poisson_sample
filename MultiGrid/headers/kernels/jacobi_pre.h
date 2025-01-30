#ifndef __JACOBI_PRE_H__
#define __JACOBI_PRE_H__

namespace kernels
{

template <class IdxND, class Scalar, class VectorType, class GridStep, class BoundaryCond>
struct jacobi_preconditioner
{

    VectorType       v;
    IdxND        range;
    GridStep      step;
    BoundaryCond  cond;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        Scalar diagonal{0};

        #pragma unroll
        for (int j = 0u; j < IdxND::dim; ++j)
        {
            const auto N     = range[j];
            const auto hj    = step[j];

            Scalar diag_j{2};

            if(idx[j] ==     0u)
                diag_j -= cond. left[j];

            if(idx[j] == N - 1u)
                diag_j -= cond.right[j];

            diagonal   += diag_j / (hj * hj);
        }

        v(idx) /= diagonal; // divide by diagonal element
    }
};

}// namespace kernels

#endif
