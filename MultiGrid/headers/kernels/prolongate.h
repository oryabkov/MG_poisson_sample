#ifndef __PROLONGATE_H__
#define __PROLONGATE_H__

#include <scfd/utils/device_tag.h>

namespace kernels
{

template <class IdxND, class Ord, class VectorType>
struct prolongate
{
    VectorType dom, img;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    {
        const auto & curr = idx;
        const auto   half = idx / Ord{2};

        img(curr) = dom(half);
    }
};

}// namespace kernels

#endif
