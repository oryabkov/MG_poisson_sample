#ifndef __IDENTITY_H__
#define __IDENTITY_H__

#include <scfd/utils/device_tag.h>

namespace kernels
{

template <class IdxND, class VectorType>
struct identity
{
    VectorType dom, img;

    __DEVICE_TAG__ void operator()(const IdxND idx) const
    { 
        img(idx) = dom(idx);  
    }
};

}// namespace kernels

#endif
