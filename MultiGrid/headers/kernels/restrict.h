#ifndef __RESTRICT_H__
#define __RESTRICT_H__

#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/rect.h>

namespace kernels
{

template <class IdxND, class Ord, class VectorType>
struct restrict 
{
    VectorType dom, img;
    
    __DEVICE_TAG__ void operator()(const IdxND idx) const // traversing image space
    {
        using Rect   = typename scfd::static_vec::rect<Ord, IdxND::dim>;
        using Scalar = typename VectorType::value_type;
        
        const Ord num_cells = Ord{1} << IdxND::dim;
        
        const IdxND begin   = Ord{2} * (idx + IdxND::make_zero());
        const IdxND end     = Ord{2} * (idx + IdxND::make_ones());
        
        Rect r{begin, end};
        Scalar sum{0};
    
        #pragma unroll
        for(IdxND i = r._bypass_start(); r.is_own(i); r._bypass_step(i))
        {
            sum += dom(i);
        }
        
        img(idx) = sum / num_cells;
    }
};

}// namespace kernels

#endif
