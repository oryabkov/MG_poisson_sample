#ifndef __DEVICE_PROLONGATOR_H__
#define __DEVICE_PROLONGATOR_H__

#include <memory>

#include "kernels/prolongate.h"

namespace tests 
{

template <class VectorSpace, class Log>
class device_prolongator
{
public:
    using scalar_type        = typename VectorSpace::scalar_type;
    using vector_type        = typename VectorSpace::vector_type;
    using vector_space_type  = VectorSpace; // defines Vector Space working in
    using ordinal_type       = typename VectorSpace::ordinal_type;
    
    using Ord = ordinal_type;

    using vector_space_ptr   = std::shared_ptr<VectorSpace>;
    using idx_nd_type        = typename VectorSpace::idx_nd_type;
    using for_each_nd_type   = typename VectorSpace::for_each_nd_type;

public: // Especially for SYCL
    using prolongator_kernel = kernels::prolongate
    <
        idx_nd_type, ordinal_type, vector_type
    >;
private:
    idx_nd_type      range; // in im space
public:
    device_prolongator(idx_nd_type r) : range(r) // in im space 
    {
        for (int i=0; i<idx_nd_type::dim; ++i)
        {
            if (r[i] % 2u != 0)
                throw std::logic_error("nmfd::prolongator: encountered odd value in vector_space range! not supported case");
        }
    }

    idx_nd_type get_size() const noexcept { return range; }
    
    std::shared_ptr<vector_space_type> get_dom_space() const
    {
        return std::make_shared<vector_space_type>(range / Ord{2u});
    }
    std::shared_ptr<vector_space_type> get_im_space() const
    {
        return std::make_shared<vector_space_type>(range);
    }
    
    // domain -> (prolongate) -> image
    void apply(vector_type &from, vector_type &to) const
    {
       for_each_nd_type for_each_nd_inst;
       for_each_nd_inst(prolongator_kernel{from, to}, range);
    };
};

}// namespace tests

#endif
