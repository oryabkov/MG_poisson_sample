#ifndef __DEVICE_RESTRICTOR_H__
#define __DEVICE_RESTRICTOR_H__

#include "kernels/restrict.h"

namespace tests
{

template <class VectorSpace, class Log>
class device_restrictor 
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
    using restrictor_kernel = kernels::restrict
    <
        idx_nd_type, scalar_type, vector_type
    >;
private:
    idx_nd_type      range; // in dom space
public:
    device_restrictor(idx_nd_type r): range(r) // in dom space 
    {
        for (int i=0; i<idx_nd_type::dim; ++i)
        {
            if (r[i] % 2u != 0)
                throw std::logic_error("nmfd::restrictor: encountered odd value in vector_space range! not supported case");
        }
    }

    idx_nd_type get_size() const noexcept { return range; }

    std::shared_ptr<vector_space_type> get_dom_space()const
    {
        return std::make_shared<vector_space_type>(range / Ord{2u});
    }
    std::shared_ptr<vector_space_type> get_im_space()const
    {
        return std::make_shared<vector_space_type>(range);
    }

    // domain -> (restrict) -> image
    void apply(vector_type &from, vector_type &to) const
    {
        auto half_r = range / Ord{2u};
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(restrictor_kernel{from, to, half_r}, half_r);
    };
};

}// namespace tests 

#endif
