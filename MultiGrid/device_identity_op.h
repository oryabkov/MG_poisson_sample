#ifndef __DEVICE_IDENTITY_H__
#define __DEVICE_IDENTITY_H__

#include <memory>

#include "kernels/identity.h"

namespace tests
{

template <class VectorSpace, class Log>
class device_identity_op
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
    using identity_kernel = kernels::identity<idx_nd_type, vector_type>;
private:
    idx_nd_type      range;
public:
    device_identity_op(idx_nd_type r = {}): range(r) {}

    idx_nd_type get_size() const noexcept { return range; }

    std::shared_ptr<vector_space_type> get_dom_space() const
    {
        return std::make_shared<vector_space_type>(range);
    }
    std::shared_ptr<vector_space_type> get_im_space() const
    {
        return std::make_shared<vector_space_type>(range);
    }

    void apply(vector_type &from, vector_type &to) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(identity_kernel{from, to, range}, range);
    };
};

}// namespace tests 

#endif
