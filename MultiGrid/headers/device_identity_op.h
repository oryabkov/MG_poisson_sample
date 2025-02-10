#ifndef __DEVICE_IDENTITY_H__
#define __DEVICE_IDENTITY_H__

#include <memory>

#include "device_laplace_op.h"
#include "kernels/identity.h"

namespace tests
{

template
<
    class VectorSpace, class Log,
    /**********************************************/
    class Backend=typename VectorSpace::backend_type
>
class device_identity_op
{
    using lin_op_t           = device_laplace_op<VectorSpace, Log>;
public:
    using scalar_type        = typename VectorSpace::scalar_type;
    using vector_type        = typename VectorSpace::vector_type;
    using vector_space_type  = VectorSpace; // defines Vector Space working in
    using ordinal_type       = typename VectorSpace::ordinal_type;

    using vector_space_ptr   = std::shared_ptr<VectorSpace>;
    using idx_nd_type        = typename VectorSpace::idx_nd_type;

    using for_each_nd_type   = typename Backend::for_each_nd_type;

public: // Especially for SYCL
    using identity_kernel = kernels::identity<idx_nd_type, vector_type>;
private:
    idx_nd_type      range;
public:
    struct params {};
    using params_hierarchy = params;
    struct utils {};
    using utils_hierarchy = utils;

    device_identity_op(const utils_hierarchy &u, const params_hierarchy &p) {}
    device_identity_op(idx_nd_type r = {}): range(r) {}

public:
    idx_nd_type get_size() const noexcept { return range; }

    void set_operator(std::shared_ptr<const lin_op_t> op)
    {
        range = op->get_size();
    }

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
        for_each_nd_inst(identity_kernel{from, to}, range);
    };
};

}// namespace tests

#endif
