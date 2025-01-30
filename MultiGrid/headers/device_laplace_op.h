#ifndef __DEVICE_LAPLACE_OP_H__
#define __DEVICE_LAPLACE_OP_H__

#include <memory>

#include "include/boundary.h" // for boundary conditions
#include "kernels/laplace.h"

namespace tests 
{

template <class VectorSpace, class Log>
class device_laplace_op
{
public:
    static const int dim     = VectorSpace::dim;

    using scalar_type        = typename VectorSpace::scalar_type;
    using vector_type        = typename VectorSpace::vector_type;
    using vector_space_type  = VectorSpace; //defines Vector Space working in
    using grid_step_type     = scfd::static_vec::vec<scalar_type, dim>;
    using boundary_cond_type = boundary_cond<dim>;
    using ordinal_type       = typename VectorSpace::ordinal_type;

    using Ord = ordinal_type;

    using vector_space_ptr   = std::shared_ptr<VectorSpace>;
    using idx_nd_type        = typename VectorSpace::idx_nd_type;
    using for_each_nd_type   = typename VectorSpace::for_each_nd_type;

public: // Especially for SYCL
    using laplace_op_kernel = kernels::laplace_op
    <
        idx_nd_type, scalar_type, vector_type,
        grid_step_type, boundary_cond_type
    >;
private:
    vector_space_ptr     vspace;
    idx_nd_type          range;
    grid_step_type       step;
    boundary_cond_type   b_cond;
public:

    device_laplace_op(idx_nd_type r, grid_step_type grid_step, 
        boundary_cond_type cond) :     
                vspace(std::make_shared<vector_space_type>(r)),
                range(r), step(grid_step), b_cond(cond) {}
    
    device_laplace_op(const vector_space_type& vec_space, 
        grid_step_type grid_step, boundary_cond_type cond) :
                device_laplace_op(vec_space.get_size(), grid_step, cond) {} 
    
    vector_space_ptr        get_space()  const 
    {
        return std::make_shared<vector_space_type>(range);
    }
    
    idx_nd_type             get_size()   const noexcept { return range;  }
    grid_step_type          get_h()      const noexcept { return step;   }
    boundary_cond_type      get_b_cond() const noexcept { return b_cond; }
    
    vector_space_ptr get_dom_space() const { return get_space(); }
    vector_space_ptr get_im_space()  const { return get_space(); }

    void apply(const vector_type &in, vector_type &out) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(laplace_op_kernel{in, out, range, step, b_cond}, range);
    };
};

}// namespace tests

#endif
