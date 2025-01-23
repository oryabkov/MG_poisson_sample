#ifndef __LAPLACE_OPERATOR_H__
#define __LAPLACE_OPERATOR_H__

#include <type_traits>

#include "LaplaceKernel.hpp"

namespace operators
{

template <class VectorSpace, class GridStep, 
class = typename std::enable_if_t<std::is_same_v
<
    typename VectorSpace::value_type, 
    typename GridStep::value_type>
>>
class laplace_operator
{
public:
    using vector_space_t    = VectorSpace; // defines Vector Space working in 
    using vector_space_ptr  = std::shared_ptr<VectorSpace>;
    
    using scalar_t          = typename VectorSpace::scalar_t;
    using ordinal_t         = typename VectorSpace::ordinal_t;
    
    using vector_t          = typename VectorSpace::vector_t;
    using idx_nd_t          = typename VectorSpace::idx_nd_t;

    using for_each_nd_t     = typename VectorSpace::for_each_nd_t;
    using grid_step_t       = GridStep; // defines dx_i 

public: // Especially for SYCL
    using laplace_op_kernel = kernels::laplace_op
    <
        idx_nd_t, scalar_t, vector_t, grid_step_t
    >;

private:
    vector_space_ptr  vspace;
    idx_nd_t          range;
    grid_step_t       step;
    
    for_each_nd_t     for_each_nd_inst;    

public:
    laplace_operator(vector_space_ptr vec_space, grid_step_t grid_step) : 
        vspace(vec_space), range(vspace->get_range()), step(grid_step) {}  

public:
    void apply(const vector_t &in, vector_t &out) const
    {
        for_each_nd_inst(laplace_op_kernel{in, out, range, step}, range);
    };
};

}// namespace operators

#endif
