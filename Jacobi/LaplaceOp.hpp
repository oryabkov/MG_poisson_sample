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
    using vector_space_type = VectorSpace; // defines Vector Space working in 
    using vector_space_ptr  = std::shared_ptr<VectorSpace>;
    
    using scalar_type       = typename VectorSpace::scalar_type;
    using ordinal_type      = typename VectorSpace::ordinal_type;
    
    using vector_type       = typename VectorSpace::vector_type;
    using idx_nd_type       = typename VectorSpace::idx_nd_type;

    using for_each_nd_type  = typename VectorSpace::for_each_nd_type;
    using grid_step_type    = GridStep; // defines dx_i 

public: // Especially for SYCL
    using laplace_op_kernel = kernels::laplace_op
    <
        idx_nd_type, scalar_type, vector_type, grid_step_type
    >;

private:
    vector_space_ptr  vspace;
    idx_nd_type          range;
    grid_step_type       step;
    
    for_each_nd_type     for_each_nd_inst;    

public:
    laplace_operator(vector_space_ptr vec_space, grid_step_type grid_step) : 
        vspace(vec_space), range(vspace->get_range()), step(grid_step) {}  

public:
    void apply(const vector_type &in, vector_type &out) const
    {
        for_each_nd_inst(laplace_op_kernel{in, out, range, step}, range);
    };
};

}// namespace operators

#endif
