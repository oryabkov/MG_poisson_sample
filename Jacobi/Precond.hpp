#ifndef __PRECONDITIONER_HPP__
#define __PRECONDITIONER_HPP__

#include "PrecondKernel.hpp"

namespace preconditioners
{

template <class VectorSpace>
class jacobi_preconditioner
{
public:
    static const int dim     = VectorSpace::dim;
    using vector_space_type  = VectorSpace; // defines Vector Space working in
    using vector_space_ptr   = std::shared_ptr<VectorSpace>;

    using scalar_type        = typename VectorSpace::scalar_type;
    using ordinal_type       = typename VectorSpace::ordinal_type;

    using vector_type        = typename VectorSpace::vector_type;
    using idx_nd_type        = typename VectorSpace::idx_nd_type;

    using for_each_nd_type   = typename VectorSpace::for_each_nd_type;

    using grid_step_type     = scfd::static_vec::vec<scalar_type, dim>;
    using boundary_cond_type = boundary_cond<dim>;

public: // Especially for SYCL
    using preconditioner_kernel = kernels::preconditioner
    <
        idx_nd_type, scalar_type, vector_type, grid_step_type, boundary_cond_type
    >;

private:
    vector_space_ptr     vspace;
    idx_nd_type          range;
    grid_step_type       step;
    boundary_cond_type   b_cond;

public:
    jacobi_preconditioner(vector_space_ptr vec_space, grid_step_type grid_step, boundary_cond_type cond) :
        vspace(vec_space), range(vspace->get_range()), step(grid_step), b_cond(cond) {}

    void apply(vector_type &v) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst(preconditioner_kernel{v, range, step, b_cond}, range);
    };
};

}// namespace operators

#endif
