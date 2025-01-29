#include <memory>
#include <cmath>
#include <scfd/utils/log.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/gmres.h>

#include "device_vector_space.h"
#include "device_restrictor.h"
#include "device_prolongator.h"
#include "device_identity_op.h"
#include "device_laplace_op.h"
#include "device_jacobi_pre.h"
#include "device_coarsening.h"

constexpr int dim =      3;
using scalar      = float;
using grid_step_type   = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type      = scfd::static_vec::vec<int   , dim>;

/***********************/
#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>

struct backend
{
    using memory_type      = scfd::memory::hip_device;
    using for_each_nd_type = scfd::for_each::hip_nd<dim>;
    using reduce_type      = scfd::thrust_reduce<>;
};
/***********************/

using log_t = scfd::utils::log_std;
using vector_space = nmfd::device_vector_space<scalar,/*dim=*/3, backend>;

using laplace_operator = tests::device_laplace_op<vector_space, log_t>;
using preconditioner   = tests::device_jacobi_pre<vector_space, log_t>;

using vector_t         = typename vector_space::vector_type;
using vector_view_t    = typename vector_t::view_type;

#define M_PIl 3.141592653589793238462643383279502884L

auto const f = [](const scalar x, const scalar y, const scalar z) noexcept
{
    return 2 * (1-y) * y * (1-z) * z +
           2 * (1-z) * z * (1-x) * x +
           2 * (1-x) * x * (1-y) * y;
};

auto const u = [](const scalar x, const scalar y, const scalar z) noexcept
{
    return x * (1-x) * y * (1-y) * z * (1-z);
};

int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    
    int error = 0;
    log_t log;
    
    using vec_ops_t     = nmfd::device_vector_space<scalar, dim, backend>;
    
    using prolongator_t = tests::device_prolongator<vec_ops_t, log_t>;
    using restrictor_t  = tests::device_restrictor <vec_ops_t, log_t>;
    using ident_op_t    = tests::device_identity_op<vec_ops_t, log_t>;
    using lin_op_t      = tests::device_laplace_op <vec_ops_t, log_t>;
    using smoother_t    = tests::device_jacobi_pre <vec_ops_t, log_t>;
    using coarsening_t  = tests::device_coarsening<lin_op_t, log_t>;
    std::shared_ptr<vec_ops_t> vec_ops;
    

    using mg_t = nmfd::preconditioners::mg
    <
        lin_op_t, restrictor_t, prolongator_t, 
        smoother_t, ident_op_t, coarsening_t, 
        log_t
    >;
    using mg_params_t = mg_t::params_hierarchy;
    using mg_utils_t = mg_t::utils_hierarchy;

    
    auto range = 100   * idx_nd_type   ::make_ones();
    auto step  = 0.01f * grid_step_type::make_ones();
    auto cond  = boundary_cond<dim>{{-1, -1, -1}, {-1, -1, -1}}; // TODO deduce?

    auto vspace  = std::make_shared<vector_space>(range);
    auto l_op    = std::make_shared<laplace_operator>(range, step, cond);
    auto precond = std::make_shared<preconditioner>(l_op);


    mg_utils_t    mg_utils;
    mg_params_t   mg_params;
     
    mg_utils.log               = &log; 
    mg_params.direct_coarse    = false;
    mg_params.num_sweeps_pre   = 3;
    mg_params.num_sweeps_post  = 3;

    mg_t mg(mg_utils, mg_params); mg.set_operator(l_op);
    
    vector_t x(range), rhs(range), res(range);
     
    {
        vspace->assign_scalar(0.f, x);  //x = 0
        vector_view_t rhs_view(rhs, false), res_view(res, false);

        for(int i=0; i<range[0]; ++i)
        for(int j=0; j<range[0]; ++j)
        for(int k=0; k<range[0]; ++k)
        {
            const auto x = step[0] * (0.5f + i);
            const auto y = step[0] * (0.5f + j);
            const auto z = step[0] * (0.5f + k);

            rhs_view(i,j,k) = f(x,y,z);
            res_view(i,j,k) = u(x,y,z);
        }

        rhs_view.release();
        res_view.release();
    }

    for (int i=0; i<100; ++i)
    {
        mg.apply(rhs, x);
    }

    return error;
}
