#include "types.h"

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
    
    using mg_params_t = mg_t::params_hierarchy;
    using mg_utils_t = mg_t::utils_hierarchy;

    
    auto range = idx_nd_type   ::make_ones() * 512;
    auto step  = grid_step_type::make_ones() / 512.;
    auto cond  = boundary_cond<dim>{{-1, -1, -1}, {-1, -1, -1}}; // TODO deduce?

    auto vspace  = std::make_shared<vec_ops_t>(range);
    auto l_op    = std::make_shared<lin_op_t>(range, step, cond);
    mg_utils_t    mg_utils;
    mg_params_t   mg_params;
     
    mg_utils.log               = &log; 
    mg_params.direct_coarse    = false;
    mg_params.num_sweeps_pre   = 3;
    mg_params.num_sweeps_post  = 3;

    auto mg = std::make_shared<mg_t>(mg_utils, mg_params); 
    mg->set_operator(l_op);
    
    vector_t x(range), tmp(range), rhs(range), res(range);
     
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

    jacobi_mg_solver solver(vspace, l_op, mg);
    
    for (std::size_t i=0; i < 40; ++i)
    {
        std::cout << "residual_i = "
                  << solver.make_step(rhs, tmp, x) << std::endl;
    }

    vspace->add_lin_comb(-1, res, 1, x);
    std::cout << "norm2(x-u) = " << vspace->norm2(x) << std::endl;

    return error;
}
