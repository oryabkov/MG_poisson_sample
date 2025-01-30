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

int main()
{
    auto range = idx_nd_type   ::make_ones() * 512;
    auto step  = grid_step_type::make_ones() / 512.;
    auto cond  = boundary_cond<dim>{{-1, -1, -1}, {-1, -1, -1}}; // TODO deduce?

    vector_t x(range), y(range), rhs(range), res(range);

    auto vspace  = std::make_shared<vec_ops_t>(range);
    auto l_op    = std::make_shared<lin_op_t>(range, step, cond);
    auto precond = std::make_shared<smoother_t>(l_op);

    jacobi_solver solver{vspace, l_op, precond};

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

    auto & tmp = y;                 //y is now temporary storage

    for (std::size_t i=0; i < 60000; ++i)
    {
        std::cout << "residual_i = "
                  << solver.make_step(rhs, tmp, x) << std::endl;
    }

    vspace->add_lin_comb(-1, res, 1, x);
    std::cout << "norm2(x-u) = " << vspace->norm2(x) << std::endl;


    return 0;
}
