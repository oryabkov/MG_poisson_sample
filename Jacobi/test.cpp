#include <utility>
#include <iostream>
#include <memory>

#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_nd.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>
#include <scfd/arrays/array_nd.h>

#include "VecSpace.hpp"
#include "LaplaceOp.hpp"
#include "Precond.hpp"

#include "Jacobi.hpp"

static constexpr int current_dim = 3;

using scalar           = float;
using grid_step_type   = scfd::static_vec::vec<scalar, current_dim>;
using idx_nd_type      = scfd::static_vec::vec<int   , current_dim>;

struct hip_backend
{
    using memory_type      = scfd::memory::hip_device;
    using for_each_nd_type = scfd::for_each::hip_nd<current_dim>;
    using reduce_type      = scfd::thrust_reduce<>;
};

using vector_space     = operations::vector_space<scalar, current_dim, hip_backend>;
using laplace_operator = operators::laplace_operator<vector_space>;
using preconditioner   = preconditioners::jacobi_preconditioner<vector_space>;

using jacobi_solver    = jacobi<vector_space, laplace_operator, preconditioner>;

using vector_type      = typename vector_space::vector_type;
using vector_view_type = typename vector_type::view_type;

int main()
{

    auto range = 10     * idx_nd_type   ::make_ones();
    auto step  = 0.005f * grid_step_type::make_ones();
    auto cond  = boundary_cond<current_dim>{{-1,1,1}, {1,1,1}}; // TODO deduce?

    vector_type x(range), y(range), rhs(range);

    auto vspace  = std::make_shared<vector_space>(range);
    auto l_op    = std::make_shared<laplace_operator>(vspace, step, cond);
    auto precond = std::make_shared<preconditioner>  (vspace, step, cond);

    jacobi_solver solver{vspace, l_op, precond};

    vspace->assign_scalar(1.f, x);               //x=1
    vspace->scale(2.f, x);                       //x=2
    vspace->assign(x, y);                        //y=x
    vspace->add_mul_scalar(3.f, -1.f, y);        //y=-2*1+3=1
    vspace->add_lin_comb(4.f, x, -3.f, y);       //y=2*4-3*1=5
    vspace->assign_lin_comb(2.f, y, x);          //x=5*2=10

    {
        vector_view_type x_view(x), y_view(y);

        std::cout << "x = " << x_view(5,5,5) << std::endl;
        std::cout << "y = " << y_view(5,5,5) << std::endl;
    }

    std::cout << "dot(x, y)   = " << vspace->scalar_prod(x,y) << std::endl;
    std::cout << "sum(x)      = " << vspace->sum(x) << std::endl;
    std::cout << "norm(x)     = " << vspace->norm(x) << std::endl;
    std::cout << "norm_sq(x)  = " << vspace->norm_sq(x) << std::endl;
    std::cout << "norm2(x)    = " << vspace->norm2(x) << std::endl;
    std::cout << "norm2_sq(x) = " << vspace->norm2_sq(x) << std::endl;

    {
        vector_view_type rhs_view(rhs, false);

        for(int i=0; i<range[0]; ++i)
        {
            const scalar hx    = step[0];
            const scalar rhs_f = (i==0) ? 2.f / (hx*hx) : 0.f;

            for(int j=0; j<range[1]; ++j)
            for(int k=0; k<range[2]; ++k)
                rhs_view(i,j,k) =  rhs_f;
        }

        rhs_view.release(); // can't we do it in destructor?
    }

    {
        vector_view_type rhs_view(rhs);

        std::cout << "rhs(0,0,0) = " << rhs_view(0,0,0) << std::endl;
        std::cout << "rhs(1,1,1) = " << rhs_view(1,1,1) << std::endl;
    }

    auto & tmp = y; //y is now temporary storage

    for (std::size_t i=0; i < 10000; ++i)
    {
        std::cout << "residual_i = "
                  << solver.make_step(rhs, tmp, x) << std::endl;
    }

    std::cout << "norm2(x)   = " << vspace->norm2(x) << std::endl;

    return 0;
}
