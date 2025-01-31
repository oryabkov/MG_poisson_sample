#include <memory>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <type_traits>

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
    if (argc != 4)
    {
        std::cout << "USAGE: " << args[0] << " precond_type " << " n " << "arch_name" << std::endl; 
        std::cout << "    where precond_type is diag or mg" << std::endl; 
        return 1;
    }
    std::string solv = "gmres";
    std::string prec = args[1];
    std::string size = args[2];
    std::string arch = args[3];
    std::string type = std::is_same_v<float, scalar> ? "f" : "d";

    log_t log;

    int N        = std::stoi(size);
    int num_iter = 100; 
    

    auto range = idx_nd_type   ::make_ones() *        N;
    auto step  = grid_step_type::make_ones() / scalar(N);
    auto cond  = boundary_cond<dim>
    {   
        {-1, -1, -1}, // left 
        {-1, -1, -1}  // right
    }; // -1 = dirichlet, +1 = neuman

    vector_t x(range), y(range), rhs(range), res(range);

    auto vspace  = std::make_shared<vec_ops_t>(range);
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
    
    
    auto l_op    = std::make_shared<lin_op_t> (range, step, cond);

    std::shared_ptr<precond_interface> precond;
    if      (prec == "diag")
    {
        precond = std::make_shared<smoother_t>(l_op);
    }
    else if (prec == "mg")
    {
        mg_utils_t    mg_utils;
        mg_params_t   mg_params;
         
        mg_utils.log               = &log; 
        mg_params.direct_coarse    = false;
        mg_params.num_sweeps_pre   = 3;
        mg_params.num_sweeps_post  = 3;

        precond = std::make_shared<mg_t>(mg_utils, mg_params); 
    }
    else
    {
        std::cout << "precond_type is either diag or mg!" << std::endl;
        return 1;
    }

    gmres_solver::params params_gmres;
    params_gmres.monitor.rel_tol = std::is_same_v<float, scalar> ? 1e-7f : 1e-10;
    //params_gmres.monitor.rel_tol = 1.0e-6;
    params_gmres.monitor.max_iters_num = num_iter;
    params_gmres.monitor.save_convergence_history = true;
    params_gmres.do_restart_on_false_ritz_convergence = true;
    params_gmres.basis_size = 25;
    params_gmres.preconditioner_side = 'L';
    params_gmres.reorthogonalization = true;
    gmres_solver gmres(l_op, vspace, &log, params_gmres, precond);
    
     
    std::chrono::duration<double, std::milli> elapsed_seconds; // aka T_solve
    {
        auto start = std::chrono::steady_clock::now();
        
        bool conv_res = gmres.solve(rhs, x);
    
        auto end = std::chrono::steady_clock::now();
        elapsed_seconds = (end - start);
    }
    auto& exec_t = elapsed_seconds;
    
    std::string
    conv_file_name("data/conv_history_");
    conv_file_name += solv; conv_file_name += "_";
    conv_file_name += prec; conv_file_name += "_";
    conv_file_name += arch; conv_file_name += "_";
    conv_file_name += size; conv_file_name += "_";
    conv_file_name += type; conv_file_name += ".dat";

    std::string exec_time_file_name("data/times.dat");

    std::ofstream conv_history(conv_file_name,      std::ios::out | std::ios::trunc);
    std::ofstream exec_times  (exec_time_file_name, std::ios::out | std::ios::app);

    auto res_by_it = gmres.monitor().convergence_history();
    //conv_history << 0 << " " << 1.00 << std::endl; 
    std::for_each(begin(res_by_it), end(res_by_it), 
    [&](std::pair<int, scalar> &pair)
    { 
        conv_history << pair.first << " " << pair.second << std::endl; 
    });
    
    auto [i_0,  init_res] = res_by_it.front();
    auto [i_n, final_res] = res_by_it.back();

    auto conv_rate = std::pow(final_res / init_res, scalar(1) / (i_n - i_0));

    exec_times <<
        solv << "," <<
        prec << "," <<
        arch << "," <<
        type << "," <<
        N    << "," <<
        exec_t.count() << "," <<
        num_iter       << "," <<
        conv_rate
    << std::endl;    

    return 0;
}
