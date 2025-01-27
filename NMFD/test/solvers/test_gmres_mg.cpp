#include <memory>
#include <cmath>
#include <scfd/utils/log.h>
#include "cpu_vector_space.h"
#include "prolongator.h"
#include "restrictor.h"
#include "ident_op.h"
#include "coarsening.h"
#include "linear_operator_elliptic.h"
#include "smoother_elliptic.h"
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/gmres.h>
#include "residual_regularization_test.h"

#define M_PIl 3.141592653589793238462643383279502884L



int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    using T = double;
    using T_vec = double*;
    using T_mvec = double*;
    using vec_ops_t = nmfd::cpu_vector_space<T, T_vec, T_mvec, log_t>;
    using prolongator_t = tests::prolongator<vec_ops_t, log_t>;
    using restrictor_t = tests::restrictor<vec_ops_t, log_t>;
    using ident_op_t = tests::ident_op<vec_ops_t, log_t>;
    using lin_op_elliptic_t = tests::linear_operator_elliptic<vec_ops_t, log_t>;
    using lin_op_t = lin_op_elliptic_t;
    using coarsening_t = tests::coarsening<lin_op_t, log_t>;
    using smoother_elliptic_t = tests::smoother_elliptic<vec_ops_t, log_t>;
    using smoother_t = smoother_elliptic_t;
    using mg_t = 
        nmfd::preconditioners::mg
        <
            lin_op_t, restrictor_t, prolongator_t, smoother_t, ident_op_t, coarsening_t, log_t
        >;
    using mg_params_t = mg_t::params_hierarchy;
    using mg_utils_t = mg_t::utils_hierarchy;
    using monitor_t = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
    using residual_reg_t = nmfd::solvers::detail::residual_regularization_test<vec_ops_t, log_t>;
    using gmres_elliptic_w_reg_t = nmfd::solvers::gmres< vec_ops_t, monitor_t, log_t, lin_op_elliptic_t, mg_t, residual_reg_t>;



    int error = 0;
    log_t log;
    log.info("test gmres with mg preconditioner");
    std::size_t N_with_preconds = 512;
    //std::size_t N_with_no_preconds = 50;
    std::shared_ptr<vec_ops_t> vec_ops;

    auto get_residual = [&log, &vec_ops](auto& A, auto& x, auto &y) 
    {
        T_vec resid;
        vec_ops->init_vector(resid);
        vec_ops->start_use_vector(resid);
        A.apply(x,resid);
        vec_ops->add_lin_comb(1,y,-1,resid);
        log.info_f("||Lx-y|| = %e", vec_ops->norm(resid) );
        vec_ops->stop_use_vector(resid);
        vec_ops->free_vector(resid);
    };

    //testing elliptic operator with constant kernel
    {
        std::size_t N = N_with_preconds;
        vec_ops = std::make_shared<vec_ops_t>(N);
        mg_utils_t mg_utils;
        mg_utils.log = &log;
        mg_params_t mg_params;
        mg_params.direct_coarse = false;
        mg_params.num_sweeps_pre = 3;
        mg_params.num_sweeps_post = 3;
        auto residual_reg = std::make_shared<residual_reg_t>(vec_ops);//, &log); //use log to see the action of the residual regularization

        T_vec x,y,resid;
        vec_ops->init_vector(x);
        vec_ops->init_vector(y);        
        vec_ops->start_use_vector(x);
        vec_ops->start_use_vector(y);

        log.info_f("=>elliptic with size %i", vec_ops->size() ); 
        auto lin_op_elliptic = std::make_shared<lin_op_elliptic_t>(*vec_ops); //with time step 5


        for(int j=0;j<N;j++)
        {
            //y[j] = std::sin(2.0*j/(N-1)*M_PIl);
            y[j] = std::sin(2.0*j/(N)*M_PIl);
        }

        gmres_elliptic_w_reg_t::params params_elliptic;
        params_elliptic.monitor.rel_tol = 1.0e-10;
        //params_elliptic.monitor.rel_tol = 1.0e-6;
        params_elliptic.monitor.max_iters_num = 300;
        params_elliptic.basis_size = 25;
        params_elliptic.reorthogonalization = true;
        {
            auto mg = std::make_shared<mg_t>(mg_utils, mg_params);
            log.info("left preconditioner");
            params_elliptic.preconditioner_side = 'L';
            gmres_elliptic_w_reg_t gmres(lin_op_elliptic, vec_ops, &log, params_elliptic, mg, residual_reg);

            vec_ops->assign_scalar(0.0, x);
            bool res = gmres.solve(y, x);
            error += (!res);

            log.info_f("pLgmres res: %s", res?"true":"false");
            log.info(" reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.99999, x);
            res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pLgmres res with x0: %s", res?"true":"false");
            log.info_f("solution final norm = %e", vec_ops->norm(x) );
            get_residual(*lin_op_elliptic, x, y);
        }
        {
            auto mg = std::make_shared<mg_t>(mg_utils, mg_params);
            log.info("right preconditioner");
            params_elliptic.preconditioner_side = 'R';
            gmres_elliptic_w_reg_t gmres(lin_op_elliptic, vec_ops, &log, params_elliptic, mg, residual_reg);

            vec_ops->assign_scalar(0.0, x);
            bool res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pRgmres res: %s", res?"true":"false");
            log.info(" reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.99999, x);
            res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pRgmres res with x0: %s", res?"true":"false");
            log.info_f("solution final norm = %e", vec_ops->norm(x) );
            get_residual(*lin_op_elliptic, x, y);
        }
        
        vec_ops->stop_use_vector(x);
        vec_ops->stop_use_vector(y);
        vec_ops->free_vector(x);
        vec_ops->free_vector(y);
    }

 
    if(error > 0)
    {
        log.error_f("Got error = %e.", error ) ;
    }
    else
    {
        log.info("No errors.") ;   
    }

    return error;
}