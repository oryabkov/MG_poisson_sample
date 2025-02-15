#ifndef __JACOBI_H__
#define __JACOBI_H__

#include <memory>

#include <nmfd/solvers/iter_solver_base.h>
#include <scfd/utils/logged_obj_base.h>

template
<
     class VectorOp,
     class LaplaceOp,
     class Precond,
     class Monitor,
     class Log
>
class jacobi : public nmfd::solvers::iter_solver_base<VectorOp, Monitor, Log, LaplaceOp, Precond>
{
    using parent_t = nmfd::solvers::iter_solver_base<VectorOp, Monitor, Log, LaplaceOp, Precond>;
    using logged_obj_t        = typename parent_t::logged_obj_t;
    using logged_obj_params_t = typename parent_t::logged_obj_params_t;

public:
    using scalar_type = typename VectorOp::scalar_type;
    using vector_type = typename VectorOp::vector_type;

    using vector_operation_type   = VectorOp;
    using laplace_operator_type   = LaplaceOp;
    using preconditioner_type     = Precond;

    using log_type     = Log;
    using monitor_type = Monitor;

    using vector_operation_ptr = std::shared_ptr<VectorOp>;
    using laplace_operator_ptr = std::shared_ptr<LaplaceOp>;
    using preconditioner_ptr   = std::shared_ptr<Precond>;

    struct params : public logged_obj_params_t
    {
        typename monitor_type::params monitor;

        params(const std::string &log_prefix = "", 
               const std::string &log_name   = "jacobi::") :
            logged_obj_params_t(0, log_prefix+log_name),
            monitor( typename Monitor::params(this->log_msg_prefix) ) // TODO
        {
        }
    };

private:
    params                  prms;  

protected:
    using parent_t::monitor_;
    using parent_t::vec_ops_;
    using parent_t::prec_;

public:
    jacobi
    (
        vector_operation_ptr      vec_ops,
        log_type *log           = nullptr, // Not sure this is a good idea
        const params& prm       =      {},
        preconditioner_ptr prec = nullptr

    ) : parent_t{std::move(vec_ops), log, prm, prm.monitor, std::move(prec)},
        prms(prm) {}

    jacobi
    (
        laplace_operator_ptr            A,
        vector_operation_ptr      vec_ops,
        log_type *log           = nullptr, // Not sure this is a good idea
        const params& prm       =      {},
        preconditioner_ptr prec = nullptr
    
    ) : jacobi(vec_ops, log, prm, prec) 
    {
        parent_t::set_operator(A);
    }
    
    bool solve(const laplace_operator_type &A, const vector_type &rhs, 
            vector_type &x) const override
    {
        vector_type     tmp;

        vec_ops_->init_vector(tmp);
        
        A.apply(x, tmp);
        vec_ops_->add_lin_comb(scalar_type{-1}, rhs, scalar_type{1}, tmp);
        // Now, tmp represents residual
        
        monitor_.start(rhs);
        while( !monitor_.check_finished(x, tmp) )
        {
            ++monitor_;
            // tmp := P(Ax - b);
            prec_->apply(tmp);
            // x   := x - P(Ax - b);
            vec_ops_->add_lin_comb(scalar_type{-1}, tmp, scalar_type{1}, x);
            // tmp := Laplace(x) = Ax;
            A.apply(x, tmp);
            // tmp := Ax - b;
            vec_ops_->add_lin_comb(scalar_type{-1}, rhs, scalar_type{1}, tmp);
        } 

        vec_ops_->free_vector(tmp);

        auto res = monitor_.converged();
        if(!res)
            logged_obj_t::error_f("solve: linear solver failed to converge");
        
        return res;
    };
    
    bool solve(const vector_type &b, vector_type &x) const override 
    {
        return solve(*parent_t::A_, b, x);
    }

    
};

#endif
