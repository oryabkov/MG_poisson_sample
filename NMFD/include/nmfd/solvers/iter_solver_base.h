// Copyright © 2020-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of NMFD.

// NMFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// NMFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with NMFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __NMFD_ITER_SOLVER_BASE_H__
#define __NMFD_ITER_SOLVER_BASE_H__

#include <memory>
#include <scfd/utils/logged_obj_base.h>
#include "detail/default_prec_creator.h"

namespace nmfd
{
namespace solvers 
{


template
<
    class VectorOperations,class Monitor,class Log,
    class LinearOperator,class Preconditioner
>
class iter_solver_base : public scfd::utils::logged_obj_base<Log>
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using linear_operator_type = LinearOperator;
    using preconditioner_type = Preconditioner;
    using vector_operations_type = VectorOperations;
    using monitor_type = Monitor;
    using log_type = Log;

protected:
    using logged_obj_t = scfd::utils::logged_obj_base<Log>;
    using logged_obj_params_t = typename logged_obj_t::params;

    mutable monitor_type           monitor_;
    std::shared_ptr<vector_operations_type> vec_ops_;
    std::shared_ptr<preconditioner_type> prec_;
    std::shared_ptr<const linear_operator_type> A_;
public:
    iter_solver_base(std::shared_ptr<vector_operations_type> vec_ops, 
                     Log *log, const logged_obj_params_t &logged_obj_params, 
                     const typename monitor_type::params& monitor_params,
                     std::shared_ptr<preconditioner_type> prec = nullptr):
        logged_obj_t(log, logged_obj_params), 
        monitor_(*vec_ops, log, monitor_params),
        vec_ops_(std::move(vec_ops)), prec_(std::move(prec)), A_(nullptr)
    {
        if (prec_ == nullptr) 
        {
            prec_ = detail::default_prec_creator<VectorOperations,LinearOperator,Preconditioner>::get(vec_ops_);
        }
        //monitor_.set_log_msg_prefix(logged_obj_params.log_msg_prefix + monitor_.get_log_msg_prefix());
    }

    Monitor         &monitor() { return monitor_; }
    const Monitor   &monitor()const { return monitor_; }
//TODO 
#if 0
    void set_preconditioner(std::shared_ptr<preconditioner_type> prec) 
    { 
        prec_ = prec; 
    }
#endif
    virtual bool solve(const linear_operator_type &A, const vector_type &b, 
                       vector_type &x)const = 0;
    
    virtual void set_operator(std::shared_ptr<const linear_operator_type> A)
    {
        A_ = std::move(A);
        if (prec_ != nullptr) 
        {
            prec_->set_operator(A_);
        }
    }
    
    virtual bool solve(const vector_type &b, vector_type &x)const = 0;

    virtual ~iter_solver_base()
    {
    }
};

}
}

#endif
