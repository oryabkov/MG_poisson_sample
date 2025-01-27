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

#ifndef __NMFD_PRECONDITIONER_MG_H__
#define __NMFD_PRECONDITIONER_MG_H__

#include <vector>
#include <memory>
#include <scfd/utils/logged_obj_base.h>
#ifdef NMFD_ENABLE_NLOHMANN
#include <nlohmann/json.hpp>
#endif
#include <nmfd/detail/vector_wrap.h>
//#include <glued_matrix_operator.h>
#include "preconditioner_interface.h"

namespace nmfd
{
namespace preconditioners 
{

/// SystemOperator is OperatorWithSpaces.
/// Restrictor, Prolongator are Operator.
/// Smoother, CoarseSolver are Preconditioner
/// Coarsening is Coarsening
/// Smoother, CoarseSolver, Coarsening are HierarchicAlgorithm
/// All vector_space_type, vector_type, scalar_type are the same
template
<
    class SystemOperator,
    class Restrictor,
    class Prolongator,
    class Smoother,
    class CoarseSolver,
    class Coarsening,
    class Log
>
class mg : 
    public preconditioner_interface<typename SystemOperator::vector_space_type,SystemOperator>,
    public scfd::utils::logged_obj_base<Log>
{
public:
    using vector_space_type = typename SystemOperator::vector_space_type;
    using vector_type = typename vector_space_type::vector_type;
    using scalar_type = typename vector_space_type::scalar_type;
    using operator_type = SystemOperator;
    using restrictor_type = Restrictor;
    using prolongator_type = Prolongator;
    using smoother_type = Smoother;
    using coarse_solver_type = CoarseSolver;
    using coarsening_type = Coarsening;

    using T = scalar_type;
    using logged_obj_t = scfd::utils::logged_obj_base<Log>;
    using logged_obj_params_t = typename logged_obj_t::params;
    
    struct params : public logged_obj_params_t
    {
        std::size_t max_levels, cycle_type, num_sweeps_pre, num_sweeps_post;
        bool direct_coarse;
        //TODO not implemented
        std::string out_prefix;
        bool set_direct_coarse_matrix_defect;
        bool regularize_after_direct_coarse;

        params(const std::string &log_prefix = "", const std::string &log_name = "mg::") : 
            logged_obj_params_t(0, log_prefix+log_name),
            max_levels(25), cycle_type(1), num_sweeps_pre(1), num_sweeps_post(1),
            direct_coarse(true), out_prefix("mg_"),
            set_direct_coarse_matrix_defect(false), regularize_after_direct_coarse(false)
        {
        }
        #ifdef NMFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
        }
        nlohmann::json to_json() const
        {
            return nlohmann::json();
        }
        #endif
    };
    using smoother_params_hierarchy_type = typename nmfd::detail::algo_params_hierarchy<smoother_type>::type;
    using coarse_solver_params_hierarchy_type = typename nmfd::detail::algo_params_hierarchy<coarse_solver_type>::type;
    using coarsening_params_hierarchy_type = typename nmfd::detail::algo_params_hierarchy<coarsening_type>::type;
    struct params_hierarchy : public params
    {
        smoother_params_hierarchy_type smoother;
        coarse_solver_params_hierarchy_type coarse_solver;
        coarsening_params_hierarchy_type coarsening;

        //TODO add prefix for other subalgorithms
        params_hierarchy(const std::string &log_prefix = "", const std::string &log_name = "mg::") : 
            params(log_prefix, log_name),
            smoother(this->log_msg_prefix)
        {
        }
        params_hierarchy(
            const params &prm_, 
            const smoother_params_hierarchy_type &smoother_,
            const coarse_solver_params_hierarchy_type &coarse_solver_,
            const coarsening_params_hierarchy_type &coarsening_
        ) : params(prm_), smoother(smoother_), coarse_solver(coarse_solver_), coarsening(coarsening_)
        {
        }
        #ifdef NMFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            params::from_json(j);
            smoother.from_json(j.at("smoother"));
            coarse_solver.from_json(j.at("coarse_solver"));
            coarsening.from_json(j.at("coarsening"));
        }
        nlohmann::json to_json() const
        {
            nlohmann::json  j = params::to_json(),
                            j_smoother = smoother.to_json(),
                            j_coarse_solver = coarse_solver.to_json(),
                            j_coarsening = coarsening.to_json();
            j["smoother"] = j_smoother;
            j["coarse_solver"] = j_coarse_solver;
            j["coarsening"] = j_coarsening;
            return j;
        }
        #endif
    };
    using smoother_utils_hierarchy_type = typename nmfd::detail::algo_utils_hierarchy<smoother_type>::type;
    using coarse_solver_utils_hierarchy_type = typename nmfd::detail::algo_utils_hierarchy<coarse_solver_type>::type;
    using coarsening_utils_hierarchy_type = typename nmfd::detail::algo_utils_hierarchy<coarsening_type>::type;
    struct utils
    { 
        Log *log;
        utils(Log *log_ = nullptr) : log(log_)
        {
        }
    };
    struct utils_hierarchy : public utils
    {
        smoother_utils_hierarchy_type smoother;
        coarse_solver_utils_hierarchy_type coarse_solver;
        coarsening_utils_hierarchy_type coarsening;
    };

    mg(const utils_hierarchy &u, const params_hierarchy &p) : 
        logged_obj_t(u.log, p), utils_(u), prm_(p)
    {
    }
    ~mg()
    {
    }

    void set_operator(std::shared_ptr<const operator_type> op)
    {
        build(op);
    }

    void apply(const vector_type &rhs, vector_type &x) const 
    {
        if (levs_.empty())
            throw std::logic_error("mg::apply: levels are empty");

        levs_[0].vec_sp->assign(rhs, *levs_[0].rhs);
        cycle(0);
        levs_[0].vec_sp->assign(*levs_[0].x, x);
    }

    /// inplace version for preconditioner interface
    void apply(vector_type &x) const 
    {
        if (levs_.empty())
            throw std::logic_error("mg::apply: levels are empty");

        levs_[0].vec_sp->assign(x, *levs_[0].rhs);
        cycle(0);
        levs_[0].vec_sp->assign(*levs_[0].x, x);
    }

private:
    //TODO make it true,false and add start_use helper class in apply
    using buf_arr_t = detail::vector_wrap<vector_space_type,true,true>;
    struct level_t
    {
        std::shared_ptr<const operator_type> sys_operator;
        std::shared_ptr<restrictor_type> restrictor;
        std::shared_ptr<prolongator_type> prolongator;

        std::shared_ptr<smoother_type> smoother;
        std::shared_ptr<coarse_solver_type> coarse_solver;

        std::shared_ptr<vector_space_type> vec_sp;
        buf_arr_t x,residual,rhs;

        level_t(std::shared_ptr<const operator_type> op, const utils_hierarchy &utils, const params_hierarchy &prm, bool create_coarse_solver = false) : 
            sys_operator(std::move(op)),
            vec_sp(sys_operator->get_dom_space()),
            x(*vec_sp), residual(*vec_sp), rhs(*vec_sp)
        {
            smoother = algo_hierarchy_creator<smoother_type>::get(utils.smoother,prm.smoother);
            smoother->set_operator(sys_operator);
            if (create_coarse_solver)
            {
                coarse_solver = algo_hierarchy_creator<coarse_solver_type>::get(utils.coarse_solver,prm.coarse_solver);
                coarse_solver->set_operator(sys_operator);
            }
        }
        level_t(const level_t&) = delete;
        level_t &operator=(const level_t&) = delete;
        level_t(level_t&&) = default;
        level_t &operator=(level_t&&) = default;

        std::shared_ptr<operator_type> create_next(coarsening_type &c)
        {
            auto transfer_ops = c.next_level(*sys_operator);
            restrictor = std::get<0>(transfer_ops);
            prolongator = std::get<1>(transfer_ops);
            if (restrictor)
                return c.coarse_operator(*sys_operator, *restrictor, *prolongator);
            else
                return std::shared_ptr<operator_type>();
        }
        /// Calcs residual using x
        void calc_residual()
        {
            sys_operator->apply(*x, *residual);
            vec_sp->add_lin_comb(T(1), *rhs, -T(1), *residual);
        }
        /// Calcs next x using previous x value and precalced residual
        void make_iter()
        {
            smoother->apply(*residual);
            vec_sp->add_lin_comb(T(1), *residual, T(1), *x);
        }
    };

    utils_hierarchy utils_;
    params_hierarchy prm_;
    /// TODO mutable - because of rhs x residual?
    mutable std::vector<level_t> levs_;

    void build(std::shared_ptr<const operator_type> op)
    {
        if (!levs_.empty())
            throw std::logic_error("mg::build: levels are alredy built!");
        
        auto c = algo_hierarchy_creator<coarsening_type>::get(utils_.coarsening,prm_.coarsening);

        int lev_i = 0;
        auto curr_op = op;
        while( !c->coarse_enough(*curr_op) ) 
        {
            levs_.emplace_back( curr_op, utils_, prm_ );

            if (levs_.size() >= prm_.max_levels) return;

            if (prm_.out_prefix != "") 
            {
                //TODO output to file
            }
            curr_op = levs_.back().create_next(*c);
            lev_i++;
            if (!curr_op) return;
        }

        if (prm_.out_prefix != "") 
        {
            //TODO output to file
        }

        if (prm_.direct_coarse) 
        {
            //int direct_coarse_matrix_defect = 0;
            if (prm_.set_direct_coarse_matrix_defect)
            {
                //TODO
            }
            levs_.emplace_back( curr_op, utils_, prm_, true );
            if (prm_.regularize_after_direct_coarse) 
            {
                //TODO
            }
        } 
        else 
        {
            levs_.emplace_back( curr_op, utils_, prm_ );
        }

        logged_obj_t::info_f("build complete: levels number = %d", levs_.size());
    }

    void cycle(size_t levi) const
    {
        auto &curr = levs_[levi];

        curr.vec_sp->assign_scalar(T(0), *curr.x);
        curr.vec_sp->assign(*curr.rhs, *curr.residual);

        if (levi+1 == levs_.size()) 
        {
            if (curr.coarse_solver) 
            {
                curr.coarse_solver->apply(*curr.residual, *curr.x);
                /*if (lvl->level_regularization) {
                    lvl->level_regularization->apply(x);
                }*/
            } 
            else 
            {
                for (size_t i = 0; i < prm_.num_sweeps_pre;  ++i) 
                {
                    curr.make_iter();
                    curr.calc_residual();
                }
                for (size_t i = 0; i < prm_.num_sweeps_post; ++i) 
                {
                    curr.make_iter();
                    ///TODO can remove last call but not very important for the last level
                    curr.calc_residual();
                }
            }
        } 
        else 
        {
            auto &next = levs_[levi+1];

            for (size_t j = 0; j < prm_.cycle_type; ++j) 
            {
                if (j > 0)
                {
                    curr.calc_residual();
                }

                for (size_t i = 0; i < prm_.num_sweeps_pre; ++i)
                {
                    curr.make_iter();
                    curr.calc_residual();
                    /*auto res_norm = curr.vec_sp->norm(*curr.residual);
                    logged_obj_t::info_f("cycle: level number = %d, pre_cycle res_norm = %e", levi, res_norm);*/
                }

                curr.restrictor->apply(*curr.residual, *next.rhs);

                cycle(levi+1);

                /// NOTE *curr.residual is used as tmp buffer here to not create extra buffers
                curr.prolongator->apply(*next.x, *curr.residual);
                curr.vec_sp->add_lin_comb(T(1), *curr.residual, T(1), *curr.x);

                for (size_t i = 0; i < prm_.num_sweeps_post; ++i)
                {
                    curr.calc_residual();
                    /*auto res_norm = curr.vec_sp->norm(*curr.residual);
                    logged_obj_t::info_f("cycle: level number = %d, post_cycle res_norm = %e", levi, res_norm);*/
                    curr.make_iter();
                }
            }
        }
    }
};


}  // preconditioners
}  // nmfd

#endif