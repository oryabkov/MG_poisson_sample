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

#ifndef __NMFD_PRECONDITIONER_DUMMY_H__
#define __NMFD_PRECONDITIONER_DUMMY_H__

#include <vector>
#include <memory>
#ifdef NMFD_ENABLE_NLOHMANN
#include <nlohmann/json.hpp>
#endif
#include <common/detail/vector_wrap.h>
#include <glued_matrix_operator.h>
#include "preconditioner_interface.h"

namespace nmfd
{
namespace preconditioners 
{

template <class VectorOperations, class LinearOperator>
class dummy : 
    public preconditioner_interface<VectorOperations,LinearOperator>
{
public:
    using vector_operations_type = VectorOperations;
    using vector_type = typename VectorOperations::vector_type;
    using operator_type = LinearOperator;
    
    struct params
    {
        params(const std::string &log_prefix = "", const std::string &log_name = "dummy::")
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
    using params_hierarchy = params;
    struct utils
    { 
        std::shared_ptr<vector_operations_type> ops;
        utils(std::shared_ptr<vector_operations_type> ops_ = nullptr) : ops(std::move(ops_))
        {
        }
    };
    using utils_hierarchy = utils;

    dummy(std::shared_ptr<vector_operations_type> ops) : ops_(std::move(ops))
    {
    }
    dummy(const utils_hierarchy &u, const params_hierarchy &) : dummy(u.ops)
    {
    }
    ~dummy()
    {
    }

    void set_operator(std::shared_ptr<const operator_type> op)
    {
    }

    void apply(const vector_type &rhs, vector_type &x) const 
    {
        ops_->assign(rhs, x);
    }

    /// inplace version for preconditioner interface
    void apply(vector_type &x) const 
    {
    }

    bool solve(const vector_type&x, vector_type&y)const
    {
        apply(x,y);
        return true;
    }


private:
    std::shared_ptr<vector_operations_type> ops_;
};


}  // preconditioners
}  // nmfd

#endif