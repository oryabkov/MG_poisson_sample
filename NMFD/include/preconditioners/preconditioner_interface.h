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

#ifndef __NMFD_PRECONDITIONER_INTERFACE_H__
#define __NMFD_PRECONDITIONER_INTERFACE_H__

#include <memory>
#include <nmfd/detail/algo_utils_hierarchy.h>
#include <nmfd/detail/algo_params_hierarchy.h>
#include <nmfd/detail/algo_hierarchy_creator.h>

namespace nmfd
{
namespace preconditioners 
{

using nmfd::detail::algo_params_hierarchy;
using nmfd::detail::algo_utils_hierarchy;
using nmfd::detail::algo_hierarchy_creator;

template <class VectorSpace, class LinearOperator>
class preconditioner_interface
{
public:
    using vector_space_type = VectorSpace;
    using vector_type = typename VectorSpace::vector_type;
    using operator_type = LinearOperator;
    
    virtual ~preconditioner_interface() 
    {
    }
    virtual void set_operator(std::shared_ptr<const operator_type> op) = 0;
    virtual void apply(const vector_type &rhs, vector_type &x) const = 0;
    /// inplace version for preconditioner interface
    virtual void apply(vector_type &x) const = 0;
};


}  // preconditioners
}  // nmfd

#endif