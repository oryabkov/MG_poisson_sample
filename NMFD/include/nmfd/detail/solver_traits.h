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

#ifndef __NMFD_SOLVER_TRAITS_H__
#define __NMFD_SOLVER_TRAITS_H__

#include <type_traits>

namespace nmfd
{
namespace lin_solvers 
{
namespace detail
{

template <typename Solver, typename = int>
struct has_hierarchy : std::false_type { };

template <typename Solver>
struct has_hierarchy<Solver, decltype((void)(typename Solver::params_hierarchy()),(void)(Solver::has_hierarchy),int(0))> : 
    std::integral_constant<bool,Solver::has_hierarchy> { };

} // namespace detail
} // namespace lin_solvers
} // namespace nmfd

#endif
