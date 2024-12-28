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

#ifndef __NMFD_ALGO_PARAMS_HIERARCHY_H__
#define __NMFD_ALGO_PARAMS_HIERARCHY_H__

#include "params_hierarchy_dummy.h"

namespace nmfd
{
namespace detail 
{

template<class Algo, class = int>
struct algo_params_hierarchy
{
    using type = params_hierarchy_dummy;
};

template<class Algo>
struct algo_params_hierarchy<Algo,decltype((void)(typename Algo::params_hierarchy()),int(0))>
{
    using type = typename Algo::params_hierarchy;
};

} // namespace detail
} // namespace nmfd

#endif
