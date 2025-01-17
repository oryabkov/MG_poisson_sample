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

#ifndef __NMFD_ALGO_HIERARCHY_CREATOR_H__
#define __NMFD_ALGO_HIERARCHY_CREATOR_H__

#include <memory>
#include "utils_hierarchy_dummy.h"
#include "params_hierarchy_dummy.h"

namespace nmfd
{
namespace detail 
{

template<class Algo, class = int>
struct algo_hierarchy_creator
{
    static std::shared_ptr<Algo> get(
        const utils_hierarchy_dummy &utils,
        const params_hierarchy_dummy &params
    )
    {
        throw std::logic_error("algo_hierarchy_creator::hierarchy constructor is not implemented");
        return std::shared_ptr<Algo>(nullptr);
    }
};

template<class Algo>
struct algo_hierarchy_creator<Algo,decltype((void)(Algo(typename Algo::utils_hierarchy(),typename Algo::params_hierarchy())),int(0))>
{
    static std::shared_ptr<Algo> get(
        const typename Algo::utils_hierarchy &utils,
        const typename Algo::params_hierarchy &params
    )
    {
        return std::make_shared<Algo>(utils,params);
    }
};

} // namespace detail
} // namespace nmfd

#endif
