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

#ifndef __NMFD_PARAMS_HIERARCHY_DUMMY_H__
#define __NMFD_PARAMS_HIERARCHY_DUMMY_H__

#include <string>

namespace nmfd
{
namespace detail 
{

struct params_hierarchy_dummy
{
    params_hierarchy_dummy(const std::string &log_prefix = "", const std::string &log_name = "")
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

} // namespace detail
} // namespace nmfd

#endif
