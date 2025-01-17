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

#ifndef __NMFD_RESIDUAL_REGULATIZATION_DUMMY_H__
#define __NMFD_RESIDUAL_REGULATIZATION_DUMMY_H__

namespace nmfd {
namespace solvers {
namespace detail {

class residual_regularization_dummy
{
public:
    
    residual_regularization_dummy()
    {}
    ~residual_regularization_dummy()
    {}
    
    template<class VecX>
    void apply(VecX &x) const
    {
    }


};

}
}
}

#endif