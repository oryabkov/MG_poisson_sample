#ifndef __DEVICE_COARSENING_H__
#define __DEVICE_COARSENING_H__

#include <tuple>

#include "device_restrictor.h"
#include "device_prolongator.h"

namespace tests
{

template<class LinearOperator, class Log> 
class device_coarsening
{
public:
    using operator_type     = LinearOperator;
    using vector_space_type = typename operator_type::vector_space_type;
    using restrictor_type   = device_restrictor<vector_space_type,Log>;
    using prolongator_type  = device_prolongator<vector_space_type,Log>;

    using ordinal_type = typename vector_space_type::ordinal_type;
    using idx_nd_type  = typename vector_space_type::idx_nd_type;
public:
    struct params
    {
    };
    using params_hierarchy = params;
    struct utils
    {
    };
    using utils_hierarchy = utils;

    device_coarsening(const utils_hierarchy &u, const params_hierarchy &p)
    {
    }

    std::tuple
    <
        std::shared_ptr<restrictor_type>, 
        std::shared_ptr<prolongator_type>
    > 
    next_level(const operator_type &op)
    {
        return std::make_tuple
        (
            std::make_shared<restrictor_type> (op.get_size()),
            std::make_shared<prolongator_type>(op.get_size())
        );

    }

    std::shared_ptr<operator_type> 
    coarse_operator(const operator_type    &op, 
                    const restrictor_type  &restrictor, 
                    const prolongator_type &prolongator)
    {
        using  Ord = ordinal_type;
        return std::make_shared<operator_type>(op.get_size() / Ord{2u});
    }

    bool coarse_enough(const operator_type &op)const
    {
        auto range = op.get_size();
        for(int i=0; i<range.dim; ++i)
            if (range[i] <= 2)
                return true;
        return false;
    }
};

}// namespace tests

#endif
