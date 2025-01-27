#ifndef __TEST_COARSENING_H__
#define __TEST_COARSENING_H__

#include <tuple>
#include "restrictor.h"
#include "prolongator.h"

namespace tests
{


template<class LinearOperator, class Log> 
class coarsening
{
public:
    using operator_type = LinearOperator;
    using vector_space_type = typename operator_type::vector_space_type;
    using restrictor_type = restrictor<vector_space_type,Log>;
    using prolongator_type = prolongator<vector_space_type,Log>;
public:
    struct params
    {
    };
    using params_hierarchy = params;
    struct utils
    {
    };
    using utils_hierarchy = utils;

    coarsening(const utils_hierarchy &u, const params_hierarchy &p)
    {
    }

    std::tuple<std::shared_ptr<restrictor_type>,std::shared_ptr<prolongator_type>> 
    next_level(const operator_type &op)
    {
        return 
            std::make_tuple(
                std::make_shared<restrictor_type>(op.get_size()),
                std::make_shared<prolongator_type>(op.get_size())
            );

    }
    std::shared_ptr<operator_type> 
    coarse_operator(const operator_type &op, const restrictor_type &restrictor, const prolongator_type &prolongator)
    {
        return std::make_shared<operator_type>(op.get_size()/2);
    }
    bool coarse_enough(const operator_type &op)const
    {
        return op.get_size() <= 2;
    }
};

}

#endif