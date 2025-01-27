#ifndef __SMOOTHER_ELLIPTIC__
#define __SMOOTHER_ELLIPTIC__


/**
*   Test class for iterative linear solver
*   Implements diagonal preconditioner to the  linear operator of the elliptic equation of the type:
*
*   -u_{xx} = f(x), x\in[0;1), u - periodic, f(x) is in the operator domain
*
*   -u_{j-1}/h^2 +2 u_{j}/h^2 - u_{j+1}/h^2 = f(x_j) 
*    
* 
*   A U = U^{n}
* 
*   preconditioner for the residual vecotr R={r_j}:
* 
*   u_{j} = (r_{j})/(2/h^2)
*
*/

#include <memory>
#include "linear_operator_elliptic.h"

namespace tests
{

template<class VectorSpace, class Log> 
class smoother_elliptic
{
public:
    using T = typename VectorSpace::scalar_type;
    using T_vec = typename VectorSpace::vector_type;
    using operator_type = linear_operator_elliptic<VectorSpace,Log>;

    struct params
    {
        params(const std::string &log_prefix = "", const std::string &log_name = "smoother_elliptic::")
        {
        }
    };
    using params_hierarchy = params;
    struct utils
    {
    };
    using utils_hierarchy = utils;

    smoother_elliptic(const utils_hierarchy &u, const params_hierarchy &p)
    {
    }
    ~smoother_elliptic()
    {
    }
    
    void set_operator(std::shared_ptr<const operator_type> op_) 
    {
        N = op_->get_size();
        h_ = op_->get_h();
        diag_coeff_ = (2/h_/h_);
    }

    void apply(T_vec& x)const
    {
        for(std::size_t j=0; j<N;j++)
        {
            x[j] = x[j]/diag_coeff_;
        }
    }

private:
    T diag_coeff_;
    std::size_t N;
    T h_;

};


}


#endif