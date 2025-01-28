#include <memory>
#include <cmath>
#include <scfd/utils/log.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/gmres.h>

#include "device_vector_space.h"
#include "device_restrictor.h"
#include "device_prolongator.h"
#include "device_identity_op.h"
#include "device_laplace_op.h"
#include "device_jacobi_pre.h"
#include "device_coarsening.h"

constexpr int dim =      3;
using scalar      = double;

/***********************/
#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>

struct backend
{
    using memory_type      = scfd::memory::hip_device;
    using for_each_nd_type = scfd::for_each::hip_nd<dim>;
    using reduce_type      = scfd::thrust_reduce<>;
};
/***********************/

#define M_PIl 3.141592653589793238462643383279502884L

int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    
    int error = 0;
    log_t log;
    
    using vec_ops_t     = nmfd::device_vector_space<scalar, dim, backend>;
    
    using prolongator_t = tests::device_prolongator<vec_ops_t, log_t>;
    using restrictor_t  = tests::device_restrictor <vec_ops_t, log_t>;
    using ident_op_t    = tests::device_identity_op<vec_ops_t, log_t>;
    using lin_op_t      = tests::device_laplace_op <vec_ops_t, log_t>;
    using smoother_t    = tests::device_jacobi_pre <vec_ops_t, log_t>;
    using coarsening_t  = tests::device_coarsening<lin_op_t, log_t>;
    std::shared_ptr<vec_ops_t> vec_ops;
    

    using mg_t = nmfd::preconditioners::mg
    <
        lin_op_t, restrictor_t, prolongator_t, 
        smoother_t, ident_op_t, coarsening_t, 
        log_t
    >;
    using mg_params_t = mg_t::params_hierarchy;
    using mg_utils_t = mg_t::utils_hierarchy;
    
    return error;
}
