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

#include "include/boundary.h"
#include "solvers/jacobi.h" //TODO move to nmfd

#include "config.h"
#include "backend.h"

using grid_step_type   = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type      = scfd::static_vec::vec<int   , dim>;

using log_t = scfd::utils::log_std;
    
using vec_ops_t     = nmfd::device_vector_space<scalar, dim, backend>;

using prolongator_t = tests::device_prolongator<vec_ops_t, log_t>;
using restrictor_t  = tests::device_restrictor <vec_ops_t, log_t>;
using ident_op_t    = tests::device_identity_op<vec_ops_t, log_t>;
using lin_op_t      = tests::device_laplace_op <vec_ops_t, log_t>;
using smoother_t    = tests::device_jacobi_pre <vec_ops_t, log_t>;
using coarsening_t  = tests::device_coarsening<lin_op_t, log_t>;

using mg_t = nmfd::preconditioners::mg
<
    lin_op_t, restrictor_t, prolongator_t,
    smoother_t, ident_op_t, coarsening_t,
    log_t
>;
using jacobi_solver    = jacobi<vec_ops_t, lin_op_t, smoother_t>;
using jacobi_mg_solver = jacobi<vec_ops_t, lin_op_t, mg_t>;

using vector_t         = typename vec_ops_t::vector_type;
using vector_view_t    = typename vector_t::view_type;


// This is due to SYCL device_copyable constraints
// since SCFD classes are not trivially copyable
#if defined(SYCL_BACKEND)

MAKE_SYCL_DEVICE_COPYABLE(vec_ops_t::shur_prod_kernel);
MAKE_SYCL_DEVICE_COPYABLE(vec_ops_t::assign_scalar_kernel);
MAKE_SYCL_DEVICE_COPYABLE(vec_ops_t::add_mul_scalar_kernel);
MAKE_SYCL_DEVICE_COPYABLE(vec_ops_t::scale_kernel);
MAKE_SYCL_DEVICE_COPYABLE(vec_ops_t::assign_kernel);
MAKE_SYCL_DEVICE_COPYABLE(vec_ops_t::assign_lin_comb_kernel);
MAKE_SYCL_DEVICE_COPYABLE(vec_ops_t::add_lin_comb_kernel);

MAKE_SYCL_DEVICE_COPYABLE(prolongator_t::prolongator_kernel);

MAKE_SYCL_DEVICE_COPYABLE(restrictor_t::restrictor_kernel);

MAKE_SYCL_DEVICE_COPYABLE(ident_op_t::identity_kernel);

MAKE_SYCL_DEVICE_COPYABLE(lin_op_t::laplace_op_kernel);

MAKE_SYCL_DEVICE_COPYABLE(smoother_t::preconditioner_kernel);
   
#endif
