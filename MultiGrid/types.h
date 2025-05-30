#ifndef __TYPES_H__
#define __TYPES_H__

#include "config.h"

#include <scfd/utils/log.h>
#include <nmfd/preconditioners/mg.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/default_monitor.h>
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

#include "backend.h"

using grid_step_type   = scfd::static_vec::vec<scalar, dim>;
using idx_nd_type      = scfd::static_vec::vec<int   , dim>;

using log_t = scfd::utils::log_std;

using vec_ops_t     = nmfd::device_vector_space<scalar, dim, backend>;

using krylov_monitor_t  = nmfd::solvers::monitor_krylov<vec_ops_t, log_t>;
using default_monitor_t = nmfd::solvers::default_monitor<vec_ops_t, log_t>; 
using monitor_funcs_t   = default_monitor_t::custom_funcs_type;
using monitor_funcs_ptr = default_monitor_t::custom_funcs_ptr;

using prolongator_t = tests::device_prolongator<vec_ops_t, log_t>;
using restrictor_t  = tests::device_restrictor <vec_ops_t, log_t>;
using ident_op_t    = tests::device_identity_op<vec_ops_t, log_t>;
using lin_op_t      = tests::device_laplace_op <vec_ops_t, log_t>;
using smoother_t    = tests::device_jacobi_pre <vec_ops_t, log_t>;
using coarsening_t  = tests::device_coarsening<lin_op_t, log_t>;

using precond_interface = nmfd::preconditioners::preconditioner_interface<vec_ops_t,lin_op_t>;

using mg_t = nmfd::preconditioners::mg
<
    lin_op_t, restrictor_t, prolongator_t,
    smoother_t, ident_op_t, coarsening_t,
    log_t
>;
using mg_params_t = mg_t::params_hierarchy;
using mg_utils_t  = mg_t::utils_hierarchy;


using jacobi_solver    =                jacobi<vec_ops_t, lin_op_t, precond_interface, default_monitor_t, log_t>;
using gmres_solver     = nmfd::solvers::gmres <vec_ops_t, krylov_monitor_t, log_t, lin_op_t, precond_interface>;

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

#endif
