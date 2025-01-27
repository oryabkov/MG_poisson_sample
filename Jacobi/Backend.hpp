#ifndef __BACKEND_HPP__
#define __BACKEND_HPP__

#include "Include.hpp"

#if   defined(HIP_BACKEND)

#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>

struct backend
{
    using memory_type      = scfd::memory::hip_device;
    using for_each_nd_type = scfd::for_each::hip_nd<current_dim>;
    using reduce_type      = scfd::thrust_reduce<>;
};

#elif defined(OMP_BACKEND)

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp_nd_impl.h>
#include <scfd/reduce/omp_reduce_impl.h>

struct backend
{
    using memory_type      = scfd::memory::host;
    using for_each_nd_type = scfd::for_each::openmp_nd<current_dim>;
    using reduce_type      = scfd::omp_reduce<>;
};

#elif

#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda_nd_impl.cuh>
#include <scfd/reduce/thrust.h>

struct backend
{
    using memory_type      = scfd::memory::hip_device;
    using for_each_nd_type = scfd::for_each::cuda_nd<current_dim>;
    using reduce_type      = scfd::thrust_reduce<>;
};

#endif

#endif
