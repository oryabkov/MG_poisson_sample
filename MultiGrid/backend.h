#ifndef __BACKEND_H__
#define __BACKEND_H__

#include "config.h"

#if   defined(CPU_BACKEND)

#include <scfd/memory/host.h>
#include <scfd/for_each/serial_cpu_nd.h>
#include <scfd/reduce/serial_cpu.h>

struct backend
{
    using memory_type      = scfd::memory::host;
    using for_each_nd_type = scfd::for_each::serial_cpu_nd<dim>;
    using reduce_type      = scfd::serial_cpu_reduce<>;
};



#elif defined(HIP_BACKEND)

#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>

struct backend
{
    using memory_type      = scfd::memory::hip_device;
    using for_each_nd_type = scfd::for_each::hip_nd<dim>;
    using reduce_type      = scfd::thrust_reduce<>;
};



#elif defined(OMP_BACKEND)

#include <scfd/memory/host.h>
#include <scfd/for_each/openmp_nd_impl.h>
#include <scfd/reduce/omp_reduce_impl.h>

struct backend
{
    using memory_type      = scfd::memory::host;
    using for_each_nd_type = scfd::for_each::openmp_nd<dim>;
    using reduce_type      = scfd::omp_reduce<>;
};



#elif defined(CUDA_BACKEND)

#include <scfd/memory/cuda.h>
#include <scfd/for_each/cuda_nd_impl.cuh>
#include <scfd/reduce/thrust.h>

struct backend
{
    using memory_type      = scfd::memory::cuda_device;
    using for_each_nd_type = scfd::for_each::cuda_nd<dim>;
    using reduce_type      = scfd::thrust_reduce<>;
};



#elif   defined(SYCL_BACKEND)

#include <scfd/memory/sycl.h>
#include <scfd/for_each/sycl_nd_impl.h>
#include <scfd/reduce/sycl_reduce_impl.h>

struct backend
{
    using memory_type      = scfd::memory::sycl_device;
    using for_each_nd_type = scfd::for_each::sycl_nd<dim>;
    using reduce_type      = scfd::sycl_reduce<>;
};

#define MAKE_SYCL_DEVICE_COPYABLE(kernel) template<>                        \
struct sycl::is_device_copyable<typename kernel> \
    : std::true_type {}

#endif


#endif
