#include <utility>
#include <iostream>

#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_nd.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>
#include <scfd/arrays/array_nd.h>

#include "VecOp.hpp"

static constexpr std::size_t current_dim = 3;

using scalar        = float;
using memory_t      = scfd::memory::hip_device;
using for_each_t    = scfd::for_each::hip_nd<current_dim>;
using reduce_t      = scfd::thrust_reduce<>;

struct hip_backend
{
    using memory_t = scfd::memory::hip_device;
    using for_each_nd_t = scfd::for_each::hip_nd<current_dim>;
    using reduce_t = scfd::thrust_reduce<>;
};

using vector_space = operations::vector_space<scalar,/*dim=*/3, hip_backend>;

using vector_t      = typename vector_space::vector_t;
using vector_view_t = typename vector_t::view_type; 

int main()
{

    vector_t x({10,10,10}), y({10,10,10});
    
    vector_space vec_space({10,10,10});
    
    vec_space.assign_scalar(1.f, x);               //x=1
    vec_space.scale(2.f, x);                       //x=2
    vec_space.assign(x, y);                        //y=x
    vec_space.add_mul_scalar(3.f, -1.f, y);        //y=-2*1+3=1
    vec_space.add_lin_comb(4.f, x, -3.f, y);       //y=2*4-3*1=5
    vec_space.assign_lin_comb(2.f, y, x);          //x=5*2=10

    vector_view_t x_view(x), y_view(y);
    
    std::cout << "x = " << x_view(5,5,5) << std::endl;
    std::cout << "y = " << y_view(5,5,5) << std::endl;    
    
    x_view.release();
    y_view.release();
    
    std::cout << "dot(x, y)   = " << vec_space.scalar_prod(x,y) << std::endl;
    std::cout << "sum(x)      = " << vec_space.sum(x) << std::endl;
    std::cout << "norm(x)     = " << vec_space.norm(x) << std::endl;
    std::cout << "norm_sq(x)  = " << vec_space.norm_sq(x) << std::endl;
    std::cout << "norm2(x)    = " << vec_space.norm2(x) << std::endl;
    std::cout << "norm2_sq(x) = " << vec_space.norm2_sq(x) << std::endl;


    return 0;
}
