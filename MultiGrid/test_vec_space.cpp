#include <utility>
#include <iostream>

#include "device_vector_space.h"

constexpr int dim =      3;
using scalar      = double;

/*******************************************************/
#include <scfd/memory/hip.h>
#include <scfd/for_each/hip_nd_impl.h>
#include <scfd/reduce/thrust.h>

struct backend
{
    using memory_type      = scfd::memory::hip_device;
    using for_each_nd_type = scfd::for_each::hip_nd<dim>;
    using reduce_type      = scfd::thrust_reduce<>;
};
/*******************************************************/


using vector_space = nmfd::device_vector_space<scalar,/*dim=*/3, backend>;

using vector_t      = typename vector_space::vector_type;
using vector_view_t = typename vector_t::view_type; 

int main()
{
    vector_t x, y;
    
    vector_space vec_space({10,10,10});
    
    vec_space.init_vector(x);
    vec_space.init_vector(y);
    
    vec_space.start_use_vector(x);
    vec_space.start_use_vector(y);

    vec_space.assign_scalar(1.f, x);               //x=1
    vec_space.scale(2.f, x);                       //x=2
    vec_space.assign(x, y);                        //y=x
    vec_space.add_mul_scalar(3.f, -1.f, y);        //y=-2*1+3=1
    vec_space.add_lin_comb(4.f, x, -3.f, y);       //y=2*4-3*1=5
    vec_space.assign_lin_comb(2.f, y, x);          //x=5*2=10
    vector_view_t x_view(x), y_view(y);
    
    std::cout << "x = " << vec_space.norm2(x) << std::endl;
    std::cout << "y = " << vec_space.norm2(y) << std::endl;    
    
    x_view.release();
    y_view.release();
    
    std::cout << "dot(x, y)   = " << vec_space.scalar_prod(x,y) << std::endl;
    std::cout << "sum(x)      = " << vec_space.sum(x) << std::endl;
    std::cout << "norm(x)     = " << vec_space.norm(x) << std::endl;
    std::cout << "norm_sq(x)  = " << vec_space.norm_sq(x) << std::endl;
    std::cout << "norm2(x)    = " << vec_space.norm2(x) << std::endl;
    std::cout << "norm2_sq(x) = " << vec_space.norm2_sq(x) << std::endl;
    
    vec_space.free_vector(x);
    vec_space.free_vector(y);
        
    return 0;
}
