#ifndef __VECTOR_OPERATIONS_H__
#define __VECTOR_OPERATIONS_H__

#include "VecSpaceKernel.hpp"
#include "VecOpBase.hpp"

#include <scfd/static_vec/vec.h>

namespace operations
{

template
<
    class Type, 
    int   Dim,
    class Backend, // class which defines for_each, reduce and memory types.
    class Ordinal=std::ptrdiff_t, 
    /****************************/ 
    class VectorType=scfd::arrays::array_nd<Type, Dim, typename Backend::memory_t>,
    class IdxType=scfd::static_vec::vec<Ordinal, Dim>
>
class vector_space : vector_operations_base<Type, VectorType, Ordinal> 
{

public:
    static const int dim    = Dim;
    using value_type        = Type;
    using scalar_t          = Type;
    using ordinal_t         = Ordinal;
    
    using for_each_nd_t     = typename Backend::for_each_nd_t;
    using reduce_t          = typename Backend::reduce_t;
    using memory_t          = typename Backend::memory_t;

    using vector_t          = VectorType;
    using idx_nd_t          = IdxType;

public: // Especially for SYCL
    using shur_prod_kernel       = kernels::shur_prod<idx_nd_t, vector_t>;
    using assign_scalar_kernel   = kernels::assign_scalar<idx_nd_t, scalar_t, vector_t>;
    using add_mul_scalar_kernel  = kernels::add_mul_scalar<idx_nd_t, scalar_t, vector_t>;
    using scale_kernel           = kernels::scale<idx_nd_t, scalar_t, vector_t>;
    using assign_kernel          = kernels::assign<idx_nd_t, scalar_t, vector_t>;
    using assign_lin_comb_kernel = kernels::assign_lin_comb<idx_nd_t, scalar_t, vector_t>;
    using add_lin_comb_kernel    = kernels::add_lin_comb<idx_nd_t, scalar_t, vector_t>;

private:
    idx_nd_t                 range;
    ordinal_t                   sz;
    mutable vector_t        helper;

    for_each_nd_t for_each_nd_inst;
    reduce_t           reduce_inst;

public:
    vector_space(idx_nd_t const v): range(v), sz(v.components_prod()), helper(range) {};

    idx_nd_t get_range() const noexcept { return range; }

    //TODO
/*
    [[nodiscard]] virtual vector_t at(multivector_t& x, Ord m, Ord k_) = 0;

    [[nodiscard]] virtual bool is_valid_number(const vector_t &x) const = 0;
    reduction operations:
*/  
    [[nodiscard]] scalar_t scalar_prod(const vector_t &x, const vector_t &y) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, y, helper}, range);
        return reduce_inst(sz, helper.raw_ptr(), scalar_t{0});
    }

    [[nodiscard]] scalar_t sum(const vector_t &x) const override
    {
        return reduce_inst(sz, x.raw_ptr(), scalar_t{0});
    }
    
    //TODO
    [[nodiscard]] scalar_t asum(const vector_t &x) const override
    {
        return 0;
    }

    //standard vector norm:=sqrt(sum(x^2))
    [[nodiscard]] scalar_t norm(const vector_t &x) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return std::sqrt(reduce_inst(sz, helper.raw_ptr(), scalar_t{0}));
    }

    //L2 emulation for the vector norm2:=sqrt(sum(x^2)/sz_)
    [[nodiscard]] scalar_t norm2(const vector_t &x) const override
    {
    
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return std::sqrt(reduce_inst(sz, helper.raw_ptr(), scalar_t{0}) / sz);
    }
    
    //standard vector norm_sq:=sum(x^2)
    [[nodiscard]] scalar_t norm_sq(const vector_t &x) const override
    {
        
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return reduce_inst(sz, helper.raw_ptr(), scalar_t{0});
    }
    
    //L2 emulation for the vector norm2_sq:=sum(x^2)/sz_
    [[nodiscard]] scalar_t norm2_sq(const vector_t &x) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return reduce_inst(sz, helper.raw_ptr(), scalar_t{0}) / sz;
    }

public:
    //calc: x := <vector_t with all elements equal to given scalar value> 
    void assign_scalar(const scalar_t scalar, vector_t& x) const override
    {
        for_each_nd_inst(assign_scalar_kernel{scalar, x}, range);
    }

    //calc: x := mul_x*x + <vector_t of all scalar value> 
    void add_mul_scalar(const scalar_t scalar, const scalar_t mul_x, vector_t& x) const override
    {
        for_each_nd_inst(add_mul_scalar_kernel{scalar, mul_x, x}, range);
    }
    
    //calc: x := scale*x 
    void scale(const scalar_t scale, vector_t &x) const override
    {
        for_each_nd_inst(scale_kernel{scale, x}, range);
    }

    //copy: y := x
    void assign(const vector_t& x, vector_t& y) const override
    {
        for_each_nd_inst(assign_kernel{x, y}, range);
    } 

    //calc: y := mul_x*x
    void assign_lin_comb(const scalar_t mul_x, const vector_t& x, vector_t& y) const override
    {
        for_each_nd_inst(assign_lin_comb_kernel{mul_x, x, y}, range);
    }

    //calc: y := mul_x*x + mul_y*y
    void add_lin_comb(const scalar_t mul_x, const vector_t& x, const scalar_t mul_y, vector_t& y) const override
    {
        for_each_nd_inst(add_lin_comb_kernel{mul_x, mul_y, x, y}, range);
    }
};

}// namespace operations

#endif
