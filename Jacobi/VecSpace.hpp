#ifndef __VECTOR_SPACE_HPP__
#define __VECTOR_SPACE_HPP__

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
    class VectorType=scfd::arrays::array_nd<Type, Dim, typename Backend::memory_type>,
    class IdxType=scfd::static_vec::vec<Ordinal, Dim>
>
class vector_space : vector_operations_base<Type, VectorType, Ordinal>
{

public:
    static const int dim       = Dim;
    using value_type           = Type;
    using scalar_type          = Type;
    using ordinal_type         = Ordinal;

    using for_each_nd_type     = typename Backend::for_each_nd_type;
    using reduce_type          = typename Backend::reduce_type;
    using memory_type          = typename Backend::memory_type;

    using vector_type          = VectorType;
    using idx_nd_type          = IdxType;

public: // Especially for SYCL
    using shur_prod_kernel       = kernels::shur_prod<idx_nd_type, vector_type>;
    using assign_scalar_kernel   = kernels::assign_scalar<idx_nd_type, scalar_type, vector_type>;
    using add_mul_scalar_kernel  = kernels::add_mul_scalar<idx_nd_type, scalar_type, vector_type>;
    using scale_kernel           = kernels::scale<idx_nd_type, scalar_type, vector_type>;
    using assign_kernel          = kernels::assign<idx_nd_type, scalar_type, vector_type>;
    using assign_lin_comb_kernel = kernels::assign_lin_comb<idx_nd_type, scalar_type, vector_type>;
    using add_lin_comb_kernel    = kernels::add_lin_comb<idx_nd_type, scalar_type, vector_type>;

private:
    idx_nd_type                 range;
    ordinal_type                   sz;
    mutable vector_type        helper;

    for_each_nd_type for_each_nd_inst;
    reduce_type           reduce_inst;

public:
    vector_space(idx_nd_type const v): range(v), sz(v.components_prod()), helper(range) {};

    idx_nd_type get_range() const noexcept { return range; }

    //TODO
/*
    [[nodiscard]] virtual vector_type at(multivector_type& x, Ord m, Ord k_) = 0;

    [[nodiscard]] virtual bool is_valid_number(const vector_type &x) const = 0;
    reduction operations:
*/
    [[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, y, helper}, range);
        return reduce_inst(sz, helper.raw_ptr(), scalar_type{0});
    }

    [[nodiscard]] scalar_type sum(const vector_type &x) const override
    {
        return reduce_inst(sz, x.raw_ptr(), scalar_type{0});
    }

    //TODO
    [[nodiscard]] scalar_type asum(const vector_type &x) const override
    {
        return 0;
    }

    //standard vector norm:=sqrt(sum(x^2))
    [[nodiscard]] scalar_type norm(const vector_type &x) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return std::sqrt(reduce_inst(sz, helper.raw_ptr(), scalar_type{0}));
    }

    //L2 emulation for the vector norm2:=sqrt(sum(x^2)/sz_)
    [[nodiscard]] scalar_type norm2(const vector_type &x) const override
    {

        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return std::sqrt(reduce_inst(sz, helper.raw_ptr(), scalar_type{0}) / sz);
    }

    //standard vector norm_sq:=sum(x^2)
    [[nodiscard]] scalar_type norm_sq(const vector_type &x) const override
    {

        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return reduce_inst(sz, helper.raw_ptr(), scalar_type{0});
    }

    //L2 emulation for the vector norm2_sq:=sum(x^2)/sz_
    [[nodiscard]] scalar_type norm2_sq(const vector_type &x) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, x, helper}, range);
        return reduce_inst(sz, helper.raw_ptr(), scalar_type{0}) / sz;
    }

public:
    //calc: x := <vector_type with all elements equal to given scalar value>
    void assign_scalar(const scalar_type scalar, vector_type& x) const override
    {
        for_each_nd_inst(assign_scalar_kernel{scalar, x}, range);
    }

    //calc: x := mul_x*x + <vector_type of all scalar value>
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const override
    {
        for_each_nd_inst(add_mul_scalar_kernel{scalar, mul_x, x}, range);
    }

    //calc: x := scale*x
    void scale(const scalar_type scale, vector_type &x) const override
    {
        for_each_nd_inst(scale_kernel{scale, x}, range);
    }

    //copy: y := x
    void assign(const vector_type& x, vector_type& y) const override
    {
        for_each_nd_inst(assign_kernel{x, y}, range);
    }

    //calc: y := mul_x*x
    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const override
    {
        for_each_nd_inst(assign_lin_comb_kernel{mul_x, x, y}, range);
    }

    //calc: y := mul_x*x + mul_y*y
    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const override
    {
        for_each_nd_inst(add_lin_comb_kernel{mul_x, mul_y, x, y}, range);
    }
};

}// namespace operations

#endif
