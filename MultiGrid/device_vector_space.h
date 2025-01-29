#ifndef __DEVICE_VECTOR_SPACE_H__
#define __DEVICE_VECTOR_SPACE_H__

#include <algorithm>
#include <cmath>
#include <vector>

#include <nmfd/operations/vector_operations_base.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/static_vec/vec.h>

#include "kernels/vector_space.h"

namespace nmfd 
{

template
<
    class Type,
    int   Dim,
    class Backend, // class which defines for_each, reduce and memory types.
    class Ordinal=std::ptrdiff_t,
    /***********************************************************/
    class VectorType=scfd::arrays::array_nd<Type, Dim, typename Backend::memory_type>,
    class MultiVectorType=std::vector<VectorType>,
    class IdxType=scfd::static_vec::vec<Ordinal, Dim>,
    class ParentType=nmfd::operations::vector_operations_base<Type, VectorType, MultiVectorType, Ordinal>
>
class device_vector_space : ParentType 
{
    using parent_t = ParentType;

public:
    static const int dim       = Dim;
    using value_type           = Type;
    using scalar_type          = Type;
    using ordinal_type         = Ordinal;

    using for_each_nd_type     = typename Backend::for_each_nd_type;
    using reduce_type          = typename Backend::reduce_type;
    using memory_type          = typename Backend::memory_type;

    using multivector_type     = MultiVectorType;
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
    vector_type mutable        helper;

    for_each_nd_type for_each_nd_inst;
    reduce_type           reduce_inst;

public:
    device_vector_space(idx_nd_type const r, bool use_high_precision = false): 
        parent_t(use_high_precision), range(r), sz(r.components_prod()), helper(range) {};

    idx_nd_type get_range() const noexcept { return range; }

public:
    void init_vector(vector_type& x) const { x.init(range); }
    void free_vector(vector_type& x) const { x.free();      }
   
    void start_use_vector(vector_type& x) const {}
    void  stop_use_vector(vector_type& x) const {}
    
    void init_multivector(multivector_type& x, ordinal_type m) const
    {
        x.clear();
        x.reserve(m);
        for(int i=0; i<m; ++i) { x.emplace_back(range); }
    }
    void free_multivector(multivector_type& x, ordinal_type m) const
    {
        std::for_each_n(begin(x), m, [](vector_type& v){ v.free(); });
    }
    
    void start_use_multivector(multivector_type& x, ordinal_type m) const {}
    void  stop_use_multivector(multivector_type& x, ordinal_type m) const {}


public: // Implementing Vector_Operations interface
    [[nodiscard]] vector_type at(multivector_type& x, ordinal_type m, ordinal_type k_) const override
    {
        if ( k_ < 0 || k_>=m  )
            throw std::out_of_range("device_vector_space: multivector.at");
        // assert preferable ?
        return x.at(k_);
    }

    [[nodiscard]] bool is_valid_number(const vector_type &x) const override
    {
        throw std::logic_error("device_vector_space: multivector.is_valid_number not implemented yet");
        return true; //TODO 
    }
    
    // reduction operations:
    [[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y) const override
    {
        for_each_nd_inst(shur_prod_kernel{x, y, helper}, range);
        return reduce_inst(sz, helper.raw_ptr(), scalar_type{0});
    }

    [[nodiscard]] scalar_type sum(const vector_type &x) const override
    {
        return reduce_inst(sz, x.raw_ptr(), scalar_type{0});
    }

    [[nodiscard]] scalar_type asum(const vector_type &x) const override
    {
        throw std::logic_error("device_vector_space: multivector.asum not implemented yet");
        return 0; //TODO
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

}// namespace nmfd 

#endif
