#ifndef __VECTOR_OPERATIONS_BASE_H__
#define __VECTOR_OPERATIONS_BASE_H__

#include <utility>

namespace operations
{

template<class Type, class VectorType, class MultiVectorType, class Ordinal = std::ptrdiff_t>
class vector_operations_base
{

public:
    using vector_type = VectorType;
    using multivector_type = MultiVectorType;
    using scalar_type = Type;
    using Ord = Ordinal;    
    using ordinal_type = Ord;
    using big_ordinal_type = Ord;  

private:
    using T = scalar_type;
/*
protected:
    bool use_high_precision_;

public:
    vector_operations_base(bool use_high_precision = false):
    use_high_precision_(use_high_precision)
    {}

    virtual ~vector_operations_base() = default;

    void set_high_precision() const
    {
        use_high_precision_ = true;
    }
    void set_regular_precision() const
    {
        use_high_precision_ = false;
    }
*/
    //[[nodiscard]] virtual vector_type at(multivector_type& x, Ord m, Ord k_) = 0;

    //[[nodiscard]] virtual bool is_valid_number(const vector_type &x) const = 0;
    //reduction operations:
    
    [[nodiscard]] virtual scalar_type scalar_prod(const vector_type &x, const vector_type &y)const = 0;
    [[nodiscard]] virtual scalar_type sum(const vector_type &x)const = 0;
    
    [[nodiscard]] virtual scalar_type asum(const vector_type &x)const = 0;

    //standard vector norm:=sqrt(sum(x^2))
    [[nodiscard]] virtual scalar_type norm(const vector_type &x) const = 0;
    //L2 emulation for the vector norm2:=sqrt(sum(x^2)/sz_)
    [[nodiscard]] virtual scalar_type norm2(const vector_type &x) const = 0;
    //standard vector norm_sq:=sum(x^2)
    [[nodiscard]] virtual scalar_type norm_sq(const vector_type &x) const = 0;
    //L2 emulation for the vector norm2_sq:=sum(x^2)/sz_
    [[nodiscard]] virtual scalar_type norm2_sq(const vector_type &x) const = 0;
    
    //calc: x := <vector_type with all elements equal to given scalar value> 
    virtual void assign_scalar(const scalar_type scalar, vector_type& x) const = 0;
    //calc: x := mul_x*x + <vector_type of all scalar value> 
    virtual void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const = 0;
    virtual void scale(const scalar_type scale, vector_type &x) const = 0;
    //copy: y := x
    virtual void assign(const vector_type& x, vector_type& y) const = 0;
    //calc: y := mul_x*x
    virtual void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const = 0;
    //calc: y := mul_x*x + mul_y*y
    virtual void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const = 0;


};

}

#endif
