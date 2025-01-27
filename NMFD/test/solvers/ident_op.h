#ifndef __TEST_IDENT_OP_H__
#define __TEST_IDENT_OP_H__

namespace tests
{


template<class VectorSpace, class Log> 
class ident_op
{
public:    
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    using ordinal_type = typename VectorSpace::ordinal_type;
private:
    using T = scalar_type;
    using T_vec = vector_type;

    ordinal_type N_;

public:

    ident_op() : N_(0)
    {
    }
    ident_op(ordinal_type N) : N_(N)
    {
    }
    ~ident_op()
    {}

    ordinal_type get_size() const
    {
        return N_;
    }
    /*T get_h() const
    {
        return h_;
    }*/

    template<class LinearOperator>
    void set_operator(std::shared_ptr<const LinearOperator> op)
    {
        N_ = op->get_size();
    }

    std::shared_ptr<vector_space_type> get_dom_space()const
    {
        return std::make_shared<vector_space_type>(N_);
    }
    std::shared_ptr<vector_space_type> get_im_space()const
    {
        return std::make_shared<vector_space_type>(N_);
    }

    void apply(const T_vec& x, T_vec& f)const
    { 
        for(ordinal_type j=0; j<N_; j++)
        {
            f[j] = x[j];
        }

    }

private:

};

}

#endif