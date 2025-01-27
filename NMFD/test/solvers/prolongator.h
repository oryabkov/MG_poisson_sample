#ifndef __TEST_PROLONGATOR_H__
#define __TEST_PROLONGATOR_H__

/**
*   Test prolongator class for (geometric) multigrid solver
*   Works for 1d space and simply duplicates value from coarse level for two adjusent cells on fine level
*   Basic size N passed to the constructor is the size of the fine level
*   Only even size is supported
*/


namespace tests
{


template<class VectorSpace, class Log> 
class prolongator
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
    //T h_;

public:

    prolongator(ordinal_type N) : N_(N)
    {
        if (N%2 != 0) throw std::logic_error("test::prolongator: N is odd! not supported case");
        //h_ = T(1.0)/static_cast<T>(N);
    }
    ~prolongator()
    {}

    ordinal_type get_size() const
    {
        return N_;
    }
    /*T get_h() const
    {
        return h_;
    }*/

    std::shared_ptr<vector_space_type> get_dom_space()const
    {
        return std::make_shared<vector_space_type>(N_/2);
    }
    std::shared_ptr<vector_space_type> get_im_space()const
    {
        return std::make_shared<vector_space_type>(N_);
    }

    /// x is vector of size N_/2, f is of N_ size
    void apply(const T_vec& x, T_vec& f)const
    { 
        for(ordinal_type j=0; j<N_; j++)
        {
            f[j] = x[j/2];
        }
        /*for(ordinal_type j=0; j<N_; j++)
        {
            if (j%2 == 0)
            {
                f[j] = x[j/2];
            }
            else
            {
                if (j < N_-1)
                    f[j] = T(0.5)*x[j/2] + T(0.5)*x[j/2+1];
                else
                    f[j] = T(0.5)*x[j/2] + T(0.5)*x[0];
            }
        }*/

    }

private:

};

}

#endif