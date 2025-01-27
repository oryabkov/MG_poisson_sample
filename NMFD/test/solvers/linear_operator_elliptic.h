#ifndef __TEST_LINEAR_OPERATOR_ELLIPTIC_H__
#define __TEST_LINEAR_OPERATOR_ELLIPTIC_H__

/**
*   Test class for iterative linear solver
*   Implements preconditioner to the  linear operator of the elliptic equation of the type:
*
*   -u_{xx} = f(x), x\in[0;1), u - periodic, f(x) is in the perator domain
*
*   -u_{j-1}/h^2 +2 u_{j}/h^2 - u_{j+1}/h^2 = f(x_j) 
*    
* 
*   A U = U^{n}
* 
*   preconditioner for the residual vecotr R={r_j}:
* 
*   u_{j}^{n+1} = (r_{j} + u_{j+1}/h^2^{n} + u_{j-1}/h^2 ^{n+1})/(2/h^2)
*
*/


namespace tests
{


template<class VectorSpace, class Log> 
class linear_operator_elliptic
{
public:    
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    using Ord = typename VectorSpace::ordinal_type;
private:
    using T = scalar_type;
    using T_vec = vector_type;
    //const VectorSpace& vec_ops_;
    Ord N_;
    T h_;

public:

    linear_operator_elliptic(Ord N):
        N_(N)
    {
        h_ = 1.0/static_cast<T>(N);
    }
    linear_operator_elliptic(const VectorSpace& vec_ops) :
        linear_operator_elliptic(vec_ops.size())
    {
    }
    ~linear_operator_elliptic()
    {}

    T diag_coefficient() const
    {
        return (2*1/h_/h_);
    }
    Ord get_size() const
    {
        return N_;
    }
    T get_h() const
    {
        return h_;
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
        for(Ord j=0; j<N_; j++)
        {
            if((j>0)&&(j<N_-1))
                f[j] = (2/h_/h_)*x[j] - (1/h_/h_)*x[j-1] - (1/h_/h_)*x[j+1];
            else if(j==0)
                f[j] = (2/h_/h_)*x[0] - (1/h_/h_)*x[N_-1] - (1/h_/h_)*x[1];
            else if(j==N_-1)
                f[j] = (2/h_/h_)*x[N_-1] - (1/h_/h_)*x[N_-2] - (1/h_/h_)*x[0];
        }

    }

private:

};

}

#endif