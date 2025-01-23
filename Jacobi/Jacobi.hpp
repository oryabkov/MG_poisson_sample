#ifndef __JACOBI_HPP__
#define __JACOBI_HPP__

#include <memory>

template
<
     class VectorOp, 
     class LaplaceOp, 
     class Precond
>
class jacobi
{
public:
    using scalar_t = typename VectorOp::scalar_t;
    using vector_t = typename VectorOp::vector_t; 
    
    using vector_operation_t   = VectorOp;
    using laplace_operator_t   = LaplaceOp;
    using preconditioner_t     = Precond;
    
    using vector_operation_ptr = std::shared_ptr<VectorOp>; 
    using laplace_operator_ptr = std::shared_ptr<LaplaceOp>; 
    using preconditioner_ptr   = std::shared_ptr<Precond>; 

private:
    vector_operation_ptr    v_op; 
    laplace_operator_ptr    l_op;
    preconditioner_ptr      p_op;

public:
    jacobi(vector_operation_ptr vec_op, laplace_operator_ptr laplace_op, 
            preconditioner_ptr precond) : v_op(vec_op), 
                                          l_op(laplace_op), 
                                          p_op(precond) {} 

    scalar_t make_step(const vector_t &rhs, vector_t &tmp, vector_t &x) // todo rm tmp 
    {
        // tmp := Laplace(x) = Ax;
        l_op->apply(x, tmp);
        // tmp := Ax - b;
        v_op->add_lin_comb(scalar_t{-1}, rhs, scalar_t{1}, tmp); 
        // residual := ||tmp||_l2;
        scalar_t residual = v_op->norm2(tmp);
        // tmp := P(Ax - b);
        p_op->apply(tmp);
        // x   := x - P(Ax - b);
        v_op->add_lin_comb(scalar_t{-1}, tmp, scalar_t{1}, x);
        
        return residual;
    }
};

#endif
