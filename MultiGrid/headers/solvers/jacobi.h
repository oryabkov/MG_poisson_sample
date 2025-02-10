#ifndef __JACOBI_H__
#define __JACOBI_H__

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
    using scalar_type = typename VectorOp::scalar_type;
    using vector_type = typename VectorOp::vector_type;

    using vector_operation_type   = VectorOp;
    using laplace_operator_type   = LaplaceOp;
    using preconditioner_type     = Precond;

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
                                          p_op(precond) { p_op->set_operator(l_op); }

    scalar_type make_step(const vector_type &rhs, vector_type &tmp, vector_type &x)
    {
        // tmp := Laplace(x) = Ax;
        l_op->apply(x, tmp);
        // tmp := Ax - b;
        v_op->add_lin_comb(scalar_type{-1}, rhs, scalar_type{1}, tmp);
        // residual := ||tmp||_l2;
        scalar_type residual = v_op->norm2(tmp);
        // tmp := P(Ax - b);
        p_op->apply(tmp);
        // x   := x - P(Ax - b);
        v_op->add_lin_comb(scalar_type{-1}, tmp, scalar_type{1}, x);

        return residual;
    }

};

#endif
