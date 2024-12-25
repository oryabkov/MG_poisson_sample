#ifndef __LIN_SOLV_JACOBI_H__
#define __LIN_SOLV_JACOBI_H__

//TODO test
#include <cstdio>


//VECTOR concept:
//
//VECTOR_OPS concept:
//BASE_T norm(const VECTOR &x)const
//BASE_T scalar_prod(const VECTOR &x, const VECTOR &y)const
//void verify_size(VECTOR&)const
//void addMul(VECTOR& B, BASE_T mulB, const VECTOR& A, BASE_T mulA) calc: B = B*mulB + A*mulA
//void addMul(VECTOR& C, BASE_T mulC, const VECTOR& A, BASE_T mulA, const VECTOR& B, BASE_T mulB) calc: C = C*mulC + A*mulA + B*mulB
//OP concept:
//void operator()(const VECTOR& x,VECTOR& f)const
template<class OP,class PREC,class VECTOR_OPS,class VECTOR,class BASE_T>
class LinSolvJacobi
{
	mutable	VECTOR	tmp;
public:
	LinSolvJacobi() : max_iters_num(100),min_iters_num(1) {}
	BASE_T	abs_tol;
	int 	max_iters_num, min_iters_num;
	void	solve(const VECTOR_OPS &vec_ops, const OP &A, const PREC &P, VECTOR &x, const VECTOR &b)const
	{
		printf("LinSolvJacobi::solve()\n");
		vec_ops.verify_size(tmp); 

		int	i = 1;
		while (1)
		{
			//x := x - P(Ax-b)
			A(x, tmp);				//tmp := Ax
			vec_ops.addMul(tmp, 1.f, b, -1.f);	//tmp := tmp - b = Ax-b
			BASE_T          resid_norm = vec_ops.norm(tmp);
			P(tmp);					//tmp := P(Ax - b)
			vec_ops.addMul(x, 1.f, tmp, -1.f);	//x := x - tmp = x - P(Ax - b)

                        i++;
                        if (i >= max_iters_num) break;

                        std::cout << "LinSolvJacobi: resid norm = " << resid_norm << " iter = " << i << std::endl;
                        std::cout << "LinSolvJacobi: abs_tol = " << abs_tol << std::endl;
                        if ((resid_norm < abs_tol)&&(i >= min_iters_num)) break;
		}
	}
};

#endif
