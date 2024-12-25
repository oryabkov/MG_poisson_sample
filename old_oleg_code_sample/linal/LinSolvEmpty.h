#ifndef __LIN_SOLV_EMPTY_H__
#define __LIN_SOLV_EMPTY_H__

//VECTOR concept:
//
//VECTOR_OPS concept:
//void addMul(VECTOR& B, BASE_T mulB, const VECTOR& A, BASE_T mulA) calc: B = B*mulB + A*mulA
//OP concept:
//void operator()(const VECTOR& x,VECTOR& f)const
//PREC concept:
//void operator()(VECTOR& x)const
template<class OP,class PREC,class VECTOR_OPS,class VECTOR,class BASE_T>
class LinSolvEmpty
{
public:
	LinSolvEmpty() {}
	void	solve(const VECTOR_OPS &vec_ops, const OP &A, const PREC &P, VECTOR &x, const VECTOR &b)const
	{
		vec_ops.addMul(x,  0.f, b, 1.f);
        	P(x);
	}
};

#endif
