#ifndef __LIN_SOLV_BICGSTAB_H__
#define __LIN_SOLV_BICGSTAB_H__

//VECTOR concept:
//
//VECTOR_OPS concept:
//BASE_T norm(const VECTOR &x)const
//BASE_T scalar_prod(const VECTOR &x, const VECTOR &y)const
//void verify_size(VECTOR&)const
//void addMul(VECTOR& B, BASE_T mulB, const VECTOR& A, BASE_T mulA) calc: B = B*mulB + A*mulA
//void addMul(VECTOR& C, BASE_T mulC, const VECTOR& A, BASE_T mulA, const VECTOR& B, BASE_T mulB) calc: C = C*mulC + A*mulA + B*mulB
//void addMulScalar(VECTOR& B, BASE_T mulB, BASE_T scalar)
//OP concept:
//void operator()(const VECTOR& x,VECTOR& f)const
template<class OP,class PREC,class VECTOR_OPS,class VECTOR,class BASE_T>
class LinSolvBiCGStab
{
	mutable	VECTOR	ri, r_, pi, s, t, nu_i;

	//mutable int	iters_performed;
public:
	LinSolvBiCGStab() : max_iters_num(100),min_iters_num(1) {}
	BASE_T	abs_tol;
	int 	max_iters_num, min_iters_num;
	mutable int	iters_performed;
	void	solve(const VECTOR_OPS &vec_ops, const OP &A, const PREC &P, VECTOR &x, const VECTOR &b)const
	{
		vec_ops.verify_size(ri); vec_ops.verify_size(r_); vec_ops.verify_size(pi); vec_ops.verify_size(s);
		vec_ops.verify_size(t); vec_ops.verify_size(nu_i);
		//ri := P*b - P*A*x0;
		A(x, ri); P(ri);                           //ri := P*A*x0
		vec_ops.addMul(s,  0.f, b, 1.f); P(s);	   //s := P*b
		vec_ops.addMul(ri, -1.f, s , 1.f);         //ri := -ri + s = -P*A*x0 + P*b
		//r_ := ri;
		vec_ops.addMul(r_,  0.f, ri, 1.f);

		int	i = 1;
		BASE_T	rho_i_2;

		BASE_T	rho_i_1 = BASE_T(1.f),
			alpha = BASE_T(1.f),
			omega_i_1 = BASE_T(1.f);
		vec_ops.addMulScalar(nu_i, 0.f, 0.f);
		vec_ops.addMulScalar(pi, 0.f, 0.f);

		while (1)
		{
			//nu_i, pi, ri are nu_{i-1}, p{i-1}, r{i-1}
			BASE_T	rho_i = vec_ops.scalar_prod(ri, r_),
				beta = (rho_i/rho_i_1)*(alpha/omega_i_1);
			if (beta != beta) { std::cout << "stop iterations because beta is ind" << std::endl; break; }
			//pi := ri + p{i-1}*beta - nu_i*beta*omega_i_1
                        vec_ops.addMul(pi, beta, ri, 1.f, nu_i, -beta*omega_i_1);
                        //pi now is pi
                        //nu_i := P*A*pi
                        A(pi, nu_i); P(nu_i);
                        //nu_i now is nu_i
                        alpha = rho_i/vec_ops.scalar_prod(nu_i, r_);
			if (alpha != alpha) { std::cout << "stop iterations because alpha is ind" << std::endl; break; }
                        vec_ops.addMul(s, 0.f, ri, 1.f, nu_i, -alpha);
                        //theta := P*A*pi
                        A(s, t); P(t);
                        BASE_T omega_i = vec_ops.scalar_prod(t, s)/vec_ops.scalar_prod(t, t);
			if (omega_i != omega_i) { std::cout << "stop iterations because omega_i is ind" << std::endl; break; }
                        vec_ops.addMul(x, 1.f, pi, alpha, s, omega_i);

			//if (alpha != alpha) { std::cout << "stop iterations because alpha is ind" << std::endl; break; }

                        //ri := s - t*omega_i
                        //ri now is ri
                        vec_ops.addMul(ri, 0.f, s, 1.f, t, -omega_i);

                        BASE_T          resid_norm = vec_ops.norm(ri);

                        if (i >= max_iters_num) break;
                        std::cout << "LinSolvBiCGStab: resid norm = " << resid_norm << " iter = " << i << std::endl;
                        std::cout << "LinSolvBiCGStab: abs_tol = " << abs_tol << std::endl;
                        if ((resid_norm < abs_tol)&&(i >= min_iters_num)) break;
                        i++;
                        omega_i_1 = omega_i; rho_i_1 = rho_i;
		}

		iters_performed = i;
	}
};

#endif
