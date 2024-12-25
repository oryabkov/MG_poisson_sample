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
	mutable BASE_T	min_resid_norm;
	mutable VECTOR	min_resid_norm_x;
	mutable VECTOR	nonprecond_ri;

	//mutable int	iters_performed;
public:
	LinSolvBiCGStab() : max_iters_num(100),min_iters_num(1),out_min_resid_norm(false),dx_tol(BASE_T(-1.f)),nonprecond_abs_tol(BASE_T(-1.f)) {}
	BASE_T	abs_tol, dx_tol, nonprecond_abs_tol;
	int 	max_iters_num, min_iters_num;
	bool	out_min_resid_norm;

	mutable int	iters_performed;
	void	solve(const VECTOR_OPS &vec_ops, const OP &A, const PREC &P, VECTOR &x, const VECTOR &b)const
	{
		vec_ops.verify_size(ri); vec_ops.verify_size(r_); vec_ops.verify_size(pi); vec_ops.verify_size(s);
		vec_ops.verify_size(t); vec_ops.verify_size(nu_i);
		if (out_min_resid_norm) {
			vec_ops.verify_size(min_resid_norm_x);
                        vec_ops.addMul(min_resid_norm_x, 0.f, x, 1.f);
		}
		if (nonprecond_abs_tol >= BASE_T(0.f)) {
			vec_ops.verify_size(nonprecond_ri);
		}

		//TEST
		BASE_T	test = vec_ops.scalar_prod(b, b);
		std::cout << "test = " << test << std::endl;
		test = vec_ops.scalar_prod(x, x);
		std::cout << "test = " << test << std::endl;
		//TEST END

		//ri := P*b - P*A*x0;
		A(x, ri); //P(ri);                           //ri := A*x0
		//test = vec_ops.scalar_prod(ri, ri);
		//std::cout << "test = " << test << std::endl;
		//P(ri);
		//test = vec_ops.scalar_prod(ri, ri);
		//std::cout << "test = " << test << std::endl;

		vec_ops.addMul(s,  0.f, b, 1.f); //P(s);	//s := b
		vec_ops.addMul(ri, -1.f, s , 1.f);         	//ri := -ri + s = -A*x0 + b
		if (nonprecond_abs_tol >= BASE_T(0.f)) {
			vec_ops.addMul(nonprecond_ri, 0.f, ri, 1.f);
			//nonprecond_ri now here is nonpreconditioned ri
		}
                P(ri);						//ri := P*ri = -P*A*x0 + P*b
		//r_ := ri;
		vec_ops.addMul(r_,  0.f, ri, 1.f);

		//TEST
		test = vec_ops.scalar_prod(r_, r_);
		std::cout << "test = " << test << std::endl;
		//TEST END

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
			std::cout << "rho_i = " << rho_i << std::endl;
			std::cout << "rho_i_1 = " << rho_i_1 << std::endl;
			std::cout << "alpha = " << alpha << std::endl;
			std::cout << "omega_i_1 = " << omega_i_1 << std::endl;
			if (beta != beta) { std::cout << "stop iterations because beta is ind" << std::endl; break; }
			if (isinf(beta)) { std::cout << "stop iterations because beta is inf" << std::endl; break; }
			//pi := ri + p{i-1}*beta - nu_i*beta*omega_i_1
                        vec_ops.addMul(pi, beta, ri, 1.f, nu_i, -beta*omega_i_1);
                        //pi now is pi
                        //nu_i := P*A*pi
                        A(pi, nu_i);
                        if (nonprecond_abs_tol >= BASE_T(0.f)) {
				//use t as buffer 4 nonprecond nu_i
				vec_ops.addMul(t, 0.f, nu_i, 1.f);
			}
			P(nu_i);
                        //nu_i now is nu_i
                        alpha = rho_i/vec_ops.scalar_prod(nu_i, r_);
			if (alpha != alpha) { std::cout << "stop iterations because alpha is ind" << std::endl; break; }
			if (isinf(alpha)) { std::cout << "stop iterations because alpha is inf" << std::endl; break; }
                        vec_ops.addMul(s, 0.f, ri, 1.f, nu_i, -alpha);
                        if (nonprecond_abs_tol >= BASE_T(0.f)) {
				//NOTE t here is nonpreconditioned nu_i
				vec_ops.addMul(nonprecond_ri, 1.f, t, -alpha);
				//nonprecond_ri here is nonpreconditioned s
			}
                        //theta := P*A*pi
                        A(s, t);
                        if (nonprecond_abs_tol >= BASE_T(0.f)) {
				//use ri as buffer 4 nonprecond t
				vec_ops.addMul(ri, 0.f, t, 1.f);
			}
			P(t);
                        BASE_T omega_i = vec_ops.scalar_prod(t, s)/vec_ops.scalar_prod(t, t);
			if (omega_i != omega_i) { std::cout << "stop iterations because omega_i is ind" << std::endl; break; }
			if (isinf(omega_i)) { std::cout << "stop iterations because omega_i is inf" << std::endl; break; }
                        vec_ops.addMul(x, 1.f, pi, alpha, s, omega_i);

			//if (alpha != alpha) { std::cout << "stop iterations because alpha is ind" << std::endl; break; }

                        BASE_T		nonprecond_resid_norm;
                        if (nonprecond_abs_tol >= BASE_T(0.f)) {
				//NOTE ri here is nonpreconditioned t
				//NOTE nonprecond_ri here is nonpreconditioned s
				vec_ops.addMul(nonprecond_ri, 1.f, ri, -omega_i);
				//nonprecond_ri now here is nonpreconditioned ri
				nonprecond_resid_norm = vec_ops.norm(nonprecond_ri);
			}

			BASE_T		dx_norm;
			if (dx_tol >= BASE_T(0.f)) {
				//if we need dx_norm we use ri as buf here to calc dx; note that in next line we update ri whithout using its previous value
                                vec_ops.addMul(ri, 0.f, pi, alpha, s, omega_i);
                                dx_norm = vec_ops.norm(ri);
			}

                        //ri := s - t*omega_i
                        //ri now is ri
                        vec_ops.addMul(ri, 0.f, s, 1.f, t, -omega_i);


                        BASE_T          resid_norm = vec_ops.norm(ri);

                        if (out_min_resid_norm) {
				if ((i == 1)||(resid_norm < min_resid_norm)) {
					min_resid_norm = resid_norm;
					vec_ops.addMul(min_resid_norm_x, 0.f, x, 1.f);
				}
			}

                        if (i >= max_iters_num) break;
                        std::cout << "LinSolvBiCGStab: resid norm = " << resid_norm << " iter = " << i << std::endl;
                        std::cout << "LinSolvBiCGStab: abs_tol = " << abs_tol << std::endl;
                        if (((abs_tol >= BASE_T(0.f))||(dx_tol >= BASE_T(0.f))||(nonprecond_abs_tol >= BASE_T(0.f)))&&          //we have at least one criteria 					//at least one criteria is present
			    (((abs_tol < BASE_T(0.f))||(resid_norm < abs_tol))&&
			     ((nonprecond_abs_tol < BASE_T(0.f))||(nonprecond_resid_norm < nonprecond_abs_tol))&&
			     ((dx_tol < BASE_T(0.f))||(dx_norm < dx_tol)))&& 							//all presented criteria are satisfied
			    (i >= min_iters_num)) break;                                                                        //minit number of iterations reached
                        i++;
                        omega_i_1 = omega_i; rho_i_1 = rho_i;
		}
		
		if (out_min_resid_norm) {
			vec_ops.addMul(x, 0.f, min_resid_norm_x, 1.f);
		}

		iters_performed = i;
	}
};

#endif
