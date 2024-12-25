#ifndef __LIN_SOLV_CGS_H__
#define __LIN_SOLV_CGS_H__

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
class LinSolvCGS
{
	mutable	VECTOR	ri, r_, pi, ui, qi, theta, tmp;
	mutable BASE_T	min_resid_norm;
	mutable VECTOR	min_resid_norm_x;
public:
	LinSolvCGS() : max_iters_num(100),min_iters_num(1),out_min_resid_norm(false) {}
	BASE_T	abs_tol;
	int 	max_iters_num, min_iters_num;
	bool	out_min_resid_norm;

	//some output stat
	mutable int	iters_performed;

	void	solve(const VECTOR_OPS &vec_ops, const OP &A, const PREC &P, VECTOR &x, const VECTOR &b)const
	{
		printf("LinSolvCGS::solve()\n");
		vec_ops.verify_size(ri); vec_ops.verify_size(r_); vec_ops.verify_size(pi); vec_ops.verify_size(ui);
		vec_ops.verify_size(qi); vec_ops.verify_size(theta); vec_ops.verify_size(tmp);
		if (out_min_resid_norm) {
			vec_ops.verify_size(min_resid_norm_x);
                        vec_ops.addMul(min_resid_norm_x, 0.f, x, 1.f);
		}
		printf("LinSolvCGS::solve(): apply A;\n");
		//ri := P*b - P*A*x0;
		A(x, ri);                        //ri := A*x0
		printf("LinSolvCGS::solve(): addMul()..;\n");
		vec_ops.addMul(ri, -1.f, b , 1.f);       //ri := -ri + b = -A*x0 + b
		printf("LinSolvCGS::solve(): apply P;\n");
		P(ri);                                   //ri := P*ri = -P*A*x0 + P*b
		//r_ := ri;
		vec_ops.addMul(r_,  0.f, ri, 1.f);

		int	i = 1;
		BASE_T	rho_i_2;
		while (1)
		{
			BASE_T	rho_i_1 = vec_ops.scalar_prod(ri, r_);
			//TODO check whether rho_k is zero
			if (i == 1) {
				vec_ops.addMul(ui, 0.f, ri, 1.f);
                                vec_ops.addMul(pi, 0.f, ui, 1.f);
			} else {
				BASE_T	beta_i_1 = rho_i_1/rho_i_2;
                                if (isnan(beta_i_1)) { std::cout << "stop iterations because beta_i_1 is nan" << std::endl; break; }
				if (isinf(beta_i_1)) { std::cout << "stop iterations because beta_i_1 is inf" << std::endl; break; }
				//std::cout << "beta_i_1 = " << beta_i_1 << std::endl;
				vec_ops.addMul(ui, 0.f, ri, 1.f, qi, beta_i_1);
                                vec_ops.addMul(pi, beta_i_1*beta_i_1, ui, 1.f, qi, beta_i_1);
			}
			//theta := P*A*pi
                        A(pi, theta); P(theta);
			BASE_T	alpha_i = rho_i_1/vec_ops.scalar_prod(theta, r_);
                        if (isnan(alpha_i)) { std::cout << "stop iterations because alpha_i is nan" << std::endl; break; }
			if (isinf(alpha_i)) { std::cout << "stop iterations because alpha_i is inf" << std::endl; break; }
			//std::cout << "vec_ops.scalar_prod(theta, r_) = " << vec_ops.scalar_prod(theta, r_) << std::endl;
			//std::cout << "alpha_i = " << alpha_i << std::endl;
			//qi := ui - alpha_i*theta
			vec_ops.addMul(qi, 0.f, ui, 1.f, theta, -alpha_i);
			//theta := ui + qi;
			vec_ops.addMul(theta, 0.f, ui, 1.f, qi, 1.);
			//xi := x{i-1} + theta*alpha_i
                        vec_ops.addMul(x, 1.f, theta, alpha_i);
                        //BASE_T          resid_norm = fabs(alpha_i)*vec_ops.norm(theta);
                        //NOTE this incremental variant of ri calculation was in original pseudocode, i changed it for explicit residual calculation
                        /*
                        //tmp := P*A*theta
                        A(theta, tmp); P(tmp);
                        //ri := r{i-1} - tmp*alpha_i
                        vec_ops.addMul(ri, 1.f, tmp, -alpha_i);
                        */
                        
                        //ri := P*b - P*A*x0;
                        A(x, ri);                            	   //ri := A*xi
                        vec_ops.addMul(ri, -1.f, b , 1.f);         //ri := -ri + b = -A*xi + b
                        P(ri);                                     //ri := P*ri = -P*A*xi + P*b

                        BASE_T          resid_norm = vec_ops.norm(ri);
                        
                        if (out_min_resid_norm) {
				if ((i == 1)||(resid_norm < min_resid_norm)) {
					min_resid_norm = resid_norm;
					vec_ops.addMul(min_resid_norm_x, 0.f, x, 1.f);
				}
			}

                        rho_i_2 = rho_i_1;

                        i++;
                        if (i >= max_iters_num) break;

                        //BASE_T        resid_norm = vec_ops.norm(ri);
                        //BASE_T                resid_norm = alpha_i*vec_ops.norm(theta);
                        std::cout << "LinSolvCGS: resid norm = " << resid_norm << " iter = " << i << std::endl;
                        std::cout << "LinSolvCGS: abs_tol = " << abs_tol << std::endl;
                        if ((resid_norm < abs_tol)&&(i >= min_iters_num)) break;
		}
		
		if (out_min_resid_norm) {
			vec_ops.addMul(x, 0.f, min_resid_norm_x, 1.f);
		}

		iters_performed = i;
	}
};

#endif
