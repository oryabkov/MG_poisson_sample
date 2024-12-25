#ifndef __T_ML_V_TML_H__
#define __T_ML_V_TML_H__

#include <vector>
#include <cassert>

//ISSUE maybe use smart pointers or something?
//0 is finest level
//NOTE don't own objects refferenced from class except for bi,xi temporal vectors
//VECTOR concept:
//DefaultConstructible
//VECTOR_OPS concept:
//void verify_size(VECTOR&)const
//void addMul(VECTOR& B, BASE_T mulB, const VECTOR& A, BASE_T mulA) calc: B = B*mulB + A*mulA
//void addMul(VECTOR& C, BASE_T mulC, const VECTOR& A, BASE_T mulA, const VECTOR& B, BASE_T mulB) calc: C = C*mulC + A*mulA + B*mulB
//void addMulScalar(VECTOR& B, BASE_T mulB, BASE_T scalar)
//OP concept:
//void operator()(const VECTOR& x,VECTOR& f)const
template<class PROJ_OP,class PROL_OP,class SMOOTHER_OP,class OP,class VECTOR_OPS,class VECTOR,class T>
class t_ML_V_tml
{
	//proj[i]'s result lies on i-th level
	std::vector<const PROJ_OP*>		proj;
	//prol[i]'s result lies on i-th level
	std::vector<const PROL_OP*>		prol;
	std::vector<const SMOOTHER_OP*>		smoother;
	std::vector<const OP*>			oper;
	std::vector<int>			hierarchy;
	std::vector<const VECTOR_OPS*>		vec_ops;
	mutable std::vector<VECTOR*>		bi,xi,ri;

	t_ML_V_tml	&operator=(const t_ML_V_tml& x)
	{
	}
	t_ML_V_tml(const t_ML_V_tml& x)
	{
	}
public:
	t_ML_V_tml()
	{
	}
	void	init0(int levs_num/*, int hierarchy_len*/)
	{
		proj.resize(levs_num);
		prol.resize(levs_num);
		bi.resize(levs_num);
		xi.resize(levs_num);
		ri.resize(levs_num);
		oper.resize(levs_num);
		smoother.resize(levs_num);
		vec_ops.resize(levs_num);
		for (int i = 0;i < levs_num;++i) {
			vec_ops[i] = NULL;
			proj[i] = NULL;
			prol[i] = NULL;
			smoother[i] = NULL;
			ri[i] = bi[i] = xi[i] = NULL;
		}
                //hierarchy.resize(hierarchy_len);
                hierarchy.resize(levs_num-1);
                for (int i = 0;i < hierarchy.size();++i) {
			hierarchy[i] = levs_num-i-1;
		}
	}
	void	set_level(int ilev, const VECTOR_OPS *_vec_ops, const PROJ_OP *_proj, const PROL_OP *_prol, const SMOOTHER_OP *_smoother, const OP *_op)
	{
		proj[ilev] = _proj;
		prol[ilev] = _prol;
		smoother[ilev] = _smoother;
		oper[ilev] = _op;
		vec_ops[ilev] = _vec_ops;
	}
	void	init()
	{
		int levs_num = smoother.size();
		for (int i = 0;i < levs_num;++i) {
			xi[i] = new VECTOR();
			bi[i] = new VECTOR();
			ri[i] = new VECTOR();
			vec_ops[i]->verify_size(*xi[i]);
			vec_ops[i]->verify_size(*bi[i]);
			vec_ops[i]->verify_size(*ri[i]);
		}
	}

	void	recursive_MG(int ilev)const
	{
		printf("t_ML_V_tml::recursive_MG(%d)\n", ilev);
		int levs_num = smoother.size();
		if (ilev == levs_num-1) {
			printf("t_ML_V_tml::recursive_MG: (*smoother[ilev])(*xi[ilev], *bi[ilev])\n");
			(*smoother[ilev])(*xi[ilev], *bi[ilev]);
		} else {
			printf("t_ML_V_tml::recursive_MG: (*smoother[ilev])(*xi[ilev], *bi[ilev])\n");
			if (ilev != 0) (*smoother[ilev])(*xi[ilev], *bi[ilev]);
			printf("t_ML_V_tml::recursive_MG: 1\n");
                        (*oper[ilev])(*xi[ilev], *ri[ilev]);
                        printf("t_ML_V_tml::recursive_MG: 2\n");
                        vec_ops[ilev]->addMul(*ri[ilev], T(-1.f), *bi[ilev], T(1.f));
                        printf("t_ML_V_tml::recursive_MG: 3\n");
                        (*proj[ilev+1])(*ri[ilev],*bi[ilev+1]);
                        printf("t_ML_V_tml::recursive_MG: 4\n");
                        vec_ops[ilev+1]->addMul(*xi[ilev+1], T(0.f), *xi[ilev+1], T(0.f));
                        recursive_MG(ilev+1);
                        printf("t_ML_V_tml::recursive_MG: 5\n");
                        (*prol[ilev])(*xi[ilev+1],*ri[ilev]);
                        printf("t_ML_V_tml::recursive_MG: 6\n");
                        vec_ops[ilev]->addMul(*xi[ilev], T(1.f), *ri[ilev], T(1.f));
                        printf("t_ML_V_tml::recursive_MG: (*smoother[ilev])(*xi[ilev], *bi[ilev])\n");
			if (ilev != 0) (*smoother[ilev])(*xi[ilev], *bi[ilev]);
		}
	}

	//apply ML-operator to x vector and returns result in y vector
	//NOTE x is usually RHS(b) and in y approximate solution will be returned
	void	operator()(const VECTOR &x, VECTOR &y)const
	{
		printf("t_ML_V_tml::operator()\n");
		printf("t_ML_V_tml::operator():1\n");
                vec_ops[0]->addMul(*xi[0], T(0.f), x, T(1.f));
                printf("t_ML_V_tml::operator():2\n");
                vec_ops[0]->addMul(*bi[0], T(0.f), y, T(1.f));
		recursive_MG(0);
		printf("t_ML_V_tml::operator():3\n");
                vec_ops[0]->addMul(y, T(0.f), *xi[0], T(1.f));
	}
	void	proj_x_all(const VECTOR &x)const
	{
		int levs_num = smoother.size();
		for (int i = 1;i < levs_num;++i) {
			if (i == 1) (*proj[i])(x,*xi[i]); else (*proj[i])(*xi[i-1],*xi[i]);
		}
	}
        VECTOR &get_xi(int ilev)const
        {
		return *xi[ilev];
	}
	~t_ML_V_tml()
	{
		int levs_num = smoother.size();
		for (int i = 0;i < levs_num;++i) {
			if (xi[i] != NULL) delete xi[i];
			if (bi[i] != NULL) delete bi[i];
		}
	}
};

#endif
