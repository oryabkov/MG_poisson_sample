#ifndef __T_EVS_MG_TML_H__
#define __T_EVS_MG_TML_H__

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
template<class PROJ_OP,class PROL_OP,class SMOOTHER_OP,class VECTOR_OPS,class VECTOR,class T>
class t_evs_MG_tml
{
	//proj[i]'s result lies on i-th level
	std::vector<const PROJ_OP*>		proj;
	//prol[i]'s result lies on i-th level
	std::vector<const PROL_OP*>		prol;
	std::vector<const SMOOTHER_OP*>		smoother;
	std::vector<int>			hierarchy;
	std::vector<const VECTOR_OPS*>		vec_ops;
	mutable std::vector<VECTOR*>		bi,xi;

	t_evs_MG_tml	&operator=(const t_evs_MG_tml& x)
	{
	}
	t_evs_MG_tml(const t_evs_MG_tml& x)
	{
	}
public:
	t_evs_MG_tml()
	{
	}
	void	init0(int levs_num/*, int hierarchy_len*/)
	{
		proj.resize(levs_num);
		prol.resize(levs_num);
		bi.resize(levs_num);
		xi.resize(levs_num);
		smoother.resize(levs_num);
		vec_ops.resize(levs_num);
		for (int i = 0;i < levs_num;++i) {
			vec_ops[i] = NULL;
			proj[i] = NULL;
			prol[i] = NULL;
			smoother[i] = NULL;
			bi[i] = xi[i] = NULL;
		}
                //hierarchy.resize(hierarchy_len);
                hierarchy.resize(levs_num);
                for (int i = 0;i < hierarchy.size();++i) {
			hierarchy[i] = levs_num-i-1;
		}
	}
	void	set_level(int ilev, const VECTOR_OPS *_vec_ops, const PROJ_OP *_proj, const PROL_OP *_prol, const SMOOTHER_OP *_smoother)
	{
		proj[ilev] = _proj;
		prol[ilev] = _prol;
		smoother[ilev] = _smoother;
		vec_ops[ilev] = _vec_ops;
	}
	void	init()
	{
		int levs_num = smoother.size();
		//TODO it's a little bit overhead with zero-level copy
		for (int i = 0;i < levs_num;++i) {
			xi[i] = new VECTOR();
			bi[i] = new VECTOR();
			vec_ops[i]->verify_size(*xi[i]);
			vec_ops[i]->verify_size(*bi[i]);
		}
	}

	//apply ML-operator to x vector and returns result in y vector
	//NOTE x is usually RHS(b) and in y approximate solution will be returned
	void	operator()(const VECTOR &x, VECTOR &y)const
	{
		//TEST
		/*printf("t_evs_MG_tml::operator()\n");
		vec_ops[0]->addMul(*bi[0], T(0.f), x, T(1.f));
		vec_ops[0]->addMul(y, T(0.f), x, T(1.f));
		(*smoother[0])(y, *bi[0]);
		return;*/
		//END TEST

		printf("t_evs_MG_tml::operator()\n");
		int levs_num = smoother.size();
		vec_ops[0]->addMul(*bi[0], T(0.f), x, T(1.f));  //TODO zero-level overhead (see above TODO)
		for (int i = 1;i < levs_num;++i) {
			printf("t_evs_MG_tml::operator(): proj x to level %d\n", i);
			if (i == 1) (*proj[i])(x,*bi[i]); else (*proj[i])(*bi[i-1],*bi[i]);
		}
		int	ilev_prev;
		for (int i = 0;i < hierarchy.size();++i) {
			int ilev = hierarchy[i];
			printf("t_evs_MG_tml::operator(): do hierarchy %d on level %d\n", i, ilev);
			//assert(ilev > 0);
			assert(ilev >= 0);
			if (i == 0) {
				//vec_ops[ilev]->addMulScalar(*xi[ilev], T(0.f), T(0.f));
				vec_ops[ilev]->addMul(*xi[ilev], T(0.f), *bi[ilev], T(1.f));
			} else {
				assert(abs(ilev_prev-ilev) == 1);
				if (ilev_prev < ilev) (*proj[ilev])(*xi[ilev_prev],*xi[ilev]); else (*prol[ilev])(*xi[ilev_prev],*xi[ilev]);
			}

			printf("t_evs_MG_tml::operator(): call smoother on level %d\n", ilev);
			(*smoother[ilev])(*xi[ilev], *bi[ilev]);
			ilev_prev = ilev;
		}
		//(*prol[0])(*xi[1], y);
		vec_ops[0]->addMul(y, T(0.f), *xi[0], T(1.f));  //TODO zero-level overhead (see above TODO)
	}
	void	proj_x_all(const VECTOR &x)const
	{
		//return;//TEST!!
		int levs_num = smoother.size();
		for (int i = 1;i < levs_num;++i) {
			if (i == 1) (*proj[i])(x,*xi[i]); else (*proj[i])(*xi[i-1],*xi[i]);
		}
	}
        VECTOR &get_xi(int ilev)const
        {
		return *xi[ilev];
	}
	~t_evs_MG_tml()
	{
		int levs_num = smoother.size();
		for (int i = 0;i < levs_num;++i) {
			if (xi[i] != NULL) delete xi[i];
			if (bi[i] != NULL) delete bi[i];
		}
	}
};

#endif
