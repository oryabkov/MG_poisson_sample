
#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include "t_ML_V_tml.h"
#include "t_evs_MG_tml.h"
//TODO hack with isinf and lnear solvers
bool isinf(float x)
{
	return false;
}
#include "../linal/LinSolvJacobi.h"
#include "../linal/LinSolvBiCGStab.h"
#include "../linal/LinSolvCGS.h"
#include "../linal/LinSolvEmpty.h"

//#define MAX_SZ	64
#define MAX_SZ	256
//#define MAX_SZ	128

struct Grid
{
	int	sz;
	//float	d[MAX_SZ][MAX_SZ];
	float	*d;
	Grid() : sz(-1) {} //uninited state
	float &operator()(int i,int j)
	{
		return d[i*MAX_SZ+j];
	}
	const float &operator()(int i,int j)const
	{
		return d[i*MAX_SZ+j];
	}
};

void	visualize_gmsh(const std::string &fn, Grid &g, int gmsh_view_cnt, int mesh_mul)
{
	FILE	*f = fopen(fn.c_str(), "w");
	float	h = 1.f/g.sz;
        fprintf(f, "View \"g_%d\" { \n", gmsh_view_cnt);
	for (int i1 = 0;i1 < g.sz;++i1)
	for (int i2 = 0;i2 < g.sz;++i2) {
		if (!(i1%mesh_mul==0 && i2%mesh_mul==0)) continue;
		int	di1 = std::min(mesh_mul,g.sz-i1),
			di2 = std::min(mesh_mul,g.sz-i2);
		float	val = 0.f;
		for (int ii1 = i1;ii1 < i1+di1;++ii1)
		for (int ii2 = i2;ii2 < i2+di2;++ii2) {
			val += g(ii1,ii2);
		}
		val /= di1*di2;
		//for (int jj = 0;jj < t_solver::t_var::__size;++jj) vars_view(t_solver::t_idx(i1,i2),jj) = 0.f;
		fprintf(f, "SQ(%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f){ %f, %f, %f, %f };\n",
			(i1-0.5f        )*h, (i2-0.5f        )*h, 0.f,
			(i1+0.5f+di1-1.f)*h, (i2-0.5f        )*h, 0.f,
			(i1+0.5f+di1-1.f)*h, (i2+0.5f+di2-1.f)*h, 0.f,
			(i1-0.5f        )*h, (i2+0.5f+di2-1.f)*h, 0.f,
			val,
			val,
			val,
			val);
	}
	fprintf(f, "}; \n");
	fclose(f);
}

struct GridOps
{
	int	sz;

	GridOps() : sz(0) {}	//assume it is uninited
	GridOps(int _sz) : sz(_sz) {}

	void verify_size(Grid &g)const
	{
		if (g.sz == sz) return;
		g.d = new float[MAX_SZ*MAX_SZ];
		g.sz = sz;
		for (int i = 0;i < sz;++i)
		for (int j = 0;j < sz;++j) {
			g(i,j) = 0.f;
		}
	}
	void addMul(Grid& B, float mulB, const Grid& A, float mulA)const
	{
		for (int i = 0;i < sz;++i)
		for (int j = 0;j < sz;++j) {
			B(i,j) = mulB*B(i,j) + mulA*A(i,j);
		}
	}
	void addMul(Grid& C, float mulC, const Grid& A, float mulA, const Grid& B, float mulB)const
	{
		for (int i = 0;i < sz;++i)
		for (int j = 0;j < sz;++j) {
			C(i,j) = mulC*C(i,j) + mulB*B(i,j) + mulA*A(i,j);
		}
	}
	void addMulScalar(Grid &B, float mulB, float scalar)const
	{
		for (int i = 0;i < sz;++i)
		for (int j = 0;j < sz;++j) {
			B(i,j) = mulB*B(i,j) + scalar;
		}
	}
	float	scalar_prod(const Grid &x, const Grid &y)const
	{
		float	res(0.f);
		float	h = 1.f/sz;
		for (int i = 0;i < sz;++i)
		for (int j = 0;j < sz;++j) {
			res += x(i,j)*y(i,j)*h*h;
		}
		return res;
	}
	float	norm(const Grid &x)const
	{
		return std::sqrt( scalar_prod(x,x) );
	}
};

//OP concept:
//void operator()(const VECTOR& x,VECTOR& f)const

struct LaplaceOp
{
	void operator()(const Grid &x,Grid &f)const
	{
		assert(x.sz == f.sz);

		int     sz = x.sz;
		float	h = 1.f/sz;
		for (int i = 0;i < sz;++i)
		for (int j = 0;j < sz;++j) {
			//account 4 borders
			float	x_c = x(i,j),
				x_l = (i-1>=0 ? x(i-1,j): -x(i,j)),
				x_r = (i+1<sz ? x(i+1,j): -x(i,j)),
				x_d = (j-1>=0 ? x(i,j-1): -x(i,j)),
				x_u = (j+1<sz ? x(i,j+1): -x(i,j));
			f(i,j) = (x_l - 2.f*x_c + x_r)/(h*h)
			       + (x_d - 2.f*x_c + x_u)/(h*h);
		}
	}

	struct t_diag_prec_op
	{
		void operator()(Grid &f)const
		{
			int     sz = f.sz;
			float h = 1.f/sz;
			for (int i = 0;i < sz;++i)
			for (int j = 0;j < sz;++j) {
				f(i,j) /= -4.f/(h*h);
			}
		}
	} diag_prec_op;

	const t_diag_prec_op	&get_diag_prec_op()const
	{
		return diag_prec_op;
	}
};

struct AdvectOp
{
	bool	expl;
	float	dt;

	void operator()(const Grid &x,Grid &f)const
	{
		assert(x.sz == f.sz);

		int     sz = x.sz;
		float	h = 1.f/sz,
			//dt = 1.f/MAX_SZ,
			//dt = 3.f/MAX_SZ,
			//dt = 6.f/MAX_SZ,
			a1 = 2.f,
			a2 = 1.f;
		for (int i = 0;i < sz;++i)
		for (int j = 0;j < sz;++j) {
			//account 4 borders
			float	x_c = x(i,j),
				/*x_l = (i-1>=0 ? x.d[i-1][j]: -x.d[i][j]),
				x_r = (i+1<sz ? x.d[i+1][j]: -x.d[i][j]),
				x_d = (j-1>=0 ? x.d[i][j-1]: -x.d[i][j]),
				x_u = (j+1<sz ? x.d[i][j+1]: -x.d[i][j]);*/
				x_l = (i-1>=0 ? x(i-1,j): -0.f),
				x_r = (i+1<sz ? x(i+1,j): -0.f),
				x_d = (j-1>=0 ? x(i,j-1): -0.f),
				x_u = (j+1<sz ? x(i,j+1): -0.f);
			//f.d[i][j] = (x_l - x_c)/(h)
			//	  + (x_d - x_c)/(h);
			//f.d[i][j] = (x_r - x_c)/(h);
			if (expl)
				//f.d[i][j] = dt*a1*(x_r - x_c)/(h) + dt*a2*(x_u - x_c)/(h);
				f(i,j) = -dt*(a1*(x_c-x_l)/(h) + a2*(x_c-x_d)/(h));
			else
				//f.d[i][j] = x_c - a1*dt*(x_r - x_c)/(h) - a2*dt*(x_u - x_c)/(h);
				f(i,j) = x_c + dt*(a1*(x_c-x_l)/(h) + a2*(x_c-x_d)/(h));
		}
	}

	//NOTE it is only for implicit
	struct t_diag_prec_op
	{
		float	dt;
		void operator()(Grid &f)const
		{
			int     sz = f.sz;
			float	h = 1.f/sz,
				//dt = 1.f/MAX_SZ,
				//dt = 3.f/MAX_SZ,
				a1 = 2.f,
				a2 = 1.f;
			for (int i = 0;i < sz;++i)
			for (int j = 0;j < sz;++j) {
				//f.d[i][j] /= -2.f/(h);
				//f.d[i][j] = x_c + dt*(a1*(x_c-x_l)/(h) + a2*(x_c-x_d)/(h));
				f(i,j) /= 1.f + a1*dt/h + a2*dt/h;
			}
		}
	};
	mutable t_diag_prec_op diag_prec_op;

	const t_diag_prec_op	&get_diag_prec_op()const
	{
                assert(!expl);
		diag_prec_op.dt = dt;
		return diag_prec_op;
	}
};

typedef	AdvectOp	SysOperator;

struct SmootherOp
{
	typedef Grid								t_grid;
	typedef GridOps								t_grid_ops;
	typedef SysOperator							t_op;
	typedef t_op::t_diag_prec_op						t_diag_prec_op;
	//typedef LinSolvJacobi<t_op,t_diag_prec_op,t_grid_ops,t_grid,float>	t_solver_jacobi;
	typedef LinSolvBiCGStab<t_op,t_diag_prec_op,t_grid_ops,t_grid,float>	t_solver_jacobi;

	t_grid_ops		*grid_ops;
	t_op			*op;
	int			solver_type;
        t_solver_jacobi		jacobi;

	void init()
	{
		//jacobi.abs_tol = 1e-16;
		jacobi.abs_tol = 1e-8;
        	jacobi.min_iters_num = 2;
	        jacobi.max_iters_num = 100;

	        //jacobi.min_iters_num = 5;
	        //jacobi.max_iters_num = 5;
	}
	void operator()(Grid &x,const Grid &b)const
	{
		jacobi.solve(*grid_ops, *op, op->get_diag_prec_op(), x, b);
	}
};

struct ProjOp
{
	void operator()(const Grid &x,Grid &f)const
	{
		assert(f.sz*2 == x.sz);
		for (int i = 0;i < f.sz;++i)
		for (int j = 0;j < f.sz;++j) {
			f(i,j) = 0.25f*(x(i*2,j*2)+x(i*2+1,j*2)+x(i*2,j*2+1)+x(i*2+1,j*2+1));
			//f.d[i][j] = x.d[i*2+1][j*2+1];
			//f.d[i][j] = x.d[i*2][j*2];
		}
	}
};

struct ProlOp
{
	void operator()(const Grid &x,Grid &f)const
	{
		assert(f.sz/2 == x.sz);
		for (int i = 0;i < f.sz;++i)
		for (int j = 0;j < f.sz;++j) {
			f(i,j) = x(i/2,j/2);
		}
	}
};

struct MLOp
{
	typedef t_evs_MG_tml<ProjOp,ProlOp,SmootherOp,GridOps,Grid,float>		t_ML;
	//typedef t_ML_V_tml<ProjOp,ProlOp,SmootherOp,SysOperator,GridOps,Grid,float>	t_ML_V;
	//typedef t_ML_V_tml<ProjOp,ProlOp,SmootherOp,SysOperator,GridOps,Grid,float>	t_ML;

	std::vector<GridOps>	grid_opses;
	SysOperator		*sys_op;
	std::vector<SmootherOp> smoothers;
	ProjOp			proj_op;
	ProlOp			prol_op;
	t_ML			ml;

	void	init(SysOperator *_sys_op,int levs_num,int sz0)
	{
		if (levs_num == 0) return;
		
		sys_op = _sys_op;

		printf("MLOp::init\n");
		printf("%d\n",levs_num);
		grid_opses.resize(levs_num);
		smoothers.resize(levs_num);
		printf("MLOp::init:ml.init0()\n");
		ml.init0(levs_num);
		int _sz = sz0;
		for (int i = 0;i < levs_num;++i) {
			smoothers[i].grid_ops = &(grid_opses[i]);
                        //smoothers[i].op = &sys_op;
                        smoothers[i].op = sys_op;
                        smoothers[i].init();

                        //test
			if (i > 0) {
				//smoothers[i].jacobi.abs_tol = 1e-10;
				//smoothers[i].jacobi.max_iters_num = 200;
			}
                        //test

			grid_opses[i].sz = _sz;
			_sz /= 2;
			ml.set_level(i, &(grid_opses[i]), &proj_op, &prol_op, &(smoothers[i]));
			//ml.set_level(i, &(grid_opses[i]), &proj_op, &prol_op, &(smoothers[i]), &sys_op);
		}
		//smoothers[0].jacobi.min_iters_num = 1;
		//smoothers[0].jacobi.max_iters_num = 1;
		printf("MLOp::init:ml.init()\n");
		ml.init();
	}

	void	operator()(Grid &x)const
	{
		if (grid_opses.size() == 0) return;

		ml(x,x);
	}
};

//typedef LinSolvJacobi<SysOperator,SysOperator::t_diag_prec_op,GridOps,Grid,float>	t_solver;
//typedef LinSolvBiCGStab<SysOperator,SysOperator::t_diag_prec_op,GridOps,Grid,float>	t_solver;
//typedef LinSolvCGS<SysOperator,SysOperator::t_diag_prec_op,GridOps,Grid,float>		t_solver;
typedef LinSolvEmpty<SysOperator,MLOp,GridOps,Grid,float>	t_solver;
//typedef LinSolvJacobi<SysOperator,MLOp,GridOps,Grid,float>				t_solver;
//typedef LinSolvBiCGStab<SysOperator,MLOp,GridOps,Grid,float>				t_solver;
//typedef LinSolvCGS<SysOperator,MLOp,GridOps,Grid,float>					t_solver;


int main(int argc, char **args)
{
	if (argc < 2) {
		printf("usage: ML_test grid_size ");
		return 1;
	}
	int levs_num = atoi(args[1]);
	int solver_type = 1;
	int init_cond_type = 2;

	int sz = MAX_SZ;
	GridOps	grid_ops(sz);
	Grid	x,b,x_new;
	grid_ops.verify_size(x);
	grid_ops.verify_size(x_new);
	grid_ops.verify_size(b);

	for (int i = 0;i < sz;++i)
	for (int j = 0;j < sz;++j) {
		//b.d[i][j] = 1.f;
		if (init_cond_type == 1) {
			x(i,j) = ((i-sz/4)*(i-sz/4)+(j-sz/4)*(j-sz/4) < (sz/6)*(sz/6)?1.f:0.f);
		} else if (init_cond_type == 2) {
			float	r_sz = float((i-sz/4)*(i-sz/4)+(j-sz/4)*(j-sz/4))/(sz*sz),
				sigma_sq = 0.01f;
                        x(i,j) = exp(-0.5f*r_sz/(sigma_sq));
			//x(i,j) = ((i-sz/4)*(i-sz/4)+(j-sz/4)*(j-sz/4) < (sz/6)*(sz/6)?1.f:0.f);
		}
		//if ((i < sz/2)&&(j < sz/2)) b.d[i][j] = 1.f; else b.d[i][j] = 0.f;
	}

	SysOperator	sys_op;
	t_solver	solver;
	float		//cfl = 5.f,
			cfl,
			t = 0.f,
			T = 0.5/2.;
			//T = -1.f;

	if (solver_type == 1) cfl = 0.9f; else cfl = 3.f;

	sys_op.dt = cfl*0.5f*(1.f/sz)/2.f;

	if (solver_type == 1) {
		sys_op.expl = true;
		while (t < T) {
                        sys_op(x, b);
                        //NOTE dt is already in operator
                        grid_ops.addMul(x, 1.f, b, 1.f);
			t += sys_op.dt;
		}
	} else if (solver_type == 2) {
		sys_op.expl = false;
		/*solver.abs_tol = 1e-7;
        	solver.min_iters_num = 2;
       		solver.max_iters_num = 300;
	        solver.solve(grid_ops, sys_op, sys_op.get_diag_prec_op(), x, b);*/
	} else if (solver_type == 3) {
		sys_op.expl = false;
		MLOp		ml_prec;
		ml_prec.init(&sys_op,levs_num,sz);
		while (t < T) {
                        //sys_op(x, b);
                        //NOTE dt is already in operator

                        grid_ops.addMul(b, 0.f, x, 1.f);
                        grid_ops.addMul(x_new, 0.f, x, 1.f);
                        solver.solve(grid_ops, sys_op, ml_prec, x_new, b);
                        //grid_ops.addMul(x, 0.f, x_new, 1.f);
                        //make post step
                        sys_op.expl = true;
                        sys_op(x_new, b);
                        sys_op.expl = false;
                        grid_ops.addMul(x, 1.f, b, 1.f);
			//solver.solve(grid_ops, sys_op, sys_op.get_diag_prec_op(), x, b);

                        //grid_ops.addMul(x, 1.f, b, 1.f);
			t += sys_op.dt;
		}

	}

	visualize_gmsh("solver_test.pos", x, 0, 1);

	return 0;
}