# MG_poisson_sample
Geometric Multigrid Poisson sample using SCFD library to verify SYCL and HIP abilities (PAVT2025 code)

# Basic interfaces

## VectorOperations + VectorSpace

TODO

VectorOperations is basically BLAS1 interface.

```
/// Nested types
using scalar_type = ...;
using vector_type = ...;

/// Methods

/// y<-x
void assign(const vector_type& x, vector_type& y) const

void assign_scalar(const scalar_type val, vector_type& x)const 
/// x<-mul_x*x+scalar
void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)

void assign_random(vector_type& x)const 
/// x<-x*scale
void scale(const scalar_type scale, vector_type& x)
/// returns sum of all elements
[[nodiscard]] scalar_type sum(const vector_type& x)const 
/// returns sum of all elements absolute values
[[nodiscard]] scalar_type asum(const vector_type &x)const
/// returns either all elements are valid numbers (not nans or infs)
[[nodiscard]] bool is_valid_number(const vector_type &x)const
/// returns inner product of vectors (sum of elements products)
[[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
/// returns some weighted inner product of vectors (problem dependent, for example L2 integral norm)
[[nodiscard]] scalar_type scalar_prod_l2(const vector_type &x, const vector_type &y)const
/// Returns ANY norm that fits problem convergence
[[nodiscard]] scalar_type norm(const vector_type &x) const
/// norm(x)^2
[[nodiscard]] scalar_type norm_sq(const vector_type &x) const
/// scalar_prod(x,x)
[[nodiscard]] scalar_type norm2_sq(const vector_type &x) const
/// sqrt(scalar_prod(x,x))
[[nodiscard]] scalar_type norm2(const vector_type &x) const
/// scalar_prod_l2(x,x)
[[nodiscard]] scalar_type norm_l2_sq(const vector_type &x) const
/// sqrt(scalar_prod_l2(x,x))
[[nodiscard]] scalar_type norm_l2(const vector_type &x) const
/// May be ommited for now
[[nodiscard]] vector_type at(multivector_type& x, ordinal_type m, ordinal_type k_)
/// y<-x*mul_x+y*mul_y
void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
/// z<-x*mul_x+y*mul_y+z*mul_z
void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, const scalar_type mul_z, vector_type& z) const
/// y<-x*mul_x
void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const
/// z<-x*mul_x+y*mul_y
void assign_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y, vector_type& z) const
```

VectorSpace has additional methods that allocate/deallocate vectors meaning that VectorSpace 'knows' 'size' of the vector space:

```
void init_vector(vector_type &v);
void start_use_vector(vector_type &v);
void stop_use_vector(vector_type &v);
void free_vector(vector_type &v);
```

## Operator 

```
class OperatorName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void apply(const vector_type &in, vector_type &out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

In Operator there is a garantee that initial value of out doesnot affects result.

## InplaceOperator 

```
class InplaceOperatorName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void apply(vector_type &in_out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

in_out is both input and output.

## Preconditioner

Preconditioner is restriction of Operator and InplaceOperator. Plus it has set_operator method for inicialization/reinitialization.

```
class PreconditionerName
{
public:
    using scalar_type = ...;
    using vector_type = ...;
    using vector_space_type = ...; /// VectorSpace
public:
    void set_operator(std::shared_ptr<const operator_type> operator);
    void apply(const vector_type &in, vector_type &out);
    void apply(vector_type &in_out);
    std::shared_ptr<vector_space_type> get_im_space() const;
    std::shared_ptr<vector_space_type> get_dom_space() const;
};
```

## Solver 

```
class SolverName
{
public:
    using vector_type = ...;
    using operator_type = ...;
public:
    void set_operator(std::shared_ptr<const operator_type> operator);
    bool solve(const vector_type &rhs, vector_type &res);
};
```

In Solver there is no strict garantee that initial value of res doesnot affects result. For example Solver can use it initial guess.

## Coarsening

Creates Restrictor,Prolongator and SystemOperator for next level based on previous SystemOperator.

```
class CoarseningName
{
public:
    using operator_type = ...;
    using restrictor_type = ...;
    using prolongator_type = ...;
public:
    std::tuple<std::shared_ptr<restrictor_type>,std::shared_ptr<prolongator_type>> next_level(const operator_type &operator);
    std::shared_ptr<operator_type> coarse_operator(const operator_type &operator, const restrictor_type &restrictor, const prolongator_type &prolongator);
};
```

This structure is almost like in amgcl.
Rationale: why do we need two methods? because of possible rebuild procedure where only operators are updated.

TODO

## HierarchicAlgorithm

Used for automized nested algorithms construction.

TODO

# MG class template

mg is Preconditioner.

```
/// SystemOperator, Restrictor, Prolongator are Operator.
/// Smoother, CoarseSolver are Preconditioner
/// Coarsening is Coarsening
/// All vector_space_type, vector_type, scalar_type are the same
template
<
    class SystemOperator,
    class Restrictor,
    class Prolongator,
    class Smoother,
    class CoarseSolver,
    class Coarsening
>
class mg
{
    //TODO
};
```


