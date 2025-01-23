#ifndef __PRECONDITIONER_HPP__
#define __PRECONDITIONER_HPP__


namespace preconditioners
{

template <class VectorSpace> 
class jacobi_preconditioner 
{
public:
    using vector_space_t    = VectorSpace; // defines Vector Space working in 
    using vector_space_ptr  = std::shared_ptr<VectorSpace>;
    
    using scalar_t          = typename VectorSpace::scalar_t;
    using ordinal_t         = typename VectorSpace::ordinal_t;
    
    using vector_t          = typename VectorSpace::vector_t;
    using idx_nd_t          = typename VectorSpace::idx_nd_t;

private:
    vector_space_ptr  vspace;
    idx_nd_t          range;

public:
    jacobi_preconditioner(vector_space_ptr vec_space) : 
        vspace(vec_space), range(vspace->get_range()){}  

public:
    void apply(vector_t &v) const
    {
        vspace->scale(1. / (2. * vector_space_t::dim), v);
    };
};

}// namespace operators

#endif
