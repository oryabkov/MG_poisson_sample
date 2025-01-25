#ifndef __PRECONDITIONER_HPP__
#define __PRECONDITIONER_HPP__


namespace preconditioners
{

template <class VectorSpace> 
class jacobi_preconditioner 
{
public:
    using vector_space_type = VectorSpace; // defines Vector Space working in 
    using vector_space_ptr  = std::shared_ptr<VectorSpace>;
    
    using scalar_type       = typename VectorSpace::scalar_type;
    using ordinal_type      = typename VectorSpace::ordinal_type;
    
    using vector_type       = typename VectorSpace::vector_type;
    using idx_nd_type       = typename VectorSpace::idx_nd_type;

private:
    vector_space_ptr  vspace;
    idx_nd_type       range;

public:
    jacobi_preconditioner(vector_space_ptr vec_space) : 
        vspace(vec_space), range(vspace->get_range()){}  

public:
    void apply(vector_type &v) const
    {
        vspace->scale(1. / (2. * vector_space_type::dim), v);
    };
};

}// namespace operators

#endif
