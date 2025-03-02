#ifndef __INCLUDE_HPP__
#define __INCLUDE_HPP__

static constexpr int current_dim = 3;

template <int Dim>
struct boundary_cond
{
    // -1 for dirichlet
    // +1 for neumann
    using conditions = int;

    conditions left [Dim];
    conditions right[Dim];
};

#endif
