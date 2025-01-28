#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

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
