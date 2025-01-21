SCFD_INCLUDE_DIR = ../SCFD/include

HEADERS=VecOpBase.hpp VecOp.hpp VecOpKernels.hpp

INCLUDE_SCFD=-I$(SCFD_INCLUDE_DIR)
COMPILE_FLAGS=-O3 $(INCLUDE_SCFD) $(CFLAGS)

poisson_jacobi_func_hip.bin: $(HEADERS) $(SOURCE) vec_op_test.cpp
	hipcc $(COMPILE_FLAGS) $(INCLUDE_SCFD) -DPOISSON_SOLVER_HIP $(SOURCE) vec_op_test.cpp -o vec_op_test.bin
