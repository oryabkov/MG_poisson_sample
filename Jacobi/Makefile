SCFD_INCLUDE_DIR = ../SCFD/include

#HEADERS=VecOpBase.hpp VecSpace.hpp VecSpaceKernel.hpp
INCLUDE_DIRS=-I./kernels -I./operations
INCLUDE_SCFD=-I$(SCFD_INCLUDE_DIR)
COMPILE_FLAGS=-O3 $(INCLUDE_SCFD) $(CFLAGS)

test_hip.bin: $(HEADERS) $(SOURCE) test.cpp
	hipcc $(COMPILE_FLAGS) $(INCLUDE_SCFD) $(INCLUDE_DIRS) -DPOISSON_SOLVER_HIP $(SOURCE) test.cpp -o test.bin
