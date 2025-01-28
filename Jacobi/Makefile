SCFD_INCLUDE_DIR = ../SCFD/include

#HEADERS=VecOpBase.hpp VecSpace.hpp VecSpaceKernel.hpp
INCLUDE_DIRS=-I./kernels -I./operations
INCLUDE_SCFD=-I$(SCFD_INCLUDE_DIR)
COMPILE_FLAGS=-O3 $(INCLUDE_SCFD) $(CFLAGS)

test_hip.bin: $(HEADERS) $(SOURCE) test.cpp
	hipcc $(COMPILE_FLAGS) $(INCLUDE_SCFD) $(INCLUDE_DIRS) -DHIP_BACKEND $(SOURCE) test.cpp -o test_hip.bin

test_omp.bin: $(HEADERS) $(SOURCE) test.cpp
	g++ $(COMPILE_FLAGS) -fopenmp $(INCLUDE_SCFD) $(INCLUDE_DIRS) -DOMP_BACKEND $(SOURCE) test.cpp -o test_omp.bin

test_cuda.bin: $(HEADERS) $(SOURCE) test.cu
	nvcc $(COMPILE_FLAGS) $(INCLUDE_SCFD) $(INCLUDE_DIRS) -DCUDA_BACKEND $(SOURCE) test.cu -o test_cuda.bin
