PROJECT_ROOT_PATH = ..

HEADERS=./headers
SCFD_INCLUDE = $(PROJECT_ROOT_PATH)/SCFD/include
NMFD_INCLUDE = $(PROJECT_ROOT_PATH)/NMFD/include

INCLUDE   = -I$(SCFD_INCLUDE) -I$(NMFD_INCLUDE) -I$(HEADERS)
HOSTFLAGS = -O3 $(CFLAGS) -std=c++17

######################## VECTOR SPACE TESTS ##################################
test_vec_space_hip.bin: test_vec_space.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND test_vec_space.cpp -o test_vec_space_hip.bin



######################## JACOBI SOLVER TESTS #################################

test_jacobi_cpu.bin: test_jacobi.cpp
	g++ $(HOSTFLAGS) $(INCLUDE) -DCPU_BACKEND test_jacobi.cpp -o test_jacobi_cpu.bin

test_jacobi_hip.bin: test_jacobi.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND test_jacobi.cpp -o test_jacobi_hip.bin

test_jacobi_omp.bin: test_jacobi.cpp
	g++ -fopenmp $(HOSTFLAGS) $(INCLUDE) -DOMP_BACKEND test_jacobi.cpp -o test_jacobi_omp.bin

test_jacobi_cuda.bin: test_jacobi.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE) -DCUDA_BACKEND -x cu test_jacobi.cpp -o test_jacobi_cuda.bin

test_jacobi_sycl.bin: test_jacobi.cpp
	icpx $(HOSTFLAGS) $(INCLUDE) -DSYCL_BACKEND test_jacobi.cpp -o test_jacobi_sycl.bin



######################### MULTIGRID TESTS ####################################
test_mg_cpu.bin: test_mg.cpp
	g++ $(HOSTFLAGS) $(INCLUDE) -DCPU_BACKEND test_mg.cpp -o test_mg_cpu.bin

test_mg_hip.bin: test_mg.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND test_mg.cpp -o test_mg_hip.bin

test_mg_omp.bin: test_mg.cpp
	g++ -fopenmp $(HOSTFLAGS) $(INCLUDE) -DOMP_BACKEND test_mg.cpp -o test_mg_omp.bin

test_mg_cuda.bin: test_mg.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE) -DCUDA_BACKEND -x cu test_mg.cpp -o test_mg_cuda.bin

test_mg_sycl.bin: test_mg.cpp
	icpx $(HOSTFLAGS) $(INCLUDE) -DSYCL_BACKEND test_mg.cpp -o test_mg_sycl.bin
