PROJECT_ROOT_PATH = ..

HEADERS=./headers
SCFD_INCLUDE = $(PROJECT_ROOT_PATH)/SCFD/include
NMFD_INCLUDE = $(PROJECT_ROOT_PATH)/NMFD/include

INCLUDE   = -I$(SCFD_INCLUDE) -I$(NMFD_INCLUDE) -I$(HEADERS)
HOSTFLAGS = -O3 $(CFLAGS) -std=c++17

######################## JACOBI SOLVER TESTS #################################


test_jacobi_cpu_d.bin: test_jacobi.cpp
	g++ $(HOSTFLAGS) $(INCLUDE) -DCPU_BACKEND -DUSE_DOUBLE_PRECISION test_jacobi.cpp -o test_jacobi_cpu_d.bin

test_jacobi_hip_d.bin: test_jacobi.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND -DUSE_DOUBLE_PRECISION test_jacobi.cpp -o test_jacobi_hip_d.bin

test_jacobi_omp_d.bin: test_jacobi.cpp
	g++ -fopenmp $(HOSTFLAGS) $(INCLUDE) -DOMP_BACKEND -DUSE_DOUBLE_PRECISION test_jacobi.cpp -o test_jacobi_omp_d.bin

test_jacobi_cuda_d.bin: test_jacobi.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE) -DCUDA_BACKEND -DUSE_DOUBLE_PRECISION -x cu test_jacobi.cpp -o test_jacobi_cuda_d.bin

test_jacobi_sycl_d.bin: test_jacobi.cpp
	icpx $(HOSTFLAGS) $(INCLUDE) -DSYCL_BACKEND -DUSE_DOUBLE_PRECISION test_jacobi.cpp -o test_jacobi_sycl_d.bin



test_jacobi_cpu_f.bin: test_jacobi.cpp
	g++ $(HOSTFLAGS) $(INCLUDE) -DCPU_BACKEND test_jacobi.cpp -o test_jacobi_cpu_f.bin

test_jacobi_hip_f.bin: test_jacobi.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND test_jacobi.cpp -o test_jacobi_hip_f.bin

test_jacobi_omp_f.bin: test_jacobi.cpp
	g++ -fopenmp $(HOSTFLAGS) $(INCLUDE) -DOMP_BACKEND test_jacobi.cpp -o test_jacobi_omp_f.bin

test_jacobi_cuda_f.bin: test_jacobi.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE) -DCUDA_BACKEND -x cu test_jacobi.cpp -o test_jacobi_cuda_f.bin

test_jacobi_sycl_f.bin: test_jacobi.cpp
	icpx $(HOSTFLAGS) $(INCLUDE) -DSYCL_BACKEND test_jacobi.cpp -o test_jacobi_sycl_f.bin


######################### MULTIGRID TESTS ####################################

test_mg_cpu_d.bin: test_mg.cpp
	g++ $(HOSTFLAGS) $(INCLUDE) -DCPU_BACKEND -DUSE_DOUBLE_PRECISION test_mg.cpp -o test_mg_cpu_d.bin

test_mg_hip_d.bin: test_mg.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND -DUSE_DOUBLE_PRECISION test_mg.cpp -o test_mg_hip_d.bin

test_mg_omp_d.bin: test_mg.cpp
	g++ -fopenmp $(HOSTFLAGS) $(INCLUDE) -DOMP_BACKEND -DUSE_DOUBLE_PRECISION test_mg.cpp -o test_mg_omp_d.bin

test_mg_cuda_d.bin: test_mg.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE) -DCUDA_BACKEND -DUSE_DOUBLE_PRECISION -x cu test_mg.cpp -o test_mg_cuda_d.bin

test_mg_sycl_d.bin: test_mg.cpp
	icpx $(HOSTFLAGS) $(INCLUDE) -DSYCL_BACKEND -DUSE_DOUBLE_PRECISION test_mg.cpp -o test_mg_sycl_d.bin




test_mg_cpu_f.bin: test_mg.cpp
	g++ $(HOSTFLAGS) $(INCLUDE) -DCPU_BACKEND test_mg.cpp -o test_mg_cpu_f.bin

test_mg_hip_f.bin: test_mg.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND test_mg.cpp -o test_mg_hip_f.bin

test_mg_omp_f.bin: test_mg.cpp
	g++ -fopenmp $(HOSTFLAGS) $(INCLUDE) -DOMP_BACKEND test_mg.cpp -o test_mg_omp_f.bin

test_mg_cuda_f.bin: test_mg.cpp
	nvcc $(HOSTFLAGS) $(INCLUDE) -DCUDA_BACKEND -x cu test_mg.cpp -o test_mg_cuda_f.bin

test_mg_sycl_f.bin: test_mg.cpp
	icpx $(HOSTFLAGS) $(INCLUDE) -DSYCL_BACKEND test_mg.cpp -o test_mg_sycl_f.bin
