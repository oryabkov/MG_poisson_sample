PROJECT_ROOT_PATH = ..

SCFD_INCLUDE = $(PROJECT_ROOT_PATH)/SCFD/include
NMFD_INCLUDE = $(PROJECT_ROOT_PATH)/NMFD/include

INCLUDE   = -I$(SCFD_INCLUDE) -I$(NMFD_INCLUDE)
HOSTFLAGS = -O3 $(CFLAGS)

######################## VECTOR SPACE TESTS ##################################
test_vec_space_hip.bin: test_vec_space.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND test_vec_space.cpp -o test_vec_space_hip.bin


######################## JACOBI SOLVER TESTS #################################
test_jacobi_hip.bin: test_jacobi.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND test_jacobi.cpp -o test_jacobi_hip.bin


######################### MULTIGRID TESTS ####################################
test_mg_hip.bin: test_mg.cpp
	hipcc $(HOSTFLAGS) $(INCLUDE) -DHIP_BACKEND test_mg.cpp -o test_mg_hip.bin
