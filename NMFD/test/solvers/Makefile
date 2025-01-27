ifndef CONFIG_FILE
CONFIG_FILE = config.inc
endif

ifeq (,$(wildcard $(CONFIG_FILE)))
$(info config file $(CONFIG_FILE) does not exist.)
$(error Create $(CONFIG_FILE) from example or specify another config via: make <target> CONFIG_FILE=<config_filename> )
endif

-include $(CONFIG_FILE)

PROJECT_ROOT_PATH = ../../..
SCFD_INCLUDE = $(PROJECT_ROOT_PATH)/SCFD/include
NMFD_INCLUDE = $(PROJECT_ROOT_PATH)/NMFD/include
#INCLUDE_ROOT = -I$(PROJECT_ROOT_PATH)/sourse
#INCLUDE_LOCAL = -I$(PROJECT_ROOT_PATH)/sourse/solver
INCLUDE_CONTRIB = -I$(SCFD_INCLUDE) -I$(NMFD_INCLUDE)
HOSTFLAGS = $(TARGET_GCC) -std=c++17
HOSTCOMPILER = g++
#CUDAFLAGS = $(TARGET_NVCC)
#MPICOMPILER = $(MPI_ROOT_PATH)/bin/mpic++
#CUDA = $(CUDA_ROOT_PATH)#/opt/cuda#/opt/cuda_all/cuda_11.2
#SM = $(CUDA_ARCH)
#MPI = $(MPI_ROOT_PATH)
#CUDACOMPILER = ${CUDA}/bin/nvcc
#HYPRELIBRARY = -lHYPRE
#CUDALIBRARIES = -lcudart -lcurand -lcusparse -lcublas
IPROJECT = ${INCLUDE_CONTRIB}
LPROJECT = -ldl



test_gmres.bin: test_gmres.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres.cpp -o test_gmres.bin
test_gmres_mg.bin: test_gmres_mg.cpp
	$(HOSTCOMPILER) $(HOSTFLAGS) $(INCLUDE_ROOT) $(INCLUDE_LOCAL) $(INCLUDE_CONTRIB) test_gmres_mg.cpp -o test_gmres_mg.bin
