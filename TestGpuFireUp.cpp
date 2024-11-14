#include <iostream>
#include <vector>

#include "catch.hpp"

#include "pdlp/CupdlpWrapper.h"

// #include <cublas_v2.h>         // cublas
// #include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
// #include <cusparse.h>          // cusparseSpMV

#include "pdlp/cupdlp/cuda/cupdlp_cuda_kernels.cuh"
#include "pdlp/cupdlp/cuda/cupdlp_cudalinalg.cuh"

TEST_CASE("start-nvidia", "[test_gpu_fire_up]") {

  // TODO add code here

  std::cout << "GPU FIRE UP TEST" << std::endl;

  // CUPDLPwork* w = cupdlp_NULL;
  // cupdlp_init_work(w, 1);

  cusparseHandle_t cusparsehandle;
  cublasHandle_t cublashandle;

  cupdlp_float cuda_prepare_time = getTimeStamp();
  // CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
    cusparseStatus_t status_cusparse = cusparseCreate(&cusparsehandle);                                      
    if (status_cusparse != CUSPARSE_STATUS_SUCCESS) {                               
      printf("CUSPARSE API failed at line %d of %s with error: %s (%d)\n", 
             __LINE__, __FILE__, 
             cusparseGetErrorString(status_cusparse), status_cusparse);  
    }

  // CHECK_CUBLAS(cublasCreate(&w->cublashandle));
    cublasStatus_t status_cublas = cublasCreate(&cublashandle);
    if (status_cublas != CUBLAS_STATUS_SUCCESS) {                               
      printf("CUBLAS API failed at line %d of %s with error: %s (%d)\n", 
             __LINE__, __FILE__,
             cublasGetStatusString(status_cublas), status_cublas);  
    }
  cuda_prepare_time = getTimeStamp() - cuda_prepare_time;

  std::cout << "cuda prepare time: " << cuda_prepare_time << std::endl;

  REQUIRE(true);

  // check cupdlp_copy_vec
  // cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);
  const int n = 3;
  double xs[] = {1, 1, 1};
  double ys[] = {1, 2, 3};

  const double* x = &xs[0];
  const double* y = &ys[0];

  double* res;

  // double x[] = [1,2,3];
  cublasStatus_t status =  cublasDdot(cublashandle, n, x, 1, y, 1, res);

  if (status != CUBLAS_STATUS_SUCCESS) {                               
      printf("CUBLAS API failed at line %d of %s with error: %s (%d)\n", 
             __LINE__, __FILE__, cublasGetStatusString(status), status); 
  }
}
