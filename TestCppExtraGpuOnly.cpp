#include <iostream>
#include <vector>

#include "catch.hpp"

#include <cublas_v2.h>         // cublas
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>          // cusparseSpMV

TEST_CASE("start-nvidia", "[test_gpu_fire_up_only]") {
  // TODO add code here

  std::cout << "Test extra GPU only." << std::endl;

  REQUIRE(true);

  cusparseHandle_t cusparsehandle;
  cublasHandle_t cublashandle;

  // double cuda_prepare_time = getTimeStamp();
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
  // cuda_prepare_time = getTimeStamp() - cuda_prepare_time;

  // std::cout << "cuda prepare time: " << cuda_prepare_time << std::endl;

  REQUIRE(true);

  // check cupdlp_copy_vec
  // cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);
  const int n = 3;
  double xs[] = {1, 1, 1};
  double ys[] = {1, 2, 3};

  double* x = &xs[0];
  double* y = &ys[0];

  double* devx;
  double* devy;
  double res;

  cudaError_t cudaErr = cudaMalloc ((void**)&devx, n*sizeof(*devx));
  if (cudaErr != cudaSuccess) 
      printf ("cudaMalloc x allocation failed");

  cublasStatus_t stat = cublasSetVector(n, sizeof(*x), x, 1, devx, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) 
        printf ("setVector x failed");

  cudaErr = cudaMalloc ((void**)&devy, n*sizeof(*devy));
  if (cudaErr != cudaSuccess) 
      printf ("cudaMalloc y allocation failed");

  stat = cublasSetVector(n, sizeof(*y), y, 1, devy, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) 
        printf ("setVector y failed");

  cublasStatus_t status =  cublasDdot(cublashandle, n, devx, 1, devy, 1, &res);

  if (status != CUBLAS_STATUS_SUCCESS) {                               
      printf("CUBLAS API failed at line %d of %s with error: %s (%d)\n", 
             __LINE__, __FILE__, cublasGetStatusString(status), status); 
  }

  std::cout << "Ddot = " << res << std::endl;
  std::cout << std::endl;

  REQUIRE(res == 6);

  cudaFree(devx);
  cudaFree(devy);
  cublasDestroy(cublashandle);
  cusparseDestroy(cusparsehandle);
}




TEST_CASE("testcublas", "[test_gpu_fire_up_only]") {

  std::cout << "Test extra GPU only: cublas" << std::endl;

  int len = 10;
 
  // alloc and init host vec memory
  double  *h_vec1 = (double *)malloc(len * sizeof(double));
  double *h_vec2 = (double *)malloc(len * sizeof(double));
  for (int i = 0; i < len; i++) {
    h_vec1[i] = 1.0;
    h_vec2[i] = i;
    // h_vec1[i] = 1.0;
    // h_vec2[i] = 2.0;
  }

  // alloc and init device vec memory
  double *d_vec1;
  double *d_vec2;
  cudaMalloc((void **)&d_vec1, len * sizeof(double));
  cudaMemcpy(d_vec1, h_vec1, len * sizeof(double),
             cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_vec2, len * sizeof(double));
  cudaMemcpy(d_vec2, h_vec2, len * sizeof(double),
             cudaMemcpyHostToDevice);

  double result_1;
  double result_2;

  cublasHandle_t cublashandle;
  cublasStatus_t status_cublas = cublasCreate(&cublashandle);
  if (status_cublas != CUBLAS_STATUS_SUCCESS) {                               
    printf("Create 2: CUBLAS API failed at line %d of %s with error: %s (%d)\n", 
            __LINE__, __FILE__,
            cublasGetStatusString(status_cublas), status_cublas);  
  }

  cublasStatus_t status = cublasDnrm2(cublashandle, len, d_vec1, 1, &result_1);
  if (status != CUBLAS_STATUS_SUCCESS) {                               
      printf("CUBLAS API failed at line %d of %s with error: %s (%d)\n", 
             __LINE__, __FILE__, cublasGetStatusString(status), status); 
  }

  status = cublasDnrm2(cublashandle, len, d_vec2, 1, &result_2);
  if (status != CUBLAS_STATUS_SUCCESS) {                               
      printf("CUBLAS API failed at line %d of %s with error: %s (%d)\n", 
             __LINE__, __FILE__, cublasGetStatusString(status), status); 
  }

  // print result
  printf("2-norm is :%f\n", result_1);
  printf("2-norm is :%f\n", result_2);

  REQUIRE(result_1 - 3.162278 < 1e06);
  REQUIRE(result_2 - 16.881943 < 1e06);
 
  cublasDestroy(cublashandle);
}
