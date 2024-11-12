#include <iostream>

#include "catch.hpp"

#include <cublas_v2.h>         // cublas
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>          // cusparseSpMV

TEST_CASE("start-nvidia", "[test_gpu_fire_up]") {
  // TODO add code here

  std::cout << "GPU FIRE UP TEST" << std::endl;

  REQUIRE(true);
}
