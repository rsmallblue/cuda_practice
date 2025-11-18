#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cutlass/barrier.h"
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <cute/tensor.hpp>

using namespace cute;
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC) {
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    __shared__ float smemA[cosize_v<ASmemLayout>];
    __shared__ float smemB[cosize_v<BSmemLayout>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);

    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});

    Tensor tCgC = local_partition(gC, tC, threadIdx.x);

    Tensor tCrC = make_tensor_like(tCgC);

    clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++) {
        copy(tAgA(_, _, k_tile), tAsA); // (_128,_8)
        copy(tBgB(_, _, k_tile), tBsB); // (_128,_8)

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tCsA, tCsB, tCrC); // (_8,_8)

        __syncthreads();

        copy(tCrC, tCgC);
    }
    

}

void gemm(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream = 0) {
    using BM = Int<128>;
    using BN = Int<128>;
    using BK = Int<8>;

    auto prob_shape = make_shape(M, N, K);

    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, K);                      // (dM, dK)
    auto dB = make_stride(Int<1>{}, K);                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, N);                      // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<  8>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (m,k) -> thr_idx
    auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (n,k) -> thr_idx
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx

    dim3 Grids(size(ceil_div(M, BM::value)), size(ceil_div(N, BN::value)));
    dim3 Blocks(size(tC));

  gemm_device<<<Grids, Blocks, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC);



}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    thrust::host_vector<float> h_A(M * K);
    thrust::host_vector<float> h_B(K * N);
    thrust::host_vector<float> h_C(M * N, 0);

    // Initialize host vectors with random values
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    thrust::generate(h_A.begin(), h_A.end(), [&]() { return dist(rng); });
    thrust::generate(h_B.begin(), h_B.end(), [&]() { return dist(rng); });

    // Copy data to device
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C = h_C;

    gemm(d_A.data().get(), d_B.data().get(), d_C.data().get(), M, N, K);
}
