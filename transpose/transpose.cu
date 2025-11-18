#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cutlass/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>


using namespace cute;

// Helper functions for retrieving optimal swizzled layouts
template <typename PrecType, int DIM> constexpr auto getSmemLayoutK() {

  constexpr int headSizeBytes = sizeof(PrecType) * DIM;

  if constexpr (headSizeBytes == 16) {
    return GMMA::Layout_K_INTER_Atom<PrecType>{};
  } else if constexpr (headSizeBytes == 32) {
    return GMMA::Layout_K_SW32_Atom<PrecType>{};
  } else if constexpr (headSizeBytes == 64) {
    return GMMA::Layout_K_SW64_Atom<PrecType>{};
  } else {
    return GMMA::Layout_K_SW128_Atom<PrecType>{};
  }
}

template <typename PrecType, int DIM> constexpr auto getSmemLayoutMN() {

  constexpr int headSizeBytes = sizeof(PrecType) * DIM;

  if constexpr (headSizeBytes == 16) {
    return GMMA::Layout_MN_INTER_Atom<PrecType>{};
  } else if constexpr (headSizeBytes == 32) {
    return GMMA::Layout_MN_SW32_Atom<PrecType>{};
  } else if constexpr (headSizeBytes == 64) {
    return GMMA::Layout_MN_SW64_Atom<PrecType>{};
  } else {
    return GMMA::Layout_MN_SW128_Atom<PrecType>{};
  }
}

void set_smem_size(int smem_size, void const *kernel) {
  // account for dynamic smem capacity if needed
  if (smem_size >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      std::cout << "  Shared Memory Allocation Failed " << std::endl
                << " cudaFuncSetAttribute() returned error: "
                << cudaGetErrorString(result) << std::endl;
    }
  }
}

template <class Element, class SmemLayout>
struct SharedStorageTranspose {
    cute::array_aligned<Element, cute::cosize_v<SmemLayout>, cutlass::detail::alignment_for_swizzle(SmemLayout{})> smem;
};

template <class TensorS, class TensorD, class ThreadLayoutS, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1) transpose_kernel0(TensorS S, TensorD D, ThreadLayoutS tS, ThreadLayoutD tD) {
    using Element = typename TensorS::value_type;

    Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);
    Tensor gD = D(make_coord(_, _), blockIdx.x, blockIdx.y);

    Tensor tSgS = local_partition(gS, tS, threadIdx.x);
    Tensor tDgD = local_partition(gD, tD, threadIdx.x);

    Tensor rmem = make_tensor_like(tSgS);

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //     print(rmem.layout()); // (_8,_2):(_2,_1)
    //     printf("\n");
    //     print(tSgS.layout()); // (_8,_2):(262144,_32)
    //     printf("\n");
    //     print(tDgD.layout()); // (_8,_2):(_8,1048576)
    //     printf("\n");
    // }

    copy(tSgS, rmem);
    copy(rmem, tDgD);
}


template <typename T>
void transpose_gpu_naive(T* input, T* output, int M, int N) {
    auto tensor_shape = make_shape(M, N);
    auto tensor_shape_T = make_shape(N, M);

    auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
    // auto gmemLayoutD = make_layout(tensor_shape_T, LayoutRight{});
    Tensor tensor_S = make_tensor(make_gmem_ptr(input), gmemLayoutS);
    // Tensor tensor_D = make_tensor(make_gmem_ptr(output), gmemLayoutD);

    auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});
    Tensor tensor_DT = make_tensor(make_gmem_ptr(output), gmemLayoutDT);

    using BM = Int<64>;
    using BN = Int<64>;

    auto block_shape = make_shape(BM{}, BN{});
    auto block_shape_T = make_shape(BN{}, BM{});

    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
    print(tiled_tensor_S.layout());    // print_tensor(tiled_tensor_S); ((_64,_64),512,512):((32768,_1),2097152,_64)
    printf("\n");
    Tensor tiled_tensor_DT = tiled_divide(tensor_DT, block_shape);
    print(tiled_tensor_DT.layout());    // print_tensor(tiled_tensor_D); ((_64,_64),512,512):((_1,32768),_64,2097152)
    printf("\n");

    auto threadLayoutS = make_layout(make_shape(Int<8>{}, Int<32>{}));
    auto threadLayoutD = make_layout(make_shape(Int<8>{}, Int<32>{}));

    dim3 gridDim(size<1>(tiled_tensor_S), size<2>(tiled_tensor_S));
    dim3 blockDim(size(threadLayoutS));

    transpose_kernel0<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_DT, threadLayoutS, threadLayoutD);
}

template <class TensorS, class TensorD, class SmemLayoutS, class ThreadLayoutS, class SmemLayoutD, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1) transpose_kernel1(
    TensorS S, TensorD D, 
    SmemLayoutS smemLayoutS, SmemLayoutD smemLayoutD, 
    ThreadLayoutS tS, ThreadLayoutD tD) {
    using namespace cute;
    using Element = typename TensorS::value_type;

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayoutS);
    Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayoutD);

    Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);
    Tensor gD = D(make_coord(_, _), blockIdx.y, blockIdx.x);

    Tensor tSgS = local_partition(gS, tS, threadIdx.x);
    Tensor tDgD = local_partition(gD, tD, threadIdx.x);

    Tensor tSsS = local_partition(sS, tS, threadIdx.x);
    Tensor tDsD = local_partition(sD, tD, threadIdx.x);

    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    //     // print(rmem.layout()); // (_8,_2):(_2,_1)
    //     // printf("\n");
    //     print(tSgS.layout()); // (_8,_2):(262144,_32)
    //     printf("\n");
    //     print(tDgD.layout()); // (_8,_2):(262144,_32)
    //     printf("\n"); 
    //     print(tSsS.layout()); // ((_2,_4),_2):((528,_1024),33)
    //     printf("\n");
    //     print(tDsD.layout()); // ((_2,_2,_2),_2):((8,16,33),_2048)
    //     printf("\n");          
    //     print_layout(tDsD.layout());      
    // }

    cute::copy(tSgS, tSsS);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    cute::copy(tDsD, tDgD);
}

template <class TensorS, class TensorD, class SmemLayoutS, class ThreadLayoutS,
          class SmemLayoutD, class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
    transposeKernelSmem(TensorS const S, TensorD const D,
                        SmemLayoutS const smemLayoutS, ThreadLayoutS const tS,
                        SmemLayoutD const smemLayoutD, ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Use Shared Storage structure to allocate aligned SMEM addresses.
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTranspose<Element, SmemLayoutD>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // two different views of smem
  Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutS); // (bM, bN)
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayoutD); // (bN, bM)

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y); // (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.y, blockIdx.x); // (bN, bM)

  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tSsS = local_partition(sS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tDgD = local_partition(gD, tD, threadIdx.x);
  Tensor tDsD = local_partition(sD, tD, threadIdx.x);

  cute::copy(tSgS, tSsS); // LDGSTS

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  cute::copy(tDsD, tDgD);
}

template <typename T>
void transpose_gpu_smem(T* input, T* output, int M, int N) {
    auto tensor_shape = make_shape(M, N);
    auto tensor_shape_T = make_shape(N, M);

    auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
    auto gmemLayoutDT = make_layout(tensor_shape_T, LayoutRight{});
    // auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{});

    Tensor tensor_S = make_tensor(make_gmem_ptr(input), gmemLayoutS);
    Tensor tensor_DT = make_tensor(make_gmem_ptr(output), gmemLayoutDT);

    using BM = Int<64>;
    using BN = Int<64>;

    auto block_shape = make_shape(BM{}, BN{});
    auto block_shape_T = make_shape(BN{}, BM{});

    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
    Tensor tiled_tensor_DT = tiled_divide(tensor_DT, block_shape_T);


    auto tiledShapeS = make_layout(block_shape, LayoutRight{});
    // print(tiledShapeS);   
    // printf("\n"); 
    auto tiledShapeD = make_layout(block_shape_T, LayoutRight{});
    // print(tiledShapeD);   
    // printf("\n"); 

    auto smemLayoutS = tiledShapeS;
    // print(smemLayoutS);   
    // printf("\n"); 
    auto smemLayoutD = composition(smemLayoutS, tiledShapeD);
    // print(smemLayoutD);   
    // printf("\n"); 
    auto smemLayoutS_swizzle = composition(Swizzle<5, 0, 5>{}, tiledShapeS); //Sw<5,0,5> o _0 o (_32,_64):(_64,_1)
    // print(smemLayoutS_swizzle);   
    // printf("\n"); 
    auto smemLayoutD_swizzle = composition(smemLayoutS_swizzle, tiledShapeD); //Sw<5,0,5> o _0 o (_64,_32):(_1,_64)
    // print(smemLayoutD_swizzle);    
    // printf("\n");

    auto threadLayoutS = make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
    auto threadLayoutD = make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

    size_t smem_size = int(sizeof(SharedStorageTranspose<T, decltype(smemLayoutS_swizzle)>));

    dim3 gridDim(size<1>(tiled_tensor_S), size<2>(tiled_tensor_S));
    dim3 blockDim(size(threadLayoutS));

    // transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
    //     tiled_tensor_S, tiled_tensor_DT, smemLayoutS_swizzle, threadLayoutS,
    //     smemLayoutD_swizzle, threadLayoutD);

    transpose_kernel1<<<gridDim, blockDim, smem_size>>>(tiled_tensor_S, tiled_tensor_DT, smemLayoutS_swizzle, smemLayoutD_swizzle, threadLayoutS, threadLayoutD);
}

template <class TensorS, class SmemLayout, class TiledCopyS, class TiledCopyD,
class GmemLayoutD, class TileShapeD, class ThreadLayoutM, class SmemLayoutM>
__global__ static void __launch_bounds__(256)
    transpose_kernel_tma(TensorS const S, SmemLayout const smemLayout,
                       TiledCopyS const tiled_copy_S,
                       CUTE_GRID_CONSTANT TiledCopyD const tmaStoreD,
                       GmemLayoutD const gmemLayoutD,
                       TileShapeD const tileShapeD, ThreadLayoutM const tM,
                       SmemLayoutM const smemLayoutM) {
    using Element = typename TensorS::value_type;

    int lane_predicate = cute::elect_one_sync();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    bool leaderWarp = warp_idx == 0;

    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorageTranspose<Element, SmemLayout>;
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sM = make_tensor(make_smem_ptr(shared_storage.smem.data()), smemLayoutM);

    Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);
    auto thr_copy_S = tiled_copy_S.get_thread_slice(threadIdx.x);

    Tensor tSgS = thr_copy_S.partition_S(gS);
    Tensor tSrS = make_fragment_like(tSgS);
    Tensor tMsM = local_partition(sM, tM, threadIdx.x);

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        // print(thr_copy_S.layout()); // 
        // printf("\n");
        print(sM.layout()); // (_4,_8,_32):(_32,_128,_1)
        printf("\n");
        print(gS.layout()); // (_32,_32):(32768,_1)
        printf("\n"); 
        print(tSgS.layout()); // ((_4,_1),_1,_1):((_1,_0),_0,_0)
        printf("\n");
        print(tSrS.layout()); // ((_4,_1),_1,_1):((_1,_0),_0,_0)
        printf("\n");      
        print(tMsM.layout()); // (_4,_1,_1):(_32,_0,_0)
        printf("\n");       
        // print_layout(tDsD.layout());      
    }

   // Copy from GMEM to RMEM to SMEM
  copy(tiled_copy_S, tSgS, tSrS);
  copy(tSrS, tMsM);

  auto synchronize = [&]() {
    cutlass::arch::NamedBarrier::sync(size(ThreadLayoutM{}), 0);
  };
  cutlass::arch::fence_view_async_shared();
  synchronize();

  // Issue the TMA store.
  Tensor mD = tmaStoreD.get_tma_tensor(shape(gmemLayoutD));
  auto blkCoordD = make_coord(blockIdx.y, blockIdx.x);
  Tensor gD = local_tile(mD, tileShapeD, blkCoordD);
  Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()),
                          smemLayout); // (bN, bM)

  auto cta_tmaD = tmaStoreD.get_slice(0);

  Tensor tDgDX = cta_tmaD.partition_D(gD);
  Tensor tDgD = group_modes<1, rank(tDgDX)>(tDgDX); // (TMA,REST)
  assert(size<1>(tDgD) == 1);

  Tensor tDsDX = cta_tmaD.partition_S(sD);
  Tensor tDsD = group_modes<1, rank(tDsDX)>(tDsDX); // (TMA,REST)
  static_assert(size<1>(tDsD) == 1);

  if (leaderWarp and lane_predicate) {
    copy(tmaStoreD, tDsD, tDgD);
  }
  // Wait for TMA store to complete.
  tma_store_wait<0>();   

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        // print(rmem.layout()); // 
        // printf("\n");
        print(mD.layout()); // (32768,32768):(_1@1,_1@0)
        printf("\n");
        print(gD.layout()); // (_32,_32):(_1@1,_1@0)
        printf("\n"); 
        print(sD.layout()); // ((_8,_4),(_32,_1)):((_32,_256),(_1,_0))
        printf("\n");
        print(tDgDX.layout()); // (((_32,_32),_1),_1,_1):(((_1@0,_1@1),_0),_0,_0)
        printf("\n");      
        print(tDgD.layout()); // (((_32,_32),_1),(_1,_1)):(((_1@0,_1@1),_0),(_0,_0))
        printf("\n");    
        print(tDsDX.layout()); // (((_32,_32),_1),_1,_1):(((_1,_32),_0),_0,_0)
        printf("\n");      
        print(tDsD.layout()); // (((_32,_32),_1),(_1,_1)):(((_1,_32),_0),(_0,_0))
        printf("\n");    
        // print_layout(tDsD.layout());      
    }

}

template <typename T>
void transpose_gpu_tma(T* input, T* output, int M, int N) {
    auto tensor_shape = make_shape(M, N);
    auto tensor_shape_T = make_shape(N, M);

    auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
    auto gmemLayoutD = make_layout(tensor_shape_T, LayoutRight{});

    Tensor tensor_S = make_tensor(make_gmem_ptr(input), gmemLayoutS);
    Tensor tensor_D = make_tensor(make_gmem_ptr(output), gmemLayoutD);

    using BM = Int<32>;
    using BN = Int<32>;

    auto block_shape = make_shape(BM{}, BN{});
    auto block_shape_T = make_shape(BN{}, BN{});

    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);
    Tensor tiled_tensor_DT = tiled_divide(tensor_D, block_shape_T);

    auto threadLayoutS =
        make_layout(make_shape(Int<32>{}, Int<8>{}), LayoutRight{});

    auto vecLayoutS = make_layout(make_shape(Int<1>{}, Int<4>{}));

    using AccessTypeS = cutlass::AlignedArray<T, size(vecLayoutS)>;
    using AtomS = Copy_Atom<UniversalCopy<AccessTypeS>, T>;
    auto tiled_copy_S = make_tiled_copy(AtomS{}, threadLayoutS, vecLayoutS);

    auto tileShapeD = block_shape_T;

    auto smemLayoutD = tile_to_shape(getSmemLayoutK<T, BM{}>(), make_shape(shape<0>(tileShapeD), shape<1>(tileShapeD)));

    auto tmaD = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayoutD, tileShapeD,
                                Int<1>{});

    auto tileShapeM = make_shape(Int<4>{}, Int<8>{}, Int<32>{});
    auto smemLayoutM = composition(smemLayoutD, make_layout(tileShapeM));
    auto threadLayoutM = make_layout(make_shape(Int<1>{}, Int<8>{}, Int<32>{}),
                                    make_stride(Int<1>{}, Int<1>{}, Int<8>{}));

    size_t smem_size =
        int(sizeof(SharedStorageTranspose<T, decltype(smemLayoutD)>));
    

    dim3 gridDim(
        size<1>(tiled_tensor_S),
        size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(threadLayoutS));

    transpose_kernel_tma<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, smemLayoutD, tiled_copy_S, tmaD, gmemLayoutD, tileShapeD,
        threadLayoutM, smemLayoutM);
}

void transpose_cpu(const half* input, half* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j*M+i] = input[i*N+j];
        }
    }
}

void check_diff(const thrust::host_vector<half> d_output, const thrust::host_vector<half> h_output_ref, int M, int N) {
    thrust::host_vector<half> h_output = d_output;
    for (int i = 0; i < M*N; i++) {
        if (abs(static_cast<float>(h_output[i] - h_output_ref[i])) > 1e-3) {
            std::cout << "Mismatch at index " << i << ": " << static_cast<float>(h_output[i]) << " (device) vs " << static_cast<float>(h_output_ref[i]) << " (host)" << std::endl;
        }
    }
}


template <typename Element, bool isSwizzled = true> void transpose_smem(Element* input, Element* output, int M, int N) {

  using namespace cute;

  //
  // Make tensors
  //
  auto tensor_shape = make_shape(M, N);
  auto tensor_shape_trans = make_shape(N, M);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape_trans, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(output), gmemLayoutD);

  //
  // Tile tensors
  //

  using bM = Int<64>;
  using bN = Int<64>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)
  auto block_shape_trans = make_shape(bN{}, bM{}); // (bN, bM)

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape_trans); // ((bN, bM), n', m')

  auto tileShapeS = make_layout(block_shape, LayoutRight{});
  auto tileShapeD = make_layout(block_shape_trans, LayoutRight{});

  auto smemLayoutS = tileShapeS;
  auto smemLayoutD = composition(smemLayoutS, tileShapeD);
  auto smemLayoutS_swizzle = composition(Swizzle<5, 0, 5>{}, tileShapeS);
  auto smemLayoutD_swizzle = composition(smemLayoutS_swizzle, tileShapeD);

  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

  size_t smem_size = int(
      sizeof(SharedStorageTranspose<Element, decltype(smemLayoutS_swizzle)>));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS)); // 256 threads

  if constexpr (isSwizzled) {
    transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, tiled_tensor_D, smemLayoutS_swizzle, threadLayoutS,
        smemLayoutD_swizzle, threadLayoutD);
  } else {
    transposeKernelSmem<<<gridDim, blockDim, smem_size>>>(
        tiled_tensor_S, tiled_tensor_D, smemLayoutS, threadLayoutS,
        smemLayoutD, threadLayoutD);
  }

    transpose_kernel1<<<gridDim, blockDim, smem_size>>>(tiled_tensor_S, tiled_tensor_D, smemLayoutS_swizzle, smemLayoutD_swizzle, threadLayoutS, threadLayoutD);
    
}

int main() {
    const int M = 32768;
    const int N = 32768;

    thrust::host_vector<float> h_input(M*N);
    thrust::host_vector<float> h_output(M*N);

    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    thrust::generate(h_input.begin(), h_input.end(), [&]() {
        float val = dist(rng);
        return static_cast<float>(val);
    });

    thrust::device_vector<float> d_input = h_input;
    thrust::device_vector<float> d_output = h_output;

    // transpose_gpu_naive(d_input.data().get(), d_output.data().get(), M, N);    
    // transpose_gpu_smem(d_input.data().get(), d_output.data().get(), M, N);
    // transpose_smem(d_input.data().get(), d_output.data().get(), M, N);
    transpose_gpu_tma(d_input.data().get(), d_output.data().get(), M, N);
    // transpose_cpu(h_input.data(), h_output.data(), M, N);
    // check_diff(d_output, h_output, M, N);
}