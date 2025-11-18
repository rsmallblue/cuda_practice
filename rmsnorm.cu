#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cutlass/barrier.h"
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#define SPLIT_K 1
#define BLOCKS_PER_ROW 4
#define BLOCK_SIZE 256
#define warpSize 32

template <int BYTES>
struct BytesToType{};

template <>
struct BytesToType<16> {
    using Type = float4;
};

template <>
struct BytesToType<8> {
    using Type = uint64_t;
};

struct GlobalSyncState {
    cutlass::Barrier::T counter;
    float partial_sum[BLOCKS_PER_ROW];
};

template <typename T>
__device__ void warp_reduce(T* value) {
    for (int mask = 16 ; mask > 0 ; mask >>= 1) {
        *value += __shfl_xor_sync(0xFFFFFFFF, *value, mask);
    }
}

template <typename T>
__device__ void block_reduce(T* value) {
    __shared__ T shm[warpSize];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warp_num = blockDim.x / warpSize;
    warp_reduce(value);
    if (lane_id == 0) {
        shm[warp_id] = *value;
    }
    __syncthreads();
    T tmp;
    if (warp_id == 0) {
        T tmp = lane_id < warp_num ? shm[lane_id] : static_cast<half>(0.0f);
        warp_reduce(&tmp);
        shm[0] = tmp;
    };
    __syncthreads();
    *value = shm[0];
}


template <int VecSize>
__global__ void rmsnorm_kernel1(
    const half* input,
    const half* weight,
    int M, int N,
    const float eps,
    GlobalSyncState* sync_state,
    half* output
) {
    using vec_t = typename BytesToType<sizeof(half) * VecSize>::Type;
    int row = blockIdx.y;
    int block_in_row = blockIdx.x;

    vec_t vec_input, vec_weight, vec_output;

    const int elems_per_block = N / BLOCKS_PER_ROW;
    const int start_col = block_in_row * elems_per_block;
    const int end_col = min((block_in_row + 1) * elems_per_block, N);

    for (int token_id = row; token_id < M; token_id += gridDim.y) {
        half thread_sum = 0.0f;
        for (int col = start_col; col < end_col; col += blockDim.x * VecSize) {
            vec_input = *(reinterpret_cast<const vec_t*>(input + token_id * N + col));
            #pragma unroll
            for (int j = 0; j < VecSize; ++j) {
                thread_sum += reinterpret_cast<const half*>(&vec_input)[j] * reinterpret_cast<const half*>(&vec_input)[j];
            }
        }
        block_reduce(&thread_sum);
        GlobalSyncState* row_sync = &sync_state[token_id];
        if (threadIdx.x == 0) {
            row_sync->partial_sum[block_in_row] = static_cast<float>(thread_sum);
        }
        __threadfence();

        cutlass::Barrier::arrive_inc(
            reinterpret_cast<void*>(&row_sync->counter),
            threadIdx.x,
            0,
            1
        ); 

        // if (threadIdx.x == 0) {
        //     printf("arrive_inc row = %d, block_in_row = %d, thread_sum = %f, %d\n", row, block_in_row, thread_sum, row_sync->counter);
        // }

        cutlass::Barrier::wait_eq(
            reinterpret_cast<void*>(&row_sync->counter),
            threadIdx.x, 
            0,
            BLOCKS_PER_ROW
        ); 

        // if (threadIdx.x == 0) {
        //     printf("wait_eq row = %d, block_in_row = %d, total_sum = %f, %d\n", row, block_in_row, row_sync->partial_sum[0], row_sync->counter);
        // }

        __syncthreads();

        if (block_in_row == 0) {
            if (threadIdx.x == 0) {
                float total_sum = 0.0f;
                for (int i = 0; i < BLOCKS_PER_ROW; i++) {
                    total_sum += row_sync->partial_sum[i];
                }
                row_sync->partial_sum[0] = total_sum;                
            }
            __syncthreads();

            cutlass::Barrier::arrive_inc(
                reinterpret_cast<void*>(&row_sync->counter),
                threadIdx.x,
                0,
                1
            ); 
        }

        cutlass::Barrier::wait_eq(
            reinterpret_cast<void*>(&row_sync->counter),
            threadIdx.x, 
            0,
            BLOCKS_PER_ROW + 1
        ); 
        __syncthreads();

        float total_sum = row_sync->partial_sum[0];

        half inv = static_cast<half>(1.0f / sqrt(static_cast<float>(total_sum) + eps));

        for (int i = threadIdx.x * VecSize; i < N; i += blockDim.x * VecSize) {
            vec_input = *(reinterpret_cast<const vec_t*>(input + token_id * N + i));
            vec_weight = *(reinterpret_cast<const vec_t*>(weight + i));
            #pragma unroll
            for (int j = 0; j < VecSize; j++) {
                reinterpret_cast<half*>(&vec_output)[j] = reinterpret_cast<const half*>(&vec_input)[j] * inv * reinterpret_cast<const half*>(&vec_weight)[j];
            }
            *(reinterpret_cast<vec_t*>(output + token_id * N + i)) = vec_output;
        }
    }
}

template <int VecSize>
__global__ void rmsnorm_kernel0(const half* input, const half* weight, int M, int N, float eps, half* output) {
    using vec_t = typename BytesToType<sizeof(half) * VecSize>::Type;
    for (int token_id = blockIdx.x; token_id < M; token_id += gridDim.x) {
        half thread_sum = 0.0f;
        vec_t vec_input, vec_weight, vec_output;
        for (int i = threadIdx.x * VecSize; i < N; i += blockDim.x * VecSize) {
            vec_input = *(reinterpret_cast<const vec_t*>(input + token_id * N + i));
            #pragma unroll
            for (int j = 0; j < VecSize; j++) {
                thread_sum += reinterpret_cast<const half*>(&vec_input)[j] * reinterpret_cast<const half*>(&vec_input)[j];
            }
        }
        block_reduce(&thread_sum);
        half inv = static_cast<half>(1.0f / sqrt(static_cast<float>(thread_sum) + eps));
        for (int i = threadIdx.x * VecSize; i < N; i += blockDim.x * VecSize) {
            vec_input = *(reinterpret_cast<const vec_t*>(input + token_id * N + i));
            vec_weight = *(reinterpret_cast<const vec_t*>(weight + i));
            #pragma unroll
            for (int j = 0; j < VecSize; j++) {
                reinterpret_cast<half*>(&vec_output)[j] = reinterpret_cast<const half*>(&vec_input)[j] * inv * reinterpret_cast<const half*>(&vec_weight)[j];
            }
            *(reinterpret_cast<vec_t*>(output + token_id * N + i)) = vec_output;
        }
    }
}

void rmsnorm_kernel_gpu(
    const half* input, 
    const half* weight, 
    half* output,   
    int M, 
    int N,
    float eps
) {
    constexpr int VecSize = 16 / sizeof(half); //8
    if (SPLIT_K) {
        dim3 grid_size(BLOCKS_PER_ROW, M);
        dim3 block_size(BLOCK_SIZE);
        
        GlobalSyncState* sync_state;
        cudaMalloc(&sync_state, M * sizeof(GlobalSyncState));
        cudaMemset(sync_state, 0, M * sizeof(GlobalSyncState));
        
        rmsnorm_kernel1<VecSize><<<grid_size, block_size>>>(input, weight, M, N, eps, sync_state, output);
        
        cudaFree(sync_state);        
    } else {
        rmsnorm_kernel0<VecSize><<<M, 256>>>(input, weight, M, N, eps, output);
    }
}


void rmsnorm_kernel_cpu(const half* input, const half* weight, half* output, int M, int N, float eps) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += static_cast<float>(input[i * N + j] * input[i * N + j]);
        }
        half inv = static_cast<half>(1.0f / sqrt(sum + eps));
        for (int j = 0; j < N; j++) {
            output[i * N + j] = input[i * N + j] * inv * weight[j];
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


int main() {
    const int M = 330;
    const int N = 65536;
    

    thrust::host_vector<half> h_input(M*N);
    thrust::host_vector<half> h_output_ref(M*N);
    thrust::host_vector<half> h_weight(N);

    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    thrust::generate(h_input.begin(), h_input.end(), [&]() {
        float val = dist(rng);
        return static_cast<half>(val);
    });

    thrust::generate(h_weight.begin(), h_weight.end(), [&]() {
        float val = dist(rng);
        return static_cast<half>(val);
    });

    thrust::fill(h_output_ref.begin(), h_output_ref.end(), half(0.0f));

    thrust::device_vector<half> d_input = h_input;
    thrust::device_vector<half> d_weight = h_weight;
    thrust::device_vector<half> d_output = h_output_ref;

    const float eps = 1e-6;


    rmsnorm_kernel_cpu(h_input.data(), h_weight.data(), h_output_ref.data(), M, N, eps);

    rmsnorm_kernel_gpu(d_input.data().get(), d_weight.data().get(), d_output.data().get(), M, N, eps);

    check_diff(d_output, h_output_ref, M, N);
    
    return 0;
}
