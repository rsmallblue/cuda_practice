import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M' : 128, 'BLOCK_SIZE_N' : 256, 'BLOCK_SIZE_K' : 64, 'GROUP_SIZE_M' : 8}, num_states=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M' : 128, 'BLOCK_SIZE_N' : 256, 'BLOCK_SIZE_K' : 64, 'GROUP_SIZE_M' : 16}, num_states=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M' : 128, 'BLOCK_SIZE_N' : 256, 'BLOCK_SIZE_K' : 64, 'GROUP_SIZE_M' : 32}, num_states=3, num_warps=8),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return []

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_tpr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_K:tl.constexpr, BLOCK_SIZE_N:tl.constexpr,
    GROUP_SIZE_M:tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n # 每个分组的pid数量
    group_id = pid // num_pid_in_group #第几个group
    first_pid_m = group_id * GROUP_SIZE_M #第group_id个group的起始pid_m
    group_size_m = min(GROUP_SIZE_M, M - first_pid_m)  #重要！勿忘
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_mSIZE_M 
    pid_n = pid % num_pid_in_group // GROUP_SIZE_M #第group_id个group的起始n偏移量

    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += stride_ak
        b_ptrs += stride_bk
    c = accumulator.to(tl.bfloat16)

    off_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (off_cm[:, None] * stride_cm + off_cn[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=off_cm[:, None] < M and off_cn[None, :] < N)



