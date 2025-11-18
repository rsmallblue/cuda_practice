import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_start * input_row_stride
        col_offset = tl.arange(0, n_cols)
        input_ptr = row_start_ptr + col_offset
        mask = col_offset < n_cols
        row = tl.load(input_ptr, mask=mask)
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)

        output = numerator / denominator

        output_ptr = output_ptr + row_start * output_row_stride + col_offset
        tl.store(output_ptr, output, mask=mask)
    