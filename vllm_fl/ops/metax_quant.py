# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


def _gemm_config(num_tokens: int) -> tuple[int, int, int]:
    return (16, 32, 4) if num_tokens <= 16 else (64, 64, 8)


@triton.jit
def _fp8_block_gemm_kernel(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    output_ptr,
    m,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_om,
    stride_on,
    block_scale_n: tl.constexpr,
    block_scale_k: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_scale_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(0, k, block_scale_k):
        a = tl.load(
            a_ptr
            + offsets_m[:, None] * stride_am
            + (k_start + offsets_k[None, :]) * stride_ak,
            mask=(offsets_m[:, None] < m)
            & (k_start + offsets_k[None, :] < k),
            other=0.0,
        ).to(tl.bfloat16)
        b = tl.load(
            b_ptr
            + offsets_n[None, :] * stride_bn
            + (k_start + offsets_k[:, None]) * stride_bk,
            mask=(offsets_n[None, :] < n)
            & (k_start + offsets_k[:, None] < k),
            other=0.0,
        ).to(tl.bfloat16)
        a_scale = tl.load(
            a_scale_ptr
            + offsets_m * stride_asm
            + (k_start // block_scale_k) * stride_ask,
            mask=offsets_m < m,
            other=0.0,
        )
        b_scale = tl.load(
            b_scale_ptr
            + (offsets_n // block_scale_n) * stride_bsn
            + (k_start // block_scale_k) * stride_bsk,
            mask=offsets_n < n,
            other=0.0,
        )
        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

    tl.store(
        output_ptr
        + offsets_m[:, None] * stride_om
        + offsets_n[None, :] * stride_on,
        accumulator.to(tl.bfloat16),
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


def metax_fp8_block_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    block_scale_n, block_scale_k = block_size
    m, k = a.shape
    n = b.shape[0]
    output = torch.empty((m, n), device=a.device, dtype=output_dtype)
    block_m, block_n, num_warps = _gemm_config(m)
    _fp8_block_gemm_kernel[
        (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    ](
        a,
        b,
        a_scale,
        b_scale,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        output.stride(0),
        output.stride(1),
        block_scale_n,
        block_scale_k,
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


@triton.jit
def _fp8_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    output_ptr,
    m,
    groups: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am,
    stride_ag,
    stride_ak,
    stride_bg,
    stride_bn,
    stride_bk,
    stride_bsg,
    stride_bsn,
    stride_bsk,
    stride_om,
    stride_og,
    stride_on,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    group = tl.program_id(2)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(0, k, block_k):
        a = tl.load(
            a_ptr
            + offsets_m[:, None] * stride_am
            + group * stride_ag
            + (k_start + offsets_k[None, :]) * stride_ak,
            mask=(offsets_m[:, None] < m)
            & (k_start + offsets_k[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + group * stride_bg
            + offsets_n[None, :] * stride_bn
            + (k_start + offsets_k[:, None]) * stride_bk,
            mask=(offsets_n[None, :] < n)
            & (k_start + offsets_k[:, None] < k),
            other=0.0,
        ).to(tl.bfloat16)
        b_scale = tl.load(
            b_scale_ptr
            + group * stride_bsg
            + (offsets_n // 128) * stride_bsn
            + (k_start // block_k) * stride_bsk,
            mask=offsets_n < n,
            other=0.0,
        )
        accumulator += tl.dot(a, b) * b_scale[None, :]

    tl.store(
        output_ptr
        + offsets_m[:, None] * stride_om
        + group * stride_og
        + offsets_n[None, :] * stride_on,
        accumulator.to(tl.bfloat16),
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


def metax_fp8_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    m, groups, k = a.shape
    _, n, _ = b.shape
    output = torch.empty((m, groups, n), device=a.device, dtype=a.dtype)
    block_m, block_n, num_warps = _gemm_config(m)
    _fp8_grouped_gemm_kernel[
        (triton.cdiv(m, block_m), triton.cdiv(n, block_n), groups)
    ](
        a,
        b,
        b_scale,
        output,
        m,
        groups,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_m,
        block_n,
        128,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


@triton.jit
def _decode_e2m1(code):
    magnitude = code & 7
    value = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(magnitude == 5, 3.0, tl.where(magnitude == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 8) == 0, value, -value)


@triton.jit
def _mxfp4_gemm_kernel(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    output_ptr,
    m,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_bsn,
    stride_bsk,
    stride_om,
    stride_on,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, 32)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(0, k, 32):
        k_offsets = k_start + offsets_k
        a = tl.load(
            a_ptr + offsets_m[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(offsets_m[:, None] < m) & (k_offsets[None, :] < k),
            other=0.0,
        )
        packed = tl.load(
            b_ptr
            + offsets_n[None, :] * stride_bn
            + (k_offsets[:, None] // 2) * stride_bk,
            mask=(offsets_n[None, :] < n) & (k_offsets[:, None] < k),
            other=0,
        )
        code = (packed >> ((k_offsets[:, None] & 1) * 4)) & 15
        value = _decode_e2m1(code).to(tl.bfloat16)
        exponent = tl.load(
            b_scale_ptr
            + offsets_n[None, :] * stride_bsn
            + (k_start // 32) * stride_bsk,
            mask=offsets_n[None, :] < n,
            other=0,
        )
        scale_bits = exponent.to(tl.int32) << 23
        scale = scale_bits.to(tl.float32, bitcast=True)
        weight = (value * scale).to(tl.bfloat16)
        accumulator += tl.dot(a, weight)

    tl.store(
        output_ptr
        + offsets_m[:, None] * stride_om
        + offsets_n[None, :] * stride_on,
        accumulator.to(tl.bfloat16),
        mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n),
    )


def metax_mxfp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[0]
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m, block_n, num_warps = _gemm_config(m)
    scale_bytes = b_scale.view(torch.uint8)
    _mxfp4_gemm_kernel[
        (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    ](
        a,
        b,
        scale_bytes,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        scale_bytes.stride(0),
        scale_bytes.stride(1),
        output.stride(0),
        output.stride(1),
        block_m,
        block_n,
        num_warps=num_warps,
        num_stages=1,
    )
    return output
