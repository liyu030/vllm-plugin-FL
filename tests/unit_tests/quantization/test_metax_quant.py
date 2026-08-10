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

import pytest
import torch

from vllm.platforms import current_platform

from vllm_fl.ops.metax_quant import (
    metax_fp8_block_gemm,
    metax_fp8_grouped_gemm,
    metax_mxfp4_gemm,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.vendor_name != "metax",
        reason="requires MetaX Triton",
    ),
]


def _fp8_weight(weight, scale):
    groups, n, k = weight.shape
    return (
        weight.view(groups, n // 128, 128, k // 128, 128).bfloat16()
        * scale.view(groups, n // 128, 1, k // 128, 1).bfloat16()
    ).reshape(groups, n, k)


def _mxfp4_weight(packed, scale):
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
        dtype=torch.bfloat16,
    )
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    scale = (scale.to(torch.int32) << 23).view(torch.float32).bfloat16()
    weight = values[codes.long()]
    return (weight.view(*weight.shape[:-1], -1, 32) * scale.unsqueeze(-1)).flatten(
        -2
    )


def test_fp8_block_gemm():
    torch.manual_seed(1)
    m, n, k = 13, 128, 256
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.rand(m, k // 128, device="cuda") + 0.5
    b_scale = torch.rand(n // 128, k // 128, device="cuda") + 0.5

    actual = metax_fp8_block_gemm(
        a, b, a_scale, b_scale, [128, 128], torch.bfloat16
    )
    a_ref = (
        a.view(m, k // 128, 128).bfloat16()
        * a_scale.view(m, k // 128, 1).bfloat16()
    ).reshape(m, k)
    expected = a_ref @ _fp8_weight(b.unsqueeze(0), b_scale.unsqueeze(0))[0].T

    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.5)


def test_fp8_grouped_gemm():
    torch.manual_seed(2)
    m, groups, n, k = 13, 2, 128, 256
    a = torch.randn(m, groups, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(groups, n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(groups, n // 128, k // 128, device="cuda") + 0.5

    actual = metax_fp8_grouped_gemm(a, b, scale)
    expected = torch.einsum("bgk,gnk->bgn", a, _fp8_weight(b, scale))

    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.5)


def test_mxfp4_gemm():
    torch.manual_seed(3)
    m, n, k = 13, 128, 256
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.randint(
        124,
        131,
        (n, k // 32),
        device="cuda",
        dtype=torch.uint8,
    )

    actual = metax_mxfp4_gemm(a, b, scale)
    expected = a @ _mxfp4_weight(b, scale).T

    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.5)
