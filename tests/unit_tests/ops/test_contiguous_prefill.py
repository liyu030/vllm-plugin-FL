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

from vllm_fl import envs
from vllm_fl.attention.contiguous_prefill import (
    can_use_contiguous_single_prefill,
    gather_contiguous_kv,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),
    ],
)
def test_contiguous_single_prefill_env(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_FL_METAX_CONTIGUOUS_SINGLE_PREFILL", value)
    assert envs.VLLM_FL_METAX_CONTIGUOUS_SINGLE_PREFILL is expected


def test_contiguous_single_prefill_env_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(
        "VLLM_FL_METAX_CONTIGUOUS_SINGLE_PREFILL",
        raising=False,
    )
    assert envs.VLLM_FL_METAX_CONTIGUOUS_SINGLE_PREFILL is False


@pytest.mark.parametrize(
    ("enabled", "num_prefills", "kv_cache_dtype", "expected"),
    [
        (True, 1, "auto", True),
        (True, 1, "bfloat16", True),
        (False, 1, "auto", False),
        (True, 2, "auto", False),
        (True, 1, "fp8", False),
        (True, 1, "fp8_e4m3", False),
    ],
)
def test_can_use_contiguous_single_prefill(
    enabled,
    num_prefills,
    kv_cache_dtype,
    expected,
):
    assert (
        can_use_contiguous_single_prefill(
            enabled,
            num_prefills,
            kv_cache_dtype,
        )
        is expected
    )


def test_gather_contiguous_kv_follows_block_table_order():
    key_cache = torch.arange(8).reshape(4, 2, 1, 1)
    value_cache = key_cache + 100
    block_table = torch.tensor([2, 0, 3], dtype=torch.int64)

    key, value = gather_contiguous_kv(
        key_cache,
        value_cache,
        block_table,
        seq_len=5,
    )

    torch.testing.assert_close(key.flatten(), torch.tensor([4, 5, 0, 1, 6]))
    torch.testing.assert_close(
        value.flatten(),
        torch.tensor([104, 105, 100, 101, 106]),
    )
    assert key.shape == (5, 1, 1)
    assert value.shape == (5, 1, 1)


def test_gather_contiguous_kv_handles_exact_block_boundary():
    key_cache = torch.arange(24).reshape(3, 2, 2, 2)
    value_cache = key_cache + 100
    block_table = torch.tensor([1, 2], dtype=torch.int32)

    key, value = gather_contiguous_kv(
        key_cache,
        value_cache,
        block_table,
        seq_len=4,
    )

    expected_key = torch.cat((key_cache[1], key_cache[2]), dim=0)
    expected_value = torch.cat((value_cache[1], value_cache[2]), dim=0)
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)


@pytest.mark.gpu
def test_gather_contiguous_kv_on_accelerator(device):
    key_cache = torch.arange(
        32,
        dtype=torch.bfloat16,
        device=device,
    ).reshape(4, 2, 2, 2)
    value_cache = key_cache + 100
    block_table = torch.tensor([3, 1, 0], dtype=torch.int32, device=device)

    key, value = gather_contiguous_kv(
        key_cache,
        value_cache,
        block_table,
        seq_len=5,
    )

    expected_key = torch.cat(
        (key_cache[3], key_cache[1], key_cache[0, :1]),
        dim=0,
    )
    expected_value = torch.cat(
        (value_cache[3], value_cache[1], value_cache[0, :1]),
        dim=0,
    )
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_value)
