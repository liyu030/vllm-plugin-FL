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

from __future__ import annotations

import torch


def can_use_contiguous_single_prefill(
    enabled: bool,
    num_prefills: int,
    kv_cache_dtype: str,
) -> bool:
    """Return whether the contiguous single-prefill path is applicable."""
    return enabled and num_prefills == 1 and not kv_cache_dtype.startswith("fp8")


def gather_contiguous_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one paged KV sequence into token-contiguous tensors."""
    block_size = key_cache.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    block_ids = block_table[:num_blocks]
    contiguous_key = torch.index_select(key_cache, 0, block_ids)
    contiguous_value = torch.index_select(value_cache, 0, block_ids)

    num_kv_heads = key_cache.shape[2]
    head_size = key_cache.shape[3]
    contiguous_key = contiguous_key.reshape(-1, num_kv_heads, head_size)[:seq_len]
    contiguous_value = contiguous_value.reshape(-1, num_kv_heads, head_size)[:seq_len]
    return contiguous_key, contiguous_value
