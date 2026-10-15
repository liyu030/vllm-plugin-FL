# Qwen3-VL MoE tuning on MetaX C550

The packaged table targets Qwen3-VL-30B-A3B-Instruct with BF16 weights and
TP2 on MetaX C550 (128 experts, top-8 routing, expert weight shapes
`(128, 768, 2048)` and `(128, 2048, 384)`).

From the plugin checkout, before starting the service:

```bash
export VLLM_TUNED_CONFIG_FOLDER="$(pwd)/vllm_fl/configs/moe/metax"
vllm serve /models/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size 2 --dtype bfloat16
```

For a wheel installation, point the same variable at the installed
`vllm_fl/configs/moe/metax` directory.

vLLM's existing loader selects the standard filename and the nearest M entry.
M is the token row count of the current MoE computation, not the prompt length.
The table contains M=8, 32, 64, 1024, 7393 and 8192; use the same directory for
1K, 4K, 16K and 64K prompts. No additional loader or dispatch code is required.
The environment variable applies to the service process; set it for the
validated deployment above.
