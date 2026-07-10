# Grouped AirLLM streaming benchmark

`benchmark_group_streaming.py` measures grouped layer streaming through the
current AirLLM API. Model construction and tokenization are outside the timed
region. Warmups run after model construction and are excluded from CSV/JSON.

The harness supports:

- `--model-path` for a local checkpoint/cache root or Hugging Face model id.
- `--group-size` (`--layers-per-gpu-group`) and `--prefetch-groups` for grouped
  streaming configuration. Use `--no-prefetch` for a no-prefetch baseline.
- `--cpu-layer-cache-gib` for the bounded pinned-RAM layer cache and
  `--no-cuda-copy-stream` for a synchronous-transfer control.
- `--persistent-gpu-residency` to preload and retain every group when the full
  model fits, plus `--awq-backend` for an explicit AWQ kernel such as Marlin.
- `--cache-implementation` for realistic dynamic, static, or offloaded KV caches.
- `--decoder-layer-count`, `--decoder-layer-profile`, and
  `--decoder-layer-indices` for reduced-depth runs. A profile ranks blocks by
  measured influence and can enforce a tested quality floor; the JSON records the
  profile and exact source indices.
- `--prompt-file` with one prompt per non-empty UTF-8 line, or repeat `--prompt`
  and use `--prompt-repeats` for repeated prompt batches.
- `--prompt-batch-size`, `--max-new-tokens`, `--min-new-tokens`, `--warmup`, and
  `--repeats`. Set minimum equal to maximum to prevent early EOS from shortening
  a throughput comparison.
- CSV and JSON output with TPS, TTFT, latency, GPU utilization, power,
  temperature, VRAM, cache hits, CPU wait, CUDA-copy wait, and compute time.

## Local Qwen3 AWQ command

From `/home/eddy/Projects/airllm`, install the checkout into an isolated Python
environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e './air_llm[awq]'
```

Run a grouped Qwen3 AWQ benchmark using the local cache root. AirLLM resolves
the cache root to its current snapshot automatically:

```bash
.venv/bin/python benchmarks/benchmark_group_streaming.py \
  --model-path /mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ \
  --device cuda:0 \
  --group-size 24 \
  --prefetch-groups 2 \
  --cpu-layer-cache-gib 4 \
  --cache-implementation dynamic \
  --prompt "Explain why layer streaming reduces peak GPU memory." \
  --prompt-repeats 4 \
  --prompt-batch-size 2 \
  --max-new-tokens 64 \
  --warmup 1 \
  --repeats 3 \
  --output-csv benchmarks/results/qwen3-awq-group2-prefetch2.csv \
  --output-json benchmarks/results/qwen3-awq-group2-prefetch2.json
```

For the fastest tested full-resident path (after installing the CUDA 13.0
toolchain listed in the top-level README), add persistent residency and Marlin:

```bash
./scripts/run_qwen3_awq_marlin.sh --max-new-tokens 64 --no-live-stats

.venv/bin/python benchmarks/benchmark_group_streaming.py \
  --model-path /mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ \
  --device cuda:0 \
  --group-size 24 \
  --prefetch-groups 8 \
  --cpu-layer-cache-gib 16 \
  --persistent-gpu-residency \
  --awq-backend marlin \
  --cache-implementation dynamic \
  --max-new-tokens 64 \
  --warmup 1 \
  --repeats 3
```

The prepared launcher now defaults to the large-model simulation: 12 decoder
layers resident per GPU group, all 36 model layers executed, persistent full-model
residency disabled, 8-group prefetch, and a 16 GiB CPU layer-cache budget.

```bash
./scripts/run_qwen3_awq_marlin.sh --benchmark \
  --max-new-tokens 128 \
  --min-new-tokens 128 \
  --warmup 1 \
  --repeats 3
```

`--layers-per-gpu-group` controls simultaneous GPU residency. It does not prune
or skip model layers.

### Research-only model-depth pruning

To compare the full 36-layer checkpoint against half depth and one-third depth,
run the same benchmark three times and force an equal decode length. This is a
speed-only experiment; `--allow-unsafe-layer-drop` is required because the half
and third-depth stacks are known to generate broken text:

```bash
for layers in 36 18 12; do
  ./scripts/run_qwen3_awq_marlin.sh --benchmark \
    --decoder-layer-count "${layers}" \
    --allow-unsafe-layer-drop \
    --max-new-tokens 128 \
    --min-new-tokens 128 \
    --warmup 1 \
    --repeats 3 \
    --output-csv "benchmarks/results/reduced-depth-${layers}.csv" \
    --output-json "benchmarks/results/reduced-depth-${layers}.json"
done
```

The selected source-layer indices are printed at startup. Reduced depth is an
inference experiment, not a distilled checkpoint: higher TPS does not imply that
the resulting text preserves the original model's quality.

The correction follows the Block Influence metric from
[ShortGPT](https://aclanthology.org/2025.findings-acl.1035/): measure cosine
distance between each block's input and output, then remove the least influential
blocks first. ShortGPT's generative variant sends generated tokens through every
layer to avoid accumulated decode errors, while NVIDIA's
[Minitron report](https://arxiv.org/abs/2408.11796) uses distillation to recover
quality after large structural reductions. Accordingly, this fork treats
50–67% untrained depth removal as an unsafe benchmark, not a usable inference
configuration.

Regenerate the local influence scores through the prepared CUDA/Marlin
environment:

```bash
./scripts/run_qwen3_awq_marlin.sh --calibrate-layer-profile \
  --output-json benchmarks/results/qwen3-4b-awq-block-influence.json
```

### Reduced-depth Qwen3-4B-AWQ result

On the RTX 5070, a clean-revision comparison used batch 1, persistent Marlin,
dynamic KV cache, one warmup, three measured runs, and exactly 128 generated
tokens per run (`min_new_tokens == max_new_tokens`).

| Active decoder layers | Mean throughput | Speedup vs. 36 | Peak allocation | Prompt sanity |
| ---: | ---: | ---: | ---: | --- |
| 36 / 36 | 10.60 tok/s | 1.00x | 2,611 MiB | Passed (`Paris`, `4`, `blue`) |
| 31 / 36, Block Influence | **13.61 tok/s** | **1.28x** | **2,355 MiB** | Passed (`Paris`, `4`, `blue`) |
| 18 / 36 | 22.51 tok/s | 2.12x | 1,691 MiB | Failed simple factual prompts |
| 12 / 36 | **28.19 tok/s** | **2.66x** | **1,385 MiB** | Failed simple factual prompts |

The 12-layer configuration remains the raw-TPS winner, but neither aggressive
stack is a usable replacement for the original checkpoint without distillation.
The 31-layer profile is the fastest configuration that passed the basic output
sanity check. At 30 layers the answers were still factually recognizable but
became verbose and repetitive, so the prepared profile conservatively requires
31. The measured source indices were:

- safe 31-layer profile: `0-28,34,35`
- historical even 18-layer speed run: `0,2,4,6,8,10,12,14,16,19,21,23,25,27,29,31,33,35`
- historical even 12-layer speed run: `0,3,6,10,13,16,19,22,25,29,32,35`

On the tested 12 GB RTX 5070, the current branch measured the following local
results. Generated result files remain gitignored by repository policy; rerun
the commands above to regenerate them. New JSON/CSV configuration records include
git revision/dirty state, model snapshot, GPU/driver, CUDA, Python, PyTorch,
Transformers, and GPTQModel versions.

| Workload | Throughput | Peak allocation |
| --- | ---: | ---: |
| Group-24 streaming, batch 1, 64-token dynamic run | 1.60 tok/s | 2,824 MiB |
| Persistent Marlin, batch 1, warm 64-token runs | 11.59 tok/s | 2,599 MiB |
| Persistent Marlin, batch 4, warm 64-token runs | 44.68 aggregate tok/s | 2,637 MiB |
| 40,952-token prefill + 8-token decode | 40,960 total context | 11,415 MiB |

The clean-revision max-context run averaged 92.5% GPU utilization and reached 21.61s TTFT.
It leaves almost no VRAM margin, so keep batch size at 1 near the native 40,960
token limit. At short context, batch 4 converts spare compute into throughput
without materially increasing per-sequence latency.
For the warm 64-token batch-1 comparison, dynamic KV reached 11.59 tok/s versus
10.70 tok/s with static KV, so the prepared launcher intentionally defaults to dynamic.

Reproduce the exact native-context test through the Marlin launcher:

```bash
./scripts/run_qwen3_awq_marlin.sh --context-limit-benchmark
```

By default, `benchmark_context_limit.py` reads `max_position_embeddings`, creates
a synthetic prompt of `native_context - 8` tokens, forces 8 decode tokens, and
writes `benchmarks/results/context-limit.json`. Override `--input-tokens`,
`--max-new-tokens`, or `--output-json` after the benchmark switch when needed.

To use a prompt file instead, pass one prompt per non-empty line:

```bash
.venv/bin/python benchmarks/benchmark_group_streaming.py \
  --model-path /mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ \
  --device cuda:0 \
  --group-size 1 \
  --prefetch-groups 1 \
  --prompt-file /absolute/path/prompts.txt \
  --prompt-batch-size 4 \
  --max-new-tokens 32 \
  --warmup 1 \
  --repeats 3 \
  --output-csv benchmarks/results/qwen3-awq-file.csv \
  --output-json benchmarks/results/qwen3-awq-file.json
```

If the prompt count is not divisible by `--prompt-batch-size`, the final batch
is padded by cycling from the beginning so every measured row has the requested
batch size. `padded_prompt_count` records how many prompts were added this way.

## Reading the results

`tokens_per_sec` counts all non-EOS generated tokens across the batch divided by
generation wall time. `time_to_first_token_s` is measured from a generation
streamer callback and is `null` when the backend provides no measurable callback.
`total_latency_s` excludes model load and tokenization. `peak_vram_mb` uses
`torch.cuda.max_memory_allocated` for the selected device and is `null` on CPU.
The runtime timing columns separate CPU-prefetch wait, GPU materialization,
copy-stream handoff wait, and grouped-layer compute. GPU utilization, board
power, and temperature are sampled through `nvidia-smi` every 250 ms.
Persistent runs also report resident-group count and hit count. Their per-token
CPU wait, GPU weight-load time, and copy wait should remain zero after preload.

## Guarded stress matrix

Run a short validation matrix:

```bash
.venv/bin/python benchmarks/run_stress_matrix.py --preset quick
```

Run the longer group-size, batch-size, copy-stream, and KV-cache comparison:

```bash
.venv/bin/python benchmarks/run_stress_matrix.py --preset overnight
```

Each case runs in a fresh process with a timeout. The runner pauses when
available RAM is below 2 GiB or the GPU is above 78 C, aborts an active case if
RAM falls below 1 GiB or the GPU exceeds 86 C, continues after failed cases, and
writes an ignored Markdown report under `benchmarks/results/`.
