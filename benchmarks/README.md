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
- `--cpu-prefetch-workers` for parallel shard staging and
  `--cpu-layer-cache-policy static` for a scan-resistant hot set.
- `--max-gpu-layer-fraction` for a hard decoder-weight residency budget that
  includes both active and CUDA-prefetched groups.
- `--cache-implementation` for realistic dynamic, static, or offloaded KV caches.
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
  --group-size 9 \
  --max-gpu-layer-fraction 0.5 \
  --prefetch-groups 8 \
  --cpu-layer-cache-gib 16 \
  --awq-backend gemm_triton \
  --cache-implementation dynamic \
  --prompt "Explain why layer streaming reduces peak GPU memory." \
  --prompt-repeats 4 \
  --prompt-batch-size 2 \
  --max-new-tokens 64 \
  --warmup 1 \
  --repeats 3 \
  --output-csv benchmarks/results/qwen3-awq-group9-prefetch8.csv \
  --output-json benchmarks/results/qwen3-awq-group9-prefetch8.json
```

The prepared launcher defaults to the large-model target: 9 decoder layers per
group, at most 18/36 decoder-layer weights on GPU after accounting for CUDA
double buffering, all 36 model layers executed, 8-group CPU prefetch, and a
16 GiB CPU layer-cache budget.

```bash
./scripts/run_qwen3_awq_marlin.sh --benchmark \
  --max-new-tokens 128 \
  --min-new-tokens 128 \
  --warmup 1 \
  --repeats 3
```

`--layers-per-gpu-group` controls grouping only. Every checkpoint layer always
executes, and the benchmark records both the model's total decoder count and the
configured peak GPU-resident decoder count.

On the tested 12 GB RTX 5070, the current branch measured the following local
results. Generated result files remain gitignored by repository policy; rerun
the commands above to regenerate them. New JSON/CSV configuration records include
git revision/dirty state, model snapshot, GPU/driver, CUDA, Python, PyTorch,
Transformers, and GPTQModel versions.

| Valid half-budget workload | Throughput | Peak allocation |
| --- | ---: | ---: |
| Previous ATen group-9 baseline, batch 1, 32 tokens | 2.04 tok/s | 1,915 MiB |
| Triton group 9, batch 1, 128 tokens | **4.01 tok/s** | 1,715 MiB |
| Triton group 9, batch 1, 512 tokens | 3.93 tok/s | 1,785 MiB |
| Triton group 9, batch 8, 16 tokens each | 28.84 aggregate tok/s | 1,737 MiB |
| Triton group 9, batch 64, 16 tokens each | **258.31 aggregate tok/s** | 2,047 MiB |

Every row executes 36/36 layers and caps simultaneous decoder weights at 18/36.
See [HALF_RESIDENCY_REPORT.md](HALF_RESIDENCY_REPORT.md) for the complete batch
sweep, cold-start comparison, quality probes, rejected alternatives, and exact
reproduction command.

Reproduce the exact native-context test through the prepared CUDA launcher:

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
