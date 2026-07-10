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
- `--cache-implementation` for realistic dynamic, static, or offloaded KV caches.
- `--prompt-file` with one prompt per non-empty UTF-8 line, or repeat `--prompt`
  and use `--prompt-repeats` for repeated prompt batches.
- `--prompt-batch-size`, `--max-new-tokens`, `--warmup`, and `--repeats`.
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
  --cache-implementation static \
  --prompt "Explain why layer streaming reduces peak GPU memory." \
  --prompt-repeats 4 \
  --prompt-batch-size 2 \
  --max-new-tokens 64 \
  --warmup 1 \
  --repeats 3 \
  --output-csv benchmarks/results/qwen3-awq-group2-prefetch2.csv \
  --output-json benchmarks/results/qwen3-awq-group2-prefetch2.json
```

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
