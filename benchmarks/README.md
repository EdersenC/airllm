# Grouped AirLLM streaming benchmark

`benchmark_group_streaming.py` measures grouped layer streaming through the
current AirLLM API. Model construction and tokenization are outside the timed
region. Warmups run after model construction and are excluded from CSV/JSON.

The harness supports:

- `--model-path` for a local checkpoint/cache root or Hugging Face model id.
- `--group-size` (`--layers-per-gpu-group`) and `--prefetch-groups` for grouped
  streaming configuration. Use `--no-prefetch` for a no-prefetch baseline.
- `--prompt-file` with one prompt per non-empty UTF-8 line, or repeat `--prompt`
  and use `--prompt-repeats` for repeated prompt batches.
- `--prompt-batch-size`, `--max-new-tokens`, `--warmup`, and `--repeats`.
- CSV and JSON output. CSV has one row per timed prompt batch/repeat; JSON also
  includes the complete configuration and aggregate statistics.

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
  --group-size 2 \
  --prefetch-groups 2 \
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
The runtime timing columns separate CPU-prefetch wait, GPU materialization, and
grouped layer compute so each run shows whether the next optimization should
target disk/CPU I/O, host-to-device movement, or model compute.
