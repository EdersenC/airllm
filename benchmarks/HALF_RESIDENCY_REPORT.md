# Half-residency streaming performance report

Date: 2026-07-10  
GPU: NVIDIA GeForce RTX 5070 12 GB  
Model: Qwen3-4B-AWQ, snapshot `74d4bd2bd4bff9cafc9345221320bffb08b406a3`

## Required invariant

- All 36/36 checkpoint decoder layers execute in original order.
- No trained model layer is removed, skipped, replaced, or reused as a substitute.
- At most 18/36 decoder-layer weights may be on GPU simultaneously.
- With CUDA overlap enabled, group 9 is the largest valid group because the active
  9 layers and prefetched 9 layers coexist during transfer.
- CPU caching, transfer overlap, batching, and KV configuration may improve TPS;
  reducing model depth may not.

## Result

Changing streamed AWQ compute from the CPU-compatible ATen kernel to the CUDA
`AwqGEMMTritonLinear` kernel raised the warm batch-1 rate from 2.04 tok/s to
4.01 tok/s on the fixed-128-token test, a 1.97x improvement. The fixed-512-token
sustained test held 3.93 tok/s while dynamic KV grew.

| Workload | Generated tokens | Aggregate TPS | Peak VRAM |
| --- | ---: | ---: | ---: |
| Previous group-9 ATen baseline, batch 1 | 32 | 2.04 | 1,915 MiB |
| Triton, batch 1 | 128 | 4.01 mean | 1,715 MiB |
| Triton sustained, batch 1 | 512 | 3.93 | 1,785 MiB |
| Triton, batch 4 | 64 total | 14.79 | 1,715 MiB |
| Triton, batch 8 | 128 total | 28.84 | 1,737 MiB |
| Triton, batch 16 | 256 total | 57.96 | 1,781 MiB |
| Triton, batch 32 | 512 total | 120.10 | 1,871 MiB |
| Triton, batch 64 | 1,024 total | 258.31 | 2,047 MiB |

Batch rows use 16 forced new tokens per prompt and report aggregate throughput,
not per-request interactive TPS. Larger models and longer contexts will support
smaller batches because their weights and KV caches consume more VRAM.

## Cold path and quality

Two CPU prefetch workers, first-group priming, and a stable CPU-cache hot set
reduced the three-prompt cold probe from 8.33 seconds to 6.60 seconds. TTFT fell
from 7.78 seconds to 5.21 seconds, and measured CPU wait fell from 6.48 seconds
to 3.24 seconds.

The final default answered the quality probes exactly: `Paris`, `4`, and `Blue.`
The runtime printed `model_decoder_layers_executed: 36/36 (100%)` and
`configured_peak_gpu_decoder_layers: 18/36 (limit 18)`.

## Rejected alternatives

- Group 12 with CUDA double buffering is rejected because it can place 24/36
  decoder-layer weights on GPU, exceeding the 50% limit.
- Group 18 with synchronous copies reached 2.01 tok/s on the old kernel and was
  slightly slower than overlapped group 9.
- Streamed Marlin reached 3.83 tok/s after a packed-cache warmup, slower than
  Triton, and would repack repeatedly when a giant model exceeds the CPU cache.
- `torch_fused_awq` in this installed GPTQModel build is CPU-only and is not used.

## Reproduce the sustained run

```bash
./scripts/run_qwen3_awq_marlin.sh --benchmark \
  --max-input-tokens 512 \
  --max-new-tokens 512 \
  --min-new-tokens 512 \
  --warmup 1 \
  --warmup-new-tokens 16 \
  --repeats 1 \
  --output-csv benchmarks/results/streaming-triton-group9-batch1-512.csv \
  --output-json benchmarks/results/streaming-triton-group9-batch1-512.json
```
