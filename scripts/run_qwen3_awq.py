#!/usr/bin/env python3
"""Run a small local-generation smoke test against a Qwen3 AWQ checkpoint."""

import argparse
import time
from pathlib import Path

import torch

from airllm import AutoModel


DEFAULT_MODEL_CACHE = Path("/mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ")
DEFAULT_PROMPT = "What is the capital of France?"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE,
                        help="Hugging Face cache root or snapshot directory")
    parser.add_argument("--prompt", action="append", dest="prompts",
                        help="Prompt to include; repeat for a prompt batch")
    parser.add_argument("--prompt-file", type=Path,
                        help="UTF-8 file with one prompt per non-empty line")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Prompts per generate call; 0 means all prompts")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=512,
                        help="Tokenizer truncation limit for each prompt")
    parser.add_argument("--layers-per-gpu-group", type=int, default=1,
                        help="Consecutive decoder layers to keep on GPU at once")
    parser.add_argument("--prefetch-groups", type=int, default=1,
                        help="Upcoming GPU groups to cache in CPU memory")
    parser.add_argument("--cpu-layer-cache-gib", type=float, default=4.0,
                        help="Bounded pinned-RAM cache for layer shards")
    parser.add_argument("--persistent-gpu-residency", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Preload and retain the complete model when it fits in VRAM")
    parser.add_argument(
        "--awq-backend",
        choices=("auto", "marlin", "gemm_triton", "torch_awq", "torch_fused_awq"),
        default=None,
        help="Explicit GPTQModel AWQ kernel; Marlin requires persistent residency",
    )
    parser.add_argument("--no-cuda-copy-stream", action="store_true",
                        help="Disable background CPU-to-GPU copies")
    parser.add_argument(
        "--cache-implementation",
        choices=("dynamic", "static", "offloaded", "offloaded_static"),
        default="dynamic",
        help="Transformers KV-cache implementation",
    )
    parser.add_argument("--no-live-stats", action="store_true",
                        help="Disable live layer/group progress output")
    parser.add_argument("--live-stats-interval", type=float, default=0.25,
                        help="Minimum seconds between non-critical live updates")
    parser.add_argument("--layer-shards-path", type=Path, default=None,
                        help="Optional directory for AirLLM's streamed layer shards")
    return parser.parse_args()


def load_prompts(args):
    prompts = list(args.prompts or [])
    if args.prompt_file:
        prompts.extend(
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return prompts or [DEFAULT_PROMPT]


class FirstTokenTimer:
    def __init__(self):
        self.prompt_seen = False
        self.first_token_at = None

    def put(self, _value):
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def end(self):
        return


def count_generated_tokens(sequences, input_width, tokenizer):
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        stop_ids = set()
    elif isinstance(eos_token_id, int):
        stop_ids = {eos_token_id}
    else:
        stop_ids = {int(token_id) for token_id in eos_token_id}

    counts = []
    for row in sequences.detach().cpu().tolist():
        generated = row[input_width:]
        count = len(generated)
        for index, token_id in enumerate(generated):
            if token_id in stop_ids:
                count = index
                break
        counts.append(count)
    return counts


def run_batch(model, prompts, args, batch_number, prompt_offset):
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("Prompt batching requires a tokenizer pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=args.max_input_tokens,
        padding=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    streamer = FirstTokenTimer()
    started = time.perf_counter()
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        cache_implementation=args.cache_implementation,
        # AirLLM swaps meta/GPU weights from Python hooks every forward. Static
        # cache may auto-compile, which cannot capture those mutations safely.
        disable_compile=True,
        streamer=streamer,
        return_dict_in_generate=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    input_tokens = int(attention_mask.sum().item())
    generated_counts = count_generated_tokens(output.sequences, input_ids.shape[-1], tokenizer)
    total_generated_tokens = sum(generated_counts)
    decoded = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    peak_vram_mb = None
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    runtime_stats = model.get_runtime_stats()

    print(
        f"batch={batch_number} prompts={len(prompts)} input_tokens={input_tokens} "
        f"generated_tokens={total_generated_tokens} "
        f"elapsed_seconds={elapsed:.2f} tokens_per_second={total_generated_tokens / elapsed:.2f} "
        f"per_prompt_tps={total_generated_tokens / elapsed / len(prompts):.2f} "
        f"time_to_first_token_seconds="
        f"{None if streamer.first_token_at is None else round(streamer.first_token_at - started, 3)} "
        f"forward_passes={runtime_stats['forward_passes']} "
        f"groups_loaded={runtime_stats['groups_loaded']} "
        f"cpu_wait_seconds={runtime_stats['group_cpu_wait_seconds']:.2f} "
        f"gpu_load_seconds={runtime_stats['group_gpu_load_seconds']:.2f} "
        f"copy_wait_seconds={runtime_stats['group_copy_wait_seconds']:.2f} "
        f"group_compute_seconds={runtime_stats['group_compute_seconds']:.2f}"
        f" cuda_prefetched_groups={runtime_stats['cuda_prefetched_groups']} "
        f"resident_group_hits={runtime_stats['resident_group_hits']} "
        f"resident_groups={runtime_stats['resident_groups']} "
        f"awq_backend={runtime_stats['awq_backend']} "
        f"quantized_kernel={runtime_stats['quantized_kernel']} "
        f"cpu_cache_hits={runtime_stats['cpu_cache_hits']} "
        f"cpu_cache_gib={runtime_stats['cpu_cache_bytes'] / (1024 ** 3):.2f}"
        + (f" peak_vram_mb={peak_vram_mb:.0f}" if peak_vram_mb is not None else "")
    )
    for index, text in enumerate(decoded):
        print(f"prompt[{prompt_offset + index}]: {text}")

    return {
        "batch": batch_number,
        "prompts": len(prompts),
        "input_tokens": input_tokens,
        "generated_tokens": total_generated_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_generated_tokens / elapsed,
        "per_prompt_tps": total_generated_tokens / elapsed / len(prompts),
        "time_to_first_token_seconds": (
            None if streamer.first_token_at is None else streamer.first_token_at - started
        ),
        "peak_vram_mb": peak_vram_mb,
    }


def main():
    args = parse_args()
    if not args.model_cache.exists():
        raise SystemExit(f"Model path does not exist: {args.model_cache}")
    if args.batch_size < 0:
        raise SystemExit("--batch-size must be non-negative")
    if args.max_input_tokens < 1:
        raise SystemExit("--max-input-tokens must be positive")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(args)
    batch_size = args.batch_size or len(prompts)
    print(f"model: {args.model_cache}")
    print(f"device: {device}")
    print(f"layers_per_gpu_group: {args.layers_per_gpu_group}")
    print(f"prefetch_groups: {args.prefetch_groups}")
    print(f"cuda_copy_stream: {not args.no_cuda_copy_stream}")
    print(f"cpu_layer_cache_gib: {args.cpu_layer_cache_gib}")
    print(f"persistent_gpu_residency: {args.persistent_gpu_residency}")
    print(f"awq_backend: {args.awq_backend or 'checkpoint default'}")
    print(f"kv_cache: {args.cache_implementation}")
    print(f"max_input_tokens: {args.max_input_tokens}")
    print(f"prompt_count: {len(prompts)} batch_size: {batch_size}")

    model = AutoModel.from_pretrained(
        str(args.model_cache),
        device=device,
        layers_per_gpu_group=args.layers_per_gpu_group,
        prefetch_groups=args.prefetch_groups,
        cuda_copy_stream=not args.no_cuda_copy_stream,
        cpu_layer_cache_gib=args.cpu_layer_cache_gib,
        persistent_gpu_residency=args.persistent_gpu_residency,
        awq_backend=args.awq_backend,
        show_live_stats=not args.no_live_stats,
        live_stats_interval=args.live_stats_interval,
        layer_shards_saving_path=str(args.layer_shards_path) if args.layer_shards_path else None,
    )
    results = []
    for offset in range(0, len(prompts), batch_size):
        results.append(run_batch(
            model,
            prompts[offset:offset + batch_size],
            args,
            batch_number=len(results) + 1,
            prompt_offset=offset,
        ))

    total_elapsed = sum(result["elapsed_seconds"] for result in results)
    total_tokens = sum(result["generated_tokens"] for result in results)
    print(
        f"total_batches={len(results)} total_tokens={total_tokens} "
        f"total_elapsed_seconds={total_elapsed:.2f} "
        f"aggregate_tokens_per_second={total_tokens / total_elapsed:.2f}"
    )


if __name__ == "__main__":
    main()
