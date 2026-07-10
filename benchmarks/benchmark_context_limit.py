#!/usr/bin/env python3
"""Stress a model at an exact total context length and record GPU/KV metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from airllm import AutoModel
try:
    from benchmarks.benchmark_group_streaming import (
        FirstTokenTimer,
        GPUStatsMonitor,
        collect_environment_metadata,
    )
except ImportError:
    from benchmark_group_streaming import (
        FirstTokenTimer,
        GPUStatsMonitor,
        collect_environment_metadata,
    )


DEFAULT_MODEL_PATH = Path("/mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "context-limit.json"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-tokens", type=positive_int, default=None,
                        help="Synthetic prompt length; default fills native context exactly")
    parser.add_argument("--max-new-tokens", type=positive_int, default=8)
    parser.add_argument("--synthetic-token-id", type=int, default=1000)
    parser.add_argument("--layers-per-gpu-group", type=positive_int, default=24)
    parser.add_argument("--prefetch-groups", type=positive_int, default=8)
    parser.add_argument("--cpu-layer-cache-gib", type=float, default=16.0)
    parser.add_argument("--awq-backend", default="marlin")
    parser.add_argument("--no-persistent-gpu-residency", action="store_true")
    parser.add_argument("--show-live-stats", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def resolve_input_tokens(native_context: int, requested_input: int | None, max_new_tokens: int) -> int:
    input_tokens = requested_input if requested_input is not None else native_context - max_new_tokens
    if input_tokens < 1:
        raise ValueError(
            f"derived input length {input_tokens} is invalid; max_new_tokens must be "
            f"smaller than native context {native_context}"
        )
    total_context = input_tokens + max_new_tokens
    if total_context > native_context:
        raise ValueError(
            f"requested total context {total_context} exceeds native limit {native_context}"
        )
    return input_tokens


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the context-limit benchmark")
    if args.cpu_layer_cache_gib < 0:
        raise SystemExit("--cpu-layer-cache-gib must be non-negative")
    if args.no_persistent_gpu_residency and args.awq_backend == "marlin":
        raise SystemExit("Marlin requires persistent GPU residency")

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(args.model_path.expanduser()),
        device=args.device,
        layers_per_gpu_group=args.layers_per_gpu_group,
        prefetch_groups=args.prefetch_groups,
        cpu_layer_cache_gib=args.cpu_layer_cache_gib,
        persistent_gpu_residency=not args.no_persistent_gpu_residency,
        awq_backend=args.awq_backend,
        show_live_stats=args.show_live_stats,
    )
    torch.cuda.synchronize(args.device)
    load_seconds = time.perf_counter() - load_started

    native_context = int(getattr(model.config, "max_position_embeddings", 0))
    if native_context < 1:
        model.close()
        raise SystemExit("model config does not define a positive max_position_embeddings")
    try:
        input_tokens = resolve_input_tokens(
            native_context, args.input_tokens, args.max_new_tokens)
    except ValueError as exc:
        model.close()
        raise SystemExit(str(exc)) from exc
    vocab_size = int(getattr(model.config, "vocab_size", 0))
    if args.synthetic_token_id < 0 or args.synthetic_token_id >= vocab_size:
        model.close()
        raise SystemExit(
            f"--synthetic-token-id must be in [0, {vocab_size - 1}]"
        )

    input_ids = torch.full(
        (1, input_tokens),
        args.synthetic_token_id,
        dtype=torch.long,
        device=args.device,
    )
    attention_mask = torch.ones_like(input_ids)
    streamer = FirstTokenTimer()
    monitor = GPUStatsMonitor(torch.device(args.device).index or 0)
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    monitor.start()
    status = "ok"
    error = None
    generated_tokens = 0

    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                cache_implementation="dynamic",
                disable_compile=True,
                pad_token_id=model.tokenizer.pad_token_id,
                streamer=streamer,
                return_dict_in_generate=True,
            )
        torch.cuda.synchronize(args.device)
        generated_tokens = int(output.sequences.shape[-1] - input_tokens)
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc)
    finally:
        finished = time.perf_counter()
        gpu_metrics = monitor.stop()

    first_token_seconds = (
        None if streamer.first_token_time is None else streamer.first_token_time - started
    )
    post_first_token_tps = None
    if first_token_seconds is not None and generated_tokens > 1:
        remaining_seconds = finished - streamer.first_token_time
        if remaining_seconds > 0:
            post_first_token_tps = (generated_tokens - 1) / remaining_seconds

    result: dict[str, Any] = {
        "status": status,
        "error": error,
        "model_path": str(args.model_path.expanduser()),
        "native_context": native_context,
        "input_tokens": input_tokens,
        "requested_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "total_context": input_tokens + generated_tokens,
        "load_seconds": load_seconds,
        "elapsed_seconds": finished - started,
        "time_to_first_token_seconds": first_token_seconds,
        "post_first_token_tps": post_first_token_tps,
        "peak_vram_mib": torch.cuda.max_memory_allocated(args.device) / (1024 ** 2),
        "configuration": {
            "layers_per_gpu_group": args.layers_per_gpu_group,
            "prefetch_groups": args.prefetch_groups,
            "cpu_layer_cache_gib": args.cpu_layer_cache_gib,
            "persistent_gpu_residency": not args.no_persistent_gpu_residency,
            "awq_backend": args.awq_backend,
            "cache_implementation": "dynamic",
        },
        "environment": collect_environment_metadata(
            torch,
            args.device,
            getattr(model, "model_local_path", args.model_path.expanduser()),
        ),
        "runtime_stats": model.get_runtime_stats(),
        **gpu_metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CONTEXT_LIMIT_RESULT=" + json.dumps(result, default=str, sort_keys=True))
    print(f"json: {args.output_json}")
    model.close()
    return 0 if status == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
