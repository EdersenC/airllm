#!/usr/bin/env python3
"""Benchmark grouped AirLLM layer streaming with repeatable prompt batches.

The benchmark keeps model loading outside the timed region. Warmups are run after
the model is loaded and are excluded from the result files. Each timed row is one
prompt batch in one repeat, so a prompt file can exercise several batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_PATH = Path("/mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ")
DEFAULT_PROMPT = "Explain why layer streaming can reduce peak GPU memory usage in one paragraph."
DEFAULT_CSV_PATH = Path("benchmarks/results.csv")
DEFAULT_JSON_PATH = Path("benchmarks/results.json")

CSV_FIELDS = (
    "repeat_index",
    "batch_index",
    "prompt_batch_size",
    "input_tokens",
    "generated_tokens",
    "tokens_per_sec",
    "time_to_first_token_s",
    "total_latency_s",
    "peak_vram_mb",
    "forward_passes",
    "groups_loaded",
    "group_cpu_wait_seconds",
    "group_gpu_load_seconds",
    "group_compute_seconds",
    "configuration",
)


def positive_int(value: str) -> int:
    """Argparse type for values that must be at least one."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    """Argparse type for values that may be zero but not negative."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local checkpoint/cache path or a Hugging Face model id.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device passed to AirLLM; 'auto' selects cuda:0 when available.",
    )
    parser.add_argument(
        "--group-size",
        "--layers-per-gpu-group",
        dest="group_size",
        type=positive_int,
        default=1,
        help="Consecutive decoder layers kept resident in one GPU group.",
    )
    parser.add_argument(
        "--prefetch-groups",
        type=positive_int,
        default=1,
        help="Number of upcoming groups loaded into CPU memory ahead of execution.",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable asynchronous group prefetching while retaining the configured count.",
    )

    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument(
        "--prompt",
        dest="prompt_values",
        action="append",
        help="Prompt text; repeat this option to provide several prompt variants.",
    )
    prompt_source.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 file containing one prompt per non-empty line.",
    )
    parser.add_argument(
        "--prompt-repeats",
        type=positive_int,
        default=1,
        help="Repeat the selected prompt source before forming batches.",
    )
    parser.add_argument(
        "--prompt-batch-size",
        type=positive_int,
        default=1,
        help="Number of prompts sent to generate in each timed batch.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=positive_int,
        default=512,
        help="Tokenizer truncation length for each prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=32,
        help="Maximum number of new tokens generated per prompt.",
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=1,
        help="Unrecorded generations on the first prompt batch after model load.",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=3,
        help="Number of recorded passes over all prompt batches.",
    )
    parser.add_argument(
        "--layer-shards-path",
        type=Path,
        default=None,
        help="Optional directory for AirLLM's streamed layer shards.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV output path; one row is written per timed batch/repeat.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="JSON output path with configuration, rows, and aggregate statistics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prompt batching and print the planned configuration without importing AirLLM.",
    )
    return parser


class FirstTokenTimer:
    """Minimal Transformers streamer interface used to timestamp the first token.

    Transformers calls ``put`` once for the input prompt and then for generated
    tokens. The first callback after that prompt callback is therefore a
    process-level TTFT measurement. No token data is copied or decoded.
    """

    def __init__(self) -> None:
        self._prompt_callback_seen = False
        self.first_token_time: float | None = None

    def put(self, _value: Any) -> None:
        if self.first_token_time is not None:
            return
        if not self._prompt_callback_seen:
            self._prompt_callback_seen = True
            return
        self.first_token_time = time.perf_counter()

    def end(self) -> None:
        return


def read_prompts(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.prompt_file is not None:
        prompt_file = args.prompt_file.expanduser()
        if not prompt_file.is_file():
            raise SystemExit(f"prompt file does not exist: {prompt_file}")
        prompts = [
            line.rstrip("\r\n")
            for line in prompt_file.read_text(encoding="utf-8").splitlines(keepends=True)
            if line.strip()
        ]
        source = f"file:{prompt_file}"
    elif args.prompt_values:
        prompts = list(args.prompt_values)
        source = "cli"
    else:
        prompts = [DEFAULT_PROMPT]
        source = "default"

    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise SystemExit("prompts must contain at least one non-empty value")
    return prompts * args.prompt_repeats, source


def build_prompt_batches(prompts: Iterable[str], batch_size: int) -> tuple[list[list[str]], int]:
    """Chunk prompts and pad only the final batch by cycling from the start."""

    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("at least one prompt is required")

    batches: list[list[str]] = []
    padded_count = 0
    for start in range(0, len(prompt_list), batch_size):
        batch = prompt_list[start:start + batch_size]
        missing = batch_size - len(batch)
        if missing:
            padded_count += missing
            batch.extend(prompt_list[index % len(prompt_list)] for index in range(missing))
        batches.append(batch)
    return batches, padded_count


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def tokenize_batches(tokenizer: Any, batches: list[list[str]], max_input_tokens: int) -> list[dict[str, Any]]:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("tokenizer has neither a pad token nor an EOS token for batching")
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only generation expects left padding when prompts in a batch have
    # different lengths. Qwen3 and the other AirLLM causal models use this path.
    tokenizer.padding_side = "left"
    encoded_batches = []
    for prompts in batches:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=max_input_tokens,
            padding=True,
        )
        encoded_batches.append(dict(encoded))
    return encoded_batches


def move_inputs(encoded: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def eos_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token_id) for token_id in value}


def count_generated_tokens(sequences: Any, input_width: int, tokenizer: Any) -> int:
    sequence_rows = sequences.detach().cpu().tolist()
    stop_ids = eos_ids(tokenizer)
    total = 0
    for row in sequence_rows:
        generated = row[input_width:]
        count = len(generated)
        if stop_ids:
            for index, token_id in enumerate(generated):
                if token_id in stop_ids:
                    count = index + 1
                    break
        total += count
    return total


def generate_once(
    torch: Any,
    model: Any,
    tokenizer: Any,
    encoded: dict[str, Any],
    device: Any,
    max_new_tokens: int,
) -> dict[str, float | int | None]:
    model_inputs = move_inputs(encoded, device)
    input_ids = model_inputs["input_ids"]
    input_width = int(input_ids.shape[-1])
    input_tokens = int(model_inputs["attention_mask"].sum().item())

    synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    streamer = FirstTokenTimer()
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": True,
        "streamer": streamer,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**model_inputs, **generation_kwargs)
    synchronize(torch, device)
    finished = time.perf_counter()

    sequences = getattr(output, "sequences", output)
    generated_tokens = count_generated_tokens(sequences, input_width, tokenizer)
    total_latency = finished - started
    first_token_time = streamer.first_token_time
    ttft = None if first_token_time is None else max(0.0, first_token_time - started)
    peak_vram = None
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    runtime_stats = model.get_runtime_stats() if hasattr(model, "get_runtime_stats") else {}
    return {
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": generated_tokens / total_latency if total_latency > 0 else None,
        "time_to_first_token_s": ttft,
        "total_latency_s": total_latency,
        "peak_vram_mb": peak_vram,
        "forward_passes": runtime_stats.get("forward_passes"),
        "groups_loaded": runtime_stats.get("groups_loaded"),
        "group_cpu_wait_seconds": runtime_stats.get("group_cpu_wait_seconds"),
        "group_gpu_load_seconds": runtime_stats.get("group_gpu_load_seconds"),
        "group_compute_seconds": runtime_stats.get("group_compute_seconds"),
    }


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = numeric_values(rows, key)
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], configuration: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    configuration_json = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            output_row = {field: row.get(field) for field in CSV_FIELDS}
            output_row["configuration"] = configuration_json
            writer.writerow(output_row)


def write_json(path: Path, rows: list[dict[str, Any]], configuration: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "configuration": configuration,
        "runs": rows,
        "summary": {
            "run_count": len(rows),
            "tokens_per_sec": summarize_metric(rows, "tokens_per_sec"),
            "time_to_first_token_s": summarize_metric(rows, "time_to_first_token_s"),
            "total_latency_s": summarize_metric(rows, "total_latency_s"),
            "peak_vram_mb": summarize_metric(rows, "peak_vram_mb"),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dry_run_report(args: argparse.Namespace, batches: list[list[str]], source: str, padded_count: int) -> None:
    report = {
        "model_path": str(args.model_path.expanduser()),
        "device_requested": args.device,
        "group_size": args.group_size,
        "prefetch_groups": args.prefetch_groups,
        "prefetching": not args.no_prefetch,
        "prompt_source": source,
        "prompt_batch_size": args.prompt_batch_size,
        "prompt_batch_count": len(batches),
        "padded_prompt_count": padded_count,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup,
        "repeats": args.repeats,
        "output_csv": str(args.output_csv),
        "output_json": str(args.output_json),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> int:
    args = build_parser().parse_args()
    prompts, prompt_source = read_prompts(args)
    batches, padded_count = build_prompt_batches(prompts, args.prompt_batch_size)

    if args.dry_run:
        dry_run_report(args, batches, prompt_source, padded_count)
        return 0

    try:
        import torch
        from airllm import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "AirLLM runtime dependencies are unavailable. Install the checkout first, "
            "for example: uv pip install --python .venv/bin/python -e './air_llm[awq]'"
        ) from exc

    device = resolve_device(torch, args.device)
    model_path = args.model_path.expanduser()
    print(f"model_path: {model_path}")
    print(f"device: {device}")
    print(f"group_size: {args.group_size}")
    print(f"prefetch_groups: {args.prefetch_groups}")
    print(f"prefetching: {not args.no_prefetch}")
    print(f"prompt_batches: {len(batches)} x {args.prompt_batch_size}")

    model_kwargs: dict[str, Any] = {
        "device": str(device),
        "layers_per_gpu_group": args.group_size,
        "prefetch_groups": args.prefetch_groups,
        "prefetching": not args.no_prefetch,
    }
    if args.layer_shards_path is not None:
        model_kwargs["layer_shards_saving_path"] = str(args.layer_shards_path.expanduser())
    model = AutoModel.from_pretrained(str(model_path), **model_kwargs)
    encoded_batches = tokenize_batches(model.tokenizer, batches, args.max_input_tokens)

    print(f"warmup_runs: {args.warmup}")
    for _ in range(args.warmup):
        generate_once(
            torch,
            model,
            model.tokenizer,
            encoded_batches[0],
            device,
            args.max_new_tokens,
        )

    configuration: dict[str, Any] = {
        "model_path": str(model_path),
        "device": str(device),
        "group_size": args.group_size,
        "prefetch_groups": args.prefetch_groups,
        "prefetching": not args.no_prefetch,
        "prompt_source": prompt_source,
        "prompt_repeats": args.prompt_repeats,
        "prompt_batch_size": args.prompt_batch_size,
        "prompt_batch_count": len(batches),
        "padded_prompt_count": padded_count,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup,
        "repeats": args.repeats,
        "layer_shards_path": (
            str(args.layer_shards_path.expanduser()) if args.layer_shards_path is not None else None
        ),
        "metrics": {
            "tokens_per_sec": "all non-EOS generated tokens across the batch divided by total latency",
            "time_to_first_token_s": "time until the first generation streamer callback after the prompt callback",
            "total_latency_s": "generation wall time, excluding model load and tokenization",
            "peak_vram_mb": "torch.cuda.max_memory_allocated for the selected device; null on CPU",
            "group_cpu_wait_seconds": "time spent waiting for each requested group to arrive from the CPU prefetch path",
            "group_gpu_load_seconds": "time spent materializing requested group weights on the GPU",
            "group_compute_seconds": "time spent executing grouped layer modules",
        },
    }

    rows: list[dict[str, Any]] = []
    for repeat_index in range(1, args.repeats + 1):
        for batch_index, encoded in enumerate(encoded_batches, start=1):
            measured = generate_once(
                torch,
                model,
                model.tokenizer,
                encoded,
                device,
                args.max_new_tokens,
            )
            row = {
                "repeat_index": repeat_index,
                "batch_index": batch_index,
                "prompt_batch_size": args.prompt_batch_size,
                **measured,
            }
            rows.append(row)
            ttft = "n/a" if row["time_to_first_token_s"] is None else f"{row['time_to_first_token_s']:.3f}s"
            peak = "n/a" if row["peak_vram_mb"] is None else f"{row['peak_vram_mb']:.1f}MB"
            throughput = "n/a" if row["tokens_per_sec"] is None else f"{row['tokens_per_sec']:.2f}"
            print(
                f"run {repeat_index}/{args.repeats} batch {batch_index}/{len(encoded_batches)}: "
                f"{throughput} tok/s, ttft {ttft}, "
                f"latency {row['total_latency_s']:.3f}s, peak_vram {peak}, "
                f"cpu_wait {row['group_cpu_wait_seconds']:.3f}s, "
                f"gpu_load {row['group_gpu_load_seconds']:.3f}s, "
                f"compute {row['group_compute_seconds']:.3f}s"
            )

    write_csv(args.output_csv, rows, configuration)
    write_json(args.output_json, rows, configuration)
    summary = summarize_metric(rows, "tokens_per_sec")
    print(f"tokens_per_sec_mean: {summary['mean']:.2f}" if summary["mean"] is not None else "tokens_per_sec_mean: n/a")
    print(f"csv: {args.output_csv}")
    print(f"json: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
