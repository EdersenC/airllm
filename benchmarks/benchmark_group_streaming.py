#!/usr/bin/env python3
"""Benchmark grouped AirLLM layer streaming with repeatable prompt batches.

The benchmark keeps model loading outside the timed region. Warmups are run after
the model is loaded and are excluded from the result files. Each timed row is one
prompt batch in one repeat, so a prompt file can exercise several batches.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.layer_profiles import (  # noqa: E402 - keep dry-run independent of AirLLM imports
    LayerProfileError,
    load_layer_profile,
    parse_layer_indices,
    select_profile_layers,
    validate_profile_identity,
)


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
    "gpu_util_avg_pct",
    "gpu_util_p95_pct",
    "gpu_util_max_pct",
    "gpu_power_avg_w",
    "gpu_power_max_w",
    "gpu_temp_max_c",
    "forward_passes",
    "groups_loaded",
    "group_cpu_wait_seconds",
    "group_gpu_load_seconds",
    "group_copy_wait_seconds",
    "group_compute_seconds",
    "cuda_prefetched_groups",
    "resident_group_hits",
    "resident_groups",
    "quantized_kernel",
    "cpu_cache_hits",
    "cpu_cache_misses",
    "cpu_cache_evictions",
    "cpu_cache_gib",
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


def layer_indices_arg(value: str) -> list[int]:
    try:
        return parse_layer_indices(value)
    except LayerProfileError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
        "--decoder-layer-count",
        type=positive_int,
        default=None,
        help=(
            "Experimental reduced-depth mode: retain this many decoder layers. A supplied "
            "Block Influence profile chooses them; otherwise AirLLM samples evenly."
        ),
    )
    parser.add_argument(
        "--decoder-layer-indices",
        type=layer_indices_arg,
        default=None,
        help="Explicit comma-separated source-layer indices to retain.",
    )
    parser.add_argument(
        "--decoder-layer-profile",
        type=Path,
        default=None,
        help="Block Influence JSON used for layer ranking and its measured quality floor.",
    )
    parser.add_argument(
        "--allow-unsafe-layer-drop",
        action="store_true",
        help="Bypass a profile's quality floor for speed-only broken-output experiments.",
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
    parser.add_argument(
        "--no-cuda-copy-stream",
        action="store_true",
        help="Disable staging the next group on a dedicated CUDA copy stream.",
    )
    parser.add_argument(
        "--cpu-layer-cache-gib",
        type=float,
        default=4.0,
        help="Bounded pinned CPU-RAM cache for layer shards; zero disables retention.",
    )
    parser.add_argument(
        "--persistent-gpu-residency",
        action="store_true",
        help="Preload and retain the complete model when it fits in VRAM.",
    )
    parser.add_argument(
        "--awq-backend",
        choices=("auto", "marlin", "gemm_triton", "torch_awq", "torch_fused_awq"),
        default=None,
        help="Explicit GPTQModel AWQ kernel; Marlin requires persistent residency.",
    )
    parser.add_argument(
        "--cache-implementation",
        choices=("dynamic", "static", "offloaded", "offloaded_static"),
        default="dynamic",
        help="Transformers KV-cache implementation used during generation.",
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
        "--min-new-tokens",
        type=non_negative_int,
        default=0,
        help="Minimum generated tokens per prompt; set equal to max for fixed-length TPS runs.",
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


class GPUStatsMonitor:
    """Low-frequency nvidia-smi sampler for utilization, power, and temperature."""

    def __init__(self, device_index: int, interval: float = 0.25) -> None:
        self.device_index = device_index
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 4))
        if not self.samples:
            return {
                "gpu_util_avg_pct": None,
                "gpu_util_p95_pct": None,
                "gpu_util_max_pct": None,
                "gpu_power_avg_w": None,
                "gpu_power_max_w": None,
                "gpu_temp_max_c": None,
            }
        utilization = [sample[0] for sample in self.samples]
        power = [sample[1] for sample in self.samples]
        temperatures = [sample[2] for sample in self.samples]
        ordered_utilization = sorted(utilization)
        p95_index = min(len(ordered_utilization) - 1, int(0.95 * len(ordered_utilization)))
        return {
            "gpu_util_avg_pct": statistics.fmean(utilization),
            "gpu_util_p95_pct": ordered_utilization[p95_index],
            "gpu_util_max_pct": max(utilization),
            "gpu_power_avg_w": statistics.fmean(power),
            "gpu_power_max_w": max(power),
            "gpu_temp_max_c": max(temperatures),
        }

    def _run(self) -> None:
        command = (
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        )
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                values = [float(value.strip()) for value in result.stdout.strip().split(",")]
                if len(values) == 3:
                    self.samples.append((values[0], values[1], values[2]))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)


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


def _metadata_command(command: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_environment_metadata(torch: Any, device: Any, model_path: Any) -> dict[str, Any]:
    """Collect enough runtime identity to make benchmark JSON independently useful."""
    try:
        import transformers
        transformers_version = transformers.__version__
    except (ImportError, AttributeError):
        transformers_version = None
    try:
        gptqmodel_version = importlib.metadata.version("gptqmodel")
    except importlib.metadata.PackageNotFoundError:
        gptqmodel_version = None

    resolved_model_path = Path(model_path).expanduser().resolve()
    snapshot_revision = (
        resolved_model_path.name
        if resolved_model_path.parent.name == "snapshots" else None
    )
    git_status = _metadata_command(("git", "status", "--porcelain"))
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers_version,
        "gptqmodel_version": gptqmodel_version,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "git_revision": _metadata_command(("git", "rev-parse", "HEAD")),
        "git_branch": _metadata_command(("git", "rev-parse", "--abbrev-ref", "HEAD")),
        "git_dirty": git_status is not None,
        "resolved_model_path": str(resolved_model_path),
        "model_snapshot_revision": snapshot_revision,
        "device": str(device),
    }
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        device_index = torch_device.index or 0
        properties = torch.cuda.get_device_properties(device_index)
        metadata.update({
            "gpu_name": properties.name,
            "gpu_total_memory_mib": properties.total_memory / (1024 ** 2),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "nvidia_driver_version": _metadata_command((
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            )),
        })
    return metadata


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
                    count = index
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
    min_new_tokens: int,
    cache_implementation: str,
) -> dict[str, float | int | None]:
    model_inputs = move_inputs(encoded, device)
    input_ids = model_inputs["input_ids"]
    input_width = int(input_ids.shape[-1])
    input_tokens = int(model_inputs["attention_mask"].sum().item())

    synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    streamer = FirstTokenTimer()
    gpu_monitor = GPUStatsMonitor(device.index or 0) if device.type == "cuda" else None
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "cache_implementation": cache_implementation,
        # Group hooks materialize different parameters every forward. Static
        # cache may auto-compile, so disable compilation for every comparable run.
        "disable_compile": True,
        "return_dict_in_generate": True,
        "streamer": streamer,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if min_new_tokens:
        generation_kwargs["min_new_tokens"] = min_new_tokens

    started = time.perf_counter()
    if gpu_monitor is not None:
        gpu_monitor.start()
    try:
        with torch.inference_mode():
            output = model.generate(**model_inputs, **generation_kwargs)
        synchronize(torch, device)
        finished = time.perf_counter()
    finally:
        gpu_metrics = gpu_monitor.stop() if gpu_monitor is not None else {}

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
        **gpu_metrics,
        "forward_passes": runtime_stats.get("forward_passes"),
        "groups_loaded": runtime_stats.get("groups_loaded"),
        "group_cpu_wait_seconds": runtime_stats.get("group_cpu_wait_seconds"),
        "group_gpu_load_seconds": runtime_stats.get("group_gpu_load_seconds"),
        "group_copy_wait_seconds": runtime_stats.get("group_copy_wait_seconds"),
        "group_compute_seconds": runtime_stats.get("group_compute_seconds"),
        "cuda_prefetched_groups": runtime_stats.get("cuda_prefetched_groups"),
        "resident_group_hits": runtime_stats.get("resident_group_hits"),
        "resident_groups": runtime_stats.get("resident_groups"),
        "quantized_kernel": runtime_stats.get("quantized_kernel"),
        "cpu_cache_hits": runtime_stats.get("cpu_cache_hits"),
        "cpu_cache_misses": runtime_stats.get("cpu_cache_misses"),
        "cpu_cache_evictions": runtime_stats.get("cpu_cache_evictions"),
        "cpu_cache_gib": (
            runtime_stats.get("cpu_cache_bytes", 0) / (1024 ** 3)
            if runtime_stats else None
        ),
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
            "gpu_util_avg_pct": summarize_metric(rows, "gpu_util_avg_pct"),
            "gpu_power_avg_w": summarize_metric(rows, "gpu_power_avg_w"),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dry_run_report(args: argparse.Namespace, batches: list[list[str]], source: str, padded_count: int) -> None:
    report = {
        "model_path": str(args.model_path.expanduser()),
        "device_requested": args.device,
        "group_size": args.group_size,
        "decoder_layer_count": args.decoder_layer_count,
        "decoder_layer_indices": args.decoder_layer_indices,
        "decoder_layer_profile": (
            str(args.decoder_layer_profile) if args.decoder_layer_profile is not None else None
        ),
        "allow_unsafe_layer_drop": args.allow_unsafe_layer_drop,
        "prefetch_groups": args.prefetch_groups,
        "prefetching": not args.no_prefetch,
        "cuda_copy_stream": not args.no_cuda_copy_stream,
        "cpu_layer_cache_gib": args.cpu_layer_cache_gib,
        "persistent_gpu_residency": args.persistent_gpu_residency,
        "awq_backend": args.awq_backend,
        "cache_implementation": args.cache_implementation,
        "prompt_source": source,
        "prompt_batch_size": args.prompt_batch_size,
        "prompt_batch_count": len(batches),
        "padded_prompt_count": padded_count,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "warmup_runs": args.warmup,
        "repeats": args.repeats,
        "output_csv": str(args.output_csv),
        "output_json": str(args.output_json),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def resolve_decoder_layer_selection(args: argparse.Namespace) -> tuple[list[int] | None, dict | None]:
    if args.decoder_layer_count is None and args.decoder_layer_indices is None:
        return None, None

    selected_count = args.decoder_layer_count
    if args.decoder_layer_indices is not None:
        if selected_count is None:
            selected_count = len(args.decoder_layer_indices)
            args.decoder_layer_count = selected_count
        elif selected_count != len(args.decoder_layer_indices):
            raise LayerProfileError(
                "--decoder-layer-count must match --decoder-layer-indices length"
            )

    profile = load_layer_profile(args.decoder_layer_profile) \
        if args.decoder_layer_profile is not None else None
    if profile is not None:
        profile_indices = select_profile_layers(
            profile,
            selected_count,
            allow_unsafe=args.allow_unsafe_layer_drop,
        )
        if args.decoder_layer_indices is None:
            return profile_indices, profile
        if args.decoder_layer_indices != profile_indices and not args.allow_unsafe_layer_drop:
            raise LayerProfileError(
                "explicit layer indices differ from the calibrated profile; use "
                "--allow-unsafe-layer-drop to run an unvalidated selection"
            )
    return args.decoder_layer_indices, profile


def main() -> int:
    args = build_parser().parse_args()
    if args.min_new_tokens > args.max_new_tokens:
        raise SystemExit("--min-new-tokens cannot exceed --max-new-tokens")
    try:
        decoder_layer_indices, layer_profile = resolve_decoder_layer_selection(args)
    except LayerProfileError as exc:
        raise SystemExit(str(exc)) from exc
    args.decoder_layer_indices = decoder_layer_indices
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
    print(f"decoder_layer_count: {args.decoder_layer_count or 'all'}")
    print(f"decoder_layer_selection: {args.decoder_layer_indices or 'all/even'}")
    print(
        "decoder_layer_profile: "
        f"{layer_profile['profile_path'] if layer_profile is not None else 'none'}"
    )
    print(f"allow_unsafe_layer_drop: {args.allow_unsafe_layer_drop}")
    print(f"prefetch_groups: {args.prefetch_groups}")
    print(f"prefetching: {not args.no_prefetch}")
    print(f"cuda_copy_stream: {not args.no_cuda_copy_stream}")
    print(f"cpu_layer_cache_gib: {args.cpu_layer_cache_gib}")
    print(f"persistent_gpu_residency: {args.persistent_gpu_residency}")
    print(f"awq_backend: {args.awq_backend or 'checkpoint default'}")
    print(f"kv_cache: {args.cache_implementation}")
    print(f"prompt_batches: {len(batches)} x {args.prompt_batch_size}")

    model_kwargs: dict[str, Any] = {
        "device": str(device),
        "layers_per_gpu_group": args.group_size,
        "decoder_layer_count": args.decoder_layer_count,
        "decoder_layer_indices": args.decoder_layer_indices,
        "prefetch_groups": args.prefetch_groups,
        "prefetching": not args.no_prefetch,
        "cuda_copy_stream": not args.no_cuda_copy_stream,
        "cpu_layer_cache_gib": args.cpu_layer_cache_gib,
        "persistent_gpu_residency": args.persistent_gpu_residency,
        "awq_backend": args.awq_backend,
    }
    if args.layer_shards_path is not None:
        model_kwargs["layer_shards_saving_path"] = str(args.layer_shards_path.expanduser())
    model = AutoModel.from_pretrained(str(model_path), **model_kwargs)
    initial_runtime_stats = model.get_runtime_stats()
    if layer_profile is not None:
        try:
            validate_profile_identity(
                layer_profile,
                original_decoder_layer_count=initial_runtime_stats[
                    "original_decoder_layer_count"
                ],
                resolved_model_path=model.model_local_path,
            )
        except LayerProfileError as exc:
            model.close()
            raise SystemExit(str(exc)) from exc
    print(
        "active_decoder_layers: "
        f"{initial_runtime_stats['decoder_layer_count']}/"
        f"{initial_runtime_stats['original_decoder_layer_count']} "
        f"source_indices={initial_runtime_stats['decoder_layer_indices']}"
    )
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
            args.min_new_tokens,
            args.cache_implementation,
        )

    configuration: dict[str, Any] = {
        "model_path": str(model_path),
        "device": str(device),
        "group_size": args.group_size,
        "decoder_layer_count": initial_runtime_stats["decoder_layer_count"],
        "original_decoder_layer_count": initial_runtime_stats["original_decoder_layer_count"],
        "decoder_layer_indices": initial_runtime_stats["decoder_layer_indices"],
        "decoder_layer_selection": initial_runtime_stats["decoder_layer_selection"],
        "decoder_layer_profile": (
            layer_profile["profile_path"] if layer_profile is not None else None
        ),
        "allow_unsafe_layer_drop": args.allow_unsafe_layer_drop,
        "prefetch_groups": args.prefetch_groups,
        "prefetching": not args.no_prefetch,
        "cuda_copy_stream": not args.no_cuda_copy_stream,
        "cpu_layer_cache_gib": args.cpu_layer_cache_gib,
        "persistent_gpu_residency": args.persistent_gpu_residency,
        "awq_backend": args.awq_backend,
        "cache_implementation": args.cache_implementation,
        "prompt_source": prompt_source,
        "prompt_repeats": args.prompt_repeats,
        "prompt_batch_size": args.prompt_batch_size,
        "prompt_batch_count": len(batches),
        "padded_prompt_count": padded_count,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "warmup_runs": args.warmup,
        "repeats": args.repeats,
        "layer_shards_path": (
            str(args.layer_shards_path.expanduser()) if args.layer_shards_path is not None else None
        ),
        "environment": collect_environment_metadata(
            torch,
            device,
            getattr(model, "model_local_path", model_path),
        ),
        "metrics": {
            "tokens_per_sec": "all non-EOS generated tokens across the batch divided by total latency",
            "time_to_first_token_s": "time until the first generation streamer callback after the prompt callback",
            "total_latency_s": "generation wall time, excluding model load and tokenization",
            "peak_vram_mb": "torch.cuda.max_memory_allocated for the selected device; null on CPU",
            "gpu_util_avg_pct": "mean nvidia-smi GPU utilization sampled every 250 ms",
            "gpu_util_p95_pct": "95th percentile sampled GPU utilization",
            "gpu_power_avg_w": "mean sampled board power draw",
            "gpu_temp_max_c": "maximum sampled GPU temperature",
            "group_cpu_wait_seconds": "time spent waiting for each requested group to arrive from the CPU prefetch path",
            "group_gpu_load_seconds": "time spent materializing requested group weights on the GPU",
            "group_copy_wait_seconds": "handoff time blocked waiting for the CUDA copy stream",
            "group_compute_seconds": "time spent executing grouped layer modules",
            "cpu_cache_gib": "current bounded CPU layer-cache payload in GiB",
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
                args.min_new_tokens,
                args.cache_implementation,
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
            utilization = (
                "n/a" if row["gpu_util_avg_pct"] is None
                else f"{row['gpu_util_avg_pct']:.1f}% avg/{row['gpu_util_p95_pct']:.0f}% p95"
            )
            throughput = "n/a" if row["tokens_per_sec"] is None else f"{row['tokens_per_sec']:.2f}"
            print(
                f"run {repeat_index}/{args.repeats} batch {batch_index}/{len(encoded_batches)}: "
                f"{throughput} tok/s, ttft {ttft}, "
                f"latency {row['total_latency_s']:.3f}s, peak_vram {peak}, "
                f"gpu_util {utilization}, "
                f"cpu_wait {row['group_cpu_wait_seconds']:.3f}s, "
                f"gpu_load {row['group_gpu_load_seconds']:.3f}s, "
                f"copy_wait {row['group_copy_wait_seconds']:.3f}s, "
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
