#!/usr/bin/env python3
"""Run guarded AirLLM throughput cases and produce a Markdown performance report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "benchmark_group_streaming.py"
DEFAULT_MODEL = Path("/mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ")


@dataclass(frozen=True)
class Case:
    name: str
    group: int
    batch: int
    tokens: int
    cache: str = "static"
    cpu_cache_gib: float = 4.0
    prefetch_groups: int = 2
    copy_stream: bool = True


QUICK_CASES = (
    Case("group8-static", group=8, batch=1, tokens=16),
    Case("group24-static", group=24, batch=1, tokens=32),
    Case("group24-batch2", group=24, batch=2, tokens=32),
)

OVERNIGHT_CASES = (
    Case("baseline-group2-no-cache", group=2, batch=1, tokens=16,
         cache="dynamic", cpu_cache_gib=0.0, copy_stream=False),
    Case("group2-static", group=2, batch=1, tokens=64),
    Case("group8-static", group=8, batch=1, tokens=64),
    Case("group16-static", group=16, batch=1, tokens=64),
    Case("group24-dynamic", group=24, batch=1, tokens=64, cache="dynamic"),
    Case("group24-static-copy-off", group=24, batch=1, tokens=64, copy_stream=False),
    Case("group24-static", group=24, batch=1, tokens=64),
    Case("group24-static-batch2", group=24, batch=2, tokens=64),
    Case("group24-static-batch4", group=24, batch=4, tokens=64),
    Case("group24-offloaded-kv", group=24, batch=2, tokens=64, cache="offloaded"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--preset", choices=("quick", "overnight"), default="quick")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-timeout", type=int, default=1200, help="Seconds per case")
    parser.add_argument("--min-free-ram-gib", type=float, default=2.0)
    parser.add_argument("--max-start-temp-c", type=float, default=78.0)
    parser.add_argument("--abort-free-ram-gib", type=float, default=1.0)
    parser.add_argument("--abort-temp-c", type=float, default=86.0)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    return parser.parse_args()


def free_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0])
    return values["MemAvailable"] / (1024 ** 2)


def gpu_snapshot() -> dict[str, float] | None:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--id=0",
                "--query-gpu=temperature.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        temperature, used, total, power = (
            float(value.strip()) for value in result.stdout.strip().split(",")
        )
        return {"temp_c": temperature, "used_mib": used, "total_mib": total, "power_w": power}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def wait_until_safe(args: argparse.Namespace) -> tuple[bool, str]:
    deadline = time.monotonic() + 300
    while True:
        ram = free_ram_gib()
        gpu = gpu_snapshot()
        temperature = gpu["temp_c"] if gpu else 0.0
        if ram >= args.min_free_ram_gib and temperature <= args.max_start_temp_c:
            return True, f"available_ram={ram:.1f}GiB gpu_temp={temperature:.0f}C"
        if time.monotonic() >= deadline:
            return False, f"safety gate timed out: available_ram={ram:.1f}GiB gpu_temp={temperature:.0f}C"
        time.sleep(10)


def run_case(case: Case, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    json_path = output_dir / f"{case.name}.json"
    csv_path = output_dir / f"{case.name}.csv"
    log_path = output_dir / f"{case.name}.log"
    command = [
        sys.executable,
        str(BENCHMARK),
        "--model-path", str(args.model_path),
        "--device", "cuda:0",
        "--group-size", str(case.group),
        "--prefetch-groups", str(case.prefetch_groups),
        "--cpu-layer-cache-gib", str(case.cpu_cache_gib),
        "--cache-implementation", case.cache,
        "--prompt-batch-size", str(case.batch),
        "--max-new-tokens", str(case.tokens),
        "--warmup", "0",
        "--repeats", "1",
        "--output-csv", str(csv_path),
        "--output-json", str(json_path),
    ]
    if not case.copy_stream:
        command.append("--no-cuda-copy-stream")

    safe, safety_message = wait_until_safe(args)
    result: dict[str, Any] = {
        "name": case.name,
        "configuration": case.__dict__,
        "safety": safety_message,
        "command": command,
    }
    if not safe:
        result.update(status="skipped", error=safety_message)
        return result

    started = time.perf_counter()
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        abort_reason = None
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            gpu = gpu_snapshot()
            if elapsed > args.case_timeout:
                abort_reason = f"exceeded {args.case_timeout}s"
            elif free_ram_gib() < args.abort_free_ram_gib:
                abort_reason = f"available RAM fell below {args.abort_free_ram_gib:.1f} GiB"
            elif gpu and gpu["temp_c"] > args.abort_temp_c:
                abort_reason = f"GPU temperature exceeded {args.abort_temp_c:.0f} C"
            if abort_reason:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
                result.update(status="safety-abort", error=abort_reason)
                break
            time.sleep(2)

        if abort_reason is None:
            if process.returncode == 0 and json_path.is_file():
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                result.update(status="passed", metrics=payload["runs"][0])
            else:
                result.update(status="failed", error=f"exit code {process.returncode}")
    result["wall_seconds"] = time.perf_counter() - started
    result["log"] = str(log_path)
    return result


def number(value: Any, suffix: str = "", precision: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{precision}f}{suffix}"


def write_report(output_dir: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    passed = [result for result in results if result["status"] == "passed"]
    best = max(passed, key=lambda result: result["metrics"]["tokens_per_sec"]) if passed else None
    lines = [
        "# AirLLM stress benchmark report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Preset: `{args.preset}`",
        f"Model: `{args.model_path}`",
        "",
    ]
    if best:
        metric = best["metrics"]
        lines.extend([
            f"Best throughput: **{metric['tokens_per_sec']:.2f} tok/s** with `{best['name']}` "
            f"at {metric['gpu_util_avg_pct']:.1f}% average GPU utilization.",
            "",
        ])
    lines.extend([
        "| Case | Status | tok/s | TTFT | GPU avg/p95 | VRAM | CPU wait | Copy wait | Power | Temp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        metrics = result.get("metrics", {})
        gpu_util = (
            "n/a" if metrics.get("gpu_util_avg_pct") is None
            else f"{metrics['gpu_util_avg_pct']:.1f}/{metrics['gpu_util_p95_pct']:.0f}%"
        )
        lines.append(
            f"| {result['name']} | {result['status']} | {number(metrics.get('tokens_per_sec'))} | "
            f"{number(metrics.get('time_to_first_token_s'), 's')} | {gpu_util} | "
            f"{number(metrics.get('peak_vram_mb'), ' MiB', 0)} | "
            f"{number(metrics.get('group_cpu_wait_seconds'), 's')} | "
            f"{number(metrics.get('group_copy_wait_seconds'), 's', 3)} | "
            f"{number(metrics.get('gpu_power_avg_w'), ' W', 1)} | "
            f"{number(metrics.get('gpu_temp_max_c'), ' C', 0)} |"
        )
    lines.extend([
        "",
        "Each case runs in a fresh process. A case starts only with at least "
        f"{args.min_free_ram_gib:.1f} GiB available RAM and GPU temperature at or below "
        f"{args.max_start_temp_c:.0f} C. It is stopped if available RAM falls below "
        f"{args.abort_free_ram_gib:.1f} GiB, GPU temperature exceeds {args.abort_temp_c:.0f} C, "
        "or the case timeout expires; failures do not stop later cases.",
        "",
    ])
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    if not args.model_path.exists():
        raise SystemExit(f"model path does not exist: {args.model_path}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or ROOT / "benchmarks" / "results" / f"stress-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = QUICK_CASES if args.preset == "quick" else OVERNIGHT_CASES
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}", flush=True)
        result = run_case(case, args, output_dir)
        results.append(result)
        print(f"  {result['status']} in {result.get('wall_seconds', 0):.1f}s", flush=True)
        if index < len(cases):
            time.sleep(args.cooldown_seconds)
    (output_dir / "summary.json").write_text(
        json.dumps({"preset": args.preset, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    report = write_report(output_dir, results, args)
    print(f"report: {report}")
    return 0 if any(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
