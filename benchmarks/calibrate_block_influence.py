#!/usr/bin/env python3
"""Measure Qwen decoder Block Influence and write a reproducible layer profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from airllm import AutoModel


DEFAULT_MODEL_PATH = Path("/mnt/s/ai-cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ")
DEFAULT_OUTPUT = Path("benchmarks/results/qwen3-4b-awq-block-influence.json")
DEFAULT_TEXTS = [
    (
        "The history of computing spans mechanical calculators, vacuum tubes, transistors, "
        "integrated circuits, and modern parallel processors. Each generation traded physical "
        "size and energy use for greater reliability and speed."
    ),
    (
        "In a healthy ecosystem, energy flows from sunlight through plants to herbivores and "
        "predators, while decomposers return nutrients to the soil. Biodiversity makes the "
        "system more resilient to disturbance."
    ),
    (
        "To debug a race condition, first reproduce it reliably, record the order of concurrent "
        "events, identify shared mutable state, and synchronize only the smallest critical section."
    ),
    (
        "A train travels 120 kilometers in two hours and then 90 kilometers in one hour. Explain "
        "how to calculate its average speed across the complete trip."
    ),
    (
        "Write a concise explanation of why the sky appears blue during the day and often red "
        "near sunset, using the idea of wavelength-dependent scattering."
    ),
    (
        "The capital of France is Paris. Water freezes at zero degrees Celsius at standard "
        "pressure. Two plus two equals four. These are simple factual statements."
    ),
    (
        "User: Please summarize the meeting. Assistant: The team agreed to ship the tested "
        "feature on Friday, postpone the database migration, and document the rollback procedure."
    ),
    (
        "Python functions accept positional and keyword arguments. A context manager guarantees "
        "cleanup by entering a scope and running its exit behavior even when an exception occurs."
    ),
    (
        "Economic inflation describes a broad rise in prices over time. A single expensive "
        "product is not sufficient evidence; analysts compare baskets of goods across periods."
    ),
    (
        "Once upon a time, a cartographer found an island that appeared on every old map but "
        "nowhere on the sea. She followed the stars until it appeared beneath a bank of fog."
    ),
    (
        "Translate this into plain language: correlation measures association but does not by "
        "itself establish that one variable caused the other."
    ),
    (
        "A secure service validates untrusted input, uses least-privilege credentials, avoids "
        "logging secrets, patches dependencies, and records enough audit data to investigate."
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="Optional UTF-8 file with one calibration sample per non-empty line")
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--layers-per-gpu-group", type=int, default=24)
    parser.add_argument("--prefetch-groups", type=int, default=8)
    parser.add_argument("--cpu-layer-cache-gib", type=float, default=16.0)
    parser.add_argument("--awq-backend", default="marlin")
    parser.add_argument("--recommended-minimum-layer-count", type=int, default=None,
                        help="Optional quality floor established by a separate generation gate")
    return parser


def calibration_texts(prompt_file: Path | None) -> list[str]:
    if prompt_file is None:
        return DEFAULT_TEXTS
    texts = [
        line.strip()
        for line in prompt_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not texts:
        raise ValueError("calibration prompt file contains no non-empty samples")
    return texts


def main() -> int:
    args = build_parser().parse_args()
    if args.max_input_tokens < 1:
        raise SystemExit("--max-input-tokens must be positive")
    texts = calibration_texts(args.prompt_file)
    device = "cuda:0" if torch.cuda.is_available() else None
    if device is None:
        raise SystemExit("CUDA is required for the prepared AWQ Block Influence calibration")

    model = AutoModel.from_pretrained(
        str(args.model_path),
        device=device,
        layers_per_gpu_group=args.layers_per_gpu_group,
        prefetch_groups=args.prefetch_groups,
        cpu_layer_cache_gib=args.cpu_layer_cache_gib,
        persistent_gpu_residency=True,
        awq_backend=args.awq_backend,
        show_live_stats=False,
    )
    decoder_layers = model.model.model.layers
    score_sums = torch.zeros(len(decoder_layers), dtype=torch.float64)
    score_counts = torch.zeros(len(decoder_layers), dtype=torch.int64)
    handles = []
    for layer_index, layer in enumerate(decoder_layers):
        def record_influence(_module, inputs, output, layer_index=layer_index):
            block_input = inputs[0].detach().float()
            block_output = (output[0] if isinstance(output, tuple) else output).detach().float()
            influence = 1.0 - F.cosine_similarity(block_input, block_output, dim=-1)
            score_sums[layer_index] += influence.double().sum().cpu()
            score_counts[layer_index] += influence.numel()

        handles.append(layer.register_forward_hook(record_influence))

    try:
        with torch.inference_mode():
            for text in texts:
                encoded = model.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_tokens,
                )
                inputs = {name: tensor.to(model.device) for name, tensor in encoded.items()}
                model.model.model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    scores = (score_sums / score_counts).tolist()
    prune_order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    resolved_model_path = Path(model.model_local_path).resolve()
    payload = {
        "schema_version": 1,
        "profile_type": "block_influence",
        "model": str(args.model_path),
        "model_snapshot_revision": (
            resolved_model_path.name if resolved_model_path.parent.name == "snapshots" else None
        ),
        "original_decoder_layer_count": len(scores),
        "metric": "1 - mean(cosine_similarity(block_input, block_output))",
        "score_direction": "higher_is_more_influential",
        "calibration": {
            "sample_count": len(texts),
            "max_tokens_per_sample": args.max_input_tokens,
            "attention_backend": "sdpa",
            "weight_backend": f"awq_{args.awq_backend}",
            "prompt_set": str(args.prompt_file) if args.prompt_file else "diverse-prose-v1",
        },
        "scores": scores,
        "prune_order": prune_order,
    }
    if args.recommended_minimum_layer_count is not None:
        payload["recommended_minimum_layer_count"] = args.recommended_minimum_layer_count

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"scores: {scores}")
    print(f"prune_order: {prune_order}")
    print(f"json: {args.output_json}")
    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
