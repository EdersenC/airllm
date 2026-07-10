"""Validated Block Influence profile loading for reduced-depth experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class LayerProfileError(ValueError):
    """Raised when a layer profile or requested pruning level is unsafe/invalid."""


def parse_layer_indices(value: str) -> list[int]:
    """Parse a comma-separated, strictly increasing source-layer list."""
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise LayerProfileError("layer indices must be comma-separated integers") from exc
    if not indices:
        raise LayerProfileError("layer indices cannot be empty")
    if indices != sorted(set(indices)) or indices[0] < 0:
        raise LayerProfileError("layer indices must be non-negative, unique, and increasing")
    return indices


def load_layer_profile(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a Block Influence JSON profile."""
    profile_path = Path(path).expanduser()
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LayerProfileError(f"cannot read decoder-layer profile {profile_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LayerProfileError(f"invalid JSON in decoder-layer profile {profile_path}: {exc}") from exc

    if profile.get("schema_version") != 1:
        raise LayerProfileError("decoder-layer profile schema_version must be 1")
    if profile.get("profile_type") != "block_influence":
        raise LayerProfileError("decoder-layer profile_type must be 'block_influence'")

    original_count = profile.get("original_decoder_layer_count")
    scores = profile.get("scores")
    prune_order = profile.get("prune_order")
    if not isinstance(original_count, int) or isinstance(original_count, bool) \
            or original_count < 1:
        raise LayerProfileError("profile original_decoder_layer_count must be positive")
    if not isinstance(scores, list) or len(scores) != original_count \
            or any(
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
                for score in scores
            ):
        raise LayerProfileError("profile scores must contain one numeric value per decoder layer")
    if not isinstance(prune_order, list):
        raise LayerProfileError("profile prune_order must be a list")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in prune_order) \
            or sorted(prune_order) != list(range(original_count)):
        raise LayerProfileError("profile prune_order must be a permutation of every layer index")
    expected_order = sorted(range(original_count), key=lambda index: (scores[index], index))
    if prune_order != expected_order:
        raise LayerProfileError("profile prune_order must sort scores from least to most influential")

    minimum = profile.get("recommended_minimum_layer_count")
    if minimum is not None and (
            not isinstance(minimum, int) or isinstance(minimum, bool)
            or not 1 <= minimum <= original_count):
        raise LayerProfileError(
            "profile recommended_minimum_layer_count must be within the decoder stack"
        )
    profile["profile_path"] = str(profile_path.resolve())
    return profile


def select_profile_layers(
    profile: dict[str, Any],
    selected_count: int,
    *,
    allow_unsafe: bool = False,
) -> list[int]:
    """Keep the most influential layers, enforcing the profile's measured quality floor."""
    original_count = profile["original_decoder_layer_count"]
    if not isinstance(selected_count, int) or isinstance(selected_count, bool) \
            or not 1 <= selected_count <= original_count:
        raise LayerProfileError(
            f"decoder layer count must be between 1 and {original_count} for this profile"
        )

    minimum = profile.get("recommended_minimum_layer_count")
    if minimum is not None and selected_count < minimum and not allow_unsafe:
        removed = original_count - selected_count
        raise LayerProfileError(
            f"refusing unsafe layer drop: {selected_count}/{original_count} keeps only "
            f"{selected_count / original_count:.1%} of the stack ({removed} removed), while "
            f"this profile's output sanity gate requires at least {minimum}/{original_count}. "
            "Use --allow-unsafe-layer-drop only for speed-only experiments; coherent half/third "
            "depth requires distillation or retraining."
        )

    removed = set(profile["prune_order"][: original_count - selected_count])
    return [index for index in range(original_count) if index not in removed]


def validate_profile_identity(
    profile: dict[str, Any],
    *,
    original_decoder_layer_count: int,
    resolved_model_path: str | Path,
) -> None:
    """Reject silently applying a calibrated profile to a different checkpoint."""
    expected_count = profile["original_decoder_layer_count"]
    if original_decoder_layer_count != expected_count:
        raise LayerProfileError(
            f"decoder-layer profile expects {expected_count} layers, but the model has "
            f"{original_decoder_layer_count}"
        )

    expected_revision = profile.get("model_snapshot_revision")
    model_path = Path(resolved_model_path).resolve()
    actual_revision = model_path.name if model_path.parent.name == "snapshots" else None
    if expected_revision and actual_revision and actual_revision != expected_revision:
        raise LayerProfileError(
            f"decoder-layer profile targets snapshot {expected_revision}, but the resolved "
            f"model snapshot is {actual_revision}"
        )
