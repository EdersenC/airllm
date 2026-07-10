import json
import tempfile
import unittest
from pathlib import Path

from scripts.layer_profiles import (
    LayerProfileError,
    load_layer_profile,
    parse_layer_indices,
    select_profile_layers,
    validate_profile_identity,
)


class TestLayerProfiles(unittest.TestCase):
    PROFILE = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "profiles"
        / "qwen3-4b-awq-block-influence.json"
    )

    def test_qwen_profile_selects_measured_safe_stack(self):
        profile = load_layer_profile(self.PROFILE)

        self.assertEqual(
            select_profile_layers(profile, 31),
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 34, 35,
            ],
        )

    def test_qwen_profile_rejects_known_broken_half_depth_by_default(self):
        profile = load_layer_profile(self.PROFILE)

        with self.assertRaisesRegex(LayerProfileError, "requires at least 31/36"):
            select_profile_layers(profile, 18)

        self.assertEqual(len(select_profile_layers(profile, 18, allow_unsafe=True)), 18)

    def test_profile_prune_order_must_match_scores(self):
        profile = json.loads(self.PROFILE.read_text(encoding="utf-8"))
        profile["prune_order"][0], profile["prune_order"][1] = (
            profile["prune_order"][1], profile["prune_order"][0]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(LayerProfileError, "must sort scores"):
                load_layer_profile(path)

    def test_explicit_layer_index_parser_is_strict(self):
        self.assertEqual(parse_layer_indices("0, 4, 5"), [0, 4, 5])
        for value in ("", "2,1", "1,1", "-1,2", "a,2"):
            with self.subTest(value=value), self.assertRaises(LayerProfileError):
                parse_layer_indices(value)

    def test_profile_identity_rejects_another_snapshot(self):
        profile = load_layer_profile(self.PROFILE)

        with self.assertRaisesRegex(LayerProfileError, "resolved model snapshot is different"):
            validate_profile_identity(
                profile,
                original_decoder_layer_count=36,
                resolved_model_path="/tmp/snapshots/different",
            )


if __name__ == "__main__":
    unittest.main()
