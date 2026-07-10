import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.benchmark_context_limit import resolve_input_tokens
from benchmarks.benchmark_group_streaming import (
    collect_environment_metadata,
    count_generated_tokens as benchmark_count_generated_tokens,
)
from scripts.run_qwen3_awq import (
    count_generated_tokens as runner_count_generated_tokens,
    format_prompts,
)


class TestBenchmarkMetrics(unittest.TestCase):
    class _ChatTokenizer:
        chat_template = "present"

        def apply_chat_template(self, messages, **kwargs):
            return f"<user>{messages[0]['content']}</user><assistant>"

    def test_eos_is_excluded_from_generated_token_throughput(self):
        sequences = torch.tensor([
            [8, 9, 10, 11, 2, 0],
            [8, 9, 2, 0, 0, 0],
            [8, 9, 10, 11, 12, 13],
        ])
        tokenizer = SimpleNamespace(eos_token_id=2)

        self.assertEqual(
            benchmark_count_generated_tokens(sequences, input_width=2, tokenizer=tokenizer),
            6,
        )
        self.assertEqual(
            runner_count_generated_tokens(sequences, input_width=2, tokenizer=tokenizer),
            [2, 0, 4],
        )

    def test_context_default_fills_native_limit_exactly(self):
        self.assertEqual(resolve_input_tokens(40960, None, 8), 40952)

    def test_context_rejects_zero_or_negative_derived_prompt(self):
        for new_tokens in (40960, 40961):
            with self.subTest(new_tokens=new_tokens):
                with self.assertRaisesRegex(ValueError, "derived input length"):
                    resolve_input_tokens(40960, None, new_tokens)

    def test_context_rejects_explicit_total_above_native_limit(self):
        with self.assertRaisesRegex(ValueError, "exceeds native limit"):
            resolve_input_tokens(40960, 40960, 1)

    def test_environment_metadata_records_revision_and_runtime(self):
        metadata = collect_environment_metadata(
            torch,
            torch.device("cpu"),
            Path("models--Qwen--Qwen3-4B-AWQ/snapshots/test-revision"),
        )

        self.assertEqual(metadata["model_snapshot_revision"], "test-revision")
        self.assertEqual(metadata["device"], "cpu")
        self.assertTrue(metadata["torch_version"])
        self.assertTrue(metadata["git_revision"])

    def test_runner_auto_applies_chat_template_and_can_force_raw(self):
        tokenizer = self._ChatTokenizer()

        formatted, active_format = format_prompts(tokenizer, ["hello"], "auto", False)
        raw, raw_format = format_prompts(tokenizer, ["hello"], "raw", False)

        self.assertEqual(formatted, ["<user>hello</user><assistant>"])
        self.assertEqual(active_format, "chat")
        self.assertEqual(raw, ["hello"])
        self.assertEqual(raw_format, "raw")

    def test_runner_rejects_forced_chat_without_template(self):
        with self.assertRaisesRegex(ValueError, "no chat template"):
            format_prompts(SimpleNamespace(chat_template=None), ["hello"], "chat", False)


if __name__ == "__main__":
    unittest.main()
