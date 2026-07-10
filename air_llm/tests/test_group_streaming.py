import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from ..airllm.airllm_base import AirLLMBaseModel


class TestGroupStreaming(unittest.TestCase):
    class _Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, _fn, group_id):
            self.submitted.append(group_id)
            return object()

    class _CUDAExecutor:
        def __init__(self):
            self.submitted = []
            self.result = object()

        def submit(self, function, *args):
            self.submitted.append((function, args))
            return self.result

    def test_prefetch_window_starts_with_immediate_next_group(self):
        model = object.__new__(AirLLMBaseModel)
        model.prefetching = True
        model.prefetch_groups = 1
        model._streaming_groups = [[0], [1, 2], [3, 4], [5]]
        model._prefetch_futures = {}
        model._executor = self._Executor()

        model._schedule_group_prefetch(0)

        self.assertEqual(model._executor.submitted, [1])

    def test_prefetch_window_can_hold_multiple_upcoming_groups(self):
        model = object.__new__(AirLLMBaseModel)
        model.prefetching = True
        model.prefetch_groups = 2
        model._streaming_groups = [[0], [1], [2], [3]]
        model._prefetch_futures = {}
        model._executor = self._Executor()

        model._schedule_group_prefetch(0)

        self.assertEqual(model._executor.submitted, [1, 2])

    def test_prefetch_window_respects_bounded_cpu_memory_budget(self):
        model = object.__new__(AirLLMBaseModel)
        model.prefetching = True
        model.prefetch_groups = 3
        model.cpu_prefetch_budget_bytes = 15
        model._streaming_groups = [[0], [1], [2], [3]]
        model._prefetch_futures = {}
        model._gpu_prefetch_futures = {}
        model._group_cpu_sources = {}
        model._executor = self._Executor()
        model._estimated_group_shard_bytes = Mock(return_value=10)

        model._schedule_group_prefetch(0)

        self.assertEqual(model._executor.submitted, [1])

    def test_cuda_prefetch_promotes_immediate_cpu_future(self):
        model = object.__new__(AirLLMBaseModel)
        model.cuda_copy_stream = True
        model._streaming_groups = [[0], [1, 2], [3]]
        cpu_future = object()
        model._prefetch_futures = {1: cpu_future}
        model._gpu_prefetch_futures = {}
        model._gpu_prefetch_executor = self._CUDAExecutor()

        model._schedule_cuda_group_prefetch(0)

        self.assertNotIn(1, model._prefetch_futures)
        self.assertIs(model._gpu_prefetch_futures[1], model._gpu_prefetch_executor.result)
        function, args = model._gpu_prefetch_executor.submitted[0]
        self.assertEqual(function, model._prepare_group_on_cuda_stream)
        self.assertEqual(args, (1, cpu_future))

    def test_cuda_prefetch_does_not_skip_missing_immediate_group(self):
        model = object.__new__(AirLLMBaseModel)
        model.cuda_copy_stream = True
        model._streaming_groups = [[0], [1], [2]]
        model._prefetch_futures = {2: object()}
        model._gpu_prefetch_futures = {}
        model._gpu_prefetch_executor = self._CUDAExecutor()

        model._schedule_cuda_group_prefetch(0)

        self.assertFalse(model._gpu_prefetch_executor.submitted)
        self.assertEqual(set(model._prefetch_futures), {2})

    def test_non_tied_groups_keep_model_edges_separate(self):
        groups = AirLLMBaseModel._build_streaming_groups(
            layer_count=9,
            tie_word_embeddings=False,
            layers_per_gpu_group=2,
        )
        self.assertEqual(groups, [[0], [1, 2], [3, 4], [5, 6], [7], [8]])

    def test_tied_groups_skip_resident_embedding_and_lm_head(self):
        groups = AirLLMBaseModel._build_streaming_groups(
            layer_count=9,
            tie_word_embeddings=True,
            layers_per_gpu_group=2,
        )
        self.assertEqual(groups, [[1, 2], [3, 4], [5, 6], [7]])

    def test_group_size_larger_than_decoder_stack_is_supported(self):
        groups = AirLLMBaseModel._build_streaming_groups(
            layer_count=6,
            tie_word_embeddings=True,
            layers_per_gpu_group=99,
        )
        self.assertEqual(groups, [[1, 2, 3], [4]])

    def test_group_size_must_be_positive_integer(self):
        for group_size in (0, -1, True, 1.5):
            with self.subTest(group_size=group_size):
                with self.assertRaises(ValueError):
                    AirLLMBaseModel._build_streaming_groups(6, True, group_size)

    def test_cuda_double_buffer_counts_current_and_next_decoder_groups(self):
        groups = AirLLMBaseModel._build_streaming_groups(
            layer_count=39,
            tie_word_embeddings=True,
            layers_per_gpu_group=9,
        )

        peak = AirLLMBaseModel._peak_decoder_layer_residency(
            groups,
            decoder_layer_count=36,
            cuda_copy_stream=True,
            persistent_gpu_residency=False,
        )

        self.assertEqual(peak, 18)

    def test_synchronous_copy_counts_only_the_active_decoder_group(self):
        groups = AirLLMBaseModel._build_streaming_groups(
            layer_count=39,
            tie_word_embeddings=True,
            layers_per_gpu_group=18,
        )

        peak = AirLLMBaseModel._peak_decoder_layer_residency(
            groups,
            decoder_layer_count=36,
            cuda_copy_stream=False,
            persistent_gpu_residency=False,
        )

        self.assertEqual(peak, 18)

    def test_persistent_residency_counts_the_complete_decoder_stack(self):
        self.assertEqual(
            AirLLMBaseModel._peak_decoder_layer_residency(
                groups=[[1, 2], [3, 4]],
                decoder_layer_count=36,
                cuda_copy_stream=True,
                persistent_gpu_residency=True,
            ),
            36,
        )

    def test_awq_backend_override_is_copied_into_model_config(self):
        model = object.__new__(AirLLMBaseModel)
        original = {"quant_method": "awq", "bits": 4}
        model.awq_backend = "marlin"
        model.config = SimpleNamespace(quantization_config=original)

        model._configure_awq_backend()

        self.assertEqual(model.config.quantization_config["backend"], "marlin")
        self.assertNotIn("backend", original)

    def test_awq_backend_override_rejects_non_awq_checkpoint(self):
        model = object.__new__(AirLLMBaseModel)
        model.awq_backend = "marlin"
        model.config = SimpleNamespace(quantization_config={"quant_method": "gptq"})

        with self.assertRaisesRegex(ValueError, "only applies to AWQ"):
            model._configure_awq_backend()

    def test_persistent_pre_hook_reuses_resident_group(self):
        model = object.__new__(AirLLMBaseModel)
        module = SimpleNamespace(_airllm_idx=1)
        model._streaming_groups = [[1, 2]]
        model._group_by_idx = {1: 0, 2: 0}
        model._resident_group_ids = {0}
        model.persistent_gpu_residency = True
        model._runtime_stats = {"resident_group_hits": 0}
        model._group_compute_started = {}
        model._forward_count = 0
        model._live_batch_size = "?"
        model._live_sequence_length = "?"
        model._emit_live_stats = Mock()
        model._load_group_to_device = Mock()
        model._unload_group = Mock()

        model._pre_hook(module, (SimpleNamespace(shape=(2, 7, 16)),))

        self.assertEqual(model._runtime_stats["resident_group_hits"], 1)
        self.assertEqual(model._forward_count, 1)
        self.assertEqual(model._live_batch_size, 2)
        self.assertEqual(model._live_sequence_length, 7)
        model._load_group_to_device.assert_not_called()
        model._unload_group.assert_not_called()

    def test_persistent_post_hook_keeps_group_loaded(self):
        model = object.__new__(AirLLMBaseModel)
        module = SimpleNamespace(_airllm_idx=2)
        model._group_by_idx = {2: 0}
        model._group_last_idx = {0: 2}
        model._group_compute_started = {0: 0.0}
        model._runtime_stats = {"group_compute_seconds": 0.0}
        model.persistent_gpu_residency = True
        model._emit_live_stats = Mock()
        model._unload_group = Mock()
        output = object()

        returned = model._post_hook(module, (), output)

        self.assertIs(returned, output)
        self.assertGreater(model._runtime_stats["group_compute_seconds"], 0.0)
        model._unload_group.assert_not_called()
        self.assertEqual(model._emit_live_stats.call_args.args[2], "resident")

    def test_persistent_oom_closes_workers_and_reports_streaming_fallback(self):
        model = object.__new__(AirLLMBaseModel)
        model._streaming_groups = [[0]]
        model._resident_group_ids = set()
        model._load_group_to_device = Mock(side_effect=torch.cuda.OutOfMemoryError())
        model.close = Mock()

        with self.assertRaisesRegex(RuntimeError, "does not fit in available VRAM"):
            model._materialize_persistent_model()

        model.close.assert_called_once_with()

    def test_persistent_materialization_always_replaces_layer_hooks(self):
        model = object.__new__(AirLLMBaseModel)
        model._streaming_groups = []
        model._resident_group_ids = set()
        model._cancel_prefetch_futures = Mock()
        model.hf_quantizer = None
        model.device = torch.device("cuda:0")
        model._group_cpu_sources = {}
        model._group_copy_events = {}
        model.cpu_layer_cache = SimpleNamespace(clear=Mock())
        model._remove_streaming_hooks = Mock()
        model._install_persistent_runtime_hooks = Mock()
        model._live_stats = SimpleNamespace(enabled=True)

        with patch("air_llm.airllm.airllm_base.torch.cuda.synchronize"):
            model._materialize_persistent_model()

        model._remove_streaming_hooks.assert_called_once_with()
        model._install_persistent_runtime_hooks.assert_called_once_with()

    def test_persistent_generate_error_cleans_transient_cuda_cache(self):
        model = object.__new__(AirLLMBaseModel)
        model.model = SimpleNamespace(generate=Mock(side_effect=RuntimeError("boom")))
        model._cancel_prefetch_futures = Mock()
        model._resident_group_ids = {0, 1}
        model._unload_group = Mock()
        model.persistent_gpu_residency = True
        model._live_stats = SimpleNamespace(start=Mock(), finish=Mock())

        with patch("air_llm.airllm.airllm_base.clean_memory") as clean:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                model.generate()

        clean.assert_called_once_with()
        model._unload_group.assert_not_called()
        model._live_stats.finish.assert_called_once_with()

    def test_close_is_idempotent_for_hook_and_worker_cleanup(self):
        model = object.__new__(AirLLMBaseModel)
        streaming_handle = Mock()
        persistent_handle = Mock()
        model._streaming_hook_handles = [streaming_handle]
        model._persistent_model_hook_handles = [persistent_handle]
        model._prefetch_futures = {}
        model._gpu_prefetch_futures = {}
        model._resident_group_ids = set()
        model.tie_word_embeddings = False
        model._executor = None
        model._gpu_prefetch_executor = None
        model.cpu_layer_cache = SimpleNamespace(close=Mock())

        with patch("air_llm.airllm.airllm_base.clean_memory"):
            model.close()
            model.close()

        streaming_handle.remove.assert_called_once_with()
        persistent_handle.remove.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
