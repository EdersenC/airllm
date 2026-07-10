import unittest

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


if __name__ == "__main__":
    unittest.main()
