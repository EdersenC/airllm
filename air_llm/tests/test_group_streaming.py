import unittest

from ..airllm.airllm_base import AirLLMBaseModel


class TestGroupStreaming(unittest.TestCase):
    class _Executor:
        def __init__(self):
            self.submitted = []

        def submit(self, _fn, group_id):
            self.submitted.append(group_id)
            return object()

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
