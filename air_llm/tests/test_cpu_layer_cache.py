"""Dependency-light unit tests for :mod:`airllm.cpu_layer_cache`."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

import torch


# Import the module directly so these tests do not need transformers, accelerate, or model files.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "airllm"))
from cpu_layer_cache import CPULayerCache  # noqa: E402


def state_dict(values: int) -> dict[str, torch.Tensor]:
    return {"weight": torch.arange(values, dtype=torch.float32)}


class CPULayerCacheTests(unittest.TestCase):
    def test_lru_eviction_tracks_bytes_and_stats(self) -> None:
        cache = CPULayerCache(max_bytes=16)
        first, second, third = state_dict(2), state_dict(2), state_dict(2)

        self.assertTrue(cache.put("first", first))
        self.assertTrue(cache.put("second", second))
        self.assertIs(cache.get("first")["weight"], first["weight"])
        self.assertTrue(cache.put("third", third))

        self.assertIsNone(cache.get("second"))
        self.assertIs(cache.get("first")["weight"], first["weight"])
        self.assertIs(cache.get("third")["weight"], third["weight"])
        self.assertEqual(
            cache.stats(),
            cache.stats().__class__(
                hits=3, misses=1, evictions=1, bytes=16, max_bytes=16, entries=2
            ),
        )

    def test_oversized_entry_does_not_flush_hot_entries(self) -> None:
        cache = CPULayerCache(max_bytes=8)
        hot = state_dict(2)

        cache.put("hot", hot)
        self.assertFalse(cache.put("too-large", state_dict(3)))

        self.assertIs(cache.get("hot")["weight"], hot["weight"])
        stats = cache.stats()
        self.assertEqual((stats.entries, stats.bytes, stats.evictions), (1, 8, 0))

    def test_replacing_an_entry_reclaims_its_old_size_before_lru_eviction(self) -> None:
        cache = CPULayerCache(max_bytes=24)
        cache.put("a", state_dict(2))
        cache.put("b", state_dict(2))
        cache.put("c", state_dict(2))

        replacement = state_dict(4)
        self.assertTrue(cache.put("a", replacement))

        self.assertIsNone(cache.get("b"))
        self.assertIs(cache.get("a")["weight"], replacement["weight"])
        stats = cache.stats()
        self.assertEqual((stats.entries, stats.bytes, stats.evictions), (2, 24, 1))

    def test_mapping_is_copied_but_cpu_tensor_references_are_not_cloned(self) -> None:
        cache = CPULayerCache(max_bytes=8)
        layer = state_dict(2)
        tensor = layer["weight"]
        cache.put("layer", layer)
        layer.clear()

        cached = cache.get("layer")
        self.assertEqual(set(cached), {"weight"})
        self.assertIs(cached["weight"], tensor)

    def test_invalid_configurations_and_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CPULayerCache()
        with self.assertRaises(ValueError):
            CPULayerCache(max_bytes=1, max_gib=1)
        with self.assertRaises(ValueError):
            CPULayerCache(max_bytes=-1)
        with self.assertRaises(ValueError):
            CPULayerCache(max_gib=float("inf"))
        with self.assertRaises(TypeError):
            CPULayerCache(max_bytes=1.5)

        cache = CPULayerCache(max_bytes=8)
        with self.assertRaises(TypeError):
            cache.put("bad", {"weight": object()})
        with self.assertRaises(TypeError):
            cache.put("bad", {1: torch.zeros(1)})

    def test_zero_capacity_coalesces_loads_without_retaining_them(self) -> None:
        cache = CPULayerCache(max_bytes=0)
        loaded = cache.get_or_load("layer", lambda: state_dict(2))

        self.assertEqual(loaded["weight"].tolist(), [0.0, 1.0])
        self.assertEqual((cache.stats().entries, cache.stats().bytes), (0, 0))

    def test_get_or_load_runs_one_loader_for_concurrent_misses(self) -> None:
        cache = CPULayerCache(max_bytes=8)
        start_loader = threading.Event()
        release_loader = threading.Event()
        calls = 0
        calls_lock = threading.Lock()
        results = []
        errors = []

        def loader():
            nonlocal calls
            with calls_lock:
                calls += 1
            start_loader.set()
            self.assertTrue(release_loader.wait(timeout=2))
            return state_dict(2)

        def worker() -> None:
            try:
                results.append(cache.get_or_load("layer", loader))
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        self.assertTrue(start_loader.wait(timeout=2))
        release_loader.set()
        for thread in threads:
            thread.join(timeout=2)

        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(calls, 1)
        self.assertEqual(len(results), 6)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertEqual((cache.stats().hits, cache.stats().misses), (0, 6))

    def test_clear_and_close_release_memory_and_close_is_terminal(self) -> None:
        cache = CPULayerCache(max_gib=1 / (1024 ** 3))
        cache.put("layer", state_dict(1))
        cache.clear()
        self.assertEqual((cache.stats().entries, cache.stats().bytes), (0, 0))

        cache.close()
        cache.close()  # idempotent
        with self.assertRaisesRegex(RuntimeError, "closed"):
            cache.get("layer")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            cache.put("layer", state_dict(1))

    @unittest.skipUnless(torch.cuda.is_available(), "pinned memory requires a CUDA-capable torch runtime")
    def test_pin_memory_option_pins_cached_cpu_tensors(self) -> None:
        cache = CPULayerCache(max_bytes=8, pin_memory=True)
        cache.put("layer", state_dict(2))
        self.assertTrue(cache.get("layer")["weight"].is_pinned())


if __name__ == "__main__":
    unittest.main()
