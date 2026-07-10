"""A bounded, thread-safe CPU cache for AirLLM layer state dictionaries.

``CPULayerCache`` is intentionally independent of the model streaming code.  It keeps
complete layer ``state_dict`` objects in CPU RAM, making a second autoregressive
forward pass able to reuse hot shards instead of reading them from disk again.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
import math
from threading import RLock
from typing import Callable, Dict, Hashable, Mapping, Optional

import torch


LayerStateDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class CPULayerCacheStats:
    """A point-in-time snapshot of :class:`CPULayerCache` counters."""

    hits: int
    misses: int
    evictions: int
    admission_rejections: int
    bytes: int
    max_bytes: int
    entries: int


class CPULayerCache:
    """Bounded LRU cache for CPU-resident layer ``state_dict`` objects.

    Args:
        max_bytes: Maximum tensor payload bytes to retain.  ``0`` disables
            retention while still allowing :meth:`get_or_load` to coalesce
            simultaneous loads.
        max_gib: Convenience alternative to ``max_bytes`` using binary GiB.
        pin_memory: Pin cached CPU tensors when CUDA is available, allowing a
            later CPU-to-CUDA transfer to use ``non_blocking=True``.
        admission_policy: ``lru`` evicts old entries for new ones. ``static``
            retains the first hot set that fits and rejects later overflow,
            preventing a repeated sequential model scan from thrashing the cache.

    The cache stores tensor references, rather than cloning tensor contents, so
    caching does not briefly double a layer's RAM use.  Callers must therefore
    treat a state dict successfully passed to :meth:`put` as immutable while it
    may be cached.  The mapping itself is copied, so adding or removing keys in
    the caller's dictionary cannot change an existing entry.
    """

    def __init__(
        self,
        *,
        max_bytes: Optional[int] = None,
        max_gib: Optional[float] = None,
        pin_memory: bool = False,
        admission_policy: str = "lru",
    ) -> None:
        if (max_bytes is None) == (max_gib is None):
            raise ValueError("provide exactly one of max_bytes or max_gib")

        if max_gib is not None:
            if isinstance(max_gib, bool) or not isinstance(max_gib, (int, float)):
                raise TypeError("max_gib must be a non-negative number")
            if not math.isfinite(max_gib) or max_gib < 0:
                raise ValueError("max_gib must be a finite non-negative number")
            max_bytes = int(max_gib * (1024 ** 3))

        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise TypeError("max_bytes must be a non-negative integer")
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")

        self._max_bytes = max_bytes
        self._pin_memory = bool(pin_memory)
        if admission_policy not in {"lru", "static"}:
            raise ValueError("admission_policy must be 'lru' or 'static'")
        self._admission_policy = admission_policy
        self._entries: "OrderedDict[Hashable, LayerStateDict]" = OrderedDict()
        self._entry_bytes: Dict[Hashable, int] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._admission_rejections = 0
        self._closed = False
        self._lock = RLock()
        self._inflight: Dict[Hashable, Future[LayerStateDict]] = {}

    @property
    def max_bytes(self) -> int:
        """The configured capacity in bytes."""
        return self._max_bytes

    @property
    def pin_memory(self) -> bool:
        """Whether entries are pinned when CUDA can support pinned memory."""
        return self._pin_memory

    def get(self, key: Hashable) -> Optional[LayerStateDict]:
        """Return a cached state dict and mark it most recently used, if present."""
        with self._lock:
            self._require_open()
            try:
                state_dict = self._entries.pop(key)
            except KeyError:
                self._misses += 1
                return None
            self._entries[key] = state_dict
            self._hits += 1
            return state_dict

    def put(self, key: Hashable, state_dict: Mapping[str, torch.Tensor]) -> bool:
        """Insert ``state_dict`` and return whether it fit in the cache.

        Entries larger than the complete cache are not retained, and importantly
        do not evict useful existing entries.  Replacing a key counts only any
        *other* LRU entries removed to make room as evictions.
        """
        prepared, entry_bytes = self._prepare_state_dict(state_dict)
        with self._lock:
            self._require_open()
            return self._put_locked(key, prepared, entry_bytes)

    def get_or_load(
        self, key: Hashable, loader: Callable[[], Mapping[str, torch.Tensor]]
    ) -> LayerStateDict:
        """Return ``key`` or run ``loader`` once when concurrent callers miss.

        The loader runs outside the cache lock.  Concurrent misses for the same
        key wait on its result, avoiding duplicate disk reads from a prefetch
        worker and a foreground forward pass.  A too-large loaded entry is
        returned to all callers but is not retained.
        """
        with self._lock:
            self._require_open()
            try:
                state_dict = self._entries.pop(key)
            except KeyError:
                self._misses += 1
            else:
                self._entries[key] = state_dict
                self._hits += 1
                return state_dict

            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                is_loader = True
            else:
                is_loader = False

        if not is_loader:
            return future.result()

        try:
            prepared, entry_bytes = self._prepare_state_dict(loader())
            with self._lock:
                if self._closed:
                    raise RuntimeError("CPULayerCache is closed")
                self._put_locked(key, prepared, entry_bytes)
                self._inflight.pop(key, None)
                future.set_result(prepared)
            return prepared
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
                if not future.done():
                    future.set_exception(exc)
            raise

    def clear(self) -> None:
        """Release all cached entries while preserving lifetime hit/miss counters."""
        with self._lock:
            self._require_open()
            self._entries.clear()
            self._entry_bytes.clear()
            self._bytes = 0

    def close(self) -> None:
        """Release entries and wake waiting callers; a closed cache cannot be reused."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._entries.clear()
            self._entry_bytes.clear()
            self._bytes = 0
            inflight = tuple(self._inflight.values())
            self._inflight.clear()
            for future in inflight:
                if not future.done():
                    future.set_exception(RuntimeError("CPULayerCache is closed"))

    def stats(self) -> CPULayerCacheStats:
        """Return hit/miss/eviction counters and current bounded memory usage."""
        with self._lock:
            return CPULayerCacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                admission_rejections=self._admission_rejections,
                bytes=self._bytes,
                max_bytes=self._max_bytes,
                entries=len(self._entries),
            )

    def __enter__(self) -> "CPULayerCache":
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def _put_locked(self, key: Hashable, state_dict: LayerStateDict, entry_bytes: int) -> bool:
        if entry_bytes > self._max_bytes:
            return False

        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= self._entry_bytes.pop(key)

        if self._admission_policy == "static" and self._bytes + entry_bytes > self._max_bytes:
            if previous is not None:
                self._entries[key] = previous
                previous_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in previous.values()
                )
                self._entry_bytes[key] = previous_bytes
                self._bytes += previous_bytes
            self._admission_rejections += 1
            return False

        while self._bytes + entry_bytes > self._max_bytes:
            evicted_key, _ = self._entries.popitem(last=False)
            self._bytes -= self._entry_bytes.pop(evicted_key)
            self._evictions += 1

        self._entries[key] = state_dict
        self._entry_bytes[key] = entry_bytes
        self._bytes += entry_bytes
        return True

    def _prepare_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> tuple[LayerStateDict, int]:
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping of tensor names to tensors")

        prepared: LayerStateDict = {}
        entry_bytes = 0
        for name, tensor in state_dict.items():
            if not isinstance(name, str):
                raise TypeError("state_dict keys must be strings")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("state_dict values must be torch.Tensor instances")
            if tensor.device.type != "cpu":
                raise ValueError("CPULayerCache accepts CPU tensors only")
            # This intentionally over-counts overlapping views.  Conservative accounting is
            # preferable to undercounting a cache whose primary safety guarantee is bounded RAM.
            entry_bytes += tensor.numel() * tensor.element_size()
            prepared[name] = tensor

        if self._pin_memory and torch.cuda.is_available():
            prepared = {
                name: tensor if tensor.is_pinned() else tensor.pin_memory()
                for name, tensor in prepared.items()
            }
        return prepared, entry_bytes

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CPULayerCache is closed")
