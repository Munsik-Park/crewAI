"""Recall over-fetch honours ``recall_oversample_factor`` (issue #5).

Each test observes the ``limit=`` that ``storage.search`` is asked for and the
number of search calls, through the public ``Memory(...)`` / ``recall`` /
``MemorySlice.bind`` surface. The query is shorter than
``query_analysis_threshold``, so no LLM is called.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from crewai.memory.memory_scope import MemorySlice
from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory
from pydantic import ValidationError
import pytest


N = 5
QUERY = "short q"


def _embedder() -> MagicMock:
    return MagicMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])


def _storage(fill: bool = False) -> MagicMock:
    """Mock storage; with ``fill`` each search returns as many records as asked."""
    storage = MagicMock()
    storage.list_scopes.return_value = []
    if fill:

        def _search(
            *_args: Any, limit: int = 10, **_kwargs: Any
        ) -> list[tuple[MemoryRecord, float]]:
            return [(MemoryRecord(content=f"r{i}"), 0.9) for i in range(limit)]

        storage.search.side_effect = _search
    else:
        storage.search.return_value = []
    return storage


def _memory(fill: bool = False, **kwargs: Any) -> tuple[Memory, MagicMock]:
    storage = _storage(fill)
    return Memory(
        storage=storage, embedder=_embedder(), llm=MagicMock(), **kwargs
    ), storage


def _limits(storage: MagicMock) -> list[int]:
    return [c.kwargs["limit"] for c in storage.search.call_args_list]


def _slice(memory: Memory) -> MemorySlice:
    return MemorySlice(scopes=["/a", "/b"]).bind(memory)


# C1: a configured factor governs every over-fetch of the recall flow.


def test_deep_recall_uses_configured_factor() -> None:
    mem, storage = _memory(recall_oversample_factor=3)
    mem.recall(QUERY, limit=N, depth="deep")
    assert _limits(storage) == [3 * N, 3 * N]


def test_slice_deep_recall_reads_bound_memory_factor_at_recall_time() -> None:
    first, first_storage = _memory()
    second, second_storage = _memory(recall_oversample_factor=3)
    sl = MemorySlice(scopes=["/a", "/b"])
    sl.bind(first)
    sl.bind(second)

    sl.recall(QUERY, limit=N, depth="deep")

    assert _limits(first_storage) == []
    assert _limits(second_storage) == [9 * N] * 4


def test_slice_shallow_recall_uses_configured_factor() -> None:
    mem, storage = _memory(recall_oversample_factor=3)
    _slice(mem).recall(QUERY, limit=N, depth="shallow")
    assert _limits(storage) == [3 * N, 3 * N]


# C2: with no factor configured the counts are unchanged, and results are capped.


def test_default_factor_search_counts_unchanged() -> None:
    mem, storage = _memory()
    mem.recall(QUERY, limit=N, depth="deep")
    assert _limits(storage) == [2 * N, 2 * N]

    mem, storage = _memory()
    mem.recall(QUERY, limit=N, depth="shallow")
    assert _limits(storage) == [N]

    mem, storage = _memory()
    _slice(mem).recall(QUERY, limit=N, depth="deep")
    assert _limits(storage) == [4 * N] * 4

    mem, storage = _memory()
    _slice(mem).recall(QUERY, limit=N, depth="shallow")
    assert _limits(storage) == [2 * N, 2 * N]


def test_recall_results_capped_at_limit_under_oversample() -> None:
    mem, storage = _memory(fill=True, recall_oversample_factor=3)
    assert len(mem.recall(QUERY, limit=N, depth="deep")) == N
    assert storage.search.call_count >= 1

    mem, storage = _memory(fill=True, recall_oversample_factor=3)
    assert len(_slice(mem).recall(QUERY, limit=N, depth="deep")) == N
    assert storage.search.call_count >= 1


# C4: the factor is a validated ``Memory(...)`` keyword.


def test_factor_below_one_rejected() -> None:
    with pytest.raises(ValidationError):
        _memory(recall_oversample_factor=0)


def test_factor_of_one_turns_oversample_off() -> None:
    mem, storage = _memory(recall_oversample_factor=1)
    mem.recall(QUERY, limit=N, depth="deep")
    assert _limits(storage) == [N, N]
