"""Tests that recall over-fetches from storage by the configured factor.

The storage stub records the ``limit`` each ``StorageBackend.search`` call
receives. Its records are old, unimportant and score 0.0, so deep recall is
below the low-confidence threshold and takes an exploration round whenever
``exploration_budget`` allows one.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import threading
from typing import Any
from unittest.mock import MagicMock

from crewai.memory.memory_scope import MemorySlice
from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory
import pytest


L = 5
TWO_SCOPES = ["/a", "/b"]


class _Recorder:
    """Storage stub that records the ``limit`` passed to each search."""

    def __init__(self, scopes: list[str]) -> None:
        self.limits: list[int] = []
        self._lock = threading.Lock()
        self.storage = MagicMock()
        self.storage.search.side_effect = self._search
        self.storage.list_scopes.return_value = list(scopes)

    def _search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        with self._lock:
            self.limits.append(limit)
        old = datetime.now() - timedelta(days=3650)
        scope = scope_prefix or "/"
        return [
            (
                MemoryRecord(
                    content=f"c{scope}{i}",
                    scope=scope,
                    importance=0.0,
                    created_at=old,
                ),
                0.0,
            )
            for i in range(limit)
        ]


def _memory(
    scopes: list[str] | None = None, **kwargs: Any
) -> tuple[Memory, _Recorder, MagicMock]:
    rec = _Recorder(scopes or [])
    llm = MagicMock()
    llm.call.return_value = "ok"
    mem = Memory(
        storage=rec.storage,
        llm=llm,
        embedder=MagicMock(return_value=[[0.1] * 8]),
        read_only=True,
        **kwargs,
    )
    return mem, rec, llm


# --- Deep recall fetches limit x factor on both branches and every round ---


def test_deep_recall_parallel_branch_uses_factor_on_every_round() -> None:
    mem, rec, llm = _memory(
        TWO_SCOPES, recall_oversample_factor=3, exploration_budget=1
    )

    result = mem.recall("q", limit=L, depth="deep")

    assert llm.call.called  # an exploration round ran
    assert len(rec.limits) == 4  # 2 rounds x 2 scopes
    assert rec.limits == [3 * L] * 4
    assert len(result) <= L


def test_deep_recall_sequential_branch_uses_factor() -> None:
    mem, rec, _ = _memory([], recall_oversample_factor=3, exploration_budget=0)

    result = mem.recall("q", limit=L, depth="deep")

    assert rec.limits == [3 * L]
    assert len(result) <= L


def test_deep_recall_sequential_branch_uses_factor_on_every_round() -> None:
    mem, rec, llm = _memory([], recall_oversample_factor=3, exploration_budget=1)

    result = mem.recall("q", limit=L, depth="deep")

    assert llm.call.called  # an exploration round ran
    assert rec.limits == [3 * L, 3 * L]  # 2 rounds x 1 search
    assert len(result) <= L


# --- A slice fetches limit x factor (shallow) or limit x factor^2 (deep) per scope ---


@pytest.mark.parametrize(
    ("depth", "expected"),
    [("deep", 9 * L), ("shallow", 3 * L)],
)
def test_slice_recall_uses_factor_per_scope(depth: str, expected: int) -> None:
    mem, rec, _ = _memory([], recall_oversample_factor=3, exploration_budget=0)
    sl = MemorySlice(memory=mem, scopes=TWO_SCOPES)

    result = sl.recall("q", limit=L, depth=depth)  # type: ignore[arg-type]

    assert rec.limits == [expected, expected]
    assert len(result) <= L


# --- A slice takes the factor from the Memory bound when it recalls ---


def test_slice_reads_factor_from_bound_memory_at_recall_time() -> None:
    sl = MemorySlice(scopes=TWO_SCOPES)

    m3, rec3, _ = _memory([], recall_oversample_factor=3, exploration_budget=0)
    sl.bind(m3)
    assert len(sl.recall("q", limit=L, depth="shallow")) <= L
    assert rec3.limits == [3 * L, 3 * L]

    m4, rec4, _ = _memory([], recall_oversample_factor=4, exploration_budget=0)
    sl.bind(m4)
    assert len(sl.recall("q", limit=L, depth="shallow")) <= L
    assert rec4.limits == [4 * L, 4 * L]


# --- Default factor 2; plain shallow recall fetches the limit at any factor ---


def test_default_deep_recall_fetches_twice_limit() -> None:
    mem, rec, llm = _memory(TWO_SCOPES, exploration_budget=1)

    result = mem.recall("q", limit=L, depth="deep")

    assert llm.call.called
    assert rec.limits == [2 * L] * 4
    assert len(result) <= L


def test_default_shallow_recall_fetches_limit() -> None:
    mem, rec, _ = _memory(TWO_SCOPES, exploration_budget=1)

    result = mem.recall("q", limit=L, depth="shallow")

    assert rec.limits == [L]
    assert len(result) <= L


@pytest.mark.parametrize(
    ("depth", "expected"),
    [("shallow", 2 * L), ("deep", 4 * L)],
)
def test_default_slice_recall_matches_base(depth: str, expected: int) -> None:
    mem, rec, _ = _memory([], exploration_budget=0)
    sl = MemorySlice(memory=mem, scopes=TWO_SCOPES)

    result = sl.recall("q", limit=L, depth=depth)  # type: ignore[arg-type]

    assert rec.limits == [expected, expected]
    assert len(result) <= L


def test_shallow_recall_ignores_factor() -> None:
    mem, rec, _ = _memory(TWO_SCOPES, recall_oversample_factor=3)

    result = mem.recall("q", limit=L, depth="shallow")

    assert rec.limits == [L]
    assert len(result) <= L
