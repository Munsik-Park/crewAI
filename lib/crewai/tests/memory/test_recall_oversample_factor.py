"""Tests for the recall over-fetch factor (``recall_oversample_factor``).

Each test records the ``limit`` that every ``storage.search`` call receives
from a fake storage. The fake reports two candidate scopes under each prefix
and returns low-scoring records, so deep recall always runs the re-search
round: 2 candidate scopes x 2 rounds = 4 storage calls per deep recall.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.memory.memory_scope import MemorySlice
from crewai.memory.recall_flow import RecallFlow
from crewai.memory.types import MemoryConfig, MemoryRecord
from crewai.memory.unified_memory import Memory


LIMIT = 5
CALLS_PER_DEEP_RECALL = 4


class FakeStorage:
    """Storage that records each search and returns as many records as asked.

    ``list_scopes(prefix)`` returns two scopes under ``prefix``, and every
    returned record carries the ``scope_prefix`` it was searched with, so a
    call can be traced back to the slice scope that issued it.
    """

    def __init__(self) -> None:
        self.searches: list[tuple[str | None, int]] = []
        self._mock = MagicMock()

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        self.searches.append((scope_prefix, limit))
        old = datetime.utcnow() - timedelta(days=3650)
        return [
            (
                MemoryRecord(
                    content=f"record {i} in {scope_prefix}",
                    scope=scope_prefix or "/",
                    importance=0.0,
                    created_at=old,
                ),
                0.0,
            )
            for i in range(limit)
        ]

    def list_scopes(self, prefix: str = "/") -> list[str]:
        base = prefix.rstrip("/")
        return [f"{base}/x", f"{base}/y"]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mock, name)

    @property
    def limits(self) -> list[int]:
        return [limit for _, limit in self.searches]

    @property
    def scopes(self) -> list[str | None]:
        return [scope for scope, _ in self.searches]


def _embedder() -> MagicMock:
    return MagicMock(side_effect=lambda texts: [[0.1] * 8 for _ in texts])


def _memory(storage: FakeStorage, **kwargs: Any) -> Memory:
    return Memory(storage=storage, llm=MagicMock(), embedder=_embedder(), **kwargs)


def _slice(storage: FakeStorage, **kwargs: Any) -> MemorySlice:
    return MemorySlice(scopes=["/a", "/b"], memory=_memory(storage, **kwargs))


# --- The configured factor is used wherever recall over-fetches (C1) ---


def test_memory_deep_recall_asks_storage_for_limit_times_factor() -> None:
    storage = FakeStorage()

    _memory(storage, recall_oversample_factor=3).recall("q", limit=LIMIT)

    assert len(storage.searches) == CALLS_PER_DEEP_RECALL
    assert storage.limits == [LIMIT * 3] * CALLS_PER_DEEP_RECALL
    assert set(storage.scopes) == {"/x", "/y"}


def test_recall_flow_uses_factor_from_its_config() -> None:
    storage = FakeStorage()
    flow = RecallFlow(
        storage=storage,
        llm=MagicMock(),
        embedder=_embedder(),
        config=MemoryConfig(recall_oversample_factor=1),
    )

    flow.kickoff(
        inputs={
            "query": "q",
            "scope": None,
            "categories": [],
            "limit": LIMIT,
            "source": None,
            "include_private": False,
        }
    )

    assert len(storage.searches) == CALLS_PER_DEEP_RECALL
    assert storage.limits == [LIMIT * 1] * CALLS_PER_DEEP_RECALL


def test_slice_deep_recall_compounds_factor_per_slice_scope() -> None:
    storage = FakeStorage()

    _slice(storage, recall_oversample_factor=3).recall("q", limit=LIMIT)

    assert len(storage.searches) == 2 * CALLS_PER_DEEP_RECALL
    assert storage.limits == [LIMIT * 3 * 3] * (2 * CALLS_PER_DEEP_RECALL)
    slice_roots = [scope.split("/")[1] for scope in storage.scopes]
    assert slice_roots.count("a") == CALLS_PER_DEEP_RECALL
    assert slice_roots.count("b") == CALLS_PER_DEEP_RECALL


def test_slice_shallow_recall_asks_each_slice_scope_for_limit_times_factor() -> None:
    storage = FakeStorage()

    _slice(storage, recall_oversample_factor=3).recall(
        "q", limit=LIMIT, depth="shallow"
    )

    assert sorted(storage.searches) == [("/a", LIMIT * 3), ("/b", LIMIT * 3)]


def test_memory_shallow_recall_does_not_over_fetch() -> None:
    storage = FakeStorage()

    _memory(storage, recall_oversample_factor=3).recall(
        "q", limit=LIMIT, depth="shallow"
    )

    assert storage.limits == [LIMIT]


# --- The default factor (2) is unchanged (C2) ---


def test_default_memory_deep_recall_over_fetches_by_two() -> None:
    storage = FakeStorage()

    _memory(storage).recall("q", limit=LIMIT)

    assert storage.limits == [LIMIT * 2] * CALLS_PER_DEEP_RECALL


def test_default_memory_shallow_recall_asks_for_limit() -> None:
    storage = FakeStorage()

    _memory(storage).recall("q", limit=LIMIT, depth="shallow")

    assert storage.limits == [LIMIT]


def test_default_slice_deep_recall_over_fetches_by_four() -> None:
    storage = FakeStorage()

    _slice(storage).recall("q", limit=LIMIT)

    assert storage.limits == [LIMIT * 4] * (2 * CALLS_PER_DEEP_RECALL)


def test_default_slice_shallow_recall_over_fetches_by_two() -> None:
    storage = FakeStorage()

    _slice(storage).recall("q", limit=LIMIT, depth="shallow")

    assert sorted(storage.searches) == [("/a", LIMIT * 2), ("/b", LIMIT * 2)]


# --- The final result cap is unchanged (C2) ---


@pytest.mark.parametrize(
    ("make_target", "depth"),
    [
        (_memory, "deep"),
        (_slice, "deep"),
        (_slice, "shallow"),
    ],
    ids=["memory-deep", "slice-deep", "slice-shallow"],
)
def test_recall_returns_at_most_limit_with_a_larger_factor(
    make_target: Any, depth: str
) -> None:
    storage = FakeStorage()

    matches = make_target(storage, recall_oversample_factor=3).recall(
        "q", limit=LIMIT, depth=depth
    )

    assert sum(limit for _, limit in storage.searches) > LIMIT
    assert len(matches) <= LIMIT
