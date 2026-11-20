"""Regression tests: recall honours the configured recall_oversample_factor.

Each test records the ``limit`` that recall passes to every storage ``search``
call. The fake storage returns as many records as it is asked for, so every
fetch yields more than ``limit`` candidates and the final cap is exercised.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from pydantic import ValidationError
import pytest

from crewai.memory.memory_scope import MemorySlice
from crewai.memory.recall_flow import RecallFlow
from crewai.memory.types import MemoryConfig, MemoryRecord
from crewai.memory.unified_memory import Memory


LIMIT = 4
FACTOR = 3
LOW_SCORE = 0.1  # low confidence: deep recall explores and re-searches
HIGH_SCORE = 0.9  # high confidence: deep recall runs one pass


def _make_storage(score: float) -> tuple[MagicMock, list[int]]:
    calls: list[int] = []
    storage = MagicMock()

    def search(
        embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        **_: Any,
    ) -> list[tuple[MemoryRecord, float]]:
        calls.append(limit)
        return [
            (
                MemoryRecord(
                    content=f"record {i}",
                    scope=scope_prefix or "/",
                    embedding=[1.0] * 4,
                ),
                score - i * 0.001,
            )
            for i in range(limit)
        ]

    storage.search.side_effect = search
    storage.list_scopes.return_value = []
    return storage, calls


def _make_llm() -> MagicMock:
    llm = MagicMock()
    llm.supports_function_calling.return_value = False
    return llm


def _make_embedder() -> MagicMock:
    return MagicMock(side_effect=lambda texts: [[1.0] * 4 for _ in texts])


def _memory(storage: MagicMock, **kwargs: Any) -> Memory:
    return Memory(storage=storage, llm=_make_llm(), embedder=_make_embedder(), **kwargs)


def _assert_passes(calls: list[int], expected: int) -> None:
    assert len(calls) >= 2, calls
    assert all(c == expected for c in calls), calls


# --- factor other than 2: recall asks storage for limit times the factor --


def test_deep_recall_uses_memory_kwarg_factor_on_each_pass() -> None:
    storage, calls = _make_storage(LOW_SCORE)
    mem = _memory(storage, recall_oversample_factor=FACTOR)

    results = mem.recall("short q", limit=LIMIT, depth="deep")

    _assert_passes(calls, LIMIT * FACTOR)
    assert len(results) <= LIMIT


def test_recall_flow_uses_factor_from_given_memory_config() -> None:
    storage, calls = _make_storage(LOW_SCORE)
    flow = RecallFlow(
        storage=storage,
        llm=_make_llm(),
        embedder=_make_embedder(),
        config=MemoryConfig(recall_oversample_factor=FACTOR),
    )

    flow.kickoff(inputs={"query": "short q", "limit": LIMIT})

    _assert_passes(calls, LIMIT * FACTOR)


def test_memory_slice_deep_recall_applies_factor_at_slice_and_deep_search() -> None:
    storage, calls = _make_storage(HIGH_SCORE)
    mem = _memory(storage, recall_oversample_factor=FACTOR)
    sl = MemorySlice(scopes=["/a", "/b"], memory=mem)

    results = sl.recall("short q", limit=LIMIT, depth="deep")

    assert len(calls) >= 2, calls
    assert all(c == LIMIT * FACTOR * FACTOR for c in calls), calls
    assert len(results) <= LIMIT


def test_memory_slice_shallow_recall_applies_factor_per_scope() -> None:
    storage, calls = _make_storage(HIGH_SCORE)
    mem = _memory(storage, recall_oversample_factor=FACTOR)
    sl = MemorySlice(scopes=["/a", "/b"], memory=mem)

    results = sl.recall("short q", limit=LIMIT, depth="shallow")

    assert calls == [LIMIT * FACTOR, LIMIT * FACTOR]
    assert len(results) <= LIMIT


def test_factor_survives_model_dump_and_rebuild() -> None:
    storage, calls = _make_storage(LOW_SCORE)
    original = _memory(storage, recall_oversample_factor=FACTOR)

    dump = original.model_dump()
    rebuilt = Memory(
        **{
            **dump,
            "storage": storage,
            "llm": _make_llm(),
            "embedder": _make_embedder(),
        }
    )
    rebuilt.recall("short q", limit=LIMIT, depth="deep")

    _assert_passes(calls, LIMIT * FACTOR)


def test_memory_rejects_factor_below_one() -> None:
    storage, _ = _make_storage(HIGH_SCORE)
    with pytest.raises(ValidationError):
        _memory(storage, recall_oversample_factor=0)


# --- no factor set: default factor 2 and the result cap -------------------


def test_default_factor_and_cap_unchanged() -> None:
    default = 2

    storage, calls = _make_storage(LOW_SCORE)
    results = _memory(storage).recall("short q", limit=LIMIT, depth="deep")
    _assert_passes(calls, LIMIT * default)
    assert len(results) <= LIMIT

    storage, calls = _make_storage(LOW_SCORE)
    RecallFlow(storage=storage, llm=_make_llm(), embedder=_make_embedder()).kickoff(
        inputs={"query": "short q", "limit": LIMIT}
    )
    _assert_passes(calls, LIMIT * default)

    storage, calls = _make_storage(HIGH_SCORE)
    sl = MemorySlice(scopes=["/a", "/b"], memory=_memory(storage))
    results = sl.recall("short q", limit=LIMIT, depth="deep")
    assert calls == [LIMIT * default * default] * 2
    assert len(results) <= LIMIT

    calls.clear()
    results = sl.recall("short q", limit=LIMIT, depth="shallow")
    assert calls == [LIMIT * default] * 2
    assert len(results) <= LIMIT
