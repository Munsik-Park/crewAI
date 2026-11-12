"""Recall over-fetches from storage by the oversample factor configured on Memory."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai import Agent, Crew, Task
from crewai.memory.recall_flow import RecallFlow
import crewai.memory.storage.factory as storage_factory
from crewai.memory.types import MemoryConfig, MemoryRecord
import crewai.memory.unified_memory as unified_memory
from crewai.memory.unified_memory import Memory
from crewai.state.checkpoint_config import CheckpointConfig
from crewai.state.runtime import RuntimeState


FACTOR = 3
LIMIT = 4
QUERY = "short query"


class RecordingStorage:
    """Storage double that records the limit of every search it is asked for."""

    def __init__(self) -> None:
        self.search_limits: list[int] = []

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        self.search_limits.append(limit)
        # A low score and zero importance keep recall confidence below the
        # exploration threshold, so a deep recall re-searches after exploring.
        return [
            (
                MemoryRecord(content=f"record {i}", scope=scope_prefix or "/", importance=0.0),
                0.01,
            )
            for i in range(limit)
        ]

    def list_scopes(self, parent: str = "/") -> list[str]:
        return []

    def touch_records(self, record_ids: list[str]) -> None:
        pass


def embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3] for _ in texts]


def make_memory(storage: RecordingStorage, **kwargs: Any) -> Memory:
    return Memory(storage=storage, llm=MagicMock(), embedder=embed, **kwargs)


def make_crew(agent_memory: Memory, **kwargs: Any) -> Crew:
    agent = Agent(
        role="researcher",
        goal="find facts",
        backstory="careful",
        llm="gpt-4o-mini",
        memory=agent_memory,
    )
    task = Task(description="research", expected_output="facts", agent=agent)
    return Crew(agents=[agent], tasks=[task], **kwargs)


@pytest.fixture
def storages_by_spec() -> Iterator[dict[str, list[RecordingStorage]]]:
    created: dict[str, list[RecordingStorage]] = {}

    def factory(spec: str) -> RecordingStorage:
        storage = RecordingStorage()
        created.setdefault(spec, []).append(storage)
        return storage

    original = storage_factory._factory
    storage_factory.set_memory_storage_factory(factory)
    yield created
    storage_factory.set_memory_storage_factory(original)


def test_deep_recall_asks_storage_for_limit_times_configured_factor() -> None:
    storage = RecordingStorage()
    memory = make_memory(storage, recall_oversample_factor=FACTOR, exploration_budget=1)

    matches = memory.recall(QUERY, limit=LIMIT, depth="deep")

    assert len(storage.search_limits) >= 2, (
        f"expected the first search and a re-search, got {storage.search_limits}"
    )
    assert storage.search_limits == [LIMIT * FACTOR] * len(storage.search_limits), (
        f"every deep-recall search must ask for limit x {FACTOR}"
    )
    assert len(matches) == LIMIT, "deep recall must return at most the requested limit"


def test_recall_flow_asks_storage_for_limit_times_its_config_factor() -> None:
    storage = RecordingStorage()
    flow = RecallFlow(
        storage=storage,
        llm=MagicMock(),
        embedder=embed,
        config=MemoryConfig(recall_oversample_factor=FACTOR, exploration_budget=1),
    )

    flow.kickoff(inputs={"query": QUERY, "limit": LIMIT, "categories": []})

    assert len(storage.search_limits) >= 2, (
        f"expected the first search and a re-search, got {storage.search_limits}"
    )
    assert storage.search_limits == [LIMIT * FACTOR] * len(storage.search_limits), (
        f"every RecallFlow search must ask for limit x {FACTOR}"
    )


def test_slice_shallow_recall_asks_storage_for_limit_times_memory_factor() -> None:
    storage = RecordingStorage()
    memory = make_memory(storage, recall_oversample_factor=FACTOR)

    matches = memory.slice(scopes=["/a", "/b"]).recall(QUERY, limit=LIMIT, depth="shallow")

    assert storage.search_limits == [LIMIT * FACTOR, LIMIT * FACTOR], (
        f"a shallow slice recall must search each scope once for limit x {FACTOR}"
    )
    assert len(matches) == LIMIT, "slice recall must return at most the requested limit"


def test_slice_deep_recall_asks_storage_for_limit_times_factor_squared() -> None:
    storage = RecordingStorage()
    memory = make_memory(storage, recall_oversample_factor=FACTOR, exploration_budget=1)

    matches = memory.slice(scopes=["/a", "/b"]).recall(QUERY, limit=LIMIT, depth="deep")

    assert len(storage.search_limits) >= 2, (
        f"expected at least one search per scope, got {storage.search_limits}"
    )
    assert storage.search_limits == [LIMIT * FACTOR * FACTOR] * len(storage.search_limits), (
        f"a deep slice recall must ask storage for limit x {FACTOR} x {FACTOR}"
    )
    assert len(matches) == LIMIT, "slice recall must return at most the requested limit"


def test_crew_copy_keeps_configured_factor_of_agent_memory() -> None:
    storage = RecordingStorage()
    agent_memory = Memory(
        storage=storage, embedder=embed, recall_oversample_factor=FACTOR, exploration_budget=0
    )

    copied_memory = make_crew(agent_memory).copy().agents[0].memory

    assert isinstance(copied_memory, Memory)
    assert copied_memory is not agent_memory, "Crew.copy() must rebuild the agent's Memory"
    copied_memory.recall(QUERY, limit=LIMIT, depth="deep")
    assert storage.search_limits == [LIMIT * FACTOR], (
        f"the copied Memory must over-fetch by the configured factor {FACTOR}"
    )


def test_checkpoint_restore_keeps_configured_factor_of_crew_and_agent_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storages_by_spec: dict[str, list[RecordingStorage]],
) -> None:
    # The checkpoint stores no embedder callable, so a restored Memory builds the
    # default embedder; it is replaced to keep the test offline.
    monkeypatch.setattr(unified_memory, "_default_embedder", lambda: embed)
    crew = make_crew(
        Memory(storage="agent-store", recall_oversample_factor=FACTOR, exploration_budget=0),
        memory=Memory(storage="crew-store", recall_oversample_factor=FACTOR, exploration_budget=0),
    )
    location = RuntimeState(root=[crew]).checkpoint(str(tmp_path))

    restored = Crew.from_checkpoint(CheckpointConfig(restore_from=location))

    restored_agent_memory = restored.agents[0].memory
    assert isinstance(restored.memory, Memory)
    assert isinstance(restored_agent_memory, Memory)
    assert restored.memory is not crew.memory, "the restore must rebuild the crew's Memory"
    assert restored_agent_memory is not crew.agents[0].memory, (
        "the restore must rebuild the agent's Memory"
    )
    for spec, memory in (("crew-store", restored.memory), ("agent-store", restored_agent_memory)):
        memory.recall(QUERY, limit=LIMIT, depth="deep")
        assert storages_by_spec[spec][-1].search_limits == [LIMIT * FACTOR], (
            f"the {spec} Memory restored from the checkpoint must over-fetch by {FACTOR}"
        )


def test_default_memory_deep_recall_asks_storage_for_twice_the_limit() -> None:
    storage = RecordingStorage()
    memory = make_memory(storage)

    memory.recall(QUERY, limit=LIMIT, depth="deep")

    assert storage.search_limits, "deep recall must search storage"
    assert storage.search_limits == [LIMIT * 2] * len(storage.search_limits), (
        "the default factor must stay 2"
    )


def test_default_memory_shallow_recall_asks_storage_for_exactly_the_limit() -> None:
    storage = RecordingStorage()
    memory = make_memory(storage)

    memory.recall(QUERY, limit=LIMIT, depth="shallow")

    assert storage.search_limits == [LIMIT], "shallow recall must not over-fetch"
