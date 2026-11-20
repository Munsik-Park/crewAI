"""The executor a coworker holds, after a delegation whose setup straddled the limit.

A later call on the coworker may already be running on that executor. The
abandoned run's setup ends after the limit, so it must not write its own
task into the executor.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any

import pytest

from tests.agents.test_agent_execution_timeout import (
    JOIN_BOUND,
    _join_timed_runs,
    _task,
    _TrackingExecutor,
)
from tests.agents.test_agent_execution_timeout_coworker_setup import _Coworker


def _assert_executor_untouched(coworker: _Coworker) -> None:
    executor = coworker.agent.agent_executor
    assert executor is coworker.executor, "the coworker's executor was replaced"
    assert executor is not None
    assert executor.task is coworker.pre_task, (
        "the timed-out run wrote its task into the coworker's executor"
    )


def test_delegation_setup_across_the_limit_leaves_the_coworkers_executor_as_it_was() -> (
    None
):
    """A delegation whose tool conversion ends after the limit writes nothing into the executor the coworker holds."""
    coworker = _Coworker("delegation", "conversion")
    coworker.record(coworker.agent.execute_task(coworker.pre_task))
    agent = coworker.timed_agent()

    before = set(threading.enumerate())
    try:
        with pytest.raises(TimeoutError):
            agent.execute_task(_task(agent, "TASK_ONE"))
        assert coworker.gate.entered, "the coworker's setup was never reached"
    finally:
        coworker.gate.release.set()
        still_alive = _join_timed_runs(before)
    assert still_alive == []
    _assert_executor_untouched(coworker)


@pytest.mark.asyncio
async def test_delegation_setup_across_the_async_limit_leaves_the_coworkers_executor_as_it_was() -> (
    None
):
    """After an async timeout, a delegation whose tool conversion ends after the limit writes nothing into the executor the coworker holds."""
    loop = asyncio.get_running_loop()
    pool = _TrackingExecutor()
    loop.set_default_executor(pool)
    coworker = _Coworker("delegation", "conversion")
    coworker.record(await coworker.agent.aexecute_task(coworker.pre_task))
    agent = coworker.timed_agent()
    in_flight_at_timeout: list[set[concurrent.futures.Future[Any]]] = []

    try:
        with pytest.raises(TimeoutError):
            await agent.aexecute_task(_task(agent, "TASK_ONE"))
        in_flight_at_timeout.append(pool.snapshot())
        assert coworker.gate.entered, "the coworker's setup was never reached"
    finally:
        coworker.gate.release.set()
        if in_flight_at_timeout:
            concurrent.futures.wait(in_flight_at_timeout[0], timeout=JOIN_BOUND)
        pool.shutdown(wait=True)
    _assert_executor_untouched(coworker)
