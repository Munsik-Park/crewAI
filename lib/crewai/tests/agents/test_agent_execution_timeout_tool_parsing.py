"""A tool call parsed by ``function_calling_llm`` when the parse ends after the limit.

When the agent's own tool input cannot be read, the tool call is parsed by
``function_calling_llm``. That parse is an LLM call of its own, and the tool
runs after it within the same step. When the parse is the call in flight at
``max_execution_time``, the run left behind does not go on to run the tool.
"""

from __future__ import annotations

import json
import threading

from crewai.tools import tool
from tests.agents.test_agent_execution_timeout import (
    ORPHAN_BLOCK,
    _action,
    _agent,
    _final,
    _ScriptedLLM,
    _task,
)
from tests.agents.test_agent_execution_timeout_in_step import _time_out


def test_timed_out_run_does_not_run_the_tool_whose_call_was_parsed_after_the_limit() -> (
    None
):
    """A tool call that function_calling_llm finishes parsing after the limit is not run."""
    release = threading.Event()
    parses: list[str] = []
    tool_runs: list[str] = []

    @tool("lookup")
    def lookup() -> str:
        """Look something up."""
        tool_runs.append("lookup")
        return "LOOKUP_RESULT"

    def agent_script(text: str) -> str:
        if "LOOKUP_RESULT" in text:
            return _final("orphan answer")
        # Not an object, so the agent's own parsing cannot read it.
        return _action("lookup", '"look it up"')

    def parsing_script(text: str) -> str:
        parses.append(text)
        if len(parses) == 1:
            release.wait(ORPHAN_BLOCK)
        return json.dumps({"tool_name": "lookup", "arguments": {}})

    agent = _agent(
        _ScriptedLLM(agent_script),
        tools=[lookup],
        function_calling_llm=_ScriptedLLM(parsing_script),
    )

    _time_out(agent, _task(agent, "TASK_ONE"), release)

    assert len(parses) == 1, "the parse was not the call in flight"
    assert tool_runs == [], "the tool ran after the limit"
