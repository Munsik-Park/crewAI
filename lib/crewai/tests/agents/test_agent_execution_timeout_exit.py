"""Program exit after a ``max_execution_time`` timeout.

A program that returns right after a timed-out call exits once the step that
was in progress at the limit ends. That step ends while the interpreter is
shutting down, when no further work can be scheduled. The run must still end
quietly: no error printed or logged at the default level, and nothing after
the step runs. Its real error stays available at debug level.

Interpreter exit cannot happen inside the test process, so the program runs
in a subprocess.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


PROGRAM = textwrap.dedent(
    '''
    import logging
    import sys
    import time
    from typing import Any

    from crewai import Agent, Task
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool

    debug_records = logging.FileHandler(sys.argv[1])
    executor_logger = logging.getLogger("crewai.agents.agent_executor")
    executor_logger.addHandler(debug_records)
    executor_logger.setLevel(logging.DEBUG)


    class SlowLLM(BaseLLM):
        def __init__(self) -> None:
            super().__init__(model="scripted-exit-test")

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 8192

        def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
            if "NEXT_STEP_OUTPUT" in str(messages):
                return "Thought: done\\nFinal Answer: late"
            time.sleep(2)
            print("STEP_IN_FLIGHT_ENDED", flush=True)
            return "Thought: tool\\nAction: next_step\\nAction Input: {}"


    @tool("next_step")
    def next_step() -> str:
        """The step after the one in flight."""
        print("NEXT_STEP_RAN", flush=True)
        return "NEXT_STEP_OUTPUT"


    agent = Agent(
        role="Exit tester",
        goal="answer",
        backstory="b",
        llm=SlowLLM(),
        tools=[next_step],
        max_execution_time=1,
        max_retry_limit=0,
    )
    try:
        agent.execute_task(
            Task(description="TASK: answer", expected_output="x", agent=agent)
        )
    except TimeoutError:
        print("CALLER_TIMED_OUT", flush=True)
    '''
)

QUIET_END_VIOLATIONS = (
    "cannot schedule new futures",
    "Failed to signal flow failure",
    "Error executing listener",
    "An unknown error occurred",
    "Traceback",
)


def test_program_exits_quietly_after_the_step_in_flight(tmp_path: Path) -> None:
    """The process exits after the step in flight, with no error output and no further step."""
    program = tmp_path / "timed_out_program.py"
    program.write_text(PROGRAM)
    debug_log = tmp_path / "debug.log"
    env = dict(os.environ)
    env.update(
        {
            "CREWAI_DISABLE_TELEMETRY": "true",
            "OTEL_SDK_DISABLED": "true",
            "CREWAI_TRACING_ENABLED": "false",
        }
    )

    done = subprocess.run(
        [sys.executable, str(program), str(debug_log)],
        capture_output=True,
        text=True,
        timeout=45,
        cwd=tmp_path,
        env=env,
    )
    stdout_lines = done.stdout.splitlines()
    debug_text = debug_log.read_text() if debug_log.exists() else ""
    report = (
        f"exit {done.returncode}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
        f"\ndebug log:\n{debug_text}"
    )

    assert done.returncode == 0, report
    assert "CALLER_TIMED_OUT" in stdout_lines, report
    assert "STEP_IN_FLIGHT_ENDED" in stdout_lines, report
    assert "NEXT_STEP_RAN" not in stdout_lines, report
    for text in QUIET_END_VIOLATIONS:
        assert text not in done.stderr, report
        assert text not in done.stdout, report
    assert "cannot schedule new futures" in debug_text, (
        "the step's own error is not available at debug level\n" + report
    )
