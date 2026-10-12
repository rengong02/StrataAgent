"""A batch may only go to the thread pool when every call in it is independent.

The order of calls inside one assistant message is the model's *intent*. A pool
does not honour it, so a batch holding a writer and a reader of the same key
races -- and loses. This was found in the GeoBenchX adapter, where every tool
shares one `data_store` and a scripted writer/reader pair failed in 27 of 30
batches; this file pins the mechanism with no GeoBenchX and no datasets.

The pair below is the point: the same batch is *expected to fail* when the tool
declares itself concurrency-safe, and to work when it does not. Without that
contrast the fix would look arbitrary.
"""

from __future__ import annotations

import time
from typing import ClassVar

from corecoder.agent import Agent
from corecoder.demo import ScriptedLLM
from corecoder.llm import LLMResponse, ToolCall
from corecoder.tools.base import Tool

_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "key": {"type": "string"},
        "value": {"type": "string"},
    },
    "required": ["action", "key"],
}


class _Store(Tool):
    """A writer/reader over one shared dict, like GeoBenchX's `data_store`."""

    name = "store"
    description = "write or read one key of a shared store"
    parameters: ClassVar[dict] = _SCHEMA

    def __init__(self, shared: dict, delay: float = 0.0):
        self.shared = shared
        self.delay = delay

    def execute(self, action: str, key: str, value: str = "") -> str:
        if action == "write":
            # The delay is what makes the race deterministic instead of a coin
            # flip: it stands in for a real tool doing real work before it
            # publishes its result.
            time.sleep(self.delay)
            self.shared[key] = value
            return f"wrote {key}"
        if key not in self.shared:
            return f"Error: {key} not found"
        return self.shared[key]


class _PooledStore(_Store):
    """Declares nothing, so the batch it is in goes to the pool."""


class _SerialStore(_Store):
    concurrency_safe = False


def _batch(tool: Tool, shared: dict) -> Agent:
    """One response asking to write a key and then read it back."""
    return Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="c1", name="store",
                         arguments={"action": "write", "key": "k", "value": "v"}),
                ToolCall(id="c2", name="store",
                         arguments={"action": "read", "key": "k"}),
            ]),
            LLMResponse(content="done"),
        ]),
        tools=[tool],
    )


def _reader_result(agent: Agent) -> str:
    """The text the model was handed for the second call."""
    replies = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    return replies[-1]


# --- the declaration itself ---------------------------------------------------


def test_tools_are_concurrency_safe_by_default():
    """Upstream's tools are independent and the pool is a win there.

    Pinned so flipping the default cannot happen silently: that would serialise
    every existing batch and change behaviour nothing asked to change.
    """
    assert Tool.concurrency_safe is True

    class _Plain(Tool):
        name, description = "plain", "d"
        parameters: ClassVar[dict] = _SCHEMA

        def execute(self, **kwargs):
            return "ok"

    assert _Plain().concurrency_safe is True


def test_one_unsafe_tool_serialises_its_whole_batch():
    """The siblings must keep the model's order too, so the whole batch waits."""
    shared: dict = {}
    agent = _batch(_SerialStore(shared, delay=0.05), shared)

    calls = [ToolCall(id="a", name="store", arguments={"action": "read", "key": "k"})]
    assert agent._all_concurrency_safe(calls) is False

    safe_agent = Agent(llm=ScriptedLLM([]), tools=[_PooledStore(shared)])
    assert safe_agent._all_concurrency_safe(calls) is True


# --- what that declaration buys -----------------------------------------------


def test_a_pooled_batch_loses_the_writer_reader_race():
    """The reason the flag exists, reproduced on purpose.

    This asserts a *failure*. It is deterministic only because the writer sleeps
    before publishing; a real tool does work too, which is why the GeoBenchX
    version of this failed 27 times out of 30 rather than every time.
    """
    shared: dict = {}
    agent = _batch(_PooledStore(shared, delay=0.05), shared)
    agent.chat("go")

    assert _reader_result(agent).startswith("Error"), (
        "the reader was expected to reach the store before the writer published; "
        "if this now passes, the race is gone for another reason and this test's "
        "premise needs re-checking"
    )


def test_an_unsafe_batch_runs_in_order_and_the_reader_sees_the_write():
    """The regression: same batch, same tools, only the declaration differs."""
    shared: dict = {}
    agent = _batch(_SerialStore(shared, delay=0.05), shared)
    agent.chat("go")

    assert _reader_result(agent) == "v", "the reader did not see the writer's value"


def test_the_write_reaches_the_store_in_a_serial_batch():
    """And the effect is real, not just a well-ordered reply."""
    shared: dict = {}
    _batch(_SerialStore(shared, delay=0.05), shared).chat("go")
    assert shared == {"k": "v"}
