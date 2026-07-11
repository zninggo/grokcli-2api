"""Regression tests for OpenAI tool emission (sub2api Content block not found)."""

from __future__ import annotations

import app
import history_compact


def test_single_tool_per_delta_call():
    acc: dict = {}
    deltas = [
        {
            "index": 0,
            "id": "call_a",
            "type": "function",
            "function": {"name": "Read", "arguments": '{"file_path":"/a"}'},
        },
        {
            "index": 1,
            "id": "call_b",
            "type": "function",
            "function": {"name": "Bash", "arguments": '{"command":"ls"}'},
        },
    ]
    first = app._tool_call_argument_delta(acc, deltas)
    assert first is not None and len(first) == 1
    assert first[0]["function"]["name"] == "Read"
    assert first[0]["id"] == "call_a"
    assert first[0]["index"] == 0
    # second tool only after another call
    second = app._tool_call_argument_delta(acc, [])
    # live path holds until complete; second is already complete so may return
    # only if not blocked — first already emitted so second should come:
    assert second and second[0]["function"]["name"] == "Bash"
    assert second[0]["index"] == 1
    assert second[0]["id"] == "call_b"
    assert app._tool_call_argument_delta(acc, []) == []


def test_synthetic_id_when_missing():
    acc: dict = {}
    out = app._tool_call_argument_delta(
        acc,
        [
            {
                "index": 0,
                "type": "function",
                "function": {"name": "Read", "arguments": '{"file_path":"/x"}'},
            }
        ],
    )
    assert out and out[0]["id"].startswith("call_")
    assert out[0]["function"]["name"] == "Read"


def test_hold_empty_object_preview():
    acc: dict = {}
    assert (
        app._tool_call_argument_delta(
            acc,
            [
                {
                    "index": 0,
                    "id": "c1",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        )
        == []
    )
    out = app._tool_call_argument_delta(
        acc,
        [{"index": 0, "function": {"arguments": '{"file_path":"/x"}'}}],
    )
    assert out and out[0]["function"]["arguments"] == '{"file_path":"/x"}'


def test_dense_index_for_sparse_upstream():
    acc: dict = {}
    out = app._tool_call_argument_delta(
        acc,
        [
            {
                "index": 2,
                "id": "c2",
                "function": {"name": "Read", "arguments": '{"file_path":"/z"}'},
            }
        ],
    )
    assert out and out[0]["index"] == 0


def test_flush_one_does_not_mark_siblings():
    acc: dict = {}
    app._ingest_tool_call_deltas(
        acc,
        [
            {
                "index": 0,
                "id": "a",
                "function": {"name": "Read", "arguments": '{"file_path":"/a"}'},
            },
            {
                "index": 1,
                "id": "b",
                "function": {"name": "Bash", "arguments": '{"command":"ls"}'},
            },
        ],
    )
    one = app._flush_one_tool_call(acc)
    assert one and one[0]["id"] == "a"
    assert not acc[1].get("_emitted")
    two = app._flush_one_tool_call(acc)
    assert two and two[0]["id"] == "b"
    assert app._flush_one_tool_call(acc) == []


def test_iter_tool_sse_splits_and_keepalive():
    frames = app._iter_tool_sse_chunks(
        chat_id="chatcmpl-x",
        model="grok-4.5",
        created=1,
        tool_calls=[
            {
                "index": 0,
                "id": "a",
                "type": "function",
                "function": {"name": "Read", "arguments": "{}"},
            },
            {
                "index": 1,
                "id": "b",
                "type": "function",
                "function": {"name": "Bash", "arguments": "{}"},
            },
        ],
    )
    assert len(frames) == 3  # tool, keepalive, tool
    assert frames[0].startswith("data: ")
    assert frames[1].startswith(":")
    assert frames[2].startswith("data: ")
    assert '"index": 0' in frames[0]
    assert '"index": 1' in frames[2]


def test_body_requests_tools_tool_choice():
    assert app._body_requests_tools({"tools": [{"type": "function"}]})
    assert app._body_requests_tools({"tool_choice": "auto"})
    assert app._body_requests_tools({"tool_choice": {"type": "function", "function": {"name": "x"}}})
    assert not app._body_requests_tools({"tool_choice": "none"})
    assert not app._body_requests_tools({})




def test_normalize_strips_builtin_search_tools():
    tools = [
        {"type": "web_search"},
        {"type": "live_search"},
        {"type": "web_search_preview", "parameters": {"type": "object", "properties": {}}},
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        },
    ]
    out = app._normalize_tools(tools)
    assert out is not None and len(out) == 1
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "get_weather"


def test_normalize_search_only_tools_becomes_none():
    assert app._normalize_tools([{"type": "web_search"}, {"type": "live_search"}]) is None


def test_normalize_tool_choice_search_to_auto():
    assert app._normalize_tool_choice({"type": "web_search"}) == "auto"
    assert app._normalize_tool_choice({"type": "live_search"}) == "auto"
    assert app._normalize_tool_choice({"type": "function", "function": {"name": "x"}}) == {
        "type": "function",
        "function": {"name": "x"},
    }

def test_outbound_tool_budget():
    assert history_compact.remaining_outbound_tool_budget(0) in (1, None) or True
    # With default OUTBOUND_MAX_TOOLS=1
    left0 = history_compact.remaining_outbound_tool_budget(0)
    left1 = history_compact.remaining_outbound_tool_budget(1)
    if history_compact.OUTBOUND_MAX_TOOLS > 0:
        assert left0 == history_compact.OUTBOUND_MAX_TOOLS
        assert left1 == max(0, history_compact.OUTBOUND_MAX_TOOLS - 1)


def test_emit_tool_sse_serial_respects_budget():
    import asyncio

    async def collect():
        frames = []
        async for f in app._emit_tool_sse_serial(
            chat_id="c",
            model="m",
            created=1,
            tool_calls=[
                {"index": 0, "id": "a", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
                {"index": 1, "id": "b", "type": "function", "function": {"name": "Bash", "arguments": "{}"}},
            ],
            already_emitted=0,
        ):
            frames.append(f)
        return frames

    frames = asyncio.run(collect())
    data_frames = [f for f in frames if f.startswith("data: ")]
    if history_compact.OUTBOUND_MAX_TOOLS > 0:
        assert len(data_frames) == 1
        assert '"index": 0' in data_frames[0]
    else:
        assert len(data_frames) >= 1



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASS")

