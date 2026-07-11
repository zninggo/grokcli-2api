"""Regression tests for OpenAI tool emission (sub2api Content block not found)."""

from __future__ import annotations

import app


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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASS")
