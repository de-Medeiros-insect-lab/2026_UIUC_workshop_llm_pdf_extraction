from types import SimpleNamespace
from workshop_lib import run_tool_loop, TOOL_SCHEMAS


def _reply(content="", tool_calls=None):
    calls = [
        SimpleNamespace(function=SimpleNamespace(name=n, arguments=a))
        for n, a in (tool_calls or [])
    ]
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=calls or None)
    )


def test_returns_text_when_no_tools_requested():
    fake = lambda **kw: _reply(content="done")
    text, calls = run_tool_loop([{"role": "user", "content": "hi"}], [], {},
                                chat=fake)
    assert text == "done"
    assert calls == []


def test_executes_a_tool_then_finishes():
    replies = iter([
        _reply(tool_calls=[("get_page_text", {"page": 5})]),
        _reply(content="the answer"),
    ])
    fake = lambda **kw: next(replies)
    impls = {"get_page_text": lambda page: f"text of {page}"}
    text, calls = run_tool_loop([{"role": "user", "content": "go"}],
                                TOOL_SCHEMAS, impls, chat=fake)
    assert text == "the answer"
    assert calls == [("get_page_text", {"page": 5})]


def test_tool_errors_are_fed_back_not_raised():
    replies = iter([
        _reply(tool_calls=[("ocr_page", {"page": 99})]),
        _reply(content="recovered"),
    ])
    captured_messages = []

    def fake(**kw):
        captured_messages.append(kw.get("messages"))
        return next(replies)

    def boom(page):
        raise ValueError("page 99 out of range")

    text, calls = run_tool_loop([{"role": "user", "content": "go"}],
                                TOOL_SCHEMAS, {"ocr_page": boom}, chat=fake)
    assert text == "recovered"
    assert calls == [("ocr_page", {"page": 99})]
    # Verify error was fed back to the model in the second chat call
    second_call_messages = captured_messages[1]
    error_message = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "page 99 out of range" in error_message["content"]


def test_unknown_tool_is_reported_back():
    replies = iter([
        _reply(tool_calls=[("no_such_tool", {})]),
        _reply(content="ok"),
    ])
    captured_messages = []

    def fake(**kw):
        captured_messages.append(kw.get("messages"))
        return next(replies)

    text, _ = run_tool_loop([{"role": "user", "content": "go"}], [], {},
                            chat=fake)
    assert text == "ok"
    # Verify unknown tool name was reported back to the model
    second_call_messages = captured_messages[1]
    error_message = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "no_such_tool" in error_message["content"]


def test_max_turns_is_enforced():
    fake = lambda **kw: _reply(tool_calls=[("get_page_text", {"page": 1})])
    impls = {"get_page_text": lambda page: "x"}
    text, calls = run_tool_loop([{"role": "user", "content": "go"}],
                                TOOL_SCHEMAS, impls, max_turns=3, chat=fake)
    assert len(calls) == 3
    assert "max_turns" in text


def test_tool_schemas_describe_both_tools():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"get_page_text", "ocr_page"}
    ocr = next(t for t in TOOL_SCHEMAS
               if t["function"]["name"] == "ocr_page")
    assert "page" in ocr["function"]["parameters"]["properties"]


def test_think_defaults_to_true_and_is_passed_through():
    """Reasoning is what makes the loop escalate to OCR; guard the default."""
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return _reply(content="ok")

    run_tool_loop([{"role": "user", "content": "hi"}], [], {}, chat=fake)
    assert seen["think"] is True

    run_tool_loop([{"role": "user", "content": "hi"}], [], {}, think=False,
                  chat=fake)
    assert seen["think"] is False
