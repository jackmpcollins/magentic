"""Regression behavior callable on unchanged source without the new public class."""

from magentic import SystemMessage, UserMessage
from magentic.chat_model.anthropic_chat_model import _extract_system_message


def test_multiple_system_messages_keep_order() -> None:
    system, messages = _extract_system_message(
        [SystemMessage("one"), SystemMessage("two"), UserMessage("hello")]
    )
    assert system == [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    assert messages == [UserMessage("hello")]


def test_single_pass_input_does_not_lose_user_message() -> None:
    system, messages = _extract_system_message(
        iter([SystemMessage("one"), UserMessage("hello")])
    )
    assert system == "one"
    assert messages == [UserMessage("hello")]


def test_plain_single_system_is_still_a_string() -> None:
    assert _extract_system_message([SystemMessage("one"), UserMessage("hello")]) == (
        "one",
        [UserMessage("hello")],
    )
