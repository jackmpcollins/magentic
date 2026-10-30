from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import anthropic
import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from magentic import Chat, SystemMessage, UserMessage
from magentic.chat_model.anthropic_chat_model import (
    AnthropicChatModel,
    AnthropicSystemMessage,
    _extract_system_message,
)
from magentic.chat_model.message import AssistantMessage, Message


def test_cache_control_is_a_text_block_not_a_message_level_field() -> None:
    cached = AnthropicSystemMessage("stable", cache_control={"type": "ephemeral"})
    system, other = _extract_system_message([cached, UserMessage("question")])
    assert system == [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}
    ]
    assert other == [UserMessage("question")]


def test_multiple_system_blocks_preserve_order_and_only_explicit_breakpoints() -> None:
    system, other = _extract_system_message(
        [
            SystemMessage("first"),
            AnthropicSystemMessage("cached", cache_control={"type": "ephemeral"}),
            AnthropicSystemMessage("uncached"),
            UserMessage("question"),
        ]
    )
    assert system == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "uncached"},
    ]
    assert other == [UserMessage("question")]


def test_ordinary_single_system_retains_string_payload() -> None:
    assert _extract_system_message([SystemMessage("ordinary")]) == ("ordinary", [])
    assert _extract_system_message([UserMessage("hello")]) == (
        anthropic.NOT_GIVEN,
        [UserMessage("hello")],
    )


def test_one_shot_message_iterable_keeps_conversation() -> None:
    def messages() -> Iterator[Message[Any]]:
        yield SystemMessage("system")
        yield UserMessage("first")
        yield AssistantMessage("answer")
        yield UserMessage("second")

    assert _extract_system_message(messages()) == (
        "system",
        [UserMessage("first"), AssistantMessage("answer"), UserMessage("second")],
    )


def test_format_preserves_cache_metadata_and_does_not_mutate_original() -> None:
    message = AnthropicSystemMessage(
        "hello {name}", cache_control={"type": "ephemeral"}
    )
    formatted = message.format(name="Ada")
    assert isinstance(formatted, AnthropicSystemMessage)
    assert formatted.content == "hello Ada"
    assert formatted.cache_control == {"type": "ephemeral"}
    assert message.content == "hello {name}"


def test_cache_control_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        AnthropicSystemMessage.model_validate(
            {"content": "text", "cache_control": {"type": "persistent"}}
        )


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chat_complete_passes_blocks_to_sdk_without_network(
    mocker: MockerFixture, *, async_mode: bool
) -> None:
    model = AnthropicChatModel("test", api_key="test-not-a-real-key")
    cached = AnthropicSystemMessage("stable", cache_control={"type": "ephemeral"})
    chat = Chat([SystemMessage("first"), cached, UserMessage("question")], model=model)
    if async_mode:
        stream_context = Mock()
        stream_context.__aenter__ = AsyncMock(return_value=Mock())
        sdk = mocker.patch.object(
            model._async_client.messages, "stream", return_value=stream_context
        )
        mocker.patch(
            "magentic.chat_model.anthropic_chat_model.aparse_stream",
            new_callable=AsyncMock,
            return_value="answer",
        )
        result = await chat.asubmit()
    else:
        stream_context = Mock()
        stream_context.__enter__ = Mock(return_value=iter(()))
        sdk = mocker.patch.object(
            model._client.messages, "stream", return_value=stream_context
        )
        mocker.patch(
            "magentic.chat_model.anthropic_chat_model.parse_stream",
            return_value="answer",
        )
        result = chat.submit()
    assert result.last_message.content == "answer"
    assert sdk.call_args.kwargs["system"] == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
    ]
    assert sdk.call_args.kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "question"}]}
    ]


def test_multiple_ordinary_system_messages_are_ordered_blocks() -> None:
    assert _extract_system_message(
        [SystemMessage("first"), SystemMessage("second")]
    ) == ([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}], [])


def test_multiple_explicit_breakpoints_are_not_merged_or_added() -> None:
    system, _ = _extract_system_message(
        [
            AnthropicSystemMessage("one", cache_control={"type": "ephemeral"}),
            AnthropicSystemMessage("two", cache_control={"type": "ephemeral"}),
        ]
    )
    assert system == [
        {"type": "text", "text": "one", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "two", "cache_control": {"type": "ephemeral"}},
    ]
