"""Tests for @chatprompt decorator."""

from inspect import getdoc
from unittest.mock import AsyncMock, Mock

import pytest

from magentic.chat_model.message import AssistantMessage, SystemMessage, UserMessage
from magentic.chatprompt import AsyncChatPromptFunction, ChatPromptFunction, chatprompt


def test_chatpromptfunction_format():
    @chatprompt(
        SystemMessage("This is a system message with {one}."),
        UserMessage("This is a {two} user message."),
        AssistantMessage("This {three} is an assistant message."),
        AssistantMessage([1, 2, 3]),
    )
    def func(one: int, two: bool, three: str) -> str:  # noqa: FBT001
        ...

    assert func.format(one=1, two=True, three="three") == [
        SystemMessage("This is a system message with 1."),
        UserMessage("This is a True user message."),
        AssistantMessage("This three is an assistant message."),
        AssistantMessage([1, 2, 3]),
    ]


def test_chatpromptfunction_call():
    mock_model = Mock()
    mock_model.complete.return_value = AssistantMessage(content="Hello!")

    @chatprompt(
        UserMessage("Hello {name}."),
        model=mock_model,
    )
    def say_hello(name: str) -> str | bool:
        ...

    assert say_hello("World") == "Hello!"
    assert mock_model.complete.call_count == 1
    assert mock_model.complete.call_args.kwargs["messages"] == [
        UserMessage("Hello World.")
    ]
    assert mock_model.complete.call_args.kwargs["output_types"] == [str, bool]


def test_chatprompt_decorator_docstring():
    @chatprompt(UserMessage("This is a user message."))
    def func(one: int) -> str:
        """This is the docstring."""
        ...

    assert isinstance(func, ChatPromptFunction)
    assert getdoc(func) == "This is the docstring."


@pytest.mark.asyncio
async def test_asyncchatpromptfunction_call():
    mock_model = AsyncMock()
    mock_model.acomplete.return_value = AssistantMessage(content="Hello!")

    @chatprompt(
        UserMessage("Hello {name}."),
        model=mock_model,
    )
    async def say_hello(name: str) -> str | bool:
        ...

    assert await say_hello("World") == "Hello!"
    assert mock_model.acomplete.call_count == 1
    assert mock_model.acomplete.call_args.kwargs["messages"] == [
        UserMessage("Hello World.")
    ]
    assert mock_model.acomplete.call_args.kwargs["output_types"] == [str, bool]


@pytest.mark.asyncio
async def test_async_chatprompt_decorator_docstring():
    @chatprompt(UserMessage("This is a user message."))
    async def func(one: int) -> str:
        """This is the docstring."""
        ...

    assert isinstance(func, AsyncChatPromptFunction)
    assert getdoc(func) == "This is the docstring."
