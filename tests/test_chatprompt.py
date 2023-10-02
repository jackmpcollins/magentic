"""Tests for @chatprompt decorator."""

from inspect import getdoc
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from magentic.chat_model.message import AssistantMessage, SystemMessage, UserMessage
from magentic.chatprompt import AsyncChatPromptFunction, ChatPromptFunction, chatprompt


@pytest.mark.parametrize(
    ("message_templates", "expected_messages"),
    [
        (
            [AssistantMessage([1, 2, 3])],
            [AssistantMessage([1, 2, 3])],
        ),
        (
            [SystemMessage("System message with {param}.")],
            [SystemMessage("System message with arg.")],
        ),
        (
            [UserMessage("User message with {param}.")],
            [UserMessage("User message with arg.")],
        ),
        (
            [AssistantMessage("Assistant message with {param}.")],
            [AssistantMessage("Assistant message with arg.")],
        ),
    ],
)
def test_chatpromptfunction_format(message_templates, expected_messages):
    @chatprompt(*message_templates)
    def func(param: str) -> str:
        ...

    assert func.format(param="arg") == expected_messages


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


@pytest.mark.openai
def test_chatprompt_readme_example():
    class Quote(BaseModel):
        quote: str
        character: str

    @chatprompt(
        SystemMessage("You are a movie buff."),
        UserMessage("What is your favorite quote from Harry Potter?"),
        AssistantMessage(
            Quote(
                quote="It does not do to dwell on dreams and forget to live.",
                character="Albus Dumbledore",
            )
        ),
        UserMessage("What is your favorite quote from {movie}?"),
    )
    def get_movie_quote(movie: str) -> Quote:
        ...

    movie_quote = get_movie_quote("Iron Man")
    assert isinstance(movie_quote, Quote)
