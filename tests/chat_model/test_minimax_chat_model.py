import os

import openai
import pytest
from pydantic import BaseModel

from magentic.chat_model.message import (
    AssistantMessage,
    SystemMessage,
    Usage,
    UserMessage,
)
from magentic.chat_model.minimax_chat_model import (
    MiniMaxChatModel,
    MiniMaxStreamParser,
    _MiniMaxOpenaiChatModel,
    _strip_think_tags,
)
from magentic.function_call import FunctionCall
from magentic.streaming import AsyncStreamedStr, StreamedStr


# --- Unit tests (no API calls) ---


def test_strip_think_tags_empty():
    assert _strip_think_tags("Hello world") == "Hello world"


def test_strip_think_tags_simple():
    assert _strip_think_tags("<think>reasoning here</think>Hello") == "Hello"


def test_strip_think_tags_multiline():
    text = "<think>\nLet me think...\nStep 1\nStep 2\n</think>\nThe answer is 42."
    assert _strip_think_tags(text) == "The answer is 42."


def test_strip_think_tags_multiple():
    text = "<think>first</think>Hello <think>second</think>World"
    assert _strip_think_tags(text) == "Hello World"


def test_strip_think_tags_trailing_whitespace():
    assert _strip_think_tags("<think>x</think> Hello") == "Hello"


def test_minimax_stream_parser_no_think_tags():
    """MiniMaxStreamParser passes through normal content."""
    from unittest.mock import MagicMock

    parser = MiniMaxStreamParser()
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "Hello world"
    result = parser.get_content(chunk)
    assert result == "Hello world"


def test_minimax_stream_parser_no_content():
    """MiniMaxStreamParser returns None when no content."""
    from unittest.mock import MagicMock

    parser = MiniMaxStreamParser()
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = None
    result = parser.get_content(chunk)
    assert result is None


def test_minimax_stream_parser_empty_choices():
    """MiniMaxStreamParser returns None when choices is empty."""
    from unittest.mock import MagicMock

    parser = MiniMaxStreamParser()
    chunk = MagicMock()
    chunk.choices = []
    result = parser.get_content(chunk)
    assert result is None


def test_minimax_stream_parser_tools_expected_skips_content():
    """MiniMaxStreamParser skips content when tools_expected=True."""
    from unittest.mock import MagicMock

    parser = MiniMaxStreamParser(tools_expected=True)
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "I'll use the tool"
    assert parser.is_content(chunk) is False


def test_minimax_stream_parser_think_tags_skipped():
    """MiniMaxStreamParser skips think-tag-only content."""
    from unittest.mock import MagicMock

    parser = MiniMaxStreamParser()
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "<think>reasoning</think>"
    assert parser.is_content(chunk) is False


def test_minimax_chat_model_api_key_required(monkeypatch):
    """MiniMaxChatModel raises error when no API key is provided."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(openai.OpenAIError, match="MINIMAX_API_KEY"):
        MiniMaxChatModel("MiniMax-M2.7")


def test_minimax_chat_model_api_key_from_env(monkeypatch):
    """MiniMaxChatModel uses MINIMAX_API_KEY env var."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-123")
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    assert chat_model.api_key == "test-key-123"


def test_minimax_chat_model_api_key_explicit(monkeypatch):
    """MiniMaxChatModel uses explicit api_key over env var."""
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
    chat_model = MiniMaxChatModel("MiniMax-M2.7", api_key="explicit-key")
    assert chat_model.api_key == "explicit-key"


def test_minimax_chat_model_default_base_url(monkeypatch):
    """MiniMaxChatModel defaults to MiniMax API base URL."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    assert chat_model.base_url == "https://api.minimax.io/v1"


def test_minimax_chat_model_custom_base_url(monkeypatch):
    """MiniMaxChatModel accepts custom base URL."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel(
        "MiniMax-M2.7", base_url="http://localhost:8080"
    )
    assert chat_model.base_url == "http://localhost:8080"


def test_minimax_chat_model_properties(monkeypatch):
    """MiniMaxChatModel exposes all configuration properties."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel(
        "MiniMax-M2.7",
        max_tokens=512,
        seed=42,
        temperature=0.7,
    )
    assert chat_model.model == "MiniMax-M2.7"
    assert chat_model.max_tokens == 512
    assert chat_model.seed == 42
    assert chat_model.temperature == 0.7


def test_minimax_temperature_clamping_high(monkeypatch):
    """Temperature > 1.0 is clamped to 1.0."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel("MiniMax-M2.7", temperature=2.0)
    assert chat_model.temperature == 1.0


def test_minimax_temperature_clamping_low(monkeypatch):
    """Temperature <= 0 is clamped to 0.01."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel("MiniMax-M2.7", temperature=0.0)
    assert chat_model.temperature == 0.01


def test_minimax_temperature_clamping_negative(monkeypatch):
    """Negative temperature is clamped to 0.01."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel("MiniMax-M2.7", temperature=-1.0)
    assert chat_model.temperature == 0.01


def test_minimax_temperature_none(monkeypatch):
    """Temperature=None is passed through as-is."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel("MiniMax-M2.7", temperature=None)
    assert chat_model.temperature is None


def test_minimax_temperature_in_range(monkeypatch):
    """Temperature within (0, 1] is not modified."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel("MiniMax-M2.7", temperature=0.5)
    assert chat_model.temperature == 0.5


def test_minimax_context_manager(monkeypatch):
    """MiniMaxChatModel works as a context manager."""
    from magentic.backend import get_chat_model

    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    with chat_model:
        assert get_chat_model() is chat_model


# --- Integration tests (require MINIMAX_API_KEY) ---


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        ("List three fruits", [list[str]], list),
    ],
)
@pytest.mark.minimax
def test_minimax_chat_model_complete(prompt, output_types, expected_output_type):
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.minimax
def test_minimax_chat_model_complete_usage():
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = chat_model.complete(
        messages=[UserMessage("Say hello!")], output_types=[StreamedStr]
    )
    str(message.content)  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.minimax
def test_minimax_chat_model_complete_usage_structured_output():
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = chat_model.complete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.minimax
def test_minimax_chat_model_complete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.minimax
def test_minimax_chat_model_few_shot_prompt():
    class Quote(BaseModel):
        quote: str
        character: str

    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = chat_model.complete(
        messages=[
            SystemMessage("You are a movie buff."),
            UserMessage("What is your favorite quote from Harry Potter?"),
            AssistantMessage(
                Quote(
                    quote="It does not do to dwell on dreams and forget to live.",
                    character="Albus Dumbledore",
                )
            ),
            AssistantMessage("."),
            UserMessage("What is your favorite quote from {movie}?"),
        ],
        output_types=[Quote],
    )
    assert isinstance(message.content, Quote)


@pytest.mark.minimax
def test_minimax_chat_model_complete_pydantic_model():
    class CapitalCity(BaseModel):
        capital: str
        country: str

    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = chat_model.complete(
        messages=[UserMessage("What is the capital of Ireland?")],
        output_types=[CapitalCity],
    )
    assert isinstance(message.content, CapitalCity)


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        ("List three fruits", [list[str]], list),
    ],
)
@pytest.mark.minimax
async def test_minimax_chat_model_acomplete(prompt, output_types, expected_output_type):
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.minimax
async def test_minimax_chat_model_acomplete_usage():
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = await chat_model.acomplete(
        messages=[UserMessage("Say hello!")], output_types=[AsyncStreamedStr]
    )
    await message.content.to_string()  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.minimax
async def test_minimax_chat_model_acomplete_usage_structured_output():
    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = await chat_model.acomplete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.minimax
async def test_minimax_chat_model_acomplete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = MiniMaxChatModel("MiniMax-M2.7")
    message = await chat_model.acomplete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)
