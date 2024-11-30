import os
from typing import Annotated, Any
from unittest.mock import ANY

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import AfterValidator, BaseModel

from magentic._pydantic import ConfigDict, with_config
from magentic._streamed_response import AsyncStreamedResponse, StreamedResponse
from magentic.chat_model.base import ToolSchemaParseError
from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    SystemMessage,
    Usage,
    UserMessage,
    _RawMessage,
)
from magentic.chat_model.openai_chat_model import (
    OpenaiChatModel,
    message_to_openai_message,
)
from magentic.function_call import FunctionCall, ParallelFunctionCall
from magentic.streaming import AsyncStreamedStr, StreamedStr


def plus(a: int, b: int) -> int:
    return a + b


@pytest.mark.parametrize(
    ("message", "expected_openai_message"),
    [
        (
            _RawMessage({"role": "user", "content": "Hello"}),
            {"role": "user", "content": "Hello"},
        ),
        (SystemMessage("Hello"), {"role": "system", "content": "Hello"}),
        (UserMessage("Hello"), {"role": "user", "content": "Hello"}),
        (AssistantMessage("Hello"), {"role": "assistant", "content": "Hello"}),
        (
            AssistantMessage(42),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "return_int", "arguments": '{"value":42}'},
                    }
                ],
            },
        ),
        (
            AssistantMessage(FunctionCall(plus, 1, 2)),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "plus", "arguments": '{"a":1,"b":2}'},
                    }
                ],
            },
        ),
        (
            AssistantMessage(
                ParallelFunctionCall(
                    [FunctionCall(plus, 1, 2), FunctionCall(plus, 3, 4)]
                )
            ),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "plus", "arguments": '{"a":1,"b":2}'},
                    },
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "plus", "arguments": '{"a":3,"b":4}'},
                    },
                ],
            },
        ),
        (
            AssistantMessage(
                StreamedResponse([StreamedStr(["Hello"]), FunctionCall(plus, 1, 2)])
            ),
            {
                "role": "assistant",
                "content": "Hello",
                "tool_calls": [
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "plus", "arguments": '{"a":1,"b":2}'},
                    },
                ],
            },
        ),
        (
            FunctionResultMessage(3, FunctionCall(plus, 1, 2)),
            {
                "role": "tool",
                "tool_call_id": ANY,
                "content": '{"value":3}',
            },
        ),
    ],
)
def test_message_to_openai_message(message, expected_openai_message):
    assert message_to_openai_message(message) == expected_openai_message


def test_message_to_openai_message_raises():
    class CustomMessage(Message[str]):
        def __init__(self, content: str, **data: Any):
            super().__init__(content, **data)

        def format(self, **kwargs):
            del kwargs
            return CustomMessage(self.content)

    with pytest.raises(NotImplementedError):
        message_to_openai_message(CustomMessage("Hello"))

    @message_to_openai_message.register
    def _(message: CustomMessage) -> ChatCompletionMessageParam:
        return {"role": "user", "content": message.content}

    assert message_to_openai_message(CustomMessage("Hello")) == {
        "role": "user",
        "content": "Hello",
    }


@pytest.mark.openai
def test_openai_chat_model_api_key(monkeypatch):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(openai.OpenAIError):
        chat_model = OpenaiChatModel("gpt-4o")

    chat_model = OpenaiChatModel("gpt-4o", api_key=openai_api_key)
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.openai
def test_openai_chat_model_complete_base_url():
    chat_model = OpenaiChatModel("gpt-4o", base_url="https://api.openai.com/v1")
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.openai
def test_openai_chat_model_complete_seed():
    chat_model = OpenaiChatModel("gpt-4o", seed=42)
    message1 = chat_model.complete(messages=[UserMessage("Say hello!")])
    message2 = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert message1.content == message2.content


@pytest.mark.openai
def test_openai_chat_model_complete_streamed_response():
    def get_weather(location: str) -> None:
        """Get the weather for a location."""

    chat_model = OpenaiChatModel("gpt-4o")
    message = chat_model.complete(
        messages=[UserMessage("Tell me your favorite city. Then get its weather.")],
        functions=[get_weather],
        output_types=[StreamedResponse],
    )
    assert isinstance(message.content, StreamedResponse)
    response_items = list(message.content)
    assert len(response_items) == 2
    streamed_str, function_call = response_items
    assert isinstance(streamed_str, StreamedStr)
    assert len(streamed_str.to_string()) > 1  # Check StreamedStr was cached
    assert isinstance(function_call, FunctionCall)
    assert function_call() is None  # Check FunctionCall is successfully called


@pytest.mark.openai
def test_openai_chat_model_complete_pydantic_model_openai_strict():
    class CapitalCity(BaseModel):
        model_config = ConfigDict(openai_strict=True)
        capital: str
        country: str

    chat_model = OpenaiChatModel("gpt-4o")
    message = chat_model.complete(
        messages=[UserMessage("What is the capital of Ireland?")],
        output_types=[CapitalCity],
    )
    assert isinstance(message.content, CapitalCity)


@pytest.mark.openai
def test_openai_chat_model_complete_function_call_openai_strict():
    @with_config(ConfigDict(openai_strict=True))
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = OpenaiChatModel("gpt-4o")
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.openai
def test_openai_chat_model_complete_usage():
    chat_model = OpenaiChatModel("gpt-4o")
    message = chat_model.complete(
        messages=[UserMessage("Say hello!")], output_types=[StreamedStr]
    )
    str(message.content)  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.openai
def test_openai_chat_model_complete_usage_structured_output():
    chat_model = OpenaiChatModel("gpt-4o")
    message = chat_model.complete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.openai
def test_openai_chat_model_complete_no_structured_output_error():
    chat_model = OpenaiChatModel("gpt-4o")
    # Should not raise StructuredOutputError because forced to make tool call
    message: Message[int | bool] = chat_model.complete(
        messages=[
            UserMessage("Tell me a short joke. Return a string, not a tool call.")
        ],
        output_types=[int, bool],
    )
    assert isinstance(message.content, int | bool)


@pytest.mark.openai
def test_openai_chat_model_complete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = OpenaiChatModel("gpt-4o")
    with pytest.raises(ToolSchemaParseError):
        chat_model.complete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.mark.openai
async def test_openai_chat_model_acomplete_async_streamed_response():
    def get_weather(location: str) -> None:
        """Get the weather for a location."""

    chat_model = OpenaiChatModel("gpt-4o")
    message = await chat_model.acomplete(
        messages=[UserMessage("Tell me your favorite city. Then get its weather.")],
        functions=[get_weather],
        output_types=[AsyncStreamedResponse],
    )
    assert isinstance(message.content, AsyncStreamedResponse)
    response_items = [x async for x in message.content]
    assert len(response_items) == 2
    streamed_str, function_call = response_items
    assert isinstance(streamed_str, AsyncStreamedStr)
    assert len(await streamed_str.to_string()) > 1  # Check AsyncStreamedStr was cached
    assert isinstance(function_call, FunctionCall)
    assert function_call() is None  # Check FunctionCall is successfully called


@pytest.mark.openai
async def test_openai_chat_model_acomplete_usage():
    chat_model = OpenaiChatModel("gpt-4o")
    message = await chat_model.acomplete(
        messages=[UserMessage("Say hello!")], output_types=[AsyncStreamedStr]
    )
    await message.content.to_string()  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.openai
async def test_openai_chat_model_acomplete_usage_structured_output():
    chat_model = OpenaiChatModel("gpt-4o")
    message = await chat_model.acomplete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.openai
async def test_openai_chat_model_acomplete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = OpenaiChatModel("gpt-4o")
    with pytest.raises(ToolSchemaParseError):
        await chat_model.acomplete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


def test_openai_chat_model_azure_omits_stream_options(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "test")
    monkeypatch.setenv("OPENAI_API_VERSION", "test")
    chat_model = OpenaiChatModel("gpt-4o", api_type="azure")
    assert chat_model._get_stream_options() == openai.NOT_GIVEN
