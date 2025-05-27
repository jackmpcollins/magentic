import os
from typing import Annotated

import openai
import pytest
from pydantic import AfterValidator, BaseModel

from magentic._pydantic import ConfigDict, with_config
from magentic._streamed_response import AsyncStreamedResponse, StreamedResponse
from magentic.chat_model.base import ToolSchemaParseError
from magentic.chat_model.message import ImageBytes, Message, Usage, UserMessage
from magentic.chat_model.openrouter_chat_model import OpenRouterChatModel
from magentic.function_call import FunctionCall
from magentic.streaming import AsyncStreamedStr, StreamedStr


@pytest.mark.openrouter
def test_openrouter_chat_model_api_key(monkeypatch):
    openrouter_api_key = os.environ["OPENROUTER_API_KEY"]
    monkeypatch.delenv("OPENROUTER_API_KEY")

    with pytest.raises(openai.OpenAIError):
        chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")

    chat_model = OpenRouterChatModel(
        "deepseek/deepseek-chat-v3-0324", api_key=openrouter_api_key
    )
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_base_url():
    chat_model = OpenRouterChatModel(
        "deepseek/deepseek-chat-v3-0324", base_url="https://openrouter.ai/api/v1"
    )
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_seed():
    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324", seed=42)
    message1 = chat_model.complete(messages=[UserMessage("Say hello!")])
    message2 = chat_model.complete(messages=[UserMessage("Say hello!")])
    # Check that both responses are greetings
    assert any(
        greeting in message1.content.lower() for greeting in ["hello", "hi", "hey"]
    )
    assert any(
        greeting in message2.content.lower() for greeting in ["hello", "hi", "hey"]
    )


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_streamed_response():
    def get_weather(location: str) -> None:
        """Get the weather for a location."""

    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
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


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_pydantic_model_openai_strict():
    class CapitalCity(BaseModel):
        model_config = ConfigDict(openai_strict=True)
        capital: str
        country: str

    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
    message = chat_model.complete(
        messages=[UserMessage("What is the capital of Ireland?")],
        output_types=[CapitalCity],
    )
    assert isinstance(message.content, CapitalCity)


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_function_call_openai_strict():
    @with_config(ConfigDict(openai_strict=True))
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_usage():
    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
    message = chat_model.complete(
        messages=[UserMessage("Say hello!")], output_types=[StreamedStr]
    )
    str(message.content)  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_usage_structured_output():
    chat_model = OpenRouterChatModel(
        "deepseek/deepseek-chat-v3-0324", require_parameters=True
    )
    message = chat_model.complete(
        messages=[UserMessage("Count to 5. Tool call please.")],
        output_types=[list[int]],
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_no_structured_output_error():
    chat_model = OpenRouterChatModel(
        "deepseek/deepseek-chat-v3-0324",
        require_parameters=True,  # This will force tool calls
    )
    message: Message[int | bool] = chat_model.complete(
        messages=[UserMessage("What is 2+2? Integer please.")],
        output_types=[int, bool],
    )
    assert isinstance(message.content, int | bool)


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
    with pytest.raises(ToolSchemaParseError):
        chat_model.complete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.mark.openrouter
def test_openrouter_chat_model_complete_image_bytes(image_bytes_jpg):
    chat_model = OpenRouterChatModel(
        "opengvlab/internvl3-14b:free", require_parameters=True
    )
    message = chat_model.complete(
        messages=[
            UserMessage(
                ("Describe this image in one word.", ImageBytes(image_bytes_jpg))
            )
        ]
    )
    assert isinstance(message.content, str)


@pytest.mark.openrouter
async def test_openrouter_chat_model_acomplete_async_streamed_response():
    def get_weather(location: str) -> None:
        """Get the weather for a location."""

    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
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


@pytest.mark.openrouter
async def test_openrouter_chat_model_acomplete_usage():
    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
    message = await chat_model.acomplete(
        messages=[UserMessage("Say hello!")], output_types=[AsyncStreamedStr]
    )
    await message.content.to_string()  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.openrouter
async def test_openrouter_chat_model_acomplete_usage_structured_output():
    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
    message = await chat_model.acomplete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.openrouter
async def test_openrouter_chat_model_acomplete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = OpenRouterChatModel("deepseek/deepseek-chat-v3-0324")
    with pytest.raises(ToolSchemaParseError):
        await chat_model.acomplete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.mark.openrouter
def test_openrouter_chat_model_extra_body():
    chat_model = OpenRouterChatModel(
        "deepseek/deepseek-chat-v3-0324",
        route="fallback",
        models=["anthropic/claude-3-opus"],
        require_parameters=True,
        reasoning={"effort": "high"},
    )
    assert chat_model._get_extra_body() == {
        "route": "fallback",
        "models": ["anthropic/claude-3-opus"],
        "provider": {"require_parameters": True},
        "reasoning": {"effort": "high"},
    }
