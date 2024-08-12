from typing import Annotated

import pytest
from pydantic import AfterValidator, BaseModel

from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from magentic.chat_model.base import ToolSchemaParseError
from magentic.chat_model.message import Message, Usage, UserMessage
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.streaming import AsyncStreamedStr, StreamedStr


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        ("List three fruits", [list[str]], list),
    ],
)
@pytest.mark.anthropic
def test_anthropic_chat_model_complete(prompt, output_types, expected_output_type):
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_usage():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage("Say hello!")], output_types=[StreamedStr]
    )
    str(message.content)  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_usage_structured_output():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_no_structured_output_error():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    # Should not raise StructuredOutputError because forced to make tool call
    message: Message[int | bool] = chat_model.complete(
        messages=[
            UserMessage("Tell me a short joke. Return a string, not a tool call.")
        ],
        output_types=[int, bool],
    )
    assert isinstance(message.content, int | bool)


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    with pytest.raises(ToolSchemaParseError):
        chat_model.complete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    chat_model = AnthropicChatModel("claude-3-opus-20240229")
    message = chat_model.complete(
        messages=[
            UserMessage(
                "Use the plus tool to sum 1 and 2."
                " Use the minus tool to subtract 1 from 2."
                " Use both tools at the same time."
            )
        ],
        functions=[plus, minus],
        output_types=[ParallelFunctionCall[int]],
    )
    assert isinstance(message.content, ParallelFunctionCall)
    assert len(list(message.content)) == 2


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        ("List three fruits", [list[str]], list),
    ],
)
@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete(
    prompt, output_types, expected_output_type
):
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_usage():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage("Say hello!")], output_types=[AsyncStreamedStr]
    )
    await message.content.to_string()  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_usage_structured_output():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    with pytest.raises(ToolSchemaParseError):
        await chat_model.acomplete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_async_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    chat_model = AnthropicChatModel("claude-3-opus-20240229")
    message = await chat_model.acomplete(
        messages=[
            UserMessage(
                "Use the plus tool to sum 1 and 2."
                " Use the minus tool to subtract 1 from 2."
                " Use both tools at the same time."
            )
        ],
        functions=[plus, minus],
        output_types=[AsyncParallelFunctionCall[int]],
    )
    assert isinstance(message.content, AsyncParallelFunctionCall)
    assert len([x async for x in message.content]) == 2
