from typing import Annotated, Any, Iterator

import litellm
import pytest
from litellm.integrations.custom_logger import CustomLogger
from pydantic import AfterValidator, BaseModel

from magentic.chat_model.base import ToolSchemaParseError
from magentic.chat_model.litellm_chat_model import LitellmChatModel
from magentic.chat_model.message import UserMessage
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        ("List three fruits", [list[str]], list),
    ],
)
@pytest.mark.litellm_openai
def test_litellm_chat_model_complete_openai(prompt, output_types, expected_output_type):
    chat_model = LitellmChatModel("gpt-3.5-turbo")
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.fixture()
def litellm_success_callback_calls() -> Iterator[list[dict[str, Any]]]:
    """A list of calls to the `litellm.success_callback`"""
    original_success_callback = litellm.success_callback.copy()
    callback_calls: list[dict[str, Any]] = []

    def _add_call_to_list(kwargs, completion_response, start_time, end_time):
        callback_calls.append({"kwargs": kwargs})

    litellm.success_callback = [_add_call_to_list]
    yield callback_calls
    litellm.success_callback = original_success_callback


@pytest.mark.litellm_openai
def test_litellm_chat_model_metadata(litellm_success_callback_calls):
    """Test that provided metadata is passed to the litellm success callback."""
    chat_model = LitellmChatModel("gpt-3.5-turbo", metadata={"foo": "bar"})
    assert chat_model.metadata == {"foo": "bar"}
    chat_model.complete(messages=[UserMessage("Say hello!")])
    # There are multiple callback calls due to streaming
    # Take the last one because the first is occasionally from another test
    callback_call = litellm_success_callback_calls[-1]
    assert callback_call["kwargs"]["litellm_params"]["metadata"] == {"foo": "bar"}


@pytest.mark.litellm_openai
def test_litellm_chat_model_custom_llm_provider(litellm_success_callback_calls):
    """Test that provided custom_llm_provider is passed to the litellm success callback."""
    chat_model = LitellmChatModel("gpt-3.5-turbo", custom_llm_provider="openai")
    assert chat_model.custom_llm_provider == "openai"
    chat_model.complete(messages=[UserMessage("Say hello!")])
    callback_call = litellm_success_callback_calls[-1]
    assert callback_call["kwargs"]["litellm_params"]["custom_llm_provider"] == "openai"


@pytest.mark.litellm_openai
def test_litellm_chat_model_complete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = LitellmChatModel("gpt-3.5-turbo")
    with pytest.raises(ToolSchemaParseError):
        chat_model.complete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.fixture()
def litellm_async_success_callback_calls() -> Iterator[list[dict[str, Any]]]:
    """A list of calls to the `async_log_success_event` callback"""
    original_success_callback = litellm.success_callback.copy()
    callback_calls: list[dict[str, Any]] = []

    class AddCallToList(CustomLogger):
        async def async_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            callback_calls.append({"kwargs": kwargs})

    litellm.callbacks = [AddCallToList()]  # type: ignore[list-item]
    yield callback_calls
    litellm.callbacks = original_success_callback  # type: ignore[assignment]


@pytest.mark.asyncio
@pytest.mark.litellm_openai
async def test_litellm_chat_model_metadata_async(litellm_async_success_callback_calls):
    """Test that provided metadata is passed to the litellm success callback."""
    chat_model = LitellmChatModel("gpt-3.5-turbo", metadata={"foo": "bar"})
    assert chat_model.metadata == {"foo": "bar"}
    await chat_model.acomplete(messages=[UserMessage("Say hello!")])
    # There are multiple callback calls due to streaming
    # Take the last one because the first is occasionally from another test
    callback_call = litellm_async_success_callback_calls[-1]
    assert callback_call["kwargs"]["litellm_params"]["metadata"] == {"foo": "bar"}


@pytest.mark.asyncio
@pytest.mark.litellm_openai
async def test_litellm_chat_model_custom_llm_provider_async(
    litellm_async_success_callback_calls,
):
    """Test that provided custom_llm_provider is passed to the litellm success callback."""
    chat_model = LitellmChatModel("gpt-3.5-turbo", custom_llm_provider="openai")
    assert chat_model.custom_llm_provider == "openai"
    await chat_model.acomplete(messages=[UserMessage("Say hello!")])
    callback_call = litellm_async_success_callback_calls[-1]
    assert callback_call["kwargs"]["litellm_params"]["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
@pytest.mark.litellm_openai
async def test_litellm_chat_model_acomplete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = LitellmChatModel("gpt-3.5-turbo")
    with pytest.raises(ToolSchemaParseError):
        await chat_model.acomplete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        pytest.param(
            "Use the tool to return a list of three fruits",
            [list[str]],
            list,
            marks=pytest.mark.skip(reason="Claude fails to format list of strings."),
        ),
    ],
)
@pytest.mark.litellm_anthropic
def test_litellm_chat_model_complete_anthropic(
    prompt, output_types, expected_output_type
):
    chat_model = LitellmChatModel("anthropic/claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.litellm_anthropic
def test_litellm_chat_model_complete_anthropic_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = LitellmChatModel("anthropic/claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.litellm_anthropic
def test_litellm_chat_model_complete_anthropic_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    chat_model = LitellmChatModel("anthropic/claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[
            UserMessage(
                "Use the plus tool to sum 1 and 2. Use the minus tool to subtract 1 from 2."
            )
        ],
        functions=[plus, minus],
        output_types=[ParallelFunctionCall[int]],
    )
    assert isinstance(message.content, ParallelFunctionCall)
    # Claude does not return multiple tool calls, so this returns a single function call


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True.", [bool], bool),
        ("Return [1, 2, 3, 4, 5]", [list[int]], list),
        ('Return ["apple", "banana"]', [list[str]], list),
    ],
)
@pytest.mark.litellm_ollama
def test_litellm_chat_model_complete_ollama(prompt, output_types, expected_output_type):
    chat_model = LitellmChatModel(
        "ollama_chat/llama3", api_base="http://localhost:11434"
    )
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


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
@pytest.mark.litellm_openai
async def test_litellm_chat_model_acomplete_openai(
    prompt, output_types, expected_output_type
):
    chat_model = LitellmChatModel("gpt-3.5-turbo")
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        pytest.param(
            "Use the tool to return a list of three fruits",
            [list[str]],
            list,
            marks=pytest.mark.skip(reason="Claude fails to format list of strings."),
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.litellm_anthropic
async def test_litellm_chat_model_acomplete_anthropic(
    prompt, output_types, expected_output_type
):
    chat_model = LitellmChatModel("anthropic/claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.asyncio
@pytest.mark.litellm_anthropic
async def test_litellm_chat_model_acomplete_anthropic_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = LitellmChatModel("anthropic/claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.asyncio
@pytest.mark.litellm_anthropic
async def test_litellm_chat_model_acomplete_anthropic_async_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    chat_model = LitellmChatModel("anthropic/claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[
            UserMessage(
                "Use the plus tool to sum 1 and 2. Use the minus tool to subtract 1 from 2."
            )
        ],
        functions=[plus, minus],
        output_types=[AsyncParallelFunctionCall[int]],
    )
    assert isinstance(message.content, AsyncParallelFunctionCall)
    # Claude does not return multiple tool calls, so this returns a single function call


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True.", [bool], bool),
        ("Return [1, 2, 3, 4, 5]", [list[int]], list),
        ('Return ["apple", "banana"]', [list[str]], list),
    ],
)
@pytest.mark.asyncio
@pytest.mark.litellm_ollama
async def test_litellm_chat_model_acomplete_ollama(
    prompt, output_types, expected_output_type
):
    chat_model = LitellmChatModel(
        "ollama_chat/llama3", api_base="http://localhost:11434"
    )
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)
