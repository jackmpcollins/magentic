import pytest

from magentic.chat_model.message import UserMessage
from magentic.chat_model.mistral_chat_model import MistralChatModel
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
@pytest.mark.mistral
def test_mistral_chat_model_complete(prompt, output_types, expected_output_type):
    chat_model = MistralChatModel("mistral-large-latest")
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.mistral
def test_mistral_chat_model_complete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = MistralChatModel("mistral-large-latest")
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],  # type: ignore[misc]
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.mistral
def test_mistral_chat_model_complete_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    chat_model = MistralChatModel("mistral-large-latest")
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
@pytest.mark.mistral
async def test_mistral_chat_model_acomplete(prompt, output_types, expected_output_type):
    chat_model = MistralChatModel("mistral-large-latest")
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.asyncio
@pytest.mark.mistral
async def test_mistral_chat_model_acomplete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = MistralChatModel("mistral-large-latest")
    message = await chat_model.acomplete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],  # type: ignore[misc]
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.asyncio
@pytest.mark.mistral
async def test_mistral_chat_model_acomplete_async_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    chat_model = MistralChatModel("mistral-large-latest")
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
