import pytest
from pydantic import BaseModel

from magentic.chat_model.message import (
    AssistantMessage,
    SystemMessage,
    Usage,
    UserMessage,
)
from magentic.chat_model.mistral_chat_model import MistralChatModel
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
@pytest.mark.mistral
def test_mistral_chat_model_complete(prompt, output_types, expected_output_type):
    chat_model = MistralChatModel("mistral-large-latest")
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.mistral
def test_mistral_chat_model_complete_usage():
    chat_model = MistralChatModel("mistral-large-latest")
    message = chat_model.complete(
        messages=[UserMessage("Say hello!")], output_types=[StreamedStr]
    )
    str(message.content)  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.mistral
def test_mistral_chat_model_complete_usage_structured_output():
    chat_model = MistralChatModel("mistral-large-latest")
    message = chat_model.complete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.mistral
def test_mistral_chat_model_complete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = MistralChatModel("mistral-large-latest")
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
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
                " Make sure to use both tools at once."
            )
        ],
        functions=[plus, minus],
        output_types=[ParallelFunctionCall[int]],
    )
    assert isinstance(message.content, ParallelFunctionCall)
    assert len(list(message.content)) == 2


@pytest.mark.mistral
def test_mistral_chat_model_few_shot_prompt():
    class Quote(BaseModel):
        quote: str
        character: str

    chat_model = MistralChatModel("mistral-large-latest")
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
            # Mistral requires AssistantMessage after tool output
            # TODO: Automatically add this in ChatModel.complete like for tool calls
            AssistantMessage("."),
            UserMessage("What is your favorite quote from {movie}?"),
        ],
        output_types=[Quote],
    )
    assert isinstance(message.content, Quote)


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
async def test_mistral_chat_model_acomplete(prompt, output_types, expected_output_type):
    chat_model = MistralChatModel("mistral-large-latest")
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.mistral
async def test_mistral_chat_model_acomplete_usage():
    chat_model = MistralChatModel("mistral-large-latest")
    message = await chat_model.acomplete(
        messages=[UserMessage("Say hello!")], output_types=[AsyncStreamedStr]
    )
    await message.content.to_string()  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.mistral
async def test_mistral_chat_model_acomplete_usage_structured_output():
    chat_model = MistralChatModel("mistral-large-latest")
    message = await chat_model.acomplete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.mistral
async def test_mistral_chat_model_acomplete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = MistralChatModel("mistral-large-latest")
    message = await chat_model.acomplete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


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
                " Make sure to use both tools at once."
            )
        ],
        functions=[plus, minus],
        output_types=[AsyncParallelFunctionCall[int]],
    )
    assert isinstance(message.content, AsyncParallelFunctionCall)
    assert len([x async for x in message.content]) == 2
