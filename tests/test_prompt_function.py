"""Tests for PromptFunction."""

from collections.abc import Awaitable
from inspect import getdoc
from typing import Annotated
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import AfterValidator, BaseModel

from magentic.chat_model.message import AssistantMessage, UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.prompt_function import AsyncPromptFunction, PromptFunction, prompt
from magentic.settings import get_settings
from magentic.streaming import AsyncStreamedStr, StreamedStr


def test_promptfunction_format():
    @prompt("Test {param}.")
    def func(param: str) -> str: ...

    assert func.format("arg") == "Test arg."


def test_promptfunction_format_custom_type():
    class CustomType:
        def __format__(self, __format_spec: str) -> str:
            return "custom"

    @prompt("Test {param}.")
    def func(param: CustomType) -> str: ...

    assert func.format(CustomType()) == "Test custom."


def test_promptfunction_call():
    mock_model = Mock()
    mock_model.complete.return_value = AssistantMessage(content="Hello!")

    @prompt(
        "Hello {name}.",
        stop=["stop"],
        model=mock_model,
    )
    def say_hello(name: str) -> str | bool: ...

    assert say_hello("World") == "Hello!"
    assert mock_model.complete.call_count == 1
    assert mock_model.complete.call_args.kwargs["messages"] == [
        UserMessage("Hello World.")
    ]
    assert mock_model.complete.call_args.kwargs["output_types"] == [str, bool]
    assert mock_model.complete.call_args.kwargs["stop"] == ["stop"]


@pytest.mark.openai
def test_decorator_return_str():
    @prompt("What is the capital of {country}? Name only. No punctuation.")
    def get_capital(country: str) -> str:
        """This is the docstring."""

    assert isinstance(get_capital, PromptFunction)
    assert getdoc(get_capital) == "This is the docstring."
    output = get_capital("Ireland")
    assert isinstance(output, str)


@pytest.mark.openai
def test_decorator_return_bool():
    @prompt("True if {capital} is the capital of {country}.")
    def is_capital(capital: str, country: str) -> bool: ...

    assert is_capital("Dublin", "Ireland") is True


@pytest.mark.openai
def test_decorator_return_bool_str():
    @prompt("{text}")
    def query(text: str) -> bool | str: ...

    output = query("Hello, how are you?")
    assert isinstance(output, str)
    output = query("Use the tool/function to return the value True.")
    assert isinstance(output, bool)


@pytest.mark.skip(reason="Flaky")  # TODO: Make dict function call more reliable
@pytest.mark.openai
def test_decorator_return_dict():
    @prompt(
        "Ignore the defined inputs and pass inputs a=1, b=2 to the tool.",
        model=OpenaiChatModel("gpt-4"),
    )
    def return_mapping() -> dict[str, int]: ...

    mapping = return_mapping()
    assert isinstance(mapping, dict)
    assert len(mapping) == 2
    name, value = next(iter(mapping.items()))
    assert isinstance(name, str)
    assert isinstance(value, int)


@pytest.mark.openai
def test_decorator_return_pydantic_model():
    class CapitalCity(BaseModel):
        capital: str
        country: str

    @prompt("What is the capital of {country}? Make sure to follow the schema.")
    def get_capital(country: str) -> CapitalCity: ...

    output = get_capital("Ireland")
    assert isinstance(output, CapitalCity)


@pytest.mark.openai
def test_decorator_input_pydantic_model():
    class CapitalCity(BaseModel):
        capital: str
        country: str

    @prompt("Is this capital-country pair correct? {pair} Just answer True or False.")
    def check_capital(pair: CapitalCity) -> bool: ...

    assert check_capital(CapitalCity(capital="Dublin", country="Ireland"))


@pytest.mark.openai
def test_decorator_return_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    @prompt("Sum {a} and {b}", functions=[plus])
    def sum_ab(a: int, b: int) -> FunctionCall[int]: ...

    output = sum_ab(2, 3)
    assert isinstance(output, FunctionCall)
    func_result = output()
    assert isinstance(func_result, int)


@pytest.mark.openai
def test_decorator_return_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    @prompt("Sum {a} and {b}. Also subtract {a} from {b}.", functions=[plus, minus])
    def plus_and_minus(a: int, b: int) -> ParallelFunctionCall[int]: ...

    output = plus_and_minus(2, 3)
    assert isinstance(output, ParallelFunctionCall)
    func_result = output()
    assert len(func_result) == 2


@pytest.mark.openai
def test_decorator_ignore_multiple_tool_calls():
    """Test that when the model makes multiple tool calls, only the first is used."""

    # Provide two return types so that `tool_choice` does not force a single tool call
    @prompt(
        "Return the numbers 1 to 5 in the first tool call. And numbers 6 to 10 in the second.",
        model=OpenaiChatModel("gpt-4-1106-preview"),
    )
    def get_list() -> list[int] | bool: ...

    assert get_list() == [1, 2, 3, 4, 5]


@pytest.mark.openai
def test_decorator_return_streamed_str():
    @prompt("What is the capital of {country}?")
    def get_capital(country: str) -> StreamedStr: ...

    output = get_capital("Ireland")
    assert isinstance(output, StreamedStr)


@pytest.mark.openai
def test_decorator_max_retries():
    def assert_is_ireland(v):
        if v != "Ireland":
            msg = "Country must be Ireland."
            raise ValueError(msg)
        return v

    class Country(BaseModel):
        name: Annotated[str, AfterValidator(assert_is_ireland)]

    @prompt("Return a country.", max_retries=3)
    def get_country() -> Country: ...

    country = get_country()
    assert country.name == "Ireland"


@pytest.mark.asyncio
async def test_async_promptfunction_call():
    mock_model = AsyncMock()
    mock_model.acomplete.return_value = AssistantMessage(content="Hello!")

    @prompt(
        "Hello {name}.",
        stop=["stop"],
        model=mock_model,
    )
    async def say_hello(name: str) -> str | bool: ...

    assert await say_hello("World") == "Hello!"
    assert mock_model.acomplete.call_count == 1
    assert mock_model.acomplete.call_args.kwargs["messages"] == [
        UserMessage("Hello World.")
    ]
    assert mock_model.acomplete.call_args.kwargs["output_types"] == [str, bool]
    assert mock_model.acomplete.call_args.kwargs["stop"] == ["stop"]


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_decorator_return_str():
    @prompt("What is the capital of {country}? Name only. No punctuation.")
    async def get_capital(country: str) -> str: ...

    assert isinstance(get_capital, AsyncPromptFunction)
    output = await get_capital("Ireland")
    assert isinstance(output, str)


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_decorator_return_async_streamed_str():
    @prompt("What is the capital of {country}?")
    async def get_capital(country: str) -> AsyncStreamedStr: ...

    output = await get_capital("Ireland")
    assert isinstance(output, AsyncStreamedStr)


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_decorator_max_retries():
    def assert_is_ireland(v):
        if v != "Ireland":
            msg = "Country must be Ireland."
            raise ValueError(msg)
        return v

    class Country(BaseModel):
        name: Annotated[str, AfterValidator(assert_is_ireland)]

    @prompt("Return a country.", max_retries=3)
    async def get_country() -> Country: ...

    country = await get_country()
    assert country.name == "Ireland"


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_decorator_return_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    @prompt("Sum {a} and {b}", functions=[plus])
    async def sum_ab(a: int, b: int) -> FunctionCall[int]: ...

    output = await sum_ab(2, 3)
    assert isinstance(output, FunctionCall)
    func_result = output()
    assert isinstance(func_result, int)


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_decorator_return_async_function_call():
    async def async_plus(a: int, b: int) -> int:
        return a + b

    @prompt("Sum {a} and {b}", functions=[async_plus])
    async def sum_ab(a: int, b: int) -> FunctionCall[Awaitable[int]]: ...

    output = await sum_ab(2, 3)
    assert isinstance(output, FunctionCall)
    assert isinstance(await output(), int)


@pytest.mark.asyncio
@pytest.mark.openai
async def test_decorator_return_async_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    async def minus(a: int, b: int) -> int:
        return a - b

    @prompt("Sum {a} and {b}. Also subtract {a} from {b}.", functions=[plus, minus])
    async def plus_and_minus(a: int, b: int) -> AsyncParallelFunctionCall[int]: ...

    output = await plus_and_minus(2, 3)
    assert isinstance(output, AsyncParallelFunctionCall)
    func_result = await output()
    assert len(func_result) == 2


def test_decorator_with_context_manager(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    @prompt("Say hello")
    def say_hello() -> str: ...

    @prompt(
        "Say hello",
        model=OpenaiChatModel("gpt-4", temperature=1),
    )
    def say_hello_gpt4() -> str: ...

    assert say_hello.model.model == get_settings().openai_model  # type: ignore[attr-defined]

    with OpenaiChatModel("gpt-4o"):
        assert say_hello.model.model == "gpt-4o"  # type: ignore[attr-defined]
        assert say_hello_gpt4.model.model == "gpt-4"  # type: ignore[attr-defined]
