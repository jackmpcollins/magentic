"""Tests for PromptFunction."""

from inspect import getdoc
from typing import Awaitable

import pytest
from pydantic import BaseModel

from magentic.chat_model.openai_chat_model import StructuredOutputError
from magentic.function_call import FunctionCall
from magentic.prompt_function import AsyncPromptFunction, PromptFunction, prompt
from magentic.streaming import AsyncStreamedStr, StreamedStr


def test_promptfunction_format():
    @prompt("Test {param}.")
    def func(param: str) -> str:
        ...

    assert func.format("arg") == "Test arg."


@pytest.mark.openai
def test_decorator_return_str():
    @prompt("What is the capital of {country}? Name only. No punctuation.")
    def get_capital(country: str) -> str:
        """This is the docstring."""
        ...

    assert isinstance(get_capital, PromptFunction)
    assert getdoc(get_capital) == "This is the docstring."
    output = get_capital("Ireland")
    assert isinstance(output, str)


@pytest.mark.openai
def test_decorator_return_bool():
    @prompt("True if {capital} is the capital of {country}.")
    def is_capital(capital: str, country: str) -> bool:
        ...

    assert is_capital("Dublin", "Ireland") is True


@pytest.mark.openai
def test_decorator_return_bool_str():
    @prompt("{text}")
    def query(text: str) -> bool | str:
        ...

    output = query("Reply to me with just the word hello.")
    assert isinstance(output, str)
    output = query("Use the function to return the value True.")
    assert isinstance(output, bool)


@pytest.mark.openai
def test_decorator_return_dict():
    @prompt('Return the mapping {{"one": 1, "two": 2}}')
    def return_mapping() -> dict[str, int]:
        ...

    mapping = return_mapping()
    assert isinstance(mapping, dict)
    name, value = next(iter(mapping.items()))
    assert isinstance(name, str)
    assert isinstance(value, int)


@pytest.mark.openai
def test_decorator_return_pydantic_model():
    class CapitalCity(BaseModel):
        capital: str
        country: str

    @prompt("What is the capital of {country}?")
    def get_capital(country: str) -> CapitalCity:
        ...

    assert get_capital("Ireland") == CapitalCity(capital="Dublin", country="Ireland")


@pytest.mark.openai
def test_decorator_input_pydantic_model():
    class CapitalCity(BaseModel):
        capital: str
        country: str

    @prompt("Is this capital-country pair correct? {pair} Just answer True or False.")
    def check_capital(pair: CapitalCity) -> bool:
        ...

    assert check_capital(CapitalCity(capital="Dublin", country="Ireland"))


@pytest.mark.openai
def test_decorator_return_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    @prompt("Sum {a} and {b}", functions=[plus])
    def sum_ab(a: int, b: int) -> FunctionCall[int]:
        ...

    output = sum_ab(2, 3)
    assert isinstance(output, FunctionCall)
    func_result = output()
    assert isinstance(func_result, int)


@pytest.mark.openai
def test_decorator_return_streamed_str():
    @prompt("What is the capital of {country}?")
    def get_capital(country: str) -> StreamedStr:
        ...

    output = get_capital("Ireland")
    assert isinstance(output, StreamedStr)


@pytest.mark.openai
def test_decorator_raise_structured_output_error():
    @prompt("How many days between {start_date} and {end_date}? Do out the math.")
    def days_between(start_date: str, end_date: str) -> int:
        ...

    with pytest.raises(StructuredOutputError):
        # The model will return a math expression, not an integer
        days_between("Jan 4th 2019", "Jul 3rd 2019")


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_decorator_return_str():
    @prompt("What is the capital of {country}? Name only. No punctuation.")
    async def get_capital(country: str) -> str:
        ...

    assert isinstance(get_capital, AsyncPromptFunction)
    output = await get_capital("Ireland")
    assert isinstance(output, str)


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_decorator_return_async_streamed_str():
    @prompt("What is the capital of {country}?")
    async def get_capital(country: str) -> AsyncStreamedStr:
        ...

    output = await get_capital("Ireland")
    assert isinstance(output, AsyncStreamedStr)


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_decorator_return_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    @prompt("Sum {a} and {b}", functions=[plus])
    async def sum_ab(a: int, b: int) -> FunctionCall[int]:
        ...

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
    async def sum_ab(a: int, b: int) -> FunctionCall[Awaitable[int]]:
        ...

    output = await sum_ab(2, 3)
    assert isinstance(output, FunctionCall)
    assert isinstance(await output(), int)
