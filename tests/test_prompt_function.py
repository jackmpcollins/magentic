"""Tests for PromptFunction."""

from inspect import getdoc

import pytest
from pydantic import BaseModel

from magentic.function_call import FunctionCall
from magentic.prompt_function import PromptFunction, prompt


@pytest.mark.openai
def test_decorator_return_str():
    @prompt()
    def get_capital(country: str) -> str:
        """What is the capital of {country}? Name only. No punctuation."""
        ...

    assert get_capital("Ireland") == "Dublin"


@pytest.mark.openai
def test_decorator_template_with_docstring():
    @prompt(template="What is the capital of {country}? Name only. No punctuation.")
    def get_capital(country: str) -> str:
        """This is the docstring."""
        ...

    assert get_capital("Ireland") == "Dublin"
    assert isinstance(get_capital, PromptFunction)
    assert getdoc(get_capital) == "This is the docstring."


@pytest.mark.openai
def test_decorator_return_bool():
    @prompt()
    def is_capital(capital: str, country: str) -> bool:
        """True if {capital} is the capital of {country}."""
        ...

    assert is_capital("Dublin", "Ireland") is True


@pytest.mark.openai
def test_decorator_return_bool_str():
    @prompt()
    def answer_question(question: str) -> bool | str:
        """Answer the following question: {question}."""
        ...

    assert answer_question("What is the capital of Ireland? Name only") == "Dublin"
    assert answer_question("Dublin is the capital of Ireland: True or False?") is True


@pytest.mark.openai
def test_decorator_return_dict():
    @prompt("Return a mapping of the 5 tallest mountains to their height in metres")
    def get_tallest_mountains() -> dict[str, int]:
        ...

    height_by_mountain = get_tallest_mountains()
    assert isinstance(height_by_mountain, dict)
    assert len(height_by_mountain) == 5
    name, height = next(iter(height_by_mountain.items()))
    assert isinstance(name, str)
    assert isinstance(height, int)


@pytest.mark.openai
def test_decorator_return_pydantic_model():
    class CapitalCity(BaseModel):
        capital: str
        country: str

    @prompt(template="What is the capital of {country}?")
    def get_capital(country: str) -> CapitalCity:
        ...

    assert get_capital("Ireland") == CapitalCity(capital="Dublin", country="Ireland")


@pytest.mark.openai
def test_decorator_input_pydantic_model():
    class CapitalCity(BaseModel):
        capital: str
        country: str

    @prompt(template="Is this capital-country pair correct? {pair}")
    def check_capital(pair: CapitalCity) -> bool:
        ...

    assert check_capital(CapitalCity(capital="Dublin", country="Ireland"))


@pytest.mark.openai
def test_decorator_return_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    @prompt(functions=[plus])
    def sum_populations(country_one: str, country_two: str) -> FunctionCall[int]:
        """Sum the populations of {country_one} and {country_two}."""
        ...

    output = sum_populations("Ireland", "UK")
    assert isinstance(output, FunctionCall)
    func_result = output()
    assert isinstance(func_result, int)
    assert isinstance(func_result, int)
    assert isinstance(func_result, int)
