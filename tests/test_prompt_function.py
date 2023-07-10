"""Tests for PromptFunction."""

from inspect import getdoc

import pytest

from agentic.function_call import FunctionCall
from agentic.prompt_function import PromptFunction, prompt


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
def test_decorator_function_call():
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
