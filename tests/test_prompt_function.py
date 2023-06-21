"""Tests for PromptFunction."""

from pydantic import BaseModel
from agentic.function_call import FunctionCall
from agentic.prompt_function import (
    AnyFunctionSchema,
    BaseModelFunctionSchema,
    FunctionCallFunctionSchema,
    prompt,
)


def test_any_function_schema():
    function_schema = AnyFunctionSchema(str)

    assert function_schema.name == "return_str"
    assert function_schema.dict() == {
        "name": "return_str",
        "parameters": {
            "type": "object",
            "properties": {
                # TODO: Remove "title" keys from schema
                "value": {"title": "Value", "type": "string"},
            },
            "required": ["value"],
        },
    }
    assert function_schema.parse('{"value": "Dublin"}') == "Dublin"


def test_base_model_function_schema():
    class User(BaseModel):
        name: str
        age: int

    function_schema = BaseModelFunctionSchema(User)

    assert function_schema.name == "return_user"
    assert function_schema.dict() == {
        "name": "return_user",
        "parameters": {
            "type": "object",
            "properties": {
                # TODO: Remove "title" keys from schema
                "name": {"title": "Name", "type": "string"},
                "age": {"title": "Age", "type": "integer"},
            },
            "required": ["name", "age"],
        },
    }
    assert function_schema.parse('{"name": "Alice", "age": 99}') == User(
        name="Alice", age=99
    )


def test_base_model_function_schema():
    def plus(a: int, b: int) -> int:
        return a + b

    function_schema = FunctionCallFunctionSchema(plus)

    assert function_schema.name == "plus"
    assert function_schema.dict() == {
        "name": "plus",
        "parameters": {
            "type": "object",
            "properties": {
                # TODO: Remove "title" keys from schema
                "a": {"title": "A", "type": "integer"},
                "b": {"title": "B", "type": "integer"},
            },
            "required": ["a", "b"],
        },
    }
    output = function_schema.parse('{"a": 1, "b": 2}')
    assert isinstance(output, FunctionCall)
    assert output() == 3


def test_decorator_return_str():
    @prompt()
    def get_capital(country: str) -> str:
        """What is the capital of {country}? Name only. No punctuation."""

    assert get_capital("Ireland") == "Dublin"


def test_decorator_return_bool():
    @prompt()
    def is_capital(capital: str, country: str) -> bool:
        """True if {capital} is the capital of {country}."""

    assert is_capital("Dublin", "Ireland") is True


def test_decorator_return_bool_str():
    @prompt()
    def answer_question(question: str) -> bool | str:
        """Answer the following question: {question}."""

    assert answer_question("What is the capital of Ireland?") == "Dublin"
    assert answer_question("Dublin is the capital of Ireland: True or False?") is True


def test_decorator_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    @prompt(functions=[plus])
    def sum_populations(country_one: str, country_two: str) -> FunctionCall[int]:
        """Sum the populations of {country_one} and {country_two}."""

    output = sum_populations("Ireland", "UK")
    assert isinstance(output, FunctionCall)
    func_result = output()
    assert isinstance(func_result, int)
