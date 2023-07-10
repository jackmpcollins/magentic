from typing import Annotated

from pydantic import BaseModel, Field

from agentic.chat_model.openai_chat_model import (
    AnyFunctionSchema,
    BaseModelFunctionSchema,
    FunctionCallFunctionSchema,
)
from agentic.function_call import FunctionCall


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
    assert function_schema.parse_args('{"value": "Dublin"}') == "Dublin"


def test_any_function_schema_parse_union():
    function_schema = AnyFunctionSchema(list[str | int | bool])

    assert function_schema.name == "return_list"
    assert function_schema.parameters == {
        "type": "object",
        "properties": {
            "value": {
                "title": "Value",
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "boolean"},
                    ]
                },
            }
        },
        "required": ["value"],
    }
    assert function_schema.parse_args('{"value": ["hello", 42, true]}') == [
        "hello",
        42,
        True,
    ]


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
    assert function_schema.parse_args('{"name": "Alice", "age": 99}') == User(
        name="Alice", age=99
    )


def test_function_call_function_schema():
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
    output = function_schema.parse_args('{"a": 1, "b": 2}')
    assert isinstance(output, FunctionCall)
    assert output() == 3


def test_function_call_function_schema_with_annotated():
    def plus(
        a: Annotated[int, Field(description="First number")],
        b: Annotated[int, Field(description="Second number")],
    ) -> int:
        return a + b

    function_schema = FunctionCallFunctionSchema(plus)

    assert function_schema.name == "plus"
    assert function_schema.dict() == {
        "name": "plus",
        "parameters": {
            "type": "object",
            "properties": {
                # TODO: Remove "title" keys from schema
                "a": {"title": "A", "type": "integer", "description": "First number"},
                "b": {"title": "B", "type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    }
    output = function_schema.parse_args('{"a": 1, "b": 2}')
    assert isinstance(output, FunctionCall)
    assert output() == 3
