from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from magentic.chat_model.openai_chat_model import (
    AnyFunctionSchema,
    BaseModelFunctionSchema,
    FunctionCallFunctionSchema,
)
from magentic.function_call import FunctionCall


@pytest.mark.parametrize(
    ["type_", "json_schema"],
    [
        (
            str,
            {
                "name": "return_str",
                "parameters": {
                    "properties": {"value": {"title": "Value", "type": "string"}},
                    "required": ["value"],
                    "type": "object",
                },
            },
        ),
        (
            int,
            {
                "name": "return_int",
                "parameters": {
                    "properties": {"value": {"title": "Value", "type": "integer"}},
                    "required": ["value"],
                    "type": "object",
                },
            },
        ),
        (
            bool | str,
            {
                "name": "return_bool_or_str",
                "parameters": {
                    "properties": {
                        "value": {
                            "title": "Value",
                            "anyOf": [{"type": "boolean"}, {"type": "string"}],
                        }
                    },
                    "required": ["value"],
                    "type": "object",
                },
            },
        ),
        (
            list[str],
            {
                "name": "return_list_of_str",
                "parameters": {
                    "properties": {
                        "value": {
                            "title": "Value",
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["value"],
                    "type": "object",
                },
            },
        ),
        (
            list[str | int],
            {
                "name": "return_list_of_str_or_int",
                "parameters": {
                    "properties": {
                        "value": {
                            "title": "Value",
                            "type": "array",
                            "items": {
                                "anyOf": [{"type": "string"}, {"type": "integer"}]
                            },
                        }
                    },
                    "required": ["value"],
                    "type": "object",
                },
            },
        ),
        (
            str | None,
            {
                "name": "return_str_or_null",
                "parameters": {
                    "properties": {
                        "value": {
                            "title": "Value",
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        }
                    },
                    "required": ["value"],
                    "type": "object",
                },
            },
        ),
    ],
)
def test_any_function_schema(type_, json_schema):
    function_schema = AnyFunctionSchema(type_)
    assert function_schema.dict() == json_schema


@pytest.mark.parametrize(
    ["type_", "args_str", "output"],
    [
        (str, '{"value": "Dublin"}', "Dublin"),
        (int, '{"value": 42}', 42),
        (bool | str, '{"value": true}', True),
        (bool | str, '{"value": "Dublin"}', "Dublin"),
        (list[str], '{"value": ["Dublin", "London"]}', ["Dublin", "London"]),
        (list[str | int], '{"value": ["Dublin", 42]}', ["Dublin", 42]),
    ],
)
def test_any_function_schema_parse_args(type_, args_str, output):
    assert AnyFunctionSchema(type_).parse_args(args_str) == output


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


def plus(a: int, b: int) -> int:
    return a + b


def plus_no_type_hints(a, b):
    return a + b


def plus_default_value(a: int, b: int = 3) -> int:
    return a + b


def plus_with_annotated(
    a: Annotated[int, Field(description="First number")],
    b: Annotated[int, Field(description="Second number")],
) -> int:
    return a + b


@pytest.mark.parametrize(
    ["function", "json_schema"],
    [
        (
            plus,
            {
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
            },
        ),
        (
            plus_no_type_hints,
            {
                "name": "plus_no_type_hints",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # TODO: Remove "title" keys from schema
                        "a": {"title": "A"},
                        "b": {"title": "B"},
                    },
                    "required": ["a", "b"],
                },
            },
        ),
        (
            plus_default_value,
            {
                "name": "plus_default_value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # TODO: Remove "title" keys from schema
                        "a": {
                            "title": "A",
                            "type": "integer",
                        },
                        "b": {
                            "default": 3,
                            "title": "B",
                            "type": "integer",
                        },
                    },
                    "required": ["a"],
                },
            },
        ),
        (
            plus_with_annotated,
            {
                "name": "plus_with_annotated",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # TODO: Remove "title" keys from schema
                        "a": {
                            "title": "A",
                            "type": "integer",
                            "description": "First number",
                        },
                        "b": {
                            "title": "B",
                            "type": "integer",
                            "description": "Second number",
                        },
                    },
                    "required": ["a", "b"],
                },
            },
        ),
    ],
)
def test_function_call_function_schema(function, json_schema):
    function_schema = FunctionCallFunctionSchema(function)
    assert function_schema.dict() == json_schema
    output = function_schema.parse_args('{"a": 1, "b": 2}')
    assert isinstance(output, FunctionCall)
    assert output() == 3


def test_function_call_function_schema_with_default_value():
    function_schema = FunctionCallFunctionSchema(plus_default_value)
    output = function_schema.parse_args('{"a": 1}')
    assert isinstance(output, FunctionCall)
    assert output() == 4
