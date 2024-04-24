import collections.abc
import json
import typing
from collections import OrderedDict
from typing import Annotated, Any, Generic, TypeVar, get_origin

import pytest
from pydantic import BaseModel, Field

from magentic.chat_model.function_schema import (
    AnyFunctionSchema,
    AsyncIterableFunctionSchema,
    BaseModelFunctionSchema,
    DictFunctionSchema,
    FunctionCallFunctionSchema,
    IterableFunctionSchema,
)
from magentic.function_call import FunctionCall
from magentic.streaming import async_iter
from magentic.typing import is_origin_subclass


@pytest.mark.parametrize(
    ("type_", "json_schema"),
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
        (
            dict[str, int],
            {
                "name": "return_dict_of_str_to_int",
                "parameters": {
                    "properties": {
                        "value": {
                            "title": "Value",
                            "type": "object",
                            "additionalProperties": {"type": "integer"},
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


any_function_schema_args_test_cases = [
    (str, '{"value": "Dublin"}', "Dublin"),
    (int, '{"value": 42}', 42),
    (bool | str, '{"value": true}', True),
    (bool | str, '{"value": "Dublin"}', "Dublin"),
    (list[str], '{"value": ["Dublin", "London"]}', ["Dublin", "London"]),
    (list[str | int], '{"value": ["Dublin", 42]}', ["Dublin", 42]),
    (str | None, '{"value": "Dublin"}', "Dublin"),
    (str | None, '{"value": null}', None),
]


@pytest.mark.parametrize(
    ("type_", "args_str", "expected_args"), any_function_schema_args_test_cases
)
def test_any_function_schema_parse_args(type_, args_str, expected_args):
    parsed_args = AnyFunctionSchema(type_).parse_args(args_str)
    assert parsed_args == expected_args


@pytest.mark.parametrize(
    ("type_", "args_str", "expected_args"), any_function_schema_args_test_cases
)
@pytest.mark.asyncio
async def test_any_function_schema_aparse_args(type_, args_str, expected_args):
    parsed_args = await AnyFunctionSchema(type_).aparse_args(async_iter(args_str))
    assert parsed_args == expected_args


@pytest.mark.parametrize(
    ("type_", "expected_args_str", "args"), any_function_schema_args_test_cases
)
def test_any_function_schema_serialize_args(type_, expected_args_str, args):
    serialized_args = AnyFunctionSchema(type_).serialize_args(args)
    assert json.loads(serialized_args) == json.loads(expected_args_str)


@pytest.mark.parametrize(
    ("type_", "json_schema"),
    [
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
            list,
            {
                "name": "return_list",
                "parameters": {
                    "properties": {
                        "value": {
                            "title": "Value",
                            "type": "array",
                            "items": {},
                        }
                    },
                    "required": ["value"],
                    "type": "object",
                },
            },
        ),
        (
            typing.Iterable[str],
            {
                "name": "return_iterable_of_str",
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
            collections.abc.Iterable[str],
            {
                "name": "return_iterable_of_str",
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
    ],
)
def test_iterable_function_schema(type_, json_schema):
    function_schema = IterableFunctionSchema(type_)
    assert function_schema.dict() == json_schema


iterable_function_schema_args_test_cases = [
    (list[str], '{"value": ["One", "Two"]}', ["One", "Two"]),
    (typing.Iterable[str], '{"value": ["One", "Two"]}', ["One", "Two"]),
    (collections.abc.Iterable[str], '{"value": ["One", "Two"]}', ["One", "Two"]),
]


@pytest.mark.parametrize(
    ("type_", "args_str", "expected_args"), iterable_function_schema_args_test_cases
)
def test_iterable_function_schema_parse_args(type_, args_str, expected_args):
    parsed_args = IterableFunctionSchema(type_).parse_args(args_str)
    assert isinstance(parsed_args, get_origin(type_))
    assert list(parsed_args) == expected_args


@pytest.mark.parametrize(
    ("type_", "expected_args_str", "args"), iterable_function_schema_args_test_cases
)
def test_iterable_function_schema_serialize_args(type_, expected_args_str, args):
    serialized_args = IterableFunctionSchema(type_).serialize_args(args)
    assert json.loads(serialized_args) == json.loads(expected_args_str)


@pytest.mark.parametrize(
    ("type_", "json_schema"),
    [
        (
            typing.AsyncIterable[str],
            {
                "name": "return_asynciterable_of_str",
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
            typing.AsyncIterable,
            {
                "name": "return_asynciterable",
                "parameters": {
                    "properties": {
                        "value": {
                            "title": "Value",
                            "type": "array",
                            "items": {},
                        }
                    },
                    "required": ["value"],
                    "type": "object",
                },
            },
        ),
        (
            collections.abc.AsyncIterable[str],
            {
                "name": "return_asynciterable_of_str",
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
    ],
)
def test_async_iterable_function_schema(type_, json_schema):
    function_schema = AsyncIterableFunctionSchema(type_)
    assert function_schema.dict() == json_schema


async_iterable_function_schema_args_test_cases = [
    (typing.AsyncIterable[str], '{"value": ["One", "Two"]}', ["One", "Two"]),
    (collections.abc.AsyncIterable[str], '{"value": ["One", "Two"]}', ["One", "Two"]),
]


@pytest.mark.parametrize(
    ("type_", "args_str", "expected_args"),
    async_iterable_function_schema_args_test_cases,
)
@pytest.mark.asyncio
async def test_async_iterable_function_schema_aparse_args(
    type_, args_str, expected_args
):
    parsed_args = await AsyncIterableFunctionSchema(type_).aparse_args(
        async_iter(args_str)
    )
    assert isinstance(parsed_args, get_origin(type_))
    assert [arg async for arg in parsed_args] == expected_args


@pytest.mark.parametrize(
    ("type_", "expected_args_str", "args"),
    async_iterable_function_schema_args_test_cases,
)
@pytest.mark.asyncio
async def test_async_iterable_function_schema_aserialize_args(
    type_, expected_args_str, args
):
    serialized_args = await AsyncIterableFunctionSchema(type_).aserialize_args(
        async_iter(args)
    )
    assert json.loads(serialized_args) == json.loads(expected_args_str)


class User(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize(
    ("type_", "json_schema"),
    [
        (
            dict,
            {
                "name": "return_dict",
                "parameters": {
                    "properties": {},
                    "type": "object",
                },
            },
        ),
        (
            dict[str, Any],
            {
                "name": "return_dict_of_str_to_any",
                "parameters": {
                    "properties": {},
                    "type": "object",
                },
            },
        ),
        (
            dict[str, int],
            {
                "name": "return_dict_of_str_to_int",
                "parameters": {
                    "additionalProperties": {"type": "integer"},
                    "properties": {},
                    "type": "object",
                },
            },
        ),
        (
            dict[str, User],
            {
                "name": "return_dict_of_str_to_user",
                "parameters": {
                    "$defs": {
                        "User": {
                            "properties": {
                                "name": {"title": "Name", "type": "string"},
                                "age": {"title": "Age", "type": "integer"},
                            },
                            "required": ["name", "age"],
                            "title": "User",
                            "type": "object",
                        }
                    },
                    "additionalProperties": {"$ref": "#/$defs/User"},
                    "properties": {},
                    "type": "object",
                },
            },
        ),
    ],
)
def test_dict_function_schema(type_, json_schema):
    function_schema = DictFunctionSchema(type_)
    assert function_schema.dict() == json_schema


dict_function_schema_args_test_cases = [
    (dict, '{"name": "Alice"}', {"name": "Alice"}),
    (dict[str, Any], '{"name": "Alice"}', {"name": "Alice"}),
    (dict[str, str], '{"name": "Alice"}', {"name": "Alice"}),
    (dict[str, int], '{"age": 99}', {"age": 99}),
    (
        dict[str, User],
        '{"alice": {"name": "Alice", "age": 99}}',
        {"alice": User(name="Alice", age=99)},
    ),
    (OrderedDict[str, int], '{"age": 99}', OrderedDict({"age": 99})),
]


@pytest.mark.parametrize(
    ("type_", "args_str", "expected_args"), dict_function_schema_args_test_cases
)
def test_dict_function_schema_parse_args(type_, args_str, expected_args):
    parsed_args = DictFunctionSchema(type_).parse_args(args_str)
    assert parsed_args == expected_args
    assert is_origin_subclass(type_, type(parsed_args))


@pytest.mark.parametrize(
    ("type_", "expected_args_str", "args"), dict_function_schema_args_test_cases
)
def test_dict_function_schema_serialize_args(type_, expected_args_str, args):
    serialized_args = DictFunctionSchema(type_).serialize_args(args)
    assert json.loads(serialized_args) == json.loads(expected_args_str)


def test_base_model_function_schema():
    class User(BaseModel):
        name: str
        age: Annotated[int, Field(description="Age", gt=0, examples=[10, 20])]

    function_schema = BaseModelFunctionSchema(User)

    assert function_schema.name == "return_user"
    assert function_schema.dict() == {
        "name": "return_user",
        "parameters": {
            "type": "object",
            "properties": {
                # TODO: Remove "title" keys from schema
                "name": {"title": "Name", "type": "string"},
                "age": {
                    "description": "Age",
                    "examples": [10, 20],
                    "exclusiveMinimum": 0,
                    "title": "Age",
                    "type": "integer",
                },
            },
            "required": ["name", "age"],
        },
    }
    assert function_schema.parse_args('{"name": "Alice", "age": 99}') == User(
        name="Alice", age=99
    )


def test_base_model_function_schema_generic_model():
    T = TypeVar("T")

    class User(BaseModel, Generic[T]):
        name: str
        age: T

    function_schema = BaseModelFunctionSchema(User[int])

    assert function_schema.name == "return_user_int"
    assert function_schema.dict() == {
        "name": "return_user_int",
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


def plus_with_args(a: int, *args: int) -> int:
    return a + sum(args)


def plus_with_args_no_type_hints(a, *args):
    return a + sum(args)


def plus_with_kwargs(a: int, **kwargs: int) -> int:
    return a + sum(kwargs.values())


def plus_with_kwargs_no_type_hints(a, **kwargs):
    return a + sum(kwargs.values())


def plus_with_annotated(
    a: Annotated[int, Field(description="First number")],
    b: Annotated[int, Field(description="Second number")],
) -> int:
    return a + b


class IntModel(BaseModel):
    value: int


def plus_with_basemodel(a: IntModel, b: IntModel) -> IntModel:
    return IntModel(value=a.value + b.value)


@pytest.mark.parametrize(
    ("function", "json_schema"),
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
            plus_with_args,
            {
                "name": "plus_with_args",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"title": "A", "type": "integer"},
                        "args": {
                            "default": [],
                            "items": {"type": "integer"},
                            "title": "Args",
                            "type": "array",
                        },
                    },
                    "required": ["a"],
                },
            },
        ),
        (
            plus_with_args_no_type_hints,
            {
                "name": "plus_with_args_no_type_hints",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"title": "A"},
                        "args": {
                            "default": [],
                            "items": {},
                            "title": "Args",
                            "type": "array",
                        },
                    },
                    "required": ["a"],
                },
            },
        ),
        (
            plus_with_kwargs,
            {
                "name": "plus_with_kwargs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"title": "A", "type": "integer"},
                        "kwargs": {
                            "additionalProperties": {"type": "integer"},
                            "default": {},
                            "title": "Kwargs",
                            "type": "object",
                        },
                    },
                    "required": ["a"],
                },
            },
        ),
        (
            plus_with_kwargs_no_type_hints,
            {
                "name": "plus_with_kwargs_no_type_hints",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"title": "A"},
                        "kwargs": {
                            "default": {},
                            "title": "Kwargs",
                            "type": "object",
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
        (
            plus_with_basemodel,
            {
                "name": "plus_with_basemodel",
                "parameters": {
                    "$defs": {
                        "IntModel": {
                            "properties": {
                                "value": {"title": "Value", "type": "integer"}
                            },
                            "required": ["value"],
                            "title": "IntModel",
                            "type": "object",
                        }
                    },
                    "properties": {
                        "a": {"$ref": "#/$defs/IntModel"},
                        "b": {"$ref": "#/$defs/IntModel"},
                    },
                    "required": ["a", "b"],
                    "type": "object",
                },
            },
        ),
    ],
)
def test_function_call_function_schema(function, json_schema):
    function_schema = FunctionCallFunctionSchema(function)
    assert function_schema.dict() == json_schema


def test_function_call_function_schema_with_default_value():
    function_schema = FunctionCallFunctionSchema(plus_default_value)
    output = function_schema.parse_args('{"a": 1}')
    assert isinstance(output, FunctionCall)
    assert output() == 4


function_call_function_schema_args_test_cases = [
    (plus, '{"a": 1, "b": 2}', FunctionCall(plus, 1, 2)),
    (
        plus_no_type_hints,
        '{"a": 1, "b": 2}',
        FunctionCall(plus_no_type_hints, 1, 2),
    ),
    (plus_default_value, '{"a": 1}', FunctionCall(plus_default_value, 1)),
    (plus_with_args, '{"a": 1, "args": [2, 3]}', FunctionCall(plus_with_args, 1, 2, 3)),
    (
        plus_with_args_no_type_hints,
        '{"a": 1, "args": [2, 3]}',
        FunctionCall(plus_with_args_no_type_hints, 1, 2, 3),
    ),
    (
        plus_with_kwargs,
        '{"a": 1, "kwargs": {"b": 2, "c": 3}}',
        FunctionCall(plus_with_kwargs, 1, b=2, c=3),
    ),
    (
        plus_with_kwargs_no_type_hints,
        '{"a": 1, "kwargs": {"b": 2, "c": 3}}',
        FunctionCall(plus_with_kwargs_no_type_hints, 1, b=2, c=3),
    ),
    (
        plus_with_annotated,
        '{"a": 1, "b": 2}',
        FunctionCall(plus_with_annotated, 1, 2),
    ),
    (
        plus_with_basemodel,
        '{"a": {"value": 1}, "b": {"value": 2}}',
        FunctionCall(plus_with_basemodel, IntModel(value=1), IntModel(value=2)),
    ),
]


@pytest.mark.parametrize(
    ("function", "args_str", "expected_args"),
    function_call_function_schema_args_test_cases,
)
def test_function_call_function_schema_parse_args(function, args_str, expected_args):
    parsed_args = FunctionCallFunctionSchema(function).parse_args(args_str)
    assert parsed_args == expected_args


@pytest.mark.parametrize(
    ("function", "expected_args_str", "args"),
    function_call_function_schema_args_test_cases,
)
def test_function_call_function_schema_serialize_args(
    function, expected_args_str, args
):
    serialized_args = FunctionCallFunctionSchema(function).serialize_args(args)
    assert json.loads(serialized_args) == json.loads(expected_args_str)
