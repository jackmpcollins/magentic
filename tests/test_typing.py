import typing
from types import NoneType
from typing import Any, Generic, Iterable, TypeVar

import pytest
from pydantic import BaseModel

from magentic.typing import (
    is_any_origin_subclass,
    is_origin_abstract,
    is_origin_subclass,
    name_type,
    split_union_type,
)


@pytest.mark.parametrize(
    ("type_", "expected_types"),
    [
        (str, ["str"]),
        (int, ["int"]),
        (str | int, ["str", "int"]),
    ],
)
def test_split_union_type(type_, expected_types):
    assert [t.__name__ for t in split_union_type(type_)] == expected_types


@pytest.mark.parametrize(
    ("type_", "expected_result"),
    [
        (str, False),
        (list[str], False),
        (typing.Iterable[str], True),
    ],
)
def test_is_origin_abstract(type_, expected_result):
    assert is_origin_abstract(type_) == expected_result


@pytest.mark.parametrize(
    ("type_", "cls_or_tuple", "expected_result"),
    [
        (str, str, True),
        (str, int, False),
        (str, (str, int), True),
        (list[str], list, True),
        (list[str], Iterable, True),
        (list[str], int, False),
        (dict, dict, True),
        (dict[str, int], dict, True),
        (NoneType, dict, False),
        (Any, dict, False),
    ],
)
def test_is_origin_subclass(type_, cls_or_tuple, expected_result):
    assert is_origin_subclass(type_, cls_or_tuple) == expected_result


@pytest.mark.parametrize(
    ("types", "cls_or_tuple", "expected_result"),
    [
        ([str, int], str, True),
        ([list[str], str], list, True),
        ([list[str], str], int, False),
    ],
)
def test_is_any_origin_subclass(types, cls_or_tuple, expected_result):
    assert is_any_origin_subclass(types, cls_or_tuple) == expected_result


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class GenericModel(BaseModel, Generic[T1]):
    one: T1


class DoubleGenericModel(BaseModel, Generic[T1, T2]):
    one: T1
    two: T2


@pytest.mark.parametrize(
    ("type_", "expected_name"),
    [
        (str, "str"),
        (int, "int"),
        (list[str], "list_of_str"),
        (str | bool, "str_or_bool"),
        (list[str | bool], "list_of_str_or_bool"),
        (list[str] | bool, "list_of_str_or_bool"),
        (dict[str, int], "dict_of_str_to_int"),
        (typing.Iterable[str], "iterable_of_str"),
        (GenericModel[int], "genericmodel_int"),
        (GenericModel[GenericModel[int]], "genericmodel_genericmodel_int"),
        (DoubleGenericModel[int, str], "doublegenericmodel_int_str"),
    ],
)
def test_name_type(type_, expected_name):
    assert name_type(type_) == expected_name
