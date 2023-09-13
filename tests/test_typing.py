import typing
from types import NoneType
from typing import Any

import pytest

from magentic.typing import (
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
        (dict, dict, True),
        (NoneType, dict, False),
        (Any, dict, False),
    ],
)
def test_is_origin_subclass(type_, cls_or_tuple, expected_result):
    assert is_origin_subclass(type_, cls_or_tuple) == expected_result


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
    ],
)
def test_name_type(type_, expected_name):
    assert name_type(type_) == expected_name
