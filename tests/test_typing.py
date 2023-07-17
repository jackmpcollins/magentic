import pytest

from magentic.typing import name_type, split_union_type


@pytest.mark.parametrize(
    ["type_", "expected_types"],
    [
        (str, ["str"]),
        (int, ["int"]),
        (str | int, ["str", "int"]),
    ],
)
def test_split_union_type(type_, expected_types):
    assert [t.__name__ for t in split_union_type(type_)] == expected_types


@pytest.mark.parametrize(
    ["type_", "expected_name"],
    [
        (str, "str"),
        (int, "int"),
        (list[str], "list_of_str"),
        (str | bool, "str_or_bool"),
        (list[str | bool], "list_of_str_or_bool"),
        (list[str] | bool, "list_of_str_or_bool"),
        (dict[str, int], "dict_of_str_to_int"),
    ],
)
def test_name_type(type_, expected_name):
    assert name_type(type_) == expected_name
