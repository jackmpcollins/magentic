import pytest

from magentic.function_call import FunctionCall


def plus(a: int, b: int) -> int:
    return a + b


def plus_default_value(a: int, b: int = 3) -> int:
    return a + b


@pytest.mark.parametrize(
    ["left", "right", "equal"],
    [
        (FunctionCall(plus, a=1, b=2), FunctionCall(plus, a=1, b=2), True),
        (FunctionCall(plus, a=1, b=2), FunctionCall(plus, a=1, b=33), False),
        (
            FunctionCall(plus, a=1, b=2),
            FunctionCall(plus_default_value, a=1, b=2),
            False,
        ),
        (
            FunctionCall(plus_default_value, a=1),
            FunctionCall(plus_default_value, a=1, b=3),
            False,  # TODO: Should default values be considered? That would make this True.
        ),
        (
            FunctionCall(plus_default_value, a=1),
            FunctionCall(plus_default_value, a=1, b=44),
            False,
        ),
    ],
)
def test_function_call_eq(left, right, equal):
    assert (left == right) is equal, (left, right, equal)


@pytest.mark.parametrize(
    ["function_call", "arguments"],
    [
        (FunctionCall(plus, a=1, b=2), {"a": 1, "b": 2}),
        (FunctionCall(plus, 1, 2), {"a": 1, "b": 2}),
        (FunctionCall(plus_default_value, a=1), {"a": 1}),
    ],
)
def test_function_call_arguments(function_call, arguments):
    assert function_call.arguments == arguments
