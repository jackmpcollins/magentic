import inspect
from typing import Awaitable

import pytest
from typing_extensions import assert_type

from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.streaming import async_iter


def plus(a: int, b: int) -> int:
    return a + b


async def async_plus(a: int, b: int) -> int:
    return a + b


def plus_default_value(a: int, b: int = 3) -> int:
    return a + b


def return_hello() -> str:
    return "hello"


@pytest.mark.parametrize(
    ("left", "right", "equal"),
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
    ("function_call", "arguments"),
    [
        (FunctionCall(plus, a=1, b=2), {"a": 1, "b": 2}),
        (FunctionCall(plus, 1, 2), {"a": 1, "b": 2}),
        (FunctionCall(plus_default_value, a=1), {"a": 1}),
    ],
)
def test_function_call_arguments(function_call, arguments):
    assert function_call.arguments == arguments


@pytest.mark.asyncio
async def test_function_call_async_function():
    async def async_plus(a: int, b: int) -> int:
        return a + b

    function_call = FunctionCall(async_plus, a=1, b=2)
    result = function_call()
    assert inspect.isawaitable(result)
    assert await result == 3


def test_parallel_function_call_call():
    parallel_function_call: ParallelFunctionCall[int | str] = ParallelFunctionCall(
        [FunctionCall(plus, a=1, b=2), FunctionCall(return_hello)]
    )
    assert_type(parallel_function_call, ParallelFunctionCall[int | str])
    result = parallel_function_call()
    assert_type(result, tuple[int | str, ...])
    assert result == (3, "hello")


def test_parallel_function_call_iter():
    function_calls: list[FunctionCall[int | str]] = [
        FunctionCall(plus, a=1, b=2),
        FunctionCall(return_hello),
    ]
    parallel_function_call = ParallelFunctionCall(iter(function_calls))
    assert list(parallel_function_call) == function_calls
    assert list(parallel_function_call) == function_calls


@pytest.mark.asyncio
async def test_async_parallel_function_call_call():
    function_calls: list[FunctionCall[int | Awaitable[int]]] = [
        FunctionCall(plus, a=1, b=2),
        FunctionCall(async_plus, a=3, b=4),
    ]
    async_parallel_function_call = AsyncParallelFunctionCall(async_iter(function_calls))
    assert_type(async_parallel_function_call, AsyncParallelFunctionCall[int])
    result = await async_parallel_function_call()
    assert_type(result, tuple[int, ...])
    assert result == (3, 7)


@pytest.mark.asyncio
async def test_async_parallel_function_call_aiter():
    function_calls: list[FunctionCall[int | Awaitable[int]]] = [
        FunctionCall(plus, a=1, b=2),
        FunctionCall(async_plus, a=3, b=4),
    ]
    async_parallel_function_call = AsyncParallelFunctionCall(async_iter(function_calls))
    assert [x async for x in async_parallel_function_call] == function_calls
    assert [x async for x in async_parallel_function_call] == function_calls
