import asyncio
import inspect
from collections.abc import Awaitable

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


async def test_async_parallel_function_call_aiter():
    function_calls: list[FunctionCall[int | Awaitable[int]]] = [
        FunctionCall(plus, a=1, b=2),
        FunctionCall(async_plus, a=3, b=4),
    ]
    async_parallel_function_call = AsyncParallelFunctionCall(async_iter(function_calls))
    assert [x async for x in async_parallel_function_call] == function_calls
    assert [x async for x in async_parallel_function_call] == function_calls


class CustomAwaitable:
    def __init__(self, future: asyncio.Future[int]):
        self.future = future

    def __await__(self):
        return self.future.__await__()


async def test_async_parallel_function_call_awaitables_preserve_order():
    loop = asyncio.get_running_loop()
    future: asyncio.Future[int] = loop.create_future()
    custom_future: asyncio.Future[int] = loop.create_future()
    completed: list[int] = []

    async def coroutine() -> int:
        custom_future.set_result(4)
        completed.append(4)
        future.set_result(2)
        completed.append(2)
        return 3

    def return_future() -> asyncio.Future[int]:
        return future

    def return_custom_awaitable() -> CustomAwaitable:
        return CustomAwaitable(custom_future)

    def return_task() -> asyncio.Task[int]:
        return asyncio.create_task(async_plus(2, 3))

    function_calls: list[FunctionCall[Awaitable[int] | int]] = [
        FunctionCall(plus, 0, 1),
        FunctionCall(return_future),
        FunctionCall(coroutine),
        FunctionCall(return_custom_awaitable),
        FunctionCall(return_task),
    ]
    result = await AsyncParallelFunctionCall(async_iter(function_calls))()
    assert result == (1, 2, 3, 4, 5)
    assert completed == [4, 2]


@pytest.mark.parametrize("custom", [False, True])
async def test_async_parallel_function_call_awaitable_exception(custom):
    def function() -> Awaitable[int]:
        future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        future.set_exception(ValueError("tool failed"))
        return CustomAwaitable(future) if custom else future

    call = AsyncParallelFunctionCall[int](async_iter([FunctionCall(function)]))
    with pytest.raises(ValueError, match="tool failed"):
        await call()


@pytest.mark.parametrize("custom", [False, True])
async def test_async_parallel_function_call_awaitable_cancellation(custom):
    def function() -> Awaitable[int]:
        future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        future.cancel()
        return CustomAwaitable(future) if custom else future

    call = AsyncParallelFunctionCall[int](async_iter([FunctionCall(function)]))
    with pytest.raises(asyncio.CancelledError):
        await call()


async def test_async_parallel_function_call_empty():
    call: AsyncParallelFunctionCall[int] = AsyncParallelFunctionCall(async_iter([]))
    assert await call() == ()


def test_async_parallel_function_call_future_uses_current_loop():
    def function() -> asyncio.Future[int]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[int] = loop.create_future()
        loop.call_soon(future.set_result, 7)
        return future

    call = AsyncParallelFunctionCall[int](async_iter([FunctionCall(function)]))
    assert asyncio.run(call()) == (7,)
    assert asyncio.run(call()) == (7,)


async def test_async_parallel_function_call_sync_results():
    sentinel = object()

    def function() -> object:
        return sentinel

    call = AsyncParallelFunctionCall[object](async_iter([FunctionCall(function)]))
    assert await call() == (sentinel,)


async def test_async_parallel_function_call_coroutine_exception():
    async def function() -> int:
        msg = "tool failed"
        raise ValueError(msg)

    call = AsyncParallelFunctionCall[int](async_iter([FunctionCall(function)]))
    with pytest.raises(ValueError, match="tool failed"):
        await call()


async def test_async_parallel_function_call_sync_exception():
    def function() -> int:
        msg = "tool failed"
        raise ValueError(msg)

    call = AsyncParallelFunctionCall[int](async_iter([FunctionCall(function)]))
    with pytest.raises(ValueError, match="tool failed"):
        await call()
