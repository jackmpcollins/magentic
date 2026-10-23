import asyncio
import inspect
from collections.abc import Awaitable, Generator
from typing import Any

import pytest

from magentic.function_call import AsyncParallelFunctionCall, FunctionCall
from magentic.streaming import async_iter


def identity(value: object) -> object:
    return value


class FutureLike:
    """Non-subclass Future implementing asyncio's documented protocol."""

    _asyncio_future_blocking = False

    def __init__(self, future: asyncio.Future[Any]):
        self.future = future

    def __await__(self) -> Generator[Any, None, Any]:
        return self.future.__await__()

    def _make_cancelled_error(self):
        # asyncio.gather uses this hook to preserve cancellation details.
        try:
            self.future.result()
        except asyncio.CancelledError as error:
            return error
        message = "Future is not cancelled"
        raise AssertionError(message)

    def get_loop(self):
        return self.future.get_loop()

    def result(self):
        return self.future.result()

    def done(self):
        return self.future.done()

    def cancelled(self):
        return self.future.cancelled()

    def cancel(self, *args, **kwargs):
        return self.future.cancel(*args, **kwargs)

    def exception(self):
        return self.future.exception()

    def add_done_callback(self, callback, *, context=None):
        self.future.add_done_callback(lambda _: callback(self), context=context)


class GenericAwaitable:
    def __init__(self, future: asyncio.Future[Any]):
        self.future = future

    def __await__(self) -> Generator[Any, None, Any]:
        return self.future.__await__()


def wrap(future: asyncio.Future[Any], kind: str) -> Awaitable[Any]:
    if kind == "future_like":
        result = FutureLike(future)
        assert asyncio.isfuture(result)
        assert inspect.isawaitable(result)
        assert not isinstance(result, asyncio.Future)
        assert asyncio.ensure_future(result) is result
        return result
    if kind == "generic":
        return GenericAwaitable(future)
    return future


@pytest.mark.parametrize("kind", ["native", "future_like", "generic"])
@pytest.mark.parametrize("state", ["resolved", "pending", "exception", "cancelled"])
async def test_future_protocol_states(kind, state):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    sentinel = object()
    if state == "resolved":
        future.set_result(sentinel)
    elif state == "pending":
        loop.call_soon(future.set_result, sentinel)
    elif state == "exception":
        future.set_exception(ValueError("tool failed"))
    else:
        future.cancel()
    value = wrap(future, kind)
    call = AsyncParallelFunctionCall(async_iter([FunctionCall(lambda: value)]))
    try:
        if state == "exception":
            with pytest.raises(ValueError, match="tool failed"):
                await call()
        elif state == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await call()
        else:
            result = await call()
            assert future.done()
            assert result[0] is sentinel
    finally:
        # Consume exceptions even on the intentionally failing baseline run.
        if future.done() and not future.cancelled():
            future.exception()
        await asyncio.sleep(0)


async def test_mixed_protocol_order_identity_and_concurrency():
    loop = asyncio.get_running_loop()
    first, second = loop.create_future(), loop.create_future()
    sync_value, async_value = object(), object()
    completion_order = []

    async def complete() -> int:
        second.set_result(2)
        completion_order.append(2)
        await asyncio.sleep(0)
        first.set_result(async_value)
        completion_order.append(1)
        return 3

    task = asyncio.create_task(complete())
    values = [sync_value, FutureLike(first), GenericAwaitable(second), task]
    calls = [FunctionCall(identity, value) for value in values]
    result = await AsyncParallelFunctionCall(async_iter(calls))()
    assert result == (sync_value, async_value, 2, 3)
    assert result[0] is sync_value
    assert result[1] is async_value
    assert completion_order == [2, 1]


async def test_duplicate_future_like_result():
    future = asyncio.get_running_loop().create_future()
    sentinel = object()
    future.set_result(sentinel)
    value = FutureLike(future)
    calls = [FunctionCall(identity, value), FunctionCall(identity, value)]
    result = await AsyncParallelFunctionCall(async_iter(calls))()
    assert result[0] is result[1] is sentinel


@pytest.mark.parametrize("kind", ["native", "future_like", "generic"])
async def test_caller_cancellation_propagates(kind):
    future = asyncio.get_running_loop().create_future()
    value = wrap(future, kind)
    call = AsyncParallelFunctionCall(async_iter([FunctionCall(lambda: value)]))
    task = asyncio.create_task(call())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert future.cancelled()
    finally:
        future.cancel()


class SyncMarkerData:
    _asyncio_future_blocking = False


class SyncFutureFacade:
    _asyncio_future_blocking = False

    def __init__(self, future: asyncio.Future[int]):
        self.future = future

    def get_loop(self):
        return self.future.get_loop()

    def done(self):
        return self.future.done()

    def result(self):
        return self.future.result()

    def cancelled(self):
        return self.future.cancelled()

    def exception(self):
        return self.future.exception()


@pytest.mark.parametrize("facade", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
async def test_non_awaitable_future_marker_preserves_sync_identity(facade, mixed):
    future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    future.set_result(42)
    value = SyncFutureFacade(future) if facade else SyncMarkerData()
    assert asyncio.isfuture(value)
    assert not inspect.isawaitable(value)
    awaited = FutureLike(future)
    middle = awaited if mixed else 42
    calls = [FunctionCall(identity, item) for item in [value, middle, value]]
    result = await AsyncParallelFunctionCall(async_iter(calls))()
    assert result[0] is result[2] is value
    assert result[1] == 42
