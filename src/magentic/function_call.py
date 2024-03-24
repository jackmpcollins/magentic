import asyncio
import inspect
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Iterator,
    ParamSpec,
    Tuple,
    TypeVar,
    cast,
)
from uuid import uuid4

from magentic.streaming import CachedAsyncIterable, CachedIterable

T = TypeVar("T")
P = ParamSpec("P")


class FunctionCall(Generic[T]):
    """A function with arguments supplied.

    Calling the instance will call the function with the supplied arguments.
    """

    def __init__(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        self._function = function
        self._args = args
        self._kwargs = kwargs

        # Used to correlate function call with result on serialization
        self._unique_id = str(uuid4())

    def __call__(self) -> T:
        return self._function(*self._args, **self._kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._function == other._function
            and self._args == other._args
            and self._kwargs == other._kwargs
        )

    def __repr__(self) -> str:
        args_kwargs_repr = ", ".join(
            [
                *(repr(arg) for arg in self._args),
                *(f"{key}={value!r}" for key, value in self._kwargs.items()),
            ]
        )
        return f"{type(self).__name__}({self._function!r}, {args_kwargs_repr})"

    @property
    def function(self):
        return self._function

    @property
    def arguments(self) -> dict[str, Any]:
        signature = inspect.signature(self._function)
        bound_args = signature.bind(*self._args, **self._kwargs)
        return bound_args.arguments.copy()


class ParallelFunctionCall(Generic[T]):
    """A collection of FunctionCalls that can be made concurrently."""

    def __init__(self, function_calls: Iterable[FunctionCall[T]]):
        self._function_calls = CachedIterable(function_calls)

    def __call__(self) -> tuple[T, ...]:
        return tuple(function_call() for function_call in self._function_calls)

    def __iter__(self) -> Iterator[FunctionCall[T]]:
        yield from self._function_calls


class AsyncParallelFunctionCall(Generic[T]):
    """Async version of `ParallelFunctionCall`."""

    def __init__(self, function_calls: AsyncIterable[FunctionCall[Awaitable[T] | T]]):
        self._function_calls = CachedAsyncIterable(function_calls)

    async def __call__(self) -> Tuple[T, ...]:
        tasks_and_results: list[asyncio.Task[T] | T] = []
        async for function_call in self._function_calls:
            result = function_call()
            if inspect.iscoroutine(result):
                tasks_and_results.append(asyncio.create_task(result))
            else:
                result = cast(T, result)
                tasks_and_results.append(result)

        tasks = [task for task in tasks_and_results if isinstance(task, asyncio.Task)]
        await asyncio.gather(*tasks)
        results = [
            task_or_result.result()
            if isinstance(task_or_result, asyncio.Task)
            else task_or_result
            for task_or_result in tasks_and_results
        ]
        return tuple(results)

    async def __aiter__(self) -> AsyncIterator[FunctionCall[Awaitable[T] | T]]:
        async for function_call in self._function_calls:
            yield function_call
