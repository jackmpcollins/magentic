from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from typing import Any

from magentic.function_call import FunctionCall
from magentic.streaming import (
    AsyncStreamedStr,
    CachedAsyncIterable,
    CachedIterable,
    StreamedStr,
)


class StreamedResponse:
    """A streamed LLM response consisting of text output and tool calls.

    This is an iterable of StreamedStr and FunctionCall instances.

    Examples
    --------
    >>> from magentic import prompt, StreamedResponse, StreamedStr, FunctionCall
    >>>
    >>> def get_weather(city: str) -> str:
    >>>     return f"The weather in {city} is 20°C."
    >>>
    >>> @prompt(
    >>>     "Say hello, then get the weather for: {cities}",
    >>>     functions=[get_weather],
    >>> )
    >>> def describe_weather(cities: list[str]) -> StreamedResponse: ...
    >>>
    >>> response = describe_weather(["Cape Town", "San Francisco"])
    >>>
    >>> for item in response:
    >>>     if isinstance(item, StreamedStr):
    >>>         for chunk in item:
    >>>             print(chunk, sep="", end="")
    >>>         print()
    >>>     if isinstance(item, FunctionCall):
    >>>         print(item)
    >>>         print(item())
    Hello! I'll get the weather for Cape Town and San Francisco for you.
    FunctionCall(<function get_weather at 0x1109825c0>, 'Cape Town')
    The weather in Cape Town is 20°C.
    FunctionCall(<function get_weather at 0x1109825c0>, 'San Francisco')
    The weather in San Francisco is 20°C.
    """

    def __init__(self, stream: Iterable[StreamedStr | FunctionCall[Any]]):
        self._stream = CachedIterable(stream)

    def __iter__(self) -> Iterator[StreamedStr | FunctionCall[Any]]:
        yield from self._stream


class AsyncStreamedResponse:
    """Async version of `StreamedResponse`."""

    def __init__(self, stream: AsyncIterable[AsyncStreamedStr | FunctionCall[Any]]):
        self._stream = CachedAsyncIterable(stream)

    async def __aiter__(self) -> AsyncIterator[AsyncStreamedStr | FunctionCall[Any]]:
        async for item in self._stream:
            yield item
