from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from typing import Any

from typing_extensions import Self

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

    def __init__(
        self,
        stream: Iterable[StreamedStr | FunctionCall[Any]],
        *,
        on_close: Callable[[], None] | None = None,
    ):
        self._stream = CachedIterable(stream)
        self._on_close = on_close
        self._closed = False

    def close(self) -> None:
        """关闭响应并保留已缓存的对象"""
        if not self._closed:
            self._closed = True
            try:
                self._stream.close()
            finally:
                if self._on_close:
                    self._on_close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[StreamedStr | FunctionCall[Any]]:
        yield from self._stream


class AsyncStreamedResponse:
    """Async version of `StreamedResponse`."""

    def __init__(
        self,
        stream: AsyncIterable[AsyncStreamedStr | FunctionCall[Any]],
        *,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ):
        self._stream = CachedAsyncIterable(stream)
        self._on_close = on_close
        self._closed = False

    async def aclose(self) -> None:
        """关闭响应并保留已缓存的对象"""
        if not self._closed:
            self._closed = True
            try:
                await self._stream.aclose()
            finally:
                if self._on_close:
                    await self._on_close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def __aiter__(self) -> AsyncIterator[AsyncStreamedStr | FunctionCall[Any]]:
        async for item in self._stream:
            yield item
