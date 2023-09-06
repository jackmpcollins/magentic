from collections.abc import AsyncIterable, Iterable
from typing import AsyncIterator, Iterator


async def async_iter(iterable: Iterable[str]) -> AsyncIterator[str]:
    """Get an AsyncIterator for an Iterable."""
    for item in iterable:
        yield item


class StreamedStr(Iterable[str]):
    """A string that is generated in chunks."""

    def __init__(self, chunks: Iterable[str]):
        self._chunks = chunks
        self._cached_chunks: list[str] = []

    def __iter__(self) -> Iterator[str]:
        yield from self._cached_chunks
        for chunk in self._chunks:
            self._cached_chunks.append(chunk)
            yield chunk

    def __str__(self) -> str:
        return "".join(self)

    def to_string(self) -> str:
        """Convert the streamed string to a string."""
        return str(self)


class AsyncStreamedStr(AsyncIterable[str]):
    """Async version of `StreamedStr`."""

    def __init__(self, chunks: AsyncIterable[str]):
        self._chunks = chunks
        self._cached_chunks: list[str] = []

    async def __aiter__(self) -> AsyncIterator[str]:
        # Cannot use `yield from` inside an async function
        # https://peps.python.org/pep-0525/#asynchronous-yield-from
        for chunk in self._cached_chunks:
            yield chunk
        async for chunk in self._chunks:
            self._cached_chunks.append(chunk)
            yield chunk

    async def to_string(self) -> str:
        """Convert the streamed string to a string."""
        return "".join([item async for item in self])
