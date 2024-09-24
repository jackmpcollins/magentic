import asyncio
import textwrap
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, dropwhile
from typing import Any, TypeVar

T = TypeVar("T")


async def async_iter(iterable: Iterable[T]) -> AsyncIterator[T]:
    """Get an AsyncIterator for an Iterable."""
    for item in iterable:
        yield item


def apply(func: Callable[[T], Any], iterable: Iterable[T]) -> Iterator[T]:
    """Apply a function to each item in an iterable and yield the original item."""
    for chunk in iterable:
        func(chunk)
        yield chunk


async def aapply(
    func: Callable[[T], Any], aiterable: AsyncIterable[T]
) -> AsyncIterator[T]:
    """Async version of `apply`."""
    async for chunk in aiterable:
        func(chunk)
        yield chunk


async def azip(*aiterables: AsyncIterable[T]) -> AsyncIterator[tuple[T, ...]]:
    """Async version of `zip`."""
    aiterators = [aiter(aiterable) for aiterable in aiterables]
    try:
        while True:
            yield tuple(
                await asyncio.gather(*(anext(aiterator) for aiterator in aiterators))
            )
    except StopAsyncIteration:
        return


async def achain(*aiterables: AsyncIterable[T]) -> AsyncIterator[T]:
    """Async version of `itertools.chain`."""
    for aiterable in aiterables:
        async for item in aiterable:
            yield item


def peek(iterator: Iterator[T]) -> tuple[T, Iterator[T]]:
    """Returns the first item in the Iterator and a copy of the Iterator."""
    first_item = next(iterator)
    return first_item, chain([first_item], iterator)


async def apeek(aiterator: AsyncIterator[T]) -> tuple[T, AsyncIterator[T]]:
    """Async version of `peek`."""
    first_item = await anext(aiterator)
    return first_item, achain(async_iter([first_item]), aiterator)


async def adropwhile(
    predicate: Callable[[T], object], aiterable: AsyncIterable[T]
) -> AsyncIterator[T]:
    """Async version of `itertools.dropwhile`."""
    aiterator = aiter(aiterable)
    async for item in aiterator:
        if not predicate(item):
            yield item
            break
    async for item in aiterator:
        yield item


async def atakewhile(
    predicate: Callable[[T], object], aiterable: AsyncIterable[T]
) -> AsyncIterator[T]:
    """Async version of `itertools.takewhile`."""
    async for item in aiterable:
        if not predicate(item):
            break
        yield item


async def agroupby(
    aiterable: AsyncIterable[T], key: Callable[[T], object]
) -> AsyncIterator[tuple[object, AsyncIterator[T]]]:
    """Async version of `itertools.groupby`."""
    aiterator = aiter(aiterable)
    transition = [await anext(aiterator)]

    async def agroup(
        aiterator: AsyncIterator[T], group_key: object
    ) -> AsyncIterator[T]:
        async for item in aiterator:
            if key(item) != group_key:
                transition.append(item)
                return
            yield item

    while transition:
        transition_item = transition.pop()
        group_key = key(transition_item)
        yield (
            group_key,
            agroup(achain(async_iter([transition_item]), aiterator), group_key),
        )


@dataclass
class JsonArrayParserState:
    """State of the parser for a streamed JSON array."""

    array_level: int = 0
    object_level: int = 0
    in_string: bool = False
    is_escaped: bool = False
    is_element_separator: bool = False

    def update(self, char: str) -> None:
        if self.in_string:
            if char == '"' and not self.is_escaped:
                self.in_string = False
        elif char == '"':
            self.in_string = True
        elif char == ",":
            if self.array_level == 1 and self.object_level == 0:
                self.is_element_separator = True
                return
        elif char == "[":
            self.array_level += 1
        elif char == "]":
            self.array_level -= 1
            if self.array_level == 0:
                self.is_element_separator = True
                return
        elif char == "{":
            self.object_level += 1
        elif char == "}":
            self.object_level -= 1
        elif char == "\\":
            self.is_escaped = not self.is_escaped
        else:
            self.is_escaped = False
        self.is_element_separator = False


def iter_streamed_json_array(chunks: Iterable[str]) -> Iterable[str]:
    """Convert a streamed JSON array into an iterable of JSON object strings.

    This ignores all characters before the start of the first array i.e. the first "["
    """
    iter_chars: Iterator[str] = chain.from_iterable(chunks)
    parser_state = JsonArrayParserState()

    iter_chars = dropwhile(lambda x: x != "[", iter_chars)
    parser_state.update(next(iter_chars))

    item_chars: list[str] = []
    for char in iter_chars:
        parser_state.update(char)
        if parser_state.is_element_separator:
            if item_chars:
                yield "".join(item_chars).strip()
                item_chars = []
        else:
            item_chars.append(char)


async def aiter_streamed_json_array(chunks: AsyncIterable[str]) -> AsyncIterable[str]:
    """Async version of `iter_streamed_json_array`."""

    async def chars_generator() -> AsyncIterable[str]:
        async for chunk in chunks:
            for char in chunk:
                yield char

    iter_chars = chars_generator()
    parser_state = JsonArrayParserState()

    async for char in iter_chars:
        if char == "[":
            break
    parser_state.update("[")

    item_chars: list[str] = []
    async for char in iter_chars:
        parser_state.update(char)
        if parser_state.is_element_separator:
            if item_chars:
                yield "".join(item_chars).strip()
                item_chars = []
        else:
            item_chars.append(char)


class CachedIterable(Iterable[T]):
    """Wraps an Iterable and caches the items after the first iteration."""

    def __init__(self, iterable: Iterable[T]):
        self._iterator = iter(iterable)
        self._cached_items: list[T] = []

    def __iter__(self) -> Iterator[T]:
        yield from self._cached_items
        for item in self._iterator:
            self._cached_items.append(item)
            yield item


class CachedAsyncIterable(AsyncIterable[T]):
    """Async version of `CachedIterable`."""

    def __init__(self, aiterable: AsyncIterable[T]):
        self._aiterator = aiter(aiterable)
        self._cached_items: list[T] = []

    async def __aiter__(self) -> AsyncIterator[T]:
        for item in self._cached_items:
            yield item
        async for item in self._aiterator:
            self._cached_items.append(item)
            yield item


class StreamedStr(Iterable[str]):
    """A string that is generated in chunks."""

    def __init__(self, chunks: Iterable[str]):
        self._chunks = CachedIterable(chunks)

    def __iter__(self) -> Iterator[str]:
        yield from self._chunks

    def __str__(self) -> str:
        return "".join(self)

    def to_string(self) -> str:
        """Convert the streamed string to a string."""
        return str(self)

    def truncate(self, length: int) -> str:
        """Truncate the streamed string to the specified length."""
        chunks = []
        current_length = 0
        for chunk in self._chunks:
            chunks.append(chunk)
            current_length += len(chunk)
            if current_length > length:
                break
        return textwrap.shorten("".join(chunks), width=length)


class AsyncStreamedStr(AsyncIterable[str]):
    """Async version of `StreamedStr`."""

    def __init__(self, chunks: AsyncIterable[str]):
        self._chunks = CachedAsyncIterable(chunks)

    async def __aiter__(self) -> AsyncIterator[str]:
        async for chunk in self._chunks:
            yield chunk

    async def to_string(self) -> str:
        """Convert the streamed string to a string."""
        return "".join([item async for item in self])

    async def truncate(self, length: int) -> str:
        """Truncate the streamed string to the specified length."""
        chunks = []
        current_length = 0
        async for chunk in self._chunks:
            chunks.append(chunk)
            current_length += len(chunk)
            if current_length > length:
                break
        return textwrap.shorten("".join(chunks), width=length)
