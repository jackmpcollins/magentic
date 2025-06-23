import asyncio
import collections
import json
import textwrap
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, TypeVar

from pydantic_core import from_json

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


def consume(iterator: Iterable[T]) -> None:
    """Consume an iterator."""
    collections.deque(iterator, maxlen=0)


async def aconsume(aiterable: AsyncIterable[T]) -> None:
    """Async version of `consume`."""
    async for _ in aiterable:
        pass


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
        aiterator = achain(async_iter([transition_item]), aiterator)
        yield (group_key, agroup(aiterator, group_key))
        # Finish the group to allow advancing to the next one
        if not transition:
            await aconsume(agroup(aiterator, group_key))


@dataclass
class JsonArrayParserState:
    """State of the parser for a streamed JSON array."""

    array_level: int = 0
    object_level: int = 0
    in_string: bool = False
    is_escaped: bool = False
    current_item: list[str] = field(default_factory=list)
    is_element_separator: bool = False

    def update(self, char: str) -> None:
        if self.in_string:
            if char == '"' and not self.is_escaped:
                self.in_string = False
            self.current_item.append(char)
        elif char == '"':
            self.in_string = True
            self.current_item.append(char)
        elif char.isspace():
            self.current_item.append(char)
        elif char == ",":
            if self.array_level == 1 and self.object_level == 0:
                self.is_element_separator = True
                return
            self.current_item.append(char)
        elif char == "[":
            self.array_level += 1
        elif char == "]":
            self.array_level -= 1
            if self.array_level == 0:
                self.is_element_separator = True
                return
        elif char == "{":
            self.object_level += 1
            self.current_item.append(char)
        elif char == "}":
            self.object_level -= 1
            self.current_item.append(char)
        elif char == "\\":
            self.is_escaped = not self.is_escaped
            self.current_item.append(char)
        else:
            self.is_escaped = False
            self.current_item.append(char)
        self.is_element_separator = False


def iter_streamed_json_array(chunks: Iterable[str]) -> Iterable[str]:
    """Convert a streamed JSON array into an iterable of JSON object strings.

    This function processes a stream of JSON chunks and yields complete array elements
    as they become available. It uses two strategies:
    1. Try Pydantic's from_json for efficient parsing of complete chunks
    2. Fall back to character-by-character parsing for handling edge cases

    Args:
        chunks: An iterable of JSON string chunks that together form a JSON array
               or an object with a "value" field containing an array.

    Yields:
        Complete JSON-encoded strings for each array element.

    Example:
        >>> chunks = ["[1, 2", ", 3]"]
        >>> list(iter_streamed_json_array(chunks))
        ['1', '2', '3']
    """
    accumulated = ""
    yielded_items: set[str] = set()

    for chunk in chunks:
        accumulated += chunk

        # Ensure array starts with '['.  Remove leading garbage.
        if accumulated and accumulated[0] != "[":
            first_bracket_index = accumulated.find("[")
            if first_bracket_index == -1:
                continue  # Keep accumulating until we find the first bracket
            else:
                accumulated = accumulated[first_bracket_index:]

        try:
            result = from_json(accumulated, allow_partial=True)
            items = result.get("value", result) if isinstance(result, dict) else result

            if isinstance(items, list):
                for item in items:
                    item_str = json.dumps(item, ensure_ascii=False)
                    if item_str not in yielded_items:
                        yield item_str
                        yielded_items.add(item_str)

        except (ValueError, TypeError, KeyError):
            # Character-by-character parsing
            parser_state = JsonArrayParserState()
            for char in accumulated:
                parser_state.update(char)
                if parser_state.is_element_separator and parser_state.current_item:
                    item = "".join(parser_state.current_item).strip()
                    if item not in yielded_items and item:
                        yield item
                        yielded_items.add(item)
                    parser_state.current_item = []
            # yield anything that is left
            if parser_state.current_item:
                item = "".join(parser_state.current_item).strip()
                if item not in yielded_items:
                    yield item
                    yielded_items.add(item)


async def aiter_streamed_json_array(chunks: AsyncIterable[str]) -> AsyncIterable[str]:
    """Async version of `iter_streamed_json_array`."""
    accumulated = ""
    yielded_items: set[str] = set()

    async for chunk in chunks:
        accumulated += chunk

        # Ensure array starts with '['.  Remove leading garbage.
        if accumulated and accumulated[0] != "[":
            first_bracket_index = accumulated.find("[")
            if first_bracket_index == -1:
                continue  # Keep accumulating until we find the first bracket
            else:
                accumulated = accumulated[first_bracket_index:]

        try:
            result = from_json(accumulated, allow_partial=True)
            items = result.get("value", result) if isinstance(result, dict) else result

            if isinstance(items, list):
                for item in items:
                    item_str = json.dumps(item, ensure_ascii=False)
                    if item_str not in yielded_items:
                        yield item_str
                        yielded_items.add(item_str)

        except (ValueError, TypeError, KeyError):
            # Character-by-character parsing
            parser_state = JsonArrayParserState()
            for char in accumulated:
                parser_state.update(char)
                if parser_state.is_element_separator and parser_state.current_item:
                    item = "".join(parser_state.current_item).strip()
                    if item not in yielded_items and item:
                        yield item
                        yielded_items.add(item)
                    parser_state.current_item = []

            # yield anything that is left
            if parser_state.current_item:
                item = "".join(parser_state.current_item).strip()
                if item not in yielded_items:
                    yield item
                    yielded_items.add(item)


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


# TODO: Add close method to close the underlying stream if chunks is a stream
# TODO: Make it a context manager to automatically close
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
