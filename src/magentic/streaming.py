from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass
from itertools import chain, dropwhile
from typing import AsyncIterator, Iterator, TypeVar

T = TypeVar("T")


async def async_iter(iterable: Iterable[T]) -> AsyncIterator[T]:
    """Get an AsyncIterator for an Iterable."""
    for item in iterable:
        yield item


async def achain(*aiterables: AsyncIterable[T]) -> AsyncIterator[T]:
    """Async version of `itertools.chain`."""
    for aiterable in aiterables:
        async for item in aiterable:
            yield item


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
