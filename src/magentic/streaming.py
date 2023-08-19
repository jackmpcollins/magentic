from itertools import chain
from typing import AsyncIterator, Iterator


def iter_streamed_json_array(generator: Iterator[str]) -> Iterator[str]:
    """Convert a stream of text chunks into a stream of objects.

    The text chunks must represent an array of objects.
    """
    iter_chars = chain.from_iterable(generator)

    first_char = next(iter_chars)
    if not first_char == "[":
        raise ValueError("Expected array")

    array_level = 1
    object_level = 0
    in_string = False
    is_escaped = False

    item_chars: list[str] = []
    for char in iter_chars:
        if in_string:
            if char == '"' and not is_escaped:
                in_string = False
        elif char == '"':
            in_string = True
        elif char == ",":
            if array_level == 1 and object_level == 0:
                yield "".join(item_chars).strip()
                item_chars = []
                continue
        elif char == "[":
            array_level += 1
        elif char == "]":
            array_level -= 1
            if array_level == 0:
                if item_chars:
                    yield "".join(item_chars).strip()
                return
        elif char == "{":
            object_level += 1
        elif char == "}":
            object_level -= 1

        item_chars.append(char)
        is_escaped = (char == "\\") and not is_escaped


class StreamedStr:
    """A string that is generated in chunks."""

    def __init__(self, generator: Iterator[str]):
        self._generator = generator
        self._cached_chunks: list[str] = []

    def __iter__(self) -> Iterator[str]:
        yield from self._cached_chunks
        for chunk in self._generator:
            self._cached_chunks.append(chunk)
            yield chunk

    def __str__(self) -> str:
        return "".join(self)

    def to_string(self) -> str:
        """Convert the streamed string to a string."""
        return str(self)


class AsyncStreamedStr:
    """Async version of `StreamedStr`."""

    def __init__(self, generator: AsyncIterator[str]):
        self._generator = generator
        self._cached_chunks: list[str] = []

    async def __aiter__(self) -> AsyncIterator[str]:
        # Cannot use `yield from` inside an async function
        # https://peps.python.org/pep-0525/#asynchronous-yield-from
        for chunk in self._cached_chunks:
            yield chunk
        async for chunk in self._generator:
            self._cached_chunks.append(chunk)
            yield chunk

    async def to_string(self) -> str:
        """Convert the streamed string to a string."""
        return "".join([item async for item in self])
