from typing import Iterator


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
