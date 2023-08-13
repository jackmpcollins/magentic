from typing import Iterator

from magentic import StreamedStr


def test_streamed_str_iter():
    def generator() -> Iterator[str]:
        yield from ["Hello", " World"]

    streamed_str = StreamedStr(generator())
    assert list(streamed_str) == ["Hello", " World"]
    assert list(streamed_str) == ["Hello", " World"]


def test_streamed_str_str():
    def generator() -> Iterator[str]:
        yield from ["Hello", " World"]

    streamed_str = StreamedStr(generator())
    assert str(streamed_str) == "Hello World"
