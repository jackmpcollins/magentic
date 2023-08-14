from typing import AsyncIterator, Iterator

import pytest

from magentic import AsyncStreamedStr, StreamedStr


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


@pytest.mark.asyncio
async def test_async_streamed_str_iter():
    async def generator() -> AsyncIterator[str]:
        for chunk in ["Hello", " World"]:
            yield chunk

    async_streamed_str = AsyncStreamedStr(generator())
    assert [chunk async for chunk in async_streamed_str] == ["Hello", " World"]
    assert [chunk async for chunk in async_streamed_str] == ["Hello", " World"]


@pytest.mark.asyncio
async def test_async_streamed_str_to_string():
    async def generator() -> AsyncIterator[str]:
        for chunk in ["Hello", " World"]:
            yield chunk

    async_streamed_str = AsyncStreamedStr(generator())
    assert await async_streamed_str.to_string() == "Hello World"
