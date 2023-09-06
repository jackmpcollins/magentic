from typing import AsyncIterator

import pytest

from magentic import AsyncStreamedStr, StreamedStr
from magentic.streamed_str import async_iter


@pytest.mark.asyncio
async def test_async_iter():
    output = async_iter(["Hello", " World"])
    assert isinstance(output, AsyncIterator)
    assert [chunk async for chunk in output] == ["Hello", " World"]


def test_streamed_str_iter():
    iter_chunks = iter(["Hello", " World"])
    streamed_str = StreamedStr(iter_chunks)
    assert list(streamed_str) == ["Hello", " World"]
    assert list(iter_chunks) == []  # iterator is exhausted
    assert list(streamed_str) == ["Hello", " World"]


def test_streamed_str_str():
    streamed_str = StreamedStr(["Hello", " World"])
    assert str(streamed_str) == "Hello World"


@pytest.mark.asyncio
async def test_async_streamed_str_iter():
    async_streamed_str = AsyncStreamedStr(async_iter(["Hello", " World"]))
    assert [chunk async for chunk in async_streamed_str] == ["Hello", " World"]
    assert [chunk async for chunk in async_streamed_str] == ["Hello", " World"]


@pytest.mark.asyncio
async def test_async_streamed_str_to_string():
    async_streamed_str = AsyncStreamedStr(async_iter(["Hello", " World"]))
    assert await async_streamed_str.to_string() == "Hello World"
