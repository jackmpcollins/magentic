from typing import AsyncIterator

import pytest

from magentic import AsyncStreamedStr, StreamedStr
from magentic.streaming import (
    aiter_streamed_json_array,
    async_iter,
    iter_streamed_json_array,
)


@pytest.mark.asyncio
async def test_async_iter():
    output = async_iter(["Hello", " World"])
    assert isinstance(output, AsyncIterator)
    assert [chunk async for chunk in output] == ["Hello", " World"]


iter_streamed_json_array_test_cases = [
    (["[]"], []),
    (['["He', 'llo", ', '"Wo', 'rld"]'], ['"Hello"', '"World"']),
    (["[1, 2, 3]"], ["1", "2", "3"]),
    (["[1, ", "2, 3]"], ["1", "2", "3"]),
    (['[{"a": 1}, {2: "b"}]'], ['{"a": 1}', '{2: "b"}']),
    (["{\n", '"value', '":', " [", "1, ", "2, 3", "]"], ["1", "2", "3"]),
]


@pytest.mark.parametrize(("input", "expected"), iter_streamed_json_array_test_cases)
def test_iter_streamed_json_array(input, expected):
    assert list(iter_streamed_json_array(iter(input))) == expected


@pytest.mark.parametrize(("input", "expected"), iter_streamed_json_array_test_cases)
@pytest.mark.asyncio
async def test_aiter_streamed_json_array(input, expected):
    assert [x async for x in aiter_streamed_json_array(async_iter(input))] == expected


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
    aiter_chunks = async_iter(["Hello", " World"])
    async_streamed_str = AsyncStreamedStr(aiter_chunks)
    assert [chunk async for chunk in async_streamed_str] == ["Hello", " World"]
    assert [chunk async for chunk in aiter_chunks] == []  # iterator is exhausted
    assert [chunk async for chunk in async_streamed_str] == ["Hello", " World"]


@pytest.mark.asyncio
async def test_async_streamed_str_to_string():
    async_streamed_str = AsyncStreamedStr(async_iter(["Hello", " World"]))
    assert await async_streamed_str.to_string() == "Hello World"
