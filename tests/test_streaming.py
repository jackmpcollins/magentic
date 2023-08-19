from typing import AsyncIterator, Iterator

import pytest

from magentic import AsyncStreamedStr, StreamedStr
from magentic.streaming import aiter_streamed_json_array, iter_streamed_json_array

iter_streamed_json_array_test_cases = [
    (["[]"], []),
    (['["He', 'llo", ', '"Wo', 'rld"]'], ['"Hello"', '"World"']),
    (["[1, 2, 3]"], ["1", "2", "3"]),
    (["[1, ", "2, 3]"], ["1", "2", "3"]),
    (['[{"a": 1}, {2: "b"}]'], ['{"a": 1}', '{2: "b"}']),
]


@pytest.mark.parametrize(["input", "expected"], iter_streamed_json_array_test_cases)
def test_iter_streamed_json_array(input: list[str], expected: list[str]):
    assert list(iter_streamed_json_array(iter(input))) == expected


@pytest.mark.parametrize(["input", "expected"], iter_streamed_json_array_test_cases)
@pytest.mark.asyncio
async def test_aiter_streamed_json_array(input: list[str], expected: list[str]):
    async def generator() -> AsyncIterator[str]:
        for chunk in input:
            yield chunk

    assert [x async for x in aiter_streamed_json_array(generator())] == expected


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
