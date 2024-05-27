from typing import AsyncIterator

import pytest

from magentic import AsyncStreamedStr, StreamedStr
from magentic.streaming import (
    CachedAsyncIterable,
    CachedIterable,
    aapply,
    adropwhile,
    agroupby,
    aiter_streamed_json_array,
    apeek,
    apply,
    async_iter,
    atakewhile,
    azip,
    iter_streamed_json_array,
    peek,
)


@pytest.mark.asyncio
async def test_async_iter():
    output = async_iter(["Hello", " World"])
    assert isinstance(output, AsyncIterator)
    assert [chunk async for chunk in output] == ["Hello", " World"]


def test_apply():
    items: list[int] = []
    iterable = apply(items.append, range(3))
    assert list(iterable) == [0, 1, 2]
    assert items == [0, 1, 2]


@pytest.mark.asyncio
async def test_aapply():
    items: list[int] = []
    aiterable = aapply(items.append, async_iter(range(3)))
    assert [x async for x in aiterable] == [0, 1, 2]
    assert items == [0, 1, 2]


@pytest.mark.parametrize(
    ("aiterable", "expected"),
    [
        (azip(async_iter([1, 2, 3])), [(1,), (2,), (3,)]),
        (azip(async_iter([1, 2, 3]), async_iter([4, 5, 6])), [(1, 4), (2, 5), (3, 6)]),
    ],
)
@pytest.mark.asyncio
async def test_azip(aiterable, expected):
    assert [x async for x in aiterable] == expected


@pytest.mark.parametrize(
    ("iterator", "expected_first", "expected_remaining"),
    [
        (iter([1, 2, 3]), 1, [1, 2, 3]),
        (iter([1]), 1, [1]),
    ],
)
def test_peek(iterator, expected_first, expected_remaining):
    first, remaining = peek(iterator)
    assert first == expected_first
    assert list(remaining) == expected_remaining


@pytest.mark.parametrize(
    ("aiterator", "expected_first", "expected_remaining"),
    [
        (async_iter([1, 2, 3]), 1, [1, 2, 3]),
        (async_iter([1]), 1, [1]),
    ],
)
@pytest.mark.asyncio
async def test_apeek(aiterator, expected_first, expected_remaining):
    first, remaining = await apeek(aiterator)
    assert first == expected_first
    assert [x async for x in remaining] == expected_remaining


@pytest.mark.parametrize(
    ("predicate", "input", "expected"),
    [
        (lambda x: x < 3, async_iter(range(5)), [3, 4]),
        (lambda x: x < 6, async_iter(range(5)), []),
        (lambda x: x < 0, async_iter(range(5)), [0, 1, 2, 3, 4]),
    ],
)
@pytest.mark.asyncio
async def test_adropwhile(predicate, input, expected):
    assert [x async for x in adropwhile(predicate, input)] == expected


@pytest.mark.parametrize(
    ("predicate", "input", "expected"),
    [
        (lambda x: x < 3, async_iter(range(5)), [0, 1, 2]),
        (lambda x: x < 6, async_iter(range(5)), [0, 1, 2, 3, 4]),
        (lambda x: x < 0, async_iter(range(5)), []),
    ],
)
@pytest.mark.asyncio
async def test_atakewhile(predicate, input, expected):
    assert [x async for x in atakewhile(predicate, input)] == expected


@pytest.mark.parametrize(
    ("aiterable", "key", "expected"),
    [
        (async_iter([1, 1]), lambda x: x, [(1, [1, 1])]),
        (async_iter([1, 1, 2]), lambda x: x, [(1, [1, 1]), (2, [2])]),
    ],
)
@pytest.mark.asyncio
async def test_agroupby(aiterable, key, expected):
    assert [
        (k, [x async for x in g]) async for k, g in agroupby(aiterable, key)
    ] == expected


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


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ([1, 2, 3], [1, 2, 3]),
        (iter([1, 2, 3]), [1, 2, 3]),
        (range(3), [0, 1, 2]),
    ],
)
def test_iter_cached_iterable(input, expected):
    cached_iterable = CachedIterable(input)
    assert list(cached_iterable) == list(expected)
    assert list(cached_iterable) == list(expected)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ([1, 2, 3], [1, 2, 3]),
        (iter([1, 2, 3]), [1, 2, 3]),
        (range(3), [0, 1, 2]),
    ],
)
@pytest.mark.asyncio
async def test_aiter_cached_async_iterable(input, expected):
    cached_aiterable = CachedAsyncIterable(async_iter(input))
    assert [x async for x in cached_aiterable] == list(expected)
    assert [x async for x in cached_aiterable] == list(expected)


def test_streamed_str_iter():
    iter_chunks = iter(["Hello", " World"])
    streamed_str = StreamedStr(iter_chunks)
    assert list(streamed_str) == ["Hello", " World"]
    assert list(iter_chunks) == []  # iterator is exhausted
    assert list(streamed_str) == ["Hello", " World"]


def test_streamed_str_str():
    streamed_str = StreamedStr(["Hello", " World"])
    assert str(streamed_str) == "Hello World"


def test_streamed_str_truncate():
    streamed_str = StreamedStr(["First", " Second", " Third"])
    assert streamed_str.truncate(length=12) == "First [...]"
    assert streamed_str.truncate(length=99) == "First Second Third"


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


@pytest.mark.asyncio
async def test_async_streamed_str_truncate():
    async_streamed_str = AsyncStreamedStr(async_iter(["First", " Second", " Third"]))
    assert await async_streamed_str.truncate(length=12) == "First [...]"
    assert await async_streamed_str.truncate(length=99) == "First Second Third"
