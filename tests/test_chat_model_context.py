import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from magentic.backend import get_chat_model
from magentic.chat_model.base import _chat_model_context
from magentic.chat_model.openai_chat_model import OpenaiChatModel


@pytest.fixture
def model():
    token = _chat_model_context.set(None)
    try:
        yield OpenaiChatModel("gpt-4", api_key="test")
    finally:
        _chat_model_context.reset(token)


def test_reenter_same_model(model):
    previous = _chat_model_context.get()
    with model:
        with model:
            assert get_chat_model() is model
        assert get_chat_model() is model
    assert _chat_model_context.get() is previous


def test_reenter_model_inside_other_model(model):
    previous = _chat_model_context.get()
    other = OpenaiChatModel("gpt-5", api_key="test")
    with model:
        with other:
            with model:
                assert get_chat_model() is model
            assert get_chat_model() is other
        assert get_chat_model() is model
    assert _chat_model_context.get() is previous


def test_reenter_model_exception(model):
    previous = _chat_model_context.get()
    message = "inner failure"
    with model:
        with pytest.raises(ValueError, match=message), model:
            raise ValueError(message)
        assert get_chat_model() is model
    assert _chat_model_context.get() is previous


@pytest.mark.parametrize("first_exits_first", [True, False])
async def test_shared_model_async_tasks(model, first_exits_first):
    previous = _chat_model_context.get()
    entered = [asyncio.Event(), asyncio.Event()]
    leave = [asyncio.Event(), asyncio.Event()]

    async def worker(index: int) -> None:
        before = _chat_model_context.get()
        with model:
            entered[index].set()
            await leave[index].wait()
            assert get_chat_model() is model
        assert _chat_model_context.get() is before

    tasks = []
    try:
        for index in range(2):
            tasks.append(asyncio.create_task(worker(index)))
            await entered[index].wait()
        order = [0, 1] if first_exits_first else [1, 0]
        for index in order:
            leave[index].set()
            await tasks[index]
    finally:
        for event in leave:
            event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert _chat_model_context.get() is previous


async def test_shared_model_inherited_context(model):
    previous = _chat_model_context.get()

    async def worker() -> None:
        assert get_chat_model() is model
        with model:
            await asyncio.sleep(0)
            assert get_chat_model() is model
        assert get_chat_model() is model

    with model:
        await asyncio.create_task(worker())
        assert get_chat_model() is model
    assert _chat_model_context.get() is previous


async def test_shared_model_cancelled_task(model):
    previous = _chat_model_context.get()
    entered = asyncio.Event()

    async def worker() -> None:
        with model:
            entered.set()
            await asyncio.Event().wait()

    with model:
        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert get_chat_model() is model
    assert _chat_model_context.get() is previous


def test_shared_model_threads(model):
    previous = _chat_model_context.get()
    entered = [Event(), Event()]
    leave = [Event(), Event()]

    def worker(index: int) -> None:
        before = _chat_model_context.get()
        with model:
            entered[index].set()
            assert leave[index].wait(timeout=5)
            assert get_chat_model() is model
        assert _chat_model_context.get() is before

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, i) for i in range(2)]
        try:
            assert all(event.wait(timeout=5) for event in entered)
            leave[0].set()
            futures[0].result(timeout=5)
        finally:
            for event in leave:
                event.set()
            for future in futures:
                future.result(timeout=5)
    assert _chat_model_context.get() is previous


def test_reuse_model_sequentially(model):
    previous = _chat_model_context.get()
    for _ in range(3):
        with model:
            assert get_chat_model() is model
        assert _chat_model_context.get() is previous


def test_distinct_models_restore_outer_context(model):
    previous = _chat_model_context.get()
    other = OpenaiChatModel("gpt-5", api_key="test")
    with model:
        with other:
            assert get_chat_model() is other
        assert get_chat_model() is model
    assert _chat_model_context.get() is previous
