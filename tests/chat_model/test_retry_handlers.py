import asyncio
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NoReturn, cast

import pytest
from pydantic import BaseModel, ValidationError

from magentic.chat_model.base import (
    ChatModel,
    OutputT,
    ToolSchemaParseError,
    UnknownToolError,
)
from magentic.chat_model.message import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from magentic.chat_model.retry_chat_model import RetryChatModel


class FakeModel(ChatModel):
    def __init__(self, outcomes: Iterable[str | BaseException]) -> None:
        self.outcomes = iter(outcomes)
        self.calls: list[list[Message[Any]]] = []

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        self.calls.append(list(messages))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        # This deterministic test double supplies preselected string outputs.
        return AssistantMessage(cast(OutputT, outcome))

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        return self.complete(messages, functions, output_types, stop=stop)


async def run(model: RetryChatModel, *, asynchronous: bool) -> AssistantMessage[str]:
    if asynchronous:
        return await model.acomplete([UserMessage("original")])
    return model.complete([UserMessage("original")])


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_registered_handler_sync_async(*, asynchronous: bool) -> None:
    error = UnknownToolError(AssistantMessage("bad"), "id", "missing")
    fake = FakeModel([error, "ok"])
    model = RetryChatModel(fake, max_retries=1)
    calls: list[Any] = []

    @model.register_handler(UnknownToolError)
    def handler(self: RetryChatModel, exc: UnknownToolError) -> Sequence[Message[Any]]:
        calls.append((self, exc))
        return [exc.output_message, ToolResultMessage(str(exc), exc.tool_call_id)]

    assert (await run(model, asynchronous=asynchronous)).content == "ok"
    assert calls == [(model, error)]
    assert fake.calls[1][0] == UserMessage("original")
    assert fake.calls[1][1] is error.output_message
    assert isinstance(fake.calls[1][2], ToolResultMessage)
    assert fake.calls[1][2].tool_call_id == "id"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unregistered_exception_identity(*, asynchronous: bool) -> None:
    error = ValueError("unregistered")
    model = RetryChatModel(FakeModel([error]), max_retries=3)
    with pytest.raises(ValueError, match=".*") as caught:
        await run(model, asynchronous=asynchronous)
    assert caught.value is error


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("budget", [0, 2])
async def test_budget_and_final_error(*, asynchronous: bool, budget: int) -> None:
    errors = [ValueError(str(i)) for i in range(budget + 1)]
    fake = FakeModel(errors)
    model = RetryChatModel(fake, max_retries=budget)
    calls: list[Any] = []

    @model.register_handler(ValueError)
    def handler(self: RetryChatModel, exc: ValueError) -> Sequence[Message[Any]]:
        calls.append(exc)
        return [UserMessage("retry")]

    with pytest.raises(ValueError, match=".*") as caught:
        await run(model, asynchronous=asynchronous)
    assert caught.value is errors[-1]
    assert calls == errors[:-1]
    assert len(fake.calls) == budget + 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_handler_failure_propagates_without_retry(*, asynchronous: bool) -> None:
    fake = FakeModel([ValueError("original"), "unreached"])
    model = RetryChatModel(fake, max_retries=2)
    failure = RuntimeError("handler failed")

    @model.register_handler(ValueError)
    def handler(self: RetryChatModel, exc: ValueError) -> NoReturn:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        await run(model, asynchronous=asynchronous)
    assert caught.value is failure
    assert len(fake.calls) == 1


async def test_specific_dispatch_override_and_instance_isolation() -> None:
    class ChildError(ValueError):
        pass

    first = RetryChatModel(FakeModel([ChildError(), ChildError(), "ok"]), max_retries=2)
    second_error = ChildError("second")
    second = RetryChatModel(FakeModel([second_error]), max_retries=1)
    seen: list[Any] = []

    @first.register_handler(ValueError)
    def parent(self: RetryChatModel, exc: ValueError) -> Sequence[Message[Any]]:
        seen.append("parent")
        return []

    @first.register_handler(ChildError)
    def child(self: RetryChatModel, exc: ChildError) -> Sequence[Message[Any]]:
        seen.append("child")
        return []

    @first.register_handler(ChildError)
    def replacement(self: RetryChatModel, exc: ChildError) -> Sequence[Message[Any]]:
        seen.append("replacement")
        return []

    assert (await run(first, asynchronous=False)).content == "ok"
    assert seen == ["replacement", "replacement"]
    with pytest.raises(ChildError) as caught:
        await run(second, asynchronous=True)
    assert caught.value is second_error


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_default_parse_handler_preserved(*, asynchronous: bool) -> None:
    class Number(BaseModel):
        value: int

    try:
        Number.model_validate({"value": "bad"})
    except ValidationError as error:
        parse_error = ToolSchemaParseError(AssistantMessage("bad"), "call-id", error)
    fake = FakeModel([parse_error, "ok"])
    model = RetryChatModel(fake, max_retries=1)
    assert (await run(model, asynchronous=asynchronous)).content == "ok"
    assert fake.calls[1][1] is parse_error.output_message
    assert fake.calls[1][2] == ToolResultMessage(
        str(parse_error.validation_error), "call-id"
    )


async def test_parse_handler_override_is_local() -> None:
    model = RetryChatModel(FakeModel([]), max_retries=1)
    original = RetryChatModel(FakeModel([]), max_retries=1)

    @model.register_handler(ToolSchemaParseError)
    def handler(
        self: RetryChatModel, exc: ToolSchemaParseError
    ) -> Sequence[Message[Any]]:
        return [UserMessage("custom")]

    assert model._exception_handlers.dispatch(
        ToolSchemaParseError
    ) != original._exception_handlers.dispatch(ToolSchemaParseError)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_base_exception_and_cancellation_not_caught(
    *, asynchronous: bool
) -> None:
    error = asyncio.CancelledError()
    model = RetryChatModel(FakeModel([error]), max_retries=2)

    @model.register_handler(Exception)
    def handler(self: RetryChatModel, exc: Exception) -> NoReturn:
        pytest.fail("must not catch cancellation")

    with pytest.raises(asyncio.CancelledError) as caught:
        await run(model, asynchronous=asynchronous)
    assert caught.value is error
    with pytest.raises(TypeError, match="Exception subclass"):
        model.register_handler(cast(type[Exception], BaseException))


def test_exception_exports_are_aliases() -> None:
    from magentic import exceptions

    assert exceptions.ToolSchemaParseError is ToolSchemaParseError
    assert exceptions.UnknownToolError is UnknownToolError


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_parent_registration_dispatches_subclass(*, asynchronous: bool) -> None:
    class ChildError(ValueError):
        pass

    error = ChildError("child")
    model = RetryChatModel(FakeModel([error, "ok"]), max_retries=1)
    seen: list[Any] = []

    @model.register_handler(ValueError)
    def handler(self: RetryChatModel, exc: ValueError) -> Sequence[Message[Any]]:
        seen.append(exc)
        return []

    assert (await run(model, asynchronous=asynchronous)).content == "ok"
    assert seen == [error]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_registered_handler_error_is_not_retried(*, asynchronous: bool) -> None:
    fake = FakeModel([ValueError("completion"), "unreachable"])
    model = RetryChatModel(fake, max_retries=2)
    failure = ValueError("handler")

    @model.register_handler(ValueError)
    def handler(self: RetryChatModel, exc: ValueError) -> NoReturn:
        raise failure

    with pytest.raises(ValueError, match="handler") as caught:
        await run(model, asynchronous=asynchronous)
    assert caught.value is failure
    assert len(fake.calls) == 1


def test_decorated_prompt_uses_registered_completion_handler() -> None:
    from magentic import prompt

    fake = FakeModel([ValueError("invalid"), "ok"])
    model = RetryChatModel(fake, max_retries=1)

    @model.register_handler(ValueError)
    def handler(self: RetryChatModel, exc: ValueError) -> Sequence[Message[Any]]:
        return [UserMessage("correct the output")]

    @prompt("hello", model=model)
    def ask() -> str: ...

    assert ask() == "ok"
    assert fake.calls[1][-1] == UserMessage("correct the output")
