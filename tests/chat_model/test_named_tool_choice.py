from collections.abc import Iterable
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel, ValidationError

from magentic import Chat, chatprompt, prompt
from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from magentic.chat_model.base import ChatModel, OutputT, ToolSchemaParseError
from magentic.chat_model.litellm_chat_model import LitellmChatModel
from magentic.chat_model.message import AssistantMessage, Message, UserMessage
from magentic.chat_model.mistral_chat_model import MistralChatModel
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.chat_model.openrouter_chat_model import OpenRouterChatModel
from magentic.chat_model.retry_chat_model import RetryChatModel
from magentic.function_call import FunctionCall


def first() -> int:
    return 1


def second() -> int:
    return 2


class Answer(BaseModel):
    value: int


class RequestCapturedError(Exception):
    pass


@pytest.fixture(params=["openai", "anthropic", "litellm", "openrouter"])
def backend(request, monkeypatch):
    name = request.param
    model: Any
    if name == "openai":
        model = OpenaiChatModel("test", api_key="test")
    elif name == "anthropic":
        model = AnthropicChatModel("test", api_key="test")
    elif name == "openrouter":
        model = OpenRouterChatModel("test", api_key="test")
    else:
        model = LitellmChatModel("test")
    sync = Mock(side_effect=RequestCapturedError)
    async_ = AsyncMock(side_effect=RequestCapturedError)
    if name == "anthropic":
        # Anthropic's async stream factory itself is synchronous.
        async_ = Mock(side_effect=RequestCapturedError)
        monkeypatch.setattr(model._client.messages, "stream", sync)
        monkeypatch.setattr(model._async_client.messages, "stream", async_)
    elif name == "litellm":
        monkeypatch.setattr(
            "magentic.chat_model.litellm_chat_model.litellm.completion", sync
        )
        monkeypatch.setattr(
            "magentic.chat_model.litellm_chat_model.litellm.acompletion", async_
        )
    else:
        inner = model if name == "openai" else model._openrouter_openai_chat_model
        monkeypatch.setattr(inner._client.chat.completions, "create", sync)
        monkeypatch.setattr(inner._async_client.chat.completions, "create", async_)
    return name, model, sync, async_


@pytest.mark.parametrize("choice", ["second", "return_answer"])
@pytest.mark.asyncio
async def test_named_request_sync_async(backend, choice):
    name, model, sync, async_ = backend
    for method, factory in [
        (model.complete_with_tool_choice, sync),
        (model.acomplete_with_tool_choice, async_),
    ]:
        with pytest.raises(RequestCapturedError):  # noqa: PT012
            result = method(
                [UserMessage("choose")],
                functions=iter([first, second]),
                output_types=iter([str, FunctionCall, Answer]),
                tool_choice=choice,
            )
            if method == model.acomplete_with_tool_choice:
                await result
        kwargs = factory.call_args.kwargs
        expected = (
            {"type": "tool", "name": choice}
            if name == "anthropic"
            else {"type": "function", "function": {"name": choice}}
        )
        assert kwargs["tool_choice"] == expected
        tools = kwargs["tools"]
        assert len(tools) == 3  # forcing does not remove the other advertised tools
        assert kwargs["messages"]


@pytest.mark.parametrize("choice", ["", "absent", "return_str"])
@pytest.mark.asyncio
async def test_invalid_choice_never_requests(backend, choice):
    _, model, sync, async_ = backend
    with pytest.raises(ValueError, match="exactly one advertised tool"):
        model.complete_with_tool_choice(
            [], functions=[first, second], output_types=[str], tool_choice=choice
        )
    with pytest.raises(ValueError, match="exactly one advertised tool"):
        await model.acomplete_with_tool_choice(
            [], functions=[first, second], output_types=[str], tool_choice=choice
        )
    sync.assert_not_called()
    async_.assert_not_called()


@pytest.mark.asyncio
async def test_ambiguous_choice_never_requests(backend):
    _, model, sync, async_ = backend
    for method in [model.complete_with_tool_choice, model.acomplete_with_tool_choice]:
        with pytest.raises(ValueError, match="exactly one advertised tool"):  # noqa: PT012
            result = method([], functions=[first, first], tool_choice="first")
            if method == model.acomplete_with_tool_choice:
                await result
    sync.assert_not_called()
    async_.assert_not_called()


@pytest.mark.asyncio
async def test_mistral_explicit_unsupported():
    model = MistralChatModel("test", api_key="test")
    with pytest.raises(NotImplementedError, match="Named tool_choice"):
        model.complete_with_tool_choice([], functions=[first], tool_choice="first")
    with pytest.raises(NotImplementedError, match="Named tool_choice"):
        await model.acomplete_with_tool_choice(
            [], functions=[first], tool_choice="first"
        )


class OldCustomModel(ChatModel):
    # This pre-feature custom model intentionally has no tool_choice keyword.
    def complete(self, messages, functions=None, output_types=None, *, stop=None):
        return AssistantMessage("old")

    async def acomplete(
        self, messages, functions=None, output_types=None, *, stop=None
    ):
        return AssistantMessage("old")


class RecordingModel(ChatModel):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AssistantMessage[Any]:
        self.calls.append(kwargs)
        return AssistantMessage("done")

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AssistantMessage[Any]:
        return self.complete(messages, functions, output_types, stop=stop, **kwargs)

    def complete_with_tool_choice(
        self, messages, functions=None, output_types=None, *, stop=None, tool_choice
    ):
        return self.complete(
            messages, functions, output_types, stop=stop, tool_choice=tool_choice
        )

    async def acomplete_with_tool_choice(
        self, messages, functions=None, output_types=None, *, stop=None, tool_choice
    ):
        return self.complete(
            messages, functions, output_types, stop=stop, tool_choice=tool_choice
        )


@pytest.mark.asyncio
async def test_chat_selection_is_one_submission_only():
    model = RecordingModel()
    chat = Chat([UserMessage("hello")], model=model)
    chat = chat.submit(tool_choice="second").submit()
    chat = await chat.asubmit(tool_choice="first")
    await chat.asubmit()
    assert model.calls == [{"tool_choice": "second"}, {}, {"tool_choice": "first"}, {}]


@pytest.mark.parametrize("decorator", [prompt, chatprompt])
@pytest.mark.asyncio
async def test_decorators_propagate_sync_async(decorator: Any) -> None:
    model = RecordingModel()
    sync_decorator = (
        prompt("hello", model=model, tool_choice="second", max_retries=1)
        if decorator is prompt
        else chatprompt(
            UserMessage("hello"), model=model, tool_choice="second", max_retries=1
        )
    )
    async_decorator = (
        prompt("hello", model=model, tool_choice="first", max_retries=1)
        if decorator is prompt
        else chatprompt(
            UserMessage("hello"), model=model, tool_choice="first", max_retries=1
        )
    )

    def sync_function() -> str: ...

    sync = sync_decorator(sync_function)

    async def async_function() -> str: ...

    async_ = async_decorator(async_function)

    assert sync() == "done"
    assert await async_() == "done"
    assert model.calls == [{"tool_choice": "second"}, {"tool_choice": "first"}]


@pytest.mark.asyncio
async def test_old_custom_default_compatibility() -> None:
    model = OldCustomModel()
    assert Chat(model=model).submit().last_message.content == "old"
    assert (await Chat(model=model).asubmit()).last_message.content == "old"
    retry = RetryChatModel(model, max_retries=1)
    assert retry.complete([]).content == "old"
    assert (await retry.acomplete([])).content == "old"
    for decorator in (prompt, chatprompt):
        decorate = (
            prompt("hello", model=model, max_retries=1)
            if decorator is prompt
            else chatprompt(UserMessage("hello"), model=model, max_retries=1)
        )

        def sync_function() -> str: ...

        sync = decorate(sync_function)

        async def async_function() -> str: ...

        async_ = decorate(async_function)

        assert sync() == "old"
        assert await async_() == "old"


@pytest.mark.asyncio
async def test_retry_every_attempt_preserves_selection(monkeypatch):
    model = RecordingModel()
    error = ToolSchemaParseError(
        AssistantMessage("bad"), "id", ValidationError.from_exception_data("Answer", [])
    )
    monkeypatch.setattr(RetryChatModel, "_make_retry_messages", lambda *args: [])
    for async_ in [False, True]:
        calls: list[dict[str, Any]] = []

        def complete(
            *args: Any, calls: list[dict[str, Any]] = calls, **kwargs: Any
        ) -> AssistantMessage[str]:
            calls.append(kwargs)
            if len(calls) == 1:
                raise error
            return AssistantMessage("done")

        async def acomplete(*args, **kwargs):
            return complete(*args, **kwargs)

        monkeypatch.setattr(model, "complete", complete)
        monkeypatch.setattr(model, "acomplete", acomplete)
        retry = RetryChatModel(model, max_retries=1)
        if async_:
            assert (
                await retry.acomplete_with_tool_choice([], tool_choice="first")
            ).content == "done"
        else:
            assert (
                retry.complete_with_tool_choice([], tool_choice="first").content
                == "done"
            )
        assert len(calls) == 2
        assert all(c["tool_choice"] == "first" for c in calls)


@pytest.mark.asyncio
async def test_selected_function_requires_allowed_output(backend):
    _, model, sync, async_ = backend
    with pytest.raises(ValueError, match="does not allow function calls"):
        model.complete_with_tool_choice(
            [], functions=[first], output_types=[str], tool_choice="first"
        )
    with pytest.raises(ValueError, match="does not allow function calls"):
        await model.acomplete_with_tool_choice(
            [], functions=[first], output_types=[Answer], tool_choice="first"
        )
    sync.assert_not_called()
    async_.assert_not_called()


@pytest.mark.asyncio
async def test_function_output_name_collision(backend):
    _, model, sync, async_ = backend

    def return_answer() -> int:
        return 1

    with pytest.raises(ValueError, match="exactly one advertised tool"):
        model.complete_with_tool_choice(
            [],
            functions=[return_answer],
            output_types=[FunctionCall, Answer],
            tool_choice="return_answer",
        )
    with pytest.raises(ValueError, match="exactly one advertised tool"):
        await model.acomplete_with_tool_choice(
            [],
            functions=[return_answer],
            output_types=[FunctionCall, Answer],
            tool_choice="return_answer",
        )
    sync.assert_not_called()
    async_.assert_not_called()


@pytest.mark.asyncio
async def test_chat_default_output_rejects_named_function(backend):
    _, model, sync, async_ = backend
    chat = Chat(functions=[first], model=model)
    with pytest.raises(ValueError, match="does not allow function calls"):
        chat.submit(tool_choice="first")
    with pytest.raises(ValueError, match="does not allow function calls"):
        await chat.asubmit(tool_choice="first")
    sync.assert_not_called()
    async_.assert_not_called()


@pytest.mark.parametrize(
    "provider",
    [
        OpenaiChatModel,
        AnthropicChatModel,
        LitellmChatModel,
        OpenRouterChatModel,
        MistralChatModel,
        RetryChatModel,
    ],
)
@pytest.mark.asyncio
async def test_provider_derived_legacy_override_not_bypassed(provider):
    def init(self: Any) -> None:
        self.called = []

    def complete(self, messages, functions=None, output_types=None, *, stop=None):
        self.called.append("sync")
        return AssistantMessage("custom")

    async def acomplete(
        self, messages, functions=None, output_types=None, *, stop=None
    ):
        self.called.append("async")
        return AssistantMessage("custom")

    derived = type(
        "LegacyDerived",
        (provider,),
        {"__init__": init, "complete": complete, "acomplete": acomplete},
    )
    model: Any = derived()
    assert Chat(model=model).submit().last_message.content == "custom"
    assert (await Chat(model=model).asubmit()).last_message.content == "custom"
    with pytest.raises(
        NotImplementedError,
        match="custom complete override|current MistralChatModel adapter",
    ):
        Chat(model=model).submit(tool_choice="first")
    with pytest.raises(
        NotImplementedError,
        match="custom acomplete override|current MistralChatModel adapter",
    ):
        await Chat(model=model).asubmit(tool_choice="first")
    assert model.called == ["sync", "async"]


@pytest.mark.asyncio
async def test_base_legacy_explicit_selection_unsupported():
    model = OldCustomModel()
    with pytest.raises(NotImplementedError, match="does not support named"):
        Chat(model=model).submit(tool_choice="first")
    with pytest.raises(NotImplementedError, match="does not support named"):
        await Chat(model=model).asubmit(tool_choice="first")


@pytest.mark.asyncio
async def test_legacy_private_methods_not_intercepted(backend):
    _, original, sync, async_ = backend

    def collision(*args, **kwargs):
        msg = "Legacy private helper was incorrectly dispatched"
        raise AssertionError(msg)

    derived = type(
        "PrivateHelpers",
        (type(original),),
        {"_complete": collision, "_acomplete": collision},
    )
    original.__class__ = derived
    if hasattr(original, "_openrouter_openai_chat_model"):
        inner = original._openrouter_openai_chat_model
        inner.__class__ = type(
            "InnerPrivateHelpers",
            (type(inner),),
            {"_complete": collision, "_acomplete": collision},
        )
    with pytest.raises(RequestCapturedError):
        original.complete([], functions=[first], output_types=[FunctionCall])
    with pytest.raises(RequestCapturedError):
        await original.acomplete([], functions=[first], output_types=[FunctionCall])
    with pytest.raises(RequestCapturedError):
        original.complete_with_tool_choice(
            [], functions=[first], output_types=[FunctionCall], tool_choice="first"
        )
    with pytest.raises(RequestCapturedError):
        await original.acomplete_with_tool_choice(
            [], functions=[first], output_types=[FunctionCall], tool_choice="first"
        )
    assert sync.call_count == 2
    assert async_.call_count == 2


@pytest.mark.asyncio
async def test_retry_legacy_private_methods_not_intercepted():
    def collision(*args, **kwargs):
        msg = "Legacy private helper was incorrectly dispatched"
        raise AssertionError(msg)

    derived = type(
        "PrivateHelpers",
        (RetryChatModel,),
        {"_complete": collision, "_acomplete": collision},
    )
    underlying = RecordingModel()
    model = derived(underlying, max_retries=0)
    assert model.complete([]).content == "done"
    assert (await model.acomplete([])).content == "done"
    assert model.complete_with_tool_choice([], tool_choice="first").content == "done"
    assert (
        await model.acomplete_with_tool_choice([], tool_choice="first")
    ).content == "done"
