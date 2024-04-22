import os
from enum import Enum
from typing import Any, Callable, Iterable, Sequence, TypeVar, overload

import openai

from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import AssistantMessage, Message
from magentic.chat_model.openai_chat_model import (
    BaseFunctionToolSchema,
    OpenaiChatModel,
)


class _MistralToolChoice(Enum):
    AUTO = "auto"
    """default mode. Model decides if it uses the tool or not."""
    ANY = "any"
    """forces tool use."""
    NONE = "none"
    """prevents tool use."""


class _MistralOpenaiChatModel(OpenaiChatModel):
    """Modified OpenaiChatModel to be compatible with Mistral API."""

    @staticmethod
    def _get_tool_choice(  # type: ignore[override]
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        allow_string_output: bool,
    ) -> str | openai.NotGiven:
        """Create the tool choice argument.

        Mistral API has different options than the OpenAI API for `tool_choice`.
        See https://docs.mistral.ai/capabilities/function_calling/#tool_choice
        """
        return openai.NOT_GIVEN if allow_string_output else _MistralToolChoice.ANY.value


R = TypeVar("R")


class MistralChatModel(ChatModel):
    """An LLM chat model for the Mistral API.

    Currently this uses the `openai` Python package. This may change in the future.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ):
        self._mistral_openai_chat_model = _MistralOpenaiChatModel(
            model,
            api_key=api_key or os.getenv("MISTRAL_API_KEY"),
            base_url=base_url or "https://api.mistral.ai/v1/",
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
        )

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]: ...

    @overload
    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]: ...

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        """Request an LLM message."""
        return self._mistral_openai_chat_model.complete(
            messages=messages,
            functions=functions,
            output_types=output_types,
            stop=stop,
        )

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: None = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[str]: ...

    @overload
    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Any = ...,
        output_types: Iterable[type[R]] = ...,
        *,
        stop: list[str] | None = ...,
    ) -> AssistantMessage[R]: ...

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[R]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[R] | AssistantMessage[str]:
        """Async version of `complete`."""
        return await self._mistral_openai_chat_model.acomplete(
            messages=messages,
            functions=functions,
            output_types=output_types,
            stop=stop,
        )
