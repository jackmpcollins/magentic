from collections.abc import Callable, Iterable
from typing import Any, TypeVar, overload

import logfire_api as logfire

from magentic.chat_model.base import (
    ChatModel,
    StructuredOutputError,
)
from magentic.chat_model.message import (
    AssistantMessage,
    Message,
    ToolResultMessage,
)

R = TypeVar("R")


class RetryChatModel(ChatModel):
    """Wraps another ChatModel to add LLM-assisted retries."""

    def __init__(
        self,
        chat_model: ChatModel,
        *,
        max_retries: int,
    ):
        self._chat_model = chat_model
        self._max_retries = max_retries

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
        with logfire.span(
            "LLM-assisted retries enabled. Max {max_retries}",
            max_retries=self._max_retries,
        ):
            messages = list(messages)
            num_retry = 0
            while True:
                try:
                    message = self._chat_model.complete(
                        messages=messages,
                        functions=functions,
                        output_types=output_types,
                        stop=stop,
                    )
                except StructuredOutputError as e:
                    if num_retry >= self._max_retries:
                        raise
                    messages.append(e.output_message)
                    messages.append(
                        ToolResultMessage(
                            content=str(e.validation_error), tool_call_id=e.tool_call_id
                        )
                    )
                else:
                    return message

                num_retry += 1
                # TODO: Also log using Python logger
                logfire.warn(
                    "Retrying Chat Completion. Attempt {num_retry}",
                    num_retry=num_retry,
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
    ) -> AssistantMessage[str] | AssistantMessage[R]:
        """Async version of `complete`."""
        with logfire.span(
            "LLM-assisted retries enabled. Max {max_retries}",
            max_retries=self._max_retries,
        ):
            messages = list(messages)
            num_retry = 0
            while True:
                try:
                    message = await self._chat_model.acomplete(
                        messages=messages,
                        functions=functions,
                        output_types=output_types,
                        stop=stop,
                    )
                except StructuredOutputError as e:
                    if num_retry >= self._max_retries:
                        raise
                    messages.append(e.output_message)
                    messages.append(
                        ToolResultMessage(
                            content=str(e.validation_error), tool_call_id=e.tool_call_id
                        )
                    )
                else:
                    return message

                num_retry += 1
                # TODO: Also log using Python logger
                logfire.warn(
                    "Retrying Chat Completion. Attempt {num_retry}",
                    num_retry=num_retry,
                )
