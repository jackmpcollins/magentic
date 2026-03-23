import os
import re
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from typing import Any, cast

import openai
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionStreamOptionsParam,
)

from magentic._parsing import contains_string_type
from magentic.chat_model.base import ChatModel, OutputT, aparse_stream, parse_stream
from magentic.chat_model.function_schema import (
    get_async_function_schemas,
    get_function_schemas,
)
from magentic.chat_model.message import AssistantMessage, Message
from magentic.chat_model.openai_chat_model import (
    BaseFunctionToolSchema,
    OpenaiChatModel,
    OpenaiStreamParser,
    OpenaiStreamState,
    _add_missing_tool_calls_responses,
    _if_given,
    async_message_to_openai_message,
    message_to_openai_message,
)
from magentic.chat_model.stream import AsyncOutputStream, OutputStream

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# MiniMax ignores tool_choice (named/required/auto all behave the same).
# A system message is required to force the model to call tools.
_TOOL_USE_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You MUST use the available tools to respond."
        " Do NOT answer with plain text. Always call a tool."
    ),
}


def _strip_think_tags(text: str) -> str:
    """Strip MiniMax thinking tags from model output."""
    return _THINK_TAG_RE.sub("", text)


def _filter_content_chunks(
    stream: Iterator[ChatCompletionChunk],
) -> Iterator[ChatCompletionChunk]:
    """Skip content-only chunks so ``OutputStream`` sees tool calls first.

    MiniMax M2.7 always emits ``<think>...</think>`` reasoning and sometimes
    explanatory text before producing tool-call deltas.  The base
    ``OutputStream.__stream__`` exits its main loop when the first chunk is
    neither ``is_content`` nor ``is_tool_call``.  By stripping content-only
    chunks here we guarantee the first chunk reaching ``OutputStream``
    carries a tool-call delta.
    """
    for chunk in stream:
        if (
            chunk.choices
            and chunk.choices[0].delta.content
            and not chunk.choices[0].delta.tool_calls
        ):
            continue
        yield chunk


async def _afilter_content_chunks(
    stream: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[ChatCompletionChunk]:
    """Async version of :func:`_filter_content_chunks`."""
    async for chunk in stream:
        if (
            chunk.choices
            and chunk.choices[0].delta.content
            and not chunk.choices[0].delta.tool_calls
        ):
            continue
        yield chunk


class MiniMaxStreamParser(OpenaiStreamParser):
    """Stream parser that strips MiniMax thinking tags from content.

    MiniMax models may emit ``<think>...</think>`` blocks before the actual
    response.  These must be hidden from both ``is_content`` (so the
    ``OutputStream`` does not create a ``StreamedStr`` for them) and
    ``get_content`` (so think-tag text never leaks into the output).
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_think_block: bool = False

    def is_content(self, item: ChatCompletionChunk) -> bool:
        if not (item.choices and item.choices[0].delta.content):
            return False
        content = item.choices[0].delta.content
        if "<think>" in content:
            self._in_think_block = True
        if self._in_think_block:
            if "</think>" in content:
                self._in_think_block = False
                # Only treat as content if there's text after the closing tag
                after = content.split("</think>", 1)[1]
                return bool(after.strip())
            return False
        return True

    def get_content(self, item: ChatCompletionChunk) -> str | None:
        if not (item.choices and item.choices[0].delta.content):
            return None
        content = item.choices[0].delta.content
        # Strip any remaining think tags from the chunk
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
            return content if content else None
        return content


class _MiniMaxOpenaiChatModel(OpenaiChatModel):
    """Modified OpenaiChatModel to be compatible with MiniMax API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = "https://api.minimax.io/v1",
        max_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ):
        # Clamp temperature to MiniMax's accepted range (0, 1]
        if temperature is not None:
            temperature = max(0.01, min(temperature, 1.0))

        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
        )

    def _get_stream_options(self) -> ChatCompletionStreamOptionsParam | openai.Omit:
        return openai.omit

    @staticmethod
    def _get_tool_choice(  # type: ignore[override]
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        output_types: Iterable[type],
    ) -> str | openai.Omit:
        """Create the tool choice argument.

        MiniMax ignores named tool choice and ``"required"``, so we always
        use ``"auto"`` (when tools are present and string output is not
        expected) and rely on a system-message prompt to steer the model
        toward calling a tool.
        """
        if contains_string_type(output_types):
            return openai.omit
        if tool_schemas:
            return "auto"
        return openai.omit

    def _get_parallel_tool_calls(
        self, *, tools_specified: bool, output_types: Iterable[type]
    ) -> bool | openai.Omit:
        return openai.omit

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Request an LLM message."""
        if output_types is None:
            output_types = cast("Iterable[type[OutputT]]", [] if functions else [str])

        function_schemas = get_function_schemas(functions, output_types)
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        openai_messages = _add_missing_tool_calls_responses(
            [message_to_openai_message(m) for m in messages]
        )
        # MiniMax ignores tool_choice; inject a system prompt to force tool use
        tools_only = bool(tool_schemas) and not contains_string_type(output_types)
        if tools_only:
            openai_messages = [_TOOL_USE_SYSTEM_MSG, *openai_messages]

        response: Iterator[ChatCompletionChunk] = self._client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=_if_given(self.max_tokens),
            seed=_if_given(self.seed),
            stop=_if_given(stop),
            stream=True,
            stream_options=self._get_stream_options(),
            temperature=_if_given(self.temperature),
            tools=[schema.to_dict() for schema in tool_schemas] or openai.omit,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, output_types=output_types
            ),
            parallel_tool_calls=self._get_parallel_tool_calls(
                tools_specified=bool(tool_schemas), output_types=output_types
            ),
        )
        # Strip pre-tool content so OutputStream sees tool-call chunks first
        if tools_only:
            response = _filter_content_chunks(response)
        stream = OutputStream(
            response,
            function_schemas=function_schemas,
            parser=MiniMaxStreamParser(),
            state=OpenaiStreamState(),
        )
        return AssistantMessage._with_usage(
            parse_stream(stream, output_types), usage_ref=stream.usage_ref
        )

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Async version of `complete`."""
        if output_types is None:
            output_types = [] if functions else cast("list[type[OutputT]]", [str])

        function_schemas = get_async_function_schemas(functions, output_types)
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        openai_messages = _add_missing_tool_calls_responses(
            [await async_message_to_openai_message(m) for m in messages]
        )
        tools_only = bool(tool_schemas) and not contains_string_type(output_types)
        if tools_only:
            openai_messages = [_TOOL_USE_SYSTEM_MSG, *openai_messages]

        response: AsyncIterator[
            ChatCompletionChunk
        ] = await self._async_client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=_if_given(self.max_tokens),
            seed=_if_given(self.seed),
            stop=_if_given(stop),
            stream=True,
            stream_options=self._get_stream_options(),
            temperature=_if_given(self.temperature),
            tools=[schema.to_dict() for schema in tool_schemas] or openai.omit,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, output_types=output_types
            ),
            parallel_tool_calls=self._get_parallel_tool_calls(
                tools_specified=bool(tool_schemas), output_types=output_types
            ),
        )
        if tools_only:
            response = _afilter_content_chunks(response)
        stream = AsyncOutputStream(
            response,
            function_schemas=function_schemas,
            parser=MiniMaxStreamParser(),
            state=OpenaiStreamState(),
        )
        return AssistantMessage._with_usage(
            await aparse_stream(stream, output_types), usage_ref=stream.usage_ref
        )


class MiniMaxChatModel(ChatModel):
    """An LLM chat model for the MiniMax API.

    Currently this uses the ``openai`` Python package. MiniMax provides an
    OpenAI-compatible API at ``https://api.minimax.io/v1``. Available models
    include ``MiniMax-M2.7`` and ``MiniMax-M2.5-highspeed``.

    Temperature is automatically clamped to the MiniMax-accepted range of (0, 1].
    Thinking tags (``<think>...</think>``) produced by reasoning models are stripped
    from streamed output so they do not appear in the final response.
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
        if not (api_key or os.getenv("MINIMAX_API_KEY")):
            msg = (
                "MINIMAX_API_KEY environment variable or api_key argument is required."
            )
            raise openai.OpenAIError(msg)
        self._minimax_openai_chat_model = _MiniMaxOpenaiChatModel(
            model,
            api_key=api_key or os.getenv("MINIMAX_API_KEY"),
            base_url=base_url or "https://api.minimax.io/v1",
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
        )

    @property
    def model(self) -> str:
        return self._minimax_openai_chat_model.model

    @property
    def api_key(self) -> str | None:
        return self._minimax_openai_chat_model.api_key

    @property
    def base_url(self) -> str | None:
        return self._minimax_openai_chat_model.base_url

    @property
    def max_tokens(self) -> int | None:
        return self._minimax_openai_chat_model.max_tokens

    @property
    def seed(self) -> int | None:
        return self._minimax_openai_chat_model.seed

    @property
    def temperature(self) -> float | None:
        return self._minimax_openai_chat_model.temperature

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Request an LLM message."""
        return self._minimax_openai_chat_model.complete(
            messages=messages,
            functions=functions,
            output_types=output_types,
            stop=stop,
        )

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Async version of `complete`."""
        return await self._minimax_openai_chat_model.acomplete(
            messages=messages,
            functions=functions,
            output_types=output_types,
            stop=stop,
        )
