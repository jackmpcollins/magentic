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


def _strip_think_tags(text: str) -> str:
    """Strip MiniMax thinking tags from model output."""
    return _THINK_TAG_RE.sub("", text)


class MiniMaxStreamParser(OpenaiStreamParser):
    """Stream parser that strips MiniMax thinking tags from content.

    MiniMax models may emit ``<think>...</think>`` blocks before the actual
    response.  These must be hidden from both ``is_content`` (so the
    ``OutputStream`` does not create a ``StreamedStr`` for them) and
    ``get_content`` (so think-tag text never leaks into the output).

    When *tools_expected* is True, all content is skipped so that only
    tool-call chunks are processed.  MiniMax often emits explanatory text
    before calling tools, and that text must not be treated as content.
    """

    def __init__(self, *, tools_expected: bool = False) -> None:
        super().__init__()
        self._in_think_block: bool = False
        self._tools_expected: bool = tools_expected

    def is_content(self, item: ChatCompletionChunk) -> bool:
        if not (item.choices and item.choices[0].delta.content):
            return False
        # When tools are expected, skip all text content – it is just
        # pre-tool-call commentary produced by the model.
        if self._tools_expected:
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
        """Create the tool choice argument."""
        if contains_string_type(output_types):
            return openai.omit
        if len(tool_schemas) == 1:
            return tool_schemas[0].as_tool_choice()
        return "required"

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

        response: Iterator[ChatCompletionChunk] = self._client.chat.completions.create(
            model=self.model,
            messages=_add_missing_tool_calls_responses(
                [message_to_openai_message(m) for m in messages]
            ),
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
        stream = OutputStream(
            response,
            function_schemas=function_schemas,
            parser=MiniMaxStreamParser(tools_expected=bool(tool_schemas)),
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

        response: AsyncIterator[
            ChatCompletionChunk
        ] = await self._async_client.chat.completions.create(
            model=self.model,
            messages=_add_missing_tool_calls_responses(
                [await async_message_to_openai_message(m) for m in messages]
            ),
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
        stream = AsyncOutputStream(
            response,
            function_schemas=function_schemas,
            parser=MiniMaxStreamParser(tools_expected=bool(tool_schemas)),
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
