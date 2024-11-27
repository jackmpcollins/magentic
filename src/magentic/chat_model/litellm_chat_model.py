from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeVar, cast, overload

import litellm
import openai
from litellm.litellm_core_utils.streaming_handler import StreamingChoices
from openai.lib.streaming.chat._completions import ChatCompletionStreamState

from magentic.chat_model.base import (
    ChatModel,
    aparse_stream,
    parse_stream,
)
from magentic.chat_model.function_schema import (
    FunctionCallFunctionSchema,
    async_function_schema_for_type,
    function_schema_for_type,
)
from magentic.chat_model.message import (
    AssistantMessage,
    Message,
    Usage,
    _RawMessage,
)
from magentic.chat_model.openai_chat_model import (
    STR_OR_FUNCTIONCALL_TYPE,
    BaseFunctionToolSchema,
    message_to_openai_message,
)
from magentic.chat_model.stream import (
    AsyncOutputStream,
    FunctionCallChunk,
    OutputStream,
    StreamParser,
    StreamState,
)
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
)
from magentic.typing import is_any_origin_subclass, is_origin_subclass

try:
    import litellm
    from litellm.types.utils import ModelResponse
except ImportError as error:
    msg = "To use LitellmChatModel you must install the `litellm` package using `pip install 'magentic[litellm]'`."
    raise ImportError(msg) from error


class LitellmStreamParser(StreamParser[ModelResponse]):
    def is_content(self, item: ModelResponse) -> bool:
        assert isinstance(item.choices[0], StreamingChoices)  # noqa: S101
        return bool(item.choices[0].delta.content)

    def is_content_ended(self, item: ModelResponse) -> bool:
        return self.is_tool_call(item)

    def get_content(self, item: ModelResponse) -> str | None:
        assert isinstance(item.choices[0], StreamingChoices)  # noqa: S101
        return item.choices[0].delta.content

    def is_tool_call(self, item: ModelResponse) -> bool:
        assert isinstance(item.choices[0], StreamingChoices)  # noqa: S101
        return bool(item.choices[0].delta.tool_calls)

    def iter_tool_calls(self, item: ModelResponse) -> Iterable[FunctionCallChunk]:
        assert isinstance(item.choices[0], StreamingChoices)  # noqa: S101
        if item.choices and item.choices[0].delta.tool_calls:
            for tool_call in item.choices[0].delta.tool_calls:
                if tool_call.function:
                    yield FunctionCallChunk(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        args=tool_call.function.arguments,
                    )


class LitellmStreamState(StreamState[ModelResponse]):
    def __init__(self):
        self._chat_completion_stream_state = ChatCompletionStreamState(
            input_tools=openai.NOT_GIVEN,
            response_format=openai.NOT_GIVEN,
        )
        self.usage_ref: list[Usage] = []

    def update(self, item: ModelResponse) -> None:
        # Patch attributes required inside ChatCompletionStreamState.handle_chunk
        if not hasattr(item, "usage"):
            # litellm requires usage is not None for its total usage calculation
            item.usage = litellm.Usage()  # type: ignore[attr-defined]
        if not hasattr(item, "refusal"):
            item.choices[0].delta.refusal = None  # type: ignore[attr-defined]
        self._chat_completion_stream_state.handle_chunk(item)  # type: ignore[arg-type]
        usage = cast(litellm.Usage, item.usage)  # type: ignore[attr-defined]
        # Ignore usages with 0 tokens
        if usage and usage.prompt_tokens and usage.completion_tokens:
            assert not self.usage_ref  # noqa: S101
            self.usage_ref.append(
                Usage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
            )

    @property
    def current_message_snapshot(self) -> Message:
        snapshot = self._chat_completion_stream_state.current_completion_snapshot
        message = snapshot.choices[0].message
        # Fix incorrectly concatenated role
        message.role = "assistant"
        # TODO: Possible to return AssistantMessage here?
        return _RawMessage(message.model_dump())


R = TypeVar("R")


class LitellmChatModel(ChatModel):
    """An LLM chat model that uses the `litellm` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        custom_llm_provider: str | None = None,
    ):
        self._model = model
        self._api_base = api_base
        self._max_tokens = max_tokens
        self._metadata = metadata
        self._temperature = temperature
        self._custom_llm_provider = custom_llm_provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_base(self) -> str | None:
        return self._api_base

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    @property
    def metadata(self) -> dict[str, Any] | None:
        return self._metadata

    @property
    def temperature(self) -> float | None:
        return self._temperature

    @property
    def custom_llm_provider(self) -> str | None:
        return self._custom_llm_provider

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        allow_string_output: bool,
    ) -> dict | Literal["none", "auto", "required"] | None:
        """Create the tool choice argument."""
        if allow_string_output:
            return None
        if len(tool_schemas) == 1:
            return tool_schemas[0].as_tool_choice()  # type: ignore[return-value]
        return "required"

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
        if output_types is None:
            output_types = cast(Iterable[type[R]], [] if functions else [str])

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        streamed_str_in_output_types = is_any_origin_subclass(output_types, StreamedStr)
        allow_string_output = str_in_output_types or streamed_str_in_output_types

        response = litellm.completion(
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            api_base=self.api_base,
            custom_llm_provider=self.custom_llm_provider,
            max_tokens=self.max_tokens,
            metadata=self.metadata,
            stop=stop,
            stream=True,
            # TODO: Add usage for LitellmChatModel
            temperature=self.temperature,
            tools=[schema.to_dict() for schema in tool_schemas] or None,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, allow_string_output=allow_string_output
            ),
        )
        assert not isinstance(response, ModelResponse)  # noqa: S101
        stream = OutputStream(
            stream=response,
            function_schemas=function_schemas,
            parser=LitellmStreamParser(),
            state=LitellmStreamState(),
        )
        return AssistantMessage(parse_stream(stream, output_types))

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
        if output_types is None:
            output_types = cast(Iterable[type[R]], [] if functions else [str])

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            async_function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        async_streamed_str_in_output_types = is_any_origin_subclass(
            output_types, AsyncStreamedStr
        )
        allow_string_output = str_in_output_types or async_streamed_str_in_output_types

        response = await litellm.acompletion(
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            api_base=self.api_base,
            custom_llm_provider=self.custom_llm_provider,
            max_tokens=self.max_tokens,
            metadata=self.metadata,
            stop=stop,
            stream=True,
            # TODO: Add usage for LitellmChatModel
            temperature=self.temperature,
            tools=[schema.to_dict() for schema in tool_schemas] or None,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, allow_string_output=allow_string_output
            ),  # type: ignore[arg-type]
        )
        assert not isinstance(response, ModelResponse)  # noqa: S101
        stream = AsyncOutputStream(
            stream=response,
            function_schemas=function_schemas,
            parser=LitellmStreamParser(),
            state=LitellmStreamState(),
        )
        return AssistantMessage(await aparse_stream(stream, output_types))
