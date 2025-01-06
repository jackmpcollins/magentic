from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TypeVar, cast, overload

import openai
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionNamedToolChoiceParam

from magentic._parsing import contains_string_type
from magentic.chat_model.base import ChatModel, aparse_stream, parse_stream
from magentic.chat_model.function_schema import (
    get_async_function_schemas,
    get_function_schemas,
)
from magentic.chat_model.message import AssistantMessage, Message, Usage, _RawMessage
from magentic.chat_model.openai_chat_model import (
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

try:
    import litellm
    from litellm.litellm_core_utils.streaming_handler import (  # type: ignore[attr-defined]
        StreamingChoices,
    )
    from litellm.types.utils import ModelResponse
except ImportError as error:
    msg = "To use LitellmChatModel you must install the `litellm` package using `pip install 'magentic[litellm]'`."
    raise ImportError(msg) from error


class LitellmStreamParser(StreamParser[ModelResponse]):
    def is_content(self, item: ModelResponse) -> bool:
        assert isinstance(item.choices[0], StreamingChoices)
        return bool(item.choices[0].delta.content)

    def get_content(self, item: ModelResponse) -> str | None:
        assert isinstance(item.choices[0], StreamingChoices)
        assert isinstance(item.choices[0].delta.content, str | None)
        return item.choices[0].delta.content

    def is_tool_call(self, item: ModelResponse) -> bool:
        assert isinstance(item.choices[0], StreamingChoices)
        return bool(item.choices[0].delta.tool_calls)

    def iter_tool_calls(self, item: ModelResponse) -> Iterable[FunctionCallChunk]:
        assert isinstance(item.choices[0], StreamingChoices)
        if item.choices and item.choices[0].delta.tool_calls:
            for tool_call in item.choices[0].delta.tool_calls:
                if tool_call.function:
                    yield FunctionCallChunk(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        args=tool_call.function.arguments,
                    )


class LitellmStreamState(StreamState[ModelResponse]):
    def __init__(self) -> None:
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
            assert isinstance(item.choices[0], StreamingChoices)
            item.choices[0].delta.refusal = None  # type: ignore[attr-defined]
        self._chat_completion_stream_state.handle_chunk(item)  # type: ignore[arg-type]
        usage = cast(litellm.Usage, item.usage)  # type: ignore[attr-defined,name-defined]
        # Ignore usages with 0 tokens
        if usage and usage.prompt_tokens and usage.completion_tokens:
            assert not self.usage_ref
            self.usage_ref.append(
                Usage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                )
            )

    @property
    def current_message_snapshot(self) -> Message[Any]:
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
        output_types: Iterable[type[R]],
    ) -> ChatCompletionNamedToolChoiceParam | Literal["required"] | None:
        """Create the tool choice argument."""
        if contains_string_type(output_types):
            return None
        if len(tool_schemas) == 1:
            return tool_schemas[0].as_tool_choice()
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

        function_schemas = get_function_schemas(functions, output_types)
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

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
                tool_schemas=tool_schemas, output_types=output_types
            ),  # type: ignore[arg-type,unused-ignore]
        )
        assert not isinstance(response, ModelResponse)
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

        function_schemas = get_async_function_schemas(functions, output_types)
        tool_schemas = [BaseFunctionToolSchema(schema) for schema in function_schemas]

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
                tool_schemas=tool_schemas, output_types=output_types
            ),  # type: ignore[arg-type,unused-ignore]
        )
        assert not isinstance(response, ModelResponse)
        stream = AsyncOutputStream(
            stream=response,
            function_schemas=function_schemas,
            parser=LitellmStreamParser(),
            state=LitellmStreamState(),
        )
        return AssistantMessage(await aparse_stream(stream, output_types))
