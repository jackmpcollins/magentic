from collections.abc import Callable, Iterable
from itertools import chain
from typing import Any, Sequence, TypeVar, cast, overload

from openai.types.chat import ChatCompletionToolChoiceOptionParam
from pydantic import ValidationError

from magentic.chat_model.base import (
    ChatModel,
    StructuredOutputError,
    avalidate_str_content,
    validate_str_content,
)
from magentic.chat_model.function_schema import (
    FunctionCallFunctionSchema,
    async_function_schema_for_type,
    function_schema_for_type,
)
from magentic.chat_model.message import (
    AssistantMessage,
    Message,
)
from magentic.chat_model.openai_chat_model import (
    STR_OR_FUNCTIONCALL_TYPE,
    AsyncFunctionToolSchema,
    BaseFunctionToolSchema,
    FunctionToolSchema,
    _aparse_streamed_tool_calls,
    _parse_streamed_tool_calls,
    discard_none_arguments,
    message_to_openai_message,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    ParallelFunctionCall,
)
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    achain,
    async_iter,
)
from magentic.typing import is_any_origin_subclass, is_origin_subclass

try:
    import litellm
    from litellm.types.utils import ModelResponse
except ImportError as error:
    msg = "To use LitellmChatModel you must install the `litellm` package using `pip install 'magentic[litellm]'`."
    raise ImportError(msg) from error


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
    ) -> ChatCompletionToolChoiceOptionParam | None:
        """Create the tool choice argument."""
        if allow_string_output:
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
            output_types = [] if functions else cast(list[type[R]], [str])

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [FunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        streamed_str_in_output_types = is_any_origin_subclass(output_types, StreamedStr)
        allow_string_output = str_in_output_types or streamed_str_in_output_types

        response = discard_none_arguments(litellm.completion)(
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            api_base=self.api_base,
            custom_llm_provider=self.custom_llm_provider,
            max_tokens=self.max_tokens,
            metadata=self.metadata,
            stop=stop,
            stream=True,
            temperature=self.temperature,
            tools=[schema.to_dict() for schema in tool_schemas] or None,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, allow_string_output=allow_string_output
            ),
        )
        assert not isinstance(response, ModelResponse)  # noqa: S101

        first_chunk = next(response)
        # Azure OpenAI sends a chunk with empty choices first
        if len(first_chunk.choices) == 0:
            first_chunk = next(response)
        if (
            first_chunk.choices[0].delta.content is None
            and first_chunk.choices[0].delta.tool_calls is None
        ):
            first_chunk = next(response)
        response = chain([first_chunk], response)

        # Check tool calls before content because both might be present
        if first_chunk.choices[0].delta.tool_calls is not None:
            try:
                tool_calls = _parse_streamed_tool_calls(response, tool_schemas)
                if is_any_origin_subclass(output_types, ParallelFunctionCall):
                    content = ParallelFunctionCall(tool_calls)
                    return AssistantMessage(content)  # type: ignore[return-value]
                # Take only the first tool_call, silently ignore extra chunks
                # TODO: Create generator here that raises error or warns if multiple tool_calls
                content = next(tool_calls)
                return AssistantMessage(content)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        if first_chunk.choices[0].delta.content is not None:
            streamed_str = StreamedStr(
                chunk.choices[0].delta.get("content", None)
                for chunk in response
                if chunk.choices[0].delta.get("content", None) is not None
            )
            str_content = validate_str_content(
                streamed_str,
                allow_string_output=allow_string_output,
                streamed=streamed_str_in_output_types,
            )
            return AssistantMessage(str_content)  # type: ignore[return-value]

        msg = f"Could not determine response type for first chunk: {first_chunk.model_dump_json()}"
        raise ValueError(msg)

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
            output_types = [] if functions else cast(list[type[R]], [str])

        function_schemas = [FunctionCallFunctionSchema(f) for f in functions or []] + [
            async_function_schema_for_type(type_)
            for type_ in output_types
            if not is_origin_subclass(type_, STR_OR_FUNCTIONCALL_TYPE)
        ]
        tool_schemas = [AsyncFunctionToolSchema(schema) for schema in function_schemas]

        str_in_output_types = is_any_origin_subclass(output_types, str)
        async_streamed_str_in_output_types = is_any_origin_subclass(
            output_types, AsyncStreamedStr
        )
        allow_string_output = str_in_output_types or async_streamed_str_in_output_types

        response = await discard_none_arguments(litellm.acompletion)(
            model=self.model,
            messages=[message_to_openai_message(m) for m in messages],
            api_base=self.api_base,
            custom_llm_provider=self.custom_llm_provider,
            max_tokens=self.max_tokens,
            metadata=self.metadata,
            stop=stop,
            stream=True,
            temperature=self.temperature,
            tools=[schema.to_dict() for schema in tool_schemas] or None,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, allow_string_output=allow_string_output
            ),
        )
        assert not isinstance(response, ModelResponse)  # noqa: S101

        first_chunk = await anext(response)
        # Azure OpenAI sends a chunk with empty choices first
        if len(first_chunk.choices) == 0:
            first_chunk = await anext(response)
        if (
            first_chunk.choices[0].delta.content is None
            and first_chunk.choices[0].delta.tool_calls is None
        ):
            first_chunk = await anext(response)
        response = achain(async_iter([first_chunk]), response)

        # Check tool calls before content because both might be present
        if first_chunk.choices[0].delta.tool_calls is not None:
            try:
                tool_calls = _aparse_streamed_tool_calls(response, tool_schemas)
                if is_any_origin_subclass(output_types, AsyncParallelFunctionCall):
                    content = AsyncParallelFunctionCall(tool_calls)
                    return AssistantMessage(content)  # type: ignore[return-value]
                # Take only the first tool_call, silently ignore extra chunks
                content = await anext(tool_calls)
                return AssistantMessage(content)  # type: ignore[return-value]
            except ValidationError as e:
                msg = (
                    "Failed to parse model output. You may need to update your prompt"
                    " to encourage the model to return a specific type."
                )
                raise StructuredOutputError(msg) from e

        if first_chunk.choices[0].delta.content is not None:
            async_streamed_str = AsyncStreamedStr(
                chunk.choices[0].delta.get("content", None)
                async for chunk in response
                if chunk.choices[0].delta.get("content", None) is not None
            )
            str_content = await avalidate_str_content(
                async_streamed_str,
                allow_string_output=allow_string_output,
                streamed=async_streamed_str_in_output_types,
            )
            return AssistantMessage(str_content)  # type: ignore[return-value]

        msg = f"Could not determine response type for first chunk: {first_chunk.model_dump_json()}"
        raise ValueError(msg)
