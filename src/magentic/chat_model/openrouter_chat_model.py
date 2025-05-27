import os
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from typing import Any, Literal, cast

import openai
from openai.types.chat import ChatCompletionChunk, ChatCompletionStreamOptionsParam

from magentic._parsing import contains_parallel_function_call_type, contains_string_type
from magentic.chat_model.base import ChatModel, OutputT
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


class _OpenRouterOpenaiChatModel(OpenaiChatModel):
    """Modified OpenaiChatModel to be compatible with OpenRouter API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = "https://openrouter.ai/api/v1",
        max_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        route: Literal["fallback"] | None = None,
        models: list[str] | None = None,
        require_parameters: bool = True,
        reasoning: dict[str, Any] | None = None,
    ):
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
        )
        self._route = route
        self._models = models
        self._require_parameters = require_parameters
        self._reasoning = reasoning

    def _get_stream_options(self) -> ChatCompletionStreamOptionsParam | openai.NotGiven:
        return {"include_usage": True}

    @staticmethod
    def _get_tool_choice(
        *,
        tool_schemas: Sequence[BaseFunctionToolSchema[Any]],
        output_types: Iterable[type],
    ) -> str | openai.NotGiven:
        if contains_string_type(output_types):
            return openai.NOT_GIVEN
        if len(tool_schemas) == 1:
            return tool_schemas[0].as_tool_choice()
        return "required"

    def _get_parallel_tool_calls(
        self, *, tools_specified: bool, output_types: Iterable[type]
    ) -> bool | openai.NotGiven:
        if not tools_specified:
            return openai.NOT_GIVEN
        if contains_parallel_function_call_type(output_types):
            return openai.NOT_GIVEN
        return False

    def _get_extra_body(self) -> dict[str, Any] | None:
        """Get extra body parameters for OpenRouter API."""
        extra_body = {}
        if self._route:
            extra_body["route"] = self._route
        if self._models:
            extra_body["models"] = self._models
        if self._require_parameters:
            extra_body.setdefault("provider", {})["require_parameters"] = True
        if self._reasoning:
            extra_body["reasoning"] = self._reasoning
        return extra_body if extra_body else None

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Request an LLM message."""
        from magentic.chat_model.base import parse_stream
        from magentic.chat_model.function_schema import get_function_schemas
        from magentic.chat_model.stream import OutputStream

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
            tools=[schema.to_dict() for schema in tool_schemas] or openai.NOT_GIVEN,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, output_types=output_types
            ),
            parallel_tool_calls=self._get_parallel_tool_calls(
                tools_specified=bool(tool_schemas), output_types=output_types
            ),
            extra_body=self._get_extra_body(),
        )
        stream = OutputStream(
            response,
            function_schemas=function_schemas,
            parser=OpenaiStreamParser(),
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
        from magentic.chat_model.base import aparse_stream
        from magentic.chat_model.function_schema import get_async_function_schemas
        from magentic.chat_model.stream import AsyncOutputStream

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
            tools=[schema.to_dict() for schema in tool_schemas] or openai.NOT_GIVEN,
            tool_choice=self._get_tool_choice(
                tool_schemas=tool_schemas, output_types=output_types
            ),
            parallel_tool_calls=self._get_parallel_tool_calls(
                tools_specified=bool(tool_schemas), output_types=output_types
            ),
            extra_body=self._get_extra_body(),
        )
        stream = AsyncOutputStream(
            response,
            function_schemas=function_schemas,
            parser=OpenaiStreamParser(),
            state=OpenaiStreamState(),
        )
        return AssistantMessage._with_usage(
            await aparse_stream(stream, output_types), usage_ref=stream.usage_ref
        )


class OpenRouterChatModel(ChatModel):
    """An LLM chat model that uses OpenRouter's API via the `openai` python package."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = "https://openrouter.ai/api/v1",
        max_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        route: Literal["fallback"] | None = None,
        models: list[str] | None = None,
        require_parameters: bool = True,
        reasoning: dict[str, Any] | None = None,
    ):
        if not (api_key or os.getenv("OPENROUTER_API_KEY")):
            exception_string = "OPENROUTER_API_KEY variable or api_key required."
            raise openai.OpenAIError(exception_string)
        self._openrouter_openai_chat_model = _OpenRouterOpenaiChatModel(
            model,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url,
            max_tokens=max_tokens,
            seed=seed,
            temperature=temperature,
            route=route,
            models=models,
            require_parameters=require_parameters,
            reasoning=reasoning,
        )

    def _get_extra_body(self) -> dict[str, Any] | None:
        """Get extra body parameters for OpenRouter API."""
        return self._openrouter_openai_chat_model._get_extra_body()

    @property
    def model(self) -> str:
        return self._openrouter_openai_chat_model.model

    @property
    def api_key(self) -> str | None:
        return self._openrouter_openai_chat_model.api_key

    @property
    def base_url(self) -> str | None:
        return self._openrouter_openai_chat_model.base_url

    @property
    def max_tokens(self) -> int | None:
        return self._openrouter_openai_chat_model.max_tokens

    @property
    def seed(self) -> int | None:
        return self._openrouter_openai_chat_model.seed

    @property
    def temperature(self) -> float | None:
        return self._openrouter_openai_chat_model.temperature

    @property
    def route(self) -> Literal["fallback"] | None:
        return self._openrouter_openai_chat_model._route

    @property
    def models(self) -> list[str] | None:
        return self._openrouter_openai_chat_model._models

    @property
    def require_parameters(self) -> bool:
        return self._openrouter_openai_chat_model._require_parameters

    @property
    def reasoning(self) -> dict[str, Any] | None:
        return self._openrouter_openai_chat_model._reasoning

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Request an LLM message."""
        return self._openrouter_openai_chat_model.complete(
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
        return await self._openrouter_openai_chat_model.acomplete(
            messages=messages,
            functions=functions,
            output_types=output_types,
            stop=stop,
        )
