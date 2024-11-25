from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator
from itertools import chain
from typing import Generic, TypeVar

from litellm.llms.files_apis.azure import Any
from pydantic import ValidationError

from magentic.chat_model.base import ToolSchemaParseError
from magentic.chat_model.function_schema import FunctionSchema, select_function_schema
from magentic.chat_model.message import Message, Usage
from magentic.streaming import (
    AsyncStreamedStr,
    StreamedStr,
    aapply,
    achain,
    apply,
    async_iter,
)

ItemT = TypeVar("ItemT")
OutputT = TypeVar("OutputT")


class StreamParser(ABC, Generic[ItemT, OutputT]):
    """Filters and transforms items from an iterator until the end condition is met."""

    def is_member(self, item: ItemT) -> bool:
        return True

    @abstractmethod
    def is_end(self, item: ItemT) -> bool: ...

    @abstractmethod
    def transform(self, item: ItemT) -> OutputT: ...

    def iter(
        self, iterator: Iterator[ItemT], transition: list[ItemT]
    ) -> Iterator[OutputT]:
        for item in iterator:
            if self.is_member(item):
                yield self.transform(item)
            if self.is_end(item):
                assert not transition  # noqa: S101
                transition.append(item)
                return

    async def aiter(
        self, aiterator: AsyncIterator[ItemT], transition: list[ItemT]
    ) -> AsyncIterator[OutputT]:
        async for item in aiterator:
            if self.is_member(item):
                yield self.transform(item)
            if self.is_end(item):
                assert not transition  # noqa: S101
                transition.append(item)
                return


class StreamState(ABC, Generic[ItemT]):
    usage_ref: list[Usage]

    @abstractmethod
    def update(self, item: ItemT) -> None: ...

    @property
    @abstractmethod
    def current_tool_call_id(self) -> str | None: ...

    @property
    @abstractmethod
    def current_message_snapshot(self) -> Message[Any]: ...


class OutputStream(Generic[ItemT, OutputT]):
    """Converts streamed LLM output into a stream of magentic objects."""

    def __init__(
        self,
        stream: Iterator[ItemT],
        function_schemas: Iterable[FunctionSchema[OutputT]],
        content_parser: StreamParser,
        tool_parser: StreamParser,
        state: StreamState[ItemT],
    ):
        self._stream = stream
        self._function_schemas = function_schemas
        self._iterator = self.__stream__()

        self._content_parser = content_parser
        self._tool_parser = tool_parser
        self._state = state

        self._wrapped_stream = apply(self._state.update, stream)

    def __next__(self) -> StreamedStr | OutputT:
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[StreamedStr | OutputT]:
        yield from self._iterator

    def __stream__(self) -> Iterator[StreamedStr | OutputT]:
        transition = [next(self._wrapped_stream)]
        while transition:
            transition_item = transition.pop()
            stream_with_transition = chain([transition_item], self._wrapped_stream)
            if self._content_parser.is_member(transition_item):
                yield StreamedStr(
                    self._content_parser.iter(stream_with_transition, transition)
                )
            elif self._tool_parser.is_member(transition_item):
                # TODO: Add new base class for tool parser
                tool_name = self._tool_parser.get_tool_name(transition_item)
                function_schema = select_function_schema(
                    self._function_schemas, tool_name
                )
                try:
                    yield function_schema.parse_args(
                        self._tool_parser.iter(stream_with_transition, transition)
                    )
                # TODO: Catch/raise unknown tool call error here
                except ValidationError as e:
                    assert self._state.current_tool_call_id is not None  # noqa: S101
                    raise ToolSchemaParseError(
                        output_message=self._state.current_message_snapshot,
                        tool_call_id=self._state.current_tool_call_id,
                        validation_error=e,
                    ) from e
            elif new_transition_item := next(self._wrapped_stream, None):
                transition.append(new_transition_item)

    @property
    def usage_ref(self) -> list[Usage]:
        return self._state.usage_ref

    def close(self):
        self._stream.close()


class AsyncOutputStream(Generic[ItemT, OutputT]):
    """Async version of `OutputStream`."""

    def __init__(
        self,
        stream: AsyncIterator[ItemT],
        function_schemas: Iterable[FunctionSchema[OutputT]],
        content_parser: StreamParser,
        tool_parser: StreamParser,
        state: StreamState[ItemT],
    ):
        self._stream = stream
        self._function_schemas = function_schemas
        self._iterator = self.__stream__()

        self._content_parser = content_parser
        self._tool_parser = tool_parser
        self._state = state

        self._wrapped_stream = aapply(self._state.update, stream)

    async def __anext__(self) -> AsyncStreamedStr | OutputT:
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[AsyncStreamedStr | OutputT]:
        async for item in self._iterator:
            yield item

    async def __stream__(self) -> AsyncIterator[AsyncStreamedStr | OutputT]:
        transition = [await anext(self._wrapped_stream)]
        while transition:
            transition_item = transition.pop()
            stream_with_transition = achain(
                async_iter([transition_item]), self._wrapped_stream
            )
            if self._content_parser.is_member(transition_item):
                yield AsyncStreamedStr(
                    self._content_parser.aiter(stream_with_transition, transition)
                )
            elif self._tool_parser.is_member(transition_item):
                # TODO: Add new base class for tool parser
                tool_name = self._tool_parser.get_tool_name(transition_item)
                function_schema = select_function_schema(
                    self._function_schemas, tool_name
                )
                try:
                    yield await function_schema.aparse_args(
                        self._tool_parser.aiter(stream_with_transition, transition)
                    )
                # TODO: Catch/raise unknown tool call error here
                except ValidationError as e:
                    assert self._state.current_tool_call_id is not None  # noqa: S101
                    raise ToolSchemaParseError(
                        output_message=self._state.current_message_snapshot,
                        tool_call_id=self._state.current_tool_call_id,
                        validation_error=e,
                    ) from e
            elif new_transition_item := await anext(self._wrapped_stream, None):
                transition.append(new_transition_item)

    @property
    def usage_ref(self) -> list[Usage]:
        return self._state.usage_ref

    async def close(self):
        await self._stream.close()
