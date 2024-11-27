from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator
from itertools import chain
from typing import Generic, NamedTuple, TypeVar

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


class FunctionCallChunk(NamedTuple):
    id: str | None
    name: str | None
    args: str | None


class StreamParser(ABC, Generic[ItemT]):
    @abstractmethod
    def is_content(self, item: ItemT) -> bool: ...

    @abstractmethod
    def is_content_ended(self, item: ItemT) -> bool: ...

    @abstractmethod
    def get_content(self, item: ItemT) -> str | None: ...

    @abstractmethod
    def is_tool_call(self, item: ItemT) -> bool: ...

    @abstractmethod
    def iter_tool_calls(self, item: ItemT) -> Iterable[FunctionCallChunk]: ...


class StreamState(ABC, Generic[ItemT]):
    """Tracks the state of the LLM output stream.

    - message snapshot
    - usage
    - stop reason
    """

    usage_ref: list[Usage]

    @abstractmethod
    def update(self, item: ItemT) -> None: ...

    @property
    @abstractmethod
    def current_message_snapshot(self) -> Message[Any]: ...


class OutputStream(Generic[ItemT, OutputT]):
    """Converts streamed LLM output into a stream of magentic objects."""

    def __init__(
        self,
        stream: Iterator[ItemT],
        function_schemas: Iterable[FunctionSchema[OutputT]],
        parser: StreamParser[ItemT],
        state: StreamState[ItemT],
    ):
        self._stream = stream
        self._function_schemas = function_schemas
        self._parser = parser
        self._state = state

        self._iterator = self.__stream__()

    def __next__(self) -> StreamedStr | OutputT:
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[StreamedStr | OutputT]:
        yield from self._iterator

    def _streamed_str(
        self, stream: Iterator[ItemT], current_item_ref: list[ItemT]
    ) -> Iterator[str]:
        for item in stream:
            if content := self._parser.get_content(item):
                yield content
            if self._parser.is_content_ended(item):
                # TODO: Check if output types allow for early return and raise if not
                assert not current_item_ref  # noqa: S101
                current_item_ref.append(item)
                return

    def _tool_call(
        self,
        stream: Iterator[FunctionCallChunk],
        current_tool_call_ref: list[FunctionCallChunk],
        current_tool_call_id: str,
    ) -> Iterator[str]:
        for item in stream:
            # Only end the stream if we encounter a new tool call
            # so that the whole stream is consumed including stop_reason/usage chunks
            if item.id and item.id != current_tool_call_id:
                # TODO: Check if output types allow for early return and raise if not
                assert not current_tool_call_ref  # noqa: S101
                current_tool_call_ref.append(item)
                return
            if item.args:
                yield item.args

    def __stream__(self) -> Iterator[StreamedStr | OutputT]:
        stream = apply(self._state.update, self._stream)
        current_item_ref = [next(stream)]
        while current_item_ref:
            current_item = current_item_ref.pop()
            if self._parser.is_content(current_item):
                stream = chain([current_item], stream)
                yield StreamedStr(self._streamed_str(stream, current_item_ref))
            elif self._parser.is_tool_call(current_item):
                tool_calls_stream = (
                    tool_call_chunk
                    for item in chain([current_item], stream)
                    for tool_call_chunk in self._parser.iter_tool_calls(item)
                )
                tool_call_ref = [next(tool_calls_stream)]
                while tool_call_ref:
                    current_tool_call_chunk = tool_call_ref.pop()
                    current_tool_call_id = current_tool_call_chunk.id
                    assert current_tool_call_id is not None  # noqa: S101
                    assert current_tool_call_chunk.name is not None  # noqa: S101
                    function_schema = select_function_schema(
                        self._function_schemas, current_tool_call_chunk.name
                    )
                    try:
                        yield function_schema.parse_args(
                            self._tool_call(
                                chain([current_tool_call_chunk], tool_calls_stream),
                                tool_call_ref,
                                current_tool_call_id,
                            )
                        )
                    # TODO: Catch/raise unknown tool call error here
                    except ValidationError as e:
                        assert current_tool_call_id is not None  # noqa: S101
                        raise ToolSchemaParseError(
                            output_message=self._state.current_message_snapshot,
                            tool_call_id=current_tool_call_id,
                            validation_error=e,
                        ) from e
            elif new_current_item := next(stream, None):
                current_item_ref.append(new_current_item)

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
        parser: StreamParser[ItemT],
        state: StreamState[ItemT],
    ):
        self._stream = stream
        self._function_schemas = function_schemas
        self._parser = parser
        self._state = state

        self._iterator = self.__stream__()

    async def __anext__(self) -> AsyncStreamedStr | OutputT:
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[AsyncStreamedStr | OutputT]:
        async for item in self._iterator:
            yield item

    async def _streamed_str(
        self, stream: AsyncIterator[ItemT], current_item_ref: list[ItemT]
    ) -> AsyncIterator[str]:
        async for item in stream:
            if content := self._parser.get_content(item):
                yield content
            if self._parser.is_content_ended(item):
                # TODO: Check if output types allow for early return
                assert not current_item_ref  # noqa: S101
                current_item_ref.append(item)
                return

    async def _tool_call(
        self,
        stream: AsyncIterator[FunctionCallChunk],
        current_tool_call_ref: list[FunctionCallChunk],
        current_tool_call_id: str,
    ) -> AsyncIterator[str]:
        async for item in stream:
            if item.id and item.id != current_tool_call_id:
                # TODO: Check if output types allow for early return
                assert not current_tool_call_ref  # noqa: S101
                current_tool_call_ref.append(item)
                return
            if item.args:
                yield item.args

    async def __stream__(self) -> AsyncIterator[AsyncStreamedStr | OutputT]:
        stream = aapply(self._state.update, self._stream)
        current_item_ref = [await anext(stream)]
        while current_item_ref:
            current_item = current_item_ref.pop()
            if self._parser.is_content(current_item):
                stream = achain(async_iter([current_item]), stream)
                yield AsyncStreamedStr(self._streamed_str(stream, current_item_ref))
            elif self._parser.is_tool_call(current_item):
                tool_calls_stream = (
                    tool_call_chunk
                    async for item in achain(async_iter([current_item]), stream)
                    for tool_call_chunk in self._parser.iter_tool_calls(item)
                )
                tool_call_ref = [await anext(tool_calls_stream)]
                while tool_call_ref:
                    current_tool_call_chunk = tool_call_ref.pop()
                    current_tool_call_id = current_tool_call_chunk.id
                    assert current_tool_call_id is not None  # noqa: S101
                    assert current_tool_call_chunk.name is not None  # noqa: S101
                    function_schema = select_function_schema(
                        self._function_schemas, current_tool_call_chunk.name
                    )
                    try:
                        yield await function_schema.aparse_args(
                            self._tool_call(
                                achain(
                                    async_iter([current_tool_call_chunk]),
                                    tool_calls_stream,
                                ),
                                tool_call_ref,
                                current_tool_call_id,
                            )
                        )
                    # TODO: Catch/raise unknown tool call error here
                    except ValidationError as e:
                        assert current_tool_call_id is not None  # noqa: S101
                        raise ToolSchemaParseError(
                            output_message=self._state.current_message_snapshot,
                            tool_call_id=current_tool_call_id,
                            validation_error=e,
                        ) from e
            elif new_current_item := await anext(stream, None):
                current_item_ref.append(new_current_item)

    @property
    def usage_ref(self) -> list[Usage]:
        return self._state.usage_ref

    async def close(self):
        await self._stream.close()
