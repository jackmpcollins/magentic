from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING, Generic, TypeVar

from magentic.chat_model.function_schema import FunctionSchema, select_function_schema
from magentic.streaming import AsyncStreamedStr, StreamedStr, achain, async_iter

if TYPE_CHECKING:
    from magentic.chat_model.message import Usage


T = TypeVar("T")
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


class OutputStream(Generic[T]):
    """Converts streamed LLM output into a stream of magentic objects."""

    def __init__(
        self,
        stream: Iterator,  # TODO: Fix typing
        function_schemas: Iterable[FunctionSchema[T]],
        content_parser: StreamParser,
        tool_parser: StreamParser,
        usage_parser: StreamParser,
    ):
        self._stream = stream
        self._function_schemas = function_schemas
        self._iterator = self.__stream__()

        self._content_parser = content_parser
        self._tool_parser = tool_parser
        self._usage_parser = usage_parser

        self.usage: Usage | None = None

    def __next__(self) -> StreamedStr | T:
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[StreamedStr | T]:
        yield from self._iterator

    def __stream__(self) -> Iterator[StreamedStr | T]:
        transition = [next(self._stream)]
        while transition:
            transition_item = transition.pop()
            stream_with_transition = chain([transition_item], self._stream)
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
                # TODO: Catch/raise ToolSchemaParseError here for retry logic
                yield function_schema.parse_args(
                    self._tool_parser.iter(stream_with_transition, transition)
                )
            elif self._usage_parser.is_member(transition_item):
                self.usage = self._usage_parser.transform(transition_item)
            elif new_transition_item := next(self._stream, None):
                transition.append(new_transition_item)

    def close(self):
        self._stream.close()


class AsyncOutputStream(Generic[T]):
    """Async version of `OutputStream`."""

    def __init__(
        self,
        stream: AsyncIterator,  # TODO: Fix typing
        function_schemas: Iterable[FunctionSchema[T]],
        content_parser: StreamParser,
        tool_parser: StreamParser,
        usage_parser: StreamParser,
    ):
        self._stream = stream
        self._function_schemas = function_schemas
        self._iterator = self.__stream__()

        self._content_parser = content_parser
        self._tool_parser = tool_parser
        self._usage_parser = usage_parser

        self.usage: Usage | None = None

    async def __anext__(self) -> AsyncStreamedStr | T:
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[AsyncStreamedStr | T]:
        async for item in self._iterator:
            yield item

    async def __stream__(self) -> AsyncIterator[AsyncStreamedStr | T]:
        transition = [await anext(self._stream)]
        while transition:
            transition_item = transition.pop()
            stream_with_transition = achain(async_iter([transition_item]), self._stream)
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
                # TODO: Catch/raise ToolSchemaParseError here for retry logic
                yield await function_schema.aparse_args(
                    self._tool_parser.aiter(stream_with_transition, transition)
                )
            elif self._usage_parser.is_member(transition_item):
                self.usage = self._usage_parser.transform(transition_item)
            elif new_transition_item := await anext(self._stream, None):
                transition.append(new_transition_item)

    async def close(self):
        await self._stream.close()
