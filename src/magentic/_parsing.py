"""Functions for parsing and checking return types."""

from collections.abc import Iterable

from magentic._streamed_response import AsyncStreamedResponse, StreamedResponse
from magentic.function_call import AsyncParallelFunctionCall, ParallelFunctionCall
from magentic.streaming import AsyncStreamedStr, StreamedStr
from magentic.typing import is_any_origin_subclass


def contains_string_type(types: Iterable[type]) -> bool:
    return is_any_origin_subclass(
        types,
        (str, StreamedStr, AsyncStreamedStr, StreamedResponse, AsyncStreamedResponse),
    )


def contains_parallel_function_call_type(types: Iterable[type]) -> bool:
    return is_any_origin_subclass(
        types,
        (
            ParallelFunctionCall,
            AsyncParallelFunctionCall,
            StreamedResponse,
            AsyncStreamedResponse,
        ),
    )
