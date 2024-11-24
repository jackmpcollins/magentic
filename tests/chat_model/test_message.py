from unittest.mock import ANY, MagicMock

import pytest
from pydantic import BaseModel, TypeAdapter
from typing_extensions import assert_type

from magentic.chat_model.message import (
    AnyMessage,
    AssistantMessage,
    FunctionResultMessage,
    Placeholder,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from magentic.function_call import FunctionCall


def test_placeholder():
    class Country(BaseModel):
        name: str

    placeholder = Placeholder(Country, "country")

    assert_type(placeholder, Placeholder[Country])
    assert placeholder.name == "country"


@pytest.mark.parametrize(
    ("message", "message_model_dump"),
    [
        (SystemMessage("Hello"), {"role": "system", "content": "Hello"}),
        (UserMessage("Hello"), {"role": "user", "content": "Hello"}),
        (AssistantMessage("Hello"), {"role": "assistant", "content": "Hello"}),
        (AssistantMessage(42), {"role": "assistant", "content": 42}),
        (
            ToolResultMessage(3, "unique_id"),
            {"role": "tool", "content": 3, "tool_call_id": "unique_id"},
        ),
        (
            FunctionResultMessage(3, FunctionCall(MagicMock(), 1, 2)),
            {"role": "tool", "content": 3, "tool_call_id": ANY},
        ),
    ],
)
def test_message_model_dump(message, message_model_dump):
    assert message.model_dump() == message_model_dump


@pytest.mark.parametrize(
    ("message", "message_repr"),
    [
        (SystemMessage("Hello"), "SystemMessage('Hello')"),
        (UserMessage("Hello"), "UserMessage('Hello')"),
        (AssistantMessage("Hello"), "AssistantMessage('Hello')"),
        (AssistantMessage(42), "AssistantMessage(42)"),
        (
            FunctionResultMessage(
                3, FunctionCall(MagicMock(__repr__=lambda x: "plus_repr"), 1, 2)
            ),
            "FunctionResultMessage(3, FunctionCall(plus_repr, 1, 2))",
        ),
    ],
)
def test_message_repr(message, message_repr):
    assert repr(message) == message_repr


def test_user_message_format():
    user_message = UserMessage("Hello {x}")
    user_message_formatted = user_message.format(x="world")

    assert_type(user_message_formatted, UserMessage)
    assert_type(user_message_formatted.content, str)
    assert user_message_formatted == UserMessage("Hello world")


def test_assistant_message_usage():
    assistant_message = AssistantMessage("Hello")
    assert assistant_message.usage is None
    assistant_message = AssistantMessage._with_usage(
        "Hello", [Usage(input_tokens=1, output_tokens=2)]
    )
    assert assistant_message.usage == Usage(input_tokens=1, output_tokens=2)


def test_assistant_message_format_str():
    assistant_message = AssistantMessage("Hello {x}")
    assistant_message_formatted = assistant_message.format(x="world")

    assert_type(assistant_message_formatted, AssistantMessage[str])
    assert_type(assistant_message_formatted.content, str)
    assert assistant_message_formatted == AssistantMessage("Hello world")


def test_assistant_message_format_placeholder():
    class Country(BaseModel):
        name: str

    assistant_message = AssistantMessage(Placeholder(Country, "country"))
    assistant_message_formatted = assistant_message.format(country=Country(name="USA"))

    assert_type(assistant_message_formatted, AssistantMessage[Country])
    assert_type(assistant_message_formatted.content, Country)
    assert assistant_message_formatted == AssistantMessage(Country(name="USA"))


def test_function_result_message_eq():
    def plus(a: int, b: int) -> int:
        return a + b

    func_call = FunctionCall(plus, 1, 2)
    function_result_message = FunctionResultMessage(3, func_call)
    assert function_result_message == function_result_message
    assert function_result_message == FunctionResultMessage(3, func_call)
    assert function_result_message != FunctionResultMessage(7, FunctionCall(plus, 3, 4))
    # Different unique ids internally => not equal, despite equal FunctionCalls
    func_call_copy = FunctionCall(plus, 1, 2)
    func_call_copy._unique_id = "999999999"
    assert function_result_message != FunctionResultMessage(3, func_call_copy)


def test_function_result_message_format():
    def plus(a: int, b: int) -> int:
        return a + b

    func_call = FunctionCall(plus, 1, 2)
    function_result_message = FunctionResultMessage(3, func_call)
    function_result_message_formatted = function_result_message.format(foo="bar")

    assert_type(function_result_message_formatted, FunctionResultMessage[int])
    assert_type(function_result_message_formatted.content, int)
    assert function_result_message_formatted == FunctionResultMessage(3, func_call)


def test_any_message():
    messages = [
        {"role": "system", "content": "Hello"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "tool", "content": 3, "tool_call_id": "unique_id"},
    ]
    assert TypeAdapter(list[AnyMessage]).validate_python(messages) == [
        SystemMessage("Hello"),
        UserMessage("Hello"),
        AssistantMessage("Hello"),
        ToolResultMessage(3, "unique_id"),
    ]
