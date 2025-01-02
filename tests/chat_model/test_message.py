from typing import TYPE_CHECKING, cast
from unittest.mock import ANY, MagicMock

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import assert_type

from magentic.chat_model.message import (
    AnyMessage,
    AssistantMessage,
    FunctionResultMessage,
    ImageBytes,
    Placeholder,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from magentic.function_call import FunctionCall


def test_placeholder_format():
    class Country(BaseModel):
        name: str

    placeholder = Placeholder(Country, "country")

    assert_type(placeholder, Placeholder[Country])
    assert placeholder.name == "country"
    assert placeholder.format(country=Country(name="USA")) == Country(name="USA")


def test_placeholder_coercion():
    placeholder_str = Placeholder(str, "my_string")
    assert placeholder_str.format(my_string="test") == "test"
    placeholder_list_str = Placeholder(list[str], "my_list_str")
    assert placeholder_list_str.format(my_list_str=["test"]) == ["test"]
    assert placeholder_list_str.format(my_list_str={"test"}) == ["test"]  # set -> list
    with pytest.raises(ValueError):  # noqa: PT011
        placeholder_str.format(my_string=True)  # bool does not coerce to str


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


def test_image_bytes_jpg(image_bytes_jpg: bytes) -> None:
    image_bytes = ImageBytes(image_bytes_jpg)
    assert image_bytes.mime_type == "image/jpeg"
    assert image_bytes.as_base64() == snapshot(
        "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"
    )


def test_image_bytes_png(image_bytes_png: bytes) -> None:
    image_bytes = ImageBytes(image_bytes_png)
    assert image_bytes.mime_type == "image/png"
    assert image_bytes.as_base64() == snapshot(
        "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"
    )


def test_image_bytes_invalid():
    with pytest.raises(ValidationError):
        ImageBytes(b"invalid")


def test_user_message_format():
    user_message = UserMessage("Hello {x}")
    user_message_formatted = user_message.format(x="world")

    assert_type(user_message_formatted, UserMessage[str])
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

    if TYPE_CHECKING:  # Avoid runtime error for None missing `format` method
        assert_type(cast(AssistantMessage[str], None).format(), AssistantMessage[str])
        assert_type(
            cast(AssistantMessage[Placeholder[int]], None).format(),
            AssistantMessage[int],
        )
        assert_type(  # type: ignore[assert-type]  # Breaks mypy but okay with pyright
            cast(AssistantMessage[str | Placeholder[int]], None).format(),
            AssistantMessage[str | int],
        )


def test_function_result_message_eq():
    def plus(a: int, b: int) -> int:
        return a + b

    func_call = FunctionCall(plus, 1, 2)
    function_result_message = FunctionResultMessage(3, func_call)
    assert function_result_message == function_result_message
    assert function_result_message == FunctionResultMessage(3, func_call)
    # Different unique ids internally => not equal, despite equal FunctionCalls
    assert function_result_message != FunctionResultMessage(3, FunctionCall(plus, 1, 2))
    assert function_result_message != FunctionResultMessage(7, FunctionCall(plus, 3, 4))


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
