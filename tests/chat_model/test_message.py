from pydantic import BaseModel
from typing_extensions import assert_type

from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    UserMessage,
)


def test_user_message_format():
    user_message = UserMessage("Hello {x}")
    user_message_formatted = user_message.format(x="world")

    assert_type(user_message_formatted, UserMessage)
    assert_type(user_message_formatted.content, str)
    assert user_message_formatted == UserMessage("Hello world")


def test_assistant_message_format():
    class Country(BaseModel):
        name: str

    assistant_message = AssistantMessage(Country(name="USA"))
    assistant_message_formatted = assistant_message.format(foo="bar")

    assert_type(assistant_message_formatted, AssistantMessage[Country])
    assert_type(assistant_message_formatted.content, Country)
    assert assistant_message_formatted == AssistantMessage(Country(name="USA"))


def test_function_result_message_format():
    def plus(a: int, b: int) -> int:
        return a + b

    function_result_message = FunctionResultMessage(3, plus)
    function_result_message_formatted = function_result_message.format(foo="bar")

    assert_type(function_result_message_formatted, FunctionResultMessage[int])
    assert_type(function_result_message_formatted.content, int)
    assert function_result_message_formatted == FunctionResultMessage(3, plus)
