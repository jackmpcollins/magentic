from pydantic import BaseModel
from typing_extensions import assert_type

from magentic.chat_model.message import AssistantMessage, Placeholder, UserMessage


def test_placeholder():
    class Country(BaseModel):
        name: str

    placeholder = Placeholder(Country, "country")

    assert_type(placeholder, Placeholder[Country])
    assert placeholder.name == "country"


def test_user_message_format():
    user_message = UserMessage("Hello {x}")
    user_message_formatted = user_message.format(x="world")

    assert_type(user_message_formatted, UserMessage)
    assert_type(user_message_formatted.content, str)
    assert user_message_formatted == UserMessage("Hello world")


def test_assistant_message_format():
    class Country(BaseModel):
        name: str

    assistant_message = AssistantMessage(Placeholder(Country, "country"))
    assistant_message_formatted = assistant_message.format(country=Country(name="USA"))

    assert_type(assistant_message_formatted, AssistantMessage[Country])
    assert_type(assistant_message_formatted.content, Country)
    assert assistant_message_formatted == AssistantMessage(Country(name="USA"))
