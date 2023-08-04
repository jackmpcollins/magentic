import pytest

from magentic.chat import Chat
from magentic.chat_model.base import AssistantMessage, UserMessage
from magentic.prompt_function import prompt


def test_chat_from_prompt():
    """Test creating a chat from a prompt function."""

    def plus(a: int, b: int) -> int:
        return a + b

    @prompt("What is {a} plus {b}?", functions=[plus])
    def add_text_numbers(a: str, b: str) -> int:
        ...

    chat = Chat.from_prompt(add_text_numbers, "one", "two")
    assert chat.messages == [UserMessage(content="What is one plus two?")]


def test_chat_add_message():
    chat1 = Chat()
    chat2 = chat1.add_message(UserMessage(content="Hello"))
    assert chat1.messages == []
    assert chat2.messages == [UserMessage(content="Hello")]


@pytest.mark.openai
def test_chat_submit():
    chat1 = Chat(
        messages=[UserMessage(content="Hello")],
    )
    chat2 = chat1.submit()
    assert chat1.messages == [UserMessage(content="Hello")]
    assert chat2.messages[0] == UserMessage(content="Hello")
    assert isinstance(chat2.messages[1], AssistantMessage)
