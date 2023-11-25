import pytest

from magentic.chat_model.message import UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel


@pytest.mark.openai
def test_openai_chat_model_complete_base_url():
    chat_model = OpenaiChatModel("gpt-3.5-turbo", base_url="https://api.openai.com/v1")
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)
