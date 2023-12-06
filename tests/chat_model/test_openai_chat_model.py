import os

import openai
import pytest

from magentic.chat_model.message import UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel


@pytest.mark.openai
def test_openai_chat_model_api_key(monkeypatch):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    monkeypatch.delenv("OPENAI_API_KEY")

    chat_model = OpenaiChatModel("gpt-3.5-turbo")
    with pytest.raises(openai.OpenAIError):
        chat_model.complete(messages=[UserMessage("Say hello!")])

    chat_model = OpenaiChatModel("gpt-3.5-turbo", api_key=openai_api_key)
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.openai
def test_openai_chat_model_complete_base_url():
    chat_model = OpenaiChatModel("gpt-3.5-turbo", base_url="https://api.openai.com/v1")
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.openai
def test_openai_chat_model_complete_seed():
    chat_model = OpenaiChatModel("gpt-3.5-turbo", seed=42)
    message1 = chat_model.complete(messages=[UserMessage("Say hello!")])
    message2 = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert message1.content == message2.content
