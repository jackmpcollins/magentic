import pytest

from magentic.chat_model.message import AssistantMessage, UserMessage
from magentic.chat_model.openai_chat_model import (
    OpenaiChatModel,
)


def test_openai_chat_model_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_OPENAI_MODEL", "gpt-4")
    assert OpenaiChatModel().model == "gpt-4"


def test_openai_chat_model_max_tokens(monkeypatch):
    monkeypatch.setenv("MAGENTIC_OPENAI_MAX_TOKENS", "1024")
    assert OpenaiChatModel().max_tokens == 1024


def test_openai_chat_model_temperature(monkeypatch):
    monkeypatch.setenv("MAGENTIC_OPENAI_TEMPERATURE", "2")
    assert OpenaiChatModel().temperature == 2


@pytest.mark.openai
def test_openai_chat_model_completion():
    model = OpenaiChatModel(
        model="gpt-3.5-turbo",
        max_tokens=5,
        temperature=0.5,
    )
    response = model.complete(messages=[UserMessage("Hello!")])
    assert isinstance(response, AssistantMessage)
    assert isinstance(response.content, str)
    # TODO: test for num_tokens here
