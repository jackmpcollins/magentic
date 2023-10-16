import pytest

from magentic.backend import get_chat_model
from magentic.chat_model.message import AssistantMessage, UserMessage
from magentic.chat_model.openai_chat_model import (
    OpenaiChatModel,
)


def test_openai_chat_model_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "openai")
    monkeypatch.setenv("MAGENTIC_OPENAI_MODEL", "gpt-4")
    chat_model = get_chat_model()
    assert isinstance(chat_model, OpenaiChatModel)
    assert chat_model.model == "gpt-4"


def test_openai_chat_model_max_tokens(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "openai")
    monkeypatch.setenv("MAGENTIC_OPENAI_MAX_TOKENS", "1024")
    chat_model = get_chat_model()
    assert isinstance(chat_model, OpenaiChatModel)
    assert chat_model.max_tokens == 1024


def test_openai_chat_model_temperature(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "openai")
    monkeypatch.setenv("MAGENTIC_OPENAI_TEMPERATURE", "2")
    chat_model = get_chat_model()
    assert isinstance(chat_model, OpenaiChatModel)
    assert chat_model.temperature == 2


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
