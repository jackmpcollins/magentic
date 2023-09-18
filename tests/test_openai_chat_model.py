from magentic.chat_model.openai_chat_model import (
    OpenaiChatModel,
)


def test_openai_chat_model_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_OPENAI_MODEL", "gpt-4")
    assert OpenaiChatModel().model == "gpt-4"


def test_openai_chat_model_temperature(monkeypatch):
    monkeypatch.setenv("MAGENTIC_OPENAI_TEMPERATURE", "2")
    assert OpenaiChatModel().temperature == 2
