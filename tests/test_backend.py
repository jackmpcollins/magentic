from magentic.backend import get_chat_model


def test_openai_chat_model_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "openai")
    monkeypatch.setenv("MAGENTIC_OPENAI_MODEL", "gpt-4")
    assert get_chat_model().model == "gpt-4"


def test_openai_chat_model_temperature(monkeypatch):
    monkeypatch.setenv("MAGENTIC_OPENAI_TEMPERATURE", "2")
    assert get_chat_model().temperature == 2
