import pytest

from magentic.backend import get_chat_model
from magentic.chat_model.litellm_chat_model import LitellmChatModel
from magentic.chat_model.message import AssistantMessage, UserMessage
from magentic.chat_model.openai_chat_model import (
    OpenaiChatModel,
)


def test_backend_openai_chat_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "openai")
    monkeypatch.setenv("MAGENTIC_OPENAI_MODEL", "gpt-4")
    monkeypatch.setenv("MAGENTIC_OPENAI_MAX_TOKENS", "1024")
    monkeypatch.setenv("MAGENTIC_OPENAI_TEMPERATURE", "2")
    chat_model = get_chat_model()
    assert isinstance(chat_model, OpenaiChatModel)
    assert chat_model.model == "gpt-4"
    assert chat_model.max_tokens == 1024
    assert chat_model.temperature == 2


def test_backend_litellm_chat_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "litellm")
    monkeypatch.setenv("MAGENTIC_LITELLM_MODEL", "claude-2")
    monkeypatch.setenv("MAGENTIC_LITELLM_MAX_TOKENS", "1024")
    monkeypatch.setenv("MAGENTIC_LITELLM_TEMPERATURE", "2")
    chat_model = get_chat_model()
    assert isinstance(chat_model, LitellmChatModel)
    assert chat_model.model == "claude-2"
    assert chat_model.max_tokens == 1024
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


def test_chat_model_context():
    chat_model = OpenaiChatModel("gpt-4")
    with chat_model:
        assert get_chat_model() is chat_model


def test_chat_model_context_within_context():
    with OpenaiChatModel("gpt-4"):
        assert get_chat_model().model == "gpt-4"  # type: ignore[attr-defined]

        with OpenaiChatModel("gpt-5"):
            assert get_chat_model().model == "gpt-5"  # type: ignore[attr-defined]

        assert get_chat_model().model == "gpt-4"  # type: ignore[attr-defined]
