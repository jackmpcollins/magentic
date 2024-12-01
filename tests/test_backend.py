import pytest

from magentic.backend import get_chat_model
from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from magentic.chat_model.litellm_chat_model import LitellmChatModel
from magentic.chat_model.message import AssistantMessage, UserMessage
from magentic.chat_model.mistral_chat_model import MistralChatModel
from magentic.chat_model.openai_chat_model import OpenaiChatModel


def test_backend_anthropic_chat_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "anthropic")
    monkeypatch.setenv("MAGENTIC_ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    monkeypatch.setenv("MAGENTIC_ANTHROPIC_API_KEY", "sk-1234567890")
    monkeypatch.setenv("MAGENTIC_ANTHROPIC_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("MAGENTIC_ANTHROPIC_MAX_TOKENS", "10")
    monkeypatch.setenv("MAGENTIC_ANTHROPIC_TEMPERATURE", "2")
    chat_model = get_chat_model()
    assert isinstance(chat_model, AnthropicChatModel)
    assert chat_model.model == "claude-3-haiku-20240307"
    assert chat_model.api_key == "sk-1234567890"
    assert chat_model.base_url == "http://localhost:8080"
    assert chat_model.max_tokens == 10
    assert chat_model.temperature == 2


def test_backend_mistral_chat_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "mistral")
    monkeypatch.setenv("MAGENTIC_MISTRAL_MODEL", "mistral-large-latest")
    monkeypatch.setenv("MAGENTIC_MISTRAL_API_KEY", "sk-1234567890")
    monkeypatch.setenv("MAGENTIC_MISTRAL_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("MAGENTIC_MISTRAL_MAX_TOKENS", "1024")
    monkeypatch.setenv("MAGENTIC_MISTRAL_SEED", "42")
    monkeypatch.setenv("MAGENTIC_MISTRAL_TEMPERATURE", "2")
    chat_model = get_chat_model()
    assert isinstance(chat_model, MistralChatModel)
    assert chat_model.model == "mistral-large-latest"
    assert chat_model.api_key == "sk-1234567890"
    assert chat_model.base_url == "http://localhost:8080"
    assert chat_model.max_tokens == 1024
    assert chat_model.seed == 42
    assert chat_model.temperature == 2


def test_backend_openai_chat_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "openai")
    monkeypatch.setenv("MAGENTIC_OPENAI_MODEL", "gpt-4")
    monkeypatch.setenv("MAGENTIC_OPENAI_API_KEY", "sk-1234567890")
    monkeypatch.setenv("MAGENTIC_OPENAI_API_TYPE", "openai")
    monkeypatch.setenv("MAGENTIC_OPENAI_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("MAGENTIC_OPENAI_MAX_TOKENS", "1024")
    monkeypatch.setenv("MAGENTIC_OPENAI_SEED", "42")
    monkeypatch.setenv("MAGENTIC_OPENAI_TEMPERATURE", "2")
    chat_model = get_chat_model()
    assert isinstance(chat_model, OpenaiChatModel)
    assert chat_model.model == "gpt-4"
    assert chat_model.api_key == "sk-1234567890"
    assert chat_model.api_type == "openai"
    assert chat_model.base_url == "http://localhost:8080"
    assert chat_model.max_tokens == 1024
    assert chat_model.seed == 42
    assert chat_model.temperature == 2


def test_backend_litellm_chat_model(monkeypatch):
    monkeypatch.setenv("MAGENTIC_BACKEND", "litellm")
    monkeypatch.setenv("MAGENTIC_LITELLM_API_BASE", "http://localhost:11434")
    monkeypatch.setenv("MAGENTIC_LITELLM_MODEL", "claude-2")
    monkeypatch.setenv("MAGENTIC_LITELLM_MAX_TOKENS", "1024")
    monkeypatch.setenv("MAGENTIC_LITELLM_TEMPERATURE", "2")
    chat_model = get_chat_model()
    assert isinstance(chat_model, LitellmChatModel)
    assert chat_model.api_base == "http://localhost:11434"
    assert chat_model.model == "claude-2"
    assert chat_model.max_tokens == 1024
    assert chat_model.temperature == 2


@pytest.mark.openai
def test_openai_chat_model_completion():
    model = OpenaiChatModel(
        model="gpt-4o",
        max_tokens=5,
        temperature=0.5,
    )
    response = model.complete(messages=[UserMessage("Hello!")])
    assert isinstance(response, AssistantMessage)
    assert isinstance(response.content, str)
    # TODO: test for num_tokens here


def test_chat_model_context(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    chat_model = OpenaiChatModel("gpt-4")
    with chat_model:
        assert get_chat_model() is chat_model


def test_chat_model_context_within_context(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with OpenaiChatModel("gpt-4"):
        assert get_chat_model().model == "gpt-4"  # type: ignore[attr-defined]

        with OpenaiChatModel("gpt-5"):
            assert get_chat_model().model == "gpt-5"  # type: ignore[attr-defined]

        assert get_chat_model().model == "gpt-4"  # type: ignore[attr-defined]
