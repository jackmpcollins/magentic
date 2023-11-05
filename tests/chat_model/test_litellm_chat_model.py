import litellm
import pytest

from magentic.chat_model.litellm_chat_model import LitellmChatModel
from magentic.chat_model.message import UserMessage


@pytest.mark.openai
def test_litellm_chat_model_complete_openai():
    chat_model = LitellmChatModel("gpt-3.5-turbo")
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.anthropic
def test_litellm_chat_model_complete_anthropic():
    chat_model = LitellmChatModel("claude-2")
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.anthropic
def test_litellm_chat_model_complete_anthropic_function_calling_error():
    def sum(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = LitellmChatModel("claude-2")
    with pytest.raises(litellm.exceptions.ServiceUnavailableError):
        chat_model.complete(messages=[UserMessage("Say hello!")], functions=[sum])


@pytest.mark.asyncio
@pytest.mark.openai
async def test_litellm_chat_model_acomplete_openai():
    chat_model = LitellmChatModel("gpt-3.5-turbo")
    message = await chat_model.acomplete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_litellm_chat_model_acomplete_anthropic():
    chat_model = LitellmChatModel("claude-2")
    message = await chat_model.acomplete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)
