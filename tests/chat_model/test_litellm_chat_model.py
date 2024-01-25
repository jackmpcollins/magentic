import litellm
import pytest

from magentic.chat_model.litellm_chat_model import LitellmChatModel
from magentic.chat_model.message import UserMessage
from magentic.function_call import FunctionCall


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


@pytest.mark.ollama
def test_litellm_chat_model_complete_ollama():
    chat_model = LitellmChatModel("ollama/llama2", api_base="http://localhost:11434")
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


@pytest.mark.skip(
    reason="LiteLLM function calling with streaming is indistinguishable from normal text."
)
@pytest.mark.ollama
def test_litellm_chat_model_complete_ollama_function_calling():
    def sum(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = LitellmChatModel("ollama/llama2", api_base="http://localhost:11434")
    message = chat_model.complete(
        messages=[UserMessage("Sum 1 and 2")], functions=[sum]
    )
    assert isinstance(message.content, FunctionCall)


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
