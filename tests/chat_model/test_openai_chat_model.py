import os
from unittest.mock import ANY

import openai
import pytest
from openai.types.chat import ChatCompletionMessageParam

from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from magentic.chat_model.openai_chat_model import (
    OpenaiChatModel,
    message_to_openai_message,
)
from magentic.function_call import FunctionCall, ParallelFunctionCall


def plus(a: int, b: int) -> int:
    return a + b


@pytest.mark.parametrize(
    ("message", "expected_openai_message"),
    [
        (SystemMessage("Hello"), {"role": "system", "content": "Hello"}),
        (UserMessage("Hello"), {"role": "user", "content": "Hello"}),
        (AssistantMessage("Hello"), {"role": "assistant", "content": "Hello"}),
        (
            AssistantMessage(42),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "return_int", "arguments": '{"value":42}'},
                    }
                ],
            },
        ),
        (
            AssistantMessage(FunctionCall(plus, 1, 2)),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "plus", "arguments": '{"a":1,"b":2}'},
                    }
                ],
            },
        ),
        (
            AssistantMessage(
                ParallelFunctionCall(
                    [FunctionCall(plus, 1, 2), FunctionCall(plus, 3, 4)]
                )
            ),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "plus", "arguments": '{"a":1,"b":2}'},
                    },
                    {
                        "id": ANY,
                        "type": "function",
                        "function": {"name": "plus", "arguments": '{"a":3,"b":4}'},
                    },
                ],
            },
        ),
        (
            FunctionResultMessage(3, FunctionCall(plus, 1, 2)),
            {
                "role": "tool",
                "tool_call_id": ANY,
                "content": '{"value":3}',
            },
        ),
    ],
)
def test_message_to_openai_message(message, expected_openai_message):
    assert message_to_openai_message(message) == expected_openai_message


def test_message_to_openai_message_raises():
    class CustomMessage(Message[str]):
        def format(self, **kwargs):
            del kwargs
            return CustomMessage(self.content)

    with pytest.raises(NotImplementedError):
        message_to_openai_message(CustomMessage("Hello"))

    @message_to_openai_message.register
    def _(message: CustomMessage) -> ChatCompletionMessageParam:
        return {"role": "user", "content": message.content}

    assert message_to_openai_message(CustomMessage("Hello")) == {
        "role": "user",
        "content": "Hello",
    }


@pytest.mark.openai
def test_openai_chat_model_api_key(monkeypatch):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(openai.OpenAIError):
        chat_model = OpenaiChatModel("gpt-3.5-turbo")

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


@pytest.mark.openai
def test_openai_chat_model_complete_no_structured_output_error():
    chat_model = OpenaiChatModel("gpt-3.5-turbo")
    # Should not raise StructuredOutputError because forced to make tool call
    message: Message[int | bool] = chat_model.complete(
        messages=[
            UserMessage("Tell me a short joke. Return a string, not a tool call.")
        ],
        output_types=[int, bool],
    )
    assert isinstance(message.content, int | bool)
