import pytest

from magentic.chat_model.message import UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True.", [bool], bool),
        ("Return [1, 2, 3, 4, 5]", [list[int]], list),
        ('Return ["apple", "banana"]', [list[str]], list),
    ],
)
@pytest.mark.openai_ollama  # TODO: Add to pytest markers in pyproject.toml
def test_openai_chat_model_complete_ollama(prompt, output_types, expected_output_type):
    chat_model = OpenaiChatModel("llama3.1", base_url="http://localhost:11434/v1/")
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


# TODO: Add asyncio tests
