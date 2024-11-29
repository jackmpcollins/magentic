import os

import pytest

from magentic.chat_model.message import UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True.", [bool], bool),
        ("Return [1, 2, 3, 4, 5]", [list[int]], list),
        ("Return a list of fruit", [list[str]], list),
    ],
)
@pytest.mark.openai_gemini
def test_openai_chat_model_complete_gemini(prompt, output_types, expected_output_type):
    chat_model = OpenaiChatModel(
        "gemini-1.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True.", [bool], bool),
        ("Return [1, 2, 3, 4, 5]", [list[int]], list),
        ("Return a list of fruit", [list[str]], list),
    ],
)
@pytest.mark.openai_gemini
async def test_openai_chat_model_acomplete_gemini(
    prompt, output_types, expected_output_type
):
    chat_model = OpenaiChatModel(
        "gemini-1.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)
