import os

import pytest

from magentic.chat_model.message import ImageBytes, UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel


@pytest.fixture
def chat_model():
    return OpenaiChatModel(
        "grok-2", base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"]
    )


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True.", [bool], bool),
        ("Return [1, 2, 3, 4, 5]", [list[int]], list),
        ("Return a list of fruit", [list[str]], list),
    ],
)
@pytest.mark.openai_xai
def test_openai_chat_model_xai_complete(
    chat_model, prompt, output_types, expected_output_type
):
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
@pytest.mark.openai_xai
async def test_openai_chat_model_xai_acomplete(
    chat_model, prompt, output_types, expected_output_type
):
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.openai_xai
def test_openai_chat_model_xai_complete_image_bytes(image_bytes_jpg):
    chat_model = OpenaiChatModel(
        "grok-2-vision",
        base_url="https://api.x.ai/v1",
        api_key=os.environ["XAI_API_KEY"],
    )
    message = chat_model.complete(
        messages=[
            UserMessage(
                ("Describe this image in one word.", ImageBytes(image_bytes_jpg))
            )
        ]
    )
    assert isinstance(message.content, str)
