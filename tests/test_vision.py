from typing_extensions import assert_type

from magentic.chat_model.message import Placeholder
from magentic.chat_model.openai_chat_model import message_to_openai_message
from magentic.vision import UserImageMessage


def test_user_image_message_format_noop():
    image_message = UserImageMessage("https://example.com/image.jpg")
    image_message_formatted = image_message.format(foo="bar")

    assert_type(image_message_formatted, UserImageMessage[str])
    assert_type(image_message_formatted.content, str)
    assert image_message_formatted == UserImageMessage("https://example.com/image.jpg")


def test_user_image_message_format_placeholder():
    image_message = UserImageMessage(Placeholder(bytes, "image"))
    image_message_formatted = image_message.format(image=b"image")

    assert_type(image_message_formatted, UserImageMessage[bytes])
    assert_type(image_message_formatted.content, bytes)
    assert image_message_formatted == UserImageMessage(b"image")


def test_message_to_openai_message_user_image_message_str():
    image_message = UserImageMessage("https://example.com/image.jpg")
    assert message_to_openai_message(image_message) == {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg", "detail": "auto"},
            }
        ],
    }
