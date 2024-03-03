import base64
from typing import Any, Generic, TypeVar, overload

import filetype
from openai.types.chat import ChatCompletionMessageParam

from magentic.chat_model.message import Message, Placeholder
from magentic.chat_model.openai_chat_model import (
    OpenaiMessageRole,
    message_to_openai_message,
)

T = TypeVar("T", bytes, str)
ImageContentT = TypeVar("ImageContentT")


class UserImageMessage(Message[ImageContentT], Generic[ImageContentT]):
    """A message containing an image sent by a user to an LLM chat model."""

    @overload
    def format(
        self: "UserImageMessage[Placeholder[T]]", **kwargs: Any
    ) -> "UserImageMessage[T]":
        ...

    @overload
    def format(self: "UserImageMessage[T]", **kwargs: Any) -> "UserImageMessage[T]":
        ...

    def format(
        self: "UserImageMessage[Placeholder[T]] | UserImageMessage[T]",
        **kwargs: Any,
    ) -> "UserImageMessage[T]":
        """Format the message using the provided substitutions."""
        if isinstance(self.content, Placeholder):
            return UserImageMessage(self.content.format(**kwargs))
        return UserImageMessage(self.content)


@message_to_openai_message.register(UserImageMessage)
def _(
    message: UserImageMessage[bytes] | UserImageMessage[str],
) -> ChatCompletionMessageParam:
    if isinstance(message.content, bytes):
        mime_type = filetype.guess_mime(message.content)
        base64_image = base64.b64encode(message.content).decode("utf-8")
        url = f"data:{mime_type};base64,{base64_image}"
    elif isinstance(message.content, str):
        url = message.content
    else:
        msg = f"Invalid content type: {type(message.content)}"
        raise TypeError(msg)

    return {
        "role": OpenaiMessageRole.USER.value,
        "content": [{"type": "image_url", "image_url": {"url": url, "detail": "auto"}}],
    }
