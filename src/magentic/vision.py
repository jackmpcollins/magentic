import base64
from typing import Any, Generic, TypeVar, overload

from openai.types.chat import ChatCompletionMessageParam

from magentic.chat_model.message import Message, Placeholder
from magentic.chat_model.openai_chat_model import (
    OpenaiMessageRole,
    message_to_openai_message,
)

T = TypeVar("T")
ImageContentT = TypeVar("ImageContentT")


class UserImageMessage(Message[ImageContentT], Generic[ImageContentT]):
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
        """Format the message using the given function arguments."""
        if isinstance(self.content, Placeholder):
            return UserImageMessage(self.content.format(**kwargs))
        return UserImageMessage(self.content)


@message_to_openai_message.register(UserImageMessage)
def _(
    message: UserImageMessage[bytes] | UserImageMessage[str],
) -> ChatCompletionMessageParam:
    if isinstance(message.content, bytes):
        base64_image = base64.b64encode(message.content).decode("utf-8")
        url = f"data:image/jpeg;base64,{base64_image}"
    elif isinstance(message.content, str):
        url = message.content
    else:
        msg = f"Invalid content type: {type(message.content)}"
        raise TypeError(msg)

    return {
        "role": OpenaiMessageRole.USER.value,
        "content": [{"type": "image_url", "image_url": {"url": url, "detail": "auto"}}],
    }
